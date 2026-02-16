#pragma once

/**
 * @file epoch.hpp
 * @brief Epoch-Based Reclamation (EBR) for lock-free read-side memory safety.
 *
 * Provides zero-overhead read-side synchronization for the Aeon memory kernel.
 * Writers defer munmap of old mmap regions until all readers have advanced
 * past the retirement epoch. Readers pay only a single atomic store on
 * entry/exit — no locks, no CAS on the hot path.
 *
 * Memory ordering contract:
 *   Reader enter:  slot.store(global_epoch, release)
 *   Reader exit:   slot.store(NOT_READING, release)
 *   Writer grow:   data_.store(new_ptr, release) → retire(old) → epoch++
 *   Reclaim scan:  slot.load(acquire) for each active reader
 *
 * Cache-line isolation:
 *   All per-reader atomic slots are padded to CACHE_LINE_SIZE (64 bytes)
 *   to eliminate false sharing across CPU cores. The global epoch counter
 *   is similarly isolated.
 *
 * @see storage.hpp — MemoryFile::grow() uses retire() for deferred munmap
 * @see atlas.hpp   — Atlas::navigate() acquires EpochGuard before mmap access
 */

#include "aeon/platform.hpp"
#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace aeon {

// ---------------------------------------------------------------------------
// Cache-line size: 64 bytes is correct for all target architectures
// (x86-64 L1d, ARM Cortex-A, Apple M-series). We avoid
// std::hardware_destructive_interference_size because Apple Clang
// advertises the feature-test macro but does not provide the constant.
// ---------------------------------------------------------------------------
inline constexpr std::size_t CACHE_LINE_SIZE = 64;

/// Maximum concurrent readers. Sized for industrial multi-tenant workloads.
constexpr size_t MAX_READERS = 64;

/// Sentinel: slot is claimed but reader is not currently in an epoch.
constexpr uint64_t EPOCH_NOT_READING = UINT64_MAX;

/// Region of memory retired for deferred munmap.
struct RetiredRegion {
  void *addr;
  size_t size;
  uint64_t retired_at_epoch;
};

// ---------------------------------------------------------------------------
// Cache-line padded atomics — each lives on its own cache line to prevent
// false sharing when multiple cores concurrently access adjacent reader slots.
// ---------------------------------------------------------------------------

/// Cache-line aligned atomic uint64_t for per-reader epoch announcements.
struct alignas(CACHE_LINE_SIZE) AlignedAtomicU64 {
  std::atomic<uint64_t> value{EPOCH_NOT_READING};
};

/// Cache-line aligned atomic bool for per-reader slot ownership flags.
struct alignas(CACHE_LINE_SIZE) AlignedAtomicBool {
  std::atomic<bool> value{false};
};

/// Cache-line aligned global epoch counter (isolated from reader arrays).
struct alignas(CACHE_LINE_SIZE) AlignedEpochCounter {
  std::atomic<uint64_t> value{1};
};

// Forward declaration
class EpochGuard;

/**
 * @brief Manages epoch-based reclamation for mmap memory regions.
 *
 * Thread-safe. Readers call enter_guard() to protect their read window.
 * Writers call retire() to defer munmap, then advance_epoch() to trigger
 * reclamation of regions no longer referenced by any active reader.
 *
 * All per-reader slots are cache-line padded (alignas(64)) to eliminate
 * false sharing. The global epoch counter is similarly isolated.
 */
class EpochManager {
public:
  EpochManager() = default;

  ~EpochManager() {
    // Invalidate thread-local cached slot before destruction to prevent
    // stale pointer matches in subsequent EpochManager instances (e.g.
    // sequential GTest cases allocating EpochManager on the same stack
    // address).
    reset_thread_local_slot();

    // Reclaim all remaining retired regions on destruction
    std::lock_guard<std::mutex> lock(retired_mutex_);
    for (auto &r : retired_) {
      if (r.addr) {
        platform::mem_unmap(r.addr, r.size);
      }
    }
    retired_.clear();
  }

  // Non-copyable, non-moveable
  EpochManager(const EpochManager &) = delete;
  EpochManager &operator=(const EpochManager &) = delete;

  /**
   * @brief Allocate a reader slot. Uses thread_local caching to avoid
   *        CAS contention on the hot path. First call per thread does CAS
   *        scan; subsequent calls reuse the cached slot.
   * @return Slot index in [0, MAX_READERS).
   * @throws std::runtime_error if all slots are exhausted after spin-backoff.
   */
  size_t acquire_slot() {
    auto &tls = tls_slot_cache();
    if (tls.mgr == this && tls.slot < MAX_READERS) {
      return tls.slot;
    }

    // Spin with exponential backoff to find a free slot
    constexpr int MAX_ATTEMPTS = 1000;
    for (int attempt = 0; attempt < MAX_ATTEMPTS; ++attempt) {
      for (size_t i = 0; i < MAX_READERS; ++i) {
        bool expected = false;
        if (slot_claimed_[i].value.compare_exchange_strong(
                expected, true, std::memory_order_acq_rel)) {
          // Successfully claimed slot i
          reader_epochs_[i].value.store(EPOCH_NOT_READING,
                                        std::memory_order_release);
          tls.slot = i;
          tls.mgr = this;
          return i;
        }
      }
      // Exponential backoff: yield then sleep
      if (attempt < 10) {
        std::this_thread::yield();
      } else {
        std::this_thread::sleep_for(
            std::chrono::microseconds(1 << std::min(attempt - 10, 10)));
      }
    }

    throw std::runtime_error("EBR: All reader slots exhausted (MAX_READERS=" +
                             std::to_string(MAX_READERS) + ")");
  }

  /**
   * @brief Reset the thread-local cached slot for this EpochManager.
   *
   * Must be called when an EpochManager is being destroyed or reset between
   * test runs. Without this, thread-local slot caching causes stale references
   * across test boundaries (the root cause of the TSan flake in
   * RetireBlockedByActiveReader).
   */
  void reset_thread_local_slot() {
    auto &tls = tls_slot_cache();
    if (tls.mgr == this) {
      // Release the slot so it can be reused
      if (tls.slot < MAX_READERS) {
        reader_epochs_[tls.slot].value.store(EPOCH_NOT_READING,
                                             std::memory_order_release);
        slot_claimed_[tls.slot].value.store(false, std::memory_order_release);
      }
      tls.slot = SIZE_MAX;
      tls.mgr = nullptr;
    }
  }

  /**
   * @brief Enter a read epoch. Stores current global epoch to the slot.
   *        Must be paired with exit().
   *
   * Memory ordering: release — ensures the slot write is visible to
   * the writer's reclamation scan before the reader proceeds to access data.
   */
  void enter(size_t slot) {
    uint64_t epoch = global_epoch_.value.load(std::memory_order_acquire);
    reader_epochs_[slot].value.store(epoch, std::memory_order_release);
  }

  /**
   * @brief Exit a read epoch. Marks slot as not-reading.
   *
   * Memory ordering: release — ensures all preceding reads complete
   * before the slot is marked as idle (happens-before the writer's scan).
   */
  void exit(size_t slot) {
    reader_epochs_[slot].value.store(EPOCH_NOT_READING,
                                     std::memory_order_release);
  }

  /**
   * @brief Create an RAII epoch guard that auto-enters and auto-exits.
   */
  EpochGuard enter_guard();

  /**
   * @brief Advance the global epoch counter. Called by writers after
   *        structural mutations (grow, compaction). Triggers try_reclaim().
   */
  void advance_epoch() {
    global_epoch_.value.fetch_add(1, std::memory_order_acq_rel);
    try_reclaim();
  }

  /**
   * @brief Schedule a memory region for deferred munmap.
   *        The region will be freed once all readers have advanced past
   *        the current epoch.
   */
  void retire(void *addr, size_t size) {
    uint64_t epoch = global_epoch_.value.load(std::memory_order_acquire);
    std::lock_guard<std::mutex> lock(retired_mutex_);
    retired_.push_back({addr, size, epoch});
  }

  /**
   * @brief Attempt to free retired regions that no active reader references.
   *        A region is reclaimable if min_active_epoch() > retired_at_epoch.
   */
  void try_reclaim() {
    std::lock_guard<std::mutex> lock(retired_mutex_);
    if (retired_.empty())
      return;

    uint64_t safe_epoch = min_active_epoch();

    // Partition: move reclaimable regions to the end
    auto it = std::remove_if(retired_.begin(), retired_.end(),
                             [safe_epoch](const RetiredRegion &r) {
                               if (r.retired_at_epoch < safe_epoch) {
                                 if (r.addr) {
                                   platform::mem_unmap(r.addr, r.size);
                                 }
                                 return true; // Remove from list
                               }
                               return false;
                             });
    retired_.erase(it, retired_.end());
  }

  /**
   * @brief Block until all active readers have released their guards.
   *        Used by MemoryFile::close() for safe shutdown.
   */
  void drain_readers() {
    constexpr int MAX_DRAIN_ATTEMPTS = 10000;
    for (int attempt = 0; attempt < MAX_DRAIN_ATTEMPTS; ++attempt) {
      bool all_idle = true;
      for (size_t i = 0; i < MAX_READERS; ++i) {
        if (slot_claimed_[i].value.load(std::memory_order_acquire)) {
          uint64_t val =
              reader_epochs_[i].value.load(std::memory_order_acquire);
          if (val != EPOCH_NOT_READING) {
            all_idle = false;
            break;
          }
        }
      }
      if (all_idle)
        return;

      if (attempt < 100) {
        std::this_thread::yield();
      } else {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }
    }
    // If we get here, readers are stuck — proceed anyway (best effort)
  }

  /// Get current global epoch (for debugging/testing).
  uint64_t current_epoch() const {
    return global_epoch_.value.load(std::memory_order_acquire);
  }

  /// Get count of pending retired regions (for testing).
  size_t retired_count() const {
    std::lock_guard<std::mutex> lock(retired_mutex_);
    return retired_.size();
  }

private:
  /// Thread-local slot cache shared between acquire_slot() and
  /// reset_thread_local_slot(). Must be a static method returning a reference
  /// to a thread_local, since C++ thread_local at function scope creates
  /// separate instances per function.
  struct TLSSlotCache {
    size_t slot = SIZE_MAX;
    EpochManager *mgr = nullptr;
  };
  static TLSSlotCache &tls_slot_cache() {
    thread_local TLSSlotCache cache;
    return cache;
  }

  /// Global monotonically increasing epoch counter (cache-line isolated).
  AlignedEpochCounter global_epoch_;

  /// Per-reader epoch announcement slots (each on its own cache line).
  std::array<AlignedAtomicU64, MAX_READERS> reader_epochs_;

  /// Ownership bitfield for slot allocation (each on its own cache line).
  std::array<AlignedAtomicBool, MAX_READERS> slot_claimed_;

  /// Deferred reclamation queue.
  std::vector<RetiredRegion> retired_;
  mutable std::mutex retired_mutex_;

  /**
   * @brief Find minimum epoch across all active (non-idle) readers.
   *        Returns UINT64_MAX if no readers are active (all reclaimable).
   */
  uint64_t min_active_epoch() const {
    uint64_t min_epoch = UINT64_MAX;
    for (size_t i = 0; i < MAX_READERS; ++i) {
      if (slot_claimed_[i].value.load(std::memory_order_acquire)) {
        uint64_t val = reader_epochs_[i].value.load(std::memory_order_acquire);
        if (val != EPOCH_NOT_READING && val < min_epoch) {
          min_epoch = val;
        }
      }
    }
    return min_epoch;
  }
};

/**
 * @brief RAII guard that enters an epoch on construction and exits
 *        on destruction. Prevents mmap reclamation while active.
 *
 * Non-copyable, moveable. Use EpochManager::enter_guard() to create.
 */
class EpochGuard {
public:
  EpochGuard(EpochManager &mgr, size_t slot)
      : mgr_(&mgr), slot_(slot), active_(true) {
    mgr_->enter(slot_);
  }

  ~EpochGuard() { release(); }

  // Non-copyable
  EpochGuard(const EpochGuard &) = delete;
  EpochGuard &operator=(const EpochGuard &) = delete;

  // Moveable
  EpochGuard(EpochGuard &&other) noexcept
      : mgr_(other.mgr_), slot_(other.slot_), active_(other.active_) {
    other.active_ = false;
  }

  EpochGuard &operator=(EpochGuard &&other) noexcept {
    if (this != &other) {
      release();
      mgr_ = other.mgr_;
      slot_ = other.slot_;
      active_ = other.active_;
      other.active_ = false;
    }
    return *this;
  }

  /// Explicitly release the guard (idempotent).
  void release() {
    if (active_) {
      mgr_->exit(slot_);
      active_ = false;
    }
  }

  /// Check if guard is still active.
  bool is_active() const { return active_; }

private:
  EpochManager *mgr_;
  size_t slot_;
  bool active_;
};

// Implementation of enter_guard (needs full EpochGuard definition)
inline EpochGuard EpochManager::enter_guard() {
  size_t slot = acquire_slot();
  return EpochGuard(*this, slot);
}

} // namespace aeon
