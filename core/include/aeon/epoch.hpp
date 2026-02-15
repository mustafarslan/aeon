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
 * @see storage.hpp — MemoryFile::grow() uses retire() for deferred munmap
 * @see atlas.hpp   — Atlas::navigate() acquires EpochGuard before mmap access
 */

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <sys/mman.h>
#include <thread>
#include <vector>

namespace aeon {

/// Maximum concurrent readers. Sized to fit in a single cache line group.
constexpr size_t MAX_READERS = 64;

/// Sentinel: slot is claimed but reader is not currently in an epoch.
constexpr uint64_t EPOCH_NOT_READING = UINT64_MAX;

/// Region of memory retired for deferred munmap.
struct RetiredRegion {
  void *addr;
  size_t size;
  uint64_t retired_at_epoch;
};

// Forward declaration
class EpochGuard;

/**
 * @brief Manages epoch-based reclamation for mmap memory regions.
 *
 * Thread-safe. Readers call enter_guard() to protect their read window.
 * Writers call retire() to defer munmap, then advance_epoch() to trigger
 * reclamation of regions no longer referenced by any active reader.
 */
class EpochManager {
public:
  EpochManager() {
    for (auto &slot : reader_epochs_) {
      slot.store(EPOCH_NOT_READING, std::memory_order_relaxed);
    }
    for (auto &claimed : slot_claimed_) {
      claimed.store(false, std::memory_order_relaxed);
    }
  }

  ~EpochManager() {
    // Reclaim all remaining retired regions on destruction
    std::lock_guard<std::mutex> lock(retired_mutex_);
    for (auto &r : retired_) {
      if (r.addr) {
        munmap(r.addr, r.size);
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
    // Thread-local cache: each thread claims a slot once
    thread_local size_t cached_slot = SIZE_MAX;
    thread_local EpochManager *cached_mgr = nullptr;

    if (cached_mgr == this && cached_slot < MAX_READERS) {
      return cached_slot;
    }

    // Spin with exponential backoff to find a free slot
    constexpr int MAX_ATTEMPTS = 1000;
    for (int attempt = 0; attempt < MAX_ATTEMPTS; ++attempt) {
      for (size_t i = 0; i < MAX_READERS; ++i) {
        bool expected = false;
        if (slot_claimed_[i].compare_exchange_strong(
                expected, true, std::memory_order_acq_rel)) {
          // Successfully claimed slot i
          reader_epochs_[i].store(EPOCH_NOT_READING,
                                  std::memory_order_release);
          cached_slot = i;
          cached_mgr = this;
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

    throw std::runtime_error(
        "EBR: All reader slots exhausted (MAX_READERS=" +
        std::to_string(MAX_READERS) + ")");
  }

  /**
   * @brief Enter a read epoch. Stores current global epoch to the slot.
   *        Must be paired with exit().
   *
   * Memory ordering: release — ensures the slot write is visible to
   * the writer's reclamation scan before the reader proceeds to access data.
   */
  void enter(size_t slot) {
    uint64_t epoch = global_epoch_.load(std::memory_order_acquire);
    reader_epochs_[slot].store(epoch, std::memory_order_release);
  }

  /**
   * @brief Exit a read epoch. Marks slot as not-reading.
   *
   * Memory ordering: release — ensures all preceding reads complete
   * before the slot is marked as idle (happens-before the writer's scan).
   */
  void exit(size_t slot) {
    reader_epochs_[slot].store(EPOCH_NOT_READING, std::memory_order_release);
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
    global_epoch_.fetch_add(1, std::memory_order_acq_rel);
    try_reclaim();
  }

  /**
   * @brief Schedule a memory region for deferred munmap.
   *        The region will be freed once all readers have advanced past
   *        the current epoch.
   */
  void retire(void *addr, size_t size) {
    uint64_t epoch = global_epoch_.load(std::memory_order_acquire);
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
                                   munmap(r.addr, r.size);
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
        if (slot_claimed_[i].load(std::memory_order_acquire)) {
          uint64_t val =
              reader_epochs_[i].load(std::memory_order_acquire);
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
    return global_epoch_.load(std::memory_order_acquire);
  }

  /// Get count of pending retired regions (for testing).
  size_t retired_count() const {
    std::lock_guard<std::mutex> lock(retired_mutex_);
    return retired_.size();
  }

private:
  /// Global monotonically increasing epoch counter.
  std::atomic<uint64_t> global_epoch_{1};

  /// Per-reader epoch announcement slots.
  std::array<std::atomic<uint64_t>, MAX_READERS> reader_epochs_;

  /// Ownership bitfield for slot allocation (1 = claimed by a thread).
  std::array<std::atomic<bool>, MAX_READERS> slot_claimed_;

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
      if (slot_claimed_[i].load(std::memory_order_acquire)) {
        uint64_t val = reader_epochs_[i].load(std::memory_order_acquire);
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
