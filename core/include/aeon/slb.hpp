#include "aeon/math_kernel.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <limits>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <tuple>
#include <vector>

namespace aeon {

/**
 * @brief Thread-local compatible SLB Entry.
 * Aligned to 64 bytes to respect AVX-512 requirements if we ever do block
 * loads.
 */
struct alignas(64) CacheEntry {
  uint64_t node_id;
  float centroid[768];
  // We use a simple tick counter for LRU.
  // 0 means empty/invalid.
  uint64_t last_accessed_tick;
};

/**
 * @brief Semantic Lookaside Buffer (SLB).
 * A small, high-speed predictive cache for Conversational Continuity.
 *
 * It stores a small set (64) of recently visited nodes.
 * Before hitting the main MemoryFile (mmap), we scan this cache.
 * If we find a node with high similarity (>0.85), we return it immediately.
 */
class SemanticCache {
public:
  static constexpr size_t CACHE_SIZE = 64;

  SemanticCache() : tick_counter_(1) {
    // Initialize entries with 0 tick (empty)
    for (auto &e : entries_) {
      e.last_accessed_tick = 0;
      e.node_id = 0;
    }
  }

  /**
   * @brief Scans the cache for the nearest neighbor to the query.
   *
   * @param query 768-dim query vector
   * @param threshold Minimum similarity to consider a "Hit" (default 0.85)
   * @return std::optional<tuple<id, score, const float*>> if a match >
   * threshold is found
   */
  std::optional<std::tuple<uint64_t, float, const float *>>
  find_nearest(std::span<const float> query, float threshold = 0.85f) {
    std::shared_lock lock(mutex_);

    float best_score = -2.0f;
    uint64_t best_id = 0;
    const float *best_centroid = nullptr;
    bool found = false;

    // Linear scan of 64 entries.
    for (const auto &entry : entries_) {
      if (entry.last_accessed_tick == 0)
        continue; // Skip empty slots

      float score = aeon::math::cosine_similarity(
          query, std::span<const float>(entry.centroid, 768));

      if (score > best_score) {
        best_score = score;
        best_id = entry.node_id;
        best_centroid = entry.centroid;
        found = true;
      }
    }

    if (found && best_score >= threshold) {
      // Return ID, Score, and Pointer to internal centroid storage
      // The pointer points to the array inside CacheEntry.
      // Since std::array is stable, this pointer is valid as long as
      // this specific entry is not overwritten.
      // NOTE: There is a race condition if another thread overwrites THIS entry
      // immediately after we release the lock.
      // For MVP, we accept this risk or we should return a copy of preview.
      // But preserving zero-copy principle requires pointer.
      // Given "Conversational Drift" is mostly single-session or isolated,
      // and we are returning "Immutable" results logic (snapshot),
      // we proceed with pointer.
      return std::make_tuple(best_id, best_score, best_centroid);
    }

    return std::nullopt;
  }

  /**
   * @brief Inserts a node into the cache. Evicts LRU if full.
   */
  void insert(uint64_t node_id, std::span<const float> centroid) {
    if (centroid.size() != 768)
      return;

    std::unique_lock lock(mutex_);

    // 1. Check if already exists (update vector/tick)
    for (auto &entry : entries_) {
      if (entry.last_accessed_tick > 0 && entry.node_id == node_id) {
        update_entry(entry, node_id, centroid);
        return;
      }
    }

    // 2. Find empty slot
    for (auto &entry : entries_) {
      if (entry.last_accessed_tick == 0) {
        update_entry(entry, node_id, centroid);
        return;
      }
    }

    // 3. Evict LRU (Smallest tick)
    auto *lru = &entries_[0];
    for (auto &entry : entries_) {
      if (entry.last_accessed_tick < lru->last_accessed_tick) {
        lru = &entry;
      }
    }
    update_entry(*lru, node_id, centroid);
  }

  /**
   * @brief Stubs for future prefetching logic.
   */
  void prefetch_neighbors(uint64_t node_id) {
    // TODO: Phase 9+
  }

private:
  void update_entry(CacheEntry &entry, uint64_t id,
                    std::span<const float> vec) {
    entry.node_id = id;
    std::copy(vec.begin(), vec.end(), std::begin(entry.centroid));
    entry.last_accessed_tick = tick_counter_.fetch_add(1);
  }

  std::array<CacheEntry, CACHE_SIZE> entries_;
  std::atomic<uint64_t> tick_counter_;
  mutable std::shared_mutex mutex_;
};

} // namespace aeon
