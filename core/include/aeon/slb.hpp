#pragma once

#include "aeon/math_kernel.hpp"
#include "aeon/schema.hpp"
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
  float centroid[EMBEDDING_DIM];
  /// LRU tick counter; 0 indicates an empty/invalid slot.
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
   * @brief Thread-safe SLB hit result. Copies centroid preview instead of
   * returning a raw pointer into CacheEntry storage (which can be evicted
   * concurrently after the shared_lock releases).
   */
  struct SLBHit {
    uint64_t node_id;
    float similarity;
    std::array<float, 3> centroid_preview;
  };

  /**
   * @brief Scans the cache for the nearest neighbor to the query.
   *
   * @param query 768-dim query vector
   * @param threshold Minimum similarity to consider a "Hit" (default 0.85)
   * @return std::optional<SLBHit> Safe result with copied preview data
   */
  std::optional<SLBHit> find_nearest(std::span<const float> query,
                                     float threshold = 0.85f) {
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
      // Return safe copy — no dangling pointer risk after lock release
      return SLBHit{best_id,
                    best_score,
                    {best_centroid[0], best_centroid[1], best_centroid[2]}};
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
   * @brief Hook for future speculative prefetching of neighbor pages.
   */
  void prefetch_neighbors([[maybe_unused]] uint64_t node_id) {
    /// Reserved: future implementation will issue madvise(MADV_WILLNEED)
    /// on the parent's child page to reduce TLB misses during descent.
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
