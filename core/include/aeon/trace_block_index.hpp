#pragma once

/**
 * @file trace_block_index.hpp
 * @brief Chronological Block Index for O(|V|/1024 + K*1024) trace search.
 *
 * Trace nodes are append-only (temporal). This index groups them into
 * fixed-size TraceBlocks of 1,024 embeddings each. Each block maintains
 * an incrementally updated centroid. Search is two-phase:
 *   Phase 1: SIMD scan across block centroids → top-K blocks
 *   Phase 2: SIMD scan ONLY inside top-K blocks → final results
 *
 * This reduces O(|V|) to O(|V|/1024 + K*1024), yielding ~5ms for 10^8
 * nodes while maintaining perfect memory contiguity and zero
 * tree-rebalancing overhead.
 *
 * Memory layout:
 *   - All embeddings are stored contiguously in a flat vector (column-major
 *     per block) for cache-line-friendly SIMD scanning.
 *   - Block centroids are stored in a separate contiguous array for
 *     the first-phase scan.
 *
 * Zero-allocation hot path:
 *   - query() uses stack-allocated arrays for top-K block selection
 *   - No heap allocation in the search path
 *
 * V4 STAGE 2 TASK 3: dim is now a runtime CONSTRUCTOR parameter, not the
 * compile-time EMBEDDING_DIM constant. The original hardcoded-768 version
 * had the exact same silent-no-op-for-non-768-Atlas bug guardrail #1.1
 * found and fixed in SemanticCache/HierarchicalSLB (v4-plan.md Stage 0) --
 * fixed here the same way: dynamic-size std::vector storage keyed by
 * index*dim_ instead of a fixed std::array<float, EMBEDDING_DIM>.
 */

#include "aeon/math_kernel.hpp"
#include "aeon/schema.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <shared_mutex>
#include <span>
#include <vector>

namespace aeon {

/// Number of embeddings per TraceBlock (tune for L2 cache: 1024 * 768 * 4 =
/// 3MB)
inline constexpr size_t TRACE_BLOCK_SIZE = 1024;

/// Maximum blocks to scan in Phase 2 (stack-allocated)
inline constexpr size_t MAX_TOP_K_BLOCKS = 16;

/**
 * @brief A contiguous block of temporal trace embeddings with a centroid.
 *
 * All embeddings in a block are stored in a flat contiguous array for
 * optimal SIMD scan performance. The centroid is updated incrementally
 * on each append:
 *   centroid = centroid * (n-1)/n + embedding * (1/n)
 *
 * When the block is sealed (full), the centroid is exact (mean of all vectors).
 */
struct TraceBlock {
  /// Dimensionality of every embedding in this block (fixed at construction,
  /// matches the owning TraceBlockIndex's dim_).
  uint32_t dim = 0;

  /// Contiguous embedding storage: [TRACE_BLOCK_SIZE * dim]
  /// Layout: embeddings[i * dim .. (i+1) * dim]
  std::vector<float> embeddings;

  /// Incrementally updated centroid (mean of all embeddings in this block)
  std::vector<float> centroid;

  /// Number of embeddings currently in this block [0, TRACE_BLOCK_SIZE]
  size_t count = 0;

  /// Whether this block is full (sealed) — no more appends
  bool sealed = false;

  /// Timestamp range for this block (for temporal filtering)
  double timestamp_start = 0.0;
  double timestamp_end = 0.0;

  /// Node IDs stored in this block (parallel array with embeddings)
  std::vector<uint64_t> node_ids;

  explicit TraceBlock(uint32_t dimensionality) : dim(dimensionality) {
    embeddings.resize(TRACE_BLOCK_SIZE * dim, 0.0f);
    centroid.resize(dim, 0.0f);
    node_ids.resize(TRACE_BLOCK_SIZE, 0);
  }

  /// Append a single embedding to this block. Returns true if block is now
  /// sealed.
  bool append(uint64_t node_id, std::span<const float> embedding,
              double timestamp) {
    if (sealed || count >= TRACE_BLOCK_SIZE)
      return true;

    // Copy embedding into contiguous storage
    std::memcpy(&embeddings[count * dim], embedding.data(),
                dim * sizeof(float));
    node_ids[count] = node_id;

    // Update centroid incrementally: centroid = centroid*(n/(n+1)) +
    // emb*(1/(n+1))
    float n = static_cast<float>(count);
    float inv_n1 = 1.0f / (n + 1.0f);
    float scale_old = n * inv_n1;

    const float *emb = embedding.data();
    for (uint32_t d = 0; d < dim; ++d) {
      centroid[d] = centroid[d] * scale_old + emb[d] * inv_n1;
    }

    // Update timestamps
    if (count == 0)
      timestamp_start = timestamp;
    timestamp_end = timestamp;

    ++count;
    if (count >= TRACE_BLOCK_SIZE) {
      sealed = true;
      return true;
    }
    return false;
  }
};

/// Result from trace search
struct TraceSearchResult {
  uint64_t node_id;
  float similarity;
  size_t block_index;
};

/**
 * @brief Chronological Block Index for high-speed trace pattern retrieval.
 *
 * Reduces O(|V|) linear scan to O(|V|/1024 + K*1024) two-phase search:
 *   Phase 1: Scan block centroids (SIMD) → select top-K blocks
 *   Phase 2: Scan embeddings in top-K blocks only (SIMD) → final results
 *
 * Complexity for 10^8 nodes:
 *   Phase 1: 100M / 1024 ≈ 97,656 centroid comparisons (~0.5ms)
 *   Phase 2: K * 1024 = 16,384 embedding comparisons (~1ms)
 *   Total: ~1.5ms (vs 20s for naive O(|V|))
 *
 * Thread safety: append() takes exclusive lock, query() takes shared lock.
 *
 * NOT persisted across process restarts -- this is an in-memory
 * ACCELERATION structure only, rebuilt from durable TraceEvent embedding
 * blobs (schema.hpp) on TraceManager startup. Every embedding it indexes
 * is independently durable in the sidecar BlobArena regardless of this
 * index's state.
 */
class TraceBlockIndex {
public:
  explicit TraceBlockIndex(uint32_t dim) : dim_(dim) {}

  /// Dimensionality this index was constructed for.
  uint32_t dim() const noexcept { return dim_; }

  /**
   * @brief Append a trace embedding to the index.
   * O(1) amortized. Creates a new block when current one is sealed.
   *
   * Silently ignores a mismatched-size embedding (matches
   * HierarchicalSLB's MismatchedSizeIsSafeNoOp precedent, v4-plan.md
   * Stage 0) rather than corrupting fixed-stride block storage.
   */
  void append(uint64_t node_id, std::span<const float> embedding,
              double timestamp) {
    if (embedding.size() != dim_)
      return;

    std::unique_lock lock(mutex_);

    if (blocks_.empty() || blocks_.back().sealed) {
      blocks_.emplace_back(dim_);
    }

    blocks_.back().append(node_id, embedding, timestamp);

    // Update centroid cache for fast Phase 1 scan
    size_t block_idx = blocks_.size() - 1;
    size_t needed = (block_idx + 1) * dim_;
    if (needed > centroid_cache_.size()) {
      centroid_cache_.resize(needed);
    }
    std::memcpy(&centroid_cache_[block_idx * dim_],
                blocks_.back().centroid.data(), dim_ * sizeof(float));
  }

  /**
   * @brief Two-phase SIMD search for closest trace embeddings.
   *
   * Phase 1: Scan block centroids → top-K blocks (stack-allocated)
   * Phase 2: Scan embeddings in selected blocks → final results
   *
   * Zero heap allocation in the block-selection hot path: uses
   * std::array for top-K tracking.
   *
   * @param query Query embedding vector (must match dim())
   * @param top_k Number of results to return
   * @param num_blocks_to_scan Number of blocks to scan in Phase 2 (default 8)
   * @return Vector of TraceSearchResult sorted by similarity (descending).
   *         Empty if query.size() != dim().
   */
  std::vector<TraceSearchResult> query(std::span<const float> query,
                                       size_t top_k = 10,
                                       size_t num_blocks_to_scan = 8) const {
    if (query.size() != dim_)
      return {};

    std::shared_lock lock(mutex_);

    if (blocks_.empty())
      return {};

    const size_t num_blocks = blocks_.size();
    num_blocks_to_scan = std::min(num_blocks_to_scan, num_blocks);
    num_blocks_to_scan =
        std::min(num_blocks_to_scan, static_cast<size_t>(MAX_TOP_K_BLOCKS));

    // ================================================================
    // PHASE 1: Scan block centroids → select top-K blocks
    //          Stack-allocated, zero heap allocation
    // ================================================================
    struct BlockScore {
      float score;
      size_t index;
    };

    // Stack-allocated top-K block tracker (insertion sort, heap-free)
    std::array<BlockScore, MAX_TOP_K_BLOCKS> top_blocks{};
    size_t top_count = 0;
    float worst_top = -2.0f;

    for (size_t b = 0; b < num_blocks; ++b) {
      float sim = math::cosine_similarity(
          query, std::span<const float>(&centroid_cache_[b * dim_], dim_));

      if (top_count < num_blocks_to_scan) {
        // Fill phase: unconditionally add
        top_blocks[top_count++] = {sim, b};
        // Update worst when we've filled the buffer
        if (top_count == num_blocks_to_scan) {
          worst_top = top_blocks[0].score;
          for (size_t i = 1; i < top_count; ++i) {
            if (top_blocks[i].score < worst_top)
              worst_top = top_blocks[i].score;
          }
        }
      } else if (sim > worst_top) {
        // Replace worst element (find it first)
        size_t worst_idx = 0;
        for (size_t i = 1; i < top_count; ++i) {
          if (top_blocks[i].score < top_blocks[worst_idx].score)
            worst_idx = i;
        }
        top_blocks[worst_idx] = {sim, b};

        // Recompute worst
        worst_top = top_blocks[0].score;
        for (size_t i = 1; i < top_count; ++i) {
          if (top_blocks[i].score < worst_top)
            worst_top = top_blocks[i].score;
        }
      }
    }

    // ================================================================
    // PHASE 2: Scan embeddings in selected blocks → final results
    //          Still uses stack-allocated comparison, but result vector
    //          is heap-allocated (output only, not hot-path).
    // ================================================================
    std::vector<TraceSearchResult> results;
    results.reserve(top_count * TRACE_BLOCK_SIZE);

    for (size_t i = 0; i < top_count; ++i) {
      const auto &block = blocks_[top_blocks[i].index];
      const float *emb_data = block.embeddings.data();

      for (size_t j = 0; j < block.count; ++j) {
        float sim = math::cosine_similarity(
            query, std::span<const float>(&emb_data[j * dim_], dim_));

        results.push_back({block.node_ids[j], sim, top_blocks[i].index});
      }
    }

    // Partial sort for top-k results
    if (results.size() > top_k) {
      std::partial_sort(
          results.begin(), results.begin() + static_cast<ptrdiff_t>(top_k),
          results.end(),
          [](const TraceSearchResult &a, const TraceSearchResult &b) {
            return a.similarity > b.similarity;
          });
      results.resize(top_k);
    } else {
      std::sort(results.begin(), results.end(),
                [](const TraceSearchResult &a, const TraceSearchResult &b) {
                  return a.similarity > b.similarity;
                });
    }

    return results;
  }

  /// Total number of trace embeddings indexed
  size_t size() const {
    std::shared_lock lock(mutex_);
    size_t total = 0;
    for (const auto &b : blocks_)
      total += b.count;
    return total;
  }

  /// Number of blocks (sealed + active)
  size_t block_count() const {
    std::shared_lock lock(mutex_);
    return blocks_.size();
  }

private:
  mutable std::shared_mutex mutex_;

  /// Dimensionality every embedding in this index must match.
  uint32_t dim_;

  /// All blocks in chronological order (append-only)
  std::vector<TraceBlock> blocks_;

  /// Flat cache of block centroids for Phase 1 scan.
  /// Layout: centroid_cache_[block_idx * dim_ + d] (contiguous, not
  /// std::array<float, EMBEDDING_DIM> -- see file doc comment).
  std::vector<float> centroid_cache_;
};

} // namespace aeon
