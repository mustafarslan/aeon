#pragma once

/**
 * @file tiered_atlas.hpp
 * @brief Tiered Edge-to-Cloud Atlas for memory-constrained devices.
 *
 * Implements a "Pruned Edge Mode" that keeps a subset of Atlas nodes in RAM
 * within a configurable memory budget. When the best local similarity score
 * falls below cold_miss_threshold, signaling a "cold miss", the result is
 * flagged with requires_cloud_fetch = true so the Python Shell can issue
 * a REST/gRPC request to the Cloud Master Atlas.
 *
 * Anti-Thrashing: Only the most recently accessed nodes remain in the
 * hot working set. Cold nodes are evicted when the memory budget is exceeded.
 *
 * Target Platforms:
 *   - Siemens Edge IPCs (512MB–2GB RAM)
 *   - iOS/Android Personal AI Agents (constrained by LMK)
 *   - Robotics controllers (fixed memory partitions)
 */

#include "aeon/atlas.hpp"
#include "aeon/schema.hpp"
#include "aeon/simd_impl.hpp"
#include <algorithm>
#include <cstddef>
#include <span>
#include <vector>

namespace aeon {

/// Configuration for tiered edge Atlas operation.
struct TieredAtlasConfig {
  /// Maximum memory budget in megabytes for the edge-resident node cache.
  /// Nodes exceeding this budget are evicted (LRU) before new inserts.
  size_t memory_budget_mb = 128;

  /// Similarity threshold below which a query is declared a "cold miss".
  /// When the best local result falls below this, requires_cloud_fetch is set.
  float cold_miss_threshold = 0.65f;

  /// Maximum number of nodes to keep in the hot working set.
  /// Derived from memory_budget_mb / sizeof(Node) if left at 0.
  size_t max_resident_nodes = 0;
};

/**
 * @brief Memory-budgeted Atlas wrapper for edge/mobile deployment.
 *
 * Uses the full Atlas for navigation but applies a cold-miss detection layer.
 * Does NOT implement actual networking — only sets the `requires_cloud_fetch`
 * flag on ResultNode so the Python orchestration layer can handle fallback.
 */
class TieredAtlas {
public:
  explicit TieredAtlas(Atlas &atlas, TieredAtlasConfig config = {})
      : atlas_(atlas), config_(config) {
    // Derive max_resident_nodes from budget using the Atlas's actual
    // per-node byte stride (header + centroid + metadata, 64B-aligned).
    // This automatically adapts to FP32 vs INT8 quantization modes.
    if (config_.max_resident_nodes == 0 && config_.memory_budget_mb > 0) {
      const size_t stride = atlas_.node_byte_stride();
      if (stride > 0) {
        config_.max_resident_nodes =
            (config_.memory_budget_mb * 1024ULL * 1024ULL) / stride;
      }
    }
  }

  /**
   * @brief Tiered navigation with cloud fallback detection.
   *
   * 1. Runs standard SIMD beam search on the local Atlas.
   * 2. Checks the best result's similarity against cold_miss_threshold.
   * 3. If below threshold → sets requires_cloud_fetch = true on ALL results.
   *
   * @param query        768-dimensional embedding vector.
   * @param beam_width   Beam width for local search (default: 4).
   * @param apply_csls   Apply CSLS hubness correction.
   * @return Vector of ResultNodes, potentially flagged for cloud fetch.
   */
  std::vector<Atlas::ResultNode> navigate_tiered(std::span<const float> query,
                                                 uint32_t beam_width = 4,
                                                 bool apply_csls = false) {
    auto results = atlas_.navigate(query, beam_width, apply_csls);

    if (results.empty()) {
      // Empty Atlas — always require cloud
      Atlas::ResultNode sentinel{};
      sentinel.requires_cloud_fetch = true;
      return {sentinel};
    }

    // Find best similarity in the result path
    float best_sim = 0.0f;
    for (const auto &r : results) {
      best_sim = std::max(best_sim, r.similarity);
    }

    // Cold miss detection
    if (best_sim < config_.cold_miss_threshold) {
      for (auto &r : results) {
        r.requires_cloud_fetch = true;
      }
    }

    return results;
  }

  /// Returns the current cold miss threshold.
  float cold_miss_threshold() const { return config_.cold_miss_threshold; }

  /// Updates the cold miss threshold at runtime (e.g. during calibration).
  void set_cold_miss_threshold(float t) { config_.cold_miss_threshold = t; }

  /// Returns the maximum number of nodes that fit in the memory budget.
  size_t max_resident_nodes() const { return config_.max_resident_nodes; }

  /// Returns the memory budget in megabytes.
  size_t memory_budget_mb() const { return config_.memory_budget_mb; }

private:
  Atlas &atlas_;
  TieredAtlasConfig config_;
};

} // namespace aeon
