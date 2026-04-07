#pragma once

/**
 * @file benchmark_harness.hpp
 * @brief Real-World Benchmarking Matrix & Recall Evaluators
 *
 * Empirically proves mathematical assertions on AVQ vs Symmetric degradation
 * models. Replaces older synthetic Gaussian loaders with real-world
 * distribution loaders extracting geometries from `.fvecs` and `.bvecs` data
 * structures.
 */

#include "aeon/metric_dispatch.hpp"
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace aeon::bench {

/**
 * @brief Read-only contiguous view of loaded dataset matrices.
 */
struct DatasetView {
  std::vector<float> data;
  uint32_t dim;
  uint32_t count;

  std::span<const float> get_vector(uint32_t idx) const noexcept {
    return std::span<const float>(data.data() + (idx * dim), dim);
  }
};

/**
 * @brief Parses widely adopted ANN benchmark dataset formats.
 */
class DatasetLoader {
public:
  /**
   * @brief Ingests standard `.fvecs` formatted binary sets.
   * Format: [dimension: 4 bytes int32] [vector: dim * 4 bytes float] ...
   */
  static DatasetView load_fvecs(const std::filesystem::path &path);

  /**
   * @brief Legacy fallback: Synthetic generator enforcing strict anisotropic
   * cluster skew.
   */
  static DatasetView generate_clustered(uint32_t count, uint32_t dim,
                                        uint32_t clusters, int seed = 42);
};

/**
 * @brief Computes strict exact nearest-neighbors using maximum-tier SIMD
 * dispatcher capability.
 *
 * Enforces FP64 (double precision) for all internal distance accumulations to
 * guarantee mathematically perfect "Recall@10" baselines for academic
 * reporting, immune to FP32 catastrophic cancellation on enormous dimensions.
 */
class GroundTruthCalculator {
public:
  /**
   * @brief Computes O(N * Q) cross-matrix dot products avoiding quantization.
   * @return `out[q][k]` = node_idx of the k-th most similar data point.
   */
  static std::vector<std::vector<uint64_t>> compute(const DatasetView &db,
                                                    const DatasetView &queries,
                                                    uint32_t top_k,
                                                    simd::MetricType metric);
};

/**
 * @brief Empirically evaluates search effectiveness matrices.
 */
class RecallCalculator {
public:
  /**
   * @brief Calculates intersection ratio between retrieved candidate nodes and
   * exact labels.
   */
  static float
  recall_at_k(const std::vector<std::vector<uint64_t>> &predicted,
              const std::vector<std::vector<uint64_t>> &ground_truth,
              uint32_t k);
};

/**
 * @brief Trajectory aggregator dumping CSV telemetry for Pareto-front
 * visualizations (Latency vs Recall).
 */
class ParetoRecorder {
public:
  void record(float latency_ns, float recall, const std::string &kernel_label);
  void write_csv(const std::filesystem::path &path) const;

private:
  struct Point {
    float latency;
    float recall;
    std::string tag;
  };
  std::vector<Point> points_;
};

} // namespace aeon::bench
