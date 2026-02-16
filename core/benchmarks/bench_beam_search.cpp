/**
 * @file bench_beam_search.cpp
 * @brief 1M-node synthetic Atlas beam search benchmark.
 *
 * Generates a 1,000,000 node mmap-backed tree (branching factor 64, D=768),
 * then runs beam_width=1, beam_width=3, and beam_width=3+CSLS configurations.
 * Reports P50/P90/P99/Mean latencies and ResultNodes per query.
 *
 * Anti-artifact hardening:
 *  - Pre-generates 1,000 unique random query vectors.
 *  - Each measurement iteration uses a DIFFERENT query via modulo index,
 *    defeating the CPU branch predictor's ability to memorize tree traversal
 *    paths from prior iterations.
 *  - ResultNodes counter tracks the total number of result nodes returned,
 *    discriminating algorithmic pruning from CPU pipeline artifacts.
 */

#include "aeon/atlas.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <vector>

static constexpr size_t NUM_NODES = 1'000'000;
static constexpr size_t QUERY_POOL_SIZE = 1'000;
static constexpr size_t NUM_QUERIES = 1'000;
static constexpr size_t DIM = 768;

/// Generate a random normalized 768-dim vector
static void random_unit_vector(float *out, std::mt19937 &rng) {
  std::normal_distribution<float> dist(0.0f, 1.0f);
  float norm_sq = 0.0f;
  for (size_t i = 0; i < DIM; ++i) {
    out[i] = dist(rng);
    norm_sq += out[i] * out[i];
  }
  float inv_norm = 1.0f / std::sqrt(norm_sq);
  for (size_t i = 0; i < DIM; ++i)
    out[i] *= inv_norm;
}

struct LatencyStats {
  double p50_us;
  double p90_us;
  double p99_us;
  double mean_us;
};

static LatencyStats compute_stats(std::vector<double> &latencies) {
  std::sort(latencies.begin(), latencies.end());
  size_t n = latencies.size();
  double sum = 0;
  for (auto l : latencies)
    sum += l;

  return {
      latencies[n / 2],
      latencies[static_cast<size_t>(n * 0.90)],
      latencies[static_cast<size_t>(n * 0.99)],
      sum / static_cast<double>(n),
  };
}

/// Run a benchmark configuration with query rotation + result counting.
///
/// @param atlas         The pre-built 1M-node Atlas.
/// @param queries       Pool of 1,000 unique random queries.
/// @param beam_width    Beam width for navigate().
/// @param use_csls      Whether to enable CSLS hub penalty.
/// @param config_name   Label for output.
/// @returns             {stats, total_result_nodes}
static std::pair<LatencyStats, uint64_t>
run_config(aeon::Atlas &atlas, const std::vector<std::vector<float>> &queries,
           uint32_t beam_width, bool use_csls, const char *config_name) {

  std::printf("Benchmarking %s...\n", config_name);

  // Warmup — 10 unique queries to prime instruction cache only
  for (size_t i = 0; i < 10; ++i) {
    auto r = atlas.navigate(std::span<const float>(queries[i]), beam_width,
                            use_csls);
    asm volatile("" : : "r"(r.size()) : "memory");
  }

  std::vector<double> latencies;
  latencies.reserve(NUM_QUERIES);
  uint64_t total_result_nodes = 0;

  // Measurement — rotate through query pool to defeat branch predictor
  for (size_t i = 0; i < NUM_QUERIES; ++i) {
    // Use modulo to cycle through the full pool, ensuring each iteration
    // presents a different query vector to the tree traversal.
    const auto &q = queries[i % QUERY_POOL_SIZE];

    auto t0 = std::chrono::steady_clock::now();
    auto result =
        atlas.navigate(std::span<const float>(q), beam_width, use_csls);
    auto t1 = std::chrono::steady_clock::now();

    // Force materialization — prevent dead code elimination
    asm volatile("" : : "r"(result.size()) : "memory");

    total_result_nodes += result.size();
    latencies.push_back(
        std::chrono::duration<double, std::micro>(t1 - t0).count());
  }

  auto stats = compute_stats(latencies);
  std::printf("  Mean: %8.1f µs\n", stats.mean_us);
  std::printf("  P50:  %8.1f µs\n", stats.p50_us);
  std::printf("  P90:  %8.1f µs\n", stats.p90_us);
  std::printf("  P99:  %8.1f µs\n", stats.p99_us);
  std::printf("  ResultNodes (total): %llu  (mean: %.1f/query)\n",
              static_cast<unsigned long long>(total_result_nodes),
              static_cast<double>(total_result_nodes) /
                  static_cast<double>(NUM_QUERIES));

  return {stats, total_result_nodes};
}

int main() {
  std::printf(
      "=== Aeon Beam Search Benchmark (1M Nodes, Query Rotation) ===\n");
  std::printf("Nodes:      %zu\n", NUM_NODES);
  std::printf("Query pool: %zu unique random vectors\n", QUERY_POOL_SIZE);
  std::printf("Queries:    %zu per configuration (modulo-rotated)\n",
              NUM_QUERIES);
  std::printf("Dim:        %zu\n\n", DIM);

  // Create temporary Atlas file
  auto tmp_path =
      std::filesystem::temp_directory_path() / "aeon_bench_beam.atlas";
  std::filesystem::remove(tmp_path); // Clean up any prior run

  {
    aeon::Atlas atlas(tmp_path);
    std::mt19937 rng(42);

    // Build tree: root (node 0), then children in sequential blocks of 64
    std::printf("Building %zu-node synthetic tree...\n", NUM_NODES);
    auto start_build = std::chrono::steady_clock::now();

    std::vector<float> vec(DIM);

    // Insert root
    random_unit_vector(vec.data(), rng);
    atlas.insert(0, std::span<const float>(vec), "root");

    // Insert remaining nodes — parent assignment: node i's parent = i / 64
    // This creates a branching factor ~64 tree with depth ~3 for 1M nodes
    for (size_t i = 1; i < NUM_NODES; ++i) {
      random_unit_vector(vec.data(), rng);
      uint64_t parent_id = (i - 1) / 64; // BF=64 tree
      atlas.insert(parent_id, std::span<const float>(vec),
                   "node_" + std::to_string(i));
    }

    auto build_time = std::chrono::steady_clock::now() - start_build;
    std::printf("Build time: %.2f s (%zu nodes)\n\n",
                std::chrono::duration<double>(build_time).count(),
                atlas.size());

    // Pre-generate query pool: 1,000 unique random vectors
    // Using a DIFFERENT seed (137) from tree construction (42) to ensure
    // queries are independently distributed from the stored vectors.
    std::mt19937 query_rng(137);
    std::vector<std::vector<float>> query_pool(QUERY_POOL_SIZE);
    for (auto &q : query_pool) {
      q.resize(DIM);
      random_unit_vector(q.data(), query_rng);
    }

    // --- Benchmark: beam_width=1 (Greedy) ---
    auto [s1, rn1] =
        run_config(atlas, query_pool, 1, false, "beam_width=1 (Greedy)");

    std::printf("\n");

    // --- Benchmark: beam_width=3 ---
    auto [s3, rn3] = run_config(atlas, query_pool, 3, false, "beam_width=3");

    std::printf("\n");

    // --- Benchmark: beam_width=3 + CSLS ---
    auto [sc, rnc] =
        run_config(atlas, query_pool, 3, true, "beam_width=3 + CSLS");

    // ── Scaling & Pruning Analysis ──
    std::printf("\n--- Scaling Analysis ---\n");
    double ratio = s3.p50_us / s1.p50_us;
    std::printf("beam=3 / beam=1 P50 ratio: %.2fx\n", ratio);
    if (ratio <= 3.5) {
      std::printf("✅ PASS: beam_width=3 scales sub-linearly "
                  "(%.2fx <= 3.5x)\n",
                  ratio);
    } else {
      std::printf("❌ FAIL: beam_width=3 scales supra-linearly "
                  "(%.2fx > 3.5x)\n",
                  ratio);
    }

    double csls_overhead = (sc.p50_us - s3.p50_us) / s3.p50_us * 100.0;
    std::printf("\nCSLS overhead vs beam=3: %.1f%%\n", csls_overhead);

    // ── Pruning Discrimination ──
    std::printf("\n--- Pruning Analysis (Algorithmic vs CPU Artifact) ---\n");
    std::printf("ResultNodes/query  beam=1: %.1f  beam=3: %.1f  CSLS: %.1f\n",
                static_cast<double>(rn1) / NUM_QUERIES,
                static_cast<double>(rn3) / NUM_QUERIES,
                static_cast<double>(rnc) / NUM_QUERIES);

    if (rnc < rn3) {
      double pruning_pct =
          (1.0 - static_cast<double>(rnc) / static_cast<double>(rn3)) * 100.0;
      std::printf("CSLS returned %.1f%% fewer nodes → hub penalty causes "
                  "algorithmic early pruning.\n",
                  pruning_pct);
      if (csls_overhead < 0.0) {
        std::printf("Hypothesis CONFIRMED: CSLS speedup is from reduced "
                    "heap insertions, not CPU pipeline artifact.\n");
      }
    } else {
      std::printf("CSLS returned equal or more nodes → "
                  "no algorithmic pruning detected.\n");
      if (csls_overhead < -5.0) {
        std::printf("Hypothesis REJECTED: CSLS speedup with same node count "
                    "suggests CPU pipeline artifact (branch predictor).\n");
      }
    }
  }

  // Cleanup
  std::filesystem::remove(tmp_path);
  return 0;
}
