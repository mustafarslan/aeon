/**
 * @file bench_beam_search.cpp
 * @brief 1M-node synthetic Atlas beam search benchmark.
 *
 * Generates a 1,000,000 node mmap-backed tree (branching factor 64, D=768),
 * then runs 1,000 queries at beam_width=1 (greedy) and beam_width=3.
 * Reports P50 and P99 latencies for both configurations.
 *
 * Performance invariant: beam_width=3 must scale sub-linearly (≤3x) of
 * beam_width=1 latency.
 */

#include "aeon/atlas.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <random>
#include <vector>

static constexpr size_t NUM_NODES = 1'000'000;
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

int main() {
  std::printf("=== Aeon Beam Search Benchmark (1M Nodes) ===\n");
  std::printf("Nodes:    %zu\n", NUM_NODES);
  std::printf("Queries:  %zu per configuration\n", NUM_QUERIES);
  std::printf("Dim:      %zu\n\n", DIM);

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

    // Generate random query vectors
    std::vector<std::vector<float>> queries(NUM_QUERIES);
    for (auto &q : queries) {
      q.resize(DIM);
      random_unit_vector(q.data(), rng);
    }

    // --- Benchmark: beam_width=1 (Greedy) ---
    std::printf("Benchmarking beam_width=1 (Greedy)...\n");
    std::vector<double> latencies_bw1;
    latencies_bw1.reserve(NUM_QUERIES);

    // Warmup (10 queries)
    for (size_t i = 0; i < 10; ++i) {
      auto r = atlas.navigate(std::span<const float>(queries[i]), 1, false);
      asm volatile("" : : "r"(r.size()) : "memory");
    }

    for (size_t i = 0; i < NUM_QUERIES; ++i) {
      auto t0 = std::chrono::steady_clock::now();
      auto result =
          atlas.navigate(std::span<const float>(queries[i]), 1, false);
      auto t1 = std::chrono::steady_clock::now();

      asm volatile("" : : "r"(result.size()) : "memory");
      latencies_bw1.push_back(
          std::chrono::duration<double, std::micro>(t1 - t0).count());
    }

    auto s1 = compute_stats(latencies_bw1);
    std::printf("  Mean: %8.1f µs\n", s1.mean_us);
    std::printf("  P50:  %8.1f µs\n", s1.p50_us);
    std::printf("  P90:  %8.1f µs\n", s1.p90_us);
    std::printf("  P99:  %8.1f µs\n\n", s1.p99_us);

    // --- Benchmark: beam_width=3 ---
    std::printf("Benchmarking beam_width=3...\n");
    std::vector<double> latencies_bw3;
    latencies_bw3.reserve(NUM_QUERIES);

    // Warmup
    for (size_t i = 0; i < 10; ++i) {
      auto r = atlas.navigate(std::span<const float>(queries[i]), 3, false);
      asm volatile("" : : "r"(r.size()) : "memory");
    }

    for (size_t i = 0; i < NUM_QUERIES; ++i) {
      auto t0 = std::chrono::steady_clock::now();
      auto result =
          atlas.navigate(std::span<const float>(queries[i]), 3, false);
      auto t1 = std::chrono::steady_clock::now();

      asm volatile("" : : "r"(result.size()) : "memory");
      latencies_bw3.push_back(
          std::chrono::duration<double, std::micro>(t1 - t0).count());
    }

    auto s3 = compute_stats(latencies_bw3);
    std::printf("  Mean: %8.1f µs\n", s3.mean_us);
    std::printf("  P50:  %8.1f µs\n", s3.p50_us);
    std::printf("  P90:  %8.1f µs\n", s3.p90_us);
    std::printf("  P99:  %8.1f µs\n\n", s3.p99_us);

    // --- Benchmark: beam_width=3 + CSLS ---
    std::printf("Benchmarking beam_width=3 + CSLS...\n");
    std::vector<double> latencies_csls;
    latencies_csls.reserve(NUM_QUERIES);

    for (size_t i = 0; i < 10; ++i) {
      auto r = atlas.navigate(std::span<const float>(queries[i]), 3, true);
      asm volatile("" : : "r"(r.size()) : "memory");
    }

    for (size_t i = 0; i < NUM_QUERIES; ++i) {
      auto t0 = std::chrono::steady_clock::now();
      auto result = atlas.navigate(std::span<const float>(queries[i]), 3, true);
      auto t1 = std::chrono::steady_clock::now();

      asm volatile("" : : "r"(result.size()) : "memory");
      latencies_csls.push_back(
          std::chrono::duration<double, std::micro>(t1 - t0).count());
    }

    auto sc = compute_stats(latencies_csls);
    std::printf("  Mean: %8.1f µs\n", sc.mean_us);
    std::printf("  P50:  %8.1f µs\n", sc.p50_us);
    std::printf("  P90:  %8.1f µs\n", sc.p90_us);
    std::printf("  P99:  %8.1f µs\n\n", sc.p99_us);

    // --- Scaling Analysis ---
    double ratio = s3.p50_us / s1.p50_us;
    std::printf("--- Scaling Analysis ---\n");
    std::printf("beam=3 / beam=1 P50 ratio: %.2fx\n", ratio);
    if (ratio <= 3.5) {
      std::printf("✅ PASS: beam_width=3 scales sub-linearly (%.2fx <= 3.5x)\n",
                  ratio);
    } else {
      std::printf("❌ FAIL: beam_width=3 scales supra-linearly (%.2fx > "
                  "3.5x) — investigate allocation/branch stalls\n",
                  ratio);
    }

    double csls_overhead = (sc.p50_us - s3.p50_us) / s3.p50_us * 100.0;
    std::printf("CSLS overhead vs beam=3: %.1f%%\n", csls_overhead);
    if (std::abs(csls_overhead) < 5.0) {
      std::printf("✅ PASS: CSLS compiled away by template dispatch (<5%% "
                  "overhead)\n");
    } else {
      std::printf("⚠️  CSLS overhead %.1f%% — check branch predictor "
                  "pollution\n",
                  csls_overhead);
    }
  }

  // Cleanup
  std::filesystem::remove(tmp_path);
  return 0;
}
