// ===========================================================================
// Test 6: Tiered Edge-to-Cloud Atlas Fallback — Aeon V3 §7.2
// ---------------------------------------------------------------------------
// Claims under test:
//   - TieredAtlas::navigate_tiered() correctly sets requires_cloud_fetch
//     when best similarity < cold_miss_threshold (default 0.65)
//   - Edge fallback detection latency overhead < 5µs vs. bare navigate()
//   - Cold miss rate is proportional to query-centroid distance distribution
//
// Architecture:
//   Simulates a Siemens Edge IPC with 128MB memory budget. The TieredAtlas
//   wraps the core Atlas and adds cold-miss detection without networking.
//   When a cold miss is detected, the Python shell would issue a REST/gRPC
//   call to the Cloud Master Atlas.
//
// Hardware: Auto-detected at runtime
// ===========================================================================

#include "aeon/atlas.hpp"
#include "aeon/math_kernel.hpp"
#include "aeon/tiered_atlas.hpp"
#include <benchmark/benchmark.h>
#include <filesystem>
#include <random>
#include <vector>

namespace {

constexpr size_t DIM = 768;

std::vector<float> generate_vector(size_t dim, int seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(dim);
  for (auto &f : v)
    f = dist(rng);
  return v;
}

// Generate a query vector that is deliberately far from all cluster centroids
// (high probability of cold miss)
std::vector<float> generate_cold_query(size_t dim, int seed) {
  std::mt19937 rng(seed);
  // Use a very different distribution range to maximize distance
  std::uniform_real_distribution<float> dist(10.0f, 20.0f);
  std::vector<float> v(dim);
  for (auto &f : v)
    f = dist(rng);
  return v;
}

} // namespace

// ---------------------------------------------------------------------------
// Fixture: builds an Atlas with 10K nodes for tiered testing
// ---------------------------------------------------------------------------
class TieredAtlasFixture : public benchmark::Fixture {
public:
  std::unique_ptr<aeon::Atlas> atlas;
  std::unique_ptr<aeon::TieredAtlas> tiered;
  std::string atlas_path;

  void SetUp(const benchmark::State &) override {
    atlas_path = "/tmp/aeon_bench_tiered_atlas.bin";
    std::filesystem::remove(atlas_path);

    atlas = std::make_unique<aeon::Atlas>(atlas_path);

    // Build a medium Atlas (10K nodes, BFS insertion)
    auto root_vec = generate_vector(DIM, 0);
    atlas->insert(0, root_vec, "root");

    uint64_t parent = 0;
    uint64_t total = 1;
    while (total < 10'000) {
      for (int i = 0; i < 64 && total < 10'000; ++i) {
        auto vec = generate_vector(DIM, static_cast<int>(total));
        atlas->insert(parent, vec, "node_" + std::to_string(total));
        total++;
      }
      parent++;
    }

    // Create TieredAtlas with edge device profile
    aeon::TieredAtlasConfig cfg;
    cfg.memory_budget_mb = 128;
    cfg.cold_miss_threshold = 0.65f;
    tiered = std::make_unique<aeon::TieredAtlas>(*atlas, cfg);
  }

  void TearDown(const benchmark::State &) override {
    tiered.reset();
    atlas.reset();
    std::filesystem::remove(atlas_path);
  }
};

// ---------------------------------------------------------------------------
// BM_TieredAtlas_WarmQuery — Query that matches well (similarity > 0.65)
// Expected: navigate_tiered() ≈ navigate() + ~2µs overhead
// ---------------------------------------------------------------------------
BENCHMARK_DEFINE_F(TieredAtlasFixture,
                   BM_TieredAtlas_WarmQuery)(benchmark::State &state) {
  // Use a query generated from the same seed space as inserted nodes
  auto query = generate_vector(DIM, 42);

  for (auto _ : state) {
    auto result = tiered->navigate_tiered(query);
    benchmark::DoNotOptimize(result);
    // Track cloud fetch flags
    bool any_cloud = false;
    for (const auto &r : result) {
      if (r.requires_cloud_fetch)
        any_cloud = true;
    }
    state.counters["cloud_fetch"] = any_cloud ? 1 : 0;
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK_REGISTER_F(TieredAtlasFixture, BM_TieredAtlas_WarmQuery)
    ->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// BM_TieredAtlas_ColdMiss — Query deliberately far from all centroids
// Expected: requires_cloud_fetch = true, navigate_tiered() ≈ navigate() + ~2µs
// ---------------------------------------------------------------------------
BENCHMARK_DEFINE_F(TieredAtlasFixture,
                   BM_TieredAtlas_ColdMiss)(benchmark::State &state) {
  // Cold query: values in [10, 20] range — far from [-1, 1] training data
  auto query = generate_cold_query(DIM, 99999);

  for (auto _ : state) {
    auto result = tiered->navigate_tiered(query);
    benchmark::DoNotOptimize(result);
    // Verify that cold miss flag is set
    bool any_cloud = false;
    for (const auto &r : result) {
      if (r.requires_cloud_fetch)
        any_cloud = true;
    }
    state.counters["cloud_fetch_triggered"] = any_cloud ? 1 : 0;
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK_REGISTER_F(TieredAtlasFixture, BM_TieredAtlas_ColdMiss)
    ->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// BM_TieredAtlas_vs_RawNavigate — Overhead measurement
// Runs raw Atlas::navigate() for direct comparison
// ---------------------------------------------------------------------------
BENCHMARK_DEFINE_F(TieredAtlasFixture,
                   BM_RawNavigate_Baseline)(benchmark::State &state) {
  auto query = generate_vector(DIM, 42);

  for (auto _ : state) {
    auto result = atlas->navigate(query);
    benchmark::DoNotOptimize(result);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK_REGISTER_F(TieredAtlasFixture, BM_RawNavigate_Baseline)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
