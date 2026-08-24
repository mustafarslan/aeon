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

// Generate a query that matches an EXISTING node well (similarity > 0.65,
// the fixture's cold_miss_threshold) WITHOUT being a bit-for-bit exact
// copy -- guardrail #0 (v4-plan.md): an exact-duplicate query guarantees
// cosine==1.0 against a stored centroid, which the SLB cache treats as a
// hit on every call after the first, collapsing "warm query" benchmarks
// into "SLB cache hit" benchmarks that no longer test what they claim to
// (found auditing for this exact self-hit-artifact class, Stage 2 task
// 5 -- the same pattern guardrail #0 already excludes
// BM_AtlasTraversal_Only for). Perturbing an existing node's vector keeps
// it a realistic "matches well" query while avoiding the guaranteed
// cache-hit degenerate case.
std::vector<float> generate_warm_query(size_t dim, int node_seed,
                                       int noise_seed) {
  auto v = generate_vector(dim, node_seed);
  std::mt19937 rng(noise_seed);
  // Empirically calibrated (not guessed): for these 768-dim uniform(-1,1)
  // vectors, N(0, 0.5) additive noise lands cosine similarity around
  // ~0.77 -- comfortably inside the fixture's intended "matches well"
  // band (> cold_miss_threshold=0.65) while staying safely BELOW
  // SLB_HIT_THRESHOLD (0.85, schema.hpp), so this genuinely exercises a
  // real navigate() call every time rather than degenerating into an SLB
  // cache hit after the first iteration. A small uniform(-0.05, 0.05)
  // perturbation (tried first) was NOT enough -- it still landed at
  // ~0.997 similarity, well above the cache-hit threshold.
  std::normal_distribution<float> noise(0.0f, 0.5f);
  for (auto &f : v)
    f += noise(rng);
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
  // Perturbed near-neighbor of node #42, NOT an exact copy -- see
  // generate_warm_query()'s doc comment (guardrail #0 self-hit fix).
  auto query = generate_warm_query(DIM, 42, /*noise_seed=*/1042);

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
  // SAME query as BM_TieredAtlas_WarmQuery (identical node_seed/noise_seed)
  // -- this benchmark exists specifically to be directly comparable to
  // it, so both must see the identical query, not just "similarly warm"
  // ones. See generate_warm_query()'s doc comment for why this isn't an
  // exact node copy (guardrail #0 self-hit fix).
  auto query = generate_warm_query(DIM, 42, /*noise_seed=*/1042);

  for (auto _ : state) {
    auto result = atlas->navigate(query);
    benchmark::DoNotOptimize(result);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK_REGISTER_F(TieredAtlasFixture, BM_RawNavigate_Baseline)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
