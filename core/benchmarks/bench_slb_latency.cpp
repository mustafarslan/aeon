// ===========================================================================
// Test 2: SLB Latency — Aeon §6.2
// ---------------------------------------------------------------------------
// Claims under test:
//   - Cache hit latency:  0.05ms (50µs)
//   - Cache miss latency: 2.50ms (full Atlas traversal)
//   - 85% hit rate under conversational workload
//   - 64-entry brute-force scan: 3.2µs (derived: 64 × 50ns)
//
// Hardware: Apple M4 Max (ARM64, NEON via SIMDe)
// Methodology: 5 repetitions, median with 25/75 percentiles
// ===========================================================================

#include "aeon/atlas.hpp"
#include "aeon/math_kernel.hpp"
#include "aeon/slb.hpp"
#include <benchmark/benchmark.h>
#include <cstring>
#include <filesystem>
#include <random>
#include <vector>

namespace {

constexpr size_t DIM = 768;
constexpr float SLB_THRESHOLD = 0.85f;

std::vector<float> generate_vector(size_t dim, int seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(dim);
  for (auto &f : v)
    f = dist(rng);
  return v;
}

// Normalize a vector to unit length
void normalize(std::vector<float> &v) {
  float norm = 0.0f;
  for (auto f : v)
    norm += f * f;
  norm = std::sqrt(norm);
  if (norm > 1e-9f)
    for (auto &f : v)
      f /= norm;
}

} // namespace

// ---------------------------------------------------------------------------
// BM_SLB_RawScan_64 — Pure scan cost: 64 cosine similarities
// Validates derived claim: 64 × 50ns = 3.2µs
// ---------------------------------------------------------------------------
static void BM_SLB_RawScan_64(benchmark::State &state) {
  aeon::SemanticCache slb;

  // Populate SLB with 64 entries using correct API: insert(node_id, centroid)
  for (int i = 0; i < 64; ++i) {
    auto vec = generate_vector(DIM, 100 + i);
    slb.insert(static_cast<uint64_t>(i), vec);
  }

  auto query = generate_vector(DIM, 999);
  std::span<const float> q{query};

  for (auto _ : state) {
    auto result = slb.find_nearest(q, SLB_THRESHOLD);
    benchmark::DoNotOptimize(result);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_SLB_RawScan_64)->Unit(benchmark::kNanosecond);

// ---------------------------------------------------------------------------
// BM_SLB_CacheHit — Hit path: query matching a cached entry (>0.85 sim)
// Expected: ~50µs
// ---------------------------------------------------------------------------
static void BM_SLB_CacheHit(benchmark::State &state) {
  aeon::SemanticCache slb;

  // Insert 63 random entries
  for (int i = 0; i < 63; ++i) {
    auto vec = generate_vector(DIM, 100 + i);
    slb.insert(static_cast<uint64_t>(i), vec);
  }

  // Insert a known "target" as the 64th entry
  auto target = generate_vector(DIM, 500);
  normalize(target);
  slb.insert(63, target);

  // Create a query that is very similar to the target (small perturbation)
  auto query = target; // Start from exact copy
  std::mt19937 rng(777);
  std::uniform_real_distribution<float> noise(-0.01f, 0.01f);
  for (auto &f : query)
    f += noise(rng);
  normalize(query);

  std::span<const float> q{query};

  for (auto _ : state) {
    auto result = slb.find_nearest(q, SLB_THRESHOLD);
    benchmark::DoNotOptimize(result);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_SLB_CacheHit)->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// Atlas fixture for cache miss tests
// ---------------------------------------------------------------------------
class AtlasFixture : public benchmark::Fixture {
public:
  std::unique_ptr<aeon::Atlas> atlas;
  std::string atlas_path;
  aeon::SemanticCache slb;

  void SetUp(const benchmark::State &) override {
    atlas_path = "/tmp/aeon_bench_atlas_slb.bin";
    std::filesystem::remove(atlas_path);

    atlas = std::make_unique<aeon::Atlas>(atlas_path);

    // Insert 10K nodes as children of root (parent_id=0)
    for (int i = 0; i < 10'000; ++i) {
      auto vec = generate_vector(DIM, 2000 + i);
      atlas->insert(0, vec, "bench_node_" + std::to_string(i));
    }

    // Populate SLB (but query will miss)
    for (int i = 0; i < 64; ++i) {
      auto vec = generate_vector(DIM, 3000 + i);
      slb.insert(static_cast<uint64_t>(i), vec);
    }
  }

  void TearDown(const benchmark::State &) override {
    atlas.reset();
    std::filesystem::remove(atlas_path);
  }
};

// ---------------------------------------------------------------------------
// BM_SLB_CacheMiss_WarmAtlas — Miss path with warm Atlas (OS-cached pages)
// Expected: ~0.8ms (10K node tree)
// ---------------------------------------------------------------------------
BENCHMARK_DEFINE_F(AtlasFixture,
                   BM_SLB_CacheMiss_WarmAtlas)(benchmark::State &state) {
  auto query = generate_vector(DIM, 9999);

  for (auto _ : state) {
    // SLB miss → fallback to Atlas
    auto slb_result = slb.find_nearest(query, SLB_THRESHOLD);
    if (!slb_result.has_value()) {
      auto atlas_result = atlas->navigate(query);
      benchmark::DoNotOptimize(atlas_result);
    }
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK_REGISTER_F(AtlasFixture, BM_SLB_CacheMiss_WarmAtlas)
    ->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// BM_AtlasTraversal_Only — Pure Atlas navigate (no SLB overhead)
// Expected: ~0.8ms for 10K nodes
// ---------------------------------------------------------------------------
BENCHMARK_DEFINE_F(AtlasFixture,
                   BM_AtlasTraversal_Only)(benchmark::State &state) {
  auto query = generate_vector(DIM, 9999);

  for (auto _ : state) {
    auto result = atlas->navigate(query);
    benchmark::DoNotOptimize(result);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK_REGISTER_F(AtlasFixture, BM_AtlasTraversal_Only)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
