// ===========================================================================
// Test 3: Spatial Index Scalability — Aeon §6.3 (FIXED)
// ---------------------------------------------------------------------------
// Claims under test:
//   - Logarithmic O(log₆₄ N) scaling from 10K to 1M nodes
//   - 10K: ~0.8ms, 100K: ~1.8ms, 1M: ~2.5ms
//   - 40x faster than flat linear scan at 1M
//   - Flat scan at 1M: ~100ms
//
// CRITICAL FIX: Previous version inserted all nodes with parent_id=0,
// creating a degenerate 1-level tree. This version uses BFS level-order
// sequential insertion to build a balanced 64-ary tree, respecting the
// Atlas's contiguous memory constraint (children of the same parent
// must be inserted in a tight loop for sequential file layout).
//
// Hardware: Apple M4 Max (ARM64, NEON via SIMDe)
// ===========================================================================

#include "aeon/atlas.hpp"
#include "aeon/math_kernel.hpp"
#include <benchmark/benchmark.h>
#include <filesystem>
#include <queue>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace {

constexpr size_t DIM = 768;
constexpr int BRANCHING_FACTOR = 64;

std::vector<float> generate_vector(size_t dim, int seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(dim);
  for (auto &f : v)
    f = dist(rng);
  return v;
}

// -------------------------------------------------------------------------
// Build a balanced 64-ary Atlas tree using BFS level-order insertion.
//
// Algorithm (user-specified):
//   1. Insert root node (parent_id=0)
//   2. For each parent in BFS order, insert B=64 children in a tight
//      sequential loop. This guarantees contiguous file layout, which is
//      critical for Atlas's mmap-based read performance.
//   3. Advance to next parent when all 64 children are inserted.
//   4. Stop when total_nodes >= TARGET_N.
//
// Tree shape for N=266,305:
//   Level 0: 1 node (root)
//   Level 1: 64 nodes
//   Level 2: 4,096 nodes
//   Level 3: 262,144 nodes
// -------------------------------------------------------------------------
void build_balanced_atlas(aeon::Atlas &atlas, size_t target_n) {
  constexpr int B = BRANCHING_FACTOR;

  // Insert root
  auto root_vec = generate_vector(DIM, 0);
  atlas.insert(0, root_vec, "root");

  uint64_t next_parent = 0;
  uint64_t total_nodes = 1;

  while (total_nodes < target_n) {
    for (int i = 0; i < B && total_nodes < target_n; ++i) {
      auto vec = generate_vector(DIM, static_cast<int>(total_nodes));
      atlas.insert(next_parent, vec, "node_" + std::to_string(total_nodes));
      total_nodes++;
    }
    next_parent++; // Move to next parent in BFS order
  }
}

} // namespace

// ---------------------------------------------------------------------------
// BM_FlatScan — Brute-force linear scan computing cosine similarity
// ---------------------------------------------------------------------------
static void BM_FlatScan(benchmark::State &state) {
  const size_t N = static_cast<size_t>(state.range(0));

  // Pre-generate all vectors in memory
  std::vector<std::vector<float>> db;
  db.reserve(N);
  for (size_t i = 0; i < N; ++i) {
    db.push_back(generate_vector(DIM, static_cast<int>(i)));
  }

  auto query = generate_vector(DIM, 99999);
  std::span<const float> q{query};

  for (auto _ : state) {
    // Top-5 via min-heap
    std::priority_queue<std::pair<float, size_t>,
                        std::vector<std::pair<float, size_t>>, std::greater<>>
        min_heap;

    for (size_t i = 0; i < N; ++i) {
      float sim = aeon::math::cosine_similarity(q, db[i]);
      if (min_heap.size() < 5) {
        min_heap.push({sim, i});
      } else if (sim > min_heap.top().first) {
        min_heap.pop();
        min_heap.push({sim, i});
      }
    }
    benchmark::DoNotOptimize(min_heap);
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N));
}
BENCHMARK(BM_FlatScan)
    ->Arg(10'000)
    ->Arg(100'000)
    ->Arg(1'000'000)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(3);

// ---------------------------------------------------------------------------
// Atlas fixture for scalability tests — builds balanced 64-ary tree
// ---------------------------------------------------------------------------
class ScalabilityFixture : public benchmark::Fixture {
public:
  std::unique_ptr<aeon::Atlas> atlas;
  std::string atlas_path;
  std::vector<float> query;

  void SetUp(benchmark::State &state) {
    const size_t N = static_cast<size_t>(state.range(0));
    atlas_path = "/tmp/aeon_bench_atlas_balanced_" + std::to_string(N) + ".bin";

    // Always rebuild to guarantee correct tree structure
    std::filesystem::remove(atlas_path);
    atlas = std::make_unique<aeon::Atlas>(atlas_path);

    // Build balanced 64-ary tree using BFS insertion
    build_balanced_atlas(*atlas, N);

    query = generate_vector(DIM, 99999);
  }

  void TearDown(benchmark::State &) {
    atlas.reset();
    std::filesystem::remove(atlas_path);
  }
};

// ---------------------------------------------------------------------------
// BM_AtlasTraversal — Atlas navigate() at various scales
// ---------------------------------------------------------------------------
BENCHMARK_DEFINE_F(ScalabilityFixture,
                   BM_AtlasTraversal)(benchmark::State &state) {
  for (auto _ : state) {
    auto result = atlas->navigate(query);
    benchmark::DoNotOptimize(result);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK_REGISTER_F(ScalabilityFixture, BM_AtlasTraversal)
    ->Arg(10'000)
    ->Arg(100'000)
    ->Arg(1'000'000)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
