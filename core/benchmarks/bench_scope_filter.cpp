// Scope-filter overhead on navigate() (v4-plan.md Stage 2 / hive-mind verdict doc).
//
// THE CLAIM THIS EXISTS TO CHECK. The hive-mind verdict document estimates the scope gate
// at "about 0.4% of query time", reasoning that a 100K-node query descends depth 3 and
// scores roughly 193 nodes rather than 100,000. That estimate describes the EARLY-REJECT
// mechanism the same document then rejects as incorrect (it deleted 95.8% of results at
// 0.50 selectivity by discarding whole subtrees during descent). It was never re-derived
// for the mechanism that actually shipped, and never measured here at all.
//
// The shipped mechanism enforces scope at RESULT EMISSION and, because descent can no
// longer prune reliably, escalates the beam to MAX_BEAM_WIDTH whenever filtering is active
// (atlas.cpp: `effective_beam_width = scope_filtering ? MAX_BEAM_WIDTH : beam_width`).
// That escalation is a deliberate recall-preservation measure with a documented reason, and
// it is also where nearly all of the cost lives.
//
// Four arms, so the cost decomposes rather than arriving as one number:
//
//   Unfiltered              caller's beam width, no filtering        the baseline
//   UnfilteredBeam16        no filtering, but at the escalated width isolates the escalation
//   FilterPathAllVisible    filtering ON, mask covers every node     isolates the filter test
//   ScopeFiltered           filtering ON, half the tree invisible    adds selectivity
//
// Queries rotate through a pool so the SLB does not serve the measurement from cache,
// matching bench_quantization_efficiency's own fix for that artifact.
#include <benchmark/benchmark.h>
#include <aeon/atlas.hpp>
#include <filesystem>
#include <random>
#include <vector>

namespace {
constexpr size_t DIM = 768, NODES = 100000, POOL = 4096;

struct Fixture : public benchmark::Fixture {
  std::unique_ptr<aeon::Atlas> atlas;
  std::vector<std::vector<float>> pool;
  std::filesystem::path path;

  void SetUp(const benchmark::State &) override {
    if (atlas) return;
    path = std::filesystem::temp_directory_path() / "aeon_bench_scope.bin";
    std::filesystem::remove(path);
    atlas = std::make_unique<aeon::Atlas>(path.string(), DIM);
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    auto rnd = [&] { std::vector<float> v(DIM); for (auto &x : v) x = d(rng); return v; };
    uint64_t root = atlas->insert(0, rnd(), "root");
    std::vector<uint64_t> ids{root};
    for (size_t i = 1; i < NODES; ++i)
      ids.push_back(atlas->insert(ids[i / 8], rnd(), "n"));
    // Half the nodes in scope 0x1, half in 0x2, so filtering has real work to do.
    for (size_t i = 0; i < ids.size(); ++i)
      atlas->set_node_scope(ids[i], (i % 2) ? 0x1 : 0x2);
    for (size_t i = 0; i < POOL; ++i) pool.push_back(rnd());
  }
  void TearDown(const benchmark::State &) override {}
};
}  // namespace

BENCHMARK_DEFINE_F(Fixture, Unfiltered)(benchmark::State &st) {
  size_t i = 0;
  for (auto _ : st) {
    auto r = atlas->navigate(pool[i++ % POOL], 4, false, 0, aeon::ALL_SCOPES_VISIBLE);
    benchmark::DoNotOptimize(r);
  }
}
BENCHMARK_DEFINE_F(Fixture, UnfilteredBeam16)(benchmark::State &st) {
  // Unfiltered, but at the beam width filtering silently escalates to. If this matches the
  // filtered arms, the entire cost is the escalation and not the filter test itself.
  size_t i = 0;
  for (auto _ : st) {
    auto r = atlas->navigate(pool[i++ % POOL], 16, false, 0, aeon::ALL_SCOPES_VISIBLE);
    benchmark::DoNotOptimize(r);
  }
}
BENCHMARK_DEFINE_F(Fixture, FilterPathAllVisible)(benchmark::State &st) {
  // scope_filtering is ON (mask != ALL_SCOPES_VISIBLE) but every node matches, so
  // selectivity is 1.0. Separates the COST OF THE FILTER PATH from the cost of having
  // to search further because half the tree is invisible.
  size_t i = 0;
  for (auto _ : st) {
    auto r = atlas->navigate(pool[i++ % POOL], 4, false, 0, 0x1 | 0x2);
    benchmark::DoNotOptimize(r);
  }
}
BENCHMARK_DEFINE_F(Fixture, ScopeFiltered)(benchmark::State &st) {
  size_t i = 0;
  for (auto _ : st) {
    auto r = atlas->navigate(pool[i++ % POOL], 4, false, 0, 0x1);
    benchmark::DoNotOptimize(r);
  }
}
BENCHMARK_REGISTER_F(Fixture, Unfiltered)->Unit(benchmark::kMicrosecond)
    ->Iterations(POOL)->Repetitions(5)->ReportAggregatesOnly(true);
BENCHMARK_REGISTER_F(Fixture, UnfilteredBeam16)->Unit(benchmark::kMicrosecond)
    ->Iterations(POOL)->Repetitions(5)->ReportAggregatesOnly(true);
BENCHMARK_REGISTER_F(Fixture, FilterPathAllVisible)->Unit(benchmark::kMicrosecond)
    ->Iterations(POOL)->Repetitions(5)->ReportAggregatesOnly(true);
BENCHMARK_REGISTER_F(Fixture, ScopeFiltered)->Unit(benchmark::kMicrosecond)
    ->Iterations(POOL)->Repetitions(5)->ReportAggregatesOnly(true);
BENCHMARK_MAIN();
