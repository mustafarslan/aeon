#include <aeon/atlas.hpp>
#include <aeon/math_kernel.hpp>
#include <benchmark/benchmark.h>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

namespace {

// Helper to generate random vector
std::vector<float> generate_vector(size_t dim, int seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(dim);
  for (auto &f : v)
    f = dist(rng);
  return v;
}

// Function to flush CPU caches by reading/writing a large buffer
void flush_cache() {
  constexpr size_t CACHE_SIZE =
      64 * 1024 * 1024; // 64 MB (Larger than typical L3)
  static std::vector<char> dummy(CACHE_SIZE);

  // Read/Write to force cache lines to be replaced
  for (size_t i = 0; i < CACHE_SIZE; i += 64) {
    dummy[i] += 1;
  }
  benchmark::DoNotOptimize(dummy.data());
}

// WarmSearch/ColdSearch previously reused one static `query` across every
// measured iteration -- the self-hit-artifact class audited across this
// benchmark suite (Stage 2 task 5, v4-plan.md): navigate()'s SLB cache
// stores an entry after the FIRST call, so every later bit-for-bit-
// identical query (cosine==1.0 against its own cached entry) hits that
// cache instead of performing a real traversal. This is especially
// misleading for ColdSearch, whose entire point is to flush CPU caches
// (flush_cache()) and measure a genuine cold traversal -- an unflushable
// SLB hit from the second iteration onward silently defeats that
// methodology regardless of CPU cache state. (ConversationalDrift, just
// below, deliberately reuses near-duplicate queries to measure cache-HIT
// speed by its own design/comment -- that one is correct as-is.)
constexpr size_t QUERY_POOL_SIZE = 4096;
constexpr int QUERY_SEED_BASE = 5'000'000;

std::vector<std::vector<float>> generate_query_pool(size_t dim, size_t count,
                                                     int seed_base) {
  std::vector<std::vector<float>> pool;
  pool.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    pool.push_back(generate_vector(dim, seed_base + static_cast<int>(i)));
  }
  return pool;
}

} // namespace

// ----------------------------------------------------------------------------
// 1. Math Kernel Benchmark
// ----------------------------------------------------------------------------
static void BM_MathKernel(benchmark::State &state) {
  constexpr size_t DIM = 768;
  auto a = generate_vector(DIM, 42);
  auto b = generate_vector(DIM, 43);

  std::span<const float> span_a{a};
  std::span<const float> span_b{b};

  for (auto _ : state) {
    float similarity = aeon::math::cosine_similarity(span_a, span_b);
    benchmark::DoNotOptimize(similarity);
  }
}
BENCHMARK(BM_MathKernel);

// ----------------------------------------------------------------------------
// 2. Atlas Search Benchmarks (Integration)
// ----------------------------------------------------------------------------
class AtlasFixture : public benchmark::Fixture {
public:
  std::unique_ptr<aeon::Atlas> atlas;
  std::string db_path = "bench_atlas.bin";
  std::vector<std::vector<float>> query_pool;

  void SetUp([[maybe_unused]] const benchmark::State &state) override {
    // Create a temporary Atlas file
    // Note: For a serious benchmark, we'd want a large pre-built DB.
    // Here we build a small one on the fly to measure pure latency overhead.

    std::filesystem::remove(db_path);
    atlas = std::make_unique<aeon::Atlas>(db_path);

    constexpr int NUM_NODES = 10000;
    constexpr size_t DIM = 768;

    // Insert nodes (Greedy build not optimized here, just populating)
    for (int i = 0; i < NUM_NODES; ++i) {
      auto vec = generate_vector(DIM, i);
      atlas->insert(0, vec, "bench_node");
    }

    query_pool = generate_query_pool(DIM, QUERY_POOL_SIZE, QUERY_SEED_BASE);
  }

  void TearDown([[maybe_unused]] const benchmark::State &state) override {
    atlas.reset();
    std::filesystem::remove(db_path);
  }
};

// BENCHMARK_DEFINE_F/BENCHMARK_REGISTER_F (not the BENCHMARK_F shorthand)
// -- needed so ->Iterations(QUERY_POOL_SIZE) below can actually be chained
// on; BENCHMARK_F's registration happens inside the macro itself and
// can't be chained.
BENCHMARK_DEFINE_F(AtlasFixture, WarmSearch)(benchmark::State &state) {
  size_t idx = 0;
  for (auto _ : state) {
    auto results = atlas->navigate(std::span<const float>{query_pool[idx % QUERY_POOL_SIZE]});
    benchmark::DoNotOptimize(results);
    ++idx;
  }
  if (idx > QUERY_POOL_SIZE) {
    state.SkipWithError(
        "query pool exhausted (iterations > QUERY_POOL_SIZE) -- "
        "results may include self-hit-artifact cache hits on repeated "
        "queries; increase QUERY_POOL_SIZE");
  }
}
BENCHMARK_REGISTER_F(AtlasFixture, WarmSearch)->Iterations(QUERY_POOL_SIZE);

BENCHMARK_DEFINE_F(AtlasFixture, ColdSearch)(benchmark::State &state) {
  size_t idx = 0;
  for (auto _ : state) {
    state.PauseTiming();
    flush_cache();
    state.ResumeTiming();

    auto results = atlas->navigate(std::span<const float>{query_pool[idx % QUERY_POOL_SIZE]});
    benchmark::DoNotOptimize(results);
    ++idx;
  }
  if (idx > QUERY_POOL_SIZE) {
    state.SkipWithError(
        "query pool exhausted (iterations > QUERY_POOL_SIZE) -- "
        "results may include self-hit-artifact cache hits on repeated "
        "queries; increase QUERY_POOL_SIZE");
  }
}
BENCHMARK_REGISTER_F(AtlasFixture, ColdSearch)->Iterations(QUERY_POOL_SIZE);

BENCHMARK_F(AtlasFixture, ConversationalDrift)(benchmark::State &state) {
  // Generate 10 related queries (simulating conversation about a topic)
  std::vector<std::vector<float>> queries;
  // Pick a base vector. To ensure we have results, we pick one of the inserted
  // vectors if possible, or just random. Random in high-dim space is far from
  // others. But for this bench, we just want to test cache HIT speed.
  // So we run one query to warm it up.
  auto base = generate_vector(768, 123);

  // Create variations
  for (int i = 0; i < 10; ++i) {
    auto q = base;
    for (int j = 0; j < 768; ++j) {
      // Very slight drift
      q[j] += (static_cast<float>(i) * 0.001f);
    }
    queries.push_back(q);
  }

  int idx = 0;
  for (auto _ : state) {
    auto &q = queries[idx++ % queries.size()];
    auto results = atlas->navigate(std::span<const float>(q));
    benchmark::DoNotOptimize(results);
  }
}

BENCHMARK_MAIN();
