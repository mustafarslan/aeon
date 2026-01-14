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
  std::vector<float> query;

  void SetUp(const benchmark::State &state) override {
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

    query = generate_vector(DIM, 99999);
  }

  void TearDown(const benchmark::State &state) override {
    atlas.reset();
    std::filesystem::remove(db_path);
  }
};

static void BM_WarmSearch(benchmark::State &state) {
  // Setup (called once per run, but we need the fixture data)
  // Actually, Fixture SetUp is called per-case.
  // We need to manage the resource manually or use the fixture logic.
  // Using simple manual setup inside the function for clarity if Fixture is
  // overkill but Fixed is better for larger setups. Let's rely on the Fixture
  // logic above if we register it correctly.
}

BENCHMARK_F(AtlasFixture, WarmSearch)(benchmark::State &state) {
  std::span<const float> q{query};
  for (auto _ : state) {
    auto results = atlas->navigate(q);
    benchmark::DoNotOptimize(results);
  }
}

BENCHMARK_F(AtlasFixture, ColdSearch)(benchmark::State &state) {
  std::span<const float> q{query};
  for (auto _ : state) {
    state.PauseTiming();
    flush_cache();
    state.ResumeTiming();

    auto results = atlas->navigate(q);
    benchmark::DoNotOptimize(results);
  }
}

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
