// ===========================================================================
// V4.1 Phase 3: Quantization Efficiency — INT8 vs FP32
// ---------------------------------------------------------------------------
// Claims under test:
//   - INT8 symmetric quantization provides 4× spatial compression
//   - Navigate latency ≤ FP32 (decompression cost amortized by cache wins)
//   - Insert throughput: quantize-on-write cost < 5% overhead
//
// Methodology:
//   - Balanced 64-ary Atlas trees at production dim=768
//   - Deterministic vector generation (seeded mt19937)
//   - Custom counters: BytesPerNode, FileSizeBytes, CompressionRatio
//   - 5 repetitions, median with 25/75 percentiles
//
// Hardware: Auto-detected at runtime
// ===========================================================================

#include "aeon/atlas.hpp"
#include "aeon/schema.hpp"
#include <benchmark/benchmark.h>
#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr size_t DIM = 768;
constexpr int BRANCHING_FACTOR = 64;

// ── Deterministic vector generation (matching paper methodology) ──
std::vector<float> generate_vector(size_t dim, int seed) {
  std::mt19937 rng(static_cast<unsigned>(seed));
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(dim);
  for (auto &f : v)
    f = dist(rng);
  return v;
}

// ── Build a balanced 64-ary Atlas tree via BFS level-order insertion ──
void build_balanced_atlas(aeon::Atlas &atlas, size_t target_n) {
  auto root_vec = generate_vector(DIM, 0);
  atlas.insert(0, root_vec, "root");

  uint64_t next_parent = 0;
  uint64_t total_nodes = 1;

  while (total_nodes < target_n) {
    for (int i = 0; i < BRANCHING_FACTOR && total_nodes < target_n; ++i) {
      auto vec = generate_vector(DIM, static_cast<int>(total_nodes));
      atlas.insert(next_parent, vec, "node_" + std::to_string(total_nodes));
      total_nodes++;
    }
    next_parent++;
  }
}

} // namespace

// ===========================================================================
// Fixture: parameterized by quantization type
// ===========================================================================
class QuantizationFixture : public benchmark::Fixture {
public:
  std::unique_ptr<aeon::Atlas> atlas;
  std::string atlas_path;
  std::vector<float> query;
  uint32_t quant_type = aeon::QUANT_FP32;

  void SetUp(benchmark::State &state) override {
    const size_t N = static_cast<size_t>(state.range(0));
    quant_type = static_cast<uint32_t>(state.range(1));

    const char *quant_tag =
        (quant_type == aeon::QUANT_INT8_SYMMETRIC) ? "int8" : "fp32";
    atlas_path = "/tmp/aeon_bench_quant_" + std::string(quant_tag) + "_" +
                 std::to_string(N) + ".bin";

    std::filesystem::remove(atlas_path);

    // Create Atlas with explicit quantization type, WAL disabled for
    // pure compute measurement (no I/O noise).
    aeon::AtlasOptions opts{};
    opts.dim = static_cast<uint32_t>(DIM);
    opts.quantization_type = quant_type;
    opts.enable_wal = false;

    atlas = std::make_unique<aeon::Atlas>(atlas_path, opts);
    build_balanced_atlas(*atlas, N);

    query = generate_vector(DIM, 99999);

    // Report file size as a custom counter
    auto file_size = std::filesystem::file_size(atlas_path);
    state.counters["FileSizeBytes"] = static_cast<double>(file_size);
    state.counters["BytesPerNode"] =
        static_cast<double>(file_size) / static_cast<double>(N);
  }

  void TearDown(benchmark::State &) override {
    atlas.reset();
    std::filesystem::remove(atlas_path);
  }
};

// ---------------------------------------------------------------------------
// BM_Navigate — Search latency at production scale
// ---------------------------------------------------------------------------
BENCHMARK_DEFINE_F(QuantizationFixture, BM_Navigate)
(benchmark::State &state) {
  std::span<const float> q{query};

  for (auto _ : state) {
    auto results = atlas->navigate(q);
    benchmark::DoNotOptimize(results);
  }
  state.SetItemsProcessed(state.iterations());
}

// Register: {node_count, quant_type}
// FP32 variants
BENCHMARK_REGISTER_F(QuantizationFixture, BM_Navigate)
    ->Args({10'000, aeon::QUANT_FP32})
    ->Args({100'000, aeon::QUANT_FP32})
    ->Unit(benchmark::kMicrosecond)
    ->Repetitions(5)
    ->ReportAggregatesOnly(true);

// INT8 variants
BENCHMARK_REGISTER_F(QuantizationFixture, BM_Navigate)
    ->Args({10'000, aeon::QUANT_INT8_SYMMETRIC})
    ->Args({100'000, aeon::QUANT_INT8_SYMMETRIC})
    ->Unit(benchmark::kMicrosecond)
    ->Repetitions(5)
    ->ReportAggregatesOnly(true);

// ---------------------------------------------------------------------------
// BM_Insert — Write throughput with quantize-on-write overhead
// ---------------------------------------------------------------------------
BENCHMARK_DEFINE_F(QuantizationFixture, BM_Insert)
(benchmark::State &state) {
  // Measure marginal insert cost after the tree is built.
  // Each iteration inserts into the already-populated tree.
  uint64_t parent_id = 0;
  int seed_counter = 500'000;

  for (auto _ : state) {
    auto vec = generate_vector(DIM, seed_counter++);
    auto id = atlas->insert(parent_id, vec, "bench_insert");
    benchmark::DoNotOptimize(id);
    // Rotate parent to spread across the tree
    parent_id = id % atlas->size();
  }
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK_REGISTER_F(QuantizationFixture, BM_Insert)
    ->Args({10'000, aeon::QUANT_FP32})
    ->Args({10'000, aeon::QUANT_INT8_SYMMETRIC})
    ->Unit(benchmark::kMicrosecond)
    ->Repetitions(5)
    ->ReportAggregatesOnly(true);

BENCHMARK_MAIN();
