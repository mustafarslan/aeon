// ===========================================================================
// V4.1: WAL Overhead — Durability Cost Measurement
// ---------------------------------------------------------------------------
// Claims under test:
//   - Write-Ahead Logging (WAL) provides crash-safe durability
//   - WAL overhead on insert throughput is measurable but bounded
//   - AtlasOptions::enable_wal cleanly toggles WAL at construction time
//
// Methodology:
//   - Fixed 10K-node pre-built Atlas (BFS balanced 64-ary tree)
//   - Marginal insert throughput measured with WAL enabled vs disabled
//   - Custom counters: InsertOpsPerSec
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
constexpr size_t PREBUILT_NODES = 10'000;

std::vector<float> generate_vector(size_t dim, int seed) {
  std::mt19937 rng(static_cast<unsigned>(seed));
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(dim);
  for (auto &f : v)
    f = dist(rng);
  return v;
}

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
// Fixture: parameterized by enable_wal (range(0) = 0 or 1)
// ===========================================================================
class WalFixture : public benchmark::Fixture {
public:
  std::unique_ptr<aeon::Atlas> atlas;
  std::string atlas_path;
  std::string wal_path;

  void SetUp(benchmark::State &state) override {
    bool wal_enabled = (state.range(0) != 0);
    const char *wal_tag = wal_enabled ? "wal_on" : "wal_off";
    atlas_path = "/tmp/aeon_bench_wal_" + std::string(wal_tag) + ".bin";
    wal_path = atlas_path + ".wal";

    std::filesystem::remove(atlas_path);
    std::filesystem::remove(wal_path);

    aeon::AtlasOptions opts{};
    opts.dim = static_cast<uint32_t>(DIM);
    opts.quantization_type = aeon::QUANT_FP32;
    opts.enable_wal = wal_enabled;

    atlas = std::make_unique<aeon::Atlas>(atlas_path, opts);
    build_balanced_atlas(*atlas, PREBUILT_NODES);
  }

  void TearDown(benchmark::State &) override {
    atlas.reset();
    std::filesystem::remove(atlas_path);
    std::filesystem::remove(wal_path);
  }
};

// ---------------------------------------------------------------------------
// BM_Insert — Marginal insert throughput with/without WAL
// ---------------------------------------------------------------------------
BENCHMARK_DEFINE_F(WalFixture, BM_Insert)(benchmark::State &state) {
  uint64_t parent_id = 0;
  int seed_counter = 500'000;

  for (auto _ : state) {
    auto vec = generate_vector(DIM, seed_counter++);
    auto id = atlas->insert(parent_id, vec, "wal_bench");
    benchmark::DoNotOptimize(id);
    parent_id = id % atlas->size();
  }
  state.SetItemsProcessed(state.iterations());
}

// WAL disabled (baseline — pure mmap speed)
BENCHMARK_REGISTER_F(WalFixture, BM_Insert)
    ->Arg(0)
    ->Unit(benchmark::kMicrosecond)
    ->Repetitions(5)
    ->ReportAggregatesOnly(true);

// WAL enabled (durability cost)
BENCHMARK_REGISTER_F(WalFixture, BM_Insert)
    ->Arg(1)
    ->Unit(benchmark::kMicrosecond)
    ->Repetitions(5)
    ->ReportAggregatesOnly(true);

BENCHMARK_MAIN();
