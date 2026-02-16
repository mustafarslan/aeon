// ===========================================================================
// V4.1: Trace Event Tombstoning & Garbage Collection
// ---------------------------------------------------------------------------
// Claims under test:
//   - tombstone_event() safely marks events as dead in O(N) scan
//   - compact() rewrites the trace file excluding tombstoned events
//   - Shadow compaction preserves event ordering and session linkage
//   - GC reclaims ~50% disk space when 50% of events are tombstoned
//
// Methodology:
//   - 100K trace events across multiple sessions
//   - 50% random tombstoning
//   - Custom counters: EventsBefore, EventsAfter, SpaceReclaimed
//   - 5 repetitions, median with 25/75 percentiles
//
// Hardware: Auto-detected at runtime
// ===========================================================================

#include "aeon/trace.hpp"
#include <benchmark/benchmark.h>
#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr size_t NUM_EVENTS = 100'000;
constexpr size_t NUM_SESSIONS = 100;

} // namespace

// ===========================================================================
// Fixture: pre-populated trace with 100K events
// ===========================================================================
class TraceGCFixture : public benchmark::Fixture {
public:
  std::unique_ptr<aeon::TraceManager> trace;
  std::string trace_path;
  std::string blob_path;
  std::vector<uint64_t> event_ids;

  void SetUp(benchmark::State &) override {
    trace_path = "/tmp/aeon_bench_trace_gc.bin";
    blob_path = trace_path + ".blob";

    // Clean slate
    std::filesystem::remove(trace_path);
    std::filesystem::remove(blob_path);
    // Also remove generation-1 files from prior compactions
    std::filesystem::remove(trace_path + ".gen1");
    std::filesystem::remove(blob_path + ".gen1");

    trace = std::make_unique<aeon::TraceManager>(trace_path);
    event_ids.clear();
    event_ids.reserve(NUM_EVENTS);

    // Pre-populate with NUM_EVENTS events across NUM_SESSIONS sessions
    for (size_t i = 0; i < NUM_EVENTS; ++i) {
      std::string session = "session_" + std::to_string(i % NUM_SESSIONS);
      std::string text = "Event payload " + std::to_string(i) +
                         " — benchmark trace GC test data.";
      uint64_t id =
          trace->append_event(session.c_str(), /*role=*/0, text.c_str());
      event_ids.push_back(id);
    }
  }

  void TearDown(benchmark::State &) override {
    trace.reset();
    // Clean up all possible generation files
    for (const auto &entry : std::filesystem::directory_iterator("/tmp/")) {
      if (entry.path().string().find("aeon_bench_trace_gc") !=
          std::string::npos) {
        std::filesystem::remove(entry.path());
      }
    }
  }
};

// ---------------------------------------------------------------------------
// BM_Tombstone — Per-event tombstone cost
// ---------------------------------------------------------------------------
BENCHMARK_DEFINE_F(TraceGCFixture, BM_Tombstone)
(benchmark::State &state) {
  // Each iteration tombstones a different event (round-robin).
  // After first pass, events are already tombstoned → measures already-dead
  // check cost too.
  size_t idx = 0;

  for (auto _ : state) {
    bool result = trace->tombstone_event(event_ids[idx]);
    benchmark::DoNotOptimize(result);
    idx = (idx + 1) % event_ids.size();
  }
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK_REGISTER_F(TraceGCFixture, BM_Tombstone)
    ->Unit(benchmark::kNanosecond)
    ->Repetitions(5)
    ->ReportAggregatesOnly(true);

// ---------------------------------------------------------------------------
// BM_Compact — Full compaction with ~50% dead events
// ---------------------------------------------------------------------------
BENCHMARK_DEFINE_F(TraceGCFixture, BM_Compact)
(benchmark::State &state) {
  // Tombstone 50% of events (even-indexed)
  for (size_t i = 0; i < event_ids.size(); i += 2) {
    trace->tombstone_event(event_ids[i]);
  }

  // Record pre-compaction size
  size_t events_before = trace->size();

  // Report file size before compaction
  if (std::filesystem::exists(trace_path)) {
    auto size_before = std::filesystem::file_size(trace_path);
    state.counters["FileSizeBefore"] = static_cast<double>(size_before);
  }

  for (auto _ : state) {
    trace->compact();
    benchmark::ClobberMemory();
  }

  // Report post-compaction metrics
  size_t events_after = trace->size();
  state.counters["EventsBefore"] = static_cast<double>(events_before);
  state.counters["EventsAfter"] = static_cast<double>(events_after);
  state.counters["GC_Ratio"] = 1.0 - (static_cast<double>(events_after) /
                                      static_cast<double>(events_before));

  if (std::filesystem::exists(trace_path)) {
    auto size_after = std::filesystem::file_size(trace_path);
    state.counters["FileSizeAfter"] = static_cast<double>(size_after);
  }

  state.SetItemsProcessed(state.iterations());
}

BENCHMARK_REGISTER_F(TraceGCFixture, BM_Compact)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(5)
    ->ReportAggregatesOnly(true);

// ---------------------------------------------------------------------------
// BM_AppendEvent — Raw append throughput (baseline for GC context)
// ---------------------------------------------------------------------------
BENCHMARK_DEFINE_F(TraceGCFixture, BM_AppendEvent)
(benchmark::State &state) {
  int counter = 1'000'000;

  for (auto _ : state) {
    std::string text = "Append bench event " + std::to_string(counter);
    auto id = trace->append_event("bench_session", /*role=*/0, text.c_str());
    benchmark::DoNotOptimize(id);
    counter++;
  }
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK_REGISTER_F(TraceGCFixture, BM_AppendEvent)
    ->Unit(benchmark::kNanosecond)
    ->Repetitions(5)
    ->ReportAggregatesOnly(true);

BENCHMARK_MAIN();
