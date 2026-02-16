// ===========================================================================
// bench_ebr_contention.cpp — Hostile EBR Contention Benchmark
// ---------------------------------------------------------------------------
// Purpose:
//   Prove that cache-line-padded EBR reader slots deliver wait-free reads
//   under maximal cross-core contention. Reports TRUE P50/P99/P99.9/Max
//   latency from raw high-resolution clock measurements.
//
// Methodology:
//   - Spawns std::thread::hardware_concurrency() reader threads
//   - Uses std::barrier to create thundering-herd synchronization:
//     all readers enter the epoch at the EXACT same nanosecond
//   - One dedicated writer thread performs mmap retire + epoch advance
//     in a tight loop to create maximum reclamation pressure
//   - Each reader records per-iteration latency via steady_clock
//   - Post-run: sorts all latency samples, reports percentiles
//
// Compiler Defeat:
//   Uses asm volatile ("" : : "r"(val) : "memory") to force the CPU
//   to actually execute the atomic load inside the epoch guard scope,
//   preventing -O3/-ffast-math from optimizing the read path to a no-op.
//
// Build:
//   cmake --build . --target bench_ebr_contention
//   ./bench_ebr_contention
// ===========================================================================

#include "aeon/epoch.hpp"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <sys/mman.h>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// Compiler defeat: prevent dead-code elimination of the epoch read.
// Forces the value into a register and issues a compiler memory barrier.
// ---------------------------------------------------------------------------
template <typename T> inline void do_not_optimize(T const &val) {
#if defined(__GNUC__) || defined(__clang__)
  asm volatile("" : : "r"(val) : "memory");
#else
  // MSVC fallback: volatile read
  static volatile T sink;
  sink = val;
#endif
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
static constexpr int WARMUP_ITERATIONS = 1'000;
static constexpr int MEASURE_ITERATIONS = 100'000;
static constexpr size_t RETIRE_REGION_SIZE = 4096; // 1 page

int main() {
  using Clock = std::chrono::steady_clock;
  using ns = std::chrono::nanoseconds;

  const unsigned hw_threads = std::thread::hardware_concurrency();
  const unsigned num_readers =
      (hw_threads > 2) ? (hw_threads - 1) : 2; // Reserve 1 core for writer

  std::printf("=== Aeon EBR Hostile Contention Benchmark ===\n"
              "Hardware threads: %u\n"
              "Reader threads:   %u\n"
              "Writer threads:   1\n"
              "Warmup iters:     %d\n"
              "Measure iters:    %d per reader\n"
              "Total samples:    %u\n\n",
              hw_threads, num_readers, WARMUP_ITERATIONS, MEASURE_ITERATIONS,
              num_readers * MEASURE_ITERATIONS);

  aeon::EpochManager mgr;
  std::atomic<bool> stop_writer{false};
  std::atomic<uint64_t> writer_retires{0};

  // Per-reader latency storage (pre-allocated to avoid allocation noise)
  std::vector<std::vector<int64_t>> all_latencies(num_readers);
  for (auto &v : all_latencies) {
    v.reserve(MEASURE_ITERATIONS);
  }

  // Barrier: all readers + main thread synchronize before measurement phase
  std::barrier sync_point(static_cast<std::ptrdiff_t>(num_readers + 1));

  // -----------------------------------------------------------------------
  // Writer thread: retire mmap pages and advance epoch in a tight loop
  // -----------------------------------------------------------------------
  std::thread writer([&]() {
    while (!stop_writer.load(std::memory_order_acquire)) {
      void *ptr = mmap(nullptr, RETIRE_REGION_SIZE, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (ptr != MAP_FAILED) {
        // Touch the page to force physical allocation
        *static_cast<volatile uint8_t *>(ptr) = 0xAA;
        do_not_optimize(*static_cast<volatile uint8_t *>(ptr));

        mgr.retire(ptr, RETIRE_REGION_SIZE);
        mgr.advance_epoch();
        writer_retires.fetch_add(1, std::memory_order_relaxed);
      }
      // Tight loop — no yield, maximum pressure on reclamation path
    }
  });

  // -----------------------------------------------------------------------
  // Reader threads: thundering-herd epoch guard acquisition
  // -----------------------------------------------------------------------
  std::vector<std::thread> readers;
  readers.reserve(num_readers);

  for (unsigned r = 0; r < num_readers; ++r) {
    readers.emplace_back([&, r]() {
      // Phase 1: Warmup (populate thread-local slot cache, warm caches)
      for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        auto guard = mgr.enter_guard();
        uint64_t epoch = mgr.current_epoch();
        do_not_optimize(epoch);
        guard.release();
      }

      // Synchronize: all readers arrive, then main thread releases
      sync_point.arrive_and_wait();

      // Phase 2: Measurement
      auto &latencies = all_latencies[r];
      for (int i = 0; i < MEASURE_ITERATIONS; ++i) {
        auto t0 = Clock::now();

        {
          auto guard = mgr.enter_guard();
          // Force a real read of the epoch inside the guard scope.
          // Without this, -O3 eliminates the entire guard as dead code.
          uint64_t epoch = mgr.current_epoch();
          do_not_optimize(epoch);
        } // guard.release() via RAII

        auto t1 = Clock::now();
        latencies.push_back(std::chrono::duration_cast<ns>(t1 - t0).count());
      }
    });
  }

  // Main thread: arrive at barrier to release all readers simultaneously
  sync_point.arrive_and_wait();

  // Wait for all readers to finish measurement
  for (auto &t : readers) {
    t.join();
  }

  // Stop writer
  stop_writer.store(true, std::memory_order_release);
  writer.join();

  // -----------------------------------------------------------------------
  // Aggregate and report percentiles
  // -----------------------------------------------------------------------
  std::vector<int64_t> combined;
  combined.reserve(static_cast<size_t>(num_readers) * MEASURE_ITERATIONS);
  for (const auto &v : all_latencies) {
    combined.insert(combined.end(), v.begin(), v.end());
  }

  std::sort(combined.begin(), combined.end());

  auto percentile = [&](double p) -> int64_t {
    size_t idx = static_cast<size_t>(p * static_cast<double>(combined.size()));
    if (idx >= combined.size())
      idx = combined.size() - 1;
    return combined[idx];
  };

  double mean = static_cast<double>(std::accumulate(
                    combined.begin(), combined.end(), int64_t{0})) /
                static_cast<double>(combined.size());

  std::printf("--- Results ---\n");
  std::printf("Writer retirements: %llu\n",
              static_cast<unsigned long long>(writer_retires.load()));
  std::printf("Total read samples: %zu\n", combined.size());
  std::printf("\n");
  std::printf("  Mean:   %10.1f ns\n", mean);
  std::printf("  P50:    %10lld ns\n",
              static_cast<long long>(percentile(0.50)));
  std::printf("  P90:    %10lld ns\n",
              static_cast<long long>(percentile(0.90)));
  std::printf("  P99:    %10lld ns\n",
              static_cast<long long>(percentile(0.99)));
  std::printf("  P99.9:  %10lld ns\n",
              static_cast<long long>(percentile(0.999)));
  std::printf("  Max:    %10lld ns\n", static_cast<long long>(combined.back()));
  std::printf("\n");

  // Verdict
  int64_t p99 = percentile(0.99);
  if (p99 < 1000) {
    std::printf("✅ PASS: P99 = %lld ns (< 1µs) — cache-line padding "
                "eliminates false sharing.\n",
                static_cast<long long>(p99));
  } else if (p99 < 10000) {
    std::printf("⚠️  WARN: P99 = %lld ns (1-10µs) — minor contention "
                "detected.\n",
                static_cast<long long>(p99));
  } else {
    std::printf("❌ FAIL: P99 = %lld ns (> 10µs) — false sharing or "
                "lock contention present.\n",
                static_cast<long long>(p99));
  }

  return (p99 < 10000) ? 0 : 1;
}
