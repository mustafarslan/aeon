#include "aeon/avq.hpp"
#include "aeon/benchmark_harness.hpp"
#include "aeon/metric_dispatch.hpp"
#include "aeon/quantization.hpp"

#include <benchmark/benchmark.h>
#include <chrono>
#include <iostream>
#include <queue>

using namespace aeon;
using namespace aeon::bench;
using namespace aeon::simd;

static DatasetView db;
static DatasetView queries;
static std::vector<std::vector<uint64_t>> gt_cosine;
static constexpr uint32_t TOP_K = 10;

// Need to populate ground truth before benchmarking
void SetupDatasets() {
  db = DatasetLoader::generate_clustered(10000, 384, 10, 42);
  queries = DatasetLoader::generate_clustered(100, 384, 10, 100);
  gt_cosine =
      GroundTruthCalculator::compute(db, queries, TOP_K, MetricType::Cosine);
}

static void BM_AVQ_Quantization_Time(benchmark::State &state) {
  std::vector<int8_t> q_out(384);
  float scale_out;
  for (auto _ : state) {
    // Just quantize one vector repeatedly to measure throughput of the
    // quantizer itself.
    aeon::quant::quantize_anisotropic(db.get_vector(0), q_out, scale_out);
    benchmark::DoNotOptimize(q_out);
    benchmark::DoNotOptimize(scale_out);
  }
}

static void BM_Symmetric_Quantization_Time(benchmark::State &state) {
  std::vector<int8_t> q_out(384);
  float scale_out;
  for (auto _ : state) {
    aeon::quant::quantize_symmetric(db.get_vector(0), q_out, scale_out);
    benchmark::DoNotOptimize(q_out);
    benchmark::DoNotOptimize(scale_out);
  }
}

// Full evaluation across the database to measure recall vs latency.
static void BM_Recall_AVQ(benchmark::State &state) {
  auto best_kernel =
      MetricDispatcher::resolve(MetricType::Cosine, QuantType::INT8);

  // Pre-quantize the DB with AVQ
  std::vector<std::vector<int8_t>> q_db(db.count, std::vector<int8_t>(db.dim));
  for (uint32_t i = 0; i < db.count; ++i) {
    float dummy_scale;
    aeon::quant::quantize_anisotropic(db.get_vector(i), q_db[i], dummy_scale);
  }
  std::vector<std::vector<int8_t>> q_queries(queries.count,
                                             std::vector<int8_t>(queries.dim));
  for (uint32_t q = 0; q < queries.count; ++q) {
    float dummy_scale;
    aeon::quant::quantize_anisotropic(queries.get_vector(q), q_queries[q],
                                      dummy_scale);
  }

  ParetoRecorder recorder;

  for (auto _ : state) {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<uint64_t>> predicted(queries.count,
                                                 std::vector<uint64_t>(TOP_K));

    for (uint32_t q = 0; q < queries.count; ++q) {
      struct Score {
        int32_t s;
        uint64_t id;
        bool operator<(const Score &o) const { return s > o.s; }
      };
      std::priority_queue<Score> pq;

      for (uint32_t i = 0; i < db.count; ++i) {
        int32_t score = best_kernel->compute_i8(q_queries[q].data(),
                                                q_db[i].data(), db.dim);
        pq.push({score, static_cast<uint64_t>(i)});
        if (pq.size() > TOP_K)
          pq.pop();
      }

      for (int32_t k = TOP_K - 1; k >= 0; --k) {
        predicted[q][k] = pq.top().id;
        pq.pop();
      }
    }

    auto end = std::chrono::high_resolution_clock::now();
    float latency_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count() /
        (float)queries.count;
    float recall = RecallCalculator::recall_at_k(predicted, gt_cosine, TOP_K);

    state.counters["Recall@10"] = recall;
    recorder.record(latency_ns, recall, "AVQ_INT8_Dispatch");
  }

  recorder.write_csv("pareto_front_avq.csv");
}

static void BM_Recall_Symmetric(benchmark::State &state) {
  auto best_kernel =
      MetricDispatcher::resolve(MetricType::Cosine, QuantType::INT8);

  // Pre-quantize the DB with Symmetric
  std::vector<std::vector<int8_t>> q_db(db.count, std::vector<int8_t>(db.dim));
  for (uint32_t i = 0; i < db.count; ++i) {
    float dummy_scale;
    aeon::quant::quantize_symmetric(db.get_vector(i), q_db[i], dummy_scale);
  }
  std::vector<std::vector<int8_t>> q_queries(queries.count,
                                             std::vector<int8_t>(queries.dim));
  for (uint32_t q = 0; q < queries.count; ++q) {
    float dummy_scale;
    aeon::quant::quantize_symmetric(queries.get_vector(q), q_queries[q],
                                    dummy_scale);
  }

  ParetoRecorder recorder;

  for (auto _ : state) {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<uint64_t>> predicted(queries.count,
                                                 std::vector<uint64_t>(TOP_K));

    for (uint32_t q = 0; q < queries.count; ++q) {
      struct Score {
        int32_t s;
        uint64_t id;
        bool operator<(const Score &o) const { return s > o.s; }
      };
      std::priority_queue<Score> pq;

      for (uint32_t i = 0; i < db.count; ++i) {
        int32_t score = best_kernel->compute_i8(q_queries[q].data(),
                                                q_db[i].data(), db.dim);
        pq.push({score, static_cast<uint64_t>(i)});
        if (pq.size() > TOP_K)
          pq.pop();
      }

      for (int32_t k = TOP_K - 1; k >= 0; --k) {
        predicted[q][k] = pq.top().id;
        pq.pop();
      }
    }

    auto end = std::chrono::high_resolution_clock::now();
    float latency_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count() /
        (float)queries.count;
    float recall = RecallCalculator::recall_at_k(predicted, gt_cosine, TOP_K);

    state.counters["Recall@10"] = recall;
    recorder.record(latency_ns, recall, "Sym_INT8_Dispatch");
  }

  recorder.write_csv("pareto_front_sym.csv");
}

int main(int argc, char **argv) {
  SetupDatasets();
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv))
    return 1;
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
  return 0;
}

BENCHMARK(BM_AVQ_Quantization_Time);
BENCHMARK(BM_Symmetric_Quantization_Time);
BENCHMARK(BM_Recall_AVQ)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Recall_Symmetric)->Unit(benchmark::kMillisecond);
