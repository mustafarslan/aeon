/**
 * @file bench_quantization.cpp
 * @brief Micro-benchmarks for V4.1 Phase 3 — INT8 Quantization Performance.
 *
 * Measures:
 *  1. FP32 cosine similarity vs INT8 dot product throughput
 *  2. Quantization overhead (FP32 → INT8)
 *  3. Memory bandwidth utilization (4× spatial compression benefit)
 *
 * Anti-optimization hardening:
 *  - benchmark::DoNotOptimize on ALL function inputs to force physical RAM
 *    fetches (defeats register promotion and constant propagation).
 *  - benchmark::DoNotOptimize on ALL function outputs to prevent Dead Code
 *    Elimination (the compiler cannot prove the result is unused).
 *  - benchmark::ClobberMemory after every iteration to force a full memory
 *    fence, defeating store-to-load forwarding and write coalescing.
 */

#include "aeon/math_kernel.hpp"
#include "aeon/quantization.hpp"
#include "aeon/schema.hpp"
#include "aeon/simd_impl.hpp"
#include <benchmark/benchmark.h>
#include <cmath>
#include <random>
#include <span>
#include <vector>

namespace {

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

constexpr uint32_t DIM = 384;

std::vector<float> make_random_fp32(uint32_t dim, unsigned seed = 42) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(dim);
  for (auto &x : v)
    x = dist(rng);
  return v;
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════════
// Fixture: pre-allocates FP32 and INT8 vectors once
// ═══════════════════════════════════════════════════════════════════════════
class QuantMicroFixture : public benchmark::Fixture {
public:
  std::vector<float> vec_a;
  std::vector<float> vec_b;
  std::vector<int8_t> q_a;
  std::vector<int8_t> q_b;
  float scale_a = 0.0f;
  float scale_b = 0.0f;

  void SetUp(benchmark::State &) override {
    vec_a = make_random_fp32(DIM, 42);
    vec_b = make_random_fp32(DIM, 99);
    q_a.resize(DIM);
    q_b.resize(DIM);
    aeon::quant::quantize_symmetric(vec_a, q_a, scale_a);
    aeon::quant::quantize_symmetric(vec_b, q_b, scale_b);
  }
};

// ---------------------------------------------------------------------------
// BM_FP32_CosineSimilarity — Baseline: dispatched FP32 cosine
// ---------------------------------------------------------------------------
BENCHMARK_DEFINE_F(QuantMicroFixture, BM_FP32_CosineSimilarity)
(benchmark::State &state) {
  std::span<const float> a{vec_a};
  std::span<const float> b{vec_b};

  for (auto _ : state) {
    benchmark::DoNotOptimize(a.data());
    benchmark::DoNotOptimize(b.data());
    float result = aeon::math::cosine_similarity(a, b);
    benchmark::DoNotOptimize(result);
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK_REGISTER_F(QuantMicroFixture, BM_FP32_CosineSimilarity)
    ->Unit(benchmark::kNanosecond)
    ->Repetitions(5)
    ->ReportAggregatesOnly(true);

// ---------------------------------------------------------------------------
// BM_INT8_DotScalar — INT8 scalar fallback path
// ---------------------------------------------------------------------------
BENCHMARK_DEFINE_F(QuantMicroFixture, BM_INT8_DotScalar)
(benchmark::State &state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(q_a.data());
    benchmark::DoNotOptimize(q_b.data());
    int32_t result = aeon::simd::dot_int8_scalar(q_a, q_b, DIM);
    benchmark::DoNotOptimize(result);
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations());
  state.counters["Speedup_vs_FP32"] = 0.0; // Filled by analysis script
}

BENCHMARK_REGISTER_F(QuantMicroFixture, BM_INT8_DotScalar)
    ->Unit(benchmark::kNanosecond)
    ->Repetitions(5)
    ->ReportAggregatesOnly(true);

// ---------------------------------------------------------------------------
// BM_INT8_DotBest — INT8 dispatched (NEON or AVX-512, best available)
// ---------------------------------------------------------------------------
BENCHMARK_DEFINE_F(QuantMicroFixture, BM_INT8_DotBest)
(benchmark::State &state) {
  auto best_fn = aeon::simd::get_best_int8_dot_impl();

  for (auto _ : state) {
    benchmark::DoNotOptimize(q_a.data());
    benchmark::DoNotOptimize(q_b.data());
    int32_t result = best_fn(q_a, q_b, DIM);
    benchmark::DoNotOptimize(result);
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK_REGISTER_F(QuantMicroFixture, BM_INT8_DotBest)
    ->Unit(benchmark::kNanosecond)
    ->Repetitions(5)
    ->ReportAggregatesOnly(true);

// ---------------------------------------------------------------------------
// BM_INT8_DotDequantize — Full scoring path: dot + dequantize
// ---------------------------------------------------------------------------
BENCHMARK_DEFINE_F(QuantMicroFixture, BM_INT8_DotDequantize)
(benchmark::State &state) {
  auto best_fn = aeon::simd::get_best_int8_dot_impl();

  for (auto _ : state) {
    benchmark::DoNotOptimize(q_a.data());
    benchmark::DoNotOptimize(q_b.data());
    int32_t raw = best_fn(q_a, q_b, DIM);
    float result = aeon::quant::dequantize_dot_product(raw, scale_a, scale_b);
    benchmark::DoNotOptimize(result);
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK_REGISTER_F(QuantMicroFixture, BM_INT8_DotDequantize)
    ->Unit(benchmark::kNanosecond)
    ->Repetitions(5)
    ->ReportAggregatesOnly(true);

// ---------------------------------------------------------------------------
// BM_QuantizeSymmetric — Quantization overhead (FP32 → INT8)
// ---------------------------------------------------------------------------
BENCHMARK_DEFINE_F(QuantMicroFixture, BM_QuantizeSymmetric)
(benchmark::State &state) {
  std::vector<int8_t> q_out(DIM);
  float scale_out;

  for (auto _ : state) {
    benchmark::DoNotOptimize(vec_a.data());
    aeon::quant::quantize_symmetric(vec_a, q_out, scale_out);
    benchmark::DoNotOptimize(q_out.data());
    benchmark::DoNotOptimize(scale_out);
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK_REGISTER_F(QuantMicroFixture, BM_QuantizeSymmetric)
    ->Unit(benchmark::kNanosecond)
    ->Repetitions(5)
    ->ReportAggregatesOnly(true);

// ---------------------------------------------------------------------------
// Memory Footprint & Accuracy — reported as custom counters
// ---------------------------------------------------------------------------
static void BM_FootprintAndAccuracy(benchmark::State &state) {
  auto vec_a = make_random_fp32(DIM, 42);
  auto vec_b = make_random_fp32(DIM, 99);

  std::vector<int8_t> q_a(DIM), q_b(DIM);
  float scale_a, scale_b;
  aeon::quant::quantize_symmetric(vec_a, q_a, scale_a);
  aeon::quant::quantize_symmetric(vec_b, q_b, scale_b);

  auto best_fn = aeon::simd::get_best_int8_dot_impl();

  for (auto _ : state) {
    benchmark::DoNotOptimize(q_a.data());
    benchmark::DoNotOptimize(q_b.data());
    int32_t raw = best_fn(q_a, q_b, DIM);
    float approx = aeon::quant::dequantize_dot_product(raw, scale_a, scale_b);
    benchmark::DoNotOptimize(approx);
    benchmark::ClobberMemory();
  }

  // Report stride comparison as counters
  constexpr size_t fp32_stride = aeon::compute_node_stride(DIM, 256);
  constexpr size_t int8_stride =
      aeon::compute_node_stride(DIM, 256, aeon::QUANT_INT8_SYMMETRIC);

  state.counters["FP32_Stride"] = static_cast<double>(fp32_stride);
  state.counters["INT8_Stride"] = static_cast<double>(int8_stride);
  state.counters["CompressionRatio"] =
      static_cast<double>(fp32_stride) / static_cast<double>(int8_stride);

  // Accuracy check
  float fp32_dot = 0.0f;
  for (uint32_t i = 0; i < DIM; ++i)
    fp32_dot += vec_a[i] * vec_b[i];
  int32_t int8_raw = best_fn(q_a, q_b, DIM);
  float int8_approx =
      aeon::quant::dequantize_dot_product(int8_raw, scale_a, scale_b);
  float rel_err = std::fabs(fp32_dot) > 1e-6f
                      ? std::fabs(int8_approx - fp32_dot) / std::fabs(fp32_dot)
                      : 0.0f;

  state.counters["RelativeError_pct"] = rel_err * 100.0;
}

BENCHMARK(BM_FootprintAndAccuracy)
    ->Unit(benchmark::kNanosecond)
    ->Repetitions(3)
    ->ReportAggregatesOnly(true);

BENCHMARK_MAIN();
