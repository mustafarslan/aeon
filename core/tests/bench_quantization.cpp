/**
 * @file bench_quantization.cpp
 * @brief Micro-benchmarks for V4.1 Phase 3 — INT8 Quantization Performance.
 *
 * Measures:
 *  1. FP32 cosine similarity vs INT8 dot product throughput
 *  2. Quantization overhead (FP32 → INT8)
 *  3. Memory bandwidth utilization (4× spatial compression benefit)
 */

#include "aeon/math_kernel.hpp"
#include "aeon/quantization.hpp"
#include "aeon/schema.hpp"
#include "aeon/simd_impl.hpp"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace {

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

std::vector<float> make_random_fp32(uint32_t dim, unsigned seed = 42) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(dim);
  for (auto &x : v)
    x = dist(rng);
  return v;
}

template <typename Fn> double bench_ns(Fn &&fn, size_t iterations = 100'000) {
  // Warmup
  for (size_t i = 0; i < 1000; ++i)
    fn();

  auto start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < iterations; ++i) {
    fn();
  }
  auto end = std::chrono::high_resolution_clock::now();

  double total_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return total_ns / static_cast<double>(iterations);
}

void print_header() {
  std::cout
      << "\n"
      << "╔══════════════════════════════════════════════════════════════╗"
      << "\n"
      << "║        Aeon V4.1 — INT8 Quantization Benchmark Suite        ║"
      << "\n"
      << "╚══════════════════════════════════════════════════════════════╝"
      << "\n\n";
}

void print_section(const char *title) {
  std::cout << "── " << title << " ──\n";
}

void print_result(const char *label, double ns, double baseline_ns = 0.0) {
  std::cout << "  " << std::left << std::setw(35) << label << std::right
            << std::setw(10) << std::fixed << std::setprecision(1) << ns
            << " ns";
  if (baseline_ns > 0.0) {
    double speedup = baseline_ns / ns;
    std::cout << "  (" << std::setprecision(2) << speedup << "x)";
  }
  std::cout << "\n";
}

} // namespace

int main() {
  print_header();

  constexpr uint32_t DIM = 384;
  constexpr size_t ITERS = 200'000;

  // ── Prepare data ──
  auto vec_a = make_random_fp32(DIM, 42);
  auto vec_b = make_random_fp32(DIM, 99);

  std::vector<int8_t> q_a(DIM), q_b(DIM);
  float scale_a, scale_b;
  aeon::quant::quantize_symmetric(vec_a, q_a, scale_a);
  aeon::quant::quantize_symmetric(vec_b, q_b, scale_b);

  // ═══════════════════════════════════════════════════════════════════════
  // Benchmark 1: FP32 Cosine Similarity vs INT8 Dot Product
  // ═══════════════════════════════════════════════════════════════════════

  print_section("Dot Product: FP32 vs INT8");

  // FP32 cosine similarity (dispatched: best available)
  volatile float sink_f = 0.0f;
  double fp32_ns = bench_ns(
      [&]() {
        sink_f =
            aeon::math::cosine_similarity(vec_a, std::span<const float>(vec_b));
      },
      ITERS);
  print_result("FP32 cosine_similarity (best)", fp32_ns);

  // INT8 scalar dot
  volatile int32_t sink_i = 0;
  double int8_scalar_ns = bench_ns(
      [&]() { sink_i = aeon::simd::dot_int8_scalar(q_a, q_b, DIM); }, ITERS);
  print_result("INT8 dot_scalar", int8_scalar_ns, fp32_ns);

  // INT8 dispatched (NEON or AVX-512)
  auto best_fn = aeon::simd::get_best_int8_dot_impl();
  double int8_best_ns =
      bench_ns([&]() { sink_i = best_fn(q_a, q_b, DIM); }, ITERS);
  print_result("INT8 dot_best (dispatched)", int8_best_ns, fp32_ns);

  // INT8 dispatched + dequantize (full scoring path)
  volatile float sink_dq = 0.0f;
  double int8_full_ns = bench_ns(
      [&]() {
        int32_t raw = best_fn(q_a, q_b, DIM);
        sink_dq = aeon::quant::dequantize_dot_product(raw, scale_a, scale_b);
      },
      ITERS);
  print_result("INT8 dot+dequantize (full)", int8_full_ns, fp32_ns);

  // ═══════════════════════════════════════════════════════════════════════
  // Benchmark 2: Quantization Overhead
  // ═══════════════════════════════════════════════════════════════════════

  std::cout << "\n";
  print_section("Quantization Overhead");

  std::vector<int8_t> q_out(DIM);
  float scale_out;
  double quant_ns = bench_ns(
      [&]() { aeon::quant::quantize_symmetric(vec_a, q_out, scale_out); },
      ITERS);
  print_result("quantize_symmetric (384-dim)", quant_ns);

  // ═══════════════════════════════════════════════════════════════════════
  // Benchmark 3: Throughput (ops/sec)
  // ═══════════════════════════════════════════════════════════════════════

  std::cout << "\n";
  print_section("Throughput Summary");

  double fp32_ops =
      1e9 / fp32_ns; // ops/sec (each op = 1 cosine sim on 384 dims)
  double int8_ops = 1e9 / int8_best_ns;
  double int8_full_ops = 1e9 / int8_full_ns;

  std::cout << "  FP32:           " << std::fixed << std::setprecision(0)
            << fp32_ops / 1e6 << " Mops/sec\n";
  std::cout << "  INT8 (raw):     " << int8_ops / 1e6 << " Mops/sec\n";
  std::cout << "  INT8 (full):    " << int8_full_ops / 1e6 << " Mops/sec\n";

  // ═══════════════════════════════════════════════════════════════════════
  // Benchmark 4: Memory Footprint
  // ═══════════════════════════════════════════════════════════════════════

  std::cout << "\n";
  print_section("Memory Footprint (per node, dim=384, meta=256B)");

  constexpr size_t fp32_stride = aeon::compute_node_stride(DIM, 256);
  constexpr size_t int8_stride =
      aeon::compute_node_stride(DIM, 256, aeon::QUANT_INT8_SYMMETRIC);

  std::cout << "  FP32 node stride:  " << fp32_stride << " bytes\n";
  std::cout << "  INT8 node stride:  " << int8_stride << " bytes\n";
  std::cout << "  Compression ratio: " << std::setprecision(2)
            << (double)fp32_stride / int8_stride << "x\n";
  std::cout << "  Savings:           "
            << 100.0 * (1.0 - (double)int8_stride / fp32_stride) << "%\n";

  // ═══════════════════════════════════════════════════════════════════════
  // Accuracy Check
  // ═══════════════════════════════════════════════════════════════════════

  std::cout << "\n";
  print_section("Accuracy (dot product approximation)");

  float fp32_dot = 0.0f;
  for (uint32_t i = 0; i < DIM; ++i)
    fp32_dot += vec_a[i] * vec_b[i];

  int32_t int8_raw = best_fn(q_a, q_b, DIM);
  float int8_approx =
      aeon::quant::dequantize_dot_product(int8_raw, scale_a, scale_b);

  float abs_err = std::fabs(int8_approx - fp32_dot);
  float rel_err =
      std::fabs(fp32_dot) > 1e-6f ? abs_err / std::fabs(fp32_dot) : abs_err;

  std::cout << "  FP32 dot:         " << std::setprecision(6) << fp32_dot
            << "\n";
  std::cout << "  INT8 approx:      " << int8_approx << "\n";
  std::cout << "  Absolute error:   " << abs_err << "\n";
  std::cout << "  Relative error:   " << std::setprecision(4)
            << (rel_err * 100.0f) << "%\n";

  std::cout << "\n";

  // Suppress unused volatile warnings
  (void)sink_f;
  (void)sink_i;
  (void)sink_dq;

  return 0;
}
