#include "aeon/math_kernel.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

int main() {
  constexpr size_t DIM = 768;
  constexpr size_t ITERATIONS = 1'000'000;

  // 1. Setup Data
  std::vector<float> a(DIM);
  std::vector<float> b(DIM);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (auto &v : a)
    v = dist(rng);
  for (auto &v : b)
    v = dist(rng);

  std::span<const float> span_a{a};
  std::span<const float> span_b{b};

  // Warmup
  volatile float res = 0.0f;
  for (int i = 0; i < 1000; ++i) {
    res = aeon::math::cosine_similarity(span_a, span_b);
  }

  // 2. Benchmark
  std::cout << "Benchmarking cosine_similarity (768-dim) over " << ITERATIONS
            << " iterations..." << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < ITERATIONS; ++i) {
    // We accumulate result to "res" to ensure compiler doesn't optimize out
    res += aeon::math::cosine_similarity(span_a, span_b);
    // Slightly perturb b's pointer? No, keep it simple latency test.
    // To prevent loop unrolling optimizations that are *too* smart:
    // usually this function call is complex enough.
  }

  auto end = std::chrono::high_resolution_clock::now();

  // prevent optimization of result
  if (res == 12345.0f)
    std::cout << "Magic" << std::endl;

  // 3. Report
  auto elapsed_us =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count();
  auto elapsed_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  double avg_ns = static_cast<double>(elapsed_ns) / ITERATIONS;
  double avg_us = static_cast<double>(elapsed_us) / ITERATIONS;

  std::cout << "Total Time: " << elapsed_us << " us" << std::endl;
  std::cout << "Average Latency: " << avg_ns << " ns (" << avg_us << " us)"
            << std::endl;

  // Check Build Info
  std::cout << "SIMD Arch: "
#if defined(__AVX512F__)
            << "AVX-512"
#elif defined(__AVX2__)
            << "AVX2"
#elif defined(__ARM_NEON)
            << "NEON"
#else
            << "Scalar"
#endif
            << std::endl;

  if (avg_us > 1.0) {
    std::cerr << "FAIL: Performance constraint violated (> 1.0 us)"
              << std::endl;
    return 1;
  }

  std::cout << "PASS: Performance optimized (< 1.0 us)" << std::endl;
  return 0;
}
