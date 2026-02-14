// ===========================================================================
// Test 1: Math Kernel Throughput — Aeon §6.1
// ---------------------------------------------------------------------------
// Claims under test:
//   - AVX-512 cosine similarity: ~50ns per 768-dim comparison
//   - Throughput: 20M ops/sec
//   - 20x faster than scalar C++
//   - ~2000x faster than Python/NumPy (measured separately)
//
// Hardware: Apple M4 Max (ARM64, NEON via SIMDe translation)
// Methodology: 5 repetitions, median with 25/75 percentiles
// ===========================================================================

#include "aeon/math_kernel.hpp"
#include "aeon/simd_impl.hpp"
#include <benchmark/benchmark.h>
#include <random>
#include <span>
#include <vector>

namespace {

// Deterministic vector generation (matching paper methodology)
std::vector<float> generate_vector(size_t dim, int seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(dim);
  for (auto &f : v)
    f = dist(rng);
  return v;
}

// Manual 4x unrolled scalar (no SIMD intrinsics, tests compiler autovec)
float similarity_scalar_4x(std::span<const float> a, std::span<const float> b) {
  float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
  size_t n = a.size();
  size_t i = 0;

  // 4x unrolled loop
  for (; i + 3 < n; i += 4) {
    dot += a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] +
           a[i + 3] * b[i + 3];
    norm_a += a[i] * a[i] + a[i + 1] * a[i + 1] + a[i + 2] * a[i + 2] +
              a[i + 3] * a[i + 3];
    norm_b += b[i] * b[i] + b[i + 1] * b[i + 1] + b[i + 2] * b[i + 2] +
              b[i + 3] * b[i + 3];
  }
  for (; i < n; ++i) {
    dot += a[i] * b[i];
    norm_a += a[i] * a[i];
    norm_b += b[i] * b[i];
  }

  if (norm_a <= 1e-9f || norm_b <= 1e-9f)
    return 0.0f;
  return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

} // namespace

// ---------------------------------------------------------------------------
// BM_Scalar — Baseline scalar loop (no SIMD, no unrolling)
// ---------------------------------------------------------------------------
static void BM_Scalar(benchmark::State &state) {
  constexpr size_t DIM = 768;
  auto a = generate_vector(DIM, 42);
  auto b = generate_vector(DIM, 43);
  std::span<const float> sa{a}, sb{b};

  for (auto _ : state) {
    float sim = aeon::simd::similarity_scalar(sa, sb);
    benchmark::DoNotOptimize(sim);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Scalar)->Unit(benchmark::kNanosecond);

// ---------------------------------------------------------------------------
// BM_Scalar_4xUnrolled — Manual 4-element unrolled scalar
// ---------------------------------------------------------------------------
static void BM_Scalar_4xUnrolled(benchmark::State &state) {
  constexpr size_t DIM = 768;
  auto a = generate_vector(DIM, 42);
  auto b = generate_vector(DIM, 43);
  std::span<const float> sa{a}, sb{b};

  for (auto _ : state) {
    float sim = similarity_scalar_4x(sa, sb);
    benchmark::DoNotOptimize(sim);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Scalar_4xUnrolled)->Unit(benchmark::kNanosecond);

// ---------------------------------------------------------------------------
// BM_SIMDe_AVX2 — AVX2 via SIMDe → NEON translation
// ---------------------------------------------------------------------------
static void BM_SIMDe_AVX2(benchmark::State &state) {
  constexpr size_t DIM = 768;
  auto a = generate_vector(DIM, 42);
  auto b = generate_vector(DIM, 43);
  std::span<const float> sa{a}, sb{b};

  for (auto _ : state) {
    float sim = aeon::simd::similarity_avx2(sa, sb);
    benchmark::DoNotOptimize(sim);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_SIMDe_AVX2)->Unit(benchmark::kNanosecond);

// ---------------------------------------------------------------------------
// BM_SIMDe_AVX512 — AVX-512 via SIMDe → NEON (paper's claimed kernel)
// ---------------------------------------------------------------------------
static void BM_SIMDe_AVX512(benchmark::State &state) {
  constexpr size_t DIM = 768;
  auto a = generate_vector(DIM, 42);
  auto b = generate_vector(DIM, 43);
  std::span<const float> sa{a}, sb{b};

  for (auto _ : state) {
    float sim = aeon::simd::similarity_avx512(sa, sb);
    benchmark::DoNotOptimize(sim);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_SIMDe_AVX512)->Unit(benchmark::kNanosecond);

// ---------------------------------------------------------------------------
// BM_RuntimeDispatched — Production path (get_best_similarity_impl)
// ---------------------------------------------------------------------------
static void BM_RuntimeDispatched(benchmark::State &state) {
  constexpr size_t DIM = 768;
  auto a = generate_vector(DIM, 42);
  auto b = generate_vector(DIM, 43);
  std::span<const float> sa{a}, sb{b};

  for (auto _ : state) {
    float sim = aeon::math::cosine_similarity(sa, sb);
    benchmark::DoNotOptimize(sim);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_RuntimeDispatched)->Unit(benchmark::kNanosecond);

BENCHMARK_MAIN();
