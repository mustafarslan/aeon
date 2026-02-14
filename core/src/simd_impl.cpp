#include "aeon/simd_impl.hpp"
#include <iostream>
#include <vector>

// ---------------------------------------------------------------------------
// SIMDe: Portable SIMD — translates AVX-512/AVX2 intrinsics to native ARM
// NEON on Apple Silicon. This enables the same x86 SIMD code paths to compile
// and execute on ARM64 via compile-time instruction translation.
// See: https://github.com/simd-everywhere/simde
// ---------------------------------------------------------------------------
#define SIMDE_ENABLE_NATIVE_ALIASES
#include <simde/x86/avx2.h>
#include <simde/x86/avx512.h>
#include <simde/x86/fma.h>
#include <simde/x86/sse.h>

namespace aeon::simd {

// ----------------------------------------------------------------------------
// 1. Scalar Implementation (baseline, no SIMD)
// ----------------------------------------------------------------------------
float similarity_scalar(std::span<const float> a, std::span<const float> b) {
  float dot = 0.0f;
  float norm_a = 0.0f;
  float norm_b = 0.0f;
  size_t n = a.size();

  for (size_t i = 0; i < n; ++i) {
    float val_a = a[i];
    float val_b = b[i];
    dot += val_a * val_b;
    norm_a += val_a * val_a;
    norm_b += val_b * val_b;
  }

  if (norm_a <= 1e-9f || norm_b <= 1e-9f) [[unlikely]] {
    return 0.0f;
  }
  return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

// ----------------------------------------------------------------------------
// 2. AVX2 Implementation (via SIMDe → NEON on ARM64)
// ----------------------------------------------------------------------------

// Helper for horizontal sum of 256-bit register
static inline float hsum256_ps(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 sum = _mm_add_ps(lo, hi);
  __m128 shuf = _mm_movehdup_ps(sum);
  __m128 sums = _mm_add_ps(sum, shuf);
  shuf = _mm_movehl_ps(shuf, sums);
  sums = _mm_add_ss(sums, shuf);
  return _mm_cvtss_f32(sums);
}

float similarity_avx2(std::span<const float> a, std::span<const float> b) {
  size_t n = a.size();
  size_t i = 0;

  __m256 sum_dot = _mm256_setzero_ps();
  __m256 sum_aa = _mm256_setzero_ps();
  __m256 sum_bb = _mm256_setzero_ps();

  // 4x Unrolling (32 floats per iteration)
  for (; i + 31 < n; i += 32) {
    __m256 va0 = _mm256_loadu_ps(a.data() + i);
    __m256 vb0 = _mm256_loadu_ps(b.data() + i);
    __m256 va1 = _mm256_loadu_ps(a.data() + i + 8);
    __m256 vb1 = _mm256_loadu_ps(b.data() + i + 8);
    __m256 va2 = _mm256_loadu_ps(a.data() + i + 16);
    __m256 vb2 = _mm256_loadu_ps(b.data() + i + 16);
    __m256 va3 = _mm256_loadu_ps(a.data() + i + 24);
    __m256 vb3 = _mm256_loadu_ps(b.data() + i + 24);

    sum_dot = _mm256_fmadd_ps(va0, vb0, sum_dot);
    sum_aa = _mm256_fmadd_ps(va0, va0, sum_aa);
    sum_bb = _mm256_fmadd_ps(vb0, vb0, sum_bb);

    sum_dot = _mm256_fmadd_ps(va1, vb1, sum_dot);
    sum_aa = _mm256_fmadd_ps(va1, va1, sum_aa);
    sum_bb = _mm256_fmadd_ps(vb1, vb1, sum_bb);

    sum_dot = _mm256_fmadd_ps(va2, vb2, sum_dot);
    sum_aa = _mm256_fmadd_ps(va2, va2, sum_aa);
    sum_bb = _mm256_fmadd_ps(vb2, vb2, sum_bb);

    sum_dot = _mm256_fmadd_ps(va3, vb3, sum_dot);
    sum_aa = _mm256_fmadd_ps(va3, va3, sum_aa);
    sum_bb = _mm256_fmadd_ps(vb3, vb3, sum_bb);
  }

  // Remainder loop for 8s
  for (; i + 7 < n; i += 8) {
    __m256 va = _mm256_loadu_ps(a.data() + i);
    __m256 vb = _mm256_loadu_ps(b.data() + i);
    sum_dot = _mm256_fmadd_ps(va, vb, sum_dot);
    sum_aa = _mm256_fmadd_ps(va, va, sum_aa);
    sum_bb = _mm256_fmadd_ps(vb, vb, sum_bb);
  }

  float dot = hsum256_ps(sum_dot);
  float norm_a = hsum256_ps(sum_aa);
  float norm_b = hsum256_ps(sum_bb);

  // Scalar cleanup
  for (; i < n; ++i) {
    dot += a[i] * b[i];
    norm_a += a[i] * a[i];
    norm_b += b[i] * b[i];
  }

  if (norm_a <= 1e-9f || norm_b <= 1e-9f) [[unlikely]]
    return 0.0f;
  return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

// Aligned version delegates to unaligned (SIMDe handles alignment)
float similarity_avx2_aligned(std::span<const float> a,
                              std::span<const float> b) {
  return similarity_avx2(a, b);
}

// ----------------------------------------------------------------------------
// 3. AVX-512 Implementation (via SIMDe → NEON on ARM64)
// ----------------------------------------------------------------------------

// Manual horizontal sum for 512-bit register (SIMDe lacks _mm512_reduce_add_ps)
static inline float hsum512_ps(__m512 v) {
  __m256 lo = _mm512_castps512_ps256(v);
  __m256 hi = _mm256_castpd_ps(
      _mm256_castsi256_pd(_mm256_castps_si256(_mm512_extractf32x8_ps(v, 1))));
  __m256 sum = _mm256_add_ps(lo, hi);
  return hsum256_ps(sum);
}

float similarity_avx512(std::span<const float> a, std::span<const float> b) {
  size_t n = a.size();
  size_t i = 0;

  __m512 sum_dot = _mm512_setzero_ps();
  __m512 sum_aa = _mm512_setzero_ps();
  __m512 sum_bb = _mm512_setzero_ps();

  for (; i + 15 < n; i += 16) {
    __m512 va = _mm512_loadu_ps(a.data() + i);
    __m512 vb = _mm512_loadu_ps(b.data() + i);
    sum_dot = _mm512_fmadd_ps(va, vb, sum_dot);
    sum_aa = _mm512_fmadd_ps(va, va, sum_aa);
    sum_bb = _mm512_fmadd_ps(vb, vb, sum_bb);
  }

  float dot = hsum512_ps(sum_dot);
  float norm_a = hsum512_ps(sum_aa);
  float norm_b = hsum512_ps(sum_bb);

  for (; i < n; ++i) {
    dot += a[i] * b[i];
    norm_a += a[i] * a[i];
    norm_b += b[i] * b[i];
  }

  if (norm_a <= 1e-9f || norm_b <= 1e-9f) [[unlikely]]
    return 0.0f;
  return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

// ----------------------------------------------------------------------------
// Runtime Dispatch
// ----------------------------------------------------------------------------
SimilarityFn get_best_similarity_impl() {
  // With SIMDe, AVX-512 intrinsics are translated to NEON at compile time.
  // On ARM64, we always use the AVX-512 path (best vectorization width).
  // On x86, we could still do runtime CPUID checks, but SIMDe handles
  // the translation transparently.
  return similarity_avx512;
}

} // namespace aeon::simd
