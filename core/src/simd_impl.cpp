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
// 4. ARM NEON Implementation (native intrinsics)
//    Explicit 4x loop unrolling for ARM Cortex A-series pipeline throughput.
//    Uses vld1q_f32/vmlaq_f32 for direct NEON execution without SIMDe
//    translation overhead. This is the preferred path on Apple Silicon
//    and ARM Cortex A7x/A7xx series.
// ----------------------------------------------------------------------------

#if defined(__aarch64__) || defined(__ARM_NEON) || defined(_M_ARM64)
#include <arm_neon.h>

// Horizontal sum of a float32x4_t register
static inline float vhsumq_f32(float32x4_t v) {
  // pairwise add: [a+b, c+d]
  float32x2_t sum = vadd_f32(vget_low_f32(v), vget_high_f32(v));
  // final pairwise: [a+b+c+d]
  sum = vpadd_f32(sum, sum);
  return vget_lane_f32(sum, 0);
}

float similarity_neon(std::span<const float> a, std::span<const float> b) {
  size_t n = a.size();
  size_t i = 0;

  // 4 accumulators per quantity for 4x unrolling (16 floats/iter)
  float32x4_t sum_dot0 = vdupq_n_f32(0.0f);
  float32x4_t sum_dot1 = vdupq_n_f32(0.0f);
  float32x4_t sum_dot2 = vdupq_n_f32(0.0f);
  float32x4_t sum_dot3 = vdupq_n_f32(0.0f);

  float32x4_t sum_aa0 = vdupq_n_f32(0.0f);
  float32x4_t sum_aa1 = vdupq_n_f32(0.0f);
  float32x4_t sum_aa2 = vdupq_n_f32(0.0f);
  float32x4_t sum_aa3 = vdupq_n_f32(0.0f);

  float32x4_t sum_bb0 = vdupq_n_f32(0.0f);
  float32x4_t sum_bb1 = vdupq_n_f32(0.0f);
  float32x4_t sum_bb2 = vdupq_n_f32(0.0f);
  float32x4_t sum_bb3 = vdupq_n_f32(0.0f);

  const float *pa = a.data();
  const float *pb = b.data();

  // 4x unrolled: 16 floats per iteration
  for (; i + 15 < n; i += 16) {
    float32x4_t va0 = vld1q_f32(pa + i);
    float32x4_t vb0 = vld1q_f32(pb + i);
    float32x4_t va1 = vld1q_f32(pa + i + 4);
    float32x4_t vb1 = vld1q_f32(pb + i + 4);
    float32x4_t va2 = vld1q_f32(pa + i + 8);
    float32x4_t vb2 = vld1q_f32(pb + i + 8);
    float32x4_t va3 = vld1q_f32(pa + i + 12);
    float32x4_t vb3 = vld1q_f32(pb + i + 12);

    sum_dot0 = vmlaq_f32(sum_dot0, va0, vb0);
    sum_dot1 = vmlaq_f32(sum_dot1, va1, vb1);
    sum_dot2 = vmlaq_f32(sum_dot2, va2, vb2);
    sum_dot3 = vmlaq_f32(sum_dot3, va3, vb3);

    sum_aa0 = vmlaq_f32(sum_aa0, va0, va0);
    sum_aa1 = vmlaq_f32(sum_aa1, va1, va1);
    sum_aa2 = vmlaq_f32(sum_aa2, va2, va2);
    sum_aa3 = vmlaq_f32(sum_aa3, va3, va3);

    sum_bb0 = vmlaq_f32(sum_bb0, vb0, vb0);
    sum_bb1 = vmlaq_f32(sum_bb1, vb1, vb1);
    sum_bb2 = vmlaq_f32(sum_bb2, vb2, vb2);
    sum_bb3 = vmlaq_f32(sum_bb3, vb3, vb3);
  }

  // Reduce 4 accumulators → 1
  float32x4_t total_dot =
      vaddq_f32(vaddq_f32(sum_dot0, sum_dot1), vaddq_f32(sum_dot2, sum_dot3));
  float32x4_t total_aa =
      vaddq_f32(vaddq_f32(sum_aa0, sum_aa1), vaddq_f32(sum_aa2, sum_aa3));
  float32x4_t total_bb =
      vaddq_f32(vaddq_f32(sum_bb0, sum_bb1), vaddq_f32(sum_bb2, sum_bb3));

  float dot = vhsumq_f32(total_dot);
  float norm_a = vhsumq_f32(total_aa);
  float norm_b = vhsumq_f32(total_bb);

  // Scalar cleanup for remainder
  for (; i < n; ++i) {
    dot += pa[i] * pb[i];
    norm_a += pa[i] * pa[i];
    norm_b += pb[i] * pb[i];
  }

  if (norm_a <= 1e-9f || norm_b <= 1e-9f) [[unlikely]]
    return 0.0f;
  return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

#else
// Stub for non-ARM builds — delegates to scalar
float similarity_neon(std::span<const float> a, std::span<const float> b) {
  return similarity_scalar(a, b);
}
#endif

// ----------------------------------------------------------------------------
// Runtime Dispatch
// ----------------------------------------------------------------------------
SimilarityFn get_best_similarity_impl() {
#if defined(__aarch64__) || defined(__ARM_NEON) || defined(_M_ARM64)
  // On ARM64, use native NEON intrinsics (bypasses SIMDe translation layer).
  // This is ~10-15% faster than SIMDe-translated AVX-512 on Apple M-series
  // due to eliminated instruction mapping overhead.
  return similarity_neon;
#else
  // On x86, use SIMDe-translated AVX-512 (best vectorization width).
  // SIMDe handles compile-time instruction translation transparently.
  return similarity_avx512;
#endif
}

// ═══════════════════════════════════════════════════════════════════════════
// INT8 Dot Product — Phase 3: 4× Spatial Compression
// ═══════════════════════════════════════════════════════════════════════════

// ----------------------------------------------------------------------------
// INT8 Scalar Baseline (portable — works on any platform)
// ----------------------------------------------------------------------------
int32_t dot_int8_scalar(std::span<const int8_t> a, std::span<const int8_t> b,
                        uint32_t dim) {
  int32_t acc = 0;
  uint32_t n =
      std::min(dim, static_cast<uint32_t>(std::min(a.size(), b.size())));

  // 4× manual unrolling for ILP
  uint32_t i = 0;
  for (; i + 3 < n; i += 4) {
    acc += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
    acc += static_cast<int32_t>(a[i + 1]) * static_cast<int32_t>(b[i + 1]);
    acc += static_cast<int32_t>(a[i + 2]) * static_cast<int32_t>(b[i + 2]);
    acc += static_cast<int32_t>(a[i + 3]) * static_cast<int32_t>(b[i + 3]);
  }
  for (; i < n; ++i) {
    acc += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
  }
  return acc;
}

// ----------------------------------------------------------------------------
// INT8 NEON — uses vdotq_s32 (SDOT) on ARMv8.2+ (Apple M-series)
// ----------------------------------------------------------------------------
#if defined(__aarch64__) || defined(__ARM_NEON) || defined(_M_ARM64)
int32_t dot_int8_neon(std::span<const int8_t> a, std::span<const int8_t> b,
                      uint32_t dim) {
  uint32_t n =
      std::min(dim, static_cast<uint32_t>(std::min(a.size(), b.size())));
  uint32_t i = 0;

  // SDOT accumulates 4 × (s8 * s8) → s32 per lane, 4 lanes = 16 elements
  int32x4_t acc0 = vdupq_n_s32(0);
  int32x4_t acc1 = vdupq_n_s32(0);

  const int8_t *pa = a.data();
  const int8_t *pb = b.data();

  // Process 32 elements per iteration (2× unroll of 16-element vdotq)
  for (; i + 31 < n; i += 32) {
    int8x16_t va0 = vld1q_s8(pa + i);
    int8x16_t vb0 = vld1q_s8(pb + i);
    int8x16_t va1 = vld1q_s8(pa + i + 16);
    int8x16_t vb1 = vld1q_s8(pb + i + 16);

    acc0 = vdotq_s32(acc0, va0, vb0);
    acc1 = vdotq_s32(acc1, va1, vb1);
  }

  // Process 16 elements
  for (; i + 15 < n; i += 16) {
    int8x16_t va = vld1q_s8(pa + i);
    int8x16_t vb = vld1q_s8(pb + i);
    acc0 = vdotq_s32(acc0, va, vb);
  }

  // Horizontal sum
  int32x4_t total = vaddq_s32(acc0, acc1);
  int32_t result = vaddvq_s32(total);

  // Scalar cleanup
  for (; i < n; ++i) {
    result += static_cast<int32_t>(pa[i]) * static_cast<int32_t>(pb[i]);
  }

  return result;
}
#else
int32_t dot_int8_neon(std::span<const int8_t> a, std::span<const int8_t> b,
                      uint32_t dim) {
  // Fallback to scalar on non-ARM platforms
  return dot_int8_scalar(a, b, dim);
}
#endif

// ----------------------------------------------------------------------------
// INT8 AVX-512 VNNI via SIMDe — uses _mm512_dpbusd_epi32 with offset trick
//
// VNNI dpbusd expects unsigned × signed (u8 * s8 → s32).
// Since both vectors are signed int8, we apply the offset trick:
//   1. Convert query a[i] to unsigned: a'[i] = a[i] + 128 (flip MSB)
//   2. Compute dot(a', b) via dpbusd
//   3. Correct: dot(a, b) = dot(a', b) - 128 * sum(b)
// ----------------------------------------------------------------------------
int32_t dot_int8_avx512(std::span<const int8_t> a, std::span<const int8_t> b,
                        uint32_t dim) {
  uint32_t n =
      std::min(dim, static_cast<uint32_t>(std::min(a.size(), b.size())));
  uint32_t i = 0;

  // Accumulators for dpbusd result and sum(b) correction
  __m512i acc_dot = _mm512_setzero_si512();
  __m512i acc_sum_b = _mm512_setzero_si512();

  // Broadcast 128 as bytes for the offset trick
  const __m512i offset_128 = _mm512_set1_epi8(static_cast<char>(128u));
  // For vpsadbw, we need all-zero to compute sum of unsigned bytes
  const __m512i zero = _mm512_setzero_si512();

  for (; i + 63 < n; i += 64) {
    // Load 64 × int8 from each vector
    __m512i va =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a.data() + i));
    __m512i vb =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(b.data() + i));

    // Step 1: Convert query to unsigned by XOR with 0x80 (equivalent to +128
    // for signed)
    __m512i va_unsigned = _mm512_xor_si512(va, offset_128);

    // Step 2: dpbusd — unsigned(a') × signed(b) → int32 accumulator
    acc_dot = _mm512_dpbusd_epi32(acc_dot, va_unsigned, vb);

    // Step 3: Accumulate sum(b) for correction
    // Convert b to unsigned for vpsadbw: b_unsigned = b ^ 0x80
    __m512i vb_unsigned = _mm512_xor_si512(vb, offset_128);
    // vpsadbw computes sum of absolute differences against zero = sum of
    // unsigned bytes Each 8-byte group → one uint64 in the result (8 groups per
    // 512-bit register)
    __m512i sad = _mm512_sad_epu8(vb_unsigned, zero);
    acc_sum_b = _mm512_add_epi64(acc_sum_b, sad);
  }

  // ── Horizontal reduction of dot product accumulator (16 × int32 → 1) ──
  // Pure register extraction chain — no memory stores.
  // Step 1: 512 → two 256-bit halves, add them
  __m256i dot_lo = _mm512_castsi512_si256(acc_dot);
  __m256i dot_hi = _mm512_extracti32x8_epi32(acc_dot, 1);
  __m256i dot_256 = _mm256_add_epi32(dot_lo, dot_hi);
  // Step 2: 256 → two 128-bit halves, add them
  __m128i dot_128_lo = _mm256_castsi256_si128(dot_256);
  __m128i dot_128_hi = _mm256_extracti128_si256(dot_256, 1);
  __m128i dot_128 = _mm_add_epi32(dot_128_lo, dot_128_hi);
  // Step 3: horizontal add within 128-bit register (4 → 2 → 1)
  dot_128 = _mm_hadd_epi32(dot_128, dot_128);
  dot_128 = _mm_hadd_epi32(dot_128, dot_128);
  int32_t dot_result = _mm_cvtsi128_si32(dot_128);

  // ── Horizontal reduction of sum(b) accumulator (8 × int64 → 1) ──
  // Pure register extraction chain — no memory stores.
  // vpsadbw gave us sums of unsigned bytes. We need signed sum:
  // b_unsigned[j] = b[j] ^ 0x80 = b[j] + 128 (for signed b[j])
  // So sum(b_unsigned) = sum(b) + n_processed * 128
  // Thus sum(b) = sum(b_unsigned) - n_processed * 128
  // Step 1: 512 → two 256-bit halves, add them
  __m256i sb_lo = _mm512_castsi512_si256(acc_sum_b);
  __m256i sb_hi = _mm512_extracti64x4_epi64(acc_sum_b, 1);
  __m256i sb_256 = _mm256_add_epi64(sb_lo, sb_hi);
  // Step 2: 256 → two 128-bit halves, add them
  __m128i sb_128_lo = _mm256_castsi256_si128(sb_256);
  __m128i sb_128_hi = _mm256_extracti128_si256(sb_256, 1);
  __m128i sb_128 = _mm_add_epi64(sb_128_lo, sb_128_hi);
  // Step 3: extract two i64 lanes and add
  int64_t sum_b_unsigned =
      _mm_cvtsi128_si64(sb_128) + _mm_extract_epi64(sb_128, 1);
  int64_t sum_b = sum_b_unsigned - static_cast<int64_t>(i) * 128;

  // Correction: dot(a, b) = dot(a', b) - 128 * sum(b)
  int64_t corrected = static_cast<int64_t>(dot_result) - 128 * sum_b;

  // Scalar cleanup for remainder
  for (; i < n; ++i) {
    corrected += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
  }

  return static_cast<int32_t>(corrected);
}

// ----------------------------------------------------------------------------
// INT8 Runtime Dispatch
// ----------------------------------------------------------------------------
Int8DotFn get_best_int8_dot_impl() {
#if defined(__aarch64__) || defined(__ARM_NEON) || defined(_M_ARM64)
  return dot_int8_neon;
#else
  return dot_int8_avx512;
#endif
}

} // namespace aeon::simd
