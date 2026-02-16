// Header for SIMD implementation
#pragma once
#include <cmath>
#include <cstdint>
#include <span>

namespace aeon::simd {

// Function pointer type for similarity
using SimilarityFn = float (*)(std::span<const float>, std::span<const float>);

// --- Implementations ---

// 1. Scalar (Portable)
float similarity_scalar(std::span<const float> a, std::span<const float> b);

// 2. AVX2 (x86 only)
float similarity_avx2(std::span<const float> a, std::span<const float> b);
float similarity_avx2_aligned(std::span<const float> a,
                              std::span<const float> b);

// 3. AVX-512 (x86 only)
float similarity_avx512(std::span<const float> a, std::span<const float> b);

// 4. ARM NEON (ARM64 / Apple Silicon)
//    Explicit intrinsics with 4x unroll for Cortex A-series pipeline.
float similarity_neon(std::span<const float> a, std::span<const float> b);

// --- Dispatch ---
// Returns the best implementation for the current CPU.
// ARM64: NEON (native) → Scalar fallback
// x86:   AVX-512 (via SIMDe) → AVX2 → Scalar
SimilarityFn get_best_similarity_impl();

// ═══════════════════════════════════════════════════════════════════════════
// INT8 Dot Product — Phase 3: 4× Spatial Compression
// ═══════════════════════════════════════════════════════════════════════════

/// Function pointer type for INT8 dot product.
/// Returns raw int32 accumulator — caller must dequantize via
/// aeon::quant::dequantize_dot_product().
using Int8DotFn = int32_t (*)(std::span<const int8_t>, std::span<const int8_t>,
                              uint32_t dim);

// --- INT8 Implementations ---

/// 1. Scalar (portable baseline)
int32_t dot_int8_scalar(std::span<const int8_t> a, std::span<const int8_t> b,
                        uint32_t dim);

/// 2. ARM NEON with SDOT (ARMv8.2+ / Apple M-series)
int32_t dot_int8_neon(std::span<const int8_t> a, std::span<const int8_t> b,
                      uint32_t dim);

/// 3. AVX-512 VNNI via SIMDe (x86 with offset trick for s8×s8)
int32_t dot_int8_avx512(std::span<const int8_t> a, std::span<const int8_t> b,
                        uint32_t dim);

/// Returns the best INT8 dot product implementation for the current CPU.
Int8DotFn get_best_int8_dot_impl();

} // namespace aeon::simd
