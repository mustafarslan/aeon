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

} // namespace aeon::simd
