#pragma once

#include "aeon/simd_impl.hpp"
#include <cmath>
#include <concepts>
#include <numeric>
#include <vector>

namespace aeon::math {

// --- Concepts ---
template <typename T>
concept FloatingPointRange = std::ranges::contiguous_range<T> &&
                             std::floating_point<std::ranges::range_value_t<T>>;

// --- Public API ---

/**
 * @brief Computes dot product with best available SIMD instruction set.
 *
 * Dispatch is by TARGET ARCHITECTURE at build time (ARM64 -> NEON, x86-64 ->
 * AVX-512-via-SIMDe), not by runtime CPUID probing -- see
 * simd::get_best_similarity_impl() in simd_impl.cpp. There is currently no
 * AVX2-only or scalar x86 fallback selected at runtime; the binary's actual
 * ISA floor is set once, at compile time, by the build's -march flag (see
 * the `release` CMake preset in CMakePresets.json, which intentionally
 * builds at a portable baseline -- x86-64-v3 / Apple M1 -- rather than
 * -march=native, so the shipped binary doesn't assume instructions the
 * deploy host may lack). Opportunistic runtime dispatch to a wider ISA
 * (e.g. use real AVX-512/VNNI only on hosts that have it, portable
 * baseline otherwise) is a real future optimization -- v4-plan.md
 * guardrail #1.2 -- deferred because verifying it correctly requires x86
 * hardware to confirm SIMDe's codegen actually honors a per-function
 * target attribute rather than only the translation unit's baseline
 * -march (unverified either way without that hardware).
 */
inline float dot_product(std::span<const float> a, std::span<const float> b) {
  if (a.size() != b.size()) [[unlikely]] {
    return 0.0f;
  }

  // Thread-safe one-time initialization of best kernel
  static const auto kernel = aeon::simd::get_best_similarity_impl();
  return kernel(a, b);
}

/**
 * @brief Computes Cosine Similarity: (A · B) / (|A| × |B|).
 * Assumes vectors are NOT pre-normalized.
 *
 * The underlying SIMD kernel computes dot product and norms in a single
 * fused pass for cache efficiency. See dot_product() above for the actual
 * dispatch mechanism (compile-time architecture selection, not a runtime
 * AVX-512 -> AVX-2 -> scalar cascade).
 */
inline float cosine_similarity(std::span<const float> a,
                               std::span<const float> b) {
  if (a.size() != b.size()) [[unlikely]] {
    return 0.0f;
  }

  // The kernels in simd_impl return cosine similarity (dot / (normA * normB))
  static const auto kernel = aeon::simd::get_best_similarity_impl();
  return kernel(a, b);
}

/**
 * @brief Cosine similarity optimized for pre-normalized (unit) vectors.
 * For unit vectors, cos(A, B) = dot(A, B). Currently delegates to the
 * general kernel; a dedicated dot-product-only SIMD path can be added
 * as a future optimization.
 */
inline float cosine_similarity_normalized(std::span<const float> a,
                                          std::span<const float> b) {
  return cosine_similarity(a, b);
}

} // namespace aeon::math
