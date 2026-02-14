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
 * Uses runtime dynamic dispatch (AVX512 -> AVX2 -> Scalar).
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
 * The underlying SIMD kernel computes dot product and norms in a single fused
 * pass for cache efficiency. Dispatch selects AVX-512 → AVX-2 → scalar at
 * static initialization time.
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
