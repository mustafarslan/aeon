#pragma once

/**
 * @file quantization.hpp
 * @brief INT8 Symmetric Scalar Quantization — Header-only utilities.
 *
 * V4.1 Phase 3: Provides FP32 ↔ INT8 conversion for 4× spatial compression
 * of embedding vectors stored in the Atlas mmap file.
 *
 * Quantization scheme (symmetric, zero_point = 0):
 *   scale   = max(|v|) / 127.0f
 *   q[i]    = clamp(round(v[i] / scale), -127, 127)
 *   v'[i]   ≈ q[i] * scale
 *
 * Dot product dequantization:
 *   dot(A, B) ≈ raw_int32_dot * scale_A * scale_B
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>

namespace aeon::quant {

/**
 * @brief Quantize a FP32 vector to INT8 using symmetric quantization.
 *
 * @param input   Source FP32 vector (dim elements)
 * @param output  Destination INT8 vector (must be pre-allocated, dim elements)
 * @param[out] scale  Computed scale factor: max(|v|) / 127.0f
 *
 * Safety invariants:
 *   - If max(|v|) == 0 → scale = 1.0f, output all zeros (no divide-by-zero)
 *   - Clamps to [-127, +127] (NOT -128 — symmetric range for zero_point = 0)
 *   - Uses std::round() for unbiased rounding
 */
inline void quantize_symmetric(std::span<const float> input,
                               std::span<int8_t> output,
                               float &scale) noexcept {
  // Find maximum absolute value
  float max_abs = 0.0f;
  for (float v : input) {
    float abs_v = std::fabs(v);
    if (abs_v > max_abs)
      max_abs = abs_v;
  }

  // Divide-by-zero safety: if vector is all zeros, scale = 1.0
  if (max_abs < 1e-10f) {
    scale = 1.0f;
    for (size_t i = 0; i < output.size(); ++i) {
      output[i] = 0;
    }
    return;
  }

  scale = max_abs / 127.0f;
  float inv_scale = 1.0f / scale;

  for (size_t i = 0; i < input.size() && i < output.size(); ++i) {
    // Quantize: round(x / scale), clamp to [-127, 127]
    float q = std::round(input[i] * inv_scale);
    q = std::max(-127.0f, std::min(127.0f, q));
    output[i] = static_cast<int8_t>(q);
  }
}

/**
 * @brief Dequantize a raw INT32 dot product back to FP32.
 *
 * Given: dot_int8(Q_quantized, D_quantized) = raw_dot (int32)
 * The true FP32 approximation is: raw_dot * query_scale * node_scale
 *
 * This MUST be called BEFORE subtracting the FP32 hub_penalty.
 *
 * @param raw_dot       Raw INT32 dot product from SIMD kernel
 * @param query_scale   Scale used to quantize the query vector
 * @param node_scale    Scale stored in NodeHeader::quant_scale
 * @return Approximate FP32 dot product
 */
inline float dequantize_dot_product(int32_t raw_dot, float query_scale,
                                    float node_scale) noexcept {
  return static_cast<float>(raw_dot) * query_scale * node_scale;
}

} // namespace aeon::quant
