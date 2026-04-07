#pragma once

/**
 * @file avq.hpp
 * @brief Anisotropic Vector Quantization (AVQ) — Header-only API
 *
 * ScaNN-inspired anisotropic quantization designed for real-world embeddings
 * (e.g., OpenAI text-embedding-3-large) where variance is heavily skewed
 * across dimensions.
 *
 * The standard symmetric quantization minimizes the raw L2 reconstruction
 * error: L_sym(x, x̃) = ‖x - x̃‖²
 *
 * AVQ decomposes the error into parallel (e_∥) and orthogonal (e_⊥) components
 * relative to the original vector `x`, and penalizes them differently: L_avq(x,
 * x̃) = w_parallel * ‖e_∥‖² + w_orthogonal * ‖e_⊥‖²
 *
 * By aggressively penalizing parallel error (w_parallel > w_orthogonal), we
 * preserve the norm and direction of the vector, which is critical for
 * retaining high recall on Inner Product ranking tasks.
 *
 * This header defines a **stateless scale optimization search** that executes
 * at insert-time, meaning the read-path remains identical (pure INT8 SDOT).
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>

namespace aeon::quant {

/**
 * @brief Configuration for the stateless Anisotropic Quantization search.
 * Enforces deterministic write-time budgets suitable for 60 FPS update loops.
 */
struct AVQConfig {
  float w_parallel = 1.0f;      ///< Weight for direction-distorting error
  float w_orthogonal = 0.2f;    ///< Weight for benign orthogonal error
  uint32_t max_iterations = 10; ///< O(log N) budget for Golden Section Search
  float epsilon = 1e-4f;        ///< Early exit threshold for scale convergence
};

/**
 * @brief Computes the anisotropic loss between a true vector and its quantized
 * reconstruction.
 *
 * L(x, x̃) = w_parallel * ‖e_∥‖² + w_orthogonal * ‖e_⊥‖²
 *
 * @param original     The original FP32 vector
 * @param quantized    The INT8 quantized vector
 * @param scale        The scale factor used for reconstruction (x̃ = quantized *
 * scale)
 * @param config       The AVQ weights
 * @return The scalar anisotropic loss
 */
float compute_anisotropic_loss(std::span<const float> original,
                               std::span<const int8_t> quantized, float scale,
                               const AVQConfig &config) noexcept;

/**
 * @brief Quantizes a FP32 vector to INT8 by minimizing anisotropic loss via
 * Golden Section Search.
 *
 * Rather than blindly setting scale = max(|v|) / 127, this function evaluates
 * the candidate scale bracket [avg/127, max/127] to find the `scale` and
 * quantized vector `q` that mathematically minimizes the anisotropic loss
 * `L_avq(x, q * scale)`.
 *
 * Uses Golden Section Search or Brent's method to guarantee convergence in
 * O(log N) iterations bounded by `config.max_iterations`, preventing
 * high-frequency write stalls.
 *
 * @param input         Source FP32 vector
 * @param output        Destination INT8 vector (must be pre-allocated)
 * @param[out] scale    The optimized scale factor
 * @param config        The deterministic AVQ search budget and weights
 */
void quantize_anisotropic(std::span<const float> input,
                          std::span<int8_t> output, float &scale,
                          const AVQConfig &config = AVQConfig{}) noexcept;

} // namespace aeon::quant
