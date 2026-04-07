#include "aeon/avq.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace aeon::quant {

float compute_anisotropic_loss(std::span<const float> original,
                               std::span<const int8_t> quantized, float scale,
                               const AVQConfig &config) noexcept {
  // 1. Reconstruct x̃
  // e = x - x̃
  // To compute parallel component without full vector allocations:
  // e_parallel = ((e · x) / ‖x‖²) * x
  // ‖e_parallel‖² = (e · x)² / ‖x‖²
  // e_orthogonal = e - e_parallel
  // ‖e_orthogonal‖² = ‖e‖² - ‖e_parallel‖²

  double dot_x_x = 0.0;
  double dot_e_x = 0.0;
  double dot_e_e = 0.0;

  for (size_t i = 0; i < original.size(); ++i) {
    double x_i = original[i];
    double x_tilde_i = static_cast<double>(quantized[i]) * scale;
    double e_i = x_i - x_tilde_i;

    dot_x_x += x_i * x_i;
    dot_e_x += e_i * x_i;
    dot_e_e += e_i * e_i;
  }

  // If original vector is exactly 0, loss is trivially ‖e‖²
  if (dot_x_x <= 1e-12) {
    return static_cast<float>(config.w_orthogonal * dot_e_e);
  }

  double norm_e_parallel_sq = (dot_e_x * dot_e_x) / dot_x_x;
  double norm_e_orthogonal_sq = std::max(0.0, dot_e_e - norm_e_parallel_sq);

  return static_cast<float>(config.w_parallel * norm_e_parallel_sq +
                            config.w_orthogonal * norm_e_orthogonal_sq);
}

// Helper to quantize a vector given a fixed scale and return its anisotropic
// loss
static float evaluate_scale(std::span<const float> input,
                            std::span<int8_t> output, float scale,
                            const AVQConfig &config) noexcept {
  // If scale is arbitrarily small, all points clamp to extremes
  float inv_scale = (scale > 1e-9f) ? (1.0f / scale) : 0.0f;

  for (size_t i = 0; i < input.size(); ++i) {
    float q = std::round(input[i] * inv_scale);
    // Clamp to [-127, 127] to reserve -128 as safety/tombstone if needed
    output[i] = static_cast<int8_t>(std::clamp(q, -127.0f, 127.0f));
  }

  return compute_anisotropic_loss(input, output, scale, config);
}

void quantize_anisotropic(std::span<const float> input,
                          std::span<int8_t> output, float &out_scale,
                          const AVQConfig &config) noexcept {
  // 1. Find vector extents
  float max_abs = 0.0f;
  double sum_abs = 0.0;
  for (float v : input) {
    float abs_v = std::abs(v);
    if (abs_v > max_abs) {
      max_abs = abs_v;
    }
    sum_abs += abs_v;
  }

  if (max_abs <= 1e-12f) {
    std::fill(output.begin(), output.end(), static_cast<int8_t>(0));
    out_scale = 0.0f;
    return;
  }

  // 2. Define search bracket for optimal scale
  // Traditional symmetric quantization uses max_abs / 127.0f
  // The optimal anisotropic scale is often smaller, pushing more values to the
  // clamp limits but preserving inner-product geometry. We bracket between
  // [avg_abs/127, max_abs/127].

  float avg_abs = static_cast<float>(sum_abs / input.size());
  float scale_high = max_abs / 127.0f;
  float scale_low = avg_abs / 127.0f;
  // ensure a viable interval
  if (scale_low < 1e-9f) {
    scale_low = 1e-9f;
  }

  // Golden Section Search Constants
  constexpr float invphi = 0.61803398875f;
  constexpr float invphi2 = 0.38196601125f;

  float a = scale_low;
  float b = scale_high;
  float h = b - a;

  if (h <= config.epsilon) {
    out_scale = scale_high;
    evaluate_scale(input, output, out_scale, config);
    return;
  }

  // Initial points
  float c = a + invphi2 * h;
  float d = a + invphi * h;

  float loss_c = evaluate_scale(input, output, c, config);
  // Scratchpad to hold the best quantized output vector found inside the loop
  // To avoid dynamic allocation, we rely on the caller's output span and
  // re-eval at the end.
  float loss_d = evaluate_scale(input, output, d, config);

  // 3. Search Loop constrained by `max_iterations`
  for (uint32_t iter = 0; iter < config.max_iterations; ++iter) {
    if (std::abs(b - a) < config.epsilon) {
      break;
    }

    if (loss_c < loss_d) {
      b = d;
      d = c;
      loss_d = loss_c;
      h = b - a;
      c = a + invphi2 * h;
      // evaluate_scale inherently clobbers the `output` array with temp INT8s,
      // which is fine since we just need it as scratch space.
      loss_c = evaluate_scale(input, output, c, config);
    } else {
      a = c;
      c = d;
      loss_c = loss_d;
      h = b - a;
      d = a + invphi * h;
      loss_d = evaluate_scale(input, output, d, config);
    }
  }

  // 4. Final selection
  float best_scale = 0.5f * (a + b);
  float final_loss = evaluate_scale(input, output, best_scale, config);

  // 5. Symmetric Fallback Guarantee
  // Since quantization is a step-function, the loss landscape is not strictly
  // unimodal. GSS can get trapped in local minima. We guarantee we never do
  // worse than symmetric.
  float sym_loss = evaluate_scale(input, output, scale_high, config);
  if (sym_loss < final_loss) {
    out_scale = scale_high;
    evaluate_scale(input, output, out_scale, config);
  } else {
    out_scale = best_scale;
    // output is currently clobbered by the sym_loss eval, so we must re-eval
    // best_scale
    evaluate_scale(input, output, out_scale, config);
  }
}

} // namespace aeon::quant
