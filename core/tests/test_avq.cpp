/**
 * @file test_avq.cpp
 * @brief Unit tests for Anisotropic Vector Quantization (AVQ).
 *
 * Tests:
 *  1. Golden Section Search correctly minimizes anisotropic distortion compared
 * to symmetric.
 *  2. AVQ handles near-zero or perfectly isotropic vectors smoothly.
 *  3. Bounded iteration parameters strictly cap execution branches.
 */

#include "aeon/avq.hpp"
#include "aeon/quantization.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <vector>

namespace aeon::test {

class AVQTest : public ::testing::Test {
protected:
  std::vector<float> make_anisotropic_vector(uint32_t dim) {
    std::vector<float> v(dim, 0.05f);
    // Create strong outlier dimensions
    v[0] = 8.0f;
    v[1] = -7.5f;
    if (dim > 2)
      v[2] = 2.0f;
    return v;
  }

  std::vector<float> make_zero_vector(uint32_t dim) {
    return std::vector<float>(dim, 0.0f);
  }
};

TEST_F(AVQTest, OutperformsSymmetricOnAnisotropicData) {
  auto input = make_anisotropic_vector(128);

  // Baseline: Symmetric
  std::vector<int8_t> sym_out(128);
  float sym_scale = 0.0f;
  quant::quantize_symmetric(input, std::span<int8_t>(sym_out.data(), 128),
                            sym_scale);

  // Proposed: AVQ
  std::vector<int8_t> avq_out(128);
  float avq_scale = 0.0f;
  quant::AVQConfig config;
  config.w_parallel = 1.0f;
  config.w_orthogonal = 0.2f;
  config.max_iterations = 20;
  quant::quantize_anisotropic(input, std::span<int8_t>(avq_out.data(), 128),
                              avq_scale, config);

  // Compute Loss
  float loss_sym = quant::compute_anisotropic_loss(
      input, std::span<const int8_t>(sym_out.data(), 128), sym_scale, config);
  float loss_avq = quant::compute_anisotropic_loss(
      input, std::span<const int8_t>(avq_out.data(), 128), avq_scale, config);

  // AVQ must not be worse than symmetric fallback
  EXPECT_LE(loss_avq, loss_sym + 1e-5f);
}

TEST_F(AVQTest, ZeroVectorHandledGracefully) {
  auto input = make_zero_vector(128);
  std::vector<int8_t> out(128, 1);
  float scale = 99.0f;

  quant::AVQConfig config;
  quant::quantize_anisotropic(input, std::span<int8_t>(out.data(), 128), scale,
                              config);

  EXPECT_EQ(scale, 0.0f);
  for (int8_t v : out) {
    EXPECT_EQ(v, 0);
  }
}

TEST_F(AVQTest, EarlyExitGracefullyHonorsEpsilon) {
  auto input = make_anisotropic_vector(64);
  std::vector<int8_t> out(64);
  float scale = 0.0f;

  quant::AVQConfig config;
  // Set epsilon massively high, making it break immediately
  config.epsilon = 1000.0f;
  config.max_iterations = 100;

  quant::quantize_anisotropic(input, std::span<int8_t>(out.data(), 64), scale,
                              config);

  // With epsilon breaking immediately, the scale should just be the upper
  // bracket (max/127)
  float max_abs = 8.0f;
  float expected_high = max_abs / 127.0f;
  EXPECT_NEAR(scale, expected_high, 1e-5f);
}

} // namespace aeon::test
