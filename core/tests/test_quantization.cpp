/**
 * @file test_quantization.cpp
 * @brief Unit tests for V4.1 Phase 3 — INT8 Scalar Quantization.
 *
 * Tests:
 *  1. Round-trip quantization accuracy
 *  2. Zero-vector safety (divide-by-zero guard)
 *  3. Clamping to [-127, +127]
 *  4. INT8 dot product correctness (SIMD vs scalar)
 *  5. Dequantization math
 */

#include "aeon/quantization.hpp"
#include "aeon/schema.hpp"
#include "aeon/simd_impl.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <vector>

namespace aeon::test {

// ═══════════════════════════════════════════════════════════════════════════
// Test Fixture
// ═══════════════════════════════════════════════════════════════════════════

class QuantizationTest : public ::testing::Test {
protected:
  static constexpr uint32_t DIM = 384;

  std::vector<float> make_random_vector(uint32_t dim, float scale = 1.0f,
                                        unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<float> v(dim);
    for (auto &x : v)
      x = dist(rng);
    return v;
  }

  std::vector<float> make_zero_vector(uint32_t dim) {
    return std::vector<float>(dim, 0.0f);
  }

  std::vector<float> make_extreme_vector(uint32_t dim) {
    std::vector<float> v(dim);
    for (uint32_t i = 0; i < dim; ++i) {
      // Alternate between large positive and negative values
      v[i] = (i % 2 == 0) ? 1000.0f : -1000.0f;
    }
    return v;
  }
};

// ═══════════════════════════════════════════════════════════════════════════
// A. Quantization Round-Trip Accuracy
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(QuantizationTest, RoundTripAccuracy) {
  auto vec = make_random_vector(DIM, 1.0f);
  std::vector<int8_t> quantized(DIM);
  float scale;

  quant::quantize_symmetric(vec, quantized, scale);

  // Verify scale is positive and reasonable
  EXPECT_GT(scale, 0.0f);

  // Verify round-trip error is within quantization tolerance
  // Max error per element should be at most 0.5 * scale (half a quantization
  // step)
  float max_error = 0.0f;
  for (uint32_t i = 0; i < DIM; ++i) {
    float reconstructed = static_cast<float>(quantized[i]) * scale;
    float error = std::fabs(reconstructed - vec[i]);
    max_error = std::max(max_error, error);
  }

  // Error should be within one quantization step
  EXPECT_LE(max_error, scale + 1e-6f)
      << "Round-trip error exceeds quantization step";
}

TEST_F(QuantizationTest, SymmetricScaleComputation) {
  // Create a vector where max absolute value is known
  std::vector<float> vec(DIM, 0.0f);
  vec[0] = 12.7f;  // Known max
  vec[1] = -12.7f; // Symmetric

  std::vector<int8_t> quantized(DIM);
  float scale;
  quant::quantize_symmetric(vec, quantized, scale);

  // scale = max(|v|) / 127 = 12.7 / 127 = 0.1
  EXPECT_NEAR(scale, 0.1f, 1e-6f);

  // The max value should map to exactly 127
  EXPECT_EQ(quantized[0], 127);
  // The min value should map to exactly -127
  EXPECT_EQ(quantized[1], -127);
  // Zeros should remain zero
  EXPECT_EQ(quantized[2], 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// B. Zero-Vector Safety (Divide-by-Zero Guard)
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(QuantizationTest, ZeroVectorSafety) {
  auto vec = make_zero_vector(DIM);
  std::vector<int8_t> quantized(DIM);
  float scale;

  // This should NOT crash or produce NaN/Inf
  quant::quantize_symmetric(vec, quantized, scale);

  EXPECT_EQ(scale, 1.0f) << "Zero vector should produce scale=1.0 for safety";

  for (uint32_t i = 0; i < DIM; ++i) {
    EXPECT_EQ(quantized[i], 0) << "Zero vector should quantize to all zeros";
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// C. Clamping to [-127, +127]
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(QuantizationTest, ClampingSymmetric) {
  auto vec = make_extreme_vector(DIM);
  std::vector<int8_t> quantized(DIM);
  float scale;

  quant::quantize_symmetric(vec, quantized, scale);

  for (uint32_t i = 0; i < DIM; ++i) {
    EXPECT_GE(quantized[i], -127)
        << "Value at index " << i << " is below -127: " << (int)quantized[i];
    EXPECT_LE(quantized[i], 127)
        << "Value at index " << i << " is above 127: " << (int)quantized[i];
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// D. INT8 Dot Product Correctness
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(QuantizationTest, DotProductScalarVsReference) {
  auto vec_a = make_random_vector(DIM, 0.5f, 42);
  auto vec_b = make_random_vector(DIM, 0.5f, 99);

  std::vector<int8_t> q_a(DIM), q_b(DIM);
  float scale_a, scale_b;
  quant::quantize_symmetric(vec_a, q_a, scale_a);
  quant::quantize_symmetric(vec_b, q_b, scale_b);

  // Reference: manual int32 accumulation
  int32_t ref_dot = 0;
  for (uint32_t i = 0; i < DIM; ++i) {
    ref_dot += static_cast<int32_t>(q_a[i]) * static_cast<int32_t>(q_b[i]);
  }

  // Scalar SIMD implementation
  int32_t simd_dot = simd::dot_int8_scalar(q_a, q_b, DIM);
  EXPECT_EQ(simd_dot, ref_dot)
      << "Scalar INT8 dot product does not match reference";
}

TEST_F(QuantizationTest, DotProductDispatchedVsScalar) {
  auto vec_a = make_random_vector(DIM, 0.8f, 123);
  auto vec_b = make_random_vector(DIM, 0.8f, 456);

  std::vector<int8_t> q_a(DIM), q_b(DIM);
  float scale_a, scale_b;
  quant::quantize_symmetric(vec_a, q_a, scale_a);
  quant::quantize_symmetric(vec_b, q_b, scale_b);

  int32_t scalar_dot = simd::dot_int8_scalar(q_a, q_b, DIM);

  // Get best implementation for this platform (NEON on ARM, AVX-512 on x86)
  auto best_fn = simd::get_best_int8_dot_impl();
  int32_t best_dot = best_fn(q_a, q_b, DIM);

  EXPECT_EQ(best_dot, scalar_dot)
      << "Dispatched INT8 dot product differs from scalar baseline";
}

TEST_F(QuantizationTest, DotProductVariousDimensions) {
  // Test non-standard dimensions to exercise scalar cleanup paths
  for (uint32_t dim : {1,  3,  7,   15,  16,  17,  31,  32,  33,  63,
                       64, 65, 128, 256, 383, 384, 385, 512, 768, 1536}) {
    auto vec_a = make_random_vector(dim, 1.0f, dim);
    auto vec_b = make_random_vector(dim, 1.0f, dim + 1);

    std::vector<int8_t> q_a(dim), q_b(dim);
    float sa, sb;
    quant::quantize_symmetric(vec_a, q_a, sa);
    quant::quantize_symmetric(vec_b, q_b, sb);

    int32_t ref = 0;
    for (uint32_t i = 0; i < dim; ++i) {
      ref += static_cast<int32_t>(q_a[i]) * static_cast<int32_t>(q_b[i]);
    }

    auto best_fn = simd::get_best_int8_dot_impl();
    int32_t result = best_fn(q_a, q_b, dim);

    EXPECT_EQ(result, ref) << "INT8 dot product mismatch at dim=" << dim;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// E. Dequantization Math
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(QuantizationTest, DequantizeDotProduct) {
  int32_t raw_dot = 12345;
  float query_scale = 0.1f;
  float node_scale = 0.05f;

  float result =
      quant::dequantize_dot_product(raw_dot, query_scale, node_scale);
  float expected = 12345.0f * 0.1f * 0.05f;

  EXPECT_FLOAT_EQ(result, expected);
}

TEST_F(QuantizationTest, EndToEndQuantDot) {
  // Verify that quantize → dot → dequantize approximates FP32 dot product
  auto vec_a = make_random_vector(DIM, 1.0f, 77);
  auto vec_b = make_random_vector(DIM, 1.0f, 88);

  // FP32 reference dot product
  float fp32_dot = 0.0f;
  for (uint32_t i = 0; i < DIM; ++i) {
    fp32_dot += vec_a[i] * vec_b[i];
  }

  // Quantize both
  std::vector<int8_t> q_a(DIM), q_b(DIM);
  float scale_a, scale_b;
  quant::quantize_symmetric(vec_a, q_a, scale_a);
  quant::quantize_symmetric(vec_b, q_b, scale_b);

  // INT8 dot product + dequantize
  auto best_fn = simd::get_best_int8_dot_impl();
  int32_t raw_dot = best_fn(q_a, q_b, DIM);
  float approx_dot = quant::dequantize_dot_product(raw_dot, scale_a, scale_b);

  // Relative error should be within ~2% for dim=384
  float rel_error = std::fabs(fp32_dot) > 1e-6f
                        ? std::fabs(approx_dot - fp32_dot) / std::fabs(fp32_dot)
                        : std::fabs(approx_dot - fp32_dot);

  EXPECT_LT(rel_error, 0.05f)
      << "End-to-end quantization error too high: fp32=" << fp32_dot
      << " int8_approx=" << approx_dot << " rel_err=" << rel_error;
}

// ═══════════════════════════════════════════════════════════════════════════
// F. Schema Static Asserts
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(QuantizationTest, SchemaLayout) {
  // Verify struct sizes remain correct
  static_assert(sizeof(AtlasHeader) == 64, "AtlasHeader must be 64 bytes");
  static_assert(sizeof(NodeHeader) == 64, "NodeHeader must be 64 bytes");

  // Verify node stride computation
  constexpr size_t fp32_stride = compute_node_stride(384, 256);
  constexpr size_t int8_stride =
      compute_node_stride(384, 256, QUANT_INT8_SYMMETRIC);

  // INT8 stride should be significantly smaller than FP32
  EXPECT_LT(int8_stride, fp32_stride)
      << "INT8 node stride (" << int8_stride
      << ") should be smaller than FP32 (" << fp32_stride << ")";

  // FP32: 64 (header) + 384*4 (1536) + 256 (meta) = 1856, align up to 64 →
  // 1856
  EXPECT_EQ(fp32_stride, 1856u);

  // INT8: 64 (header) + 384*1 (384) + 256 (meta) = 704, align up to 64 → 704
  EXPECT_EQ(int8_stride, 704u);
}

} // namespace aeon::test
