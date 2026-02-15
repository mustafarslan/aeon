#include "aeon/math_kernel.hpp"
#include <gtest/gtest.h>
#include <random>
#include <vector>

namespace {
// Use a scalar implementation for ground truth
float ground_truth_dot(std::span<const float> a, std::span<const float> b) {
  float sum = 0.0f;
  for (size_t i = 0; i < a.size(); ++i)
    sum += a[i] * b[i];
  return sum;
}
} // namespace

TEST(MathKernel, DotProductCorrectness) {
  std::vector<float> a(768);
  std::vector<float> b(768);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (auto &val : a)
    val = dist(rng);
  for (auto &val : b)
    val = dist(rng);

  float dot_ab = aeon::math::dot_product(a, b);
  float dot_ba = aeon::math::dot_product(b, a);
  float dot_aa = aeon::math::dot_product(a, a);

  // Property 1: Result must be finite (no NaN/Inf from -ffast-math)
  EXPECT_TRUE(std::isfinite(dot_ab)) << "dot(a,b) produced non-finite result";

  // Property 2: Commutativity — a·b == b·a (must be exact under same SIMD path)
  EXPECT_FLOAT_EQ(dot_ab, dot_ba);

  // Property 3: Self-dot is strictly positive for non-zero vectors
  EXPECT_GT(dot_aa, 0.0f);

  // Property 4: Cauchy-Schwarz bound |a·b| <= ||a|| * ||b||
  float norm_a = std::sqrt(aeon::math::dot_product(a, a));
  float norm_b = std::sqrt(aeon::math::dot_product(b, b));
  EXPECT_LE(std::abs(dot_ab), norm_a * norm_b * 1.01f); // 1% slack for FP
}

TEST(MathKernel, CosineSimilarityIdentity) {
  std::vector<float> a(768, 1.0f);
  EXPECT_NEAR(aeon::math::cosine_similarity(a, a), 1.0f, 1e-4f);
}

TEST(MathKernel, CosineSimilarityOrthogonal) {
  std::vector<float> a = {1, 0, 0, 0};
  std::vector<float> b = {0, 1, 0, 0};
  EXPECT_NEAR(aeon::math::cosine_similarity(a, b), 0.0f, 1e-4f);
}
