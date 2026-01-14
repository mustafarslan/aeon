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

  float expected = ground_truth_dot(a, b);
  float actual = aeon::math::dot_product(a, b);

  EXPECT_NEAR(actual, expected, 1e-3f);
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
