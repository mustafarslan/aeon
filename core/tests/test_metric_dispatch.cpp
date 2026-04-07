#include "aeon/metric_dispatch.hpp"
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace aeon::simd;

class MetricDispatchTest : public ::testing::Test {
protected:
  static constexpr uint32_t DIM = 384;
  std::vector<float> f_a;
  std::vector<float> f_b;
  std::vector<int8_t> i_a;
  std::vector<int8_t> i_b;

  void SetUp() override {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> f_dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int32_t> i_dist(-128, 127);

    f_a.resize(DIM);
    f_b.resize(DIM);
    i_a.resize(DIM);
    i_b.resize(DIM);

    for (uint32_t i = 0; i < DIM; ++i) {
      f_a[i] = f_dist(rng);
      f_b[i] = f_dist(rng);
      i_a[i] = static_cast<int8_t>(i_dist(rng));
      i_b[i] = static_cast<int8_t>(i_dist(rng));
    }
  }
};

TEST_F(MetricDispatchTest, CosineFp32DispatchEquivalence) {
  auto scalar_kernel = MetricDispatcher::resolve(
      MetricType::Cosine, SimdTier::Scalar, QuantType::FP32);
  auto auto_kernel =
      MetricDispatcher::resolve(MetricType::Cosine, QuantType::FP32);

  ASSERT_NE(scalar_kernel, nullptr);
  ASSERT_NE(auto_kernel, nullptr);
  ASSERT_NE(scalar_kernel->compute_f32, nullptr);
  ASSERT_NE(auto_kernel->compute_f32, nullptr);

  float scalar_res = scalar_kernel->compute_f32(f_a.data(), f_b.data(), DIM);
  float auto_res = auto_kernel->compute_f32(f_a.data(), f_b.data(), DIM);

  EXPECT_NEAR(scalar_res, auto_res, 1e-5f);
}

TEST_F(MetricDispatchTest, L2Fp32DispatchEquivalence) {
  auto scalar_kernel = MetricDispatcher::resolve(
      MetricType::L2, SimdTier::Scalar, QuantType::FP32);
  auto auto_kernel = MetricDispatcher::resolve(MetricType::L2, QuantType::FP32);

  ASSERT_NE(scalar_kernel, nullptr);
  ASSERT_NE(auto_kernel, nullptr);
  ASSERT_NE(scalar_kernel->compute_f32, nullptr);
  ASSERT_NE(auto_kernel->compute_f32, nullptr);

  float scalar_res = scalar_kernel->compute_f32(f_a.data(), f_b.data(), DIM);
  float auto_res = auto_kernel->compute_f32(f_a.data(), f_b.data(), DIM);

  EXPECT_NEAR(scalar_res, auto_res, 1e-4f);
}

TEST_F(MetricDispatchTest, InnerProductInt8DispatchEquivalence) {
  auto scalar_kernel = MetricDispatcher::resolve(
      MetricType::InnerProduct, SimdTier::Scalar, QuantType::INT8);
  auto auto_kernel =
      MetricDispatcher::resolve(MetricType::InnerProduct, QuantType::INT8);

  ASSERT_NE(scalar_kernel, nullptr);
  ASSERT_NE(auto_kernel, nullptr);
  ASSERT_NE(scalar_kernel->compute_i8, nullptr);
  ASSERT_NE(auto_kernel->compute_i8, nullptr);

  int32_t scalar_res = scalar_kernel->compute_i8(i_a.data(), i_b.data(), DIM);
  int32_t auto_res = auto_kernel->compute_i8(i_a.data(), i_b.data(), DIM);

  EXPECT_EQ(scalar_res, auto_res);
}
