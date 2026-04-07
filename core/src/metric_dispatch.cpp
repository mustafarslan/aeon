#include "aeon/metric_dispatch.hpp"
#include "aeon/simd_impl.hpp"
#include <stdexcept>

namespace aeon::simd {

// Build the kernel tables statically
static const DistanceKernel KERNEL_SCALAR_COSINE = {
    MetricType::Cosine, SimdTier::Scalar, QuantType::FP32, similarity_scalar,
    nullptr};

static const DistanceKernel KERNEL_AVX2_COSINE = {
    MetricType::Cosine, SimdTier::AVX2, QuantType::FP32, similarity_avx2,
    nullptr};

static const DistanceKernel KERNEL_AVX512_COSINE = {
    MetricType::Cosine, SimdTier::AVX512, QuantType::FP32, similarity_avx512,
    nullptr};

static const DistanceKernel KERNEL_NEON_COSINE = {
    MetricType::Cosine, SimdTier::NEON, QuantType::FP32, similarity_neon,
    nullptr};

static const DistanceKernel KERNEL_SCALAR_IP_INT8 = {
    MetricType::InnerProduct, SimdTier::Scalar, QuantType::INT8, nullptr,
    dot_int8_scalar};

static const DistanceKernel KERNEL_AVX512_IP_INT8 = {
    MetricType::InnerProduct, SimdTier::AVX512, QuantType::INT8, nullptr,
    dot_int8_avx512};

static const DistanceKernel KERNEL_NEON_IP_INT8 = {
    MetricType::InnerProduct, SimdTier::NEON, QuantType::INT8, nullptr,
    dot_int8_neon};

SimdTier MetricDispatcher::detected_tier() noexcept {
#if defined(__aarch64__) || defined(__ARM_NEON) || defined(_M_ARM64)
  return SimdTier::NEON;
#else
  // TODO: Implement actual CPUID for x86 to differentiate AVX2 / AVX512.
  // For now, default to AVX512 if compiled for it, else AVX2.
  // Assuming SIMDe AVX512 is our target for x86 if not ARM.
  return SimdTier::AVX512;
#endif
}

const DistanceKernel *MetricDispatcher::resolve(MetricType metric,
                                                QuantType quant) noexcept {
  return resolve(metric, detected_tier(), quant);
}

const DistanceKernel *MetricDispatcher::resolve(MetricType metric,
                                                SimdTier tier,
                                                QuantType quant) noexcept {
  if (quant == QuantType::FP32) {
    if (metric == MetricType::Cosine) {
      switch (tier) {
      case SimdTier::AVX512:
        return &KERNEL_AVX512_COSINE;
      case SimdTier::AVX2:
        return &KERNEL_AVX2_COSINE;
      case SimdTier::NEON:
        return &KERNEL_NEON_COSINE;
      case SimdTier::Scalar:
      default:
        return &KERNEL_SCALAR_COSINE;
      }
    }
    // Fallbacks for unimplemented L2 / IP FP32
    return &KERNEL_SCALAR_COSINE;
  } else if (quant == QuantType::INT8) {
    if (metric == MetricType::InnerProduct || metric == MetricType::Cosine) {
      // INT8 always operates via inner product (dot) because scale takes care
      // of norm.
      switch (tier) {
      case SimdTier::AVX512:
        return &KERNEL_AVX512_IP_INT8;
      // We fallback AVX2 to AVX512 for now since SIMDe handles it
      case SimdTier::AVX2:
        return &KERNEL_AVX512_IP_INT8;
      case SimdTier::NEON:
        return &KERNEL_NEON_IP_INT8;
      case SimdTier::Scalar:
      default:
        return &KERNEL_SCALAR_IP_INT8;
      }
    }
    return &KERNEL_SCALAR_IP_INT8;
  }

  return &KERNEL_SCALAR_COSINE;
}

} // namespace aeon::simd
