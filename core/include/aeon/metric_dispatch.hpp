#pragma once

/**
 * @file metric_dispatch.hpp
 * @brief Hardware-Agnostic Metric Dispatch — Zero-Overhead Kernel Router
 *
 * USearch-inspired dynamic dispatch system. Rather than relying on monolithic
 * CPUID wrapper functions executing indirect branches on every cycle, this
 * dispatcher returns a strongly-typed `DistanceKernel` struct at `Atlas`
 * initialization time. The `Atlas` struct caches these raw function pointers
 * for zero virtual overhead traverses inside the M-ary tree traversal hot loop.
 */

#include <cstdint>
#include <span>

namespace aeon::simd {

enum class MetricType : uint8_t { Cosine = 0, L2 = 1, InnerProduct = 2 };

enum class SimdTier : uint8_t {
  Scalar = 0, // Fallback pipeline
  AVX2 = 1,   // x86_64
  AVX512 = 2, // x86_64 (Cloud variants)
  NEON = 3    // ARM64 / Apple Silicon
};

enum class QuantType : uint8_t { FP32 = 0, INT8 = 1 };

// Raw function pointer signatures enforcing absolute C++ ABI adherence.
typedef float (*aeon_metric_f32_t)(const float *a, const float *b,
                                   uint32_t dim);
typedef int32_t (*aeon_metric_i8_t)(const int8_t *a, const int8_t *b,
                                    uint32_t dim);

/**
 * @brief Represents a fully resolved and validated similarity kernel.
 *
 * ALIGNAS(64): Forces isolation of this struct onto a separate CPU cache line.
 * This prevents false sharing degradation during high-concurrency retrieval
 * loops when thousands of concurrent NPCs access metric configurations.
 */
struct alignas(64) DistanceKernel {
  MetricType metric;
  SimdTier tier;
  QuantType quant;

  // Guaranteed that only one of these function pointers matches 'quant'
  // specification.
  aeon_metric_f32_t compute_f32;
  aeon_metric_i8_t compute_i8;
};

/**
 * @brief Singleton router mapping metric geometries and quantization formats to
 * hardware ISAs.
 *
 * Invokes CPUID/feature register detection strictly on first invocation block.
 */
class MetricDispatcher {
public:
  /**
   * @brief Resolves the mathematically optimal distance kernel for the current
   * CPU. Guaranteed to return a const pointer to prevent "torn function
   * pointers" at runtime.
   *
   * @param metric   The similarity metric (e.g., Cosine/L2)
   * @param quant    The quantization domain (FP32/INT8)
   * @return Const pointer to the cached 64-byte aligned kernel definition.
   */
  static const DistanceKernel *resolve(MetricType metric,
                                       QuantType quant) noexcept;

  /**
   * @brief Explicitly resolves a specific instruction tier (used strictly for
   * Benchmarking).
   */
  static const DistanceKernel *resolve(MetricType metric, SimdTier tier,
                                       QuantType quant) noexcept;

  /**
   * @brief Introspects the auto-detected top-tier SIMD instruction set
   * executing locally.
   */
  static SimdTier detected_tier() noexcept;
};

} // namespace aeon::simd
