#include "aeon/core.hpp"
#include <format>

namespace aeon::core {

std::string_view version() noexcept { return "0.1.0"; }

BuildInfo get_build_info() {
  BuildInfo info;

// Detect Compiler
#if defined(__clang__)
  info.compiler = std::format("Clang {}", __clang_version__);
#elif defined(__GNUC__)
  info.compiler = std::format("GCC {}.{}.{}", __GNUC__, __GNUC_MINOR__,
                              __GNUC_PATCHLEVEL__);
#elif defined(_MSC_VER)
  info.compiler = std::format("MSVC {}", _MSC_VER);
#else
  info.compiler = "Unknown";
#endif

// Detect Architecture
#if defined(__x86_64__) || defined(_M_X64)
  info.architecture = "x86_64";
#elif defined(__aarch64__) || defined(_M_ARM64)
  info.architecture = "arm64";
#else
  info.architecture = "Unknown";
#endif

  // Detect SIMD (Compile-time detection for this scaffolding)
  // In a real high-perf engine, we'd also do runtime checks.
  info.simd_level = "None";
#if defined(__AVX512F__)
  info.simd_level = "AVX-512";
#elif defined(__AVX2__)
  info.simd_level = "AVX2";
#elif defined(__AVX__)
  info.simd_level = "AVX";
#elif defined(__ARM_NEON)
  info.simd_level = "NEON";
#endif

  // C++ Standard
  info.standard = std::format("C++{}", __cplusplus / 100 % 100);

  return info;
}

} // namespace aeon::core
