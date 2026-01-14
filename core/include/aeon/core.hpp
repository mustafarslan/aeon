#pragma once

#include <string>
#include <string_view>

// Symbol visibility macros
#if defined(_WIN32)
  #if defined(AEON_CORE_EXPORTS)
    #define AEON_CORE_EXPORT __declspec(dllexport)
  #else
    #define AEON_CORE_EXPORT __declspec(dllimport)
  #endif
#else
  #define AEON_CORE_EXPORT __attribute__((visibility("default")))
#endif

namespace aeon::core {

    struct BuildInfo {
        std::string compiler;
        std::string architecture;
        std::string simd_level;
        std::string standard;
    };

    /**
     * @brief Returns the version of the Aeon Core.
     */
    AEON_CORE_EXPORT std::string_view version() noexcept;

    /**
     * @brief Returns build-time information about the library.
     */
    AEON_CORE_EXPORT BuildInfo get_build_info();

} // namespace aeon::core
