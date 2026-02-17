# =============================================================================
# Aeon Node Bridge — cmake-js Build Configuration
# Target: macOS Apple Silicon (M4 Max) — NEON SDOT, C++23, Zero-Copy N-API
# =============================================================================
cmake_minimum_required(VERSION 3.26)
project(aeon_node LANGUAGES CXX)

# ---------------------------------------------------------------------------
# C++23 Standard (matches core kernel)
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Apple Silicon M4 Max — Hardware Intrinsics
# Enforces NEON SDOT utilization via -mcpu=apple-m4.
# -ffast-math enables FMA contraction and reciprocal approximations.
# -flto enables cross-TU inlining between bridge and libaeon.
# ---------------------------------------------------------------------------
add_compile_options(
    -mcpu=apple-m4
    -O3
    -flto
    -ffast-math
    -Wall
    -Wextra
    -Wpedantic
    -Wno-unused-parameter
)

# ---------------------------------------------------------------------------
# node-addon-api include path (resolved at configure time)
# cmake-js injects CMAKE_JS_INC with Node.js + N-API headers.
# ---------------------------------------------------------------------------
execute_process(
    COMMAND node -p "require('node-addon-api').include.replace(/\"/g, '')"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE NODE_ADDON_API_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# ---------------------------------------------------------------------------
# Locate libaeon.dylib from the workspace build directory
# ---------------------------------------------------------------------------
set(AEON_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(AEON_DYLIB "${AEON_ROOT}/build/libaeon.dylib")

if(NOT EXISTS "${AEON_DYLIB}")
    message(FATAL_ERROR
        "libaeon.dylib not found at ${AEON_DYLIB}. "
        "Build the core library first: cd ${AEON_ROOT} && cmake --build build --target aeon_shared"
    )
endif()

# ---------------------------------------------------------------------------
# Native Module Target: aeon_node
# Produces aeon_node.node (shared library with .node suffix via cmake-js)
# ---------------------------------------------------------------------------
add_library(aeon_node SHARED
    src/aeon_node.cpp
)

# cmake-js provides CMAKE_JS_SRC for the node stub source.
# Only add if it exists (cmake-js >= 7.x populates this).
if(CMAKE_JS_SRC)
    target_sources(aeon_node PRIVATE ${CMAKE_JS_SRC})
endif()

# ---------------------------------------------------------------------------
# Include Directories
# ---------------------------------------------------------------------------
target_include_directories(aeon_node PRIVATE
    # Aeon C-API headers
    "${AEON_ROOT}/core/include"
    # node-addon-api C++ wrappers
    "${NODE_ADDON_API_DIR}"
    # cmake-js injected Node.js + N-API headers
    ${CMAKE_JS_INC}
)

# ---------------------------------------------------------------------------
# Link against libaeon.dylib
# ---------------------------------------------------------------------------
target_link_libraries(aeon_node PRIVATE "${AEON_DYLIB}")

# LTO for the link step
target_link_options(aeon_node PRIVATE -flto)

# ---------------------------------------------------------------------------
# N-API Configuration
# ---------------------------------------------------------------------------
target_compile_definitions(aeon_node PRIVATE
    # Target N-API version 9 (Node 18+ stable ABI)
    NAPI_VERSION=9
    # Disable C++ exceptions in node-addon-api.
    # We handle errors via Napi::Error::New() + early return.
    NAPI_DISABLE_CPP_EXCEPTIONS
)

# ---------------------------------------------------------------------------
# Output Configuration
# cmake-js expects a .node suffix. Set prefix to empty (no "lib" prefix).
# ---------------------------------------------------------------------------
set_target_properties(aeon_node PROPERTIES
    PREFIX ""
    SUFFIX ".node"
    # macOS RPATH: resolve libaeon.dylib from the same directory as .node
    MACOSX_RPATH ON
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "@loader_path"
    # Hide all symbols except those exported by N-API
    CXX_VISIBILITY_PRESET hidden
    C_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# ---------------------------------------------------------------------------
# Post-Build: Copy libaeon.dylib alongside aeon_node.node
# This makes the build output self-contained — no fragile RPATH to ../../build.
# ---------------------------------------------------------------------------
add_custom_command(TARGET aeon_node POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${AEON_DYLIB}"
        "$<TARGET_FILE_DIR:aeon_node>/libaeon.dylib"
    COMMENT "Copying libaeon.dylib alongside aeon_node.node for self-contained deployment"
)
