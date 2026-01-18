cmake_minimum_required(VERSION 3.26)

# --- Nanobind Setup ---
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
# Use the nanobind package installed via pip
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT
)
list(APPEND CMAKE_PREFIX_PATH "${nanobind_ROOT}")
find_package(nanobind CONFIG REQUIRED)

# --- Compiler Optimizations ---
if(MSVC)
    add_compile_options(/arch:AVX2 /O2)
else()
    # Base aggressive flags - generally safe
    add_compile_options(-march=native -Wall -Wextra -pedantic)
endif()

# --- Core Library (C++) ---
# This library holds the pure C++ logic, usable by C++ apps independently
add_library(aeon_core STATIC
    src/core.cpp
    src/atlas.cpp
    src/simd_impl.cpp
    src/trace.cpp
)
# Apply dangerous optimizations ONLY to the core library (math kernels)
# This prevents breaking GTest which relies on Infinity/NaN
if(NOT MSVC)
    target_compile_options(aeon_core PRIVATE -O3 -ffast-math -flto)
    # Set debug flags for non-MSVC
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
endif()

target_include_directories(aeon_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Export headers for other targets
set_target_properties(aeon_core PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# --- Python Extension ---
# Links the core library and exposes it to Python
nanobind_add_module(aeon_py_core
    src/bindings.cpp
)

target_link_libraries(aeon_py_core PRIVATE aeon_core)

set_target_properties(aeon_py_core PROPERTIES OUTPUT_NAME "core")


# Output definitions (install rules handled by scikit-build-core)
install(TARGETS aeon_py_core DESTINATION .)

# --- Testing ---
# --- Testing ---
if(BUILD_TESTING)
    enable_testing()

    include(FetchContent)
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    add_executable(aeon_tests 
        tests/test_main.cpp
        tests/test_math.cpp
        tests/test_storage.cpp
        tests/test_atlas.cpp
    )

    target_link_libraries(aeon_tests PRIVATE 
        aeon_core 
        GTest::gtest_main
    )

    # Default to C++23 for tests
    target_compile_features(aeon_tests PRIVATE cxx_std_23)

    # --- Benchmarks ---
    # Add Google Benchmark
    FetchContent_Declare(
      googlebenchmark
      URL https://github.com/google/benchmark/archive/refs/tags/v1.8.3.zip
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googlebenchmark)

    # Previous manual benchmark (keep if needed, or remove. Keeping for legacy)
    add_executable(aeon_benchmark_legacy
        tests/benchmark_math.cpp
    )
    target_link_libraries(aeon_benchmark_legacy PRIVATE aeon_core)
    target_compile_features(aeon_benchmark_legacy PRIVATE cxx_std_23)

    # New Google Benchmark Suite
    add_executable(aeon_bench
        benchmarks/bench_main.cpp
    )
    target_link_libraries(aeon_bench PRIVATE aeon_core benchmark::benchmark_main)
    target_compile_features(aeon_bench PRIVATE cxx_std_23)
endif()

# --- Advanced Optimizations ---
# Profile Guided Optimization (PGO)
# Step 1: Generate Profile
add_custom_target(pgo_generate
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/pgo_profile
    COMMAND ${CMAKE_COMMAND} -DCMAKE_CXX_FLAGS="-fprofile-generate=${CMAKE_BINARY_DIR}/pgo_profile" ${CMAKE_SOURCE_DIR}
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target aeon_bench
    COMMAND ${CMAKE_BINARY_DIR}/bin/aeon_bench
    COMMENT "Building with PGO instrumentation and running benchmark..."
)

# Step 2: Use Profile
add_custom_target(pgo_use
    COMMAND ${CMAKE_COMMAND} -DCMAKE_CXX_FLAGS="-fprofile-use=${CMAKE_BINARY_DIR}/pgo_profile -Wno-missing-profile" ${CMAKE_SOURCE_DIR}
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target aeon_py_core
    COMMENT "Building optimized binary using PGO profile..."
)
