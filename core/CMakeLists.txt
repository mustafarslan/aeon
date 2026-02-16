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
option(AEON_CI_BUILD "Use portable SIMD flags for CI runners" OFF)

if(MSVC)
    # MSVC: AVX2 + fast-math + latest C++ standard (C++23 experimental)
    add_compile_options(/O2 /fp:fast /arch:AVX2 /std:c++latest /permissive-)
else()
    # Clang/GCC: architecture-specific optimization
    if(AEON_CI_BUILD)
        if(APPLE)
            # CI macOS ARM64: target Apple M1 baseline (covers M1/M2/M3/M4)
            add_compile_options(-mcpu=apple-m1 -O3 -ffast-math -Wall -Wextra)
        else()
            # CI Linux x86-64: portable AVX2 level (x86-64-v3)
            add_compile_options(-march=x86-64-v3 -O3 -ffast-math -Wall -Wextra)
        endif()
    else()
        # Local development: use native arch for maximum performance
        add_compile_options(-march=native -Wall -Wextra -pedantic)
    endif()
endif()

# --- Threads (cross-platform: pthreads on POSIX, win32 threads on Windows) ---
find_package(Threads REQUIRED)

# --- BLAS Abstraction ---
if(APPLE)
    # macOS: Use Accelerate framework
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if(NOT ACCELERATE_FRAMEWORK)
        message(FATAL_ERROR "Accelerate framework not found on macOS")
    endif()
    # We will link this later to the core target
elseif(WIN32)
    # Windows: Use OpenBLAS (vcpkg or system)
    find_package(OpenBLAS CONFIG)
    if(NOT OpenBLAS_FOUND)
         # Try generic BLAS/LAPACK lookup if pkg-config/vcpkg fails
         find_package(BLAS)
         find_package(LAPACK)
    endif()
else()
    # Linux: Standard BLAS/LAPACK
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
endif()

# --- Core Library (C++) ---
# This library holds the pure C++ logic, usable by C++ apps independently
add_library(aeon_core STATIC
    src/core.cpp
    src/atlas.cpp
    src/simd_impl.cpp
    src/trace.cpp
)

# Link BLAS
if(APPLE)
    target_link_libraries(aeon_core PUBLIC ${ACCELERATE_FRAMEWORK})
elseif(WIN32)
    if(OpenBLAS_FOUND)
       target_link_libraries(aeon_core PUBLIC OpenBLAS::OpenBLAS)
    elseif(BLAS_FOUND)
       target_link_libraries(aeon_core PUBLIC ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
    endif()
else()
    target_link_libraries(aeon_core PUBLIC ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()

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

# Require C++23. MSVC /std:c++latest enables all experimental C++23 features.
if(MSVC)
    target_compile_features(aeon_core PUBLIC cxx_std_23)
else()
    target_compile_features(aeon_core PUBLIC cxx_std_23)
endif()

# --- SIMDe: Portable SIMD (AVX-512/AVX2 → NEON on ARM64) ---
find_path(SIMDE_INCLUDE_DIR NAMES simde/x86/avx512.h
    HINTS /opt/homebrew/include /usr/local/include
)
if(SIMDE_INCLUDE_DIR)
    message(STATUS "SIMDe found at: ${SIMDE_INCLUDE_DIR}")
    target_include_directories(aeon_core PUBLIC ${SIMDE_INCLUDE_DIR})
else()
    message(WARNING "SIMDe not found — SIMD kernels will not compile on ARM64")
endif()

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

# --- Universal C-API Shared Library ---
# Builds libaeon.so (Linux), libaeon.dylib (macOS), aeon.dll (Windows)
# Target platforms: Unreal Engine, Godot, Unity (C#/P/Invoke), iOS, Android JNI
add_library(aeon_shared SHARED
    src/aeon_c_api.cpp
)

target_link_libraries(aeon_shared PRIVATE aeon_core Threads::Threads)
target_compile_features(aeon_shared PRIVATE cxx_std_23)

# AEON_BUILDING_SHARED triggers __declspec(dllexport) on Windows
# and __attribute__((visibility("default"))) on POSIX for AEON_API symbols.
target_compile_definitions(aeon_shared PRIVATE AEON_BUILDING_SHARED)

# Hide all symbols by default; only AEON_API-annotated symbols are exported.
# This prevents C++ name mangling from leaking into the flat C ABI.
set_target_properties(aeon_shared PROPERTIES
    OUTPUT_NAME "aeon"
    CXX_VISIBILITY_PRESET hidden
    C_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    # macOS: set install_name for framework embedding
    MACOSX_RPATH ON
    # Windows: put DLL next to executables for testing
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

if(NOT MSVC)
    # Safe optimizations for the C-API wrapper (no -ffast-math — it's a thin wrapper)
    target_compile_options(aeon_shared PRIVATE -O2 -flto)
endif()

# Install rules for SDK distribution
install(TARGETS aeon_shared
    LIBRARY DESTINATION lib      # .so / .dylib
    ARCHIVE DESTINATION lib      # .lib (Windows import library)
    RUNTIME DESTINATION bin      # .dll (Windows)
)
install(FILES include/aeon/aeon_c_api.h DESTINATION include/aeon)

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
        tests/test_epoch.cpp
        tests/test_concurrency.cpp
        tests/test_wal.cpp
        tests/test_blob_arena.cpp
    )

    target_link_libraries(aeon_tests PRIVATE 
        aeon_core 
        GTest::gtest_main
        Threads::Threads
    )

    target_compile_features(aeon_tests PRIVATE cxx_std_23)

    # Register GTest cases with CTest for discovery
    include(GoogleTest)
    gtest_discover_tests(aeon_tests)

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

    # --- IEEE WCCI 2026 Benchmark Suite ---
    # Test 1: Math Kernel Throughput (§6.1)
    add_executable(bench_kernel_throughput
        benchmarks/bench_kernel_throughput.cpp
    )
    target_link_libraries(bench_kernel_throughput PRIVATE aeon_core benchmark::benchmark)
    target_compile_features(bench_kernel_throughput PRIVATE cxx_std_23)

    # Test 2: SLB Latency (§6.2)
    add_executable(bench_slb_latency
        benchmarks/bench_slb_latency.cpp
    )
    target_link_libraries(bench_slb_latency PRIVATE aeon_core benchmark::benchmark)
    target_compile_features(bench_slb_latency PRIVATE cxx_std_23)

    # Test 3: Spatial Index Scalability (§6.3)
    add_executable(bench_scalability
        benchmarks/bench_scalability.cpp
    )
    target_link_libraries(bench_scalability PRIVATE aeon_core benchmark::benchmark)
    target_compile_features(bench_scalability PRIVATE cxx_std_23)

    # Test 4: EBR Hostile Contention (Phase 1 — P99 latency proof)
    add_executable(bench_ebr_contention
        benchmarks/bench_ebr_contention.cpp
    )
    target_link_libraries(bench_ebr_contention PRIVATE aeon_core Threads::Threads)
    target_compile_features(bench_ebr_contention PRIVATE cxx_std_23)

    # Test 5: Beam Search Scalability (Phase 3 — 1M node proof)
    add_executable(bench_beam_search
        benchmarks/bench_beam_search.cpp
    )
    target_link_libraries(bench_beam_search PRIVATE aeon_core Threads::Threads)
    target_compile_features(bench_beam_search PRIVATE cxx_std_23)

    # Test 6: Multi-Tenant SLB Thrashing (V3 §7.1 — 10K NPC proof)
    add_executable(bench_multitenant_slb
        benchmarks/bench_multitenant_slb.cpp
    )
    target_link_libraries(bench_multitenant_slb PRIVATE aeon_core benchmark::benchmark Threads::Threads)
    target_compile_features(bench_multitenant_slb PRIVATE cxx_std_23)

    # Test 7: Tiered Edge-to-Cloud Atlas (V3 §7.2 — cold miss detection)
    add_executable(bench_tiered_atlas
        benchmarks/bench_tiered_atlas.cpp
    )
    target_link_libraries(bench_tiered_atlas PRIVATE aeon_core benchmark::benchmark)
    target_compile_features(bench_tiered_atlas PRIVATE cxx_std_23)
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
