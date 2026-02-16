@echo off
REM ===========================================================================
REM Aeon Memory OS — One-Command Build Script (Windows)
REM ===========================================================================
REM Usage:
REM   build.bat             Build (dev) + run tests
REM   build.bat release     Optimized production build + tests
REM   build.bat clean       Wipe build directory
REM   build.bat bench       Build + run benchmark suite
REM ===========================================================================
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set "MODE=%~1"
if "%MODE%"=="" set "MODE=dev"

if /i "%MODE%"=="clean" (
    echo ==^> Cleaning all build directories...
    if exist build\dev     rmdir /s /q build\dev
    if exist build\release rmdir /s /q build\release
    if exist build\ci-windows rmdir /s /q build\ci-windows
    echo     Done.
    exit /b 0
)

set "PRESET=ci-windows"
set "RUN_BENCH=0"

if /i "%MODE%"=="release" (
    set "PRESET=release"
) else if /i "%MODE%"=="bench" (
    set "PRESET=ci-windows"
    set "RUN_BENCH=1"
) else if /i "%MODE%"=="ci" (
    set "PRESET=ci-windows"
) else (
    set "PRESET=ci-windows"
)

echo ===========================================================
echo   Aeon Memory OS — Build
echo   Preset:   %PRESET%
echo   Platform: Windows x86-64
echo   Compiler: MSVC
echo ===========================================================

REM ── Check for Visual Studio environment ──
where cl >nul 2>&1
if errorlevel 1 (
    echo [ERROR] MSVC compiler ^(cl.exe^) not found.
    echo         Run this script from a "Developer Command Prompt for VS 2022"
    echo         or execute: "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
    exit /b 1
)

REM ── Configure ──
echo.
echo ==^> Configuring (cmake --preset %PRESET%)...
cmake --preset %PRESET%
if errorlevel 1 (
    echo [ERROR] CMake configuration failed.
    exit /b 1
)

REM ── Build ──
echo.
echo ==^> Building...
cmake --build --preset %PRESET%
if errorlevel 1 (
    echo [ERROR] Build failed.
    exit /b 1
)

REM ── Test ──
echo.
echo ==^> Running tests (ctest --preset %PRESET%)...
ctest --preset %PRESET% --output-on-failure
if errorlevel 1 (
    echo [WARN] Some tests failed. Check output above.
)

REM ── Benchmarks (optional) ──
if "%RUN_BENCH%"=="1" (
    echo.
    echo ==^> Running benchmark suite...
    set "BUILD_DIR=build\%PRESET%"
    
    for %%b in (bench_kernel_throughput bench_slb_latency bench_scalability) do (
        if exist "!BUILD_DIR!\bin\%%b.exe" (
            echo.
            echo --- %%b ---
            "!BUILD_DIR!\bin\%%b.exe" --benchmark_repetitions=3 --benchmark_report_aggregates_only=true
        )
    )
)

echo.
echo ===========================================================
echo   BUILD COMPLETE
echo   Preset:    %PRESET%
echo   Artifacts: build\%PRESET%\bin\
echo ===========================================================
endlocal
