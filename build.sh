#!/usr/bin/env bash
# ===========================================================================
# Aeon Memory OS — One-Command Build Script (macOS / Linux)
# ===========================================================================
# Usage:
#   ./build.sh            Build (dev) + run tests
#   ./build.sh release    Optimized production build + tests
#   ./build.sh clean      Wipe build directory
#   ./build.sh bench      Build + run benchmark suite
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Detect preset ──
MODE="${1:-dev}"

case "$MODE" in
  release)
    PRESET="release"
    ;;
  clean)
    echo "==> Cleaning all build directories..."
    rm -rf build/dev build/release build/ci-*
    echo "    Done."
    exit 0
    ;;
  bench)
    PRESET="dev"
    RUN_BENCH=1
    ;;
  ci)
    # Auto-detect CI preset based on OS
    if [[ "$(uname -s)" == "Darwin" ]]; then
      PRESET="ci-macos"
    else
      PRESET="ci-linux"
    fi
    ;;
  *)
    PRESET="dev"
    ;;
esac

RUN_BENCH="${RUN_BENCH:-0}"

echo "==========================================================="
echo "  Aeon Memory OS — Build"
echo "  Preset:   $PRESET"
echo "  Platform: $(uname -s) $(uname -m)"
echo "  Compiler: ${CXX:-$(c++ --version 2>&1 | head -1)}"
echo "==========================================================="

# ── Ensure Ninja is available (prefer Ninja, fall back to Make) ──
if command -v ninja &>/dev/null; then
  GENERATOR_FLAG=""  # CMakePresets specifies Ninja
else
  echo "  [WARN] Ninja not found. Installing or falling back to Make."
  if [[ "$(uname -s)" == "Darwin" ]]; then
    brew install ninja 2>/dev/null || true
  else
    sudo apt-get install -y ninja-build 2>/dev/null || true
  fi
fi

# ── Configure ──
echo ""
echo "==> Configuring (cmake --preset $PRESET)..."
cmake --preset "$PRESET"

# ── Build ──
NPROC=$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
echo ""
echo "==> Building (${NPROC} parallel jobs)..."
cmake --build --preset "$PRESET" -- -j"$NPROC"

# ── Test ──
echo ""
echo "==> Running tests (ctest --preset $PRESET)..."
ctest --preset "$PRESET"

# ── Benchmarks (optional) ──
if [[ "$RUN_BENCH" == "1" ]]; then
  echo ""
  echo "==> Running benchmark suite..."
  BUILD_DIR="build/$PRESET"
  
  for bench in bench_kernel_throughput bench_slb_latency bench_scalability; do
    BENCH_BIN="$BUILD_DIR/bin/$bench"
    if [[ -x "$BENCH_BIN" ]]; then
      echo ""
      echo "--- $bench ---"
      "$BENCH_BIN" --benchmark_repetitions=3 --benchmark_report_aggregates_only=true
    fi
  done
fi

echo ""
echo "==========================================================="
echo "  BUILD COMPLETE"
echo "  Preset:    $PRESET"
echo "  Artifacts: build/$PRESET/bin/"
echo "==========================================================="
