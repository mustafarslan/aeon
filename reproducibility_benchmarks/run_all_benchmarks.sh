#!/usr/bin/env bash
# ===========================================================================
# Aeon V4.1 — Master Benchmark Runner
# ---------------------------------------------------------------------------
# Compiles the project in strict Release mode, then executes every benchmark
# binary, piping all output to master_metrics.txt.
#
# Usage:
#   chmod +x run_all_benchmarks.sh
#   ./run_all_benchmarks.sh
#
# Output:
#   reproducibility_benchmarks/master_metrics.txt
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build"
OUTPUT_FILE="${SCRIPT_DIR}/master_metrics.txt"

# Number of parallel build jobs
NPROC=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)

# ---------------------------------------------------------------------------
# 1. Configure — strict Release with LTO + fast-math
# ---------------------------------------------------------------------------
echo "══════════════════════════════════════════════════════════════"
echo "  [1/3] CMake Configure (Release, -O3, -march=native, LTO)"
echo "══════════════════════════════════════════════════════════════"

cmake -S "${REPO_ROOT}/core" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-O3 -march=native -flto -ffast-math" \
  -Wno-dev

# ---------------------------------------------------------------------------
# 2. Build — all benchmark targets
# ---------------------------------------------------------------------------
BENCH_TARGETS=(
  # Legacy benchmarks (V3 §6)
  aeon_bench
  bench_kernel_throughput
  bench_slb_latency
  bench_scalability
  bench_ebr_contention
  bench_beam_search
  bench_quantization
  bench_multitenant_slb
  bench_tiered_atlas
  # V4.1 benchmarks
  bench_quantization_efficiency
  bench_wal_overhead
  bench_trace_gc
)

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  [2/3] Building ${#BENCH_TARGETS[@]} benchmark targets"
echo "══════════════════════════════════════════════════════════════"

cmake --build "${BUILD_DIR}" \
  --target "${BENCH_TARGETS[@]}" \
  -j"${NPROC}"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  [3/3] Executing all benchmarks → ${OUTPUT_FILE}"
echo "══════════════════════════════════════════════════════════════"

# ---------------------------------------------------------------------------
# 3. Execute — each benchmark with 5 repetitions, aggregates only
# ---------------------------------------------------------------------------
BENCH_ARGS="--benchmark_repetitions=5 --benchmark_report_aggregates_only=true"

# Google Benchmark binaries (use repetitions + aggregates)
GBENCH_BINARIES=(
  bench_kernel_throughput
  bench_slb_latency
  bench_multitenant_slb
  bench_tiered_atlas
  bench_quantization_efficiency
  bench_wal_overhead
  bench_trace_gc
  aeon_bench
)

# Non-Google-Benchmark binaries (custom timing, no --benchmark_* flags)
CUSTOM_BINARIES=(
  bench_ebr_contention
  bench_beam_search
  bench_quantization
)

# Scalability benchmark — reduced iterations for 1M-node tests
SCALE_ARGS="--benchmark_repetitions=3 --benchmark_report_aggregates_only=true"

{
  echo "============================================================"
  echo "  Aeon V4.1 — Master Benchmark Report"
  echo "  Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "  Host:      $(uname -mnrs)"
  echo "  CPU:       $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'unknown')"
  echo "  Cores:     ${NPROC}"
  echo "============================================================"
  echo ""

  # ── Google Benchmark binaries ──
  for bin in "${GBENCH_BINARIES[@]}"; do
    echo "────────────────────────────────────────────────────────────"
    echo "  RUNNING: ${bin}"
    echo "────────────────────────────────────────────────────────────"
    # shellcheck disable=SC2086
    "${BUILD_DIR}/${bin}" ${BENCH_ARGS} 2>&1 || echo "[WARN] ${bin} exited with non-zero status"
    echo ""
  done

  # ── Scalability (separate due to long runtime at 1M nodes) ──
  echo "────────────────────────────────────────────────────────────"
  echo "  RUNNING: bench_scalability (long-running, 3 reps)"
  echo "────────────────────────────────────────────────────────────"
  # shellcheck disable=SC2086
  "${BUILD_DIR}/bench_scalability" ${SCALE_ARGS} 2>&1 || echo "[WARN] bench_scalability exited with non-zero status"
  echo ""

  # ── Custom binaries (no Google Benchmark flags) ──
  for bin in "${CUSTOM_BINARIES[@]}"; do
    echo "────────────────────────────────────────────────────────────"
    echo "  RUNNING: ${bin}"
    echo "────────────────────────────────────────────────────────────"
    "${BUILD_DIR}/${bin}" 2>&1 || echo "[WARN] ${bin} exited with non-zero status"
    echo ""
  done

  echo "============================================================"
  echo "  COMPLETE — $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "============================================================"

} > "${OUTPUT_FILE}" 2>&1

echo ""
echo "✅ All benchmarks complete. Results written to:"
echo "   ${OUTPUT_FILE}"
echo ""
wc -l "${OUTPUT_FILE}"
