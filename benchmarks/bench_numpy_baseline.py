#!/usr/bin/env python3
"""
Test 1 (Python companion): NumPy Cosine Similarity Baseline — Aeon §6.1

Claims under test:
  - Aeon SIMD kernel is ~2000x faster than Python/NumPy
  - Measures per-call overhead including Python interpreter dispatch

Methodology: 100K iterations, reports median + 25/75 percentiles
"""

import time
import numpy as np
import statistics

DIM = 768
N_ITER = 100_000
N_WARMUP = 1_000

# Deterministic seed matching C++ benchmarks
rng = np.random.default_rng(seed=42)
a = rng.uniform(-1.0, 1.0, size=DIM).astype(np.float32)
rng2 = np.random.default_rng(seed=43)
b = rng2.uniform(-1.0, 1.0, size=DIM).astype(np.float32)


def cosine_numpy(x: np.ndarray, y: np.ndarray) -> float:
    """NumPy BLAS-backed cosine similarity (what real code would use)."""
    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))


def cosine_pure_python(x: np.ndarray, y: np.ndarray) -> float:
    """Pure Python loop (no NumPy) — the interpreted baseline."""
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for i in range(len(x)):
        dot += float(x[i]) * float(y[i])
        norm_a += float(x[i]) * float(x[i])
        norm_b += float(y[i]) * float(y[i])
    return dot / (norm_a**0.5 * norm_b**0.5)


def benchmark_function(fn, name: str, n_iter: int, n_warmup: int):
    """Benchmark a function with warmup and per-call timing."""
    # Warmup
    for _ in range(n_warmup):
        fn(a, b)

    # Timed run: measure each call individually
    timings_ns = []
    for _ in range(n_iter):
        t0 = time.perf_counter_ns()
        result = fn(a, b)
        t1 = time.perf_counter_ns()
        timings_ns.append(t1 - t0)

    timings_ns.sort()
    median = statistics.median(timings_ns)
    p25 = timings_ns[int(len(timings_ns) * 0.25)]
    p75 = timings_ns[int(len(timings_ns) * 0.75)]
    mean = statistics.mean(timings_ns)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Iterations:       {n_iter:,}")
    print(f"  Median:           {median:,.0f} ns/op")
    print(f"  Mean:             {mean:,.0f} ns/op")
    print(f"  P25:              {p25:,.0f} ns/op")
    print(f"  P75:              {p75:,.0f} ns/op")
    print(f"  Ops/sec (median): {1e9 / median:,.0f}")
    print(f"  Result sanity:    {result:.6f}")

    return median


if __name__ == "__main__":
    print("=" * 60)
    print("  IEEE WCCI 2026 — Test 1: NumPy Baseline")
    print(f"  Vector dimension: {DIM}")
    print(f"  Hardware: Apple M4 Max (ARM64)")
    print(f"  NumPy version: {np.__version__}")

    try:
        config = np.show_config(mode="dicts")
        if isinstance(config, dict) and "Build Dependencies" in config:
            blas_info = config["Build Dependencies"].get("blas", {})
            print(f"  BLAS: {blas_info.get('name', 'unknown')}")
    except Exception:
        print("  BLAS: unknown")

    print("=" * 60)

    # Benchmark 1: NumPy BLAS-backed
    numpy_median = benchmark_function(
        cosine_numpy,
        "NumPy BLAS Cosine Similarity (numpy.dot + norm)",
        n_iter=N_ITER,
        n_warmup=N_WARMUP,
    )

    # Benchmark 2: Pure Python loop (the "interpreted" baseline)
    # Reduce iterations since pure Python is MUCH slower
    python_median = benchmark_function(
        cosine_pure_python,
        "Pure Python Loop Cosine Similarity (no NumPy)",
        n_iter=1_000,  # Much fewer — too slow otherwise
        n_warmup=10,
    )

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  NumPy median:       {numpy_median:,.0f} ns/op")
    print(f"  Pure Python median: {python_median:,.0f} ns/op")
    print(f"  Python/NumPy ratio: {python_median / numpy_median:,.1f}x")
    print(f"\n  Paper claims Aeon SIMD kernel: ~50 ns/op")
    print(f"  If true, NumPy ratio:          {numpy_median / 50:,.1f}x")
    print(f"  If true, Python ratio:         {python_median / 50:,.1f}x")
    print(f"\n  NOTE: Paper claims '~2000x vs Python' — this metric")
    print(f"  likely refers to the pure Python loop, not NumPy BLAS.")
