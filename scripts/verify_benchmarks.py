#!/usr/bin/env python3
"""
Aeon Benchmark Verification Suite

A rigorous statistical benchmark runner that wraps C++ and Python benchmarks
to produce statistically significant results for paper publication.

Features:
- 10 warm-up iterations to prime CPU caches
- 50 measurement iterations for statistical significance
- Calculates Mean, Median, P99, and Standard Deviation
- Warns if stddev > 10% (noisy environment)
- Fails if Warm Search > 200ms (Phase 0 constraint)

Output: results/benchmark_final.json

Usage:
    python scripts/verify_benchmarks.py [--iterations N] [--warmup N]
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import statistics

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_WARMUP_ITERATIONS = 10
DEFAULT_MEASUREMENT_ITERATIONS = 50
NOISE_THRESHOLD = 0.10  # 10% stddev/mean ratio
WARM_SEARCH_LATENCY_LIMIT_MS = 200.0  # Phase 0 constraint

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
BUILD_DIR = PROJECT_ROOT / "build"
BENCHMARK_BINARY = BUILD_DIR / "bin" / "aeon_bench"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_FILE = RESULTS_DIR / "benchmark_final.json"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class BenchmarkStats:
    """Statistical summary of a benchmark run."""
    name: str
    unit: str
    iterations: int
    warmup_iterations: int
    samples: list[float] = field(default_factory=list)
    mean: float = 0.0
    median: float = 0.0
    p99: float = 0.0
    stddev: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    noise_warning: bool = False
    
    def calculate_stats(self) -> None:
        """Calculate all statistics from samples."""
        if not self.samples:
            return
        
        self.mean = statistics.mean(self.samples)
        self.median = statistics.median(self.samples)
        self.stddev = statistics.stdev(self.samples) if len(self.samples) > 1 else 0.0
        self.min_val = min(self.samples)
        self.max_val = max(self.samples)
        
        # P99: 99th percentile
        sorted_samples = sorted(self.samples)
        p99_index = int(len(sorted_samples) * 0.99)
        self.p99 = sorted_samples[min(p99_index, len(sorted_samples) - 1)]
        
        # Check for noisy environment
        if self.mean > 0:
            noise_ratio = self.stddev / self.mean
            self.noise_warning = noise_ratio > NOISE_THRESHOLD


@dataclass
class BenchmarkResult:
    """Complete benchmark results with metadata."""
    timestamp: str
    platform: dict
    configuration: dict
    benchmarks: dict[str, dict] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    passed: bool = True
    failure_reason: Optional[str] = None


# =============================================================================
# Utility Functions
# =============================================================================

def get_platform_info() -> dict:
    """Collect platform information for reproducibility."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }


def print_header(text: str, char: str = "=") -> None:
    """Print a formatted header."""
    width = 70
    print(f"\n{char * width}")
    print(f" {text}")
    print(f"{char * width}")


def print_stats(stats: BenchmarkStats) -> None:
    """Print formatted statistics for a benchmark."""
    print(f"\n  {stats.name}:")
    print(f"    Mean:   {stats.mean:12.3f} {stats.unit}")
    print(f"    Median: {stats.median:12.3f} {stats.unit}")
    print(f"    P99:    {stats.p99:12.3f} {stats.unit}")
    print(f"    Stddev: {stats.stddev:12.3f} {stats.unit}")
    print(f"    Range:  [{stats.min_val:.3f}, {stats.max_val:.3f}] {stats.unit}")
    
    if stats.noise_warning:
        noise_pct = (stats.stddev / stats.mean) * 100 if stats.mean > 0 else 0
        print(f"    ⚠️  WARNING: High variance ({noise_pct:.1f}%) - environment may be noisy")


# =============================================================================
# Benchmark Runners
# =============================================================================

def run_cpp_benchmark_gbench(
    warmup: int,
    iterations: int,
    benchmark_filter: str = ""
) -> dict[str, BenchmarkStats]:
    """
    Run C++ benchmarks using Google Benchmark with JSON output.
    
    Google Benchmark handles its own iterations, so we run the benchmark
    multiple times to get our statistical samples.
    """
    if not BENCHMARK_BINARY.exists():
        raise FileNotFoundError(
            f"Benchmark binary not found: {BENCHMARK_BINARY}\n"
            "Please build with: cmake --build build --target aeon_bench"
        )
    
    results: dict[str, BenchmarkStats] = {}
    
    # Build command
    cmd = [str(BENCHMARK_BINARY), "--benchmark_format=json"]
    if benchmark_filter:
        cmd.append(f"--benchmark_filter={benchmark_filter}")
    
    print(f"\n  Running: {' '.join(cmd)}")
    print(f"  Warm-up iterations: {warmup}")
    print(f"  Measurement iterations: {iterations}")
    
    all_samples: dict[str, list[float]] = {}
    
    total_runs = warmup + iterations
    for run_idx in range(total_runs):
        is_warmup = run_idx < warmup
        phase = "warmup" if is_warmup else "measure"
        run_num = run_idx + 1 if is_warmup else run_idx - warmup + 1
        total_phase = warmup if is_warmup else iterations
        
        sys.stdout.write(f"\r  [{phase}] {run_num}/{total_phase}...")
        sys.stdout.flush()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
                timeout=60
            )
            
            if result.returncode != 0:
                print(f"\n  ERROR: Benchmark failed with code {result.returncode}")
                print(f"  stderr: {result.stderr[:500]}")
                continue
            
            # Parse JSON output
            try:
                data = json.loads(result.stdout)
            except json.JSONDecodeError:
                # Google Benchmark might output extra text before JSON
                json_start = result.stdout.find("{")
                if json_start >= 0:
                    data = json.loads(result.stdout[json_start:])
                else:
                    print(f"\n  ERROR: Could not parse JSON output")
                    continue
            
            # Extract benchmark results
            for bench in data.get("benchmarks", []):
                name = bench.get("name", "unknown")
                # Google Benchmark reports in nanoseconds by default
                time_ns = bench.get("real_time", 0)
                time_unit = bench.get("time_unit", "ns")
                
                # Convert to milliseconds for consistency
                if time_unit == "ns":
                    time_ms = time_ns / 1_000_000
                elif time_unit == "us":
                    time_ms = time_ns / 1_000
                elif time_unit == "ms":
                    time_ms = time_ns
                else:
                    time_ms = time_ns
                
                if name not in all_samples:
                    all_samples[name] = []
                
                # Only collect measurement samples, not warmup
                if not is_warmup:
                    all_samples[name].append(time_ms)
                    
        except subprocess.TimeoutExpired:
            print(f"\n  ERROR: Benchmark timed out")
            continue
        except Exception as e:
            print(f"\n  ERROR: {e}")
            continue
    
    print()  # Newline after progress
    
    # Build result stats
    for name, samples in all_samples.items():
        # Determine appropriate unit based on magnitude
        avg = statistics.mean(samples) if samples else 0
        if avg < 0.001:  # < 1 microsecond
            unit = "ns"
            factor = 1_000_000
        elif avg < 1:  # < 1 millisecond
            unit = "µs"
            factor = 1_000
        else:
            unit = "ms"
            factor = 1
        
        # Convert samples to appropriate unit
        converted_samples = [s * factor for s in samples]
        
        stats = BenchmarkStats(
            name=name,
            unit=unit,
            iterations=iterations,
            warmup_iterations=warmup,
            samples=converted_samples
        )
        stats.calculate_stats()
        results[name] = stats
    
    return results


def run_python_benchmarks(
    warmup: int,
    iterations: int
) -> dict[str, BenchmarkStats]:
    """
    Run Python-level benchmarks (e.g., binding overhead, full pipeline).
    """
    results: dict[str, BenchmarkStats] = {}
    
    try:
        # Try to import the Python module
        import aeon_py.core as aeon_core
    except ImportError:
        print("  ⚠️  aeon_py.core not found - skipping Python benchmarks")
        print("     Install with: pip install -e .")
        return results
    
    # Benchmark: Python binding overhead (create Atlas instance)
    print("\n  Running Python binding benchmarks...")
    
    samples = []
    total_runs = warmup + iterations
    
    for run_idx in range(total_runs):
        is_warmup = run_idx < warmup
        
        # Prepare test data
        test_db = PROJECT_ROOT / "results" / "_bench_temp.bin"
        
        start = time.perf_counter_ns()
        atlas = aeon_core.Atlas(str(test_db))
        # Quick create and destroy
        del atlas
        end = time.perf_counter_ns()
        
        if not is_warmup:
            samples.append((end - start) / 1_000)  # Convert to microseconds
        
        # Cleanup
        if test_db.exists():
            test_db.unlink()
    
    stats = BenchmarkStats(
        name="python_binding_init",
        unit="µs",
        iterations=iterations,
        warmup_iterations=warmup,
        samples=samples
    )
    stats.calculate_stats()
    results["python_binding_init"] = stats
    
    return results


# =============================================================================
# Main Execution
# =============================================================================

def run_all_benchmarks(args: argparse.Namespace) -> BenchmarkResult:
    """Execute all benchmark suites and aggregate results."""
    
    result = BenchmarkResult(
        timestamp=datetime.now().isoformat(),
        platform=get_platform_info(),
        configuration={
            "warmup_iterations": args.warmup,
            "measurement_iterations": args.iterations,
            "noise_threshold": NOISE_THRESHOLD,
            "latency_limit_ms": WARM_SEARCH_LATENCY_LIMIT_MS,
        }
    )
    
    print_header("AEON BENCHMARK VERIFICATION SUITE")
    print(f"  Timestamp: {result.timestamp}")
    print(f"  Platform:  {result.platform['system']} {result.platform['release']}")
    print(f"  CPU:       {result.platform['processor'] or result.platform['machine']}")
    print(f"  Cores:     {result.platform['cpu_count']}")
    
    # Run C++ benchmarks
    print_header("C++ Core Benchmarks", "-")
    try:
        cpp_results = run_cpp_benchmark_gbench(
            warmup=args.warmup,
            iterations=args.iterations
        )
        
        for name, stats in cpp_results.items():
            print_stats(stats)
            result.benchmarks[name] = asdict(stats)
            
            if stats.noise_warning:
                result.warnings.append(
                    f"{name}: High variance detected (stddev/mean > {NOISE_THRESHOLD*100:.0f}%)"
                )
                
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}")
        result.warnings.append(str(e))
    
    # Run Python benchmarks
    print_header("Python Binding Benchmarks", "-")
    try:
        py_results = run_python_benchmarks(
            warmup=args.warmup,
            iterations=args.iterations
        )
        
        for name, stats in py_results.items():
            print_stats(stats)
            result.benchmarks[name] = asdict(stats)
            
            if stats.noise_warning:
                result.warnings.append(
                    f"{name}: High variance detected"
                )
                
    except Exception as e:
        print(f"\n  ERROR: {e}")
        result.warnings.append(f"Python benchmarks failed: {e}")
    
    # Validate constraints
    print_header("CONSTRAINT VALIDATION", "-")
    
    # Check Warm Search latency constraint (Phase 0 requirement)
    warm_search_keys = [k for k in result.benchmarks if "Warm" in k or "warm" in k.lower()]
    
    for key in warm_search_keys:
        bench = result.benchmarks[key]
        # Convert to ms if needed
        p99 = bench.get("p99", 0)
        unit = bench.get("unit", "ms")
        
        if unit == "ns":
            p99_ms = p99 / 1_000_000
        elif unit == "µs":
            p99_ms = p99 / 1_000
        else:
            p99_ms = p99
        
        if p99_ms > WARM_SEARCH_LATENCY_LIMIT_MS:
            result.passed = False
            result.failure_reason = (
                f"FAIL: {key} P99 latency ({p99_ms:.2f}ms) exceeds "
                f"limit ({WARM_SEARCH_LATENCY_LIMIT_MS}ms)"
            )
            print(f"\n  ❌ {result.failure_reason}")
        else:
            print(f"\n  ✓ {key}: P99 = {p99_ms:.2f}ms (limit: {WARM_SEARCH_LATENCY_LIMIT_MS}ms)")
    
    if not warm_search_keys:
        print("\n  ⚠️  No Warm Search benchmark found - skipping latency validation")
    
    # Summary
    print_header("SUMMARY")
    print(f"  Benchmarks run: {len(result.benchmarks)}")
    print(f"  Warnings:       {len(result.warnings)}")
    print(f"  Status:         {'PASSED ✓' if result.passed else 'FAILED ❌'}")
    
    if result.warnings:
        print("\n  Warnings:")
        for w in result.warnings:
            print(f"    ⚠️  {w}")
    
    return result


def save_results(result: BenchmarkResult) -> None:
    """Save benchmark results to JSON file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Convert dataclass to dict
    output_data = {
        "timestamp": result.timestamp,
        "platform": result.platform,
        "configuration": result.configuration,
        "benchmarks": result.benchmarks,
        "warnings": result.warnings,
        "passed": result.passed,
        "failure_reason": result.failure_reason,
    }
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n  Results saved to: {OUTPUT_FILE}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Aeon Benchmark Verification Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--warmup", "-w",
        type=int,
        default=DEFAULT_WARMUP_ITERATIONS,
        help=f"Number of warmup iterations (default: {DEFAULT_WARMUP_ITERATIONS})"
    )
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=DEFAULT_MEASUREMENT_ITERATIONS,
        help=f"Number of measurement iterations (default: {DEFAULT_MEASUREMENT_ITERATIONS})"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick mode: 3 warmup, 10 iterations (for testing)"
    )
    
    args = parser.parse_args()
    
    if args.quick:
        args.warmup = 3
        args.iterations = 10
    
    try:
        result = run_all_benchmarks(args)
        save_results(result)
        
        return 0 if result.passed else 1
        
    except KeyboardInterrupt:
        print("\n\n  Interrupted by user.")
        return 130
    except Exception as e:
        print(f"\n  FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
