#!/usr/bin/env python3
"""
Aeon CI latency-regression gate.

Compares a curated subset of benchmark binaries between two already-built
CMake build directories (typically merge-base vs. HEAD, built on the SAME
runner in the SAME CI job) and fails (non-zero exit) if any benchmark
regresses beyond a relative tolerance.

Deliberately NOT an absolute-threshold check against a recorded baseline
file (e.g. reproducibility_benchmarks/master_metrics.txt): those numbers
were captured on an M4 Max with active cooling and comparing hosted CI
runners against them at microsecond/nanosecond scale flakes and gets
disabled. Instead this script builds both revisions on the same runner and
compares them to each other, which cancels out most runner-to-runner noise
while still catching an algorithmic regression (e.g. an O(1) -> O(N) scan).

Curated benchmark subset (see v4-plan.md, guardrail #0):
  - bench_kernel_throughput      (Google Benchmark, JSON)
  - bench_wal_overhead           (Google Benchmark, JSON)
  - bench_quantization_efficiency(Google Benchmark, JSON)
  - bench_beam_search            (custom main(), plain-text P50/P90/P99)
  - bench_ebr_contention         (custom main(), already gates itself on
                                   an absolute P99 < 10us threshold and
                                   returns a real exit code -- that
                                   threshold is coarse/generous by design
                                   and is not expected to flake, so it is
                                   run once against HEAD only rather than
                                   folded into the relative-diff table)

Usage:
  python3 scripts/ci_perf_gate.py \\
      --baseline-dir build/ci-baseline --head-dir build/ci-linux \\
      --tolerance 0.25
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# Google-Benchmark (JSON-capable) binaries in the curated per-PR gate set.
JSON_BENCHMARKS = [
    "bench_kernel_throughput",
    "bench_wal_overhead",
    "bench_quantization_efficiency",
]

# Benchmark-name substrings that are known SLB self-hit artifacts (query
# vector is a bit-for-bit copy of a stored node, cosine == 1.0) rather than
# real traversal measurements -- see v4-plan.md guardrail #0. Extend this
# list before adding any new benchmark to JSON_BENCHMARKS/BEAM_SEARCH_LABELS.
#
# As of Stage 2 task 5's benchmark audit, every known self-hit artifact
# (BM_AtlasTraversal_Only among them -- it lives in bench_slb_latency.cpp,
# not bench_scalability.cpp as an earlier version of this comment said) has
# been FIXED at the source (query-pool cycling + a matching ->Iterations()
# cap, so navigate()'s own internal SLB cache can never serve a repeat).
# This list is intentionally left empty rather than deleted: it's the
# checkpoint to extend if a *future* benchmark reintroduces the same
# static-query-reused-across-every-iteration pattern.
SKIP_NAME_SUBSTRINGS = []

# bench_beam_search prints plain text per configuration; these are the
# config labels it currently emits (core/benchmarks/bench_beam_search.cpp).
BEAM_SEARCH_LABELS = [
    "beam_width=1 (Greedy)",
    "beam_width=3",
    "beam_width=3 + CSLS",
]

METRIC = "real_time"  # Google Benchmark field to compare (wall-clock, ns)


def run_json_benchmark(binary: Path, out_json: Path) -> dict:
    subprocess.run(
        [str(binary), "--benchmark_format=json", f"--benchmark_out={out_json}",
         "--benchmark_out_format=json"],
        check=True, capture_output=True, text=True,
    )
    data = json.loads(out_json.read_text())
    results = {}
    for b in data.get("benchmarks", []):
        name = b["name"]
        if any(s in name for s in SKIP_NAME_SUBSTRINGS):
            continue
        run_type = b.get("run_type", "iteration")
        if run_type == "iteration":
            # No --benchmark_repetitions: a single real run per benchmark.
            results[name] = b[METRIC]
        elif run_type == "aggregate":
            # --benchmark_repetitions>1 emits only aggregate rows (mean,
            # median, stddev, cv) with no raw per-iteration rows. Compare
            # on the median -- robust to outliers, which repetitions exist
            # to smooth out in the first place.
            if b.get("aggregate_name") == "median":
                results[name] = b[METRIC]
        # else: unknown run_type, skip rather than silently comparing
        # something unexpected.
    return results


def run_beam_search(binary: Path) -> dict:
    proc = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    out = proc.stdout
    results = {}
    # Each config block looks like:
    #   Benchmarking beam_width=3...
    #     Mean:   12.3 µs
    #     P50:     9.1 µs
    #     P90:    15.2 µs
    #     P99:    22.0 µs
    blocks = re.split(r"Benchmarking ", out)[1:]
    for block in blocks:
        header, _, rest = block.partition("...\n")
        label = header.strip()
        if label not in BEAM_SEARCH_LABELS:
            continue
        m = re.search(r"P50:\s+([\d.]+)\s*\xb5s", rest)
        if m:
            results[f"beam_search::{label}::P50"] = float(m.group(1))
    return results


def collect(build_dir: Path) -> dict:
    bin_dir = build_dir / "bin"
    metrics = {}
    for name in JSON_BENCHMARKS:
        binary = bin_dir / name
        if not binary.exists():
            print(f"WARN: {binary} not found, skipping", file=sys.stderr)
            continue
        out_json = build_dir / f"{name}.result.json"
        metrics.update(run_json_benchmark(binary, out_json))

    beam_binary = bin_dir / "bench_beam_search"
    if beam_binary.exists():
        metrics.update(run_beam_search(beam_binary))
    else:
        print(f"WARN: {beam_binary} not found, skipping", file=sys.stderr)

    return metrics


def check_ebr_contention(build_dir: Path) -> bool:
    """bench_ebr_contention already gates itself on an absolute P99 < 10us
    threshold and returns a real exit code -- run it once against HEAD and
    surface its own verdict rather than duplicating the threshold here."""
    binary = build_dir / "bin" / "bench_ebr_contention"
    if not binary.exists():
        print(f"WARN: {binary} not found, skipping EBR contention check", file=sys.stderr)
        return True
    proc = subprocess.run([str(binary)], capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print("FAIL: bench_ebr_contention exceeded its own absolute P99 threshold (>10us)",
              file=sys.stderr)
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-dir", required=True, type=Path,
                     help="CMake build dir for the baseline revision (e.g. merge-base)")
    ap.add_argument("--head-dir", required=True, type=Path,
                     help="CMake build dir for the revision under test (e.g. HEAD)")
    ap.add_argument("--tolerance", type=float, default=0.25,
                     help="Relative regression tolerance (default 0.25 = 25%%)")
    args = ap.parse_args()

    print(f"Collecting baseline metrics from {args.baseline_dir} ...")
    baseline = collect(args.baseline_dir)
    print(f"Collecting head metrics from {args.head_dir} ...")
    head = collect(args.head_dir)

    common = sorted(set(baseline) & set(head))
    missing_baseline = sorted(set(head) - set(baseline))
    missing_head = sorted(set(baseline) - set(head))
    if missing_baseline:
        print(f"NOTE: new benchmarks not present in baseline (not gated): {missing_baseline}")
    if missing_head:
        print(f"WARN: benchmarks present in baseline but missing from head: {missing_head}",
              file=sys.stderr)

    ok = True
    print(f"\n{'Benchmark':<55} {'Baseline':>12} {'Head':>12} {'Delta':>8}  Verdict")
    print("-" * 100)
    for name in common:
        b, h = baseline[name], head[name]
        delta = (h - b) / b if b else 0.0
        verdict = "PASS"
        if delta > args.tolerance:
            verdict = "FAIL"
            ok = False
        print(f"{name:<55} {b:>12.1f} {h:>12.1f} {delta:>+7.1%}  {verdict}")

    print()
    if not check_ebr_contention(args.head_dir):
        ok = False

    if not ok:
        print("\nCI PERF GATE: FAIL -- one or more benchmarks regressed beyond "
              f"{args.tolerance:.0%} tolerance vs. merge-base, or an absolute "
              "safety threshold was exceeded.", file=sys.stderr)
        return 1

    print("\nCI PERF GATE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
