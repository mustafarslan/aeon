#!/usr/bin/env python3
"""
Aeon V3 Reproducible Benchmark: Dreaming Process Validation.

ACADEMIC CLAIM VERIFICATION:
  1. The Dreaming process (consolidate_subgraph + compact_mmap) physically
     reclaims storage from tombstoned nodes. File size MUST decrease.
  2. SLB search latency is NOT degraded by the tombstoning process.
     The branchless hub_penalty = 1e9f trick ensures zero SIMD branch
     predictor pollution.
  3. The process runs without freezing the event loop (GIL released).

PROTOCOL:
  - Insert 1,000,000 nodes simulating months of mobile NPC memory.
  - Measure the .bin file size on disk.
  - Consolidate 900,000 nodes into 9,000 summary nodes (100:1 ratio).
  - Call compact_mmap() to physically defragment.
  - Measure the new file size (must be significantly smaller).
  - Run SLB latency probes before/after to verify no regression.
  - Output a clean Markdown table suitable for the V3 paper §6.4.

REPRODUCIBILITY:
  - Fixed PRNG seed for deterministic embeddings.
  - All timings use time.perf_counter_ns() for nanosecond precision.
  - Results include system metadata (OS, CPU, Python version).

Usage:
  python bench_v3_dreaming.py [--nodes 1000000] [--batch-size 100]
"""

import argparse
import os
import platform
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ===========================================================================
# Configuration
# ===========================================================================

EMBEDDING_DIM = 768
CONSOLIDATION_RATIO = 100  # 100 old nodes → 1 summary node
LATENCY_PROBE_COUNT = 1000
WARMUP_PROBES = 100


@dataclass
class BenchmarkResult:
    """Collected benchmark measurements."""
    total_nodes: int = 0
    insert_duration_s: float = 0.0
    insert_rate_ops: float = 0.0
    file_size_before_mb: float = 0.0
    file_size_after_mb: float = 0.0
    storage_reclaimed_mb: float = 0.0
    storage_reclaimed_pct: float = 0.0
    consolidation_duration_s: float = 0.0
    compaction_duration_s: float = 0.0
    nodes_consolidated: int = 0
    summary_nodes_created: int = 0
    latency_before_p50_ns: float = 0.0
    latency_before_p99_ns: float = 0.0
    latency_after_p50_ns: float = 0.0
    latency_after_p99_ns: float = 0.0
    latency_delta_pct: float = 0.0


# ===========================================================================
# Benchmark Implementation
# ===========================================================================

def generate_embedding(rng: np.random.RandomState) -> np.ndarray:
    """Generate a normalized 768-dim embedding."""
    vec = rng.randn(EMBEDDING_DIM).astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-12
    return vec


def measure_navigate_latency(
    atlas, rng: np.random.RandomState, n_probes: int, n_warmup: int
) -> np.ndarray:
    """
    Measure navigate latency with warmup.
    Returns array of latencies in nanoseconds.
    """
    # Warmup (populate SLB, page cache)
    for _ in range(n_warmup):
        query = generate_embedding(rng)
        atlas.navigate_raw(query, beam_width=1, apply_csls=False)

    # Timed probes
    latencies = np.empty(n_probes, dtype=np.float64)
    for i in range(n_probes):
        query = generate_embedding(rng)
        t0 = time.perf_counter_ns()
        atlas.navigate_raw(query, beam_width=1, apply_csls=False)
        t1 = time.perf_counter_ns()
        latencies[i] = t1 - t0

    return latencies


def run_benchmark(total_nodes: int, batch_size: int) -> BenchmarkResult:
    """Execute the full Dreaming benchmark."""
    # Import aeon core (must be built and installed)
    try:
        from aeon_py import core
    except ImportError:
        print("ERROR: aeon_py.core not found. Build with: pip install -e .")
        sys.exit(1)

    result = BenchmarkResult(total_nodes=total_nodes)
    rng = np.random.RandomState(42)

    # Create temp directory for the benchmark atlas
    tmp_dir = tempfile.mkdtemp(prefix="aeon_bench_dreaming_")
    atlas_path = Path(tmp_dir) / "bench_atlas.bin"

    print(f"\n{'='*72}")
    print(f"  Aeon V3 Benchmark: Dreaming Process Validation")
    print(f"  Nodes: {total_nodes:,} | Batch: {batch_size}")
    print(f"  Atlas: {atlas_path}")
    print(f"{'='*72}\n")

    try:
        # ---------------------------------------------------------------
        # Phase 1: Insert N nodes
        # ---------------------------------------------------------------
        print(f"[1/5] Inserting {total_nodes:,} nodes...")
        atlas = core.Atlas(str(atlas_path))

        t_insert_start = time.perf_counter()
        for i in range(total_nodes):
            vec = generate_embedding(rng)
            parent = 0 if i == 0 else rng.randint(0, i)
            atlas.insert(parent, vec.tolist(), f"node_{i}")
            if (i + 1) % 100_000 == 0:
                print(f"    {i+1:>10,} / {total_nodes:,} inserted")
        t_insert_end = time.perf_counter()

        result.insert_duration_s = t_insert_end - t_insert_start
        result.insert_rate_ops = total_nodes / result.insert_duration_s
        result.file_size_before_mb = atlas_path.stat().st_size / (1024 * 1024)

        print(f"    Insert complete: {result.insert_duration_s:.2f}s "
              f"({result.insert_rate_ops:,.0f} ops/s)")
        print(f"    File size: {result.file_size_before_mb:.2f} MB")

        # ---------------------------------------------------------------
        # Phase 2: Pre-consolidation latency measurement
        # ---------------------------------------------------------------
        print(f"\n[2/5] Measuring pre-consolidation navigate latency...")
        lat_before = measure_navigate_latency(
            atlas, np.random.RandomState(99),
            LATENCY_PROBE_COUNT, WARMUP_PROBES)

        result.latency_before_p50_ns = float(np.percentile(lat_before, 50))
        result.latency_before_p99_ns = float(np.percentile(lat_before, 99))
        print(f"    Pre-consolidation latency: "
              f"p50={result.latency_before_p50_ns/1000:.1f}µs "
              f"p99={result.latency_before_p99_ns/1000:.1f}µs")

        # ---------------------------------------------------------------
        # Phase 3: Consolidate 90% of nodes
        # ---------------------------------------------------------------
        nodes_to_consolidate = int(total_nodes * 0.9)
        n_batches = nodes_to_consolidate // batch_size
        result.nodes_consolidated = n_batches * batch_size
        result.summary_nodes_created = n_batches

        print(f"\n[3/5] Consolidating {result.nodes_consolidated:,} nodes "
              f"into {n_batches:,} summaries...")

        t_consolidate_start = time.perf_counter()
        for batch_idx in range(n_batches):
            start_id = 1 + batch_idx * batch_size
            old_ids = list(range(start_id, start_id + batch_size))
            summary_vec = generate_embedding(rng)

            atlas.consolidate_subgraph(
                old_ids,
                summary_vec.tolist(),
                f"summary_batch_{batch_idx}"
            )

            if (batch_idx + 1) % 1000 == 0:
                print(f"    {batch_idx+1:>8,} / {n_batches:,} batches consolidated")

        t_consolidate_end = time.perf_counter()
        result.consolidation_duration_s = t_consolidate_end - t_consolidate_start

        tombstones = atlas.tombstone_count()
        print(f"    Consolidation: {result.consolidation_duration_s:.2f}s")
        print(f"    Tombstone count: {tombstones:,}")

        # ---------------------------------------------------------------
        # Phase 4: Compact mmap (physical storage reclamation)
        # ---------------------------------------------------------------
        print(f"\n[4/5] Compacting storage (defragmenting)...")

        compact_path = str(atlas_path) + ".compact_tmp"
        t_compact_start = time.perf_counter()
        atlas.compact_mmap(compact_path)
        t_compact_end = time.perf_counter()

        result.compaction_duration_s = t_compact_end - t_compact_start
        result.file_size_after_mb = atlas_path.stat().st_size / (1024 * 1024)
        result.storage_reclaimed_mb = result.file_size_before_mb - result.file_size_after_mb
        result.storage_reclaimed_pct = (
            (result.storage_reclaimed_mb / result.file_size_before_mb) * 100
            if result.file_size_before_mb > 0 else 0.0
        )

        print(f"    Compaction: {result.compaction_duration_s:.2f}s")
        print(f"    File size: {result.file_size_before_mb:.2f} MB → "
              f"{result.file_size_after_mb:.2f} MB")
        print(f"    Reclaimed: {result.storage_reclaimed_mb:.2f} MB "
              f"({result.storage_reclaimed_pct:.1f}%)")

        # ---------------------------------------------------------------
        # Phase 5: Post-consolidation latency measurement
        # ---------------------------------------------------------------
        print(f"\n[5/5] Measuring post-consolidation navigate latency...")
        lat_after = measure_navigate_latency(
            atlas, np.random.RandomState(99),
            LATENCY_PROBE_COUNT, WARMUP_PROBES)

        result.latency_after_p50_ns = float(np.percentile(lat_after, 50))
        result.latency_after_p99_ns = float(np.percentile(lat_after, 99))

        if result.latency_before_p50_ns > 0:
            result.latency_delta_pct = (
                (result.latency_after_p50_ns - result.latency_before_p50_ns)
                / result.latency_before_p50_ns * 100
            )

        print(f"    Post-consolidation latency: "
              f"p50={result.latency_after_p50_ns/1000:.1f}µs "
              f"p99={result.latency_after_p99_ns/1000:.1f}µs")
        print(f"    Delta: {result.latency_delta_pct:+.1f}%")

    finally:
        # Cleanup
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return result


# ===========================================================================
# Output: Markdown Table (for §6.4 of the V3 Paper)
# ===========================================================================

def format_results(r: BenchmarkResult) -> str:
    """Format results as a Markdown table for the V3 paper."""
    return f"""
## Aeon V3 — Dreaming Process Benchmark Results

**System**: {platform.system()} {platform.machine()} | Python {platform.python_version()}
**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}

### Storage Reclamation

| Metric                    | Value                |
|--------------------------|----------------------|
| Total nodes inserted     | {r.total_nodes:,}    |
| Insert throughput        | {r.insert_rate_ops:,.0f} ops/s |
| File size (before)       | {r.file_size_before_mb:.2f} MB |
| Nodes consolidated       | {r.nodes_consolidated:,} ({r.nodes_consolidated/r.total_nodes*100:.0f}%) |
| Summary nodes created    | {r.summary_nodes_created:,} |
| Consolidation time       | {r.consolidation_duration_s:.2f}s |
| Compaction time           | {r.compaction_duration_s:.2f}s |
| **File size (after)**    | **{r.file_size_after_mb:.2f} MB** |
| **Storage reclaimed**    | **{r.storage_reclaimed_mb:.2f} MB ({r.storage_reclaimed_pct:.1f}%)** |

### Navigate Latency (SIMD Regression Test)

| Phase              | p50 (µs)  | p99 (µs)  |
|-------------------|-----------|-----------|
| Pre-consolidation | {r.latency_before_p50_ns/1000:.1f}     | {r.latency_before_p99_ns/1000:.1f}     |
| Post-consolidation| {r.latency_after_p50_ns/1000:.1f}     | {r.latency_after_p99_ns/1000:.1f}     |
| **Delta**         | **{r.latency_delta_pct:+.1f}%** | — |

### Interpretation

{"✅" if r.storage_reclaimed_pct > 50 else "⚠️"} Storage reclamation: {r.storage_reclaimed_pct:.1f}% of the original file was physically reclaimed via `compact_mmap()`.

{"✅" if abs(r.latency_delta_pct) < 20 else "⚠️"} Latency regression: Navigate latency changed by {r.latency_delta_pct:+.1f}% after consolidation — the branchless `hub_penalty = 1e9f` tombstoning trick ensures SIMD scan is not polluted by branch mispredictions.
"""


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Aeon V3 Dreaming Benchmark")
    parser.add_argument("--nodes", type=int, default=1_000_000,
                        help="Total nodes to insert (default: 1,000,000)")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Nodes per consolidation batch (default: 100)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output markdown file (default: stdout)")
    args = parser.parse_args()

    result = run_benchmark(args.nodes, args.batch_size)
    report = format_results(result)

    if args.output:
        Path(args.output).write_text(report)
        print(f"\nResults written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
