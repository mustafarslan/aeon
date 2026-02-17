#!/usr/bin/env node
/**
 * @file bench_bridge.js
 * @brief Empirical microbenchmark for the aeon-node-mac V8→C++ bridge.
 *
 * Measures the TOTAL V8-to-C++ roundtrip latency for:
 *   1. atlasNavigate — SIMD beam search (dim=768, top_k=10)
 *   2. traceAppend   — WAL insert
 *
 * METHODOLOGY:
 *   - Uses process.hrtime.bigint() for nanosecond-precision timing.
 *   - Pre-allocates all buffers BEFORE the benchmark loop (no GC noise).
 *   - 1,000 iteration warmup to stabilize JIT and CPU branch predictors.
 *   - 100,000 iteration measurement for statistical significance.
 *   - Reports Median (P50) and P99 latency in microseconds.
 *
 * EXPECTED RESULTS (M4 Max, INT8 quantization):
 *   - Navigate: Median < 10µs (SLB hit ~3.56µs + FFI overhead ~1µs)
 *   - Trace:    Median < 10µs (WAL append ~2.23µs + FFI overhead ~1µs)
 *
 * @copyright 2024–2026 Aeon Project. All rights reserved.
 */

'use strict';

const { unlinkSync, existsSync } = require('fs');
const path = require('path');

// ─── Configuration ───────────────────────────────────────────────────────────

const DIM = 768;
const QUANT_TYPE = 1;  // INT8_SYMMETRIC
const WARMUP_ITERS = 1_000;
const BENCH_ITERS = 100_000;
const TOP_K = 10;
const SEED_NODES = 100;

const ATLAS_PATH = '/tmp/aeon_bench_atlas.bin';
const TRACE_PATH = '/tmp/aeon_bench_trace.wal';

// Cleanup leftover files from previous runs
const CLEANUP_PATTERNS = [
    ATLAS_PATH,
    ATLAS_PATH + '.wal',
    TRACE_PATH,
    TRACE_PATH.replace('.wal', '') + '.blob',
    TRACE_PATH + '.blob',
];

function cleanup() {
    for (const p of CLEANUP_PATTERNS) {
        try { if (existsSync(p)) unlinkSync(p); } catch (_) { /* ignore */ }
    }
    // Also clean up atlas blob/wal companion files
    for (const ext of ['.wal', '.blob', '.bak']) {
        const f = ATLAS_PATH + ext;
        try { if (existsSync(f)) unlinkSync(f); } catch (_) { /* ignore */ }
    }
}

// ─── Statistics Helpers ──────────────────────────────────────────────────────

function percentile(sorted, p) {
    const idx = Math.ceil(sorted.length * p) - 1;
    return sorted[Math.max(0, idx)];
}

function nsToUs(ns) {
    return Number(ns) / 1000.0;
}

// ─── Main ────────────────────────────────────────────────────────────────────

function main() {
    // ── Load native module ───────────────────────────────────────────────
    let mod;
    try {
        mod = require('./build/Release/aeon_node.node');
    } catch (e) {
        console.error('Failed to load aeon_node.node. Did you run `npm run build`?');
        console.error(e.message);
        process.exit(1);
    }

    console.log(`\n${'═'.repeat(60)}`);
    console.log(`  aeon-node-mac Bridge Benchmark`);
    console.log(`  Aeon SDK Version: ${mod.version}`);
    console.log(`  Dimension: ${DIM}  |  Quantization: INT8  |  Top-K: ${TOP_K}`);
    console.log(`  Warmup: ${WARMUP_ITERS.toLocaleString()} iters  |  Measure: ${BENCH_ITERS.toLocaleString()} iters`);
    console.log(`${'═'.repeat(60)}\n`);

    // ── Cleanup previous benchmark artifacts ─────────────────────────────
    cleanup();

    // ── Initialize AeonDB ────────────────────────────────────────────────
    const db = new mod.AeonDB(ATLAS_PATH, TRACE_PATH, DIM, QUANT_TYPE);

    // ── Pre-allocate query vector (OUTSIDE benchmark loop) ───────────────
    const query = new Float32Array(DIM);
    for (let i = 0; i < DIM; i++) {
        query[i] = (Math.random() - 0.5) * 2.0; // Uniform [-1, 1]
    }

    // ── Seed Atlas with nodes (navigate needs data to search) ────────────
    console.log(`Seeding Atlas with ${SEED_NODES} nodes...`);
    const seedVec = new Float32Array(DIM);
    for (let n = 0; n < SEED_NODES; n++) {
        for (let i = 0; i < DIM; i++) {
            seedVec[i] = (Math.random() - 0.5) * 2.0;
        }
        db.atlasInsert(0n, seedVec, `seed_node_${n}`);
    }
    console.log(`Atlas size: ${db.atlasSize()} nodes\n`);

    // ══════════════════════════════════════════════════════════════════════
    // Benchmark 1: atlasNavigate
    // ══════════════════════════════════════════════════════════════════════

    console.log(`[1/2] Benchmarking atlasNavigate...`);

    // Warmup — stabilize JIT, branch predictors, SLB cache
    for (let i = 0; i < WARMUP_ITERS; i++) {
        db.atlasNavigate(query, TOP_K);
    }

    // Measure
    const navLatencies = new BigInt64Array(BENCH_ITERS);
    for (let i = 0; i < BENCH_ITERS; i++) {
        const t0 = process.hrtime.bigint();
        db.atlasNavigate(query, TOP_K);
        const t1 = process.hrtime.bigint();
        navLatencies[i] = t1 - t0;
    }

    // Sort for percentile calculation
    const navSorted = Array.from(navLatencies).sort((a, b) =>
        a < b ? -1 : a > b ? 1 : 0
    );
    const navMedian = percentile(navSorted, 0.50);
    const navP99 = percentile(navSorted, 0.99);
    const navMin = navSorted[0];
    const navMax = navSorted[navSorted.length - 1];

    console.log(`  Median (P50): ${nsToUs(navMedian).toFixed(2)} µs`);
    console.log(`  P99:          ${nsToUs(navP99).toFixed(2)} µs`);
    console.log(`  Min:          ${nsToUs(navMin).toFixed(2)} µs`);
    console.log(`  Max:          ${nsToUs(navMax).toFixed(2)} µs`);
    console.log();

    // ══════════════════════════════════════════════════════════════════════
    // Benchmark 2: traceAppend
    // ══════════════════════════════════════════════════════════════════════

    console.log(`[2/2] Benchmarking traceAppend...`);

    // Warmup
    for (let i = 0; i < WARMUP_ITERS; i++) {
        db.traceAppend(0, 'warmup event');
    }

    // Measure
    const traceLatencies = new BigInt64Array(BENCH_ITERS);
    for (let i = 0; i < BENCH_ITERS; i++) {
        const t0 = process.hrtime.bigint();
        db.traceAppend(0, 'bench event payload for latency measurement');
        const t1 = process.hrtime.bigint();
        traceLatencies[i] = t1 - t0;
    }

    // Sort for percentile calculation
    const traceSorted = Array.from(traceLatencies).sort((a, b) =>
        a < b ? -1 : a > b ? 1 : 0
    );
    const traceMedian = percentile(traceSorted, 0.50);
    const traceP99 = percentile(traceSorted, 0.99);
    const traceMin = traceSorted[0];
    const traceMax = traceSorted[traceSorted.length - 1];

    console.log(`  Median (P50): ${nsToUs(traceMedian).toFixed(2)} µs`);
    console.log(`  P99:          ${nsToUs(traceP99).toFixed(2)} µs`);
    console.log(`  Min:          ${nsToUs(traceMin).toFixed(2)} µs`);
    console.log(`  Max:          ${nsToUs(traceMax).toFixed(2)} µs`);
    console.log();

    // ══════════════════════════════════════════════════════════════════════
    // Summary
    // ══════════════════════════════════════════════════════════════════════

    const navPass = nsToUs(navMedian) < 10.0;
    const tracePass = nsToUs(traceMedian) < 10.0;

    console.log(`${'─'.repeat(60)}`);
    console.log(`  RESULTS SUMMARY`);
    console.log(`${'─'.repeat(60)}`);
    console.log(`  Navigate Median: ${nsToUs(navMedian).toFixed(2)} µs  ${navPass ? '✅ PASS' : '❌ FAIL'} (<10µs)`);
    console.log(`  Trace Median:    ${nsToUs(traceMedian).toFixed(2)} µs  ${tracePass ? '✅ PASS' : '❌ FAIL'} (<10µs)`);
    console.log(`  Trace Size:      ${db.traceSize().toLocaleString()} events`);
    console.log(`  Atlas Size:      ${db.atlasSize()} nodes`);
    console.log(`${'─'.repeat(60)}`);

    if (navPass && tracePass) {
        console.log(`\n  ✅ ALL LATENCY TARGETS MET — Bridge is production-ready.\n`);
    } else {
        console.log(`\n  ❌ LATENCY TARGETS NOT MET — Investigate FFI overhead.\n`);
        process.exitCode = 1;
    }

    // ── Cleanup ──────────────────────────────────────────────────────────
    db.close();
    cleanup();
}

main();
