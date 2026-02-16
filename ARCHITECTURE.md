# Aeon Memory OS — Architecture

> **Version:** 4.1.0 · **Language Standard:** C++23 · **Target:** Universal (x86-64, ARM64, Edge, Cloud, Game Engines)

---

## 1. System Overview

Aeon is a **persistent, crash-recoverable semantic memory engine** that provides high-dimensional vector search over memory-mapped storage. It is designed for AI agents, game engines, robotics, and any system that needs to *remember, retrieve, and forget* knowledge in real time — including on memory-constrained edge devices.

**V4.1 Frontier Architecture Pillars:**

1. **Dynamic Dimensionality** — The engine supports variable embedding dimensions (384, 768, 1536, etc.) per Atlas file, determined at runtime from the `AtlasHeader`.
2. **Double-Buffered Shadow Compaction** — Stutter-free garbage collection using the Redis BGSAVE pattern. Microsecond pointer swaps enable 60 FPS game engine integration.
3. **Write-Ahead Log (WAL)** — Crash-recoverable durability with a 3-step lock ordering protocol that decouples disk flush latency from RAM updates. Empirically verified < 1% overhead (2.23µs insert with WAL vs 2.23µs without).
4. **Sidecar Blob Arena** — Unlimited-length text storage for `TraceEvent` via an append-only mmap-backed blob file. The 512-byte struct retains a 64-byte inline `text_preview` for zero-touch listings while full text lives in the sidecar.
5. **INT8 Scalar Quantization** — 3.1× file compression (440MB → 141MB at 100K nodes) via symmetric INT8 quantization with NEON SDOT achieving 4.70ns per 768-dim dot product (5.6× math speedup vs FP32). The SLB exclusively stores FP32 vectors (dequantized on-the-fly) to protect the < 5µs cache hit latency.
6. **Multi-Tenant C-API** — A strictly typed C interface where every function accepts a `session_id`, routing queries through the `HierarchicalSLB` sharded semantic cache for isolation.

---

## 2. Core-Shell Paradigm

Aeon adheres to a strict **Core-Shell** separation of concerns:

- **Kernel (Ring 0):** A C++23 shared library (`libaeon.so` / `aeon.dll`) handling all critical path operations: mmap I/O, SIMD vector search, shadow compaction, WAL journaling, INT8 quantization, and concurrency control. It exposes a minimal, opaque C-API.
- **Shell (Ring 3):** A Python orchestration layer (`aeon.py`) that binds to the kernel via `nanobind`. It handles high-level logic, telemetry, and distributed coordination, but **never** touches raw memory or performs heavy computation.

This separation ensures that the performance-critical "hot path" remains in optimized machine code, while the "control plane" retains the flexibility of Python.

---

## 3. Dual-Layer Memory Architecture

Aeon manages two distinct types of persistent memory, both backed by memory-mapped files:

### 3.1 Atlas (Semantic Memory)

- **Purpose:** Spatial index for concept retrieval (RAG).
- **Structure:** A flat tree of `Node` structs (variable stride, quantization-aware).
- **Storage:** `atlas.bin` (mmap) + Delta Buffer (RAM) + `.wal` (crash journal).
- **Key Feature:** `node_byte_stride` is computed from the file header's `dim`, `metadata_size`, and `quantization_type`, allowing a single binary to handle any embedding model at either FP32 or INT8 precision.

### 3.2 Trace (Episodic Memory)

- **Purpose:** Chronological log of agent experiences ("stream of consciousness").
- **Structure:** A linear append-only log of `TraceEvent` structs (512 bytes each).
- **Storage:** `trace.bin` (mmap) + Delta Buffer (RAM) + `trace_blobs_genN.bin` (Sidecar Blob Arena) + `.wal` (crash journal).
- **Key Feature:** Full event text of unlimited length is stored in the Sidecar Blob Arena. Each `TraceEvent` carries a `(blob_offset, blob_size)` pointer into the blob file and a 64-byte `text_preview` for fast listings without touching the sidecar.

---

## 4. Write-Ahead Log (WAL) — Crash Recovery

V4.1 introduces a **Write-Ahead Log** for both Atlas and Trace, providing crash-recoverable durability without sacrificing the hot-path insert latency.

### 4.1 The Problem

Without a WAL, data inserted into the delta buffer but not yet compacted to disk is lost on crash. The naive solution — `fsync()` on every insert — would destroy throughput by coupling every insert to disk latency.

### 4.2 3-Step Lock Ordering Protocol

The WAL uses a dedicated `wal_mutex_` with a strict lock ordering that **decouples disk flush latency from RAM updates**:

```text
Step 1: SERIALIZE (no lock)
  ┌──────────────────────────────────────────┐
  │ Serialize node/event to a byte buffer.   │
  │ Compute FNV-1a checksum of payload.      │
  │ Build WalRecordHeader (16 bytes):        │
  │   record_type + payload_size + checksum  │
  │ No lock held — pure CPU work.            │
  └──────────────────────────────────────────┘
           │
           ▼
Step 2: FLUSH TO WAL (wal_mutex_ only)
  ┌──────────────────────────────────────────┐
  │ Acquire wal_mutex_                       │
  │ Write header + payload to .wal file      │
  │ fflush() — data hits kernel buffer cache │
  │ Release wal_mutex_                       │
  │                                          │
  │ ⚠ This step NEVER holds delta_mutex_     │
  │   → no reader/writer contention on RAM   │
  └──────────────────────────────────────────┘
           │
           ▼
Step 3: APPLY TO RAM (delta_mutex_ only)
  ┌──────────────────────────────────────────┐
  │ Acquire delta_mutex_                     │
  │ memcpy node/event into delta buffer      │
  │ Release delta_mutex_                     │
  │                                          │
  │ ⚠ Disk I/O is already done.             │
  │   RAM update is sub-microsecond.         │
  └──────────────────────────────────────────┘
```

**Lock ordering invariant:** `serialize (no lock) → wal_mutex_ → delta_mutex_`. Since WAL disk I/O is completed before the delta lock is acquired, game engine threads waiting on `navigate()` are never blocked by `fdatasync()` latency.

### 4.3 Crash Recovery

On next open, `replay_wal()` scans the `.wal` file:

1. Read each `WalRecordHeader` (16 bytes).
2. Read the payload (`payload_size` bytes).
3. Recompute FNV-1a checksum. If it doesn't match, discard this record and all subsequent records (torn write).
4. Replay the payload into the delta buffer.

After compaction, `truncate_wal()` deletes the journal file.

### 4.4 Empirical Proof: < 1% Overhead

From `master_metrics.txt` (`bench_wal_overhead`):

| Configuration | Insert (CPU) | Throughput |
|---|---|---|
| WAL disabled (`enable_wal=false`) | 2.23 µs | 447,870 ops/sec |
| WAL enabled (`enable_wal=true`) | 2.23 µs | 449,105 ops/sec |

The overhead is within noise (< 0.3%), confirming that the 3-step lock ordering successfully decouples disk latency from the insert hot path.

---

## 5. Sidecar Blob Arena — Unlimited Text Storage

V4.1 removes the legacy 440-character text ceiling from `TraceEvent` by introducing a **sidecar blob file**.

### 5.1 Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│ TraceEvent (512 bytes, mmap-backed)                         │
│                                                              │
│   id, prev_id, atlas_id, timestamp, role, flags, session_id │
│   blob_offset = 4096 ──┐                                    │
│   blob_size   = 2048   │                                    │
│   text_preview = "The agent observed a thermal anomaly in"  │
│   reserved[364]        │                                    │
└────────────────────────┼────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ trace_blobs_gen0.bin (Sidecar Blob Arena, mmap-backed)      │
│                                                              │
│   [blob_0][blob_1][blob_2]...[blob at offset 4096]          │
│                               ↑                              │
│                               Full text: "The agent observed │
│                               a thermal anomaly in sector 7  │
│                               of the manufacturing floor..." │
│                               (2048 bytes, unlimited)        │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Design Decisions

1. **64-byte inline preview (`text_preview`):** The first 63 characters (+ null terminator) are copied inline into the `TraceEvent`. This enables `ls`-style event listings by reading only the mmap trace file — the blob sidecar is never touched for simple enumeration.
2. **Append-only:** Blobs are never modified in-place. `BlobArena::append()` writes to the current end of the file and returns a `BlobRef{offset, size}`.
3. **Zero-copy reads:** `BlobArena::read()` returns a `std::string_view` directly over the mmap'd region.
4. **Generational GC:** During Trace compaction, only blobs referenced by live (non-tombstoned) events are copied to the new generation blob file (`trace_blobs_gen1.bin`). Dead blobs are garbage-collected.

### 5.3 File Layout

The blob file has **no header and no framing**. It is a raw concatenation of byte blobs. All framing metadata (`offset`, `size`) lives in the corresponding `TraceEvent` struct. This keeps the blob file dead simple and maximizes I/O throughput.

---

## 6. INT8 Scalar Quantization — 4× Spatial Compression

V4.1 adds symmetric INT8 quantization to the Atlas for environments where memory footprint is critical (edge, mobile, large-scale deployments).

### 6.1 Quantization Scheme

```text
scale    = max(|v|) / 127.0f
q[i]     = clamp(round(v[i] / scale), -127, 127)
v'[i]    ≈ q[i] * scale

Dot product (dequantized):
  dot(A, B) ≈ (Σ qa[i] * qb[i]) * scale_A * scale_B
```

- **Symmetric:** `zero_point = 0`, range `[-127, +127]` (not -128, to keep the range symmetric around zero).
- **Per-vector scale:** Stored in `NodeHeader::quant_scale` (4 bytes per node).
- **Safety invariant:** If `max(|v|) == 0`, scale defaults to `1.0f` and output is all zeros (no divide-by-zero).

### 6.2 On-Disk Layout: INT8 vs FP32

| Parameter | FP32 | INT8 |
|---|---|---|
| Centroid storage | `dim × 4` bytes | `dim × 1` byte |
| Node stride (dim=768, meta=256) | 3,392 bytes | 1,088 bytes |
| File size (100K nodes) | 440 MB | 141 MB |
| **Compression ratio** | 1.0× | **3.1×** |

The `quantization_type` field in `AtlasHeader` (offset 0x30) determines the layout. `compute_node_stride(dim, metadata_size, quant_type)` is used at open time to calculate the correct byte stride.

### 6.3 SIMD INT8 Dot Product Performance

From `master_metrics.txt` (`bench_quantization`):

| Kernel | Time (768-dim) | Throughput |
|---|---|---|
| FP32 Cosine (NEON) | 26.5 ns | 37.8M/s |
| INT8 Scalar (baseline) | 15.3 ns | 65.5M/s |
| INT8 NEON SDOT (best) | 4.44 ns | 225.5M/s |
| **INT8 Dequantize-on-the-fly** | **4.70 ns** | **212.9M/s** |

The NEON SDOT kernel achieves a **5.6× math speedup** over FP32 cosine similarity (4.70ns vs 26.5ns). The dequantize-on-the-fly variant (`BM_INT8_DotDequantize`) adds only 0.26ns over raw SDOT for the scale multiplication — negligible.

### 6.4 Architectural Decision: FP32 in the SLB

**Critical design choice:** The `HierarchicalSLB` (Semantic Lookaside Buffer) stores **exclusively FP32 vectors**, even when the Atlas is INT8-quantized.

**Rationale:**

The SLB cache hit path must remain < 5µs (our empirical target: 3.56µs median). Storing INT8 in the SLB would mean dequantizing on every cache hit — adding unnecessary latency to the hottest path. Since the SLB holds at most 64 entries per session (L1) and 256 globally (L2), the FP32 memory overhead is negligible (64 × 768 × 4 = ~192KB per session). Vectors are dequantized once on initial insertion into the SLB and served as FP32 thereafter.

This ensures the cache hit latency remains at the empirically verified **3.56µs** regardless of the on-disk quantization format.

---

## 7. Navigate Query Flow

The `navigate` operation is the primary mechanism for retrieving knowledge. It executes a **SIMD-accelerated beam search** with quantization-aware dispatch:

1. **L1 Cache Lookup:** The query is first checked against the `SessionRingBuffer` (L1) for the specific `session_id`. FP32 similarity always — SLB stores FP32 exclusively.
2. **L2 Cache Lookup:** If L1 misses, the global `SemanticCache` (L2) is checked.
3. **Atlas Search:** If both caches miss, the kernel performs a beam search on the Atlas:
    - **Dispatch:** If `quantization_type == QUANT_INT8_SYMMETRIC`, the query is quantized once, and the INT8 SIMD kernel (`dot_int8_neon` / `dot_int8_avx512`) is used. The raw `int32` accumulator is dequantized via `scale_query × scale_node`.
    - **Roots:** Starts at root nodes.
    - **Expansion:** Computes similarity to children using the dispatched kernel (FP32 cosine or INT8 SDOT).
    - **Pruning:** Keeps the top-K candidates (beam width).
    - **Tombstones:** Dead nodes are branchlessly penalized (score ≈ -1e9) to exclude them without conditional jumps.
4. **Delta Scan:** Simultaneously, the delta buffer (recent inserts) is scanned.
5. **Merge:** Results from mmap and delta are merged.
6. **SLB Insert:** Results are inserted into the SLB as FP32 (dequantized if INT8 source).

---

## 8. Double-Buffered Shadow Compaction

To support real-time applications (game engines at 60 FPS), Aeon V4.1 implements a **stutter-free garbage collection** mechanism inspired by Redis `BGSAVE`.

**The 4-Step Process:**

1. **Microsecond Freeze:** The kernel acquires a lock, swaps the active `delta_buffer` with a `frozen_delta_buffer`, and snapshots the current state. This takes < 10µs.
2. **Background Copy:** A background thread iterates over the live (non-tombstoned) nodes in the mmap file and the frozen delta buffer, writing them contiguously to a new generation file (`atlas_gen2.bin`). The main thread continues to serve reads and accepts new writes into the *fresh* delta buffer.
3. **Hot Swap:** Once the copy is complete, the kernel briefly locks again to swap the `MemoryFile` handle to the new file.
4. **Cleanup:** The old generation file is closed and deleted. The WAL is truncated (all data is now durably on disk in the new generation file).

This ensures that the main thread is never blocked by I/O proportional to the dataset size.

---

## 9. Concurrency: Epoch-Based Reclamation (EBR)

Aeon uses **Epoch-Based Reclamation (EBR)** to guarantee memory safety during concurrent mmap operations without heavy reader locks.

- **Readers:** Enter a "read epoch" by writing the global epoch counter to a thread-local atomic slot. This signals to writers that they are active.
- **Writers:** When a file grows (unmap/remap), the old memory region is "retired" but not freed.
- **Reclamation:** A background process frees retired regions only when *all* active readers have advanced past the epoch in which the region was retired.

This lock-free read path ensures that readers never segfault on unmapped memory, even if a writer resizes the file mid-read. Empirically verified: **P99 read latency = 750ns** under hostile 15-reader / 1-writer contention (`bench_ebr_contention`).

---

## 10. Hierarchical SLB (Semantic Lookaside Buffer)

Structure for multi-tenant scalability (100,000+ sessions):

- **L1: Sharded Session Map:** A hash map split into 64 lock-striped shards. Each shard stores `SessionRingBuffer` instances (size 64) for active sessions.
- **L2: Global Cache:** A shared ring buffer (size 256) for cold-start acceleration and cross-session common concepts.
- **Lifecycle:**
  - **Insert:** Updates L1 (session) and L2 (global). For INT8 Atlas files, vectors are dequantized to FP32 before SLB insertion.
  - **Drop:** `aeon_atlas_drop_session` removes the L1 buffer.
  - **Eviction:** LRU policy evicts oldest sessions when a shard fills up.
- **Empirical:** Cache hit latency = **3.56µs** (median), cache isolation verified across 64 shards.

---

## 11. Tiered Edge-Cloud Architecture

For memory-constrained devices (mobile, IoT), Aeon runs in a **Tiered Mode**:

- **RAM Budget:** The `TieredAtlas` is configured with a maximum resident memory budget (e.g., 512MB). INT8 quantization extends the effective capacity by 3.1×.
- **Cold Miss Detection:** If the best local search result has a similarity score below `cold_miss_threshold`, the result is flagged with `requires_cloud_fetch = true`.
- **Orchestration:** The Python Shell detects this flag and routes the query to a Cloud Master Atlas for a full-fidelity search.

---

## 12. Trace Block Index (TBI)

Trace search uses a **Chronological Block Index** to achieve sub-linear time pattern retrieval `O(|V|/1024 + K*1024)`:

- **Structure:** Trace events are grouped into `TraceBlock`s of 1024 events.
- **Centroids:** Each block maintains an incrementally updated centroid of its event embeddings.
- **Two-Phase Search:**
    1. **Block Scan:** SIMD scan of block centroids to find the top-K most relevant time windows.
    2. **Event Scan:** Deep scan of *only* the events within those top-K blocks.

This allows Aeon to search huge episodic timelines (100M+ events) in milliseconds without indexing overhead.

---

## 13. C-API Surface

The kernel exposes a pure C interface (`extern "C"`) in `aeon_c_api.h` ensuring binary compatibility across languages (Python, C#, C++, Swift).

- **Opaque Pointers:** Types like `aeon_atlas_t*` and `aeon_trace_t*` hide C++ implementation details.
- **Caller Allocation:** The caller owns all memory. Result buffers are pre-allocated by the caller and passed to the kernel to fill.
- **Error Codes:** Functions return `aeon_error_t` enums, never exceptions.
- **V4.1 Extensions:**
  - `aeon_atlas_create_ex()` — Accepts `aeon_atlas_options_t` with `quantization_type` and `enable_wal` fields.
  - `aeon_trace_get_event_text()` — Retrieves full text from the Sidecar Blob Arena via caller-allocated buffer.

```c
// V4.1 Extended creation with quantization + WAL
aeon_atlas_options_t opts = {
    .dim = 768,
    .quantization_type = 1,  // INT8_SYMMETRIC
    .enable_wal = 1
};
aeon_atlas_t *atlas = NULL;
aeon_atlas_create_ex("memory.bin", &opts, &atlas);
```
