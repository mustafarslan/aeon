# Aeon Memory OS — Architecture

> **Version:** 4.0.0 · **Language Standard:** C++23 · **Target:** Universal (x86-64, ARM64, Edge, Cloud, Game Engines)

---

## 1. System Overview

Aeon is a **persistent, crash-recoverable semantic memory engine** that provides high-dimensional vector search over memory-mapped storage. It is designed for AI agents, game engines, robotics, and any system that needs to *remember, retrieve, and forget* knowledge in real time — including on memory-constrained edge devices.

**V4.0 Enterprise Architecture Pillars:**

1. **Dynamic Dimensionality** — The engine supports variable embedding dimensions (384, 768, 1536, etc.) per Atlas file, determined at runtime from the `AtlasHeader`.
2. **Double-Buffered Shadow Compaction** — Stutter-free garbage collection using the Redis BGSAVE pattern. Microsecond pointer swaps enable 60 FPS game engine integration.
3. **Binary C++ MMap Trace** — Episodic memory is stored as a fixed-size 512-byte C++ binary struct (`TraceEvent`) in a memory-mapped file, enabling zero-copy access and O(1) page-in.
4. **Multi-Tenant C-API** — A strictly typed C interface where every function accepts a `session_id`, routing queries through the `HierarchicalSLB` sharded semantic cache for isolation.

---

## 2. Core-Shell Paradigm

Aeon adheres to a strict **Core-Shell** separation of concerns:

- **Kernel (Ring 0):** A C++23 shared library (`libaeon.so` / `aeon.dll`) handling all critical path operations: mmap I/O, SIMD vector search, shadow compaction, and concurrency control. It exposes a minimal, opaque C-API.
- **Shell (Ring 3):** A Python orchestration layer (`aeon.py`) that binds to the kernel via `nanobind`. It handles high-level logic, telemetry, and distributed coordination, but **never** touches raw memory or performs heavy computation.

This separation ensures that the performance-critical "hot path" remains in optimized machine code, while the "control plane" retains the flexibility of Python.

---

## 3. Dual-Layer Memory Architecture

Aeon manages two distinct types of persistent memory, both backed by memory-mapped files:

### 3.1 Atlas (Semantic Memory)

- **Purpose:** Spatial index for concept retrieval (RAG).
- **Structure:** A flat tree of `Node` structs (variable stride).
- **Storage:** `atlas.bin` (mmap) + Delta Buffer (RAM).
- **Key Feature:** Nodes are accessed via `node_byte_stride` calculated from the file header, allowing a single binary to handle any embedding model.

### 3.2 Trace (Episodic Memory)

- **Purpose:** Chronological log of agent experiences ("stream of consciousness").
- **Structure:** A linear append-only log of `TraceEvent` structs (512 bytes).
- **Storage:** `trace.bin` (mmap) + Delta Buffer (RAM).
- **Key Feature:** Zero heap usage. Fixed-size C++ structs allow direct `memcpy` to/from disk, eliminating serialization overhead.

---

## 4. Navigate Query Flow

The `navigate` operation is the primary mechanism for retrieving knowledge. It executes a **SIMD-accelerated beam search**:

1. **L1 Cache Lookup:** The query is first checked against the `SessionRingBuffer` (L1) for the specific `session_id`.
2. **L2 Cache Lookup:** If L1 misses, the global `SemanticCache` (L2) is checked.
3. **Atlas Search:** If both caches miss, the kernel performs a beam search on the Atlas:
    - **Roots:** Starts at root nodes.
    - **Expansion:** Computes cosine similarity to children using AVX-512/NEON.
    - **Pruning:** Keeps the top-K candidates (beam width).
    - **Tombstones:** Dead nodes are branchlessly penalized (score ≈ -1e9) to exclude them without conditional jumps.
4. **Delta Scan:** Simultaneously, the delta buffer (recent inserts) is scanned.
5. **Merge:** Results from mmap and delta are merged and returned.

---

## 5. Double-Buffered Shadow Compaction

To support real-time applications (game engines at 60 FPS), Aeon V4.0 implements a **stutter-free garbage collection** mechanism inspired by Redis `BGSAVE`.

**The 4-Step Process:**

1. **Microsecond Freeze:** The kernel acquires a lock, swaps the active `delta_buffer` with a `frozen_delta_buffer`, and snapshots the current state. This takes < 10µs.
2. **Background Copy:** A background thread iterates over the live (non-tombstoned) nodes in the mmap file and the frozen delta buffer, writing them contiguously to a new generation file (`atlas.gen2.bin`). The main thread continues to serve reads and accepts new writes into the *fresh* delta buffer.
3. **Hot Swap:** Once the copy is complete, the kernel briefly locks again to swap the `MemoryFile` handle to the new file.
4. **Cleanup:** The old generation file is closed and deleted.

This ensures that the main thread is never blocked by I/O proportional to the dataset size.

---

## 6. Concurrency: Epoch-Based Reclamation (EBR)

Aeon uses **Epoch-Based Reclamation (EBR)** to guarantee memory safety during concurrent mmap operations without heavy reader locks.

- **Readers:** Enter a "read epoch" by writing the global epoch counter to a thread-local atomic slot. This signals to writers that they are active.
- **Writers:** When a file grows (unmap/remap), the old memory region is "retired" but not freed.
- **Reclamation:** A background process frees retired regions only when *all* active readers have advanced fast the epoch in which the region was retired.

This lock-free read path ensures that readers never segfault on unmapped memory, even if a writer resizes the file mid-read.

---

## 7. Hierarchical SLB (Semantic Lookaside Buffer)

Structure for multi-tenant scalability (100,000+ sessions):

- **L1: Sharded Session Map:** A hash map split into 64 lock-striped shards. Each shard stores `SessionRingBuffer` instances (size 64) for active sessions.
- **L2: Global Cache:** A shared ring buffer (size 256) for cold-start acceleration and cross-session common concepts.
- **Lifecycle:**
  - **Insert:** Updates L1 (session) and L2 (global).
  - **Drop:** `aeon_atlas_drop_session` removes the L1 buffer.
  - **Eviction:** LRU policy evicts oldest sessions when a shard fills up.

---

## 8. Tiered Edge-Cloud Architecture

For memory-constrained devices (mobile, IoT), Aeon runs in a **Tiered Mode**:

- **RAM Budget:** The `TieredAtlas` is configured with a maximum resident memory budget (e.g., 512MB).
- **Cold Miss Detection:** If the best local search result has a similarity score below `cold_miss_threshold`, the result is flagged with `requires_cloud_fetch = true`.
- **Orchestration:** The Python Shell detects this flag and routes the query to a Cloud Master Atlas for a full-fidelity search.

---

## 9. Trace Block Index (TBI)

Trace search uses a **Chronological Block Index** to achieve sub-linear time pattern retrieval `O(|V|/1024 + K*1024)`:

- **Structure:** Trace events are grouped into `TraceBlock`s of 1024 events.
- **Centroids:** Each block maintains an incrementally updated centroid of its event embeddings.
- **Two-Phase Search:**
    1. **Block Scan:** SIMD scan of block centroids to find the top-K most relevant time windows.
    2. **Event Scan:** Deep scan of *only* the events within those top-K blocks.

This allows Aeon to search huge episodic timelines (100M+ events) in milliseconds without indexing overhead.

---

## 10. C-API Surface

The kernel exposes a pure C interface (`extern "C"`) in `aeon_c_api.h` ensures binary compatibility across languages (Python, C#, C++, Swift).

- **Opaque Pointers:** Types like `aeon_atlas_t*` hide C++ implementation details.
- **Caller Allocation:** The caller owns all memory. Result buffers are pre-allocated by the caller and passed to the kernel to fill.
- **Error Codes:** functions return `aeon_error_t` enums, never exceptions.

```c
// Example C-API signature
aeon_error_t aeon_atlas_navigate(
    aeon_atlas_t *atlas, const float *query, size_t dim,
    aeon_result_node_t *out_results, size_t max_results, size_t *out_count
);
```
