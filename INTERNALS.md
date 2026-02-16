# Aeon Internals & Data Structures

> **Target Audience:** Kernel Developers, Performance Engineers, Systems Architects.

---

## 1. On-Disk Format (Atlas)

Aeon stores the semantic Atlas in a memory-mapped binary file designed for **pointer-chasing-free** access. Offsets are calculated using a dynamic `node_byte_stride` to support variable embedding dimensions and quantization levels.

### 1.1 AtlasHeader (64 bytes)

Located at offset 0. `magic` = `0x41544C41535F3031` (ASCII "ATLAS_01").

| Offset | Field | Type | Description |
|---|---|---|---|
| 0x00 | `magic` | `uint64` | `0x41544C41535F3031` literal. |
| 0x08 | `version` | `uint64` | Version identifier (current: 2). |
| 0x10 | `node_count` | `uint64` | Number of live + tombstoned nodes. |
| 0x18 | `capacity` | `uint64` | Total capacity (nodes) before resize needed. |
| 0x20 | `dim` | `uint32` | Embedding dimension (e.g., 384, 768, 1536). |
| 0x24 | `metadata_size` | `uint32` | Fixed byte size for metadata storage (default 256). |
| 0x28 | `node_byte_stride` | `uint64` | **Critical:** 64-byte-aligned stride per node. |
| 0x30 | `quantization_type` | `uint32` | `0` = FP32, `1` = INT8_SYMMETRIC. |
| 0x34 | `reserved[12]` | `uint8[12]` | Zeroed on creation. Future expansion. |

`static_assert(sizeof(AtlasHeader) == 64)`

### 1.2 Node Memory Layout (Variable Stride)

Nodes are stored contiguously starting at `sizeof(AtlasHeader)`. Address of node `i`:

```cpp
ptr = base_addr + 64 + (i * header.node_byte_stride);
```

**Node Structure:**

1. **NodeHeader (64 bytes)**
    - `id` (8B), `parent_offset` (8B), `first_child_offset` (8B)
    - `child_count` (2B), `flags` (2B), `hub_penalty` (4B) for CSLS
    - `quant_scale` (4B) — `max(|v|) / 127.0f` for INT8; `0.0f` for FP32
    - `quant_zero_point` (4B) — always `0.0f` (symmetric quantization)
    - `reserved[20]` — padding to 64-byte cache line boundary
2. **Centroid Data** (variable)
    - FP32 mode: `dim × sizeof(float)` bytes. Access: `(float*)(ptr + 64)`
    - INT8 mode: `dim × sizeof(int8_t)` bytes. Access: `(int8_t*)(ptr + 64)`
3. **Metadata** (`metadata_size` bytes)
    - Null-terminated string or binary blob.
4. **Alignment padding** to the next 64-byte boundary.

**Stride Computation (quantization-aware):**

```cpp
// FP32:  align_up(64 + 768*4 + 256, 64) = 3392 bytes
// INT8:  align_up(64 + 768*1 + 256, 64) = 1088 bytes  (3.1× smaller)
constexpr size_t compute_node_stride(uint32_t dim, uint32_t metadata_size,
                                     uint32_t quant_type) noexcept {
    size_t payload = (quant_type == QUANT_INT8_SYMMETRIC)
                         ? dim * sizeof(int8_t)
                         : dim * sizeof(float);
    return align_up(sizeof(NodeHeader) + payload + metadata_size, 64);
}
```

---

## 2. Flat Byte Arena Delta Buffer

To support high-throughput inserts (10k+/sec) without frequent mmap resizing, Aeon uses a **Flat Byte Arena** in RAM.

### 2.1 Design: Flat Arena vs. Object-Per-Node

| Feature | Legacy `std::vector<Node>` | V4.0+ `std::vector<uint8_t>` Arena |
|---|---|---|
| **Memory Layout** | Heap-allocated objects, pointer chasing | Contiguous byte array, stride-indexed |
| **Stride** | Fixed at compile time | Dynamic per file (runtime configurable) |
| **Cache locality** | Poor (scattered objects) | Excellent (linear scan friendly) |
| **Serialization** | Requires deep copy/serialization | `memcpy` directly to/from disk |
| **Quantization** | N/A | Same arena layout for FP32 and INT8 |

### 2.2 Delta Node IDs

Delta nodes are assigned temporary IDs with the **Most Significant Bit (MSB)** set:
`id = 0x8000000000000000 | delta_index`
This allows the kernel to distinguish between mmap-backed nodes (stable IDs) and delta nodes (ephemeral IDs) during query merging.

---

## 3. TraceEvent Struct (512 bytes)

Episodic memory events are stored as fixed-size, page-aligned structs. This design enables O(1) random access and zero-copy ingestion.

```cpp
struct alignas(64) TraceEvent {
    uint64_t id;              // 0x000: Monotonic ID
    uint64_t prev_id;         // 0x008: Previous event in session (linked list)
    uint64_t atlas_id;        // 0x010: Linked Atlas concept ID
    uint64_t timestamp;       // 0x018: Epoch microseconds
    uint16_t role;            // 0x020: User/System/Concept/Summary
    uint16_t flags;           // 0x022: TOMBSTONE / ARCHIVED
    char     session_id[36];  // 0x024: UUID string (null-terminated)

    // ─── V4.1: Sidecar Blob Arena pointers ───
    uint64_t blob_offset;     // 0x048: Offset into sidecar blob file
    uint32_t blob_size;       // 0x050: Byte length of full text in blob
    char     text_preview[64];// 0x054: Null-terminated 63-char inline prefix
    uint8_t  reserved[364];   // 0x094: Padding to 512 bytes
};
static_assert(sizeof(TraceEvent) == 512); // 8 events per 4KB page
```

**V4.1 Changes from V4.0:**

| Field | V4.0 | V4.1 |
|---|---|---|
| `text[440]` | Full text inline (440B limit) | **Removed** |
| `blob_offset` | N/A | Offset into sidecar blob file |
| `blob_size` | N/A | Byte length of full text |
| `text_preview[64]` | N/A | 63-char inline prefix for fast listing |
| `reserved` | Smaller | 364 bytes (accommodates new fields at same 512B total) |

The struct size remains **exactly 512 bytes**. The blob pointers replace the old `text[440]` field, which was the bottleneck for long-form agent traces, LLM transcript storage, and full conversation logging.

---

## 4. Sidecar Blob Arena (Internals)

The `BlobArena` class manages the sidecar blob file (`trace_blobs_genN.bin`).

### 4.1 File Layout

```text
┌───────────────────────────────────────────────────────────┐
│ trace_blobs_gen0.bin                                       │
│ (No header. Raw concatenated blobs. Offset metadata lives  │
│  in TraceEvent::blob_offset / blob_size.)                  │
│                                                            │
│ [blob_0: 142 bytes][blob_1: 4096 bytes][blob_2: 23 bytes] │
│ ↑ offset=0         ↑ offset=142        ↑ offset=4238      │
└───────────────────────────────────────────────────────────┘
```

### 4.2 Growth Strategy

- **Initial size:** 4KB (`kInitialSize = 4096`)
- **Growth:** 2× doubling (`kGrowthFactor = 2`)
- **Mechanism:** `ftruncate()` → `munmap()` → `mmap()` (no `mremap` for portability)

### 4.3 Generational Garbage Collection

During `TraceManager::compact()`:

1. **Freeze:** Swap delta buffer (same as Trace compaction Step 1).
2. **Copy live events:** For each non-tombstoned event, copy its blob from the old arena to a new `trace_blobs_gen{N+1}.bin`.
3. **Update blob_offset:** The new event's `blob_offset` points to the new position in the new blob file.
4. **Delete old arena:** The old generation blob file is removed after the hot swap.

Dead blobs (those only referenced by tombstoned events) are simply not copied — zero-overhead garbage collection.

---

## 5. Write-Ahead Log (WAL) Record Format

### 5.1 WalRecordHeader (16 bytes)

```cpp
struct WalRecordHeader {
    uint32_t record_type;   // WAL_RECORD_ATLAS (0x01) or WAL_RECORD_TRACE (0x02)
    uint32_t payload_size;  // Bytes of payload following this header
    uint64_t checksum;      // FNV-1a 64-bit hash of payload bytes
};
static_assert(sizeof(WalRecordHeader) == 16);
```

### 5.2 WAL File Layout

```text
┌────────────────────────────────────────────────────────┐
│ atlas.bin.wal (or trace.bin.wal)                        │
│                                                         │
│ [WalRecordHeader (16B)][Payload (node_byte_stride B)]  │
│ [WalRecordHeader (16B)][Payload (node_byte_stride B)]  │
│ [WalRecordHeader (16B)][Payload (512B)]  ← Trace event │
│ ...                                                     │
│ [torn record — checksum mismatch → discard from here]  │
└────────────────────────────────────────────────────────┘
```

### 5.3 Checksum Validation (FNV-1a)

Each record's payload is checksummed with **FNV-1a 64-bit**. On replay:

- If `computed_checksum == header.checksum` → replay the record into the delta buffer.
- If mismatched → the record (and all subsequent records) is a torn write from a crash. Discard and stop.

This is **best-effort** crash recovery: committed records are replayed, torn writes are safely discarded.

### 5.4 Lock Ordering

```text
Invariant: serialize (no lock) → wal_mutex_ → delta_mutex_

wal_mutex_:   Protects WAL file I/O (fwrite + fflush).
delta_mutex_: Protects in-memory delta_buffer_bytes_ / delta_bytes_.

Key insight: wal_mutex_ is NEVER held while delta_mutex_ is held,
             and delta_mutex_ is NEVER held while wal_mutex_ is held.
             This prevents game engine writers from blocking on disk I/O.
```

---

## 6. Trace Block Index (TBI)

To search episodic history efficiently without an external index (like Faiss), Aeon implements a **Chronological Block Index**.

- **TraceBlock:** Covers 1024 consecutive `TraceEvent`s.
- **Centroid:** Maintains a running mean vector of all embeddings in the block.
- **Two-Phase Search:**
    1. **Block Selector:** Scans block centroids to find the `top_k` most relevant blocks. This reduces the search space by ~1000×.
    2. **Fine-Grained Scan:** Loads embeddings *only* for events in the selected blocks and performs a full dot product.

**Performance:** Scans 100M events in < 50ms (single core AVX-512).

---

## 7. Shadow Compaction (Internals)

Code flow for the stutter-free `compact_mmap()` operation:

```cpp
void Atlas::compact_mmap() {
    // Step 1: Microsecond Freeze
    {
        std::unique_lock lock(write_mutex_);
        frozen_delta_buffer_bytes_ = std::move(delta_buffer_bytes_);
        delta_buffer_bytes_.reserve(1000);
        compact_in_progress_ = true;
    } // Lock released immediately (~2 µs)

    // Step 2: Background Copy (no exclusive lock)
    // - Iterate active mmap nodes (skip tombstones)
    // - Iterate frozen_delta_buffer_bytes_
    // - Write tightly packed stream to "atlas_gen{N+1}.bin"

    // Step 3: Hot Swap
    {
        std::unique_lock lock(write_mutex_);
        file_->swap_handle("atlas_gen{N+1}.bin");
        frozen_delta_buffer_bytes_.clear();
        slb_cache_.clear();  // Node IDs changed
    }

    // Step 4: Background Cleanup
    compact_in_progress_ = false;
    // Close old file → drain EBR readers → delete old gen
    // Truncate WAL (all delta data now safely on disk)
    truncate_wal();
}
```

---

## 8. Epoch-Based Reclamation (EBR)

Aeon's concurrency model relies on a lock-free read path.

- **Global Epoch:** A monotonic counter `atomic<uint64_t>`.
- **Reader Slots:** Fixed array `[MAX_READERS=64]`. Each reader claims a slot and publishes the current global epoch when entering a critical section.
- **Cache Isolation:** Each slot is padded to `CACHE_LINE_SIZE` (64 bytes) to prevent **false sharing** on multi-core concurrent reads.

```cpp
struct alignas(64) ReaderSlot {
    std::atomic<uint64_t> epoch;
};
```

Empirically verified under hostile contention (15 readers, 1 writer, 100K iterations each):

| Percentile | Latency |
|---|---|
| P50 | 167 ns |
| P90 | 417 ns |
| **P99** | **750 ns** |
| P99.9 | 1,083 ns |

---

## 9. Hierarchical SLB Internals

The **Semantic Lookaside Buffer** is sharded to minimize lock contention.

- **Sharding:** `session_id` is hashed to one of 64 shards.
- **Lock Striping:** Each shard has its own `std::shared_mutex`. Operations on Session A (Shard 1) never block Session B (Shard 2).
- **Safe References:** The map stores `std::shared_ptr<SessionRingBuffer>`. Even if a session is dropped, any active reader holding the pointer can safely finish its scan before the memory is freed.
- **FP32 Only:** The SLB always stores FP32 vectors. When the Atlas is INT8-quantized, vectors are dequantized on insert into the SLB. This preserves the 3.56µs cache hit latency.

---

## 10. SIMD Math Kernel

The `math_kernel.hpp` module uses compile-time dispatch to select the optimal instruction set:

### 10.1 FP32 Cosine Similarity

1. **AVX-512:** Used if `__AVX512F__` is defined. Processes 16 floats per cycle.
2. **AVX-2:** Fallback for older x86 (Zen 2, Skylake). Processes 8 floats per cycle.
3. **NEON:** Used on ARM64 (Apple Silicon, AWS Graviton). Processes 4 floats per cycle.
4. **Scalar:** Maximum portability fallback.

### 10.2 INT8 Dot Product (V4.1)

1. **NEON SDOT (ARMv8.2+):** Processes 16 int8s per cycle via `vdotq_s32`. **4.70ns** at dim=768.
2. **AVX-512 VNNI:** Uses SIMDe offset trick for signed×signed int8 multiplication.
3. **Scalar:** Portable baseline (15.3ns at dim=768).

The raw `int32` result from any INT8 kernel must be dequantized: `float_result = raw_int32 * scale_query * scale_node`.

---

## 11. INT8 Quantization Internals

### 11.1 Quantize Path (Insert)

```text
float[768] input → quantize_symmetric() → int8[768] output + float scale

1. Find max_abs = max(|v[i]|) over all i
2. scale = max_abs / 127.0f  (safety: if max_abs < 1e-10, scale = 1.0)
3. q[i] = clamp(round(v[i] / scale), -127, 127)
4. Store q[i] in node centroid, scale in NodeHeader::quant_scale
```

### 11.2 Navigate Path (Query)

```text
1. Quantize query vector: quantize_symmetric(query) → q_query, scale_query
2. For each node:
   a. Load int8 centroid from mmap
   b. Load node scale from NodeHeader::quant_scale
   c. raw_dot = dot_int8_neon(q_query, q_node, dim)  ← 4.70ns
   d. similarity = raw_dot * scale_query * node_scale  ← dequantize
   e. similarity -= hub_penalty  ← branchless tombstone/CSLS
3. Insert top-K results into SLB as FP32 (dequantized)
```

### 11.3 Compression Results

From `master_metrics.txt` (`bench_quantization_efficiency`, 100K nodes, dim=768):

| Metric | FP32 | INT8 | Ratio |
|---|---|---|---|
| Bytes/node | 4,400 | 1,411 | 3.1× smaller |
| File size | 440.0 MB | 141.1 MB | 3.1× smaller |
| Navigate (mean) | 10.5 µs | 3.09 µs | 3.4× faster |
| Insert (mean) | 2.23 µs | 2.11 µs | ~same |
| Relative error | — | 0.069% | Negligible |

---

## 12. Storage Layer

The `MemoryFile` class abstracts platform-specific `mmap` details.

- **OS Abstraction:** Wraps Windows `CreateFileMapping` and POSIX `mmap`.
- **Huge Pages:** Calls `madvise(MADV_HUGEPAGE)` on Linux to reduce TLB misses for large Atlas files.
- **Growth Strategy:** `grow(capacity)` allocates a larger region, copies data (if not using `mremap`), and retires the old pointer via EBR.
