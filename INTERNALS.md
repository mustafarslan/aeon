# Aeon Internals & Data Structures

> **Target Audience:** Kernel Developers, Performance Engineers, Systems Architects.

---

## 1. On-Disk Format (Atlas)

Aeon stores the semantic Atlas in a memory-mapped binary file designed for **pointer-chasing-free** access. Offsets are calculated using a dynamic `node_byte_stride` to support variable embedding dimensions.

### 1.1 AtlasHeader (128 bytes)

Located at offset 0. `magic` = `0xAEO11` (Aeon V1).

| Offset | Field | Type | Description |
|---|---|---|---|
| 0 | `magic` | `uint64` | `0xAEO11` literal. |
| 8 | `version` | `uint32` | Version identifier (current: 2). |
| 12 | `flags` | `uint32` | Reserved for feature flags. |
| 16 | `node_count` | `uint64` | Number of live + tombstoned nodes. |
| 24 | `capacity` | `uint64` | Total capacity (nodes) before resize needed. |
| 32 | `dim` | `uint32` | Embedding dimension (e.g., 384, 768, 1536). |
| 36 | `metadata_size` | `uint32` | Fixed byte size for metadata storage (default 256). |
| 40 | `node_byte_stride` | `uint64` | **Critical:** `sizeof(NodeHeader) + dim*4 + metadata_size`. |
| 48 | `reserved[10]` | `uint64` | Reserved for future expansion (padding to 128B). |

### 1.2 Node Memory Layout (Variable Stride)

Nodes are stored contiguously starting at `sizeof(AtlasHeader)`. Address of node `i`:

```cpp
ptr = base_addr + 128 + (i * header.node_byte_stride);
```

**Node Structure:**

1. **NodeHeader (64 bytes)**
    - `id` (8B), `parent_id` (8B), `created_at` (8B)
    - `child_count` (4B), `flags` (2B), `alignment_pad` (2B)
    - `hub_penalty` (4B) for CSLS
2. **Vector Data** (`dim * sizeof(float)`)
    - Raw float array. Access: `(float*)(ptr + 64)`.
3. **Metadata** (`metadata_size` bytes)
    - Null-terminated string or binary blob.

---

## 2. Flat Byte Arena Delta Buffer

To support high-throughput inserts (10k+/sec) without frequent mmap resizing, Aeon uses a **Flat Byte Arena** in RAM.

### 2.1 Design: Flat Arena vs. Object-Per-Node

| Feature | Legacy `std::vector<Node>` | V4.0 `std::vector<uint8_t>` Arena |
|---|---|---|
| **Memory Layout** | Heap-allocated objects, pointer chasing | Contiguous byte array, stride-indexed |
| **Stride** | Fixed at compile time | Dynamic per file (runtime configurable) |
| **Cache locality** | Poor (scattered objects) | Excellent (linear scan friendly) |
| **Serialization** | Requires deep copy/serialization | `memcpy` directly to/from disk |

### 2.2 Delta Node IDs

Delta nodes are assigned temporary IDs with the **Most Significant Bit (MSB)** set:
`id = 0x8000000000000000 | delta_index`
This allows the kernel to distinguish between mmap-backed nodes (stable IDs) and delta nodes (ephemeral IDs) during query merging.

---

## 3. TraceEvent Struct (512 bytes)

Episodic memory events are stored as fixed-size, page-aligned structs. This design enables O(1) random access and zero-copy ingestion.

```cpp
struct alignas(64) TraceEvent {
  uint64_t id;              // 8B   Monotonic ID
  uint64_t prev_id;         // 8B   Previous event in session (linked list)
  uint64_t atlas_id;        // 8B   Linked Atlas concept ID
  uint64_t timestamp;       // 8B   Epoch microseconds
  uint16_t role;            // 2B   User/System/Concept/Summary
  uint16_t flags;           // 2B   TOMBSTONE / ARCHIVED
  char     session_id[36];  // 36B  UUID string (null-terminated)
  char     text[440];       // 440B Content preview
};
static_assert(sizeof(TraceEvent) == 512); // Fits exactly 8 per 4KB page
```

---

## 4. Trace Block Index (TBI)

To search episodic history efficiently without an external index (like Faiss), Aeon implements a **Chronological Block Index**.

- **TraceBlock:** Covers 1024 consecutive `TraceEvent`s.
- **Centroid:** Maintains a running mean vector of all embeddings in the block.
- **Two-Phase Search:**
    1. **Block Selector:** Scans block centroids to find the `top_k` most relevant blocks. This reduces the search space by ~1000x.
    2. **Fine-Grained Scan:** Loads embeddings *only* for events in the selected blocks and performs a full dot product.

**Performance:** Scans 100M events in < 50ms (single core AVX-512).

---

## 5. Shadow Compaction (Internals)

Code flow for the stutter-free `compact_mmap()` operation:

```cpp
void Atlas::compact_mmap() {
    // Step 1: Microsecond Freeze
    {
        std::unique_lock lock(write_mutex_);
        frozen_delta_ = std::move(delta_buffer_); // Pointer swap
        delta_buffer_.reserve(1000);              // Fresh buffer
    } // Lock released immediately

    // Step 2: Background Copy
    // - Iterate active mmap nodes (skip tombstones)
    // - Iterate frozen_delta_
    // - Write tightly packed stream to "atlas.gen2.bin"

    // Step 3: Hot Swap
    {
        std::unique_lock lock(write_mutex_);
        file_.swap_handle("atlas.gen2.bin");
        frozen_delta_.clear();
    }
}
```

---

## 6. Epoch-Based Reclamation (EBR)

Aeon's concurrency model relies on a lock-free read path.

- **Global Epoch:** A monotonic counter `atomic<uint64_t>`.
- **Reader Slots:** Fixed array `[MAX_READERS=64]`. Each reader claims a slot and publishes the current global epoch when entering a critical section.
- **Cache Isolation:** Each slot is padded to `CACHE_LINE_SIZE` (64 bytes) to prevent **false sharing** on multi-core concurrent reads.

```cpp
struct alignas(64) ReaderSlot {
    std::atomic<uint64_t> epoch;
};
```

---

## 7. Hierarchical SLB Internals

The **Semantic Lookaside Buffer** is sharded to minimize lock contention.

- **Sharding:** `session_id` is hashed to one of 64 shards.
- **Lock Striping:** Each shard has its own `std::shared_mutex`. Operations on Session A (Shard 1) never block Session B (Shard 2).
- **Safe References:** The map stores `std::shared_ptr<SessionRingBuffer>`. Even if a session is dropped, any active reader holding the pointer can safely finish its scan before the memory is freed.

---

## 8. SIMD Math Kernel

The `math_kernel.hpp` module uses compile-time dispatch to select the optimal instruction set:

1. **AVX-512:** Used if `__AVX512F__` is defined. Processes 16 floats per cycle.
2. **AVX-2:** Fallback for older x86 (Zen 2, Skylake). Processes 8 floats per cycle.
3. **NEON:** Used on ARM64 (Apple Silicon, AWS Graviton). Processes 4 floats per cycle.
4. **Scalar:** Maximum portability fallback.

---

## 9. Storage Layer

The `MemoryFile` class abstracts platform-specific `mmap` details.

- **OS Abstraction:** Wraps Windows `CreateFileMapping` and POSIX `mmap`.
- **Huge Pages:** Calls `madvise(MADV_HUGEPAGE)` on Linux to reduce TLB misses for large Atlas files.
- **Growth Strategy:** `grow(capacity)` allocates a larger region, copies data (if not using `mremap`), and retires the old pointer via EBR.
