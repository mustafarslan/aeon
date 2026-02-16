# Aeon Memory OS — Code Walkthrough

> **Version:** 4.0.0 · **Audience:** Developers integrating Aeon or contributing to the kernel

This document walks through the key code paths, API surfaces, and operational workflows of the Aeon Memory OS. For data structure details, see [INTERNALS.md](./INTERNALS.md). For architectural design, see [ARCHITECTURE.md](./ARCHITECTURE.md).

---

## 1. Repository Layout

```text
aeon/
├── core/                          # C++23 Kernel
│   ├── include/aeon/
│   │   ├── aeon_c_api.h           # Universal C-API (FFI boundary)
│   │   ├── atlas.hpp              # Atlas spatial memory engine
│   │   ├── epoch.hpp              # Epoch-Based Reclamation (EBR)
│   │   ├── hierarchical_slb.hpp   # L1/L2 Sharded Semantic Cache
│   │   ├── math_kernel.hpp        # SIMD dispatch interface
│   │   ├── platform.hpp           # Cross-platform mmap abstraction
│   │   ├── schema.hpp             # Binary schema (NodeHeader, TraceEvent)
│   │   ├── simd_impl.hpp          # SIMD implementation (AVX-2 / NEON)
│   │   ├── slb.hpp                # Single-session SemanticCache (L2)
│   │   ├── storage.hpp            # MemoryFile (mmap lifecycle)
│   │   ├── tiered_atlas.hpp       # Edge-cloud cold-miss detection
│   │   ├── trace.hpp              # mmap-backed Episodic Trace Engine
│   │   └── trace_block_index.hpp  # Block index for trace range queries
│   ├── src/
│   │   ├── aeon_c_api.cpp         # C-API implementation
│   │   ├── atlas.cpp              # Atlas implementation
│   │   ├── bindings.cpp           # nanobind Python bindings
│   │   ├── simd_impl.cpp          # SIMD codepaths
│   │   └── trace.cpp              # Trace implementation
│   ├── tests/                     # Google Test suites
│   └── benchmarks/                # Google Benchmark suites
├── shell/aeon_py/                 # Python Shell (Ring 3)
│   ├── client.py                  # AeonClient, TieredClient, DriftMonitor
│   ├── dreamer.py                 # DreamingWorker (background GC)
│   ├── session.py                 # Session management
│   ├── trace.py                   # Python TraceGraph wrapper
│   ├── server.py                  # HTTP/REST server
│   └── loop.py                    # Main event loop
├── bindings/                      # Language binding examples
├── benchmarks/                    # Python-level benchmarks
├── ARCHITECTURE.md
├── INTERNALS.md
├── CODE_WALKTHROUGH.md            # (this file)
├── README.md
├── CMakeLists.txt                 # Root CMake
├── build.sh                       # One-command build (macOS/Linux)
└── build.bat                      # One-command build (Windows)
```

---

## 2. C-API Walkthrough — Multi-Tenant Session Routing

The C-API (`aeon_c_api.h` / `aeon_c_api.cpp`) is the universal FFI boundary for all non-Python consumers. Every function follows a strict safety contract.

### 2.1 Function Signature Pattern

```c
AEON_API aeon_error_t aeon_atlas_navigate(
    aeon_atlas_t     *atlas,          // Opaque handle
    const float      *query_vector,   // Caller-owned buffer
    size_t            query_dim,      // Validated against atlas dim
    uint32_t          beam_width,     // Beam search width (1 = greedy)
    int               apply_csls,     // CSLS hubness correction flag
    const char       *session_id,     // Multi-tenant routing key
    aeon_result_node_t *results,      // Caller-allocated output buffer
    size_t            max_results,    // Capacity of output buffer
    size_t           *out_actual_count // Actual results written
);
```

**Safety guarantees:**

1. **NULL check:** All pointer parameters are validated; returns `AEON_ERR_NULL_PTR`
2. **Dimension validation:** `query_dim` is checked against `atlas->dim()`; returns `AEON_ERR_INVALID_ARG`
3. **Exception containment:** Every function body is wrapped in `try/catch(...)` to prevent C++ exceptions from crossing the FFI boundary
4. **Caller-allocated buffers:** The kernel never allocates memory that the caller must free

### 2.2 session_id Routing to HierarchicalSLB

The `session_id` parameter enables multi-tenant SLB cache isolation:

```text
┌─────────────────────────────────────────────────┐
│  aeon_atlas_navigate(atlas, query, ...,          │
│                      session_id="npc-goblin-42", │
│                      results, ...)               │
│                                                  │
│  1. NULL/dim validation                          │
│  2. Route session_id → HierarchicalSLB L1       │
│     → shard = hash("npc-goblin-42") % 64        │
│     → check SessionRingBuffer for cache hit      │
│  3. If miss → Atlas::navigate() beam search      │
│  4. Copy results → caller buffer                 │
│  5. Return AEON_OK                               │
└─────────────────────────────────────────────────┘
```

### 2.3 aeon_atlas_drop_session — Memory Leak Prevention

```c
AEON_API aeon_error_t aeon_atlas_drop_session(
    aeon_atlas_t *atlas,
    const char   *session_id    // Session to evict from SLB
);
```

**When to call:**

| Platform | Trigger |
|----------|---------|
| Unreal Engine | `AActor::EndPlay()` (NPC despawn) |
| Unity | `OnDestroy()` / scene transition |
| Mobile | `onTrimMemory()` / background transition |
| Server | WebSocket disconnect / session timeout |

**Why mandatory:** With 100,000+ concurrent sessions (e.g., MMO server), each session's L1 `SessionRingBuffer` (64 entries × ~3KB each ≈ 200KB per session) would leak if not explicitly dropped. At 100K sessions, this represents ~20GB of leaked SLB cache.

### 2.4 Trace C-API

```c
// Create or open mmap-backed trace file
aeon_error_t aeon_trace_create(const char *path, aeon_trace_t **out);

// Append event — links prev_id via session_tails_ map
aeon_error_t aeon_trace_append_event(
    aeon_trace_t *trace,
    const char   *session_id,    // Session for prev_id chaining
    uint16_t      role,          // TraceRole: User(0), System(1), Concept(2), Summary(3)
    const char   *text,          // Truncated to 439 chars (null-terminated in 440B field)
    uint64_t      atlas_id,      // Linked Atlas concept node (0 = none)
    uint64_t     *out_id         // Receives the new monotonic event ID
);

// Retrieve session history via prev_id chain (newest first)
aeon_error_t aeon_trace_get_history(
    aeon_trace_t       *trace,
    const char         *session_id,
    aeon_trace_event_t *out_events,  // Caller-allocated buffer
    size_t              max_events,
    size_t             *out_count
);
```

**FFI struct identity:** `aeon_trace_event_t` (C-API) and `TraceEvent` (C++ kernel) have identical memory layout. The C-API implementation uses a `static_assert` to enforce this and flat `memcpy` for zero-overhead transfer:

```cpp
static_assert(sizeof(aeon_trace_event_t) == sizeof(aeon::TraceEvent),
              "FFI struct must match C++ struct size");
std::memcpy(&out_events[i], &history[i], sizeof(aeon_trace_event_t));
```

---

## 3. Trace Workflow — append_event Lifecycle

### 3.1 Session Tail Lookup and prev_id Linking

When `aeon_trace_append_event()` is called:

```text
┌──────────────────────────────────────────────────┐
│  append_event("session-abc", ROLE_USER,           │
│               "What is the capital of France?",   │
│               atlas_id=42)                        │
│                                                   │
│  1. Acquire unique_lock(rw_mutex_)                │
│  2. Lookup session_tails_["session-abc"]          │
│     → prev_id = 38 (last event for this session) │
│  3. Check write diversion:                        │
│     compact_in_progress_? → append_delta()        │
│     else                  → append_mmap()         │
│  4. Fill TraceEvent (512B):                       │
│     id = next_event_id_++                         │
│     prev_id = 38                                  │
│     atlas_id = 42                                 │
│     timestamp = now_micros()                      │
│     role = ROLE_USER                              │
│     session_id = "session-abc"                    │
│     text = "What is the capital..."               │
│  5. Update session_tails_["session-abc"] = new_id │
│  6. Release lock                                  │
│  7. Return new event ID                           │
└──────────────────────────────────────────────────┘
```

### 3.2 get_history — Backward Chain Traversal

```text
get_history("session-abc", limit=5)

session_tails_["session-abc"] → event_id=42
  resolve_event(42) → event at mmap offset
    copy → result[0], follow prev_id=38
  resolve_event(38) → event at delta buffer
    copy → result[1], follow prev_id=25
  resolve_event(25) → event at mmap offset
    copy → result[2], follow prev_id=0  (root)
  Stop: prev_id = 0

Return: [event_42, event_38, event_25]  (newest first)
```

`resolve_event()` searches three locations in order:

1. **mmap region** (most events live here)
2. **Active delta buffer** (recent events during normal operation)
3. **Frozen delta buffer** (recent events during compaction)

---

## 4. compact_mmap() — Full Lifecycle

### 4.1 Atlas Compaction

```text
compact_mmap() — V4.0 Shadow Compaction
═══════════════════════════════════════

STEP 1: µs FREEZE (exclusive lock)
┌─────────────────────────────────────┐
│ write_lock + delta_lock acquired    │
│                                     │
│ delta_buffer_ ──move──→ frozen_     │
│ delta_buffer_ ← empty + reserve    │
│ snapshot_node_count = header->count │
│ compact_in_progress_ = true        │
│                                     │
│ Lock released (~2 µs)              │
└─────────────────────────────────────┘
      │
      │  insert() → insert_delta()  (Write Diversion active)
      │  navigate() continues normally
      ▼
STEP 2: BACKGROUND COPY (no exclusive lock)
┌─────────────────────────────────────┐
│ EpochGuard only (prevents munmap)   │
│                                     │
│ Count live mmap nodes (skip tombs)  │
│ Create atlas_gen{N+1}.bin           │
│ ftruncate + mmap new file          │
│                                     │
│ for each live mmap node:            │
│   memcpy(dst, src, node_byte_stride)│
│   dst_hdr->id = new_sequential_id  │
│                                     │
│ for each frozen delta node:         │
│   memcpy(dst, src, node_byte_stride)│
│   dst_hdr->id = promoted_id        │
│                                     │
│ Re-index: parent_offset,            │
│   first_child_offset via old_to_new │
│                                     │
│ munmap + close new file             │
│ (~seconds for GB-scale files)       │
└─────────────────────────────────────┘
      │
      ▼
STEP 3: µs HOT-SWAP (exclusive lock)
┌─────────────────────────────────────┐
│ write_lock acquired                 │
│                                     │
│ old_file = move(file_)              │
│ file_ = move(new_file)  ← swap!    │
│                                     │
│ frozen_delta_.clear()               │
│ slb_cache_.clear()  (IDs changed)  │
│ atlas_path_ = new_path             │
│ generation_ = new_gen              │
│                                     │
│ Lock released (~1 µs)              │
└─────────────────────────────────────┘
      │
      ▼
STEP 4: BACKGROUND CLEANUP (no lock)
┌─────────────────────────────────────┐
│ compact_in_progress_ = false        │
│ old_file->close()  ← blocks here   │
│   (drain_readers waits for EBR)     │
│ old_file.reset()                    │
│ filesystem::remove(old_path)        │
│ epoch_mgr_.advance_epoch()          │
└─────────────────────────────────────┘
```

### 4.2 Trace Compaction

The Trace compaction follows the same pattern but operates directly on raw POSIX `mmap`/`munmap` (not through `MemoryFile`). Key differences:

- Events are 512 bytes each (vs variable `node_byte_stride` for Atlas)
- After Step 3, `rebuild_session_tails()` is called to reconstruct the `session_tails_` map from the new file
- Events added to `delta_bytes_` during Step 2 are accounted for in the session tail rebuild

---

## 5. nanobind Python Bindings

### 5.1 GIL Release Pattern

All blocking C++ operations release the Python GIL to enable concurrency:

```cpp
.def("navigate_raw",
    [](aeon::Atlas &self, const std::vector<float> &query, ...) {
        std::vector<aeon::Atlas::ResultNode> results;
        {
            nb::gil_scoped_release release;  // Release GIL
            results = self.navigate(
                std::span<const float>(query.data(), self.dim()),
                beam_width, apply_csls);
        }
        // GIL re-acquired — safe to allocate Python objects
        
        size_t num_bytes = results.size() * sizeof(ResultNode);
        uint8_t *data = new uint8_t[num_bytes];
        std::memcpy(data, results.data(), num_bytes);
        
        nb::capsule owner(data,
            [](void *p) noexcept { delete[] (uint8_t *)p; });
        
        return nb::ndarray<uint8_t, nb::numpy, nb::shape<-1>, nb::ro>(
            data, {num_bytes}, owner);
    })
```

**Key pattern:** C++ work runs with GIL released → results are copied to a capsule-owned buffer → returned as a read-only NumPy byte array → Python client views as structured dtype.

### 5.2 Zero-Copy Result Pipeline

```text
C++ Atlas::navigate()  →  std::vector<ResultNode>  →  memcpy to capsule
    ↓
Python client.py: byte_view = atlas.navigate_raw(embedding)
    ↓
result = byte_view.view(RESULT_DTYPE)    # Zero-copy cast
result.flags.writeable = False           # Enforce read-only
```

The `RESULT_DTYPE` in `client.py` mirrors the C++ `ResultNode` layout:

```python
RESULT_DTYPE = np.dtype([
    ('id', 'u8'),                    # uint64_t
    ('similarity', 'f4'),            # float
    ('preview', 'f4', (3,)),         # float[3]
    ('requires_cloud_fetch', '?'),   # bool
], align=True)   # Total: 32 bytes
```

A startup-time schema validation ensures the C++ and Python struct sizes match:

```python
cpp_size = core.get_result_node_size()    # sizeof(Atlas::ResultNode)
py_size = RESULT_DTYPE.itemsize           # 32
assert cpp_size == py_size                # CRITICAL: exact match required
```

### 5.3 EpochGuard as Python Context Manager

```python
# Python usage
with client.safe_memory_view():
    results = client.query(embedding)
    # Memory is pinned — safe to slice, index, pass around
    process(results)
```

```cpp
// C++ binding
nb::class_<aeon::EpochGuard>(m, "EpochGuard")
    .def("__enter__", [](nb::object self) -> nb::object { return self; })
    .def("__exit__", [](aeon::EpochGuard &self, nb::args) { self.release(); })
    .def("release", &aeon::EpochGuard::release)
    .def("is_active", &aeon::EpochGuard::is_active);
```

---

## 6. Python Shell — Key Components

### 6.1 DreamingWorker Lifecycle

```python
# Initialize
worker = DreamingWorker(
    atlas=atlas,
    atlas_path=Path("memory.bin"),
    config=DreamConfig(memory_budget_mb=128, tombstone_ratio_threshold=0.25),
    summarizer=StubSummarizer(),  # Or CloudSummarizer, LocalSummarizer
)

# Start background thread
worker.start(daemon=True)
# ... application runs ...

# Manual trigger (e.g., iOS Background App Refresh)
report = worker.dream_now()
print(f"Reclaimed: {report.storage_reclaimed_mb:.1f} MB")

# Stop
worker.stop(timeout=10.0)
```

### 6.2 TieredClient — Edge-Cloud Fallback

```python
client = TieredClient(
    atlas_path="edge_memory.bin",
    session_id="robot-arm-01",
    cold_miss_threshold=0.65,
    cloud_endpoint="https://cloud.example.com/aeon",
)

result = client.query_tiered(embedding)
# result["requires_cloud_fetch"]  → True if local similarity < 0.65
# result["telemetry"]["hit_rate"] → Rolling SLB hit rate
# result["telemetry"]["drift_score"] → Jensen-Shannon concept drift
```

### 6.3 DriftMonitor Alerts

When the SLB hit rate drops below 70% or JS-divergence exceeds 0.15:

```text
WARNING  SEMANTIC INERTIA COLLAPSED — Dreaming/Compaction Required
         session=robot-arm-01 hit_rate=0.582 threshold=0.700
         drift=0.1823 window=1000 total_queries=15420
```

---

## 7. Build and Test

### 7.1 One-Command Build

```bash
# Development build + tests
./build.sh

# Optimized release build
./build.sh release

# Benchmarks
./build.sh bench

# Windows
build.bat
build.bat release
```

### 7.2 CMake Presets

The project uses `CMakePresets.json` with four configurations:

| Preset | Generator | Build Type | Description |
|--------|-----------|------------|-------------|
| `dev` | Ninja | Debug | Development with sanitizers |
| `release` | Ninja | Release | Optimized production build |
| `ci-macos` | Ninja | Release | CI for macOS ARM64 |
| `ci-linux` | Ninja | Release | CI for Linux x86-64 |

### 7.3 Python Shell Installation

```bash
pip install -e ./shell
```

This installs the `aeon_py` package and compiles the nanobind C++ extension module (`core.cpython-*.so`).

---

## 8. Integration Examples

### 8.1 C-API Usage (Unreal Engine / C++)

```c
#include "aeon/aeon_c_api.h"

// Create Atlas with 768-dim embeddings
aeon_atlas_t *atlas = NULL;
aeon_error_t err = aeon_atlas_create("memory.bin", 768, &atlas);
if (err != AEON_OK) { /* handle error */ }

// Insert a memory
float embedding[768] = { /* ... from your model ... */ };
uint64_t node_id;
err = aeon_atlas_insert(atlas, 0, embedding, 768,
                        "Goblin spotted near village",
                        "npc-goblin-42",  // session_id
                        &node_id);

// Query with session-isolated SLB
aeon_result_node_t results[50];
size_t count;
err = aeon_atlas_navigate(atlas, query_vec, 768, 4, 0,
                          "npc-goblin-42",
                          results, 50, &count);

// NPC despawn — cleanup SLB cache
aeon_atlas_drop_session(atlas, "npc-goblin-42");

// Append trace event
aeon_trace_t *trace = NULL;
aeon_trace_create("trace.bin", &trace);

uint64_t event_id;
aeon_trace_append_event(trace, "npc-goblin-42",
                        0,  // ROLE_USER
                        "Player attacked goblin",
                        node_id, &event_id);

// Cleanup
aeon_atlas_destroy(atlas);
aeon_trace_destroy(trace);
```

### 8.2 C# Usage (Unity)

```csharp
[DllImport("aeon", CallingConvention = CallingConvention.Cdecl)]
static extern int aeon_atlas_create(string path, uint dim, out IntPtr atlas);

[DllImport("aeon", CallingConvention = CallingConvention.Cdecl)]
static extern int aeon_atlas_navigate(
    IntPtr atlas, float[] query, UIntPtr queryDim,
    uint beamWidth, int applyCsls, string sessionId,
    [Out] AeonResultNode[] results, UIntPtr maxResults,
    out UIntPtr actualCount);

[DllImport("aeon", CallingConvention = CallingConvention.Cdecl)]
static extern int aeon_atlas_drop_session(IntPtr atlas, string sessionId);

// Usage
IntPtr atlas;
aeon_atlas_create("memory.bin", 768, out atlas);

var results = new AeonResultNode[50];
UIntPtr count;
aeon_atlas_navigate(atlas, queryVec, (UIntPtr)768, 4, 0,
                    "player-session-1",
                    results, (UIntPtr)50, out count);

// Scene transition
aeon_atlas_drop_session(atlas, "player-session-1");
```

### 8.3 Python Usage

```python
from aeon_py.client import AeonClient
from aeon_py import core

# Create Atlas
client = AeonClient("memory.bin")
client.warmup()  # Touch mmap pages

# Query
import numpy as np
embedding = np.random.randn(768).astype(np.float32)
results = client.query(embedding)
print(f"Best match: node {results['id'][0]}, sim={results['similarity'][0]:.4f}")

# Trace
trace = core.TraceManager("trace.bin")
event_id = trace.append_event("session-abc", 0, "User asked a question", 0)
history = trace.get_history("session-abc", limit=10)
for ev in history:
    print(f"  [{ev['role']}] {ev['text']}")
```
