# Aeon Memory OS

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/aeon-ag/aeon)
[![Version](https://img.shields.io/badge/version-4.0.0-blue)](https://github.com/aeon-ag/aeon/releases)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Aeon** is a persistent, crash-recoverable **Semantic Memory Engine** for AI agents, game engines, and robotics. It provides a shared memory substrate where thousands of independent agents — or a single massively parallel system — can *remember, retrieve, and forget* knowledge in real time.

> **V4.0 Enterprise Hardening:** Now featuring Dynamic Dimensionality, C++ Binary Trace, and stutter-free Double-Buffered Shadow Compaction.

---

## Key Features

| Feature | Description |
|---|---|
| **Dual-Layer Memory** | Combines **Atlas** (Spatial Index) for concepts and **Trace** (Episodic Log) for experiences. |
| **Dynamic Dimensionality** | Single binary supports any embedding dim (384, 768, 1536) via runtime stride calculation. |
| **Zero-Copy Architecture** | Mmap-backed C++ kernel with direct zero-copy bindings to Python (`nanobind`). |
| **Shadow Compaction** | "Redis-style" stutter-free garbage collection for 60 FPS real-time apps. |
| **Multi-Tenant C-API** | Strictly typed, session-routed C interface for Unity, Unreal, and Godot. |
| **Hierarchical SLB** | Sharded Semantic Lookaside Buffer (L1/L2) handling 100,000+ concurrent sessions. |
| **Trace Block Index** | Sub-linear `O(|V|/1024)` search over episodic history using block centroid scanning. |

---

## Architecture

```mermaid
graph TD
    User[User / Agent] -->|Request| Py[Python Shell (aeon.py)]
    Py -->|nanobind| C[C++ Kernel (libaeon)]
    
    subgraph "Kernel (Ring 0)"
        C -->|Insert/Query| Atlas[Atlas (Spatial)]
        C -->|Log Event| Trace[Trace (Episodic)]
        C -->|Cache| SLB[Hierarchical SLB]
        
        Atlas -->|MMap| AFile[atlas.bin]
        Trace -->|MMap| TFile[trace.bin]
    end
    
    subgraph "Storage"
        AFile
        TFile
    end
```

---

## Quick Start

### Prerequisites

- CMake 3.25+
- C++23 Compiler (Clang 16+, GCC 13+, MSVC 19.34+)
- Python 3.10+

### Build from Source

```bash
# Clone repository
git clone https://github.com/aeon-ag/aeon.git
cd aeon

# Configure and build (Development Preset)
cmake --preset dev
cmake --build build/dev --parallel
```

### Python Usage

```python
import aeon

# 1. Open an Atlas (768-dim)
atlas = aeon.Atlas("memory/atlas.bin", dim=768)

# 2. Insert detailed knowledge
node_id = atlas.insert(
    vector=[0.1, 0.5, ...], 
    metadata="The mitochondrion is the powerhouse of the cell."
)

# 3. Search (Navigate)
results = atlas.navigate(
    query=[0.1, 0.4, ...], 
    top_k=5
)
print(f"Nearest concept: {results[0].metadata}")
```

---

## C-API Example

Aeon provides a stable ABI for integration with game engines.

```c
#include "aeon_c_api.h"

int main() {
    aeon_atlas_t* atlas;
    aeon_error_t err = aeon_atlas_create("game_memory.bin", 768, &atlas);
    
    if (err == AEON_OK) {
        // ... game logic ...
        aeon_atlas_destroy(atlas);
    }
    return 0;
}
```

---

## Performance

Benchmarks run on Apple M3 Max (Active Cooling):

- **Insert Throughput:** 12,500 ops/sec (Threaded)
- **Query Latency:** 2.1ms (p95) @ 1M vectors
- **Compaction Pause:** < 15µs (Main thread stall)
- **Memory Overhead:** ~1.2x raw vector size (Zero-copy)

---

## License

MIT License. See [LICENSE](LICENSE) for details.
