<div align="center">

# Aeon

**Neuro-Symbolic Memory Kernel for Long-Horizon AI Agents**

[![C++23](https://img.shields.io/badge/C++-23-00599C?style=for-the-badge&logo=cplusplus&logoColor=white)](https://isocpp.org/) [![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE) [![CMake 3.26+](https://img.shields.io/badge/CMake-3.26+-064F8C?style=for-the-badge&logo=cmake&logoColor=white)](https://cmake.org/)

*A production C++23 memory system that gives AI agents the ability to remember, retrieve, and consolidate experience across unbounded conversation horizons.*

[Quick Start](#quick-start) · [Architecture](#architecture) · [Benchmarks](#benchmarks) · [Paper](#paper) · [Contributing](#contributing)

</div>

---

## Why Aeon?

Large language models are stateless. Every turn starts from scratch. **Aeon** provides the missing **persistent, queryable memory** that transforms a stateless LLM into a long-horizon agent:

| Problem | Aeon Solution |
|---------|--------------|
| Context window overflow | **Atlas** — page-clustered vector index with O(log N) greedy descent |
| No episodic recall | **Trace** — directed acyclic graph of typed episodes |
| Python GIL bottleneck | C++23 core with zero-copy **Nanobind** bindings |
| Cache-miss latency | **Semantic Lookaside Buffer (SLB)** — L1-resident similarity cache |
| SIMD portability | **SIMDe** abstraction: AVX-512 ↔ AVX2 ↔ NEON at compile time |

---

## Architecture

```
                        ┌─────────────────────────────────────┐
                        │         Python Shell (aeon_py)       │
                        │  client · session · trace · server   │
                        └──────────────┬──────────────────────┘
                                       │  Nanobind (zero-copy)
                        ┌──────────────▼──────────────────────┐
                        │          C++23 Kernel (core/)        │
                        │                                      │
                        │  ┌────────────┐   ┌──────────────┐  │
                        │  │   Atlas    │   │    Trace      │  │
                        │  │  (Spatial) │   │  (Episodic)   │  │
                        │  │  mmap B+   │   │   DAG + RAT   │  │
                        │  │  + SLB     │   │              │  │
                        │  └────────────┘   └──────────────┘  │
                        │                                      │
                        │  ┌──────────────────────────────────┐│
                        │  │  SIMD Math Kernel (SIMDe)        ││
                        │  │  AVX-512 · AVX2 · NEON           ││
                        │  └──────────────────────────────────┘│
                        │                                      │
                        │  ┌──────────────────────────────────┐│
                        │  │  Storage (mmap · zero-copy)      ││
                        │  └──────────────────────────────────┘│
                        └──────────────────────────────────────┘
```

### Core Abstractions

| Component | Header | Purpose |
|-----------|--------|---------|
| **Atlas** | `core/include/aeon/atlas.hpp` | Page-clustered spatial vector index over memory-mapped storage |
| **Trace** | `core/include/aeon/trace.hpp` | Episodic DAG with TEMPORAL, CAUSAL, and SEMANTIC edge types |
| **SLB** | `core/include/aeon/slb.hpp` | Semantic Lookaside Buffer — 64-entry L1-resident similarity cache |
| **Math Kernel** | `core/include/aeon/math_kernel.hpp` | SIMD-dispatched cosine similarity (AVX-512 → AVX2 → scalar) |
| **Storage** | `core/include/aeon/storage.hpp` | `mmap`-backed allocator with in-place growth via `ftruncate` |
| **Schema** | `core/include/aeon/schema.hpp` | Binary-stable `Node` layout (64-byte aligned, 3392 bytes) |

### Key Design Decisions

- **Immutable + Mutable Layers**: Atlas writes go to an in-memory delta buffer; reads merge the mmap-backed B+ tree with the delta layer. This avoids `msync` on every insert.
- **SLB Hit Path**: Queries first probe the SLB (cosine similarity ≥ `SLB_HIT_THRESHOLD`). On hit, the full tree traversal is bypassed entirely — yielding sub-microsecond latency for conversational locality.
- **Compile-Time Layout Verification**: `static_assert` enforces that `Node` is exactly 3392 bytes, trivially copyable, and that the `centroid` array starts at cache-line offset 64.

---

## Quick Start

### Prerequisites

| Dependency | Version | Notes |
|-----------|---------|-------|
| C++ compiler | Clang 16+ / GCC 13+ | Must support C++23 |
| CMake | 3.26+ | |
| Python | 3.10+ | |
| nanobind | 2.0+ | `pip install nanobind` |
| SIMDe | latest | `brew install simde` (macOS) or system package |

### Install via pip

```bash
# Clone and install (scikit-build-core handles CMake automatically)
git clone https://github.com/mustafarslan/aeon.git
cd aeon
pip install -e .
```

### Build from source (C++ only)

```bash
cd core

# Configure and build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Build with tests and benchmarks
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j$(nproc)

# Run tests
./build/aeon_tests

# Run benchmarks
./build/aeon_bench
```

### Python Usage

```python
from aeon_py import AeonClient

# Initialize with a storage directory
client = AeonClient(storage_dir="./memory")

# Add an episode
client.add_episode(
    content="The user asked about quantum computing fundamentals",
    embedding=model.encode("quantum computing fundamentals"),
)

# Semantic search (zero-copy: NumPy → C++ without allocation)
results = client.navigate(query_embedding, top_k=10)
```

---

## Repository Structure

```
aeon/
├── core/                          # C++23 kernel
│   ├── include/aeon/             # Public headers
│   │   ├── schema.hpp            #   Binary-stable Node layout + constants
│   │   ├── atlas.hpp             #   Spatial vector index
│   │   ├── trace.hpp             #   Episodic DAG
│   │   ├── slb.hpp               #   Semantic Lookaside Buffer
│   │   ├── math_kernel.hpp       #   SIMD math dispatch
│   │   ├── storage.hpp           #   mmap allocator
│   │   └── simd_impl.hpp         #   AVX-512/AVX2/NEON kernels
│   ├── src/                      # Implementation
│   │   ├── atlas.cpp             #   Atlas: navigate, insert, delta merge
│   │   ├── trace.cpp             #   Trace: DAG operations, consolidation
│   │   ├── bindings.cpp          #   Nanobind Python ↔ C++ bridge
│   │   └── simd_impl.cpp         #   SIMD kernel implementations
│   ├── tests/                    # GTest unit tests
│   ├── benchmarks/               # Google Benchmark suite
│   └── CMakeLists.txt
├── shell/aeon_py/                # Python shell
│   ├── client.py                 #   High-level AeonClient API
│   ├── session.py                #   Session management
│   ├── trace.py                  #   Python-side Trace operations
│   ├── server.py                 #   FastAPI server
│   └── ...
├── paper/                        # Academic paper (LaTeX)
├── reproducibility_benchmarks/   # Benchmark suite for paper
├── pyproject.toml                # scikit-build-core configuration
└── LICENSE                       # MIT
```

---

## Benchmarks

Benchmark suite located in `core/benchmarks/`. Run with:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j$(nproc)

./build/bench_kernel_throughput   # §6.1: SIMD math throughput
./build/bench_slb_latency         # §6.2: SLB hit/miss latency
./build/bench_scalability         # §6.3: Atlas scaling characteristics
```

### Benchmark Targets

| Binary | Measures |
|--------|----------|
| `bench_kernel_throughput` | Cosine similarity throughput (ops/sec) across vector dimensions |
| `bench_slb_latency` | SLB cache hit vs. cold-start navigate latency |
| `bench_scalability` | Insert and navigate scaling from 10³ to 10⁶ nodes |
| `aeon_bench` | Combined Google Benchmark suite |

---

## Thread Safety

Aeon uses `std::shared_mutex` with a reader-writer pattern:

| Operation | Lock Type | Contention |
|-----------|-----------|------------|
| `navigate()` | `shared_lock` (read) | Concurrent readers allowed |
| `insert()` / `insert_delta()` | `unique_lock` (write) | Exclusive; blocks readers |
| SLB `find_nearest()` | `shared_lock` (read) | Lock-free on hot path |
| SLB `insert()` | `unique_lock` (write) | < 1μs critical section |

The Python bindings release the GIL during all C++ operations, enabling true multi-threaded concurrency from Python.

---

## Paper

The theoretical foundations, design rationale, and experimental evaluation are documented in:

> **Aeon: A Neuro-Symbolic Memory System for Long-Horizon LLM Agents**

LaTeX source is in `paper/`. Pre-built PDFs:

- `aeon_neuro_symbolic_memory_llm.pdf` — Full paper
- `aeon_neuro_symbolic_memory_llm_v2.pdf` — Revised submission

---

## Configuration Constants

All architectural constants are defined in [`schema.hpp`](core/include/aeon/schema.hpp):

```cpp
constexpr size_t  EMBEDDING_DIM      = 768;    // Vector dimensionality
constexpr size_t  TOP_K_LIMIT        = 50;     // Max navigate() results
constexpr float   SLB_HIT_THRESHOLD  = 0.85f;  // SLB cache hit threshold
```

The `Node` struct is binary-stable at **3392 bytes** with `centroid` at cache-line offset 64, verified by `static_assert` at compile time.

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Ensure all tests pass: `cd core && cmake -B build -DBUILD_TESTING=ON && cmake --build build && ./build/aeon_tests`
4. Submit a pull request

### Code Style

- C++23 with Doxygen-style `///` comments
- `clang-format` with project `.clang-format` (if present)
- All magic numbers must be `constexpr` in `schema.hpp`

---

## License

[MIT](LICENSE) © 2026 Mustafa Arslan

---

## Citation

If you use Aeon in your research, please cite:

```bibtex
@article{arslan2026aeon,
  title   = {Aeon: A Neuro-Symbolic Memory System for Long-Horizon LLM Agents},
  author  = {Arslan, Mustafa},
  year    = {2026},
}
```
