<div align="center">
  <h1>Aeon</h1>
  <h3>A Neuro-Symbolic Cognitive Operating System</h3>
  
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a> <a href="https://isocpp.org/"><img src="https://img.shields.io/badge/Standard-C%2B%2B23-informational.svg" alt="Standard: C++23"></a> <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.12%2B-3776AB.svg" alt="Python: 3.12+"></a> <a href="https://cmake.org/"><img src="https://img.shields.io/badge/Build-CMake-064F8C.svg" alt="Build: CMake"></a> <img src="https://img.shields.io/badge/Status-Experimental-orange.svg" alt="Status: Experimental">
</div>

---

## Abstract

Aeon is a neuro-symbolic cognitive operating system that replaces flat Retrieval-Augmented Generation (RAG) architectures with a hierarchical **Memory Palace** abstraction. The system is organized around two core subsystems: **Atlas**, a SIMD-accelerated, memory-mapped B+ tree for O(log N) spatial navigation of semantic memory, and **Trace**, a directed acyclic graph (DAG) for episodic context tracking with backtracking and anchoring capabilities. A **Semantic Lookaside Buffer (SLB)** provides predictive caching of recently-accessed concept embeddings, reducing retrieval latency to sub-200ms for conversational workloads. The architecture enforces strict separation between the high-performance C++23 Core and the orchestration logic in the Python Shell, using zero-copy nanobind interop to eliminate serialization overhead.

---

## Architecture

The system follows a **Core-Shell** design pattern, partitioning responsibilities between a latency-critical C++ kernel and a flexible Python orchestration layer.

```
                        ┌─────────────────────────────────────────────────────────┐
                        │                     PYTHON SHELL                        │
                        │  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
                        │  │    Trace    │  │   Architect  │  │ CognitiveLoop │  │
                        │  │ (NetworkX)  │  │(Delta Graft) │  │  (FastAPI)    │  │
                        │  └──────┬──────┘  └──────┬───────┘  └───────┬───────┘  │
                        │         │                │                  │          │
                        │         └────────────────┴──────────────────┘          │
                        │                          │                             │
                        │                   Zero-Copy Interop                    │
                        │                     (nanobind)                         │
                        └──────────────────────────┬──────────────────────────────┘
                                                   │
                        ┌──────────────────────────┴──────────────────────────────┐
                        │                      C++23 CORE                         │
                        │  ┌─────────────────────────────────────────────────┐   │
                        │  │                     Atlas                       │   │
                        │  │  ┌───────────────┐       ┌───────────────────┐  │   │
                        │  │  │  MMap B+ Tree │       │        SLB        │  │   │
                        │  │  │   (Storage)   │       │ (Predictive Cache)│  │   │
                        │  │  └───────────────┘       └───────────────────┘  │   │
                        │  └─────────────────────────────────────────────────┘   │
                        │                          │                             │
                        │                   SIMD Math Kernel                     │
                        │                 (AVX2/AVX-512 Dispatch)                │
                        └─────────────────────────────────────────────────────────┘
```

### Terminology

| Term | Definition |
|------|------------|
| **Atlas** | The spatial memory engine. A memory-mapped hierarchical tree storing 768-dimensional embeddings with O(log N) navigational retrieval. Supports real-time ingestion via a write-through Delta Buffer. |
| **Trace** | The episodic memory graph. A NetworkX-backed DAG that records conversation history as User, System, and Concept nodes linked by typed edges (CAUSAL, NEXT, REFERS_TO). Enables backtracking and context anchoring. |
| **Architect** | The ingestion orchestrator. Implements Delta Grafting for real-time insertion of new knowledge without full tree reconstruction. |
| **SLB** | Semantic Lookaside Buffer. A predictive cache that stores recently-accessed concept centroids in contiguous memory, enabling sub-millisecond lookups for conversational drift queries. |
| **Glass Box** | The observability layer. Exposes internal state (active memory room, traversal path, cache statistics) via a streaming API for frontend visualization. |

---

## Key Innovations

- **Zero-Copy Interop.** C++ search results are exposed to Python as read-only NumPy structured arrays via nanobind capsules. No serialization or memory duplication occurs across the language boundary.

- **Spatial Indexing.** Hierarchical tree navigation achieves O(log N) retrieval complexity, compared to O(N) brute-force vector scans in flat RAG systems. The greedy descent algorithm uses SIMD-accelerated cosine similarity for branch selection.

- **Predictive Caching.** The Semantic Lookaside Buffer (SLB) pre-fetches concept embeddings based on episodic access patterns. Warm-start initialization from session history reduces cold-start latency by approximately 100x for repeat queries.

- **SIMD Acceleration.** The math kernel implements runtime dispatch to AVX2/AVX-512 instruction sets with 4x loop unrolling. Fallback to scalar operations ensures portability across architectures.

- **Real-Time Ingestion.** The Delta Buffer enables immediate availability of newly-inserted knowledge without blocking navigational queries. A shared mutex separates read and write paths.

---

## Quick Start

### Prerequisites

- Docker 24.0+
- Docker Compose v2

### Build and Run

```bash
# Clone the repository
git clone https://github.com/yourusername/aeon.git
cd aeon

# Build the containerized environment
docker compose up --build

# Execute a shell in the container
docker exec -it aeon-dev bash

# Run the benchmark suite (inside container)
./scripts/run_benchmarks.sh
```

### Local Development (without Docker)

```bash
# Create Python environment
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .

# Build C++ core with tests
cmake -B build -S core -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run C++ tests
./build/aeon_tests

# Run benchmarks
./build/aeon_bench
```

---

## Project Structure

```
aeon/
├── core/                   # C++23 High-Performance Kernel
│   ├── include/aeon/       # Public headers (atlas.hpp, slb.hpp, schema.hpp, ...)
│   ├── src/                # Implementation (atlas.cpp, simd_impl.cpp, bindings.cpp)
│   ├── benchmarks/         # Google Benchmark suite
│   └── tests/              # GTest unit tests
├── shell/                  # Python Orchestration Layer
│   └── aeon_py/            # Package (trace.py, context.py, client.py, ...)
├── tests/                  # Python integration tests
├── scripts/                # Utility scripts (run_benchmarks.sh, verify_setup.sh)
├── deploy/                 # Dockerfile and deployment configurations
├── docker-compose.yml      # Container orchestration
├── pyproject.toml          # Python package metadata (scikit-build-core)
└── CMakeLists.txt          # Root CMake configuration
```

---

## Benchmarks

Performance was measured on an **Apple M4 Max** (16-core CPU, 40-core GPU, 16-core Neural Engine) with 64GB Unified Memory. The Atlas was populated with 10,000 nodes (768-dimensional embeddings).

| Benchmark | Latency (median) | Notes |
|-----------|------------------|-------|
| Math Kernel (cosine similarity) | ~50 ns | Single 768-dim vector pair, AVX2 |
| Warm Search | ~1.2 ms | SLB hit path |
| Cold Search | ~45 ms | Full tree traversal, cache flushed |
| Conversational Drift (10 queries) | ~0.8 ms/query | Simulated topic coherence |

---

## Citation

If you use Aeon in your research, please cite:

```bibtex
@software{aeon2026,
  author       = {Arslan, Mustafa},
  title        = {Aeon: A Neuro-Symbolic Cognitive Operating System},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/mustafarslan/aeon},
  note         = {Experimental software. Version 0.1.0}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Copyright & Attribution Notice
### Official Research Title
Aeon: High-Performance Neuro-Symbolic Memory Management for Long-Horizon LLM Agents

arXiv Identifier: 2601.15311

Release Date: January 14, 2026

Author: Mustafa Arslan

### Legal & Usage Terms
#### © 2026 Mustafa Arslan. All rights reserved.

The architecture, methodology, and unique technical terminology—specifically the Semantic Lookaside Buffer (SLB), Atlas: Spatial Memory Kernel, and Neuro-Symbolic Trace (NST)—described in this research are the intellectual property of Mustafa Arslan.

1. Proper Attribution: Any reference, discussion, or summary of this work on social media (LinkedIn, X, Reddit) or academic platforms must explicitly credit Mustafa Arslan and link to the original arXiv:2601.15311 submission. 
2. Commercial Use: Unauthorized commercial implementation or "white-labeling" of the SLB architecture or Atlas Index methodology is strictly prohibited without prior written consent from the author. 
3. Social Media Scrapers: AI aggregators and newsletters utilizing this content for "Daily AI Digests" must include the author's name in the post preview/caption to avoid misattribution.
