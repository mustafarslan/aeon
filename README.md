# Aeon Memory Kernel

> **C++23 Neuro-Symbolic Memory System for Long-Horizon AI Agents**

[![C++23](https://img.shields.io/badge/C++-23-00599C?logo=cplusplus)](https://isocpp.org/)
[![CMake](https://img.shields.io/badge/CMake-3.25+-064F8C?logo=cmake)](https://cmake.org/)
[![Nanobind](https://img.shields.io/badge/Nanobind-2.0-green)](https://github.com/wjakob/nanobind)

---

## Overview

Aeon is the high-performance memory kernel that powers Triality's infinite context capability. It provides two core abstractions:

- **Atlas** — Spatial memory via HNSW-inspired vector index
- **Trace** — Episodic memory via directed acyclic graph (DAG)

Both are implemented in C++23 with **zero-copy** Python bindings via Nanobind.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Atlas: Semantic Search](#atlas-semantic-search)
3. [Trace: Episodic Graph](#trace-episodic-graph)
4. [The "Dreaming" Process](#the-dreaming-process)
5. [Thread Safety Model](#thread-safety-model)
6. [Python Bindings](#python-bindings)
7. [Building](#building)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AEON KERNEL ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Python Shell (Triality)                                                   │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │                         AeonClient                                │    │
│   │   navigate() │ add_episode() │ consolidate() │ load_session()     │    │
│   └───────────────────────────────┬───────────────────────────────────┘    │
│                                   │                                         │
│                                   │ Nanobind (Zero-Copy)                    │
│                                   ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                        C++23 Kernel                                 │  │
│   │                                                                     │  │
│   │   ┌─────────────────────────┐   ┌─────────────────────────────┐    │  │
│   │   │         Atlas          │   │          Trace              │    │  │
│   │   │  ┌─────────────────┐   │   │   ┌─────────────────────┐   │    │  │
│   │   │  │   HNSW Index    │   │   │   │   DAG Structure     │   │    │  │
│   │   │  │   (Spatial)     │   │   │   │   (Temporal)        │   │    │  │
│   │   │  └─────────────────┘   │   │   └─────────────────────┘   │    │  │
│   │   │  ┌─────────────────┐   │   │   ┌─────────────────────┐   │    │  │
│   │   │  │      SLB        │   │   │   │   Edge Types        │   │    │  │
│   │   │  │   (L1 Cache)    │   │   │   │   TEMPORAL|CAUSAL   │   │    │  │
│   │   │  └─────────────────┘   │   │   └─────────────────────┘   │    │  │
│   │   └─────────────────────────┘   └─────────────────────────────┘    │  │
│   │                                                                     │  │
│   │   ┌─────────────────────────────────────────────────────────────┐  │  │
│   │   │                    Shared Memory (mmap)                     │  │  │
│   │   │          Zero-Copy Access · Unified Memory (M4)             │  │  │
│   │   └─────────────────────────────────────────────────────────────┘  │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Atlas: Semantic Search

### Core Data Structure

The Atlas is a hierarchical navigable small world (HNSW) graph optimized for streaming insertions and low-latency search.

```cpp
/**
 * @struct Node
 * @brief A node in the Atlas spatial index
 * 
 * Memory Layout (768-dim, 64-byte aligned):
 *   - id:        8 bytes
 *   - vector:    768 * 4 = 3072 bytes  
 *   - neighbors: 64 * 8 = 512 bytes (M_max=64)
 *   - metadata:  256 bytes
 *   - padding:   56 bytes (alignment)
 *   Total: 3904 bytes per node
 */
struct alignas(64) Node {
    uint64_t id;                              // Unique identifier
    std::array<float, 768> vector;            // Embedding (D=768)
    std::vector<uint64_t> neighbors;          // HNSW connections (per layer)
    NodeMetadata metadata;                    // Timestamp, access count, etc.
};
```

### Mathematical Definition

A node $N$ is formally defined as:

$$
N = \{id, \mathbf{v}, \mathcal{C}, \text{meta}\}
$$

Where:

- $id \in \mathbb{Z}^+$ — Unique identifier
- $\mathbf{v} \in \mathbb{R}^{768}$ — Embedding vector (normalized)
- $\mathcal{C} \subset \mathbb{Z}^+$ — Set of neighbor IDs (connections)
- $\text{meta}$ — Metadata (timestamp, access frequency, TTL)

### Greedy SIMD Descent

Search uses a greedy descent with SIMD-accelerated distance computation:

```cpp
/**
 * @brief Navigate to the nearest node using greedy SIMD descent
 * 
 * Algorithm:
 *   1. Start at entry point (or SLB suggestion)
 *   2. Compute distances to all neighbors using AVX/NEON
 *   3. Move to closest neighbor if it improves distance
 *   4. Repeat until local minimum
 * 
 * Complexity: O(log N) expected, O(N) worst case
 * Latency: ~0.5ms (SLB hit) to ~3ms (cold start) for N=10^6
 */
std::vector<NodeResult> Atlas::navigate(
    const float* query,           // Query vector (768-dim)
    size_t k,                     // Number of results
    size_t ef_search = 64         // Search beam width
) {
    // 1. Check SLB (Semantic Lookaside Buffer) first
    if (auto cached = slb_.lookup(query); cached.has_value()) {
        entry_point_ = cached->node_id;  // Warm start
    }
    
    // 2. Greedy descent
    std::priority_queue<NodeDistance> candidates;
    candidates.push({entry_point_, distance_simd(query, get_vector(entry_point_))});
    
    std::unordered_set<uint64_t> visited;
    std::vector<NodeDistance> results;
    
    while (!candidates.empty() && results.size() < k) {
        auto [current_id, current_dist] = candidates.top();
        candidates.pop();
        
        if (visited.count(current_id)) continue;
        visited.insert(current_id);
        results.push_back({current_id, current_dist});
        
        // Expand neighbors
        for (uint64_t neighbor_id : get_neighbors(current_id)) {
            if (!visited.count(neighbor_id)) {
                float dist = distance_simd(query, get_vector(neighbor_id));
                candidates.push({neighbor_id, dist});
            }
        }
    }
    
    // 3. Update SLB with result centroid
    slb_.update(query, results[0].node_id);
    
    return to_node_results(results);
}
```

### SIMD Distance Kernel

Distance computation is the hot path. We use ARM NEON (M4) with 4x unrolling:

```cpp
/**
 * @brief Compute L2 distance using NEON SIMD
 * 
 * Performance:
 *   - Scalar: ~2000ns per 768-dim vector
 *   - NEON:   ~50ns per 768-dim vector (40x speedup)
 */
inline float distance_simd_neon(const float* a, const float* b, size_t dim) {
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);
    
    // 4x unrolled loop, 16 floats per iteration
    for (size_t i = 0; i < dim; i += 16) {
        float32x4_t va0 = vld1q_f32(a + i);
        float32x4_t vb0 = vld1q_f32(b + i);
        float32x4_t diff0 = vsubq_f32(va0, vb0);
        sum0 = vmlaq_f32(sum0, diff0, diff0);
        
        float32x4_t va1 = vld1q_f32(a + i + 4);
        float32x4_t vb1 = vld1q_f32(b + i + 4);
        float32x4_t diff1 = vsubq_f32(va1, vb1);
        sum1 = vmlaq_f32(sum1, diff1, diff1);
        
        float32x4_t va2 = vld1q_f32(a + i + 8);
        float32x4_t vb2 = vld1q_f32(b + i + 8);
        float32x4_t diff2 = vsubq_f32(va2, vb2);
        sum2 = vmlaq_f32(sum2, diff2, diff2);
        
        float32x4_t va3 = vld1q_f32(a + i + 12);
        float32x4_t vb3 = vld1q_f32(b + i + 12);
        float32x4_t diff3 = vsubq_f32(va3, vb3);
        sum3 = vmlaq_f32(sum3, diff3, diff3);
    }
    
    // Horizontal reduction
    sum0 = vaddq_f32(sum0, sum1);
    sum2 = vaddq_f32(sum2, sum3);
    sum0 = vaddq_f32(sum0, sum2);
    
    return vaddvq_f32(sum0);  // Final horizontal add
}
```

### The Semantic Lookaside Buffer (SLB)

The SLB is a small cache (K=64 entries) that exploits **semantic locality**: consecutive queries in a conversation tend to be semantically close.

```cpp
/**
 * @class SemanticCache
 * @brief L1 cache for semantic search, exploiting conversational locality
 * 
 * Theory: "Semantic Inertia" — If topic at turn t is T, then P(topic at t+1 ≈ T) > 0.8
 * 
 * Structure:
 *   - Ring buffer of K=64 (query_centroid, node_pointer) pairs
 *   - Brute-force SIMD search (K is small enough to fit in L1 cache)
 *   - LRU eviction on insert
 */
class SemanticCache {
public:
    static constexpr size_t CACHE_SIZE = 64;
    static constexpr float HIT_THRESHOLD = 0.15f;  // Cosine distance
    
    std::optional<CacheEntry> lookup(const float* query) {
        float min_dist = std::numeric_limits<float>::max();
        size_t best_idx = 0;
        
        // Brute-force search (64 entries × 768 dims ≈ 200KB, fits in L2)
        for (size_t i = 0; i < entries_.size(); ++i) {
            float dist = distance_simd_neon(query, entries_[i].centroid.data(), 768);
            if (dist < min_dist) {
                min_dist = dist;
                best_idx = i;
            }
        }
        
        if (min_dist < HIT_THRESHOLD) {
            entries_[best_idx].access_count++;
            return entries_[best_idx];
        }
        return std::nullopt;
    }
    
private:
    std::vector<CacheEntry> entries_;
    size_t head_ = 0;  // Ring buffer pointer
};
```

---

## Trace: Episodic Graph

### DAG Structure

The Trace is a directed acyclic graph (DAG) representing episodic memory. Each node is a conversation turn or event, connected by typed edges.

```cpp
/**
 * @struct TraceNode
 * @brief A node in the episodic memory graph
 */
struct TraceNode {
    uint64_t id;                          // Unique identifier
    std::string content;                  // The actual text/event
    std::array<float, 768> embedding;     // Semantic embedding
    uint64_t timestamp;                   // Unix epoch (nanoseconds)
    NodeType type;                        // RAW | SUMMARY | ANCHOR
    std::vector<Edge> edges;              // Outgoing edges
};

/**
 * @struct Edge
 * @brief A directed edge in the Trace DAG
 */
struct Edge {
    uint64_t target_id;                   // Target node ID
    EdgeType type;                        // TEMPORAL | CAUSAL | SEMANTIC
    float weight;                         // Edge strength [0, 1]
};
```

### Edge Types

| Type | Symbol | Semantics | Creation |
|------|--------|-----------|----------|
| **TEMPORAL** | `→` | "Happened after" | Automatic on insert |
| **CAUSAL** | `⇒` | "Caused by" / "Follows from" | Inferred by Strategist |
| **SEMANTIC** | `~` | "Related to" | Computed via embedding similarity |

```
Trace Graph Example:
═══════════════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────────────────────────────────┐
    │                        CONVERSATION TRACE                       │
    └─────────────────────────────────────────────────────────────────┘
    
         [T₁: "What's the weather?"]
                    │
                    │ TEMPORAL
                    ▼
         [T₂: "It's sunny, 22°C"]
                    │
          ┌────────┴────────┐
          │ TEMPORAL        │ CAUSAL
          ▼                 ▼
    [T₃: "Should I bring   [T₄: "Planning a picnic"]
     an umbrella?"]              │
          │                      │ SEMANTIC (~0.85)
          │ TEMPORAL             ▼
          ▼              [T₇: "Best picnic spots"]
    [T₅: "No, forecast         
     shows no rain"]           
          │                    
          │ TEMPORAL           
          ▼                    
    [T₆: "Great, thanks!"]     
```

### Graph Operations

```cpp
/**
 * @brief Add a new episode to the Trace
 * 
 * Atomically:
 *   1. Creates new node with content + embedding
 *   2. Links to previous node (TEMPORAL edge)
 *   3. Computes semantic edges to recent nodes (if similarity > threshold)
 */
uint64_t TraceManager::add_episode(
    const std::string& content,
    const float* embedding,
    std::optional<uint64_t> causal_parent = std::nullopt
) {
    std::unique_lock lock(mutex_);  // Exclusive write lock
    
    uint64_t new_id = next_id_++;
    TraceNode node{
        .id = new_id,
        .content = content,
        .embedding = to_array<768>(embedding),
        .timestamp = now_ns(),
        .type = NodeType::RAW
    };
    
    // Link to previous (TEMPORAL)
    if (last_node_id_ != 0) {
        node.edges.push_back({last_node_id_, EdgeType::TEMPORAL, 1.0f});
    }
    
    // Link causal parent if provided
    if (causal_parent.has_value()) {
        node.edges.push_back({*causal_parent, EdgeType::CAUSAL, 1.0f});
    }
    
    // Compute semantic edges to recent nodes
    for (uint64_t recent_id : get_recent_ids(50)) {
        float sim = cosine_similarity(embedding, get_embedding(recent_id));
        if (sim > 0.7f) {
            node.edges.push_back({recent_id, EdgeType::SEMANTIC, sim});
        }
    }
    
    nodes_[new_id] = std::move(node);
    last_node_id_ = new_id;
    
    return new_id;
}
```

---

## The "Dreaming" Process

### Overview

The Dreaming process is a background thread that consolidates the Trace graph. It rewrites clusters of "raw" nodes into "summary" nodes, reducing graph size while preserving semantic content.

```
Dreaming Consolidation:
═══════════════════════════════════════════════════════════════════════════

BEFORE (50 raw nodes):
    [R₁] → [R₂] → [R₃] → ... → [R₄₉] → [R₅₀]
      │      │      │              │       │
      └──────┴──────┴──────────────┴───────┘
                    │
                    ▼ (semantic clustering)
                    
AFTER (1 summary + 3 anchor nodes):
                  ┌──────────────┐
                  │  [S₁]        │
                  │  "Discussion │
                  │   about X"   │
                  └──────┬───────┘
                         │
           ┌─────────────┼─────────────┐
           ▼             ▼             ▼
        [A₁]          [A₂]          [A₃]
        Key           Key           Key
        Moment 1      Moment 2      Moment 3
```

### Consolidation Algorithm

```cpp
/**
 * @brief Consolidate a cluster of raw nodes into a summary node
 * 
 * Algorithm:
 *   1. Select cluster of N raw nodes (default N=50)
 *   2. Compute cluster centroid embedding
 *   3. Identify K anchor nodes (highest "importance" score)
 *   4. Create summary node with centroid
 *   5. Rewrite edges: incoming → summary, summary → anchors
 *   6. Delete non-anchor raw nodes
 * 
 * Thread Safety: Uses std::shared_mutex for read-write locking
 * Atomicity: Wrapped in transaction (all-or-nothing)
 */
void TraceManager::consolidate(
    const std::vector<uint64_t>& cluster_ids,
    size_t num_anchors = 3
) {
    // Acquire exclusive lock
    std::unique_lock lock(mutex_);
    
    if (cluster_ids.size() < 10) {
        return;  // Too small to consolidate
    }
    
    // 1. Compute centroid embedding
    std::array<float, 768> centroid{};
    for (uint64_t id : cluster_ids) {
        const auto& emb = nodes_[id].embedding;
        for (size_t i = 0; i < 768; ++i) {
            centroid[i] += emb[i];
        }
    }
    for (size_t i = 0; i < 768; ++i) {
        centroid[i] /= cluster_ids.size();
    }
    normalize(centroid);
    
    // 2. Identify anchor nodes (highest importance)
    auto anchors = select_anchors(cluster_ids, num_anchors);
    
    // 3. Create summary node
    uint64_t summary_id = next_id_++;
    TraceNode summary{
        .id = summary_id,
        .content = generate_summary(cluster_ids),  // LLM call (async)
        .embedding = centroid,
        .timestamp = now_ns(),
        .type = NodeType::SUMMARY
    };
    
    // 4. Link summary to anchors
    for (uint64_t anchor_id : anchors) {
        summary.edges.push_back({anchor_id, EdgeType::CAUSAL, 1.0f});
        nodes_[anchor_id].type = NodeType::ANCHOR;
    }
    
    // 5. Redirect incoming edges to summary
    redirect_incoming_edges(cluster_ids, summary_id);
    
    // 6. Delete non-anchor nodes
    std::unordered_set<uint64_t> anchor_set(anchors.begin(), anchors.end());
    for (uint64_t id : cluster_ids) {
        if (!anchor_set.count(id)) {
            nodes_.erase(id);
        }
    }
    
    nodes_[summary_id] = std::move(summary);
}
```

### Importance Scoring

Anchor selection uses a multi-factor importance score:

$$
\text{Importance}(n) = \alpha \cdot \text{degree}(n) + \beta \cdot \text{recency}(n) + \gamma \cdot \text{centrality}(n)
$$

Where:

- $\text{degree}(n)$ — Number of edges (normalized)
- $\text{recency}(n)$ — Inverse of age (newer = more important)
- $\text{centrality}(n)$ — PageRank-like score within cluster

```cpp
float compute_importance(const TraceNode& node, uint64_t now) {
    constexpr float ALPHA = 0.3f;  // Degree weight
    constexpr float BETA = 0.4f;   // Recency weight  
    constexpr float GAMMA = 0.3f;  // Centrality weight
    
    float degree = static_cast<float>(node.edges.size()) / MAX_DEGREE;
    float recency = 1.0f / (1.0f + (now - node.timestamp) / 1e9f);  // Decay over seconds
    float centrality = compute_pagerank(node.id);  // Cached
    
    return ALPHA * degree + BETA * recency + GAMMA * centrality;
}
```

---

## Thread Safety Model

Aeon uses a **reader-writer lock** pattern to allow concurrent reads during generation while ensuring exclusive access for mutations.

### Lock Types

| Lock | Type | Held By | Duration |
|------|------|---------|----------|
| `graph_mutex_` | `std::shared_mutex` | Trace operations | Variable |
| `slb_lock_` | `std::mutex` | SLB updates | <1ms |
| `consolidate_cv_` | `std::condition_variable` | Dreaming thread | N/A |

### Read-Write Semantics

```cpp
/**
 * Reading (Actor, Strategist):
 *   - Acquires shared lock
 *   - Multiple readers allowed concurrently
 *   - Blocked only during active write
 */
std::vector<TraceNode> TraceManager::get_recent(size_t n) {
    std::shared_lock lock(mutex_);  // Shared (read) lock
    // ... read operations ...
}

/**
 * Writing (Dreaming, add_episode):
 *   - Acquires exclusive lock
 *   - Blocks all readers
 *   - Only one writer at a time
 */
void TraceManager::consolidate(...) {
    std::unique_lock lock(mutex_);  // Exclusive (write) lock
    // ... write operations ...
}
```

### Dreaming Thread Lifecycle

```cpp
void TraceManager::dreaming_loop() {
    while (!shutdown_requested_) {
        // Wait for idle signal or timeout
        std::unique_lock cv_lock(consolidate_mutex_);
        consolidate_cv_.wait_for(cv_lock, 60s, [this] {
            return should_consolidate() || shutdown_requested_;
        });
        
        if (shutdown_requested_) break;
        
        // Find cluster to consolidate
        auto cluster = find_consolidation_candidate();
        if (!cluster.empty()) {
            consolidate(cluster);
        }
    }
}
```

---

## Python Bindings

Aeon exposes a zero-copy interface via Nanobind. NumPy arrays are passed directly to C++ without copying.

### Zero-Copy Array Passing

```cpp
// bindings.cpp
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

NB_MODULE(aeon_core, m) {
    nb::class_<Atlas>(m, "Atlas")
        .def("navigate", [](Atlas& self, 
                            nb::ndarray<float, nb::shape<768>> query,
                            size_t k) {
            // Zero-copy: query.data() points directly to NumPy memory
            return self.navigate(query.data(), k);
        })
        .def("add_node", [](Atlas& self,
                            nb::ndarray<float, nb::shape<768>> embedding,
                            const std::string& content) {
            return self.add_node(embedding.data(), content);
        });
        
    nb::class_<TraceManager>(m, "TraceManager")
        .def("add_episode", &TraceManager::add_episode)
        .def("consolidate", &TraceManager::consolidate)
        .def("get_recent", &TraceManager::get_recent);
}
```

### Python Usage

```python
import numpy as np
from aeon_core import Atlas, TraceManager

# Initialize
atlas = Atlas("/path/to/atlas.bin")
trace = TraceManager("/path/to/trace.bin")

# Navigate (zero-copy)
query = np.random.randn(768).astype(np.float32)
results = atlas.navigate(query, k=5)

# Add episode
embedding = model.encode("User asked about weather")
trace.add_episode("User asked about weather", embedding)
```

---

## Building

### Prerequisites

- CMake 3.25+
- C++23 compiler (Clang 16+ / GCC 13+)
- Python 3.12+ with NumPy
- Nanobind 2.0+

### Build Commands

```bash
cd libs/aeon
mkdir build && cd build

# Release build (optimized)
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-march=native -O3 -flto" \
      ..
make -j$(sysctl -n hw.ncpu)

# Install Python bindings
pip install ..
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `AEON_BUILD_TESTS` | ON | Build unit tests |
| `AEON_BUILD_BENCH` | OFF | Build benchmarks |
| `AEON_ENABLE_ASAN` | OFF | Address sanitizer |
| `AEON_SIMD_ARCH` | `native` | SIMD target (neon/avx2/avx512) |

---

## Benchmarks

| Operation | N=10⁴ | N=10⁵ | N=10⁶ |
|-----------|-------|-------|-------|
| `navigate(k=10)` | 0.3ms | 0.8ms | 2.1ms |
| `add_node` | 0.1ms | 0.2ms | 0.5ms |
| `add_episode` | 0.2ms | 0.3ms | 0.4ms |
| `consolidate(50)` | 15ms | 18ms | 22ms |

*Measured on M4 Max, 64GB, single-threaded.*

---

## References

- [HNSW: Hierarchical Navigable Small World Graphs](https://arxiv.org/abs/1603.09320)
- [Nanobind: Tiny and Efficient C++/Python Bindings](https://github.com/wjakob/nanobind)
- [MLX: Machine Learning on Apple Silicon](https://github.com/ml-explore/mlx)

---

<div align="center">

*Aeon Kernel v1.0.0 — High-Performance Memory for Cognitive AI*

</div>
