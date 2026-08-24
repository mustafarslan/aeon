# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Aeon is a persistent, crash-recoverable **semantic memory engine** (vector search over memory-mapped
storage) for AI agents, game engines, and robotics. It follows a strict **Core-Shell** split:

- **Kernel / "Ring 0"** (`core/`): C++23, compiled to a static lib (`aeon_core`), a Python extension
  (`aeon_py_core`, via nanobind), and a universal C-ABI shared library (`aeon_shared` /
  `libaeon.{so,dylib,dll}`) for game engines. All hot-path work (mmap I/O, SIMD search, WAL,
  quantization, compaction, concurrency) lives here and nowhere else.
- **Shell / "Ring 3"** (`shell/aeon_py/`): Python orchestration layer that binds to the kernel via
  nanobind. Handles session management, prompt construction, LLM calls, telemetry — never touches
  raw memory or does heavy computation itself.
- **Language bindings** (`bindings/`): C# (Unity/Godot), C++ header-only (Unreal), Node.js N-API
  (macOS ARM64), all consuming the same `aeon_c_api.h` C-ABI as the Python shell.

Read `ARCHITECTURE.md` before making non-trivial changes to `core/` — it documents the WAL 3-step
lock ordering protocol, the Sidecar Blob Arena, INT8 quantization scheme, EBR concurrency model, and
navigate query flow in detail. `CODE_WALKTHROUGH.md` and `INTERNALS.md` cover API usage and on-disk
struct layouts respectively.

## Build & test (C++ core)

Requires CMake 3.26+, a C++23 compiler, Python 3.10+, SIMDe (`brew install simde` /
`apt install libsimde-dev`), and BLAS (Accelerate on macOS, OpenBLAS elsewhere).

```bash
# One-shot: configure + build + test
./build.sh              # dev preset (native arch, RelWithDebInfo)
./build.sh release      # release preset (LTO, fast-math)
./build.sh bench        # dev build + runs the 3 core benchmark binaries
./build.sh clean        # wipe build/dev, build/release, build/ci-*

# Equivalent manual steps
cmake --preset dev
cmake --build --preset dev
ctest --preset dev --output-on-failure
```

Presets are defined in `CMakePresets.json` (`dev`, `release`, `ci-linux`, `ci-macos`, `ci-windows`,
`ci-ios`). CI presets (`AEON_CI_BUILD=ON`) use portable arch flags (`x86-64-v3` / `apple-m1`) instead
of `-march=native`, and set `CMAKE_BUILD_TYPE=Release`.

Run a single GTest binary/filter directly after building:

```bash
ctest --preset dev -R test_wal              # by CTest-registered name (gtest_discover_tests)
build/dev/bin/aeon_tests --gtest_filter=WalTest.*
```

Test sources live in `core/tests/` (`test_atlas.cpp`, `test_wal.cpp`, `test_blob_arena.cpp`,
`test_quantization.cpp`, `test_epoch.cpp`, `test_concurrency.cpp`, `test_storage.cpp`,
`test_math.cpp`) and are all linked into the single `aeon_tests` target.

### Benchmarks

`core/CMakeLists.txt` registers many standalone Google Benchmark executables (built only when
`BUILD_TESTING=ON`), e.g. `bench_wal_overhead`, `bench_quantization_efficiency`,
`bench_ebr_contention`, `bench_beam_search`, `bench_trace_gc`, `bench_multitenant_slb`,
`bench_tiered_atlas`. The full reproducibility sweep:

```bash
./reproducibility_benchmarks/run_all_benchmarks.sh   # writes reproducibility_benchmarks/master_metrics.txt
```

Numbers quoted in README.md/ARCHITECTURE.md (e.g. 2.23µs insert, 3.09µs INT8 navigate, 4.70ns SDOT
dot product) are pulled from that file — regenerate it rather than hand-editing quoted benchmark
numbers in docs.

## Build & test (Python shell)

```bash
pip install -e .          # builds the nanobind extension + installs shell/aeon_py as `aeon_py`
                           # (root pyproject.toml drives this: cmake.targets=["aeon_py_core"],
                           #  wheel.packages=["shell/aeon_py"] — NOT `pip install -e ./shell`,
                           #  there is no pyproject.toml under shell/)
pytest tests/              # root-level integration tests import `aeon_py.*`
```

Regenerate Python type stubs for the compiled extension after changing `core/src/bindings.cpp`:

```bash
./scripts/gen_stubs.sh    # writes shell/aeon_py/core.pyi via nanobind.stubgen
```

Run the FastAPI dev server (chat/session API over the cognitive loop):

```bash
./scripts/run_server.sh   # uvicorn aeon_py.server:app --reload, port 8000
```

## Node.js bridge (macOS Apple Silicon only)

```bash
cmake --build build --target aeon_shared    # build libaeon.dylib first (from repo root)
cd bindings/node
npm install
npm run build              # cmake-js compile -T aeon_node -a arm64
npm run bench               # node bench_bridge.js
```

## Architecture notes worth knowing before editing

- **Dual-layer memory:** `Atlas` (`core/include/aeon/atlas.hpp`) is the spatial/concept index — a
  tree of `NodeHeader` structs reached by descent through `first_child_offset` (it's a tree walk,
  not a flat array — any per-node filtering scheme must account for subtree skipping). `Trace`
  (`core/include/aeon/trace.hpp`) is the episodic/chronological log of `TraceEvent` structs, backed
  by an unlimited-length Sidecar `BlobArena` for full text plus a 64-byte inline preview.
- **Dynamic dimensionality:** node byte stride is computed at runtime from `AtlasHeader` (`dim`,
  `metadata_size`, `quantization_type`) via `compute_node_stride()` — one binary serves any
  embedding width (384/768/1536/...) at either FP32 or INT8 precision.
- **WAL:** strict lock ordering `serialize (no lock) → wal_mutex_ → delta_mutex_`. Never acquire
  `delta_mutex_` before the WAL write completes — that's what keeps insert latency decoupled from
  `fflush()`.
- **INT8 quantization:** symmetric, per-vector `scale` stored in `NodeHeader`. The `HierarchicalSLB`
  cache deliberately stores **FP32 only** (dequantized on insert) even for INT8-backed Atlases, to
  keep cache-hit latency off the dequantization path — don't "optimize" this by caching INT8.
  L1 is a sharded per-session `SessionRingBuffer` (64 shards), L2 is a global `SemanticCache`.
- **Concurrency:** Epoch-Based Reclamation (`core/include/aeon/epoch.hpp`, `EpochManager` /
  `EpochGuard`) — readers never lock; retired mmap regions are only freed once all active readers
  have advanced past the retiring epoch. Compaction uses double-buffered shadow copies (Redis
  `BGSAVE`-style) so the main thread is never blocked proportional to dataset size.
- **C-API surface** (`core/include/aeon/aeon_c_api.h`, `core/src/aeon_c_api.cpp`): the only supported
  integration point for non-Python consumers. Opaque pointers, caller-allocated result buffers, no
  exceptions across the boundary — every function returns an `aeon_error_t`. All new kernel
  capabilities exposed to game engines / other languages must be added here, mirroring the pattern
  of `aeon_atlas_create_ex()` (options struct) and `aeon_trace_get_event_text()` (caller buffer).
- **Compiler flags differ by target within `core/CMakeLists.txt`:** `aeon_core` (the math kernels)
  gets `-ffast-math -flto`; `aeon_shared` (the C-ABI wrapper) deliberately does not, since it's a
  thin dispatch layer, not a numerics hot path. GTest links against `aeon_core` and relies on
  IEEE Inf/NaN behavior — don't apply `-ffast-math` to the test target.
- **Python shell layering** (`shell/aeon_py/`): `client.py` (`AeonClient`/`TieredClient` — zero-copy
  Atlas queries + edge/cloud fallback), `context.py` (`ContextManager` — orchestrates Atlas + Trace
  together per turn), `loop.py` (`CognitiveLoop` — input → memory → LLM → response),
  `session.py` (`SessionManager` — multi-tenant LRU session isolation), `dreamer.py`
  (`DreamingWorker` — background consolidation: aging subgraphs get LLM-summarized then
  tombstoned/compacted via the kernel), `architect.py` (short-term delta ingestion),
  `server.py` (FastAPI app wiring these together over SSE).
- **`v4/docs/`** (currently untracked, not yet reflected in code) holds forward-looking
  design/roadmap notes for a next-generation redesign — distinct from the shipped "V4.1" release
  described in README.md/ARCHITECTURE.md. Don't conflate the two when reading docs or planning work.
