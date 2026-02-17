# Aeon Universal Language Bindings

Production-ready bindings for the Aeon Memory OS C-API (`aeon_c_api.h`).

## Available Bindings

| Language | Path | Target Platform |
|----------|------|-----------------|
| **C# (.NET)** | [`csharp/AeonMemory.cs`](csharp/AeonMemory.cs) | Unity 2022+, Godot 4.x, .NET 6/8 |
| **C++ (Header-Only)** | [`cpp_unreal/AeonClient.hpp`](cpp_unreal/AeonClient.hpp) | Unreal Engine 5.x, Standalone C++ |
| **Node.js (N-API)** | [`node/`](node/) | OpenClaw Agent Framework, macOS Apple Silicon |
| **Python** | `shell/aeon_py/` (nanobind) | Cloud, HPC, Edge Orchestration |

---

## C# — Unity / Godot

### GC Safety Contract

> [!CAUTION]
> The .NET GC can relocate managed objects at any time. All arrays passed to native code
> are pinned via `fixed` statements. **Never** pass unpinned managed arrays to the C-API.

```csharp
// Unity MonoBehaviour
using Aeon;

public class NPCBrain : MonoBehaviour
{
    private AeonAtlas _atlas;

    void OnEnable()
    {
        _atlas = new AeonAtlas("Assets/Memory/npc.bin");
    }

    void Update()
    {
        float[] query = GetEmbedding(); // 768-dim from your model
        var results = _atlas.Navigate(query);

        foreach (var r in results)
            Debug.Log($"Memory {r.Id}: similarity={r.Similarity:F4}");
    }

    void OnDisable() => _atlas?.Dispose();
}
```

### Unity Plugin Setup

1. Build `libaeon.dylib` / `aeon.dll` / `libaeon.so` from CMake.
2. Copy to `Assets/Plugins/<platform>/`.
3. Add `AeonMemory.cs` to your Unity project.

---

## C++ — Unreal Engine

### Exception-Free Design

> [!IMPORTANT]
> Unreal Engine compiles with `-fno-exceptions`. The `FAeonClient` wrapper
> never throws — all functions return `aeon_error_t` error codes.

```cpp
// In your AActor or UObject
#include "AeonClient.hpp"

FAeonClient Memory;

void AMyNPC::BeginPlay()
{
    if (!Memory.Open(TEXT("Content/Memory/npc_brain.bin")))
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to open Atlas"));
        return;
    }

    TArray<float> Query; // 768-dim
    Query.SetNum(768);
    // ... fill from your embedding model ...

    TArray<FAeonResult> Results;
    Memory.Navigate(Query, Results);

    for (const auto& R : Results)
        UE_LOG(LogTemp, Log, TEXT("Node %llu sim=%.4f"), R.Id, R.Similarity);
}
```

### Unreal Plugin Setup

1. Build `libaeon` as a shared library from CMake.
2. Add to your Unreal plugin's `ThirdParty/` folder.
3. Update your `.Build.cs` to link `aeon` and add the include path.

---

## Node.js — OpenClaw Agent Framework (macOS Apple Silicon)

### Zero-Copy V8 Blood-Brain Barrier

> [!CAUTION]
> All vector data MUST be passed as `Float32Array`. The bridge extracts the raw `float*`
> pointer directly — no copies, no V8 heap allocations. Do NOT modify the array during
> a synchronous call.

```javascript
const { AeonDB } = require('@aeon/node-mac');

const db = new AeonDB('/data/atlas.bin', '/data/trace.wal', 768, 1); // INT8

// Zero-copy insert
const vec = new Float32Array(768);
crypto.getRandomValues(new Uint8Array(vec.buffer)); // fill with data
const nodeId = db.atlasInsert(0n, vec, 'concept:memory');

// Sub-10µs synchronous navigate
const results = db.atlasNavigate(vec, 10);
for (const r of results) {
    console.log(`Node ${r.id}: similarity=${r.score.toFixed(4)}`);
}

// WAL-backed trace append
const eventId = db.traceAppend(0, 'User asked about quantum computing');

db.close(); // MUST call — prevents file handle leaks
```

### Node.js Bridge Build

```bash
cd bindings/node
npm install
npm run build    # cmake-js compile -T aeon_node -a arm64
npm run bench    # Run latency benchmark
```

> [!IMPORTANT]
> Requires `libaeon.dylib` to be pre-built. From the project root:
> `cmake --build build --target aeon_shared`

---

## JNI — Android (Planned)

### Architecture

```
┌───────────────────┐     JNI      ┌──────────────────┐
│   Kotlin/Java     │ ──────────→  │  libaeon.so      │
│   AeonMemory.kt   │              │  (C-API)         │
│                   │ ←──────────  │                  │
│   ByteBuffer      │   direct     │  aeon_c_api.h    │
└───────────────────┘              └──────────────────┘
```

### Key Implementation Notes

1. **Use `DirectByteBuffer`** for the result arrays — these are NOT managed by the JVM GC
   and provide a stable native pointer for JNI.
2. **Load the library** with `System.loadLibrary("aeon")`.
3. **Android NDK**: Build `libaeon.so` with the NDK cross-compiler targeting
   `armeabi-v7a`, `arm64-v8a`, and `x86_64` ABIs.
4. **Result marshalling**: Map `aeon_result_node_t` to a Kotlin data class using
   `ByteBuffer.getLong()`, `.getFloat()`, etc.

### Skeleton

```kotlin
// AeonMemory.kt
object AeonMemory {
    init { System.loadLibrary("aeon") }

    external fun nativeCreate(path: String): Long       // Returns atlas ptr
    external fun nativeDestroy(atlas: Long)
    external fun nativeNavigate(
        atlas: Long,
        query: FloatArray,
        beamWidth: Int,
        results: ByteBuffer,    // DirectByteBuffer — GC-immune
        maxResults: Int
    ): Int                       // Returns actual count
}
```

---

## Swift — iOS (Planned)

### Architecture

```
┌───────────────────┐    C interop  ┌──────────────────┐
│   Swift           │ ──────────→   │  libaeon.dylib   │
│   AeonMemory.swift│               │  (C-API)         │
│                   │ ←──────────   │                  │
│   UnsafePointer   │   zero-copy   │  aeon_c_api.h    │
└───────────────────┘               └──────────────────┘
```

### Key Implementation Notes

1. **Swift C Interop**: Import the C header via a bridging header or a
   Swift Package `.modulemap`.
2. **Memory safety**: Use `UnsafeMutablePointer<aeon_result_node_t>` with
   `allocate(capacity:)` to create the caller-allocated result buffer.
3. **iOS Low Memory**: Hook into `UIApplication.didReceiveMemoryWarning`
   to call `aeon_atlas_release_pages()` and `aeon_atlas_compact()`.
4. **Background Dreaming**: Use `BGProcessingTask` (iOS 13+) to schedule
   `consolidate_subgraph` + `compact_mmap` during overnight charging.

### Skeleton

```swift
// AeonMemory.swift
import Foundation

class AeonAtlas {
    private var handle: OpaquePointer?

    init(path: String) throws {
        var ptr: OpaquePointer?
        let err = aeon_atlas_create(path, &ptr)
        guard err == AEON_OK else {
            throw AeonError(rawValue: err.rawValue)!
        }
        self.handle = ptr
    }

    func navigate(query: [Float]) -> [AeonResultNode] {
        let buffer = UnsafeMutablePointer<aeon_result_node_t>
            .allocate(capacity: Int(AEON_TOP_K_LIMIT))
        defer { buffer.deallocate() }

        var count: Int = 0
        query.withUnsafeBufferPointer { queryPtr in
            aeon_atlas_navigate(
                handle, queryPtr.baseAddress, 768,
                1, 0, buffer, 50, &count)
        }
        return (0..<count).map { AeonResultNode(buffer[$0]) }
    }

    deinit { aeon_atlas_destroy(handle) }
}
```

---

## Build Instructions

### Shared Library (all platforms)

```bash
cd core
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target aeon_shared

# Output:
#   macOS:   lib/libaeon.dylib
#   Linux:   lib/libaeon.so
#   Windows: bin/aeon.dll + lib/aeon.lib
```

### Cross-Compilation (Android NDK)

```bash
cmake -B build-android \
    -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-26 \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build-android --target aeon_shared
```

### Cross-Compilation (iOS)

```bash
cmake -B build-ios \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build-ios --target aeon_shared
```
