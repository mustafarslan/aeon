// =============================================================================
// AeonMemory.cs — Unity / Godot C# Bindings for the Aeon Memory OS
//
// PLATFORM TARGETS:
//   - Unity 2022+ (Mono / IL2CPP / .NET 8 Backend)
//   - Godot 4.x (C# via .NET 6/8)
//   - Any .NET project on Windows/macOS/Linux/iOS/Android
//
// CRITICAL GC SAFETY RULES:
//   The .NET Garbage Collector can RELOCATE managed heap objects at ANY time.
//   If we pass a managed array pointer to a native C function and the GC
//   moves it mid-call, the C++ kernel will read/write garbage memory.
//   This causes VIOLENT CRASHES in Unity (SIGSEGV, hung editor, data loss).
//
//   MANDATORY: All buffers passed to native code MUST be either:
//     1. stackalloc'd  (stack memory — GC cannot relocate)
//     2. Pinned via `fixed` or GCHandle (GC pin prevents relocation)
//
//   We NEVER pass unpinned managed arrays across the FFI boundary.
//
// USAGE IN UNITY:
//   // Inspector-serialized path to the .bin Atlas file
//   [SerializeField] private string atlasPath = "Assets/Memory/atlas.bin";
//
//   void Start() {
//       AeonMemory.Create(atlasPath, out var atlas);
//       // Use atlas for NPC memory...
//       AeonMemory.Destroy(atlas);
//   }
//
// Copyright 2024-2026 Aeon Project. MIT License.
// =============================================================================

using System;
using System.Runtime.InteropServices;

namespace Aeon
{
    // =========================================================================
    // Error Codes — Must match aeon_c_api.h exactly
    // =========================================================================
    public enum AeonError : int
    {
        Ok              =  0,
        ErrNullPtr      = -1,
        ErrInvalidArg   = -2,
        ErrFileIO       = -3,
        ErrInvalidFmt   = -4,
        ErrOutOfMemory  = -5,
        ErrNodeNotFound = -6,
        ErrAlreadyDead  = -7,
        ErrUnknown      = -99,
    }

    // =========================================================================
    // Result Node — Must match aeon_result_node_t in aeon_c_api.h
    // Sequential layout prevents the CLR from reordering fields.
    // =========================================================================
    [StructLayout(LayoutKind.Sequential)]
    public struct AeonResultNode
    {
        public ulong Id;
        public float Similarity;
        public float PreviewX;
        public float PreviewY;
        public float PreviewZ;
        public int   RequiresCloudFetch;
    }

    // =========================================================================
    // AeonMemory — Static P/Invoke Wrapper
    // =========================================================================
    public static class AeonMemory
    {
        // Library name: resolved to libaeon.dylib (macOS), libaeon.so (Linux),
        // aeon.dll (Windows) by the runtime. Unity copies the native plugin
        // from Assets/Plugins/<platform>/.
        private const string LibName = "aeon";

        // --- Raw P/Invoke Declarations (private) ---

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern AeonError aeon_atlas_create(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string path,
            out IntPtr outAtlas);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern AeonError aeon_atlas_destroy(IntPtr atlas);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern unsafe AeonError aeon_atlas_navigate(
            IntPtr atlas,
            float* queryVector,
            UIntPtr queryDim,
            uint beamWidth,
            int applyCsls,
            AeonResultNode* results,
            UIntPtr maxResults,
            UIntPtr* outActualCount);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern unsafe AeonError aeon_atlas_insert(
            IntPtr atlas,
            ulong parentId,
            float* vector,
            UIntPtr vectorDim,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string metadata,
            ulong* outId);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern AeonError aeon_atlas_size(
            IntPtr atlas,
            out UIntPtr outSize);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern AeonError aeon_atlas_tombstone_count(
            IntPtr atlas,
            out UIntPtr outCount);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern unsafe AeonError aeon_atlas_consolidate_subgraph(
            IntPtr atlas,
            ulong* oldNodeIds,
            UIntPtr oldNodeCount,
            float* summaryVector,
            UIntPtr summaryDim,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string summaryMetadata,
            ulong* outSummaryId);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern AeonError aeon_atlas_compact(
            IntPtr atlas,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string newPath);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr aeon_version();

        // =====================================================================
        // PUBLIC API — Safe Wrappers with GC Pinning
        // =====================================================================

        /// <summary>SDK version string.</summary>
        public static string Version()
        {
            IntPtr ptr = aeon_version();
            return Marshal.PtrToStringAnsi(ptr) ?? "unknown";
        }

        /// <summary>Create an Atlas instance.</summary>
        public static AeonError Create(string path, out IntPtr atlas)
        {
            return aeon_atlas_create(path, out atlas);
        }

        /// <summary>Destroy an Atlas instance (idempotent, NULL-safe).</summary>
        public static AeonError Destroy(IntPtr atlas)
        {
            return aeon_atlas_destroy(atlas);
        }

        /// <summary>Total node count (including tombstoned).</summary>
        public static AeonError Size(IntPtr atlas, out int size)
        {
            var err = aeon_atlas_size(atlas, out UIntPtr s);
            size = (int)s;
            return err;
        }

        /// <summary>Count of tombstoned (dead) nodes.</summary>
        public static AeonError TombstoneCount(IntPtr atlas, out int count)
        {
            var err = aeon_atlas_tombstone_count(atlas, out UIntPtr c);
            count = (int)c;
            return err;
        }

        /// <summary>
        /// Navigate the Atlas using a 768-dim query embedding.
        ///
        /// GC SAFETY: Uses `fixed` to pin the query array and stackalloc
        /// for the result buffer. The GC CANNOT move these during the call.
        /// </summary>
        /// <param name="atlas">Opaque atlas handle from Create().</param>
        /// <param name="query">768-element float array (must be normalized).</param>
        /// <param name="beamWidth">Beam width (1 = greedy, max 16).</param>
        /// <param name="applyCsls">Apply CSLS hubness correction.</param>
        /// <param name="results">Output array (caller-allocated or stackalloc).</param>
        /// <param name="actualCount">Number of results written.</param>
        /// <returns>AEON_OK on success.</returns>
        public static unsafe AeonError Navigate(
            IntPtr atlas,
            float[] query,
            uint beamWidth,
            bool applyCsls,
            out AeonResultNode[] results,
            out int actualCount)
        {
            const int MaxResults = 50; // AEON_TOP_K_LIMIT
            results = new AeonResultNode[MaxResults];
            actualCount = 0;

            if (query == null || query.Length != 768)
                return AeonError.ErrInvalidArg;

            // PIN the query array so GC cannot relocate it during the C call.
            // PIN the results array so C++ writes to a stable address.
            fixed (float* queryPtr = query)
            fixed (AeonResultNode* resultPtr = results)
            {
                UIntPtr outCount;
                var err = aeon_atlas_navigate(
                    atlas,
                    queryPtr,
                    (UIntPtr)768,
                    beamWidth,
                    applyCsls ? 1 : 0,
                    resultPtr,
                    (UIntPtr)MaxResults,
                    &outCount);

                actualCount = (int)outCount;

                // Trim results array to actual count
                if (actualCount < MaxResults)
                    Array.Resize(ref results, actualCount);

                return err;
            }
        }

        /// <summary>
        /// Insert a new node into the Atlas.
        ///
        /// GC SAFETY: The vector array is pinned via `fixed`.
        /// </summary>
        public static unsafe AeonError Insert(
            IntPtr atlas,
            ulong parentId,
            float[] vector,
            string metadata,
            out ulong nodeId)
        {
            nodeId = 0;

            if (vector == null || vector.Length != 768)
                return AeonError.ErrInvalidArg;

            fixed (float* vecPtr = vector)
            {
                ulong id;
                var err = aeon_atlas_insert(
                    atlas, parentId, vecPtr, (UIntPtr)768, metadata, &id);
                nodeId = id;
                return err;
            }
        }

        /// <summary>
        /// Consolidate a subgraph into a single summary node (Dreaming).
        ///
        /// GC SAFETY: Both nodeIds and summaryVector are pinned.
        /// </summary>
        public static unsafe AeonError ConsolidateSubgraph(
            IntPtr atlas,
            ulong[] oldNodeIds,
            float[] summaryVector,
            string summaryMetadata,
            out ulong summaryId)
        {
            summaryId = 0;

            if (oldNodeIds == null || oldNodeIds.Length == 0)
                return AeonError.ErrInvalidArg;
            if (summaryVector == null || summaryVector.Length != 768)
                return AeonError.ErrInvalidArg;

            fixed (ulong* idsPtr = oldNodeIds)
            fixed (float* vecPtr = summaryVector)
            {
                ulong id;
                var err = aeon_atlas_consolidate_subgraph(
                    atlas,
                    idsPtr,
                    (UIntPtr)oldNodeIds.Length,
                    vecPtr,
                    (UIntPtr)768,
                    summaryMetadata,
                    &id);
                summaryId = id;
                return err;
            }
        }

        /// <summary>
        /// Compact (defragment) the Atlas storage file.
        /// Call during idle windows (loading screens, sleep, etc.).
        /// </summary>
        public static AeonError Compact(IntPtr atlas, string newPath = null)
        {
            return aeon_atlas_compact(atlas, newPath ?? "");
        }
    }

    // =========================================================================
    // AeonAtlas — RAII Disposable Wrapper (Recommended for Unity MonoBehaviour)
    // =========================================================================

    /// <summary>
    /// Managed lifecycle wrapper for the native Atlas handle.
    /// Implements IDisposable for deterministic cleanup.
    ///
    /// Usage in Unity:
    ///   using var atlas = new AeonAtlas("Assets/Memory/npc_brain.bin");
    ///   var results = atlas.Navigate(embedding);
    /// </summary>
    public sealed class AeonAtlas : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed;

        public AeonAtlas(string path)
        {
            var err = AeonMemory.Create(path, out _handle);
            if (err != AeonError.Ok)
                throw new InvalidOperationException(
                    $"Failed to create Atlas: {err}");
        }

        public int Size
        {
            get
            {
                AeonMemory.Size(_handle, out int s);
                return s;
            }
        }

        public int TombstoneCount
        {
            get
            {
                AeonMemory.TombstoneCount(_handle, out int c);
                return c;
            }
        }

        public AeonResultNode[] Navigate(
            float[] query,
            uint beamWidth = 1,
            bool applyCsls = false)
        {
            ThrowIfDisposed();
            var err = AeonMemory.Navigate(
                _handle, query, beamWidth, applyCsls,
                out var results, out _);
            if (err != AeonError.Ok)
                throw new InvalidOperationException($"Navigate failed: {err}");
            return results;
        }

        public ulong Insert(ulong parentId, float[] vector, string metadata)
        {
            ThrowIfDisposed();
            var err = AeonMemory.Insert(
                _handle, parentId, vector, metadata, out ulong id);
            if (err != AeonError.Ok)
                throw new InvalidOperationException($"Insert failed: {err}");
            return id;
        }

        public ulong ConsolidateSubgraph(
            ulong[] oldNodeIds,
            float[] summaryVector,
            string summaryMetadata)
        {
            ThrowIfDisposed();
            var err = AeonMemory.ConsolidateSubgraph(
                _handle, oldNodeIds, summaryVector, summaryMetadata,
                out ulong id);
            if (err != AeonError.Ok)
                throw new InvalidOperationException(
                    $"Consolidation failed: {err}");
            return id;
        }

        public void Compact(string newPath = null)
        {
            ThrowIfDisposed();
            var err = AeonMemory.Compact(_handle, newPath);
            if (err != AeonError.Ok)
                throw new InvalidOperationException($"Compact failed: {err}");
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(AeonAtlas));
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                if (_handle != IntPtr.Zero)
                {
                    AeonMemory.Destroy(_handle);
                    _handle = IntPtr.Zero;
                }
                _disposed = true;
            }
        }

        ~AeonAtlas() => Dispose();
    }
}
