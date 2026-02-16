/**
 * @file aeon_c_api.h
 * @brief Universal C-API for the Aeon Memory OS Kernel.
 *
 * PLATFORM TARGETS:
 *   - Game Engines:  Unreal Engine (C++), Godot (C++/C#/GDScript), Unity (C#)
 *   - Operating Systems:  Windows, macOS, Linux
 *   - Mobile/Edge:  iOS, Android, Robotics hardware, IoT, Factory Edge IPCs
 *   - Infrastructure:  Cloud Data Centers, Enterprise LLMs, HPC clusters
 *
 * DESIGN INVARIANTS:
 *   1. All functions are `extern "C"` for flat ABI compatibility.
 *   2. All functions return `aeon_error_t` (integer enum) for FFI safety.
 *   3. C++ exceptions NEVER cross the FFI boundary (UB prevention).
 *   4. Caller-allocated buffers: the caller passes pre-allocated arrays +
 *      max_count + out_actual_count. NEVER malloc inside and return across FFI.
 *   5. Opaque pointer pattern: `aeon_atlas_t` hides all C++ internals.
 *   6. AEON_API macro handles DLL export on Windows and visibility on POSIX.
 *
 * @copyright 2024-2026 Aeon Project. All rights reserved.
 */

#ifndef AEON_C_API_H
#define AEON_C_API_H

#include <stddef.h>
#include <stdint.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * DLL EXPORT MACRO — The Windows/Mobile DLL Export Trap
 * ═══════════════════════════════════════════════════════════════════════════
 * On Windows, shared library symbols are hidden by default. We must
 * explicitly export them with __declspec(dllexport) when building and
 * __declspec(dllimport) when consuming. On POSIX (Linux/macOS/iOS/Android),
 * we use __attribute__((visibility("default"))) with -fvisibility=hidden.
 */
#if defined(_WIN32) || defined(_WIN64)
#ifdef AEON_BUILDING_SHARED
#define AEON_API __declspec(dllexport)
#else
#define AEON_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define AEON_API __attribute__((visibility("default")))
#else
#define AEON_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * ERROR CODES — aeon_error_t
 *
 * Every extern "C" function returns one of these. The C++ implementation
 * wraps all logic in try/catch(...) to prevent UB from exceptions crossing
 * the FFI boundary. Game engine integrators (Unreal/Godot/Unity) should
 * check these return values for robust error handling.
 * ═══════════════════════════════════════════════════════════════════════════
 */
typedef enum {
  AEON_OK = 0,                  /**< Success */
  AEON_ERR_NULL_PTR = -1,       /**< A required pointer argument was NULL */
  AEON_ERR_INVALID_ARG = -2,    /**< Invalid argument (wrong dimension, etc.) */
  AEON_ERR_FILE_IO = -3,        /**< File I/O error (open, mmap, resize) */
  AEON_ERR_INVALID_FORMAT = -4, /**< Atlas file has invalid magic/version */
  AEON_ERR_OUT_OF_MEMORY = -5,  /**< Memory allocation or mmap resize failed */
  AEON_ERR_NODE_NOT_FOUND = -6, /**< Referenced node ID does not exist */
  AEON_ERR_ALREADY_DEAD = -7,   /**< Node is already tombstoned */
  AEON_ERR_BUFFER_TOO_SMALL = -8, /**< Caller buffer too small for result */
  AEON_ERR_UNKNOWN = -99          /**< Unknown internal error (catch-all) */
} aeon_error_t;

/* ═══════════════════════════════════════════════════════════════════════════
 * OPAQUE POINTER — aeon_atlas_t
 *
 * The C-API consumer (Unreal Blueprint, Unity C# P/Invoke, Godot GDExtension,
 * Android JNI) sees only an opaque handle. All C++ internals (mmap, EBR,
 * SLB cache, delta buffer) are completely hidden behind this pointer.
 * ═══════════════════════════════════════════════════════════════════════════
 */
typedef struct aeon_atlas_s aeon_atlas_t;

/** Opaque handle to a Trace episodic memory store. */
typedef struct aeon_trace_s aeon_trace_t;

/**
 * @brief A single trace event (FFI-safe, matches TraceEvent layout).
 * 512 bytes, trivially copyable. Caller allocates buffer of these.
 *
 * V4.1: Full text stored in sidecar blob file. Use aeon_trace_get_event_text()
 * to retrieve. text_preview contains the first 63 chars for fast listing.
 */
typedef struct {
  uint64_t id;           /**< Unique monotonic event ID */
  uint64_t prev_id;      /**< Previous event in session (0 = root) */
  uint64_t atlas_id;     /**< Linked Atlas concept (0 = none) */
  uint64_t timestamp;    /**< Epoch microseconds */
  uint16_t role;         /**< 0=User, 1=System, 2=Concept, 3=Summary */
  uint16_t flags;        /**< Tombstone/archive flags */
  char session_id[36];   /**< Multi-tenant session UUID */
  uint64_t blob_offset;  /**< Offset into sidecar blob file */
  uint32_t blob_size;    /**< Byte length of full text in blob */
  char text_preview[64]; /**< Null-terminated 63-char inline prefix */
  uint8_t reserved[364]; /**< Padding to 512 bytes */
} aeon_trace_event_t;

/* ═══════════════════════════════════════════════════════════════════════════
 * RESULT STRUCTURES — Caller-Allocated Buffers
 *
 * STRICT RULE: The caller MUST pass a pre-allocated array of these structs,
 * plus max_results and an out_actual_count pointer. We NEVER allocate memory
 * inside C++ and return it across the FFI boundary.
 * ═══════════════════════════════════════════════════════════════════════════
 */

/**
 * @brief A single navigation result node (FFI-safe, trivially copyable).
 * Mirrors Atlas::ResultNode but is a flat C struct for cross-language use.
 */
typedef struct {
  uint64_t id;               /**< Node unique identifier */
  float similarity;          /**< Cosine similarity score (or CSLS-adjusted) */
  float centroid_preview[3]; /**< First 3 dims of the centroid vector */
  int requires_cloud_fetch;  /**< 1 if edge cold miss, 0 otherwise */
} aeon_result_node_t;

/* ═══════════════════════════════════════════════════════════════════════════
 * CONSTANTS — Exported for FFI consumers
 * ═══════════════════════════════════════════════════════════════════════════
 */

/**
 * Default embedding dimensionality (768 for all-MiniLM-L12-v2).
 * V4.0: This is now a DEFAULT for new Atlas files. The actual dim
 * is runtime-queryable via aeon_atlas_get_dim(). Existing code using
 * AEON_EMBEDDING_DIM as a buffer size remains correct for 768-dim models.
 */
#define AEON_EMBEDDING_DIM 768
#define AEON_EMBEDDING_DIM_DEFAULT 768

/** Maximum result set size from navigate. */
#define AEON_TOP_K_LIMIT 50

/* ═══════════════════════════════════════════════════════════════════════════
 * LIFECYCLE — Create / Destroy
 * ═══════════════════════════════════════════════════════════════════════════
 */

/**
 * @brief Opens or creates an Atlas memory-mapped file.
 *
 * V4.0: Accepts an embedding dimensionality parameter. For NEW files,
 * this configures the node layout. For EXISTING files, dim is read
 * from the on-disk header and this parameter is ignored.
 *
 * @param[in]  path     Null-terminated file path (UTF-8).
 * @param[in]  dim      Embedding dimensionality. 0 = use default (768).
 * @param[out] out_atlas Pointer to receive the opaque atlas handle.
 * @return AEON_OK on success, error code on failure.
 *
 * @note Caller MUST call aeon_atlas_destroy() when done.
 * @note Thread-safe: each Atlas instance has its own internal locks.
 */
AEON_API aeon_error_t aeon_atlas_create(const char *path, uint32_t dim,
                                        aeon_atlas_t **out_atlas);

/**
 * @brief Returns the embedding dimensionality of an Atlas instance.
 *
 * @param[in]  atlas   Atlas handle.
 * @param[out] out_dim Receives the dimensionality (e.g. 384, 768, 1536).
 * @return AEON_OK on success.
 */
AEON_API aeon_error_t aeon_atlas_get_dim(aeon_atlas_t *atlas,
                                         uint32_t *out_dim);

/**
 * @brief Destroys an Atlas instance and releases all resources.
 *
 * @param[in] atlas The atlas handle to destroy (may be NULL — no-op).
 * @return AEON_OK always.
 */
AEON_API aeon_error_t aeon_atlas_destroy(aeon_atlas_t *atlas);

/* ═══════════════════════════════════════════════════════════════════════════
 * QUERY — Navigate (Beam Search)
 * ═══════════════════════════════════════════════════════════════════════════
 */

/**
 * @brief Navigates the Atlas tree to find nearest nodes to the query vector.
 *
 * Uses SIMD-accelerated beam search. Tombstoned nodes are branchlessly
 * eliminated via the hub_penalty trick (score ≈ -1e9f).
 *
 * CALLER-ALLOCATED BUFFER CONTRACT:
 *   - `results` must point to a pre-allocated array of `max_results` elements.
 *   - `out_actual_count` receives the number of results actually written.
 *   - If max_results < actual results, output is truncated (no error).
 *
 * @param[in]  atlas            Atlas handle.
 * @param[in]  query_vector     Pointer to `dim` floats (use
 * aeon_atlas_get_dim).
 * @param[in]  query_dim        Must equal Atlas dim.
 * @param[in]  beam_width       Beam width (1 = greedy, max 16).
 * @param[in]  apply_csls       1 to apply CSLS hubness correction, 0 otherwise.
 * @param[out] results          Caller-allocated result buffer.
 * @param[in]  max_results      Capacity of the results buffer.
 * @param[out] out_actual_count Actual number of results written.
 * @return AEON_OK on success.
 */
/**
 * @param[in]  session_id       Session UUID for L1 SLB routing.
 *                              NULL or "" routes to global L2 cache.
 */
AEON_API aeon_error_t aeon_atlas_navigate(
    aeon_atlas_t *atlas, const float *query_vector, size_t query_dim,
    uint32_t beam_width, int apply_csls, const char *session_id,
    aeon_result_node_t *results, size_t max_results, size_t *out_actual_count);

/* ═══════════════════════════════════════════════════════════════════════════
 * MUTATION — Insert Node
 * ═══════════════════════════════════════════════════════════════════════════
 */

/**
 * @brief Inserts a new node into the Atlas as a child of parent_id.
 *
 * @param[in]  atlas      Atlas handle.
 * @param[in]  parent_id  Parent node ID (0 for root if Atlas is empty).
 * @param[in]  vector     AEON_EMBEDDING_DIM floats.
 * @param[in]  vector_dim Must equal AEON_EMBEDDING_DIM.
 * @param[in]  metadata   Null-terminated metadata string (max 255 chars).
 * @param[in]  session_id  Session UUID for SLB cache routing. NULL = global.
 * @param[out] out_id     Receives the ID of the newly inserted node.
 * @return AEON_OK on success.
 */
AEON_API aeon_error_t aeon_atlas_insert(aeon_atlas_t *atlas, uint64_t parent_id,
                                        const float *vector, size_t vector_dim,
                                        const char *metadata,
                                        const char *session_id,
                                        uint64_t *out_id);

/**
 * @brief Drop a session's L1 SLB cache. MANDATORY for NPC despawn.
 *
 * When an NPC dies or a user session ends, call this to free the
 * per-session L1 cache memory. Without this, 100K+ sessions will OOM.
 *
 * @param[in] atlas      Atlas handle.
 * @param[in] session_id Session UUID to drop.
 * @return AEON_OK on success, AEON_ERR_NODE_NOT_FOUND if session didn't exist.
 */
AEON_API aeon_error_t aeon_atlas_drop_session(aeon_atlas_t *atlas,
                                              const char *session_id);

/* ═══════════════════════════════════════════════════════════════════════════
 * INSPECTION — Size, Children, Tombstones
 * ═══════════════════════════════════════════════════════════════════════════
 */

/**
 * @brief Returns the total number of nodes (including tombstoned).
 */
AEON_API aeon_error_t aeon_atlas_size(aeon_atlas_t *atlas, size_t *out_size);

/**
 * @brief Returns the number of tombstoned (dead) nodes.
 * Useful for deciding when to trigger compact.
 */
AEON_API aeon_error_t aeon_atlas_tombstone_count(aeon_atlas_t *atlas,
                                                 size_t *out_count);

/**
 * @brief Retrieves direct children of a node.
 * Caller-allocated buffer contract (same as navigate).
 */
AEON_API aeon_error_t aeon_atlas_get_children(aeon_atlas_t *atlas,
                                              uint64_t parent_id,
                                              aeon_result_node_t *results,
                                              size_t max_results,
                                              size_t *out_actual_count);

/* ═══════════════════════════════════════════════════════════════════════════
 * DREAMING — Memory Consolidation (Edge/Mobile)
 * ═══════════════════════════════════════════════════════════════════════════
 */

/**
 * @brief Consolidates a subgraph into a single summary node.
 *
 * Atomically: inserts summary → re-wires children → tombstones old nodes.
 * The tombstoned nodes are branchlessly eliminated from future beam searches.
 *
 * @param[in]  atlas            Atlas handle.
 * @param[in]  old_node_ids     Array of node IDs to consolidate.
 * @param[in]  old_node_count   Number of IDs in old_node_ids.
 * @param[in]  summary_vector   AEON_EMBEDDING_DIM floats (the LLM summary).
 * @param[in]  summary_dim      Must equal AEON_EMBEDDING_DIM.
 * @param[in]  summary_metadata Null-terminated summary text (max 255 chars).
 * @param[out] out_summary_id   Receives the ID of the new summary node.
 * @return AEON_OK on success.
 */
AEON_API aeon_error_t aeon_atlas_consolidate_subgraph(
    aeon_atlas_t *atlas, const uint64_t *old_node_ids, size_t old_node_count,
    const float *summary_vector, size_t summary_dim,
    const char *summary_metadata, uint64_t *out_summary_id);

/**
 * @brief Background shadow compaction (V4.0 — stutter-free).
 *
 * Uses Redis BGSAVE double-buffer pattern: freeze delta buffer, copy
 * live nodes to a new generation file, hot-swap with µs-level lock.
 * Game engines can continue inserting during compaction.
 *
 * @param[in] atlas Atlas handle.
 * @return AEON_OK on success.
 *
 * @note V4.0: No longer requires exclusive access. Safe to call during
 *       active reads/writes. Internally manages generational file naming.
 */
AEON_API aeon_error_t aeon_atlas_compact(aeon_atlas_t *atlas);

/* ═══════════════════════════════════════════════════════════════════════════
 * EDGE-CLOUD — Page Release (iOS/Android Low Memory)
 * ═══════════════════════════════════════════════════════════════════════════
 */

/**
 * @brief Releases resident pages back to the OS without unmapping.
 *
 * Call when the OS signals low memory (iOS LMK, Android
 * onTrimMemory, Linux cgroups pressure). Pages will be demand-paged
 * back on next access with a minor fault.
 *
 * @param[in] atlas       Atlas handle.
 * @param[in] start_node  First node index to release.
 * @param[in] count       Number of nodes to release.
 * @return AEON_OK on success.
 */
AEON_API aeon_error_t aeon_atlas_release_pages(aeon_atlas_t *atlas,
                                               size_t start_node, size_t count);

/* ═══════════════════════════════════════════════════════════════════════════
 * TRACE DAG — Episodic Memory (mmap-backed)
 * ═══════════════════════════════════════════════════════════════════════════
 */

/**
 * @brief Creates or opens a Trace episodic memory store.
 *
 * @param[in]  path      File path for trace storage (e.g. "memory/trace.bin").
 * @param[out] out_trace Receives the opaque trace handle.
 * @return AEON_OK on success.
 */
AEON_API aeon_error_t aeon_trace_create(const char *path,
                                        aeon_trace_t **out_trace);

/**
 * @brief Destroys a Trace instance and releases all resources.
 * @param[in] trace  Trace handle (may be NULL — no-op).
 */
AEON_API aeon_error_t aeon_trace_destroy(aeon_trace_t *trace);

/**
 * @brief Append an episodic event to the trace.
 *
 * V4.1: text is stored in sidecar blob file (unlimited length).
 * The first 63 characters are kept as an inline preview.
 *
 * @param[in]  trace      Trace handle.
 * @param[in]  session_id Multi-tenant session UUID (max 35 chars).
 * @param[in]  role       0=User, 1=System, 2=Concept, 3=Summary.
 * @param[in]  text       Full text (unlimited length).
 * @param[in]  atlas_id   Linked Atlas concept node ID (0 = none).
 * @param[out] out_id     Receives the new event's unique ID.
 * @return AEON_OK on success.
 */
AEON_API aeon_error_t aeon_trace_append_event(aeon_trace_t *trace,
                                              const char *session_id,
                                              uint16_t role, const char *text,
                                              uint64_t atlas_id,
                                              uint64_t *out_id);

/**
 * @brief Retrieve session history (newest first).
 *
 * Traverses the prev_id chain backwards from the session's tail event.
 * Caller allocates the buffer.
 *
 * @param[in]  trace       Trace handle.
 * @param[in]  session_id  Session UUID.
 * @param[out] out_events  Caller-allocated array of aeon_trace_event_t.
 * @param[in]  max_events  Capacity of out_events buffer.
 * @param[out] out_count   Actual number of events written.
 * @return AEON_OK on success.
 */
AEON_API aeon_error_t aeon_trace_get_history(aeon_trace_t *trace,
                                             const char *session_id,
                                             aeon_trace_event_t *out_events,
                                             size_t max_events,
                                             size_t *out_count);

/**
 * @brief Returns total event count (mmap + delta buffer).
 */
AEON_API aeon_error_t aeon_trace_size(aeon_trace_t *trace, size_t *out_size);

/**
 * @brief Trigger shadow compaction on the trace file.
 *
 * V4.1: Also GC's the sidecar blob file — only live event blobs
 * are copied to the new generation blob file.
 */
AEON_API aeon_error_t aeon_trace_compact(aeon_trace_t *trace);

/**
 * @brief Retrieve the full text for a trace event from the blob arena.
 *
 * Zero-heap: caller provides the output buffer. Returns
 * AEON_ERR_BUFFER_TOO_SMALL if buf_capacity is insufficient (out_len
 * still receives the required size).
 *
 * @param[in]  trace        Trace handle.
 * @param[in]  blob_offset  Byte offset from aeon_trace_event_t::blob_offset.
 * @param[in]  blob_size    Byte length from aeon_trace_event_t::blob_size.
 * @param[out] out_buf      Caller-allocated buffer for the text.
 * @param[in]  buf_capacity Capacity of out_buf in bytes.
 * @param[out] out_len      Receives the actual text length (excluding null).
 * @return AEON_OK on success, AEON_ERR_BUFFER_TOO_SMALL if buffer too small.
 */
AEON_API aeon_error_t aeon_trace_get_event_text(
    aeon_trace_t *trace, uint64_t blob_offset, uint32_t blob_size,
    char *out_buf, size_t buf_capacity, size_t *out_len);

/* ═══════════════════════════════════════════════════════════════════════════
 * VERSION — Runtime Introspection
 * ═══════════════════════════════════════════════════════════════════════════
 */

/**
 * @brief Returns the Aeon SDK version string.
 * @return Null-terminated static string (never freed by caller).
 */
AEON_API const char *aeon_version(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* AEON_C_API_H */
