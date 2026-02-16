/**
 * @file aeon_c_api.cpp
 * @brief Universal C-API Implementation — Exception-Safe FFI Boundary.
 *
 * FATAL CRASH PREVENTION:
 *   C++ exceptions crossing an FFI boundary cause Undefined Behavior.
 *   Every extern "C" function is wrapped in:
 *     try { ... } catch (const std::exception& e) { return AEON_ERR_...; }
 *                  catch (...) { return AEON_ERR_UNKNOWN; }
 *
 * This file is compiled into the shared library (libaeon.so / aeon.dll /
 * libaeon.dylib).
 */

#include "aeon/aeon_c_api.h"
#include "aeon/atlas.hpp"
#include "aeon/trace.hpp"

#include <cstring>
#include <exception>
#include <filesystem>
#include <new>
#include <stdexcept>
#include <string>

// ===========================================================================
// Internal: cast opaque pointers to C++ objects
// ===========================================================================

static aeon::Atlas *to_atlas(aeon_atlas_t *handle) {
  return reinterpret_cast<aeon::Atlas *>(handle);
}

static const aeon::Atlas *to_atlas(const aeon_atlas_t *handle) {
  return reinterpret_cast<const aeon::Atlas *>(handle);
}

static aeon::TraceManager *to_trace(aeon_trace_t *handle) {
  return reinterpret_cast<aeon::TraceManager *>(handle);
}

static const aeon::TraceManager *to_trace(const aeon_trace_t *handle) {
  return reinterpret_cast<const aeon::TraceManager *>(handle);
}

// ===========================================================================
// Version
// ===========================================================================

extern "C" {

AEON_API const char *aeon_version(void) { return "1.0.0-dreaming"; }

// ===========================================================================
// Lifecycle
// ===========================================================================

AEON_API aeon_error_t aeon_atlas_create(const char *path, uint32_t dim,
                                        aeon_atlas_t **out_atlas) {
  if (!path || !out_atlas)
    return AEON_ERR_NULL_PTR;

  try {
    auto *atlas = new aeon::Atlas(std::filesystem::path(path), dim);
    *out_atlas = reinterpret_cast<aeon_atlas_t *>(atlas);
    return AEON_OK;
  } catch (const std::bad_alloc &) {
    return AEON_ERR_OUT_OF_MEMORY;
  } catch (const std::runtime_error &) {
    return AEON_ERR_FILE_IO;
  } catch (...) {
    return AEON_ERR_UNKNOWN;
  }
}

AEON_API aeon_error_t aeon_atlas_create_ex(const char *path,
                                           const aeon_atlas_options_t *opts,
                                           aeon_atlas_t **out_atlas) {
  if (!path || !opts || !out_atlas)
    return AEON_ERR_NULL_PTR;

  try {
    aeon::AtlasOptions cpp_opts;
    cpp_opts.dim = opts->dim;
    cpp_opts.quantization_type = opts->quantization_type;
    cpp_opts.enable_wal = (opts->enable_wal != 0);

    auto *atlas = new aeon::Atlas(std::filesystem::path(path), cpp_opts);
    *out_atlas = reinterpret_cast<aeon_atlas_t *>(atlas);
    return AEON_OK;
  } catch (const std::bad_alloc &) {
    return AEON_ERR_OUT_OF_MEMORY;
  } catch (const std::runtime_error &) {
    return AEON_ERR_FILE_IO;
  } catch (...) {
    return AEON_ERR_UNKNOWN;
  }
}

AEON_API aeon_error_t aeon_atlas_destroy(aeon_atlas_t *atlas) {
  if (!atlas)
    return AEON_OK; // No-op for NULL — safe idempotent behavior

  try {
    delete to_atlas(atlas);
    return AEON_OK;
  } catch (...) {
    return AEON_ERR_UNKNOWN;
  }
}

// ===========================================================================
// Introspection — Dimension Query
// ===========================================================================

AEON_API aeon_error_t aeon_atlas_get_dim(aeon_atlas_t *atlas,
                                         uint32_t *out_dim) {
  if (!atlas || !out_dim)
    return AEON_ERR_NULL_PTR;

  try {
    *out_dim = to_atlas(atlas)->dim();
    return AEON_OK;
  } catch (...) {
    return AEON_ERR_UNKNOWN;
  }
}

// ===========================================================================
// Query — Navigate
// ===========================================================================

AEON_API aeon_error_t aeon_atlas_navigate(
    aeon_atlas_t *atlas, const float *query_vector, size_t query_dim,
    uint32_t beam_width, int apply_csls, const char *session_id,
    aeon_result_node_t *results, size_t max_results, size_t *out_actual_count) {
  if (!atlas || !query_vector || !results || !out_actual_count)
    return AEON_ERR_NULL_PTR;

  if (query_dim != to_atlas(atlas)->dim())
    return AEON_ERR_INVALID_ARG;

  try {
    auto *a = to_atlas(atlas);
    std::span<const float> query{query_vector, query_dim};

    // TODO: Route through HierarchicalSLB L1 when session_id is provided.
    // For now, session_id is accepted but SLB routing is pending integration.
    (void)session_id;

    auto path = a->navigate(query, beam_width, apply_csls != 0);

    size_t count = std::min(path.size(), max_results);
    for (size_t i = 0; i < count; ++i) {
      results[i].id = path[i].id;
      results[i].similarity = path[i].similarity;
      results[i].centroid_preview[0] = path[i].centroid_preview[0];
      results[i].centroid_preview[1] = path[i].centroid_preview[1];
      results[i].centroid_preview[2] = path[i].centroid_preview[2];
      results[i].requires_cloud_fetch = path[i].requires_cloud_fetch ? 1 : 0;
    }
    *out_actual_count = count;
    return AEON_OK;
  } catch (const std::invalid_argument &) {
    return AEON_ERR_INVALID_ARG;
  } catch (...) {
    return AEON_ERR_UNKNOWN;
  }
}

// ===========================================================================
// Mutation — Insert
// ===========================================================================

AEON_API aeon_error_t aeon_atlas_insert(aeon_atlas_t *atlas, uint64_t parent_id,
                                        const float *vector, size_t vector_dim,
                                        const char *metadata,
                                        const char *session_id,
                                        uint64_t *out_id) {
  if (!atlas || !vector || !out_id)
    return AEON_ERR_NULL_PTR;

  if (vector_dim != to_atlas(atlas)->dim())
    return AEON_ERR_INVALID_ARG;

  try {
    auto *a = to_atlas(atlas);
    std::span<const float> vec{vector, vector_dim};
    std::string_view meta = metadata ? metadata : "";

    uint64_t id = a->insert(parent_id, vec, meta);

    // TODO: Insert into HierarchicalSLB L1 cache for session_id.
    // For now, session_id is accepted but SLB routing is pending integration.
    (void)session_id;

    *out_id = id;
    return AEON_OK;
  } catch (const std::invalid_argument &) {
    return AEON_ERR_INVALID_ARG;
  } catch (const std::runtime_error &) {
    return AEON_ERR_OUT_OF_MEMORY;
  } catch (...) {
    return AEON_ERR_UNKNOWN;
  }
}

// ===========================================================================
// Session Management — Drop Session
// ===========================================================================

AEON_API aeon_error_t aeon_atlas_drop_session(aeon_atlas_t *atlas,
                                              const char *session_id) {
  if (!atlas || !session_id)
    return AEON_ERR_NULL_PTR;

  try {
    // TODO: Forward to Atlas's HierarchicalSLB::drop_session().
    // This requires Atlas to expose its SLB reference. For now, this
    // is a validated stub that will be wired once Atlas gains a
    // public drop_session() forwarder.
    (void)to_atlas(atlas);
    return AEON_OK;
  } catch (...) {
    return AEON_ERR_UNKNOWN;
  }
}

// ===========================================================================
// Inspection — Size, Children, Tombstones
// ===========================================================================

AEON_API aeon_error_t aeon_atlas_size(aeon_atlas_t *atlas, size_t *out_size) {
  if (!atlas || !out_size)
    return AEON_ERR_NULL_PTR;

  try {
    *out_size = to_atlas(atlas)->size();
    return AEON_OK;
  } catch (...) {
    return AEON_ERR_UNKNOWN;
  }
}

AEON_API aeon_error_t aeon_atlas_tombstone_count(aeon_atlas_t *atlas,
                                                 size_t *out_count) {
  if (!atlas || !out_count)
    return AEON_ERR_NULL_PTR;

  try {
    *out_count = to_atlas(atlas)->tombstone_count();
    return AEON_OK;
  } catch (...) {
    return AEON_ERR_UNKNOWN;
  }
}

AEON_API aeon_error_t aeon_atlas_get_children(aeon_atlas_t *atlas,
                                              uint64_t parent_id,
                                              aeon_result_node_t *results,
                                              size_t max_results,
                                              size_t *out_actual_count) {
  if (!atlas || !results || !out_actual_count)
    return AEON_ERR_NULL_PTR;

  try {
    auto *a = to_atlas(atlas);
    auto children = a->get_children(parent_id);

    size_t count = std::min(children.size(), max_results);
    for (size_t i = 0; i < count; ++i) {
      results[i].id = children[i].id;
      results[i].similarity = children[i].similarity;
      results[i].centroid_preview[0] = children[i].centroid_preview[0];
      results[i].centroid_preview[1] = children[i].centroid_preview[1];
      results[i].centroid_preview[2] = children[i].centroid_preview[2];
      results[i].requires_cloud_fetch =
          children[i].requires_cloud_fetch ? 1 : 0;
    }
    *out_actual_count = count;
    return AEON_OK;
  } catch (...) {
    return AEON_ERR_UNKNOWN;
  }
}

// ===========================================================================
// Dreaming — Consolidation & Compaction
// ===========================================================================

AEON_API aeon_error_t aeon_atlas_consolidate_subgraph(
    aeon_atlas_t *atlas, const uint64_t *old_node_ids, size_t old_node_count,
    const float *summary_vector, size_t summary_dim,
    const char *summary_metadata, uint64_t *out_summary_id) {
  if (!atlas || !old_node_ids || !summary_vector || !out_summary_id)
    return AEON_ERR_NULL_PTR;

  if (old_node_count == 0)
    return AEON_ERR_INVALID_ARG;

  if (summary_dim != to_atlas(atlas)->dim())
    return AEON_ERR_INVALID_ARG;

  try {
    auto *a = to_atlas(atlas);
    std::span<const uint64_t> ids{old_node_ids, old_node_count};
    std::span<const float> vec{summary_vector, summary_dim};
    std::string_view meta = summary_metadata ? summary_metadata : "";

    uint64_t id = a->consolidate_subgraph(ids, vec, meta);
    *out_summary_id = id;
    return AEON_OK;
  } catch (const std::invalid_argument &) {
    return AEON_ERR_INVALID_ARG;
  } catch (const std::runtime_error &e) {
    // Differentiate between "node not found" and "already tombstoned"
    std::string msg = e.what();
    if (msg.find("invalid node id") != std::string::npos ||
        msg.find("null node") != std::string::npos) {
      return AEON_ERR_NODE_NOT_FOUND;
    }
    if (msg.find("already tombstoned") != std::string::npos) {
      return AEON_ERR_ALREADY_DEAD;
    }
    return AEON_ERR_FILE_IO;
  } catch (...) {
    return AEON_ERR_UNKNOWN;
  }
}

AEON_API aeon_error_t aeon_atlas_compact(aeon_atlas_t *atlas) {
  if (!atlas)
    return AEON_ERR_NULL_PTR;

  try {
    to_atlas(atlas)->compact_mmap();
    return AEON_OK;
  } catch (const std::runtime_error &) {
    return AEON_ERR_FILE_IO;
  } catch (...) {
    return AEON_ERR_UNKNOWN;
  }
}

// ===========================================================================
// Edge-Cloud — Page Release
// ===========================================================================

AEON_API aeon_error_t aeon_atlas_release_pages(aeon_atlas_t *atlas,
                                               size_t start_node,
                                               size_t count) {
  if (!atlas)
    return AEON_ERR_NULL_PTR;

  try {
    // Access the internal storage file's release_pages through a helper.
    // Since release_pages is on MemoryFile (private member), we need to
    // expose it. For now, this is a no-op stub that will be connected
    // once we add a public Atlas::release_pages() forwarder.
    // The platform::advise_dontneed call is already implemented in storage.hpp.
    (void)start_node;
    (void)count;
    // TODO: Add Atlas::release_pages() public forwarder
    return AEON_OK;
  } catch (...) {
    return AEON_ERR_UNKNOWN;
  }
}

} // extern "C"

// ===========================================================================
// Trace DAG — Episodic Memory C-API
// ===========================================================================

extern "C" {

AEON_API aeon_error_t aeon_trace_create(const char *path,
                                        aeon_trace_t **out_trace) {
  if (!path || !out_trace)
    return AEON_ERR_NULL_PTR;

  try {
    auto *trace = new aeon::TraceManager(std::filesystem::path(path));
    *out_trace = reinterpret_cast<aeon_trace_t *>(trace);
    return AEON_OK;
  } catch (const std::bad_alloc &) {
    return AEON_ERR_OUT_OF_MEMORY;
  } catch (const std::runtime_error &) {
    return AEON_ERR_FILE_IO;
  } catch (...) {
    return AEON_ERR_UNKNOWN;
  }
}

AEON_API aeon_error_t aeon_trace_destroy(aeon_trace_t *trace) {
  if (!trace)
    return AEON_OK;

  try {
    delete to_trace(trace);
    return AEON_OK;
  } catch (...) {
    return AEON_ERR_UNKNOWN;
  }
}

AEON_API aeon_error_t aeon_trace_append_event(aeon_trace_t *trace,
                                              const char *session_id,
                                              uint16_t role, const char *text,
                                              uint64_t atlas_id,
                                              uint64_t *out_id) {
  if (!trace || !out_id)
    return AEON_ERR_NULL_PTR;

  try {
    uint64_t id =
        to_trace(trace)->append_event(session_id, role, text, atlas_id);
    *out_id = id;
    return AEON_OK;
  } catch (const std::invalid_argument &) {
    return AEON_ERR_INVALID_ARG;
  } catch (...) {
    return AEON_ERR_UNKNOWN;
  }
}

AEON_API aeon_error_t aeon_trace_get_history(aeon_trace_t *trace,
                                             const char *session_id,
                                             aeon_trace_event_t *out_events,
                                             size_t max_events,
                                             size_t *out_count) {
  if (!trace || !out_events || !out_count)
    return AEON_ERR_NULL_PTR;

  try {
    auto history = to_trace(trace)->get_history(session_id, max_events);

    size_t count = std::min(history.size(), max_events);
    for (size_t i = 0; i < count; ++i) {
      // Flat memcpy — TraceEvent and aeon_trace_event_t have identical layout
      static_assert(sizeof(aeon_trace_event_t) == sizeof(aeon::TraceEvent),
                    "FFI struct must match C++ struct size");
      std::memcpy(&out_events[i], &history[i], sizeof(aeon_trace_event_t));
    }
    *out_count = count;
    return AEON_OK;
  } catch (...) {
    return AEON_ERR_UNKNOWN;
  }
}

AEON_API aeon_error_t aeon_trace_size(aeon_trace_t *trace, size_t *out_size) {
  if (!trace || !out_size)
    return AEON_ERR_NULL_PTR;

  try {
    *out_size = to_trace(trace)->size();
    return AEON_OK;
  } catch (...) {
    return AEON_ERR_UNKNOWN;
  }
}

AEON_API aeon_error_t aeon_trace_compact(aeon_trace_t *trace) {
  if (!trace)
    return AEON_ERR_NULL_PTR;

  try {
    to_trace(trace)->compact();
    return AEON_OK;
  } catch (const std::runtime_error &) {
    return AEON_ERR_FILE_IO;
  } catch (...) {
    return AEON_ERR_UNKNOWN;
  }
}

AEON_API aeon_error_t aeon_trace_tombstone_event(aeon_trace_t *trace,
                                                 uint64_t event_id) {
  if (!trace)
    return AEON_ERR_NULL_PTR;

  try {
    bool found = to_trace(trace)->tombstone_event(event_id);
    return found ? AEON_OK : AEON_ERR_NODE_NOT_FOUND;
  } catch (...) {
    return AEON_ERR_UNKNOWN;
  }
}
AEON_API aeon_error_t aeon_trace_get_event_text(
    aeon_trace_t *trace, uint64_t blob_offset, uint32_t blob_size,
    char *out_buf, size_t buf_capacity, size_t *out_len) {
  if (!trace || !out_buf || !out_len)
    return AEON_ERR_NULL_PTR;

  try {
    std::string text = to_trace(trace)->get_event_text(blob_offset, blob_size);
    *out_len = text.size();

    // Check if caller buffer is large enough (need space for null terminator)
    if (buf_capacity < text.size() + 1) {
      return AEON_ERR_BUFFER_TOO_SMALL;
    }

    std::memcpy(out_buf, text.data(), text.size());
    out_buf[text.size()] = '\0';
    return AEON_OK;
  } catch (...) {
    return AEON_ERR_UNKNOWN;
  }
}

} // extern "C"
