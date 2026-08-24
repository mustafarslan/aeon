#include "aeon/atlas.hpp"
#include "aeon/core.hpp"
#include "aeon/hash.hpp"
#include "aeon/hierarchical_slb.hpp"
#include "aeon/trace.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>
#include <optional>

namespace nb = nanobind;
using namespace nb::literals;

namespace {
/// Atlas's session-aware SLB cache is keyed by uint64_t; Python callers
/// (e.g. the FastAPI server's authenticated user_id) work with strings.
/// Same FNV-1a mapping as aeon_c_api.cpp's session_id_to_u64() so a given
/// session string routes to the same shard whether reached via the C-API
/// or these Python bindings. std::nullopt/empty maps to Atlas's "no
/// session" default (session_id=0), matching the C++ layer's own default.
uint64_t py_session_id_to_u64(const std::optional<std::string> &session_id) {
  if (!session_id || session_id->empty())
    return 0;
  return aeon::hash::fnv1a_64(session_id->data(), session_id->size());
}
} // namespace

NB_MODULE(core, m) {
  m.doc() = "Aeon Core C++23 High-Performance Backend";

  // --- EBR Guard (Python Context Manager) ---
  nb::class_<aeon::EpochGuard>(m, "EpochGuard")
      .def("__enter__", [](nb::object self) -> nb::object { return self; })
      .def("__exit__", [](aeon::EpochGuard &self, nb::args) { self.release(); })
      .def("release", &aeon::EpochGuard::release,
           "Explicitly release the epoch guard (idempotent)")
      .def("is_active", &aeon::EpochGuard::is_active,
           "Check if the guard is still protecting memory");

  // --- Core Utils ---
  m.def("version", &aeon::core::version, "Get the library version");

  nb::class_<aeon::core::BuildInfo>(m, "BuildInfo")
      .def_ro("compiler", &aeon::core::BuildInfo::compiler)
      .def_ro("architecture", &aeon::core::BuildInfo::architecture)
      .def_ro("simd_level", &aeon::core::BuildInfo::simd_level)
      .def_ro("standard", &aeon::core::BuildInfo::standard)
      .def_prop_ro("repr", [](const aeon::core::BuildInfo &b) {
        return "<BuildInfo arch='" + b.architecture + "' simd='" +
               b.simd_level + "' compiler='" + b.compiler + "'>";
      });

  m.def("get_build_info", &aeon::core::get_build_info,
        "Get build environment details");

  // --- Safety Latches ---
  m.def(
      "get_result_node_size", []() { return sizeof(aeon::Atlas::ResultNode); },
      "Return size of ResultNode struct for schema validation");

  // --- Atlas Engine ---

  nb::class_<aeon::Atlas>(m, "Atlas")
      .def(
          "__init__",
          [](aeon::Atlas *self, std::filesystem::path path, uint32_t dim,
             uint32_t quantization_type, uint32_t metadata_size) {
            aeon::AtlasOptions opts;
            opts.dim = dim;
            opts.quantization_type = quantization_type;
            opts.metadata_size = metadata_size;
            new (self) aeon::Atlas(std::move(path), opts);
          },
          "path"_a, "dim"_a = 0, "quantization_type"_a = 0,
          "metadata_size"_a = 0,
          "dim/quantization_type/metadata_size are new-file-only (0 = "
          "default); an existing file's on-disk values are always "
          "authoritative. metadata_size (v4-plan.md Stage 4 task 6 Phase "
          "B) lets a store be opened with a larger metadata field, e.g. "
          "for the shared Atlas store to absorb encrypted-payload "
          "overhead.")
      .def("size", &aeon::Atlas::size)
      .def_prop_ro("metadata_size", &aeon::Atlas::metadata_size,
                   "Metadata field size (bytes) of this Atlas instance -- "
                   "callers writing an encoded payload into the metadata "
                   "field must length-check against this BEFORE calling "
                   "insert(), since insert() truncates silently rather "
                   "than raising on overflow.")

      .def(
          "insert",
          [](aeon::Atlas &self, uint64_t parent, const std::vector<float> &vec,
             const std::string &meta,
             const std::optional<std::string> &session_id) {
            if (vec.size() != self.dim())
              throw std::invalid_argument("Vector dim mismatch (expected " +
                                          std::to_string(self.dim()) + ")");
            return self.insert(parent, std::span<const float>(vec), meta,
                              py_session_id_to_u64(session_id));
          },
          "parent_id"_a, "vector"_a, "metadata"_a, "session_id"_a = nb::none())

      .def(
          "insert_delta",
          [](aeon::Atlas &self, const std::vector<float> &vec,
             const std::string &meta) {
            if (vec.size() != self.dim())
              throw std::invalid_argument("Vector dim mismatch (expected " +
                                          std::to_string(self.dim()) + ")");
            // Release GIL while waiting for mutex/inserting
            nb::gil_scoped_release release;
            return self.insert_delta(std::span<const float>(vec), meta);
          },
          "vector"_a, "metadata"_a,
          "Insert into in-memory Delta Buffer (immediate availability)")

      .def(
          "prune_delta_tail",
          [](aeon::Atlas &self, size_t n) {
            nb::gil_scoped_release release;
            return self.prune_delta_tail(n);
          },
          "n"_a, "Remove last N nodes from delta buffer (for rollback)")

      .def(
          "navigate_raw",
          [](aeon::Atlas &self, const std::vector<float> &query,
             uint32_t beam_width, bool apply_csls,
             const std::optional<std::string> &session_id,
             uint64_t scope_mask) {
            if (query.size() != self.dim())
              throw std::invalid_argument("Vector dim mismatch (expected " +
                                          std::to_string(self.dim()) + ")");

            // Release GIL during C++ search for Python-side concurrency.

            size_t num_bytes = 0;
            std::vector<aeon::Atlas::ResultNode> results;
            uint64_t sid = py_session_id_to_u64(session_id);

            {
              nb::gil_scoped_release release;
              results = self.navigate(
                  std::span<const float>(query.data(), self.dim()), beam_width,
                  apply_csls, sid, scope_mask);
            }
            // GIL is re-acquired here

            num_bytes = results.size() * sizeof(aeon::Atlas::ResultNode);

            // Allocate raw byte buffer
            uint8_t *data = new uint8_t[num_bytes];
            if (num_bytes > 0) {
              std::memcpy(data, results.data(), num_bytes);
            }

            // Create capsule with destructor for uint8_t array
            nb::capsule owner(data,
                              [](void *p) noexcept { delete[] (uint8_t *)p; });

            // Return Byte Array (uint8 view, Read-Only)
            return nb::ndarray<uint8_t, nb::numpy, nb::shape<-1>, nb::ro>(
                data, {num_bytes}, owner);
          },
          "query"_a, "beam_width"_a = 1, "apply_csls"_a = false,
          "session_id"_a = nb::none(),
          "scope_mask"_a = aeon::ALL_SCOPES_VISIBLE,
          "Beam search navigate. beam_width=1 is greedy. apply_csls=True "
          "applies hub penalty. session_id scopes the SLB cache lookup to "
          "the caller's session (v4-plan.md Stage 0) -- omit for the "
          "shared default session. scope_mask (v4-plan.md Stage 2) filters "
          "results to nodes matching scope_mask & node.scope_bitmap != 0; "
          "omit (ALL_SCOPES_VISIBLE) for unfiltered pre-Stage-2 behavior.")

      .def(
          "drop_session",
          [](aeon::Atlas &self, const std::string &session_id) {
            return self.drop_session(
                aeon::hash::fnv1a_64(session_id.data(), session_id.size()));
          },
          "session_id"_a,
          "Remove a session's SLB L1 cache entry and free its memory "
          "(prevents unbounded growth across many short-lived sessions)")

      .def("sync", &aeon::Atlas::sync,
           "Explicitly flush pending mmap writes to disk (see Atlas::insert "
           "doc comment for the durability model this closes the gap on)")

      .def("set_node_scope", &aeon::Atlas::set_node_scope, "node_id"_a,
           "scope_bitmap"_a,
           "Sets a node's scope_bitmap in place (v4-plan.md Stage 1/2). "
           "Mmap nodes only -- raises for a delta-arena id or if "
           "compaction is in progress.")
      .def("get_node_scope", &aeon::Atlas::get_node_scope, "node_id"_a,
           "Reads a node's current scope_bitmap.")
      .def("supersede_node", &aeon::Atlas::supersede_node, "node_id"_a,
           "Reversibly excludes a node from beam search results "
           "(v4-plan.md Stage 2), branchless like tombstoning but "
           "reversible via revoke_node_supersede(). Mmap nodes only.")
      .def("revoke_node_supersede", &aeon::Atlas::revoke_node_supersede,
           "node_id"_a,
           "Reverses a prior supersede_node() call. Mmap nodes only.")
      .def("is_node_superseded", &aeon::Atlas::is_node_superseded,
           "node_id"_a,
           "Reads whether a node currently has NODE_FLAG_SUPERSEDED set.")

      .def("set_node_governance_id", &aeon::Atlas::set_node_governance_id,
           "node_id"_a, "governance_record_id"_a,
           "Sets a node's governance_record_id in place (v4-plan.md Stage "
           "4) -- an opaque link into the control plane. Mmap nodes only.")
      .def("get_node_governance_id", &aeon::Atlas::get_node_governance_id,
           "node_id"_a,
           "Reads a node's current governance_record_id.")
      .def("get_node_metadata", &aeon::Atlas::get_node_metadata,
           "node_id"_a,
           "Reads a node's metadata string back out (v4-plan.md Stage 4 "
           "task 2, promotion). Works for both mmap and delta-arena ids.")
      .def("get_node_centroid", &aeon::Atlas::get_node_centroid,
           "node_id"_a,
           "Reads a node's full centroid vector back out, dequantized to "
           "FP32 if this Atlas is INT8-quantized (v4-plan.md Stage 4 task "
           "2, promotion). Works for both mmap and delta-arena ids.")
      .def("list_nodes_by_scope", &aeon::Atlas::list_nodes_by_scope,
           "scope_mask"_a,
           "Lists live (non-tombstoned) node ids whose scope_bitmap "
           "overlaps scope_mask (v4-plan.md Stage 4 console primitive). "
           "Superseded nodes ARE included; tombstoned nodes are not.")
      .def("bulk_set_node_scope", &aeon::Atlas::bulk_set_node_scope,
           "updates"_a,
           "Applies many (node_id, scope_bitmap) updates under a single "
           "lock/WAL-flush pass (v4-plan.md Stage 4 bulk bit remap). "
           "All-or-nothing: every entry is validated before any node is "
           "mutated.")
      .def("tombstone_node", &aeon::Atlas::tombstone_node, "node_id"_a,
           "Logically deletes a single mmap node by id (v4-plan.md Stage 4 "
           "console/erasure-workflow primitive). WAL-protected, idempotent, "
           "TERMINAL (no revoke) -- see Atlas::tombstone_node's doc "
           "comment for the physical-vs-logical deletion distinction the "
           "erasure workflow must account for.")

      .def(
          "get_children_raw",
          [](aeon::Atlas &self, uint64_t parent_id, uint64_t scope_mask) {
            std::vector<aeon::Atlas::ResultNode> results =
                self.get_children(parent_id, scope_mask);

            size_t num_bytes = results.size() * sizeof(aeon::Atlas::ResultNode);

            uint8_t *data = new uint8_t[num_bytes];
            if (num_bytes > 0) {
              std::memcpy(data, results.data(), num_bytes);
            }

            nb::capsule owner(data,
                              [](void *p) noexcept { delete[] (uint8_t *)p; });

            return nb::ndarray<uint8_t, nb::numpy, nb::shape<-1>, nb::ro>(
                data, {num_bytes}, owner);
          },
          "parent_id"_a, "scope_mask"_a = aeon::ALL_SCOPES_VISIBLE,
          "Returns byte array of child nodes (view as structured in "
          "Python). scope_mask (v4-plan.md Stage 2 task 2): the "
          "Atlas->Trace->Atlas graph-expansion-boundary enforcement point "
          "-- see Atlas::get_children()'s doc comment (atlas.hpp).")

      .def(
          "load_context",
          [](aeon::Atlas &self, const std::vector<uint64_t> &node_ids,
             const std::optional<std::string> &session_id) {
            nb::gil_scoped_release release;
            self.load_context(
                std::span<const uint64_t>(node_ids.data(), node_ids.size()),
                py_session_id_to_u64(session_id));
          },
          "node_ids"_a, "session_id"_a = nb::none(),
          "Pre-fill SLB cache with node IDs for warm start, scoped to "
          "session_id (v4-plan.md Stage 0)")

      // --- Dreaming Kernel (Phase 3) ---

      .def(
          "consolidate_subgraph",
          [](aeon::Atlas &self, const std::vector<uint64_t> &old_ids,
             const std::vector<float> &summary_vec, const std::string &meta) {
            if (summary_vec.size() != self.dim())
              throw std::invalid_argument(
                  "Summary vector dim mismatch (expected " +
                  std::to_string(self.dim()) + ")");
            nb::gil_scoped_release release;
            return self.consolidate_subgraph(
                std::span<const uint64_t>(old_ids),
                std::span<const float>(summary_vec), meta);
          },
          "old_node_ids"_a, "summary_vector"_a, "summary_metadata"_a,
          "Atomically: insert summary → re-wire children → tombstone old "
          "nodes. Returns the new summary node ID.")

      .def(
          "compact_mmap",
          [](aeon::Atlas &self) {
            nb::gil_scoped_release release;
            self.compact_mmap();
          },
          "Shadow compaction: defragment Atlas file with generational "
          "naming (stutter-free, no path needed).")

      .def(
          "tombstone_count",
          [](aeon::Atlas &self) {
            nb::gil_scoped_release release;
            return self.tombstone_count();
          },
          "Returns count of tombstoned (dead) nodes for compaction triggers.")

      .def(
          "acquire_read_guard",
          [](aeon::Atlas &self) { return self.acquire_read_guard(); },
          nb::rv_policy::move, nb::keep_alive<0, 1>(),
          "Acquire EBR read guard for safe zero-copy memory access");

  // --- Hierarchical SLB (Multi-Tenant Semantic Cache) ---
  nb::class_<aeon::HierarchicalSLB>(m, "HierarchicalSLB")
      .def(nb::init<uint32_t>(), "dim"_a = aeon::EMBEDDING_DIM_DEFAULT,
           "Create a session-aware cache for the given embedding dimension "
           "(must match the owning Atlas's dim)")
      .def_prop_ro("dim", &aeon::HierarchicalSLB::dim)
      .def(
          "find_nearest",
          [](aeon::HierarchicalSLB &self, uint64_t session_id,
             const std::vector<float> &query,
             float threshold) -> nb::object {
            std::optional<aeon::HierarchicalSLB::SLBHit> hit;
            {
              nb::gil_scoped_release release;
              hit = self.find_nearest(
                  session_id,
                  std::span<const float>(query.data(), query.size()),
                  threshold);
            }
            if (!hit)
              return nb::none();
            // Convert to dict for Python ergonomics (same pattern as
            // TraceManager::get_history below) -- SLBHit/Hit is a plain
            // C++ struct, not a registered nanobind type.
            nb::dict d;
            d["node_id"] = nb::int_(hit->node_id);
            d["similarity"] = nb::float_(hit->similarity);
            d["centroid_preview"] =
                nb::make_tuple(hit->centroid_preview[0],
                               hit->centroid_preview[1],
                               hit->centroid_preview[2]);
            return d;
          },
          "session_id"_a, "query"_a, "threshold"_a = 0.85f,
          "Hierarchical L1/L2 lookup: session cache then global cache. "
          "Returns a dict with node_id/similarity/centroid_preview, or "
          "None on a cache miss.")
      .def(
          "insert",
          [](aeon::HierarchicalSLB &self, uint64_t session_id, uint64_t node_id,
             const std::vector<float> &centroid) {
            nb::gil_scoped_release release;
            self.insert(
                session_id, node_id,
                std::span<const float>(centroid.data(), centroid.size()));
          },
          "session_id"_a, "node_id"_a, "centroid"_a,
          "Insert into session L1 cache and global L2 cache")
      .def(
          "drop_session",
          [](aeon::HierarchicalSLB &self, uint64_t session_id) {
            nb::gil_scoped_release release;
            return self.drop_session(session_id);
          },
          "session_id"_a,
          "Remove session and free its L1 cache (prevents OOM leaks)")
      .def("active_session_count", &aeon::HierarchicalSLB::active_session_count,
           "Count of active sessions across all shards (diagnostic)")
      .def_prop_ro_static(
          "shard_count",
          [](nb::handle) { return aeon::HierarchicalSLB::shard_count(); },
          "Number of lock-striped shards (64)");

  // --- Trace Manager (mmap-backed Episodic Memory) ---
  nb::class_<aeon::TraceManager>(m, "TraceManager")
      .def(nb::init<>(), "Create in-memory-only trace manager")
      .def(nb::init<std::filesystem::path>(), "path"_a,
           "Create or open mmap-backed trace file")
      .def("size", &aeon::TraceManager::size,
           "Total event count (mmap + delta)")
      .def("mmap_event_count", &aeon::TraceManager::mmap_event_count,
           "Event count in mmap file")
      .def("delta_event_count", &aeon::TraceManager::delta_event_count,
           "Event count in delta buffer")
      .def(
          "append_event",
          [](aeon::TraceManager &self, const std::string &session_id,
             uint16_t role, const std::string &text, uint64_t atlas_id,
             const std::vector<float> &embedding, uint8_t edge_type,
             uint64_t supersedes_id, uint8_t reason_code,
             uint64_t event_time) {
            nb::gil_scoped_release release;
            return self.append_event(
                session_id.c_str(), role, text.c_str(), atlas_id,
                std::span<const float>(embedding.data(), embedding.size()),
                edge_type, supersedes_id, reason_code, event_time);
          },
          "session_id"_a, "role"_a, "text"_a, "atlas_id"_a = 0,
          "embedding"_a = std::vector<float>{}, "edge_type"_a = 0,
          "supersedes_id"_a = 0, "reason_code"_a = 0, "event_time"_a = 0,
          "Append an episodic event. Returns the new event ID. embedding "
          "(v4-plan.md Stage 2 task 3): optional, empty by default = not "
          "embedded (excluded from semantic_search()). The FIRST non-empty "
          "embedding ever appended to this trace file fixes its "
          "embedding_dim; a later mismatched size raises ValueError. "
          "edge_type/supersedes_id/reason_code (v4-plan.md Stage 1/2 task "
          "4): EdgeType/ReasonCode enum values (see schema.hpp) for a "
          "version/admission edge this event carries -- 0/None (default) "
          "means no edge. supersedes_id is in practice always a "
          "store-encoded Atlas node id, not a TraceEvent id -- see "
          "schema.hpp's TraceEvent doc comment (V4 STAGE 4 note). "
          "event_time (v4-plan.md Stage 7 Track 2): optional caller-"
          "supplied event time (epoch microseconds), distinct from the "
          "event's timestamp (always Aeon's own insertion wall-clock). "
          "0/default = not supplied -- callers ordering by 'when this "
          "happened' should fall back to timestamp when event_time is 0.")
      .def(
          "semantic_search",
          [](aeon::TraceManager &self, const std::vector<float> &query,
             size_t top_k) -> nb::list {
            std::vector<aeon::TraceEvent> events;
            {
              nb::gil_scoped_release release;
              events = self.semantic_search(
                  std::span<const float>(query.data(), query.size()), top_k);
            }
            nb::list result;
            for (const auto &ev : events) {
              nb::dict d;
              d["id"] = nb::int_(ev.id);
              d["atlas_id"] = nb::int_(ev.atlas_id);
              d["timestamp"] = nb::int_(ev.timestamp);
              d["event_time"] = nb::int_(ev.event_time);
              d["role"] = nb::int_(ev.role);
              d["session_id"] = nb::str(ev.session_id);
              d["text_preview"] = nb::str(ev.text_preview);
              if (ev.blob_size > 0) {
                d["text"] = nb::str(
                    self.get_event_text(ev.blob_offset, ev.blob_size).c_str());
              } else {
                d["text"] = nb::str(ev.text_preview);
              }
              result.append(d);
            }
            return result;
          },
          "query"_a, "top_k"_a = 10,
          "Semantic search over embedded trace events (v4-plan.md Stage 2 "
          "task 3), via TraceBlockIndex's two-phase O(|V|/1024 + K*1024) "
          "search. Only events appended with a non-empty embedding are "
          "indexed. Returns an empty list if no embedding has ever been "
          "appended to this file, or if query's length doesn't match "
          "embedding_dim.")
      .def_prop_ro("embedding_dim", &aeon::TraceManager::embedding_dim,
                   "Dimensionality of indexed embeddings, or 0 if none have "
                   "been appended to this trace file yet.")
      .def(
          "get_history",
          [](aeon::TraceManager &self, const std::string &session_id,
             size_t limit) -> nb::list {
            std::vector<aeon::TraceEvent> events;
            {
              nb::gil_scoped_release release;
              events = self.get_history(session_id.c_str(), limit);
            }
            // Convert to list of dicts for Python ergonomics
            nb::list result;
            for (const auto &ev : events) {
              nb::dict d;
              d["id"] = nb::int_(ev.id);
              d["prev_id"] = nb::int_(ev.prev_id);
              d["atlas_id"] = nb::int_(ev.atlas_id);
              d["timestamp"] = nb::int_(ev.timestamp);
              d["event_time"] = nb::int_(ev.event_time);
              d["role"] = nb::int_(ev.role);
              d["flags"] = nb::int_(ev.flags);
              d["session_id"] = nb::str(ev.session_id);
              d["text_preview"] = nb::str(ev.text_preview);
              d["blob_offset"] = nb::int_(ev.blob_offset);
              d["blob_size"] = nb::int_(ev.blob_size);
              // V4 Stage 1: version/supersession edge fields.
              d["edge_type"] = nb::int_(ev.edge_type);
              d["supersedes_id"] = nb::int_(ev.supersedes_id);
              d["reason_code"] = nb::int_(ev.reason_code);
              // Fetch full text from blob arena
              if (ev.blob_size > 0) {
                d["text"] = nb::str(
                    self.get_event_text(ev.blob_offset, ev.blob_size).c_str());
              } else {
                d["text"] = nb::str(ev.text_preview);
              }
              result.append(d);
            }
            return result;
          },
          "session_id"_a, "limit"_a = 100,
          "Retrieve session history (newest first). Returns list of dicts.")
      .def(
          "compact",
          [](aeon::TraceManager &self) {
            nb::gil_scoped_release release;
            self.compact();
          },
          "Shadow compaction: defragment trace file.")
      .def("has_session", &aeon::TraceManager::has_session, "session_id"_a,
           "Check if a session has any events")
      .def(
          "drop_session",
          [](aeon::TraceManager &self, const std::string &session_id) {
            nb::gil_scoped_release release;
            return self.drop_session(session_id.c_str());
          },
          "session_id"_a, "Drop session tail pointer (session cleanup)");
}
