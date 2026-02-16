#include "aeon/atlas.hpp"
#include "aeon/core.hpp"
#include "aeon/hierarchical_slb.hpp"
#include "aeon/trace.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;

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
      .def(nb::init<std::filesystem::path>(), "path"_a)
      .def("size", &aeon::Atlas::size)

      .def(
          "insert",
          [](aeon::Atlas &self, uint64_t parent, const std::vector<float> &vec,
             const std::string &meta) {
            if (vec.size() != self.dim())
              throw std::invalid_argument("Vector dim mismatch (expected " +
                                          std::to_string(self.dim()) + ")");
            return self.insert(parent, std::span<const float>(vec), meta);
          },
          "parent_id"_a, "vector"_a, "metadata"_a)

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
             uint32_t beam_width, bool apply_csls) {
            if (query.size() != self.dim())
              throw std::invalid_argument("Vector dim mismatch (expected " +
                                          std::to_string(self.dim()) + ")");

            // Release GIL during C++ search for Python-side concurrency.

            size_t num_bytes = 0;
            std::vector<aeon::Atlas::ResultNode> results;

            {
              nb::gil_scoped_release release;
              results = self.navigate(
                  std::span<const float>(query.data(), self.dim()), beam_width,
                  apply_csls);
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
          "Beam search navigate. beam_width=1 is greedy. apply_csls=True "
          "applies hub penalty.")

      .def(
          "get_children_raw",
          [](aeon::Atlas &self, uint64_t parent_id) {
            std::vector<aeon::Atlas::ResultNode> results =
                self.get_children(parent_id);

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
          "parent_id"_a,
          "Returns byte array of child nodes (view as structured in Python)")

      .def(
          "load_context",
          [](aeon::Atlas &self, const std::vector<uint64_t> &node_ids) {
            nb::gil_scoped_release release;
            self.load_context(
                std::span<const uint64_t>(node_ids.data(), node_ids.size()));
          },
          "node_ids"_a, "Pre-fill SLB cache with node IDs for warm start")

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
      .def(nb::init<>())
      .def(
          "find_nearest",
          [](aeon::HierarchicalSLB &self, uint64_t session_id,
             const std::vector<float> &query, float threshold) {
            nb::gil_scoped_release release;
            auto hit = self.find_nearest(
                session_id, std::span<const float>(query.data(), query.size()),
                threshold);
            return hit;
          },
          "session_id"_a, "query"_a, "threshold"_a = 0.85f,
          "Hierarchical L1/L2 lookup: session cache then global cache")
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
             uint16_t role, const std::string &text, uint64_t atlas_id) {
            nb::gil_scoped_release release;
            return self.append_event(session_id.c_str(), role, text.c_str(),
                                     atlas_id);
          },
          "session_id"_a, "role"_a, "text"_a, "atlas_id"_a = 0,
          "Append an episodic event. Returns the new event ID.")
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
              d["role"] = nb::int_(ev.role);
              d["flags"] = nb::int_(ev.flags);
              d["session_id"] = nb::str(ev.session_id);
              d["text_preview"] = nb::str(ev.text_preview);
              d["blob_offset"] = nb::int_(ev.blob_offset);
              d["blob_size"] = nb::int_(ev.blob_size);
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
