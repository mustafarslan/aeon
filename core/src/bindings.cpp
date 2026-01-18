#include "aeon/atlas.hpp"
#include "aeon/core.hpp"
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
            if (vec.size() != 768)
              throw std::invalid_argument("Vector must be 768-dim");
            return self.insert(parent, std::span<const float>(vec), meta);
          },
          "parent_id"_a, "vector"_a, "metadata"_a)

      .def(
          "insert_delta",
          [](aeon::Atlas &self, const std::vector<float> &vec,
             const std::string &meta) {
            if (vec.size() != 768)
              throw std::invalid_argument("Vector must be 768-dim");
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
          [](aeon::Atlas &self, const std::vector<float> &query) {
            if (query.size() != 768)
              throw std::invalid_argument("Vector must be 768-dim");

            // 1. Run C++ Search (Block GIL during search? No, we removed
            // release logic for debugging) Ideally we release GIL, but
            // investigating SegFault.

            size_t num_bytes = 0;
            std::vector<aeon::Atlas::ResultNode> results;

            {
              // RE-ENABLED GIL RELEASE for production concurrency
              nb::gil_scoped_release release;
              results =
                  self.navigate(std::span<const float>(query.data(), 768));
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
          "query"_a,
          "Returns byte array of results (view as structured in Python)")

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
          "node_ids"_a, "Pre-fill SLB cache with node IDs for warm start");

  // --- Trace Manager (Graph Consolidation) ---
  nb::class_<aeon::TraceManager>(m, "TraceManager")
      .def(nb::init<>())
      .def("size", &aeon::TraceManager::size)
      .def("add_node", &aeon::TraceManager::add_node, "id"_a, "role"_a,
           "text"_a, "timestamp"_a)
      .def("add_edge", &aeon::TraceManager::add_edge, "source"_a, "target"_a,
           "type"_a)
      .def("has_node", &aeon::TraceManager::has_node, "id"_a)
      .def("consolidate", &aeon::TraceManager::consolidate, "node_ids"_a,
           "summary"_a,
           "Consolidate multiple nodes into a single summary node, rewiring "
           "edges.")
      .def(
          "get_successors",
          [](aeon::TraceManager &self, const std::string &id) {
            (void)self;
            (void)id;
            // Stub for now.
            return std::vector<std::string>{};
          },
          "id"_a);
}
