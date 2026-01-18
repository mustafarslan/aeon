#include "aeon/trace.hpp"
#include <stdexcept>
#include <uuid/uuid.h> // Need UUID generation or passed from Python?
// For C++ UUID, we usually use libuuid or just dummy random.
// Let's generate a simple ID in C++ or rely on Python passing it?
// The prompt signature says: consolidate(node_ids, summary) -> string (new ID).
// So C++ must generate ID. We'll use a simple random hex generator to avoid ext
// deps if possible, or just use time-based.
#include <iomanip>
#include <random>
#include <sstream>

namespace aeon {

// Helper: UUID v4 ish
std::string generate_uuid() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 15);

  std::stringstream ss;
  ss << "sum_";
  for (int i = 0; i < 8; ++i)
    ss << std::hex << dis(gen);
  return ss.str();
}

EdgeType TraceManager::parse_edge_type(const std::string &s) {
  if (s == "CAUSAL")
    return EdgeType::CAUSAL;
  if (s == "REFERS_TO")
    return EdgeType::REFERS_TO;
  return EdgeType::NEXT;
}

void TraceManager::add_node(const std::string &id, const std::string &role,
                            const std::string &text, double timestamp) {
  std::unique_lock lock(mutex_);
  nodes_[id] = TraceNode{id, text, role, timestamp, false, {}, {}};
}

void TraceManager::add_edge(const std::string &source,
                            const std::string &target,
                            const std::string &type_str) {
  std::unique_lock lock(mutex_);
  if (nodes_.find(source) == nodes_.end() ||
      nodes_.find(target) == nodes_.end()) {
    return; // Or throw
  }

  EdgeType type = parse_edge_type(type_str);
  nodes_[source].out_edges.push_back({target, type});
  nodes_[target].in_edges.push_back({source, type});
}

bool TraceManager::has_node(const std::string &id) const {
  std::shared_lock lock(mutex_);
  return nodes_.find(id) != nodes_.end();
}

size_t TraceManager::size() const {
  std::shared_lock lock(mutex_);
  return nodes_.size();
}

std::string TraceManager::consolidate(const std::vector<std::string> &node_ids,
                                      const std::string &summary_text) {
  std::unique_lock lock(mutex_); // Exclusive write access

  if (node_ids.empty())
    throw std::invalid_argument("Cannot consolidate empty list");

  // 1. Validate all nodes exist
  for (const auto &nid : node_ids) {
    if (nodes_.find(nid) == nodes_.end()) {
      throw std::runtime_error("Node not found: " + nid);
    }
  }

  // 2. Determine Subgraph Boundaries
  // We assume node_ids are ordered temporally or we verify connectivity.
  // For general consolidation, we care about:
  // - Edges entering the set from OUTSIDE
  // - Edges leaving the set to OUTSIDE
  // - Internal edges (can be ignored/archived)

  std::string start_node = node_ids.front();
  std::string end_node = node_ids.back(); // Assuming order!

  // Create Summary Node
  std::string summary_id = generate_uuid();
  TraceNode summary;
  summary.id = summary_id;
  summary.text = summary_text;
  summary.role = "summary";
  summary.timestamp = nodes_[start_node].timestamp; // Inherit start time

  // 3. Rewire Incoming Edges (to the group) -> Point to Summary
  for (const auto &nid : node_ids) {
    TraceNode &node = nodes_[nid];

    for (const auto &inc : node.in_edges) {
      std::string source_id = inc.first;

      // If source is NOT in the group, it's an external incoming edge
      bool is_internal = false;
      for (const auto &x : node_ids)
        if (x == source_id)
          is_internal = true;

      if (!is_internal) {
        // Rewire: Source -> Summary
        TraceNode &source_node = nodes_[source_id];

        // Update source's out_edges: replace 'nid' with 'summary_id'
        for (auto &edge : source_node.out_edges) {
          if (edge.first == nid) {
            edge.first = summary_id;
          }
        }

        // Add to summary in_edges
        summary.in_edges.push_back({source_id, inc.second});
      }
    }

    // 4. Rewire Outgoing Edges (from the group) -> Point from Summary
    for (const auto &out : node.out_edges) {
      std::string target_id = out.first;

      bool is_internal = false;
      for (const auto &x : node_ids)
        if (x == target_id)
          is_internal = true;

      if (!is_internal) {
        // Rewire: Summary -> Target
        TraceNode &target_node = nodes_[target_id];

        // Update target's in_edges: replace 'nid' with 'summary_id'
        for (auto &edge : target_node.in_edges) {
          if (edge.first == nid) {
            edge.first = summary_id;
          }
        }

        // Add to summary out_edges
        summary.out_edges.push_back({target_id, out.second});
      }
    }

    // 5. Archive Original Node
    node.archived = true;
  }

  // 6. Insert Summary Node
  nodes_[summary_id] = summary;

  return summary_id;
}

} // namespace aeon
