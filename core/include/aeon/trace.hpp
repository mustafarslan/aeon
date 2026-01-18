#pragma once

#include <algorithm>
#include <iostream>
#include <map>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace aeon {

enum class EdgeType { NEXT, CAUSAL, REFERS_TO };

struct TraceNode {
  std::string id;
  std::string text;
  std::string role; // "user", "system", "summary", "concept"
  double timestamp;
  bool archived = false;

  // Adjacency List: target_id -> type
  // Using a map for simple lookup, or vector of pairs
  std::vector<std::pair<std::string, EdgeType>> out_edges;
  std::vector<std::pair<std::string, EdgeType>> in_edges;
};

class TraceManager {
public:
  TraceManager() = default;

  // Core Graph Ops
  void add_node(const std::string &id, const std::string &role,
                const std::string &text, double timestamp);
  void add_edge(const std::string &source, const std::string &target,
                const std::string &type_str);

  bool has_node(const std::string &id) const;

  // Consolidation Logic
  // Returns string ID of new summary node
  std::string consolidate(const std::vector<std::string> &node_ids,
                          const std::string &summary_text);

  size_t size() const;

private:
  mutable std::shared_mutex mutex_;
  std::unordered_map<std::string, TraceNode> nodes_;

  // Helpers
  EdgeType parse_edge_type(const std::string &s);
};

} // namespace aeon
