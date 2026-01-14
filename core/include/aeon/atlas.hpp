#pragma once

#include "aeon/schema.hpp"
#include "aeon/slb.hpp"
#include "aeon/storage.hpp"
#include <filesystem>
#include <shared_mutex>
#include <span>
#include <string_view>
#include <vector>

namespace aeon {

class Atlas {
public:
  explicit Atlas(std::filesystem::path path);
  ~Atlas();

  /**
   * @brief Navigates the tree to find the closest leaf to the query vector.
   * Uses SIMD-accelerated greedy search.
   *
   * @param query 768-dimensional vector
   * @return std::vector<Node> Path from root to leaf (breadcrumbs)
   */
  /**
   * @brief Lightweight node representation for search results.
   * Optimized for Python Zero-Copy views.
   */
  struct ResultNode {
    uint64_t id;
    float similarity;
    float centroid_preview[3]; // First 3 dims for visualization
  };

  /**
   * @brief Navigates the tree to find the closest leaf to the query vector.
   * Uses SIMD-accelerated greedy search.
   *
   * @param query 768-dimensional vector
   * @return std::vector<ResultNode> Path from root to leaf (breadcrumbs)
   */
  std::vector<ResultNode> navigate(std::span<const float> query);

  /**
   * @brief Retrieves all direct children of the specified node.
   * Used for inspecting the "Room" options.
   *
   * @param parent_id ID of the parent node
   * @return std::vector<ResultNode> List of child nodes (ResultNode format)
   */
  std::vector<ResultNode> get_children(uint64_t parent_id);

  /**
   * @brief Inserts a new node as a child of the specified parent.
   *
   * @param parent_id ID of the parent node (0 for new Root, if empty)
   * @param vector 768-dimensional vector
   * @param metadata Description string (truncated to 255 chars)
   * @return uint64_t ID of the new node
   */
  uint64_t insert(uint64_t parent_id, std::span<const float> vector,
                  std::string_view metadata);

  /**
   * @brief Inserts a new node into the in-memory Delta Buffer.
   * Thread-safe (exclusive lock).
   *
   * @param vector 768-dimensional embedding
   * @param metadata Description string
   * @return uint64_t Temporary ID (MSB set high: 0x8000... + index)
   */
  uint64_t insert_delta(std::span<const float> vector,
                        std::string_view metadata);

  /**
   * @brief Returns the total number of nodes in the atlas.
   */
  size_t size() const;

  /**
   * @brief Pre-fills the SLB cache with the specified nodes.
   * Useful for "Warm Start" from session history.
   */
  void load_context(std::span<const uint64_t> node_ids);

private:
  storage::MemoryFile file_;
  std::vector<Node> delta_buffer_;
  mutable std::shared_mutex delta_mutex_;
  SemanticCache slb_cache_;
};

} // namespace aeon
