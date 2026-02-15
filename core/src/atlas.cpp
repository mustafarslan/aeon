#include "aeon/atlas.hpp"
#include "aeon/math_kernel.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

#if defined(__SSE__) || defined(__x86_64__) || defined(_M_X64)
#include <xmmintrin.h>
#endif

namespace aeon {

Atlas::Atlas(std::filesystem::path path) {
  // Wire EBR manager BEFORE opening file (grow() needs it)
  file_.set_epoch_manager(&epoch_mgr_);
  auto result = file_.open(path);
  if (!result) {
    throw std::runtime_error("Failed to open Atlas storage");
  }
  // Reserve memory to prevent reallocations during initial ingestion
  delta_buffer_.reserve(10000);
}

Atlas::~Atlas() = default;

EpochGuard Atlas::acquire_read_guard() { return epoch_mgr_.enter_guard(); }

size_t Atlas::size() const {
  if (auto *header = file_.get_header()) {
    return header->node_count;
  }
  return 0;
}

std::vector<Atlas::ResultNode> Atlas::navigate(std::span<const float> query) {
  std::vector<Atlas::ResultNode> path;

  if (query.size() != 768) {
    return path;
  }

  /// Fast-path: consult the Semantic Lookaside Buffer for L1-resident cache
  /// hits.
  if (auto hit = slb_cache_.find_nearest(query, SLB_HIT_THRESHOLD)) {
    /// SLB hit — return immediately, bypassing the full tree traversal.
    return {{hit->node_id,
             hit->similarity,
             {hit->centroid_preview[0], hit->centroid_preview[1],
              hit->centroid_preview[2]}}};
  }

  // CRITICAL: Acquire epoch guard before ANY mmap pointer access.
  // This prevents grow() from munmapping the data we're reading.
  auto guard = epoch_mgr_.enter_guard();

  // Shared lock: allows concurrent readers, blocks only during insert().
  std::shared_lock<std::shared_mutex> read_lock(write_mutex_);

  auto *header = file_.get_header();

  /// Greedy descent through the memory-mapped B+ tree (immutable layer).
  if (header && header->node_count > 0) {
    // Start at Root (Index 0)
    uint64_t current_idx = 0;
    Node *current = file_.get_node(current_idx);

    if (current) {
      float root_score = math::cosine_similarity(
          query, std::span<const float>(current->centroid, 768));
      path.push_back(
          {current->id,
           root_score,
           {current->centroid[0], current->centroid[1], current->centroid[2]}});

      // Greedy Descent
      while (current->child_count > 0) {
        float best_score = -2.0f; // Cosine similarity is [-1, 1]
        uint64_t best_child_idx = 0;
        bool found_child = false;

        if (current->first_child_offset == 0)
          break;

        intptr_t file_base = (intptr_t)header;
        intptr_t child_addr = file_base + current->first_child_offset;
        Node *children_start = reinterpret_cast<Node *>(child_addr);

        for (uint16_t i = 0; i < current->child_count; ++i) {
          // --- Prefetch Next Child ---
          // Accessing children[i].centroid is a distinct memory region.
          // We prefetch i+1 to hide memory latency during math_kernel(i).
          if (i + 1 < current->child_count) {
            const float *next_centroid = children_start[i + 1].centroid;
            // _MM_HINT_T0: Prefetch into L1 cache
#if defined(__SSE__) || defined(__x86_64__) || defined(_M_X64)
            _mm_prefetch((const char *)next_centroid, _MM_HINT_T0);
#else
            __builtin_prefetch(next_centroid, 0, 3); // 0=read, 3=high locality
#endif
          }

          std::span<const float> child_vec{children_start[i].centroid, 768};
          // ...
          float score = math::cosine_similarity(query, child_vec);

          if (score > best_score) {
            best_score = score;
            best_child_idx =
                (&children_start[i] - file_.get_node(0)); // relative to base
            found_child = true;
          }
        }

        if (!found_child)
          break;

        current = file_.get_node(best_child_idx);
        if (!current)
          break;

        path.push_back({current->id,
                        best_score,
                        {current->centroid[0], current->centroid[1],
                         current->centroid[2]}});
        current_idx = best_child_idx;
      }
    }
  }

  /// Linear scan of the write-ahead delta buffer (mutable layer).
  /// Recent insertions not yet compacted into the mmap are scanned here.
  std::vector<ResultNode> delta_candidates;
  {
    std::shared_lock lock(delta_mutex_);
    delta_candidates.reserve(delta_buffer_.size());
    for (const auto &node : delta_buffer_) {
      float score = math::cosine_similarity(
          query, std::span<const float>(node.centroid, EMBEDDING_DIM));

      /// Retain all delta candidates; top-K filtering occurs after merge.
      delta_candidates.push_back(
          {node.id,
           score,
           {node.centroid[0], node.centroid[1], node.centroid[2]}});
    }
  }

  /// Merge immutable-layer path with delta-layer candidates and return top-K
  /// nearest neighbors ranked by cosine similarity.

  // Combine lists
  path.insert(path.end(), delta_candidates.begin(), delta_candidates.end());

  // Sort by similarity descending
  std::sort(path.begin(), path.end(),
            [](const ResultNode &a, const ResultNode &b) {
              return a.similarity > b.similarity;
            });

  /// Cap result set to prevent unbounded output on large delta buffers.
  if (path.size() > TOP_K_LIMIT) {
    path.resize(TOP_K_LIMIT);
  }

  /// Populate the SLB with the best result for future hit acceleration.
  /// Only mmap-backed nodes are cached; delta nodes are always linear-scanned.
  if (!path.empty()) {
    const auto &best = path[0];
    // Check if ID is from MMap (MSB is 0)
    if ((best.id & 0x8000000000000000ULL) == 0) {
      if (auto *node = file_.get_node(best.id)) {
        slb_cache_.insert(best.id, std::span<const float>(node->centroid, 768));
      }
    }
  }

  return path;
}

std::vector<Atlas::ResultNode> Atlas::get_children(uint64_t parent_id) {
  std::vector<Atlas::ResultNode> children;

  auto guard = epoch_mgr_.enter_guard();
  std::shared_lock<std::shared_mutex> read_lock(write_mutex_);

  auto *header = file_.get_header();
  if (!header || parent_id >= header->node_count) {
    return children;
  }

  Node *parent = file_.get_node(parent_id);
  if (!parent || parent->child_count == 0 || parent->first_child_offset == 0) {
    return children;
  }

  intptr_t file_base = (intptr_t)header;
  intptr_t child_addr = file_base + parent->first_child_offset;
  Node *children_start = reinterpret_cast<Node *>(child_addr);

  children.reserve(parent->child_count);
  for (uint16_t i = 0; i < parent->child_count; ++i) {
    Node *child = &children_start[i];

    // Safety check: Don't read past valid memory
    if ((uint8_t *)child >=
        (uint8_t *)file_base + (header->capacity * sizeof(Node)))
      break;

    children.push_back(
        {child->id,
         0.0f, // No similarity score in simple traversal
         {child->centroid[0], child->centroid[1], child->centroid[2]}});
  }

  return children;
}

uint64_t Atlas::insert_delta(std::span<const float> vector,
                             std::string_view metadata) {
  std::unique_lock lock(delta_mutex_);

  /// Delta node ID uses MSB=1 to distinguish from mmap-backed nodes (MSB=0).
  static const uint64_t DELTA_MASK = 0x8000000000000000ULL;
  uint64_t new_id = DELTA_MASK | delta_buffer_.size();

  Node node{};
  node.id = new_id;

  // Copy Vector
  if (vector.size() == 768) {
    std::copy(vector.begin(), vector.end(), std::begin(node.centroid));
  } else {
    std::fill(std::begin(node.centroid), std::end(node.centroid), 0.0f);
  }

  // Copy Metadata
  std::memset(node.metadata, 0, sizeof(node.metadata));
  size_t meta_len = std::min(metadata.size(), sizeof(node.metadata) - 1);
  std::memcpy(node.metadata, metadata.data(), meta_len);

  // Other fields default to 0 (parent, children, etc.)
  node.child_count = 0;
  node.first_child_offset = 0;
  node.parent_offset = 0;

  delta_buffer_.push_back(node);
  return new_id;
}

size_t Atlas::prune_delta_tail(size_t n) {
  std::unique_lock lock(delta_mutex_);
  size_t to_remove = std::min(n, delta_buffer_.size());
  if (to_remove > 0) {
    // Vector resize removes from the end
    delta_buffer_.resize(delta_buffer_.size() - to_remove);
  }
  return to_remove;
}

uint64_t Atlas::insert(uint64_t parent_id, std::span<const float> vector,
                       std::string_view metadata) {
  // Serialize all mmap-mutating operations (exclusive lock)
  std::unique_lock<std::shared_mutex> write_lock(write_mutex_);

  auto *header = file_.get_header();

  // Check capacity
  if (header->node_count >= header->capacity) {
    // Grow by 50% or +100
    size_t new_cap = header->capacity * 1.5;
    if (new_cap < header->capacity + 100)
      new_cap = header->capacity + 100;

    if (!file_.grow(new_cap)) {
      throw std::runtime_error("Failed to grow Atlas file");
    }
    // After grow, old mmap is retired — advance epoch for reclamation
    epoch_mgr_.advance_epoch();
    // reload header after remap
    header = file_.get_header();
  }

  uint64_t new_id = header->node_count; // Simple sequential ID
  uint64_t new_idx = header->node_count;

  // Get pointer to new node
  Node *node = file_.get_node(new_idx);

  // Initialize Node
  node->id = new_id;
  node->child_count = 0;
  node->first_child_offset = 0;
  node->parent_offset = 0; // Fix below
  node->flags = 0;
  std::memset(node->reserved, 0, sizeof(node->reserved));

  // Copy Vector
  if (vector.size() == EMBEDDING_DIM) {
    std::copy(vector.begin(), vector.end(), std::begin(node->centroid));
  } else {
    throw std::invalid_argument(
        "Vector dimension mismatch: expected EMBEDDING_DIM");
  }

  // Copy Metadata
  std::memset(node->metadata, 0, sizeof(node->metadata));
  size_t meta_len = std::min(metadata.size(), sizeof(node->metadata) - 1);
  std::memcpy(node->metadata, metadata.data(), meta_len);

  // Link to Parent
  if (new_idx > 0) {
    Node *parent = file_.get_node(parent_id);
    if (!parent) {
      // Fallback: treat as independent root (should not happen if logic is
      // strict)
      node->parent_offset = 0;
    } else {
      /// Compute absolute byte offsets from file base for parent linkage.

      uint64_t parent_abs_offset = (uint8_t *)parent - (uint8_t *)header;
      uint64_t node_abs_offset = (uint8_t *)node - (uint8_t *)header;

      node->parent_offset = parent_abs_offset;

      /// Update parent-child linkage.
      /// The navigate() traversal requires children to be contiguous in the
      /// mmap region. Contiguity is enforced here: if the new node is not
      /// adjacent to the parent's existing children, the insertion is
      /// silently accepted but the node will not be navigable from that parent.
      /// This constraint arises from the append-only, page-clustered layout.
      bool is_contiguous = false;
      if (parent->child_count == 0) {
        parent->first_child_offset = node_abs_offset;
        is_contiguous = true;
      } else {
        uint64_t expected_addr =
            parent->first_child_offset + (parent->child_count * sizeof(Node));
        if (expected_addr == node_abs_offset) {
          is_contiguous = true;
        }
      }

      if (is_contiguous) {
        parent->child_count++;
      } else {
        /// Non-contiguous insertion: node is stored but not navigable from
        /// this parent. A future compaction pass can resolve fragmentation.
      }
    }
  }

  // Commit
  header->node_count++;

  return new_id;
}

void Atlas::load_context(std::span<const uint64_t> node_ids) {
  for (uint64_t id : node_ids) {
    // Only load MMap nodes (MSB 0)
    if ((id & 0x8000000000000000ULL) == 0) {
      if (auto *node = file_.get_node(id)) {
        slb_cache_.insert(id, std::span<const float>(node->centroid, 768));
      }
    }
  }
}

} // namespace aeon
