#include "aeon/atlas.hpp"
#include "aeon/hash.hpp"
#include "aeon/math_kernel.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

#if defined(__SSE__) || defined(__x86_64__) || defined(_M_X64)
#include <xmmintrin.h>
#endif

namespace aeon {

// ═══════════════════════════════════════════════════════════════════════════
// Construction / Destruction
// ═══════════════════════════════════════════════════════════════════════════

Atlas::Atlas(std::filesystem::path path, uint32_t dim)
    : Atlas(std::move(path), AtlasOptions{.dim = dim}) {}

Atlas::Atlas(std::filesystem::path path, AtlasOptions opts)
    : atlas_path_(std::move(path)), enable_wal_(opts.enable_wal) {
  // Resolve effective dim: 0 means "use file's dim or default"
  uint32_t effective_dim = (opts.dim == 0) ? EMBEDDING_DIM_DEFAULT : opts.dim;

  // Determine generation from existing files
  // Look for atlas_genN.bin pattern, or use path directly
  generation_ = 0;

  file_ = std::make_unique<storage::MemoryFile>();
  file_->set_epoch_manager(&epoch_mgr_);

  auto result =
      file_->open(atlas_path_, /*initial_capacity=*/1000, effective_dim,
                  METADATA_SIZE_DEFAULT, opts.quantization_type);
  if (!result) {
    throw std::runtime_error("Failed to open Atlas storage");
  }

  // Read authoritative layout from the on-disk header
  auto *header = file_->get_header();
  dim_ = header->dim;
  metadata_size_ = header->metadata_size;
  node_byte_stride_ = header->node_byte_stride;
  quantization_type_ = header->quantization_type; // V4.1 Phase 3

  // Pre-allocate delta arena for ~10,000 nodes worth of contiguous memory
  delta_buffer_bytes_.reserve(10000 * node_byte_stride_);

  // ── V4.1 WAL: crash recovery (optional) ──
  wal_path_ = atlas_path_;
  wal_path_ += ".wal";
  if (enable_wal_) {
    replay_wal();
    open_wal();
  }
}

Atlas::~Atlas() = default;

EpochGuard Atlas::acquire_read_guard() { return epoch_mgr_.enter_guard(); }

size_t Atlas::size() const {
  if (auto *header = file_->get_header()) {
    return header->node_count;
  }
  return 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// Delta Buffer — Flat Byte Arena Helpers
// ═══════════════════════════════════════════════════════════════════════════

size_t Atlas::delta_node_count() const noexcept {
  return (node_byte_stride_ > 0)
             ? delta_buffer_bytes_.size() / node_byte_stride_
             : 0;
}

size_t
Atlas::delta_node_count(const std::vector<uint8_t> &arena) const noexcept {
  return (node_byte_stride_ > 0) ? arena.size() / node_byte_stride_ : 0;
}

NodeHeader *Atlas::delta_get_node(size_t index) noexcept {
  size_t offset = index * node_byte_stride_;
  if (offset + node_byte_stride_ > delta_buffer_bytes_.size())
    return nullptr;
  return reinterpret_cast<NodeHeader *>(delta_buffer_bytes_.data() + offset);
}

const NodeHeader *Atlas::delta_get_node(size_t index) const noexcept {
  size_t offset = index * node_byte_stride_;
  if (offset + node_byte_stride_ > delta_buffer_bytes_.size())
    return nullptr;
  return reinterpret_cast<const NodeHeader *>(delta_buffer_bytes_.data() +
                                              offset);
}

const NodeHeader *Atlas::delta_get_node(const std::vector<uint8_t> &arena,
                                        size_t index) const noexcept {
  size_t offset = index * node_byte_stride_;
  if (offset + node_byte_stride_ > arena.size())
    return nullptr;
  return reinterpret_cast<const NodeHeader *>(arena.data() + offset);
}

// ═══════════════════════════════════════════════════════════════════════════
// navigate() — public entry point
// ═══════════════════════════════════════════════════════════════════════════

std::vector<Atlas::ResultNode> Atlas::navigate(std::span<const float> query,
                                               uint32_t beam_width,
                                               bool apply_csls) {
  if (query.size() != dim_)
    return {};

  beam_width = std::clamp(beam_width, uint32_t{1}, MAX_BEAM_WIDTH);

  if (apply_csls) {
    return navigate_internal<true>(query, beam_width);
  } else {
    return navigate_internal<false>(query, beam_width);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// navigate_internal<ApplyCSLS>() — zero-allocation beam search
// ═══════════════════════════════════════════════════════════════════════════

struct BeamCandidate {
  uint64_t node_idx;
  float score;
};

template <bool ApplyCSLS>
std::vector<Atlas::ResultNode>
Atlas::navigate_internal(std::span<const float> query, uint32_t beam_width) {
  std::vector<Atlas::ResultNode> path;

  // Fast-path: SLB cache hit
  if (auto hit = slb_cache_.find_nearest(query, SLB_HIT_THRESHOLD)) {
    return {{hit->node_id,
             hit->similarity,
             {hit->centroid_preview[0], hit->centroid_preview[1],
              hit->centroid_preview[2]}}};
  }

  // ── V4.1 Phase 3: Quantize query ONCE if atlas is INT8 ──
  const bool is_int8 = (quantization_type_ == QUANT_INT8_SYMMETRIC);
  std::vector<int8_t> query_q;
  float query_scale = 0.0f;
  static const auto int8_dot_fn = simd::get_best_int8_dot_impl();

  if (is_int8) {
    query_q.resize(dim_);
    quant::quantize_symmetric(query, query_q, query_scale);
  }

  // Acquire epoch guard before ANY mmap pointer access
  auto guard = epoch_mgr_.enter_guard();
  std::shared_lock<std::shared_mutex> read_lock(write_mutex_);

  auto *header = file_->get_header();

  // ── Beam search through memory-mapped B+ tree (immutable layer) ──
  if (header && header->node_count > 0) {
    std::array<BeamCandidate, MAX_BEAM_WIDTH> beam{};
    uint32_t beam_size = 0;

    std::array<BeamCandidate, MAX_BEAM_WIDTH> next_beam{};
    uint32_t next_beam_size = 0;

    // Seed beam with root node
    NodeHeader *root = file_->get_node(0);
    if (!root)
      return path;

    float root_score;
    if (is_int8) {
      // INT8 path: dot product + dequantize
      const int8_t *root_q = node_centroid_int8(root);
      int32_t raw_dot =
          int8_dot_fn(query_q, std::span<const int8_t>(root_q, dim_), dim_);
      root_score = quant::dequantize_dot_product(raw_dot, query_scale,
                                                 root->quant_scale);
    } else {
      const float *root_centroid = node_centroid(root);
      root_score = math::cosine_similarity(
          query, std::span<const float>(root_centroid, dim_));
    }
    if constexpr (ApplyCSLS) {
      root_score -= root->hub_penalty;
    }

    beam[0] = {0, root_score};
    beam_size = 1;

    BeamCandidate overall_best = beam[0];

    // Preview: use FP32 centroid for preview even in INT8 mode (first 3 dims)
    const float *root_preview = is_int8 ? nullptr : node_centroid(root);
    float p0 = root_preview ? root_preview[0] : 0.0f;
    float p1 = root_preview ? root_preview[1] : 0.0f;
    float p2 = root_preview ? root_preview[2] : 0.0f;
    path.push_back({root->id, root_score, {p0, p1, p2}});

    // Beam descent loop
    bool has_children = true;
    while (has_children) {
      has_children = false;
      next_beam_size = 0;

      for (uint32_t b = 0; b < beam_size; ++b) {
        NodeHeader *current = file_->get_node(beam[b].node_idx);
        if (!current || current->child_count == 0 ||
            current->first_child_offset == 0)
          continue;

        has_children = true;

        // Children are contiguous in the byte stream at first_child_offset
        // Each child is node_byte_stride_ bytes apart
        uint8_t *file_base = reinterpret_cast<uint8_t *>(header);
        uint8_t *child_base = file_base + current->first_child_offset;

        for (uint16_t i = 0; i < current->child_count; ++i) {
          NodeHeader *child = reinterpret_cast<NodeHeader *>(
              child_base + i * node_byte_stride_);

          // Prefetch next child's centroid into L1
          if (i + 1 < current->child_count) {
            auto *next_hdr = reinterpret_cast<NodeHeader *>(
                child_base + (i + 1) * node_byte_stride_);
            const void *next_data =
                reinterpret_cast<const uint8_t *>(next_hdr) +
                sizeof(NodeHeader);
#if defined(__SSE__) || defined(__x86_64__) || defined(_M_X64)
            _mm_prefetch((const char *)next_data, _MM_HINT_T0);
#else
            __builtin_prefetch(next_data, 0, 3);
#endif
          }

          float score;
          if (is_int8) {
            const int8_t *child_q = node_centroid_int8(child);
            int32_t raw_dot = int8_dot_fn(
                query_q, std::span<const int8_t>(child_q, dim_), dim_);
            // CRITICAL: dequantize BEFORE hub_penalty subtraction
            score = quant::dequantize_dot_product(raw_dot, query_scale,
                                                  child->quant_scale);
          } else {
            const float *child_centroid = node_centroid(child);
            score = math::cosine_similarity(
                query, std::span<const float>(child_centroid, dim_));
          }

          if constexpr (ApplyCSLS) {
            score -= child->hub_penalty;
          }

          // Compute child's node index from byte offset
          uint64_t child_offset = reinterpret_cast<uint8_t *>(child) -
                                  file_base - sizeof(AtlasHeader);
          uint64_t child_node_idx = child_offset / node_byte_stride_;

          if (next_beam_size < beam_width) {
            next_beam[next_beam_size] = {child_node_idx, score};
            ++next_beam_size;
            if (score > overall_best.score) {
              overall_best = next_beam[next_beam_size - 1];
            }
          } else {
            uint32_t worst_idx = 0;
            float worst_score = next_beam[0].score;
            for (uint32_t k = 1; k < next_beam_size; ++k) {
              if (next_beam[k].score < worst_score) {
                worst_score = next_beam[k].score;
                worst_idx = k;
              }
            }
            if (score > worst_score) {
              next_beam[worst_idx] = {child_node_idx, score};
              if (score > overall_best.score) {
                overall_best = next_beam[worst_idx];
              }
            }
          }
        }
      }

      if (!has_children || next_beam_size == 0)
        break;

      beam = next_beam;
      beam_size = next_beam_size;

      uint32_t best_in_beam = 0;
      for (uint32_t k = 1; k < beam_size; ++k) {
        if (beam[k].score > beam[best_in_beam].score)
          best_in_beam = k;
      }

      NodeHeader *best_node = file_->get_node(beam[best_in_beam].node_idx);
      if (best_node) {
        float bp0 = 0.0f, bp1 = 0.0f, bp2 = 0.0f;
        if (!is_int8) {
          const float *bc = node_centroid(best_node);
          bp0 = bc[0];
          bp1 = bc[1];
          bp2 = bc[2];
        }
        path.push_back(
            {best_node->id, beam[best_in_beam].score, {bp0, bp1, bp2}});
      }
    }

    // Populate SLB with overall best (mmap-backed nodes only, MSB=0)
    // SLB stores exclusively FP32 vectors. For INT8 nodes, we dequantize
    // on-the-fly to FP32 before insertion. This keeps HierarchicalSLB
    // completely unaware of quantization.
    NodeHeader *best = file_->get_node(overall_best.node_idx);
    if (best && (best->id & 0x8000000000000000ULL) == 0) {
      if (is_int8) {
        // Dequantize INT8 → FP32 on-the-fly for SLB insertion
        const int8_t *q_vec = node_centroid_int8(best);
        std::vector<float> fp32_vec(dim_);
        float s = best->quant_scale;
        for (uint32_t d = 0; d < dim_; ++d) {
          fp32_vec[d] = static_cast<float>(q_vec[d]) * s;
        }
        slb_cache_.insert(best->id,
                          std::span<const float>(fp32_vec.data(), dim_));
      } else {
        slb_cache_.insert(best->id,
                          std::span<const float>(node_centroid(best), dim_));
      }
    }
  }

  // ── Linear scan of flat byte arena delta buffer (mutable layer) ──
  std::vector<ResultNode> delta_candidates;
  {
    std::shared_lock lock(delta_mutex_);
    size_t count = delta_node_count();
    delta_candidates.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      const NodeHeader *dnode = delta_get_node(i);
      if (!dnode)
        continue;

      float score;
      if (is_int8) {
        const int8_t *dq = node_centroid_int8(dnode);
        int32_t raw_dot =
            int8_dot_fn(query_q, std::span<const int8_t>(dq, dim_), dim_);
        score = quant::dequantize_dot_product(raw_dot, query_scale,
                                              dnode->quant_scale);
      } else {
        const float *dc = node_centroid(dnode);
        score =
            math::cosine_similarity(query, std::span<const float>(dc, dim_));
      }

      if constexpr (ApplyCSLS) {
        score -= dnode->hub_penalty;
      }

      float dp0 = 0.0f, dp1 = 0.0f, dp2 = 0.0f;
      if (!is_int8) {
        const float *dc = node_centroid(dnode);
        dp0 = dc[0];
        dp1 = dc[1];
        dp2 = dc[2];
      }
      delta_candidates.push_back({dnode->id, score, {dp0, dp1, dp2}});
    }
  }

  // Merge + sort + cap
  path.insert(path.end(), delta_candidates.begin(), delta_candidates.end());
  std::sort(path.begin(), path.end(),
            [](const ResultNode &a, const ResultNode &b) {
              return a.similarity > b.similarity;
            });
  if (path.size() > TOP_K_LIMIT) {
    path.resize(TOP_K_LIMIT);
  }

  return path;
}

// Explicit template instantiation
template std::vector<Atlas::ResultNode>
Atlas::navigate_internal<false>(std::span<const float>, uint32_t);
template std::vector<Atlas::ResultNode>
Atlas::navigate_internal<true>(std::span<const float>, uint32_t);

// ═══════════════════════════════════════════════════════════════════════════
// get_children, insert, insert_delta, prune_delta_tail, load_context
// ═══════════════════════════════════════════════════════════════════════════

std::vector<Atlas::ResultNode> Atlas::get_children(uint64_t parent_id) {
  std::vector<Atlas::ResultNode> children;

  auto guard = epoch_mgr_.enter_guard();
  std::shared_lock<std::shared_mutex> read_lock(write_mutex_);

  auto *header = file_->get_header();
  if (!header || parent_id >= header->node_count) {
    return children;
  }

  NodeHeader *parent = file_->get_node(parent_id);
  if (!parent || parent->child_count == 0 || parent->first_child_offset == 0) {
    return children;
  }

  uint8_t *file_base = reinterpret_cast<uint8_t *>(header);
  uint8_t *child_base = file_base + parent->first_child_offset;

  children.reserve(parent->child_count);
  for (uint16_t i = 0; i < parent->child_count; ++i) {
    NodeHeader *child =
        reinterpret_cast<NodeHeader *>(child_base + i * node_byte_stride_);

    // Bounds check
    if (reinterpret_cast<uint8_t *>(child) >=
        file_base + sizeof(AtlasHeader) +
            (header->capacity * node_byte_stride_))
      break;

    const float *cc = node_centroid(child);
    children.push_back({child->id, 0.0f, {cc[0], cc[1], cc[2]}});
  }

  return children;
}

uint64_t Atlas::insert_delta(std::span<const float> vector,
                             std::string_view metadata) {
  // ── Step 1: Serialize data & compute checksum (NO LOCK) ──
  // Build the node payload into a temporary buffer on the stack/heap
  // so that all expensive work happens without holding any lock.
  std::vector<uint8_t> payload(node_byte_stride_, 0);

  auto *hdr = reinterpret_cast<NodeHeader *>(payload.data());
  // ID will be assigned under delta_mutex_ (needs delta_node_count)
  hdr->parent_offset = 0;
  hdr->first_child_offset = 0;
  hdr->child_count = 0;
  hdr->flags = 0;
  hdr->hub_penalty = 0.0f;
  std::memset(hdr->reserved, 0, sizeof(hdr->reserved));

  // Copy centroid — quantize if INT8
  if (quantization_type_ == QUANT_INT8_SYMMETRIC) {
    int8_t *centroid_q = node_centroid_int8(hdr);
    if (vector.size() == dim_) {
      float scale;
      quant::quantize_symmetric(vector, std::span<int8_t>(centroid_q, dim_),
                                scale);
      hdr->quant_scale = scale;
      hdr->quant_zero_point = 0.0f;
    } else {
      std::memset(centroid_q, 0, dim_ * sizeof(int8_t));
      hdr->quant_scale = 1.0f;
      hdr->quant_zero_point = 0.0f;
    }
  } else {
    float *centroid = node_centroid(hdr);
    if (vector.size() == dim_) {
      std::memcpy(centroid, vector.data(), dim_ * sizeof(float));
    } else {
      std::memset(centroid, 0, dim_ * sizeof(float));
    }
    hdr->quant_scale = 0.0f;
    hdr->quant_zero_point = 0.0f;
  }

  // Copy metadata (uses quant-aware accessor)
  char *meta = node_metadata_q(hdr, dim_, quantization_type_);
  std::memset(meta, 0, metadata_size_);
  size_t meta_len =
      std::min(metadata.size(), static_cast<size_t>(metadata_size_ - 1));
  std::memcpy(meta, metadata.data(), meta_len);

  // Compute FNV-1a checksum of the full payload
  uint64_t checksum =
      hash::fnv1a_64(payload.data(), static_cast<size_t>(node_byte_stride_));

  // ── Step 2: lock(wal_mutex_) → WAL write + flush → unlock ──
  if (enable_wal_) {
    std::lock_guard<std::mutex> wal_lock(wal_mutex_);
    if (wal_stream_.is_open()) {
      WalRecordHeader wal_hdr{};
      wal_hdr.record_type = WAL_RECORD_ATLAS;
      wal_hdr.payload_size = static_cast<uint32_t>(node_byte_stride_);
      wal_hdr.checksum = checksum;

      wal_stream_.write(reinterpret_cast<const char *>(&wal_hdr),
                        sizeof(WalRecordHeader));
      wal_stream_.write(reinterpret_cast<const char *>(payload.data()),
                        static_cast<std::streamsize>(node_byte_stride_));
      wal_stream_.flush();
    }
  }

  // ── Step 3: lock(delta_mutex_) → append to RAM buffer → unlock ──
  std::unique_lock lock(delta_mutex_);

  static const uint64_t DELTA_MASK = 0x8000000000000000ULL;
  uint64_t new_id = DELTA_MASK | delta_node_count();

  // Set the ID now that we know the position
  hdr->id = new_id;

  // Extend the flat byte arena and copy the payload in
  size_t old_size = delta_buffer_bytes_.size();
  delta_buffer_bytes_.resize(old_size + node_byte_stride_, 0);
  std::memcpy(delta_buffer_bytes_.data() + old_size, payload.data(),
              node_byte_stride_);

  return new_id;
}

size_t Atlas::prune_delta_tail(size_t n) {
  std::unique_lock lock(delta_mutex_);
  size_t count = delta_node_count();
  size_t to_remove = std::min(n, count);
  if (to_remove > 0) {
    delta_buffer_bytes_.resize(delta_buffer_bytes_.size() -
                               to_remove * node_byte_stride_);
  }
  return to_remove;
}

uint64_t Atlas::insert(uint64_t parent_id, std::span<const float> vector,
                       std::string_view metadata) {
  // ── WRITE DIVERSION: if background compaction in progress, divert to delta
  // ──
  if (compact_in_progress_.load(std::memory_order_acquire)) {
    return insert_delta(vector, metadata);
  }

  // Serialize all mmap-mutating operations
  std::unique_lock<std::shared_mutex> write_lock(write_mutex_);

  auto *header = file_->get_header();

  // Check capacity — grow if needed
  if (header->node_count >= header->capacity) {
    size_t new_cap = header->capacity * 1.5;
    if (new_cap < header->capacity + 100)
      new_cap = header->capacity + 100;

    if (!file_->grow(new_cap)) {
      throw std::runtime_error("Failed to grow Atlas file");
    }
    epoch_mgr_.advance_epoch();
    header = file_->get_header();
  }

  uint64_t new_id = header->node_count;
  uint64_t new_idx = header->node_count;

  NodeHeader *node = file_->get_node(new_idx);

  // Initialize header
  node->id = new_id;
  node->child_count = 0;
  node->first_child_offset = 0;
  node->parent_offset = 0;
  node->flags = 0;
  node->hub_penalty = 0.0f;
  std::memset(node->reserved, 0, sizeof(node->reserved));

  // Copy vector — quantize if INT8
  if (vector.size() != dim_) {
    throw std::invalid_argument("Vector dimension mismatch: expected " +
                                std::to_string(dim_));
  }

  if (quantization_type_ == QUANT_INT8_SYMMETRIC) {
    int8_t *centroid_q = node_centroid_int8(node);
    float scale;
    quant::quantize_symmetric(vector, std::span<int8_t>(centroid_q, dim_),
                              scale);
    node->quant_scale = scale;
    node->quant_zero_point = 0.0f;
  } else {
    std::memcpy(node_centroid(node), vector.data(), dim_ * sizeof(float));
    node->quant_scale = 0.0f;
    node->quant_zero_point = 0.0f;
  }

  // Copy metadata
  char *meta = node_metadata_q(node, dim_, quantization_type_);
  std::memset(meta, 0, metadata_size_);
  size_t meta_len =
      std::min(metadata.size(), static_cast<size_t>(metadata_size_ - 1));
  std::memcpy(meta, metadata.data(), meta_len);

  // Link to parent
  if (new_idx > 0) {
    NodeHeader *parent = file_->get_node(parent_id);
    if (!parent) {
      node->parent_offset = 0;
    } else {
      uint64_t parent_abs_offset = reinterpret_cast<uint8_t *>(parent) -
                                   reinterpret_cast<uint8_t *>(header);
      uint64_t node_abs_offset = reinterpret_cast<uint8_t *>(node) -
                                 reinterpret_cast<uint8_t *>(header);

      node->parent_offset = parent_abs_offset;

      bool is_contiguous = false;
      if (parent->child_count == 0) {
        parent->first_child_offset = node_abs_offset;
        is_contiguous = true;
      } else {
        uint64_t expected_addr = parent->first_child_offset +
                                 (parent->child_count * node_byte_stride_);
        if (expected_addr == node_abs_offset) {
          is_contiguous = true;
        }
      }

      if (is_contiguous) {
        parent->child_count++;
      }
    }
  }

  header->node_count++;
  return new_id;
}

void Atlas::load_context(std::span<const uint64_t> node_ids) {
  for (uint64_t id : node_ids) {
    if ((id & 0x8000000000000000ULL) == 0) {
      if (auto *node = file_->get_node(id)) {
        slb_cache_.insert(id,
                          std::span<const float>(node_centroid(node), dim_));
      }
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Dreaming Kernel — consolidate_subgraph
// ═══════════════════════════════════════════════════════════════════════════

uint64_t Atlas::consolidate_subgraph(std::span<const uint64_t> old_node_ids,
                                     std::span<const float> summary_vector,
                                     std::string_view summary_meta) {
  if (old_node_ids.empty()) {
    throw std::invalid_argument("consolidate_subgraph: empty old_node_ids");
  }
  if (summary_vector.size() != dim_) {
    throw std::invalid_argument(
        "consolidate_subgraph: summary_vector must match Atlas dim (" +
        std::to_string(dim_) + ")");
  }
  // Consolidation during compaction is unsafe (mmap is being rewritten)
  if (compact_in_progress_.load(std::memory_order_acquire)) {
    throw std::runtime_error("consolidate_subgraph: cannot consolidate while "
                             "compaction in progress");
  }

  std::unique_lock<std::shared_mutex> write_lock(write_mutex_);

  auto *header = file_->get_header();
  if (!header) {
    throw std::runtime_error("consolidate_subgraph: Atlas file not open");
  }

  // Phase 1: Validate
  for (uint64_t id : old_node_ids) {
    if (id >= header->node_count) {
      throw std::runtime_error("consolidate_subgraph: invalid node id " +
                               std::to_string(id));
    }
    NodeHeader *node = file_->get_node(id);
    if (!node) {
      throw std::runtime_error("consolidate_subgraph: null node at id " +
                               std::to_string(id));
    }
    if (is_tombstoned(*node)) {
      throw std::runtime_error(
          "consolidate_subgraph: node already tombstoned: " +
          std::to_string(id));
    }
  }

  // Phase 2: Insert summary node
  NodeHeader *first_old = file_->get_node(old_node_ids[0]);
  uint64_t summary_parent_offset = first_old->parent_offset;

  if (header->node_count >= header->capacity) {
    size_t new_cap = header->capacity * 1.5;
    if (new_cap < header->capacity + 100)
      new_cap = header->capacity + 100;
    if (!file_->grow(new_cap)) {
      throw std::runtime_error("consolidate_subgraph: failed to grow Atlas");
    }
    epoch_mgr_.advance_epoch();
    header = file_->get_header();
  }

  uint64_t summary_id = header->node_count;
  NodeHeader *summary = file_->get_node(summary_id);

  summary->id = summary_id;
  summary->parent_offset = summary_parent_offset;
  summary->first_child_offset = 0;
  summary->child_count = 0;
  summary->flags = NODE_FLAG_SUMMARY;
  summary->hub_penalty = 0.0f;
  std::memset(summary->reserved, 0, sizeof(summary->reserved));

  std::memcpy(node_centroid(summary), summary_vector.data(),
              dim_ * sizeof(float));

  char *meta = node_metadata(summary, dim_);
  std::memset(meta, 0, metadata_size_);
  size_t meta_len =
      std::min(summary_meta.size(), static_cast<size_t>(metadata_size_ - 1));
  std::memcpy(meta, summary_meta.data(), meta_len);

  header->node_count++;

  // Phase 3: Re-wire children → summary
  uint8_t *file_base = reinterpret_cast<uint8_t *>(header);
  uint64_t summary_abs_offset =
      reinterpret_cast<uint8_t *>(summary) - file_base;

  for (uint64_t id : old_node_ids) {
    NodeHeader *old_node = file_->get_node(id);
    if (!old_node || old_node->child_count == 0 ||
        old_node->first_child_offset == 0)
      continue;

    uint8_t *child_base = file_base + old_node->first_child_offset;

    for (uint16_t i = 0; i < old_node->child_count; ++i) {
      NodeHeader *child =
          reinterpret_cast<NodeHeader *>(child_base + i * node_byte_stride_);

      // Skip children being tombstoned in this batch
      bool skip = false;
      for (uint64_t dead_id : old_node_ids) {
        if (child->id == dead_id) {
          skip = true;
          break;
        }
      }
      if (!skip) {
        child->parent_offset = summary_abs_offset;
      }
    }
  }

  // Phase 4: Tombstone
  for (uint64_t id : old_node_ids) {
    NodeHeader *old_node = file_->get_node(id);
    if (old_node) {
      tombstone_node(*old_node);
    }
  }

  epoch_mgr_.advance_epoch();
  return summary_id;
}

// ═══════════════════════════════════════════════════════════════════════════
// compact_mmap() — V4.0 Background Shadow Compaction
// ═══════════════════════════════════════════════════════════════════════════
//
// Uses the Redis BGSAVE double-buffer pattern with:
//   Correction #1: Flat byte arena (no heap pointer chasing)
//   Correction #2: Write Diversion (insert() → insert_delta() during compact)
//   Correction #3: Non-blocking swap (unique_ptr<MemoryFile>)
//   Correction #4: Generational naming (Windows MapViewOfFile safety)

void Atlas::compact_mmap() {
  // Prevent concurrent compactions
  bool expected = false;
  if (!compact_in_progress_.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel)) {
    return; // Another compaction is already running
  }

  // ── Step 1: Microsecond Freeze ──
  // Acquire exclusive lock. Move delta buffer into frozen snapshot.
  // Capture current mmap node_count. Release immediately.
  uint64_t snapshot_node_count = 0;
  {
    std::unique_lock<std::shared_mutex> write_lock(write_mutex_);
    std::unique_lock<std::shared_mutex> delta_lock(delta_mutex_);

    // Swap: active delta → frozen, leave empty active for new inserts
    frozen_delta_buffer_bytes_ = std::move(delta_buffer_bytes_);
    delta_buffer_bytes_.clear();
    delta_buffer_bytes_.reserve(1000 * node_byte_stride_);

    auto *header = file_->get_header();
    if (!header || header->node_count == 0) {
      compact_in_progress_.store(false, std::memory_order_release);
      return;
    }
    snapshot_node_count = header->node_count;
  }
  // ── Lock released. Game engine can now insert_delta() freely. ──

  // ── Step 2: Background Copy (NO exclusive lock, just EpochGuard) ──
  // Copy live mmap nodes + frozen deltas → new generation file.
  auto guard = epoch_mgr_.enter_guard();

  auto *header = file_->get_header();
  (void)header; // Reserved for future use

  // Count live mmap nodes
  size_t live_mmap_count = 0;
  for (size_t i = 0; i < snapshot_node_count; ++i) {
    const NodeHeader *node = file_->get_node(i);
    if (node && !is_tombstoned(*node)) {
      ++live_mmap_count;
    }
  }

  size_t frozen_count = delta_node_count(frozen_delta_buffer_bytes_);
  size_t total_live = live_mmap_count + frozen_count;

  if (total_live == snapshot_node_count && frozen_count == 0) {
    // Nothing to compact
    compact_in_progress_.store(false, std::memory_order_release);
    return;
  }

  // Generational file naming (Windows MapViewOfFile safety)
  uint64_t new_gen = generation_ + 1;
  std::filesystem::path new_path =
      atlas_path_.parent_path() /
      (atlas_path_.stem().string() + "_gen" + std::to_string(new_gen) + ".bin");

  // Create new generation file
  size_t new_file_size = sizeof(AtlasHeader) + (total_live * node_byte_stride_);

  platform::FileHandle new_handle = platform::file_open(
#if defined(AEON_PLATFORM_WINDOWS)
      new_path.string().c_str()
#else
      new_path.c_str(), 0644
#endif
  );

  if (new_handle == platform::INVALID_FILE_HANDLE) {
    compact_in_progress_.store(false, std::memory_order_release);
    throw std::runtime_error("compact_mmap: failed to create generation file");
  }

  if (!platform::file_resize(new_handle, new_file_size)) {
    platform::file_close(new_handle);
    compact_in_progress_.store(false, std::memory_order_release);
    throw std::runtime_error("compact_mmap: failed to resize generation file");
  }

  void *new_raw = platform::mem_map(new_handle, new_file_size);
  if (new_raw == platform::MAP_FAILED_PTR) {
    platform::file_close(new_handle);
    compact_in_progress_.store(false, std::memory_order_release);
    throw std::runtime_error("compact_mmap: failed to mmap generation file");
  }

  auto *new_data = static_cast<uint8_t *>(new_raw);

  // Write header
  auto *new_header = reinterpret_cast<AtlasHeader *>(new_data);
  new_header->magic = ATLAS_MAGIC;
  new_header->version = ATLAS_VERSION;
  new_header->node_count = total_live;
  new_header->capacity = total_live;
  new_header->dim = dim_;
  new_header->metadata_size = metadata_size_;
  new_header->node_byte_stride = node_byte_stride_;
  std::fill(std::begin(new_header->reserved), std::end(new_header->reserved),
            0);

  // Build old→new mapping and copy live mmap nodes
  std::vector<uint64_t> old_to_new(snapshot_node_count, UINT64_MAX);
  size_t new_idx = 0;

  for (size_t i = 0; i < snapshot_node_count; ++i) {
    const NodeHeader *old_node = file_->get_node(i);
    if (!old_node || is_tombstoned(*old_node))
      continue;

    old_to_new[i] = new_idx;

    // Copy entire node stride (header + centroid + metadata + padding)
    uint8_t *dst =
        new_data + sizeof(AtlasHeader) + (new_idx * node_byte_stride_);
    std::memcpy(dst, old_node, node_byte_stride_);

    // Assign new sequential ID
    auto *dst_hdr = reinterpret_cast<NodeHeader *>(dst);
    dst_hdr->id = new_idx;

    ++new_idx;
  }

  // Append frozen delta nodes (promoted from delta to mmap layer)
  for (size_t i = 0; i < frozen_count; ++i) {
    const NodeHeader *delta_node =
        delta_get_node(frozen_delta_buffer_bytes_, i);
    if (!delta_node)
      continue;

    uint8_t *dst =
        new_data + sizeof(AtlasHeader) + (new_idx * node_byte_stride_);
    std::memcpy(dst, delta_node, node_byte_stride_);

    auto *dst_hdr = reinterpret_cast<NodeHeader *>(dst);
    dst_hdr->id = new_idx; // Promote: replace delta ID with sequential mmap ID

    // Clear parent/child linkage for promoted deltas (they were unlinked)
    dst_hdr->parent_offset = 0;
    dst_hdr->first_child_offset = 0;
    dst_hdr->child_count = 0;

    ++new_idx;
  }

  // Re-index byte offsets for mmap-origin nodes
  for (size_t i = 0; i < live_mmap_count; ++i) {
    NodeHeader *node = reinterpret_cast<NodeHeader *>(
        new_data + sizeof(AtlasHeader) + (i * node_byte_stride_));

    // Re-index parent_offset
    if (node->parent_offset != 0) {
      uint64_t old_parent_idx =
          (node->parent_offset - sizeof(AtlasHeader)) / node_byte_stride_;
      if (old_parent_idx < snapshot_node_count &&
          old_to_new[old_parent_idx] != UINT64_MAX) {
        node->parent_offset = sizeof(AtlasHeader) +
                              old_to_new[old_parent_idx] * node_byte_stride_;
      } else {
        node->parent_offset = 0;
      }
    }

    // Re-index first_child_offset
    if (node->first_child_offset != 0 && node->child_count > 0) {
      uint64_t old_first_child_idx =
          (node->first_child_offset - sizeof(AtlasHeader)) / node_byte_stride_;

      uint64_t new_first_child = UINT64_MAX;
      uint16_t new_child_count = 0;

      for (uint16_t c = 0; c < node->child_count; ++c) {
        uint64_t old_child_idx = old_first_child_idx + c;
        if (old_child_idx < snapshot_node_count &&
            old_to_new[old_child_idx] != UINT64_MAX) {
          if (new_first_child == UINT64_MAX) {
            new_first_child = old_to_new[old_child_idx];
          }
          if (new_first_child != UINT64_MAX &&
              old_to_new[old_child_idx] == new_first_child + new_child_count) {
            ++new_child_count;
          }
        }
      }

      if (new_first_child != UINT64_MAX && new_child_count > 0) {
        node->first_child_offset =
            sizeof(AtlasHeader) + new_first_child * node_byte_stride_;
        node->child_count = new_child_count;
      } else {
        node->first_child_offset = 0;
        node->child_count = 0;
      }
    }
  }

  // Unmap the new file (we'll reopen it via MemoryFile)
  platform::mem_unmap(new_raw, new_file_size);
  platform::file_close(new_handle);

  // Release the EBR guard from Step 2
  guard.release();

  // ── Step 3: Microsecond Freeze (Non-Blocking Swap) ──
  // Create new MemoryFile for the compacted generation file.
  auto new_file = std::make_unique<storage::MemoryFile>();
  new_file->set_epoch_manager(&epoch_mgr_);
  auto open_result = new_file->open(new_path, total_live, dim_, metadata_size_);
  if (!open_result) {
    compact_in_progress_.store(false, std::memory_order_release);
    throw std::runtime_error("compact_mmap: failed to reopen compacted file");
  }

  // Capture old file pointer for background cleanup
  std::unique_ptr<storage::MemoryFile> old_file;
  std::filesystem::path old_path = atlas_path_;

  {
    std::unique_lock<std::shared_mutex> write_lock(write_mutex_);

    // Pointer swap: O(1), no I/O, no drain_readers()
    old_file = std::move(file_);
    file_ = std::move(new_file);

    // Clear frozen buffer and SLB (node IDs have changed)
    frozen_delta_buffer_bytes_.clear();
    frozen_delta_buffer_bytes_.shrink_to_fit();
    slb_cache_.clear();

    // Update path and generation
    atlas_path_ = new_path;
    generation_ = new_gen;
  }
  // ── Lock released immediately. Game engine never blocked > µs. ──

  // ── Step 4: Background Cleanup ──
  // Close old file OUTSIDE the lock. drain_readers() blocks here safely
  // without freezing any game engine threads.
  compact_in_progress_.store(false, std::memory_order_release);

  old_file->close();
  old_file.reset();

  // Delete the old generation file (safe: no more MapViewOfFile handles)
  std::error_code ec;
  std::filesystem::remove(old_path, ec);
  // Silently ignore deletion errors (file may already be gone)

  // ── V4.1: Truncate WAL — all delta data is now in the compacted file ──
  if (enable_wal_) {
    truncate_wal();
    open_wal();
  }

  epoch_mgr_.advance_epoch();
}

// ═══════════════════════════════════════════════════════════════════════════
// tombstone_count
// ═══════════════════════════════════════════════════════════════════════════

size_t Atlas::tombstone_count() const {
  auto *header = file_->get_header();
  if (!header)
    return 0;

  size_t count = 0;
  for (size_t i = 0; i < header->node_count; ++i) {
    const NodeHeader *node = file_->get_node(i);
    if (node && is_tombstoned(*node)) {
      ++count;
    }
  }
  return count;
}

} // namespace aeon

// ═══════════════════════════════════════════════════════════════════════════
// WAL Methods (V4.1)
// ═══════════════════════════════════════════════════════════════════════════

namespace aeon {

void Atlas::open_wal() {
  std::lock_guard<std::mutex> lock(wal_mutex_);
  if (wal_stream_.is_open())
    wal_stream_.close();
  wal_stream_.open(wal_path_, std::ios::binary | std::ios::app);
}

void Atlas::replay_wal() {
  if (!std::filesystem::exists(wal_path_))
    return;

  auto file_size = std::filesystem::file_size(wal_path_);
  if (file_size == 0)
    return;

  std::ifstream in(wal_path_, std::ios::binary);
  if (!in.is_open())
    return;

  while (in.good() && !in.eof()) {
    // Read WAL record header
    WalRecordHeader wal_hdr{};
    in.read(reinterpret_cast<char *>(&wal_hdr), sizeof(WalRecordHeader));
    if (in.gcount() != sizeof(WalRecordHeader))
      break; // Truncated header — stop replay

    // Validate record type
    if (wal_hdr.record_type != WAL_RECORD_ATLAS)
      break; // Wrong record type — corruption

    // Validate payload size is sane (must match node_byte_stride_)
    if (wal_hdr.payload_size != node_byte_stride_)
      break; // Size mismatch — corruption

    // Read payload
    std::vector<uint8_t> payload(wal_hdr.payload_size);
    in.read(reinterpret_cast<char *>(payload.data()),
            static_cast<std::streamsize>(wal_hdr.payload_size));
    if (static_cast<uint32_t>(in.gcount()) != wal_hdr.payload_size)
      break; // Truncated payload — stop replay

    // Verify checksum
    uint64_t computed = hash::fnv1a_64(payload.data(), wal_hdr.payload_size);
    if (computed != wal_hdr.checksum)
      break; // Checksum mismatch — stop replay

    // ── Record is valid: reconstruct delta buffer ──
    // Assign a delta ID based on current buffer position
    static const uint64_t DELTA_MASK = 0x8000000000000000ULL;
    auto *hdr = reinterpret_cast<NodeHeader *>(payload.data());
    hdr->id = DELTA_MASK | delta_node_count();

    size_t old_size = delta_buffer_bytes_.size();
    delta_buffer_bytes_.resize(old_size + node_byte_stride_);
    std::memcpy(delta_buffer_bytes_.data() + old_size, payload.data(),
                node_byte_stride_);
  }
}

void Atlas::truncate_wal() {
  std::lock_guard<std::mutex> lock(wal_mutex_);
  if (wal_stream_.is_open())
    wal_stream_.close();

  std::error_code ec;
  std::filesystem::remove(wal_path_, ec);
  // Ignore errors — file may not exist
}

} // namespace aeon
