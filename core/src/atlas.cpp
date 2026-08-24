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
  // Resolve effective metadata_size: 0 means "use the default" -- same
  // convention as dim above. New files only; an existing file's on-disk
  // metadata_size (read from the header below) is authoritative regardless
  // of what's passed here.
  uint32_t effective_metadata_size =
      (opts.metadata_size == 0) ? METADATA_SIZE_DEFAULT : opts.metadata_size;

  // Determine generation from existing files
  // Look for atlas_genN.bin pattern, or use path directly
  generation_ = 0;

  file_ = std::make_unique<storage::MemoryFile>();
  file_->set_epoch_manager(&epoch_mgr_);

  auto result =
      file_->open(atlas_path_, /*initial_capacity=*/1000, effective_dim,
                  effective_metadata_size, opts.quantization_type);
  if (!result) {
    throw std::runtime_error("Failed to open Atlas storage");
  }

  // Read authoritative layout from the on-disk header
  auto *header = file_->get_header();
  dim_ = header->dim;
  metadata_size_ = header->metadata_size;
  node_byte_stride_ = header->node_byte_stride;
  quantization_type_ = header->quantization_type; // V4.1 Phase 3

  // Session-aware SLB cache (v4-plan.md Stage 0). Must be constructed here,
  // not in the member-initializer list: dim_ is only known once the header
  // has been read above. Cross-session L2 sharing is disabled -- see the
  // slb_cache_ member doc comment in atlas.hpp.
  slb_cache_ = std::make_unique<HierarchicalSLB>(
      dim_, /*enable_cross_session_l2=*/false);

  // Pre-allocate delta arena for ~10,000 nodes worth of contiguous memory
  delta_buffer_bytes_.reserve(10000 * node_byte_stride_);

  // ── V4.1 WAL: crash recovery (optional) ──
  wal_path_ = atlas_path_;
  wal_path_ += ".wal";
  if (enable_wal_) {
    replay_wal();
    open_wal();
  }

  // V4 Stage 2 follow-up: build the scope-union admission index AFTER
  // replay_wal() so it reflects final on-disk state (including any
  // WAL_RECORD_ATLAS_SCOPE records replayed above). No lock needed here --
  // this Atlas isn't reachable by any other thread yet.
  rebuild_scope_union_locked();
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
// scope_union_ maintenance — V4 Stage 2 follow-up (recall-gap fix)
// ═══════════════════════════════════════════════════════════════════════════

void Atlas::rebuild_scope_union_locked() {
  auto *header = file_->get_header();
  size_t n = header ? static_cast<size_t>(header->node_count) : 0;
  scope_union_.assign(n, 0);
  if (n == 0)
    return;

  // Seed: every node's own bits, queued if non-zero (nothing to push
  // upward for a node contributing zero new bits).
  std::vector<uint64_t> queue;
  queue.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    NodeHeader *node = file_->get_node(i);
    if (!node || node->scope_bitmap == 0)
      continue;
    scope_union_[i] |= node->scope_bitmap;
    queue.push_back(i);
  }

  // Worklist fixpoint -- see rebuild_scope_union_locked()'s doc comment
  // (atlas.hpp) for why this can't be a single reverse index pass.
  size_t head = 0;
  while (head < queue.size()) {
    uint64_t i = queue[head++];
    NodeHeader *node = file_->get_node(i);
    if (!node || node->parent_offset == 0)
      continue;
    uint64_t parent_idx =
        (node->parent_offset - sizeof(AtlasHeader)) / node_byte_stride_;
    if (parent_idx >= n)
      continue;
    uint64_t before = scope_union_[parent_idx];
    scope_union_[parent_idx] |= scope_union_[i];
    if (scope_union_[parent_idx] != before) {
      queue.push_back(parent_idx);
    }
  }
}

void Atlas::propagate_scope_union_locked(uint64_t node_idx) {
  if (node_idx >= scope_union_.size()) {
    // Defensive only -- every mutator keeps scope_union_ sized to
    // node_count, but this guarantees correctness even if that invariant
    // is ever violated rather than silently reading/writing out of range.
    scope_union_.resize(node_idx + 1, 0);
  }
  NodeHeader *node = file_->get_node(node_idx);
  if (!node)
    return;

  uint64_t before = scope_union_[node_idx];
  scope_union_[node_idx] |= node->scope_bitmap;
  if (scope_union_[node_idx] == before)
    return; // no new bits -- nothing to push upward

  std::vector<uint64_t> queue{node_idx};
  size_t head = 0;
  while (head < queue.size()) {
    uint64_t i = queue[head++];
    NodeHeader *cur = file_->get_node(i);
    if (!cur || cur->parent_offset == 0)
      continue;
    uint64_t parent_idx =
        (cur->parent_offset - sizeof(AtlasHeader)) / node_byte_stride_;
    if (parent_idx >= scope_union_.size())
      continue;
    uint64_t p_before = scope_union_[parent_idx];
    scope_union_[parent_idx] |= scope_union_[i];
    if (scope_union_[parent_idx] != p_before) {
      queue.push_back(parent_idx);
    }
  }
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
                                               bool apply_csls,
                                               uint64_t session_id,
                                               uint64_t scope_mask) {
  if (query.size() != dim_)
    return {};

  beam_width = std::clamp(beam_width, uint32_t{1}, MAX_BEAM_WIDTH);

  if (apply_csls) {
    return navigate_internal<true>(query, beam_width, session_id, scope_mask);
  } else {
    return navigate_internal<false>(query, beam_width, session_id, scope_mask);
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
Atlas::navigate_internal(std::span<const float> query, uint32_t beam_width,
                         uint64_t session_id, uint64_t scope_mask) {
  std::vector<Atlas::ResultNode> path;

  const bool scope_filtering = (scope_mask != ALL_SCOPES_VISIBLE);
  auto node_in_scope = [scope_mask](const NodeHeader &n) noexcept {
    return (n.scope_bitmap & scope_mask) != 0;
  };

  // V4 Stage 2 follow-up: whether node_idx's SUBTREE (scope_union_, not
  // just its own bit) provably contains nothing matching scope_mask. Used
  // ONLY for beam-admission priority below -- never for emission, which
  // still checks node_in_scope() (the real per-node bit) exclusively.
  // Fails OPEN to true (never excludes) when not filtering, or when
  // node_idx is out of scope_union_'s current range -- the same
  // fail-safe direction as the pre-follow-up blind admission this
  // replaces, so a stale/undersized union array can only forgo a
  // pruning opportunity, never wrongly exclude a real candidate.
  auto in_union = [scope_filtering, scope_mask,
                   this](uint64_t node_idx) noexcept {
    return !scope_filtering || node_idx >= scope_union_.size() ||
           (scope_union_[node_idx] & scope_mask) != 0;
  };

  // Acquire epoch guard before ANY mmap pointer access -- moved ahead of
  // the SLB cache check (below) so the staleness validation can safely
  // dereference file_->get_node() too.
  auto guard = epoch_mgr_.enter_guard();
  std::shared_lock<std::shared_mutex> read_lock(write_mutex_);

  // Fast-path: SLB cache hit (session-scoped L1, optionally L2 -- see
  // slb_cache_ member doc comment). VALIDATED, not trusted blindly: the
  // cache has no reverse index from node id -> cache entries, so
  // supersede_node()/tombstone_node()/consolidate_subgraph()/
  // set_node_scope() cannot evict a stale hit when a node's exclusion
  // state changes after it was cached. Without this check, a query that
  // had previously cached a node as its best match would keep returning
  // that node via the fast path even after it's superseded, tombstoned,
  // or scope-changed -- silently bypassing both the branchless
  // hub_penalty exclusion mechanism AND scope filtering entirely (found
  // while testing Stage 2's supersede_node() exclusion gate). Delta-arena
  // hits (MSB set) were never eligible for tombstone/supersede/scope in
  // the first place, so only mmap ids need the check.
  if (auto hit = slb_cache_->find_nearest(session_id, query, SLB_HIT_THRESHOLD)) {
    bool cache_hit_valid = true;
    if ((hit->node_id & NODE_ID_DELTA_MASK) == 0) {
      const NodeHeader *cached_node = file_->get_node(hit->node_id);
      if (!cached_node || is_tombstoned(*cached_node) ||
          is_superseded(*cached_node) ||
          (scope_filtering && !node_in_scope(*cached_node))) {
        cache_hit_valid = false;
      }
    } else if (scope_filtering) {
      // Delta nodes always have scope_bitmap == 0 (set_node_scope()
      // rejects delta ids) -- never a match under an active filter.
      cache_hit_valid = false;
    }
    if (cache_hit_valid) {
      return {{hit->node_id,
               hit->similarity,
               {hit->centroid_preview[0], hit->centroid_preview[1],
                hit->centroid_preview[2]}}};
    }
    // else: fall through to a full beam search below, same as a cache miss.
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

  auto *header = file_->get_header();

  // V4 Stage 2: when scope-filtering, widen the INTERNAL beam beyond the
  // caller's requested width -- gives the per-level REPORTING choice
  // (below) a larger pool to find an in-scope candidate in. Measured
  // (test_scope_recall.cpp) as helpful but INSUFFICIENT alone at low
  // selectivity (0.93/0.37 recall at 10%/2%, gate requires >=0.99).
  // A per-candidate scope-affinity steering bonus at intermediate-level
  // beam admission was also tried and measured ACTIVELY HARMFUL (recall
  // at 50% selectivity dropped from ~1.0 to ~0.52) -- reverted; see the
  // comment at the (former) use site below in the descent loop for why.
  // Closing the remaining gap needs accurate ancestor scope-union hints
  // to steer descent by, which deferred/option-b propagation doesn't
  // track by design (v4-plan.md Stage 2 status).
  const uint32_t effective_beam_width =
      scope_filtering ? MAX_BEAM_WIDTH : beam_width;

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
    // V4 Stage 2: emission-time scope filter -- root is still explored
    // and seeds the beam regardless (descent stays scope-blind), only
    // whether it's REPORTED in path is scope-aware.
    if (!scope_filtering || node_in_scope(*root)) {
      path.push_back({root->id, root_score, {p0, p1, p2}});
    }

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

          // V4 Stage 2: a per-candidate scope-affinity steering BONUS at
          // this intermediate-level admission decision was tried and
          // measured HARMFUL (recall at 50% selectivity DROPPED from ~1.0
          // to ~0.52) -- reverted. That approach used a candidate's own
          // scope_bitmap, which under deferred propagation says nothing
          // about its descendants, so it hijacked descent toward random
          // in-scope leaves rather than the true best match.
          //
          // V4 Stage 2 follow-up (replaces the above): admission now
          // consults scope_union_ -- a candidate whose SUBTREE (not just
          // its own bit) provably contains nothing matching scope_mask is
          // deprioritized, not hard-excluded. This is a sound
          // over-approximation (unlike a leaf's own bit), so it cannot
          // repeat the harmful-steering failure mode: a candidate is only
          // deprioritized when its entire subtree is PROVABLY irrelevant,
          // never because a real, reachable in-scope path scored lower on
          // raw similarity. Composite ordering (union-tier first, raw
          // score as tiebreak) degenerates to today's pure-score ordering
          // whenever scope_filtering is false OR every candidate ties on
          // union tier (including if scope_union_ is somehow out of range
          // for a candidate -- in_union() below fails OPEN to true, same
          // as the pre-follow-up blind behavior, never fails closed).
          // "a worse than b" under the composite (union-tier, score)
          // order -- identical to plain `a.score < b.score` whenever
          // scope_filtering is false or both share a union tier, so this
          // is a strict generalization of the pre-follow-up comparison,
          // not a behavior change outside the scope-filtered path.
          auto worse_than = [scope_filtering](bool a_ok, float a_score,
                                              bool b_ok, float b_score) {
            if (scope_filtering && a_ok != b_ok)
              return b_ok; // a is worse iff b is union-ok and a isn't
            return a_score < b_score;
          };

          if (next_beam_size < effective_beam_width) {
            next_beam[next_beam_size] = {child_node_idx, score};
            ++next_beam_size;
            if (score > overall_best.score) {
              overall_best = next_beam[next_beam_size - 1];
            }
          } else {
            uint32_t worst_idx = 0;
            bool worst_ok = in_union(next_beam[0].node_idx);
            for (uint32_t k = 1; k < next_beam_size; ++k) {
              bool k_ok = in_union(next_beam[k].node_idx);
              if (worse_than(k_ok, next_beam[k].score, worst_ok,
                             next_beam[worst_idx].score)) {
                worst_idx = k;
                worst_ok = k_ok;
              }
            }
            if (worse_than(worst_ok, next_beam[worst_idx].score,
                           in_union(child_node_idx), score)) {
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

      // V4 Stage 2: which beam member gets REPORTED for this level is
      // scope-aware; the beam itself (assigned above, used for the NEXT
      // level's descent) is not -- descent stays scope-blind regardless.
      uint32_t best_in_beam = 0;
      bool have_reportable_best = true;
      if (scope_filtering) {
        have_reportable_best = false;
        // aeon_core is built with -ffast-math (CLAUDE.md), which makes
        // std::numeric_limits<float>::infinity() undefined behavior --
        // lowest() is the most-negative FINITE float, safe under
        // fast-math and still smaller than any real achievable score
        // (cosine similarity is bounded to [-1, 1] even before any
        // hub_penalty adjustment).
        float best_score = std::numeric_limits<float>::lowest();
        for (uint32_t k = 0; k < beam_size; ++k) {
          const NodeHeader *cand = file_->get_node(beam[k].node_idx);
          if (cand && node_in_scope(*cand) && beam[k].score > best_score) {
            best_score = beam[k].score;
            best_in_beam = k;
            have_reportable_best = true;
          }
        }
      } else {
        for (uint32_t k = 1; k < beam_size; ++k) {
          if (beam[k].score > beam[best_in_beam].score)
            best_in_beam = k;
        }
      }

      if (have_reportable_best) {
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
        slb_cache_->insert(session_id, best->id,
                          std::span<const float>(fp32_vec.data(), dim_));
      } else {
        slb_cache_->insert(session_id, best->id,
                          std::span<const float>(node_centroid(best), dim_));
      }
    }
  }

  // ── Linear scan of flat byte arena delta buffer (mutable layer) ──
  // V4 Stage 2: skipped entirely when scope-filtering is active -- delta
  // nodes always have scope_bitmap == 0 (set_node_scope() rejects delta
  // ids outright), so none could ever match a real filter; no point
  // paying the scan cost.
  std::vector<ResultNode> delta_candidates;
  if (!scope_filtering) {
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
Atlas::navigate_internal<false>(std::span<const float>, uint32_t, uint64_t,
                                uint64_t);
template std::vector<Atlas::ResultNode>
Atlas::navigate_internal<true>(std::span<const float>, uint32_t, uint64_t,
                               uint64_t);

// ═══════════════════════════════════════════════════════════════════════════
// get_children, insert, insert_delta, prune_delta_tail, load_context
// ═══════════════════════════════════════════════════════════════════════════

std::vector<Atlas::ResultNode> Atlas::get_children(uint64_t parent_id,
                                                   uint64_t scope_mask) {
  std::vector<Atlas::ResultNode> children;
  const bool scope_filtering = (scope_mask != ALL_SCOPES_VISIBLE);

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

    // V4 Stage 2 task 2: the graph-expansion-boundary enforcement point --
    // see this function's doc comment in atlas.hpp.
    if (scope_filtering && (child->scope_bitmap & scope_mask) == 0)
      continue;

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
  // scope_bitmap defaults to 0 (unscoped) -- no scope-assignment authority
  // exists yet (Stage 3/4); see NodeHeader's doc comment in schema.hpp.
  hdr->scope_bitmap = 0;
  hdr->governance_record_id = 0;
  hdr->saved_hub_penalty = 0.0f;

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

  uint64_t new_id = NODE_ID_DELTA_MASK | delta_node_count();

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
                       std::string_view metadata,
                       [[maybe_unused]] uint64_t session_id) {
  // session_id is accepted (see atlas.hpp doc comment) but not yet used --
  // insert() doesn't populate the SLB cache today, only navigate() does.
  // V4 Stage 1 landed the scope_bitmap field (defaults to 0, see below) but
  // deliberately does NOT derive it from session_id here: there is no
  // scope-assignment authority yet (Stage 3/4's control plane). insert()
  // has no caller-supplied scope parameter by design -- once that
  // authority exists, session_id is the only input it will ever consult.

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
  // scope_bitmap defaults to 0 -- see insert_delta()'s comment above; the
  // same rationale applies (session_id above is not yet a scope authority).
  node->scope_bitmap = 0;
  node->governance_record_id = 0;
  node->saved_hub_penalty = 0.0f;

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
  // V4 Stage 2 follow-up: keep scope_union_ sized to node_count. A fresh
  // node's own scope_bitmap is always 0 here (see the comment above), so
  // this entry contributes nothing yet -- real propagation happens later,
  // when set_node_scope() actually assigns this node a scope.
  scope_union_.push_back(0);
  return new_id;
}

void Atlas::load_context(std::span<const uint64_t> node_ids,
                         uint64_t session_id) {
  for (uint64_t id : node_ids) {
    if ((id & NODE_ID_DELTA_MASK) == 0) {
      if (auto *node = file_->get_node(id)) {
        slb_cache_->insert(session_id, id,
                          std::span<const float>(node_centroid(node), dim_));
      }
    }
  }
}

bool Atlas::drop_session(uint64_t session_id) {
  return slb_cache_->drop_session(session_id);
}

void Atlas::sync() { file_->sync(); }

// ═══════════════════════════════════════════════════════════════════════════
// set_node_scope / get_node_scope — V4 Stage 1/2 scope mutation primitive
// ═══════════════════════════════════════════════════════════════════════════

void Atlas::set_node_scope(uint64_t node_id, uint64_t scope_bitmap) {
  if (node_id & NODE_ID_DELTA_MASK) {
    throw std::invalid_argument(
        "set_node_scope: delta-arena node ids are not supported -- delta "
        "nodes get a fresh id when compact_mmap() promotes them, so a "
        "scope set against the old id would be silently lost");
  }

  // Mutating an existing node while compact_mmap() is concurrently copying
  // it to the new generation file risks the write landing in the old
  // generation and being lost -- same reasoning as consolidate_subgraph()'s
  // guard on this flag.
  if (compact_in_progress_.load(std::memory_order_acquire)) {
    throw std::runtime_error(
        "set_node_scope: cannot mutate a node while compaction is in "
        "progress");
  }

  // Serializes against insert()/consolidate_subgraph()'s mmap mutations
  // (same write_mutex_ they take) and against compact_mmap()'s copy pass.
  std::unique_lock<std::shared_mutex> write_lock(write_mutex_);

  auto *header = file_->get_header();
  if (!header || node_id >= header->node_count) {
    throw std::runtime_error("set_node_scope: invalid node id " +
                             std::to_string(node_id));
  }
  NodeHeader *node = file_->get_node(node_id);
  if (!node) {
    throw std::runtime_error("set_node_scope: null node at id " +
                             std::to_string(node_id));
  }

  // WAL-write BEFORE mutating, same write-ahead ordering as insert_delta():
  // if the mmap write itself doesn't survive a crash (insert() has no WAL
  // of its own -- see guardrail #1.3 -- so a fresh node from the same
  // session might not have either), replay can still reapply this scope
  // set from the durable WAL record. New lock chain (write_mutex_ then
  // wal_mutex_) -- distinct from insert_delta()'s (wal_mutex_ then
  // delta_mutex_), and safe to nest since nothing else ever holds both
  // write_mutex_ and wal_mutex_ at once.
  if (enable_wal_) {
    std::lock_guard<std::mutex> wal_lock(wal_mutex_);
    if (wal_stream_.is_open()) {
      WalScopeRecord rec{node_id, scope_bitmap};
      WalRecordHeader wal_hdr{};
      wal_hdr.record_type = WAL_RECORD_ATLAS_SCOPE;
      wal_hdr.payload_size = static_cast<uint32_t>(sizeof(WalScopeRecord));
      wal_hdr.checksum = hash::fnv1a_64(&rec, sizeof(WalScopeRecord));

      wal_stream_.write(reinterpret_cast<const char *>(&wal_hdr),
                        sizeof(WalRecordHeader));
      wal_stream_.write(reinterpret_cast<const char *>(&rec),
                        sizeof(WalScopeRecord));
      wal_stream_.flush();
    }
  }

  node->scope_bitmap = scope_bitmap;

  // V4 Stage 2 follow-up: push the new bits up into scope_union_ (the
  // navigate() beam-admission steering index) -- see this function's own
  // doc comment (atlas.hpp) and propagate_scope_union_locked()'s comment
  // for why this correctly reaches a consolidate_subgraph() summary even
  // if that summary's index is numerically LATER than node_id.
  propagate_scope_union_locked(node_id);
}

uint64_t Atlas::get_node_scope(uint64_t node_id) const {
  if (node_id & NODE_ID_DELTA_MASK) {
    throw std::invalid_argument(
        "get_node_scope: delta-arena node ids are not supported");
  }

  // EBR-guarded: unlike tombstone_count() (a diagnostic), this is the read
  // half of a primitive Stage 4's control plane will call from a different
  // thread than any writer -- without the guard, a concurrent
  // compact_mmap() could retire the mmap region this reads through.
  auto guard = epoch_mgr_.enter_guard();
  std::shared_lock<std::shared_mutex> read_lock(write_mutex_);

  auto *header = file_->get_header();
  if (!header || node_id >= header->node_count) {
    throw std::runtime_error("get_node_scope: invalid node id " +
                             std::to_string(node_id));
  }
  const NodeHeader *node = file_->get_node(node_id);
  if (!node) {
    throw std::runtime_error("get_node_scope: null node at id " +
                             std::to_string(node_id));
  }
  return node->scope_bitmap;
}

// ═══════════════════════════════════════════════════════════════════════════
// supersede_node / revoke_node_supersede / is_node_superseded
// ═══════════════════════════════════════════════════════════════════════════

void Atlas::supersede_node(uint64_t node_id) {
  if (node_id & NODE_ID_DELTA_MASK) {
    throw std::invalid_argument(
        "supersede_node: delta-arena node ids are not supported -- same "
        "reasoning as set_node_scope()");
  }
  if (compact_in_progress_.load(std::memory_order_acquire)) {
    throw std::runtime_error(
        "supersede_node: cannot mutate a node while compaction is in "
        "progress");
  }

  std::unique_lock<std::shared_mutex> write_lock(write_mutex_);

  auto *header = file_->get_header();
  if (!header || node_id >= header->node_count) {
    throw std::runtime_error("supersede_node: invalid node id " +
                             std::to_string(node_id));
  }
  NodeHeader *node = file_->get_node(node_id);
  if (!node) {
    throw std::runtime_error("supersede_node: null node at id " +
                             std::to_string(node_id));
  }

  // WAL-write BEFORE mutating -- same write-ahead ordering as
  // set_node_scope()/insert_delta(). Idempotent on replay: supersede_node()
  // (the schema.hpp free function) is a documented no-op if already
  // superseded.
  if (enable_wal_) {
    std::lock_guard<std::mutex> wal_lock(wal_mutex_);
    if (wal_stream_.is_open()) {
      WalSupersedeRecord rec{node_id, /*revoke=*/0};
      WalRecordHeader wal_hdr{};
      wal_hdr.record_type = WAL_RECORD_ATLAS_SUPERSEDE;
      wal_hdr.payload_size = static_cast<uint32_t>(sizeof(WalSupersedeRecord));
      wal_hdr.checksum = hash::fnv1a_64(&rec, sizeof(WalSupersedeRecord));

      wal_stream_.write(reinterpret_cast<const char *>(&wal_hdr),
                        sizeof(WalRecordHeader));
      wal_stream_.write(reinterpret_cast<const char *>(&rec),
                        sizeof(WalSupersedeRecord));
      wal_stream_.flush();
    }
  }

  aeon::supersede_node(*node); // schema.hpp free function (see doc comment)
}

void Atlas::revoke_node_supersede(uint64_t node_id) {
  if (node_id & NODE_ID_DELTA_MASK) {
    throw std::invalid_argument(
        "revoke_node_supersede: delta-arena node ids are not supported");
  }
  if (compact_in_progress_.load(std::memory_order_acquire)) {
    throw std::runtime_error(
        "revoke_node_supersede: cannot mutate a node while compaction is "
        "in progress");
  }

  std::unique_lock<std::shared_mutex> write_lock(write_mutex_);

  auto *header = file_->get_header();
  if (!header || node_id >= header->node_count) {
    throw std::runtime_error("revoke_node_supersede: invalid node id " +
                             std::to_string(node_id));
  }
  NodeHeader *node = file_->get_node(node_id);
  if (!node) {
    throw std::runtime_error("revoke_node_supersede: null node at id " +
                             std::to_string(node_id));
  }

  if (enable_wal_) {
    std::lock_guard<std::mutex> wal_lock(wal_mutex_);
    if (wal_stream_.is_open()) {
      WalSupersedeRecord rec{node_id, /*revoke=*/1};
      WalRecordHeader wal_hdr{};
      wal_hdr.record_type = WAL_RECORD_ATLAS_SUPERSEDE;
      wal_hdr.payload_size = static_cast<uint32_t>(sizeof(WalSupersedeRecord));
      wal_hdr.checksum = hash::fnv1a_64(&rec, sizeof(WalSupersedeRecord));

      wal_stream_.write(reinterpret_cast<const char *>(&wal_hdr),
                        sizeof(WalRecordHeader));
      wal_stream_.write(reinterpret_cast<const char *>(&rec),
                        sizeof(WalSupersedeRecord));
      wal_stream_.flush();
    }
  }

  aeon::revoke_supersede(*node); // schema.hpp free function
}

bool Atlas::is_node_superseded(uint64_t node_id) const {
  if (node_id & NODE_ID_DELTA_MASK) {
    throw std::invalid_argument(
        "is_node_superseded: delta-arena node ids are not supported");
  }

  auto guard = epoch_mgr_.enter_guard();
  std::shared_lock<std::shared_mutex> read_lock(write_mutex_);

  auto *header = file_->get_header();
  if (!header || node_id >= header->node_count) {
    throw std::runtime_error("is_node_superseded: invalid node id " +
                             std::to_string(node_id));
  }
  const NodeHeader *node = file_->get_node(node_id);
  if (!node) {
    throw std::runtime_error("is_node_superseded: null node at id " +
                             std::to_string(node_id));
  }
  return aeon::is_superseded(*node);
}

// ═══════════════════════════════════════════════════════════════════════════
// tombstone_node(uint64_t)  (V4 Stage 4 task 5/6 -- console/erasure "delete")
// ═══════════════════════════════════════════════════════════════════════════

void Atlas::tombstone_node(uint64_t node_id) {
  if (node_id & NODE_ID_DELTA_MASK) {
    throw std::invalid_argument(
        "tombstone_node: delta-arena node ids are not supported -- same "
        "reasoning as set_node_scope()/supersede_node()");
  }
  if (compact_in_progress_.load(std::memory_order_acquire)) {
    throw std::runtime_error(
        "tombstone_node: cannot mutate a node while compaction is in "
        "progress");
  }

  std::unique_lock<std::shared_mutex> write_lock(write_mutex_);

  auto *header = file_->get_header();
  if (!header || node_id >= header->node_count) {
    throw std::runtime_error("tombstone_node: invalid node id " +
                             std::to_string(node_id));
  }
  NodeHeader *node = file_->get_node(node_id);
  if (!node) {
    throw std::runtime_error("tombstone_node: null node at id " +
                             std::to_string(node_id));
  }

  // WAL-write BEFORE mutating -- same write-ahead ordering as
  // supersede_node()/set_node_scope().
  if (enable_wal_) {
    std::lock_guard<std::mutex> wal_lock(wal_mutex_);
    if (wal_stream_.is_open()) {
      WalTombstoneRecord rec{node_id};
      WalRecordHeader wal_hdr{};
      wal_hdr.record_type = WAL_RECORD_ATLAS_TOMBSTONE;
      wal_hdr.payload_size = static_cast<uint32_t>(sizeof(WalTombstoneRecord));
      wal_hdr.checksum = hash::fnv1a_64(&rec, sizeof(WalTombstoneRecord));

      wal_stream_.write(reinterpret_cast<const char *>(&wal_hdr),
                        sizeof(WalRecordHeader));
      wal_stream_.write(reinterpret_cast<const char *>(&rec),
                        sizeof(WalTombstoneRecord));
      wal_stream_.flush();
    }
  }

  aeon::tombstone_node(*node); // schema.hpp free function (see doc comment)
}

// ═══════════════════════════════════════════════════════════════════════════
// set_node_governance_id / get_node_governance_id / list_nodes_by_scope /
// bulk_set_node_scope  (V4 Stage 4 task 1)
// ═══════════════════════════════════════════════════════════════════════════

void Atlas::set_node_governance_id(uint64_t node_id,
                                   uint64_t governance_record_id) {
  if (node_id & NODE_ID_DELTA_MASK) {
    throw std::invalid_argument(
        "set_node_governance_id: delta-arena node ids are not supported -- "
        "same reasoning as set_node_scope()");
  }
  if (compact_in_progress_.load(std::memory_order_acquire)) {
    throw std::runtime_error(
        "set_node_governance_id: cannot mutate a node while compaction is "
        "in progress");
  }

  std::unique_lock<std::shared_mutex> write_lock(write_mutex_);

  auto *header = file_->get_header();
  if (!header || node_id >= header->node_count) {
    throw std::runtime_error("set_node_governance_id: invalid node id " +
                             std::to_string(node_id));
  }
  NodeHeader *node = file_->get_node(node_id);
  if (!node) {
    throw std::runtime_error("set_node_governance_id: null node at id " +
                             std::to_string(node_id));
  }

  // WAL-write BEFORE mutating -- same write-ahead ordering as
  // set_node_scope().
  if (enable_wal_) {
    std::lock_guard<std::mutex> wal_lock(wal_mutex_);
    if (wal_stream_.is_open()) {
      WalGovernanceRecord rec{node_id, governance_record_id};
      WalRecordHeader wal_hdr{};
      wal_hdr.record_type = WAL_RECORD_ATLAS_GOVERNANCE;
      wal_hdr.payload_size =
          static_cast<uint32_t>(sizeof(WalGovernanceRecord));
      wal_hdr.checksum = hash::fnv1a_64(&rec, sizeof(WalGovernanceRecord));

      wal_stream_.write(reinterpret_cast<const char *>(&wal_hdr),
                        sizeof(WalRecordHeader));
      wal_stream_.write(reinterpret_cast<const char *>(&rec),
                        sizeof(WalGovernanceRecord));
      wal_stream_.flush();
    }
  }

  node->governance_record_id = governance_record_id;
}

uint64_t Atlas::get_node_governance_id(uint64_t node_id) const {
  if (node_id & NODE_ID_DELTA_MASK) {
    throw std::invalid_argument(
        "get_node_governance_id: delta-arena node ids are not supported");
  }

  auto guard = epoch_mgr_.enter_guard();
  std::shared_lock<std::shared_mutex> read_lock(write_mutex_);

  auto *header = file_->get_header();
  if (!header || node_id >= header->node_count) {
    throw std::runtime_error("get_node_governance_id: invalid node id " +
                             std::to_string(node_id));
  }
  const NodeHeader *node = file_->get_node(node_id);
  if (!node) {
    throw std::runtime_error("get_node_governance_id: null node at id " +
                             std::to_string(node_id));
  }
  return node->governance_record_id;
}

std::string Atlas::get_node_metadata(uint64_t node_id) const {
  // Unlike get_node_scope()/get_node_governance_id(), delta-arena ids are
  // supported here -- see this method's doc comment in atlas.hpp for why.
  if (node_id & NODE_ID_DELTA_MASK) {
    std::shared_lock<std::shared_mutex> delta_lock(delta_mutex_);
    size_t index = node_id & ~NODE_ID_DELTA_MASK;
    const NodeHeader *node = delta_get_node(index);
    if (!node) {
      throw std::runtime_error("get_node_metadata: invalid delta node id " +
                               std::to_string(node_id));
    }
    const char *meta = node_metadata_q(node, dim_, quantization_type_);
    return std::string(meta);
  }

  auto guard = epoch_mgr_.enter_guard();
  std::shared_lock<std::shared_mutex> read_lock(write_mutex_);

  auto *header = file_->get_header();
  if (!header || node_id >= header->node_count) {
    throw std::runtime_error("get_node_metadata: invalid node id " +
                             std::to_string(node_id));
  }
  const NodeHeader *node = file_->get_node(node_id);
  if (!node) {
    throw std::runtime_error("get_node_metadata: null node at id " +
                             std::to_string(node_id));
  }
  const char *meta = node_metadata_q(node, dim_, quantization_type_);
  return std::string(meta);
}

std::vector<float> Atlas::get_node_centroid(uint64_t node_id) const {
  auto extract = [this](const NodeHeader *node) -> std::vector<float> {
    std::vector<float> out(dim_);
    if (quantization_type_ == QUANT_INT8_SYMMETRIC) {
      const int8_t *q = node_centroid_int8(node);
      quant::dequantize_vector(std::span<const int8_t>(q, dim_),
                               node->quant_scale, std::span<float>(out));
    } else {
      const float *c = node_centroid(node);
      std::copy(c, c + dim_, out.begin());
    }
    return out;
  };

  if (node_id & NODE_ID_DELTA_MASK) {
    std::shared_lock<std::shared_mutex> delta_lock(delta_mutex_);
    size_t index = node_id & ~NODE_ID_DELTA_MASK;
    const NodeHeader *node = delta_get_node(index);
    if (!node) {
      throw std::runtime_error("get_node_centroid: invalid delta node id " +
                               std::to_string(node_id));
    }
    return extract(node);
  }

  auto guard = epoch_mgr_.enter_guard();
  std::shared_lock<std::shared_mutex> read_lock(write_mutex_);

  auto *header = file_->get_header();
  if (!header || node_id >= header->node_count) {
    throw std::runtime_error("get_node_centroid: invalid node id " +
                             std::to_string(node_id));
  }
  const NodeHeader *node = file_->get_node(node_id);
  if (!node) {
    throw std::runtime_error("get_node_centroid: null node at id " +
                             std::to_string(node_id));
  }
  return extract(node);
}

std::vector<uint64_t> Atlas::list_nodes_by_scope(uint64_t scope_mask) const {
  auto guard = epoch_mgr_.enter_guard();
  std::shared_lock<std::shared_mutex> read_lock(write_mutex_);

  std::vector<uint64_t> result;
  auto *header = file_->get_header();
  if (!header)
    return result;

  // ALL_SCOPES_VISIBLE is special-cased to mean "no filtering" (every live
  // node, including unscoped ones), matching navigate()'s documented
  // semantics for the same sentinel -- NOT treated as an ordinary mask.
  // A plain AND check would get this backwards: unscoped nodes default to
  // scope_bitmap == 0, and 0 & ALL_SCOPES_VISIBLE == 0 (falsy), so an
  // ordinary mask check would EXCLUDE every unscoped node from a query
  // that's supposed to mean "everything" -- the exact inverse of what a
  // console caller expects (found via advisor review, v4-plan.md).
  bool unfiltered = (scope_mask == ALL_SCOPES_VISIBLE);
  for (uint64_t i = 0; i < header->node_count; ++i) {
    const NodeHeader *node = file_->get_node(i);
    if (!node || aeon::is_tombstoned(*node))
      continue;
    if (unfiltered || (node->scope_bitmap & scope_mask) != 0) {
      result.push_back(i);
    }
  }
  return result;
}

void Atlas::bulk_set_node_scope(
    const std::vector<std::pair<uint64_t, uint64_t>> &updates) {
  if (updates.empty())
    return;
  if (compact_in_progress_.load(std::memory_order_acquire)) {
    throw std::runtime_error(
        "bulk_set_node_scope: cannot mutate nodes while compaction is in "
        "progress");
  }

  std::unique_lock<std::shared_mutex> write_lock(write_mutex_);

  auto *header = file_->get_header();

  // Pass 1: validate every entry BEFORE mutating any (all-or-nothing) --
  // an invalid id partway through the batch must not leave earlier entries
  // applied and later ones not.
  std::vector<NodeHeader *> nodes;
  nodes.reserve(updates.size());
  for (const auto &[node_id, scope_bitmap] : updates) {
    if (node_id & NODE_ID_DELTA_MASK) {
      throw std::invalid_argument(
          "bulk_set_node_scope: delta-arena node ids are not supported -- "
          "same reasoning as set_node_scope()");
    }
    if (!header || node_id >= header->node_count) {
      throw std::runtime_error("bulk_set_node_scope: invalid node id " +
                               std::to_string(node_id));
    }
    NodeHeader *node = file_->get_node(node_id);
    if (!node) {
      throw std::runtime_error("bulk_set_node_scope: null node at id " +
                               std::to_string(node_id));
    }
    nodes.push_back(node);
  }

  // Pass 2: WAL-write every record, single flush at the end -- the actual
  // "bulk" efficiency win over N sequential set_node_scope() calls (each
  // of which would otherwise pay its own lock acquisition and fsync cost).
  if (enable_wal_) {
    std::lock_guard<std::mutex> wal_lock(wal_mutex_);
    if (wal_stream_.is_open()) {
      for (const auto &[node_id, scope_bitmap] : updates) {
        WalScopeRecord rec{node_id, scope_bitmap};
        WalRecordHeader wal_hdr{};
        wal_hdr.record_type = WAL_RECORD_ATLAS_SCOPE;
        wal_hdr.payload_size = static_cast<uint32_t>(sizeof(WalScopeRecord));
        wal_hdr.checksum = hash::fnv1a_64(&rec, sizeof(WalScopeRecord));

        wal_stream_.write(reinterpret_cast<const char *>(&wal_hdr),
                          sizeof(WalRecordHeader));
        wal_stream_.write(reinterpret_cast<const char *>(&rec),
                          sizeof(WalScopeRecord));
      }
      wal_stream_.flush();
    }
  }

  // Pass 3: apply. Safe to do after WAL writes complete -- same
  // write-ahead ordering as every other mutator in this file. Each
  // mutation immediately propagates into scope_union_ (V4 Stage 2
  // follow-up), same as set_node_scope()'s own ordering.
  for (size_t i = 0; i < updates.size(); ++i) {
    nodes[i]->scope_bitmap = updates[i].second;
    propagate_scope_union_locked(updates[i].first);
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

  // Phase 1: Validate (and accumulate scope union -- the summary must
  // remain visible under any scope its sources were visible under, or a
  // future scope-filtered retrieval would silently drop consolidated
  // content). V4 Stage 5 task 1 (advisor review): also enforce that every
  // source shares the IDENTICAL scope_bitmap -- see this function's own
  // doc comment (atlas.hpp) for why a mere OR-union is not enough on its
  // own to prevent a consolidation from WIDENING visibility.
  uint64_t summary_scope_union = 0;
  bool have_reference_scope = false;
  uint64_t reference_scope = 0;
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
    if (!have_reference_scope) {
      reference_scope = node->scope_bitmap;
      have_reference_scope = true;
    } else if (node->scope_bitmap != reference_scope) {
      throw std::invalid_argument(
          "consolidate_subgraph: old_node_ids must share the same "
          "scope_bitmap -- node " +
          std::to_string(id) + " has scope " +
          std::to_string(node->scope_bitmap) + ", expected " +
          std::to_string(reference_scope) +
          " (consolidating across scopes would widen visibility of the "
          "summary beyond any single source)");
    }
    summary_scope_union |= node->scope_bitmap;
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
  uint8_t *file_base = reinterpret_cast<uint8_t *>(header);
  uint64_t summary_abs_offset =
      reinterpret_cast<uint8_t *>(summary) - file_base;

  summary->id = summary_id;
  summary->parent_offset = summary_parent_offset;
  summary->first_child_offset = 0;
  summary->child_count = 0;
  summary->flags = NODE_FLAG_SUMMARY;
  summary->hub_penalty = 0.0f;
  summary->scope_bitmap = summary_scope_union;
  summary->governance_record_id = 0;
  summary->saved_hub_penalty = 0.0f;

  std::memcpy(node_centroid(summary), summary_vector.data(),
              dim_ * sizeof(float));

  char *meta = node_metadata(summary, dim_);
  std::memset(meta, 0, metadata_size_);
  size_t meta_len =
      std::min(summary_meta.size(), static_cast<size_t>(metadata_size_ - 1));
  std::memcpy(meta, summary_meta.data(), meta_len);

  // V4 Stage 2 follow-up chased a SEPARATE finding surfaced while proving
  // the recall fix's discrimination: this function set summary->parent_offset
  // (so the summary "knows" its logical parent) but NEVER updated that
  // PARENT's own child_count/first_child_offset to enumerate the summary
  // back -- confirmed empirically (a scratch root->child->consolidate_subgraph
  // check, see v4-plan.md) that the resulting summary was NEVER returned by
  // navigate() at all, regardless of scope filtering. Fixed with EXACTLY
  // insert()'s own contiguity check (atlas.cpp's insert(), "is_contiguous"):
  // the summary is always placed at the current tail (header->node_count,
  // same as insert()'s new_idx), so it registers correctly whenever nothing
  // else was inserted under a DIFFERENT parent since the parent's own last
  // child -- true in every case this codebase's real callers exercise today
  // (every production Atlas.insert() call uses parent_id=0, so root's
  // children are always contiguous by construction) and in the common
  // Dreaming case (consolidating LEAF fragments with no children of their
  // own, so Phase 3 below never needs to rewire anything for this specific
  // consolidation). Left un-registered (parent_node->child_count untouched,
  // exactly insert()'s own existing behavior for a non-contiguous target)
  // when it doesn't land contiguously -- a pre-existing property of this
  // tree's flat-array child representation, not something this fix
  // resolves in general; see this function's own doc comment (atlas.hpp)
  // for the narrower residual gap this leaves (Phase 3's rewired children,
  // when old_node_ids are non-leaves, are not similarly registered under
  // the summary -- flagged for Stage 5, not fixed here).
  if (summary_parent_offset != 0) {
    uint64_t parent_idx =
        (summary_parent_offset - sizeof(AtlasHeader)) / node_byte_stride_;
    NodeHeader *parent_node = file_->get_node(parent_idx);
    if (parent_node) {
      bool parent_contiguous = false;
      if (parent_node->child_count == 0) {
        parent_node->first_child_offset = summary_abs_offset;
        parent_contiguous = true;
      } else {
        uint64_t expected_addr =
            parent_node->first_child_offset +
            (parent_node->child_count * node_byte_stride_);
        if (expected_addr == summary_abs_offset) {
          parent_contiguous = true;
        }
      }
      if (parent_contiguous) {
        parent_node->child_count++;
      }
    }
  }

  header->node_count++;
  // V4 Stage 2 follow-up: keep scope_union_ sized to node_count, same as
  // insert()'s own bookkeeping (this function creates the summary node
  // directly rather than via insert()). Seed the summary's own union
  // entry now (summary_scope_union, already the OR of every source's OWN
  // bit -- Phase 1 above) and push it toward summary's ancestors; the
  // rewired children below push THEIR existing subtree unions into
  // summary next, and propagate_scope_union_locked()'s worklist
  // automatically continues propagating further up if that changes
  // summary's own union in turn.
  scope_union_.push_back(0);
  propagate_scope_union_locked(summary_id);

  // Phase 3: Re-wire children → summary
  // (file_base/summary_abs_offset already computed above, alongside the
  // new parent-registration logic that also needs them)

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
        // V4 Stage 2 follow-up: this child's parent just changed to
        // summary_id, which is NUMERICALLY LATER than the child (the
        // exact case rebuild_scope_union_locked()'s doc comment explains
        // a naive single reverse pass would miss). Re-propagating from
        // the child pushes its already-known subtree union into its new
        // parent explicitly, rather than relying on index ordering.
        propagate_scope_union_locked(child->id);
      }
    }
  }

  // Phase 4: Tombstone
  //
  // Explicitly namespace-qualified (aeon::tombstone_node, the schema.hpp
  // free function) -- Atlas now also has a PUBLIC MEMBER tombstone_node
  // (uint64_t) (V4 Stage 4 task 5/6), and inside a member function an
  // unqualified call resolves to class scope first, which would try (and
  // fail) to convert `*old_node` (NodeHeader&) into a node id. Same
  // disambiguation supersede_node()/revoke_node_supersede() already need
  // for aeon::supersede_node()/aeon::revoke_supersede().
  for (uint64_t id : old_node_ids) {
    NodeHeader *old_node = file_->get_node(id);
    if (old_node) {
      aeon::tombstone_node(*old_node);
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
  // V4 Stage 2 fix: a SEVERE pre-existing bug, same root cause and fix as
  // TraceManager::compact() (v4-plan.md Stage 2 status has the full
  // writeup). This function used to build the new generation at a
  // PERMANENTLY generation-suffixed name (atlas_gen1.bin, atlas_gen2.bin,
  // ...) and delete the file at atlas_path_ -- but atlas_path_/
  // generation_ are only tracked in-memory, reset to (constructor's path
  // argument, 0) on every fresh Atlas construction. Any restart using the
  // caller's originally-configured path (the normal case -- see
  // dependencies.py's AEON_ATLAS_PATH) would find that path GONE (deleted
  // by the prior compaction) and silently create a new, EMPTY Atlas:
  // total, silent loss of every long-term memory node on the very first
  // restart after the very first compaction.
  //
  // Fix: build the new generation at a temporary path, durably flush it
  // (already done below), then ATOMICALLY RENAME it onto the stable,
  // caller-facing atlas_path_ before reopening via MemoryFile -- POSIX
  // rename() only rewrites the directory entry, so this is safe even
  // though the OLD file at that name is still open via `old_file` until
  // Step 4 closes it below. atlas_path_ itself never changes again after
  // construction; only generation_ (an internal temp-filename
  // disambiguator only, now) still increments.
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

  // Generational TEMPORARY file naming (Windows MapViewOfFile safety) --
  // renamed onto the stable atlas_path_ once fully populated and durable,
  // below. Never a permanent name (see this function's opening comment).
  uint64_t new_gen = generation_ + 1;
  std::filesystem::path new_path = atlas_path_;
  new_path += (".compacting" + std::to_string(new_gen));

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

  // Durably flush the fully-built new generation file BEFORE the old
  // generation (which is about to be deleted below) stops being a fallback
  // copy. Unlike per-insert mmap writes -- process-crash-safe already via
  // MAP_SHARED page-cache backing, see Atlas::insert() doc comment -- this
  // is the one point in compaction where skipping an explicit sync creates
  // a real irrecoverable-on-power-loss window: after the old file is
  // unlinked, the new file is the ONLY copy, so it must actually be on
  // disk first (v4-plan.md guardrail #1.3).
  platform::mem_sync(new_raw, new_file_size);

  // Unmap the new file (we'll reopen it via MemoryFile)
  platform::mem_unmap(new_raw, new_file_size);
  platform::file_close(new_handle);

  // ATOMICALLY install the new generation at the stable, caller-facing
  // path -- see this function's opening comment. Must happen after the
  // durable flush above (a crash between "rename" and "flush" could
  // otherwise leave the stable path pointing at incompletely-written
  // data) and before reopening below (MemoryFile::open() must open the
  // NOW-correct stable name, not the temp name that no longer exists
  // after this rename).
  {
    std::error_code ec;
    std::filesystem::rename(new_path, atlas_path_, ec);
    if (ec) {
      compact_in_progress_.store(false, std::memory_order_release);
      throw std::runtime_error("compact_mmap: failed to install new "
                               "generation at stable path: " +
                               ec.message());
    }
  }

  // Release the EBR guard from Step 2
  guard.release();

  // ── Step 3: Microsecond Freeze (Non-Blocking Swap) ──
  // Create new MemoryFile for the compacted generation file (now living
  // at the stable atlas_path_, having just been renamed onto it above).
  auto new_file = std::make_unique<storage::MemoryFile>();
  new_file->set_epoch_manager(&epoch_mgr_);
  auto open_result =
      new_file->open(atlas_path_, total_live, dim_, metadata_size_);
  if (!open_result) {
    compact_in_progress_.store(false, std::memory_order_release);
    throw std::runtime_error("compact_mmap: failed to reopen compacted file");
  }

  // Capture old file pointer for background cleanup. Its underlying inode
  // is already unlinked from atlas_path_ (replaced by the rename above)
  // but remains valid via `old_file`'s own open fd/mmap until closed in
  // Step 4 -- nothing left to separately remove by path afterward.
  std::unique_ptr<storage::MemoryFile> old_file;

  {
    std::unique_lock<std::shared_mutex> write_lock(write_mutex_);

    // Pointer swap: O(1), no I/O, no drain_readers()
    old_file = std::move(file_);
    file_ = std::move(new_file);

    // Clear frozen buffer and SLB (node IDs have changed)
    frozen_delta_buffer_bytes_.clear();
    frozen_delta_buffer_bytes_.shrink_to_fit();
    // HierarchicalSLB has no bulk clear() (unlike the old SemanticCache) --
    // reconstruct instead. Cheap relative to the compaction this runs
    // inside; correctness-critical since every cached node_id is now stale.
    slb_cache_ = std::make_unique<HierarchicalSLB>(
        dim_, /*enable_cross_session_l2=*/false);

    // atlas_path_ deliberately NOT reassigned -- it already names the new
    // content, having been the rename() target above. generation_ still
    // advances (an internal temp-filename disambiguator only now).
    generation_ = new_gen;

    // V4 Stage 2 follow-up: node ids/indices just changed (same reason
    // the SLB cache above gets reconstructed rather than carried over) --
    // rebuild scope_union_ from scratch against the NEW file_ rather than
    // trying to remap the old array through old_to_new. Still O(N), the
    // same class of cost this function already pays to re-index every
    // surviving node's parent_offset/first_child_offset above; still
    // under write_lock_, matching every other scope_union_ mutation site.
    rebuild_scope_union_locked();
  }
  // ── Lock released immediately. Game engine never blocked > µs. ──

  // ── Step 4: Background Cleanup ──
  // Close old file OUTSIDE the lock. drain_readers() blocks here safely
  // without freezing any game engine threads.
  compact_in_progress_.store(false, std::memory_order_release);

  old_file->close();
  old_file.reset();
  // Nothing left to remove by path -- the rename() above already
  // atomically replaced the old generation at atlas_path_.

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

  uint64_t bytes_consumed = 0;

  // WAL_RECORD_ATLAS_SCOPE targets an already-inserted mmap node by id
  // (set_node_scope() rejects delta-arena ids outright) rather than
  // reconstructing a delta-buffer entry. Buffered here and applied to
  // file_ in a second pass AFTER the main loop, rather than in-line: even
  // though the current invariant (scope-set targets are always mmap ids,
  // never delta ids) means the two record kinds can't actually target the
  // same node within one replay pass, applying scope records in-line would
  // make correctness depend on that invariant holding forever. Two-pass
  // costs nothing (replay is a rare startup-only path, not the hot path)
  // and removes the dependency entirely.
  std::vector<WalScopeRecord> pending_scope_records;
  // Same two-pass rationale as pending_scope_records. Applied in original
  // WAL (chronological) order, since supersede_node()/revoke_supersede()
  // are stateful read-modify-writes -- a later record's correctness
  // depends on earlier ones for the same node having already been applied.
  std::vector<WalSupersedeRecord> pending_supersede_records;
  // Same two-pass rationale and plain-field-set shape as
  // pending_scope_records (order-independent -- unlike supersede, later
  // records just overwrite earlier ones for the same node, which is
  // correct either way since a WAL is always applied in on-disk order
  // regardless of pass ordering here).
  std::vector<WalGovernanceRecord> pending_governance_records;
  // Same two-pass rationale as pending_scope_records -- order-independent
  // (tombstoning is a one-way flag set, so replaying the same node's
  // record twice, or out of order relative to a DIFFERENT node's record,
  // produces the same end state either way).
  std::vector<WalTombstoneRecord> pending_tombstone_records;

  while (in.good() && !in.eof()) {
    // Read WAL record header
    WalRecordHeader wal_hdr{};
    in.read(reinterpret_cast<char *>(&wal_hdr), sizeof(WalRecordHeader));
    if (in.gcount() != sizeof(WalRecordHeader))
      break; // Truncated header — stop replay
    bytes_consumed += sizeof(WalRecordHeader);

    // Bound payload_size against bytes remaining in the file BEFORE
    // trusting it for anything — including skipping an unrecognized
    // record_type below. A corrupted/adversarial payload_size must never
    // drive an out-of-bounds read or an unbounded allocation.
    uint64_t remaining = static_cast<uint64_t>(file_size) - bytes_consumed;
    if (wal_hdr.payload_size > remaining)
      break; // Declared payload exceeds file — truncated tail, stop replay

    // Read payload
    std::vector<uint8_t> payload(wal_hdr.payload_size);
    in.read(reinterpret_cast<char *>(payload.data()),
            static_cast<std::streamsize>(wal_hdr.payload_size));
    if (static_cast<uint32_t>(in.gcount()) != wal_hdr.payload_size)
      break; // Truncated payload — stop replay
    bytes_consumed += wal_hdr.payload_size;

    // Verify checksum
    uint64_t computed = hash::fnv1a_64(payload.data(), wal_hdr.payload_size);
    if (computed != wal_hdr.checksum)
      break; // Checksum mismatch — stop replay

    // Dispatch on record_type. An unrecognized type, or a recognized type
    // with an unexpected payload_size (e.g. a WAL written for a different
    // node_byte_stride_), is skipped rather than treated as fatal: forward
    // compatibility requires that a binary which doesn't understand a
    // record can still recover everything written after it.
    if (wal_hdr.record_type == WAL_RECORD_ATLAS &&
        wal_hdr.payload_size == node_byte_stride_) {
      // ── Record is valid: reconstruct delta buffer ──
      // Assign a delta ID based on current buffer position
      auto *hdr = reinterpret_cast<NodeHeader *>(payload.data());
      hdr->id = NODE_ID_DELTA_MASK | delta_node_count();

      size_t old_size = delta_buffer_bytes_.size();
      delta_buffer_bytes_.resize(old_size + node_byte_stride_);
      std::memcpy(delta_buffer_bytes_.data() + old_size, payload.data(),
                  node_byte_stride_);
    } else if (wal_hdr.record_type == WAL_RECORD_ATLAS_SCOPE &&
               wal_hdr.payload_size == sizeof(WalScopeRecord)) {
      WalScopeRecord rec{};
      std::memcpy(&rec, payload.data(), sizeof(WalScopeRecord));
      pending_scope_records.push_back(rec);
    } else if (wal_hdr.record_type == WAL_RECORD_ATLAS_SUPERSEDE &&
               wal_hdr.payload_size == sizeof(WalSupersedeRecord)) {
      WalSupersedeRecord rec{};
      std::memcpy(&rec, payload.data(), sizeof(WalSupersedeRecord));
      pending_supersede_records.push_back(rec);
    } else if (wal_hdr.record_type == WAL_RECORD_ATLAS_GOVERNANCE &&
               wal_hdr.payload_size == sizeof(WalGovernanceRecord)) {
      WalGovernanceRecord rec{};
      std::memcpy(&rec, payload.data(), sizeof(WalGovernanceRecord));
      pending_governance_records.push_back(rec);
    } else if (wal_hdr.record_type == WAL_RECORD_ATLAS_TOMBSTONE &&
               wal_hdr.payload_size == sizeof(WalTombstoneRecord)) {
      WalTombstoneRecord rec{};
      std::memcpy(&rec, payload.data(), sizeof(WalTombstoneRecord));
      pending_tombstone_records.push_back(rec);
    }
    // else: skip (payload already fully consumed above), continue replay.
  }

  // Second pass: apply scope mutations directly to the mmap node. Bounds
  // checked against the CURRENT header->node_count (reflecting whatever
  // insert()'s no-WAL mmap durability actually preserved) rather than
  // trusted blindly — an out-of-range node_id here means the insert that
  // created it didn't survive the crash even though this scope-set's WAL
  // record did (insert() has no WAL of its own, see guardrail #1.3), and
  // is skipped rather than treated as fatal, same forward-compat spirit
  // as an unrecognized record_type above.
  if (!pending_scope_records.empty()) {
    auto *header = file_->get_header();
    for (const auto &rec : pending_scope_records) {
      if (!header || rec.node_id >= header->node_count)
        continue;
      NodeHeader *node = file_->get_node(rec.node_id);
      if (!node)
        continue;
      node->scope_bitmap = rec.scope_bitmap;
    }
  }

  // Same bounds-checked, skip-on-out-of-range application, in original
  // chronological order (see pending_supersede_records' declaration for
  // why order matters here).
  if (!pending_supersede_records.empty()) {
    auto *header = file_->get_header();
    for (const auto &rec : pending_supersede_records) {
      if (!header || rec.node_id >= header->node_count)
        continue;
      NodeHeader *node = file_->get_node(rec.node_id);
      if (!node)
        continue;
      if (rec.revoke) {
        aeon::revoke_supersede(*node);
      } else {
        aeon::supersede_node(*node);
      }
    }
  }

  // Same bounds-checked, skip-on-out-of-range application as
  // pending_scope_records above.
  if (!pending_governance_records.empty()) {
    auto *header = file_->get_header();
    for (const auto &rec : pending_governance_records) {
      if (!header || rec.node_id >= header->node_count)
        continue;
      NodeHeader *node = file_->get_node(rec.node_id);
      if (!node)
        continue;
      node->governance_record_id = rec.governance_record_id;
    }
  }

  // Same bounds-checked, skip-on-out-of-range application as
  // pending_scope_records above.
  if (!pending_tombstone_records.empty()) {
    auto *header = file_->get_header();
    for (const auto &rec : pending_tombstone_records) {
      if (!header || rec.node_id >= header->node_count)
        continue;
      NodeHeader *node = file_->get_node(rec.node_id);
      if (!node)
        continue;
      aeon::tombstone_node(*node);
    }
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
