#pragma once

#include "aeon/epoch.hpp"
#include "aeon/hash.hpp"
#include "aeon/hierarchical_slb.hpp"
#include "aeon/quantization.hpp"
#include "aeon/schema.hpp"
#include "aeon/simd_impl.hpp"
#include "aeon/storage.hpp"
#include <atomic>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace aeon {

/// Configuration for Atlas construction (V4.1).
struct AtlasOptions {
  uint32_t dim = 0; ///< 0 = default 768
  uint32_t quantization_type =
      QUANT_FP32;         ///< QUANT_FP32 or QUANT_INT8_SYMMETRIC
  bool enable_wal = true; ///< WAL for crash recovery
  /// 0 = default METADATA_SIZE_DEFAULT (256). New files only -- existing
  /// files read metadata_size from the on-disk AtlasHeader, same as dim.
  /// v4-plan.md Stage 4 task 6 Phase B: lets the shared Atlas store be
  /// opened with a larger metadata field to absorb encrypted-payload
  /// overhead (nonce + base64) without shrinking the plaintext budget
  /// below what dreamer.py's existing 250-char convention already
  /// assumes for node text.
  uint32_t metadata_size = 0;
};

class Atlas {
public:
  /**
   * @brief Opens or creates an Atlas file with the given embedding dimension.
   *
   * For NEW files: creates with the specified dim (default 768).
   * For EXISTING files: dim is read from the on-disk AtlasHeader.
   *
   * @param path File path (.bin)
   * @param dim  Embedding dimensionality (new files only; 0 = default 768)
   */
  explicit Atlas(std::filesystem::path path, uint32_t dim = 0);

  /**
   * @brief Opens or creates an Atlas with explicit configuration options.
   *
   * For NEW files: uses opts.dim, opts.quantization_type to initialize the
   * on-disk layout. opts.enable_wal controls whether the Write-Ahead Log
   * is created for crash recovery.
   *
   * For EXISTING files: dim and quantization_type are read from the on-disk
   * header. enable_wal still controls WAL behavior for this session.
   *
   * @param path File path (.bin)
   * @param opts Configuration options
   */
  Atlas(std::filesystem::path path, AtlasOptions opts);
  ~Atlas();

  // Non-copyable, non-movable (owns epoch state, mutexes, mmap)
  Atlas(const Atlas &) = delete;
  Atlas &operator=(const Atlas &) = delete;

  /**
   * @brief Acquire an EBR read guard for safe zero-copy memory access.
   * While the guard is active, mmap regions will not be reclaimed.
   */
  EpochGuard acquire_read_guard();

  /**
   * @brief Lightweight node representation for search results.
   * Optimized for Python zero-copy views.
   */
  struct ResultNode {
    uint64_t id;
    float similarity;
    float centroid_preview[3]; // First 3 dims for visualization
    /// When true, the local edge Atlas produced a cold miss. The Python Shell
    /// should route to the Cloud Master Atlas for higher-fidelity navigation.
    bool requires_cloud_fetch = false;
  };

  /**
   * @brief SIMD-accelerated beam search.
   *
   * WRITE DIVERSION: If background compaction is in progress, mmap inserts
   * are diverted to the delta buffer. Reads are unaffected — they scan
   * both the current mmap AND the active delta buffer.
   *
   * @param query       dim-dimensional vector (must match Atlas dim)
   * @param beam_width  Candidates per level (1 = greedy, max = MAX_BEAM_WIDTH)
   * @param apply_csls  When true, applies CSLS hubness correction
   * @param session_id  Caller's session/tenant identity, routed to the SLB
   *                     cache's L1 (session-scoped) lookup. 0 = the default
   *                     "no session" bucket used by callers that don't (yet)
   *                     have a session concept; this is a single shared
   *                     bucket, not a bypass, so results inserted by one
   *                     session_id=0 caller ARE visible to another -- pass a
   *                     real per-tenant ID for any multi-tenant caller.
   *                     Cross-session L2 fallback is disabled (see
   *                     slb_cache_ member) until Stage 3's scoped shared
   *                     tier lands (v4-plan.md).
   * @param scope_mask   V4 Stage 2 (widen-only), V4 Stage 2 follow-up
   *                     (ancestor scope-union admission). ALL_SCOPES_VISIBLE
   *                     (default) = no filtering, unchanged pre-Stage-2
   *                     behavior. Any other value filters RESULTS to nodes
   *                     where (node.scope_bitmap & scope_mask) != 0 --
   *                     that emission-time check is still exact and
   *                     unchanged. What DID change: beam ADMISSION (which
   *                     candidates survive to be explored at the next
   *                     level) now also consults an auxiliary, RAM-only
   *                     subtree scope-union index (scope_union_, maintained
   *                     by set_node_scope()/bulk_set_node_scope()/
   *                     consolidate_subgraph(), rebuilt wholesale at open()
   *                     and after compact_mmap()) -- a candidate whose
   *                     subtree provably contains no node matching
   *                     scope_mask is deprioritized (never HARD-excluded:
   *                     if too few union-positive candidates exist to fill
   *                     the beam, the remaining slots still go to the
   *                     next-best union-negative candidates by raw score,
   *                     identical to pre-follow-up behavior -- this can
   *                     only improve recall, never regress it below the
   *                     widen-only baseline). This is NOT the per-candidate
   *                     "own bit as a bonus" steering that was tried and
   *                     measured actively harmful (see navigate_internal()'s
   *                     doc comment) -- a subtree union bit is a sound
   *                     over-approximation of "does this subtree contain a
   *                     match," unlike a leaf's own bit, which says nothing
   *                     about its descendants under deferred propagation.
   *                     The union index is best-effort and directional
   *                     (monotonic OR-only: a scope NARROWED via
   *                     set_node_scope() leaves ancestor unions stale-wide,
   *                     safe for steering since emission-time filtering is
   *                     still exact) -- see scope_union_'s member comment.
   *                     Delta-buffer candidates always have scope_bitmap
   *                     == 0 (set_node_scope() rejects delta ids), so they
   *                     are excluded entirely whenever filtering is active.
   */
  std::vector<ResultNode> navigate(std::span<const float> query,
                                   uint32_t beam_width = 1,
                                   bool apply_csls = false,
                                   uint64_t session_id = 0,
                                   uint64_t scope_mask = ALL_SCOPES_VISIBLE);

  /**
   * @brief Returns a node's direct children.
   *
   * @param scope_mask  V4 Stage 2 task 2: same semantics as navigate()'s
   *                     scope_mask -- ALL_SCOPES_VISIBLE (default) means
   *                     no filtering. This is the enforcement point for
   *                     the Atlas->Trace->Atlas "graph-expansion boundary"
   *                     the plan calls out: a caller that reached
   *                     parent_id via a Trace event's atlas_id (crossing
   *                     from episodic memory back into the concept graph)
   *                     must not enumerate children outside its own
   *                     scope just because the STARTING node was
   *                     legitimately theirs -- get_children() itself is
   *                     otherwise completely unscoped, unlike navigate().
   */
  std::vector<ResultNode> get_children(uint64_t parent_id,
                                       uint64_t scope_mask = ALL_SCOPES_VISIBLE);

  /**
   * @brief Inserts a new node as a child of parent_id.
   *
   * WRITE DIVERSION: If compact_in_progress_, silently falls through to
   * insert_delta() to prevent data loss during background compaction.
   *
   * Durability: writes go directly into the mmap'd file (MAP_SHARED), so
   * they are visible to any other process re-mapping this file and survive
   * a crash of THIS process -- the data lives in the OS page cache, not
   * this process's heap. There is no per-insert WAL entry or msync (either
   * would seriously regress insert latency). The remaining exposure is
   * narrower than "no durability at all": an OS crash or power loss before
   * the kernel's own writeback timer flushes the dirty page. Call sync()
   * for an explicit durability checkpoint (e.g. before a controlled
   * shutdown); compact_mmap() calls it automatically before deleting the
   * old generation file, which is the one place skipping it would be a
   * real irrecoverable-data-loss risk rather than a bounded window
   * (v4-plan.md guardrail #1.3).
   *
   * @param session_id  Caller's session/tenant identity. Currently accepted
   *                     for API/C-API consistency but not yet used to
   *                     populate the SLB cache on insert (only navigate()
   *                     populates it today) -- this is the plumbing point
   *                     Stage 1 will use to assign write-time scope from
   *                     authenticated session context (v4-plan.md).
   */
  uint64_t insert(uint64_t parent_id, std::span<const float> vector,
                  std::string_view metadata, uint64_t session_id = 0);

  /**
   * @brief Explicitly flush pending mmap writes to disk. See insert()'s
   * doc comment for the durability model this closes the gap on.
   */
  void sync();

  /**
   * @brief Inserts into the flat byte arena delta buffer.
   * Thread-safe. δ-node ID has MSB=1 to distinguish from mmap nodes.
   */
  uint64_t insert_delta(std::span<const float> vector,
                        std::string_view metadata);

  size_t prune_delta_tail(size_t n);

  size_t size() const;

  void load_context(std::span<const uint64_t> node_ids, uint64_t session_id = 0);

  /**
   * @brief Remove a session's L1 SLB cache entry and free its memory.
   *
   * Forwards to HierarchicalSLB::drop_session(). Real implementation (not a
   * stub) as of v4-plan.md Stage 0 -- callers (e.g. NPC despawn via the
   * C-API) can rely on this actually freeing memory rather than being a
   * validated no-op.
   *
   * @return true if the session existed and was removed.
   */
  bool drop_session(uint64_t session_id);

  /**
   * @brief Sets a node's scope_bitmap in place (V4 Stage 1/2 prerequisite).
   *
   * This is the only supported write path for scope_bitmap after a node
   * has already been inserted (insert()/insert_delta() always create new
   * nodes with scope_bitmap = 0 -- see their doc comments). WAL-protected
   * (WAL_RECORD_ATLAS_SCOPE): unlike insert()'s mmap-direct durability
   * model, a scope mutation is logged to the WAL before being applied, so
   * it survives a crash even if the OS hasn't flushed the mmap page yet.
   *
   * Mmap nodes only -- throws std::invalid_argument for a delta-arena node
   * id (MSB set), since a delta node's id is replaced when compact_mmap()
   * promotes it, and a scope set against the old id would be silently
   * lost. Throws std::runtime_error if compaction is in progress (the node
   * may be mid-copy to the new generation file) or if node_id is invalid.
   *
   * V4 Stage 2 follow-up: also incrementally updates the auxiliary
   * scope_union_ index -- pushes scope_bitmap's new bits up through
   * parent_offset ancestors (a bounded worklist fixpoint, not a naive
   * single reverse pass, since consolidate_subgraph() can rewire a
   * surviving child's parent_offset to a NUMERICALLY LATER summary node,
   * breaking the "parent index < child index" ordering a simpler pass
   * would assume). Monotonic OR-only: if this call NARROWS an existing
   * scope_bitmap, ancestor union entries are left stale-wide (still
   * claim the old, now-removed bit) rather than recomputed -- safe for a
   * steering hint (over-inclusion just means a missed pruning
   * opportunity, never a wrong result, since emission-time filtering in
   * navigate() is still exact against the real scope_bitmap). This closes
   * the KNOWN GAP this comment used to describe (a consolidate_subgraph()
   * summary's union going stale after a source's scope changed) --
   * propagate_scope_union_locked() is called here as one node like any
   * other; see that function's comment for the general mechanism.
   */
  void set_node_scope(uint64_t node_id, uint64_t scope_bitmap);

  /// Reads a node's current scope_bitmap. Mmap nodes only -- see
  /// set_node_scope()'s doc comment for the delta-id restriction.
  uint64_t get_node_scope(uint64_t node_id) const;

  /**
   * @brief Reversibly excludes a node from beam search results (V4 Stage 2
   * prerequisite for the "superseded fragments are excluded from
   * navigate()" gate).
   *
   * Applies schema.hpp's supersede_node() (branchless: stashes the real
   * hub_penalty, overwrites it with TOMBSTONE_PENALTY -- the same
   * exclusion mechanism tombstoning uses) to an already-inserted mmap
   * node. WAL-protected (WAL_RECORD_ATLAS_SUPERSEDE); idempotent (a
   * second call is a no-op, matching supersede_node()'s own guard).
   *
   * Same restrictions as set_node_scope(): mmap nodes only (throws
   * std::invalid_argument for a delta-arena id), throws
   * std::runtime_error if compaction is in progress or the id is invalid.
   */
  void supersede_node(uint64_t node_id);

  /// Reverses a prior supersede_node() call -- restores the real
  /// hub_penalty (schema.hpp's revoke_supersede(), which correctly leaves
  /// hub_penalty at TOMBSTONE_PENALTY if the node was ALSO tombstoned
  /// since being superseded). Same restrictions as supersede_node().
  void revoke_node_supersede(uint64_t node_id);

  /// Reads whether a node currently has NODE_FLAG_SUPERSEDED set. Mmap
  /// nodes only.
  bool is_node_superseded(uint64_t node_id) const;

  /**
   * @brief Sets a node's governance_record_id in place (V4 Stage 4 task 1)
   * -- an opaque link into the Stage 3/4 control plane (e.g. a Postgres
   * governance-record primary key), or a version-lineage pointer. Stage 1
   * allocated this NodeHeader field but never gave it a writer; this is
   * that writer. Same restrictions, locking, and WAL-before-mutate
   * ordering as set_node_scope() (WAL_RECORD_ATLAS_GOVERNANCE): mmap nodes
   * only, throws std::invalid_argument for a delta-arena id, throws
   * std::runtime_error if compaction is in progress or node_id is
   * invalid.
   */
  void set_node_governance_id(uint64_t node_id, uint64_t governance_record_id);

  /// Reads a node's current governance_record_id. Mmap nodes only -- see
  /// set_node_governance_id()'s doc comment for the delta-id restriction.
  uint64_t get_node_governance_id(uint64_t node_id) const;

  /**
   * @brief Reads a node's metadata string back out (V4 Stage 4 task 2 --
   * promotion's mint-and-recontextualize needs to read a source fragment's
   * text before it can classify/re-embed/copy it into the shared store).
   * insert()/insert_delta() have always WRITTEN this field
   * (node_metadata_q(), schema.hpp) but nothing ever read it back until
   * now -- found while wiring the promotion pipeline, the same class of
   * gap as governance_record_id's missing writer in Stage 1 (allocated,
   * never given the other half of its API). Works for BOTH mmap and
   * delta-arena node ids (unlike the scope/governance accessors) --
   * promotion needs to read fresh delta-buffer content too, e.g. a
   * same-turn admission that hasn't been compacted yet, and metadata is a
   * plain byte-copy read with no WAL/version-lineage implications, so
   * there's no correctness reason to restrict it the way mutation
   * primitives are restricted.
   */
  std::string get_node_metadata(uint64_t node_id) const;

  /**
   * @brief Reads a node's full centroid vector back out, dequantized to
   * FP32 if this Atlas is INT8-quantized (V4 Stage 4 task 2 -- promotion
   * needs the SOURCE node's actual vector to insert a copy into the
   * shared store; query()/get_children() only ever return a 3-float
   * preview, not the full dim_-length vector, so this is a genuinely new
   * accessor, not a rename of an existing one). Same mmap-and-delta-arena
   * support as get_node_metadata() and the same reasoning for it: a
   * same-turn admission not yet compacted is still promotable content.
   */
  std::vector<float> get_node_centroid(uint64_t node_id) const;

  /**
   * @brief Lists live (non-tombstoned) node ids whose scope_bitmap
   * overlaps scope_mask (V4 Stage 4 task 1 -- the console's "list-by-scope"
   * primitive). No new scan mechanism: a flat pass over
   * MemoryFile::get_node(i), the same one tombstone_count()/compact_mmap()
   * already use, EBR-guarded like get_node_scope() since this is a
   * control-plane-facing read that can run concurrently with writers.
   * Superseded nodes ARE included (supersession is reversible, so they're
   * still live data an admin console needs to see -- callers can check
   * is_node_superseded() per id if they need to distinguish); tombstoned
   * nodes are excluded (logically deleted, pending physical reclaim at the
   * next compaction). Delta-buffer candidates are never included --
   * set_node_scope() rejects delta ids outright, so they always have
   * scope_bitmap == 0.
   *
   * scope_mask == ALL_SCOPES_VISIBLE is special-cased to mean "every live
   * node, no filtering" -- matching navigate()'s documented semantics for
   * the same sentinel. This is NOT the same as treating it as an ordinary
   * mask: unscoped nodes default to scope_bitmap == 0, and
   * 0 & ALL_SCOPES_VISIBLE == 0 (falsy), so a plain AND check would
   * exclude every unscoped node from a query meant to return "everything"
   * -- the exact inverse of the intended result (found via review before
   * this ever shipped, v4-plan.md Stage 4).
   */
  std::vector<uint64_t> list_nodes_by_scope(uint64_t scope_mask) const;

  /**
   * @brief Applies many scope_bitmap updates under a single lock/WAL-flush
   * pass (V4 Stage 4 task 1 -- the console's "bulk bit remap" primitive),
   * rather than N separate set_node_scope() calls each paying their own
   * lock-acquisition and fsync cost. All-or-nothing: every (node_id,
   * scope_bitmap) pair is validated BEFORE any node is mutated, so an
   * invalid id anywhere in the batch throws without partially applying the
   * rest. Semantically identical to N sequential set_node_scope() calls
   * otherwise -- reuses WAL_RECORD_ATLAS_SCOPE (replay doesn't distinguish
   * which call produced a given record), including the scope_union_
   * propagation each individual set_node_scope() call now also performs
   * (V4 Stage 2 follow-up) -- applied once per update in Pass 3, after the
   * node mutation, same ordering set_node_scope() uses.
   */
  void bulk_set_node_scope(
      const std::vector<std::pair<uint64_t, uint64_t>> &updates);

  /**
   * @brief Logically deletes a single, already-inserted mmap node by id
   * (V4 Stage 4 task 5/6 -- the console/erasure-workflow "delete" primitive).
   *
   * Applies schema.hpp's tombstone_node() (flags |= NODE_FLAG_TOMBSTONE,
   * hub_penalty = TOMBSTONE_PENALTY -- the same branchless beam-exclusion
   * mechanism consolidate_subgraph() already uses on old nodes) directly to
   * an arbitrary node id. WAL-protected (WAL_RECORD_ATLAS_TOMBSTONE);
   * idempotent (schema.hpp's tombstone_node() has no guard against a second
   * call, but setting the same flag bit and the same penalty value twice is
   * itself a no-op).
   *
   * Unlike supersede_node(), this is TERMINAL -- there is no
   * revoke_node_tombstone(): schema.hpp's tombstone_node() does not stash
   * the prior hub_penalty (only supersede_node() does), so there is nothing
   * to restore. If the node was ALSO NODE_FLAG_SUPERSEDED, this leaves that
   * bit set (revoke_node_supersede() still works afterward, correctly
   * leaving hub_penalty at TOMBSTONE_PENALTY per its own doc comment) --
   * calling this on a superseded node does not lose the stashed
   * saved_hub_penalty, it simply becomes moot.
   *
   * This is a logical delete only: the node's bytes remain physically
   * present in the mmap file until the next compact_mmap() reclaims them
   * (see tombstone_count()) -- callers needing to state a guarantee about
   * physical erasure (e.g. the console's erasure workflow) must account for
   * that gap explicitly, not imply it from this call succeeding.
   *
   * Same restrictions as set_node_scope()/supersede_node(): mmap nodes only
   * (throws std::invalid_argument for a delta-arena id -- a delta node that
   * hasn't been compacted yet has no durable identity to tombstone against;
   * discard it at the caller's own layer instead), throws
   * std::runtime_error if compaction is in progress or the id is invalid.
   */
  void tombstone_node(uint64_t node_id);

  /// Returns the embedding dimensionality of this Atlas instance.
  uint32_t dim() const noexcept { return dim_; }

  /// Returns the metadata field size (bytes) of this Atlas instance --
  /// v4-plan.md Stage 4 task 6 Phase B: callers writing an encoded
  /// (nonce+ciphertext) payload into the metadata field need this to
  /// length-check BEFORE calling insert(), since insert() silently
  /// truncates at metadata_size() - 1 rather than raising.
  uint32_t metadata_size() const noexcept { return metadata_size_; }

  /// Returns the node byte stride of this Atlas instance.
  size_t node_byte_stride() const noexcept { return node_byte_stride_; }

  // ═══════════════════════════════════════════════════════════════════════
  // Dreaming Kernel — Memory Consolidation for Edge/Mobile Devices
  // ═══════════════════════════════════════════════════════════════════════

  /**
   * @brief Atomically consolidates a subgraph into a single summary node.
   *
   * Dreaming Process: insert summary → re-wire children → tombstone old nodes.
   * Thread-safe: acquires exclusive write lock.
   *
   * WRITE DIVERSION: If compact_in_progress_, throws runtime_error.
   * Consolidation during compaction is logically unsafe.
   *
   * REQUIRES every id in `old_node_ids` to share the IDENTICAL scope_bitmap
   * (V4 Stage 5 task 1 -- advisor review). The summary's own scope is the
   * OR of its sources' bits (so consolidated content stays visible to
   * everyone who could see any source), which is exactly the failure mode
   * this precondition closes: consolidating a scope-0x1 node together with
   * a scope-0x2 node would silently mint a scope-0x3 summary readable by
   * BOTH scopes -- widening visibility as a side effect of a storage-GC
   * operation, with nothing else in this file positioned to catch it (the
   * caller's clustering logic is Python, a second caller could easily miss
   * this, and the kernel is the one layer that already owns scope_bitmap
   * as ground truth). Throws std::invalid_argument naming the offending id
   * if scopes differ -- checked in Phase 1, before any node is mutated, so
   * a rejected call leaves the Atlas completely unchanged. Private-store
   * nodes (scope_bitmap == 0 uniformly, scoping is a shared-tier-only
   * concept) trivially satisfy this and are unaffected by this change --
   * verified via the existing single-tenant Dreaming tests, which never
   * call set_node_scope() on their nodes.
   *
   * Does NOT check `governance_record_id`/subject attribution -- the
   * kernel has no visibility into which Postgres `subject_id` a
   * governance_record_id resolves to (that mapping lives entirely in the
   * control plane, control_plane/db.py). A caller consolidating
   * PROMOTED shared-store nodes across different subjects would violate
   * task 6's one-subject-per-node invariant even though scope matches;
   * that check must be enforced by the Python-layer caller (the shared-
   * tier Dreamer, Stage 5 task 1) BEFORE it ever calls this function, by
   * resolving each candidate's subject_id via governance_db and refusing
   * to cluster across subjects.
   */
  uint64_t consolidate_subgraph(std::span<const uint64_t> old_node_ids,
                                std::span<const float> summary_vector,
                                std::string_view summary_meta);

  /**
   * @brief Background Shadow Compaction (V4.0 — stutter-free).
   *
   * Uses the Redis BGSAVE double-buffer pattern to avoid holding an
   * exclusive lock during the multi-second file copy:
   *
   *   Step 1 (µs freeze):  Swap delta buffers, snapshot node_count.
   *   Step 2 (background):  Copy live nodes + frozen deltas → new gen file.
   *   Step 3 (µs freeze):  Hot-swap MemoryFile, clear frozen buffer.
   *   Step 4 (background):  Close + delete old generation file.
   *
   * Game engines can continue inserting into the active delta_buffer_
   * while Step 2 copies gigabytes of data.
   */
  void compact_mmap();

  size_t tombstone_count() const;

private:
  /// Template-dispatched beam search inner loop (CSLS branch hoisted).
  template <bool ApplyCSLS>
  std::vector<ResultNode> navigate_internal(std::span<const float> query,
                                            uint32_t beam_width,
                                            uint64_t session_id,
                                            uint64_t scope_mask);

  /// Count delta nodes in a flat byte arena.
  size_t delta_node_count() const noexcept;
  size_t delta_node_count(const std::vector<uint8_t> &arena) const noexcept;

  /// Get a NodeHeader* from the delta byte arena at the given index.
  NodeHeader *delta_get_node(size_t index) noexcept;
  const NodeHeader *delta_get_node(size_t index) const noexcept;
  const NodeHeader *delta_get_node(const std::vector<uint8_t> &arena,
                                   size_t index) const noexcept;

  /**
   * @brief Full O(N) rebuild of scope_union_ from scratch (V4 Stage 2
   * follow-up -- closes the recall gap documented in navigate()'s
   * scope_mask doc comment). Called at Atlas construction (after
   * replay_wal(), so it reflects final on-disk state) and after
   * compact_mmap() installs a new generation (fresh indices). Caller must
   * already hold write_mutex_ exclusively, or be in a context where no
   * concurrent access is possible yet (the constructor, before this Atlas
   * is reachable by any other thread).
   *
   * Uses a worklist fixpoint, not a single reverse index pass: a naive
   * "process indices N-1 down to 0, OR each node's accumulated union into
   * its parent" pass would be O(N) and correct IF parent index were
   * always < child index -- true for every insert()-created edge, but
   * consolidate_subgraph() rewires a SURVIVING child's parent_offset to
   * point at the new summary node, which has a NUMERICALLY LATER index
   * than the child. A single reverse pass would silently miss that
   * child's contribution to the summary (and anything further up).
   * Termination is guaranteed regardless of edge ordering: scope bits
   * only ever grow (monotonic OR, ≤64 bits/node), so the worklist can
   * only push a bounded number of times.
   */
  void rebuild_scope_union_locked();

  /**
   * @brief Incrementally pushes node_idx's own scope_bitmap (already
   * mutated by the caller) up through parent_offset ancestors into
   * scope_union_ (V4 Stage 2 follow-up). Caller must already hold
   * write_mutex_ exclusively. Same worklist-fixpoint mechanism as
   * rebuild_scope_union_locked(), seeded from a single node -- correctly
   * handles consolidate_subgraph()'s child-to-later-summary rewiring
   * because it re-queues any ancestor whose union actually changes,
   * regardless of index direction. Monotonic OR-only: never clears a bit
   * an ancestor already claims, even if this specific call narrowed
   * node_idx's own scope_bitmap -- see set_node_scope()'s doc comment for
   * why that staleness is safe. Defensively grows scope_union_ if
   * node_idx is out of its current range (should not happen given every
   * mutator keeps it sized to node_count, but this is cheap insurance
   * against relying on that perfectly).
   */
  void propagate_scope_union_locked(uint64_t node_idx);

  // ─── Layout constants (set once at construction, never change) ───
  uint32_t dim_ = 0;
  uint32_t metadata_size_ = METADATA_SIZE_DEFAULT;
  size_t node_byte_stride_ = 0;
  uint32_t quantization_type_ = QUANT_FP32; // V4.1 Phase 3: cached from header
  bool enable_wal_ = true;                  // V4.1: WAL toggle for benchmarking

  // ─── Core state ───
  // mutable: enter_guard() bumps an internal atomic reader count, not the
  // object's logical state -- needed so const read paths (get_node_scope())
  // can still take a proper EBR guard instead of reading through file_
  // unprotected while a concurrent compact_mmap() retires the mmap region.
  mutable EpochManager epoch_mgr_;
  std::unique_ptr<storage::MemoryFile> file_;
  std::filesystem::path atlas_path_;
  uint64_t generation_ = 0; ///< Generational file naming counter

  // ─── Concurrency ───
  // mutable for the same reason as epoch_mgr_ above: a shared_lock from a
  // const read path is a read, not a mutation of Atlas's logical state.
  mutable std::shared_mutex write_mutex_; ///< RW lock: shared reads, exclusive writes
  mutable std::shared_mutex delta_mutex_;

  // ─── Flat byte arena delta buffers ───
  // Contiguous memory:
  // [NodeHeader|centroid|metadata|pad][NodeHeader|centroid|metadata|pad]...
  // SIMD prefetcher can stream through without chasing heap pointers.
  std::vector<uint8_t> delta_buffer_bytes_;
  std::vector<uint8_t>
      frozen_delta_buffer_bytes_; ///< Frozen snapshot for bg compaction

  // ─── Scope union index (V4 Stage 2 follow-up) ───
  // RAM-only, NOT part of the on-disk format, NOT NodeHeader::scope_bitmap
  // (deliberately a separate array -- see the rejected alternative below).
  // scope_union_[i] = OR of node i's own scope_bitmap and every
  // descendant's scope_bitmap, indexed identically to file_->get_node(i)
  // (mmap node ids ARE their array index throughout this codebase).
  // Consulted ONLY by navigate()'s beam-ADMISSION step as a steering hint
  // (a candidate whose union doesn't overlap scope_mask is deprioritized,
  // never hard-excluded -- see navigate()'s scope_mask doc comment);
  // emission-time filtering still checks the real per-node scope_bitmap
  // exclusively and is unaffected by anything in this array.
  //
  // REJECTED ALTERNATIVE: reusing NodeHeader::scope_bitmap itself to mean
  // "subtree union" for internal nodes (own-scope for leaves) would need
  // zero new storage, but every node in this Atlas holds real content
  // (there is no routing-only/leaf-only distinction), so a node's OWN
  // scope assignment and "what its descendants need" would collide in one
  // field -- get_node_scope()/list_nodes_by_scope() and, critically, the
  // admin console's _require_scope_containment()/erasure DEK lookup
  // (server.py, shell/aeon_py) all read scope_bitmap as ground truth for
  // AUTHORIZATION. Polluting it with a descendant's bit would let an
  // admin's grant over scope Y wrongly authorize acting on an ancestor
  // node whose own real content belongs to a different scope entirely,
  // purely because one of its descendants happens to be scoped to Y --
  // a real privilege-escalation risk, not a theoretical one, given how
  // carefully Stage 4 had to fix containment-vs-overlap bugs in exactly
  // this authorization path. Keeping the union in its own array, read
  // only by navigate()'s internal admission heuristic, makes this
  // impossible by construction.
  std::vector<uint64_t> scope_union_;

  // ─── Compaction state ───
  std::atomic<bool> compact_in_progress_{false};

  // ─── Write-Ahead Log (V4.1) ───
  // Separate mutex to avoid blocking game engine threads on disk I/O.
  // Lock ordering: serialize (no lock) → wal_mutex_ → delta_mutex_
  std::mutex wal_mutex_;
  std::ofstream wal_stream_;
  std::filesystem::path wal_path_;

  /// Open or create the WAL file for append-only writes.
  void open_wal();

  /// Replay WAL records to reconstruct delta_buffer_bytes_ after crash.
  void replay_wal();

  /// Truncate (delete) the WAL file after successful compaction.
  void truncate_wal();

  // ─── Cache ───
  // Session-aware L1/L2 semantic cache (v4-plan.md Stage 0). Constructed in
  // the .cpp constructor body once dim_ is known from the on-disk header
  // (unique_ptr because HierarchicalSLB contains std::shared_mutex members
  // and is neither copyable nor movable, so it can't be a plain member
  // initialized before dim_ is resolved). Cross-session L2 fallback is
  // disabled at construction: HierarchicalSLB's global cache has no
  // per-entry scope predicate yet, so until Stage 3's scoped shared tier
  // lands, allowing L2 hits across sessions would leak one tenant's cached
  // results to another. This intentionally gives up the (currently unused)
  // cross-session cold-start benefit HierarchicalSLB was designed for.
  std::unique_ptr<HierarchicalSLB> slb_cache_;
};

} // namespace aeon
