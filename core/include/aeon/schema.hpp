#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace aeon {

// ═══════════════════════════════════════════════════════════════════════════
// Magic & Version
// ═══════════════════════════════════════════════════════════════════════════

/// Magic bytes: "ATLAS_01" in hex.
constexpr uint64_t ATLAS_MAGIC = 0x41544C41535F3031;

/// File format version. Bumped to 2 for dynamic dimensionality (V4.0).
constexpr uint64_t ATLAS_VERSION = 2;

// ═══════════════════════════════════════════════════════════════════════════
// Compile-Time Constants (Defaults & Limits)
// ═══════════════════════════════════════════════════════════════════════════

/// Default embedding dimensionality. Used when creating a NEW Atlas file
/// without an explicit dim parameter. Matches all-MiniLM-L12-v2 output.
constexpr uint32_t EMBEDDING_DIM_DEFAULT = 768;

/// Alias used by HierarchicalSLB and other components that require a
/// fixed compile-time embedding dimension for cache-line-aligned arrays.
constexpr uint32_t EMBEDDING_DIM = EMBEDDING_DIM_DEFAULT;

/// Maximum result set size returned by navigate().
constexpr size_t TOP_K_LIMIT = 50;

/// Maximum beam width for beam search navigate (stack-allocated).
constexpr uint32_t MAX_BEAM_WIDTH = 16;

/// Sentinel scope_mask value for Atlas::navigate() meaning "no scope
/// filtering -- return results regardless of scope_bitmap" (V4 Stage 2).
/// The default, preserving pre-Stage-2 behavior for existing callers. Any
/// other value is treated as a real filter: a node is visible only if
/// (node.scope_bitmap & scope_mask) != 0. Deliberately distinct from 0
/// (which, as a filter, would mean "caller has zero granted scopes" --
/// matching nothing, including unscoped nodes, since 0 & anything == 0)
/// so "no filtering" and "filter to nothing" are never confusable.
constexpr uint64_t ALL_SCOPES_VISIBLE = UINT64_MAX;

/// Default SLB similarity threshold for cache hit classification.
constexpr float SLB_HIT_THRESHOLD = 0.85f;

/// Default metadata size in bytes (null-terminated UTF-8).
constexpr uint32_t METADATA_SIZE_DEFAULT = 256;

/// CPU cache line size for alignment (AVX-512 / ARM NEON friendly).
constexpr size_t CACHE_LINE_SIZE_NODE = 64;

// ═══════════════════════════════════════════════════════════════════════════
// Node Flags — bitfield stored in NodeHeader::flags (uint16_t)
// ═══════════════════════════════════════════════════════════════════════════

/// Tombstone: node consolidated by the Dreaming process.
/// hub_penalty overwritten to TOMBSTONE_PENALTY for branchless SIMD
/// elimination.
constexpr uint16_t NODE_FLAG_TOMBSTONE = 1 << 0;

/// Summary: created by consolidate_subgraph() to replace older verbose nodes.
constexpr uint16_t NODE_FLAG_SUMMARY = 1 << 1;

/// Superseded: reversibly excluded from beam search (V4 Stage 1). Unlike
/// NODE_FLAG_TOMBSTONE, this is undoable — see supersede_node()/
/// revoke_supersede() below, which stash/restore the real hub_penalty in
/// NodeHeader::saved_hub_penalty rather than destroying it.
constexpr uint16_t NODE_FLAG_SUPERSEDED = 1 << 2;

/// O(1) branchless tombstone penalty.
/// score = cosine_similarity - hub_penalty → tombstoned nodes score ≈ -1e9f.
constexpr float TOMBSTONE_PENALTY = 1e9f;

// ═══════════════════════════════════════════════════════════════════════════
// Quantization Type Constants (V4.1 Phase 3)
// ═══════════════════════════════════════════════════════════════════════════

/// FP32 (unquantized) — default for all existing Atlas files.
constexpr uint32_t QUANT_FP32 = 0;

/// INT8 Symmetric Quantization: scale = max(|v|) / 127, zero_point = 0.
/// 4× spatial compression vs FP32.
constexpr uint32_t QUANT_INT8_SYMMETRIC = 1;

// ═══════════════════════════════════════════════════════════════════════════
// AtlasHeader — 64-byte file header with dynamic layout fields
// ═══════════════════════════════════════════════════════════════════════════

/**
 * @brief Global file header for the memory-mapped region.
 *
 * V4.0 CHANGE: The previously unused `reserved` bytes now store the
 * embedding dimension, metadata size, and computed node byte stride.
 * This enables a single binary to open Atlas files of ANY dimensionality
 * (384 for mobile, 768 for MiniLM, 1536 for OpenAI).
 */
struct alignas(64) AtlasHeader {
  uint64_t magic;      // 0x00: Magic number identifier
  uint64_t version;    // 0x08: Format version (now 2)
  uint64_t node_count; // 0x10: Current number of actively used nodes
  uint64_t capacity;   // 0x18: Total capacity (allocated slots)
  uint32_t dim;        // 0x20: Embedding dimensionality (e.g., 384, 768, 1536)
  uint32_t metadata_size;     // 0x24: Metadata block size in bytes
  uint64_t node_byte_stride;  // 0x28: Byte stride per node (64-byte aligned)
  uint32_t quantization_type; // 0x30: 0=FP32, 1=INT8_SYMMETRIC
  uint8_t reserved[12];       // 0x34: Future use — zeroed on creation
};

static_assert(sizeof(AtlasHeader) == 64,
              "AtlasHeader must be exactly 64 bytes");
static_assert(std::is_standard_layout_v<AtlasHeader>);
static_assert(std::is_trivially_copyable_v<AtlasHeader>);

// ═══════════════════════════════════════════════════════════════════════════
// NodeHeader — 64-byte fixed header (cache-line 0)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * @brief Fixed-size header for every node in the mmap and delta arenas.
 *
 * The centroid embedding and metadata follow IMMEDIATELY after this header
 * in the byte stream. Their sizes are determined at Atlas open time from
 * AtlasHeader::dim and AtlasHeader::metadata_size.
 *
 * Layout in the byte arena (per node_byte_stride):
 *
 *   [NodeHeader: 64 bytes][centroid: dim*4 bytes][metadata: metadata_size
 * bytes][padding → stride]
 *
 * TOMBSTONE INVARIANT: When consolidated, flags |= NODE_FLAG_TOMBSTONE and
 * hub_penalty = TOMBSTONE_PENALTY (1e9f). Beam search eliminates it
 * branchlessly: score = cosine_sim - 1e9f ≈ -1e9f.
 *
 * V4 STAGE 1: the former `reserved[20]` is now three named fields
 * (8 + 8 + 4 = 20 bytes, filling the region exactly — no further per-node
 * field fits without a side structure). scope_bitmap defaults to 0
 * ("no scope membership", the most restrictive default) on every new node;
 * there is no scope-assignment authority yet (that's Stage 3/4's control
 * plane), and by design insert()/insert_delta() accept no caller-supplied
 * scope parameter — the only path by which it can ever be populated is
 * authenticated session/org context once that authority exists.
 */
struct alignas(64) NodeHeader {
  uint64_t id;                 // 0x00: Unique ID (MSB=1 for delta nodes)
  uint64_t parent_offset;      // 0x08: Byte offset to parent (0 if root)
  uint64_t first_child_offset; // 0x10: Byte offset to first child
  uint16_t child_count;        // 0x18: Number of contiguous children
  uint16_t flags; // 0x1A: NODE_FLAG_TOMBSTONE|SUMMARY|SUPERSEDED
  float hub_penalty;           // 0x1C: CSLS penalty or TOMBSTONE_PENALTY
  float quant_scale;           // 0x20: scale = max(|v|) / 127.0f (0.0 for FP32)
  float quant_zero_point;      // 0x24: always 0.0 for symmetric quantization
  uint64_t scope_bitmap;         // 0x28: up to 64 named scopes (0 = unscoped)
  uint64_t governance_record_id; // 0x30: opaque link into control-plane / version lineage
  float saved_hub_penalty;       // 0x38: stashed hub_penalty while NODE_FLAG_SUPERSEDED
  // 0x3C–0x40: compiler padding (struct size rounded up to alignas(64))
};

static_assert(sizeof(NodeHeader) == 64,
              "NodeHeader must be exactly 64 bytes (1 cache line)");
static_assert(std::is_standard_layout_v<NodeHeader>);
static_assert(std::is_trivially_copyable_v<NodeHeader>);
static_assert(offsetof(NodeHeader, scope_bitmap) == 0x28);
static_assert(offsetof(NodeHeader, governance_record_id) == 0x30);
static_assert(offsetof(NodeHeader, saved_hub_penalty) == 0x38);

// ═══════════════════════════════════════════════════════════════════════════
// Alignment & Stride Utilities
// ═══════════════════════════════════════════════════════════════════════════

/// Round `size` up to the nearest multiple of `alignment`.
/// alignment MUST be a power of 2.
constexpr size_t align_up(size_t size, size_t alignment) noexcept {
  return (size + alignment - 1) & ~(alignment - 1);
}

/**
 * @brief Compute the 64-byte-aligned byte stride for a single node (FP32).
 *
 * stride = align_up(sizeof(NodeHeader) + dim * sizeof(float) + metadata_size,
 * 64)
 */
constexpr size_t compute_node_stride(uint32_t dim,
                                     uint32_t metadata_size) noexcept {
  return align_up(sizeof(NodeHeader) + dim * sizeof(float) + metadata_size,
                  CACHE_LINE_SIZE_NODE);
}

/**
 * @brief Compute the 64-byte-aligned byte stride with quantization awareness.
 *
 * @param quant_type  QUANT_FP32 → dim * sizeof(float)
 *                    QUANT_INT8_SYMMETRIC → dim * sizeof(int8_t)
 *
 * INT8 Example:
 *   dim=768, meta=256 → align_up(64 + 768 + 256, 64) = align_up(1088, 64) =
 *   1088  (vs 3392 for FP32 — 3.1× compression)
 */
constexpr size_t compute_node_stride(uint32_t dim, uint32_t metadata_size,
                                     uint32_t quant_type) noexcept {
  size_t payload_size = (quant_type == QUANT_INT8_SYMMETRIC)
                            ? dim * sizeof(int8_t)
                            : dim * sizeof(float);
  return align_up(sizeof(NodeHeader) + payload_size + metadata_size,
                  CACHE_LINE_SIZE_NODE);
}

// ═══════════════════════════════════════════════════════════════════════════
// NodeHeader Inline Accessors (zero-overhead pointer arithmetic)
// ═══════════════════════════════════════════════════════════════════════════

/// Returns a pointer to the centroid embedding (starts at byte 64 of the node).
/// The returned span has exactly `dim` elements.
inline float *node_centroid(NodeHeader *hdr) noexcept {
  return reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(hdr) +
                                   sizeof(NodeHeader));
}
inline const float *node_centroid(const NodeHeader *hdr) noexcept {
  return reinterpret_cast<const float *>(
      reinterpret_cast<const uint8_t *>(hdr) + sizeof(NodeHeader));
}

/// Returns a pointer to the INT8 centroid embedding (starts at byte 64 of the
/// node). Used only when AtlasHeader::quantization_type ==
/// QUANT_INT8_SYMMETRIC.
inline int8_t *node_centroid_int8(NodeHeader *hdr) noexcept {
  return reinterpret_cast<int8_t *>(reinterpret_cast<uint8_t *>(hdr) +
                                    sizeof(NodeHeader));
}
inline const int8_t *node_centroid_int8(const NodeHeader *hdr) noexcept {
  return reinterpret_cast<const int8_t *>(
      reinterpret_cast<const uint8_t *>(hdr) + sizeof(NodeHeader));
}

/// Returns a pointer to the metadata string (starts after the centroid).
/// @param payload_bytes  dim * sizeof(float) for FP32, dim * sizeof(int8_t)
/// for INT8.
inline char *node_metadata(NodeHeader *hdr, uint32_t dim) noexcept {
  return reinterpret_cast<char *>(reinterpret_cast<uint8_t *>(hdr) +
                                  sizeof(NodeHeader) + dim * sizeof(float));
}
inline const char *node_metadata(const NodeHeader *hdr, uint32_t dim) noexcept {
  return reinterpret_cast<const char *>(reinterpret_cast<const uint8_t *>(hdr) +
                                        sizeof(NodeHeader) +
                                        dim * sizeof(float));
}

/// Returns a pointer to the metadata string for INT8 nodes.
/// For INT8, the centroid is dim * sizeof(int8_t) bytes, NOT dim *
/// sizeof(float).
inline char *node_metadata_q(NodeHeader *hdr, uint32_t dim,
                             uint32_t quant_type) noexcept {
  size_t payload = (quant_type == QUANT_INT8_SYMMETRIC) ? dim * sizeof(int8_t)
                                                        : dim * sizeof(float);
  return reinterpret_cast<char *>(reinterpret_cast<uint8_t *>(hdr) +
                                  sizeof(NodeHeader) + payload);
}
inline const char *node_metadata_q(const NodeHeader *hdr, uint32_t dim,
                                   uint32_t quant_type) noexcept {
  size_t payload = (quant_type == QUANT_INT8_SYMMETRIC) ? dim * sizeof(int8_t)
                                                        : dim * sizeof(float);
  return reinterpret_cast<const char *>(reinterpret_cast<const uint8_t *>(hdr) +
                                        sizeof(NodeHeader) + payload);
}

// ═══════════════════════════════════════════════════════════════════════════
// Node Flag Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Check if a node is tombstoned (O(1), reads flags in the header cache line).
inline bool is_tombstoned(const NodeHeader &n) noexcept {
  return (n.flags & NODE_FLAG_TOMBSTONE) != 0;
}

/// Check if a node is a consolidation summary.
inline bool is_summary(const NodeHeader &n) noexcept {
  return (n.flags & NODE_FLAG_SUMMARY) != 0;
}

/// Tombstone a node: sets flag + overwrites hub_penalty for branchless SIMD.
/// This is the ONLY correct way to mark a node as dead.
inline void tombstone_node(NodeHeader &n) noexcept {
  n.flags |= NODE_FLAG_TOMBSTONE;
  n.hub_penalty = TOMBSTONE_PENALTY;
}

/// Check if a node is currently superseded (reversibly excluded from beam
/// search).
inline bool is_superseded(const NodeHeader &n) noexcept {
  return (n.flags & NODE_FLAG_SUPERSEDED) != 0;
}

/// Supersede a node: same branchless beam-exclusion as tombstone_node()
/// (hub_penalty = TOMBSTONE_PENALTY), but reversible — the real hub_penalty
/// is stashed in saved_hub_penalty first. Idempotent: a second call while
/// already superseded is a no-op, since otherwise it would stash
/// TOMBSTONE_PENALTY itself as the "real" value and permanently poison the
/// node's score once revoked.
inline void supersede_node(NodeHeader &n) noexcept {
  if (n.flags & NODE_FLAG_SUPERSEDED)
    return;
  n.saved_hub_penalty = n.hub_penalty;
  n.hub_penalty = TOMBSTONE_PENALTY;
  n.flags |= NODE_FLAG_SUPERSEDED;
}

/// Revoke a supersession: restore the original hub_penalty and clear the
/// flag. No-op if the node isn't currently superseded.
///
/// If the node has ALSO been tombstoned since being superseded (e.g.
/// consolidate_subgraph() ran on it), hub_penalty is deliberately left at
/// TOMBSTONE_PENALTY rather than restored — tombstoning is terminal, and
/// restoring a real CSLS value here would silently break the branchless
/// beam-search exclusion that is the entire point of the tombstone
/// invariant, even though is_tombstoned() still correctly reports true.
/// Only the NODE_FLAG_SUPERSEDED bit is cleared in that case.
inline void revoke_supersede(NodeHeader &n) noexcept {
  if (!(n.flags & NODE_FLAG_SUPERSEDED))
    return;
  if (!(n.flags & NODE_FLAG_TOMBSTONE)) {
    n.hub_penalty = n.saved_hub_penalty;
  }
  n.flags &= static_cast<uint16_t>(~NODE_FLAG_SUPERSEDED);
}

// ===========================================================================
// Trace Event — O(1) mmap-indexed episodic memory record
// ===========================================================================

/// Trace event roles (stored as uint16_t in TraceEvent).
enum class TraceRole : uint16_t {
  User = 0,
  System = 1,
  Concept = 2,
  Summary = 3,
};

/// Trace event flags.
inline constexpr uint16_t TRACE_FLAG_TOMBSTONE = 0x0001;
inline constexpr uint16_t TRACE_FLAG_ARCHIVED = 0x0002;
inline constexpr uint16_t TRACE_FLAG_SUPERSEDED = 0x0004;

/// Version/supersession edge type (TraceEvent::edge_type). V4 Stage 1.
/// Additive-only: new values may be appended, existing values must never
/// change meaning — already-written TraceEvents encode these as raw bytes
/// on disk.
enum class EdgeType : uint8_t {
  None = 0,
  Supersedes = 1,   // this event supersedes supersedes_id (reversible)
  Refines = 2,      // this event extends supersedes_id; both remain live
  Contradicts = 3,  // this event conflicts with supersedes_id
  Revokes = 4,      // this event un-supersedes supersedes_id
  MergesWith = 5,   // this event and supersedes_id were consolidated (Dreaming)
  PromotedFrom = 6, // this event was minted from supersedes_id (Stage 4 promotion)
};

/// Reason code for a version/supersession edge (TraceEvent::reason_code).
/// Additive-only, same stability contract as EdgeType.
enum class ReasonCode : uint8_t {
  Unspecified = 0,
  Correction = 1,           // human/reviewer correction
  BugFixVerified = 2,       // outcome-verified fix (Stage 5)
  Deprecated = 3,           // superseded by newer guidance, no defect implied
  PolicyOrRedaction = 4,    // governance/erasure-driven change (Stage 4)
  ConsolidatedByDreaming = 5,
};

/**
 * @brief Binary trace event for mmap-backed episodic memory.
 *
 * Exactly 512 bytes (8 cache lines) — guarantees O(1) page-in from disk.
 * NO std::string, NO std::vector, NO heap pointers. Every field is
 * trivially copyable for safe mmap serialization.
 *
 * V4.1: Full text moved to sidecar BlobArena file. TraceEvent stores a
 * (blob_offset, blob_size) pointer plus a 64-byte inline preview for
 * fast ls-style listings without touching the blob file.
 *
 * Layout (byte offsets):
 *   0x000: id             (8B)
 *   0x008: prev_id        (8B)  — chronological linked list per session
 *   0x010: atlas_id       (8B)  — linked spatial concept (0 if none)
 *   0x018: timestamp      (8B)  — epoch microseconds
 *   0x020: role           (2B)  — TraceRole enum
 *   0x022: flags          (2B)  — tombstone/archive flags
 *   0x024: session_id     (36B) — UUID string for multi-tenant isolation
 *   0x048: blob_offset    (8B)  — offset into sidecar blob file
 *   0x050: blob_size      (4B)  — byte length of full text in blob
 *   0x054: text_preview   (64B) — null-terminated 63-char prefix
 *   0x094: edge_type            (1B)  — EdgeType (V4 Stage 1)
 *   0x095: reason_code          (1B)  — ReasonCode (V4 Stage 1)
 *   0x096: _pad0                (2B)  — explicit padding, 8-align supersedes_id
 *   0x098: supersedes_id        (8B)  — id this event's edge_type relates to
 *                                        (see V4 STAGE 4 note below: NOT a
 *                                        TraceEvent::id in practice)
 *   0x0A0: evidence_blob_offset (8B)  — sidecar BlobArena offset for rationale text
 *   0x0A8: evidence_blob_size   (4B)  — byte length of evidence blob (0 = none)
 *   0x0AC: _pad1                 (4B)  — explicit padding, 8-align embedding_blob_offset
 *   0x0B0: embedding_blob_offset (8B)  — sidecar BlobArena offset for this event's embedding
 *   0x0B8: embedding_blob_size   (4B)  — byte length of the embedding (0 = not embedded)
 *   0x0BC: _pad2                 (4B)  — explicit padding, 8-align event_time
 *   0x0C0: event_time            (8B)  — caller-supplied event time, epoch microseconds
 *                                        (0 = unset; distinct from `timestamp`, which is
 *                                        always Aeon's own insertion wall-clock)
 *   0x0C8: reserved       (312B) — padding to 512 bytes
 *
 * V4 STAGE 1: supersedes_id is deliberately NOT prev_id — prev_id is the
 * per-session chronological chain pointer; conflating it with version
 * lineage would break both. evidence_blob_* is deliberately a separate
 * (offset,size) pair from blob_offset/blob_size, which already belong to
 * the event's own text.
 *
 * V4 STAGE 4 (advisor-caught correction to the STAGE 1 design above):
 * supersedes_id's two real callers -- Stage 2 task 4's REFINES admission
 * dedup (context.py's process_turn()) and Stage 4 task 2's PROMOTED_FROM
 * (promotion.py's promote_fragment()) -- both write an ATLAS NODE id
 * (store-encoded via encode_store_id(), client.py, where store
 * discrimination applies), not a TraceEvent::id. This is correct usage,
 * not a bug: what each of those two edge types actually relates to is a
 * fragment of Atlas content (the near-duplicate node being refined; the
 * source node being promoted from), which the caller only ever has as an
 * Atlas node id, not as the TraceEvent that originally surfaced it. The
 * C++ layer never dereferences this field either way (it's opaque
 * payload to the WAL/replay/storage code), so nothing forces one
 * interpretation over the other -- this comment documents actual usage
 * rather than the original, never-implemented TraceEvent::id design.
 * SUPERSEDES/CONTRADICTS/REVOKES/MERGES_WITH have no caller yet; a future
 * one should follow the same Atlas-node-id convention for consistency
 * unless a concrete need for episodic-event-level lineage emerges.
 *
 * V4 STAGE 2: embedding_blob_* is a THIRD separate (offset,size) pair for
 * the same reason -- an event's embedding vector is neither its text nor
 * its evidence. Stored in the sidecar BlobArena (not inline: even at
 * dim=384 an FP32 embedding is 1536 bytes, far larger than TraceEvent's
 * entire reserved budget). Fed by TraceManager::append_event()'s optional
 * embedding parameter into TraceBlockIndex for O(|V|/1024 + K*1024)
 * semantic trace search (trace_block_index.hpp) -- an event with
 * embedding_blob_size == 0 was never embedded (e.g. no embedding model
 * configured) and is excluded from that index, not a zero vector.
 */
struct alignas(64) TraceEvent {
  uint64_t id;        // 0x000: Unique monotonic event ID
  uint64_t prev_id;   // 0x008: Previous event in this session (0 = root)
  uint64_t atlas_id;  // 0x010: Linked Atlas concept node (0 = none)
  uint64_t timestamp; // 0x018: Epoch microseconds

  uint16_t role;  // 0x020: TraceRole (User/System/Concept/Summary)
  uint16_t flags; // 0x022: TRACE_FLAG_TOMBSTONE|ARCHIVED|SUPERSEDED
  char session_id[36]; // 0x024: Multi-tenant session UUID (null-terminated)

  uint64_t blob_offset;  // 0x048: Offset into sidecar blob arena file
  uint32_t blob_size;    // 0x050: Byte length of full text in blob file
  char text_preview[64]; // 0x054: Null-terminated 63-char inline prefix

  uint8_t edge_type;             // 0x094: EdgeType enum value
  uint8_t reason_code;           // 0x095: ReasonCode enum value
  uint8_t _pad0[2];               // 0x096: explicit padding (8-align supersedes_id)
  uint64_t supersedes_id;         // 0x098: TraceEvent::id superseded (0 = none)
  uint64_t evidence_blob_offset;  // 0x0A0: sidecar BlobArena offset for evidence text
  uint32_t evidence_blob_size;    // 0x0A8: byte length of evidence blob (0 = none)

  uint8_t _pad1[4];                // 0x0AC: explicit padding (8-align embedding_blob_offset)
  uint64_t embedding_blob_offset;  // 0x0B0: sidecar BlobArena offset for this event's
                                    //        embedding vector (V4 Stage 2 task 3; 0 = none)
  uint32_t embedding_blob_size;    // 0x0B8: byte length of the embedding (dim * sizeof(float);
                                    //        0 = not embedded -- excluded from TraceBlockIndex)

  // V4 Stage 7 Track 2: caller-supplied event time, distinct from
  // `timestamp` (always Aeon's own insertion wall-clock, set internally by
  // append_event() and never caller-controlled). Real gap found via
  // LongMemEval (v4-plan.md): a caller ingesting historical or backdated
  // content (a chat import, a game engine replaying past events, any
  // agent backfilling memory) had nowhere to record WHEN an event actually
  // happened, only when Aeon received it -- LongMemEval's own harness had
  // to smuggle synthetic dates into event TEXT as a prefix, unable to
  // express them any other way. 0 = not supplied; callers ordering by
  // "when this happened" should fall back to `timestamp` when
  // `event_time == 0`, not treat 0 as a real epoch value.
  uint8_t _pad2[4];    // 0x0BC: explicit padding (8-align event_time)
  uint64_t event_time; // 0x0C0: Caller-supplied event time (epoch microseconds; 0 = unset)

  uint8_t reserved[312]; // 0x0C8: padding to 512 bytes
};
static_assert(sizeof(TraceEvent) == 512,
              "TraceEvent must be exactly 512 bytes for O(1) mmap indexing");
static_assert(std::is_standard_layout_v<TraceEvent>);
static_assert(std::is_trivially_copyable_v<TraceEvent>);
// V4 Stage 1: pin the exact byte offsets of the new fields. This is the
// layout contract aeon_trace_event_t (aeon_c_api.h) must mirror exactly —
// if either drifts, the C-API struct silently misreads.
static_assert(offsetof(TraceEvent, edge_type) == 0x094);
static_assert(offsetof(TraceEvent, reason_code) == 0x095);
static_assert(offsetof(TraceEvent, supersedes_id) == 0x098);
static_assert(offsetof(TraceEvent, evidence_blob_offset) == 0x0A0);
static_assert(offsetof(TraceEvent, evidence_blob_size) == 0x0A8);
static_assert(offsetof(TraceEvent, embedding_blob_offset) == 0x0B0);
static_assert(offsetof(TraceEvent, embedding_blob_size) == 0x0B8);
static_assert(offsetof(TraceEvent, event_time) == 0x0C0);
static_assert(offsetof(TraceEvent, reserved) == 0x0C8);

/// Trace file magic bytes (ASCII "AETR" = Aeon Trace).
inline constexpr uint32_t TRACE_MAGIC = 0x52544541; // "AETR" little-endian

/**
 * @brief On-disk header for trace mmap files (trace_genN.bin).
 *
 * 64 bytes, aligned to cache line. Sits at offset 0 of the file.
 * TraceEvent[0] begins at offset 64 (sizeof(TraceFileHeader)).
 *
 * V4 Stage 2 task 3: embedding_dim carved from reserved, mirroring
 * AtlasHeader::dim -- set once, at file creation, from the FIRST
 * embedding ever appended (TraceManager has no separate "create with
 * dim" constructor the way Atlas does; the file format doesn't need one
 * since embeddings are optional per-event and stored in the sidecar
 * BlobArena, not inline). 0 means no embedding has ever been appended to
 * this file yet. All subsequent embeddings must match this dim once set
 * -- see TraceManager::append_event()'s embedding parameter.
 */
struct alignas(64) TraceFileHeader {
  uint32_t magic;         // 0x00: TRACE_MAGIC
  uint32_t version;       // 0x04: File format version (1)
  uint64_t event_count;   // 0x08: Number of events in file
  uint64_t next_event_id; // 0x10: Next ID to assign
  uint32_t embedding_dim; // 0x18: Dim of indexed embeddings (0 = none set yet)
  uint8_t reserved[36];   // 0x1C: Padding to 64 bytes
};
static_assert(sizeof(TraceFileHeader) == 64,
              "TraceFileHeader must be 64 bytes (1 cache line)");
static_assert(offsetof(TraceFileHeader, embedding_dim) == 0x18);

// ═══════════════════════════════════════════════════════════════════════════
// WAL Record Header (V4.1 — Write-Ahead Log)
// ═══════════════════════════════════════════════════════════════════════════

/// WAL record types.
constexpr uint32_t WAL_RECORD_ATLAS = 0x01;
constexpr uint32_t WAL_RECORD_TRACE = 0x02;

/// In-place scope_bitmap mutation on an ALREADY-INSERTED mmap node, keyed
/// by node id (V4 Stage 1/2 prerequisite). Distinct from WAL_RECORD_ATLAS,
/// whose payload reconstructs a brand-new delta node -- that record type
/// cannot express "node N's field F is now V". See Atlas::set_node_scope().
constexpr uint32_t WAL_RECORD_ATLAS_SCOPE = 0x03;

/// Payload for WAL_RECORD_ATLAS_SCOPE records.
struct WalScopeRecord {
  uint64_t node_id;
  uint64_t scope_bitmap;
};
static_assert(sizeof(WalScopeRecord) == 16, "WalScopeRecord must be 16 bytes");
static_assert(std::is_trivially_copyable_v<WalScopeRecord>);

/// In-place NODE_FLAG_SUPERSEDED toggle on an ALREADY-INSERTED mmap node
/// (V4 Stage 2 prerequisite). Same rationale as WAL_RECORD_ATLAS_SCOPE --
/// this is a stateful read-modify-write (supersede_node()/
/// revoke_supersede() stash/restore hub_penalty based on current flag
/// state), not a payload WAL_RECORD_ATLAS can express. Both operations are
/// idempotent (see schema.hpp's guards), so replaying one against a node
/// already in the target state is a safe no-op. See
/// Atlas::supersede_node()/Atlas::revoke_node_supersede().
constexpr uint32_t WAL_RECORD_ATLAS_SUPERSEDE = 0x04;

/// Payload for WAL_RECORD_ATLAS_SUPERSEDE records.
struct WalSupersedeRecord {
  uint64_t node_id;
  uint8_t revoke; // 0 = supersede_node(), 1 = revoke_supersede()
};
static_assert(sizeof(WalSupersedeRecord) == 16,
              "WalSupersedeRecord must be 16 bytes (padding included)");
static_assert(std::is_trivially_copyable_v<WalSupersedeRecord>);

/// In-place governance_record_id mutation on an ALREADY-INSERTED mmap node,
/// keyed by node id (V4 Stage 4 task 1) -- same rationale and shape as
/// WAL_RECORD_ATLAS_SCOPE. NodeHeader::governance_record_id (schema.hpp,
/// allocated in Stage 1's byte budget) had no writer at all until
/// Atlas::set_node_governance_id().
constexpr uint32_t WAL_RECORD_ATLAS_GOVERNANCE = 0x05;

/// Payload for WAL_RECORD_ATLAS_GOVERNANCE records.
struct WalGovernanceRecord {
  uint64_t node_id;
  uint64_t governance_record_id;
};
static_assert(sizeof(WalGovernanceRecord) == 16,
              "WalGovernanceRecord must be 16 bytes");
static_assert(std::is_trivially_copyable_v<WalGovernanceRecord>);

/// In-place NODE_FLAG_TOMBSTONE set on an ALREADY-INSERTED mmap node, keyed
/// by node id (V4 Stage 4 task 5/6 -- the console/erasure-workflow "delete"
/// primitive). Same rationale and shape as WAL_RECORD_ATLAS_SUPERSEDE, minus
/// the revoke flag: unlike supersede_node()/revoke_supersede(), tombstoning
/// via this path is TERMINAL (schema.hpp's tombstone_node() does not stash
/// the prior hub_penalty, so there is nothing to restore), so there is no
/// symmetric "un-tombstone" operation to encode a direction for. Idempotent
/// on replay (setting the same flag bit and penalty value twice is a
/// no-op). See Atlas::tombstone_node(uint64_t).
constexpr uint32_t WAL_RECORD_ATLAS_TOMBSTONE = 0x06;

/// Payload for WAL_RECORD_ATLAS_TOMBSTONE records.
struct WalTombstoneRecord {
  uint64_t node_id;
};
static_assert(sizeof(WalTombstoneRecord) == 8,
              "WalTombstoneRecord must be 8 bytes");
static_assert(std::is_trivially_copyable_v<WalTombstoneRecord>);

/// Marks a node id as living in the delta byte arena (MSB set) rather than
/// the mmap file (sequential, MSB clear). Hoisted from Atlas::insert_delta()
/// /Atlas::replay_wal(), which each previously redefined this locally.
constexpr uint64_t NODE_ID_DELTA_MASK = 0x8000000000000000ULL;

/**
 * @brief WAL record header — prepended to each payload in the .wal file.
 *
 * Layout (16 bytes):
 *   [0x00]  uint32_t record_type    — WAL_RECORD_ATLAS or WAL_RECORD_TRACE
 *   [0x04]  uint32_t payload_size   — byte count of payload following header
 *   [0x08]  uint64_t checksum       — FNV-1a 64-bit hash of payload bytes
 *
 * On replay, if the checksum doesn't match the payload, the record and
 * all subsequent records are discarded (best-effort crash recovery).
 *
 * V4 STAGE 1 forward-compat contract: replay must never OOB-read a
 * corrupted/adversarial payload_size (bound it against bytes remaining in
 * the file before trusting it), and must never abort the whole replay just
 * because record_type is one this binary doesn't recognize — after the
 * checksum passes, an unrecognized record_type (or a recognized type with
 * an unexpected payload_size) is skipped, not fatal. See
 * Atlas::replay_wal()/TraceManager::replay_wal().
 */
struct WalRecordHeader {
  uint32_t record_type;  // WAL_RECORD_ATLAS or WAL_RECORD_TRACE
  uint32_t payload_size; // Bytes of payload following this header
  uint64_t checksum;     // FNV-1a 64-bit of payload bytes
};
static_assert(sizeof(WalRecordHeader) == 16,
              "WalRecordHeader must be 16 bytes");

} // namespace aeon
