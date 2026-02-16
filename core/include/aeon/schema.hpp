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
 */
struct alignas(64) NodeHeader {
  uint64_t id;                 // 0x00: Unique ID (MSB=1 for delta nodes)
  uint64_t parent_offset;      // 0x08: Byte offset to parent (0 if root)
  uint64_t first_child_offset; // 0x10: Byte offset to first child
  uint16_t child_count;        // 0x18: Number of contiguous children
  uint16_t flags;              // 0x1A: NODE_FLAG_TOMBSTONE | NODE_FLAG_SUMMARY
  float hub_penalty;           // 0x1C: CSLS penalty or TOMBSTONE_PENALTY
  float quant_scale;           // 0x20: scale = max(|v|) / 127.0f (0.0 for FP32)
  float quant_zero_point;      // 0x24: always 0.0 for symmetric quantization
  uint8_t reserved[20];        // 0x28: Padding to 64-byte boundary
};

static_assert(sizeof(NodeHeader) == 64,
              "NodeHeader must be exactly 64 bytes (1 cache line)");
static_assert(std::is_standard_layout_v<NodeHeader>);
static_assert(std::is_trivially_copyable_v<NodeHeader>);

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
 *   0x094: reserved       (364B) — padding to 512 bytes
 */
struct alignas(64) TraceEvent {
  uint64_t id;        // 0x000: Unique monotonic event ID
  uint64_t prev_id;   // 0x008: Previous event in this session (0 = root)
  uint64_t atlas_id;  // 0x010: Linked Atlas concept node (0 = none)
  uint64_t timestamp; // 0x018: Epoch microseconds

  uint16_t role;       // 0x020: TraceRole (User/System/Concept/Summary)
  uint16_t flags;      // 0x022: TRACE_FLAG_TOMBSTONE, TRACE_FLAG_ARCHIVED
  char session_id[36]; // 0x024: Multi-tenant session UUID (null-terminated)

  uint64_t blob_offset;  // 0x048: Offset into sidecar blob arena file
  uint32_t blob_size;    // 0x050: Byte length of full text in blob file
  char text_preview[64]; // 0x054: Null-terminated 63-char inline prefix
  uint8_t reserved[364]; // 0x094: Padding to 512 bytes
};
static_assert(sizeof(TraceEvent) == 512,
              "TraceEvent must be exactly 512 bytes for O(1) mmap indexing");

/// Trace file magic bytes (ASCII "AETR" = Aeon Trace).
inline constexpr uint32_t TRACE_MAGIC = 0x52544541; // "AETR" little-endian

/**
 * @brief On-disk header for trace mmap files (trace_genN.bin).
 *
 * 64 bytes, aligned to cache line. Sits at offset 0 of the file.
 * TraceEvent[0] begins at offset 64 (sizeof(TraceFileHeader)).
 */
struct alignas(64) TraceFileHeader {
  uint32_t magic;         // 0x00: TRACE_MAGIC
  uint32_t version;       // 0x04: File format version (1)
  uint64_t event_count;   // 0x08: Number of events in file
  uint64_t next_event_id; // 0x10: Next ID to assign
  uint8_t reserved[40];   // 0x18: Padding to 64 bytes
};
static_assert(sizeof(TraceFileHeader) == 64,
              "TraceFileHeader must be 64 bytes (1 cache line)");

// ═══════════════════════════════════════════════════════════════════════════
// WAL Record Header (V4.1 — Write-Ahead Log)
// ═══════════════════════════════════════════════════════════════════════════

/// WAL record types.
constexpr uint32_t WAL_RECORD_ATLAS = 0x01;
constexpr uint32_t WAL_RECORD_TRACE = 0x02;

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
 */
struct WalRecordHeader {
  uint32_t record_type;  // WAL_RECORD_ATLAS or WAL_RECORD_TRACE
  uint32_t payload_size; // Bytes of payload following this header
  uint64_t checksum;     // FNV-1a 64-bit of payload bytes
};
static_assert(sizeof(WalRecordHeader) == 16,
              "WalRecordHeader must be 16 bytes");

} // namespace aeon
