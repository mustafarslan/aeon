#pragma once

#include "aeon/epoch.hpp"
#include "aeon/hash.hpp"
#include "aeon/quantization.hpp"
#include "aeon/schema.hpp"
#include "aeon/simd_impl.hpp"
#include "aeon/slb.hpp"
#include "aeon/storage.hpp"
#include <atomic>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <string_view>
#include <vector>

namespace aeon {

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
   */
  std::vector<ResultNode> navigate(std::span<const float> query,
                                   uint32_t beam_width = 1,
                                   bool apply_csls = false);

  std::vector<ResultNode> get_children(uint64_t parent_id);

  /**
   * @brief Inserts a new node as a child of parent_id.
   *
   * WRITE DIVERSION: If compact_in_progress_, silently falls through to
   * insert_delta() to prevent data loss during background compaction.
   */
  uint64_t insert(uint64_t parent_id, std::span<const float> vector,
                  std::string_view metadata);

  /**
   * @brief Inserts into the flat byte arena delta buffer.
   * Thread-safe. δ-node ID has MSB=1 to distinguish from mmap nodes.
   */
  uint64_t insert_delta(std::span<const float> vector,
                        std::string_view metadata);

  size_t prune_delta_tail(size_t n);

  size_t size() const;

  void load_context(std::span<const uint64_t> node_ids);

  /// Returns the embedding dimensionality of this Atlas instance.
  uint32_t dim() const noexcept { return dim_; }

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
                                            uint32_t beam_width);

  /// Count delta nodes in a flat byte arena.
  size_t delta_node_count() const noexcept;
  size_t delta_node_count(const std::vector<uint8_t> &arena) const noexcept;

  /// Get a NodeHeader* from the delta byte arena at the given index.
  NodeHeader *delta_get_node(size_t index) noexcept;
  const NodeHeader *delta_get_node(size_t index) const noexcept;
  const NodeHeader *delta_get_node(const std::vector<uint8_t> &arena,
                                   size_t index) const noexcept;

  // ─── Layout constants (set once at construction, never change) ───
  uint32_t dim_ = 0;
  uint32_t metadata_size_ = METADATA_SIZE_DEFAULT;
  size_t node_byte_stride_ = 0;
  uint32_t quantization_type_ = QUANT_FP32; // V4.1 Phase 3: cached from header

  // ─── Core state ───
  EpochManager epoch_mgr_;
  std::unique_ptr<storage::MemoryFile> file_;
  std::filesystem::path atlas_path_;
  uint64_t generation_ = 0; ///< Generational file naming counter

  // ─── Concurrency ───
  std::shared_mutex write_mutex_; ///< RW lock: shared reads, exclusive writes
  mutable std::shared_mutex delta_mutex_;

  // ─── Flat byte arena delta buffers ───
  // Contiguous memory:
  // [NodeHeader|centroid|metadata|pad][NodeHeader|centroid|metadata|pad]...
  // SIMD prefetcher can stream through without chasing heap pointers.
  std::vector<uint8_t> delta_buffer_bytes_;
  std::vector<uint8_t>
      frozen_delta_buffer_bytes_; ///< Frozen snapshot for bg compaction

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
  SemanticCache slb_cache_;
};

} // namespace aeon
