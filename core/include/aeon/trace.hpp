#pragma once

/**
 * @file trace.hpp
 * @brief mmap-backed Episodic Trace Engine — Zero-Heap History.
 *
 * Replaces the old heap-based TraceManager (std::unordered_map<string,
 * TraceNode>) with a binary append-only log backed by a memory-mapped file.
 * Events are 512 bytes each (TraceEvent), laid out contiguously after a 64-byte
 * TraceFileHeader for O(1) indexing.
 *
 * Multi-tenant session isolation is maintained via session_tails_: a
 * lightweight RAM map of session_id → last_event_id, enabling O(1) linked-list
 * traversal per session through the prev_id chain.
 *
 * Shadow compaction mirrors the Atlas pattern:
 *   Step 1: µs freeze — swap delta buffer
 *   Step 2: Background copy — merge mmap + frozen delta → new generation file
 *   Step 3: µs freeze — swap MemoryFile pointers
 *   Step 4: Background cleanup — close old file, delete old generation
 */

#include "aeon/blob_arena.hpp"
#include "aeon/hash.hpp"
#include "aeon/schema.hpp"
#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace aeon {

/**
 * @brief mmap-backed Episodic Trace Manager.
 *
 * Provides:
 *   - append_event(): O(1) sequential append with per-session prev_id linking
 *   - get_history(): backward traversal via prev_id chain (mmap ↔ delta)
 *   - compact(): shadow compaction with generational file naming
 *
 * Concurrency:
 *   - Reads (get_history) take shared_lock on rw_mutex_
 *   - Writes (append_event) take unique_lock on rw_mutex_
 *   - session_tails_ protected by rw_mutex_ (it's small, no separate lock
 * needed)
 */
class TraceManager {
public:
  /**
   * @brief Opens or creates a trace mmap file.
   * @param path File path for trace storage (e.g., "memory/trace.bin")
   */
  explicit TraceManager(std::filesystem::path path);

  /// Default constructor for in-memory-only operation (no mmap backing).
  TraceManager();

  ~TraceManager();

  // Non-copyable
  TraceManager(const TraceManager &) = delete;
  TraceManager &operator=(const TraceManager &) = delete;

  /**
   * @brief Append an episodic event for a specific session.
   *
   * Sets prev_id to the last event for this session (from session_tails_),
   * then updates session_tails_ to point to this new event.
   *
   * If compact_in_progress_, appends to delta buffer instead of mmap file.
   *
   * @param session_id  Multi-tenant session UUID (max 35 chars + null)
   * @param role        TraceRole (User=0, System=1, Concept=2, Summary=3)
   * @param text        Full text (unlimited length, stored in blob arena)
   * @param atlas_id    Linked Atlas concept node ID (0 if none)
   * @return            The new event's unique ID
   */
  uint64_t append_event(const char *session_id, uint16_t role, const char *text,
                        uint64_t atlas_id = 0);

  /**
   * @brief Retrieve session history by backward traversal of prev_id chain.
   *
   * Starts from session_tails_[session_id], follows prev_id links backwards.
   * Seamlessly crosses the mmap ↔ delta buffer boundary.
   *
   * @param session_id  Session to retrieve history for
   * @param limit       Maximum number of events to return
   * @return            Events in reverse chronological order (newest first)
   */
  std::vector<TraceEvent> get_history(const char *session_id,
                                      size_t limit = 100) const;

  /**
   * @brief Shadow compaction — defragment trace file (generational naming).
   *
   * Mirrors Atlas::compact_mmap() pattern:
   *   1. µs freeze: swap delta buffer
   *   2. Background copy: write non-tombstoned events to new gen file
   *   3. µs freeze: swap file pointers
   *   4. Background cleanup: close + delete old generation
   */
  void compact();

  /**
   * @brief Retrieve the full text for an event from the blob arena.
   * @param blob_offset  Byte offset of the text in the blob file
   * @param blob_size    Byte length of the text in the blob file
   * @return             Full text as a string
   */
  std::string get_event_text(uint64_t blob_offset, uint32_t blob_size) const;

  /// Total event count (mmap + delta).
  size_t size() const;

  /// Event count in mmap file only.
  size_t mmap_event_count() const;

  /// Event count in delta buffer only.
  size_t delta_event_count() const;

  /// Check if a session has any events.
  bool has_session(const char *session_id) const;

  /**
   * @brief Tombstone a trace event by ID.
   *
   * Sets TRACE_FLAG_TOMBSTONE on the event's flags field. Tombstoned events
   * are excluded during compact(), enabling garbage collection of dead blobs.
   *
   * Thread-safe: acquires unique_lock on rw_mutex_.
   *
   * @param event_id  The event ID to tombstone.
   * @return true if the event was found and tombstoned, false if not found
   *         or already tombstoned.
   */
  bool tombstone_event(uint64_t event_id);

  /// Drop a session's tail pointer (does NOT delete events from disk).
  /// Used for session cleanup when an NPC despawns.
  bool drop_session(const char *session_id);

private:
  // -----------------------------------------------------------------------
  // File Management
  // -----------------------------------------------------------------------

  /// Path to current trace file (updated on compaction).
  std::filesystem::path trace_path_;

  /// Memory-mapped file handle (nullptr for in-memory-only mode).
  /// Points to: [TraceFileHeader (64B)] [TraceEvent[0]] [TraceEvent[1]] ...
  uint8_t *mapped_base_ = nullptr;
  size_t mapped_size_ = 0;
  int fd_ = -1;

  /// Events stored in the mmap file (excludes delta buffer).
  size_t mmap_event_count_ = 0;

  /// Next event ID to assign (monotonically increasing).
  uint64_t next_event_id_ = 1;

  /// Generational file naming counter.
  uint64_t generation_ = 0;

  // -----------------------------------------------------------------------
  // Delta Buffer (flat byte arena — NO std::vector<TraceEvent>)
  // -----------------------------------------------------------------------

  /// Flat contiguous storage for delta events.
  /// Layout: [TraceEvent][TraceEvent][...] — each exactly 512 bytes.
  std::vector<uint8_t> delta_bytes_;

  /// Frozen delta buffer (swapped during compaction Step 1).
  std::vector<uint8_t> frozen_delta_bytes_;

  /// Atomic flag for write diversion during compaction.
  std::atomic<bool> compact_in_progress_{false};

  // -----------------------------------------------------------------------
  // Session Tracking
  // -----------------------------------------------------------------------

  /// Maps session_id → last event_id for that session.
  /// Protected by rw_mutex_.
  std::unordered_map<std::string, uint64_t> session_tails_;

  // -----------------------------------------------------------------------
  // Concurrency
  // -----------------------------------------------------------------------

  mutable std::shared_mutex rw_mutex_;

  // -----------------------------------------------------------------------
  // Internal Helpers
  // -----------------------------------------------------------------------

  /// Get a pointer to event at index in the mmap file.
  const TraceEvent *mmap_event_at(size_t index) const;

  /// Get a pointer to event at index in the delta buffer.
  const TraceEvent *delta_event_at(size_t index) const;

  /// Resolve an event ID to a TraceEvent pointer (searches mmap then delta).
  const TraceEvent *resolve_event(uint64_t event_id) const;

  /// Append a single event to the mmap file. Returns the event ID.
  uint64_t append_mmap(const char *session_id, uint16_t role, const char *text,
                       size_t text_len, uint64_t atlas_id, uint64_t prev_id);

  /// Open or create the trace mmap file.
  void open_file(const std::filesystem::path &path);

  /// Grow the mmap file to accommodate more events.
  void grow_mmap(size_t additional_events);

  /// Close and unmap the current file.
  void close_file();

  /// Rebuild session_tails_ by scanning all events (used after file open).
  void rebuild_session_tails();

  /// Get the current timestamp in epoch microseconds.
  static uint64_t now_micros();

  // -----------------------------------------------------------------------
  // Sidecar Blob Arena (V4.1)
  // -----------------------------------------------------------------------

  /// Append-only blob store for full event text.
  std::unique_ptr<BlobArena> blob_arena_;

  // -----------------------------------------------------------------------
  // Write-Ahead Log (V4.1)
  // -----------------------------------------------------------------------

  /// Separate mutex for WAL disk I/O — does NOT block rw_mutex_ readers.
  /// Lock ordering: serialize (no lock) → wal_mutex_ → rw_mutex_
  std::mutex wal_mutex_;
  std::ofstream wal_stream_;
  std::filesystem::path wal_path_;

  /// Open or create the WAL file for append-only writes.
  void open_wal();

  /// Replay WAL records to reconstruct delta_bytes_ after crash.
  void replay_wal();

  /// Truncate (delete) the WAL file after successful compaction.
  void truncate_wal();
};

} // namespace aeon
