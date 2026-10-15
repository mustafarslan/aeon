/**
 * @file trace.cpp
 * @brief mmap-backed Episodic Trace Engine — Implementation.
 *
 * Binary append-only log with:
 *   - O(1) event append via flat byte arena delta buffer
 *   - O(K) session history retrieval via prev_id chain traversal
 *   - Shadow compaction with generational file naming
 *   - Multi-tenant isolation via session_tails_ map
 *
 * ZERO std::string or std::vector inside mmap regions — only TraceEvent
 * structs (512 bytes each, trivially copyable).
 */

#include "aeon/trace.hpp"

#include <unordered_map>
#include "aeon/hash.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace aeon {

// ===========================================================================
// Timestamp Helper
// ===========================================================================

uint64_t TraceManager::now_micros() {
  auto now = std::chrono::steady_clock::now();
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          now.time_since_epoch())
          .count());
}

// ===========================================================================
// Construction / Destruction
// ===========================================================================

TraceManager::TraceManager(std::filesystem::path path)
    : trace_path_(std::move(path)) {
  open_file(trace_path_);

  // ── V4.1: Open sidecar blob arena ──
  // V4 Stage 2 task 3 fix: STABLE name derived from trace_path_ (same
  // ".suffix" convention as wal_path_ below), not generation-suffixed.
  // See compact()'s doc comment for why a generation-suffixed name here
  // was a severe latent bug (any restart after a compaction couldn't
  // find its data).
  auto blob_path = trace_path_;
  blob_path += ".blobs";
  blob_arena_ = std::make_unique<BlobArena>(blob_path);

  // ── V4.1 WAL: crash recovery, BEFORE the index is rebuilt (Aleph Zero
  // fix): replay_wal() writes recovered events into the trace file itself,
  // so the rebuild below indexes them like any other ──
  wal_path_ = trace_path_;
  wal_path_ += ".wal";
  replay_wal();

  // ── V4 Stage 2 task 3: rebuild the (unpersisted) semantic block index
  // from durable embedding blobs, now that blob_arena_ is open ──
  rebuild_block_index();
  open_wal();
}

TraceManager::TraceManager() = default;

TraceManager::~TraceManager() {
  if (blob_arena_)
    blob_arena_->close();
  close_file();
}

// ===========================================================================
// File Management (POSIX mmap / Win32 MapViewOfFile)
// ===========================================================================

void TraceManager::open_file(const std::filesystem::path &path) {
  bool new_file = !std::filesystem::exists(path);

#ifdef _WIN32
  // --- Windows implementation ---
  DWORD access = GENERIC_READ | GENERIC_WRITE;
  DWORD creation = new_file ? CREATE_NEW : OPEN_EXISTING;
  HANDLE hFile = CreateFileW(path.wstring().c_str(), access, FILE_SHARE_READ,
                             nullptr, creation, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (hFile == INVALID_HANDLE_VALUE)
    throw std::runtime_error("Failed to open trace file: " + path.string());

  fd_ = _open_osfhandle(reinterpret_cast<intptr_t>(hFile), 0);
#else
  // --- POSIX implementation ---
  int flags = O_RDWR;
  if (new_file)
    flags |= O_CREAT;

  fd_ = ::open(path.c_str(), flags, 0644);
  if (fd_ < 0)
    throw std::runtime_error("Failed to open trace file: " + path.string());
#endif

  if (new_file) {
    // Write initial header + space for 1024 events
    size_t initial_size = sizeof(TraceFileHeader) + 1024 * sizeof(TraceEvent);

#ifndef _WIN32
    if (::ftruncate(fd_, static_cast<off_t>(initial_size)) < 0) {
      ::close(fd_);
      fd_ = -1;
      throw std::runtime_error("Failed to resize trace file");
    }
#else
    _chsize_s(fd_, static_cast<long long>(initial_size));
#endif

    mapped_size_ = initial_size;

#ifndef _WIN32
    mapped_base_ = static_cast<uint8_t *>(::mmap(
        nullptr, mapped_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));
    if (mapped_base_ == MAP_FAILED) {
      mapped_base_ = nullptr;
      ::close(fd_);
      fd_ = -1;
      throw std::runtime_error("Failed to mmap trace file");
    }
#endif

    // Initialize header
    auto *hdr = reinterpret_cast<TraceFileHeader *>(mapped_base_);
    std::memset(hdr, 0, sizeof(TraceFileHeader));
    hdr->magic = TRACE_MAGIC;
    hdr->version = 1;
    hdr->event_count = 0;
    hdr->next_event_id = 1;

    mmap_event_count_ = 0;
    next_event_id_ = 1;
  } else {
    // Read existing file
#ifndef _WIN32
    struct stat st;
    if (::fstat(fd_, &st) < 0) {
      ::close(fd_);
      fd_ = -1;
      throw std::runtime_error("Failed to stat trace file");
    }
    mapped_size_ = static_cast<size_t>(st.st_size);

    mapped_base_ = static_cast<uint8_t *>(::mmap(
        nullptr, mapped_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));
    if (mapped_base_ == MAP_FAILED) {
      mapped_base_ = nullptr;
      ::close(fd_);
      fd_ = -1;
      throw std::runtime_error("Failed to mmap trace file");
    }
#endif

    auto *hdr = reinterpret_cast<const TraceFileHeader *>(mapped_base_);
    if (hdr->magic != TRACE_MAGIC)
      throw std::runtime_error("Invalid trace file magic");

    mmap_event_count_ = static_cast<size_t>(hdr->event_count);
    next_event_id_ = hdr->next_event_id;
    embedding_dim_ = hdr->embedding_dim;

    // Rebuild session tails from on-disk events
    rebuild_session_tails();
  }
  // rebuild_block_index() needs blob_arena_, which the constructor opens
  // AFTER open_file() returns -- called from the constructor body instead
  // of here. See TraceManager::TraceManager().
}

void TraceManager::grow_mmap(size_t additional_events) {
  if (fd_ < 0 || !mapped_base_)
    return;

  size_t needed = sizeof(TraceFileHeader) +
                  (mmap_event_count_ + additional_events) * sizeof(TraceEvent);
  if (needed <= mapped_size_)
    return;

  // Round up to next power of 2 events for amortized growth
  size_t new_capacity = mmap_event_count_ + additional_events;
  new_capacity = std::max(new_capacity, mmap_event_count_ * 2);
  size_t new_size = sizeof(TraceFileHeader) + new_capacity * sizeof(TraceEvent);

#ifndef _WIN32
  ::munmap(mapped_base_, mapped_size_);

  if (::ftruncate(fd_, static_cast<off_t>(new_size)) < 0)
    throw std::runtime_error("Failed to grow trace file");

  mapped_base_ = static_cast<uint8_t *>(
      ::mmap(nullptr, new_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));
  if (mapped_base_ == MAP_FAILED) {
    mapped_base_ = nullptr;
    throw std::runtime_error("Failed to remap trace file after growth");
  }
  mapped_size_ = new_size;
#endif
}

void TraceManager::close_file() {
  if (mapped_base_) {
#ifndef _WIN32
    ::munmap(mapped_base_, mapped_size_);
#endif
    mapped_base_ = nullptr;
    mapped_size_ = 0;
  }
  if (fd_ >= 0) {
#ifndef _WIN32
    ::close(fd_);
#else
    _close(fd_);
#endif
    fd_ = -1;
  }
}

// ===========================================================================
// Event Accessors
// ===========================================================================

const TraceEvent *TraceManager::mmap_event_at(size_t index) const {
  if (!mapped_base_ || index >= mmap_event_count_)
    return nullptr;
  return reinterpret_cast<const TraceEvent *>(
      mapped_base_ + sizeof(TraceFileHeader) + index * sizeof(TraceEvent));
}

const TraceEvent *TraceManager::delta_event_at(size_t index) const {
  size_t offset = index * sizeof(TraceEvent);
  if (offset + sizeof(TraceEvent) > delta_bytes_.size())
    return nullptr;
  return reinterpret_cast<const TraceEvent *>(delta_bytes_.data() + offset);
}

const TraceEvent *TraceManager::resolve_event(uint64_t event_id) const {
  if (event_id == 0)
    return nullptr;

  // Events are 1-indexed. IDs 1..mmap_event_count_ are in the mmap file.
  // IDs mmap_event_count_+1.. are in the delta buffer.
  // But IDs might not be contiguous after compaction, so we search by ID.

  // Fast path: check mmap (events are sequential, ID == index + 1 for
  // non-compacted files, but after compaction IDs may be sparse).
  // Linear scan is acceptable because get_history is bounded by `limit`.

  // Search mmap first (most events live here)
  for (size_t i = mmap_event_count_; i > 0; --i) {
    const TraceEvent *ev = mmap_event_at(i - 1);
    if (ev && ev->id == event_id)
      return ev;
  }

  // Search delta buffer
  size_t delta_count = delta_bytes_.size() / sizeof(TraceEvent);
  for (size_t i = delta_count; i > 0; --i) {
    const TraceEvent *ev = delta_event_at(i - 1);
    if (ev && ev->id == event_id)
      return ev;
  }

  // Search frozen delta (during compaction)
  if (compact_in_progress_.load(std::memory_order_acquire)) {
    size_t frozen_count = frozen_delta_bytes_.size() / sizeof(TraceEvent);
    for (size_t i = frozen_count; i > 0; --i) {
      size_t offset = (i - 1) * sizeof(TraceEvent);
      const auto *ev = reinterpret_cast<const TraceEvent *>(
          frozen_delta_bytes_.data() + offset);
      if (ev->id == event_id)
        return ev;
    }
  }

  return nullptr;
}

// ===========================================================================
// Session Tail Rebuilding
// ===========================================================================

void TraceManager::rebuild_session_tails() {
  session_tails_.clear();

  // Scan all mmap events to find the latest event per session
  for (size_t i = 0; i < mmap_event_count_; ++i) {
    const TraceEvent *ev = mmap_event_at(i);
    if (!ev)
      continue;
    if (ev->flags & TRACE_FLAG_TOMBSTONE)
      continue;

    std::string sid(ev->session_id,
                    std::min(std::strlen(ev->session_id), size_t{35}));
    auto it = session_tails_.find(sid);
    if (it == session_tails_.end() || ev->id > it->second) {
      session_tails_[sid] = ev->id;
    }
  }
}

void TraceManager::rebuild_block_index() {
  if (embedding_dim_ == 0)
    return; // no embedding has ever been appended to this file

  block_index_ = std::make_unique<TraceBlockIndex>(embedding_dim_);
  if (!blob_arena_)
    return;

  // Mmap events only -- a KNOWN LIMITATION, not a silent gap: a delta
  // event embedded before a crash and recovered via replay_wal() is not
  // covered here (replay_wal() runs after this, and reconstructs into
  // delta_bytes_, not the mmap region this scans). During NORMAL
  // operation (not crash recovery), append_event() indexes both mmap and
  // delta writes as they happen, so this gap is narrow: crash-recovered
  // delta embeddings only, until the next compaction promotes them to
  // mmap and a subsequent restart's rebuild picks them up.
  for (size_t i = 0; i < mmap_event_count_; ++i) {
    const TraceEvent *ev = mmap_event_at(i);
    if (!ev)
      continue;
    if (ev->flags & (TRACE_FLAG_TOMBSTONE | TRACE_FLAG_SUPERSEDED))
      continue;
    if (ev->embedding_blob_size == 0)
      continue;

    auto view = blob_arena_->read(ev->embedding_blob_offset,
                                  ev->embedding_blob_size);
    if (view.size() != static_cast<size_t>(embedding_dim_) * sizeof(float))
      continue; // corrupt/mismatched blob -- skip defensively, don't crash startup

    const float *emb = reinterpret_cast<const float *>(view.data());
    block_index_->append(ev->id, std::span<const float>(emb, embedding_dim_),
                         static_cast<double>(ev->timestamp));
  }
}

// ===========================================================================
// Append Event
// ===========================================================================

namespace {

// Returns the largest prefix length <= `len` bytes of `data` that does not
// split a multi-byte UTF-8 sequence.
//
// Bug found and fixed via v4-plan.md Stage 7: `TraceEvent::text_preview`'s
// inline truncation previously used a raw `std::strncpy` to the first 63
// bytes, with no regard for UTF-8 character boundaries. Any stored text
// containing a multi-byte character (curly quotes, accented Latin, CJK,
// emoji -- i.e. most real-world non-ASCII text) whose 63-byte boundary
// happened to land mid-character left a truncated, invalid UTF-8 sequence
// in the preview buffer. `bindings.cpp`'s `get_history()`/`semantic_search()`
// unconditionally convert `text_preview` to a Python string
// (`nb::str(ev.text_preview)`) for every returned event, regardless of
// whether the caller even uses the preview field (the full text, read
// separately from the blob arena, was always correct) -- nanobind's strict
// UTF-8 decode throws `str_from_cstr(): conversion error!` on the corrupted
// bytes, crashing the entire call. This is a standing correctness bug for
// any real deployment ingesting ordinary human text, not specific to any
// one benchmark.
size_t safe_utf8_truncate_length(const char *data, size_t len) {
  if (len == 0)
    return 0;

  // Walk back over UTF-8 continuation bytes (10xxxxxx) to find the lead
  // byte of the last (possibly incomplete) sequence -- at most 3 bytes
  // back, since the longest UTF-8 sequence is 4 bytes (1 lead + 3
  // continuation).
  size_t lead_pos = len;
  size_t back = 0;
  while (lead_pos > 0 && back < 3 &&
         (static_cast<unsigned char>(data[lead_pos - 1]) & 0xC0) == 0x80) {
    --lead_pos;
    ++back;
  }
  if (lead_pos == 0)
    return 0; // Defensive: no valid lead byte found within range.

  unsigned char lead = static_cast<unsigned char>(data[lead_pos - 1]);
  size_t seq_len;
  if ((lead & 0x80) == 0x00)
    seq_len = 1; // ASCII
  else if ((lead & 0xE0) == 0xC0)
    seq_len = 2; // 110xxxxx
  else if ((lead & 0xF0) == 0xE0)
    seq_len = 3; // 1110xxxx
  else if ((lead & 0xF8) == 0xF0)
    seq_len = 4; // 11110xxx
  else
    return lead_pos - 1; // Invalid lead byte -- drop it defensively too.

  size_t seq_start = lead_pos - 1;
  if (seq_start + seq_len <= len)
    return len; // The last sequence fits entirely within `len` -- no cut needed.
  return seq_start; // Doesn't fit -- drop the whole incomplete sequence.
}

} // namespace

uint64_t TraceManager::append_event(const char *session_id, uint16_t role,
                                    const char *text, uint64_t atlas_id,
                                    std::span<const float> embedding,
                                    uint8_t edge_type, uint64_t supersedes_id,
                                    uint8_t reason_code, uint64_t event_time) {
  // ── Step 0: Establish/validate embedding dimensionality (brief lock) ──
  // V4 Stage 2 task 3. Must happen BEFORE Step 2's WAL write below -- a
  // dim-mismatch throw after the WAL record is written would leave a
  // phantom WAL entry with no corresponding committed event (replayed on
  // next restart as a ghost event the caller never intended). Kept as its
  // own narrow critical section rather than holding rw_mutex_ for this
  // whole function, preserving Step 1's "expensive work outside any lock"
  // design intent (matching Atlas::insert_delta()) for everything else.
  // Also reserves the event's id and its place in its session's chain (Aleph
  // Zero fix): both are in the WAL record written in Step 2, so a replay after
  // a crash keeps the id the caller was given and the chain it joined, instead
  // of inventing them. The lock is brief and never held across the WAL write.
  std::string sid(session_id ? session_id : "");
  uint64_t reserved_id = 0;
  uint64_t reserved_prev = 0;
  {
    std::unique_lock dim_lock(rw_mutex_);
    if (embedding.empty()) {
      // nothing to establish or validate
    } else if (embedding_dim_ == 0) {
      embedding_dim_ = static_cast<uint32_t>(embedding.size());
      block_index_ = std::make_unique<TraceBlockIndex>(embedding_dim_);
      if (mapped_base_) {
        reinterpret_cast<TraceFileHeader *>(mapped_base_)->embedding_dim =
            embedding_dim_;
      }
    } else if (embedding.size() != embedding_dim_) {
      throw std::invalid_argument(
          "TraceManager::append_event: embedding dim mismatch (this file's "
          "embedding_dim is " +
          std::to_string(embedding_dim_) + ", got " +
          std::to_string(embedding.size()) + ")");
    }
    reserved_id = next_event_id_++;
    auto pending = pending_tails_.find(sid);
    auto tail = session_tails_.find(sid);
    reserved_prev = pending != pending_tails_.end() ? pending->second
                    : tail != session_tails_.end() ? tail->second : 0;
    pending_tails_[sid] = reserved_id; // what the next append chains to; not yet what readers see
  }

  try {
  // ── Step 1: Serialize data & write blob (NO LOCK) ──
  // Build the TraceEvent payload outside of any lock.
  TraceEvent ev{};
  std::memset(&ev, 0, sizeof(TraceEvent));

  ev.id = reserved_id;
  ev.prev_id = reserved_prev;
  ev.timestamp = now_micros();
  ev.atlas_id = atlas_id;
  ev.role = role;
  ev.flags = 0;
  ev.edge_type = edge_type;
  ev.supersedes_id = supersedes_id;
  ev.reason_code = reason_code;
  ev.event_time = event_time;

  if (session_id) {
    std::strncpy(ev.session_id, session_id, sizeof(ev.session_id) - 1);
    ev.session_id[sizeof(ev.session_id) - 1] = '\0';
  }

  // V4.1: Write full text to sidecar blob arena, keep 63-char preview inline
  if (text) {
    size_t text_len = std::strlen(text);
    if (blob_arena_ && text_len > 0) {
      BlobRef ref = blob_arena_->append(text, text_len);
      ev.blob_offset = ref.offset;
      ev.blob_size = ref.size;
    }
    // Inline preview: first up to 63 bytes, snapped back to a UTF-8
    // character boundary rather than a raw byte count (see
    // safe_utf8_truncate_length's doc comment for the crash this fixes).
    size_t preview_cap = sizeof(ev.text_preview) - 1;
    size_t copy_len = std::min(text_len, preview_cap);
    copy_len = safe_utf8_truncate_length(text, copy_len);
    std::memcpy(ev.text_preview, text, copy_len);
    ev.text_preview[copy_len] = '\0';
  }

  // V4 Stage 2 task 3: write the embedding to the sidecar blob arena (a
  // THIRD independent blob, separate from text and Stage 1's evidence --
  // see TraceEvent's doc comment in schema.hpp). Dim already validated
  // in Step 0 above.
  if (!embedding.empty() && blob_arena_) {
    BlobRef ref = blob_arena_->append(
        reinterpret_cast<const char *>(embedding.data()),
        embedding.size() * sizeof(float));
    ev.embedding_blob_offset = ref.offset;
    ev.embedding_blob_size = ref.size;
  }

  // Compute FNV-1a checksum of the full TraceEvent
  uint64_t checksum = hash::fnv1a_64(&ev, sizeof(TraceEvent));

  // ── Step 2: WAL write — ONLY for delta path (volatile RAM) ──
  // Events going to mmap are already durable on disk and don't need WAL
  // protection. Check the same condition used for write diversion:
  //   compact_in_progress_ (atomic) || !mapped_base_ (pointer)
  // This is safe to read lock-free — false positives just mean an
  // unnecessary (but harmless) WAL write.
  bool needs_wal =
      compact_in_progress_.load(std::memory_order_acquire) || !mapped_base_;
  bool wal_written = false;
  auto write_wal = [&] {
    std::lock_guard<std::mutex> wal_lock(wal_mutex_);
    if (wal_stream_.is_open()) {
      WalRecordHeader wal_hdr{};
      wal_hdr.record_type = WAL_RECORD_TRACE;
      wal_hdr.payload_size = static_cast<uint32_t>(sizeof(TraceEvent));
      wal_hdr.checksum = checksum;

      wal_stream_.write(reinterpret_cast<const char *>(&wal_hdr),
                        sizeof(WalRecordHeader));
      wal_stream_.write(reinterpret_cast<const char *>(&ev),
                        sizeof(TraceEvent));
      wal_stream_.flush();
    }
    wal_written = true;
  };

  if (needs_wal)
    write_wal();

  // ── Step 3: lock(rw_mutex_) → append to buffer → unlock ──
  // (id and prev_id were reserved in Step 0.) A compaction that began after
  // the lock-free check above sends the event to the RAM delta: it must not
  // land there without its WAL record, so the record is written first, with
  // no lock held, and the lock taken again.
  std::unique_lock lock(rw_mutex_);
  if (!wal_written &&
      (compact_in_progress_.load(std::memory_order_acquire) || !mapped_base_)) {
    lock.unlock();
    write_wal();
    lock.lock();
  }

  uint64_t event_id;

  // Write diversion during compaction
  if (compact_in_progress_.load(std::memory_order_acquire) || !mapped_base_) {
    // Append to delta buffer
    size_t old_size = delta_bytes_.size();
    delta_bytes_.resize(old_size + sizeof(TraceEvent), 0);
    std::memcpy(delta_bytes_.data() + old_size, &ev, sizeof(TraceEvent));
    event_id = ev.id;
  } else {
    // Append to mmap file
    grow_mmap(1);
    auto *dst =
        reinterpret_cast<TraceEvent *>(mapped_base_ + sizeof(TraceFileHeader) +
                                       mmap_event_count_ * sizeof(TraceEvent));
    std::memcpy(dst, &ev, sizeof(TraceEvent));
    ++mmap_event_count_;

    // Update file header
    auto *hdr = reinterpret_cast<TraceFileHeader *>(mapped_base_);
    hdr->event_count = mmap_event_count_;
    hdr->next_event_id = next_event_id_;

    event_id = ev.id;
  }

  // V4 Stage 2 task 3: index the embedding (if any) under the SAME id
  // just assigned, regardless of whether it landed in mmap or delta --
  // TraceBlockIndex doesn't care where the event physically lives.
  if (!embedding.empty() && block_index_) {
    block_index_->append(event_id, embedding,
                         static_cast<double>(ev.timestamp));
  }

  // Readers see the event now: the newest landed id of its session is the tail.
  auto tail = session_tails_.find(sid);
  if (tail == session_tails_.end() || event_id > tail->second)
    session_tails_[sid] = event_id;
  auto pending = pending_tails_.find(sid);
  if (pending != pending_tails_.end() && pending->second == event_id)
    pending_tails_.erase(pending);
  return event_id;
  } catch (...) {
    // The event never landed: what the next append chains to goes back to
    // where it was, unless another append has already chained to it (then that
    // chain is cut here; a single writer, as Aleph Zero is under its file
    // lock, never sees it). A WAL record already written is replayed later as
    // an event the caller was told failed: the rare cost of a write that
    // fails after its record (out of memory, a full disk).
    std::unique_lock lock(rw_mutex_);
    auto pending = pending_tails_.find(sid);
    if (pending != pending_tails_.end() && pending->second == reserved_id) {
      auto landed = session_tails_.find(sid);
      if (landed != session_tails_.end() && landed->second == reserved_prev)
        pending_tails_.erase(pending);
      else if (reserved_prev)
        pending->second = reserved_prev;
      else
        pending_tails_.erase(pending);
    }
    throw;
  }
}

// ===========================================================================
// Get History — Backward prev_id chain traversal
// ===========================================================================

std::vector<TraceEvent> TraceManager::get_history(const char *session_id,
                                                  size_t limit) const {
  std::shared_lock lock(rw_mutex_);

  std::string sid(session_id ? session_id : "");
  auto it = session_tails_.find(sid);
  if (it == session_tails_.end())
    return {};

  std::vector<TraceEvent> result;
  result.reserve(std::min(limit, size_t{256}));

  uint64_t current_id = it->second;

  while (current_id != 0 && result.size() < limit) {
    const TraceEvent *ev = resolve_event(current_id);
    if (!ev)
      break;

    // Skip tombstoned events
    if (!(ev->flags & TRACE_FLAG_TOMBSTONE)) {
      result.push_back(*ev); // Flat 512-byte copy — trivially copyable
    }

    current_id = ev->prev_id;
  }

  return result;
}

// ===========================================================================
// Semantic Trace Search (V4 Stage 2 task 3)
// ===========================================================================

std::vector<TraceEvent>
TraceManager::semantic_search(std::span<const float> query,
                              size_t top_k) const {
  std::shared_lock lock(rw_mutex_);

  if (!block_index_ || query.size() != embedding_dim_)
    return {};

  // Over-fetch: TraceBlockIndex has no notion of tombstoned/superseded
  // events (it only knows ids), so some of its top_k may need to be
  // filtered out below -- ask for a bit more than requested to absorb
  // that without an extra round trip in the common case.
  auto raw = block_index_->query(query, top_k + top_k / 2 + 4);

  std::vector<TraceEvent> result;
  result.reserve(std::min(raw.size(), top_k));
  for (const auto &r : raw) {
    if (result.size() >= top_k)
      break;
    const TraceEvent *ev = resolve_event(r.node_id);
    if (!ev)
      continue; // compacted away since indexing; stale block-index entry
    if (ev->flags & (TRACE_FLAG_TOMBSTONE | TRACE_FLAG_SUPERSEDED))
      continue;
    result.push_back(*ev);
  }
  return result;
}

// ===========================================================================
// Shadow Compaction
// ===========================================================================

void TraceManager::compact() {
  // V4 Stage 2 task 3 fix: a SEVERE pre-existing bug, found while testing
  // semantic search across a restart. compact() used to build the new
  // generation at a PERMANENTLY generation-suffixed name (trace_gen1.bin,
  // trace_gen2.bin, ...) and delete the file at trace_path_ -- but
  // trace_path_/generation_ are only tracked in-memory, reset to
  // (trace_path_, 0) on every fresh TraceManager construction. Any
  // restart using the caller's originally-configured path (the normal
  // case -- see dependencies.py's AEON_TRACE_PATH) would find that path
  // GONE (deleted by the prior compaction) and silently create a new,
  // EMPTY file: total, silent data loss on the very first restart after
  // the very first compaction. Atlas::compact_mmap() had the identical
  // bug (v4-plan.md Stage 2), fixed the same way.
  //
  // Fix: build the new generation at a TEMPORARY path, durably flush it,
  // then ATOMICALLY RENAME it onto the STABLE, caller-facing trace_path_
  // (POSIX rename() only rewrites the directory entry -- it doesn't
  // invalidate this process's still-open old_fd/old_base, which keep
  // referencing the old, now-unlinked inode until Step 4 closes them).
  // trace_path_ itself never changes again after construction; only
  // generation_ (purely an internal temp-filename disambiguator now)
  // still increments.
  if (!mapped_base_ || fd_ < 0)
    return;

  // -----------------------------------------------------------------------
  // Step 1: µs Freeze — swap delta buffer
  // -----------------------------------------------------------------------
  size_t snapshot_mmap_count;
  {
    std::unique_lock lock(rw_mutex_);
    compact_in_progress_.store(true, std::memory_order_release);
    frozen_delta_bytes_ = std::move(delta_bytes_);
    delta_bytes_.clear();
    snapshot_mmap_count = mmap_event_count_;
  }
  // Lock released — game engine continues via write diversion to delta_bytes_

  // -----------------------------------------------------------------------
  // Step 2: Background Copy — merge live events to new gen file
  // -----------------------------------------------------------------------
  uint64_t new_gen = generation_ + 1;
  // TEMPORARY construction path -- renamed onto trace_path_ (the stable,
  // caller-facing name) once fully populated and durable, below.
  std::filesystem::path new_path = trace_path_;
  new_path += (".compacting" + std::to_string(new_gen));

  // Count live events
  size_t live_count = 0;
  for (size_t i = 0; i < snapshot_mmap_count; ++i) {
    const TraceEvent *ev = mmap_event_at(i);
    if (ev && !(ev->flags & TRACE_FLAG_TOMBSTONE))
      ++live_count;
  }
  size_t frozen_count = frozen_delta_bytes_.size() / sizeof(TraceEvent);
  for (size_t i = 0; i < frozen_count; ++i) {
    const auto *ev = reinterpret_cast<const TraceEvent *>(
        frozen_delta_bytes_.data() + i * sizeof(TraceEvent));
    if (!(ev->flags & TRACE_FLAG_TOMBSTONE))
      ++live_count;
  }

  // Create new file — pre-allocate extra for future growth
  size_t alloc_size =
      sizeof(TraceFileHeader) +
      std::max(live_count * 2, live_count + 1024) * sizeof(TraceEvent);

#ifndef _WIN32
  int new_fd = ::open(new_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
  if (new_fd < 0) {
    compact_in_progress_.store(false, std::memory_order_release);
    throw std::runtime_error("Failed to create compacted trace file");
  }

  if (::ftruncate(new_fd, static_cast<off_t>(alloc_size)) < 0) {
    ::close(new_fd);
    compact_in_progress_.store(false, std::memory_order_release);
    throw std::runtime_error("Failed to resize compacted trace file");
  }

  auto *new_base = static_cast<uint8_t *>(::mmap(
      nullptr, alloc_size, PROT_READ | PROT_WRITE, MAP_SHARED, new_fd, 0));
  if (new_base == MAP_FAILED) {
    ::close(new_fd);
    compact_in_progress_.store(false, std::memory_order_release);
    throw std::runtime_error("Failed to mmap compacted trace file");
  }
#else
  // NOT IMPLEMENTED ON WIN32 -- AND IT NOW SAYS SO.
  //
  // This used to `return` silently, so a Windows caller's compact() reported
  // success while reclaiming nothing: tombstoned events stayed on disk, the
  // blob arena never GC'd, and the file grew without bound with no error to
  // act on. A no-op that looks like success is the same failure shape as an
  // enumeration that returns a silent subset -- the caller cannot distinguish
  // "done" from "did nothing" and so cannot retry, alert, or fall back.
  //
  // Throwing is deliberate over a silent skip: compaction is invoked from a
  // maintenance path that can handle a failure, and on a platform where it
  // cannot run, unbounded growth is the strictly worse outcome. Implementing
  // it properly needs the Win32 file-mapping equivalents used elsewhere in
  // platform.hpp; that is a separate change, and shipping it untested from a
  // POSIX host would be worse than reporting honestly.
  compact_in_progress_.store(false, std::memory_order_release);
  throw std::runtime_error(
      "TraceManager::compact() is not implemented on Windows; the trace file "
      "was NOT compacted and no space was reclaimed");
#endif

  // Write header
  auto *new_hdr = reinterpret_cast<TraceFileHeader *>(new_base);
  std::memset(new_hdr, 0, sizeof(TraceFileHeader));
  new_hdr->magic = TRACE_MAGIC;
  new_hdr->version = 2; // V4.1: Blob arena format

  // ── V4.1: Create new-generation blob arena for GC ──
  // TEMPORARY construction path -- renamed onto the STABLE blob path
  // (trace_path_ + ".blobs", see the constructor) alongside the main
  // file's rename, below.
  std::filesystem::path stable_blob_path = trace_path_;
  stable_blob_path += ".blobs";
  std::filesystem::path new_blob_path = stable_blob_path;
  new_blob_path += (".compacting" + std::to_string(new_gen));
  auto new_blob_arena = std::make_unique<BlobArena>(new_blob_path);

  // GC every (offset,size) blob pair a TraceEvent carries: text,
  // evidence (Stage 1), embedding (Stage 2 task 3). Without this, a
  // surviving event's blob-referencing fields would keep pointing at the
  // OLD generation's blob file, which gets deleted at the end of this
  // function -- a dangling reference the first time anything actually
  // populated evidence_blob_* or embedding_blob_* (evidence has had no
  // live writer yet, so this was a dormant gap until embeddings, wired in
  // by this same task, made it live).
  auto gc_blob_pair = [&](uint64_t old_offset, uint32_t old_size,
                          uint64_t &new_offset, uint32_t &new_size) {
    if (old_size == 0 || !blob_arena_) {
      new_offset = 0;
      new_size = 0;
      return;
    }
    auto old_data = blob_arena_->read(old_offset, old_size);
    if (old_data.empty()) {
      new_offset = 0;
      new_size = 0;
      return;
    }
    BlobRef ref = new_blob_arena->append(old_data.data(), old_data.size());
    new_offset = ref.offset;
    new_size = ref.size;
  };

  // Copy live mmap events — re-pointing blob offsets to new blob file
  // Tombstoned events are dropped: a survivor whose prev_id named one is
  // relinked to the nearest surviving ancestor, or get_history() would stop
  // there (Aleph Zero fix, found by agy). Two passes: after a WAL replay,
  // ids in the file need not be in order.
  std::unordered_map<uint64_t, uint64_t> dropped; // dropped id -> its prev_id
  for (size_t i = 0; i < snapshot_mmap_count; ++i) {
    const TraceEvent *ev = mmap_event_at(i);
    if (ev && (ev->flags & TRACE_FLAG_TOMBSTONE))
      dropped[ev->id] = ev->prev_id;
  }
  for (size_t i = 0; i < frozen_count; ++i) {
    const auto *ev = reinterpret_cast<const TraceEvent *>(
        frozen_delta_bytes_.data() + i * sizeof(TraceEvent));
    if (ev->flags & TRACE_FLAG_TOMBSTONE)
      dropped[ev->id] = ev->prev_id;
  }
  auto survivor = [&dropped](uint64_t id) {
    for (size_t hops = 0; id != 0 && hops <= dropped.size(); ++hops) {
      auto it = dropped.find(id);
      if (it == dropped.end())
        break;
      id = it->second;
    }
    return id;
  };

  size_t write_idx = 0;
  for (size_t i = 0; i < snapshot_mmap_count; ++i) {
    const TraceEvent *ev = mmap_event_at(i);
    if (!ev || (ev->flags & TRACE_FLAG_TOMBSTONE))
      continue;

    auto *dst = reinterpret_cast<TraceEvent *>(
        new_base + sizeof(TraceFileHeader) + write_idx * sizeof(TraceEvent));
    std::memcpy(dst, ev, sizeof(TraceEvent));
    dst->prev_id = survivor(ev->prev_id);

    gc_blob_pair(ev->blob_offset, ev->blob_size, dst->blob_offset,
                dst->blob_size);
    gc_blob_pair(ev->evidence_blob_offset, ev->evidence_blob_size,
                dst->evidence_blob_offset, dst->evidence_blob_size);
    gc_blob_pair(ev->embedding_blob_offset, ev->embedding_blob_size,
                dst->embedding_blob_offset, dst->embedding_blob_size);
    ++write_idx;
  }

  // Copy live frozen delta events
  for (size_t i = 0; i < frozen_count; ++i) {
    const auto *ev = reinterpret_cast<const TraceEvent *>(
        frozen_delta_bytes_.data() + i * sizeof(TraceEvent));
    if (ev->flags & TRACE_FLAG_TOMBSTONE)
      continue;

    auto *dst = reinterpret_cast<TraceEvent *>(
        new_base + sizeof(TraceFileHeader) + write_idx * sizeof(TraceEvent));
    std::memcpy(dst, ev, sizeof(TraceEvent));
    dst->prev_id = survivor(ev->prev_id);

    gc_blob_pair(ev->blob_offset, ev->blob_size, dst->blob_offset,
                dst->blob_size);
    gc_blob_pair(ev->evidence_blob_offset, ev->evidence_blob_size,
                dst->evidence_blob_offset, dst->evidence_blob_size);
    gc_blob_pair(ev->embedding_blob_offset, ev->embedding_blob_size,
                dst->embedding_blob_offset, dst->embedding_blob_size);
    ++write_idx;
  }

  if (compaction_hook_)
    compaction_hook_(); // tests: appends and tombstones that race the compaction

  new_hdr->event_count = write_idx;
  new_hdr->next_event_id = next_event_id_;
  new_hdr->embedding_dim = embedding_dim_;

  // -----------------------------------------------------------------------
  // Step 2b: Durably flush, then ATOMICALLY install the new generation at
  // the stable, caller-facing path (see this function's opening comment).
  // Must happen BEFORE the old file/blob's name is touched: an atomic
  // rename requires the new content to already be complete and durable,
  // or a crash between "rename" and "flush" could leave the stable path
  // pointing at incompletely-written data.
  // -----------------------------------------------------------------------
#ifndef _WIN32
  ::msync(new_base, alloc_size, MS_SYNC);
#endif

  {
    std::error_code ec;
    std::filesystem::rename(new_path, trace_path_, ec);
    if (ec) {
      ::munmap(new_base, alloc_size);
      ::close(new_fd);
      compact_in_progress_.store(false, std::memory_order_release);
      throw std::runtime_error("compact: failed to install new generation "
                               "at stable path: " +
                               ec.message());
    }
    std::filesystem::rename(new_blob_path, stable_blob_path, ec);
    if (ec) {
      // Main file already renamed (now durable and self-consistent) --
      // the blob rename failing here is a real problem for any NEW
      // blob-referencing field written after this point, but not a
      // reason to unwind the already-successful main-file rename half.
      // Surface it; the caller can retry compaction.
      ::munmap(new_base, alloc_size);
      ::close(new_fd);
      compact_in_progress_.store(false, std::memory_order_release);
      throw std::runtime_error("compact: failed to install new blob "
                               "generation at stable path: " +
                               ec.message());
    }
  }

  // -----------------------------------------------------------------------
  // Step 3: µs Freeze — swap in-process pointers to the (already renamed)
  // new generation
  // -----------------------------------------------------------------------
  uint8_t *old_base = nullptr;
  size_t old_size = 0;
  int old_fd = -1;
  {
    std::unique_lock lock(rw_mutex_);

    old_base = mapped_base_;
    old_size = mapped_size_;
    old_fd = fd_;

    // Swap blob arena — close old (now-unlinked inode, still valid via
    // its own open fd until closed), install new
    if (blob_arena_) {
      blob_arena_->close();
    }
    blob_arena_ = std::move(new_blob_arena);

    mapped_base_ = new_base;
    mapped_size_ = alloc_size;
    fd_ = new_fd;
    mmap_event_count_ = write_idx;
    // trace_path_ deliberately NOT reassigned -- it already names the
    // new content, having been the rename() target above. generation_
    // still advances (an internal temp-filename disambiguator only now).
    generation_ = new_gen;

    // Clear frozen buffer
    frozen_delta_bytes_.clear();
    frozen_delta_bytes_.shrink_to_fit();

    // Tombstones set while the events were being copied landed on the old
    // generation: set them again on the new one (Aleph Zero fix).
    for (const auto &[id, erase] : pending_tombstones_) {
      if (const TraceEvent *cev = resolve_event(id))
        apply_tombstone(const_cast<TraceEvent *>(cev), erase);
    }
    pending_tombstones_.clear();

    // Events appended during the compaction went to the RAM delta, protected
    // only by WAL records that the truncate below deletes: they go into the
    // new file now, relinked past dropped events like the rest (Aleph Zero
    // fix, found by agy). The delta is empty afterwards: no event lives only
    // in RAM, so no later append to the file can chain to one.
    size_t new_delta_count = delta_bytes_.size() / sizeof(TraceEvent);
    for (size_t i = 0; i < new_delta_count; ++i) {
      TraceEvent ev{};
      std::memcpy(&ev, delta_bytes_.data() + i * sizeof(TraceEvent), sizeof(TraceEvent));
      ev.prev_id = survivor(ev.prev_id);
      grow_mmap(1);
      auto *dst = reinterpret_cast<TraceEvent *>(
          mapped_base_ + sizeof(TraceFileHeader) + mmap_event_count_ * sizeof(TraceEvent));
      std::memcpy(dst, &ev, sizeof(TraceEvent));
      ++mmap_event_count_;
    }
    delta_bytes_.clear();
    auto *hdr = reinterpret_cast<TraceFileHeader *>(mapped_base_);
    hdr->event_count = mmap_event_count_;
    hdr->next_event_id = next_event_id_;
#ifndef _WIN32
    ::msync(mapped_base_, mapped_size_, MS_SYNC);
#endif

    // Session tails from the compacted file, the drained events included
    rebuild_session_tails();
  }
  // Lock released — reads now see new file

  compact_in_progress_.store(false, std::memory_order_release);

  // -----------------------------------------------------------------------
  // Step 4: Background Cleanup — close the old (already-unlinked-by-
  // rename) file/blob handles. Nothing left to explicitly remove by path
  // -- the renames in Step 2b already atomically replaced them.
  // -----------------------------------------------------------------------
#ifndef _WIN32
  if (old_base)
    ::munmap(old_base, old_size);
  if (old_fd >= 0)
    ::close(old_fd);
#endif

  // ── V4.1: Truncate WAL — all delta data is now in the compacted file ──
  truncate_wal();
  open_wal();
}

// ===========================================================================
// Full Text Retrieval (V4.1)
// ===========================================================================

std::string TraceManager::get_event_text(uint64_t blob_offset,
                                         uint32_t blob_size) const {
  std::shared_lock lock(rw_mutex_);
  if (!blob_arena_ || blob_size == 0) {
    return {};
  }
  auto view = blob_arena_->read(blob_offset, blob_size);
  return std::string(view);
}

// ===========================================================================
// Introspection
// ===========================================================================

size_t TraceManager::size() const {
  std::shared_lock lock(rw_mutex_);
  return mmap_event_count_ + delta_bytes_.size() / sizeof(TraceEvent);
}

size_t TraceManager::mmap_event_count() const {
  std::shared_lock lock(rw_mutex_);
  return mmap_event_count_;
}

size_t TraceManager::delta_event_count() const {
  std::shared_lock lock(rw_mutex_);
  return delta_bytes_.size() / sizeof(TraceEvent);
}

bool TraceManager::has_session(const char *session_id) const {
  std::shared_lock lock(rw_mutex_);
  return session_tails_.contains(std::string(session_id ? session_id : ""));
}

bool TraceManager::tombstone_event(uint64_t event_id, bool erase_text) {
  // Nothing is logged for an event that does not exist (yet), or for one with
  // nothing left to do: a record for an id not yet given would tombstone the
  // event that later gets it, at the next replay (agy). An existing event
  // cannot vanish before the record is written: only compact() removes events,
  // and only tombstoned ones.
  {
    std::shared_lock check(rw_mutex_);
    const TraceEvent *cev = resolve_event(event_id);
    if (!cev)
      return false;
    bool dead = (cev->flags & TRACE_FLAG_TOMBSTONE) != 0;
    bool text = cev->blob_size || cev->evidence_blob_size ||
                cev->embedding_blob_size || cev->text_preview[0];
    if (dead && !(erase_text && text))
      return false;
  }
  // WAL first, under wal_mutex_ alone (ARCHITECTURE §4.2's order): a crash
  // after this cannot leave a RAM-delta event alive, and replay applies it
  // again harmlessly to an event already in the file.
  {
    WalTraceTombstoneRecord rec{};
    rec.event_id = event_id;
    rec.erase_text = erase_text ? 1 : 0;
    WalRecordHeader wal_hdr{};
    wal_hdr.record_type = WAL_RECORD_TRACE_TOMBSTONE;
    wal_hdr.payload_size = static_cast<uint32_t>(sizeof(rec));
    wal_hdr.checksum = hash::fnv1a_64(&rec, sizeof(rec));
    std::lock_guard<std::mutex> wal_lock(wal_mutex_);
    if (wal_stream_.is_open()) {
      wal_stream_.write(reinterpret_cast<const char *>(&wal_hdr), sizeof(wal_hdr));
      wal_stream_.write(reinterpret_cast<const char *>(&rec), sizeof(rec));
      wal_stream_.flush();
    }
  }

  std::unique_lock lock(rw_mutex_);
  // const_cast is safe: we hold the exclusive write lock, and the underlying
  // storage (mmap region or delta buffer) is inherently mutable.
  auto *ev = const_cast<TraceEvent *>(resolve_event(event_id));
  if (!ev)
    return false; // event not found
  bool was_dead = (ev->flags & TRACE_FLAG_TOMBSTONE) != 0;
  bool text_left = ev->blob_size || ev->evidence_blob_size ||
                   ev->embedding_blob_size || ev->text_preview[0];
  if (was_dead && !(erase_text && text_left))
    return false; // already tombstoned (and erased, if that was asked)
  apply_tombstone(ev, erase_text);
  if (compact_in_progress_.load(std::memory_order_acquire))
    pending_tombstones_.emplace_back(event_id, erase_text);
  return true; // tombstoned now, or erased now: something was done
}

void TraceManager::apply_tombstone(TraceEvent *ev, bool erase_text) {
  ev->flags |= TRACE_FLAG_TOMBSTONE;
  if (erase_text) {
    if (blob_arena_) {
      blob_arena_->scrub(ev->blob_offset, ev->blob_size);
      blob_arena_->scrub(ev->evidence_blob_offset, ev->evidence_blob_size);
      blob_arena_->scrub(ev->embedding_blob_offset, ev->embedding_blob_size);
    }
    // Sizes cleared too: compact() copies nothing and rebuild_block_index()
    // indexes nothing for it from now on.
    ev->blob_offset = ev->blob_size = 0;
    ev->evidence_blob_offset = ev->evidence_blob_size = 0;
    ev->embedding_blob_offset = ev->embedding_blob_size = 0;
    std::memset(ev->text_preview, 0, sizeof(ev->text_preview));
  }
#ifndef _WIN32
  auto *at = reinterpret_cast<uint8_t *>(ev);
  if (mapped_base_ && at >= mapped_base_ && at + sizeof(TraceEvent) <= mapped_base_ + mapped_size_) {
    size_t page_size = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
    size_t offset = static_cast<size_t>(at - mapped_base_);
    size_t page_start = (offset / page_size) * page_size;
    size_t sync_len = (offset + sizeof(TraceEvent) - page_start + page_size - 1) / page_size * page_size;
    ::msync(mapped_base_ + page_start, sync_len, MS_SYNC);
  }
#endif
}

bool TraceManager::drop_session(const char *session_id) {
  std::unique_lock lock(rw_mutex_);
  return session_tails_.erase(std::string(session_id ? session_id : "")) > 0;
}

} // namespace aeon

// ===========================================================================
// WAL Methods (V4.1)
// ===========================================================================

namespace aeon {

void TraceManager::open_wal() {
  std::lock_guard<std::mutex> lock(wal_mutex_);
  if (wal_stream_.is_open())
    wal_stream_.close();
  wal_stream_.open(wal_path_, std::ios::binary | std::ios::app);
}

void TraceManager::replay_wal() {
  if (!std::filesystem::exists(wal_path_))
    return;

  auto file_size = std::filesystem::file_size(wal_path_);
  if (file_size == 0)
    return;

  std::ifstream in(wal_path_, std::ios::binary);
  if (!in.is_open())
    return;

  uint64_t bytes_consumed = 0;
  size_t replayed = 0;

  while (in.good() && !in.eof()) {
    // Read WAL record header
    WalRecordHeader wal_hdr{};
    in.read(reinterpret_cast<char *>(&wal_hdr), sizeof(WalRecordHeader));
    if (in.gcount() != sizeof(WalRecordHeader))
      break; // Truncated header
    bytes_consumed += sizeof(WalRecordHeader);

    // Bound payload_size against bytes remaining in the file BEFORE
    // trusting it for anything — see Atlas::replay_wal() for the same
    // fix and rationale.
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
      break;

    // Dispatch on record_type. An unrecognized type, or a recognized type
    // with an unexpected payload_size, is skipped rather than fatal — see
    // Atlas::replay_wal() for the same forward-compat rationale.
    if (wal_hdr.record_type == WAL_RECORD_TRACE &&
        wal_hdr.payload_size == sizeof(TraceEvent)) {
      TraceEvent ev{};
      std::memcpy(&ev, payload.data(), sizeof(TraceEvent));
      std::string sid(ev.session_id,
                      std::min(std::strlen(ev.session_id), size_t{35}));

      if (ev.id == 0) {
        // A record from before ids were reserved ahead of the WAL write: it
        // gets the next id and joins its session's chain, as it always did.
        auto it = session_tails_.find(sid);
        ev.prev_id = (it != session_tails_.end()) ? it->second : 0;
        ev.id = next_event_id_++;
      } else if (resolve_event(ev.id)) {
        next_event_id_ = std::max(next_event_id_, ev.id + 1);
        continue; // already in the file: a crash after a replay, before the log was emptied
      } else {
        // The id and prev_id the event was given when it was appended.
        next_event_id_ = std::max(next_event_id_, ev.id + 1);
      }

      // The first embedding of a file may be one only the log holds.
      if (embedding_dim_ == 0 && ev.embedding_blob_size > 0 &&
          ev.embedding_blob_size % sizeof(float) == 0) {
        embedding_dim_ = ev.embedding_blob_size / sizeof(float);
        if (mapped_base_)
          reinterpret_cast<TraceFileHeader *>(mapped_base_)->embedding_dim = embedding_dim_;
      }

      // Into the trace file, not a RAM buffer: an event only in RAM made a
      // later append to the file chain to it, and the next replay chain it
      // back again (a prev_id cycle).
      grow_mmap(1);
      if (mapped_base_) {
        auto *dst = reinterpret_cast<TraceEvent *>(
            mapped_base_ + sizeof(TraceFileHeader) + mmap_event_count_ * sizeof(TraceEvent));
        std::memcpy(dst, &ev, sizeof(TraceEvent));
        ++mmap_event_count_;
        auto *hdr = reinterpret_cast<TraceFileHeader *>(mapped_base_);
        hdr->event_count = mmap_event_count_;
        hdr->next_event_id = next_event_id_;
      } else {
        size_t old_size = delta_bytes_.size();
        delta_bytes_.resize(old_size + sizeof(TraceEvent));
        std::memcpy(delta_bytes_.data() + old_size, &ev, sizeof(TraceEvent));
      }
      auto it = session_tails_.find(sid);
      if (it == session_tails_.end() || ev.id > it->second)
        session_tails_[sid] = ev.id;
      ++replayed;
    }
    else if (wal_hdr.record_type == WAL_RECORD_TRACE_TOMBSTONE &&
             wal_hdr.payload_size == sizeof(WalTraceTombstoneRecord)) {
      WalTraceTombstoneRecord rec{};
      std::memcpy(&rec, payload.data(), sizeof(rec));
      next_event_id_ = std::max(next_event_id_, rec.event_id + 1);
      if (const TraceEvent *cev = resolve_event(rec.event_id)) {
        apply_tombstone(const_cast<TraceEvent *>(cev), rec.erase_text != 0);
        ++replayed;
      }
    }
    // else: skip (payload already fully consumed above), continue replay.
  }
  in.close();

  // What was replayed is in the file now: make it durable, then empty the log
  // (a crash between the two replays the same records again, and those already
  // in the file are skipped by id).
  // Also when nothing was new (every record already in the file, after a crash
  // between the msync and the removal): the log is spent either way.
  if (mapped_base_) {
    reinterpret_cast<TraceFileHeader *>(mapped_base_)->next_event_id = next_event_id_;
#ifndef _WIN32
    if (replayed > 0)
      ::msync(mapped_base_, mapped_size_, MS_SYNC);
#endif
    std::error_code ec;
    std::filesystem::remove(wal_path_, ec);
  }
}

void TraceManager::truncate_wal() {
  std::lock_guard<std::mutex> lock(wal_mutex_);
  if (wal_stream_.is_open())
    wal_stream_.close();

  std::error_code ec;
  std::filesystem::remove(wal_path_, ec);
}

} // namespace aeon
