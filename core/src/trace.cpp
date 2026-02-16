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
  auto blob_path = trace_path_.parent_path() /
                   ("trace_blobs_gen" + std::to_string(generation_) + ".bin");
  blob_arena_ = std::make_unique<BlobArena>(blob_path);

  // ── V4.1 WAL: crash recovery ──
  wal_path_ = trace_path_;
  wal_path_ += ".wal";
  replay_wal();
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

    // Rebuild session tails from on-disk events
    rebuild_session_tails();
  }
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

// ===========================================================================
// Append Event
// ===========================================================================

uint64_t TraceManager::append_event(const char *session_id, uint16_t role,
                                    const char *text, uint64_t atlas_id) {
  // ── Step 1: Serialize data & write blob (NO LOCK) ──
  // Build the TraceEvent payload outside of any lock.
  TraceEvent ev{};
  std::memset(&ev, 0, sizeof(TraceEvent));

  ev.timestamp = now_micros();
  ev.atlas_id = atlas_id;
  ev.role = role;
  ev.flags = 0;

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
    // Inline preview: first 63 chars + null terminator
    std::strncpy(ev.text_preview, text, sizeof(ev.text_preview) - 1);
    ev.text_preview[sizeof(ev.text_preview) - 1] = '\0';
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

  if (needs_wal) {
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
  }

  // ── Step 3: lock(rw_mutex_) → assign IDs, append to buffer → unlock ──
  std::unique_lock lock(rw_mutex_);

  // Determine prev_id for this session
  uint64_t prev_id = 0;
  std::string sid(session_id ? session_id : "");
  auto it = session_tails_.find(sid);
  if (it != session_tails_.end()) {
    prev_id = it->second;
  }

  // Assign sequential event ID and prev_id
  ev.id = next_event_id_++;
  ev.prev_id = prev_id;

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

  // Update session tail
  session_tails_[sid] = event_id;
  return event_id;
}

uint64_t TraceManager::append_mmap(const char *session_id, uint16_t role,
                                   const char *text, size_t text_len,
                                   uint64_t atlas_id, uint64_t prev_id) {
  // Ensure we have space
  grow_mmap(1);

  auto *ev =
      reinterpret_cast<TraceEvent *>(mapped_base_ + sizeof(TraceFileHeader) +
                                     mmap_event_count_ * sizeof(TraceEvent));
  std::memset(ev, 0, sizeof(TraceEvent));

  ev->id = next_event_id_++;
  ev->prev_id = prev_id;
  ev->atlas_id = atlas_id;
  ev->timestamp = now_micros();
  ev->role = role;
  ev->flags = 0;

  if (session_id) {
    std::strncpy(ev->session_id, session_id, sizeof(ev->session_id) - 1);
    ev->session_id[sizeof(ev->session_id) - 1] = '\0';
  }

  // V4.1: Write full text to blob arena, keep 63-char preview inline
  if (text && text_len > 0) {
    if (blob_arena_) {
      BlobRef ref = blob_arena_->append(text, text_len);
      ev->blob_offset = ref.offset;
      ev->blob_size = ref.size;
    }
    std::strncpy(ev->text_preview, text, sizeof(ev->text_preview) - 1);
    ev->text_preview[sizeof(ev->text_preview) - 1] = '\0';
  }

  ++mmap_event_count_;

  // Update file header
  auto *hdr = reinterpret_cast<TraceFileHeader *>(mapped_base_);
  hdr->event_count = mmap_event_count_;
  hdr->next_event_id = next_event_id_;

  return ev->id;
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
// Shadow Compaction
// ===========================================================================

void TraceManager::compact() {
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
  std::filesystem::path new_path =
      trace_path_.parent_path() /
      ("trace_gen" + std::to_string(new_gen) + ".bin");

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
  // TODO: Win32 implementation
  compact_in_progress_.store(false, std::memory_order_release);
  return;
#endif

  // Write header
  auto *new_hdr = reinterpret_cast<TraceFileHeader *>(new_base);
  std::memset(new_hdr, 0, sizeof(TraceFileHeader));
  new_hdr->magic = TRACE_MAGIC;
  new_hdr->version = 2; // V4.1: Blob arena format

  // ── V4.1: Create new-generation blob arena for GC ──
  auto new_blob_path = new_path.parent_path() /
                       ("trace_blobs_gen" + std::to_string(new_gen) + ".bin");
  auto new_blob_arena = std::make_unique<BlobArena>(new_blob_path);

  // Copy live mmap events — re-pointing blob offsets to new blob file
  size_t write_idx = 0;
  for (size_t i = 0; i < snapshot_mmap_count; ++i) {
    const TraceEvent *ev = mmap_event_at(i);
    if (!ev || (ev->flags & TRACE_FLAG_TOMBSTONE))
      continue;

    auto *dst = reinterpret_cast<TraceEvent *>(
        new_base + sizeof(TraceFileHeader) + write_idx * sizeof(TraceEvent));
    std::memcpy(dst, ev, sizeof(TraceEvent));

    // GC: copy live blob data to new blob file, update offset
    if (ev->blob_size > 0 && blob_arena_) {
      auto old_text = blob_arena_->read(ev->blob_offset, ev->blob_size);
      if (!old_text.empty()) {
        BlobRef new_ref =
            new_blob_arena->append(old_text.data(), old_text.size());
        dst->blob_offset = new_ref.offset;
        dst->blob_size = new_ref.size;
      }
    }
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

    // GC: copy live blob data to new blob file, update offset
    if (ev->blob_size > 0 && blob_arena_) {
      auto old_text = blob_arena_->read(ev->blob_offset, ev->blob_size);
      if (!old_text.empty()) {
        BlobRef new_ref =
            new_blob_arena->append(old_text.data(), old_text.size());
        dst->blob_offset = new_ref.offset;
        dst->blob_size = new_ref.size;
      }
    }
    ++write_idx;
  }

  new_hdr->event_count = write_idx;
  new_hdr->next_event_id = next_event_id_;

  // -----------------------------------------------------------------------
  // Step 3: µs Freeze — swap file pointers
  // -----------------------------------------------------------------------
  std::filesystem::path old_path = trace_path_;
  std::filesystem::path old_blob_path;
  uint8_t *old_base = nullptr;
  size_t old_size = 0;
  int old_fd = -1;
  {
    std::unique_lock lock(rw_mutex_);

    old_base = mapped_base_;
    old_size = mapped_size_;
    old_fd = fd_;

    // Swap blob arena — close old, install new
    if (blob_arena_) {
      old_blob_path = blob_arena_->path();
      blob_arena_->close();
    }
    blob_arena_ = std::move(new_blob_arena);

    mapped_base_ = new_base;
    mapped_size_ = alloc_size;
    fd_ = new_fd;
    mmap_event_count_ = write_idx;
    trace_path_ = new_path;
    generation_ = new_gen;

    // Clear frozen buffer
    frozen_delta_bytes_.clear();
    frozen_delta_bytes_.shrink_to_fit();

    // Rebuild session tails from the compacted file
    rebuild_session_tails();

    // Also account for any events added to delta_bytes_ during compaction
    size_t new_delta_count = delta_bytes_.size() / sizeof(TraceEvent);
    for (size_t i = 0; i < new_delta_count; ++i) {
      const auto *ev = reinterpret_cast<const TraceEvent *>(
          delta_bytes_.data() + i * sizeof(TraceEvent));
      std::string sid(ev->session_id,
                      std::min(std::strlen(ev->session_id), size_t{35}));
      session_tails_[sid] = ev->id;
    }
  }
  // Lock released — reads now see new file

  compact_in_progress_.store(false, std::memory_order_release);

  // -----------------------------------------------------------------------
  // Step 4: Background Cleanup — safe to close + delete old file
  // -----------------------------------------------------------------------
#ifndef _WIN32
  if (old_base)
    ::munmap(old_base, old_size);
  if (old_fd >= 0)
    ::close(old_fd);
#endif

  std::error_code ec;
  std::filesystem::remove(old_path, ec);
  // Ignore removal errors — old file may have already been cleaned up

  // ── V4.1: Delete old blob file ──
  if (!old_blob_path.empty()) {
    std::filesystem::remove(old_blob_path, ec);
  }

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

  while (in.good() && !in.eof()) {
    // Read WAL record header
    WalRecordHeader wal_hdr{};
    in.read(reinterpret_cast<char *>(&wal_hdr), sizeof(WalRecordHeader));
    if (in.gcount() != sizeof(WalRecordHeader))
      break; // Truncated header

    // Validate record type
    if (wal_hdr.record_type != WAL_RECORD_TRACE)
      break;

    // Validate payload size
    if (wal_hdr.payload_size != sizeof(TraceEvent))
      break;

    // Read payload
    TraceEvent ev{};
    in.read(reinterpret_cast<char *>(&ev), sizeof(TraceEvent));
    if (static_cast<uint32_t>(in.gcount()) != sizeof(TraceEvent))
      break;

    // Verify checksum
    uint64_t computed = hash::fnv1a_64(&ev, sizeof(TraceEvent));
    if (computed != wal_hdr.checksum)
      break;

    // ── Record is valid: reconstruct delta buffer ──
    // Re-chain: look up current session tail to set prev_id
    std::string sid(ev.session_id,
                    std::min(std::strlen(ev.session_id), size_t{35}));
    auto it = session_tails_.find(sid);
    uint64_t prev_id = (it != session_tails_.end()) ? it->second : 0;

    // Assign sequential event ID and re-chain prev_id
    ev.id = next_event_id_++;
    ev.prev_id = prev_id;

    // Append to delta buffer
    size_t old_size = delta_bytes_.size();
    delta_bytes_.resize(old_size + sizeof(TraceEvent));
    std::memcpy(delta_bytes_.data() + old_size, &ev, sizeof(TraceEvent));

    // Update session tail for this event
    session_tails_[sid] = ev.id;
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
