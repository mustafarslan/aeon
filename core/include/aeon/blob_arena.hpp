#pragma once

/**
 * @file blob_arena.hpp
 * @brief Append-only mmap-backed blob store for Trace event text.
 *
 * V4.1: Removes the 440-character text ceiling by storing full event text
 * in a sidecar blob file. TraceEvent keeps a (blob_offset, blob_size)
 * pointer into this arena.
 *
 * Design:
 *   - Append-only: blobs are never modified in-place
 *   - Zero-copy reads via mmap (string_view over mapped region)
 *   - Generational naming for compaction GC (trace_blobs_genN.bin)
 *   - File layout: raw concatenated byte blobs, no header or framing
 *     (TraceEvent carries the offset+size metadata)
 */

#include <atomic>
#include <mutex>
#include <utility>
#include <vector>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace aeon {

/// Return type for BlobArena::append — offset and size of the written blob.
struct BlobRef {
  uint64_t offset;
  uint32_t size;
};

/**
 * @brief Append-only mmap-backed blob store.
 *
 * Each blob is a contiguous byte range in the file. Reads return a
 * string_view directly over the mmap'd region (zero-copy).
 *
 * Thread safety (Aleph Zero fix, found by agy): append, scrub, grow and close
 * are serialized by the arena's own mutex -- TraceManager::append_event()
 * writes blobs outside its locks, so two appends raced on write_offset_ and on
 * grow()'s remap. read() takes no lock: the mapping and the write offset are
 * atomics, the offset published only after the bytes are copied, and grow()
 * retires the old mapping instead of unmapping it, so a view a reader holds
 * stays valid until close().
 */
class BlobArena {
public:
  /**
   * @brief Open or create a blob arena file.
   * @param path  File path (e.g., "memory/trace_blobs_gen0.bin")
   */
  explicit BlobArena(std::filesystem::path path) : path_(std::move(path)) {
#ifndef _WIN32
    bool exists = std::filesystem::exists(path_);

    fd_ = ::open(path_.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd_ < 0) {
      throw std::runtime_error("BlobArena: failed to open " + path_.string());
    }

    size_t written = 0;
    if (exists) {
      // Get existing file size
      struct stat st{};
      if (::fstat(fd_, &st) < 0) {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("BlobArena: fstat failed");
      }
      file_size_ = static_cast<size_t>(st.st_size);
      written = file_size_;
    } else {
      file_size_ = 0;
    }

    // Ensure minimum allocation for mmap (can't mmap zero-length file)
    if (file_size_ == 0) {
      file_size_ = kInitialSize;
      if (::ftruncate(fd_, static_cast<off_t>(file_size_)) < 0) {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("BlobArena: initial ftruncate failed");
      }
    }

    auto *base = static_cast<uint8_t *>(::mmap(
        nullptr, file_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));
    if (base == MAP_FAILED) {
      ::close(fd_);
      fd_ = -1;
      throw std::runtime_error("BlobArena: mmap failed");
    }
    mapped_size_ = file_size_;
    mapped_base_.store(base, std::memory_order_release);
    write_offset_.store(written, std::memory_order_release);
#endif
  }

  ~BlobArena() { close(); }

  // Non-copyable
  BlobArena(const BlobArena &) = delete;
  BlobArena &operator=(const BlobArena &) = delete;

  /**
   * @brief Append a blob to the arena.
   * @param data  Pointer to blob data
   * @param size  Byte length of blob data
   * @return BlobRef with offset and size of the written blob
   */
  BlobRef append(const char *data, size_t size) {
    if (!data || size == 0) {
      return {0, 0};
    }

#ifndef _WIN32
    std::lock_guard<std::mutex> guard(mutex_);
    size_t at = write_offset_.load(std::memory_order_relaxed);
    // Grow file if needed
    if (at + size > file_size_) {
      grow(at + size);
    }
    uint8_t *base = mapped_base_.load(std::memory_order_relaxed);

    // Copy data into mmap, then publish the new end to readers
    std::memcpy(base + at, data, size);
    write_offset_.store(at + size, std::memory_order_release);

    BlobRef ref{};
    ref.offset = at;
    ref.size = static_cast<uint32_t>(size);

    // Flush to disk (msync the written region)
    size_t page_size = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
    size_t page_start = (ref.offset / page_size) * page_size;
    size_t sync_len =
        (at + size - page_start + page_size - 1) / page_size * page_size;
    ::msync(base + page_start, sync_len, MS_SYNC);

    return ref;
#else
    return {0, 0};
#endif
  }

  /**
   * @brief Read a blob from the arena (zero-copy via mmap).
   * @param offset  Byte offset of the blob
   * @param size    Byte length of the blob
   * @return string_view over the mapped region
   */
  std::string_view read(uint64_t offset, uint32_t size) const {
    uint64_t written = write_offset_.load(std::memory_order_acquire);
    uint8_t *base = mapped_base_.load(std::memory_order_acquire);
    if (!base || size == 0) {
      return {};
    }
    if (offset > written || size > written - offset) {
      return {}; // Out of bounds — return empty
    }
    return {reinterpret_cast<const char *>(base + offset), size};
  }

  /**
   * @brief Overwrite a blob with zeros in place and flush it (Aleph Zero fix:
   * erasing a tombstoned event's text from the disk).
   * @return false when the range is not a written blob of this arena
   */
  bool scrub(uint64_t offset, uint32_t size) {
#ifndef _WIN32
    std::lock_guard<std::mutex> guard(mutex_);
    uint64_t written = write_offset_.load(std::memory_order_relaxed);
    uint8_t *base = mapped_base_.load(std::memory_order_relaxed);
    if (!base || size == 0 || offset > written || size > written - offset)
      return false;
    std::memset(base + offset, 0, size);
    size_t page_size = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
    size_t page_start = (offset / page_size) * page_size;
    size_t sync_len = (offset + size - page_start + page_size - 1) / page_size * page_size;
    ::msync(base + page_start, sync_len, MS_SYNC);
    return true;
#else
    (void)offset;
    (void)size;
    return false;
#endif
  }

  /// Close the arena file and unmap (retired mappings too).
  void close() {
#ifndef _WIN32
    std::lock_guard<std::mutex> guard(mutex_);
    uint8_t *base = mapped_base_.exchange(nullptr, std::memory_order_acq_rel);
    if (base) {
      ::munmap(base, mapped_size_);
      mapped_size_ = 0;
    }
    for (auto &[old_base, old_size] : retired_)
      ::munmap(old_base, old_size);
    retired_.clear();
    if (fd_ >= 0) {
      // Truncate file to actual data size before closing
      uint64_t written = write_offset_.load(std::memory_order_relaxed);
      if (written > 0) {
        ::ftruncate(fd_, static_cast<off_t>(written));
      }
      ::close(fd_);
      fd_ = -1;
    }
#endif
  }

  /// Current write offset (total bytes written).
  size_t write_offset() const { return write_offset_.load(std::memory_order_acquire); }

  /// File path.
  const std::filesystem::path &path() const { return path_; }

private:
  static constexpr size_t kInitialSize = 4096; // 4KB initial allocation
  static constexpr size_t kGrowthFactor = 2;   // Double on grow

  std::filesystem::path path_;
  int fd_ = -1;
  std::atomic<uint8_t *> mapped_base_{nullptr};
  size_t mapped_size_ = 0;
  size_t file_size_ = 0;
  std::atomic<uint64_t> write_offset_{0};
  std::mutex mutex_;
  std::vector<std::pair<uint8_t *, size_t>> retired_; // mappings replaced by grow(), unmapped at close()

  /// Grow the file and map it again; the old mapping is retired, not unmapped
  /// (a reader may hold a view into it). Caller holds mutex_.
  void grow(size_t needed) {
#ifndef _WIN32
    size_t new_size = file_size_;
    while (new_size < needed) {
      new_size = std::max(new_size * kGrowthFactor, needed);
    }

    // Extend file
    if (::ftruncate(fd_, static_cast<off_t>(new_size)) < 0) {
      throw std::runtime_error("BlobArena: grow ftruncate failed");
    }

    auto *base = static_cast<uint8_t *>(::mmap(
        nullptr, new_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));
    if (base == MAP_FAILED) {
      throw std::runtime_error("BlobArena: grow mmap failed");
    }
    uint8_t *old = mapped_base_.exchange(base, std::memory_order_acq_rel);
    if (old)
      retired_.emplace_back(old, mapped_size_);
    file_size_ = new_size;
    mapped_size_ = new_size;
#endif
  }
};

} // namespace aeon
