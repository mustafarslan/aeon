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
 * Thread safety: the caller (TraceManager) holds its locks around
 * append/read calls. BlobArena itself is NOT thread-safe.
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

    if (exists) {
      // Get existing file size
      struct stat st{};
      if (::fstat(fd_, &st) < 0) {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("BlobArena: fstat failed");
      }
      file_size_ = static_cast<size_t>(st.st_size);
      write_offset_ = file_size_;
    } else {
      file_size_ = 0;
      write_offset_ = 0;
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

    mapped_base_ = static_cast<uint8_t *>(::mmap(
        nullptr, file_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));
    if (mapped_base_ == MAP_FAILED) {
      mapped_base_ = nullptr;
      ::close(fd_);
      fd_ = -1;
      throw std::runtime_error("BlobArena: mmap failed");
    }
    mapped_size_ = file_size_;
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
    // Grow file if needed
    if (write_offset_ + size > file_size_) {
      grow(write_offset_ + size);
    }

    // Copy data into mmap
    std::memcpy(mapped_base_ + write_offset_, data, size);

    BlobRef ref{};
    ref.offset = write_offset_;
    ref.size = static_cast<uint32_t>(size);

    write_offset_ += size;

    // Flush to disk (msync the written region)
    size_t page_size = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
    size_t page_start = (ref.offset / page_size) * page_size;
    size_t sync_len =
        (write_offset_ - page_start + page_size - 1) / page_size * page_size;
    ::msync(mapped_base_ + page_start, sync_len, MS_SYNC);

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
    if (!mapped_base_ || size == 0) {
      return {};
    }
    if (offset + size > write_offset_) {
      return {}; // Out of bounds — return empty
    }
    return {reinterpret_cast<const char *>(mapped_base_ + offset), size};
  }

  /// Close the arena file and unmap.
  void close() {
#ifndef _WIN32
    if (mapped_base_) {
      ::munmap(mapped_base_, mapped_size_);
      mapped_base_ = nullptr;
      mapped_size_ = 0;
    }
    if (fd_ >= 0) {
      // Truncate file to actual data size before closing
      if (write_offset_ > 0) {
        ::ftruncate(fd_, static_cast<off_t>(write_offset_));
      }
      ::close(fd_);
      fd_ = -1;
    }
#endif
  }

  /// Current write offset (total bytes written).
  size_t write_offset() const { return write_offset_; }

  /// File path.
  const std::filesystem::path &path() const { return path_; }

private:
  static constexpr size_t kInitialSize = 4096; // 4KB initial allocation
  static constexpr size_t kGrowthFactor = 2;   // Double on grow

  std::filesystem::path path_;
  int fd_ = -1;
  uint8_t *mapped_base_ = nullptr;
  size_t mapped_size_ = 0;
  size_t file_size_ = 0;
  size_t write_offset_ = 0;

  /// Grow the file and re-mmap.
  void grow(size_t needed) {
#ifndef _WIN32
    size_t new_size = file_size_;
    while (new_size < needed) {
      new_size = std::max(new_size * kGrowthFactor, needed);
    }

    // Unmap current
    if (mapped_base_) {
      ::munmap(mapped_base_, mapped_size_);
      mapped_base_ = nullptr;
    }

    // Extend file
    if (::ftruncate(fd_, static_cast<off_t>(new_size)) < 0) {
      throw std::runtime_error("BlobArena: grow ftruncate failed");
    }
    file_size_ = new_size;

    // Re-mmap
    mapped_base_ = static_cast<uint8_t *>(::mmap(
        nullptr, file_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));
    if (mapped_base_ == MAP_FAILED) {
      mapped_base_ = nullptr;
      throw std::runtime_error("BlobArena: grow mmap failed");
    }
    mapped_size_ = file_size_;
#endif
  }
};

} // namespace aeon
