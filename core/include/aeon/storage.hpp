#pragma once

#include "aeon/schema.hpp"
#include <algorithm>
#include <expected>
#include <filesystem>
#include <string>

// Platform specific includes for mmap
#if defined(_WIN32)
#error "Windows is not supported in this phase"
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace aeon::storage {

enum class StorageError {
  Ok,
  FileNotFound,
  PermissionDenied,
  InvalidFormat,
  VersionMismatch,
  AllocationFailed,
  IOError
};

class MemoryFile {
public:
  MemoryFile() = default;

  // Disable copy
  MemoryFile(const MemoryFile &) = delete;
  MemoryFile &operator=(const MemoryFile &) = delete;

  // Enable move
  MemoryFile(MemoryFile &&other) noexcept
      : fd_(other.fd_), data_(other.data_), size_(other.size_) {
    other.fd_ = -1;
    other.data_ = nullptr;
    other.size_ = 0;
  }

  MemoryFile &operator=(MemoryFile &&other) noexcept {
    if (this != &other) {
      close();
      fd_ = other.fd_;
      data_ = other.data_;
      size_ = other.size_;
      other.fd_ = -1;
      other.data_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  ~MemoryFile() { close(); }

  /**
   * @brief Opens or creates a memory mapped file.
   * If creating, initializes the global header.
   */
  std::expected<void, StorageError> open(const std::filesystem::path &path,
                                         size_t initial_capacity = 1000) {
    bool exists = std::filesystem::exists(path);
    int flags = O_RDWR | O_CREAT;
    fd_ = ::open(path.c_str(), flags, 0644);

    if (fd_ == -1) {
      return std::unexpected(StorageError::IOError);
    }

    if (!exists) {
      // New file: Initialize header
      size_ = sizeof(AtlasHeader) + (initial_capacity * sizeof(Node));
      if (ftruncate(fd_, size_) != 0) {
        return std::unexpected(StorageError::AllocationFailed);
      }

      // Initial map to write header
      void *ptr =
          mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
      if (ptr == MAP_FAILED) {
        return std::unexpected(StorageError::AllocationFailed);
      }
      data_ = static_cast<uint8_t *>(ptr);

      // Advise kernel to use Huge Pages if possible (2MB pages)
      // Reduces TLB misses for large datasets.
      // MADV_HUGEPAGE is Linux specific. On Mac/BSD it might be MADV_HUGEPAGE
      // or ignored.
#if defined(__linux__) && defined(MADV_HUGEPAGE)
      madvise(ptr, size_, MADV_HUGEPAGE);
#endif
      // Also advise random access since we jump around the tree
      madvise(ptr, size_, MADV_RANDOM);

      // Write Header
      auto *header = get_header();
      header->magic = ATLAS_MAGIC;
      header->version = ATLAS_VERSION;
      header->node_count = 0;
      header->capacity = initial_capacity;
      // Zero out padding
      std::fill(std::begin(header->reserved), std::end(header->reserved), 0);

    } else {
      // Existing file: Get size
      struct stat sb;
      if (fstat(fd_, &sb) == -1) {
        return std::unexpected(StorageError::IOError);
      }
      size_ = sb.st_size;

      void *ptr =
          mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
      if (ptr == MAP_FAILED) {
        return std::unexpected(StorageError::AllocationFailed);
      }
      data_ = static_cast<uint8_t *>(ptr);

      // Advise kernel to use Huge Pages if possible (2MB pages)
      // Reduces TLB misses for large datasets.
      // MADV_HUGEPAGE is Linux specific. On Mac/BSD it might be MADV_HUGEPAGE
      // or ignored.
#if defined(__linux__) && defined(MADV_HUGEPAGE)
      madvise(ptr, size_, MADV_HUGEPAGE);
#endif
      // Also advise random access since we jump around the tree
      madvise(ptr, size_, MADV_RANDOM);

      // Validate Header
      if (size_ < sizeof(AtlasHeader)) {
        return std::unexpected(StorageError::InvalidFormat);
      }
      auto *header = get_header();
      if (header->magic != ATLAS_MAGIC) {
        return std::unexpected(StorageError::InvalidFormat);
      }
      if (header->version != ATLAS_VERSION) {
        return std::unexpected(StorageError::VersionMismatch);
      }
    }

    return {};
  }

  void close() {
    if (data_) {
      munmap(data_, size_);
      data_ = nullptr;
    }
    if (fd_ != -1) {
      ::close(fd_);
      fd_ = -1;
    }
    size_ = 0;
  }

  /**
   * @brief Expands the file to hold new_capacity nodes.
   */
  std::expected<void, StorageError> grow(size_t new_capacity) {
    if (!data_)
      return std::unexpected(StorageError::IOError);

    auto current_header = get_header();
    if (new_capacity <= current_header->capacity) {
      return {}; // No need to grow
    }

    size_t new_size = sizeof(AtlasHeader) + (new_capacity * sizeof(Node));

    // Must unmap before truncating in some OSes/Safe practice
    munmap(data_, size_);
    data_ = nullptr;

    if (ftruncate(fd_, new_size) != 0) {
      /// ftruncate failed; the previous mapping was already released.
      return std::unexpected(StorageError::AllocationFailed);
    }

    void *ptr =
        mmap(nullptr, new_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (ptr == MAP_FAILED) {
      return std::unexpected(StorageError::AllocationFailed);
    }

    data_ = static_cast<uint8_t *>(ptr);
    size_ = new_size;

    // Update header
    get_header()->capacity = new_capacity;

    return {};
  }

  // --- Accessors ---

  AtlasHeader *get_header() { return reinterpret_cast<AtlasHeader *>(data_); }
  const AtlasHeader *get_header() const {
    return reinterpret_cast<const AtlasHeader *>(data_);
  }

  Node *get_node(size_t index) {
    if (index >= get_header()->capacity)
      return nullptr;
    // Strict pointer arithmetic: base + header + index * stride
    // We cast to uintptr_t or char* to do byte math, then cast to Node*
    size_t offset = sizeof(AtlasHeader) + (index * sizeof(Node));
    return reinterpret_cast<Node *>(data_ + offset);
  }

  const Node *get_node(size_t index) const {
    if (index >= get_header()->capacity)
      return nullptr;
    size_t offset = sizeof(AtlasHeader) + (index * sizeof(Node));
    return reinterpret_cast<const Node *>(data_ + offset);
  }

private:
  int fd_ = -1;
  uint8_t *data_ = nullptr;
  size_t size_ = 0;
};

} // namespace aeon::storage
