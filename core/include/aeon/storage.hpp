#pragma once

#include "aeon/epoch.hpp"
#include "aeon/schema.hpp"
#include <algorithm>
#include <atomic>
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
      : fd_(other.fd_), data_(other.data_.load(std::memory_order_relaxed)),
        size_(other.size_), epoch_mgr_(other.epoch_mgr_) {
    other.fd_ = -1;
    other.data_.store(nullptr, std::memory_order_relaxed);
    other.size_ = 0;
    other.epoch_mgr_ = nullptr;
  }

  MemoryFile &operator=(MemoryFile &&other) noexcept {
    if (this != &other) {
      close();
      fd_ = other.fd_;
      data_.store(other.data_.load(std::memory_order_relaxed),
                  std::memory_order_relaxed);
      size_ = other.size_;
      epoch_mgr_ = other.epoch_mgr_;
      other.fd_ = -1;
      other.data_.store(nullptr, std::memory_order_relaxed);
      other.size_ = 0;
      other.epoch_mgr_ = nullptr;
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
      data_.store(static_cast<uint8_t *>(ptr), std::memory_order_release);

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
      data_.store(static_cast<uint8_t *>(ptr), std::memory_order_release);

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

  /// Set the EpochManager for deferred reclamation in grow().
  void set_epoch_manager(aeon::EpochManager *mgr) { epoch_mgr_ = mgr; }

  void close() {
    // Drain all active readers before unmapping
    if (epoch_mgr_) {
      epoch_mgr_->drain_readers();
    }
    auto *d = data_.load(std::memory_order_acquire);
    if (d) {
      munmap(d, size_);
      data_.store(nullptr, std::memory_order_release);
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
    auto *current_data = data_.load(std::memory_order_acquire);
    if (!current_data)
      return std::unexpected(StorageError::IOError);

    auto *current_header = reinterpret_cast<AtlasHeader *>(current_data);
    if (new_capacity <= current_header->capacity) {
      return {}; // No need to grow
    }

    size_t new_size = sizeof(AtlasHeader) + (new_capacity * sizeof(Node));

    // 1. Extend file FIRST (safe while old mapping still exists)
    if (ftruncate(fd_, new_size) != 0) {
      return std::unexpected(StorageError::AllocationFailed);
    }

    // 2. Create NEW mapping covering the extended file
    void *new_ptr =
        mmap(nullptr, new_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (new_ptr == MAP_FAILED) {
      return std::unexpected(StorageError::AllocationFailed);
    }

    // 3. Capture old mapping for deferred reclamation
    void *old_data = current_data;
    size_t old_size = size_;

    // 4. Atomically swap to new mapping (release ensures visibility)
    data_.store(static_cast<uint8_t *>(new_ptr), std::memory_order_release);
    size_ = new_size;

    // 5. Retire old mapping via EBR (deferred munmap)
    if (epoch_mgr_ && old_data) {
      epoch_mgr_->retire(old_data, old_size);
    } else if (old_data) {
      munmap(old_data, old_size); // Fallback: immediate cleanup
    }

    // Update header on new mapping
    get_header()->capacity = new_capacity;

    return {};
  }

  // --- Accessors ---

  AtlasHeader *get_header() {
    auto *d = data_.load(std::memory_order_acquire);
    return d ? reinterpret_cast<AtlasHeader *>(d) : nullptr;
  }
  const AtlasHeader *get_header() const {
    auto *d = data_.load(std::memory_order_acquire);
    return d ? reinterpret_cast<const AtlasHeader *>(d) : nullptr;
  }

  Node *get_node(size_t index) {
    auto *header = get_header();
    if (!header || index >= header->capacity)
      return nullptr;
    size_t offset = sizeof(AtlasHeader) + (index * sizeof(Node));
    return reinterpret_cast<Node *>(data_.load(std::memory_order_acquire) +
                                    offset);
  }

  const Node *get_node(size_t index) const {
    auto *header = get_header();
    if (!header || index >= header->capacity)
      return nullptr;
    size_t offset = sizeof(AtlasHeader) + (index * sizeof(Node));
    return reinterpret_cast<const Node *>(
        data_.load(std::memory_order_acquire) + offset);
  }

private:
  int fd_ = -1;
  std::atomic<uint8_t *> data_{nullptr};
  size_t size_ = 0;
  aeon::EpochManager *epoch_mgr_ = nullptr;
};

} // namespace aeon::storage
