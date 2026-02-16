#pragma once

#include "aeon/epoch.hpp"
#include "aeon/platform.hpp"
#include "aeon/schema.hpp"
#include <algorithm>
#include <atomic>
#include <expected>
#include <filesystem>
#include <string>

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
      : handle_(other.handle_),
        data_(other.data_.load(std::memory_order_relaxed)), size_(other.size_),
        stride_(other.stride_), epoch_mgr_(other.epoch_mgr_) {
    other.handle_ = platform::INVALID_FILE_HANDLE;
    other.data_.store(nullptr, std::memory_order_relaxed);
    other.size_ = 0;
    other.stride_ = 0;
    other.epoch_mgr_ = nullptr;
  }

  MemoryFile &operator=(MemoryFile &&other) noexcept {
    if (this != &other) {
      close();
      handle_ = other.handle_;
      data_.store(other.data_.load(std::memory_order_relaxed),
                  std::memory_order_relaxed);
      size_ = other.size_;
      stride_ = other.stride_;
      epoch_mgr_ = other.epoch_mgr_;
      other.handle_ = platform::INVALID_FILE_HANDLE;
      other.data_.store(nullptr, std::memory_order_relaxed);
      other.size_ = 0;
      other.stride_ = 0;
      other.epoch_mgr_ = nullptr;
    }
    return *this;
  }

  ~MemoryFile() { close(); }

  /**
   * @brief Opens or creates a memory mapped file.
   *
   * For NEW files: initializes AtlasHeader with the provided dim,
   * metadata_size, and quantization_type. Computes node_byte_stride
   * with quantization awareness (INT8 = 1B/dim vs FP32 = 4B/dim).
   *
   * For EXISTING files: reads dim, metadata_size, and node_byte_stride
   * from the on-disk AtlasHeader. The caller's params are ignored —
   * the file is authoritative.
   *
   * @param path             File path
   * @param initial_capacity Initial node slots (new files only)
   * @param dim              Embedding dimensionality (new files only)
   * @param metadata_size    Metadata block size (new files only)
   * @param quantization_type QUANT_FP32 or QUANT_INT8_SYMMETRIC (new files
   * only)
   */
  std::expected<void, StorageError>
  open(const std::filesystem::path &path, size_t initial_capacity = 1000,
       uint32_t dim = EMBEDDING_DIM_DEFAULT,
       uint32_t metadata_size = METADATA_SIZE_DEFAULT,
       uint32_t quantization_type = QUANT_FP32) {
    bool exists = std::filesystem::exists(path);

#if defined(AEON_PLATFORM_WINDOWS)
    handle_ = platform::file_open(path.string().c_str());
    if (handle_ != platform::INVALID_FILE_HANDLE) {
      exists = platform::file_existed_before_open(handle_);
    }
#else
    handle_ = platform::file_open(path.c_str(), 0644);
#endif

    if (handle_ == platform::INVALID_FILE_HANDLE) {
      return std::unexpected(StorageError::IOError);
    }

    if (!exists) {
      // ── New file: compute stride with quantization awareness ──
      stride_ = compute_node_stride(dim, metadata_size, quantization_type);

      size_ = sizeof(AtlasHeader) + (initial_capacity * stride_);
      if (!platform::file_resize(handle_, size_)) {
        return std::unexpected(StorageError::AllocationFailed);
      }

      void *ptr = platform::mem_map(handle_, size_);
      if (ptr == platform::MAP_FAILED_PTR) {
        return std::unexpected(StorageError::AllocationFailed);
      }
      data_.store(static_cast<uint8_t *>(ptr), std::memory_order_release);

      platform::advise_hugepage(ptr, size_);
      platform::advise_random(ptr, size_);

      auto *header = get_header();
      header->magic = ATLAS_MAGIC;
      header->version = ATLAS_VERSION;
      header->node_count = 0;
      header->capacity = initial_capacity;
      header->dim = dim;
      header->metadata_size = metadata_size;
      header->node_byte_stride = stride_;
      header->quantization_type = quantization_type;
      std::fill(std::begin(header->reserved), std::end(header->reserved), 0);

    } else {
      // ── Existing file: read stride from header ──
      size_ = platform::file_size(handle_);
      if (size_ == 0) {
        return std::unexpected(StorageError::IOError);
      }

      void *ptr = platform::mem_map(handle_, size_);
      if (ptr == platform::MAP_FAILED_PTR) {
        return std::unexpected(StorageError::AllocationFailed);
      }
      data_.store(static_cast<uint8_t *>(ptr), std::memory_order_release);

      platform::advise_hugepage(ptr, size_);
      platform::advise_random(ptr, size_);

      if (size_ < sizeof(AtlasHeader)) {
        return std::unexpected(StorageError::InvalidFormat);
      }
      auto *header = get_header();
      if (header->magic != ATLAS_MAGIC) {
        return std::unexpected(StorageError::InvalidFormat);
      }
      // Accept both V1 and V2 files
      if (header->version != ATLAS_VERSION && header->version != 1) {
        return std::unexpected(StorageError::VersionMismatch);
      }

      // V1 backward compat: if dim/stride not set, derive from V1 layout
      if (header->version == 1 || header->dim == 0) {
        header->dim = EMBEDDING_DIM_DEFAULT;
        header->metadata_size = METADATA_SIZE_DEFAULT;
        header->node_byte_stride =
            compute_node_stride(EMBEDDING_DIM_DEFAULT, METADATA_SIZE_DEFAULT);
        header->version = ATLAS_VERSION; // Upgrade in place
      }

      stride_ = header->node_byte_stride;
    }

    return {};
  }

  /// Set the EpochManager for deferred reclamation in grow().
  void set_epoch_manager(aeon::EpochManager *mgr) { epoch_mgr_ = mgr; }

  void close() {
    if (epoch_mgr_) {
      epoch_mgr_->drain_readers();
    }
    auto *d = data_.load(std::memory_order_acquire);
    if (d) {
      platform::mem_unmap(d, size_);
      data_.store(nullptr, std::memory_order_release);
    }
    if (handle_ != platform::INVALID_FILE_HANDLE) {
      platform::file_close(handle_);
      handle_ = platform::INVALID_FILE_HANDLE;
    }
    size_ = 0;
  }

  /**
   * @brief Expands the file to hold new_capacity nodes.
   * Uses dynamic node_byte_stride for size calculations.
   */
  std::expected<void, StorageError> grow(size_t new_capacity) {
    auto *current_data = data_.load(std::memory_order_acquire);
    if (!current_data)
      return std::unexpected(StorageError::IOError);

    auto *current_header = reinterpret_cast<AtlasHeader *>(current_data);
    if (new_capacity <= current_header->capacity) {
      return {};
    }

    size_t new_size = sizeof(AtlasHeader) + (new_capacity * stride_);

    if (!platform::file_resize(handle_, new_size)) {
      return std::unexpected(StorageError::AllocationFailed);
    }

    void *new_ptr = platform::mem_map(handle_, new_size);
    if (new_ptr == platform::MAP_FAILED_PTR) {
      return std::unexpected(StorageError::AllocationFailed);
    }

    void *old_data = current_data;
    size_t old_size = size_;

    data_.store(static_cast<uint8_t *>(new_ptr), std::memory_order_release);
    size_ = new_size;

    if (epoch_mgr_ && old_data) {
      epoch_mgr_->retire(old_data, old_size);
    } else if (old_data) {
      platform::mem_unmap(old_data, old_size);
    }

    get_header()->capacity = new_capacity;

    return {};
  }

  /**
   * @brief Release resident pages back to the OS without unmapping.
   * Uses dynamic stride for offset calculations.
   */
  void release_pages(size_t start_node, size_t count) {
    auto *header = get_header();
    if (!header || start_node >= header->capacity)
      return;
    count = std::min(count, static_cast<size_t>(header->capacity) - start_node);

    size_t offset = sizeof(AtlasHeader) + (start_node * stride_);
    size_t length = count * stride_;
    auto *d = data_.load(std::memory_order_acquire);
    if (d) {
      platform::advise_dontneed(d + offset, length);
    }
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

  /**
   * @brief Returns a NodeHeader* at the given index using dynamic byte stride.
   *
   * Pointer arithmetic: base + sizeof(AtlasHeader) + index * node_byte_stride.
   * Every node starts on a 64-byte boundary (enforced by compute_node_stride).
   */
  NodeHeader *get_node(size_t index) {
    auto *header = get_header();
    if (!header || index >= header->capacity)
      return nullptr;
    size_t offset = sizeof(AtlasHeader) + (index * stride_);
    return reinterpret_cast<NodeHeader *>(
        data_.load(std::memory_order_acquire) + offset);
  }

  const NodeHeader *get_node(size_t index) const {
    auto *header = get_header();
    if (!header || index >= header->capacity)
      return nullptr;
    size_t offset = sizeof(AtlasHeader) + (index * stride_);
    return reinterpret_cast<const NodeHeader *>(
        data_.load(std::memory_order_acquire) + offset);
  }

  /// Returns the cached node byte stride.
  size_t stride() const noexcept { return stride_; }

private:
  platform::FileHandle handle_ = platform::INVALID_FILE_HANDLE;
  std::atomic<uint8_t *> data_{nullptr};
  size_t size_ = 0;
  size_t stride_ = 0; ///< Cached from AtlasHeader::node_byte_stride
  aeon::EpochManager *epoch_mgr_ = nullptr;
};

} // namespace aeon::storage
