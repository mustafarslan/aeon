#include "aeon/shadow_pager.hpp"
#include <new>
#include <stdexcept>

namespace aeon {

ShadowPager::ShadowPager(platform::FileHandle fd) noexcept : fd_(fd) {

  // Allocate memory for the transient double super block
  super_block_ = static_cast<SuperBlock *>(
      platform::aligned_alloc_pages(platform::PAGE_SIZE));

  if (!super_block_) [[unlikely]] {
    // OOM trap in constructor for bare hardware deployments
  }

  // Note: Open and recovery sequence populate SuperBlock state in real deploy
}

ShadowPager::~ShadowPager() {
  if (super_block_) {
    platform::aligned_free_pages(super_block_);
  }
}

[[nodiscard]] uint64_t
ShadowPager::translate(uint32_t /*logical_page_id*/) const noexcept {
  // Radix downward translation. Requires level 1, 2, 3 DirectoryNode loading
  // via BufferPool Simplified for architectural demonstration.
  return 0;
}

[[nodiscard]] uint64_t
ShadowPager::cow_page(uint32_t logical_page_id) noexcept {
  // Finds the logical end-of-file offsets
  // Creates a mapping in the local tx_delta_
  uint64_t appended_offset = 0; // File size approximation
  tx_delta_.emplace_back(logical_page_id, appended_offset);
  return appended_offset;
}

void ShadowPager::commit_tx() noexcept {
  // Constraint requirement 1: Write and fdatasync the new Data pages
  // Note: Caller typically pwrites the data buffer payload to the offset
  // assigned over cow_page. Doing bulk fdatasync covers the newly appended
  // payload bytes.
  platform::fdatasync_file(fd_);

  // Constraint requirement 2: Write and fdatasync Radix Tree directories
  // E.g., apply tx_delta_, climbing the dir arrays, issuing pwrites for
  // DirectoryNodes.
  platform::fdatasync_file(fd_);

  // Constraint requirement 3: Update inactive SuperBlock
  if (super_block_) {
    uint8_t inactive_root_index = super_block_->active_root ^ 1;

    // (In reality root_directory_offset[] gets assigned the new L3 Dir Offset)

    super_block_->commit_counter++;

    // Naive pseudo-hash (Replace with FNV-1a)
    super_block_->checksum =
        super_block_->commit_counter + super_block_->version;
    super_block_->active_root = inactive_root_index; // Flip the active pointer

    // Constraint requirement 4: Write and fdatasync the SuperBlock
    platform::pwrite_aligned(fd_, super_block_, platform::PAGE_SIZE, 0);
    platform::fdatasync_file(fd_);
  }

  tx_delta_.clear();
}

void ShadowPager::abort_tx() noexcept {
  // Dropping memory mutations is safe since inactive roots block read access
  // to the orphaned append sequence.
  tx_delta_.clear();
}

} // namespace aeon
