#pragma once

/**
 * @file shadow_pager.hpp
 * @brief Append-Only Shadow Pager with 4-Level Radix Tree Directory.
 *
 * Prevents I/O amplification during commits by ensuring only the modified
 * 4KB pages and their ancestral directory paths are CoW'd.
 */

#include <cstddef>
#include <cstdint>
#include <vector>

#include "aeon/platform_io.hpp"

namespace aeon {

/// Magic bytes: "SHADOW_1" in hex.
inline constexpr uint64_t AEON_SHADOW_MAGIC = 0x534841444F575F31;

/**
 * @brief A single node in the Radix Tree page directory.
 * Fits exactly within a 4KB hardware page mapping 512 physical file offsets.
 */
struct alignas(platform::PAGE_SIZE) DirectoryNode {
  // 512 * 8 bytes = 4096 bytes.
  // Each entry is a physical file offset. 0 means unmapped.
  uint64_t child_offsets[512];
};
static_assert(sizeof(DirectoryNode) == platform::PAGE_SIZE,
              "DirectoryNode must exactly map to a hardware sector block");

/**
 * @brief The double-buffered meta-page located at physical offset 0.
 */
struct alignas(platform::PAGE_SIZE) SuperBlock {
  uint64_t magic;   // 0x00
  uint64_t version; // 0x08
  uint64_t
      root_directory_offset[2]; // 0x10: Pointers to active L3 DirectoryNodes
  uint8_t active_root;          // 0x20: 0 or 1
  uint64_t commit_counter;      // 0x21: Monotonic transaction ID
  uint64_t checksum;            // 0x29: FNV-1a hash

  // Padding to 4096 bytes automatically handled by alignas(platform::PAGE_SIZE)
};
static_assert(sizeof(SuperBlock) == platform::PAGE_SIZE,
              "SuperBlock must exactly map to a hardware sector block");

/**
 * @brief Manages the append-only data file and Radix Tree directory CoW.
 */
class ShadowPager {
public:
  explicit ShadowPager(platform::FileHandle fd) noexcept;
  ~ShadowPager();

  ShadowPager(const ShadowPager &) = delete;
  ShadowPager &operator=(const ShadowPager &) = delete;

  /**
   * @brief Traverse the active Radix Tree to find a physical file offset.
   * @return Physical byte offset, or 0 if logical page is unmapped.
   */
  [[nodiscard]] uint64_t translate(uint32_t logical_page_id) const noexcept;

  /**
   * @brief Copy-on-Write a logical page.
   * Allocates a new physical offset at the end of the file. Records the
   * impending directory modifications in RAM (transaction delta).
   * @return The new physical file offset to write the page data into.
   */
  [[nodiscard]] uint64_t cow_page(uint32_t logical_page_id) noexcept;

  /**
   * @brief Commit the current transaction.
   * 1. Copies and modifies affected ancestral DirectoryNodes.
   * 2. Appends new DirectoryNodes to the file.
   * 3. fdatasync() data + directories.
   * 4. Update the inactive SuperBlock and fdatasync().
   * 5. Flip active_root and fdatasync().
   */
  void commit_tx() noexcept;

  /**
   * @brief Abort the current transaction. Drops RAM deltas.
   * Appended file garbage is naturally ignored since SuperBlock was untouched.
   */
  void abort_tx() noexcept;

private:
  platform::FileHandle fd_;
  SuperBlock *super_block_; // mmap'd or buffer-aligned

  // In-memory delta representing the current uncommitted CoW paths.
  // logical_page_id -> new_physical_offset
  std::vector<std::pair<uint32_t, uint64_t>> tx_delta_;
};

} // namespace aeon
