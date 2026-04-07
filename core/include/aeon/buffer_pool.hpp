#pragma once

/**
 * @file buffer_pool.hpp
 * @brief Lock-Free EBR-Protected Page Table.
 *
 * Implements a strictly zero-overhead read path. Readers look up frames via
 * relaxed atomic loads and perform a single relaxed fetch_or to vote for the
 * CLOCK eviction policy. Features deterministic writer backoff.
 */

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "aeon/epoch.hpp"
#include "aeon/platform_io.hpp"

namespace aeon {

struct BufferPoolConfig {
  size_t pool_capacity_frames; // Maximum total 4KB frames locked in RAM
  size_t low_watermark_frames; // Frame threshold triggering writer eviction
                               // backoff
};

/// Sentinel for an unmapped or strictly on-disk logical page.
inline constexpr uint32_t INVALID_FRAME = 0xFFFFFFFF;

/**
 * @brief Lock-free EbrBufferPool for sub-microsecond vector chunk resolution.
 */
class EbrBufferPool {
public:
  explicit EbrBufferPool(BufferPoolConfig config, platform::FileHandle fd,
                         EpochManager &epoch_manager);
  ~EbrBufferPool();

  EbrBufferPool(const EbrBufferPool &) = delete;
  EbrBufferPool &operator=(const EbrBufferPool &) = delete;

  /**
   * @brief The ultra-low-latency read path.
   * Executed by 64+ threads concurrently. STRICTLY ZERO LOCKS.
   *
   * @param logical_page_id 32-bit logical page index.
   * @return Raw pointer to the 4KB aligned frame, or nullptr if page faulted.
   */
  [[nodiscard]] const uint8_t *
  read_page_fast(uint32_t logical_page_id) noexcept {
    if (logical_page_id >= page_table_size_) [[unlikely]] {
      return nullptr;
    }

    uint32_t frame_idx =
        page_table_[logical_page_id].load(std::memory_order_consume);
    if (frame_idx == INVALID_FRAME) [[unlikely]] {
      return nullptr;
    }

    // Check-Before-Set to prevent MESI cache invalidation storms
    // Only issue fetch_or if the accessed bit (0x01) is un-set.
    uint8_t current_clock =
        clock_bits_[frame_idx].load(std::memory_order_relaxed);
    if ((current_clock & 0x01) == 0) [[unlikely]] {
      clock_bits_[frame_idx].fetch_or(0x01, std::memory_order_relaxed);
    }

    return frame_arena_ +
           (static_cast<size_t>(frame_idx) * platform::PAGE_SIZE);
  }

  /**
   * @brief Write path logic.
   * Faults the page into RAM if necessary. Employs deterministic backoff
   * (thread::yield or sleep) if available frames dip below
   * low_watermark_frames.
   *
   * @param logical_page_id The page to mutate.
   * @return Writable pointer to the 4KB aligned frame.
   */
  [[nodiscard]] uint8_t *pin_mut(uint32_t logical_page_id) noexcept;

  /**
   * @brief Semantic pinning. Flags a frame as a high-priority Hub.
   * Ensures the CLOCK algorithm grants it extra survival sweeps.
   */
  void mark_hub(uint32_t logical_page_id) noexcept;

private:
  BufferPoolConfig config_;
  platform::FileHandle fd_;
  EpochManager &epoch_manager_;

  /// Hardware-aligned contiguous block (capacity * 4KB).
  uint8_t *frame_arena_;

  /// Lock-Free Flat Mapping: logical_page_id -> physical_RAM_frame_index
  std::unique_ptr<std::atomic<uint32_t>[]> page_table_;
  size_t page_table_size_;

  /// Eviction state: 8-bit tracking per frame.
  /// bit 0: accessed, bit 7: is_hub.
  std::unique_ptr<std::atomic<uint8_t>[]> clock_bits_;

  /// Logical Page ID currently mapped to this frame.
  std::unique_ptr<std::atomic<uint32_t>[]> frame_to_page_;

  /// Number of frames currently not in use. Read primarily by writer backoff.
  std::atomic<size_t> available_free_frames_;

  /// Sweeper hand for the CLOCK algorithm.
  std::atomic<size_t> clock_hand_{0};
};

} // namespace aeon
