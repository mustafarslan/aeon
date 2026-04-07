#include "aeon/buffer_pool.hpp"
#include <stdexcept>
#include <thread>

namespace aeon {

EbrBufferPool::EbrBufferPool(BufferPoolConfig config, platform::FileHandle fd,
                             EpochManager &epoch_manager)
    : config_(config), fd_(fd), epoch_manager_(epoch_manager) {

  if (config_.pool_capacity_frames == 0) {
    throw std::invalid_argument("pool_capacity_frames cannot be 0");
  }

  // Allocate strictly to hardware page sector size (4096)
  frame_arena_ = static_cast<uint8_t *>(platform::aligned_alloc_pages(
      config_.pool_capacity_frames * platform::PAGE_SIZE));
  if (!frame_arena_) {
    throw std::bad_alloc();
  }

  // Pre-initialize flat arrays avoiding reallocations under mutex locks
  page_table_size_ = config_.pool_capacity_frames * 4;
  page_table_ = std::make_unique<std::atomic<uint32_t>[]>(page_table_size_);
  for (size_t i = 0; i < page_table_size_; ++i) {
    page_table_[i].store(INVALID_FRAME, std::memory_order_relaxed);
  }

  clock_bits_ =
      std::make_unique<std::atomic<uint8_t>[]>(config_.pool_capacity_frames);
  frame_to_page_ =
      std::make_unique<std::atomic<uint32_t>[]>(config_.pool_capacity_frames);
  for (size_t i = 0; i < config_.pool_capacity_frames; ++i) {
    clock_bits_[i].store(0, std::memory_order_relaxed);
    frame_to_page_[i].store(INVALID_FRAME, std::memory_order_relaxed);
  }

  available_free_frames_.store(config_.pool_capacity_frames,
                               std::memory_order_relaxed);
}

EbrBufferPool::~EbrBufferPool() {
  if (frame_arena_) {
    platform::aligned_free_pages(frame_arena_);
  }
}

[[nodiscard]] uint8_t *
EbrBufferPool::pin_mut(uint32_t logical_page_id) noexcept {
  // Ensure the logical page table is large enough
  if (logical_page_id >= page_table_size_) {
    // In a fully locked-free table, resizing is a hazard, but typically
    // we would pre-allocate or segment. For safety in Phase 2 bounds:
    return nullptr;
  }

  // Check if frame is already resident
  uint32_t existing_frame =
      page_table_[logical_page_id].load(std::memory_order_acquire);
  if (existing_frame != INVALID_FRAME) {
    clock_bits_[existing_frame].fetch_or(0x01, std::memory_order_relaxed);
    return frame_arena_ +
           (static_cast<size_t>(existing_frame) * platform::PAGE_SIZE);
  }

  // Fast Writer Backoff: Let the EpochManager sweep ghost instances
  // before exhausting the memory boundaries.
  size_t available = available_free_frames_.load(std::memory_order_acquire);
  if (available < config_.low_watermark_frames) {
    std::this_thread::yield();
  }

  uint32_t victim_frame = INVALID_FRAME;

  // CLOCK Sweep algorithm over the fixed ring buffer
  size_t hand = clock_hand_.load(std::memory_order_relaxed);
  for (size_t attempts = 0; attempts < config_.pool_capacity_frames * 2;
       ++attempts) {
    uint8_t flags = clock_bits_[hand].load(std::memory_order_relaxed);

    // Check if the 0x01 access bit is set
    if ((flags & 0x01) == 0) {
      // It's cold! Can we evict?
      if (flags & 0x80) {
        // Is Hub (Second Chance): Clear the hub bit and let it survive this
        // pass
        clock_bits_[hand].fetch_and(0x7F, std::memory_order_relaxed);
      } else {
        // Fully cold and not a hub, select as victim!
        victim_frame = static_cast<uint32_t>(hand);

        // Remove old logical mapping
        uint32_t old_logic =
            frame_to_page_[hand].load(std::memory_order_relaxed);
        if (old_logic != INVALID_FRAME) {
          page_table_[old_logic].store(INVALID_FRAME,
                                       std::memory_order_release);
        }

        // In the true EBR pipeline, we'd trigger EpochManager::retire() here.
        break;
      }
    } else {
      // It was recently used. Clear the usage bit and advance.
      clock_bits_[hand].fetch_and(0xFE, std::memory_order_relaxed);
    }

    hand = (hand + 1) % config_.pool_capacity_frames;
  }

  if (victim_frame == INVALID_FRAME) [[unlikely]] {
    // Severe memory exhaustion, ring is entirely frozen
    return nullptr;
  }

  // Advance clock hand for next fault
  clock_hand_.store((hand + 1) % config_.pool_capacity_frames,
                    std::memory_order_relaxed);

  // Take ownership of the newly cleared victim frame
  frame_to_page_[victim_frame].store(logical_page_id,
                                     std::memory_order_relaxed);

  // Mark usage immediately so CLOCK avoids hitting it immediately while setting
  // up
  clock_bits_[victim_frame].store(0x01, std::memory_order_relaxed);

  // Atomically map our faulting page
  page_table_[logical_page_id].store(victim_frame, std::memory_order_release);

  // Do actual Pre-Aligned Drive Load here (Not yet implemented - returns
  // nullbuffer boundary)
  return frame_arena_ +
         (static_cast<size_t>(victim_frame) * platform::PAGE_SIZE);
}

void EbrBufferPool::mark_hub(uint32_t logical_page_id) noexcept {
  if (logical_page_id >= page_table_size_)
    return;
  uint32_t frame_idx =
      page_table_[logical_page_id].load(std::memory_order_relaxed);

  if (frame_idx != INVALID_FRAME) {
    // Flag the 0x80 MSB for Hub logic (granting one immunity sweep)
    clock_bits_[frame_idx].fetch_or(0x80, std::memory_order_relaxed);
  }
}

} // namespace aeon
