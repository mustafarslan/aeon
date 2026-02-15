#include "aeon/epoch.hpp"
#include <algorithm>
#include <gtest/gtest.h>
#include <sys/mman.h>
#include <thread>
#include <vector>

using namespace aeon;

// ============================================================================
// EpochManager Unit Tests
// ============================================================================

TEST(EpochManager, InitialState) {
  EpochManager mgr;
  EXPECT_EQ(mgr.current_epoch(), 1);
  EXPECT_EQ(mgr.retired_count(), 0);
}

TEST(EpochManager, AdvanceEpoch) {
  EpochManager mgr;
  mgr.advance_epoch();
  EXPECT_EQ(mgr.current_epoch(), 2);
  mgr.advance_epoch();
  EXPECT_EQ(mgr.current_epoch(), 3);
}

TEST(EpochManager, SlotAcquisition) {
  EpochManager mgr;
  size_t slot = mgr.acquire_slot();
  EXPECT_LT(slot, MAX_READERS);
}

TEST(EpochManager, ThreadLocalSlotCaching) {
  EpochManager mgr;
  size_t first = mgr.acquire_slot();
  size_t second = mgr.acquire_slot();
  // Same thread should get same slot
  EXPECT_EQ(first, second);
}

TEST(EpochManager, MultiThreadSlotUniqueness) {
  EpochManager mgr;
  constexpr int NUM_THREADS = 16;
  std::vector<size_t> slots(NUM_THREADS);
  std::vector<std::thread> threads;

  for (int i = 0; i < NUM_THREADS; ++i) {
    threads.emplace_back(
        [&mgr, &slots, i]() { slots[i] = mgr.acquire_slot(); });
  }

  for (auto &t : threads)
    t.join();

  // All slots should be unique
  std::sort(slots.begin(), slots.end());
  for (int i = 1; i < NUM_THREADS; ++i) {
    EXPECT_NE(slots[i], slots[i - 1])
        << "Thread " << i << " and " << (i - 1) << " got same slot";
  }
}

// ============================================================================
// EpochGuard Unit Tests
// ============================================================================

TEST(EpochGuard, BasicLifecycle) {
  EpochManager mgr;
  {
    auto guard = mgr.enter_guard();
    EXPECT_TRUE(guard.is_active());
  }
  // Guard destroyed — slot should be released
}

TEST(EpochGuard, ExplicitRelease) {
  EpochManager mgr;
  auto guard = mgr.enter_guard();
  EXPECT_TRUE(guard.is_active());
  guard.release();
  EXPECT_FALSE(guard.is_active());
}

TEST(EpochGuard, MoveSemantics) {
  EpochManager mgr;
  auto guard1 = mgr.enter_guard();
  EXPECT_TRUE(guard1.is_active());

  // Move to guard2
  auto guard2 = std::move(guard1);
  EXPECT_FALSE(guard1.is_active());
  EXPECT_TRUE(guard2.is_active());
}

TEST(EpochGuard, MultipleGuardsOnDifferentThreads) {
  EpochManager mgr;
  constexpr int NUM_THREADS = 8;
  std::atomic<int> active_count{0};
  std::atomic<bool> start{false};
  std::vector<std::thread> threads;

  for (int i = 0; i < NUM_THREADS; ++i) {
    threads.emplace_back([&]() {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      auto guard = mgr.enter_guard();
      active_count.fetch_add(1, std::memory_order_relaxed);
      // Hold guard briefly
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      active_count.fetch_sub(1, std::memory_order_relaxed);
    });
  }

  start.store(true, std::memory_order_release);
  for (auto &t : threads)
    t.join();
  EXPECT_EQ(active_count.load(), 0);
}

// ============================================================================
// Deferred Reclamation Tests
// ============================================================================

TEST(EpochManager, RetireAndReclaim) {
  EpochManager mgr;

  // Allocate via mmap — matches munmap in try_reclaim()
  void *ptr = mmap(nullptr, 4096, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(ptr, MAP_FAILED);

  mgr.retire(ptr, 4096);
  EXPECT_EQ(mgr.retired_count(), 1);

  // No active readers → advance should reclaim
  mgr.advance_epoch();
  EXPECT_EQ(mgr.retired_count(), 0);
}

TEST(EpochManager, RetireBlockedByActiveReader) {
  EpochManager mgr;

  void *ptr = mmap(nullptr, 4096, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(ptr, MAP_FAILED);

  // Reader holds a guard
  auto guard = mgr.enter_guard();

  mgr.retire(ptr, 4096);
  mgr.advance_epoch();

  // Region should NOT be reclaimed while reader is active
  EXPECT_EQ(mgr.retired_count(), 1);

  // Release reader
  guard.release();

  // Now reclaim should succeed
  mgr.advance_epoch();
  EXPECT_EQ(mgr.retired_count(), 0);
}

TEST(EpochManager, MultipleRetirementsAcrossEpochs) {
  EpochManager mgr;

  void *ptr1 = mmap(nullptr, 4096, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  void *ptr2 = mmap(nullptr, 4096, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

  mgr.retire(ptr1, 4096);
  mgr.advance_epoch();
  mgr.retire(ptr2, 4096);

  // Both should be reclaimable (no active readers)
  mgr.advance_epoch();
  EXPECT_EQ(mgr.retired_count(), 0);
}

TEST(EpochManager, DrainReaders) {
  EpochManager mgr;
  std::atomic<bool> guard_released{false};

  std::thread reader([&]() {
    auto guard = mgr.enter_guard();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    guard.release();
    guard_released.store(true, std::memory_order_release);
  });

  // Give reader time to acquire guard
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  // drain_readers should block until reader releases
  mgr.drain_readers();
  EXPECT_TRUE(guard_released.load(std::memory_order_acquire));

  reader.join();
}
