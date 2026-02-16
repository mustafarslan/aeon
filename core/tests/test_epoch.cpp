#include "aeon/epoch.hpp"
#include "aeon/platform.hpp"
#include <algorithm>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

using namespace aeon;

// TSan slows execution 10-50x. Scale all timing-sensitive sleeps accordingly.
#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define AEON_TSAN_ACTIVE 1
#endif
#endif
#if defined(__SANITIZE_THREAD__) && !defined(AEON_TSAN_ACTIVE)
#define AEON_TSAN_ACTIVE 1
#endif

#ifdef AEON_TSAN_ACTIVE
constexpr int TSAN_MULTIPLIER = 20;
#else
constexpr int TSAN_MULTIPLIER = 1;
#endif

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

  // Allocate via platform::mem_map — matches platform::mem_unmap in
  // try_reclaim() On POSIX, we need anonymous mmap for testing. Use
  // platform-aware allocation.
#if defined(AEON_PLATFORM_WINDOWS)
  // On Windows, allocate via VirtualAlloc for anonymous memory
  void *ptr =
      VirtualAlloc(nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
  ASSERT_NE(ptr, nullptr);
#else
  void *ptr = mmap(nullptr, 4096, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(ptr, MAP_FAILED);
#endif

  mgr.retire(ptr, 4096);
  EXPECT_EQ(mgr.retired_count(), 1);

  // No active readers → advance should reclaim
  mgr.advance_epoch();
  EXPECT_EQ(mgr.retired_count(), 0);
}

TEST(EpochManager, RetireBlockedByActiveReader) {
  EpochManager mgr;

#if defined(AEON_PLATFORM_WINDOWS)
  void *ptr =
      VirtualAlloc(nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
  ASSERT_NE(ptr, nullptr);
#else
  void *ptr = mmap(nullptr, 4096, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(ptr, MAP_FAILED);
#endif

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

#if defined(AEON_PLATFORM_WINDOWS)
  void *ptr1 =
      VirtualAlloc(nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
  void *ptr2 =
      VirtualAlloc(nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
  void *ptr1 = mmap(nullptr, 4096, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  void *ptr2 = mmap(nullptr, 4096, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#endif

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
    std::this_thread::sleep_for(
        std::chrono::milliseconds(50 * TSAN_MULTIPLIER));
    guard.release();
    guard_released.store(true, std::memory_order_release);
  });

  // Give reader time to acquire guard
  std::this_thread::sleep_for(std::chrono::milliseconds(10 * TSAN_MULTIPLIER));

  // drain_readers should block until reader releases
  mgr.drain_readers();
  EXPECT_TRUE(guard_released.load(std::memory_order_acquire));

  reader.join();
}
