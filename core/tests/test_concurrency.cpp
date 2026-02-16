#include "aeon/atlas.hpp"
#include "aeon/epoch.hpp"
#include "aeon/storage.hpp"
#include <filesystem>
#include <gtest/gtest.h>
#include <random>
#include <thread>
#include <vector>

using namespace aeon;

namespace {

/// Helper: generate random 768-dim float32 vector
std::vector<float> random_vector(std::mt19937 &rng) {
  std::vector<float> v(768);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  for (auto &f : v)
    f = dist(rng);
  // Normalize for cosine similarity
  float norm = 0.0f;
  for (auto f : v)
    norm += f * f;
  norm = std::sqrt(norm);
  if (norm > 0)
    for (auto &f : v)
      f /= norm;
  return v;
}

/// Helper: create a temporary Atlas file
class TempAtlas {
public:
  TempAtlas() {
    path_ = std::filesystem::temp_directory_path() /
            ("aeon_test_" + std::to_string(counter_++) + ".atlas");
    atlas_ = std::make_unique<Atlas>(path_);
  }

  ~TempAtlas() {
    atlas_.reset();
    std::filesystem::remove(path_);
  }

  Atlas &get() { return *atlas_; }
  const std::filesystem::path &path() const { return path_; }

private:
  static inline int counter_ = 0;
  std::filesystem::path path_;
  std::unique_ptr<Atlas> atlas_;
};

} // namespace

// ============================================================================
// Concurrent Navigate + Grow (8 readers + 1 writer)
// ============================================================================

TEST(Concurrency, ConcurrentNavigateAndInsert) {
  TempAtlas ta;
  auto &atlas = ta.get();

  // Seed some initial data
  std::mt19937 rng(42);
  auto root_vec = random_vector(rng);
  atlas.insert(0, std::span<const float>(root_vec), "root");

  for (int i = 0; i < 50; ++i) {
    auto v = random_vector(rng);
    atlas.insert(0, std::span<const float>(v), "child_" + std::to_string(i));
  }

  constexpr int NUM_READERS = 8;
  constexpr int ITERATIONS = 100;
  std::atomic<bool> stop{false};
  std::atomic<int> read_count{0};
  std::atomic<int> write_count{0};
  std::vector<std::thread> threads;

  // Writer thread: continuous inserts that may trigger grow()
  threads.emplace_back([&]() {
    std::mt19937 local_rng(123);
    for (int i = 0; i < ITERATIONS && !stop; ++i) {
      auto v = random_vector(local_rng);
      try {
        atlas.insert(0, std::span<const float>(v),
                     "concurrent_" + std::to_string(i));
        write_count.fetch_add(1, std::memory_order_relaxed);
      } catch (...) {
        // May fail on resource limits — acceptable
      }
      std::this_thread::yield();
    }
  });

  // Reader threads: continuous navigate()
  for (int r = 0; r < NUM_READERS; ++r) {
    threads.emplace_back([&, r]() {
      std::mt19937 local_rng(r + 1000);
      for (int i = 0; i < ITERATIONS && !stop; ++i) {
        auto q = random_vector(local_rng);
        try {
          auto results = atlas.navigate(std::span<const float>(q));
          // Results should be valid (not corrupted memory)
          for (const auto &rn : results) {
            EXPECT_GE(rn.similarity, -1.1f);
            EXPECT_LE(rn.similarity, 1.1f);
          }
          read_count.fetch_add(1, std::memory_order_relaxed);
        } catch (...) {
          // Acceptable during concurrent modification
        }
      }
    });
  }

  for (auto &t : threads)
    t.join();

  // Verify both readers and writers completed
  EXPECT_GT(read_count.load(), 0);
  EXPECT_GT(write_count.load(), 0);
}

// ============================================================================
// Concurrent Delta Buffer Operations
// ============================================================================

TEST(Concurrency, ConcurrentDeltaInsertAndNavigate) {
  TempAtlas ta;
  auto &atlas = ta.get();

  constexpr int NUM_THREADS = 4;
  constexpr int INSERTS_PER_THREAD = 50;
  std::atomic<int> total_inserts{0};
  std::vector<std::thread> threads;

  // Multiple threads inserting to delta buffer
  for (int t = 0; t < NUM_THREADS; ++t) {
    threads.emplace_back([&, t]() {
      std::mt19937 rng(t + 2000);
      for (int i = 0; i < INSERTS_PER_THREAD; ++i) {
        auto v = random_vector(rng);
        atlas.insert_delta(std::span<const float>(v),
                           "delta_" + std::to_string(t) + "_" +
                               std::to_string(i));
        total_inserts.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  // Concurrent reader
  threads.emplace_back([&]() {
    std::mt19937 rng(9999);
    for (int i = 0; i < 100; ++i) {
      auto q = random_vector(rng);
      auto results = atlas.navigate(std::span<const float>(q));
      // No crash = success for concurrent access
      std::this_thread::yield();
    }
  });

  for (auto &t : threads)
    t.join();
  EXPECT_EQ(total_inserts.load(), NUM_THREADS * INSERTS_PER_THREAD);
}

// ============================================================================
// SLB Thread Safety
// ============================================================================

TEST(Concurrency, SLBConcurrentInsertAndLookup) {
  SemanticCache slb;

  constexpr int NUM_THREADS = 8;
  constexpr int OPS_PER_THREAD = 200;
  std::atomic<int> lookups{0};
  std::vector<std::thread> threads;

  // Mixed insert/lookup workload
  for (int t = 0; t < NUM_THREADS; ++t) {
    threads.emplace_back([&, t]() {
      std::mt19937 rng(t + 3000);
      for (int i = 0; i < OPS_PER_THREAD; ++i) {
        auto v = random_vector(rng);
        if (i % 3 == 0) {
          // Insert
          slb.insert(i + t * OPS_PER_THREAD, std::span<const float>(v));
        } else {
          // Lookup
          auto hit = slb.find_nearest(std::span<const float>(v), 0.1f);
          if (hit) {
            // Verify SLBHit has valid data (no dangling pointer crash)
            EXPECT_GE(hit->similarity, -1.1f);
            EXPECT_LE(hit->similarity, 1.1f);
            EXPECT_NE(hit->node_id, UINT64_MAX);
          }
          lookups.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }

  for (auto &t : threads)
    t.join();
  EXPECT_GT(lookups.load(), 0);
}

// ============================================================================
// EBR Stress: Many guards, many retirements
// ============================================================================

TEST(Concurrency, EBRStressTest) {
  EpochManager mgr;
  constexpr int NUM_THREADS = 16;
  constexpr int ITERATIONS = 500;
  std::atomic<int> total_guards{0};
  std::vector<std::thread> threads;

  for (int t = 0; t < NUM_THREADS; ++t) {
    threads.emplace_back([&]() {
      for (int i = 0; i < ITERATIONS; ++i) {
        auto guard = mgr.enter_guard();
        // Simulate brief read
        std::this_thread::yield();
        guard.release();
        total_guards.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  // Writer thread: retire and advance
  threads.emplace_back([&]() {
    for (int i = 0; i < ITERATIONS; ++i) {
      void *ptr = mmap(nullptr, 64, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (ptr != MAP_FAILED) {
        mgr.retire(ptr, 64);
        mgr.advance_epoch();
      }
      std::this_thread::yield();
    }
  });

  for (auto &t : threads)
    t.join();
  EXPECT_EQ(total_guards.load(), NUM_THREADS * ITERATIONS);

  // Final cleanup
  mgr.advance_epoch();
}
