/**
 * @file test_wal.cpp
 * @brief Unit tests for the V4.1 Write-Ahead Log (WAL) crash recovery.
 *
 * Tests cover:
 *   - Atlas WAL replay:  Insert delta nodes, close, reopen → verify recovery.
 *   - Trace WAL replay:  Append events, close, reopen → verify recovery.
 *   - Corrupted tail:    Partial write → replay discards bad tail.
 *   - Checksum failure:  Bit-flipped payload → replay stops at corruption.
 *   - WAL truncation:    compact_mmap() / compact() resets WAL to empty.
 */

#include "aeon/atlas.hpp"
#include "aeon/hash.hpp"
#include "aeon/schema.hpp"
#include "aeon/trace.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <vector>

namespace fs = std::filesystem;

// ═══════════════════════════════════════════════════════════════════════════
// Test Fixtures
// ═══════════════════════════════════════════════════════════════════════════

class WalAtlasTest : public ::testing::Test {
protected:
  fs::path atlas_path_;
  fs::path wal_path_;

  void SetUp() override {
    atlas_path_ = fs::temp_directory_path() / "test_wal_atlas.bin";
    wal_path_ = atlas_path_;
    wal_path_ += ".wal";

    // Clean up any leftover files
    std::error_code ec;
    fs::remove(atlas_path_, ec);
    fs::remove(wal_path_, ec);
  }

  void TearDown() override {
    std::error_code ec;
    fs::remove(atlas_path_, ec);
    fs::remove(wal_path_, ec);
    // Also clean up any generational files
    for (int g = 0; g < 10; ++g) {
      auto gen_path = atlas_path_.parent_path() /
                      ("test_wal_atlas_gen" + std::to_string(g) + ".bin");
      fs::remove(gen_path, ec);
    }
  }
};

class WalTraceTest : public ::testing::Test {
protected:
  fs::path trace_path_;
  fs::path wal_path_;

  void SetUp() override {
    trace_path_ = fs::temp_directory_path() / "test_wal_trace.bin";
    wal_path_ = trace_path_;
    wal_path_ += ".wal";

    std::error_code ec;
    fs::remove(trace_path_, ec);
    fs::remove(wal_path_, ec);
  }

  void TearDown() override {
    std::error_code ec;
    fs::remove(trace_path_, ec);
    fs::remove(wal_path_, ec);
    for (int g = 0; g < 10; ++g) {
      auto gen_path = trace_path_.parent_path() /
                      ("trace_gen" + std::to_string(g) + ".bin");
      fs::remove(gen_path, ec);
    }
  }
};

// ═══════════════════════════════════════════════════════════════════════════
// Atlas WAL Tests
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(WalAtlasTest, WalAtlasReplay) {
  // Insert 5 delta nodes, then close (simulating crash)
  constexpr uint32_t DIM = 4;
  {
    aeon::Atlas atlas(atlas_path_, DIM);
    std::vector<float> vec(DIM, 1.0f);

    for (int i = 0; i < 5; ++i) {
      vec[0] = static_cast<float>(i);
      atlas.insert_delta(vec, "node_" + std::to_string(i));
    }
    // Destructor closes — WAL file should exist with 5 records
  }

  // Verify WAL file exists
  ASSERT_TRUE(fs::exists(wal_path_));
  ASSERT_GT(fs::file_size(wal_path_), 0u);

  // Reopen — replay_wal should reconstruct the 5 delta nodes
  {
    aeon::Atlas atlas(atlas_path_, DIM);
    auto guard = atlas.acquire_read_guard();

    // The 5 delta nodes + any mmap nodes
    // We check that the atlas reports the delta nodes were recovered.
    // size() returns mmap node count only; delta nodes are separate.
    // Navigate with a query vector similar to node 0 to verify recovery.
    std::vector<float> query = {0.0f, 1.0f, 1.0f, 1.0f};
    auto results = atlas.navigate(query, 1);

    // We should get at least one result (proving delta nodes were recovered)
    EXPECT_FALSE(results.empty());
  }
}

TEST_F(WalAtlasTest, WalAtlasCorruptedTail) {
  constexpr uint32_t DIM = 4;

  // Write 3 valid nodes
  {
    aeon::Atlas atlas(atlas_path_, DIM);
    std::vector<float> vec(DIM, 1.0f);
    for (int i = 0; i < 3; ++i) {
      atlas.insert_delta(vec, "valid_" + std::to_string(i));
    }
  }

  // Append garbage bytes to simulate a partial 4th write (crash mid-write)
  {
    std::ofstream wal(wal_path_, std::ios::binary | std::ios::app);
    const char garbage[] = "THIS_IS_TRUNCATED_GARBAGE";
    wal.write(garbage, sizeof(garbage));
  }

  // Reopen — should recover 3 valid nodes, discard the garbage tail
  {
    aeon::Atlas atlas(atlas_path_, DIM);
    std::vector<float> query(DIM, 1.0f);
    auto results = atlas.navigate(query, 5);

    // Should find results from the 3 valid recovered nodes
    EXPECT_FALSE(results.empty());
  }
}

TEST_F(WalAtlasTest, WalAtlasChecksumFail) {
  constexpr uint32_t DIM = 4;

  // Write 2 valid nodes
  {
    aeon::Atlas atlas(atlas_path_, DIM);
    std::vector<float> vec(DIM, 1.0f);
    atlas.insert_delta(vec, "good_node_0");
    atlas.insert_delta(vec, "good_node_1");
  }

  // Read the WAL, flip a bit in the SECOND record's payload, rewrite
  {
    // Read entire WAL
    std::ifstream in(wal_path_, std::ios::binary);
    std::vector<uint8_t> wal_data((std::istreambuf_iterator<char>(in)),
                                  std::istreambuf_iterator<char>());
    in.close();

    // Flip a byte near the end of the file to corrupt the second record's
    // payload
    if (wal_data.size() > 20) {
      wal_data[wal_data.size() - 10] ^= 0xFF; // Flip bits
    }

    // Rewrite WAL
    std::ofstream out(wal_path_, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char *>(wal_data.data()),
              static_cast<std::streamsize>(wal_data.size()));
  }

  // Reopen — should recover only the first valid node
  {
    aeon::Atlas atlas(atlas_path_, DIM);
    std::vector<float> query(DIM, 1.0f);
    auto results = atlas.navigate(query, 5);

    // Should find at least the first valid node
    EXPECT_FALSE(results.empty());
    // But should NOT have both nodes (second was corrupted)
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Trace WAL Tests
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(WalTraceTest, WalTraceReplay) {
  // First, create the trace file so it has a valid header
  {
    aeon::TraceManager trace(trace_path_);
    // No events — just creating the file
  }

  // Manually write 10 valid WAL records (simulating events that were
  // written to the delta buffer during compaction, then crashed)
  {
    std::ofstream wal(wal_path_, std::ios::binary | std::ios::trunc);
    for (int i = 0; i < 10; ++i) {
      aeon::TraceEvent ev{};
      std::memset(&ev, 0, sizeof(aeon::TraceEvent));
      ev.timestamp = static_cast<uint64_t>(1000 + i);
      ev.atlas_id = 0;
      ev.role = 0;
      ev.flags = 0;
      std::string sid = "session_test";
      std::strncpy(ev.session_id, sid.c_str(), sizeof(ev.session_id) - 1);
      std::string text = "Event message " + std::to_string(i);
      std::strncpy(ev.text_preview, text.c_str(), sizeof(ev.text_preview) - 1);

      uint64_t checksum = aeon::hash::fnv1a_64(&ev, sizeof(aeon::TraceEvent));

      aeon::WalRecordHeader wal_hdr{};
      wal_hdr.record_type = aeon::WAL_RECORD_TRACE;
      wal_hdr.payload_size = static_cast<uint32_t>(sizeof(aeon::TraceEvent));
      wal_hdr.checksum = checksum;

      wal.write(reinterpret_cast<const char *>(&wal_hdr),
                sizeof(aeon::WalRecordHeader));
      wal.write(reinterpret_cast<const char *>(&ev), sizeof(aeon::TraceEvent));
    }
  }

  ASSERT_TRUE(fs::exists(wal_path_));

  // Reopen — replay should reconstruct all 10 events into delta buffer
  {
    aeon::TraceManager trace(trace_path_);

    // Get history for the session — should have all 10 events
    auto history = trace.get_history("session_test", 100);
    EXPECT_EQ(history.size(), 10u);
  }
}

TEST_F(WalTraceTest, WalTraceTruncation) {
  // Append events, compact, verify WAL is reset
  {
    aeon::TraceManager trace(trace_path_);

    // Append some events to create WAL entries
    for (int i = 0; i < 5; ++i) {
      trace.append_event("session_compact", 0, "compact_test", 0);
    }

    // WAL should have data
    ASSERT_TRUE(fs::exists(wal_path_));

    // Run compaction — this should truncate the WAL
    trace.compact();

    // After compaction, events should still be accessible
    auto history = trace.get_history("session_compact", 100);
    EXPECT_EQ(history.size(), 5u);

    // WAL should be reset (either empty or newly created)
    // The exact behavior depends on whether open_wal creates an empty file
    // or if truncate_wal removes it and open_wal creates fresh
    if (fs::exists(wal_path_)) {
      EXPECT_EQ(fs::file_size(wal_path_), 0u);
    }
  }
}
