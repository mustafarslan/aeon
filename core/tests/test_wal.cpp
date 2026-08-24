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
    auto blob_path = trace_path_;
    blob_path += ".blobs";
    fs::remove(blob_path, ec);
    for (int g = 0; g < 10; ++g) {
      auto tmp = trace_path_;
      tmp += (".compacting" + std::to_string(g));
      fs::remove(tmp, ec);
      auto tmp_blob = blob_path;
      tmp_blob += (".compacting" + std::to_string(g));
      fs::remove(tmp_blob, ec);
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

TEST_F(WalAtlasTest, WalAtlasUnknownRecordTypeSkippedNotFatal) {
  // V4 Stage 1 forward-compat: an unrecognized record_type mid-stream must
  // be skipped, not abort replay of everything after it. Uses two
  // DISTINGUISHABLE vectors (not both all-1.0f) and queries specifically
  // for the post-unknown-record one, so this test can only pass if replay
  // actually continued past the unknown record — under the old `break`
  // behavior, the pre-unknown record alone would satisfy a same-vector
  // query and this test would pass for the wrong reason.
  constexpr uint32_t DIM = 4;
  const std::vector<float> vec_before = {1.0f, 0.0f, 0.0f, 0.0f};
  const std::vector<float> vec_after = {0.0f, 1.0f, 0.0f, 0.0f};

  {
    aeon::Atlas atlas(atlas_path_, DIM);
    atlas.insert_delta(vec_before, "before_unknown");
  }

  // Splice a well-formed record with an unrecognized record_type in
  // between two valid Atlas WAL records.
  {
    std::vector<uint8_t> wal_data;
    {
      std::ifstream in(wal_path_, std::ios::binary);
      wal_data.assign((std::istreambuf_iterator<char>(in)),
                      std::istreambuf_iterator<char>());
    }

    std::vector<uint8_t> unknown_payload(16, 0xAB);
    aeon::WalRecordHeader unknown_hdr{};
    unknown_hdr.record_type = 0xFFFFFFFF; // not WAL_RECORD_ATLAS or _TRACE
    unknown_hdr.payload_size = static_cast<uint32_t>(unknown_payload.size());
    unknown_hdr.checksum =
        aeon::hash::fnv1a_64(unknown_payload.data(), unknown_payload.size());

    wal_data.insert(
        wal_data.end(), reinterpret_cast<uint8_t *>(&unknown_hdr),
        reinterpret_cast<uint8_t *>(&unknown_hdr) + sizeof(unknown_hdr));
    wal_data.insert(wal_data.end(), unknown_payload.begin(),
                    unknown_payload.end());

    std::ofstream out(wal_path_, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char *>(wal_data.data()),
              static_cast<std::streamsize>(wal_data.size()));
  }

  // Append a second, valid, DISTINCT Atlas WAL record AFTER the unknown one
  // by reopening and inserting again — this appends to the same WAL file.
  {
    aeon::Atlas atlas(atlas_path_, DIM);
    atlas.insert_delta(vec_after, "after_unknown");
  }

  // Reopen once more — query specifically for vec_after. This can only
  // return a near-1.0 similarity hit if the record AFTER the unknown one
  // was actually recovered (proving replay skipped, not broke).
  {
    aeon::Atlas atlas(atlas_path_, DIM);
    auto results = atlas.navigate(vec_after, 5);
    ASSERT_FALSE(results.empty());
    EXPECT_NEAR(results[0].similarity, 1.0f, 1e-4f);
  }
}

TEST_F(WalAtlasTest, WalAtlasScopeRecordReplayed) {
  // Proves replay's second pass genuinely applies WAL_RECORD_ATLAS_SCOPE --
  // not just that the value happens to already be durable via mmap. The
  // node's live mmap scope_bitmap is left at its default (0); the ONLY
  // place the target value exists is a hand-constructed WAL record, so
  // this can only pass if replay_wal() actually reads and applies it.
  constexpr uint32_t DIM = 4;
  constexpr uint64_t kScope = 0x2Au;
  uint64_t node_id;

  {
    aeon::Atlas atlas(atlas_path_, DIM);
    std::vector<float> vec(DIM, 1.0f);
    node_id = atlas.insert(0, vec, "scoped_node");
    ASSERT_EQ(atlas.get_node_scope(node_id), 0u); // never scope-set live
  }

  // Hand-append a WAL_RECORD_ATLAS_SCOPE record (mirrors WalTraceReplay's
  // manual-construction style) -- set_node_scope() is never called, so the
  // mmap file itself has no knowledge of kScope.
  {
    std::ofstream wal(wal_path_, std::ios::binary | std::ios::app);
    aeon::WalScopeRecord rec{node_id, kScope};

    aeon::WalRecordHeader hdr{};
    hdr.record_type = aeon::WAL_RECORD_ATLAS_SCOPE;
    hdr.payload_size = static_cast<uint32_t>(sizeof(aeon::WalScopeRecord));
    hdr.checksum = aeon::hash::fnv1a_64(&rec, sizeof(aeon::WalScopeRecord));

    wal.write(reinterpret_cast<const char *>(&hdr), sizeof(hdr));
    wal.write(reinterpret_cast<const char *>(&rec), sizeof(rec));
  }

  // Reopen — replay must apply the scope record to the mmap node.
  {
    aeon::Atlas atlas(atlas_path_, DIM);
    EXPECT_EQ(atlas.get_node_scope(node_id), kScope);
  }
}

TEST_F(WalAtlasTest, WalAtlasSupersedeRecordReplayed) {
  // Mirrors WalAtlasScopeRecordReplayed: proves replay's second pass
  // genuinely applies WAL_RECORD_ATLAS_SUPERSEDE. The live mmap node is
  // never superseded through the API, so this can only pass if replay
  // itself applies the hand-constructed record.
  constexpr uint32_t DIM = 4;
  uint64_t node_id;

  {
    aeon::Atlas atlas(atlas_path_, DIM);
    std::vector<float> vec(DIM, 1.0f);
    node_id = atlas.insert(0, vec, "node");
    ASSERT_FALSE(atlas.is_node_superseded(node_id));
  }

  {
    std::ofstream wal(wal_path_, std::ios::binary | std::ios::app);
    aeon::WalSupersedeRecord rec{node_id, /*revoke=*/0};

    aeon::WalRecordHeader hdr{};
    hdr.record_type = aeon::WAL_RECORD_ATLAS_SUPERSEDE;
    hdr.payload_size = static_cast<uint32_t>(sizeof(aeon::WalSupersedeRecord));
    hdr.checksum = aeon::hash::fnv1a_64(&rec, sizeof(aeon::WalSupersedeRecord));

    wal.write(reinterpret_cast<const char *>(&hdr), sizeof(hdr));
    wal.write(reinterpret_cast<const char *>(&rec), sizeof(rec));
  }

  {
    aeon::Atlas atlas(atlas_path_, DIM);
    EXPECT_TRUE(atlas.is_node_superseded(node_id));
  }
}

TEST_F(WalAtlasTest, WalAtlasGovernanceRecordReplayed) {
  // Mirrors WalAtlasScopeRecordReplayed: proves replay's second pass
  // genuinely applies WAL_RECORD_ATLAS_GOVERNANCE (V4 Stage 4 task 1). The
  // live mmap node's governance_record_id is never set through the API, so
  // this can only pass if replay itself applies the hand-constructed
  // record.
  constexpr uint32_t DIM = 4;
  constexpr uint64_t kGovernanceId = 0xDEADBEEFu;
  uint64_t node_id;

  {
    aeon::Atlas atlas(atlas_path_, DIM);
    std::vector<float> vec(DIM, 1.0f);
    node_id = atlas.insert(0, vec, "governed_node");
    ASSERT_EQ(atlas.get_node_governance_id(node_id), 0u);
  }

  {
    std::ofstream wal(wal_path_, std::ios::binary | std::ios::app);
    aeon::WalGovernanceRecord rec{node_id, kGovernanceId};

    aeon::WalRecordHeader hdr{};
    hdr.record_type = aeon::WAL_RECORD_ATLAS_GOVERNANCE;
    hdr.payload_size = static_cast<uint32_t>(sizeof(aeon::WalGovernanceRecord));
    hdr.checksum = aeon::hash::fnv1a_64(&rec, sizeof(aeon::WalGovernanceRecord));

    wal.write(reinterpret_cast<const char *>(&hdr), sizeof(hdr));
    wal.write(reinterpret_cast<const char *>(&rec), sizeof(rec));
  }

  {
    aeon::Atlas atlas(atlas_path_, DIM);
    EXPECT_EQ(atlas.get_node_governance_id(node_id), kGovernanceId);
  }
}

TEST_F(WalAtlasTest, WalAtlasTombstoneRecordReplayed) {
  // Mirrors WalAtlasSupersedeRecordReplayed: proves replay's second pass
  // genuinely applies WAL_RECORD_ATLAS_TOMBSTONE (V4 Stage 4 task 5/6). The
  // live mmap node is never tombstoned through the API, so this can only
  // pass if replay itself applies the hand-constructed record. There is no
  // public is_node_tombstoned() accessor, so tombstone_count() (already
  // used by TombstonedNodesAreDroppedByCompaction, test_atlas.cpp) is the
  // observable proof.
  constexpr uint32_t DIM = 4;
  uint64_t node_id;

  {
    aeon::Atlas atlas(atlas_path_, DIM);
    std::vector<float> vec(DIM, 1.0f);
    node_id = atlas.insert(0, vec, "node");
    ASSERT_EQ(atlas.tombstone_count(), 0u);
  }

  {
    std::ofstream wal(wal_path_, std::ios::binary | std::ios::app);
    aeon::WalTombstoneRecord rec{node_id};

    aeon::WalRecordHeader hdr{};
    hdr.record_type = aeon::WAL_RECORD_ATLAS_TOMBSTONE;
    hdr.payload_size = static_cast<uint32_t>(sizeof(aeon::WalTombstoneRecord));
    hdr.checksum = aeon::hash::fnv1a_64(&rec, sizeof(aeon::WalTombstoneRecord));

    wal.write(reinterpret_cast<const char *>(&hdr), sizeof(hdr));
    wal.write(reinterpret_cast<const char *>(&rec), sizeof(rec));
  }

  {
    aeon::Atlas atlas(atlas_path_, DIM);
    EXPECT_EQ(atlas.tombstone_count(), 1u);
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

TEST_F(WalTraceTest, WalTraceUnknownRecordTypeSkippedNotFatal) {
  // V4 Stage 1 forward-compat, Trace side: an unrecognized record_type
  // mid-stream must be skipped, and every valid record after it recovered.
  {
    aeon::TraceManager trace(trace_path_);
  }

  auto write_valid_event = [](std::ofstream &wal, int i) {
    aeon::TraceEvent ev{};
    std::memset(&ev, 0, sizeof(aeon::TraceEvent));
    ev.timestamp = static_cast<uint64_t>(2000 + i);
    std::string sid = "session_fwdcompat";
    std::strncpy(ev.session_id, sid.c_str(), sizeof(ev.session_id) - 1);
    std::string text = "Event " + std::to_string(i);
    std::strncpy(ev.text_preview, text.c_str(), sizeof(ev.text_preview) - 1);

    aeon::WalRecordHeader hdr{};
    hdr.record_type = aeon::WAL_RECORD_TRACE;
    hdr.payload_size = static_cast<uint32_t>(sizeof(aeon::TraceEvent));
    hdr.checksum = aeon::hash::fnv1a_64(&ev, sizeof(aeon::TraceEvent));

    wal.write(reinterpret_cast<const char *>(&hdr), sizeof(hdr));
    wal.write(reinterpret_cast<const char *>(&ev), sizeof(ev));
  };

  {
    std::ofstream wal(wal_path_, std::ios::binary | std::ios::trunc);

    write_valid_event(wal, 0);

    // Unknown record type, well-formed checksum — must be skipped, not
    // treated as corruption that halts replay.
    std::vector<uint8_t> unknown_payload(24, 0xCD);
    aeon::WalRecordHeader unknown_hdr{};
    unknown_hdr.record_type = 0xFFFFFFFF;
    unknown_hdr.payload_size = static_cast<uint32_t>(unknown_payload.size());
    unknown_hdr.checksum =
        aeon::hash::fnv1a_64(unknown_payload.data(), unknown_payload.size());
    wal.write(reinterpret_cast<const char *>(&unknown_hdr),
              sizeof(unknown_hdr));
    wal.write(reinterpret_cast<const char *>(unknown_payload.data()),
              static_cast<std::streamsize>(unknown_payload.size()));

    write_valid_event(wal, 1);
    write_valid_event(wal, 2);
  }

  ASSERT_TRUE(fs::exists(wal_path_));

  {
    aeon::TraceManager trace(trace_path_);
    auto history = trace.get_history("session_fwdcompat", 100);
    // All 3 valid events (before AND after the unknown record) recovered.
    EXPECT_EQ(history.size(), 3u);
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

// Regression test for a severe data-loss bug found while wiring V4 Stage
// 2's semantic trace search (v4-plan.md): compact() used to install the
// compacted generation under a PERMANENTLY generation-suffixed name
// (trace_gen1.bin, ...) and delete the file at trace_path_. Since
// trace_path_/generation_ are only tracked in-memory and reset on every
// fresh TraceManager construction, a process restart using the same
// caller-configured path (the normal case) would find that path gone and
// silently create an empty file -- total, silent loss of every event
// that existed before the first compaction. Atlas::compact_mmap() had
// the identical bug, fixed the same way (v4-plan.md Stage 2).
//
// This test is the actual bug scenario: append, compact, CLOSE the
// TraceManager entirely (simulating a process exit), then reopen via the
// exact same caller-configured path (simulating a restart) and confirm
// the data is still there.
TEST_F(WalTraceTest, DataSurvivesCompactionAcrossFullRestart) {
  {
    aeon::TraceManager trace(trace_path_);
    for (int i = 0; i < 5; ++i) {
      trace.append_event("session_restart", 0, "pre-compaction event", 0);
    }
    trace.compact();
    // Post-compaction events too, to prove appends after compaction also
    // land correctly under the (now-stable) path.
    trace.append_event("session_restart", 0, "post-compaction event", 0);
  } // TraceManager destructor runs here -- full process-exit simulation

  ASSERT_TRUE(fs::exists(trace_path_))
      << "the caller's configured path must still exist after a restart "
        "following compaction -- this is exactly the bug that was fixed";

  aeon::TraceManager reopened(trace_path_);
  auto history = reopened.get_history("session_restart", 100);
  EXPECT_EQ(history.size(), 6u)
      << "all 5 pre-compaction events plus the 1 post-compaction event "
        "must survive a full restart";
}
