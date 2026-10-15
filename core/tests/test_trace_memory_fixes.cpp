/**
 * @file test_trace_memory_fixes.cpp
 * @brief Trace fixes a long-term memory store relies on (Aleph Zero, ADR-006):
 * ids that never change across a crash, crash-recovered events that are
 * searchable, and an event that can be deleted with its text.
 *
 *   - WAL replay keeps each event's id, puts it in the trace file and empties
 *     the log, so a later append never chains to an event that lives only in
 *     RAM (a prev_id cycle on the next open).
 *   - A replayed event with an embedding is searchable, also when the log
 *     holds the file's first embedding.
 *   - tombstone_event(id, erase_text) leaves the event out of search and, with
 *     erase_text, overwrites its text and vector on disk; durable across reopen.
 */

#include "aeon/blob_arena.hpp"
#include "aeon/hash.hpp"
#include "aeon/schema.hpp"
#include "aeon/trace.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

std::vector<float> axis(uint32_t dim, uint32_t at) {
  std::vector<float> v(dim, 0.0f);
  v[at] = 1.0f;
  return v;
}

std::string slurp(const fs::path &path) {
  std::ifstream in(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(in), {});
}

} // namespace

class TraceMemoryFixesTest : public ::testing::Test {
protected:
  fs::path trace_path_, wal_path_, blob_path_;

  void SetUp() override {
    trace_path_ = fs::temp_directory_path() / "test_trace_memory_fixes.bin";
    wal_path_ = trace_path_;
    wal_path_ += ".wal";
    blob_path_ = trace_path_;
    blob_path_ += ".blobs";
    clean();
  }
  void TearDown() override { clean(); }

  void clean() {
    std::error_code ec;
    for (const auto &p : {trace_path_, wal_path_, blob_path_})
      fs::remove(p, ec);
  }

  // A WAL record as append_event() writes one while a compaction runs: with the
  // id and prev_id reserved before the write (0 and 0 in a record from before).
  void write_wal_event(std::ofstream &wal, uint64_t id, const std::string &sid,
                       const std::string &preview, uint64_t emb_offset = 0,
                       uint32_t emb_size = 0, uint64_t prev_id = 0) {
    aeon::TraceEvent ev{};
    std::memset(&ev, 0, sizeof(ev));
    ev.id = id;
    ev.prev_id = prev_id;
    ev.timestamp = 1000 + id;
    std::strncpy(ev.session_id, sid.c_str(), sizeof(ev.session_id) - 1);
    std::strncpy(ev.text_preview, preview.c_str(), sizeof(ev.text_preview) - 1);
    ev.embedding_blob_offset = emb_offset;
    ev.embedding_blob_size = emb_size;
    aeon::WalRecordHeader hdr{};
    hdr.record_type = aeon::WAL_RECORD_TRACE;
    hdr.payload_size = sizeof(aeon::TraceEvent);
    hdr.checksum = aeon::hash::fnv1a_64(&ev, sizeof(ev));
    wal.write(reinterpret_cast<const char *>(&hdr), sizeof(hdr));
    wal.write(reinterpret_cast<const char *>(&ev), sizeof(ev));
  }
};

TEST_F(TraceMemoryFixesTest, ReplayKeepsIdsWritesTheFileAndEmptiesTheLog) {
  { aeon::TraceManager trace(trace_path_); }
  {
    std::ofstream wal(wal_path_, std::ios::binary | std::ios::trunc);
    write_wal_event(wal, 7, "s", "seven");
    write_wal_event(wal, 8, "s", "eight", 0, 0, /*prev_id=*/7);
  }
  uint64_t next = 0;
  {
    aeon::TraceManager trace(trace_path_);
    auto history = trace.get_history("s", 10);
    ASSERT_EQ(history.size(), 2u);
    EXPECT_EQ(history[0].id, 8u); // newest first, the ids the log carried
    EXPECT_EQ(history[1].id, 7u);
    next = trace.append_event("s", 0, "nine");
    EXPECT_EQ(next, 9u);
  }
  EXPECT_TRUE(!fs::exists(wal_path_) || fs::file_size(wal_path_) == 0);
  {
    // Reopened: the three are in the file, chained 9 -> 8 -> 7, no cycle.
    aeon::TraceManager trace(trace_path_);
    auto history = trace.get_history("s", 10);
    ASSERT_EQ(history.size(), 3u);
    EXPECT_EQ(history[0].id, 9u);
    EXPECT_EQ(history[1].id, 8u);
    EXPECT_EQ(history[2].id, 7u);
  }
}

TEST_F(TraceMemoryFixesTest, TheSameLogReplayedAgainAddsNothing) {
  // A crash after the replay wrote the file but before the log was emptied.
  { aeon::TraceManager trace(trace_path_); }
  auto write_log = [&] {
    std::ofstream wal(wal_path_, std::ios::binary | std::ios::trunc);
    write_wal_event(wal, 7, "s", "seven");
    write_wal_event(wal, 8, "s", "eight", 0, 0, /*prev_id=*/7);
  };
  write_log();
  { aeon::TraceManager trace(trace_path_); }
  write_log();
  {
    aeon::TraceManager trace(trace_path_);
    EXPECT_EQ(trace.get_history("s", 10).size(), 2u);
    EXPECT_EQ(trace.size(), 2u);
    EXPECT_EQ(trace.append_event("s", 0, "next"), 9u); // the counter passed the skipped ids
  }
  // Nothing in the log was new, and it is gone all the same (agy: it stayed).
  EXPECT_TRUE(!fs::exists(wal_path_) || fs::file_size(wal_path_) == 0);
}

TEST_F(TraceMemoryFixesTest, LegacyRecordsWithoutAnIdStillGetOne) {
  { aeon::TraceManager trace(trace_path_); }
  {
    std::ofstream wal(wal_path_, std::ios::binary | std::ios::trunc);
    write_wal_event(wal, 0, "s", "old one");
    write_wal_event(wal, 0, "s", "old two");
  }
  aeon::TraceManager trace(trace_path_);
  auto history = trace.get_history("s", 10);
  ASSERT_EQ(history.size(), 2u);
  EXPECT_NE(history[0].id, history[1].id);
  EXPECT_GT(trace.append_event("s", 0, "new"), history[0].id);
}

TEST_F(TraceMemoryFixesTest, AReplayedEmbeddedEventIsSearchableEvenAsTheFirstEmbedding) {
  constexpr uint32_t dim = 8;
  { aeon::TraceManager trace(trace_path_); } // a file with no embedding yet
  aeon::BlobRef ref{};
  {
    aeon::BlobArena arena(blob_path_);
    auto v = axis(dim, 3);
    ref = arena.append(reinterpret_cast<const char *>(v.data()), v.size() * sizeof(float));
    arena.close();
  }
  {
    std::ofstream wal(wal_path_, std::ios::binary | std::ios::trunc);
    write_wal_event(wal, 1, "s", "recovered", ref.offset, ref.size);
  }
  aeon::TraceManager trace(trace_path_);
  auto found = trace.semantic_search(axis(dim, 3), 5);
  ASSERT_EQ(found.size(), 1u);
  EXPECT_EQ(found[0].id, 1u);
}

TEST_F(TraceMemoryFixesTest, ATombstoneWithErasureLeavesSearchAndTheDiskForGood) {
  constexpr uint32_t dim = 8;
  uint64_t gone = 0, kept = 0;
  const std::string secret = "SECRET-memory-text-that-must-leave-the-disk";
  {
    aeon::TraceManager trace(trace_path_);
    gone = trace.append_event("s", 0, secret.c_str(), 0, axis(dim, 1));
    kept = trace.append_event("s", 0, "kept", 0, axis(dim, 2));
    EXPECT_TRUE(trace.tombstone_event(gone, /*erase_text=*/true));
    EXPECT_FALSE(trace.tombstone_event(gone, true)); // once
    auto found = trace.semantic_search(axis(dim, 1), 5);
    for (const auto &ev : found)
      EXPECT_NE(ev.id, gone);
  }
  EXPECT_EQ(slurp(blob_path_).find("SECRET-"), std::string::npos);
  EXPECT_EQ(slurp(trace_path_).find("SECRET-"), std::string::npos);
  {
    aeon::TraceManager trace(trace_path_);
    for (const auto &ev : trace.semantic_search(axis(dim, 1), 5))
      EXPECT_NE(ev.id, gone);
    auto found = trace.semantic_search(axis(dim, 2), 1);
    ASSERT_EQ(found.size(), 1u);
    EXPECT_EQ(found[0].id, kept);
  }
}

TEST_F(TraceMemoryFixesTest, ATombstoneWithoutErasureKeepsTheTextButHidesTheEvent) {
  constexpr uint32_t dim = 8;
  aeon::TraceManager trace(trace_path_);
  uint64_t id = trace.append_event("s", 0, "still-on-disk-text", 0, axis(dim, 4));
  EXPECT_TRUE(trace.tombstone_event(id));
  EXPECT_TRUE(trace.semantic_search(axis(dim, 4), 5).empty());
  EXPECT_FALSE(trace.tombstone_event(999999)); // unknown id
}
