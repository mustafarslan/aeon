/**
 * @file test_trace_semantic_search.cpp
 * @brief V4 Stage 2 task 3: TraceManager::semantic_search() via the
 * (now dynamic-dim) TraceBlockIndex, and the block-GC-on-compaction fix
 * found while wiring it in.
 */

#include "aeon/trace.hpp"

#include <cmath>
#include <filesystem>
#include <gtest/gtest.h>
#include <vector>

namespace fs = std::filesystem;

namespace {
std::vector<float> unit_vec(float value, uint32_t dim) {
  std::vector<float> v(dim, value);
  return v;
}
} // namespace

class TraceSemanticSearchTest : public ::testing::Test {
protected:
  fs::path trace_path_;

  void SetUp() override {
    trace_path_ = fs::temp_directory_path() / "test_trace_semsearch.bin";
    std::error_code ec;
    fs::remove(trace_path_, ec);
  }

  void TearDown() override {
    std::error_code ec;
    fs::remove(trace_path_, ec);
    auto blob_path = trace_path_;
    blob_path += ".blobs";
    fs::remove(blob_path, ec);
    auto wal_path = trace_path_;
    wal_path += ".wal";
    fs::remove(wal_path, ec);
    // Stale temp-compaction files, if a test failed mid-compaction.
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

TEST_F(TraceSemanticSearchTest, EmbeddingDimStartsZeroThenLocksIn) {
  aeon::TraceManager trace(trace_path_);
  EXPECT_EQ(trace.embedding_dim(), 0u);

  trace.append_event("s1", 0, "first embedded event", 0, unit_vec(1.0f, 16));
  EXPECT_EQ(trace.embedding_dim(), 16u);
}

TEST_F(TraceSemanticSearchTest, MismatchedDimThrows) {
  aeon::TraceManager trace(trace_path_);
  trace.append_event("s1", 0, "first", 0, unit_vec(1.0f, 16));

  EXPECT_THROW(
      trace.append_event("s1", 0, "wrong dim", 0, unit_vec(1.0f, 8)),
      std::invalid_argument);
}

TEST_F(TraceSemanticSearchTest, SemanticSearchFindsClosestEmbeddedEvent) {
  aeon::TraceManager trace(trace_path_);

  // cos_sim to query=(1,1,...,1): target=1.0, near=-0.25, far=-1.0
  std::vector<float> target(32, 1.0f);
  std::vector<float> near(32, 1.0f);
  std::fill(near.begin() + 12, near.end(), -1.0f); // 12 ones + 20 neg-ones -> -0.25
  std::vector<float> far(32, -1.0f);

  uint64_t target_id =
      trace.append_event("s1", 0, "the target event", 0, target);
  trace.append_event("s1", 0, "a distant event", 0, near);
  trace.append_event("s1", 0, "the opposite event", 0, far);

  std::vector<float> query(32, 1.0f);
  auto results = trace.semantic_search(query, 3);

  ASSERT_FALSE(results.empty());
  EXPECT_EQ(results[0].id, target_id);
}

TEST_F(TraceSemanticSearchTest, UnembeddedEventsAreInvisibleToSemanticSearch) {
  aeon::TraceManager trace(trace_path_);

  trace.append_event("s1", 0, "no embedding here", 0); // embedding = {}
  std::vector<float> embedded(16, 1.0f);
  uint64_t embedded_id =
      trace.append_event("s1", 0, "this one is embedded", 0, embedded);

  std::vector<float> query(16, 1.0f);
  auto results = trace.semantic_search(query, 10);

  ASSERT_EQ(results.size(), 1u);
  EXPECT_EQ(results[0].id, embedded_id);
}

// V4 Stage 1/2 task 4: edge_type/supersedes_id/reason_code (built in Stage
// 1, never had a real caller until Stage 2 task 4's admission-time
// near-duplicate detection) round-trip correctly through append_event().
TEST_F(TraceSemanticSearchTest, EdgeFieldsRoundTripThroughAppendEvent) {
  aeon::TraceManager trace(trace_path_);

  uint64_t original_id = trace.append_event("s1", 0, "original content", 0);
  uint64_t refines_id = trace.append_event(
      "s1", 2 /* Concept */, "a near-duplicate", 0, {},
      static_cast<uint8_t>(aeon::EdgeType::Refines), original_id,
      static_cast<uint8_t>(aeon::ReasonCode::Unspecified));

  auto history = trace.get_history("s1", 10);
  const aeon::TraceEvent *refines_ev = nullptr;
  for (const auto &ev : history) {
    if (ev.id == refines_id)
      refines_ev = &ev;
  }
  ASSERT_NE(refines_ev, nullptr);
  EXPECT_EQ(refines_ev->edge_type,
           static_cast<uint8_t>(aeon::EdgeType::Refines));
  EXPECT_EQ(refines_ev->supersedes_id, original_id);
  EXPECT_EQ(refines_ev->reason_code,
           static_cast<uint8_t>(aeon::ReasonCode::Unspecified));

  // The original (non-edge) event defaults to no edge.
  const aeon::TraceEvent *original_ev = nullptr;
  for (const auto &ev : history) {
    if (ev.id == original_id)
      original_ev = &ev;
  }
  ASSERT_NE(original_ev, nullptr);
  EXPECT_EQ(original_ev->edge_type,
           static_cast<uint8_t>(aeon::EdgeType::None));
  EXPECT_EQ(original_ev->supersedes_id, 0u);
}

// V4 Stage 7 Track 2: event_time round-trips through append_event() and
// get_history(), independent of `timestamp` (Aeon's own insertion
// wall-clock, which append_event() always sets regardless of event_time).
TEST_F(TraceSemanticSearchTest, EventTimeRoundTripsThroughAppendEvent) {
  aeon::TraceManager trace(trace_path_);

  uint64_t explicit_time = 1'700'000'000'000'000ULL; // an arbitrary past epoch-micros value
  uint64_t with_time_id = trace.append_event(
      "s1", 0, "backdated content", 0, {}, 0, 0, 0, explicit_time);
  uint64_t without_time_id = trace.append_event("s1", 0, "normal content");

  auto history = trace.get_history("s1", 10);
  const aeon::TraceEvent *with_time_ev = nullptr;
  const aeon::TraceEvent *without_time_ev = nullptr;
  for (const auto &ev : history) {
    if (ev.id == with_time_id)
      with_time_ev = &ev;
    if (ev.id == without_time_id)
      without_time_ev = &ev;
  }

  ASSERT_NE(with_time_ev, nullptr);
  ASSERT_NE(without_time_ev, nullptr);
  EXPECT_EQ(with_time_ev->event_time, explicit_time);
  // event_time is independent of timestamp -- both are set on every event,
  // to different values here, and neither overwrites the other.
  EXPECT_NE(with_time_ev->timestamp, 0u);
  EXPECT_NE(with_time_ev->timestamp, explicit_time);
  // Default (not supplied) is 0, not a copy of timestamp.
  EXPECT_EQ(without_time_ev->event_time, 0u);
  EXPECT_NE(without_time_ev->timestamp, 0u);
}

TEST_F(TraceSemanticSearchTest, EmptyIndexReturnsEmptyNotCrash) {
  aeon::TraceManager trace(trace_path_);
  std::vector<float> query(16, 1.0f);
  auto results = trace.semantic_search(query, 5);
  EXPECT_TRUE(results.empty());
}

// Closes the gap found while wiring this feature in: compact() re-points
// blob offsets for text but, before this fix, NOT for embedding_blob_*
// (or Stage 1's still-unused evidence_blob_*) -- an embedded event
// surviving compaction would keep pointing at the OLD (soon-deleted)
// blob file. Not a TraceBlockIndex concern (it stores values, not
// pointers) but a real, separate durability bug in the blob GC path.
TEST_F(TraceSemanticSearchTest, EmbeddingSurvivesCompactionReadableAfterReopen) {
  std::vector<float> emb(16, 1.0f);
  {
    aeon::TraceManager trace(trace_path_);
    trace.append_event("s1", 0, "keep me", 0, emb);
    trace.compact();
  }

  // Reopen via the SAME stable, caller-configured path -- this is
  // exactly the severe data-loss bug found and fixed while writing this
  // test (v4-plan.md Stage 2): compact() used to install the new
  // generation under a DIFFERENT, permanently generation-suffixed name
  // and delete trace_path_, so reopening it here would previously find
  // nothing and silently create an empty file. Now compact() atomically
  // renames the new generation back onto trace_path_, so this is exactly
  // the shape of a real process restart after a real compaction.
  ASSERT_TRUE(fs::exists(trace_path_));

  // Reopen: rebuild_block_index() must find the embedding via the
  // POST-compaction blob offsets, not stale pre-compaction ones.
  aeon::TraceManager trace2(trace_path_);
  EXPECT_EQ(trace2.embedding_dim(), 16u);

  std::vector<float> query(16, 1.0f);
  auto results = trace2.semantic_search(query, 5);
  ASSERT_EQ(results.size(), 1u);
  EXPECT_EQ(results[0].embedding_blob_size, 16u * sizeof(float));
}

// V4 Stage 7 Track 2: unlike embedding_blob_*/blob_offset (file-relative
// offsets that compact() must explicitly re-point into the new generation
// file), event_time is a plain scalar with no indirection -- compact()'s
// whole-struct memcpy() carries it over for free. Verified directly
// rather than assumed.
TEST_F(TraceSemanticSearchTest, EventTimeSurvivesCompaction) {
  uint64_t explicit_time = 1'700'000'000'000'000ULL;
  uint64_t id;
  {
    aeon::TraceManager trace(trace_path_);
    id = trace.append_event("s1", 0, "backdated content", 0, {}, 0, 0, 0,
                            explicit_time);
    trace.compact();
  }

  aeon::TraceManager trace2(trace_path_);
  auto history = trace2.get_history("s1", 10);
  const aeon::TraceEvent *ev = nullptr;
  for (const auto &e : history) {
    if (e.id == id)
      ev = &e;
  }
  ASSERT_NE(ev, nullptr);
  EXPECT_EQ(ev->event_time, explicit_time);
}
