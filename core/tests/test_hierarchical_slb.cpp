// Regression coverage for v4-plan.md guardrail #1.1: HierarchicalSLB (and
// its predecessor bug in SessionRingBuffer/GlobalCacheEntry) used to hardcode
// centroid storage to the compile-time EMBEDDING_DIM constant (768), so any
// non-768-dim Atlas silently got zero SLB cache acceleration -- insert()
// guarded `if (centroid.size() != EMBEDDING_DIM) return;` and simply did
// nothing, with no error surfaced anywhere. These tests pin the fixed
// behavior: cache hits must actually occur for 384/1536/3072-dim sessions,
// not just for the historical 768-dim default.

#include "aeon/hierarchical_slb.hpp"
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace aeon;

namespace {

std::vector<float> random_vector(uint32_t dim, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(dim);
  for (auto &x : v)
    x = dist(rng);
  return v;
}

} // namespace

class HierarchicalSLBDimTest : public ::testing::TestWithParam<uint32_t> {};

// The core regression test: insert then immediately query with the exact
// same vector (cosine similarity == 1.0) must hit, for every supported dim.
// Before the fix, this passed only for dim == 768 (EMBEDDING_DIM) and
// silently returned nullopt for every other dimension.
TEST_P(HierarchicalSLBDimTest, InsertThenFindNearestHitsAtEveryDim) {
  const uint32_t dim = GetParam();
  HierarchicalSLB slb(dim);
  ASSERT_EQ(slb.dim(), dim);

  auto centroid = random_vector(dim, /*seed=*/dim);
  constexpr uint64_t session_id = 1;
  constexpr uint64_t node_id = 42;

  slb.insert(session_id, node_id, centroid);

  auto hit = slb.find_nearest(session_id, centroid, /*threshold=*/0.5f);
  ASSERT_TRUE(hit.has_value())
      << "dim=" << dim
      << ": expected an L1 cache hit; got a silent miss (the exact bug "
         "this test guards against)";
  EXPECT_EQ(hit->node_id, node_id);
  EXPECT_NEAR(hit->similarity, 1.0f, 1e-4f);
}

// L2 (global cache) fallback must also work at every dim -- a different
// session querying the same centroid should hit via scan_global_cache(),
// which had the identical hardcoded-768 bug in GlobalCacheEntry.
TEST_P(HierarchicalSLBDimTest, L2GlobalCacheHitsAcrossSessionsAtEveryDim) {
  const uint32_t dim = GetParam();
  HierarchicalSLB slb(dim);

  auto centroid = random_vector(dim, /*seed=*/dim + 1);
  constexpr uint64_t writer_session = 10;
  constexpr uint64_t reader_session = 20; // different session -> L1 miss
  constexpr uint64_t node_id = 7;

  slb.insert(writer_session, node_id, centroid);

  auto hit = slb.find_nearest(reader_session, centroid, /*threshold=*/0.5f);
  ASSERT_TRUE(hit.has_value())
      << "dim=" << dim << ": expected an L2 global-cache hit";
  EXPECT_EQ(hit->node_id, node_id);
}

// Multiple sessions at the same non-default dim must not corrupt each
// other's centroid storage -- this is the part that would break under a
// naive "reinterpret the flat buffer" implementation if the per-entry
// stride were computed incorrectly.
TEST_P(HierarchicalSLBDimTest, MultipleEntriesAtSameDimDoNotAlias) {
  const uint32_t dim = GetParam();
  HierarchicalSLB slb(dim);
  constexpr uint64_t session_id = 1;

  std::vector<std::vector<float>> centroids;
  for (uint64_t i = 0; i < 5; ++i) {
    centroids.push_back(random_vector(dim, static_cast<unsigned>(dim + i + 100)));
    slb.insert(session_id, /*node_id=*/i, centroids.back());
  }

  for (uint64_t i = 0; i < 5; ++i) {
    auto hit = slb.find_nearest(session_id, centroids[i], 0.99f);
    ASSERT_TRUE(hit.has_value()) << "dim=" << dim << " node=" << i;
    EXPECT_EQ(hit->node_id, i) << "dim=" << dim
                                << ": wrong node returned -- possible "
                                   "centroid storage aliasing";
  }
}

INSTANTIATE_TEST_SUITE_P(SupportedDimensions, HierarchicalSLBDimTest,
                        ::testing::Values(384u, 768u, 1536u, 3072u));

// A query/centroid whose size doesn't match the instance's dim must be a
// clean miss (or no-op on insert), never a crash or silent OOB read --
// matches the pre-existing size-mismatch contract elsewhere in the codebase.
TEST(HierarchicalSLBTest, MismatchedSizeIsSafeNoOp) {
  HierarchicalSLB slb(/*dim=*/384);
  auto wrong_size = random_vector(768, 1);

  // insert() with wrong-sized centroid must not corrupt state.
  slb.insert(/*session_id=*/1, /*node_id=*/1, wrong_size);
  EXPECT_EQ(slb.active_session_count(), 0u)
      << "insert() with mismatched dim should be a no-op, not create a "
         "session with corrupt data";

  // find_nearest() with wrong-sized query must return nullopt, not crash.
  auto hit = slb.find_nearest(/*session_id=*/1, wrong_size, 0.0f);
  EXPECT_FALSE(hit.has_value());
}

TEST(HierarchicalSLBTest, DefaultConstructorUsesEmbeddingDimDefault) {
  HierarchicalSLB slb; // no dim specified
  EXPECT_EQ(slb.dim(), EMBEDDING_DIM_DEFAULT);
}
