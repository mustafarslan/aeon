#include "aeon/atlas.hpp"
#include <algorithm>
#include <filesystem>
#include <gtest/gtest.h>
#include <vector>

class AtlasTest : public ::testing::Test {
protected:
  // A bare relative filename (the previous "test_atlas.aeon") writes into
  // whatever directory ctest is invoked from -- if that's the repo root
  // (as it was for whoever committed test_atlas.aeon.wal into git history
  // at 281da68), the WAL sidecar this fixture creates leaks into the repo
  // and never gets cleaned up (TearDown only ever removed test_path
  // itself, never its .wal). Every other fixture in this suite already
  // uses fs::temp_directory_path() (see WalAtlasTest, test_wal.cpp) --
  // this one was the sole outlier.
  std::filesystem::path test_path =
      std::filesystem::temp_directory_path() / "aeon_test_atlas.bin";

  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove(test_path, ec);
    auto wal_path = test_path;
    wal_path += ".wal";
    std::filesystem::remove(wal_path, ec);
  }
};

// compact_mmap() writes a new generation file at a DIFFERENT path
// (<stem>_genN.bin) and reassigns atlas_path_ to it, so a compaction test
// needs its own scratch directory rather than a single fixed test_path --
// otherwise generation files and the .wal file leak between test runs.
class AtlasCompactionTest : public ::testing::Test {
protected:
  std::filesystem::path test_dir;
  std::filesystem::path test_path;

  void SetUp() override {
    test_dir = std::filesystem::temp_directory_path() /
              ("aeon_compaction_test_" +
               std::to_string(reinterpret_cast<uintptr_t>(this)));
    std::filesystem::create_directories(test_dir);
    test_path = test_dir / "compact.bin";
  }

  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(test_dir, ec);
  }
};

TEST_F(AtlasTest, InsertAndRetrieve) {
  aeon::Atlas atlas(test_path);
  std::vector<float> vec(768, 0.0f);

  uint64_t id = atlas.insert(0, vec, "Root Node");
  EXPECT_EQ(id, 0); // First node is 0
  EXPECT_EQ(atlas.size(), 1);
}

// V4 Stage 1/2 prerequisite: set_node_scope()/get_node_scope() are the only
// write path for scope_bitmap on an already-inserted node. These close the
// test-coverage gap noted in v4-plan.md's Stage 1 status -- there was
// previously no way to exercise non-zero scope_bitmap at all.
TEST_F(AtlasTest, SetAndGetNodeScopeRoundTrip) {
  aeon::Atlas atlas(test_path);
  std::vector<float> vec(768, 0.0f);
  uint64_t id = atlas.insert(0, vec, "Root");

  EXPECT_EQ(atlas.get_node_scope(id), 0u); // default

  atlas.set_node_scope(id, 0x5A5Au);
  EXPECT_EQ(atlas.get_node_scope(id), 0x5A5Au);

  // Overwriting with a new value replaces, doesn't OR/accumulate.
  atlas.set_node_scope(id, 0x0Fu);
  EXPECT_EQ(atlas.get_node_scope(id), 0x0Fu);
}

// V4 Stage 2 task 1: emission-time scope filtering. Basic correctness
// before the recall-vs-exhaustive-scan benchmark: only scope-matching
// nodes are ever reported once a real scope_mask is passed.
TEST_F(AtlasTest, NavigateScopeFilterExcludesNonMatchingNodes) {
  aeon::Atlas atlas(test_path);
  std::vector<float> root(768, -1.0f);
  uint64_t root_id = atlas.insert(0, root, "Root");

  // Same direction-differentiated vector construction as the supersede
  // test -- cosine similarity is scale-invariant, so same-direction
  // vectors would tie.
  std::vector<float> best_match(768, 1.0f); // cos_sim = 1.0
  std::vector<float> second_match(768, 1.0f);
  std::fill(second_match.begin() + 384, second_match.end(), -1.0f); // 0.0

  uint64_t best_id = atlas.insert(root_id, best_match, "BestMatch");
  uint64_t second_id = atlas.insert(root_id, second_match, "SecondMatch");

  constexpr uint64_t kScopeA = 0x1u;
  constexpr uint64_t kScopeB = 0x2u;
  atlas.set_node_scope(best_id, kScopeA); // NOT in scope B
  atlas.set_node_scope(second_id, kScopeB);

  std::vector<float> query(768, 1.0f);

  // No filter (default ALL_SCOPES_VISIBLE): best_match wins as always.
  auto unfiltered = atlas.navigate(query, 2, /*apply_csls=*/false);
  bool saw_best = false;
  for (const auto &r : unfiltered)
    if (r.id == best_id)
      saw_best = true;
  EXPECT_TRUE(saw_best);

  // Filtered to scope B: best_match (scope A only) must never appear,
  // even though it's the highest raw-similarity candidate.
  auto filtered = atlas.navigate(query, 2, /*apply_csls=*/false,
                                 /*session_id=*/0, /*scope_mask=*/kScopeB);
  for (const auto &r : filtered) {
    EXPECT_NE(r.id, best_id);
  }
  bool saw_second = false;
  for (const auto &r : filtered)
    if (r.id == second_id)
      saw_second = true;
  EXPECT_TRUE(saw_second);
}

// V4 Stage 2 task 2: get_children() is the Atlas->Trace->Atlas
// graph-expansion-boundary enforcement point -- a caller that reached
// parent_id via a Trace event's atlas_id must not enumerate children
// outside its own scope just because the starting node was legitimately
// theirs (get_children() is otherwise completely unscoped, unlike
// navigate()).
TEST_F(AtlasTest, GetChildrenScopeFilterExcludesNonMatchingChildren) {
  aeon::Atlas atlas(test_path);
  std::vector<float> root_vec(768, 0.0f);
  uint64_t root_id = atlas.insert(0, root_vec, "Root");

  std::vector<float> child_vec(768, 1.0f);
  uint64_t child_a = atlas.insert(root_id, child_vec, "ChildA");
  uint64_t child_b = atlas.insert(root_id, child_vec, "ChildB");
  uint64_t child_c = atlas.insert(root_id, child_vec, "ChildC");

  constexpr uint64_t kScopeX = 0x1u;
  constexpr uint64_t kScopeY = 0x2u;
  atlas.set_node_scope(child_a, kScopeX);
  atlas.set_node_scope(child_b, kScopeY);
  // child_c left unscoped (scope_bitmap == 0)

  auto unfiltered = atlas.get_children(root_id);
  EXPECT_EQ(unfiltered.size(), 3u);

  auto filtered_x = atlas.get_children(root_id, kScopeX);
  ASSERT_EQ(filtered_x.size(), 1u);
  EXPECT_EQ(filtered_x[0].id, child_a);

  auto filtered_y = atlas.get_children(root_id, kScopeY);
  ASSERT_EQ(filtered_y.size(), 1u);
  EXPECT_EQ(filtered_y[0].id, child_b);

  // Unscoped child_c never matches any real (non-sentinel) filter.
  auto filtered_either = atlas.get_children(root_id, kScopeX | kScopeY);
  EXPECT_EQ(filtered_either.size(), 2u);
  for (const auto &r : filtered_either) {
    EXPECT_NE(r.id, child_c);
  }
}

TEST_F(AtlasTest, SetNodeScopeRejectsDeltaArenaId) {
  aeon::Atlas atlas(test_path);
  std::vector<float> vec(768, 0.0f);
  uint64_t delta_id = atlas.insert_delta(vec, "delta node");
  ASSERT_NE(delta_id & 0x8000000000000000ULL, 0u); // sanity: really a delta id

  EXPECT_THROW(atlas.set_node_scope(delta_id, 1), std::invalid_argument);
  EXPECT_THROW(atlas.get_node_scope(delta_id), std::invalid_argument);
}

// V4 Stage 2 prerequisite: supersede_node()/revoke_node_supersede() are the
// live write path for NODE_FLAG_SUPERSEDED, needed so Stage 2's "superseded
// fragments are excluded from navigate()" gate can actually be exercised.
TEST_F(AtlasTest, SupersedeNodeExcludesFromCSLSNavigation) {
  // Branchless exclusion (hub_penalty = TOMBSTONE_PENALTY) only actually
  // REMOVES a node from the returned beam once something else displaces
  // it -- with more candidate children than beam_width, exactly as a
  // realistic tree has. A beam wide enough to fit every candidate
  // (regardless of score) wouldn't exercise displacement at all; this is
  // the same characteristic tombstoning already has.
  aeon::Atlas atlas(test_path);
  // Non-zero, clearly-dissimilar root vector: cosine_similarity() against
  // an all-zero vector is 0/0 = NaN, which breaks std::sort's ordering
  // guarantee once navigate()'s final sort has enough entries to expose
  // it -- a pre-existing, out-of-scope quirk unrelated to supersession;
  // sidestepped here rather than fixed as part of this test.
  std::vector<float> root(768, -1.0f);
  uint64_t root_id = atlas.insert(0, root, "Root");

  // Cosine similarity is scale-invariant -- a positive scalar multiple of
  // the query (e.g. 0.8x) has cos_sim == 1.0, NOT 0.8, so vectors must
  // differ in DIRECTION (not just magnitude) to get genuinely distinct,
  // non-tied scores. query is all-1.0 (dim 768); target matches it
  // exactly (cos_sim = 1.0); the others flip an increasing fraction of
  // dimensions to -1.0 for cos_sim = 0.0 / -0.5 / -1.0 respectively.
  std::vector<float> target(768, 1.0f);
  std::vector<float> other_a(768, 1.0f);
  std::fill(other_a.begin() + 384, other_a.end(), -1.0f); // cos_sim = 0.0
  std::vector<float> other_b(768, 1.0f);
  std::fill(other_b.begin() + 192, other_b.end(), -1.0f); // cos_sim = -0.5
  std::vector<float> other_c(768, -1.0f);                 // cos_sim = -1.0

  uint64_t target_id = atlas.insert(root_id, target, "Target");
  atlas.insert(root_id, other_a, "OtherA");
  atlas.insert(root_id, other_b, "OtherB");
  atlas.insert(root_id, other_c, "OtherC");

  std::vector<float> query(768, 1.0f);
  constexpr uint32_t kBeamWidth = 2; // < 4 children -- displacement can occur

  // Before superseding: the exact-match node wins with apply_csls=true.
  auto before = atlas.navigate(query, kBeamWidth, /*apply_csls=*/true);
  ASSERT_FALSE(before.empty());
  EXPECT_EQ(before[0].id, target_id);

  EXPECT_FALSE(atlas.is_node_superseded(target_id));
  atlas.supersede_node(target_id);
  EXPECT_TRUE(atlas.is_node_superseded(target_id));

  // After superseding: same branchless exclusion mechanism as tombstone --
  // only takes effect with apply_csls=true (hub_penalty is only subtracted
  // in that mode, a pre-existing characteristic shared with tombstoning).
  // With a beam narrower than the candidate count, the two remaining
  // real matches (other_a, other_b) displace the now -1e9f-scored target.
  auto after = atlas.navigate(query, kBeamWidth, /*apply_csls=*/true);
  for (const auto &r : after) {
    EXPECT_NE(r.id, target_id);
  }

  // Revoke restores it.
  atlas.revoke_node_supersede(target_id);
  EXPECT_FALSE(atlas.is_node_superseded(target_id));
  auto restored = atlas.navigate(query, kBeamWidth, /*apply_csls=*/true);
  ASSERT_FALSE(restored.empty());
  EXPECT_EQ(restored[0].id, target_id);
}

TEST_F(AtlasTest, SupersedeNodeRejectsDeltaArenaId) {
  aeon::Atlas atlas(test_path);
  std::vector<float> vec(768, 0.0f);
  uint64_t delta_id = atlas.insert_delta(vec, "delta node");

  EXPECT_THROW(atlas.supersede_node(delta_id), std::invalid_argument);
  EXPECT_THROW(atlas.revoke_node_supersede(delta_id), std::invalid_argument);
  EXPECT_THROW(atlas.is_node_superseded(delta_id), std::invalid_argument);
}

// V4 Stage 4 task 5/6: Atlas::tombstone_node(uint64_t) -- the console/
// erasure-workflow "delete" primitive. Mirrors SupersedeNodeRejects*/
// SetNodeScope* test shapes.
TEST_F(AtlasTest, TombstoneNodeExcludesFromListAndCountsAsTombstoned) {
  aeon::Atlas atlas(test_path);
  std::vector<float> root(768, 0.0f);
  uint64_t root_id = atlas.insert(0, root, "Root");
  std::vector<float> vec(768, 0.5f);
  uint64_t id = atlas.insert(root_id, vec, "to be deleted");
  atlas.set_node_scope(id, 0x1u);

  ASSERT_EQ(atlas.tombstone_count(), 0u);
  auto before = atlas.list_nodes_by_scope(0x1u);
  ASSERT_NE(std::find(before.begin(), before.end(), id), before.end());

  atlas.tombstone_node(id);

  EXPECT_EQ(atlas.tombstone_count(), 1u);
  auto after = atlas.list_nodes_by_scope(0x1u);
  EXPECT_EQ(std::find(after.begin(), after.end(), id), after.end());
}

TEST_F(AtlasTest, TombstoneNodeIsIdempotent) {
  aeon::Atlas atlas(test_path);
  std::vector<float> vec(768, 0.0f);
  uint64_t id = atlas.insert(0, vec, "node");

  atlas.tombstone_node(id);
  EXPECT_EQ(atlas.tombstone_count(), 1u);
  atlas.tombstone_node(id); // second call must not double-count or throw
  EXPECT_EQ(atlas.tombstone_count(), 1u);
}

TEST_F(AtlasTest, TombstoneNodeRejectsDeltaArenaId) {
  aeon::Atlas atlas(test_path);
  std::vector<float> vec(768, 0.0f);
  uint64_t delta_id = atlas.insert_delta(vec, "delta node");

  EXPECT_THROW(atlas.tombstone_node(delta_id), std::invalid_argument);
}

TEST_F(AtlasTest, TombstoneNodeRejectsInvalidNodeId) {
  aeon::Atlas atlas(test_path);
  std::vector<float> vec(768, 0.0f);
  atlas.insert(0, vec, "Root"); // id 0 only

  EXPECT_THROW(atlas.tombstone_node(999), std::runtime_error);
}

// Advisor review (v4-plan.md Stage 4 task 5/6): tombstone_node() is
// TERMINAL, unlike supersede_node(), and does not stash/restore
// saved_hub_penalty. This pins the documented interaction -- superseding a
// node THEN tombstoning it must not corrupt the stashed saved_hub_penalty,
// and a subsequent revoke_node_supersede() on that same node must still
// leave it correctly tombstoned (schema.hpp's revoke_supersede() doc
// comment: "correctly leaves hub_penalty at TOMBSTONE_PENALTY if the node
// was ALSO tombstoned").
TEST_F(AtlasTest, SupersedeThenTombstoneThenRevokeSupersedeLeavesNodeTombstoned) {
  aeon::Atlas atlas(test_path);
  std::vector<float> vec(768, 0.0f);
  uint64_t id = atlas.insert(0, vec, "node");

  atlas.supersede_node(id);
  EXPECT_TRUE(atlas.is_node_superseded(id));

  atlas.tombstone_node(id);
  EXPECT_EQ(atlas.tombstone_count(), 1u);
  // Tombstoning doesn't touch NODE_FLAG_SUPERSEDED -- still reads as
  // superseded until explicitly revoked, independent of tombstone state.
  EXPECT_TRUE(atlas.is_node_superseded(id));

  atlas.revoke_node_supersede(id);
  EXPECT_FALSE(atlas.is_node_superseded(id));
  // The node must still be tombstoned after the revoke -- revoke_supersede()
  // only restores hub_penalty when NODE_FLAG_TOMBSTONE is NOT set.
  EXPECT_EQ(atlas.tombstone_count(), 1u);
}

TEST_F(AtlasTest, SetNodeScopeRejectsInvalidNodeId) {
  aeon::Atlas atlas(test_path);
  std::vector<float> vec(768, 0.0f);
  atlas.insert(0, vec, "Root"); // id 0 only

  EXPECT_THROW(atlas.set_node_scope(999, 1), std::runtime_error);
  EXPECT_THROW(atlas.get_node_scope(999), std::runtime_error);
}

// V4 Stage 4 task 1: set_node_governance_id()/get_node_governance_id() are
// the first writer this NodeHeader field (allocated in Stage 1's byte
// budget) has ever had -- mirrors SetAndGetNodeScopeRoundTrip exactly.
TEST_F(AtlasTest, SetAndGetNodeGovernanceIdRoundTrip) {
  aeon::Atlas atlas(test_path);
  std::vector<float> vec(768, 0.0f);
  uint64_t id = atlas.insert(0, vec, "Root");

  EXPECT_EQ(atlas.get_node_governance_id(id), 0u); // default

  atlas.set_node_governance_id(id, 0xCAFEu);
  EXPECT_EQ(atlas.get_node_governance_id(id), 0xCAFEu);

  atlas.set_node_governance_id(id, 0x1234u);
  EXPECT_EQ(atlas.get_node_governance_id(id), 0x1234u);
}

TEST_F(AtlasTest, SetNodeGovernanceIdRejectsDeltaArenaId) {
  aeon::Atlas atlas(test_path);
  std::vector<float> vec(768, 0.0f);
  uint64_t delta_id = atlas.insert_delta(vec, "delta node");

  EXPECT_THROW(atlas.set_node_governance_id(delta_id, 1),
               std::invalid_argument);
  EXPECT_THROW(atlas.get_node_governance_id(delta_id), std::invalid_argument);
}

TEST_F(AtlasTest, SetNodeGovernanceIdRejectsInvalidNodeId) {
  aeon::Atlas atlas(test_path);
  std::vector<float> vec(768, 0.0f);
  atlas.insert(0, vec, "Root"); // id 0 only

  EXPECT_THROW(atlas.set_node_governance_id(999, 1), std::runtime_error);
  EXPECT_THROW(atlas.get_node_governance_id(999), std::runtime_error);
}

// V4 Stage 4 task 2: get_node_metadata() -- promotion needs to read a
// source fragment's text back out before it can classify/re-embed/copy it.
// insert()/insert_delta() have always written this field; nothing read it
// back until now.
TEST_F(AtlasTest, GetNodeMetadataRoundTripMmap) {
  aeon::Atlas atlas(test_path);
  std::vector<float> vec(768, 0.0f);
  uint64_t id = atlas.insert(0, vec, "hello world");

  EXPECT_EQ(atlas.get_node_metadata(id), "hello world");
}

TEST_F(AtlasTest, GetNodeMetadataRoundTripDeltaArena) {
  aeon::Atlas atlas(test_path);
  std::vector<float> vec(768, 0.0f);
  uint64_t delta_id = atlas.insert_delta(vec, "fresh delta fragment");

  // Unlike scope/governance accessors, delta ids ARE supported -- promotion
  // needs to read same-turn admissions that haven't been compacted yet.
  EXPECT_EQ(atlas.get_node_metadata(delta_id), "fresh delta fragment");
}

TEST_F(AtlasTest, GetNodeMetadataRejectsInvalidNodeId) {
  aeon::Atlas atlas(test_path);
  std::vector<float> vec(768, 0.0f);
  atlas.insert(0, vec, "Root"); // id 0 only

  EXPECT_THROW(atlas.get_node_metadata(999), std::runtime_error);
}

// V4 Stage 4 task 2: get_node_centroid() -- promotion needs a source
// node's FULL vector, not just query()/get_children()'s 3-float preview.
TEST_F(AtlasTest, GetNodeCentroidRoundTripFp32Mmap) {
  aeon::Atlas atlas(test_path);
  std::vector<float> vec(768, 0.0f);
  for (size_t i = 0; i < vec.size(); ++i)
    vec[i] = static_cast<float>(i) * 0.001f;
  uint64_t id = atlas.insert(0, vec, "Root");

  auto out = atlas.get_node_centroid(id);
  ASSERT_EQ(out.size(), vec.size());
  for (size_t i = 0; i < vec.size(); ++i)
    EXPECT_FLOAT_EQ(out[i], vec[i]);
}

TEST_F(AtlasTest, GetNodeCentroidRoundTripDeltaArena) {
  aeon::Atlas atlas(test_path);
  std::vector<float> vec(768, 0.0f);
  vec[0] = 0.5f;
  vec[1] = -0.25f;
  uint64_t delta_id = atlas.insert_delta(vec, "delta fragment");

  auto out = atlas.get_node_centroid(delta_id);
  ASSERT_EQ(out.size(), vec.size());
  EXPECT_FLOAT_EQ(out[0], 0.5f);
  EXPECT_FLOAT_EQ(out[1], -0.25f);
}

TEST_F(AtlasTest, GetNodeCentroidDequantizesInt8) {
  aeon::AtlasOptions opts;
  opts.dim = 8;
  opts.quantization_type = aeon::QUANT_INT8_SYMMETRIC;
  aeon::Atlas atlas(test_path, opts);

  std::vector<float> vec = {1.0f, -1.0f, 0.5f, -0.5f, 0.25f, 0.0f, 0.75f, -0.75f};
  uint64_t id = atlas.insert(0, vec, "Root");

  auto out = atlas.get_node_centroid(id);
  ASSERT_EQ(out.size(), vec.size());
  // INT8 symmetric quantization is lossy -- confirm approximate round-trip,
  // not bit-exact (same tolerance class as the existing quantization test
  // suite's INT8 recall checks).
  for (size_t i = 0; i < vec.size(); ++i)
    EXPECT_NEAR(out[i], vec[i], 0.02f);
}

TEST_F(AtlasTest, GetNodeCentroidRejectsInvalidNodeId) {
  aeon::Atlas atlas(test_path);
  std::vector<float> vec(768, 0.0f);
  atlas.insert(0, vec, "Root"); // id 0 only

  EXPECT_THROW(atlas.get_node_centroid(999), std::runtime_error);
}

// V4 Stage 4 task 1: list_nodes_by_scope() -- the console's list-by-scope
// primitive. Confirms scope-mask overlap filtering, tombstone exclusion,
// and that superseded (reversible) nodes ARE still included.
TEST_F(AtlasTest, ListNodesByScopeReturnsMatchingLiveNodes) {
  aeon::Atlas atlas(test_path);
  std::vector<float> vec(768, 0.0f);

  uint64_t a = atlas.insert(0, vec, "A"); // scope 0x1
  uint64_t b = atlas.insert(0, vec, "B"); // scope 0x2
  uint64_t c = atlas.insert(0, vec, "C"); // scope 0x1 | 0x2, tombstoned
  uint64_t d = atlas.insert(0, vec, "D"); // scope 0x1, superseded
  uint64_t e = atlas.insert(0, vec, "E"); // unscoped (0)

  atlas.set_node_scope(a, 0x1u);
  atlas.set_node_scope(b, 0x2u);
  atlas.set_node_scope(c, 0x1u | 0x2u);
  atlas.set_node_scope(d, 0x1u);

  // Atlas has no public tombstone-a-node-directly method (schema.hpp's
  // tombstone_node() operates on a NodeHeader&, internal to
  // consolidate_subgraph()'s Dreaming-consolidation flow) -- this is the
  // same mechanism TombstonedNodesAreDroppedByCompaction uses to produce a
  // tombstoned node for testing. consolidate_subgraph() unions its sources'
  // scope_bitmap onto the new summary node (summary_scope_union in
  // atlas.cpp) -- so the summary inherits c's 0x1|0x2 scope, appearing in
  // every filter below alongside c's still-live siblings.
  uint64_t summary_id =
      atlas.consolidate_subgraph(std::vector<uint64_t>{c}, vec, "Summary");
  atlas.supersede_node(d);

  auto scope1 = atlas.list_nodes_by_scope(0x1u);
  std::sort(scope1.begin(), scope1.end());
  // a, d, and the summary match scope 0x1 -- c itself also matches but is
  // tombstoned (excluded); d is superseded but still included (reversible,
  // still live data).
  EXPECT_EQ(scope1, (std::vector<uint64_t>{a, d, summary_id}));

  auto scope2 = atlas.list_nodes_by_scope(0x2u);
  // b and the summary match (c itself matches too but is tombstoned).
  EXPECT_EQ(scope2, (std::vector<uint64_t>{b, summary_id}));

  // Combined ORDINARY mask (0x1|0x2, NOT the ALL_SCOPES_VISIBLE sentinel --
  // see ListNodesByScopeAllScopesVisibleIncludesUnscopedNodes below for
  // that case). e is never scope-set (defaults to 0), and
  // 0 & 0x3 == 0 (falsy), so an ordinary mask correctly never matches an
  // unscoped node -- this is the normal, intended AND-overlap semantics
  // for a real (non-sentinel) query.
  auto scope_combined = atlas.list_nodes_by_scope(0x1u | 0x2u);
  std::sort(scope_combined.begin(), scope_combined.end());
  EXPECT_EQ(scope_combined, (std::vector<uint64_t>{a, b, d, summary_id}));
  for (auto id : scope_combined) {
    EXPECT_NE(id, e);
  }
}

// scope_mask == ALL_SCOPES_VISIBLE is special-cased to mean "no filtering"
// (matching navigate()'s documented semantics for the same sentinel), NOT
// treated as an ordinary mask -- an ordinary-mask AND check would exclude
// every unscoped node (0 & ALL_SCOPES_VISIBLE == 0, falsy), which is the
// inverse of what "list everything" should mean. Found via review before
// this ever shipped (v4-plan.md Stage 4).
TEST_F(AtlasTest, ListNodesByScopeAllScopesVisibleIncludesUnscopedNodes) {
  aeon::Atlas atlas(test_path);
  std::vector<float> vec(768, 0.0f);

  uint64_t scoped = atlas.insert(0, vec, "Scoped");
  uint64_t unscoped = atlas.insert(0, vec, "Unscoped");
  atlas.set_node_scope(scoped, 0x1u);
  // unscoped stays at its default scope_bitmap == 0.

  auto all = atlas.list_nodes_by_scope(aeon::ALL_SCOPES_VISIBLE);
  std::sort(all.begin(), all.end());
  EXPECT_EQ(all, (std::vector<uint64_t>{scoped, unscoped}));
}

// V4 Stage 4 task 1: bulk_set_node_scope() -- the console's bulk bit remap
// primitive. Applies N updates under one lock/WAL-flush pass.
TEST_F(AtlasTest, BulkSetNodeScopeAppliesAllUpdates) {
  aeon::Atlas atlas(test_path);
  std::vector<float> vec(768, 0.0f);
  uint64_t a = atlas.insert(0, vec, "A");
  uint64_t b = atlas.insert(0, vec, "B");
  uint64_t c = atlas.insert(0, vec, "C");

  atlas.bulk_set_node_scope({{a, 0x1u}, {b, 0x2u}, {c, 0x3u}});

  EXPECT_EQ(atlas.get_node_scope(a), 0x1u);
  EXPECT_EQ(atlas.get_node_scope(b), 0x2u);
  EXPECT_EQ(atlas.get_node_scope(c), 0x3u);
}

// All-or-nothing: an invalid id anywhere in the batch must throw WITHOUT
// applying any of the valid entries that came before it.
TEST_F(AtlasTest, BulkSetNodeScopeIsAllOrNothingOnInvalidId) {
  aeon::Atlas atlas(test_path);
  std::vector<float> vec(768, 0.0f);
  uint64_t a = atlas.insert(0, vec, "A");

  EXPECT_THROW(
      atlas.bulk_set_node_scope({{a, 0x1u}, {999u, 0x2u}}),
      std::runtime_error);

  // 'a' must be untouched -- validation happens before any mutation.
  EXPECT_EQ(atlas.get_node_scope(a), 0u);
}

TEST_F(AtlasTest, BulkSetNodeScopeRejectsDeltaArenaId) {
  aeon::Atlas atlas(test_path);
  std::vector<float> vec(768, 0.0f);
  uint64_t a = atlas.insert(0, vec, "A");
  uint64_t delta_id = atlas.insert_delta(vec, "delta node");

  EXPECT_THROW(
      atlas.bulk_set_node_scope({{a, 0x1u}, {delta_id, 0x2u}}),
      std::invalid_argument);
  EXPECT_EQ(atlas.get_node_scope(a), 0u);
}

TEST_F(AtlasTest, NavigationSimple) {
  aeon::Atlas atlas(test_path);
  std::vector<float> zero(768, 0.0f);

  // Create Root
  atlas.insert(0, zero, "Root");

  // Create Child 1 (Close to target)
  std::vector<float> child1(768, 1.0f);
  atlas.insert(0, child1, "Child 1");

  // Create Child 2 (Far from target)
  std::vector<float> child2(768, -1.0f);
  atlas.insert(0, child2, "Child 2");

  // Query
  std::vector<float> query(768, 1.0f);
  auto path = atlas.navigate(query);

  // Should be Root -> Child 1
  // Should be Root -> Child 1 (or sorted by similarity)
  // Actually navigate sorts by similarity now.
  // Child 1 (1.0f) is closer to Query (1.0f) than Root (0.0f) or Child 2
  // (-1.0f). Root might be in list too. Let's just check the best one.
  ASSERT_GE(path.size(), 1);
  EXPECT_EQ(path[0].id, 1); // Child 1 has ID 1
}

// Atlas::compact_mmap() previously had zero direct test coverage (only
// TraceManager::compact() was exercised, via test_wal.cpp). Added alongside
// the explicit msync() call in compact_mmap() (v4-plan.md guardrail #1.3)
// so that change -- and compaction generally -- has real regression
// coverage rather than being verified only by inspection.
TEST_F(AtlasCompactionTest, LiveNodesSurviveWithCorrectData) {
  aeon::Atlas atlas(test_path);
  std::vector<float> root(768, 0.0f);
  atlas.insert(0, root, "Root");
  std::vector<float> child(768, 1.0f);
  atlas.insert(0, child, "Child");

  ASSERT_EQ(atlas.size(), 2u);

  atlas.compact_mmap();

  // Data must still be present and correct after compaction (not just
  // "didn't crash" -- the new generation file's writes must have actually
  // landed, which is exactly what the new msync() call is verifying).
  EXPECT_EQ(atlas.size(), 2u);
  std::vector<float> query(768, 1.0f);
  auto path = atlas.navigate(query);
  ASSERT_GE(path.size(), 1u);
  EXPECT_NEAR(path[0].similarity, 1.0f, 1e-4f);
}

// Closes the exact gap noted in v4-plan.md's Stage 1 status: proves
// scope_bitmap survives a real insert -> set-non-zero-scope -> compact ->
// read-back cycle, now that set_node_scope() provides a real non-zero
// write path (there wasn't one when Stage 1 landed the field).
TEST_F(AtlasCompactionTest, ScopeBitmapSurvivesCompaction) {
  aeon::Atlas atlas(test_path);
  std::vector<float> root(768, 0.0f);
  uint64_t root_id = atlas.insert(0, root, "Root");
  std::vector<float> child(768, 1.0f);
  uint64_t child_id = atlas.insert(root_id, child, "Child");

  atlas.set_node_scope(root_id, 0x1u);
  atlas.set_node_scope(child_id, 0x3u);

  atlas.compact_mmap();

  // Compaction renumbers surviving nodes sequentially from 0; with no
  // tombstones here, root stays 0 and child stays 1.
  EXPECT_EQ(atlas.get_node_scope(0), 0x1u);
  EXPECT_EQ(atlas.get_node_scope(1), 0x3u);
}

// V4 Stage 4 task 1: governance_record_id survives compact_mmap() the same
// way scope_bitmap does -- compaction copies the entire node stride
// (header + centroid + metadata) byte-for-byte per live node, so this is
// automatic, not something set_node_governance_id() or compact_mmap()
// itself needs special-cased logic for. Verified rather than assumed.
TEST_F(AtlasCompactionTest, GovernanceRecordIdSurvivesCompaction) {
  aeon::Atlas atlas(test_path);
  std::vector<float> root(768, 0.0f);
  uint64_t root_id = atlas.insert(0, root, "Root");
  std::vector<float> child(768, 1.0f);
  uint64_t child_id = atlas.insert(root_id, child, "Child");

  atlas.set_node_governance_id(root_id, 0x111u);
  atlas.set_node_governance_id(child_id, 0x222u);

  atlas.compact_mmap();

  EXPECT_EQ(atlas.get_node_governance_id(0), 0x111u);
  EXPECT_EQ(atlas.get_node_governance_id(1), 0x222u);
}

// Pins a real, pre-existing (not touched this session) asymmetry in
// consolidate_subgraph() flagged during Stage 4 task 1 review: the new
// summary node inherits its sources' UNIONED scope_bitmap (so consolidated
// knowledge doesn't silently drop out of scope-filtered queries -- see
// ListNodesByScopeReturnsMatchingLiveNodes) but always gets a fresh,
// zeroed governance_record_id rather than inheriting one from any source.
// This is consistent with Stage 4's mint-not-flip promotion design
// elsewhere in v4-plan.md: a summary is genuinely NEW synthesized content,
// not a copy of any one source, so it shouldn't arbitrarily inherit one
// source's governance/control-plane record among several candidates.
// Pinned with a test (not just documented) since the console's audit-log/
// knowledge-browser work will build directly on this asymmetry.
TEST_F(AtlasTest, ConsolidateSubgraphUnionsScopeButZeroesGovernanceId) {
  aeon::Atlas atlas(test_path);
  std::vector<float> root(768, 0.0f);
  uint64_t root_id = atlas.insert(0, root, "Root");
  std::vector<float> stale(768, 0.5f);
  uint64_t stale_id = atlas.insert(root_id, stale, "Stale");

  atlas.set_node_scope(stale_id, 0x1u);
  atlas.set_node_governance_id(stale_id, 0xABCu);

  std::vector<float> summary_vec(768, 0.5f);
  uint64_t summary_id = atlas.consolidate_subgraph(
      std::vector<uint64_t>{stale_id}, summary_vec, "Summary");

  EXPECT_EQ(atlas.get_node_scope(summary_id), 0x1u); // inherited
  EXPECT_EQ(atlas.get_node_governance_id(summary_id), 0u); // NOT inherited
}

// V4 Stage 5 task 1 (advisor review, before any Python clustering code was
// written): a plain scope-union alone can WIDEN visibility -- consolidating
// a scope-0x1 node with a scope-0x2 node would mint a scope-0x3 summary
// readable by both, as a silent side effect of a storage-GC operation nobody
// asked for. Rejected up front (Phase 1, before any mutation) rather than
// left to the Python-layer clustering logic to get right on its own.
TEST_F(AtlasTest, ConsolidateSubgraphRejectsMixedScopeInput) {
  aeon::Atlas atlas(test_path);
  std::vector<float> root(768, 0.0f);
  uint64_t root_id = atlas.insert(0, root, "Root");
  std::vector<float> a(768, 0.1f);
  uint64_t a_id = atlas.insert(root_id, a, "A");
  std::vector<float> b(768, 0.2f);
  uint64_t b_id = atlas.insert(root_id, b, "B");

  atlas.set_node_scope(a_id, 0x1u);
  atlas.set_node_scope(b_id, 0x2u);

  std::vector<float> summary_vec(768, 0.15f);
  EXPECT_THROW(
      atlas.consolidate_subgraph(std::vector<uint64_t>{a_id, b_id},
                                  summary_vec, "Summary"),
      std::invalid_argument);

  // Rejected BEFORE any mutation: no summary node was minted, and both
  // sources remain live and unchanged.
  EXPECT_EQ(atlas.size(), 3u); // root, a, b -- no summary appended
  EXPECT_EQ(atlas.get_node_scope(a_id), 0x1u);
  EXPECT_EQ(atlas.get_node_scope(b_id), 0x2u);
}

// Same-scope input must still succeed (the precondition rejects DIFFERING
// scopes, not scoped input generally) -- a regression guard so a future
// tightening of the check doesn't accidentally reject the common case.
TEST_F(AtlasTest, ConsolidateSubgraphAllowsUniformScopeInput) {
  aeon::Atlas atlas(test_path);
  std::vector<float> root(768, 0.0f);
  uint64_t root_id = atlas.insert(0, root, "Root");
  std::vector<float> a(768, 0.1f);
  uint64_t a_id = atlas.insert(root_id, a, "A");
  std::vector<float> b(768, 0.2f);
  uint64_t b_id = atlas.insert(root_id, b, "B");

  atlas.set_node_scope(a_id, 0x4u);
  atlas.set_node_scope(b_id, 0x4u);

  std::vector<float> summary_vec(768, 0.15f);
  uint64_t summary_id = atlas.consolidate_subgraph(
      std::vector<uint64_t>{a_id, b_id}, summary_vec, "Summary");

  EXPECT_EQ(atlas.get_node_scope(summary_id), 0x4u);
}

// A real, separate gap found while proving the V4 Stage 2 scope-recall
// follow-up's discrimination (v4-plan.md): consolidate_subgraph() set the
// new summary's OWN parent_offset (Phase 2) but never updated that
// PARENT's child_count/first_child_offset to enumerate the summary back --
// confirmed empirically (a scratch root->child->consolidate_subgraph check)
// that the resulting summary was NEVER returned by navigate() at all,
// independent of scope filtering. This directly undermines Dreaming's
// whole purpose (Stage 5's plan: consolidate "fourteen tickets" into one
// summary, tombstoning the originals -- if the summary can never be found
// again, that's a net RETRIEVAL REGRESSION, not neutral). Fixed with
// EXACTLY insert()'s own existing contiguity check: the summary always
// lands at the current tail (header->node_count), so it registers
// correctly whenever nothing else was inserted under a DIFFERENT parent
// since the parent's last child -- true for every real caller today
// (every production Atlas.insert() uses parent_id=0, confirmed by grep
// across shell/aeon_py/*.py) and for the common Dreaming case
// (consolidating LEAF fragments, so Phase 3 has nothing to rewire).
TEST_F(AtlasTest, ConsolidateSubgraphSummaryIsReachableViaNavigateWhenContiguous) {
  aeon::Atlas atlas(test_path);
  std::vector<float> root(768, 0.0f);
  uint64_t root_id = atlas.insert(0, root, "Root");
  std::vector<float> leaf(768, 1.0f);
  uint64_t leaf_id = atlas.insert(root_id, leaf, "Leaf");

  uint64_t summary_id = atlas.consolidate_subgraph(
      std::vector<uint64_t>{leaf_id}, leaf, "Summary");

  // apply_csls=true so the tombstoned leaf's branchless exclusion
  // (hub_penalty = TOMBSTONE_PENALTY) actually applies -- with
  // apply_csls=false it would score normally and the query could
  // (mis)report a "hit" via the OLD tombstoned node rather than genuinely
  // exercising whether the NEW summary is reachable.
  auto results = atlas.navigate(leaf, /*beam_width=*/4, /*apply_csls=*/true);
  bool found_summary = false;
  bool found_tombstoned_leaf = false;
  for (const auto &r : results) {
    if (r.id == summary_id) found_summary = true;
    if (r.id == leaf_id) found_tombstoned_leaf = true;
  }
  EXPECT_TRUE(found_summary)
      << "summary_id=" << summary_id << " not returned by navigate() -- "
         "the parent (root) never learned to enumerate it as a child";
  EXPECT_FALSE(found_tombstoned_leaf)
      << "tombstoned original leaf should not itself be reported";
}

// The narrower residual this fix does NOT solve, pinned so it stays a
// documented, deliberate limitation rather than an assumption: if the
// summary's placement (always the current tail) does NOT land physically
// contiguous with its parent's existing children -- because something else
// was inserted under a DIFFERENT parent in between -- the parent's
// child_count is correctly left UNCHANGED (exactly insert()'s own existing
// behavior for the identical non-contiguous scenario), not incorrectly
// incremented in a way that would make navigate() read past the real
// children into unrelated node bytes.
TEST_F(AtlasTest, ConsolidateSubgraphSummaryNotRegisteredWhenNonContiguousButNoCorruption) {
  aeon::Atlas atlas(test_path);
  std::vector<float> root(768, 0.0f);
  uint64_t root_id = atlas.insert(0, root, "Root");
  std::vector<float> leaf(768, 1.0f);
  uint64_t leaf_id = atlas.insert(root_id, leaf, "Leaf");

  // Insert an unrelated node under a DIFFERENT parent (leaf_id itself) --
  // this lands physically between leaf_id and the summary the upcoming
  // consolidate_subgraph() call will create, breaking contiguity with
  // root's existing child block.
  std::vector<float> other(768, 0.3f);
  atlas.insert(leaf_id, other, "Other"); // NOT a child of root

  uint64_t root_child_count_before =
      atlas.get_children(root_id).size(); // 1 (leaf_id) -- root's OWN
                                          // enumerated children, unaffected
                                          // by the grandchild just inserted
  uint64_t summary_id = atlas.consolidate_subgraph(
      std::vector<uint64_t>{leaf_id}, leaf, "Summary");

  auto results = atlas.navigate(leaf, /*beam_width=*/4, /*apply_csls=*/true);
  bool found_summary = false;
  for (const auto &r : results) {
    if (r.id == summary_id) found_summary = true;
  }
  // NOTE: this EXPECT_FALSE is NOT the assertion that actually
  // discriminates the contiguity gate -- it holds true both with the
  // gate enabled (correctly not registered) AND with registration
  // temporarily forced unconditionally (verified via a discrimination
  // check, v4-plan.md), because in the forced-registration case
  // root->child_count is bumped but root->first_child_offset still
  // points at leaf_id's existing block, so the "extra child" navigate()
  // would scan is actually the unrelated `Other` node sitting at that
  // address, not the summary (summary sits one slot further and is
  // still never reached). Kept as a same-symptom sanity check, not
  // deleted, since it IS true and worth asserting regardless.
  EXPECT_FALSE(found_summary)
      << "non-contiguous placement should leave the summary unregistered, "
         "not corrupt root's child enumeration by pretending otherwise";
  // THIS is the assertion that actually discriminates the contiguity
  // gate: forcing registration unconditionally makes root->child_count
  // increment despite nothing new actually landing contiguously with
  // root's existing children, so get_children(root_id) returns the
  // unrelated `Other` node as a phantom second child of root -- a real
  // structural corruption this check catches even though found_summary
  // above does not.
  EXPECT_EQ(atlas.get_children(root_id).size(), root_child_count_before);
}

// Supersession is deliberately reversible (unlike tombstoning) -- a
// superseded node must survive compact_mmap() physically, not be dropped
// like a tombstoned one, since revoke_node_supersede() may still be called
// on it later (e.g. Article-16-equivalent correction flows, Stage 4).
TEST_F(AtlasCompactionTest, SupersededNodesSurviveCompactionUnlikeTombstoned) {
  aeon::Atlas atlas(test_path);
  std::vector<float> root(768, 0.0f);
  uint64_t root_id = atlas.insert(0, root, "Root");
  std::vector<float> child(768, 1.0f);
  uint64_t child_id = atlas.insert(root_id, child, "Child");

  atlas.supersede_node(child_id);
  size_t size_before = atlas.size();

  atlas.compact_mmap();

  EXPECT_EQ(atlas.size(), size_before); // nothing dropped
  // Renumbered but present: root=0, child=1 (no tombstones to skip).
  EXPECT_TRUE(atlas.is_node_superseded(1));
}

// Regression test for a severe data-loss bug found while wiring V4 Stage
// 2's TraceManager semantic search (v4-plan.md): compact_mmap() used to
// install the compacted generation under a PERMANENTLY generation-
// suffixed name (atlas_gen1.bin, ...) and delete the file at atlas_path_.
// Since atlas_path_/generation_ are only tracked in-memory and reset on
// every fresh Atlas construction, a process restart using the same
// caller-configured path (the normal case -- see dependencies.py's
// AEON_ATLAS_PATH) would find that path gone and silently create an
// empty Atlas -- total, silent loss of every long-term memory node.
//
// This is the actual bug scenario: insert, compact, CLOSE the Atlas
// entirely (simulating a process exit), then reopen via the exact same
// caller-configured path (simulating a restart) and confirm the data is
// still there.
TEST_F(AtlasCompactionTest, DataSurvivesCompactionAcrossFullRestart) {
  {
    aeon::Atlas atlas(test_path);
    std::vector<float> root(768, 0.0f);
    uint64_t root_id = atlas.insert(0, root, "Root");
    std::vector<float> child(768, 1.0f);
    atlas.insert(root_id, child, "Child");

    atlas.compact_mmap();

    // Post-compaction insert too, to prove writes after compaction also
    // land correctly under the (now-stable) path.
    std::vector<float> after(768, 0.5f);
    atlas.insert(root_id, after, "AfterCompaction");
  } // Atlas destructor runs here -- full process-exit simulation

  ASSERT_TRUE(std::filesystem::exists(test_path))
      << "the caller's configured path must still exist after a restart "
        "following compaction -- this is exactly the bug that was fixed";

  aeon::Atlas reopened(test_path);
  EXPECT_EQ(reopened.size(), 3u)
      << "Root, Child, and the post-compaction AfterCompaction node must "
        "all survive a full restart";
}

TEST_F(AtlasCompactionTest, TombstonedNodesAreDroppedByCompaction) {
  aeon::Atlas atlas(test_path);
  std::vector<float> root(768, 0.0f);
  uint64_t root_id = atlas.insert(0, root, "Root");
  std::vector<float> stale(768, 0.5f);
  uint64_t stale_id = atlas.insert(root_id, stale, "Stale");

  std::vector<uint64_t> old_ids = {stale_id};
  std::vector<float> summary(768, 0.5f);
  atlas.consolidate_subgraph(old_ids, summary, "Summary");

  EXPECT_GT(atlas.tombstone_count(), 0u);
  size_t size_before = atlas.size();

  atlas.compact_mmap();

  // Tombstoned nodes are physically dropped by compaction -- the live set
  // shrinks and there are no tombstones left to count in the new
  // generation file.
  EXPECT_LT(atlas.size(), size_before);
  EXPECT_EQ(atlas.tombstone_count(), 0u);
}

TEST_F(AtlasCompactionTest, RepeatedCompactionIsSafe) {
  aeon::Atlas atlas(test_path);
  std::vector<float> root(768, 0.0f);
  atlas.insert(0, root, "Root");

  // Compacting an already-compacted (and otherwise idle) Atlas must not
  // crash, hang, or lose data -- exercises the new-generation-file naming
  // and the msync-before-delete ordering across multiple generations.
  atlas.compact_mmap();
  atlas.compact_mmap();
  atlas.compact_mmap();

  EXPECT_EQ(atlas.size(), 1u);
}

// v4-plan.md Stage 4 task 6 Phase B: AtlasOptions::metadata_size looked
// generalized (storage::MemoryFile::open()/compute_node_stride() both
// take/generalize over it, AtlasOptions already exists for exactly this
// kind of per-file customization) but was NEVER actually exercised at a
// non-default value -- Atlas::Atlas(path, opts) hardcoded
// METADATA_SIZE_DEFAULT at its one file_->open() call site, ignoring
// opts.metadata_size entirely. The same "parameterized-in-theory,
// hardcoded-in-practice" shape as the hardcoded-768-dim bugs guardrail
// #1.1 found. These tests prove the fix actually works, not just that it
// compiles.

TEST_F(AtlasTest, MetadataSizeDefaultsTo256WhenUnset) {
  aeon::Atlas atlas(test_path);
  EXPECT_EQ(atlas.metadata_size(), aeon::METADATA_SIZE_DEFAULT);
}

TEST_F(AtlasTest, MetadataSizeOptionRoundTrips) {
  aeon::AtlasOptions opts;
  opts.metadata_size = 512;
  aeon::Atlas atlas(test_path, opts);
  ASSERT_EQ(atlas.metadata_size(), 512u);

  // A string that would have been silently truncated at the default
  // 256-byte field (metadata_size - 1 = 255 usable bytes) but fits
  // comfortably within 512.
  std::string long_text(400, 'x');
  std::vector<float> vec(768, 0.0f);
  uint64_t id = atlas.insert(0, vec, long_text);

  EXPECT_EQ(atlas.get_node_metadata(id), long_text);
}

TEST_F(AtlasTest, MetadataSizeDefaultStillTruncatesAt255Bytes) {
  // Regression guard: confirms the default behavior (pre-existing, lossy,
  // accepted -- dreamer.py's own 250-char convention relies on it) is
  // unchanged by adding the metadata_size option.
  aeon::Atlas atlas(test_path);
  std::string long_text(300, 'y');
  std::vector<float> vec(768, 0.0f);
  uint64_t id = atlas.insert(0, vec, long_text);

  std::string stored = atlas.get_node_metadata(id);
  EXPECT_EQ(stored.size(), aeon::METADATA_SIZE_DEFAULT - 1);
  EXPECT_EQ(stored, long_text.substr(0, aeon::METADATA_SIZE_DEFAULT - 1));
}

TEST_F(AtlasCompactionTest, MetadataSizeOptionSurvivesCompaction) {
  aeon::AtlasOptions opts;
  opts.metadata_size = 512;
  aeon::Atlas atlas(test_path, opts);

  std::string long_text(400, 'z');
  std::vector<float> vec(768, 0.0f);
  uint64_t id = atlas.insert(0, vec, long_text);

  atlas.compact_mmap();

  EXPECT_EQ(atlas.metadata_size(), 512u);
  EXPECT_EQ(atlas.get_node_metadata(id), long_text);
}
