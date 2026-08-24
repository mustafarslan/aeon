/**
 * @file test_schema.cpp
 * @brief Direct unit tests for schema.hpp's pure struct-level helpers.
 *
 * V4 Stage 1: supersede_node()/revoke_supersede() are the branchless,
 * reversible counterpart to tombstone_node() -- these tests exercise them
 * directly on a stack NodeHeader rather than through a full Atlas, since
 * the behavior being verified (hub_penalty stash/restore, idempotency) is
 * entirely local to the struct.
 */

#include "aeon/schema.hpp"

#include <gtest/gtest.h>

using namespace aeon;

TEST(SupersedeNode, StashesAndOverwritesHubPenaltyBranchlessly) {
  NodeHeader n{};
  n.hub_penalty = 0.42f;

  EXPECT_FALSE(is_superseded(n));
  supersede_node(n);

  EXPECT_TRUE(is_superseded(n));
  EXPECT_FLOAT_EQ(n.hub_penalty, TOMBSTONE_PENALTY);
  EXPECT_FLOAT_EQ(n.saved_hub_penalty, 0.42f);
}

TEST(SupersedeNode, RevokeRestoresOriginalHubPenaltyAndClearsFlag) {
  NodeHeader n{};
  n.hub_penalty = -0.15f;

  supersede_node(n);
  revoke_supersede(n);

  EXPECT_FALSE(is_superseded(n));
  EXPECT_FLOAT_EQ(n.hub_penalty, -0.15f);
}

TEST(SupersedeNode, DoubleSupersedeDoesNotClobberStashedValue) {
  // Regression test for the exact bug the advisor flagged: calling
  // supersede_node() twice must not re-stash TOMBSTONE_PENALTY as the
  // "real" hub_penalty, which would permanently poison the node's score
  // once revoked.
  NodeHeader n{};
  n.hub_penalty = 1.23f;

  supersede_node(n);
  supersede_node(n); // second call must be a no-op
  revoke_supersede(n);

  EXPECT_FALSE(is_superseded(n));
  EXPECT_FLOAT_EQ(n.hub_penalty, 1.23f);
}

TEST(SupersedeNode, RevokeWithoutSupersedeIsNoOp) {
  NodeHeader n{};
  n.hub_penalty = 7.0f;

  revoke_supersede(n); // never superseded

  EXPECT_FALSE(is_superseded(n));
  EXPECT_FLOAT_EQ(n.hub_penalty, 7.0f);
}

TEST(SupersedeNode, TombstoneAndSupersedeFlagBitsAreIndependent) {
  // NODE_FLAG_TOMBSTONE (bit 0) and NODE_FLAG_SUPERSEDED (bit 2) are
  // distinct bits -- setting one must not disturb the other.
  NodeHeader n{};
  n.hub_penalty = 5.0f;

  tombstone_node(n);
  EXPECT_TRUE(is_tombstoned(n));
  EXPECT_FALSE(is_superseded(n));
  EXPECT_FLOAT_EQ(n.hub_penalty, TOMBSTONE_PENALTY);

  supersede_node(n);
  EXPECT_TRUE(is_tombstoned(n)); // still tombstoned
  EXPECT_TRUE(is_superseded(n));
}

TEST(SupersedeNode, RevokeAfterTombstoneLeavesHubPenaltyAtTombstonePenalty) {
  // A node superseded and THEN tombstoned (e.g. consolidate_subgraph() ran
  // on it, as Stage 5's Dreaming is specified to do) must stay branchlessly
  // excluded from beam search even after its supersession is revoked --
  // tombstoning is terminal, revoking a stale supersession must not
  // resurrect a real hub_penalty on a dead node.
  NodeHeader n{};
  n.hub_penalty = 0.75f;

  supersede_node(n);
  EXPECT_FLOAT_EQ(n.saved_hub_penalty, 0.75f);

  tombstone_node(n); // node is now BOTH superseded and tombstoned
  ASSERT_TRUE(is_tombstoned(n));
  ASSERT_TRUE(is_superseded(n));

  revoke_supersede(n);

  EXPECT_FALSE(is_superseded(n));
  EXPECT_TRUE(is_tombstoned(n)); // unaffected -- tombstone is terminal
  EXPECT_FLOAT_EQ(n.hub_penalty, TOMBSTONE_PENALTY); // NOT restored to 0.75f
}

TEST(NodeHeaderScope, DefaultZeroInitializedFieldsAreZero) {
  NodeHeader n{};
  EXPECT_EQ(n.scope_bitmap, 0u);
  EXPECT_EQ(n.governance_record_id, 0u);
  EXPECT_FLOAT_EQ(n.saved_hub_penalty, 0.0f);
}

TEST(TraceEventEdgeFields, DefaultZeroInitializedFieldsAreZero) {
  TraceEvent ev{};
  EXPECT_EQ(ev.edge_type, 0u);
  EXPECT_EQ(ev.reason_code, 0u);
  EXPECT_EQ(ev.supersedes_id, 0u);
  EXPECT_EQ(ev.evidence_blob_offset, 0u);
  EXPECT_EQ(ev.evidence_blob_size, 0u);
}
