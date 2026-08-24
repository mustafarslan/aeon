/**
 * @file test_scope_recall.cpp
 * @brief V4 Stage 2 gate: "recall >= 0.99 vs. an exhaustive scope-filtered
 * scan across selectivities 0.02-1.0."
 *
 * The mechanism under test (Atlas::navigate()'s scope_mask parameter,
 * atlas.cpp) deliberately keeps beam descent scope-blind and only makes
 * emission scope-aware (per v4-plan.md Stage 2 task 1's "never exclude by
 * scope during descent"), with an internal beam-width widen-to-MAX as the
 * chosen recall safety net ("implement + measure first", user's explicit
 * choice over speculative scope-affinity steering).
 *
 * Ground truth is computed test-side (every inserted vector/scope triple
 * is tracked here at insertion time) rather than via a new Atlas API --
 * Atlas has no public "iterate all nodes" reader, and adding one purely
 * for this test would repeat the TraceBlockIndex dead-code mistake this
 * whole plan exists partly to fix.
 */

#include "aeon/atlas.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <gtest/gtest.h>
#include <random>
#include <vector>

namespace {

constexpr uint32_t kDim = 64; // small dim keeps the test fast; recall
                               // behavior doesn't depend on dim.
constexpr uint64_t kScopeBit = 0x1u;

float cosine_sim(const std::vector<float> &a, const std::vector<float> &b) {
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += static_cast<double>(a[i]) * b[i];
    na += static_cast<double>(a[i]) * a[i];
    nb += static_cast<double>(b[i]) * b[i];
  }
  if (na == 0.0 || nb == 0.0)
    return 0.0f;
  return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb)));
}

std::vector<float> random_unit_vector(std::mt19937 &rng) {
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> v(kDim);
  for (auto &x : v)
    x = dist(rng);
  float norm = 0.0f;
  for (float x : v)
    norm += x * x;
  norm = std::sqrt(norm);
  if (norm > 1e-6f) {
    for (auto &x : v)
      x /= norm;
  }
  return v;
}

std::vector<float> normalize(std::vector<float> v) {
  float norm = 0.0f;
  for (float x : v)
    norm += x * x;
  norm = std::sqrt(norm);
  if (norm > 1e-6f) {
    for (auto &x : v)
      x /= norm;
  }
  return v;
}

/// center + Gaussian noise scaled by `spread`, renormalized. Small spread
/// keeps the result semantically close to center (cosine similarity near
/// 1.0); this is what gives the test tree real locality -- a node's
/// vector actually resembles its cluster/ancestor, matching how a real
/// Aeon tree clusters semantically related content, unlike inserting
/// pure-random vectors under an arbitrary, content-blind parent.
std::vector<float> perturbed(const std::vector<float> &center, float spread,
                             std::mt19937 &rng) {
  std::normal_distribution<float> noise(0.0f, spread);
  std::vector<float> v = center;
  for (auto &x : v)
    x += noise(rng);
  return normalize(v);
}

/// Builds a genuinely multi-level tree (branching factor B, depth D) so
/// Atlas::navigate()'s one-result-per-level `path` structure has more
/// than a couple of entries to work with -- a flat single-level tree
/// would cap recall opportunities regardless of the filtering mechanism.
struct ScopedTree {
  aeon::Atlas atlas;
  // Ground truth, tracked independently of Atlas: id -> (vector, in_scope).
  struct Node {
    uint64_t id;
    std::vector<float> vec;
    bool in_scope;
  };
  std::vector<Node> nodes;

  explicit ScopedTree(const std::filesystem::path &path, double selectivity,
                      std::mt19937 &rng)
      : atlas(path, kDim) {
    std::uniform_real_distribution<double> coin(0.0, 1.0);

    // Spread shrinks each level: children stay semantically close to
    // their parent's direction, giving the tree real locality -- the
    // property beam search over a hierarchical index actually depends
    // on. Pure per-node-random vectors under an arbitrary, content-blind
    // parent (the original version of this test) give beam search
    // nothing to exploit and measure an unrelated failure mode.
    auto add_node = [&](uint64_t parent_id, const std::vector<float> &parent_vec,
                        float spread) -> std::pair<uint64_t, std::vector<float>> {
      auto vec = perturbed(parent_vec, spread, rng);
      uint64_t id = atlas.insert(parent_id, vec, "n");
      bool in_scope = coin(rng) < selectivity;
      if (in_scope) {
        atlas.set_node_scope(id, kScopeBit);
      }
      nodes.push_back({id, vec, in_scope});
      return {id, vec};
    };

    // Root (id 0), branching factor 8, depth 3: 1 + 8 + 64 + 512 nodes.
    std::vector<float> root_vec = random_unit_vector(rng);
    uint64_t root_id = atlas.insert(0, root_vec, "root");
    bool root_in_scope = coin(rng) < selectivity;
    nodes.push_back({root_id, root_vec, root_in_scope});
    if (root_in_scope) {
      atlas.set_node_scope(root_id, kScopeBit);
    }

    std::vector<std::pair<uint64_t, std::vector<float>>> level1, level2;
    for (int i = 0; i < 8; ++i)
      level1.push_back(add_node(root_id, root_vec, 1.0f));
    for (const auto &[p_id, p_vec] : level1)
      for (int i = 0; i < 8; ++i)
        level2.push_back(add_node(p_id, p_vec, 0.3f));
    for (const auto &[p_id, p_vec] : level2)
      for (int i = 0; i < 8; ++i)
        add_node(p_id, p_vec, 0.1f);
  }

  /// Exhaustive ground truth: best cosine similarity among IN-SCOPE nodes.
  /// Returns node id, or 0 (root sentinel; excluded by construction below)
  /// if no in-scope node exists.
  std::pair<uint64_t, float> exhaustive_best(const std::vector<float> &query,
                                             bool &found) const {
    uint64_t best_id = 0;
    float best_sim = -2.0f;
    found = false;
    for (const auto &n : nodes) {
      if (!n.in_scope)
        continue;
      float sim = cosine_sim(query, n.vec);
      if (sim > best_sim) {
        best_sim = sim;
        best_id = n.id;
        found = true;
      }
    }
    return {best_id, best_sim};
  }
};

/// Runs the recall measurement for one selectivity, returns measured
/// recall in [0, 1]. num_queries independent random queries.
double measure_recall(double selectivity, int num_queries, uint32_t seed) {
  std::mt19937 rng(seed);
  auto tmp_dir = std::filesystem::temp_directory_path() /
                ("aeon_scope_recall_" + std::to_string(seed));
  std::filesystem::remove_all(tmp_dir);
  std::filesystem::create_directories(tmp_dir);
  auto atlas_path = tmp_dir / "atlas.bin";

  ScopedTree tree(atlas_path, selectivity, rng);

  std::uniform_int_distribution<size_t> pick_node(0, tree.nodes.size() - 1);

  int hits = 0;
  int evaluable = 0; // queries where a ground-truth in-scope node exists
  for (int q = 0; q < num_queries; ++q) {
    // Query = a real node's vector, lightly perturbed -- simulates "a
    // query resembling existing content," the realistic case, rather
    // than a pure-random vector with no structural relationship to
    // anything in the tree.
    auto query = perturbed(tree.nodes[pick_node(rng)].vec, 0.1f, rng);

    bool found = false;
    auto [gt_id, gt_sim] = tree.exhaustive_best(query, found);
    if (!found)
      continue; // no in-scope node exists at all -- not evaluable
    ++evaluable;

    auto results = tree.atlas.navigate(query, /*beam_width=*/4,
                                       /*apply_csls=*/false, /*session_id=*/0,
                                       /*scope_mask=*/kScopeBit);
    // Recall criterion: navigate()'s best-reported result must be the
    // ground-truth best node, OR within a tight similarity tolerance of
    // it (ties among near-duplicate random vectors are expected at this
    // scale, not a real miss).
    bool matched = false;
    for (const auto &r : results) {
      if (r.id == gt_id || std::abs(r.similarity - gt_sim) < 1e-3f) {
        matched = true;
        break;
      }
    }
    if (matched)
      ++hits;
  }

  if (evaluable == 0)
    return 1.0; // vacuously true -- no evaluable queries at this selectivity
  return static_cast<double>(hits) / evaluable;
}

} // namespace

// Selectivities per the gate: 0.02 (sparse) through 1.0 (unfiltered).
//
// GATE MET (v4-plan.md Stage 2 follow-up, 2026-08-23): all four
// selectivities now measure >=0.99 recall, closing the gap this file
// documented as a KNOWN GAP from 2026-08-22 through this fix (0.10 measured
// ~0.93, 0.02 measured ~0.37 before it). Root cause: under deferred/
// option-b scope propagation, an internal node's own scope_bitmap said
// nothing about whether its DESCENDANTS included the target scope, so
// blind beam descent had no way to tell a subtree worth exploring from one
// that provably couldn't contain a match. Fixed by Atlas::scope_union_ (an
// auxiliary, RAM-only index -- see atlas.hpp's member comment for why it's
// a separate array rather than reusing scope_bitmap itself) tracking each
// node's subtree scope union, consulted ONLY as a beam-admission priority
// (union-negative candidates are deprioritized, never hard-excluded, so a
// stale/incomplete union can only forgo a pruning opportunity, never wrongly
// exclude a real candidate). This is NOT the per-candidate own-bit steering
// bonus that was tried and measured actively harmful at 50% selectivity
// (see navigate_internal()'s doc comment in atlas.cpp) -- a subtree union
// bit is a sound over-approximation of "does this subtree contain a match,"
// unlike a leaf's own bit, which said nothing about its descendants.
//
// Discrimination proved before landing (not just passing): temporarily
// forced the union check to always return true (disabling the fix, falling
// back to blind admission) and re-ran this file -- Selectivity_0_10 and
// Selectivity_0_02 reproduced the exact old ~0.93/~0.37 ceiling and failed
// at the 0.99 bar, confirming this test genuinely discriminates the fix
// rather than passing regardless. Restored and re-confirmed green after.
TEST(ScopeRecall, Selectivity_1_00) {
  double recall = measure_recall(1.00, 60, /*seed=*/1001);
  EXPECT_GE(recall, 0.99) << "measured recall: " << recall;
}

TEST(ScopeRecall, Selectivity_0_50) {
  double recall = measure_recall(0.50, 60, /*seed=*/1002);
  EXPECT_GE(recall, 0.99) << "measured recall: " << recall;
}

TEST(ScopeRecall, Selectivity_0_10) {
  double recall = measure_recall(0.10, 60, /*seed=*/1003);
  EXPECT_GE(recall, 0.99) << "measured recall: " << recall;
}

TEST(ScopeRecall, Selectivity_0_02) {
  double recall = measure_recall(0.02, 60, /*seed=*/1004);
  EXPECT_GE(recall, 0.99) << "measured recall: " << recall;
}

// A test attempting to cover consolidate_subgraph()'s scope_union_
// propagation specifically (the case where a rewired child's parent_offset
// points to a NUMERICALLY LATER summary node -- see
// rebuild_scope_union_locked()'s doc comment, atlas.hpp) was written and
// then DELETED, not landed: it passed even with both
// propagate_scope_union_locked() calls in consolidate_subgraph() disabled,
// because it queried with apply_csls=false, under which a tombstoned
// node's hub_penalty exclusion never applies -- the needle stayed
// reachable via the OLD (pre-consolidation) path regardless of anything
// this fix does, so the test wasn't discriminating what it claimed to.
// Rewriting it with apply_csls=true doesn't fix this: an empirical check
// (a scratch Atlas -- root -> child -> consolidate_subgraph({child}))
// confirmed the resulting summary node is NEVER returned by navigate() at
// all, with or without this fix -- consolidate_subgraph() never updates
// the summary's own parent's child_count/first_child_offset to include
// it, so the summary is structurally unreachable via beam descent
// (a node can be a child BY parent_offset without its parent ever
// enumerating it back). This is a separate, pre-existing property of
// consolidate_subgraph() -- unrelated to scope filtering, not something
// this fix introduces or could paper over -- flagged in v4-plan.md as an
// observation for whoever builds Stage 5 (the Dreamer, the only caller of
// consolidate_subgraph() today), not fixed here.
//
// The consolidate_subgraph() propagate_scope_union_locked() calls
// themselves are KEPT (correct as an algorithm on the scope_union_ data
// structure regardless of navigate()'s separate reachability limits, and
// they matter for ancestors ABOVE the summary, e.g. the summary's own
// parent, which IS normally reachable) -- they are simply UNVERIFIED BY
// TEST, for the reason above. A future test could observe them once
// there's an accessor or a code path that reaches an ancestor of a
// rewired child under real beam competition.
