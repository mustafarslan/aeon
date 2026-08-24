"""
V4 Stage 5 task 1: shared-tier Dreaming (consolidate_shared_scope(), dreamer.py).
"""
import numpy as np
import pytest

from aeon_py.client import AeonClient, encode_store_id
from aeon_py.dreamer import consolidate_shared_scope
from aeon_py.trace import TraceGraph


def _vec(seed: float) -> list:
    return np.full(768, seed, dtype=np.float32).tolist()


class _ConsolidateSpy:
    """Wraps a real Atlas instance to record every consolidate_subgraph()
    call's node-id argument, delegating everything else unchanged.
    Necessary because the underlying nanobind Atlas object rejects
    attribute assignment ('...attribute is read-only'), so a direct
    monkeypatch of atlas.consolidate_subgraph isn't possible -- this is
    the smallest wrapper that still lets a test observe WHICH candidates
    were actually attempted, not just the final outcome. That distinction
    matters here specifically: the kernel's own scope-uniformity
    precondition (consolidate_subgraph(), V4 Stage 5 task 1) silently
    catches and the calling code logs+skips a mixed-scope attempt, so
    asserting on `reports` alone cannot tell "the Python-layer filter
    correctly excluded this candidate" apart from "the filter was broken,
    but the kernel's defense-in-depth caught the resulting bad call
    anyway" -- both produce the same empty `reports` list.
    """

    def __init__(self, real_atlas):
        self._real = real_atlas
        self.calls = []

    def __getattr__(self, name):
        return getattr(self._real, name)

    def consolidate_subgraph(self, old_node_ids, *args, **kwargs):
        self.calls.append(list(old_node_ids))
        return self._real.consolidate_subgraph(old_node_ids, *args, **kwargs)


def _orthogonal_vec(axis: int) -> list:
    """A one-hot-style vector distinct in DIRECTION, not just magnitude --
    _vec()'s uniform [c, c, ..., c] vectors are all scalar multiples of
    each other (cosine similarity 1.0 regardless of c), so they can't
    exercise the "dissimilar, don't cluster" path at all."""
    v = np.zeros(768, dtype=np.float32)
    v[axis] = 1.0
    return v.tolist()


@pytest.fixture
def shared_atlas(tmp_path):
    client = AeonClient(tmp_path / "shared.atlas")
    # Seed an explicit, unscoped root node first (matching every other test
    # file's convention, e.g. test_atlas.cpp's "Root" node) -- a real,
    # significant finding from this stage's verification pass: Atlas::
    # insert()'s parent-linking is a no-op for the very FIRST node ever
    # inserted into a fresh file (new_idx==0 skips the link-to-parent block
    # entirely), so without this, whichever node a test inserted FIRST would
    # itself become the tree's root and silently accumulate every
    # subsequent parent_id=0 insert as ITS OWN child -- exactly the
    # "non-leaf" case consolidate_shared_scope()'s new candidate filter
    # (Stage 5 task 1 verification, dreamer.py) is designed to exclude.
    # Confirmed empirically (not a bug -- promote_fragment()'s real-world
    # `insert(0, ...)` calls behave identically: the first-ever promoted
    # fragment into any shared Atlas becomes that atlas's root and
    # legitimately accumulates every later promotion as its child,
    # correctly reachable via navigate()). Left un-fixed here since it's
    # this fixture's own tests' n1/n2/etc. that must be ordinary,
    # childless candidates -- not the mechanism under test.
    client.atlas.insert(0, np.zeros(768, dtype=np.float32).tolist(), "root")
    return client


class TestConsolidateSharedScope:
    def test_clusters_similar_nodes_within_scope(self, shared_atlas):
        atlas = shared_atlas.atlas
        n1 = atlas.insert(0, _vec(0.5), "ticket A")
        n2 = atlas.insert(0, _vec(0.5001), "ticket B")  # near-identical vector
        atlas.set_node_scope(n1, 0x1)
        atlas.set_node_scope(n2, 0x1)

        reports = consolidate_shared_scope(
            atlas, scope=0x1, subject_id_of=lambda nid: "subject1",
        )

        assert len(reports) == 1
        assert reports[0].nodes_consolidated == 2
        assert atlas.tombstone_count() == 2  # sources tombstoned, not just superseded

    def test_dissimilar_nodes_not_clustered_together(self, shared_atlas):
        atlas = shared_atlas.atlas
        n1 = atlas.insert(0, _orthogonal_vec(0), "ticket A")
        n2 = atlas.insert(0, _orthogonal_vec(1), "ticket B")  # orthogonal, similarity 0.0
        atlas.set_node_scope(n1, 0x1)
        atlas.set_node_scope(n2, 0x1)

        reports = consolidate_shared_scope(
            atlas, scope=0x1, subject_id_of=lambda nid: "subject1",
        )

        # Two singleton clusters, both below the default min_cluster_size=2.
        assert reports == []
        assert atlas.tombstone_count() == 0

    def test_node_outside_requested_scope_is_not_a_candidate(self, shared_atlas):
        atlas = shared_atlas.atlas
        n1 = atlas.insert(0, _vec(0.5), "ticket A")
        n2 = atlas.insert(0, _vec(0.5001), "ticket B")
        atlas.set_node_scope(n1, 0x1)
        atlas.set_node_scope(n2, 0x2)  # different scope, near-identical vector

        reports = consolidate_shared_scope(
            atlas, scope=0x1, subject_id_of=lambda nid: "subject1",
        )

        # n2 was never a candidate for scope=0x1 -- n1 alone is a singleton.
        assert reports == []
        assert atlas.tombstone_count() == 0

    def test_excludes_node_with_broader_overlapping_scope(self, shared_atlas):
        # list_nodes_by_scope() returns OVERLAP, not exact match (its own
        # doc comment) -- a node scoped to 0x1|0x2 overlaps scope_mask=0x1
        # but must NOT be treated as an exact-0x1 candidate here. Uses
        # _ConsolidateSpy (see its own doc comment) because the kernel's
        # OWN scope-uniformity precondition would otherwise mask a broken
        # Python-layer filter: both produce an empty `reports` list, so
        # this test asserts on what was actually ATTEMPTED, not just the
        # final outcome.
        atlas = shared_atlas.atlas
        n1 = atlas.insert(0, _vec(0.5), "ticket A")
        n2 = atlas.insert(0, _vec(0.5001), "ticket B")
        atlas.set_node_scope(n1, 0x1)
        atlas.set_node_scope(n2, 0x1 | 0x2)

        spy = _ConsolidateSpy(atlas)
        reports = consolidate_shared_scope(
            spy, scope=0x1, subject_id_of=lambda nid: "subject1",
        )

        for call_ids in spy.calls:
            assert n2 not in call_ids
        assert reports == []  # n1 alone -- singleton

    def test_never_clusters_across_subject_id(self, shared_atlas):
        atlas = shared_atlas.atlas
        n1 = atlas.insert(0, _vec(0.5), "ticket A")
        n2 = atlas.insert(0, _vec(0.5001), "ticket B")
        atlas.set_node_scope(n1, 0x1)
        atlas.set_node_scope(n2, 0x1)
        subjects = {n1: "subject1", n2: "subject2"}

        reports = consolidate_shared_scope(
            atlas, scope=0x1, subject_id_of=lambda nid: subjects[nid],
        )

        # Each is a singleton within its own subject group.
        assert reports == []
        assert atlas.tombstone_count() == 0

    def test_skips_node_with_unresolvable_subject_id(self, shared_atlas):
        # advisor review: a node with governance_record_id==0 (never
        # promoted) resolves to None and must be SKIPPED, not grouped
        # under a shared "unknown" key alongside an attributed node.
        atlas = shared_atlas.atlas
        n1 = atlas.insert(0, _vec(0.5), "attributed")
        n2 = atlas.insert(0, _vec(0.5001), "unattributed")
        atlas.set_node_scope(n1, 0x1)
        atlas.set_node_scope(n2, 0x1)
        subjects = {n1: "subject1", n2: None}

        reports = consolidate_shared_scope(
            atlas, scope=0x1, subject_id_of=lambda nid: subjects[nid],
        )

        assert reports == []  # n1 alone (n2 skipped) -- singleton
        assert atlas.tombstone_count() == 0
        assert atlas.is_node_superseded(n2) is False  # untouched

    def test_default_subject_id_of_none_skips_everything(self, shared_atlas):
        # Fail-closed: no resolver supplied means nothing is ever
        # consolidated, not "treat everything as one group".
        atlas = shared_atlas.atlas
        n1 = atlas.insert(0, _vec(0.5), "A")
        n2 = atlas.insert(0, _vec(0.5001), "B")
        atlas.set_node_scope(n1, 0x1)
        atlas.set_node_scope(n2, 0x1)

        reports = consolidate_shared_scope(atlas, scope=0x1)

        assert reports == []
        assert atlas.tombstone_count() == 0

    def test_records_merges_with_trace_edge_per_source(self, shared_atlas, tmp_path):
        atlas = shared_atlas.atlas
        trace = TraceGraph(tmp_path / "trace.bin")
        n1 = atlas.insert(0, _vec(0.5), "ticket A")
        n2 = atlas.insert(0, _vec(0.5001), "ticket B")
        atlas.set_node_scope(n1, 0x1)
        atlas.set_node_scope(n2, 0x1)

        reports = consolidate_shared_scope(
            atlas, scope=0x1, subject_id_of=lambda nid: "subject1",
            trace=trace, actor="dreamer1",
        )
        summary_id = reports[0].summary_node_id

        history = trace.get_history("dreamer1", limit=10)
        assert len(history) == 2  # one event per consolidated source
        encoded_summary = encode_store_id(summary_id, is_shared=True)
        for ev in history:
            assert int(ev["atlas_id"]) == encoded_summary
            assert ev["role"] == 2  # concept

    def test_no_trace_edges_when_trace_omitted(self, shared_atlas):
        atlas = shared_atlas.atlas
        n1 = atlas.insert(0, _vec(0.5), "A")
        n2 = atlas.insert(0, _vec(0.5001), "B")
        atlas.set_node_scope(n1, 0x1)
        atlas.set_node_scope(n2, 0x1)

        # Must succeed with no trace instance at all -- not raise.
        reports = consolidate_shared_scope(
            atlas, scope=0x1, subject_id_of=lambda nid: "subject1",
        )
        assert len(reports) == 1

    def test_min_cluster_size_one_consolidates_a_singleton(self, shared_atlas):
        atlas = shared_atlas.atlas
        n1 = atlas.insert(0, _vec(0.1), "A")
        atlas.set_node_scope(n1, 0x1)

        reports = consolidate_shared_scope(
            atlas, scope=0x1, subject_id_of=lambda nid: "subject1",
            min_cluster_size=1,
        )
        assert len(reports) == 1
        assert reports[0].nodes_consolidated == 1

    def test_excludes_candidate_with_its_own_children(self, shared_atlas):
        # v4-plan.md Stage 5 task 1 verification: consolidate_subgraph()'s
        # documented residual is that a NON-LEAF old_node_id's rewired
        # grandchildren aren't registered under the new summary's own
        # enumeration, making them structurally unreachable via
        # navigate(). Verified unexercised by any real production caller
        # (promote_fragment() always uses parent_id=0, so every promoted
        # node -- other than whichever one happens to be a shared atlas's
        # very first, see the shared_atlas fixture's own comment -- is
        # childless), but explicitly guarded here rather than left as an
        # implicit assumption a future change could silently violate.
        # Uses _ConsolidateSpy since the kernel has no equivalent
        # rejection for this case (unlike the mixed-scope precondition)
        # -- only this Python-layer filter protects it.
        atlas = shared_atlas.atlas
        n1 = atlas.insert(0, _vec(0.5), "ticket A")
        n2 = atlas.insert(0, _vec(0.5001), "ticket B")
        atlas.set_node_scope(n1, 0x1)
        atlas.set_node_scope(n2, 0x1)
        atlas.insert(n1, _vec(0.5002), "grandchild")  # gives n1 a child

        spy = _ConsolidateSpy(atlas)
        reports = consolidate_shared_scope(
            spy, scope=0x1, subject_id_of=lambda nid: "subject1",
        )

        for call_ids in spy.calls:
            assert n1 not in call_ids
        assert reports == []  # n2 alone -- singleton

    def test_sources_tombstoned_not_reclaimed(self, shared_atlas):
        # advisor review: this asserts what actually happens (sources are
        # tombstoned, storage isn't reclaimed), NOT "compact_mmap() was
        # never called" -- these same assertions would hold even if a
        # compaction ran and happened to reclaim nothing, so they don't
        # by themselves prove non-compaction. The actual guarantee
        # (compaction is deliberately not a side effect of this function)
        # is enforced by there being no call site for it in
        # consolidate_shared_scope(), documented in its own doc comment --
        # not by a runtime check here.
        atlas = shared_atlas.atlas
        n1 = atlas.insert(0, _vec(0.5), "A")
        n2 = atlas.insert(0, _vec(0.5001), "B")
        atlas.set_node_scope(n1, 0x1)
        atlas.set_node_scope(n2, 0x1)
        size_before = atlas.size()

        reports = consolidate_shared_scope(
            atlas, scope=0x1, subject_id_of=lambda nid: "subject1",
        )

        # consolidate_subgraph() appends one summary node and tombstones
        # (not removes) the sources -- size only ever grows here, and a
        # compaction would have physically reclaimed the tombstoned pair.
        assert atlas.size() == size_before + 1
        assert atlas.tombstone_count() == 2
