"""Tests for the persistent semantic layer (`aeon_py.records`).

Focused on provenance, because that is the part the consolidation probe never exercised and
the part the rest of the layer depends on: a record whose origin is lost cannot recover the
conversational licensing that compression destroys, which this project measured at a cost of
12 questions (v4-plan.md, mode (ii)).
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from aeon_py.records import (
    BUCKETS, Provenance, Record, RecordStore, _as_int, _compress_indices, _expand_indices,
    _utf8_truncate,
)
from aeon_py.trace import TraceGraph

DIM = 8


@pytest.fixture
def tmp_root():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def unit_vec():
    return (np.ones(DIM, dtype=np.float32) / np.sqrt(DIM)).tolist()


@pytest.fixture
def trace(tmp_root):
    tr = TraceGraph(path=str(tmp_root / "t.trace"))
    for role, text in (
        ("user", "I have been getting into vinyl lately."),
        ("user", "I just downloaded Happier Than Ever by Billie Eilish."),
        ("system", "Great album! Billie Eilish won a Grammy for it."),
        ("user", "Also picked up a lavender shampoo at Trader Joe's."),
        ("user", "Unrelated: my flight is on Tuesday."),
    ):
        tr.add_event("sess-7", role, text)
    return tr


# --- provenance encoding ----------------------------------------------------

@pytest.mark.parametrize("idx", [(), (5,), (0, 1, 2), (1, 2, 3, 7, 8, 20)])
def test_turn_index_roundtrip(idx):
    assert _expand_indices(_compress_indices(idx)) == tuple(sorted(set(idx)))


def test_contiguous_indices_compress_to_ranges():
    # Provenance shares a fixed metadata budget with the record text, so compactness
    # here directly buys text headroom.
    assert _compress_indices((0, 1, 2, 3)) == "0-3"


def test_provenance_roundtrip_through_record():
    r = Record(kind="ITEM", text="x", bucket="MEDIA", subtype="album",
               provenance=Provenance("sess-7", (3, 4)))
    back = Record.decode(r.encode())
    assert back.provenance.session_id == "sess-7"
    assert back.provenance.turn_indices == (3, 4)


def test_session_ids_containing_colons_survive():
    # rpartition, not partition -- a session id may legitimately contain ':'.
    p = Provenance.decode(Provenance("tenant:abc:7", (1,)).encode())
    assert p.session_id == "tenant:abc:7"
    assert p.turn_indices == (1,)


# --- overflow handling ------------------------------------------------------

@pytest.mark.parametrize("text,limit", [("héllo wörld", 6), ("日本語テキスト", 7), ("ascii", 3)])
def test_utf8_truncate_never_splits_a_character(text, limit):
    out = _utf8_truncate(text, limit)
    assert len(out.encode("utf-8")) <= limit
    out.encode("utf-8").decode("utf-8")  # must not raise


def test_overflow_preserves_structure_and_provenance(tmp_root, unit_vec):
    """`Atlas.insert()` truncates silently, so an oversized record must lose its text tail,
    never its structure -- a record you can still trace back beats one whose text is intact
    but whose origin is gone."""
    store = RecordStore(tmp_root / "r.atlas", dim=DIM, metadata_size=256)
    store.add(Record(kind="ITEM", text="Y" * 400, bucket="MEDIA", subtype="film",
                     date="2023/01/01", provenance=Provenance("sess-9", (1,))), unit_vec)
    (rec,) = store.all_records()
    assert rec.kind == "ITEM" and rec.bucket == "MEDIA" and rec.subtype == "film"
    assert rec.provenance.session_id == "sess-9" and rec.provenance.turn_indices == (1,)
    assert len(rec.text) < 400


def test_multibyte_overflow_is_readable(tmp_root, unit_vec):
    """Regression: an earlier version appended a 3-byte ellipsis onto a full budget, so the
    stored payload ended mid-UTF-8-sequence and every later read raised UnicodeDecodeError.
    Same class of bug the C++ side fixed on TraceEvent.text_preview."""
    store = RecordStore(tmp_root / "r.atlas", dim=DIM, metadata_size=256)
    store.add(Record(kind="ITEM", text="日本語" * 100, bucket="TRAVEL", subtype="trip",
                     provenance=Provenance("sess-3", (2,))), unit_vec)
    (rec,) = store.all_records()          # must not raise
    assert rec.bucket == "TRAVEL"


# --- provenance rehydration -------------------------------------------------

def test_rehydrate_returns_neighbourhood(trace):
    rec = Record(kind="ITEM", text="Happier Than Ever", bucket="ACQUISITION",
                 subtype="music album", provenance=Provenance("sess-7", (1,)))
    assert len(RecordStore.rehydrate([rec], trace, neighbours=0)) == 1
    ctx = RecordStore.rehydrate([rec], trace, neighbours=1)
    assert len(ctx) == 3
    assert any("vinyl" in c for c in ctx) and any("Grammy" in c for c in ctx)


def test_rehydrate_recovers_pragmatic_licensing(trace):
    """The whole reason provenance exists: the bare record cannot answer "what brand?",
    the neighbourhood can."""
    rec = Record(kind="ITEM", text="lavender shampoo", bucket="POSSESSION",
                 subtype="toiletry", provenance=Provenance("sess-7", (3,)))
    assert "Trader Joe" not in rec.display()
    assert any("Trader Joe" in c for c in RecordStore.rehydrate([rec], trace, neighbours=1))


def test_rehydrate_uses_role_names_not_ints(trace):
    """`get_history` returns role as an int; rehydrated context must match the `- [user] ...`
    shape every other context path in this repo emits."""
    rec = Record(kind="ITEM", text="x", bucket="MEDIA", provenance=Provenance("sess-7", (1,)))
    ctx = RecordStore.rehydrate([rec], trace, neighbours=0)
    assert ctx[0].startswith("- [user]")


def test_rehydrate_respects_max_turns(trace):
    """Provenance must not quietly reintroduce the ~100k-char context this layer exists to
    avoid."""
    recs = [Record(kind="ITEM", text=f"r{i}", bucket="MEDIA",
                   provenance=Provenance("sess-7", (i,))) for i in range(5)]
    assert len(RecordStore.rehydrate(recs, trace, neighbours=1, max_turns=3)) == 3


@pytest.mark.parametrize("prov", [
    Provenance(""),                      # no provenance at all
    Provenance("no-such-session", (1,)),  # dangling session
    Provenance("sess-7", (99,)),          # out-of-range turn
])
def test_rehydrate_degrades_gracefully(trace, prov):
    assert RecordStore.rehydrate([Record(kind="FACT", text="x", provenance=prov)], trace) == []


# --- schema -----------------------------------------------------------------

def test_buckets_are_closed_and_unique():
    """The closed set is load-bearing: free-form categories were tried first and members of
    one real category scattered across labels, so counting them returned 1 against a gold
    of 3."""
    assert len(BUCKETS) == len(set(BUCKETS)) == 12


def test_records_persist_across_reopen(tmp_root, unit_vec):
    path = tmp_root / "r.atlas"
    store = RecordStore(path, dim=DIM, metadata_size=512)
    store.add(Record(kind="ITEM", text="Midnight Sky", bucket="ACQUISITION",
                     subtype="music album", provenance=Provenance("sess-1", (2,))), unit_vec)
    store.sync()
    reopened = RecordStore(path, dim=DIM, metadata_size=512)
    (rec,) = reopened.all_records()
    assert rec.text == "Midnight Sky" and rec.provenance.turn_indices == (2,)


# --- kernel capabilities adopted instead of reimplemented --------------------

def test_buckets_materialise_as_an_atlas_subtree(tmp_root, unit_vec):
    """The closed taxonomy IS a tree, and Atlas is a tree. Records become children of their
    bucket node so a category scan is a kernel subtree walk, not a Python filter."""
    store = RecordStore(tmp_root / "r.atlas", dim=DIM)
    for i in range(3):
        store.add(Record(kind="ITEM", text=f"album{i}", bucket="ACQUISITION",
                         subtype="music album", provenance=Provenance("s1", (i,))), unit_vec)
    store.add(Record(kind="ITEM", text="Dune", bucket="MEDIA", subtype="film",
                     provenance=Provenance("s1", (9,))), unit_vec)
    assert len(store.records_in_bucket("ACQUISITION")) == 3
    assert len(store.records_in_bucket("MEDIA")) == 1


def test_category_scan_returns_every_member_not_a_subset(tmp_root, unit_vec):
    """Counting requires completeness; a top-k would return a subset and turn a complete
    answer into a plausible wrong one."""
    store = RecordStore(tmp_root / "r.atlas", dim=DIM)
    for i in range(12):
        store.add(Record(kind="ITEM", text=f"m{i}", bucket="MEDIA", subtype="film",
                         provenance=Provenance("s1", (i,))), unit_vec)
    assert len(store.records_in_bucket("MEDIA")) == 12


def test_unknown_bucket_scan_is_empty_not_an_error(tmp_root):
    assert RecordStore(tmp_root / "r.atlas", dim=DIM).records_in_bucket("NOPE") == []


def test_bucket_nodes_are_not_returned_as_records(tmp_root, unit_vec):
    """The taxonomy's own nodes live in the same Atlas; they must never leak into a record
    set, or every count would be inflated by its category marker."""
    store = RecordStore(tmp_root / "r.atlas", dim=DIM)
    store.add(Record(kind="ITEM", text="Dune", bucket="MEDIA", subtype="film",
                     provenance=Provenance("s1", (0,))), unit_vec)
    recs = store.all_records()
    assert len(recs) == 1 and recs[0].text == "Dune"


def test_records_are_session_scoped_for_multi_tenancy(tmp_root, unit_vec):
    """`Atlas.insert` takes a session_id and the first version passed none -- invisible in a
    benchmark (one store per question), fatal in a multi-tenant product."""
    store = RecordStore(tmp_root / "r.atlas", dim=DIM, session_id="tenant-a")
    store.add(Record(kind="FACT", text="x", provenance=Provenance("s1", (0,))), unit_vec)
    assert store.session_id == "tenant-a"
    assert len(store.all_records()) == 1


def test_supersede_uses_the_kernel_primitive(tmp_root, unit_vec):
    """A superseded record should be excluded from retrieval, reversibly -- not shown to the
    model beside its replacement with a text marker and hoped about."""
    store = RecordStore(tmp_root / "r.atlas", dim=DIM)
    rec = Record(kind="ITEM", text="$350,000", bucket="FINANCE", subtype="pre-approval",
                 provenance=Provenance("s1", (0,)))
    store.add(rec, unit_vec)
    assert store.supersede(rec) is True
    assert store.atlas.is_node_superseded(rec.node_id) is True


def test_supersede_without_a_node_id_is_a_no_op(tmp_root):
    store = RecordStore(tmp_root / "r.atlas", dim=DIM)
    assert store.supersede(Record(kind="FACT", text="never stored")) is False


def test_provenance_doubles_as_the_erasure_cascade_index(tmp_root, unit_vec):
    """Records are PII derived from conversation. erasure.py tombstones Atlas nodes but
    nothing cascaded to derived records; provenance makes that a filter, not a new
    mechanism."""
    store = RecordStore(tmp_root / "r.atlas", dim=DIM)
    for sess in ("keep", "erase", "keep"):
        store.add(Record(kind="FACT", text=f"from {sess}",
                         provenance=Provenance(sess, (0,))), unit_vec)
    assert len(store.records_for_session("erase")) == 1
    assert len(store.records_for_session("keep")) == 2


def test_tree_refactor_does_not_change_rendered_context(tmp_root, unit_vec):
    """EQUIVALENCE GUARD. The composite's measured 72/85 depends on what render_records()
    emits; moving records into a bucket subtree must not alter that by one byte."""
    from aeon_py.compose import render_records
    recs = [Record(kind="ITEM", text="Dune", bucket="MEDIA", subtype="film",
                   provenance=Provenance("s1", (0,))),
            Record(kind="ITEM", text="album", bucket="ACQUISITION", subtype="music album",
                   provenance=Provenance("s1", (1,))),
            Record(kind="FACT", text="owns a bike", provenance=Provenance("s1", (2,)))]
    expected = render_records(recs)
    store = RecordStore(tmp_root / "r.atlas", dim=DIM)
    for r in recs:
        store.add(r, unit_vec)
    assert render_records(store.all_records()) == expected


def test_superseded_records_are_excluded_from_all_records(tmp_root, unit_vec):
    """BUG FIX. supersede_node() sets hub_penalty, which excludes a node from BEAM SEARCH
    and nothing else -- enumeration never consulted it, so a superseded record still
    rendered into every prompt. This test fails against the code as it stood."""
    store = RecordStore(tmp_root / "r.atlas", dim=DIM)
    keep = Record(kind="FACT", text="current", provenance=Provenance("s1", (0,)))
    stale = Record(kind="FACT", text="stale", provenance=Provenance("s1", (1,)))
    store.add(keep, unit_vec)
    store.add(stale, unit_vec)
    assert store.supersede(stale)
    texts = [r.text for r in store.all_records()]
    assert texts == ["current"]


def test_include_superseded_returns_them_for_audit(tmp_root, unit_vec):
    store = RecordStore(tmp_root / "r.atlas", dim=DIM)
    r = Record(kind="FACT", text="stale", provenance=Provenance("s1", (0,)))
    store.add(r, unit_vec)
    store.supersede(r)
    assert store.all_records() == []
    assert [x.text for x in store.all_records(include_superseded=True)] == ["stale"]


def test_superseded_records_do_not_reach_the_rendered_context(tmp_root, unit_vec):
    """The point of the fix: what the model is shown must not contain retired records."""
    from aeon_py.compose import render_records
    store = RecordStore(tmp_root / "r.atlas", dim=DIM)
    stale = Record(kind="ITEM", text="old salary", bucket="FINANCE", subtype="pay",
                   provenance=Provenance("s1", (0,)))
    store.add(stale, unit_vec)
    store.add(Record(kind="ITEM", text="new salary", bucket="FINANCE", subtype="pay",
                     provenance=Provenance("s1", (1,))), unit_vec)
    store.supersede(stale)
    assert "old salary" not in render_records(store.all_records())


# --- bucket index persistence (v4.1 correctness) ------------------------------------

def test_bucket_scan_survives_a_reopen(tmp_root, unit_vec):
    """BUG FIX. `_bucket_nodes` was in-memory only, so a store opened over an existing file
    started with an empty index and the subtree walk returned nothing. The bucket node and
    its children were on disk the whole time."""
    path = tmp_root / "r.atlas"
    s1 = RecordStore(path, dim=DIM)
    s1.add(Record(kind="ITEM", text="Dune", bucket="MEDIA", subtype="film",
                  provenance=Provenance("s1", (0,))), unit_vec)
    del s1
    s2 = RecordStore(path, dim=DIM)
    assert [r.text for r in s2.records_in_bucket("MEDIA")] == ["Dune"]


def test_reopen_does_not_mint_a_duplicate_bucket_node(tmp_root, unit_vec):
    """The second-order failure, and the dangerous one: a duplicate bucket node made the
    category scan return only what was written since the re-open -- a SILENT SUBSET, which
    is the exact failure the structured scan exists to prevent."""
    path = tmp_root / "r.atlas"
    s1 = RecordStore(path, dim=DIM)
    s1.add(Record(kind="ITEM", text="Dune", bucket="MEDIA", subtype="film",
                  provenance=Provenance("s1", (0,))), unit_vec)
    del s1
    s2 = RecordStore(path, dim=DIM)
    before = _as_int(s2.atlas.size)
    s2.add(Record(kind="ITEM", text="Arrival", bucket="MEDIA", subtype="film",
                  provenance=Provenance("s2", (0,))), unit_vec)
    assert _as_int(s2.atlas.size) - before == 1        # the record only, no second bucket
    markers = sum(1 for i in range(_as_int(s2.atlas.size))
                  if (s2._read_metadata(i) or "").startswith("BUCKET"))
    assert markers == 1


def test_category_scan_is_complete_across_a_reopen(tmp_root, unit_vec):
    """Counting is what this layer exists for, so a partial scan is worse than no scan."""
    path = tmp_root / "r.atlas"
    s1 = RecordStore(path, dim=DIM)
    s1.add(Record(kind="ITEM", text="Dune", bucket="MEDIA", subtype="film",
                  provenance=Provenance("s1", (0,))), unit_vec)
    del s1
    s2 = RecordStore(path, dim=DIM)
    s2.add(Record(kind="ITEM", text="Arrival", bucket="MEDIA", subtype="film",
                  provenance=Provenance("s2", (0,))), unit_vec)
    assert sorted(r.text for r in s2.records_in_bucket("MEDIA")) == ["Arrival", "Dune"]


def test_bucket_index_rebuild_ignores_ordinary_records(tmp_root, unit_vec):
    path = tmp_root / "r.atlas"
    s1 = RecordStore(path, dim=DIM)
    s1.add(Record(kind="FACT", text="BUCKET-ish text that is not a marker",
                  provenance=Provenance("s1", (0,))), unit_vec)
    del s1
    assert RecordStore(path, dim=DIM)._bucket_nodes == {}


# --- erasure cascade to derived records (v4.1 correctness) --------------------------

def test_erasure_cascades_to_records_derived_from_a_session(tmp_root, unit_vec):
    """Records are PII DERIVED from conversation. `records_for_session()` is documented as
    the right-to-erasure cascade index and had zero non-test callers, so erasing a node left
    every record extracted from that session in place."""
    from aeon_py.erasure import cascade_to_derived_records
    store = RecordStore(tmp_root / "r.atlas", dim=DIM)
    keep = Record(kind="FACT", text="from another session", provenance=Provenance("keep", (0,)))
    store.add(Record(kind="FACT", text="derived A", provenance=Provenance("erase", (0,))), unit_vec)
    store.add(Record(kind="FACT", text="derived B", provenance=Provenance("erase", (1,))), unit_vec)
    store.add(keep, unit_vec)

    cascaded, failures = cascade_to_derived_records(store, ["erase"])
    assert len(cascaded) == 2 and failures == []
    assert [r.text for r in store.all_records()] == ["from another session"]


def test_cascade_is_a_no_op_without_a_store_or_sessions(tmp_root, unit_vec):
    from aeon_py.erasure import cascade_to_derived_records
    store = RecordStore(tmp_root / "r.atlas", dim=DIM)
    store.add(Record(kind="FACT", text="x", provenance=Provenance("s1", (0,))), unit_vec)
    assert cascade_to_derived_records(None, ["s1"]) == ([], [])
    assert cascade_to_derived_records(store, []) == ([], [])
    assert [r.text for r in store.all_records()] == ["x"]


def test_cascade_reports_failures_rather_than_aborting(tmp_root, unit_vec):
    """One record that will not tombstone must not abort the rest of the cascade.

    The Atlas binding is read-only, so the flaky store is a stand-in rather than a patch --
    which also keeps this test on `cascade_to_derived_records`'s contract instead of on
    nanobind's attribute semantics."""
    from aeon_py.erasure import cascade_to_derived_records

    class _FlakyAtlas:
        def __init__(self):
            self.calls = 0
            self.tombstoned = []

        def tombstone_node(self, node_id):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient")
            self.tombstoned.append(node_id)

    class _FlakyStore:
        def __init__(self):
            self.atlas = _FlakyAtlas()

        def records_for_session(self, session_id):
            return [Record(kind="FACT", text="a", node_id=11),
                    Record(kind="FACT", text="b", node_id=12)]

    store = _FlakyStore()
    cascaded, failures = cascade_to_derived_records(store, ["s"])
    assert cascaded == [12] and len(failures) == 1
    assert "transient" in failures[0]["reason"]
    assert store.atlas.tombstoned == [12]        # the second record still went through


def test_tombstoned_records_do_not_reach_the_rendered_context(tmp_root, unit_vec):
    """The compliance twin of the supersession bug. `tombstone_node()` has no Python-visible
    per-node predicate, so `range(atlas.size)` enumeration could not tell an erased node from
    a live one -- an erased record still rendered into every prompt."""
    from aeon_py.compose import render_records
    store = RecordStore(tmp_root / "r.atlas", dim=DIM)
    gone = Record(kind="ITEM", text="erased fact", bucket="HEALTH", subtype="note",
                  provenance=Provenance("s1", (0,)))
    store.add(gone, unit_vec)
    store.add(Record(kind="ITEM", text="kept fact", bucket="HEALTH", subtype="note",
                     provenance=Provenance("s2", (0,))), unit_vec)
    store.atlas.tombstone_node(gone.node_id)
    assert "erased fact" not in render_records(store.all_records())
    assert "kept fact" in render_records(store.all_records())


def test_audit_view_does_not_resurrect_tombstoned_records(tmp_root, unit_vec):
    """`include_superseded` is for audit paths, but a TOMBSTONE is terminal and a
    right-to-erasure guarantee -- it must not be re-openable by an audit flag."""
    store = RecordStore(tmp_root / "r.atlas", dim=DIM)
    gone = Record(kind="FACT", text="erased", provenance=Provenance("s1", (0,)))
    store.add(gone, unit_vec)
    store.atlas.tombstone_node(gone.node_id)
    assert store.all_records(include_superseded=True) == []


def test_compose_from_store_bucket_filter_uses_the_subtree_after_a_reopen(tmp_root, unit_vec):
    """Ties the two fixes together: the bucket filter now goes through the kernel subtree
    walk, which is only trustworthy because the bucket index is rebuilt on open. Before both
    fixes this returned a silent subset across a restart."""
    from aeon_py.compose import compose_from_store
    path = tmp_root / "r.atlas"
    s1 = RecordStore(path, dim=DIM)
    s1.add(Record(kind="ITEM", text="Dune", bucket="MEDIA", subtype="film",
                  provenance=Provenance("s1", (0,))), unit_vec)
    s1.add(Record(kind="ITEM", text="a hammer", bucket="POSSESSION", subtype="tool",
                  provenance=Provenance("s1", (1,))), unit_vec)
    del s1
    s2 = RecordStore(path, dim=DIM)
    s2.add(Record(kind="ITEM", text="Arrival", bucket="MEDIA", subtype="film",
                  provenance=Provenance("s2", (0,))), unit_vec)

    class _NoTrace:
        def get_history(self, session_id, limit):
            return []

    out = compose_from_store(s2, _NoTrace(), "Question: how many films?", buckets=["MEDIA"])
    assert out["record_count"] == 2
    assert "Dune" in out["prompt"] and "Arrival" in out["prompt"]
    assert "a hammer" not in out["prompt"]


def test_kernel_subtree_walk_is_unsound_for_interleaved_writes(tmp_root, unit_vec):
    """PINS THE KERNEL LIMITATION that forced `records_in_bucket()` to become a filter, so it
    cannot be rediscovered as a mystery -- and so nobody "optimises" the filter back into a
    subtree walk.

    Atlas registers a child only when its byte offset is exactly the next slot after the
    parent's existing children, and `insert()` appends at the tail. Interleave two buckets and
    the later children are invisible to `get_children()` forever. This asserts the KERNEL's
    behaviour directly; if it ever starts failing, the kernel gained non-contiguous children
    and the filter can go back to being a walk."""
    store = RecordStore(tmp_root / "r.atlas", dim=DIM)
    for text, bucket in (("A", "MEDIA"), ("H", "POSSESSION"), ("C", "MEDIA")):
        store.add(Record(kind="ITEM", text=text, bucket=bucket, subtype="x",
                         provenance=Provenance("s", (0,))), unit_vec)
    walked = store._decode(np.frombuffer(
        bytes(store.atlas.get_children_raw(store._bucket_nodes["MEDIA"])), dtype=np.uint64))
    assert [r.text for r in walked] == ["A"]                                  # kernel: subset
    assert sorted(r.text for r in store.records_in_bucket("MEDIA")) == ["A", "C"]   # filter: complete


def test_category_scan_is_complete_under_interleaved_writes(tmp_root, unit_vec):
    """The property the layer actually needs. Counting is what it exists for, so a partial
    category scan turns a complete answer into a plausible wrong one."""
    store = RecordStore(tmp_root / "r.atlas", dim=DIM)
    for text, bucket in (("Dune", "MEDIA"), ("album", "ACQUISITION"), ("Arrival", "MEDIA"),
                         ("Alien", "MEDIA"), ("vinyl", "ACQUISITION")):
        store.add(Record(kind="ITEM", text=text, bucket=bucket, subtype="x",
                         provenance=Provenance("s", (0,))), unit_vec)
    assert len(store.records_in_bucket("MEDIA")) == 3
    assert len(store.records_in_bucket("ACQUISITION")) == 2


def test_erasure_cascade_index_includes_superseded_records(tmp_root, unit_vec):
    """A superseded record is excluded from PROMPTS, not from EXISTENCE. It is still PII
    derived from the session, so an erasure cascade using the read-path default would
    tombstone the live records and silently leave the retired ones on disk."""
    store = RecordStore(tmp_root / "r.atlas", dim=DIM)
    old = Record(kind="ITEM", text="three tops", bucket="ACQUISITION", subtype="clothing",
                 provenance=Provenance("erase", (0,)))
    store.add(old, unit_vec)
    store.add(Record(kind="ITEM", text="five tops", bucket="ACQUISITION", subtype="clothing",
                     provenance=Provenance("erase", (1,))), unit_vec)
    store.supersede(old)
    assert [r.text for r in store.all_records()] == ["five tops"]          # prompt view
    assert sorted(r.text for r in store.records_for_session("erase")) == [
        "five tops", "three tops"]                                          # erasure view


def test_erasure_cascade_reaches_superseded_records(tmp_root, unit_vec):
    from aeon_py.erasure import cascade_to_derived_records
    store = RecordStore(tmp_root / "r.atlas", dim=DIM)
    old = Record(kind="FACT", text="retired but still PII", provenance=Provenance("erase", (0,)))
    store.add(old, unit_vec)
    store.supersede(old)
    cascaded, failures = cascade_to_derived_records(store, ["erase"])
    assert len(cascaded) == 1 and failures == []
    assert store.all_records(include_superseded=True) == []
