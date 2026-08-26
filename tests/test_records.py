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
    BUCKETS, Provenance, Record, RecordStore, _compress_indices, _expand_indices,
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
