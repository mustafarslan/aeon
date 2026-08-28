"""Point-in-time traversal (`as_of`) -- v4.2 Track A.

NO ACCURACY CLAIM IS UNDER TEST HERE. LongMemEval asks every question at T_now and its
corpora predate the question date, so `as_of` cannot move a benchmark number and is not
gated on one. It is a product capability, and these are its correctness criteria.

The design under test is a DERIVED overlay, following `timeline.py`'s standing rule that
what is a pure function of stored state gets derived rather than stored. There is no
per-record timestamp to filter on and there cannot be one: `NodeHeader` is full at 64 bytes
and `Record.decode()` folds an eighth field into `text`. Validity is therefore derived from
provenance -- the `event_time` of the turns a record came from -- which is also the
semantics a user means by "as of March": a fact about the conversation, not about when the
consolidator happened to run.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from aeon_py.compose import compose_from_store
from aeon_py.records import Provenance, Record, RecordStore
from aeon_py.timeline import (
    edge_session, edges_before, session_event_times, validity_start, visible_at,
    write_supersession_edge,
)
from aeon_py.trace import TraceGraph, event_time_of

DIM = 8
TENANT = "tenant-a"

MARCH = 1_677_628_800_000_000     # 2023/03/01, epoch microseconds
JUNE = 1_685_577_600_000_000      # 2023/06/01
SEPTEMBER = 1_693_526_400_000_000  # 2023/09/01


@pytest.fixture
def tmp_root():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def unit_vec():
    return (np.ones(DIM, dtype=np.float32) / np.sqrt(DIM)).tolist()


@pytest.fixture
def trace(tmp_root):
    """One session whose turns straddle the query moments under test."""
    tr = TraceGraph(path=str(tmp_root / "t.trace"))
    tr.add_event("s1", "user", "I bought three tops from H&M.", event_time=MARCH)
    tr.add_event("s1", "system", "Nice.", event_time=MARCH + 1)
    tr.add_event("s1", "user", "Make that five tops from H&M.", event_time=SEPTEMBER)
    return tr


def rec(text, *, turns, session="s1", node_id=None, superseded=False):
    r = Record(kind="ITEM", text=text, bucket="ACQUISITION", subtype="clothing",
               provenance=Provenance(session, turns), node_id=node_id)
    r.superseded = superseded
    return r


# --- the time axis ------------------------------------------------------------------

def test_event_time_is_the_axis_and_timestamp_is_only_the_fallback():
    """`timestamp` is when the row was written. On a bulk-ingested corpus every row shares
    one ingest run, so it answers no temporal question -- it is a fallback, never the axis."""
    assert event_time_of({"event_time": JUNE, "timestamp": 999}) == JUNE
    assert event_time_of({"event_time": 0, "timestamp": 999}) == 999
    assert event_time_of({}) == 0


def test_session_times_are_indexed_the_way_provenance_numbers_them(trace):
    """`get_history()` is newest-first and `rehydrate()` reverses it before indexing. If
    these two ever disagree, every provenance index silently addresses the wrong turn."""
    times = session_event_times(trace, "s1")
    assert times == [MARCH, MARCH + 1, SEPTEMBER]


def test_validity_start_is_the_latest_source_turn_not_the_earliest():
    """A record citing turns 0 and 2 is not assertable until turn 2 has been said."""
    times = [MARCH, MARCH + 1, SEPTEMBER]
    assert validity_start(rec("x", turns=(0, 2)), times) == SEPTEMBER
    assert validity_start(rec("x", turns=(0, 1)), times) == MARCH + 1


def test_whole_session_provenance_falls_back_to_the_last_turn():
    """`parse_records()` assigns whole-session provenance when a citation is lost, and an
    empty index set must not read as 'valid since the beginning of time'."""
    assert validity_start(rec("x", turns=()), [MARCH, SEPTEMBER]) == SEPTEMBER


def test_out_of_range_turn_indices_do_not_crash_or_widen_validity():
    """A model-emitted citation can name a turn that does not exist."""
    assert validity_start(rec("x", turns=(0, 99)), [MARCH, SEPTEMBER]) == MARCH


def test_unknown_session_is_unresolvable_rather_than_ancient():
    assert validity_start(rec("x", turns=(0,)), ()) is None


# --- filter one: not yet true -------------------------------------------------------

def test_a_record_whose_evidence_came_later_is_invisible_in_the_past(trace):
    march_rec = rec("three tops from H&M", turns=(0,))
    sept_rec = rec("five tops from H&M", turns=(2,))
    assert visible_at([march_rec, sept_rec], trace, JUNE) == [march_rec]
    assert visible_at([march_rec, sept_rec], trace, SEPTEMBER) == [march_rec, sept_rec]


def test_as_of_none_returns_the_input_untouched(trace):
    recs = [rec("a", turns=(0,)), rec("b", turns=(2,))]
    assert visible_at(recs, trace, None) == recs


def test_the_boundary_is_inclusive(trace):
    """`as_of` names a moment; evidence said AT that moment had already been said by it."""
    assert visible_at([rec("a", turns=(0,))], trace, MARCH) != []


# --- filter two: no longer true -----------------------------------------------------

def test_a_supersession_recorded_later_is_reversed_in_the_past(trace):
    """The historical view needs the RETIRED record back -- that is the whole point of it,
    and the live view is exactly where it is not available."""
    write_supersession_edge(trace, tenant=TENANT, survivor_node_id=2, retired_node_id=1,
                            text="five supersedes three", event_time=SEPTEMBER)
    old = rec("three tops from H&M", turns=(0,), node_id=1, superseded=True)
    new = rec("five tops from H&M", turns=(2,), node_id=2)

    at_june = visible_at([old, new], trace, JUNE, tenant=TENANT)
    assert at_june == [old], "in June the user had three tops and the revision had not happened"

    at_now = visible_at([old, new], trace, SEPTEMBER + 1, tenant=TENANT)
    assert at_now == [new], "by September the earlier record is retired"


def test_edges_before_is_inclusive_at_the_boundary(trace):
    write_supersession_edge(trace, tenant=TENANT, survivor_node_id=2, retired_node_id=1,
                            text="x", event_time=JUNE)
    assert len(edges_before(trace, tenant=TENANT, as_of=JUNE)) == 1
    assert edges_before(trace, tenant=TENANT, as_of=JUNE - 1) == []


def test_without_a_tenant_it_degrades_to_the_present_view_not_a_wrong_past(trace):
    """Retirement times live in the tenant's edge session. With no tenant they are unknown,
    and an undatable retirement must not be silently treated as 'had not happened yet'."""
    old = rec("three tops", turns=(0,), node_id=1, superseded=True)
    assert visible_at([old], trace, JUNE) == []


# --- unresolvable provenance --------------------------------------------------------

def test_unresolvable_records_are_included_by_default(trace):
    """Matches `RecordStore._decode()`'s posture: degrade toward showing a record rather
    than vanishing it from a user's memory."""
    orphan = rec("from a session this trace never saw", turns=(0,), session="gone")
    assert visible_at([orphan], trace, JUNE) == [orphan]


def test_an_audit_caller_can_exclude_what_it_cannot_date(trace):
    orphan = rec("undatable", turns=(0,), session="gone")
    assert visible_at([orphan], trace, JUNE, unresolved="exclude") == []


def test_an_unknown_unresolved_mode_is_an_error_not_a_silent_default(trace):
    with pytest.raises(ValueError):
        visible_at([], trace, JUNE, unresolved="maybe")


# --- both halves of the assembly ----------------------------------------------------

def test_as_of_none_is_byte_identical(tmp_root, trace, unit_vec):
    """The measured read path must not move because a temporal parameter now exists."""
    store = RecordStore(tmp_root / "r.atlas", dim=DIM, metadata_size=256)
    store.add(rec("three tops from H&M", turns=(0,)), unit_vec)
    store.add(rec("five tops from H&M", turns=(2,)), unit_vec)

    baseline = compose_from_store(store, trace, "Q: how many tops?")
    with_param = compose_from_store(store, trace, "Q: how many tops?", as_of=None)
    assert with_param["prompt"] == baseline["prompt"]


def test_the_episodic_half_is_cut_too(tmp_root, trace, unit_vec):
    """A record block cut to March beside excerpts quoting September is not a historical
    view -- it is a leak that reads as one. Both halves move together or neither does."""
    store = RecordStore(tmp_root / "r.atlas", dim=DIM, metadata_size=256)
    store.add(rec("three tops from H&M", turns=(0,)), unit_vec)
    store.add(rec("five tops from H&M", turns=(2,)), unit_vec)

    past = compose_from_store(store, trace, "Q: how many tops?", as_of=JUNE)
    assert "three tops" in past["prompt"]
    assert "five tops" not in past["prompt"]
    assert "Make that five tops" not in past["prompt"], "the raw September turn leaked"

    now = compose_from_store(store, trace, "Q: how many tops?")
    assert "five tops" in now["prompt"]
