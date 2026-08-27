"""Tests for background consolidation (`aeon_py.consolidator`).

The properties here protect two things this project cares about most: that ingest stays at
kernel speed, and that a session's records are never silently lost -- losing them is invisible
at read time (the answer is merely worse), which is the kind of data loss that never gets
reported as a bug.
"""

import threading
import time

import pytest

from aeon_py.consolidator import DirtyQueue, SessionConsolidator
from aeon_py.records import Provenance, Record


class FakeStore:
    def __init__(self, fail_on=None):
        self.records = []
        self.fail_on = fail_on

    def add(self, rec, embedding):
        if self.fail_on and self.fail_on in rec.text:
            raise RuntimeError("disk full")
        self.records.append(rec)
        return len(self.records) - 1

    def all_records(self):
        return list(self.records)


def rec(text, session="s1"):
    return Record(kind="FACT", text=text, provenance=Provenance(session, (0,)))


def make(queue, *, turns=None, extract=None, store=None, merge=None):
    return SessionConsolidator(
        queue,
        fetch_session=turns or (lambda s: [{"role": "user", "content": f"hi from {s}"}]),
        extract=extract or (lambda t, s, d: [rec(f"fact-{s}", s)]),
        embed=lambda text: [0.1, 0.2],
        store=store if store is not None else FakeStore(),
        merge=merge,
    )


# --- the queue ---------------------------------------------------------------

def test_marking_dirty_is_idempotent():
    """A session written ten times before the worker wakes needs consolidating once --
    otherwise consolidation cost scales with write volume, not distinct sessions."""
    q = DirtyQueue()
    for _ in range(10):
        q.mark_dirty("s1")
    assert q.pending_count == 1


def test_mark_dirty_is_cheap_and_lock_safe_under_threads():
    q = DirtyQueue()

    def hammer(i):
        for j in range(200):
            q.mark_dirty(f"s{j % 20}")

    threads = [threading.Thread(target=hammer, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert q.pending_count == 20


def test_empty_session_id_is_ignored():
    q = DirtyQueue()
    q.mark_dirty("")
    assert q.pending_count == 0


def test_claim_moves_to_in_flight_not_oblivion():
    q = DirtyQueue()
    q.mark_dirty("s1")
    assert q.claim() == ["s1"]
    assert q.pending_count == 0 and q.in_flight_count == 1
    assert "s1" in q                       # still tracked


def test_claim_respects_limit_and_is_deterministic():
    q = DirtyQueue()
    for s in ("s3", "s1", "s2"):
        q.mark_dirty(s)
    assert q.claim(limit=2) == ["s1", "s2"]


def test_failed_release_requeues_for_retry():
    q = DirtyQueue()
    q.mark_dirty("s1")
    q.claim()
    q.release("s1", succeeded=False)
    assert q.pending_count == 1


def test_successful_release_clears_the_session():
    q = DirtyQueue()
    q.mark_dirty("s1")
    q.claim()
    q.release("s1", succeeded=True)
    assert q.pending_count == 0 and q.in_flight_count == 0


def test_in_flight_work_is_recovered_after_a_crash():
    """Anything in flight was never confirmed, so on restart it must go back to pending."""
    q = DirtyQueue()
    q.mark_dirty("s1")
    q.mark_dirty("s2")
    q.claim()
    assert q.requeue_in_flight() == 2
    assert q.pending_count == 2 and q.in_flight_count == 0


# --- the consolidation cycle -------------------------------------------------

def test_cycle_writes_records_and_drains_the_queue():
    q = DirtyQueue()
    q.mark_dirty("s1")
    store = FakeStore()
    stats = make(q, store=store).run_cycle()
    assert stats.sessions_consolidated == 1 and stats.records_written == 1
    assert q.pending_count == 0 and len(store.records) == 1


def test_cycle_on_an_empty_queue_is_a_no_op():
    stats = make(DirtyQueue()).run_cycle()
    assert stats.sessions_consolidated == 0 and stats.seconds_spent == 0.0


def test_a_failing_session_is_requeued_not_dropped():
    """Losing a session's records is invisible at read time, so failure must retry."""
    q = DirtyQueue()
    q.mark_dirty("s1")

    def boom(turns, session, date):
        raise RuntimeError("model down")

    stats = make(q, extract=boom).run_cycle()
    assert stats.sessions_failed == 1 and stats.sessions_consolidated == 0
    assert q.pending_count == 1


def test_a_store_write_failure_also_requeues():
    q = DirtyQueue()
    q.mark_dirty("s1")
    stats = make(q, store=FakeStore(fail_on="fact-")).run_cycle()
    assert stats.sessions_failed == 1 and q.pending_count == 1


def test_one_bad_session_does_not_block_the_others():
    q = DirtyQueue()
    for s in ("good1", "bad", "good2"):
        q.mark_dirty(s)

    def extract(turns, session, date):
        if session == "bad":
            raise RuntimeError("nope")
        return [rec(f"fact-{session}", session)]

    stats = make(q, extract=extract).run_cycle()
    assert stats.sessions_consolidated == 2 and stats.sessions_failed == 1
    assert q.pending_count == 1            # only the bad one retries


def test_empty_session_consolidates_without_writing():
    q = DirtyQueue()
    q.mark_dirty("s1")
    store = FakeStore()
    stats = make(q, turns=lambda s: [], store=store).run_cycle()
    assert stats.sessions_consolidated == 1 and len(store.records) == 0


def test_commit_order_is_deterministic_despite_concurrency():
    """Two runs over the same dirty set must produce the same store, or a background worker
    silently destroys reproducibility."""
    def run():
        q = DirtyQueue()
        for i in range(8):
            q.mark_dirty(f"s{i}")
        store = FakeStore()

        def extract(turns, session, date):
            time.sleep(0.005 if session == "s0" else 0.0)   # s0 finishes last
            return [rec(f"fact-{session}", session)]

        make(q, extract=extract, store=store).run_cycle()
        return [r.text for r in store.records]

    assert run() == run()


def test_cycle_limit_leaves_the_rest_pending():
    q = DirtyQueue()
    for i in range(5):
        q.mark_dirty(f"s{i}")
    make(q).run_cycle(limit=2)
    assert q.pending_count == 3


# --- the merge pass ----------------------------------------------------------

def test_merge_is_skipped_when_not_configured():
    assert make(DirtyQueue()).run_merge() == 0


def test_merge_refuses_to_write_back_nothing():
    """Overwriting a populated store with an empty result is indistinguishable from erasing
    the user's memory."""
    store = FakeStore()
    store.records = [rec(f"f{i}") for i in range(5)]
    c = make(DirtyQueue(), store=store, merge=lambda recs: [])
    assert c.run_merge() == 5
    assert c.stats.merges_run == 0


def test_merge_runs_on_a_populated_store():
    store = FakeStore()
    store.records = [rec(f"f{i}") for i in range(5)]
    c = make(DirtyQueue(), store=store, merge=lambda recs: list(recs)[:4])
    assert c.run_merge() == 4 and c.stats.merges_run == 1


def test_merge_on_an_empty_store_is_a_no_op():
    c = make(DirtyQueue(), store=FakeStore(), merge=lambda recs: recs)
    assert c.run_merge() == 0


# --- run_merge write-back (v4.1 Path B) ---------------------------------------------

def test_run_merge_writes_back_new_records_and_supersedes_the_old(tmp_path):
    """Until v4.1 run_merge() computed `merged` and DISCARDED it -- its docstring claimed
    to refuse an empty write-back while no write-back existed to refuse. Consolidation was
    inert on the store no matter what the merge returned."""
    from aeon_py.records import Provenance, Record, RecordStore
    import numpy as np
    store = RecordStore(tmp_path / "r.atlas", dim=8)
    vec = np.zeros(8, dtype=np.float32); vec[0] = 1.0
    old_a = Record(kind="ITEM", text="three tops", bucket="ACQUISITION", subtype="clothing",
                   provenance=Provenance("s1", (0,)))
    old_b = Record(kind="ITEM", text="five tops", bucket="ACQUISITION", subtype="clothing",
                   provenance=Provenance("s2", (0,)))
    store.add(old_a, vec); store.add(old_b, vec)

    merged_record = Record(kind="ITEM", text="five tops (previously three)",
                           bucket="ACQUISITION", subtype="clothing",
                           provenance=Provenance("s2", (0,)))
    sc = SessionConsolidator(DirtyQueue(), fetch_session=lambda s: [],
                             extract=lambda *a: [], embed=lambda t: vec, store=store,
                             merge=lambda recs: [merged_record])
    assert sc.run_merge() == 1
    live = [r.text for r in store.all_records()]
    assert live == ["five tops (previously three)"]
    assert sc.stats.records_merged_in == 1
    assert sc.stats.records_superseded == 2


def test_run_merge_is_a_no_op_when_the_merge_changes_nothing(tmp_path):
    """An idempotent merge must not churn the store into a delete-and-rewrite."""
    from aeon_py.records import Provenance, Record, RecordStore
    import numpy as np
    store = RecordStore(tmp_path / "r.atlas", dim=8)
    vec = np.zeros(8, dtype=np.float32); vec[0] = 1.0
    r = Record(kind="FACT", text="unchanged", provenance=Provenance("s1", (0,)))
    store.add(r, vec)
    before = [x.node_id for x in store.all_records()]
    sc = SessionConsolidator(DirtyQueue(), fetch_session=lambda s: [], extract=lambda *a: [],
                             embed=lambda t: vec, store=store,
                             merge=lambda recs: list(recs))
    assert sc.run_merge() == 1
    assert [x.node_id for x in store.all_records()] == before
    assert sc.stats.records_superseded == 0


def test_run_merge_still_refuses_an_empty_result(tmp_path):
    """Overwriting a populated store with nothing is indistinguishable from erasing the
    user's memory. Now actually enforced, rather than only claimed."""
    from aeon_py.records import Provenance, Record, RecordStore
    import numpy as np
    store = RecordStore(tmp_path / "r.atlas", dim=8)
    vec = np.zeros(8, dtype=np.float32); vec[0] = 1.0
    store.add(Record(kind="FACT", text="keep", provenance=Provenance("s1", (0,))), vec)
    sc = SessionConsolidator(DirtyQueue(), fetch_session=lambda s: [], extract=lambda *a: [],
                             embed=lambda t: vec, store=store, merge=lambda recs: [])
    assert sc.run_merge() == 1
    assert [r.text for r in store.all_records()] == ["keep"]


# --- durable supersession edges (v4.1 Step 5) ---------------------------------------

def _tpg_store(tmp_path, dim=8):
    from aeon_py.records import RecordStore
    return RecordStore(tmp_path / "r.atlas", dim=dim)


def test_merge_writes_one_edge_per_resolved_supersession(tmp_path):
    """`run_merge()` is the one moment both node ids are in hand -- after it returns, the
    pairing is gone. This is what turns prose (`Record.supersedes`, 4-of-9 by the derived
    resolver) into a link that answers "what replaced this?" exactly."""
    import numpy as np
    from aeon_py.records import Provenance, Record
    from aeon_py.timeline import read_supersession_edges
    from aeon_py.trace import TraceGraph
    store = _tpg_store(tmp_path)
    trace = TraceGraph(tmp_path / "t.trace")
    vec = np.zeros(8, dtype=np.float32); vec[0] = 1.0
    old = Record(kind="ITEM", text="three tops from H&M", bucket="ACQUISITION",
                 subtype="clothing", provenance=Provenance("s1", (0,)))
    store.add(old, vec)
    survivor = Record(kind="ITEM", text="five tops from H&M", bucket="ACQUISITION",
                      subtype="clothing", supersedes="three tops from H&M",
                      provenance=Provenance("s2", (0,)))
    sc = SessionConsolidator(DirtyQueue(), fetch_session=lambda s: [], extract=lambda *a: [],
                             embed=lambda t: vec, store=store, merge=lambda r: [survivor],
                             trace=trace, tenant="acme")
    sc.run_merge()
    edges = read_supersession_edges(trace, tenant="acme")
    assert len(edges) == 1
    assert int(edges[0]["supersedes_id"]) == old.node_id
    assert int(edges[0]["atlas_id"]) == survivor.node_id
    assert sc.stats.edges_written == 1


def test_merge_without_a_trace_is_unchanged(tmp_path):
    """EQUIVALENCE GUARD. Every existing caller passes no trace, and their behaviour must not
    move -- the edge log is additive."""
    import numpy as np
    from aeon_py.records import Provenance, Record
    store = _tpg_store(tmp_path)
    vec = np.zeros(8, dtype=np.float32); vec[0] = 1.0
    old = Record(kind="ITEM", text="three tops", bucket="ACQUISITION", subtype="clothing",
                 provenance=Provenance("s1", (0,)))
    store.add(old, vec)
    survivor = Record(kind="ITEM", text="five tops", bucket="ACQUISITION", subtype="clothing",
                      supersedes="three tops", provenance=Provenance("s2", (0,)))
    sc = SessionConsolidator(DirtyQueue(), fetch_session=lambda s: [], extract=lambda *a: [],
                             embed=lambda t: vec, store=store, merge=lambda r: [survivor])
    assert sc.run_merge() == 1
    assert sc.stats.edges_written == 0
    assert [r.text for r in store.all_records()] == ["five tops"]


def test_unresolved_markers_are_counted_not_silently_dropped(tmp_path):
    """4 of 9 measured markers resolve. The other 5 must be observable in production, not a
    number in a docstring."""
    import numpy as np
    from aeon_py.records import Provenance, Record
    from aeon_py.trace import TraceGraph
    store = _tpg_store(tmp_path)
    trace = TraceGraph(tmp_path / "t.trace")
    vec = np.zeros(8, dtype=np.float32); vec[0] = 1.0
    store.add(Record(kind="ITEM", text="unrelated", bucket="MEDIA", subtype="film",
                     provenance=Provenance("s1", (0,))), vec)
    survivor = Record(kind="ITEM", text="new value", bucket="MEDIA", subtype="film",
                      supersedes="something never recorded",
                      provenance=Provenance("s2", (0,)))
    sc = SessionConsolidator(DirtyQueue(), fetch_session=lambda s: [], extract=lambda *a: [],
                             embed=lambda t: vec, store=store, merge=lambda r: [survivor],
                             trace=trace, tenant="acme")
    sc.run_merge()
    assert sc.stats.edges_written == 0
    assert sc.stats.edges_unresolved == 1


def test_dangling_marker_does_not_write_an_edge(tmp_path):
    """A dangling marker means the merge asserted a supersession it did not execute. Writing
    an edge for it would record a lineage that never happened."""
    import numpy as np
    from aeon_py.records import Provenance, Record
    from aeon_py.timeline import read_supersession_edges
    from aeon_py.trace import TraceGraph
    store = _tpg_store(tmp_path)
    trace = TraceGraph(tmp_path / "t.trace")
    vec = np.zeros(8, dtype=np.float32); vec[0] = 1.0
    still_live = Record(kind="ITEM", text="three tops", bucket="ACQUISITION",
                        subtype="clothing", provenance=Provenance("s1", (0,)))
    store.add(still_live, vec)
    survivor = Record(kind="ITEM", text="five tops", bucket="ACQUISITION", subtype="clothing",
                      supersedes="three tops", provenance=Provenance("s2", (0,)))
    # Both survive the merge -- the marker is asserted but not executed.
    sc = SessionConsolidator(DirtyQueue(), fetch_session=lambda s: [], extract=lambda *a: [],
                             embed=lambda t: vec, store=store,
                             merge=lambda r: [still_live, survivor],
                             trace=trace, tenant="acme")
    sc.run_merge()
    assert read_supersession_edges(trace, tenant="acme") == []
    assert sc.stats.edges_dangling == 1
