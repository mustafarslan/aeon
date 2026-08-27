"""
Background session consolidation — the write path's production wiring.

Ingest must stay at kernel speed. Extraction is a ~1.3s LLM call, so putting it on the ingest
path would destroy the write-latency story that is half of Aeon's product argument. Instead:

    ingest  ->  mark_dirty(session_id)      O(1), microseconds, no I/O, no LLM
    later   ->  DreamingWorker drains the queue in the background

Consolidation cost is real -- measured at ~36s for 46 sessions at 4-way concurrency -- but it
lives entirely off BOTH hot paths: the write path only enqueues, and the read path only reads
finished records. That is the property that makes a semantic layer affordable in a system
whose pitch is ultra-low latency.

WHY NOT `LLMSummarizer`: `dreamer.py` already has a pluggable summarizer interface, but it is
shaped `summarize(texts) -> (text, embedding)` -- summarise-N-into-1, for summarise-to-forget.
Consolidation is a different operation (a session becomes many typed records, and records
merge across sessions), so it is wired as its own cycle rather than forced through an
interface built for something else.

FAILURE HANDLING, stated precisely: entries are cleared only once their records are committed,
so a session that fails mid-consolidation is retried rather than silently dropped. Losing a
session's records is invisible at read time -- the answer is merely worse, never an error --
which is exactly the kind of data loss that never gets reported as a bug.

**This is NOT crash safety, and an earlier version of this docstring overstated it.** The queue
is an in-memory set: process death loses every pending entry, and `requeue_in_flight()` only
recovers from in-process failures such as a cancelled cycle. The kernel-aligned fix is to make
dirty state *derivable* rather than durable -- compare a session's latest Trace event against a
per-session consolidation watermark, so recovery is a rescan and needs no second write-ahead
structure alongside the one Trace already has. Recorded as the follow-up rather than papered
over.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

from .parallel import DEFAULT_MAX_WORKERS, parallel_map
from .records import Record

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationStats:
    sessions_consolidated: int = 0
    sessions_failed: int = 0
    records_written: int = 0
    merges_run: int = 0
    records_merged_in: int = 0
    records_superseded: int = 0
    # Turns the one-off "4 of 9 markers resolve" measurement into a production-observable
    # number rather than a claim in a docstring.
    edges_written: int = 0
    edges_unresolved: int = 0
    edges_dangling: int = 0
    seconds_spent: float = 0.0


class DirtyQueue:
    """Thread-safe set of sessions awaiting consolidation.

    A *set*, not a list: a session written to ten times before the worker wakes needs
    consolidating once, and a queue that grew per-write would make consolidation cost scale
    with write volume rather than with distinct sessions.
    """

    def __init__(self) -> None:
        self._pending: set[str] = set()
        self._in_flight: set[str] = set()
        self._lock = threading.Lock()

    def mark_dirty(self, session_id: str) -> None:
        """Called on the ingest path. Must stay O(1) with no I/O."""
        if not session_id:
            return
        with self._lock:
            self._pending.add(session_id)

    def claim(self, limit: Optional[int] = None) -> list[str]:
        """Take up to `limit` sessions for processing. Claimed sessions move to in-flight so
        a second worker cannot pick them up, but are NOT forgotten -- `release` decides
        whether they are done or must be retried."""
        with self._lock:
            take = sorted(self._pending)              # deterministic order
            if limit is not None:
                take = take[:limit]
            for s in take:
                self._pending.discard(s)
                self._in_flight.add(s)
            return take

    def release(self, session_id: str, *, succeeded: bool) -> None:
        with self._lock:
            self._in_flight.discard(session_id)
            if not succeeded:
                self._pending.add(session_id)          # retry rather than lose it

    def requeue_in_flight(self) -> int:
        """Recover after a crash or shutdown mid-cycle: anything still in flight was never
        confirmed, so it goes back to pending."""
        with self._lock:
            n = len(self._in_flight)
            self._pending |= self._in_flight
            self._in_flight.clear()
            return n

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    @property
    def in_flight_count(self) -> int:
        with self._lock:
            return len(self._in_flight)

    def __contains__(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._pending or session_id in self._in_flight


class ConsolidationWorker:
    """Background drain for the dirty queue, following `DreamingWorker`'s pattern exactly.

    WHY A COPY RATHER THAN A REUSE. `dreamer.py`'s own docstring says it is "deeply tied to
    private single-tenant semantics", and its operation is summarise-N-into-1 for
    summarise-to-forget. Consolidation is a different operation over a per-tenant store, so
    this borrows the thread/Event/start/stop shape and nothing else.

    NOTE ON THE PATTERN IT COPIES: `DreamingWorker` is itself never started anywhere in
    production -- it is a pattern, not a wired example. This is the first background worker
    the server actually runs, which is why `server.py` needed a startup hook at all.

    `requeue_in_flight()` runs on start so a process that died mid-cycle re-enqueues rather
    than dropping those sessions. That only covers in-process failure: `DirtyQueue` is an
    in-memory set, so a crash still loses pending entries -- the durable fix is a Trace
    watermark, which is recorded as future work rather than claimed here.
    """

    def __init__(self, consolidator: "SessionConsolidator", *,
                 interval_seconds: float = 30.0, batch: int = 4):
        self._consolidator = consolidator
        self._interval = interval_seconds
        self._batch = batch
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self, daemon: bool = True) -> None:
        if self._running:
            logger.warning("ConsolidationWorker already running")
            return
        self._consolidator.queue.requeue_in_flight()
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop,
                                        name="aeon-consolidation-worker", daemon=daemon)
        self._thread.start()
        logger.info("ConsolidationWorker started | interval=%.1fs batch=%d",
                    self._interval, self._batch)

    def stop(self, timeout: float = 10.0) -> None:
        if not self._running:
            return
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        self._running = False
        logger.info("ConsolidationWorker stopped")

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._consolidator.run_cycle(limit=self._batch)
            except Exception:
                # A failing cycle must never kill the worker -- the queue requeues on
                # failure, so the next tick retries rather than dropping the session.
                logger.exception("ConsolidationWorker cycle failed")
            self._stop_event.wait(timeout=self._interval)


class SessionConsolidator:
    """Drains the dirty queue: session turns -> typed records -> record store.

    Every collaborator is injected. Consolidation is the one component that touches the LLM,
    the trace and the store at once, and this project has repeatedly found that the expensive
    bugs are in the seams -- so the seams are made testable without a model.
    """

    def __init__(
        self,
        queue: DirtyQueue,
        *,
        fetch_session: Callable[[str], Sequence[dict]],
        extract: Callable[[Sequence[dict], str, str], list[Record]],
        embed: Callable[[str], Sequence[float]],
        store,
        session_date: Callable[[str], str] = lambda _s: "",
        merge: Optional[Callable[[Sequence[Record]], list[Record]]] = None,
        max_workers: int = DEFAULT_MAX_WORKERS,
        trace=None,
        tenant: str = "",
    ) -> None:
        self.queue = queue
        self._fetch_session = fetch_session
        self._extract = extract
        self._embed = embed
        self._store = store
        self._session_date = session_date
        self._merge = merge
        self._max_workers = max_workers
        # Optional so every existing caller and test fake keeps working untouched -- the
        # module's existing dependency-injection discipline.
        self._trace = trace
        self._tenant = tenant
        self.stats = ConsolidationStats()

    def run_cycle(self, limit: Optional[int] = None) -> ConsolidationStats:
        """Consolidate up to `limit` dirty sessions. Returns cumulative stats.

        Sessions are extracted concurrently but committed in input order, so two runs over the
        same dirty set produce the same record store -- reproducibility that a background
        worker would otherwise quietly destroy.
        """
        claimed = self.queue.claim(limit)
        if not claimed:
            return self.stats
        t0 = time.perf_counter()

        def work(session_id: str):
            turns = self._fetch_session(session_id)
            if not turns:
                return session_id, []
            return session_id, self._extract(turns, session_id,
                                             self._session_date(session_id))

        outcomes = parallel_map(
            work, claimed, max_workers=self._max_workers,
            # A session that fails is requeued, never dropped: losing records is invisible at
            # read time (the answer is just worse), the worst kind of silent data loss.
            on_error=lambda sid, exc: (sid, None),
        )

        for session_id, records in outcomes:
            if records is None:
                self.queue.release(session_id, succeeded=False)
                self.stats.sessions_failed += 1
                continue
            try:
                for rec in records:
                    self._store.add(rec, self._embed(rec.text))
                self.stats.records_written += len(records)
                self.stats.sessions_consolidated += 1
                self.queue.release(session_id, succeeded=True)
            except Exception:
                self.queue.release(session_id, succeeded=False)
                self.stats.sessions_failed += 1

        self.stats.seconds_spent += time.perf_counter() - t0
        return self.stats

    def run_merge(self) -> int:
        """Global normalisation across accumulated records -- the operation per-session
        extraction structurally cannot do, since accumulation is global and extraction is
        local. Returns the resulting record count.

        A merge that collapses the record set is REJECTED by the injected `merge` callable's
        own guard; this method additionally refuses to write back an empty result, because
        overwriting a populated store with nothing is indistinguishable from erasing the
        user's memory.
        """
        if self._merge is None:
            return 0
        current = self._store.all_records()
        if not current:
            return 0
        merged = self._merge(current)
        if not merged:
            return len(current)

        # THE WRITE-BACK. Until v4.1 this method computed `merged` and DISCARDED it --
        # the docstring above claimed to "refuse to write back an empty result" while no
        # write-back code existed to refuse. Consolidation was therefore inert on the
        # store no matter what the merge callable returned.
        #
        # Order matters and is add-then-supersede, never the reverse: a crash between the
        # two steps must leave the user with BOTH the old and the new record (a duplicate,
        # visible and recoverable) rather than neither (silent data loss). The same
        # reasoning the WAL's lock ordering encodes.
        #
        # Superseding is by node_id, so only records that came FROM the store are retired;
        # a merged record the model invented has `node_id is None` and can supersede
        # nothing. Records the merge left untouched are matched by identity of their
        # encoded form and kept as they are, so an idempotent merge is a no-op on disk
        # rather than a churn of delete-and-rewrite.
        kept = {r.encode() for r in merged}
        survivors = [r for r in current if r.encode() in kept]
        retired = [r for r in current if r.encode() not in kept and r.node_id is not None]
        fresh = [r for r in merged if r.encode() not in {x.encode() for x in current}]

        for record in fresh:
            self._store.add(record, self._embed(record.text))
        for record in retired:
            self._store.supersede(record)

        # THE DURABLE EDGE. This is the one moment both node ids are in hand: `retired` and
        # `fresh` are the two sides of the same merge decision, and after this method returns
        # the pairing is gone. Recording it here is what turns `Record.supersedes` -- prose,
        # unresolvable, 4-of-9 by the derived resolver -- into a link that answers "what
        # replaced this?" exactly.
        #
        # Written AFTER the supersede, preserving this method's existing crash argument: a
        # crash between them leaves a superseded record with no edge, which the derived
        # resolver can still recover, rather than an edge pointing at a live record.
        #
        # Dangling links are skipped, not written. A dangling marker means the merge asserted
        # a supersession it did not execute; writing an edge for it would record a lineage
        # that never happened.
        if self._trace is not None:
            from . import timeline as _timeline
            # Resolved against the FULL merged set, not just `fresh`: a marker pointing at a
            # record the merge KEPT is the dangling case, and it is only visible if that
            # record is in the candidate set. Passing `fresh` alone would silently
            # reclassify every dangling marker as unresolved and lose the signal.
            for link in _timeline.resolve_links(merged, retired):
                if link.predecessor is None:
                    self.stats.edges_unresolved += 1
                    continue
                if link.dangling:
                    self.stats.edges_dangling += 1
                    continue
                if link.successor.node_id is None or link.predecessor.node_id is None:
                    self.stats.edges_unresolved += 1
                    continue
                _timeline.write_supersession_edge(
                    self._trace, tenant=self._tenant,
                    survivor_node_id=link.successor.node_id,
                    retired_node_id=link.predecessor.node_id,
                    text=link.successor.text)
                self.stats.edges_written += 1

        self.stats.merges_run += 1
        self.stats.records_merged_in += len(fresh)
        self.stats.records_superseded += len(retired)
        return len(survivors) + len(fresh)
