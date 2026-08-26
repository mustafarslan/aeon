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

CRASH SAFETY: the queue records intent, and entries are only cleared once their records are
committed. A session interrupted mid-consolidation is retried rather than silently dropped --
losing a session's records is invisible at read time (the answer is merely worse), which is
exactly the kind of data loss that never gets reported as a bug.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

from .parallel import DEFAULT_MAX_WORKERS, parallel_map
from .records import Record


@dataclass
class ConsolidationStats:
    sessions_consolidated: int = 0
    sessions_failed: int = 0
    records_written: int = 0
    merges_run: int = 0
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
    ) -> None:
        self.queue = queue
        self._fetch_session = fetch_session
        self._extract = extract
        self._embed = embed
        self._store = store
        self._session_date = session_date
        self._merge = merge
        self._max_workers = max_workers
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
        self.stats.merges_run += 1
        return len(merged)
