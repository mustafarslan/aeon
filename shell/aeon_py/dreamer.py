"""
Aeon Memory OS — Dreaming Worker (Memory Consolidation Orchestrator).

The "Dreaming" process is the SDK-layer garbage collector for constrained
edge devices (iOS/Android/Robotics/IoT). It monitors Atlas storage pressure,
extracts older subgraphs, calls an LLM to generate a semantic summary,
and invokes the C++ kernel's branchless tombstoning + compact_mmap to
physically reclaim storage.

Architecture:
    ┌────────────────────────────────────────────────────┐
    │             DreamingWorker (Python)                 │
    │                                                    │
    │  1. Monitor: tombstone_count() / size() > threshold│
    │  2. Select oldest subgraph (node IDs)              │
    │  3. LLM summarize → 768-dim embedding             │
    │  4. C++ consolidate_subgraph() → branchless tomb   │
    │  5. C++ compact_mmap() → physical reclamation      │
    └────────────────────────────────────────────────────┘

Thread Safety:
    - DreamingWorker runs on a background thread via asyncio/threading.
    - All C++ calls release the GIL, so dream cycles do NOT block the
      main event loop.
    - compact_mmap() acquires an exclusive lock internally; callers should
      schedule this during idle windows (nighttime, screen-off, charging).

Platforms: iOS (Background App Refresh), Android (WorkManager/Idle),
           Linux Edge (cron/systemd timer), Robotics (idle loop).

Shared tier (v4-plan.md Stage 5 task 1): `consolidate_shared_scope()`
below is a SEPARATE, module-level entry point, not a `DreamingWorker`
method -- `DreamingWorker` is deeply tied to private single-tenant
semantics (file-size/tombstone-ratio triggers, arbitrary lowest-N-ids
candidate selection, no notion of scope or subject attribution at all).
Bolting scope/subject-aware clustering onto it would produce one class
with two unrelated candidate-selection strategies and a config object
half of whose fields don't apply to the shared-tier path. See that
function's own doc comment for what it does differently and why.
"""

import asyncio
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Protocol

import numpy as np

from .client import encode_store_id
from .trace import EdgeType, ReasonCode, TraceGraph

logger = logging.getLogger("aeon.dreaming")


# ===========================================================================
# LLM Summarizer Interface (Pluggable)
# ===========================================================================

class LLMSummarizer(ABC):
    """
    Abstract interface for the LLM that generates semantic summaries
    of old memory subgraphs during the Dreaming process.

    Implementations:
      - LocalSummarizer:  Ollama / llama.cpp on-device (edge)
      - CloudSummarizer:  OpenAI / Anthropic / enterprise API (cloud)
      - StubSummarizer:   Deterministic stub for benchmarking
    """

    @abstractmethod
    def summarize(self, texts: list[str]) -> tuple[str, np.ndarray]:
        """
        Summarize a list of text snippets into a single summary.

        Args:
            texts: List of metadata strings from the nodes to consolidate.

        Returns:
            Tuple of (summary_text, summary_embedding_768d).
        """
        ...


class StubSummarizer(LLMSummarizer):
    """
    Deterministic summarizer for benchmarking and testing.
    Produces a truncated concatenation and a normalized random embedding.
    """

    def summarize(self, texts: list[str]) -> tuple[str, np.ndarray]:
        # Concatenate and truncate to 255 chars (metadata field limit)
        combined = " | ".join(t for t in texts if t)
        summary_text = combined[:250] + "..." if len(combined) > 250 else combined

        # Deterministic embedding: hash-seeded for reproducibility
        seed = hash(summary_text) % (2**31)
        rng = np.random.RandomState(seed)
        vec = rng.randn(768).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-12  # L2 normalize
        return summary_text, vec


# ===========================================================================
# Dreaming Configuration
# ===========================================================================

@dataclass
class DreamConfig:
    """Configuration for the DreamingWorker."""

    # --- Storage Pressure Thresholds ---
    memory_budget_mb: int = 128
    """Maximum allowable .bin file size in MB before triggering compaction."""

    tombstone_ratio_threshold: float = 0.25
    """Trigger compaction when tombstoned/total > this ratio."""

    min_nodes_to_consolidate: int = 10
    """Minimum number of nodes in a subgraph before consolidation."""

    max_nodes_per_dream_cycle: int = 500
    """Maximum nodes to consolidate in a single dream cycle."""

    # --- Scheduling ---
    check_interval_seconds: float = 30.0
    """How often to check storage pressure (seconds)."""

    idle_only: bool = True
    """If True, only dream when the system reports idle status."""

    # --- Compaction ---
    compact_after_consolidation: bool = True
    """Run compact_mmap() immediately after consolidation."""

    compact_temp_suffix: str = ".compact_tmp"
    """Suffix for the temporary compacted file."""


# ===========================================================================
# Dream Cycle Telemetry
# ===========================================================================

@dataclass
class DreamCycleReport:
    """Telemetry from a single dream cycle."""
    timestamp: float = 0.0
    nodes_consolidated: int = 0
    summary_node_id: int = 0
    file_size_before_mb: float = 0.0
    file_size_after_mb: float = 0.0
    storage_reclaimed_mb: float = 0.0
    compaction_duration_ms: float = 0.0
    total_duration_ms: float = 0.0
    tombstone_count_before: int = 0
    tombstone_count_after: int = 0


# ===========================================================================
# DreamingWorker — The Edge Memory GC
# ===========================================================================

class DreamingWorker:
    """
    Background Memory Consolidation Worker.

    Monitors Atlas storage pressure and orchestrates the Dreaming process:
      1. Detect pressure: file size > budget OR tombstone ratio > threshold.
      2. Select subgraph: oldest N nodes (by ID, which is insertion-order).
      3. Summarize: call the pluggable LLM summarizer.
      4. Consolidate: C++ consolidate_subgraph() — branchless tombstoning.
      5. Compact: C++ compact_mmap() — physical storage reclamation.

    Thread Safety:
      - All C++ calls release the Python GIL.
      - compact_mmap() acquires an exclusive write lock in C++.
      - The worker runs in a background thread and does NOT block the
        main event loop or Atlas query threads.
    """

    def __init__(
        self,
        atlas,
        atlas_path: Path | str,
        config: Optional[DreamConfig] = None,
        summarizer: Optional[LLMSummarizer] = None,
        node_text_extractor: Optional[Callable[[list[int]], list[str]]] = None,
    ):
        """
        Args:
            atlas:              The nanobind Atlas instance.
            atlas_path:         Path to the .bin mmap file.
            config:             DreamConfig (defaults if None).
            summarizer:         LLM summarizer (StubSummarizer if None).
            node_text_extractor: Callable that maps node IDs to their text.
                                 If None, uses empty strings (embedding-only).
        """
        self._atlas = atlas
        self._atlas_path = Path(atlas_path)
        self._config = config or DreamConfig()
        self._summarizer = summarizer or StubSummarizer()
        self._text_extractor = node_text_extractor or (lambda ids: [""] * len(ids))

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._cycle_history: list[DreamCycleReport] = []

        # Callbacks
        self._on_cycle_complete: Optional[Callable[[DreamCycleReport], None]] = None

    # --- Lifecycle ---

    def start(self, daemon: bool = True) -> None:
        """Start the background dreaming thread."""
        if self._running:
            logger.warning("DreamingWorker already running")
            return

        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="aeon-dreaming-worker",
            daemon=daemon,
        )
        self._thread.start()
        logger.info(
            "DreamingWorker started | budget=%dMB interval=%.1fs",
            self._config.memory_budget_mb,
            self._config.check_interval_seconds,
        )

    def stop(self, timeout: float = 10.0) -> None:
        """Signal the worker to stop and wait for completion."""
        if not self._running:
            return
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        self._running = False
        logger.info("DreamingWorker stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def cycle_history(self) -> list[DreamCycleReport]:
        return list(self._cycle_history)

    def on_cycle_complete(self, callback: Callable[[DreamCycleReport], None]) -> None:
        """Register a callback invoked after each dream cycle."""
        self._on_cycle_complete = callback

    # --- Core Loop ---

    def _run_loop(self) -> None:
        """Main background loop: check pressure → dream if needed."""
        while not self._stop_event.is_set():
            try:
                if self._should_dream():
                    report = self._execute_dream_cycle()
                    if report:
                        self._cycle_history.append(report)
                        if self._on_cycle_complete:
                            self._on_cycle_complete(report)
            except Exception:
                logger.exception("DreamingWorker cycle failed")

            self._stop_event.wait(timeout=self._config.check_interval_seconds)

    def _should_dream(self) -> bool:
        """Check if storage pressure warrants a dream cycle."""
        # Check file size
        try:
            file_size_mb = self._atlas_path.stat().st_size / (1024 * 1024)
        except OSError:
            return False

        if file_size_mb > self._config.memory_budget_mb:
            logger.info(
                "Storage pressure: %.1fMB > %dMB budget",
                file_size_mb, self._config.memory_budget_mb,
            )
            return True

        # Check tombstone ratio
        total = self._atlas.size()
        if total == 0:
            return False

        tombstones = self._atlas.tombstone_count()
        ratio = tombstones / total
        if ratio > self._config.tombstone_ratio_threshold:
            logger.info(
                "Tombstone pressure: %.1f%% > %.1f%% threshold",
                ratio * 100, self._config.tombstone_ratio_threshold * 100,
            )
            return True

        return False

    # --- Dream Cycle ---

    def _execute_dream_cycle(self) -> Optional[DreamCycleReport]:
        """Execute a single dream cycle: select → summarize → consolidate → compact."""
        t_start = time.monotonic()
        report = DreamCycleReport(timestamp=time.time())

        # 1. Snapshot metrics
        report.tombstone_count_before = self._atlas.tombstone_count()
        try:
            report.file_size_before_mb = self._atlas_path.stat().st_size / (1024 * 1024)
        except OSError:
            report.file_size_before_mb = 0.0

        # 2. Select oldest non-tombstoned nodes for consolidation
        #    We use the lowest node IDs (insertion order = chronological order).
        total_nodes = self._atlas.size()
        if total_nodes < self._config.min_nodes_to_consolidate:
            logger.debug("Too few nodes (%d) for consolidation", total_nodes)
            return None

        # Select the oldest N nodes (skip node 0 = root)
        n = min(self._config.max_nodes_per_dream_cycle, total_nodes - 1)
        old_ids = list(range(1, 1 + n))

        # 3. Extract text and summarize
        texts = self._text_extractor(old_ids)
        summary_text, summary_embedding = self._summarizer.summarize(texts)

        # 4. Consolidate subgraph (C++ — releases GIL)
        try:
            summary_id = self._atlas.consolidate_subgraph(
                old_ids,
                summary_embedding.tolist(),
                summary_text,
            )
            report.summary_node_id = summary_id
            report.nodes_consolidated = n
        except Exception as e:
            logger.error("Consolidation failed: %s", e)
            return None

        # 5. Compact mmap (C++ — releases GIL, exclusive lock)
        if self._config.compact_after_consolidation:
            compact_path = str(self._atlas_path) + self._config.compact_temp_suffix
            t_compact = time.monotonic()
            try:
                self._atlas.compact_mmap(compact_path)
            except Exception as e:
                logger.error("Compaction failed: %s", e)
            report.compaction_duration_ms = (time.monotonic() - t_compact) * 1000

        # 6. Post-cycle metrics
        report.tombstone_count_after = self._atlas.tombstone_count()
        try:
            report.file_size_after_mb = self._atlas_path.stat().st_size / (1024 * 1024)
        except OSError:
            report.file_size_after_mb = report.file_size_before_mb

        report.storage_reclaimed_mb = max(
            0, report.file_size_before_mb - report.file_size_after_mb
        )
        report.total_duration_ms = (time.monotonic() - t_start) * 1000

        logger.info(
            "Dream cycle complete | "
            "consolidated=%d nodes → summary_id=%d | "
            "storage: %.1fMB → %.1fMB (reclaimed %.1fMB) | "
            "duration=%.1fms",
            report.nodes_consolidated,
            report.summary_node_id,
            report.file_size_before_mb,
            report.file_size_after_mb,
            report.storage_reclaimed_mb,
            report.total_duration_ms,
        )

        return report

    # --- Manual API ---

    def dream_now(self) -> Optional[DreamCycleReport]:
        """
        Synchronously execute a dream cycle immediately.
        For programmatic use (e.g., iOS Background App Refresh handler,
        Android WorkManager, or game engine idle callback).
        """
        report = self._execute_dream_cycle()
        if report:
            self._cycle_history.append(report)
            if self._on_cycle_complete:
                self._on_cycle_complete(report)
        return report

    def summarize_and_consolidate(
        self, subgraph_ids: list[int]
    ) -> Optional[DreamCycleReport]:
        """
        Public API: summarize specific nodes and consolidate them.

        This is the explicit entry point for callers who know exactly
        which subgraph to consolidate (e.g., game engine scripts,
        robotics mission planners).

        Args:
            subgraph_ids: List of node IDs to consolidate.

        Returns:
            DreamCycleReport on success, None on failure.
        """
        if len(subgraph_ids) < 2:
            logger.warning("Need >= 2 nodes to consolidate")
            return None

        t_start = time.monotonic()
        report = DreamCycleReport(timestamp=time.time())

        try:
            report.file_size_before_mb = self._atlas_path.stat().st_size / (1024 * 1024)
        except OSError:
            report.file_size_before_mb = 0.0

        report.tombstone_count_before = self._atlas.tombstone_count()

        # Extract and summarize
        texts = self._text_extractor(subgraph_ids)
        summary_text, summary_embedding = self._summarizer.summarize(texts)

        # Consolidate
        try:
            summary_id = self._atlas.consolidate_subgraph(
                subgraph_ids,
                summary_embedding.tolist(),
                summary_text,
            )
            report.summary_node_id = summary_id
            report.nodes_consolidated = len(subgraph_ids)
        except Exception as e:
            logger.error("consolidate_subgraph failed: %s", e)
            return None

        # Compact
        if self._config.compact_after_consolidation:
            compact_path = str(self._atlas_path) + self._config.compact_temp_suffix
            t_compact = time.monotonic()
            try:
                self._atlas.compact_mmap(compact_path)
            except Exception as e:
                logger.error("compact_mmap failed: %s", e)
            report.compaction_duration_ms = (time.monotonic() - t_compact) * 1000

        # Post-metrics
        report.tombstone_count_after = self._atlas.tombstone_count()
        try:
            report.file_size_after_mb = self._atlas_path.stat().st_size / (1024 * 1024)
        except OSError:
            report.file_size_after_mb = report.file_size_before_mb

        report.storage_reclaimed_mb = max(
            0, report.file_size_before_mb - report.file_size_after_mb
        )
        report.total_duration_ms = (time.monotonic() - t_start) * 1000

        self._cycle_history.append(report)

        logger.info(
            "Manual consolidation | %d nodes → summary_id=%d | "
            "reclaimed=%.1fMB duration=%.1fms",
            report.nodes_consolidated,
            report.summary_node_id,
            report.storage_reclaimed_mb,
            report.total_duration_ms,
        )

        return report


# ═══════════════════════════════════════════════════════════════════════════
# Shared-tier Dreaming — scope/subject-aware clustering (V4 Stage 5 task 1)
# ═══════════════════════════════════════════════════════════════════════════

def _cosine_similarity(a: list, b: list) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-12
    return float(np.dot(va, vb) / denom)


def _cluster_by_similarity(
    candidates: list[tuple[int, list[float]]], threshold: float
) -> list[list[int]]:
    """Greedy single-pass clustering: each candidate joins the first
    existing cluster whose FIRST member's vector is within `threshold`
    cosine similarity, or starts a new cluster. Deterministic given input
    order -- no k-means/HDBSCAN dependency, "cluster related fragments"
    (v4-plan.md Stage 5 task 1's own example: fourteen tickets about the
    same flaky test) doesn't need more than this to be useful, and a
    simple, well-tested mechanism beats an opaque one for a storage-GC
    path that mutates data."""
    clusters: list[dict] = []
    for node_id, vec in candidates:
        placed = False
        for cluster in clusters:
            if _cosine_similarity(vec, cluster["vectors"][0]) >= threshold:
                cluster["ids"].append(node_id)
                cluster["vectors"].append(vec)
                placed = True
                break
        if not placed:
            clusters.append({"ids": [node_id], "vectors": [vec]})
    return [c["ids"] for c in clusters]


def consolidate_shared_scope(
    atlas,
    scope: int,
    summarizer: Optional[LLMSummarizer] = None,
    node_text_extractor: Optional[Callable[[list[int]], list[str]]] = None,
    subject_id_of: Optional[Callable[[int], Optional[str]]] = None,
    trace: Optional[TraceGraph] = None,
    actor: str = "dreaming-worker",
    similarity_threshold: float = 0.9,
    min_cluster_size: int = 2,
) -> list[DreamCycleReport]:
    """Point the Dreamer at the SHARED tier (v4-plan.md Stage 5 task 1):
    cluster related resolved fragments WITHIN one scope into summary
    fragments, linked back to their sources via MERGES_WITH Trace edges,
    reusing the same tombstone-and-summarize flow the private-tier path
    already performs via `consolidate_subgraph()`.

    Differs from `DreamingWorker`'s private-tier path in every dimension
    that actually matters here:

    - Candidate selection is `atlas.list_nodes_by_scope(scope)`, re-
      filtered to nodes whose OWN scope is EXACTLY `scope` (not merely
      overlapping it, which `list_nodes_by_scope()`'s own doc comment
      says it returns) -- consolidating a node scoped to `scope` together
      with one ALSO visible under other scopes would either violate the
      kernel's scope-uniformity precondition (`consolidate_subgraph()`,
      V4 Stage 5 task 1) if the bits differ, or (if it happened to match
      by accident) rely on that precondition rather than this function's
      own clustering being correct in the first place. The gate this
      function's own test proves is "clustering never PRODUCES a mixed-
      scope call" -- a Python-layer property, distinct from (and in
      addition to) the kernel's own unconditional rejection of one.
    - Candidates with any children of their own (`get_children_raw()`
      non-empty) are also excluded -- `consolidate_subgraph()`'s
      documented residual (its own doc comment, atlas.hpp) is that a
      NON-LEAF old_node_id's rewired grandchildren aren't registered
      under the new summary's own enumeration, making them structurally
      unreachable via `navigate()`. Verified today's only real shared-
      Atlas insertion path (`promote_fragment()`) always uses
      `parent_id=0`, so every real candidate is childless in practice --
      this filter turns that into an explicit, tested guarantee rather
      than an implicit assumption a future change could silently break.
    - Candidates are further grouped by `subject_id_of(node_id)` (a
      caller-supplied resolver, typically `governance_db.get_subject_id`
      composed with `atlas.get_node_governance_id`) BEFORE any similarity
      clustering -- task 6's one-subject-per-node invariant is invisible
      to the kernel (`consolidate_subgraph()`'s own doc comment: it has
      no notion of Postgres `subject_id`), so this Python layer is the
      ONLY place that can enforce it. A node whose subject_id resolves to
      None (never promoted through `promote_fragment()`, or the resolver
      itself is unavailable) is SKIPPED entirely, not grouped under a
      shared "unknown" key -- grouping unattributed nodes together would
      let the invariant be silently, retroactively violated the moment
      one of them later gains a real subject_id.
    - Clustering within each (scope, subject_id) group is threshold-based
      greedy grouping over centroid cosine similarity (`_cluster_by_
      similarity()`), not "every candidate in one blob" -- the private-
      tier path's `_execute_dream_cycle()` has no clustering step at all
      (it treats "the oldest N nodes" as a single group), which is fine
      for a single-tenant device GC but would not "cluster fourteen
      tickets about the same flaky test" the way this stage's plan text
      describes; unrelated fragments sharing a scope must NOT be forced
      into one summary.
    - Does NOT call `compact_mmap()` (advisor review): the private-tier
      path compacts because it is a storage-pressure GC running on a
      single device with no other observers. The shared tier's
      compaction has a much larger blast radius (every admin/console
      operation touching this store) and reassigns node ids -- doing it
      as an automatic side effect of a consolidation pass would silently
      invalidate the very MERGES_WITH edges this cycle just wrote
      (they carry `supersedes_id` pointing at the tombstoned source ids,
      which a subsequent compaction reclaims/renumbers). Left as a
      separate, deliberately operator-triggered call.
    - Returns a LIST of `DreamCycleReport` (one per cluster actually
      consolidated), not a single report -- one shared-scope cycle can
      produce multiple independent summaries. `compaction_duration_ms`/
      `file_size_*_mb` are always 0.0 on every report this function
      returns, since no compaction ever runs here.

    Args:
        atlas: The SHARED AeonClient/Atlas instance.
        scope: The single scope_bitmap value to consolidate within.
        summarizer: LLM summarizer (StubSummarizer if None).
        node_text_extractor: Callable mapping node ids to their text
            (empty strings if None, embedding-only).
        subject_id_of: Callable resolving a raw node id to its
            Postgres subject_id, or None if unresolvable/unattributed.
            None (the default) means every candidate is treated as
            unattributed and skipped -- callers consolidating REAL
            promoted content must supply a real resolver, or nothing
            will ever be consolidated (fail-closed, not fail-open, same
            discipline as promotion.py's IdentifierCorpus.is_empty()).
        trace: Optional -- if given, records one MERGES_WITH Trace event
            per consolidated source, pointing back at it from the new
            summary (mirrors promote_fragment()'s PROMOTED_FROM pattern).
        actor: Identity recorded as the Trace event's actor/session_id
            and this cycle's log context -- typically a fixed identity
            for the automated Dreaming process, not an end user.
        similarity_threshold: Cosine similarity a candidate must meet
            against a cluster's first member to join it.
        min_cluster_size: Clusters smaller than this are left alone (not
            worth consolidating a single fragment into a "summary" of
            one).

    Returns:
        One DreamCycleReport per cluster actually consolidated. Empty if
        no scope-matching, attributed candidates cluster to at least
        min_cluster_size.
    """
    summarizer = summarizer or StubSummarizer()
    text_extractor = node_text_extractor or (lambda ids: [""] * len(ids))

    candidate_ids = atlas.list_nodes_by_scope(scope)
    candidate_ids = [nid for nid in candidate_ids if atlas.get_node_scope(nid) == scope]

    # V4 Stage 5 task 1 verification (2026-08-23): consolidate_subgraph()'s
    # known, documented residual (atlas.hpp/v4-plan.md) is that when
    # old_node_ids are NON-LEAF nodes, Phase 3 correctly rewires their
    # surviving children's parent_offset to the new summary but never
    # registers those rewired grandchildren under the summary's OWN
    # child_count/first_child_offset -- they become structurally
    # unreachable via navigate() (though still visible via a flat
    # list_nodes_by_scope() scan). Verified today's ONLY real insertion
    # path into a shared Atlas (promote_fragment(), promotion.py) always
    # uses parent_id=0 -- every promoted node, and every summary this
    # function itself creates, is childless by construction, so this gap
    # is unexercised in practice. Rather than leave that as an implicit
    # assumption a future change could silently violate, candidates with
    # ANY children are explicitly excluded here -- skipped, not grouped,
    # same "don't silently risk it" discipline as the unattributed-
    # subject-id skip below.
    candidate_ids = [
        nid for nid in candidate_ids if len(atlas.get_children_raw(nid)) == 0
    ]

    groups: dict = {}
    for node_id in candidate_ids:
        subject_id = subject_id_of(node_id) if subject_id_of is not None else None
        if subject_id is None:
            continue
        groups.setdefault(subject_id, []).append(node_id)

    reports: list[DreamCycleReport] = []
    for subject_id, ids in groups.items():
        vectors = [atlas.get_node_centroid(nid) for nid in ids]
        clusters = _cluster_by_similarity(list(zip(ids, vectors)), similarity_threshold)

        for cluster_ids in clusters:
            if len(cluster_ids) < min_cluster_size:
                continue

            texts = text_extractor(cluster_ids)
            summary_text, summary_embedding = summarizer.summarize(texts)

            try:
                summary_id = atlas.consolidate_subgraph(
                    cluster_ids, summary_embedding.tolist(), summary_text,
                )
            except Exception as e:
                logger.error(
                    "consolidate_shared_scope: consolidation failed for "
                    "scope=%d subject_id=%s cluster=%s: %s",
                    scope, subject_id, cluster_ids, e,
                )
                continue

            if trace is not None:
                encoded_summary = encode_store_id(summary_id, is_shared=True)
                for source_id in cluster_ids:
                    trace.add_event(
                        actor,
                        "concept",
                        f"[Merged {source_id} into summary {summary_id}]",
                        atlas_id=encoded_summary,
                        edge_type=EdgeType.MERGES_WITH,
                        supersedes_id=encode_store_id(source_id, is_shared=True),
                        reason_code=ReasonCode.CONSOLIDATED_BY_DREAMING,
                    )

            reports.append(
                DreamCycleReport(
                    timestamp=time.time(),
                    nodes_consolidated=len(cluster_ids),
                    summary_node_id=summary_id,
                )
            )
            logger.info(
                "consolidate_shared_scope: scope=%d subject_id=%s merged %d "
                "nodes -> summary_id=%d",
                scope, subject_id, len(cluster_ids), summary_id,
            )

    return reports
