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
