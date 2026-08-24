"""
Aeon Episodic Trace — Python Wrapper over C++ mmap TraceManager.

This module provides the Python-facing TraceGraph API, backed by the C++ kernel's
mmap Trace Engine (a single stable trace_path, atomically swapped in place by
compaction -- v4-plan.md Stage 2's severe compaction-data-loss fix means this is
no longer a generation-suffixed filename). NetworkX has been REMOVED. All trace
storage is handled in C++ via the nanobind `core.TraceManager` binding.

Usage:
    from aeon_py.trace import TraceGraph

    trace = TraceGraph(path="memory/trace.bin")
    event_id = trace.add_event("session-1", "user", "Hello, world!")
    history = trace.get_history("session-1", limit=50)
"""

from __future__ import annotations

import logging
from enum import IntEnum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class EdgeType(IntEnum):
    """Mirrors aeon::EdgeType (schema.hpp) exactly -- raw values are what
    actually get stored in TraceEvent::edge_type via
    TraceManager.append_event()'s edge_type kwarg. Additive-only, same
    stability contract as the C++ side: values must never change meaning.

    First real caller: v4-plan.md Stage 2 task 4's admission-time
    near-duplicate detection, which records a REFINES edge (pointing at
    supersedes_id = the existing near-duplicate's id) instead of
    inserting a redundant Atlas node. Superseded the earlier placeholder
    CAUSAL-only version of this enum (nothing ever called it -- the one
    caller that needed it, context.py's concept-linking, was fixed by
    episodic prev_id adjacency instead, not an edge).
    """

    NONE = 0
    SUPERSEDES = 1
    REFINES = 2
    CONTRADICTS = 3
    REVOKES = 4
    MERGES_WITH = 5
    PROMOTED_FROM = 6


class ReasonCode(IntEnum):
    """Mirrors aeon::ReasonCode (schema.hpp) exactly. Additive-only, same
    stability contract as EdgeType."""

    UNSPECIFIED = 0
    CORRECTION = 1
    BUG_FIX_VERIFIED = 2
    DEPRECATED = 3
    POLICY_OR_REDACTION = 4
    CONSOLIDATED_BY_DREAMING = 5

# Import the C++ nanobind core module
try:
    from aeon_py import core as _core

    _HAS_CORE = True
except ImportError:
    _HAS_CORE = False
    logger.warning(
        "aeon_py.core not available — TraceGraph will run in stub mode. "
        "Build the C++ extension with `cmake --build build` first."
    )


class TraceGraph:
    """Episodic trace graph backed by C++ mmap TraceManager.

    All storage and indexing is performed in C++. This Python class is
    a thin ergonomic wrapper that:
      - Creates or opens the mmap trace file
      - Converts Python strings to C++ calls
      - Returns Python dicts from get_history()
    """

    # Role enum (mirrors C++ TraceRole)
    ROLE_USER = 0
    ROLE_SYSTEM = 1
    ROLE_CONCEPT = 2
    ROLE_SUMMARY = 3

    _ROLE_MAP = {
        "user": ROLE_USER,
        "system": ROLE_SYSTEM,
        "concept": ROLE_CONCEPT,
        "summary": ROLE_SUMMARY,
    }

    def __init__(self, path: Optional[str | Path] = None):
        """
        Args:
            path: File path for mmap-backed trace. None = in-memory only.
        """
        if not _HAS_CORE:
            self._manager = None
            logger.warning("TraceGraph running in stub mode (no C++ backend)")
            return

        if path is not None:
            self._manager = _core.TraceManager(str(path))
        else:
            self._manager = _core.TraceManager()

    def add_event(
        self,
        session_id: str,
        role: str,
        text: str,
        atlas_id: int = 0,
        embedding: Optional[list[float]] = None,
        edge_type: EdgeType = EdgeType.NONE,
        supersedes_id: int = 0,
        reason_code: ReasonCode = ReasonCode.UNSPECIFIED,
        event_time: int = 0,
    ) -> int:
        """Append an episodic event for a session.

        Args:
            session_id: Multi-tenant session UUID.
            role: One of "user", "system", "concept", "summary".
            text: Text preview (max 439 chars, truncated in C++).
            atlas_id: Linked Atlas concept node ID (0 if none).
            embedding: V4 Stage 2 task 3. Optional embedding vector for
                this event -- None (default) = not embedded, excluded
                from semantic_search(). The FIRST non-empty embedding
                ever appended to this trace file fixes its
                embedding_dim; a later mismatched size raises
                ValueError.
            edge_type: V4 Stage 1/2 task 4. This event's
                version/admission edge type -- EdgeType.NONE (default)
                means no edge.
            supersedes_id: Id edge_type relates to (0 = none). In
                practice always a store-encoded Atlas node id
                (encode_store_id(), client.py), not a TraceEvent id --
                both real callers (Stage 2 task 4's REFINES admission
                dedup, Stage 4 task 2's promote_fragment()'s
                PROMOTED_FROM) only ever have the Atlas node id on hand.
                See schema.hpp's TraceEvent doc comment (V4 STAGE 4 note)
                for the full reasoning.
            reason_code: Reason for the edge (ReasonCode.UNSPECIFIED if
                not applicable).
            event_time: V4 Stage 7 Track 2. Caller-supplied event time
                (epoch microseconds), distinct from the event's
                `timestamp` (always Aeon's own insertion wall-clock,
                unaffected by this parameter). 0 (default) = not
                supplied -- callers ordering events by "when this
                happened" (as opposed to "when Aeon recorded it")
                should treat `event_time == 0` as "fall back to
                timestamp," not as a real epoch value. Lets a caller
                backfilling historical content (a chat import, a game
                engine replaying past events) record when something
                actually happened.

        Returns:
            The new event's unique monotonic ID.
        """
        if self._manager is None:
            return 0

        role_int = self._ROLE_MAP.get(role.lower(), self.ROLE_USER)
        return self._manager.append_event(
            session_id,
            role_int,
            text,
            atlas_id,
            embedding or [],
            int(edge_type),
            supersedes_id,
            int(reason_code),
            event_time,
        )

    def semantic_search(
        self, query_embedding: list[float], top_k: int = 10
    ) -> list[dict]:
        """Semantic search over embedded trace events (v4-plan.md Stage 2
        task 3), via TraceBlockIndex's two-phase O(|V|/1024 + K*1024)
        search -- only events appended with a non-empty `embedding` (see
        add_event()) are indexed.

        Args:
            query_embedding: Must match this file's embedding_dim.
            top_k: Maximum results to return.

        Returns:
            List of event dicts (same shape as get_history()'s, minus
            prev_id/flags), sorted by descending similarity. Empty if no
            embedding has ever been appended to this file, or if
            query_embedding's length doesn't match embedding_dim.
        """
        if self._manager is None:
            return []
        return self._manager.semantic_search(query_embedding, top_k)

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of indexed embeddings, or 0 if none have been
        appended to this trace file yet."""
        if self._manager is None:
            return 0
        return self._manager.embedding_dim

    def get_history(
        self, session_id: str, limit: int = 100
    ) -> list[dict]:
        """Retrieve session history in reverse chronological order.

        Args:
            session_id: Session UUID.
            limit: Maximum events to return.

        Returns:
            List of event dicts with keys: id, prev_id, atlas_id,
            timestamp, role, flags, session_id, text.
        """
        if self._manager is None:
            return []

        return self._manager.get_history(session_id, limit)

    def compact(self) -> None:
        """Shadow compaction — defragment trace file."""
        if self._manager is not None:
            self._manager.compact()

    def has_session(self, session_id: str) -> bool:
        """Check if a session has any events."""
        if self._manager is None:
            return False
        return self._manager.has_session(session_id)

    def drop_session(self, session_id: str) -> bool:
        """Drop session tail pointer (NPC despawn cleanup)."""
        if self._manager is None:
            return False
        return self._manager.drop_session(session_id)

    @property
    def size(self) -> int:
        """Total event count (mmap + delta)."""
        if self._manager is None:
            return 0
        return self._manager.size()

    @property
    def mmap_event_count(self) -> int:
        """Event count in mmap file only."""
        if self._manager is None:
            return 0
        return self._manager.mmap_event_count()

    @property
    def delta_event_count(self) -> int:
        """Event count in delta buffer only."""
        if self._manager is None:
            return 0
        return self._manager.delta_event_count()
