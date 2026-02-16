"""
Aeon Episodic Trace — Python Wrapper over C++ mmap TraceManager.

This module provides the Python-facing TraceGraph API, backed by the C++ kernel's
mmap Trace Engine (trace_genN.bin). NetworkX has been REMOVED. All trace storage
is handled in C++ via the nanobind `core.TraceManager` binding.

Usage:
    from aeon_py.trace import TraceGraph

    trace = TraceGraph(path="memory/trace.bin")
    event_id = trace.add_event("session-1", "user", "Hello, world!")
    history = trace.get_history("session-1", limit=50)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

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
    ) -> int:
        """Append an episodic event for a session.

        Args:
            session_id: Multi-tenant session UUID.
            role: One of "user", "system", "concept", "summary".
            text: Text preview (max 439 chars, truncated in C++).
            atlas_id: Linked Atlas concept node ID (0 if none).

        Returns:
            The new event's unique monotonic ID.
        """
        if self._manager is None:
            return 0

        role_int = self._ROLE_MAP.get(role.lower(), self.ROLE_USER)
        return self._manager.append_event(session_id, role_int, text, atlas_id)

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
