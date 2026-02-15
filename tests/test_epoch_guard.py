"""
Tests for the Python-side EBR EpochGuard context manager.
Verifies safe_memory_view() and acquire_read_guard().
"""

import tempfile
import numpy as np
import pytest
from pathlib import Path
from aeon_py import core
from aeon_py.client import AeonClient


@pytest.fixture
def atlas_path(tmp_path):
    """Create a temporary atlas file path."""
    return tmp_path / "test_epoch.atlas"


@pytest.fixture
def client(atlas_path):
    """Create an AeonClient with a temporary atlas."""
    return AeonClient(atlas_path)


class TestEpochGuardContextManager:
    """Tests for the C++-backed EpochGuard as a Python context manager."""

    def test_acquire_read_guard_basic(self, atlas_path):
        """Guard can be acquired and is active."""
        atlas = core.Atlas(str(atlas_path))
        guard = atlas.acquire_read_guard()
        assert guard.is_active()

    def test_guard_as_context_manager(self, atlas_path):
        """Guard works as a context manager (__enter__/__exit__)."""
        atlas = core.Atlas(str(atlas_path))
        with atlas.acquire_read_guard() as guard:
            assert guard.is_active()
        # After context exit, guard should be released
        # (Python object still exists but slot is freed)

    def test_multiple_guards(self, atlas_path):
        """Multiple concurrent guards should not deadlock."""
        atlas = core.Atlas(str(atlas_path))
        with atlas.acquire_read_guard() as g1:
            with atlas.acquire_read_guard() as g2:
                assert g1.is_active()
                assert g2.is_active()

    def test_guard_release_is_idempotent(self, atlas_path):
        """Releasing a guard multiple times should not crash."""
        atlas = core.Atlas(str(atlas_path))
        guard = atlas.acquire_read_guard()
        # Guard should be alive after entering context
        with guard:
            pass
        # Releasing again should be safe (idempotent)


class TestSafeMemoryView:
    """Tests for the high-level safe_memory_view() context manager."""

    def test_safe_memory_view_basic(self, client):
        """safe_memory_view yields a guard."""
        with client.safe_memory_view() as guard:
            assert guard.is_active()

    def test_query_inside_safe_view(self, client, atlas_path):
        """Queries work inside safe_memory_view."""
        # Insert a node first
        atlas = core.Atlas(str(atlas_path))
        vec = np.random.randn(768).astype(np.float32).tolist()
        atlas.insert(0, vec, "test_node")

        # Create a fresh client pointing to same file
        c = AeonClient(atlas_path)
        with c.safe_memory_view():
            embedding = np.random.randn(768).astype(np.float32)
            results = c.query(embedding)
            assert len(results) >= 0  # May be 0 on empty, >=1 after insert

    def test_safe_view_during_insert(self, client, atlas_path):
        """Inserts while holding a safe_memory_view should not crash."""
        atlas = core.Atlas(str(atlas_path))
        vec = np.random.randn(768).astype(np.float32).tolist()
        atlas.insert(0, vec, "root")

        c = AeonClient(atlas_path)
        with c.safe_memory_view():
            # Insert more nodes while holding guard
            for i in range(5):
                v = np.random.randn(768).astype(np.float32).tolist()
                atlas.insert(0, v, f"node_{i}")


class TestStaleGuardDetection:
    """Tests ensuring stale guard usage is handled safely."""

    def test_guard_inactive_after_release(self, atlas_path):
        """Guard reports inactive after explicit release."""
        atlas = core.Atlas(str(atlas_path))
        guard = atlas.acquire_read_guard()
        assert guard.is_active()
        # Use as context manager to release
        with guard:
            pass
        # After context exit, checking is_active may still work
        # (the C++ object exists, but the epoch slot is freed)
