import pytest
import asyncio
import numpy as np
import sys
from pathlib import Path
from unittest.mock import MagicMock

# --- ENVIRONMENT SETUP ---
# 1. Add shell path
sys.path.insert(0, str(Path(__file__).parents[1] / "shell"))

# 2. Mock 'aeon_py.core' BEFORE importing aeon_py
# This prevents the circular import/missing binary issue during unit tests
mock_core = MagicMock()
sys.modules["aeon_py.core"] = mock_core

# Import new components
from aeon_py.session import SessionManager
from aeon_py.context import ContextManager
from aeon_py.loop import CognitiveLoop

# --- Fixtures ---

@pytest.fixture
def mock_deps():
    atlas = MagicMock()
    llm = MagicMock()
    # SessionManager.get_context() calls trace.has_session() to decide
    # whether to warm-start a resumed session (session.py) -- every user in
    # these tests is a first-time session, so this must be False, not an
    # auto-generated (truthy) MagicMock, or get_context() would go on to
    # call ContextManager.warm_start() -> trace.get_history(), which on a
    # bare MagicMock returns another MagicMock and blows up trying to
    # iterate it.
    trace = MagicMock()
    trace.has_session.return_value = False
    return atlas, trace, llm

@pytest.fixture
def session_mgr(mock_deps):
    atlas, trace, llm = mock_deps
    # Low max_sessions to test eviction easily. v4-plan.md Stage 0: one
    # shared Atlas/TraceGraph across all sessions, isolated by session_id
    # (== user_id) -- not the per-user-JSON-snapshot design this fixture
    # previously constructed (storage_dir=...), which SessionManager's own
    # class docstring (session.py) says predates the current shared-store
    # architecture and was removed.
    return SessionManager(atlas, trace, llm, max_sessions=2)

# --- Tests ---

@pytest.mark.anyio
async def test_user_isolation(session_mgr, mock_deps):
    """Verify separate users' turns are recorded under distinct
    session_ids against the ONE shared trace. v4-plan.md Stage 0: isolation
    is by session_id, not by private per-user trace objects --
    SessionManager.get_context() (session.py) hands every user's
    ContextManager the SAME shared Atlas/TraceGraph instance; there is no
    per-user `.graph` to inspect directly anymore."""
    atlas, trace, llm = mock_deps
    atlas.query.return_value = np.array(
        [], dtype=[('id', 'u8'), ('similarity', 'f4'), ('preview', 'f4', (3,))]
    )
    # Architect.ingest() (process_turn() step 4) routes through
    # atlas.atlas.insert_delta() -- give it a concrete int so
    # encode_store_id() (v4-plan.md Stage 4) doesn't choke on a bare,
    # unconfigured MagicMock (same fix as test_phase5.py's
    # test_cognitive_loop_flow).
    atlas.atlas.insert_delta.return_value = 42

    ctx_a = await session_mgr.get_context("alice")
    ctx_b = await session_mgr.get_context("bob")

    assert ctx_a is not ctx_b
    assert ctx_a.trace is ctx_b.trace  # same shared TraceGraph

    vec = np.zeros(768, dtype=np.float32)
    ctx_a.process_turn("I am Alice", vec, session_id="alice")
    ctx_b.process_turn("I am Bob", vec, session_id="bob")

    user_event_calls = [
        c for c in trace.add_event.call_args_list if c.args[1] == "user"
    ]
    alice_call = next(c for c in user_event_calls if c.args[0] == "alice")
    bob_call = next(c for c in user_event_calls if c.args[0] == "bob")
    assert alice_call.args[2] == "I am Alice"
    assert bob_call.args[2] == "I am Bob"

@pytest.mark.anyio
async def test_session_persistence(session_mgr, mock_deps):
    """Verify eviction only drops the in-memory ContextManager wrapper --
    it never deletes or mutates the underlying shared Atlas/Trace data,
    which is already durable on its own (SessionManager's class docstring,
    session.py: "There is nothing to explicitly save or load here").
    Replaces the old per-user-JSON-snapshot persistence test, which tested
    a design that docstring says predates the current shared-store
    architecture and no longer exists (no `_save_session`/`storage_dir` on
    the current SessionManager)."""
    atlas, trace, llm = mock_deps

    ctx1 = await session_mgr.get_context("charlie")

    # Evict charlie by filling the (max_sessions=2) cache with other users.
    await session_mgr.get_context("u1")
    await session_mgr.get_context("u2")
    assert "charlie" not in session_mgr._active_sessions

    # Re-fetching charlie creates a NEW wrapper (the old one was dropped,
    # not persisted-and-reloaded) -- but nothing was ever deleted from the
    # shared trace/atlas; its data was durable the whole time.
    ctx2 = await session_mgr.get_context("charlie")
    assert ctx2 is not ctx1
    assert not trace.drop_session.called
    assert not atlas.drop_session.called

@pytest.mark.anyio
async def test_lru_eviction(session_mgr):
    """Verify LRU cache limits and eviction."""
    # mgr max_sessions = 2
    
    # Load 1
    await session_mgr.get_context("u1")
    assert len(session_mgr._active_sessions) == 1
    
    # Load 2
    await session_mgr.get_context("u2")
    assert len(session_mgr._active_sessions) == 2
    
    # Access u1 to make it recent (u2 becomes oldest)
    await session_mgr.get_context("u1")
    
    # Load 3 -> Should evict u2 (the least recently used if we didn't touch u1? Wait.)
    # OrderedDict: 
    # Insert u1: [u1]
    # Insert u2: [u1, u2]
    # Access u1: move_to_end -> [u2, u1]
    # Insert u3: Evict u2 (first item) -> [u1, u3]
    
    await session_mgr.get_context("u3")
    assert len(session_mgr._active_sessions) == 2
    assert "u2" not in session_mgr._active_sessions
    assert "u1" in session_mgr._active_sessions
    assert "u3" in session_mgr._active_sessions

@pytest.mark.anyio
async def test_input_validation(session_mgr):
    with pytest.raises(ValueError):
        await session_mgr.get_context("../bad_actor")
        
@pytest.mark.anyio
async def test_concurrency_lock(session_mgr):
    """Verify thread safety with gathering."""
    # Race to create same user
    results = await asyncio.gather(
        session_mgr.get_context("racer"),
        session_mgr.get_context("racer"),
        session_mgr.get_context("racer")
    )
    
    # All should be same object
    first = results[0]
    assert all(r is first for r in results)
    assert len(session_mgr._active_sessions) == 1
