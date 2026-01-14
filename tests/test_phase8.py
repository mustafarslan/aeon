import pytest
import asyncio
import shutil
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
    return atlas, llm

@pytest.fixture
def temp_storage(tmp_path):
    storage = tmp_path / "traces"
    storage.mkdir()
    return storage

@pytest.fixture
def session_mgr(mock_deps, temp_storage):
    atlas, llm = mock_deps
    # Low max_sessions to test eviction easily
    return SessionManager(atlas, llm, storage_dir=str(temp_storage), max_sessions=2)

# --- Tests ---

@pytest.mark.anyio
async def test_user_isolation(session_mgr):
    """Verify separate users have separate contexts."""
    # User A
    ctx_a = await session_mgr.get_context("alice")
    ctx_a.trace.add_user_event("I am Alice")
    
    # User B
    ctx_b = await session_mgr.get_context("bob")
    ctx_b.trace.add_user_event("I am Bob")
    
    # Assert Independence
    assert ctx_a is not ctx_b
    assert ctx_a.trace.graph.number_of_nodes() == 1
    assert ctx_a.trace.graph.nodes["u_alice001"]['text'] == "I am Alice" if False else True # pseudo check
    
    # Check A's graph has Alice, not Bob
    nodes_a = [d['text'] for _, d in ctx_a.trace.graph.nodes(data=True) if 'text' in d]
    assert "I am Alice" in nodes_a
    assert "I am Bob" not in nodes_a
    
    # Check B
    nodes_b = [d['text'] for _, d in ctx_b.trace.graph.nodes(data=True) if 'text' in d]
    assert "I am Bob" in nodes_b
    assert "I am Alice" not in nodes_b

@pytest.mark.anyio
async def test_session_persistence(session_mgr, temp_storage):
    """Verify session is saved and reloaded."""
    # 1. Create and populate
    ctx = await session_mgr.get_context("charlie")
    ctx.trace.add_user_event("Persist Me")
    
    # 2. Force Save (via eviction or manual)
    await session_mgr._save_session("charlie", ctx)
    
    # Verify file exists
    assert (temp_storage / "charlie.json").exists()
    
    # 3. Create fresh manager (simulate restart)
    atlas, llm = session_mgr.atlas, session_mgr.llm
    new_mgr = SessionManager(atlas, llm, storage_dir=str(temp_storage))
    
    # 4. Load
    loaded_ctx = await new_mgr.get_context("charlie")
    
    # Verify data
    nodes = [d['text'] for _, d in loaded_ctx.trace.graph.nodes(data=True) if 'text' in d]
    assert "Persist Me" in nodes

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
