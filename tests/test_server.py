import json

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from aeon_py.server import (
    app,
    get_current_user_id,
    get_session_manager,
    get_atlas_client,
    get_shared_atlas_client,
)
from aeon_py.client import NODE_ID_DELTA_MASK, SHARED_STORE_BIT, encode_store_id

client = TestClient(app)

# JS's Number.MAX_SAFE_INTEGER -- any store-discriminated id above this
# would silently round through a browser's JSON.parse if sent as a bare
# JSON number.
_JS_MAX_SAFE_INTEGER = 2**53 - 1

# --- Mock Dependencies ---
# Rewritten to match the CURRENT server.py (v4-plan.md Stage 0's real-auth
# rework): routes now depend on get_current_user_id/get_session_manager
# (from aeon_py.dependencies, re-exported via server.py's import) rather
# than the removed get_cognitive_loop/get_context_manager, and
# ctx.trace.get_history() (the real TraceGraph API) rather than the
# removed to_viz_json()/networkx .graph attribute.

def mock_get_current_user_id():
    return "test_user"

def mock_get_atlas_client():
    atlas = MagicMock()
    # Mock get_children returning a list of dict-like rows (server.py only
    # needs row['id'] access).
    atlas.get_children.return_value = [
        {'id': 200, 'preview': [1.0, 0.0, 0.0]}
    ]
    return atlas

# One user event, role=0 ("user", per trace.py's ROLE_USER) with no
# atlas_id -- deliberately NOT a concept event, so /state/atlas/active
# falls back to its documented root default (matches the original test's
# expectation of room_id == 0).
_MOCK_HISTORY = [
    {
        "id": 1,
        "prev_id": 0,
        "atlas_id": 0,
        "timestamp": 123.0,
        "role": 0,
        "flags": 0,
        "session_id": "test_user",
        "text": "Hello",
        "text_preview": "Hello",
    }
]

_mock_ctx = MagicMock()
_mock_ctx.trace.get_history.return_value = _MOCK_HISTORY


def _mock_chat(text, session_id=None, event_time=0):
    yield "Hello"
    yield " "
    yield "World"


_mock_loop = MagicMock()
_mock_loop.chat = _mock_chat


async def _mock_get_context(user_id):
    return _mock_ctx


async def _mock_get_loop(user_id):
    return _mock_loop


_mock_session_mgr = MagicMock()
_mock_session_mgr.get_context = _mock_get_context
_mock_session_mgr.get_loop = _mock_get_loop


def mock_get_session_manager():
    return _mock_session_mgr

# Override dependencies
app.dependency_overrides[get_current_user_id] = mock_get_current_user_id
app.dependency_overrides[get_session_manager] = mock_get_session_manager
app.dependency_overrides[get_atlas_client] = mock_get_atlas_client

# --- Tests ---

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_trace_endpoint():
    response = client.get("/state/trace")
    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    assert "edges" in data
    assert data["nodes"][0]["id"] == "1"
    assert data["nodes"][0]["type"] == "UserNode"
    # prev_id == 0 for the only event -> no edges recorded (server.py
    # skips prev_id == 0, the "no predecessor" sentinel).
    assert data["edges"] == []

def test_active_room_endpoint():
    response = client.get("/state/atlas/active")
    assert response.status_code == 200
    data = response.json()
    # ids are strings (models.py's NeighborInfo/ActiveRoomResponse,
    # advisor review v4-plan.md Stage 4) -- a store-discriminated id can
    # exceed JS's Number.MAX_SAFE_INTEGER, so these are never bare JSON
    # numbers.
    assert data["room_id"] == "0"  # Default fallback: no concept event in history
    assert len(data["neighbors"]) == 1
    assert data["neighbors"][0]["id"] == "200"

def test_chat_streaming():
    # TestClient doesn't fully support streaming verification same as live, but we can check the content
    response = client.post("/chat", json={"text": "Hi"})
    assert response.status_code == 200
    # SSE format
    assert "event: token" in response.text
    assert "data: Hello" in response.text
    assert "event: done" in response.text

def test_active_room_endpoint_routes_to_shared_store():
    """V4 Stage 4: a concept event whose atlas_id is store-discriminated as
    SHARED must route get_children() to the shared Atlas client, not the
    private one -- this is the real Atlas->Trace->Atlas crossing advisor
    flagged, exercised through the full FastAPI request cycle (not just
    ContextManager directly, see tests/test_stage4_stores.py for that
    level)."""
    shared_room_id = 77
    shared_ctx = MagicMock()
    shared_ctx.trace.get_history.return_value = [
        {
            "id": 1,
            "prev_id": 0,
            "atlas_id": encode_store_id(shared_room_id, is_shared=True),
            "timestamp": 123.0,
            "role": 2,  # ROLE_CONCEPT
            "flags": 0,
            "session_id": "test_user",
            "text": "shared concept",
            "text_preview": "shared concept",
        }
    ]

    async def mock_get_context(user_id):
        return shared_ctx

    shared_session_mgr = MagicMock()
    shared_session_mgr.get_context = mock_get_context

    mock_shared_atlas = MagicMock()
    mock_shared_atlas.get_children.return_value = [{'id': 300, 'preview': [0.0, 1.0, 0.0]}]

    # Temporary overrides, restored after this test so it doesn't leak into
    # the module-level ones the other tests here depend on.
    saved = dict(app.dependency_overrides)
    app.dependency_overrides[get_session_manager] = lambda: shared_session_mgr
    app.dependency_overrides[get_shared_atlas_client] = lambda: mock_shared_atlas
    try:
        response = client.get("/state/atlas/active")
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(saved)

    assert response.status_code == 200
    data = response.json()
    # room_id is the store-discriminated id, unchanged from what Trace held
    # -- sent as a string (models.py, advisor review v4-plan.md Stage 4).
    assert data["room_id"] == str(shared_room_id | SHARED_STORE_BIT)
    # get_children() was called on the SHARED client with the DECODED raw
    # id, never on the private one.
    mock_shared_atlas.get_children.assert_called_once_with(shared_room_id)
    assert len(data["neighbors"]) == 1
    # The neighbor id is re-encoded as shared too (children live in the
    # same store as their parent).
    assert data["neighbors"][0]["id"] == str(300 | SHARED_STORE_BIT)


def test_large_store_discriminated_ids_survive_as_exact_strings_not_js_numbers():
    """advisor review (v4-plan.md Stage 4): a delta-arena, shared-store id
    (both SHARED_STORE_BIT and NODE_ID_DELTA_MASK set) exceeds JS's
    Number.MAX_SAFE_INTEGER -- confirm the raw HTTP response body encodes
    it as a JSON STRING (quoted), not a bare number a browser's
    JSON.parse would silently round, and that round-tripping through
    Python's own json.loads preserves the exact value either way (a
    sanity check that the fix is real, not merely "the test happens to
    still pass because Python ints have no precision limit")."""
    big_room_id = encode_store_id(NODE_ID_DELTA_MASK | 42, is_shared=True)
    assert big_room_id > _JS_MAX_SAFE_INTEGER  # the scenario actually at risk

    big_ctx = MagicMock()
    big_ctx.trace.get_history.return_value = [
        {
            "id": 1, "prev_id": 0, "atlas_id": big_room_id, "timestamp": 123.0,
            "role": 2, "flags": 0, "session_id": "test_user",
            "text": "big id concept", "text_preview": "big id concept",
        }
    ]

    async def mock_get_context(user_id):
        return big_ctx

    big_session_mgr = MagicMock()
    big_session_mgr.get_context = mock_get_context

    mock_shared = MagicMock()
    mock_shared.get_children.return_value = [{'id': NODE_ID_DELTA_MASK | 7, 'preview': [0.0, 0.0, 0.0]}]

    saved = dict(app.dependency_overrides)
    app.dependency_overrides[get_session_manager] = lambda: big_session_mgr
    app.dependency_overrides[get_shared_atlas_client] = lambda: mock_shared
    try:
        response = client.get("/state/atlas/active")
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(saved)

    assert response.status_code == 200
    # Inspect the RAW response body text, not response.json() -- a quoted
    # JSON string ("123...") is the property under test; parsing it back
    # into a Python int (which has no precision limit) would hide exactly
    # the bug this test exists to catch.
    raw = response.text
    assert f'"room_id":"{big_room_id}"' in raw.replace(" ", "")
    assert f'"room_id":{big_room_id}' not in raw.replace(" ", "")  # never a bare number

    data = json.loads(raw)
    assert data["room_id"] == str(big_room_id)
    assert int(data["room_id"]) == big_room_id  # exact, no precision loss
