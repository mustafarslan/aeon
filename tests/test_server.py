import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from aeon_py.server import app, get_cognitive_loop, get_context_manager, get_atlas_client
from aeon_py.models import TraceResponse

client = TestClient(app)

# --- Mock Dependencies ---

def mock_get_cognitive_loop():
    loop = MagicMock()
    # Mock chat generator
    def mock_chat(text):
        yield "Hello"
        yield " "
        yield "World"
    loop.chat = mock_chat
    return loop

def mock_get_context_manager():
    ctx = MagicMock()
    # Mock trace
    ctx.trace.to_viz_json.return_value = {
        "nodes": [{"id": "u1", "label": "User", "type": "UserNode", "timestamp": 123.0, "details": {}}],
        "edges": []
    }
    return ctx

def mock_get_atlas_client():
    atlas = MagicMock()
    # Mock get_children returning structured array (list of dicts for mock)
    # The server expects structured array or something iterable giving access to ['id']
    atlas.get_children.return_value = [
        {'id': 200, 'preview': [1.0, 0.0, 0.0]}
    ]
    return atlas

# Override dependencies
app.dependency_overrides[get_cognitive_loop] = mock_get_cognitive_loop
app.dependency_overrides[get_context_manager] = mock_get_context_manager
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
    assert data["nodes"][0]["id"] == "u1"

def test_active_room_endpoint():
    response = client.get("/state/atlas/active")
    assert response.status_code == 200
    data = response.json()
    assert data["room_id"] == 0 # Default fallback
    assert len(data["neighbors"]) == 1
    assert data["neighbors"][0]["id"] == 200

def test_chat_streaming():
    # TestClient doesn't fully support streaming verification same as live, but we can check the content
    response = client.post("/chat", json={"text": "Hi"})
    assert response.status_code == 200
    # SSE format
    assert "event: token" in response.text
    assert "data: Hello" in response.text
    assert "event: done" in response.text
