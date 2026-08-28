"""HTTP-level multi-tenant isolation: two authenticated users, one server, zero crosstalk.

WHY THIS EXISTS SEPARATELY FROM `test_multitenant_records.py`. That suite proves the record
layer isolates when handed per-tenant stores. This one proves the SERVER actually hands them
per-tenant stores -- which is a different claim, and the one an operator cares about. Every
other server test in this repo overrides `get_current_user_id` with a single fixed identity,
so before this file no test ever drove two identities through the real dependency graph.

It runs against the real `app` with real `SessionManager` wiring; only the LLM and the
identity header are substituted, because a smoke test that mocks the session manager would
be testing the mock.
"""

import os
import shutil
import tempfile

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def isolated_server(monkeypatch):
    """A server whose record files land in a temp dir, with a stub LLM.

    `AEON_*` vars are read at IMPORT time and the getters are `lru_cache`d, so the env var is
    not enough on its own -- the caches must be cleared and the module constant patched. That
    landmine is documented in `dependencies.py` and is exactly the kind of thing this fixture
    exists to get right once.
    """
    from aeon_py import dependencies as deps
    from aeon_py.server import app

    tmp = tempfile.mkdtemp()
    records_dir = os.path.join(tmp, "records")
    monkeypatch.setattr(deps, "DEFAULT_RECORDS_DIR", records_dir)
    monkeypatch.setattr(deps, "DEFAULT_ATLAS_PATH", os.path.join(tmp, "atlas.aeon"))
    monkeypatch.setattr(deps, "DEFAULT_TRACE_PATH", os.path.join(tmp, "trace.bin"))
    for getter in (deps.get_atlas_client, deps.get_trace_manager, deps.get_session_manager,
                   deps.get_llm_provider, deps.get_shared_atlas_client):
        getter.cache_clear()

    class _StubLLM:
        def generate(self, prompt, system_prompt="", temperature=None):
            # Echo a marker the test can find, and nothing from any other tenant.
            yield "ack"

    monkeypatch.setattr(deps, "get_llm_provider", lambda: _StubLLM())

    current = {"user": "alice"}

    async def _identity():
        return current["user"]

    # SAVE AND RESTORE, never clear. Other suites install their own overrides at module
    # scope, and `app` is a process-wide singleton -- clearing wipes theirs and the damage
    # shows up as unrelated failures in whichever file happens to run next.
    saved = dict(app.dependency_overrides)
    # Pollution runs BOTH ways on a process-wide `app`. Other suites replace the session
    # manager and stores with mocks; this suite's whole point is to exercise the REAL
    # wiring, so those overrides are removed for its duration and restored after. A smoke
    # test running against someone else's mock proves nothing.
    for dep in (deps.get_session_manager, deps.get_atlas_client, deps.get_trace_manager,
                deps.get_shared_atlas_client, deps.get_llm_provider):
        app.dependency_overrides.pop(dep, None)
    app.dependency_overrides[deps.get_current_user_id] = _identity
    try:
        yield app, current, records_dir, deps
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(saved)
        for getter in (deps.get_atlas_client, deps.get_trace_manager,
                       deps.get_session_manager, deps.get_shared_atlas_client,
                       # `get_encoder` too: it is an lru_cache singleton, so a populated
                       # cache makes a later test's `patch("sentence_transformers....")`
                       # invisible -- the model is already built. That is the singleton
                       # working as intended, and the cost is that whoever warms it must
                       # clean up after themselves.
                       deps.get_encoder):
            getter.cache_clear()
        shutil.rmtree(tmp, ignore_errors=True)


def _chat(client, text):
    r = client.post("/chat", json={"text": text})
    assert r.status_code == 200, r.text
    return r


def test_two_users_write_to_separate_record_files(isolated_server):
    """The file-level boundary, asserted through HTTP rather than at the library API."""
    app, current, records_dir, deps = isolated_server
    with TestClient(app) as client:
        current["user"] = "alice"
        _chat(client, "my salary is 400000")
        current["user"] = "bob"
        _chat(client, "my salary is 90000")

        mgr = deps.get_session_manager()
        alice = mgr.get_store("alice")
        bob = mgr.get_store("bob")
        assert alice is not None and bob is not None
        assert alice.path != bob.path
        assert alice.path.name == "alice.atlas" and bob.path.name == "bob.atlas"


def test_one_users_records_never_reach_anothers_prompt(isolated_server):
    """The property that matters: isolation at the PROMPT, which is what reaches the model."""
    from aeon_py.compose import compose_from_store
    from aeon_py.records import Provenance, Record
    import numpy as np

    app, current, records_dir, deps = isolated_server
    with TestClient(app) as client:
        current["user"] = "alice"
        _chat(client, "hello")
        current["user"] = "bob"
        _chat(client, "hello")

        mgr = deps.get_session_manager()
        vec = np.zeros(768, dtype=np.float32)
        vec[0] = 1.0
        mgr.get_store("alice").add(
            Record(kind="FACT", text="alice salary is 400000",
                   provenance=Provenance("alice", (0,))), vec)
        mgr.get_store("bob").add(
            Record(kind="FACT", text="bob salary is 90000",
                   provenance=Provenance("bob", (0,))), vec)

        trace = deps.get_trace_manager()
        bob_prompt = compose_from_store(mgr.get_store("bob"), trace,
                                        "Question: what is my salary?")["prompt"]
        assert "90000" in bob_prompt
        assert "400000" not in bob_prompt
        assert "alice" not in bob_prompt


def test_each_users_turns_are_marked_dirty_under_their_own_id(isolated_server):
    """The write hook must enqueue the TENANT, since the worker resolves a store from it.
    Marking under the wrong id would consolidate one user's turns into another's file."""
    app, current, records_dir, deps = isolated_server
    with TestClient(app) as client:
        current["user"] = "alice"
        _chat(client, "hello from alice")
        current["user"] = "bob"
        _chat(client, "hello from bob")

        queue = deps.get_session_manager().dirty_queue
        # `__contains__` covers pending AND in-flight, so this does not race the worker
        # having already claimed a session.
        assert "alice" in queue and "bob" in queue


def test_trace_history_is_scoped_per_user(isolated_server):
    """Trace shares one file across tenants and isolates by session_id -- the convention that
    does NOT transfer to records. Asserted here so the difference is visible in one place."""
    app, current, records_dir, deps = isolated_server
    with TestClient(app) as client:
        current["user"] = "alice"
        _chat(client, "alice-only-marker")
        current["user"] = "bob"
        _chat(client, "bob-only-marker")

        trace = deps.get_trace_manager()
        bob_text = " ".join(ev.get("text", "") for ev in trace.get_history("bob", limit=50))
        assert "bob-only-marker" in bob_text
        assert "alice-only-marker" not in bob_text


def test_a_supplied_event_time_reaches_the_trace(isolated_server):
    """`event_time` is the caller's "when this happened" in epoch microseconds. It was read
    back correctly by two consumers all along and written by NOTHING in production, so
    session dates fell back to insertion wall-clock -- correct for live chat, wrong for
    imported history, and `Record.date` ultimately reads it."""
    app, current, records_dir, deps = isolated_server
    when = 1_684_000_000_000_000          # 2023-05-13, well before "now"
    with TestClient(app) as client:
        current["user"] = "alice"
        r = client.post("/chat", json={"text": "imported turn", "event_time": when})
        assert r.status_code == 200, r.text

        history = deps.get_trace_manager().get_history("alice", limit=20)
        assert any(int(ev.get("event_time", 0)) == when for ev in history)


def test_omitting_event_time_leaves_live_chat_unchanged(isolated_server):
    """0 is the documented 'unset' sentinel, not a real epoch value."""
    app, current, records_dir, deps = isolated_server
    with TestClient(app) as client:
        current["user"] = "bob"
        assert client.post("/chat", json={"text": "live turn"}).status_code == 200
        history = deps.get_trace_manager().get_history("bob", limit=20)
        assert history and all(int(ev.get("event_time", 0)) == 0 for ev in history)
        assert all(int(ev.get("timestamp", 0)) > 0 for ev in history)


def test_the_session_date_resolver_prefers_a_supplied_event_time(isolated_server):
    """The payoff: the resolver already preferred `event_time` and only ever saw 0."""
    app, current, records_dir, deps = isolated_server
    when = 1_684_000_000_000_000
    with TestClient(app) as client:
        current["user"] = "carol"
        client.post("/chat", json={"text": "imported", "event_time": when})
        resolve = deps.make_session_date_resolver(deps.get_trace_manager())
        assert resolve("carol") == "2023/05/13"
