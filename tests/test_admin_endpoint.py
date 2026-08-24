"""
V4 Stage 4 tasks 2/7: POST /admin/promotions/{request_id}/execute
(server.py) -- the full FastAPI request cycle over a real, live control
plane, not just the execute_approved_promotion() unit level covered by
test_admin.py.

Postgres-backed -- same opt-in-locally / hard-fail-in-CI pattern as
test_control_plane.py/test_admin.py.
"""
import os

import pytest

DATABASE_URL = os.environ.get("AEON_CONTROL_PLANE_DATABASE_URL")
REQUIRE_DB_TESTS = os.environ.get("AEON_REQUIRE_DB_TESTS") == "1"

if not DATABASE_URL:
    if REQUIRE_DB_TESTS:
        pytest.fail(
            "AEON_REQUIRE_DB_TESTS=1 but AEON_CONTROL_PLANE_DATABASE_URL is "
            "unset -- CI must run these tests for real, not silently skip "
            "them.",
            pytrace=False,
        )
    pytest.skip(
        "AEON_CONTROL_PLANE_DATABASE_URL not set -- Postgres-backed admin "
        "endpoint tests are opt-in locally (see docker-compose.yml for a "
        "local Postgres).",
        allow_module_level=True,
    )

from datetime import datetime, timedelta, timezone

import numpy as np
from fastapi.testclient import TestClient

from aeon_py.client import AeonClient, decode_store_id
from aeon_py.control_plane.admin import AdminDB
from aeon_py.governance import AuditLog
from aeon_py.promotion import IdentifierCorpus, create_promotion_approval_request
from aeon_py.server import (
    app,
    get_admin_db,
    get_atlas_client,
    get_audit_log,
    get_current_user_id,
    get_identifier_corpus,
    get_require_code_verification,
    get_shared_atlas_client,
)

client = TestClient(app)


def _vec(seed: float) -> list:
    return np.full(768, seed, dtype=np.float32).tolist()


def _future(seconds: int = 3600):
    return datetime.now(timezone.utc) + timedelta(seconds=seconds)


@pytest.fixture
def admin_db():
    db = AdminDB(DATABASE_URL)
    yield db
    db.dispose()


@pytest.fixture
def wired_app(tmp_path, admin_db):
    """Overrides every dependency the endpoint touches with real,
    test-isolated instances -- a real AeonClient pair (not mocks, so the
    full C++ insert/set_scope/set_governance_id path actually runs), a
    tmp_path AuditLog, a permissive IdentifierCorpus, and the real
    AdminDB against the live test database. get_governance_db is left as
    the real DI (also hits the live database) to exercise the full
    Postgres-mirroring path too."""
    private = AeonClient(tmp_path / "private.atlas")
    shared = AeonClient(tmp_path / "shared.atlas")
    audit_log = AuditLog(tmp_path / "audit.jsonl")
    corpus = IdentifierCorpus(patterns=["nomatch"])

    # Save/restore, not clear() -- `app` is one shared FastAPI instance
    # across every test module in this process; test_server.py sets its
    # own module-level overrides that must survive this fixture running
    # (same discipline test_server.py's own
    # test_active_room_endpoint_routes_to_shared_store already uses for
    # exactly this reason).
    saved = dict(app.dependency_overrides)
    app.dependency_overrides[get_current_user_id] = lambda: "alice"
    app.dependency_overrides[get_atlas_client] = lambda: private
    app.dependency_overrides[get_shared_atlas_client] = lambda: shared
    app.dependency_overrides[get_admin_db] = lambda: admin_db
    app.dependency_overrides[get_audit_log] = lambda: audit_log
    app.dependency_overrides[get_identifier_corpus] = lambda: corpus
    try:
        yield {"private": private, "shared": shared, "audit_log": audit_log}
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(saved)


class TestExecutePromotionEndpoint:
    def test_succeeds_with_role_and_full_approval(self, wired_app, admin_db):
        private = wired_app["private"]
        shared = wired_app["shared"]

        node_id = private.atlas.insert(0, _vec(0.3), "clean text")
        req_id = create_promotion_approval_request(
            admin_db, source_node_id=node_id, dest_scope=0x4, subject_id="subject1",
            reason="testing", requested_by="alice", expires_at=_future(),
        )
        admin_db.grant_approval(request_id=req_id, approver="bob")
        admin_db.grant_approval(request_id=req_id, approver="carol")
        admin_db.grant_role(principal="alice", scope_mask=0x4, granted_by="root")

        response = client.post(f"/admin/promotions/{req_id}/execute")

        assert response.status_code == 200
        data = response.json()
        assert data["promoted_node_id"] is not None
        # store-discriminated -- string, not a bare number (models.py).
        assert isinstance(data["promoted_node_id"], str)
        assert shared.atlas.size() == 1

    def test_destination_embedding_overrides_source_vector(self, wired_app, admin_db):
        # Task 2's deferred destination-conditioned re-embedding, closed
        # as a caller-supplied vector threaded through as promote_fragment()'s
        # existing reembed_fn seam (models.py's PromotionExecuteRequest).
        private = wired_app["private"]
        shared = wired_app["shared"]

        node_id = private.atlas.insert(0, _vec(0.3), "clean text")
        req_id = create_promotion_approval_request(
            admin_db, source_node_id=node_id, dest_scope=0x4, subject_id="subject1",
            reason="testing", requested_by="alice", expires_at=_future(),
        )
        admin_db.grant_approval(request_id=req_id, approver="bob")
        admin_db.grant_approval(request_id=req_id, approver="carol")
        admin_db.grant_role(principal="alice", scope_mask=0x4, granted_by="root")

        dest_vec = _vec(0.9)
        response = client.post(
            f"/admin/promotions/{req_id}/execute",
            json={"destination_embedding": dest_vec},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["promoted_node_id"] is not None
        new_id, is_shared = decode_store_id(int(data["promoted_node_id"]))
        assert is_shared
        stored_vec = shared.atlas.get_node_centroid(new_id)
        assert np.allclose(stored_vec, dest_vec, atol=1e-5)
        source_vec = private.atlas.get_node_centroid(node_id)
        assert not np.allclose(stored_vec, source_vec, atol=1e-3)

    def test_require_code_verification_rejects_missing_verification(
        self, wired_app, admin_db
    ):
        # v4-plan.md Stage 4 task 3: when this deployment has
        # AEON_REQUIRE_CODE_VERIFICATION on, an execute call with no
        # verification result is rejected (200, promoted_node_id=None --
        # same shape as a classifier rejection), not minted.
        private = wired_app["private"]
        shared = wired_app["shared"]

        node_id = private.atlas.insert(0, _vec(0.5), "clean text")
        req_id = create_promotion_approval_request(
            admin_db, source_node_id=node_id, dest_scope=0x4, subject_id="subject1",
            reason="testing", requested_by="alice", expires_at=_future(),
        )
        admin_db.grant_approval(request_id=req_id, approver="bob")
        admin_db.grant_approval(request_id=req_id, approver="carol")
        admin_db.grant_role(principal="alice", scope_mask=0x4, granted_by="root")

        app.dependency_overrides[get_require_code_verification] = lambda: True
        try:
            response = client.post(f"/admin/promotions/{req_id}/execute")
        finally:
            del app.dependency_overrides[get_require_code_verification]

        assert response.status_code == 200
        assert response.json()["promoted_node_id"] is None
        assert shared.atlas.size() == 0
        # A rejection doesn't consume the four-eyes approval -- same
        # convention as a classifier rejection (execute_approved_promotion's
        # own doc comment); retrying once CI passes must still work.
        assert admin_db.get_request(req_id)["executed_at"] is None

    def test_require_code_verification_passes_with_passed_status(
        self, wired_app, admin_db
    ):
        private = wired_app["private"]
        shared = wired_app["shared"]

        node_id = private.atlas.insert(0, _vec(0.5), "clean text")
        req_id = create_promotion_approval_request(
            admin_db, source_node_id=node_id, dest_scope=0x4, subject_id="subject1",
            reason="testing", requested_by="alice", expires_at=_future(),
        )
        admin_db.grant_approval(request_id=req_id, approver="bob")
        admin_db.grant_approval(request_id=req_id, approver="carol")
        admin_db.grant_role(principal="alice", scope_mask=0x4, granted_by="root")

        app.dependency_overrides[get_require_code_verification] = lambda: True
        try:
            response = client.post(
                f"/admin/promotions/{req_id}/execute",
                json={"verification": {"status": "passed", "commit_sha": "abc123"}},
            )
        finally:
            del app.dependency_overrides[get_require_code_verification]

        assert response.status_code == 200
        assert response.json()["promoted_node_id"] is not None
        assert shared.atlas.size() == 1

    def test_403_when_caller_lacks_role_despite_full_approval(self, wired_app, admin_db):
        # Defense in depth: N approvers is not enough on its own -- the
        # CALLER triggering execution must independently hold the role.
        private = wired_app["private"]
        node_id = private.atlas.insert(0, _vec(0.4), "clean text")
        req_id = create_promotion_approval_request(
            admin_db, source_node_id=node_id, dest_scope=0x8, subject_id="subject1",
            reason="testing", requested_by="alice", expires_at=_future(),
        )
        admin_db.grant_approval(request_id=req_id, approver="bob")
        admin_db.grant_approval(request_id=req_id, approver="carol")
        # deliberately no grant_role() call for "alice" over scope 0x8

        response = client.post(f"/admin/promotions/{req_id}/execute")
        assert response.status_code == 403

    def test_409_without_enough_approvals(self, wired_app, admin_db):
        private = wired_app["private"]
        node_id = private.atlas.insert(0, _vec(0.5), "clean text")
        req_id = create_promotion_approval_request(
            admin_db, source_node_id=node_id, dest_scope=0x10, subject_id="subject1",
            reason="testing", requested_by="alice", expires_at=_future(),
        )
        admin_db.grant_approval(request_id=req_id, approver="bob")  # only 1
        admin_db.grant_role(principal="alice", scope_mask=0x10, granted_by="root")

        response = client.post(f"/admin/promotions/{req_id}/execute")
        assert response.status_code == 409

    def test_404_for_unknown_request(self, wired_app):
        response = client.post("/admin/promotions/999999999/execute")
        assert response.status_code == 404

    def test_503_when_identifier_corpus_is_not_configured(self, wired_app, admin_db):
        # Distinct from a classifier REJECTION (still a 200 with
        # promoted_node_id=None) -- an empty corpus means the deployment
        # isn't configured to promote anything at all, a standing
        # misconfiguration, not a per-fragment outcome.
        private = wired_app["private"]
        node_id = private.atlas.insert(0, _vec(0.65), "clean text")
        req_id = create_promotion_approval_request(
            admin_db, source_node_id=node_id, dest_scope=0x40, subject_id="subject1",
            reason="testing", requested_by="alice", expires_at=_future(),
        )
        admin_db.grant_approval(request_id=req_id, approver="bob")
        admin_db.grant_approval(request_id=req_id, approver="carol")
        admin_db.grant_role(principal="alice", scope_mask=0x40, granted_by="root")

        saved = dict(app.dependency_overrides)
        app.dependency_overrides[get_identifier_corpus] = lambda: IdentifierCorpus()
        try:
            response = client.post(f"/admin/promotions/{req_id}/execute")
        finally:
            app.dependency_overrides.clear()
            app.dependency_overrides.update(saved)

        assert response.status_code == 503
        # Not consumed -- the request is still executable once configured.
        assert admin_db.get_request(req_id)["executed_at"] is None

    def test_409_on_replay_after_success(self, wired_app, admin_db):
        private = wired_app["private"]
        node_id = private.atlas.insert(0, _vec(0.6), "clean text")
        req_id = create_promotion_approval_request(
            admin_db, source_node_id=node_id, dest_scope=0x20, subject_id="subject1",
            reason="testing", requested_by="alice", expires_at=_future(),
        )
        admin_db.grant_approval(request_id=req_id, approver="bob")
        admin_db.grant_approval(request_id=req_id, approver="carol")
        admin_db.grant_role(principal="alice", scope_mask=0x20, granted_by="root")

        first = client.post(f"/admin/promotions/{req_id}/execute")
        assert first.status_code == 200

        second = client.post(f"/admin/promotions/{req_id}/execute")
        assert second.status_code == 409
