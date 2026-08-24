"""
V4 Stage 4 task 5: the minimum admin console's HTTP surface (server.py) --
audit log, knowledge browser, erasure workflow. Full FastAPI request cycle
over a real, live control plane, same shape as test_admin_endpoint.py
(promotion execute).

Postgres-backed -- same opt-in-locally / hard-fail-in-CI pattern as every
other control-plane test file.
"""
import json
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
        "AEON_CONTROL_PLANE_DATABASE_URL not set -- Postgres-backed console "
        "endpoint tests are opt-in locally (see docker-compose.yml for a "
        "local Postgres).",
        allow_module_level=True,
    )

import numpy as np
from fastapi.testclient import TestClient

from aeon_py.client import AeonClient
from aeon_py.control_plane.admin import AdminDB
from aeon_py.control_plane.erasure_db import ErasureDB
from aeon_py.governance import AuditLog
from aeon_py.server import (
    app,
    get_admin_db,
    get_atlas_client,
    get_audit_log,
    get_audit_log_export_key,
    get_current_user_id,
    get_erasure_db,
    get_governance_db,
    get_shared_atlas_client,
)

client = TestClient(app)

# A principal never granted anything anywhere in this test session --
# distinct from "alice" (reused, and cumulatively granted things, across
# test_admin_endpoint.py) so 403/"no role" assertions don't depend on
# cross-file test execution order.
NOBODY = "console_nobody"


def _vec(seed: float) -> list:
    return np.full(768, seed, dtype=np.float32).tolist()


@pytest.fixture
def admin_db():
    db = AdminDB(DATABASE_URL)
    yield db
    db.dispose()


@pytest.fixture
def erasure_db():
    db = ErasureDB(DATABASE_URL)
    yield db
    db.dispose()


@pytest.fixture
def wired_app(tmp_path, admin_db, erasure_db):
    shared = AeonClient(tmp_path / "shared.atlas")
    private = AeonClient(tmp_path / "private.atlas")
    audit_log = AuditLog(tmp_path / "audit.jsonl")

    saved = dict(app.dependency_overrides)
    app.dependency_overrides[get_current_user_id] = lambda: "console_admin"
    app.dependency_overrides[get_atlas_client] = lambda: private
    app.dependency_overrides[get_shared_atlas_client] = lambda: shared
    app.dependency_overrides[get_admin_db] = lambda: admin_db
    app.dependency_overrides[get_erasure_db] = lambda: erasure_db
    app.dependency_overrides[get_audit_log] = lambda: audit_log
    app.dependency_overrides[get_audit_log_export_key] = lambda: bytes.fromhex("aa" * 32)
    admin_db.grant_role(principal="console_admin", scope_mask=0x1000, granted_by="root")
    try:
        yield {"shared": shared, "private": private, "audit_log": audit_log}
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(saved)


def _as(principal):
    """Context-manager-free override swap for a single request as a
    different caller identity -- used for the 403/"no role" cases."""
    app.dependency_overrides[get_current_user_id] = lambda: principal


class TestAuditLogEndpoints:
    def test_tail_lists_appended_records(self, wired_app):
        audit_log = wired_app["audit_log"]
        audit_log.append("promotion", "someone", {"x": 1})
        audit_log.append("promotion", "someone", {"x": 2})

        response = client.get("/admin/audit-log")
        assert response.status_code == 200
        data = response.json()
        assert [r["seq"] for r in data["records"]] == [1, 2]
        assert data["next_since_seq"] == 2

    def test_tail_403_without_admin_role(self, wired_app):
        _as(NOBODY)
        response = client.get("/admin/audit-log")
        assert response.status_code == 403

    def test_verify_valid_on_untampered_log(self, wired_app):
        audit_log = wired_app["audit_log"]
        audit_log.append("promotion", "someone", {"x": 1})

        response = client.get("/admin/audit-log/verify")
        assert response.status_code == 200
        assert response.json() == {"valid": True, "error": None}

    def test_export_503_without_key_configured(self, wired_app):
        app.dependency_overrides[get_audit_log_export_key] = lambda: None
        response = client.get("/admin/audit-log/export")
        assert response.status_code == 503

    def test_export_returns_verifiable_signed_bytes(self, wired_app):
        from aeon_py.governance import AuditLog as AL

        audit_log = wired_app["audit_log"]
        audit_log.append("promotion", "someone", {"x": 1})

        response = client.get("/admin/audit-log/export")
        assert response.status_code == 200
        key = bytes.fromhex("aa" * 32)
        assert AL.verify_export_signature(response.content, key) is True


class TestKnowledgeBrowserEndpoints:
    def test_list_only_returns_nodes_within_callers_effective_scope(self, wired_app):
        shared = wired_app["shared"]
        visible = shared.atlas.insert(0, _vec(0.1), "visible")
        shared.atlas.set_node_scope(visible, 0x1000)  # overlaps console_admin's grant
        hidden = shared.atlas.insert(0, _vec(0.2), "hidden")
        shared.atlas.set_node_scope(hidden, 0x2000)  # does NOT overlap

        response = client.get("/admin/knowledge?reason=investigating+ticket+123")
        assert response.status_code == 200
        ids = {n["id"] for n in response.json()["nodes"]}
        from aeon_py.client import encode_store_id
        assert str(encode_store_id(visible, is_shared=True)) in ids
        assert str(encode_store_id(hidden, is_shared=True)) not in ids

    def test_list_pagination_offset_and_limit(self, wired_app):
        shared = wired_app["shared"]
        for i in range(5):
            nid = shared.atlas.insert(0, _vec(0.01 * i), f"node{i}")
            shared.atlas.set_node_scope(nid, 0x1000)

        first_page = client.get("/admin/knowledge?limit=2&offset=0&reason=paging+test")
        assert first_page.status_code == 200
        data = first_page.json()
        assert len(data["nodes"]) == 2
        assert data["total"] == 5

        second_page = client.get("/admin/knowledge?limit=2&offset=2&reason=paging+test")
        assert len(second_page.json()["nodes"]) == 2

        third_page = client.get("/admin/knowledge?limit=2&offset=4&reason=paging+test")
        assert len(third_page.json()["nodes"]) == 1

    def test_list_403_without_admin_role(self, wired_app):
        _as(NOBODY)
        response = client.get("/admin/knowledge?reason=testing")
        assert response.status_code == 403

    def test_list_422_when_reason_missing(self, wired_app):
        # reason has no default -- FastAPI's own request validation rejects
        # a missing query param before the handler body (and its own
        # blank-string guard) ever runs.
        response = client.get("/admin/knowledge")
        assert response.status_code == 422

    def test_list_400_when_reason_blank(self, wired_app):
        # v4-plan.md Stage 4 task 7: "mandatory read-reason prompts" --
        # an explicitly blank/whitespace reason is a distinct failure from
        # a missing one (caught by the handler's own guard, not FastAPI's
        # request validation), same shape as create_approval_request()'s
        # existing reason guard.
        response = client.get("/admin/knowledge?reason=%20%20")
        assert response.status_code == 400

    def test_list_records_reason_in_audit_chain_not_just_a_400_check(self, wired_app):
        # Proves the reason actually lands in the hash chain, not merely
        # that a bad request is rejected -- a test that only checks the
        # 400 above wouldn't discriminate this feature from a bare
        # query-param validator.
        shared = wired_app["shared"]
        audit_log = wired_app["audit_log"]
        node_id = shared.atlas.insert(0, _vec(0.15), "node")
        shared.atlas.set_node_scope(node_id, 0x1000)

        response = client.get("/admin/knowledge?reason=quarterly+access+review&limit=1")
        assert response.status_code == 200

        records = audit_log.tail()
        knowledge_reads = [r for r in records if r.action == "knowledge_read"]
        assert len(knowledge_reads) == 1
        assert knowledge_reads[0].actor == "console_admin"
        assert knowledge_reads[0].payload["reason"] == "quarterly access review"
        assert knowledge_reads[0].payload["returned_count"] == 1
        # No node text/metadata leaked into the audit payload.
        assert "node" not in json.dumps(knowledge_reads[0].payload)

        # The chain (now carrying this read record) still verifies.
        verify_resp = client.get("/admin/audit-log/verify")
        assert verify_resp.status_code == 200
        assert verify_resp.json()["valid"] is True

    def test_supersede_then_revoke_round_trips(self, wired_app):
        shared = wired_app["shared"]
        from aeon_py.client import encode_store_id
        node_id = shared.atlas.insert(0, _vec(0.3), "node")
        shared.atlas.set_node_scope(node_id, 0x1000)
        encoded = str(encode_store_id(node_id, is_shared=True))

        r1 = client.post(
            f"/admin/knowledge/{encoded}",
            json={"action": "supersede", "reason": "testing round-trip"},
        )
        assert r1.status_code == 200
        assert shared.atlas.is_node_superseded(node_id) is True

        r2 = client.post(
            f"/admin/knowledge/{encoded}",
            json={"action": "revoke_supersede", "reason": "testing round-trip"},
        )
        assert r2.status_code == 200
        assert shared.atlas.is_node_superseded(node_id) is False

    def test_supersede_400_when_reason_blank(self, wired_app):
        shared = wired_app["shared"]
        from aeon_py.client import encode_store_id
        node_id = shared.atlas.insert(0, _vec(0.31), "node")
        shared.atlas.set_node_scope(node_id, 0x1000)
        encoded = str(encode_store_id(node_id, is_shared=True))

        response = client.post(
            f"/admin/knowledge/{encoded}", json={"action": "supersede", "reason": "  "}
        )
        assert response.status_code == 400
        assert shared.atlas.is_node_superseded(node_id) is False

    def test_supersede_records_audit_entry(self, wired_app):
        # v4-plan.md Stage 5 task 2 retrofit: this route previously called
        # Atlas.supersede_node() with no audit trail at all.
        shared = wired_app["shared"]
        audit_log = wired_app["audit_log"]
        from aeon_py.client import encode_store_id
        node_id = shared.atlas.insert(0, _vec(0.32), "node")
        shared.atlas.set_node_scope(node_id, 0x1000)
        encoded = str(encode_store_id(node_id, is_shared=True))

        response = client.post(
            f"/admin/knowledge/{encoded}",
            json={"action": "supersede", "reason": "manual correction"},
        )
        assert response.status_code == 200

        tail = audit_log.tail()
        assert tail[-1].action == "supersession"
        assert tail[-1].payload["node_id"] == node_id
        assert tail[-1].payload["reason"] == "manual correction"

    def test_tombstone_action_via_endpoint(self, wired_app):
        shared = wired_app["shared"]
        from aeon_py.client import encode_store_id
        node_id = shared.atlas.insert(0, _vec(0.4), "node")
        shared.atlas.set_node_scope(node_id, 0x1000)
        encoded = str(encode_store_id(node_id, is_shared=True))

        response = client.post(f"/admin/knowledge/{encoded}", json={"action": "tombstone"})
        assert response.status_code == 200
        assert shared.atlas.tombstone_count() == 1

    def test_action_403_when_caller_lacks_role_for_nodes_scope(self, wired_app):
        shared = wired_app["shared"]
        from aeon_py.client import encode_store_id
        node_id = shared.atlas.insert(0, _vec(0.5), "node")
        shared.atlas.set_node_scope(node_id, 0x2000)  # NOT console_admin's granted scope
        encoded = str(encode_store_id(node_id, is_shared=True))

        response = client.post(f"/admin/knowledge/{encoded}", json={"action": "supersede"})
        assert response.status_code == 403

    def test_action_403_for_multi_scope_node_caller_only_partially_covers(self, wired_app):
        # Advisor-caught bug, now fixed: containment, not overlap.
        # console_admin holds ONLY 0x1000 -- a node ALSO in 0x2000 (which
        # they have no grant over) must be rejected even though 0x1000
        # overlaps. Before the fix, has_role()'s overlap check would have
        # authorized this.
        shared = wired_app["shared"]
        from aeon_py.client import encode_store_id
        node_id = shared.atlas.insert(0, _vec(0.55), "multi-scope node")
        shared.atlas.set_node_scope(node_id, 0x1000 | 0x2000)
        encoded = str(encode_store_id(node_id, is_shared=True))

        response = client.post(f"/admin/knowledge/{encoded}", json={"action": "tombstone"})
        assert response.status_code == 403
        assert shared.atlas.tombstone_count() == 0

    def test_rejects_private_store_id(self, wired_app):
        private = wired_app["private"]
        from aeon_py.client import encode_store_id
        node_id = private.atlas.insert(0, _vec(0.6), "private node")
        encoded = str(encode_store_id(node_id, is_shared=False))

        response = client.post(f"/admin/knowledge/{encoded}", json={"action": "tombstone"})
        assert response.status_code == 400

    def test_list_decrypts_encrypted_node_metadata(self, wired_app):
        # v4-plan.md Stage 4 task 6 Phase B: the ONLY shared-store metadata
        # read site in the entire shell (confirmed by grep, see the
        # decision record) -- must transparently decrypt a node minted
        # via promote_fragment(keystore=...), not surface base64 gibberish
        # to a real caller.
        import os as _os

        from aeon_py.control_plane.db import GovernanceDB
        from aeon_py.crypto import Keystore
        from aeon_py.promotion import IdentifierCorpus, promote_fragment
        from aeon_py.server import get_governance_db, get_keystore

        private = wired_app["private"]
        shared = wired_app["shared"]

        governance_db = GovernanceDB(DATABASE_URL)
        keystore = Keystore(DATABASE_URL, _os.urandom(32))
        app.dependency_overrides[get_governance_db] = lambda: governance_db
        app.dependency_overrides[get_keystore] = lambda: keystore
        try:
            node_id = private.atlas.insert(0, _vec(0.7), "contact alice@example.com")
            new_id = promote_fragment(
                private, node_id, shared,
                dest_scope=0x1000, corpus=IdentifierCorpus(redact_emails=True),
                audit_log=wired_app["audit_log"], actor="console_admin",
                subject_id=f"subject-{_os.urandom(4).hex()}",
                keystore=keystore, governance_db=governance_db,
            )
            # Confirm it's genuinely stored as ciphertext, not plaintext.
            from aeon_py.crypto import is_encrypted_metadata
            assert is_encrypted_metadata(shared.atlas.get_node_metadata(new_id)) is True

            response = client.get("/admin/knowledge?reason=testing+decryption")
            assert response.status_code == 200
            from aeon_py.client import encode_store_id
            encoded = str(encode_store_id(new_id, is_shared=True))
            node = next(n for n in response.json()["nodes"] if n["id"] == encoded)
            assert "alice@example.com" not in node["metadata"]
            assert "[REDACTED_EMAIL]" in node["metadata"]
        finally:
            governance_db.dispose()
            keystore.dispose()

    def test_partial_erasure_leaves_survivor_marker_prefixed_not_decrypted(
        self, wired_app, admin_db, erasure_db
    ):
        # v4-plan.md Stage 4 task 6 Phase B: erasure.py's own collateral-
        # effect comment says destroying a (subject_id, scope) DEK also
        # breaks any OTHER, still-live node sharing that pair -- this is a
        # DESIGNED, reachable consequence of erasing only SOME of a
        # subject's fragments in a scope, not a theoretical edge case.
        # Confirms the survivor comes back marker-prefixed ciphertext
        # (server.py's _read_metadata), not decrypted and not a crash.
        import os as _os
        from datetime import datetime, timedelta, timezone

        from aeon_py.control_plane.db import GovernanceDB
        from aeon_py.crypto import Keystore, is_encrypted_metadata
        from aeon_py.erasure import create_erasure_case, execute_approved_erasure
        from aeon_py.promotion import IdentifierCorpus, promote_fragment
        from aeon_py.server import get_governance_db, get_keystore

        private = wired_app["private"]
        shared = wired_app["shared"]
        subject_id = f"subject-{_os.urandom(4).hex()}"
        scope = 0x1000

        governance_db = GovernanceDB(DATABASE_URL)
        keystore = Keystore(DATABASE_URL, _os.urandom(32))
        app.dependency_overrides[get_governance_db] = lambda: governance_db
        app.dependency_overrides[get_keystore] = lambda: keystore
        try:
            node_a = private.atlas.insert(0, _vec(0.71), "fragment A -- subject's data")
            node_b = private.atlas.insert(0, _vec(0.72), "fragment B -- subject's data")
            corpus = IdentifierCorpus(patterns=["nomatch"])

            new_a = promote_fragment(
                private, node_a, shared, dest_scope=scope, corpus=corpus,
                audit_log=wired_app["audit_log"], actor="console_admin",
                subject_id=subject_id, keystore=keystore, governance_db=governance_db,
            )
            new_b = promote_fragment(
                private, node_b, shared, dest_scope=scope, corpus=corpus,
                audit_log=wired_app["audit_log"], actor="console_admin",
                subject_id=subject_id, keystore=keystore, governance_db=governance_db,
            )

            # Erase ONLY new_a -- new_b survives, but shares new_a's
            # (subject_id, scope) DEK.
            case_id = create_erasure_case(
                admin_db, erasure_db, shared_atlas=shared,
                node_ids=[new_a], reason="partial erasure test",
                requested_by="alice",
                expires_at=datetime.now(timezone.utc) + timedelta(seconds=3600),
            )
            req_id = erasure_db.get_case(case_id)["approval_request_id"]
            for approver in ("bob", "carol"):
                admin_db.grant_approval(request_id=req_id, approver=approver)
            execute_approved_erasure(
                admin_db, erasure_db, case_id, actor="alice", shared_atlas=shared,
                audit_log=wired_app["audit_log"], governance_db=governance_db,
                keystore=keystore,
            )

            # The key is gone, collaterally, for the survivor too.
            assert keystore.get_dek(subject_id, scope) is None

            response = client.get("/admin/knowledge?reason=confirming+collateral+effect")
            assert response.status_code == 200
            from aeon_py.client import encode_store_id
            encoded_b = str(encode_store_id(new_b, is_shared=True))
            node = next(n for n in response.json()["nodes"] if n["id"] == encoded_b)
            # Marker-prefixed ciphertext, not decrypted (no key) and not a
            # 500 (the endpoint degrades gracefully).
            assert is_encrypted_metadata(node["metadata"]) is True
            assert "fragment B" not in node["metadata"]
        finally:
            governance_db.dispose()
            keystore.dispose()


class TestSupersedeByCommitEndpoint:
    """v4-plan.md Stage 5 task 2: outcome-verified supersession's HTTP
    entry point."""

    def test_supersedes_every_node_citing_the_reverted_commit(self, wired_app):
        from aeon_py.promotion import IdentifierCorpus, VerificationResult, promote_fragment

        private = wired_app["private"]
        shared = wired_app["shared"]
        corpus = IdentifierCorpus(patterns=["nomatch"])

        n1 = private.atlas.insert(0, _vec(0.81), "clean text 1")
        new1 = promote_fragment(
            private, n1, shared, dest_scope=0x1000, corpus=corpus,
            audit_log=wired_app["audit_log"], actor="console_admin", subject_id="subject1",
            require_verification=True,
            verification=VerificationResult(status="passed", commit_sha="deadbeef"),
        )

        response = client.post(
            "/admin/supersede-by-commit", json={"commit_sha": "deadbeef"}
        )
        assert response.status_code == 200
        data = response.json()

        from aeon_py.client import encode_store_id
        assert str(encode_store_id(new1, is_shared=True)) in data["superseded"]
        assert data["could_not_supersede"] == []
        assert shared.atlas.is_node_superseded(new1) is True

    def test_unrelated_commit_returns_empty_receipt(self, wired_app):
        response = client.post(
            "/admin/supersede-by-commit", json={"commit_sha": "not-a-real-commit"}
        )
        assert response.status_code == 200
        assert response.json() == {"superseded": [], "could_not_supersede": []}

    def test_403_for_caller_with_no_admin_role_at_all(self, wired_app):
        _as(NOBODY)
        try:
            response = client.post(
                "/admin/supersede-by-commit", json={"commit_sha": "deadbeef"}
            )
            assert response.status_code == 403
        finally:
            _as("console_admin")

    def test_scope_containment_filters_unauthorized_nodes_into_receipt(self, wired_app):
        # console_admin only holds 0x1000 -- a node promoted into a scope
        # they have no grant over must land in could_not_supersede, not
        # be silently superseded.
        from aeon_py.promotion import IdentifierCorpus, VerificationResult, promote_fragment

        private = wired_app["private"]
        shared = wired_app["shared"]
        corpus = IdentifierCorpus(patterns=["nomatch"])

        n1 = private.atlas.insert(0, _vec(0.82), "clean text")
        new1 = promote_fragment(
            private, n1, shared, dest_scope=0x2000, corpus=corpus,
            audit_log=wired_app["audit_log"], actor="console_admin", subject_id="subject1",
            require_verification=True,
            verification=VerificationResult(status="passed", commit_sha="cafef00d"),
        )

        response = client.post(
            "/admin/supersede-by-commit", json={"commit_sha": "cafef00d"}
        )
        assert response.status_code == 200
        data = response.json()

        from aeon_py.client import encode_store_id
        encoded = str(encode_store_id(new1, is_shared=True))
        assert data["superseded"] == []
        assert any(entry["node_id"] == encoded for entry in data["could_not_supersede"])
        assert shared.atlas.is_node_superseded(new1) is False


class TestErasureEndpoints:
    def test_full_lifecycle_create_approve_execute(self, wired_app, admin_db, erasure_db):
        shared = wired_app["shared"]
        from aeon_py.client import encode_store_id
        node_id = shared.atlas.insert(0, _vec(0.7), "to erase")
        shared.atlas.set_node_scope(node_id, 0x1000)
        encoded = str(encode_store_id(node_id, is_shared=True))

        create_resp = client.post(
            "/admin/erasure",
            json={"node_ids": [encoded], "reason": "GDPR request #1"},
        )
        assert create_resp.status_code == 200
        case_id = create_resp.json()["case_id"]

        get_resp = client.get(f"/admin/erasure/{case_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["completed"] is False

        request_id = erasure_db.get_case(case_id)["approval_request_id"]
        admin_db.grant_approval(request_id=request_id, approver="bob")
        admin_db.grant_approval(request_id=request_id, approver="carol")

        exec_resp = client.post(f"/admin/erasure/{case_id}/execute")
        assert exec_resp.status_code == 200
        data = exec_resp.json()
        assert data["completed"] is True
        assert data["erased"] == [encoded]
        assert data["could_not_erase"] == []
        assert shared.atlas.tombstone_count() == 1

    def test_create_403_for_multi_scope_node_caller_only_partially_covers(self, wired_app):
        # Same advisor-caught containment-vs-overlap bug, erasure side --
        # worse there, since combined_scope ORs across ALL targets, so a
        # single overlapping grant would otherwise authorize a case
        # spanning scopes the caller never held at all.
        shared = wired_app["shared"]
        from aeon_py.client import encode_store_id
        node_id = shared.atlas.insert(0, _vec(0.65), "multi-scope node")
        shared.atlas.set_node_scope(node_id, 0x1000 | 0x2000)
        encoded = str(encode_store_id(node_id, is_shared=True))

        response = client.post(
            "/admin/erasure", json={"node_ids": [encoded], "reason": "testing"}
        )
        assert response.status_code == 403

    def test_create_400_for_empty_node_ids(self, wired_app):
        response = client.post(
            "/admin/erasure", json={"node_ids": [], "reason": "testing"}
        )
        assert response.status_code == 400

    def test_create_403_when_caller_lacks_role_for_target_scope(self, wired_app):
        shared = wired_app["shared"]
        from aeon_py.client import encode_store_id
        node_id = shared.atlas.insert(0, _vec(0.8), "node")
        shared.atlas.set_node_scope(node_id, 0x2000)  # NOT console_admin's granted scope
        encoded = str(encode_store_id(node_id, is_shared=True))

        response = client.post(
            "/admin/erasure", json={"node_ids": [encoded], "reason": "testing"}
        )
        assert response.status_code == 403

    def test_execute_409_without_enough_approvals(self, wired_app, admin_db, erasure_db):
        shared = wired_app["shared"]
        node_id = shared.atlas.insert(0, _vec(0.9), "node")
        shared.atlas.set_node_scope(node_id, 0x1000)
        from aeon_py.erasure import create_erasure_case
        from datetime import datetime, timedelta, timezone
        case_id = create_erasure_case(
            admin_db, erasure_db, shared_atlas=shared,
            node_ids=[node_id], reason="testing", requested_by="console_admin",
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=3600),
        )

        response = client.post(f"/admin/erasure/{case_id}/execute")
        assert response.status_code == 409

    def test_get_404_for_unknown_case(self, wired_app):
        response = client.get("/admin/erasure/999999999")
        assert response.status_code == 404
