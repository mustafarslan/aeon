"""
V4 Stage 4 task 5(c): erasure workflow (erasure.py, control_plane/erasure_db.py).

Postgres-backed -- same opt-in-locally / hard-fail-in-CI pattern as
test_admin.py/test_control_plane.py (see either's module docstring for
the reasoning).
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
        "AEON_CONTROL_PLANE_DATABASE_URL not set -- Postgres-backed "
        "erasure-workflow tests are opt-in locally (see docker-compose.yml "
        "for a local Postgres).",
        allow_module_level=True,
    )

import json
from datetime import datetime, timedelta, timezone

import numpy as np

from aeon_py.client import AeonClient
from aeon_py.control_plane.admin import AdminDB
from aeon_py.control_plane.erasure_db import ErasureDB
from aeon_py.erasure import ErasureTransientFailure, create_erasure_case, execute_approved_erasure
from aeon_py.governance import AuditLog


def _vec(seed: float) -> list:
    return np.full(768, seed, dtype=np.float32).tolist()


def _future(seconds: int = 3600) -> datetime:
    return datetime.now(timezone.utc) + timedelta(seconds=seconds)


@pytest.fixture(scope="module")
def admin_db():
    db = AdminDB(DATABASE_URL)
    yield db
    db.dispose()


@pytest.fixture(scope="module")
def erasure_db():
    db = ErasureDB(DATABASE_URL)
    yield db
    db.dispose()


@pytest.fixture
def shared_atlas(tmp_path):
    return AeonClient(tmp_path / "shared.atlas")


@pytest.fixture
def audit_log(tmp_path):
    return AuditLog(tmp_path / "audit.jsonl")


class TestCreateErasureCase:
    def test_creates_case_and_locks_in_node_ids_and_scope(
        self, admin_db, erasure_db, shared_atlas
    ):
        n1 = shared_atlas.atlas.insert(0, _vec(0.1), "fragment one")
        n2 = shared_atlas.atlas.insert(0, _vec(0.2), "fragment two")
        shared_atlas.atlas.set_node_scope(n1, 0x1)
        shared_atlas.atlas.set_node_scope(n2, 0x2)

        case_id = create_erasure_case(
            admin_db, erasure_db, shared_atlas=shared_atlas,
            node_ids=[n1, n2], reason="GDPR erasure request, ticket #1",
            requested_by="alice", expires_at=_future(),
        )

        case = erasure_db.get_case(case_id)
        assert case["completed_at"] is None
        assert case["receipt"] is None

        req = admin_db.get_request(case["approval_request_id"])
        assert req["action"] == "erasure"
        target = json.loads(req["target"])
        assert set(target["node_ids"]) == {n1, n2}
        assert target["scope_mask"] == (0x1 | 0x2)

    def test_rejects_empty_node_ids(self, admin_db, erasure_db, shared_atlas):
        with pytest.raises(ValueError):
            create_erasure_case(
                admin_db, erasure_db, shared_atlas=shared_atlas,
                node_ids=[], reason="testing", requested_by="alice",
                expires_at=_future(),
            )


class TestExecuteApprovedErasure:
    def _approve_fully(self, admin_db, req_id, approvers=("bob", "carol")):
        for approver in approvers:
            admin_db.grant_approval(request_id=req_id, approver=approver)

    def _make_case(self, admin_db, erasure_db, shared_atlas, node_ids, scope=0x4):
        for nid in node_ids:
            shared_atlas.atlas.set_node_scope(nid, scope)
        case_id = create_erasure_case(
            admin_db, erasure_db, shared_atlas=shared_atlas,
            node_ids=node_ids, reason="testing", requested_by="alice",
            expires_at=_future(),
        )
        case = erasure_db.get_case(case_id)
        self._approve_fully(admin_db, case["approval_request_id"])
        return case_id

    def test_executes_erasure_tombstones_all_targets(
        self, admin_db, erasure_db, shared_atlas, audit_log
    ):
        n1 = shared_atlas.atlas.insert(0, _vec(0.3), "a")
        n2 = shared_atlas.atlas.insert(0, _vec(0.4), "b")
        case_id = self._make_case(admin_db, erasure_db, shared_atlas, [n1, n2])

        receipt = execute_approved_erasure(
            admin_db, erasure_db, case_id, actor="alice",
            shared_atlas=shared_atlas, audit_log=audit_log,
        )

        assert set(receipt["erased"]) == {n1, n2}
        assert receipt["could_not_erase"] == []
        assert shared_atlas.atlas.tombstone_count() == 2

        case = erasure_db.get_case(case_id)
        assert case["completed_at"] is not None
        assert json.loads(case["receipt"]) == receipt

    def test_records_could_not_erase_without_aborting_the_rest(
        self, admin_db, erasure_db, shared_atlas, audit_log
    ):
        n1 = shared_atlas.atlas.insert(0, _vec(0.5), "valid")
        invalid_id = 999999999
        case_id = self._make_case(admin_db, erasure_db, shared_atlas, [n1])
        # Hand-craft a case whose target ALSO names an invalid id -- the
        # normal create_erasure_case() path can't produce this (it would
        # fail computing scope_mask via get_node_scope() first), so this
        # exercises execute's own per-id fault isolation directly by
        # editing the locked-in target after the fact via a fresh request.
        case = erasure_db.get_case(case_id)
        req = admin_db.get_request(case["approval_request_id"])
        # Simulate node deletion-by-compaction between approval and
        # execution: re-point the SAME approved request's target at one
        # valid + one now-invalid id.
        from aeon_py.control_plane.schema import approval_requests
        with admin_db._engine.begin() as conn:
            conn.execute(
                approval_requests.update()
                .where(approval_requests.c.id == req["id"])
                .values(target=json.dumps(
                    {"node_ids": [n1, invalid_id], "scope_mask": 0x4}
                ))
            )

        receipt = execute_approved_erasure(
            admin_db, erasure_db, case_id, actor="alice",
            shared_atlas=shared_atlas, audit_log=audit_log,
        )

        assert receipt["erased"] == [n1]
        assert len(receipt["could_not_erase"]) == 1
        assert receipt["could_not_erase"][0]["node_id"] == invalid_id
        # Partial failure is still a legitimate completion, not a dangling case.
        assert erasure_db.get_case(case_id)["completed_at"] is not None

    def test_refuses_without_enough_approvals(
        self, admin_db, erasure_db, shared_atlas, audit_log
    ):
        n1 = shared_atlas.atlas.insert(0, _vec(0.6), "c")
        shared_atlas.atlas.set_node_scope(n1, 0x8)
        case_id = create_erasure_case(
            admin_db, erasure_db, shared_atlas=shared_atlas,
            node_ids=[n1], reason="testing", requested_by="alice",
            expires_at=_future(),
        )
        case = erasure_db.get_case(case_id)
        admin_db.grant_approval(request_id=case["approval_request_id"], approver="bob")  # only 1 of 2

        with pytest.raises(PermissionError):
            execute_approved_erasure(
                admin_db, erasure_db, case_id, actor="alice",
                shared_atlas=shared_atlas, audit_log=audit_log,
            )
        assert shared_atlas.atlas.tombstone_count() == 0

    def test_refuses_replay_after_completion(
        self, admin_db, erasure_db, shared_atlas, audit_log
    ):
        n1 = shared_atlas.atlas.insert(0, _vec(0.7), "d")
        case_id = self._make_case(admin_db, erasure_db, shared_atlas, [n1])

        first = execute_approved_erasure(
            admin_db, erasure_db, case_id, actor="alice",
            shared_atlas=shared_atlas, audit_log=audit_log,
        )
        assert first["erased"] == [n1]

        with pytest.raises(RuntimeError, match="already"):
            execute_approved_erasure(
                admin_db, erasure_db, case_id, actor="alice",
                shared_atlas=shared_atlas, audit_log=audit_log,
            )

    def test_unknown_case_raises(self, admin_db, erasure_db, shared_atlas, audit_log):
        with pytest.raises(ValueError):
            execute_approved_erasure(
                admin_db, erasure_db, 999999999, actor="alice",
                shared_atlas=shared_atlas, audit_log=audit_log,
            )

    def test_case_pointing_at_non_erasure_request_raises(
        self, admin_db, erasure_db, shared_atlas, audit_log
    ):
        # Defensive check: a case's approval_request_id must actually
        # name an "erasure" request -- schema.py's FK guarantees the row
        # EXISTS, not that its action is right.
        req_id = admin_db.create_approval_request(
            action="bulk_scope_remap", target="scope=0x1", reason="testing",
            requested_by="alice", expires_at=_future(), required_approvals=1,
        )
        admin_db.grant_approval(request_id=req_id, approver="bob")
        case_id = erasure_db.create_case(approval_request_id=req_id)

        with pytest.raises(ValueError):
            execute_approved_erasure(
                admin_db, erasure_db, case_id, actor="alice",
                shared_atlas=shared_atlas, audit_log=audit_log,
            )

    def test_transient_compaction_failure_leaves_case_uncompleted(
        self, admin_db, erasure_db, shared_atlas, audit_log
    ):
        # Advisor review: a transient failure (the shared store is
        # mid-compaction) must NOT be recorded as a permanent
        # could_not_erase failure and must NOT consume the four-eyes
        # approval -- unlike an invalid/delta-arena id, compaction
        # finishing shortly makes a retry the correct next action.
        n1 = shared_atlas.atlas.insert(0, _vec(0.95), "f")
        case_id = self._make_case(admin_db, erasure_db, shared_atlas, [n1])

        class _FakeAtlas:
            def tombstone_node(self, node_id):
                raise RuntimeError(
                    "tombstone_node: cannot mutate a node while "
                    "compaction is in progress"
                )

        class _FakeSharedAtlas:
            atlas = _FakeAtlas()

        with pytest.raises(ErasureTransientFailure):
            execute_approved_erasure(
                admin_db, erasure_db, case_id, actor="alice",
                shared_atlas=_FakeSharedAtlas(), audit_log=audit_log,
            )

        case = erasure_db.get_case(case_id)
        assert case["completed_at"] is None
        assert case["receipt"] is None
        # Real Atlas untouched -- the fake intercepted the call, but
        # confirm the target is still live, not erased.
        assert shared_atlas.atlas.tombstone_count() == 0

        # And the case is genuinely still executable once the (simulated)
        # transient condition clears -- retry with the REAL atlas.
        receipt = execute_approved_erasure(
            admin_db, erasure_db, case_id, actor="alice",
            shared_atlas=shared_atlas, audit_log=audit_log,
        )
        assert receipt["erased"] == [n1]
        assert erasure_db.get_case(case_id)["completed_at"] is not None

    def test_crash_resumable_reexecution_treats_already_tombstoned_as_erased(
        self, admin_db, erasure_db, shared_atlas, audit_log
    ):
        # Simulates a process crash AFTER tombstone_node() succeeded for
        # one target but BEFORE complete_case() ran -- completed_at is
        # still None, so a retry must be safe: tombstone_node() is
        # idempotent (core/tests/test_atlas.cpp's
        # TombstoneNodeIsIdempotent), so re-running reports the
        # already-tombstoned id as erased again, not as a failure.
        n1 = shared_atlas.atlas.insert(0, _vec(0.8), "e")
        case_id = self._make_case(admin_db, erasure_db, shared_atlas, [n1])

        shared_atlas.atlas.tombstone_node(n1)  # pre-tombstoned, simulating the crash
        assert erasure_db.get_case(case_id)["completed_at"] is None  # crash guard not tripped

        receipt = execute_approved_erasure(
            admin_db, erasure_db, case_id, actor="alice",
            shared_atlas=shared_atlas, audit_log=audit_log,
        )
        assert receipt["erased"] == [n1]
        assert receipt["could_not_erase"] == []
