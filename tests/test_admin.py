"""
V4 Stage 4 task 7: admin roles + four-eyes approval (control_plane/admin.py).

Postgres-backed -- same opt-in-locally / hard-fail-in-CI pattern as
test_control_plane.py (see that file's module docstring for the reasoning).
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
        "admin/approval tests are opt-in locally (see docker-compose.yml "
        "for a local Postgres).",
        allow_module_level=True,
    )

from datetime import datetime, timedelta, timezone

from aeon_py.client import ALL_SCOPES_VISIBLE
from aeon_py.control_plane.admin import AdminDB, DuplicateApprovalError, WildcardScopeError


@pytest.fixture(scope="module")
def admin_db():
    db = AdminDB(DATABASE_URL)
    yield db
    db.dispose()


def _future(seconds: int = 3600) -> datetime:
    return datetime.now(timezone.utc) + timedelta(seconds=seconds)


def _past(seconds: int = 3600) -> datetime:
    return datetime.now(timezone.utc) - timedelta(seconds=seconds)


class TestAdminRoles:
    def test_grant_and_has_role_permanent(self, admin_db):
        admin_db.grant_role(principal="alice", scope_mask=0x1, granted_by="root")
        assert admin_db.has_role(principal="alice", scope_mask=0x1) is True

    def test_has_role_false_for_non_overlapping_scope(self, admin_db):
        admin_db.grant_role(principal="bob", scope_mask=0x2, granted_by="root")
        assert admin_db.has_role(principal="bob", scope_mask=0x4) is False

    def test_has_role_true_for_partial_scope_overlap(self, admin_db):
        # Grant covers scopes 0x1|0x2; a request for just 0x2 overlaps.
        admin_db.grant_role(principal="carol", scope_mask=0x1 | 0x2, granted_by="root")
        assert admin_db.has_role(principal="carol", scope_mask=0x2) is True

    def test_wildcard_scope_rejected_by_default(self, admin_db):
        with pytest.raises(WildcardScopeError):
            admin_db.grant_role(
                principal="dave", scope_mask=ALL_SCOPES_VISIBLE, granted_by="root"
            )

    def test_wildcard_scope_allowed_when_explicit(self, admin_db):
        admin_db.grant_role(
            principal="eve", scope_mask=ALL_SCOPES_VISIBLE, granted_by="root",
            allow_wildcard=True,
        )
        assert admin_db.has_role(principal="eve", scope_mask=0x1) is True

    # ── Gate test 1 (advisor): expiry with NO sweeper running ──────

    def test_expired_role_grant_is_invalid_with_no_sweeper(self, admin_db):
        """The whole design decision made falsifiable: insert a grant
        whose expires_at is already in the past, and confirm has_role()
        treats it as invalid purely by comparing against now() at read
        time -- no background job, no sweeper, nothing mutates the row
        between insert and this assertion."""
        admin_db.grant_role(
            principal="frank", scope_mask=0x8, granted_by="root",
            expires_at=_past(),
        )
        assert admin_db.has_role(principal="frank", scope_mask=0x8) is False

    def test_non_expired_role_grant_is_valid(self, admin_db):
        # Same shape as the expiry test, opposite direction -- a grant
        # expiring in the FUTURE must still be valid.
        admin_db.grant_role(
            principal="grace", scope_mask=0x10, granted_by="root",
            expires_at=_future(),
        )
        assert admin_db.has_role(principal="grace", scope_mask=0x10) is True

    def test_break_glass_is_a_time_boxed_admin_grant(self, admin_db):
        # Break-glass (task 7) is just an "admin" role with a short
        # expires_at -- not a distinct role value or mechanism.
        admin_db.grant_role(
            principal="oncall1", scope_mask=0x20, granted_by="incident-bot",
            expires_at=_future(seconds=900),  # 15-minute break-glass window
        )
        assert admin_db.has_role(principal="oncall1", scope_mask=0x20) is True

    def test_unknown_role_rejected(self, admin_db):
        with pytest.raises(ValueError):
            admin_db.grant_role(
                principal="mallory", scope_mask=0x1, granted_by="root",
                role="superuser",
            )

    # V4 Stage 4 task 5(b), advisor review: effective_scope_mask() is what
    # the console's knowledge-browser listing route derives the caller's
    # OWN visibility from, rather than trusting a caller-supplied mask.
    def test_effective_scope_mask_ors_multiple_grants(self, admin_db):
        admin_db.grant_role(principal="heidi", scope_mask=0x1, granted_by="root")
        admin_db.grant_role(principal="heidi", scope_mask=0x4, granted_by="root")
        assert admin_db.effective_scope_mask(principal="heidi") == (0x1 | 0x4)

    def test_effective_scope_mask_is_zero_for_unknown_principal(self, admin_db):
        # Fail closed: no grants at all must resolve to "sees nothing"
        # (0 & anything == 0), never ALL_SCOPES_VISIBLE.
        assert admin_db.effective_scope_mask(principal="nobody-ever-granted") == 0

    def test_effective_scope_mask_excludes_expired_grants(self, admin_db):
        admin_db.grant_role(
            principal="ivan", scope_mask=0x8, granted_by="root",
            expires_at=_past(),
        )
        assert admin_db.effective_scope_mask(principal="ivan") == 0


class TestFourEyesApproval:
    def test_single_approval_is_insufficient_for_required_two(self, admin_db):
        req_id = admin_db.create_approval_request(
            action="bulk_scope_remap", target="scope=0x4",
            requested_by="alice", reason="testing", expires_at=_future(), required_approvals=2,
        )
        admin_db.grant_approval(request_id=req_id, approver="bob")
        assert admin_db.is_approved(req_id) is False

    def test_two_distinct_approvers_satisfies_required_two(self, admin_db):
        req_id = admin_db.create_approval_request(
            action="bulk_scope_remap", target="scope=0x8",
            requested_by="alice", reason="testing", expires_at=_future(), required_approvals=2,
        )
        admin_db.grant_approval(request_id=req_id, approver="bob")
        admin_db.grant_approval(request_id=req_id, approver="carol")
        assert admin_db.is_approved(req_id) is True

    # ── Gate test 2 (advisor): same approver twice must not satisfy N ──

    def test_same_approver_twice_does_not_satisfy_required_two(self, admin_db):
        """The second design decision made falsifiable: one person
        approving the same request twice must never satisfy
        required_approvals=2 on their own -- that's the entire point of
        four-eyes. The UNIQUE(request_id, approver) constraint
        (schema.py) is what actually enforces this; grant_approval()
        surfaces the violation as a typed DuplicateApprovalError rather
        than a raw IntegrityError."""
        req_id = admin_db.create_approval_request(
            action="bulk_scope_remap", target="scope=0x10",
            requested_by="alice", reason="testing", expires_at=_future(), required_approvals=2,
        )
        admin_db.grant_approval(request_id=req_id, approver="bob")
        with pytest.raises(DuplicateApprovalError):
            admin_db.grant_approval(request_id=req_id, approver="bob")

        assert admin_db.is_approved(req_id) is False

    def test_expired_approval_request_is_invalid_even_if_fully_granted(self, admin_db):
        # Lazy expiry applies to approval_requests too -- fully granted
        # but past its expires_at must read as NOT approved, no sweeper.
        req_id = admin_db.create_approval_request(
            action="bulk_scope_remap", target="scope=0x20",
            requested_by="alice", reason="testing", expires_at=_past(), required_approvals=1,
        )
        admin_db.grant_approval(request_id=req_id, approver="bob")
        assert admin_db.is_approved(req_id) is False

    def test_revoked_approval_request_is_invalid_even_if_fully_granted(self, admin_db):
        req_id = admin_db.create_approval_request(
            action="bulk_scope_remap", target="scope=0x40",
            requested_by="alice", reason="testing", expires_at=_future(), required_approvals=1,
        )
        admin_db.grant_approval(request_id=req_id, approver="bob")
        assert admin_db.is_approved(req_id) is True
        admin_db.revoke_request(req_id)
        assert admin_db.is_approved(req_id) is False

    def test_mark_executed_is_visible_via_get_request(self, admin_db):
        req_id = admin_db.create_approval_request(
            action="bulk_scope_remap", target="scope=0x80",
            requested_by="alice", reason="testing", expires_at=_future(), required_approvals=1,
        )
        admin_db.grant_approval(request_id=req_id, approver="bob")
        assert admin_db.get_request(req_id)["executed_at"] is None
        admin_db.mark_executed(req_id)
        assert admin_db.get_request(req_id)["executed_at"] is not None

    def test_unknown_request_id_is_not_approved(self, admin_db):
        assert admin_db.is_approved(999999999) is False

    def test_get_request_returns_none_for_unknown_id(self, admin_db):
        assert admin_db.get_request(999999999) is None


import numpy as np

from aeon_py.client import AeonClient
from aeon_py.promotion import (
    IdentifierCorpus,
    create_promotion_approval_request,
    execute_approved_promotion,
)


def _vec(seed: float) -> list:
    return np.full(768, seed, dtype=np.float32).tolist()


@pytest.fixture
def private_atlas(tmp_path):
    return AeonClient(tmp_path / "private.atlas")


@pytest.fixture
def shared_atlas(tmp_path):
    return AeonClient(tmp_path / "shared.atlas")


@pytest.fixture
def audit_log(tmp_path):
    from aeon_py.governance import AuditLog
    return AuditLog(tmp_path / "audit.jsonl")


@pytest.fixture
def governance_db():
    from aeon_py.control_plane.db import GovernanceDB
    db = GovernanceDB(DATABASE_URL)
    yield db
    db.dispose()


class TestExecuteApprovedPromotion:
    def _approve_fully(self, admin_db, req_id, approvers=("bob", "carol")):
        for approver in approvers:
            admin_db.grant_approval(request_id=req_id, approver=approver)

    def test_executes_promotion_with_params_from_the_request(
        self, admin_db, private_atlas, shared_atlas, audit_log
    ):
        node_id = private_atlas.atlas.insert(0, _vec(0.3), "clean text")
        req_id = create_promotion_approval_request(
            admin_db, source_node_id=node_id, dest_scope=0x4, subject_id="subject1",
            reason="quarterly knowledge sync", requested_by="alice",
            expires_at=_future(),
        )
        self._approve_fully(admin_db, req_id)

        corpus = IdentifierCorpus(patterns=["nomatch"])
        new_id = execute_approved_promotion(
            admin_db, req_id, actor="alice",
            source_atlas=private_atlas, dest_atlas=shared_atlas,
            corpus=corpus, audit_log=audit_log,
        )

        assert new_id is not None
        assert shared_atlas.atlas.get_node_scope(new_id) == 0x4
        assert admin_db.get_request(req_id)["executed_at"] is not None

    def test_refuses_without_enough_approvals(
        self, admin_db, private_atlas, shared_atlas, audit_log
    ):
        node_id = private_atlas.atlas.insert(0, _vec(0.4), "clean text")
        req_id = create_promotion_approval_request(
            admin_db, source_node_id=node_id, dest_scope=0x1, subject_id="subject1",
            reason="testing", requested_by="alice", expires_at=_future(),
        )
        admin_db.grant_approval(request_id=req_id, approver="bob")  # only 1 of 2

        corpus = IdentifierCorpus(patterns=["nomatch"])
        with pytest.raises(PermissionError):
            execute_approved_promotion(
                admin_db, req_id, actor="alice",
                source_atlas=private_atlas, dest_atlas=shared_atlas,
                corpus=corpus, audit_log=audit_log,
            )
        assert shared_atlas.atlas.size() == 0

    def test_refuses_replay_after_first_execution(
        self, admin_db, private_atlas, shared_atlas, audit_log
    ):
        node_id = private_atlas.atlas.insert(0, _vec(0.5), "clean text")
        req_id = create_promotion_approval_request(
            admin_db, source_node_id=node_id, dest_scope=0x1, subject_id="subject1",
            reason="testing", requested_by="alice", expires_at=_future(),
        )
        self._approve_fully(admin_db, req_id)
        corpus = IdentifierCorpus(patterns=["nomatch"])

        first = execute_approved_promotion(
            admin_db, req_id, actor="alice",
            source_atlas=private_atlas, dest_atlas=shared_atlas,
            corpus=corpus, audit_log=audit_log,
        )
        assert first is not None
        assert shared_atlas.atlas.size() == 1

        with pytest.raises(RuntimeError, match="already"):
            execute_approved_promotion(
                admin_db, req_id, actor="alice",
                source_atlas=private_atlas, dest_atlas=shared_atlas,
                corpus=corpus, audit_log=audit_log,
            )
        # Replay must not mint a second node.
        assert shared_atlas.atlas.size() == 1

    def test_classifier_rejection_does_not_consume_the_approval(
        self, admin_db, private_atlas, shared_atlas, audit_log
    ):
        # Advisor review: a classifier rejection is a corpus-config
        # problem, not a decision anyone actually made -- it must not
        # permanently burn a four-eyes approval that could otherwise be
        # retried once the corpus config is fixed.
        node_id = private_atlas.atlas.insert(0, _vec(0.55), "clean text")
        req_id = create_promotion_approval_request(
            admin_db, source_node_id=node_id, dest_scope=0x1, subject_id="subject1",
            reason="testing", requested_by="alice", expires_at=_future(),
        )
        self._approve_fully(admin_db, req_id)

        empty_corpus = IdentifierCorpus()  # fail-closed: rejects everything
        rejected = execute_approved_promotion(
            admin_db, req_id, actor="alice",
            source_atlas=private_atlas, dest_atlas=shared_atlas,
            corpus=empty_corpus, audit_log=audit_log,
        )
        assert rejected is None
        assert shared_atlas.atlas.size() == 0
        assert admin_db.get_request(req_id)["executed_at"] is None

        # Retry with a corpus that actually clears the content -- still
        # possible because the approval was never consumed.
        working_corpus = IdentifierCorpus(patterns=["nomatch"])
        promoted = execute_approved_promotion(
            admin_db, req_id, actor="alice",
            source_atlas=private_atlas, dest_atlas=shared_atlas,
            corpus=working_corpus, audit_log=audit_log,
        )
        assert promoted is not None
        assert shared_atlas.atlas.size() == 1
        assert admin_db.get_request(req_id)["executed_at"] is not None

    def test_unknown_request_raises(self, admin_db, private_atlas, shared_atlas, audit_log):
        corpus = IdentifierCorpus(patterns=["nomatch"])
        with pytest.raises(ValueError):
            execute_approved_promotion(
                admin_db, 999999999, actor="alice",
                source_atlas=private_atlas, dest_atlas=shared_atlas,
                corpus=corpus, audit_log=audit_log,
            )

    def test_non_promotion_request_raises(self, admin_db, private_atlas, shared_atlas, audit_log):
        # A request created for a DIFFERENT admin action (e.g. bulk scope
        # remap) must never be executable as a promotion.
        req_id = admin_db.create_approval_request(
            action="bulk_scope_remap", target="scope=0x1",
            reason="testing", requested_by="alice", expires_at=_future(),
            required_approvals=1,
        )
        admin_db.grant_approval(request_id=req_id, approver="bob")

        corpus = IdentifierCorpus(patterns=["nomatch"])
        with pytest.raises(ValueError):
            execute_approved_promotion(
                admin_db, req_id, actor="alice",
                source_atlas=private_atlas, dest_atlas=shared_atlas,
                corpus=corpus, audit_log=audit_log,
            )

    @pytest.mark.parametrize("blank_subject_id", ["", "   "])
    def test_create_request_rejects_blank_subject_id(
        self, admin_db, private_atlas, blank_subject_id
    ):
        # advisor review: rejecting only inside promote_fragment() lets a
        # blank subject_id collect two real approvals before ever failing
        # -- a dead request nobody could ever execute. Must fail HERE,
        # before any approval_requests row is even created.
        node_id = private_atlas.atlas.insert(0, _vec(1.0), "clean text")
        with pytest.raises(ValueError):
            create_promotion_approval_request(
                admin_db, source_node_id=node_id, dest_scope=0x1,
                subject_id=blank_subject_id,
                reason="testing", requested_by="alice", expires_at=_future(),
            )

    def test_subject_id_survives_target_json_lock_in_to_postgres_row(
        self, admin_db, private_atlas, shared_atlas, audit_log, governance_db
    ):
        # advisor review (task 6 Phase A): every prior test either passed
        # subject_id straight to governance_db.record() or asserted on a
        # request created and read back in the same call -- none proved
        # subject_id actually survives create_promotion_approval_request()'s
        # target-JSON lock-in THROUGH execute_approved_promotion() into the
        # governance_records row (the whole replay-safety point of locking
        # it in there instead of accepting it fresh at execution time).
        # This would pass even if subject_id were silently dropped between
        # the two calls, unless it specifically checks the Postgres row.
        node_id = private_atlas.atlas.insert(0, _vec(0.35), "clean text")
        req_id = create_promotion_approval_request(
            admin_db, source_node_id=node_id, dest_scope=0x100,
            subject_id="subject-round-trip",
            reason="testing", requested_by="alice", expires_at=_future(),
        )
        self._approve_fully(admin_db, req_id)

        corpus = IdentifierCorpus(patterns=["nomatch"])
        new_id = execute_approved_promotion(
            admin_db, req_id, actor="alice",
            source_atlas=private_atlas, dest_atlas=shared_atlas,
            corpus=corpus, audit_log=audit_log, governance_db=governance_db,
        )
        assert new_id is not None

        gov_id = shared_atlas.atlas.get_node_governance_id(new_id)
        from sqlalchemy import create_engine, select

        from aeon_py.control_plane.schema import governance_records

        engine = create_engine(DATABASE_URL)
        try:
            with engine.connect() as conn:
                row = conn.execute(
                    select(governance_records).where(governance_records.c.id == gov_id)
                ).mappings().first()
        finally:
            engine.dispose()

        assert row is not None
        assert row["subject_id"] == "subject-round-trip"
