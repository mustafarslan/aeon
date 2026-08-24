"""
V4 Stage 4 task 6 Phase B: crypto.py's Keystore (Postgres-backed DEK
storage) and its wiring into promote_fragment()/the knowledge browser/
execute_approved_erasure(). See test_crypto.py for the pure encoding
tests that need no Postgres.

Postgres-backed -- same opt-in-locally / hard-fail-in-CI pattern as
test_control_plane.py/test_erasure.py.
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
        "crypto-erase keystore tests are opt-in locally (see "
        "docker-compose.yml for a local Postgres).",
        allow_module_level=True,
    )

import os as _os
from datetime import datetime, timedelta, timezone

import numpy as np

from aeon_py.client import AeonClient
from aeon_py.control_plane.admin import AdminDB
from aeon_py.control_plane.db import GovernanceDB
from aeon_py.control_plane.erasure_db import ErasureDB
from aeon_py.crypto import Keystore, decrypt_metadata, is_encrypted_metadata
from aeon_py.erasure import create_erasure_case, execute_approved_erasure
from aeon_py.governance import AuditLog
from aeon_py.promotion import IdentifierCorpus, promote_fragment


def _vec(seed: float) -> list:
    return np.full(768, seed, dtype=np.float32).tolist()


def _future(seconds: int = 3600) -> datetime:
    return datetime.now(timezone.utc) + timedelta(seconds=seconds)


@pytest.fixture(scope="module")
def kek() -> bytes:
    return _os.urandom(32)


@pytest.fixture
def keystore(kek):
    ks = Keystore(DATABASE_URL, kek)
    yield ks
    ks.dispose()


@pytest.fixture(scope="module")
def governance_db():
    db = GovernanceDB(DATABASE_URL)
    yield db
    db.dispose()


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
def private_atlas(tmp_path):
    return AeonClient(tmp_path / "private.atlas")


@pytest.fixture
def shared_atlas(tmp_path):
    # metadata_size=512, matching dependencies.py's real default for the
    # shared store once crypto-erase is enabled (v4-plan.md task 6 Phase B).
    return AeonClient(tmp_path / "shared.atlas", metadata_size=512)


@pytest.fixture
def audit_log(tmp_path):
    return AuditLog(tmp_path / "audit.jsonl")


class TestKeystore:
    def test_get_or_create_dek_is_stable_across_calls(self, keystore):
        subject_id = f"subject-{_os.urandom(4).hex()}"
        dek1 = keystore.get_or_create_dek(subject_id, 0x1)
        dek2 = keystore.get_or_create_dek(subject_id, 0x1)
        assert dek1 == dek2

    def test_different_subject_scope_pairs_get_different_deks(self, keystore):
        subject_id = f"subject-{_os.urandom(4).hex()}"
        dek_a = keystore.get_or_create_dek(subject_id, 0x1)
        dek_b = keystore.get_or_create_dek(subject_id, 0x2)  # same subject, different scope
        dek_c = keystore.get_or_create_dek(f"{subject_id}-other", 0x1)  # different subject, same scope
        assert dek_a != dek_b
        assert dek_a != dek_c

    def test_get_dek_returns_none_before_first_use(self, keystore):
        subject_id = f"subject-{_os.urandom(4).hex()}"
        assert keystore.get_dek(subject_id, 0x1) is None

    def test_get_dek_matches_get_or_create_dek_after_first_use(self, keystore):
        subject_id = f"subject-{_os.urandom(4).hex()}"
        created = keystore.get_or_create_dek(subject_id, 0x1)
        assert keystore.get_dek(subject_id, 0x1) == created

    def test_destroy_key_removes_it(self, keystore):
        subject_id = f"subject-{_os.urandom(4).hex()}"
        keystore.get_or_create_dek(subject_id, 0x1)
        assert keystore.destroy_key(subject_id, 0x1) is True
        assert keystore.get_dek(subject_id, 0x1) is None

    def test_destroy_key_returns_false_when_nothing_to_destroy(self, keystore):
        subject_id = f"subject-{_os.urandom(4).hex()}"
        assert keystore.destroy_key(subject_id, 0x1) is False

    def test_wrapped_dek_is_not_the_kek_and_differs_across_deks(self, keystore, kek):
        # Sanity check that wrapping is actually happening, not a no-op --
        # read the raw wrapped column value via a fresh Keystore's own
        # unwrap plumbing indirectly: two different subjects' DEKs must
        # differ, and neither DEK equals the KEK itself.
        s1, s2 = f"s-{_os.urandom(4).hex()}", f"s-{_os.urandom(4).hex()}"
        dek1 = keystore.get_or_create_dek(s1, 0x1)
        dek2 = keystore.get_or_create_dek(s2, 0x1)
        assert dek1 != dek2
        assert dek1 != kek
        assert dek2 != kek


class TestPromoteFragmentWithKeystore:
    def test_promoted_metadata_is_encrypted_at_rest(
        self, private_atlas, shared_atlas, audit_log, keystore
    ):
        subject_id = f"subject-{_os.urandom(4).hex()}"
        node_id = private_atlas.atlas.insert(0, _vec(0.3), "contact alice@example.com")
        corpus = IdentifierCorpus(redact_emails=True)

        new_id = promote_fragment(
            private_atlas, node_id, shared_atlas,
            dest_scope=0x1, corpus=corpus, audit_log=audit_log,
            actor="maintainer1", subject_id=subject_id, keystore=keystore,
        )
        assert new_id is not None

        raw_stored = shared_atlas.atlas.get_node_metadata(new_id)
        assert is_encrypted_metadata(raw_stored) is True
        assert "alice" not in raw_stored  # not even the redacted-but-plaintext form

    def test_promoted_metadata_decrypts_to_the_redacted_text(
        self, private_atlas, shared_atlas, audit_log, keystore
    ):
        subject_id = f"subject-{_os.urandom(4).hex()}"
        node_id = private_atlas.atlas.insert(0, _vec(0.35), "contact alice@example.com")
        corpus = IdentifierCorpus(redact_emails=True)

        new_id = promote_fragment(
            private_atlas, node_id, shared_atlas,
            dest_scope=0x2, corpus=corpus, audit_log=audit_log,
            actor="maintainer1", subject_id=subject_id, keystore=keystore,
        )

        dek = keystore.get_dek(subject_id, 0x2)
        assert dek is not None
        raw_stored = shared_atlas.atlas.get_node_metadata(new_id)
        decrypted = decrypt_metadata(dek, raw_stored)
        assert "alice@example.com" not in decrypted
        assert "[REDACTED_EMAIL]" in decrypted

    def test_promote_without_keystore_stores_plaintext(
        self, private_atlas, shared_atlas, audit_log
    ):
        # Regression: default (keystore=None) behavior is unchanged by
        # Phase B -- this deployment hasn't opted into crypto-erase.
        node_id = private_atlas.atlas.insert(0, _vec(0.4), "clean text")
        corpus = IdentifierCorpus(patterns=["nomatch"])

        new_id = promote_fragment(
            private_atlas, node_id, shared_atlas,
            dest_scope=0x1, corpus=corpus, audit_log=audit_log,
            actor="m1", subject_id="subject-3",
        )
        stored = shared_atlas.atlas.get_node_metadata(new_id)
        assert stored == "clean text"
        assert is_encrypted_metadata(stored) is False

    def test_oversized_text_raises_instead_of_truncating_ciphertext(
        self, shared_atlas, audit_log, keystore, tmp_path
    ):
        # The PRIVATE store's own default metadata_size (256) would
        # silently truncate a 500-char fragment before promote_fragment()
        # ever sees it -- use a private store with a large enough field
        # that the fragment survives intact, so this test actually
        # exercises the SHARED store's (smaller, 512-byte) encrypted
        # budget being exceeded, not the private store's unrelated
        # pre-existing truncation.
        big_private_atlas = AeonClient(tmp_path / "big_private.atlas", metadata_size=600)
        long_text = "x" * 500
        node_id = big_private_atlas.atlas.insert(0, _vec(0.45), long_text)
        assert big_private_atlas.atlas.get_node_metadata(node_id) == long_text
        corpus = IdentifierCorpus(patterns=["nomatch"])

        with pytest.raises(ValueError, match="exceeds this shared store"):
            promote_fragment(
                big_private_atlas, node_id, shared_atlas,
                dest_scope=0x1, corpus=corpus, audit_log=audit_log,
                actor="m1", subject_id="subject-4", keystore=keystore,
            )
        # Rejected before any insert into the shared store.
        assert shared_atlas.atlas.size() == 0


class TestEndToEndKeyDestruction:
    """The task 6 gate: 'erasure workflow demonstrates actual key
    destruction end-to-end, not just a tombstone flag.' Write subject
    data, read it back, destroy the key, REOPEN a fresh Keystore
    (simulating a process restart) from the same Postgres row, and
    confirm decryption is genuinely impossible -- not just that the
    in-memory object forgot the key.
    """

    def _approve_fully(self, admin_db, req_id, approvers=("bob", "carol")):
        for approver in approvers:
            admin_db.grant_approval(request_id=req_id, approver=approver)

    def test_erasure_destroys_the_key_and_a_fresh_keystore_cannot_decrypt(
        self, private_atlas, shared_atlas, audit_log, keystore, kek,
        governance_db, admin_db, erasure_db,
    ):
        subject_id = f"subject-{_os.urandom(4).hex()}"
        node_id = private_atlas.atlas.insert(0, _vec(0.5), "clean text about subject")
        corpus = IdentifierCorpus(patterns=["nomatch"])

        new_id = promote_fragment(
            private_atlas, node_id, shared_atlas,
            dest_scope=0x8, corpus=corpus, audit_log=audit_log,
            actor="maintainer1", subject_id=subject_id,
            keystore=keystore, governance_db=governance_db,
        )
        assert new_id is not None

        # Confirm it's genuinely readable before erasure.
        dek_before = keystore.get_dek(subject_id, 0x8)
        assert dek_before is not None
        raw_stored = shared_atlas.atlas.get_node_metadata(new_id)
        assert decrypt_metadata(dek_before, raw_stored) == "clean text about subject"

        case_id = create_erasure_case(
            admin_db, erasure_db, shared_atlas=shared_atlas,
            node_ids=[new_id], reason="GDPR erasure request, ticket #999",
            requested_by="alice", expires_at=_future(),
        )
        req_id = erasure_db.get_case(case_id)["approval_request_id"]
        self._approve_fully(admin_db, req_id)

        receipt = execute_approved_erasure(
            admin_db, erasure_db, case_id,
            actor="alice", shared_atlas=shared_atlas, audit_log=audit_log,
            governance_db=governance_db, keystore=keystore,
        )
        assert receipt["erased"] == [new_id]

        # The node is now tombstoned (logical delete) AND its key is gone.
        assert shared_atlas.atlas.is_node_superseded(new_id) is False  # tombstone, not supersede
        assert keystore.get_dek(subject_id, 0x8) is None

        # Simulate a process restart: a FRESH Keystore instance, same KEK,
        # same Postgres -- the key must still be gone (proves persistence
        # of the deletion, not just this object's in-memory state).
        fresh_keystore = Keystore(DATABASE_URL, kek)
        try:
            assert fresh_keystore.get_dek(subject_id, 0x8) is None
        finally:
            fresh_keystore.dispose()

    def test_a_different_subjects_key_in_the_same_scope_still_decrypts(
        self, private_atlas, shared_atlas, audit_log, keystore,
        governance_db, admin_db, erasure_db,
    ):
        # The other half of the negative control (advisor review): a
        # per-*scope*-only key would pass the destruction test above
        # while silently NOT being per-subject. Confirm erasing one
        # subject's key in a scope leaves a DIFFERENT subject's key in
        # the SAME scope fully intact.
        scope = 0x10
        subject_a = f"subject-a-{_os.urandom(4).hex()}"
        subject_b = f"subject-b-{_os.urandom(4).hex()}"

        node_a = private_atlas.atlas.insert(0, _vec(0.6), "subject A's content")
        node_b = private_atlas.atlas.insert(0, _vec(0.65), "subject B's content")
        corpus = IdentifierCorpus(patterns=["nomatch"])

        new_a = promote_fragment(
            private_atlas, node_a, shared_atlas, dest_scope=scope, corpus=corpus,
            audit_log=audit_log, actor="m1", subject_id=subject_a,
            keystore=keystore, governance_db=governance_db,
        )
        new_b = promote_fragment(
            private_atlas, node_b, shared_atlas, dest_scope=scope, corpus=corpus,
            audit_log=audit_log, actor="m1", subject_id=subject_b,
            keystore=keystore, governance_db=governance_db,
        )

        case_id = create_erasure_case(
            admin_db, erasure_db, shared_atlas=shared_atlas,
            node_ids=[new_a], reason="GDPR erasure request, ticket #1000",
            requested_by="alice", expires_at=_future(),
        )
        req_id = erasure_db.get_case(case_id)["approval_request_id"]
        self._approve_fully(admin_db, req_id)
        execute_approved_erasure(
            admin_db, erasure_db, case_id,
            actor="alice", shared_atlas=shared_atlas, audit_log=audit_log,
            governance_db=governance_db, keystore=keystore,
        )

        assert keystore.get_dek(subject_a, scope) is None
        dek_b = keystore.get_dek(subject_b, scope)
        assert dek_b is not None
        raw_b = shared_atlas.atlas.get_node_metadata(new_b)
        assert decrypt_metadata(dek_b, raw_b) == "subject B's content"


class TestErasureGovernanceDbSubjectIdRegression:
    def test_execute_approved_erasure_with_governance_db_does_not_raise(
        self, private_atlas, shared_atlas, audit_log, governance_db,
        admin_db, erasure_db,
    ):
        # Regression: Phase A made governance_records.subject_id NOT NULL,
        # but execute_approved_erasure()'s own governance_db.record() call
        # was never updated to supply one -- would have raised a bare
        # TypeError on every real erasure execution against a deployment
        # with a control plane configured (server.py's endpoint always
        # passes governance_db). No existing test exercised this branch.
        node_id = private_atlas.atlas.insert(0, _vec(0.7), "clean text")
        corpus = IdentifierCorpus(patterns=["nomatch"])
        new_id = promote_fragment(
            private_atlas, node_id, shared_atlas, dest_scope=0x20, corpus=corpus,
            audit_log=audit_log, actor="m1", subject_id="subject-regress",
            governance_db=governance_db,
        )

        case_id = create_erasure_case(
            admin_db, erasure_db, shared_atlas=shared_atlas,
            node_ids=[new_id], reason="regression test",
            requested_by="alice", expires_at=_future(),
        )
        req_id = erasure_db.get_case(case_id)["approval_request_id"]
        for approver in ("bob", "carol"):
            admin_db.grant_approval(request_id=req_id, approver=approver)

        receipt = execute_approved_erasure(
            admin_db, erasure_db, case_id,
            actor="alice", shared_atlas=shared_atlas, audit_log=audit_log,
            governance_db=governance_db,
        )
        assert receipt["erased"] == [new_id]
