"""
V4 Stage 4 task 1: Postgres control plane (control_plane/).

Postgres-backed -- opt-in locally (set AEON_CONTROL_PLANE_DATABASE_URL to
run these for real; see docker-compose.yml for a local Postgres). CI sets
AEON_REQUIRE_DB_TESTS=1 alongside a real service container so a
misconfigured CI run hard-fails collection instead of silently skipping
every test in this file -- a green CI that skipped its DB tests is worse
than a red one.

Each test uses a fresh uuid4() instance id and unique actor/node values
rather than truncating tables between runs, so repeated local runs against
the same long-lived test database never interfere with each other.
"""
import os
import uuid

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
        "control-plane tests are opt-in locally (see docker-compose.yml "
        "for a local Postgres).",
        allow_module_level=True,
    )

import numpy as np
from fastapi.testclient import TestClient

from aeon_py.client import AeonClient
from aeon_py.control_plane.app import app as control_plane_app
from aeon_py.control_plane.db import GovernanceDB
from aeon_py.governance import AuditLog
from aeon_py.promotion import IdentifierCorpus, promote_fragment


def _vec(seed: float) -> list:
    return np.full(768, seed, dtype=np.float32).tolist()


@pytest.fixture(scope="module")
def governance_db():
    db = GovernanceDB(DATABASE_URL)
    yield db
    db.dispose()


class TestMigrationsApplied:
    def test_database_is_at_head_revision(self):
        # advisor review: with a single migration, every test here passes
        # against a hand-created table just as well as a real migration --
        # that stops being true the moment a second migration exists,
        # since a stale-schema database (migration 1 applied, migration 2
        # not) would pass every functional test while silently diverging
        # from what alembic/versions/ actually specifies. This is the
        # check that catches that: read alembic_version directly and
        # confirm it matches the newest revision file on disk, not just
        # "some revision applied."
        import pathlib

        from alembic.config import Config
        from alembic.script import ScriptDirectory
        from sqlalchemy import create_engine, text as sa_text

        repo_root = pathlib.Path(__file__).parent.parent
        cfg = Config(str(repo_root / "alembic.ini"))
        cfg.set_main_option("script_location", str(repo_root / "alembic"))
        script = ScriptDirectory.from_config(cfg)
        head_revision = script.get_current_head()

        engine = create_engine(DATABASE_URL)
        try:
            with engine.connect() as conn:
                db_revision = conn.execute(
                    sa_text("SELECT version_num FROM alembic_version")
                ).scalar_one()
        finally:
            engine.dispose()

        assert db_revision == head_revision, (
            f"database is at revision {db_revision!r} but the newest "
            f"migration on disk is {head_revision!r} -- run "
            "`alembic upgrade head` before running these tests"
        )


class TestGovernanceDBRecord:
    def test_record_returns_distinct_pks(self, governance_db):
        instance_id = uuid.uuid4()
        pk1 = governance_db.record(
            log_instance_id=instance_id, log_instance_path="/tmp/a.jsonl",
            log_seq=1, action="promotion", actor="alice", subject_id="subject-alice",
            source_node_id=1, dest_node_id=100, dest_scope=0x1,
        )
        pk2 = governance_db.record(
            log_instance_id=instance_id, log_instance_path="/tmp/a.jsonl",
            log_seq=2, action="promotion", actor="alice", subject_id="subject-alice",
            source_node_id=2, dest_node_id=101, dest_scope=0x1,
        )
        assert pk1 != pk2

    def test_record_is_idempotent_on_log_instance_upsert(self, governance_db):
        # Two records against the SAME instance_id must not fail on the
        # governance_log_instances upsert (ON CONFLICT DO NOTHING).
        instance_id = uuid.uuid4()
        governance_db.record(
            log_instance_id=instance_id, log_instance_path="/tmp/b.jsonl",
            log_seq=1, action="promotion", actor="bob", subject_id="subject-bob",
        )
        pk = governance_db.record(
            log_instance_id=instance_id, log_instance_path="/tmp/b.jsonl",
            log_seq=2, action="promotion", actor="bob", subject_id="subject-bob",
        )
        assert pk is not None

    def test_record_readable_via_control_plane_app(self, governance_db):
        instance_id = uuid.uuid4()
        pk = governance_db.record(
            log_instance_id=instance_id, log_instance_path="/tmp/c.jsonl",
            log_seq=1, action="promotion", actor="carol", subject_id="subject-carol",
            source_node_id=7, dest_node_id=200, dest_scope=0x4,
        )

        client = TestClient(control_plane_app)
        r = client.get(f"/governance-records/{pk}")
        assert r.status_code == 200
        data = r.json()
        assert data["id"] == pk
        assert data["actor"] == "carol"
        assert data["dest_node_id"] == 200
        # task 6 Phase A: subject_id round-trips -- this is the field a
        # future crypto-erase DEK lookup resolves through
        # governance_record_id, so it must actually be readable back, not
        # just accepted at write time.
        assert data["subject_id"] == "subject-carol"

    def test_missing_record_returns_404(self, governance_db):
        client = TestClient(control_plane_app)
        r = client.get("/governance-records/999999999")
        assert r.status_code == 404


@pytest.fixture
def private_atlas(tmp_path):
    return AeonClient(tmp_path / "private.atlas")


@pytest.fixture
def shared_atlas(tmp_path):
    return AeonClient(tmp_path / "shared.atlas")


@pytest.fixture
def audit_log(tmp_path):
    return AuditLog(tmp_path / "audit.jsonl")


class TestPromoteFragmentWithGovernanceDB:
    def test_governance_record_id_is_postgres_pk_not_jsonl_seq(
        self, private_atlas, shared_atlas, audit_log, governance_db
    ):
        node_id = private_atlas.atlas.insert(0, _vec(0.3), "clean text")
        corpus = IdentifierCorpus(patterns=["nomatch"])

        new_id = promote_fragment(
            private_atlas, node_id, shared_atlas,
            dest_scope=0x1, corpus=corpus, audit_log=audit_log,
            actor="maintainer1", subject_id="subject-under-test",
            governance_db=governance_db,
        )

        gov_id = shared_atlas.atlas.get_node_governance_id(new_id)
        # The JSONL log's own seq is always 1 for a fresh log's first
        # promotion -- if governance_db is wired correctly, gov_id must
        # be the Postgres row's id, not that seq (a Postgres bigserial
        # starts at 1 too on a fresh table, so this only distinguishes
        # the two once more than one promotion has happened across the
        # shared test DB -- assert against the control-plane app instead,
        # which is unambiguous).
        client = TestClient(control_plane_app)
        r = client.get(f"/governance-records/{gov_id}")
        assert r.status_code == 200
        assert r.json()["dest_node_id"] == new_id
        assert r.json()["log_instance_id"] == str(audit_log.instance_id)
        assert r.json()["subject_id"] == "subject-under-test"

    def test_two_promotions_get_distinct_stable_ids(
        self, private_atlas, shared_atlas, audit_log, governance_db
    ):
        corpus = IdentifierCorpus(patterns=["nomatch"])
        id1 = private_atlas.atlas.insert(0, _vec(0.1), "fragment one")
        id2 = private_atlas.atlas.insert(0, _vec(0.2), "fragment two")

        new1 = promote_fragment(
            private_atlas, id1, shared_atlas, dest_scope=0x1, corpus=corpus,
            audit_log=audit_log, actor="m1", subject_id="subject1",
            governance_db=governance_db,
        )
        new2 = promote_fragment(
            private_atlas, id2, shared_atlas, dest_scope=0x1, corpus=corpus,
            audit_log=audit_log, actor="m1", subject_id="subject1",
            governance_db=governance_db,
        )

        gov1 = shared_atlas.atlas.get_node_governance_id(new1)
        gov2 = shared_atlas.atlas.get_node_governance_id(new2)
        assert gov1 != gov2

    def test_rejected_promotion_is_mirrored_to_postgres(
        self, private_atlas, shared_atlas, audit_log, governance_db
    ):
        # advisor review: governance_db.record() was previously only
        # called on the success path -- a console querying
        # governance_records for "what was attempted" got a
        # systematically incomplete answer. Confirm the rejection path is
        # now queryable too, with dest_node_id correctly absent (no
        # shared-store node was ever created).
        node_id = private_atlas.atlas.insert(0, _vec(0.5), "unredactable secret")

        result = promote_fragment(
            private_atlas, node_id, shared_atlas, dest_scope=0x1,
            corpus=IdentifierCorpus(),  # empty -- fail closed
            audit_log=audit_log, actor="m1", subject_id="subject1",
            governance_db=governance_db,
        )
        assert result is None

        # The rejection was audit_log's seq 1 -- find its mirrored row via
        # the log_seq (there's no direct "list by seq" API on the
        # control-plane app yet, task 5; query GovernanceDB's underlying
        # engine directly, same pattern as TestMigrationsApplied).
        from sqlalchemy import create_engine, select

        from aeon_py.control_plane.schema import governance_records

        engine = create_engine(DATABASE_URL)
        try:
            with engine.connect() as conn:
                row = conn.execute(
                    select(governance_records).where(
                        governance_records.c.log_instance_id == audit_log.instance_id,
                        governance_records.c.log_seq == 1,
                    )
                ).mappings().first()
        finally:
            engine.dispose()

        assert row is not None
        assert row["action"] == "promotion_rejected"
        assert row["dest_node_id"] is None
        assert row["source_node_id"] == node_id

    def test_delta_diversion_anomaly_is_mirrored_to_postgres(
        self, private_atlas, audit_log, governance_db
    ):
        from unittest.mock import MagicMock

        from aeon_py.client import NODE_ID_DELTA_MASK
        from sqlalchemy import create_engine, select

        from aeon_py.control_plane.schema import governance_records

        node_id = private_atlas.atlas.insert(0, _vec(0.6), "clean text")
        corpus = IdentifierCorpus(patterns=["nomatch"])

        orphaned_id = NODE_ID_DELTA_MASK | 3
        mock_dest = MagicMock()
        mock_dest.atlas.insert.return_value = orphaned_id

        with pytest.raises(RuntimeError, match="delta buffer"):
            promote_fragment(
                private_atlas, node_id, mock_dest, dest_scope=0x1,
                corpus=corpus, audit_log=audit_log, actor="m1",
                subject_id="subject1", governance_db=governance_db,
            )

        engine = create_engine(DATABASE_URL)
        try:
            with engine.connect() as conn:
                row = conn.execute(
                    select(governance_records).where(
                        governance_records.c.log_instance_id == audit_log.instance_id,
                        governance_records.c.log_seq == 1,
                    )
                ).mappings().first()
        finally:
            engine.dispose()

        assert row is not None
        assert row["action"] == "promotion_unscoped_anomaly"
        assert row["dest_node_id"] == orphaned_id

    def test_postgres_mirror_failure_does_not_mask_the_original_exception(
        self, private_atlas, audit_log
    ):
        # The best-effort mirror (_mirror_governance_record) must swallow
        # its own failure and let the ORIGINAL RuntimeError propagate --
        # not raise a DB error that hides what actually happened.
        from unittest.mock import MagicMock

        from aeon_py.client import NODE_ID_DELTA_MASK

        node_id = private_atlas.atlas.insert(0, _vec(0.7), "clean text")
        corpus = IdentifierCorpus(patterns=["nomatch"])

        broken_governance_db = MagicMock()
        broken_governance_db.record.side_effect = RuntimeError("connection refused")

        mock_dest = MagicMock()
        mock_dest.atlas.insert.return_value = NODE_ID_DELTA_MASK | 9

        with pytest.raises(RuntimeError, match="delta buffer"):
            promote_fragment(
                private_atlas, node_id, mock_dest, dest_scope=0x1,
                corpus=corpus, audit_log=audit_log, actor="m1",
                subject_id="subject1", governance_db=broken_governance_db,
            )
