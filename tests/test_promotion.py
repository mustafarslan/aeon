"""
V4 Stage 4 task 2: mint-and-recontextualize promotion pipeline.
"""
import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from aeon_py.client import AeonClient, NODE_ID_DELTA_MASK, decode_store_id, encode_store_id
from aeon_py.governance import AuditLog
from aeon_py.promotion import (
    IdentifierCorpus,
    VerificationResult,
    classify_and_redact,
    promote_fragment,
)
from aeon_py.trace import EdgeType, TraceGraph


def _vec(seed: float) -> list:
    return np.full(768, seed, dtype=np.float32).tolist()


class TestClassifyAndRedact:
    def test_empty_corpus_fails_closed(self):
        result = classify_and_redact("nothing sensitive here", IdentifierCorpus())
        assert result.passed is False
        assert result.categories == ["EMPTY_CORPUS_FAIL_CLOSED"]

    def test_redacts_adopter_pattern(self):
        corpus = IdentifierCorpus(patterns=[r"PROJ-\d+"], redact_emails=False,
                                  redact_commit_shas=False)
        result = classify_and_redact("see ticket PROJ-1234 for details", corpus)
        assert result.passed is True
        assert "PROJ-1234" not in result.redacted_text
        assert "[REDACTED]" in result.redacted_text
        assert result.categories == ["corpus_pattern"]

    def test_redacts_email(self):
        corpus = IdentifierCorpus(redact_emails=True)
        result = classify_and_redact("contact alice@example.com for help", corpus)
        assert result.passed is True
        assert "alice@example.com" not in result.redacted_text
        assert "email" in result.categories

    def test_redacts_commit_sha(self):
        corpus = IdentifierCorpus(redact_commit_shas=True)
        result = classify_and_redact("fixed in a1b2c3d4e5f6", corpus)
        assert result.passed is True
        assert "a1b2c3d4e5f6" not in result.redacted_text
        assert "commit_sha" in result.categories

    def test_categories_never_contain_raw_matched_values(self):
        # The audit log this feeds must not itself become a PII store.
        corpus = IdentifierCorpus(patterns=[r"alice-personal-alias"])
        result = classify_and_redact("assigned to alice-personal-alias", corpus)
        for cat in result.categories:
            assert "alice" not in cat

    def test_clean_text_passes_with_no_categories(self):
        corpus = IdentifierCorpus(patterns=[r"PROJ-\d+"])
        result = classify_and_redact("totally unremarkable text", corpus)
        assert result.passed is True
        assert result.categories == []
        assert result.redacted_text == "totally unremarkable text"


@pytest.fixture
def private_atlas(tmp_path):
    return AeonClient(tmp_path / "private.atlas")


@pytest.fixture
def shared_atlas(tmp_path):
    return AeonClient(tmp_path / "shared.atlas")


@pytest.fixture
def audit_log(tmp_path):
    return AuditLog(tmp_path / "audit.jsonl")


class TestPromoteFragment:
    def test_rejected_fragment_returns_none_and_writes_nothing_to_dest(
        self, private_atlas, shared_atlas, audit_log
    ):
        node_id = private_atlas.atlas.insert(0, _vec(1.0), "unredactable secret")

        result = promote_fragment(
            private_atlas, node_id, shared_atlas,
            dest_scope=0x1, corpus=IdentifierCorpus(),  # empty -- fail closed
            audit_log=audit_log, actor="maintainer1", subject_id="subject1",
        )

        assert result is None
        assert shared_atlas.atlas.size() == 0

        # Rejection is still recorded.
        audit_log.verify()
        assert audit_log._seq == 1

    def test_passed_fragment_mints_new_node_in_shared_store(
        self, private_atlas, shared_atlas, audit_log
    ):
        node_id = private_atlas.atlas.insert(0, _vec(1.0), "contact alice@example.com")
        corpus = IdentifierCorpus(redact_emails=True)

        new_id = promote_fragment(
            private_atlas, node_id, shared_atlas,
            dest_scope=0x4, corpus=corpus, audit_log=audit_log, actor="maintainer1", subject_id="subject1",
        )

        assert new_id is not None
        assert shared_atlas.atlas.size() == 1

        # Original private node is untouched (mint-not-flip).
        assert private_atlas.atlas.get_node_metadata(node_id) == "contact alice@example.com"
        # New shared node holds the DE-IDENTIFIED text, never the original.
        promoted_text = shared_atlas.atlas.get_node_metadata(new_id)
        assert "alice@example.com" not in promoted_text
        assert "[REDACTED_EMAIL]" in promoted_text

    def test_promoted_node_gets_dest_scope_and_governance_id(
        self, private_atlas, shared_atlas, audit_log
    ):
        node_id = private_atlas.atlas.insert(0, _vec(1.0), "clean text")
        corpus = IdentifierCorpus(patterns=["nomatch"])

        new_id = promote_fragment(
            private_atlas, node_id, shared_atlas,
            dest_scope=0x8, corpus=corpus, audit_log=audit_log, actor="maintainer1", subject_id="subject1",
        )

        assert shared_atlas.atlas.get_node_scope(new_id) == 0x8
        gov_id = shared_atlas.atlas.get_node_governance_id(new_id)
        assert gov_id == audit_log._seq  # points at the promotion's own audit record
        assert gov_id != 0

    def test_promotion_reuses_source_vector_by_default(
        self, private_atlas, shared_atlas, audit_log
    ):
        node_id = private_atlas.atlas.insert(0, _vec(0.75), "clean text")
        corpus = IdentifierCorpus(patterns=["nomatch"])

        new_id = promote_fragment(
            private_atlas, node_id, shared_atlas,
            dest_scope=0x1, corpus=corpus, audit_log=audit_log, actor="m1", subject_id="subject1",
        )

        source_vec = private_atlas.atlas.get_node_centroid(node_id)
        dest_vec = shared_atlas.atlas.get_node_centroid(new_id)
        assert np.allclose(source_vec, dest_vec, atol=1e-5)

    def test_reembed_fn_overrides_default_vector(
        self, private_atlas, shared_atlas, audit_log
    ):
        node_id = private_atlas.atlas.insert(0, _vec(0.1), "clean text")
        corpus = IdentifierCorpus(patterns=["nomatch"])
        called_with = []

        def fake_reembed(text):
            called_with.append(text)
            return _vec(0.9)

        new_id = promote_fragment(
            private_atlas, node_id, shared_atlas,
            dest_scope=0x1, corpus=corpus, audit_log=audit_log, actor="m1", subject_id="subject1",
            reembed_fn=fake_reembed,
        )

        assert called_with == ["clean text"]
        dest_vec = shared_atlas.atlas.get_node_centroid(new_id)
        assert np.allclose(dest_vec, _vec(0.9), atol=1e-5)

    def test_require_verification_off_by_default_ignores_missing_verification(
        self, private_atlas, shared_atlas, audit_log
    ):
        node_id = private_atlas.atlas.insert(0, _vec(1.0), "clean text")
        corpus = IdentifierCorpus(patterns=["nomatch"])

        new_id = promote_fragment(
            private_atlas, node_id, shared_atlas,
            dest_scope=0x1, corpus=corpus, audit_log=audit_log, actor="m1", subject_id="subject1",
        )

        assert new_id is not None
        assert shared_atlas.atlas.size() == 1

    def test_require_verification_rejects_when_missing(
        self, private_atlas, shared_atlas, audit_log
    ):
        node_id = private_atlas.atlas.insert(0, _vec(1.0), "clean text")
        corpus = IdentifierCorpus(patterns=["nomatch"])

        result = promote_fragment(
            private_atlas, node_id, shared_atlas,
            dest_scope=0x1, corpus=corpus, audit_log=audit_log, actor="m1", subject_id="subject1",
            require_verification=True,
        )

        assert result is None
        assert shared_atlas.atlas.size() == 0
        tail = audit_log.tail()
        assert tail[-1].action == "promotion_rejected"
        assert tail[-1].payload["reason_categories"] == ["VERIFICATION_REQUIRED_BUT_MISSING"]

    def test_require_verification_rejects_when_status_not_passed(
        self, private_atlas, shared_atlas, audit_log
    ):
        node_id = private_atlas.atlas.insert(0, _vec(1.0), "clean text")
        corpus = IdentifierCorpus(patterns=["nomatch"])

        result = promote_fragment(
            private_atlas, node_id, shared_atlas,
            dest_scope=0x1, corpus=corpus, audit_log=audit_log, actor="m1", subject_id="subject1",
            require_verification=True,
            verification=VerificationResult(status="failed", commit_sha="abc123"),
        )

        assert result is None
        assert shared_atlas.atlas.size() == 0
        tail = audit_log.tail()
        assert tail[-1].action == "promotion_rejected"
        assert tail[-1].payload["reason_categories"] == ["VERIFICATION_FAILED"]
        assert tail[-1].payload["verification_status"] == "failed"

    def test_require_verification_passes_when_status_passed(
        self, private_atlas, shared_atlas, audit_log
    ):
        node_id = private_atlas.atlas.insert(0, _vec(1.0), "clean text")
        corpus = IdentifierCorpus(patterns=["nomatch"])

        new_id = promote_fragment(
            private_atlas, node_id, shared_atlas,
            dest_scope=0x1, corpus=corpus, audit_log=audit_log, actor="m1", subject_id="subject1",
            require_verification=True,
            verification=VerificationResult(
                status="passed", commit_sha="abc123", verified_by="ci-run-9182"
            ),
        )

        assert new_id is not None
        assert shared_atlas.atlas.size() == 1
        tail = audit_log.tail()
        assert tail[-1].action == "promotion"
        assert tail[-1].payload["verification_status"] == "passed"
        assert tail[-1].payload["verification_commit_sha"] == "abc123"
        assert tail[-1].payload["verification_verified_by"] == "ci-run-9182"

    def test_verification_recorded_but_not_gating_when_require_verification_false(
        self, private_atlas, shared_atlas, audit_log
    ):
        # A caller can supply verification for audit-trail purposes even
        # on a deployment that hasn't turned gating on -- it must not be
        # silently dropped, and must not (falsely) gate either.
        node_id = private_atlas.atlas.insert(0, _vec(1.0), "clean text")
        corpus = IdentifierCorpus(patterns=["nomatch"])

        new_id = promote_fragment(
            private_atlas, node_id, shared_atlas,
            dest_scope=0x1, corpus=corpus, audit_log=audit_log, actor="m1", subject_id="subject1",
            verification=VerificationResult(status="failed"),  # would gate-reject if required
        )

        assert new_id is not None
        tail = audit_log.tail()
        assert tail[-1].payload["verification_status"] == "failed"

    def test_delta_arena_source_node_is_promotable(
        self, private_atlas, shared_atlas, audit_log
    ):
        # A same-turn admission not yet compacted must still be promotable.
        delta_id = private_atlas.atlas.insert_delta(_vec(0.5), "clean text")
        corpus = IdentifierCorpus(patterns=["nomatch"])

        new_id = promote_fragment(
            private_atlas, delta_id, shared_atlas,
            dest_scope=0x1, corpus=corpus, audit_log=audit_log, actor="m1", subject_id="subject1",
        )

        assert new_id is not None
        assert shared_atlas.atlas.get_node_metadata(new_id) == "clean text"

    def test_records_promoted_from_trace_edge_when_trace_given(
        self, private_atlas, shared_atlas, audit_log
    ):
        node_id = private_atlas.atlas.insert(0, _vec(0.2), "clean text")
        corpus = IdentifierCorpus(patterns=["nomatch"])
        trace = TraceGraph()

        new_id = promote_fragment(
            private_atlas, node_id, shared_atlas,
            dest_scope=0x1, corpus=corpus, audit_log=audit_log, actor="maintainer1", subject_id="subject1",
            trace=trace,
        )

        history = trace.get_history("maintainer1", limit=10)
        assert len(history) == 1
        ev = history[0]
        assert ev["role"] == TraceGraph.ROLE_CONCEPT
        assert ev["edge_type"] == int(EdgeType.PROMOTED_FROM)

        dest_raw, dest_is_shared = decode_store_id(ev["atlas_id"])
        assert dest_raw == new_id
        assert dest_is_shared is True

        src_raw, src_is_shared = decode_store_id(ev["supersedes_id"])
        assert src_raw == node_id
        assert src_is_shared is False

    def test_no_trace_edge_when_trace_omitted(
        self, private_atlas, shared_atlas, audit_log
    ):
        node_id = private_atlas.atlas.insert(0, _vec(0.2), "clean text")
        corpus = IdentifierCorpus(patterns=["nomatch"])

        promote_fragment(
            private_atlas, node_id, shared_atlas,
            dest_scope=0x1, corpus=corpus, audit_log=audit_log, actor="m1", subject_id="subject1",
        )  # trace=None (default) -- must not raise, and there's nothing to check

    def test_dest_scope_zero_rejected(self, private_atlas, shared_atlas, audit_log):
        node_id = private_atlas.atlas.insert(0, _vec(1.0), "clean text")
        corpus = IdentifierCorpus(patterns=["nomatch"])

        with pytest.raises(ValueError):
            promote_fragment(
                private_atlas, node_id, shared_atlas,
                dest_scope=0, corpus=corpus, audit_log=audit_log, actor="m1", subject_id="subject1",
            )
        # Rejected before touching the classifier/dest store at all.
        assert shared_atlas.atlas.size() == 0

    @pytest.mark.parametrize("blank_subject_id", ["", "   "])
    def test_blank_subject_id_rejected(
        self, private_atlas, shared_atlas, audit_log, blank_subject_id
    ):
        # task 6 Phase A: an empty/whitespace subject_id can never resolve
        # to a real (subject_id, dest_scope) DEK lookup -- same fail-closed
        # treatment as dest_scope == 0, above.
        node_id = private_atlas.atlas.insert(0, _vec(1.0), "clean text")
        corpus = IdentifierCorpus(patterns=["nomatch"])

        with pytest.raises(ValueError):
            promote_fragment(
                private_atlas, node_id, shared_atlas,
                dest_scope=0x1, corpus=corpus, audit_log=audit_log, actor="m1",
                subject_id=blank_subject_id,
            )
        assert shared_atlas.atlas.size() == 0

    def test_delta_diversion_at_dest_records_anomaly_and_raises(
        self, private_atlas, audit_log
    ):
        # advisor review (v4-plan.md Stage 4 task 2): Atlas::insert()
        # diverts to the delta buffer if dest_atlas is mid-compaction,
        # returning a delta-masked id -- set_node_scope()/
        # set_node_governance_id() both reject such ids outright. Confirm
        # promote_fragment() detects this BEFORE calling either, and
        # records the orphaned node id in the audit log before raising.
        node_id = private_atlas.atlas.insert(0, _vec(1.0), "clean text")
        corpus = IdentifierCorpus(patterns=["nomatch"])

        orphaned_id = NODE_ID_DELTA_MASK | 5
        mock_dest = MagicMock()
        mock_dest.atlas.insert.return_value = orphaned_id

        with pytest.raises(RuntimeError, match="delta buffer"):
            promote_fragment(
                private_atlas, node_id, mock_dest,
                dest_scope=0x1, corpus=corpus, audit_log=audit_log, actor="m1", subject_id="subject1",
            )

        mock_dest.atlas.set_node_scope.assert_not_called()
        mock_dest.atlas.set_node_governance_id.assert_not_called()

        audit_log.verify()
        lines = audit_log.path.read_text().splitlines()
        last = json.loads(lines[-1])
        assert last["action"] == "promotion_unscoped_anomaly"
        assert last["payload"]["dest_node_id"] == orphaned_id

    def test_set_scope_failure_after_insert_records_anomaly_and_reraises(
        self, private_atlas, audit_log
    ):
        # Narrower race: insert() returns a normal mmap id, but
        # set_node_scope() itself then fails (e.g. compaction started
        # concurrently) -- same orphaned-node shape, same safety net.
        node_id = private_atlas.atlas.insert(0, _vec(1.0), "clean text")
        corpus = IdentifierCorpus(patterns=["nomatch"])

        mock_dest = MagicMock()
        mock_dest.atlas.insert.return_value = 42  # normal mmap id
        mock_dest.atlas.set_node_scope.side_effect = RuntimeError("compacting")

        with pytest.raises(RuntimeError, match="compacting"):
            promote_fragment(
                private_atlas, node_id, mock_dest,
                dest_scope=0x1, corpus=corpus, audit_log=audit_log, actor="m1", subject_id="subject1",
            )

        audit_log.verify()
        lines = audit_log.path.read_text().splitlines()
        last = json.loads(lines[-1])
        assert last["action"] == "promotion_unscoped_anomaly"
        assert last["payload"]["dest_node_id"] == 42
