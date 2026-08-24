"""
V4 Stage 5 task 2: outcome-verified supersession (supersession.py).
"""
import numpy as np
import pytest

from aeon_py.client import AeonClient, decode_store_id, encode_store_id
from aeon_py.governance import AuditLog
from aeon_py.promotion import IdentifierCorpus, VerificationResult, promote_fragment
from aeon_py.supersession import (
    find_promoted_nodes_by_commit_sha,
    revoke_node_supersession,
    supersede_by_reverted_commit,
    supersede_node,
)
from aeon_py.trace import EdgeType, ReasonCode, TraceGraph


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
    return AuditLog(tmp_path / "audit.jsonl")


class TestSupersedeNode:
    def test_supersedes_and_records_audit_entry(self, shared_atlas, audit_log):
        node_id = shared_atlas.atlas.insert(0, _vec(0.5), "clean text")
        shared_atlas.atlas.set_node_scope(node_id, 0x1)

        supersede_node(
            shared_atlas, node_id, audit_log, actor="admin1",
            reason="manual correction", reason_code=ReasonCode.CORRECTION,
        )

        assert shared_atlas.atlas.is_node_superseded(node_id) is True
        tail = audit_log.tail()
        assert tail[-1].action == "supersession"
        assert tail[-1].payload["node_id"] == node_id
        assert tail[-1].payload["reason"] == "manual correction"
        assert tail[-1].payload["reason_code"] == int(ReasonCode.CORRECTION)

    def test_records_evidence_commit_sha_when_supplied(self, shared_atlas, audit_log):
        node_id = shared_atlas.atlas.insert(0, _vec(0.5), "clean text")

        supersede_node(
            shared_atlas, node_id, audit_log, actor="ci-bot",
            reason="cited commit abc123 was reverted",
            reason_code=ReasonCode.BUG_FIX_VERIFIED,
            evidence_commit_sha="abc123",
        )

        tail = audit_log.tail()
        assert tail[-1].payload["evidence_commit_sha"] == "abc123"

    def test_records_trace_edge_when_trace_given(self, shared_atlas, audit_log, tmp_path):
        trace = TraceGraph(tmp_path / "trace.bin")
        node_id = shared_atlas.atlas.insert(0, _vec(0.5), "clean text")

        supersede_node(
            shared_atlas, node_id, audit_log, actor="admin1", reason="test",
            trace=trace,
        )

        history = trace.get_history("admin1", limit=10)
        assert len(history) == 1
        ev = history[0]
        assert ev["role"] == 2  # concept
        encoded = encode_store_id(node_id, is_shared=True)
        assert int(ev["atlas_id"]) == encoded

    def test_no_trace_edge_when_trace_omitted(self, shared_atlas, audit_log):
        node_id = shared_atlas.atlas.insert(0, _vec(0.5), "clean text")
        # No trace= argument -- must not raise, must not record anything
        # Trace-side (there's no trace instance to record into).
        supersede_node(shared_atlas, node_id, audit_log, actor="admin1", reason="test")
        assert shared_atlas.atlas.is_node_superseded(node_id) is True

    def test_invalid_node_id_raises_before_any_audit_record(self, shared_atlas, audit_log):
        with pytest.raises(RuntimeError):
            supersede_node(shared_atlas, 999999, audit_log, actor="admin1", reason="test")
        # Rejected mutation must not produce a misleading audit entry.
        assert audit_log.tail() == []


class TestRevokeNodeSupersession:
    def test_revokes_and_records_audit_entry(self, shared_atlas, audit_log):
        node_id = shared_atlas.atlas.insert(0, _vec(0.5), "clean text")
        supersede_node(shared_atlas, node_id, audit_log, actor="admin1", reason="test")

        revoke_node_supersession(
            shared_atlas, node_id, audit_log, actor="admin2",
            reason="correction was wrong", reason_code=ReasonCode.CORRECTION,
        )

        assert shared_atlas.atlas.is_node_superseded(node_id) is False
        tail = audit_log.tail()
        assert tail[-1].action == "supersession_revoked"
        assert tail[-1].payload["node_id"] == node_id


class TestFindPromotedNodesByCommitSha:
    def test_finds_matching_promotion(self, private_atlas, shared_atlas, audit_log):
        corpus = IdentifierCorpus(patterns=["nomatch"])
        node_id = private_atlas.atlas.insert(0, _vec(1.0), "clean text")

        new_id = promote_fragment(
            private_atlas, node_id, shared_atlas,
            dest_scope=0x1, corpus=corpus, audit_log=audit_log, actor="m1", subject_id="subject1",
            require_verification=True,
            verification=VerificationResult(status="passed", commit_sha="abc123"),
        )

        found = find_promoted_nodes_by_commit_sha(audit_log, "abc123")
        assert found == [new_id]

    def test_returns_empty_when_no_promotion_cites_commit(self, private_atlas, shared_atlas, audit_log):
        corpus = IdentifierCorpus(patterns=["nomatch"])
        node_id = private_atlas.atlas.insert(0, _vec(1.0), "clean text")
        promote_fragment(
            private_atlas, node_id, shared_atlas,
            dest_scope=0x1, corpus=corpus, audit_log=audit_log, actor="m1", subject_id="subject1",
        )

        assert find_promoted_nodes_by_commit_sha(audit_log, "zzz999") == []

    def test_ignores_rejected_promotions(self, private_atlas, shared_atlas, audit_log):
        # A rejected promotion's payload has no dest_node_id -- must not
        # be mistaken for a match even if a caller supplied a matching
        # commit_sha on a rejected attempt.
        node_id = private_atlas.atlas.insert(0, _vec(1.0), "unredactable secret")
        promote_fragment(
            private_atlas, node_id, shared_atlas,
            dest_scope=0x1, corpus=IdentifierCorpus(),  # empty -- fail closed
            audit_log=audit_log, actor="m1", subject_id="subject1",
            require_verification=True,
            verification=VerificationResult(status="passed", commit_sha="abc123"),
        )

        assert find_promoted_nodes_by_commit_sha(audit_log, "abc123") == []


class TestSupersedeByRevertedCommit:
    def test_supersedes_every_node_citing_the_commit(self, private_atlas, shared_atlas, audit_log):
        corpus = IdentifierCorpus(patterns=["nomatch"])
        n1 = private_atlas.atlas.insert(0, _vec(1.0), "clean text 1")
        n2 = private_atlas.atlas.insert(0, _vec(2.0), "clean text 2")

        new1 = promote_fragment(
            private_atlas, n1, shared_atlas, dest_scope=0x1, corpus=corpus,
            audit_log=audit_log, actor="m1", subject_id="subject1",
            require_verification=True,
            verification=VerificationResult(status="passed", commit_sha="abc123"),
        )
        new2 = promote_fragment(
            private_atlas, n2, shared_atlas, dest_scope=0x1, corpus=corpus,
            audit_log=audit_log, actor="m1", subject_id="subject1",
            require_verification=True,
            verification=VerificationResult(status="passed", commit_sha="abc123"),
        )

        receipt = supersede_by_reverted_commit(
            shared_atlas, audit_log, "abc123", actor="ci-bot",
        )

        assert set(receipt["superseded"]) == {new1, new2}
        assert receipt["could_not_supersede"] == []
        assert shared_atlas.atlas.is_node_superseded(new1) is True
        assert shared_atlas.atlas.is_node_superseded(new2) is True

    def test_unrelated_commit_supersedes_nothing(self, private_atlas, shared_atlas, audit_log):
        corpus = IdentifierCorpus(patterns=["nomatch"])
        n1 = private_atlas.atlas.insert(0, _vec(1.0), "clean text")
        promote_fragment(
            private_atlas, n1, shared_atlas, dest_scope=0x1, corpus=corpus,
            audit_log=audit_log, actor="m1", subject_id="subject1",
            require_verification=True,
            verification=VerificationResult(status="passed", commit_sha="abc123"),
        )

        receipt = supersede_by_reverted_commit(
            shared_atlas, audit_log, "zzz999", actor="ci-bot",
        )

        assert receipt == {"superseded": [], "could_not_supersede": []}

    def test_authorize_callback_filters_out_unauthorized_nodes(
        self, private_atlas, shared_atlas, audit_log
    ):
        corpus = IdentifierCorpus(patterns=["nomatch"])
        n1 = private_atlas.atlas.insert(0, _vec(1.0), "clean text")
        new1 = promote_fragment(
            private_atlas, n1, shared_atlas, dest_scope=0x1, corpus=corpus,
            audit_log=audit_log, actor="m1", subject_id="subject1",
            require_verification=True,
            verification=VerificationResult(status="passed", commit_sha="abc123"),
        )

        receipt = supersede_by_reverted_commit(
            shared_atlas, audit_log, "abc123", actor="ci-bot",
            authorize=lambda node_id: False,
        )

        assert receipt["superseded"] == []
        assert receipt["could_not_supersede"] == [
            {"node_id": new1, "reason": "not authorized for this node's scope"}
        ]
        # Not mutated -- an unauthorized candidate must not be touched.
        assert shared_atlas.atlas.is_node_superseded(new1) is False
