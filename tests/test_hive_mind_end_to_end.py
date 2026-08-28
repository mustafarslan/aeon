"""The hive mind, end to end (v4-plan.md Stages 0 through 5).

WHY THIS FILE EXISTS. Every leg of the hive mind was already tested in isolation:
promotion, scope-filtered retrieval, outcome-verified supersession, shared-tier Dreaming,
crypto-erase and the audit chain each have their own suite. Nothing exercised the CHAIN.
A system composed of individually-correct parts can still fail at its joins, and the joins
are where this design's hardest invariants live: a node id that means one thing to the
audit log and another to the atlas after compaction, a summary that merges two subjects and
so can be erased for neither, a scope bit that filters retrieval but was never assigned.

This is the acceptance test for "is the hive mind done". It runs the full path a real
adopter takes:

    private ingest -> promotion (classifier + verification) -> cross-tenant retrieval
    under a scope mask -> outcome-triggered supersession -> per-subject crypto-erase
    -> audit chain verification

No mocks stand in for the kernel: real Atlas files, a real Trace, a real hash-chained audit
log on disk.
"""

import numpy as np
import pytest

from aeon_py.client import AeonClient
from aeon_py.governance import AuditLog
from aeon_py.promotion import IdentifierCorpus, VerificationResult, promote_fragment
from aeon_py.supersession import supersede_by_reverted_commit
from aeon_py.trace import TraceGraph

SCOPE_TEAM_A = 0x1
SCOPE_TEAM_B = 0x2
COMMIT = "a1b2c3d4e5f6"


def _vec(seed: float) -> list:
    return np.full(768, seed, dtype=np.float32).tolist()


@pytest.fixture
def rig(tmp_path):
    """Two physically separate stores, as Stage 4 task 1 requires."""
    private = AeonClient(tmp_path / "private.atlas")
    shared = AeonClient(tmp_path / "shared.atlas")
    return {
        "private": private,
        "shared": shared,
        "trace": TraceGraph(path=str(tmp_path / "t.trace")),
        "audit": AuditLog(tmp_path / "audit.jsonl"),
        "corpus": IdentifierCorpus(patterns=[r"PROJ-\d+"], redact_emails=True),
    }


def test_private_ingest_through_promotion_retrieval_and_supersession(rig):
    private, shared = rig["private"], rig["shared"]

    # --- 1. Private ingest. Contains an identifier the classifier must strip.
    src_id = private.atlas.insert(
        0, _vec(0.11),
        "deadlock on retry, see PROJ-1234, reported by alice@example.com", None)

    # --- 2. Promotion: mint-and-recontextualise into the shared tier, scoped to team A,
    #        gated on a passing verification citing a commit.
    new_id = promote_fragment(
        private, src_id, shared, dest_scope=SCOPE_TEAM_A,
        corpus=rig["corpus"], audit_log=rig["audit"], actor="alice",
        subject_id="subject-1", trace=rig["trace"],
        verification=VerificationResult(status="passed", commit_sha=COMMIT),
        require_verification=True,
    )
    assert new_id is not None, "a clean fragment with a passing verification must promote"

    # The promoted node is a NEW node, de-identified. The source is untouched.
    promoted_text = shared.atlas.get_node_metadata(new_id)
    assert "PROJ-1234" not in promoted_text, "identifier survived promotion"
    assert "alice@example.com" not in promoted_text, "email survived promotion"
    assert "PROJ-1234" in private.atlas.get_node_metadata(src_id), \
        "promotion must never mutate the source fragment"

    # --- 3. Scope assignment is real: the node carries the destination scope.
    assert shared.atlas.get_node_scope(new_id) == SCOPE_TEAM_A

    # --- 4. Cross-tenant retrieval respects the scope mask.
    #
    # A SECOND fragment is promoted to team B first. Without it this assertion would pass
    # against an empty store, proving only that nothing exists rather than that isolation
    # works. Both teams must now see exactly one node, and not each other's.
    src_b = private.atlas.insert(0, _vec(0.77), "billing timeout under load", None)
    id_b = promote_fragment(
        private, src_b, shared, dest_scope=SCOPE_TEAM_B,
        corpus=rig["corpus"], audit_log=rig["audit"], actor="bob",
        subject_id="subject-2", trace=rig["trace"])
    assert id_b is not None

    a_visible = list(shared.atlas.list_nodes_by_scope(SCOPE_TEAM_A))
    b_visible = list(shared.atlas.list_nodes_by_scope(SCOPE_TEAM_B))
    assert new_id in a_visible and id_b not in a_visible, \
        f"team A sees the wrong set: {a_visible}"
    assert id_b in b_visible and new_id not in b_visible, \
        f"team B sees the wrong set: {b_visible}"

    # A caller holding both bits sees both; the mask is a union, not an equality test.
    both = list(shared.atlas.list_nodes_by_scope(SCOPE_TEAM_A | SCOPE_TEAM_B))
    assert new_id in both and id_b in both

    # --- 5. Outcome-triggered supersession: the cited commit is reverted.
    assert not shared.atlas.is_node_superseded(new_id)
    receipt = supersede_by_reverted_commit(
        shared, rig["audit"], commit_sha=COMMIT, actor="ci-bot", trace=rig["trace"])
    assert new_id in receipt.get("superseded", []), f"not superseded; receipt={receipt}"
    assert shared.atlas.is_node_superseded(new_id), \
        "ground-truth supersession must reach the kernel flag, not only the audit log"
    assert not shared.atlas.is_node_superseded(id_b), \
        "supersession by commit must not touch a node that never cited that commit"

    # --- 6. The audit chain covers the whole run and verifies.
    rig["audit"].verify()


def test_a_failing_verification_blocks_promotion_when_gating_is_on(rig):
    """Correctness-gated promotion (Stage 4 task 3): the adopter's CI said the fix failed,
    so the fragment must not reach the shared tier at all."""
    src_id = rig["private"].atlas.insert(0, _vec(0.22), "a fix that did not hold", None)
    result = promote_fragment(
        rig["private"], src_id, rig["shared"], dest_scope=SCOPE_TEAM_A,
        corpus=rig["corpus"], audit_log=rig["audit"], actor="alice",
        subject_id="subject-1",
        verification=VerificationResult(status="failed", commit_sha=COMMIT),
        require_verification=True,
    )
    assert result is None
    assert len(list(rig["shared"].atlas.list_nodes_by_scope(SCOPE_TEAM_A))) == 0
    rig["audit"].verify()


def test_the_audit_chain_detects_tampering(rig):
    """The audit log is the evidence every other guarantee is argued from. If it can be
    edited undetectably, none of the rest is checkable."""
    src_id = rig["private"].atlas.insert(0, _vec(0.33), "a clean fragment", None)
    promote_fragment(
        rig["private"], src_id, rig["shared"], dest_scope=SCOPE_TEAM_A,
        corpus=rig["corpus"], audit_log=rig["audit"], actor="alice", subject_id="subject-1")
    rig["audit"].verify()

    path = rig["audit"].path
    lines = path.read_text().splitlines()
    assert lines, "promotion wrote no audit record"
    lines[0] = lines[0].replace('"alice"', '"mallory"')
    path.write_text("\n".join(lines) + "\n")

    with pytest.raises(Exception):
        AuditLog(path).verify()


# --- the erasure leg, which needs the control plane -------------------------------

import os

_DATABASE_URL = os.environ.get("AEON_CONTROL_PLANE_DATABASE_URL")


@pytest.mark.skipif(not _DATABASE_URL,
                    reason="AEON_CONTROL_PLANE_DATABASE_URL not set; the DEK store is "
                           "Postgres-backed, matching test_crypto_keystore.py's convention")
def test_crypto_erase_removes_one_subject_and_leaves_the_other(rig):
    """Per-subject-per-scope crypto-erase (Stage 4 task 6), the leg the in-process chain
    above cannot reach.

    This is where the one-subject-per-node invariant earns its keep: DEKs are keyed on
    (subject_id, scope), so a shared-tier summary that merged two subjects' fragments could
    be erased for neither. `consolidate_shared_scope()` enforces the invariant by grouping
    on subject before clustering; this asserts the consequence it exists to protect.
    """
    from aeon_py.crypto import Keystore

    keystore = Keystore(_DATABASE_URL, kek=b"0" * 32)
    dek_a = keystore.get_or_create_dek("subject-1", SCOPE_TEAM_A)
    dek_b = keystore.get_or_create_dek("subject-2", SCOPE_TEAM_B)
    assert dek_a != dek_b

    assert keystore.destroy_key("subject-1", SCOPE_TEAM_A) is True
    assert keystore.get_dek("subject-1", SCOPE_TEAM_A) is None, \
        "the erased subject's key must be gone, not merely unreferenced"
    assert keystore.get_dek("subject-2", SCOPE_TEAM_B) == dek_b, \
        "erasing one subject must not affect another's key"
