"""
Aeon Supersession — audited, reversible node exclusion for the shared tier
(v4-plan.md Stage 5 task 2: outcome-verified supersession).

Aeon does not integrate with any VCS/CI provider itself -- same trust
boundary as promotion.py's `VerificationResult` (Stage 4 task 3). An
external caller (the org's own CI/commit-graph integration) determines
that a commit was reverted or replaced and tells Aeon so; this module
looks up which promoted shared-store nodes cited that commit
(`promote_fragment()`'s `VerificationResult.commit_sha`, recorded in the
audit log at promotion time) and supersedes each one, recording why.

`supersede_node()`/`revoke_node_supersession()` are also the general
audited primitive for ANY reason a shared-store node needs to be
superseded, not just a reverted commit -- closing a pre-existing gap
found while building this (advisor review is not required to see it,
grep confirms it directly): server.py's `/admin/knowledge/{node_id}`
console action route called `Atlas.supersede_node()`/
`revoke_node_supersede()` directly with NO audit trail at all, unlike
every other governance-mutating path in this codebase (promotion,
erasure). Both that route and this module's own commit-triggered entry
point now funnel through the same audited function.

Known, pre-existing limitation shared with erasure.py and the knowledge-
browser route (not introduced or worsened here): a raw Atlas node id can
shift after `compact_mmap()` reclaims tombstoned slots. Nothing in this
codebase's existing node-id-based admin operations re-verifies identity
against `governance_record_id` before mutating by raw id, and this module
follows that same established convention rather than inventing a new one
just for this path.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

logger = logging.getLogger("aeon.governance")

from .client import AeonClient, encode_store_id
from .governance import AuditLog
from .trace import EdgeType, ReasonCode, TraceGraph

if TYPE_CHECKING:
    from .control_plane.db import GovernanceDB

_ACTION_SUPERSESSION = "supersession"
_ACTION_SUPERSESSION_REVOKED = "supersession_revoked"

# tail()'s `limit` has no "give me everything" sentinel of its own --
# passing a value this large just means "read to EOF, same cost class as
# verify()/export_signed()" (both already read the whole file; this log
# is not expected to reach a size where that matters, per tail()'s own
# doc comment).
_UNBOUNDED_TAIL = 2**63 - 1


def _resolve_subject_id(
    dest_atlas: AeonClient, node_id: int, governance_db: Optional["GovernanceDB"]
) -> Optional[str]:
    """Best-effort subject_id lookup for the Postgres mirror -- None if no
    control plane is configured, or if this node has no governance_record_id
    (never promoted through promote_fragment(), e.g. an admin action on
    ordinary shared-tier content), or if the lookup itself fails. Mirrors
    promote_fragment()'s own best-effort-mirror philosophy: never blocks
    or fails the JSONL write, which is authoritative regardless."""
    if governance_db is None:
        return None
    try:
        gov_id = dest_atlas.atlas.get_node_governance_id(node_id)
    except Exception:
        return None
    if gov_id == 0:
        return None
    try:
        return governance_db.get_subject_id(gov_id)
    except Exception:
        logger.warning(
            "supersession: best-effort subject_id lookup for node %d "
            "(governance_record_id=%d) failed -- Postgres mirror will be "
            "skipped, JSONL record is unaffected",
            node_id, gov_id, exc_info=True,
        )
        return None


def supersede_node(
    dest_atlas: AeonClient,
    node_id: int,
    audit_log: AuditLog,
    actor: str,
    reason: str,
    reason_code: ReasonCode = ReasonCode.UNSPECIFIED,
    evidence_commit_sha: Optional[str] = None,
    trace: Optional[TraceGraph] = None,
    governance_db: Optional["GovernanceDB"] = None,
) -> None:
    """Reversibly excludes a shared-store node from beam search results
    (Atlas.supersede_node(), branchless, idempotent) AND records why --
    the audited primitive both this module's commit-triggered entry point
    and server.py's admin console action route use.

    Args:
        dest_atlas: The SHARED store the node lives in.
        node_id: Raw (not store-encoded) shared-store node id.
        audit_log: Where this supersession is recorded (always, regardless
            of whether governance_db/trace are supplied).
        actor: Identity of whoever/whatever triggered this -- an admin
            operator, or (for the commit-triggered path) a caller identity
            representing the external CI/commit-graph integration.
        reason: Free-text rationale, recorded in the audit log payload.
        reason_code: ReasonCode.UNSPECIFIED unless a more specific one
            applies (e.g. ReasonCode.BUG_FIX_VERIFIED for the commit-
            revert case, ReasonCode.CORRECTION for a manual admin fix).
        evidence_commit_sha: Optional -- the commit whose revert triggered
            this, if applicable. Recorded verbatim in the audit payload,
            same reasoning as VerificationResult.commit_sha (promotion.py)
            for why this is not run through the redaction classifier: it's
            operational metadata about WHY an admin action happened, not
            fragment text entering the shared store for a broader
            audience to read.
        trace: Optional -- if given, also records a SUPERSEDES Trace edge
            (atlas_id == supersedes_id == this node, since nothing NEW is
            minted here, unlike PROMOTED_FROM's cross-node edge) for the
            knowledge-browser's future provenance view.
        governance_db: Optional -- best-effort Postgres mirror, same
            optionality convention as promote_fragment(). Silently skipped
            (with a warning) if this node has no governance_record_id or
            the lookup/write fails; the JSONL record is authoritative
            regardless.

    Raises:
        Whatever Atlas.supersede_node() raises (invalid_argument for a
        delta-arena id, runtime_error for an invalid id or a compaction in
        progress) -- propagates BEFORE any audit record is written, so a
        rejected mutation never produces a misleading "this happened"
        record.
    """
    dest_atlas.atlas.supersede_node(node_id)

    subject_id = _resolve_subject_id(dest_atlas, node_id, governance_db)

    payload: Dict[str, Any] = {
        "node_id": node_id,
        "reason": reason,
        "reason_code": int(reason_code),
    }
    if evidence_commit_sha is not None:
        payload["evidence_commit_sha"] = evidence_commit_sha
    seq = audit_log.append(action=_ACTION_SUPERSESSION, actor=actor, payload=payload)

    if governance_db is not None and subject_id is not None:
        try:
            governance_db.record(
                log_instance_id=audit_log.instance_id,
                log_instance_path=str(audit_log.path),
                log_seq=seq,
                action=_ACTION_SUPERSESSION,
                actor=actor,
                subject_id=subject_id,
                dest_node_id=node_id,
                dest_scope=dest_atlas.atlas.get_node_scope(node_id),
            )
        except Exception:
            logger.warning(
                "supersede_node: best-effort Postgres mirror of "
                "supersession (log_seq=%d, node=%d) failed -- the JSONL "
                "record is still authoritative and unaffected",
                seq, node_id, exc_info=True,
            )

    if trace is not None:
        encoded = encode_store_id(node_id, is_shared=True)
        trace.add_event(
            actor,
            "concept",
            f"[Superseded {node_id}: {reason}]",
            atlas_id=encoded,
            edge_type=EdgeType.SUPERSEDES,
            supersedes_id=encoded,
            reason_code=reason_code,
        )


def revoke_node_supersession(
    dest_atlas: AeonClient,
    node_id: int,
    audit_log: AuditLog,
    actor: str,
    reason: str,
    reason_code: ReasonCode = ReasonCode.UNSPECIFIED,
    trace: Optional[TraceGraph] = None,
    governance_db: Optional["GovernanceDB"] = None,
) -> None:
    """Reverses a prior supersede_node() call (Atlas.revoke_node_supersede(),
    idempotent, no-op if not currently superseded) AND records why -- the
    Article-16-equivalent correction flow's counterpart to supersede_node()
    above. Same audit/Postgres-mirror/Trace-edge shape, action=
    "supersession_revoked", edge_type=EdgeType.REVOKES.
    """
    dest_atlas.atlas.revoke_node_supersede(node_id)

    subject_id = _resolve_subject_id(dest_atlas, node_id, governance_db)

    payload: Dict[str, Any] = {
        "node_id": node_id,
        "reason": reason,
        "reason_code": int(reason_code),
    }
    seq = audit_log.append(
        action=_ACTION_SUPERSESSION_REVOKED, actor=actor, payload=payload
    )

    if governance_db is not None and subject_id is not None:
        try:
            governance_db.record(
                log_instance_id=audit_log.instance_id,
                log_instance_path=str(audit_log.path),
                log_seq=seq,
                action=_ACTION_SUPERSESSION_REVOKED,
                actor=actor,
                subject_id=subject_id,
                dest_node_id=node_id,
                dest_scope=dest_atlas.atlas.get_node_scope(node_id),
            )
        except Exception:
            logger.warning(
                "revoke_node_supersession: best-effort Postgres mirror "
                "(log_seq=%d, node=%d) failed -- the JSONL record is "
                "still authoritative and unaffected",
                seq, node_id, exc_info=True,
            )

    if trace is not None:
        encoded = encode_store_id(node_id, is_shared=True)
        trace.add_event(
            actor,
            "concept",
            f"[Revoked supersession {node_id}: {reason}]",
            atlas_id=encoded,
            edge_type=EdgeType.REVOKES,
            supersedes_id=encoded,
            reason_code=reason_code,
        )


def find_promoted_nodes_by_commit_sha(audit_log: AuditLog, commit_sha: str) -> List[int]:
    """Scans the audit log's "promotion" records for ones whose
    VerificationResult.commit_sha (promotion.py) matches `commit_sha`,
    returning the (raw, shared-store) dest_node_id of each. This is the
    lookup half of "a fragment citing a commit SHA is auto-superseded when
    that commit is reverted" (v4-plan.md Stage 5 task 2) -- Aeon's own
    JSONL audit log is the source of truth for this, not a new Postgres
    column: `promote_fragment()` already writes `verification_commit_sha`
    into every promotion's audit payload (task 3), and the audit log is
    the authoritative record (Postgres, when configured, is a queryable
    index over it, per governance.py's own module docstring), so this
    needs no new plumbing.

    A full linear scan of the log, same cost class as AuditLog.verify()/
    tail()'s own doc comment -- this is an occasional governance operation
    (a commit revert), not a hot path.

    Returns an empty list if nothing promoted ever cited this commit (not
    an error -- most commits are never reverted).
    """
    records = audit_log.tail(limit=_UNBOUNDED_TAIL)
    return [
        rec.payload["dest_node_id"]
        for rec in records
        if rec.action == "promotion"
        and rec.payload.get("verification_commit_sha") == commit_sha
        and rec.payload.get("dest_node_id") is not None
    ]


def supersede_by_reverted_commit(
    dest_atlas: AeonClient,
    audit_log: AuditLog,
    commit_sha: str,
    actor: str,
    trace: Optional[TraceGraph] = None,
    governance_db: Optional["GovernanceDB"] = None,
    authorize=None,
) -> Dict[str, List[Any]]:
    """v4-plan.md Stage 5 task 2's concrete entry point: an external
    caller has determined `commit_sha` was reverted or replaced and tells
    Aeon so. Supersedes every promoted node whose VerificationResult
    cited this commit, with reason_code=BUG_FIX_VERIFIED.

    Per-node, best-effort, not all-or-nothing -- same "partial outcome is
    a legitimate, auditable completion" philosophy as erasure.py's own
    receipt shape (a case with 14 cited nodes where 1 was already
    physically reclaimed by a compaction since promotion should still
    supersede the other 13, not abort the whole batch on the first
    failure).

    Args:
        authorize: Optional `Callable[[int], bool]` (raw node_id ->
            authorized?), consulted per candidate BEFORE attempting to
            supersede it -- an unauthorized node is recorded in
            `could_not_supersede` with a generic reason, never mutated.
            None (the default) means "authorize everything" -- the right
            default for a trusted, non-HTTP caller (e.g. a CI script
            running with full local trust); server.py's HTTP endpoint
            supplies a per-node scope-containment check here instead of
            re-implementing this whole function's batching/receipt logic.

    Returns:
        {"superseded": [node_id, ...], "could_not_supersede": [{"node_id":
        ..., "reason": ...}, ...]}. Both lists reference RAW node ids.
        `find_promoted_nodes_by_commit_sha()` returning zero results
        (nothing cited this commit) is a legitimate outcome, not an error
        -- both lists are simply empty.
    """
    node_ids = find_promoted_nodes_by_commit_sha(audit_log, commit_sha)
    superseded: List[int] = []
    could_not_supersede: List[Dict[str, Any]] = []

    for node_id in node_ids:
        if authorize is not None and not authorize(node_id):
            could_not_supersede.append(
                {"node_id": node_id, "reason": "not authorized for this node's scope"}
            )
            continue
        try:
            supersede_node(
                dest_atlas,
                node_id,
                audit_log,
                actor,
                reason=f"cited commit {commit_sha} was reverted or replaced",
                reason_code=ReasonCode.BUG_FIX_VERIFIED,
                evidence_commit_sha=commit_sha,
                trace=trace,
                governance_db=governance_db,
            )
            superseded.append(node_id)
        except (ValueError, RuntimeError) as e:
            # e.g. the node's raw id no longer resolves to the node this
            # audit record meant (compact_mmap() reclaimed/shifted ids
            # since promotion -- a known, pre-existing limitation shared
            # with erasure.py, see this module's own doc comment) --
            # discoverable in the receipt, not a silently dropped result.
            could_not_supersede.append({"node_id": node_id, "reason": str(e)})

    return {"superseded": superseded, "could_not_supersede": could_not_supersede}
