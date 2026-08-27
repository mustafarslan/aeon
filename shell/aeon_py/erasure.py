"""
Aeon Erasure Workflow (v4-plan.md Stage 4 task 5(c)) -- tracked erasure
cases with a completion receipt and an explicit "could not be erased"
section.

Reuses the four-eyes approval infrastructure task 7 already built
(control_plane/admin.py's AdminDB) wholesale rather than inventing a
parallel approval mechanism: erasing a subject's shared-tier fragments is
exactly the "bulk operation" task 7's admin console constraints require
four-eyes approval for. A case's EXACT target node ids and the scope
authorization required to execute it are locked into the approval
request's `target` JSON at creation time (mirrors promotion.py's
create_promotion_approval_request() -- same replay-safety reasoning: an
approval for "erase nodes [5, 9]" can never be replayed to erase a
different set once approved).

Scoped to the SHARED atlas store only -- see control_plane/schema.py's
erasure_cases doc comment for why private-store erasure is explicitly
deferred, not silently unsupported.

This is a LOGICAL delete (Atlas.tombstone_node()) -- physical bytes
survive until the next compact_mmap() reclaims them. Layered on top,
v4-plan.md Stage 4 task 6's actual crypto-erase (see the task 6 decision
record, added/updated 2026-08-23): Phase A (subject attribution --
promote_fragment()'s subject_id, DONE) and Phase B (encrypted shared-store
METADATA -- reshaped to blob/metadata-only, centroid vectors stay
plaintext) both feed execute_approved_erasure()'s `keystore` param, which
destroys the (subject_id, scope) DEK for every successfully-tombstoned
node when configured -- see that function's own doc comment for the
collateral-effect caveat (one DEK covers every node sharing a subject+
scope). A case's receipt still records what was logically tombstoned,
never a physical-mmap-bytes-destruction guarantee this module doesn't
make -- key destruction makes the metadata's CIPHERTEXT unreadable, which
is a real, independent guarantee, distinct from physical byte erasure.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

logger = logging.getLogger("aeon.governance")

from .client import AeonClient
from .governance import GOVERNANCE_RECORD_ACTIONS, AuditLog

_ACTION_ERASURE = "erasure"
assert _ACTION_ERASURE in GOVERNANCE_RECORD_ACTIONS, (
    "erasure.py's action constant drifted from governance.py's "
    "GOVERNANCE_RECORD_ACTIONS"
)

_APPROVAL_ACTION_ERASURE = "erasure"

# Substring of Atlas::tombstone_node()'s std::runtime_error message
# (atlas.cpp) for the ONE transient failure mode among its throws --
# "compaction is in progress" -- distinct from the permanent ones
# (invalid node id, also a runtime_error; delta-arena id, an
# invalid_argument/ValueError). String-matched rather than a dedicated
# exception type: the C++ layer doesn't distinguish transient from
# permanent failures at the type level today, and adding that
# distinction there is a bigger change than this increment's scope.
_TRANSIENT_FAILURE_MARKER = "compaction is in progress"


class ErasureTransientFailure(RuntimeError):
    """Raised by execute_approved_erasure() when a target node's
    tombstone_node() call failed for a TRANSIENT reason (the shared
    store is mid-compaction) rather than a permanent one (advisor
    review: the promotion path already had to distinguish "burn the
    approval" from "leave it retryable" once -- this is the same
    question on the erasure path). The case is left UNCOMPLETED
    (completed_at stays None, receipt stays None, the four-eyes approval
    is NOT consumed) so a retry once compaction finishes is not just
    possible but the correct next action -- safe because
    Atlas.tombstone_node() is idempotent for whatever this run already
    erased before hitting the transient node, and the shared store's
    compaction is store-wide, not per-node, so the remaining targets in
    this run are abandoned rather than attempted (they would hit the
    same failure)."""


if TYPE_CHECKING:
    from .control_plane.admin import AdminDB
    from .control_plane.db import GovernanceDB
    from .control_plane.erasure_db import ErasureDB
    from .crypto import Keystore


def create_erasure_case(
    admin_db: "AdminDB",
    erasure_db: "ErasureDB",
    *,
    shared_atlas: AeonClient,
    node_ids: List[int],
    reason: str,
    requested_by: str,
    expires_at: datetime,
    required_approvals: int = 2,
    session_ids: Optional[List[str]] = None,
) -> int:
    """Creates a four-eyes approval request for erasing `node_ids` from
    `shared_atlas`, plus the erasure_cases row tracking its completion.
    Returns the new case id.

    The authorizing scope_mask is computed HERE, at request time, as the
    OR of every target node's CURRENT scope_bitmap, and locked into the
    approval request's target JSON alongside the node ids -- the same
    node ids and scope this request is approved against are exactly what
    execute_approved_erasure() reads back and enforces later, not
    whatever the node's scope_bitmap happens to be AT EXECUTION time
    (which a concurrent admin action could have changed in between).

    Raises:
        ValueError: node_ids is empty -- an erasure case with nothing to
            erase is not a meaningful case to create.
    """
    if not node_ids:
        raise ValueError("create_erasure_case: node_ids must be non-empty")

    scope_mask = 0
    for node_id in node_ids:
        scope_mask |= shared_atlas.atlas.get_node_scope(node_id)

    # v4.1: the DERIVED-RECORD CASCADE, locked into the approval request alongside the
    # node ids. Records are PII derived from conversation -- `records.py` names
    # `provenance.session_id` as the right-to-erasure cascade index -- and until now
    # nothing cascaded: `records_for_session()` had zero non-test callers, so erasing a
    # node left every record extracted from that session in place.
    #
    # Sessions are named EXPLICITLY rather than derived from the node ids, because they
    # cannot be derived: `Atlas.insert(..., session_id)` routes the SLB cache lookup and
    # is not stored on the node, and `drop_session()` drops a cache entry, not data. There
    # is no node -> session getter. Naming them here is also the auditable choice -- a
    # four-eyes request should state everything it will destroy, and the cascade is
    # therefore approved rather than inferred at execution time.
    target = json.dumps({"node_ids": node_ids, "scope_mask": scope_mask,
                         "session_ids": list(session_ids or [])})
    request_id = admin_db.create_approval_request(
        action=_APPROVAL_ACTION_ERASURE,
        target=target,
        reason=reason,
        requested_by=requested_by,
        expires_at=expires_at,
        required_approvals=required_approvals,
    )
    return erasure_db.create_case(approval_request_id=request_id)


def cascade_to_derived_records(store_for, session_ids) -> tuple[List[int], List[dict]]:
    """Tombstone every record derived from `session_ids`. Returns `(cascaded, failures)`.

    `store_for` is a **resolver**, `tenant -> RecordStore | None`, not a single store, and
    that shape is load-bearing: records live in per-tenant files, so the store holding a
    subject's records is the SUBJECT's, never the admin actor's who executed the erasure.
    Passing one store here would have cascaded against whoever pressed the button. A single
    store is still accepted for tests and for callers with one tenant -- see below.

    Extracted from `execute_approved_erasure()` so it is testable without a live control
    plane -- the erasure workflow tests need Postgres and are opt-in locally, which is
    exactly how a cascade could ship unverified.

    Best-effort per record: one record that will not tombstone must not abort the rest of
    the cascade, nor undo the node erasure that has already happened. Failures are returned
    and land in the receipt rather than being silently dropped -- the same discipline the
    node loop above uses for `could_not_erase`.
    """
    cascaded: List[int] = []
    failures: List[dict] = []
    if store_for is None or not session_ids:
        return cascaded, failures
    resolve = store_for if callable(store_for) else (lambda _t, _s=store_for: _s)
    for session_id in session_ids:
        record_store = resolve(session_id)
        if record_store is None:
            continue
        for rec in record_store.records_for_session(session_id):
            if rec.node_id is None:
                continue
            try:
                record_store.atlas.tombstone_node(rec.node_id)
                cascaded.append(rec.node_id)
            except Exception as exc:
                failures.append({"node_id": rec.node_id, "session_id": session_id,
                                 "reason": f"{type(exc).__name__}: {exc}"})
    return cascaded, failures


def execute_approved_erasure(
    admin_db: "AdminDB",
    erasure_db: "ErasureDB",
    case_id: int,
    *,
    actor: str,
    shared_atlas: AeonClient,
    audit_log: AuditLog,
    governance_db: Optional["GovernanceDB"] = None,
    keystore: Optional["Keystore"] = None,
    record_store: Optional[object] = None,
    store_for: Optional[object] = None,
) -> dict:
    """Executes an already-approved erasure case: tombstones every target
    node id it can, and records a completion receipt for all of them --
    the ones that succeeded AND the ones that didn't, explicitly, never
    silently dropping a failure.

    Per-node-id failures (an id that no longer exists, or was already a
    delta-arena id, or the store is mid-compaction) are caught
    individually and land in the receipt's `could_not_erase` list with
    the reason -- they do NOT abort the whole case, since a case spanning
    N node ids should erase the N-1 it can rather than erase nothing
    because of one bad id.

    keystore: Optional (crypto.py's Keystore, v4-plan.md Stage 4 task 6
        Phase B) -- when supplied ALONGSIDE governance_db (both are
        needed to resolve a node's (subject_id, scope) DEK), each
        successfully-tombstoned node's key is destroyed too, best-effort
        (a destruction failure doesn't undo the tombstone). None (the
        default) means this case only logically tombstones, same as
        before Phase B existed.

    Crash-resumable: `completed_at` is only set at the very end, after
    every node id has been attempted -- if this process is killed
    mid-loop, `completed_at` is still None (the replay guard below does
    not trigger) and calling this again for the SAME case_id is safe:
    Atlas.tombstone_node() is idempotent, so re-attempting an
    already-tombstoned id is a no-op, not a double-erasure.

    Raises:
        ValueError: no such case, or its approval request isn't actually
            an erasure request (defensive -- shouldn't happen given
            erasure_cases.approval_request_id's FK, but a request row
            reused across action types would be a real bug worth
            surfacing loudly).
        RuntimeError: this case was already completed (replay guard, same
            shape as promotion's execute_approved_promotion()).
        ErasureTransientFailure (a RuntimeError subclass): the shared
            store was mid-compaction when a target node was reached --
            see its own doc comment. Caught separately from a plain
            RuntimeError by any caller that wants to tell "retry
            shortly" apart from "already executed".
        PermissionError: not currently approved (insufficient distinct
            approvers, expired, or revoked).
    """
    case = erasure_db.get_case(case_id)
    if case is None:
        raise ValueError(f"execute_approved_erasure: no such case {case_id}")
    if case["completed_at"] is not None:
        raise RuntimeError(
            f"execute_approved_erasure: case {case_id} was already "
            f"completed at {case['completed_at']}"
        )

    request_id = case["approval_request_id"]
    req = admin_db.get_request(request_id)
    if req is None or req["action"] != _APPROVAL_ACTION_ERASURE:
        raise ValueError(
            f"execute_approved_erasure: case {case_id}'s approval request "
            f"{request_id} is not a valid erasure request"
        )
    if not admin_db.is_approved(request_id):
        raise PermissionError(
            f"execute_approved_erasure: request {request_id} is not "
            "currently approved (insufficient distinct approvers, "
            "expired, or revoked)"
        )

    params = json.loads(req["target"])
    node_ids = params["node_ids"]
    scope_mask = int(params["scope_mask"])
    session_ids = params.get("session_ids", [])

    erased: List[int] = []
    could_not_erase: List[dict] = []
    for node_id in node_ids:
        try:
            shared_atlas.atlas.tombstone_node(node_id)
            erased.append(node_id)

            # v4-plan.md Stage 4 task 6 Phase B: the ACTUAL key destruction
            # step -- task 6's own gate ("demonstrates actual key
            # destruction end-to-end, not just a tombstone flag"). Only
            # reached after a successful tombstone (not preemptively -- a
            # node that fails to erase above must not have its key
            # destroyed out from under it), and only when keystore/
            # governance_db are BOTH configured -- get_node_scope()/
            # get_node_governance_id() are real EBR-guarded reads, not
            # free, so this deployment pays that cost only when it has
            # actually opted into crypto-erase (tombstoning doesn't
            # change either value, so reading them after the tombstone
            # above is equivalent to reading them before it). Best-effort:
            # a resolution/destruction failure here must not undo the
            # tombstone above -- the logical delete is the primary,
            # always-attempted guarantee; key destruction is an
            # additional guarantee layered on top.
            #
            # Collateral-effect note, stated rather than left implicit:
            # one DEK covers EVERY node from this (subject_id, scope)
            # pair, so destroying it also makes any OTHER, still-live
            # (non-tombstoned) node sharing the same subject+scope
            # undecryptable too. Acceptable because this module's own
            # docstring frames a case as erasing "a SUBJECT's shared-tier
            # fragments" (i.e. all of them in-scope, not a hand-picked
            # subset) -- a case that deliberately erases only SOME of a
            # subject's fragments in a scope would have this side effect
            # on the ones left behind, which is a caller-workflow
            # consideration, not a bug in this function.
            if keystore is not None and governance_db is not None:
                node_scope = shared_atlas.atlas.get_node_scope(node_id)
                gov_id = shared_atlas.atlas.get_node_governance_id(node_id)
                subject_id = governance_db.get_subject_id(gov_id) if gov_id else None
                if subject_id is not None:
                    try:
                        keystore.destroy_key(subject_id, node_scope)
                    except Exception:
                        logger.warning(
                            "execute_approved_erasure: key destruction failed "
                            "for node %d (subject_id=%r, scope=%d) -- the "
                            "node is already tombstoned regardless",
                            node_id, subject_id, node_scope, exc_info=True,
                        )
        except RuntimeError as e:
            if _TRANSIENT_FAILURE_MARKER in str(e):
                raise ErasureTransientFailure(
                    f"execute_approved_erasure: case {case_id} aborted -- "
                    f"node {node_id} hit a transient failure ({e}). "
                    f"{len(erased)} of {len(node_ids)} targets were "
                    "already erased in this attempt (safe, idempotent); "
                    "the case is NOT marked completed and the approval "
                    "is NOT consumed -- retry shortly."
                ) from e
            could_not_erase.append({"node_id": node_id, "reason": str(e)})
        except ValueError as e:
            could_not_erase.append({"node_id": node_id, "reason": str(e)})

    # THE DERIVED-RECORD CASCADE. Runs after the node tombstones because the logical
    # delete of the named nodes is the primary, always-attempted guarantee -- the same
    # ordering the crypto-erase step above uses, and for the same reason. Best-effort per
    # record: one record that will not tombstone must not abort the rest of the cascade or
    # undo the node erasure, and it is reported rather than silently dropped.
    cascaded, could_not_cascade = cascade_to_derived_records(
        store_for if store_for is not None else record_store, session_ids)

    receipt = {"erased": erased, "could_not_erase": could_not_erase,
               "cascaded_records": cascaded,
               "could_not_cascade_records": could_not_cascade,
               "cascade_session_ids": list(session_ids)}

    seq = audit_log.append(
        action=_ACTION_ERASURE,
        actor=actor,
        payload={
            "case_id": case_id,
            "scope_mask": scope_mask,
            "erased_count": len(erased),
            "could_not_erase_count": len(could_not_erase),
        },
    )
    if governance_db is not None:
        try:
            # v4-plan.md Stage 4 task 6 Phase A: governance_records.subject_id
            # is now NOT NULL on every row, including erasure's -- a real
            # regression Phase A introduced here and left undetected (no
            # existing test exercised this branch with governance_db
            # configured; server.py's execute_erasure endpoint DOES pass
            # one, so any real deployment with a control plane configured
            # would have hit a TypeError on every erasure execution).
            # Resolved via the first successfully-erased node's own
            # governance_record_id -- erasure.py's own docstring frames a
            # case as "erasing a SUBJECT's shared-tier fragments"
            # (singular), so this is the representative subject for the
            # whole case, not a guess. Falls back to a documented sentinel
            # if nothing could be resolved (no nodes erased, or governance
            # lookup itself failed) -- there is no meaningful single
            # subject to name in that case, but the column still requires
            # a non-empty value.
            subject_id = "unknown"
            for node_id in erased:
                gov_id = shared_atlas.atlas.get_node_governance_id(node_id)
                resolved = governance_db.get_subject_id(gov_id) if gov_id else None
                if resolved:
                    subject_id = resolved
                    break

            governance_db.record(
                log_instance_id=audit_log.instance_id,
                log_instance_path=str(audit_log.path),
                log_seq=seq,
                action=_ACTION_ERASURE,
                actor=actor,
                subject_id=subject_id,
                dest_scope=scope_mask,
            )
        except Exception:
            # Best-effort mirror, same as promotion.py's
            # _mirror_governance_record() -- the JSONL record above is
            # already durable and authoritative regardless of whether
            # this succeeds.
            logger.warning(
                "execute_approved_erasure: best-effort Postgres mirror "
                "of erasure (case=%d, log_seq=%d) failed -- the JSONL "
                "record is still authoritative and unaffected",
                case_id, seq, exc_info=True,
            )

    erasure_db.complete_case(case_id, receipt=json.dumps(receipt))
    return receipt
