"""
Aeon Promotion — mint-and-recontextualize (v4-plan.md Stage 4 task 2).

Promotion never mutates or flips a bit on a source fragment -- it reads a
PRIVATE-store node's text and vector, runs a fail-closed deterministic
identifier-corpus classifier over the text, and if (and only if) the
classifier clears it, inserts a NEW, de-identified node into the SHARED
store, links it back via a PROMOTED_FROM Trace edge, and appends an audit
record (governance.py) whose seq number becomes the new node's
governance_record_id (Stage 1's NodeHeader field, previously allocated
but unwritten until Stage 4 task 1 gave it a setter) -- UNLESS a
`governance_db` (control_plane/db.py) is supplied, in which case the
JSONL audit record is additionally mirrored into Postgres and the
resulting, globally-stable row id is used instead (see promote_fragment's
`governance_db` parameter doc). The JSONL log remains the authoritative
tamper-evident chain either way; Postgres is a queryable index over it,
not a replacement.

Fail-closed, by design: the deterministic classifier is the ONLY layer
permitted to PASS a fragment. There is deliberately no optional
LLM/secondary layer in this module that can override a reject into a
pass -- a caller wanting an additional LLM-based reject/flag layer runs
it BEFORE calling promote_fragment() and simply doesn't call this
function if that layer flags the fragment. An empty/unconfigured
IdentifierCorpus rejects everything rather than silently passing raw
content through (see IdentifierCorpus.is_empty()'s doc comment).

Destination-conditioned re-embedding (task 2's "free retrieval-quality
upside") is closed via `reembed_fn` (text -> vector): this module never
runs an embedding model itself (there's no embedding-conditioning
pipeline anywhere in this repo), so a caller with one supplies it as a
callback; server.py's execute_promotion() wires the HTTP-level
`PromotionExecuteRequest.destination_embedding` field through as exactly
such a callback (a closure that ignores the redacted text and returns the
caller's own precomputed vector). Omitted, `reembed_fn` defaults to
reusing the source vector unmodified -- correctness-preserving even if
not quality-optimal.

Correctness-gated promotion for code knowledge (task 3) is closed the
same way: Aeon does not integrate with any VCS/CI provider itself (a
genuinely separate integration surface -- auth, rate limits, a provider
abstraction across GitHub/GitLab/etc. -- from what this module does).
Instead `VerificationResult` is a caller-supplied outcome (the adopter's
own CI/test runner already ran and knows the result); `require_verification`
turns on fail-closed gating on it. Both default to "off" (verification
recorded if given, not required) so no existing caller's behavior changes.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Callable, List, Optional

logger = logging.getLogger("aeon.governance")

from .client import AeonClient, NODE_ID_DELTA_MASK, decode_store_id, encode_store_id
from .governance import GOVERNANCE_RECORD_ACTIONS, AuditLog
from .trace import EdgeType, TraceGraph

# Indexed by name, not positional unpacking -- GOVERNANCE_RECORD_ACTIONS
# grew a 4th value (erasure.py's "erasure") after this module was
# written; positional unpacking (`a, b, c = GOVERNANCE_RECORD_ACTIONS`)
# would have broken silently-at-import-time the moment that happened.
_ACTION_PROMOTION = "promotion"
_ACTION_REJECTED = "promotion_rejected"
_ACTION_ANOMALY = "promotion_unscoped_anomaly"
assert {_ACTION_PROMOTION, _ACTION_REJECTED, _ACTION_ANOMALY} <= set(
    GOVERNANCE_RECORD_ACTIONS
), "promotion.py's action constants drifted from governance.py's GOVERNANCE_RECORD_ACTIONS"

# v4-plan.md Stage 4 task 7: the approval_requests.action value for a
# promotion request -- distinct from GOVERNANCE_RECORD_ACTIONS above,
# which enumerates governance_records.action (promotion OUTCOMES), not
# approval_requests.action (what operation is being REQUESTED). The two
# tables' action columns are independent vocabularies.
_APPROVAL_ACTION_PROMOTION = "promotion"

if TYPE_CHECKING:
    # control_plane/db.py and control_plane/admin.py need sqlalchemy/
    # psycopg (the optional `db` extras, pyproject.toml) -- NOT imported
    # eagerly here, since Postgres is optional infrastructure
    # (governance_db defaults to None below) and this module must keep
    # working with zero DB dependencies installed, same as every other
    # Stage 4 optional-infra pattern (shared_atlas_client,
    # AEON_ENABLE_DEBUG_ENDPOINTS).
    from .control_plane.admin import AdminDB
    from .control_plane.db import GovernanceDB
    from .crypto import Keystore

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_COMMIT_SHA_RE = re.compile(r"\b[0-9a-f]{7,40}\b")


@dataclass
class IdentifierCorpus:
    """Adopter-configured identifiers the deterministic classifier treats
    as internal/identifying and redacts before a fragment may enter the
    shared tier. This is CONFIGURATION, not code -- for a self-hosting
    OSS adopter (v4-plan.md Stage 4's OSS reframing), the corpus is
    whatever their own directory names/aliases, internal ticket formats,
    and hostnames are, supplied at deployment time, not hardcoded here.

    `patterns`: adopter-specific regex patterns (case-insensitive),
        applied first -- directory names/aliases, ticket-ID formats
        (e.g. r"PROJ-\\d+"), internal hostnames, anything corpus-specific.
    `redact_emails`/`redact_commit_shas`: generic, shape-based patterns
        that apply regardless of adopter (an email is identifying no
        matter whose corpus it's in). Default OFF, deliberately, even
        though they're safe to enable: a fragment can carry a name,
        internal hostname, or other adopter-specific identifier that
        neither generic pattern catches, so a bare `IdentifierCorpus()`
        clearing a fragment on generic redaction alone would be a false
        sense of protection, not real de-identification -- exactly the
        "silently passing raw content through" failure this classifier
        exists to prevent. An adopter must affirmatively configure
        something (patterns, or explicitly opt into the generic
        redactors) before anything can be promoted at all.
    """

    patterns: List[str] = field(default_factory=list)
    redact_emails: bool = False
    redact_commit_shas: bool = False

    def is_empty(self) -> bool:
        """No adopter patterns AND both generic redactors disabled means
        this classifier has nothing to check against -- the default,
        zero-config state. Fail closed: reject rather than silently pass
        raw content."""
        return not self.patterns and not self.redact_emails and not self.redact_commit_shas


@dataclass
class ClassificationResult:
    passed: bool
    redacted_text: str
    # Categories matched, NEVER the raw matched values -- the audit log
    # this feeds must not itself become a store of the PII it redacted.
    categories: List[str]


@dataclass
class VerificationResult:
    """A caller-supplied verification-run outcome for correctness-gated
    promotion of code knowledge (v4-plan.md Stage 4 task 3). Aeon does
    NOT talk to any VCS/CI system itself -- integrating with GitHub
    Checks (or any other provider) would be a genuinely separate
    integration surface (auth, rate limits, a provider abstraction for
    non-GitHub adopters) from what this module does. Instead, the
    caller's own CI/test runner has already produced a result by the
    time it calls promote_fragment()/execute_approved_promotion(), and
    hands back the outcome here -- the same trust boundary already used
    for `governance_db`/`keystore` (Aeon consumes and records what it's
    given; it does not independently re-verify it).

    Supplied at PROMOTION-EXECUTION time, not request-creation time
    (create_promotion_approval_request()'s `target`) -- unlike
    dest_scope/subject_id, a verification run's outcome is often not
    known yet when a promotion is first requested and queued for
    four-eyes approval, so locking it in early would force either
    blocking request creation on CI completing first, or re-approving
    every time a flaky check reruns. Not itself covered by the four-eyes
    replay-safety guarantee -- see execute_approved_promotion()'s own
    doc comment.

    `status`: must be exactly the string "passed" to clear a gated
        promotion -- anything else (a typo, "pending", "failed") fails
        closed, same discipline as IdentifierCorpus.is_empty().
    `commit_sha`: the commit the verification ran against. Recorded for
        audit/traceability only -- NOT validated against the fragment's
        own content (this module has no VCS access to do so).
    `verified_by`: identifies the verification run/system for audit
        purposes (e.g. a CI job id or URL) -- distinct from `actor`
        (who/what triggered the promotion itself).

    `commit_sha` here is recorded VERBATIM into the audit log (never run
    through `IdentifierCorpus`/`_COMMIT_SHA_RE`) -- deliberately, not an
    oversight: `redact_commit_shas` exists to strip commit hashes that
    happen to appear INSIDE a fragment's own TEXT before that text enters
    the shared store, where an unrelated reader might see it. This value
    is different in kind: it's metadata the promotion CALLER supplied
    about their own CI run, going into the audit log (already the
    authoritative who/what/when record of this action, readable only by
    admins), not into shared-store content a broader audience will see.
    Redacting it here would make the audit trail less useful (an operator
    investigating a bad promotion needs the real commit) for no
    corresponding privacy benefit.
    """

    status: str
    commit_sha: Optional[str] = None
    verified_by: Optional[str] = None


def classify_and_redact(text: str, corpus: IdentifierCorpus) -> ClassificationResult:
    """The fail-closed deterministic classifier. Redacts every matching
    span it finds and reports which categories fired -- never the raw
    matched text."""
    if corpus.is_empty():
        return ClassificationResult(
            passed=False, redacted_text="", categories=["EMPTY_CORPUS_FAIL_CLOSED"]
        )

    redacted = text
    categories: List[str] = []

    for pattern in corpus.patterns:
        compiled = re.compile(pattern, re.IGNORECASE)
        if compiled.search(redacted):
            redacted = compiled.sub("[REDACTED]", redacted)
            categories.append("corpus_pattern")

    if corpus.redact_emails and _EMAIL_RE.search(redacted):
        redacted = _EMAIL_RE.sub("[REDACTED_EMAIL]", redacted)
        categories.append("email")

    if corpus.redact_commit_shas and _COMMIT_SHA_RE.search(redacted):
        redacted = _COMMIT_SHA_RE.sub("[REDACTED_SHA]", redacted)
        categories.append("commit_sha")

    return ClassificationResult(passed=True, redacted_text=redacted, categories=categories)


def promote_fragment(
    source_atlas: AeonClient,
    source_node_id: int,
    dest_atlas: AeonClient,
    dest_scope: int,
    corpus: IdentifierCorpus,
    audit_log: AuditLog,
    actor: str,
    subject_id: str,
    trace: Optional[TraceGraph] = None,
    reembed_fn: Optional[Callable[[str], List[float]]] = None,
    governance_db: Optional["GovernanceDB"] = None,
    keystore: Optional["Keystore"] = None,
    verification: Optional[VerificationResult] = None,
    require_verification: bool = False,
) -> Optional[int]:
    """Mint-and-recontextualize: promotes a private-store fragment into
    the shared store as a NEW, de-identified node.

    Args:
        source_atlas: The PRIVATE store the fragment currently lives in.
        source_node_id: Raw (not store-encoded) node id within
            source_atlas -- a mmap or delta-arena id, both supported
            (get_node_metadata()/get_node_centroid(), Stage 4 task 2).
        dest_atlas: The SHARED store to insert the promoted copy into.
        dest_scope: scope_bitmap to assign the new node (Stage 1/2) --
            which team(s) within the shared tier can see it.
        corpus: The fail-closed classifier's configuration.
        audit_log: Where this promotion (or rejection) gets recorded.
            The returned node's governance_record_id is this record's
            seq number.
        actor: Identity of whoever/whatever triggered this promotion --
            recorded in the audit entry and, if `trace` is given, used
            as that Trace event's session_id (a governance action has no
            single end-user session of its own; the actor's identity is
            the natural key, mirroring process_turn()'s use of
            session_id == user_id).
        subject_id: Identity of whichever private-store owner
            `source_node_id`'s content derives from -- distinct from
            `actor` (who/what triggered the promotion; may be an admin
            or an automated process, not the data subject). Required,
            not derived: `source_atlas`/`source_node_id` carry no owner
            identity of their own (private-store isolation is at the
            session/SLB layer, not per-node ownership tagging -- see
            erasure.py's deferred-private-store-erasure note), so the
            caller wiring up the promotion (create_promotion_approval_
            request(), below) is responsible for supplying the correct
            subject alongside the node it's promoting. Recorded on every
            audit path (rejection/anomaly/success) and, when
            `governance_db` is supplied, written to `governance_records.
            subject_id` -- v4-plan.md Stage 4 task 6's Phase A, the
            (subject_id, dest_scope) pair a future crypto-erase DEK
            lookup resolves through `governance_record_id`. Rejected up
            front (`ValueError`) if empty/blank, same fail-closed
            treatment as `dest_scope == 0` -- an empty value can never
            resolve to a real DEK lookup. One `subject_id` per promoted
            node is task 6's decision-record invariant (a node must
            never derive from more than one subject, since destroying
            one subject's key would silently destroy another's too);
            this function satisfies it BY CONSTRUCTION today (it mints
            exactly one node from exactly one caller-supplied subject_id
            per call, and nothing yet merges shared-store nodes -- see
            the decision record's multi-subject-node check), not by an
            independent runtime check spanning multiple calls. Stage 5
            pointing the Dreamer at the shared tier is what would need
            to add real enforcement here.
        trace: Optional -- if given, also records a PROMOTED_FROM Trace
            edge for the console's future knowledge-browser view. The
            audit log is the authoritative record regardless of whether
            this is supplied.
        reembed_fn: Optional destination-conditioned re-embedding hook
            (text -> vector). Defaults to reusing the source vector
            unmodified -- see module docstring.
        governance_db: Optional (control_plane/db.py's GovernanceDB) --
            when supplied, this promotion's audit record is ALSO written
            to Postgres (control_plane/schema.py's governance_records
            table), and the resulting Postgres-assigned row id is used
            as governance_record_id instead of the JSONL log's own seq
            number. Fixes the seq's one real weakness (only unique
            within one specific log file -- ambiguous across log
            rotation/relocation) without requiring Postgres to promote
            anything at all when it's absent, same optionality pattern
            as shared_atlas_client (client.py). Called in-process,
            synchronously, ordered with the JSONL append and BEFORE the
            scope/governance mutation -- see control_plane/db.py's
            GovernanceDB docstring for why this isn't a network call
            through control_plane/app.py's own HTTP API.
        keystore: Optional (crypto.py's Keystore) -- v4-plan.md Stage 4
            task 6 Phase B. When supplied, the text written to the
            SHARED store's metadata field (never the centroid vector --
            see the task 6 decision record) is encrypted under a DEK
            resolved from (subject_id, dest_scope), generating one on
            first use. None (the default) means this deployment hasn't
            opted into crypto-erase -- proceeds with plaintext exactly as
            before, same optionality convention as governance_db=None.
            Raises ValueError if the redacted text's encoded length
            exceeds this store's encrypted-metadata budget
            (crypto.max_plaintext_bytes()) rather than letting
            Atlas.insert() silently truncate ciphertext.
        verification: Optional caller-supplied verification-run outcome
            (task 3, VerificationResult above). Recorded on the audit
            trail (success or rejection) whenever supplied, regardless
            of `require_verification` -- informational unless gating is
            turned on.
        require_verification: When True, promotion is REJECTED (same
            fail-closed shape as a classifier rejection -- recorded as
            "promotion_rejected", returns None, does not raise) unless
            `verification` is supplied AND `verification.status ==
            "passed"`. Checked BEFORE the identifier-corpus classifier
            runs (cheapest check first). Default False preserves every
            existing caller's behavior unchanged -- same opt-in
            convention as `governance_db`/`keystore`.

    Returns:
        The new shared-store node's RAW id (not store-encoded -- same
        convention as AeonClient.atlas's other raw-id methods; callers
        crossing into Trace/server.py encode it themselves, same as
        every other Stage 4 write site), or None if the classifier
        rejected the fragment (still recorded in the audit log either
        way).

    Raises:
        ValueError: dest_scope == 0 -- a scope-0 node is unreachable by
            any scoped query, but list_nodes_by_scope(ALL_SCOPES_VISIBLE)
            still lists it as present, which reads as data loss to
            whoever debugs it later. Rejected up front, fail-closed, same
            reasoning as the empty-corpus case.
        RuntimeError: the destination insert succeeded but scope/
            governance could not be applied (see the two anomaly cases
            below) -- the node now exists in the shared store but is not
            fully governed. Always recorded in the audit log as
            "promotion_unscoped_anomaly" BEFORE this raises, so the
            orphaned node id is discoverable, not silently lost.
    """
    if dest_scope == 0:
        raise ValueError(
            "promote_fragment: dest_scope must be non-zero -- a scope-0 "
            "node is unreachable by any scoped query"
        )
    if not subject_id or not subject_id.strip():
        raise ValueError(
            "promote_fragment: subject_id must be a non-empty identifier -- "
            "an empty/blank value can never resolve to a real (subject_id, "
            "dest_scope) DEK lookup (task 6 Phase A)"
        )

    def _mirror_governance_record(
        action: str, log_seq: int, dest_node_id: Optional[int]
    ) -> None:
        """Best-effort Postgres mirror for the rejection/anomaly paths
        below (advisor review: governance_db.record() was previously
        only ever called on the success path, so Postgres held completed
        promotions and nothing else -- a console querying it for "what
        was attempted against scope X" got a systematically incomplete
        answer, and the one row an operator most needs to find (an
        orphaned unscoped node) was invisible in the queryable store).
        Unlike the success-path mirror (folded into the same try/except
        as the mutation, because a failure THERE is itself the anomaly
        being guarded against), a failure HERE must not mask the
        JSONL write that already succeeded or block the
        ValueError/RuntimeError this is called alongside from
        propagating -- the JSONL entry is the durable, authoritative
        record regardless of whether this mirror succeeds.
        """
        if governance_db is None:
            return
        try:
            governance_db.record(
                log_instance_id=audit_log.instance_id,
                log_instance_path=str(audit_log.path),
                log_seq=log_seq,
                action=action,
                actor=actor,
                subject_id=subject_id,
                source_node_id=source_node_id,
                dest_node_id=dest_node_id,
                dest_scope=dest_scope,
            )
        except Exception:
            logger.warning(
                "promote_fragment: best-effort Postgres mirror of %s "
                "(log_seq=%d) failed -- the JSONL record is still "
                "authoritative and unaffected",
                action, log_seq, exc_info=True,
            )

    def _verification_payload() -> dict:
        if verification is None:
            return {}
        return {
            "verification_status": verification.status,
            "verification_commit_sha": verification.commit_sha,
            "verification_verified_by": verification.verified_by,
        }

    if require_verification and (verification is None or verification.status != "passed"):
        seq = audit_log.append(
            action=_ACTION_REJECTED,
            actor=actor,
            payload={
                "source_node_id": source_node_id,
                "dest_scope": dest_scope,
                "subject_id": subject_id,
                "reason_categories": [
                    "VERIFICATION_REQUIRED_BUT_MISSING"
                    if verification is None
                    else "VERIFICATION_FAILED"
                ],
                **_verification_payload(),
            },
        )
        _mirror_governance_record(_ACTION_REJECTED, seq, dest_node_id=None)
        return None

    text = source_atlas.atlas.get_node_metadata(source_node_id)
    result = classify_and_redact(text, corpus)

    if not result.passed:
        seq = audit_log.append(
            action=_ACTION_REJECTED,
            actor=actor,
            payload={
                "source_node_id": source_node_id,
                "dest_scope": dest_scope,
                "subject_id": subject_id,
                "reason_categories": result.categories,
            },
        )
        _mirror_governance_record(_ACTION_REJECTED, seq, dest_node_id=None)
        return None

    vector = (
        reembed_fn(result.redacted_text)
        if reembed_fn is not None
        else source_atlas.atlas.get_node_centroid(source_node_id)
    )

    # v4-plan.md Stage 4 task 6 Phase B: encrypt the text actually WRITTEN
    # to the shared store's metadata field, not the vector (centroid stays
    # plaintext -- see the task 6 decision record) and not the text fed to
    # reembed_fn above (embeddings must be computed over plaintext
    # meaning). keystore=None (the default) means this deployment hasn't
    # opted into crypto-erase -- proceeds with plaintext exactly as
    # before, same optionality convention as governance_db=None.
    stored_text = result.redacted_text
    if keystore is not None:
        from .crypto import encrypt_metadata, max_plaintext_bytes

        budget = max_plaintext_bytes(dest_atlas.atlas.metadata_size)
        if len(stored_text.encode("utf-8")) > budget:
            raise ValueError(
                f"promote_fragment: redacted text ({len(stored_text.encode('utf-8'))} "
                f"bytes) exceeds this shared store's encrypted-metadata budget "
                f"({budget} bytes at metadata_size={dest_atlas.atlas.metadata_size}) -- "
                "raising rather than letting Atlas.insert() silently truncate "
                "ciphertext (task 6 decision record)"
            )
        dek = keystore.get_or_create_dek(subject_id, dest_scope)
        stored_text = encrypt_metadata(dek, stored_text)

    new_id = dest_atlas.atlas.insert(0, vector, stored_text)

    def _record_anomaly(reason: str) -> None:
        seq = audit_log.append(
            action=_ACTION_ANOMALY,
            actor=actor,
            payload={
                "source_node_id": source_node_id,
                "dest_node_id": new_id,
                "dest_scope": dest_scope,
                "subject_id": subject_id,
                "reason": reason,
            },
        )
        _mirror_governance_record(_ACTION_ANOMALY, seq, dest_node_id=new_id)

    if new_id & NODE_ID_DELTA_MASK:
        # dest_atlas.atlas.insert() diverts to the delta buffer if a
        # compaction was in progress at insert time (Atlas::insert(),
        # atlas.cpp) -- and set_node_scope()/set_node_governance_id()
        # BOTH reject delta-arena ids outright ("delta nodes get a fresh
        # id when compact_mmap() promotes them, so a scope set against
        # the old id would be silently lost"). Proceeding to call them
        # would throw mid-sequence and leave this node sitting in the
        # shared store UNSCOPED -- readable by any query_stores(
        # mode="shared") caller using the default ALL_SCOPES_VISIBLE
        # mask, exactly the leak this increment exists to prevent. Fail
        # loudly instead of letting that throw happen implicitly.
        _record_anomaly(
            "insert() diverted to delta buffer during concurrent "
            "compaction; scope/governance cannot be applied to a "
            "delta-arena id"
        )
        raise RuntimeError(
            f"promote_fragment: destination store diverted node {new_id} "
            "to its delta buffer (concurrent compaction) -- it now exists "
            "UNSCOPED in the shared store; recorded as "
            "promotion_unscoped_anomaly. Retry once compaction completes."
        )

    seq = audit_log.append(
        action=_ACTION_PROMOTION,
        actor=actor,
        payload={
            "source_node_id": source_node_id,
            "dest_node_id": new_id,
            "dest_scope": dest_scope,
            "subject_id": subject_id,
            "redaction_categories": result.categories,
            **_verification_payload(),
        },
    )

    try:
        if governance_db is not None:
            # Mirrors the JSONL record into Postgres and gets back a
            # globally-stable row id -- see governance_db's param doc.
            # Folded into this same try/except: a network failure here
            # (e.g. a timeout after the row actually landed) produces
            # the identical "node exists, not fully governed" shape as
            # the set_node_scope()/set_node_governance_id() failures
            # below, so it gets the same anomaly-recording safety net.
            governance_id = governance_db.record(
                log_instance_id=audit_log.instance_id,
                log_instance_path=str(audit_log.path),
                log_seq=seq,
                action=_ACTION_PROMOTION,
                actor=actor,
                subject_id=subject_id,
                source_node_id=source_node_id,
                dest_node_id=new_id,
                dest_scope=dest_scope,
            )
        else:
            governance_id = seq

        dest_atlas.atlas.set_node_scope(new_id, dest_scope)
        dest_atlas.atlas.set_node_governance_id(new_id, governance_id)
    except Exception:
        # Narrower race than the delta-diversion case above (e.g. a
        # compaction starting on dest_atlas between insert() returning
        # and this call, or a Postgres write failure/timeout) -- same
        # failure shape (node exists, ungoverned), same fix: record
        # before raising so it's discoverable, not lost.
        _record_anomaly(
            "governance_db.record()/set_node_scope/set_node_governance_id "
            "failed after insert (e.g. compaction started concurrently, "
            "or a Postgres write failure) -- node exists but is not "
            "fully governed"
        )
        raise

    if trace is not None:
        trace.add_event(
            actor,
            "concept",
            f"[Promoted {new_id} from {source_node_id}]",
            atlas_id=encode_store_id(new_id, is_shared=True),
            edge_type=EdgeType.PROMOTED_FROM,
            supersedes_id=encode_store_id(source_node_id, is_shared=False),
        )

    return new_id


def create_promotion_approval_request(
    admin_db: "AdminDB",
    *,
    source_node_id: int,
    dest_scope: int,
    subject_id: str,
    reason: str,
    requested_by: str,
    expires_at: datetime,
    required_approvals: int = 2,
) -> int:
    """Creates a four-eyes approval request for a specific promotion
    (v4-plan.md Stage 4 task 7). `target` is JSON-encoded exact operation
    parameters, not free text -- execute_approved_promotion() reads them
    back from THIS record rather than trusting caller-supplied parameters
    at execution time, so an approval for "promote node 5 into scope 0x1"
    can never be replayed to execute "promote node 999 into scope
    ALL_SCOPES_VISIBLE" using the same N approvals.

    `subject_id` (task 6 Phase A -- see promote_fragment()'s doc comment)
    is locked into `target` for the identical replay-safety reason as
    `source_node_id`/`dest_scope`: whoever creates the request is the one
    who knows which private-store owner's content is being promoted, and
    that attribution must survive to execution unchanged, not be
    re-suppliable by whoever happens to call execute_approved_promotion().

    Raises:
        ValueError: `subject_id` is empty/blank. Checked HERE, not just in
            promote_fragment()'s own guard, because a blank value passing
            this check would still get locked into `target`, collect a
            real four-eyes approval, and only fail at execution -- a dead
            request that consumed two reviewers' attention for nothing,
            the same fail-late shape as bugs already fixed elsewhere on
            this path (advisor review).
    """
    if not subject_id or not subject_id.strip():
        raise ValueError(
            "create_promotion_approval_request: subject_id must be a "
            "non-empty identifier -- rejected at request-creation time so "
            "a blank value can't collect real approvals before failing at "
            "execution"
        )
    target = json.dumps(
        {
            "source_node_id": source_node_id,
            "dest_scope": dest_scope,
            "subject_id": subject_id,
        }
    )
    return admin_db.create_approval_request(
        action=_APPROVAL_ACTION_PROMOTION,
        target=target,
        reason=reason,
        requested_by=requested_by,
        expires_at=expires_at,
        required_approvals=required_approvals,
    )


def execute_approved_promotion(
    admin_db: "AdminDB",
    request_id: int,
    *,
    actor: str,
    source_atlas: AeonClient,
    dest_atlas: AeonClient,
    corpus: IdentifierCorpus,
    audit_log: AuditLog,
    trace: Optional[TraceGraph] = None,
    reembed_fn: Optional[Callable[[str], List[float]]] = None,
    governance_db: Optional["GovernanceDB"] = None,
    keystore: Optional["Keystore"] = None,
    verification: Optional[VerificationResult] = None,
    require_verification: bool = False,
) -> Optional[int]:
    """Executes a promotion that has already been approved via
    create_promotion_approval_request() + N calls to
    admin_db.grant_approval().

    Three checks, in this order, BEFORE promote_fragment() ever runs:
    1. The request exists and is actually a promotion request (not some
       other admin_roles action reusing the same approval_requests table).
    2. It hasn't already been executed -- replay guard. Without this, the
       SAME already-approved request could be executed twice, minting two
       promoted nodes from one four-eyes approval. Checked here, in the
       application layer, not via a database transaction spanning this
       call and promote_fragment()'s Atlas mutation -- that mutation
       happens in a completely different storage system (mmap, not SQL),
       so no SQL transaction could make this atomic with it anyway. A
       narrow window remains (promote_fragment() succeeds, then this
       process crashes before mark_executed() runs) -- same accepted-risk
       shape as promote_fragment()'s own anomaly handling: not perfectly
       atomic, but discoverable (a retry's resulting SECOND governance
       record is itself an audit trail entry pointing at the duplicate).
    3. admin_db.is_approved(request_id) -- the actual four-eyes check
       (distinct-approver count, non-expired, non-revoked).

    The promotion's exact parameters (source_node_id, dest_scope) come
    from the approval request's own `target`, not from this call's
    arguments -- see create_promotion_approval_request()'s doc comment
    for why that's load-bearing, not incidental.

    Raises:
        ValueError: no such request, or it isn't a promotion request.
        RuntimeError: already executed.
        PermissionError: not currently approved (insufficient distinct
            approvers, expired, or revoked).

    A REJECTION (promote_fragment() returning None -- either the
    fail-closed classifier didn't clear the content, or, task 3,
    `require_verification` was set and `verification` was missing/not
    "passed") does NOT mark the request executed (advisor review: the
    four-eyes approval is consent to promote SPECIFIC CONTENT into a
    scope, not consent to burn one attempt at it -- treating a rejection
    as "executed" would permanently consume the approval on what is, from
    the requester's point of view, a corpus-configuration or CI-not-done-
    yet problem, not a decision anyone actually made. The request stays
    executable; an operator who fixes the corpus config, or retries once
    CI passes, or a caller who deliberately wants to stop retrying, can
    call admin_db.revoke_request() instead). Only a successful mint (an
    ANOMALY is still a mint -- see promote_fragment()'s RuntimeError path)
    consumes the approval. `verification` is deliberately NOT locked into
    the approval request's own `target` (see VerificationResult's doc
    comment) -- it's supplied fresh at each execution attempt, same as
    `reembed_fn`/`keystore`.
    """
    req = admin_db.get_request(request_id)
    if req is None:
        raise ValueError(f"execute_approved_promotion: no such request {request_id}")
    if req["action"] != _APPROVAL_ACTION_PROMOTION:
        raise ValueError(
            f"execute_approved_promotion: request {request_id} is action "
            f"{req['action']!r}, not {_APPROVAL_ACTION_PROMOTION!r}"
        )
    if req["executed_at"] is not None:
        raise RuntimeError(
            f"execute_approved_promotion: request {request_id} was already "
            f"executed at {req['executed_at']}"
        )
    if not admin_db.is_approved(request_id):
        raise PermissionError(
            f"execute_approved_promotion: request {request_id} is not "
            "currently approved (insufficient distinct approvers, "
            "expired, or revoked)"
        )

    params = json.loads(req["target"])
    result = promote_fragment(
        source_atlas,
        params["source_node_id"],
        dest_atlas,
        params["dest_scope"],
        corpus,
        audit_log,
        actor,
        params["subject_id"],
        trace=trace,
        reembed_fn=reembed_fn,
        governance_db=governance_db,
        keystore=keystore,
        verification=verification,
        require_verification=require_verification,
    )
    if result is not None:
        admin_db.mark_executed(request_id)
    return result
