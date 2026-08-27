import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
import anyio
from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from typing import Generator, Any

from .dependencies import (
    DEFAULT_RECORDS_DIR,
    build_consolidation_worker,
    get_session_manager,
    get_current_user_id,
    get_atlas_client,
    get_shared_atlas_client,
    get_admin_db,
    get_governance_db,
    get_audit_log,
    get_audit_log_export_key,
    get_erasure_db,
    get_identifier_corpus,
    get_keystore,
    get_require_code_verification,
    AeonClient,
    SessionManager
)
from .client import ALL_SCOPES_VISIBLE, decode_store_id, encode_store_id
from .governance import AuditLogError
from .loop import CognitiveLoop
from .context import ContextManager
from .promotion import execute_approved_promotion
import logging
from .erasure import ErasureTransientFailure, create_erasure_case, execute_approved_erasure
from .supersession import supersede_node as _supersede_node_audited
from .supersession import revoke_node_supersession as _revoke_node_supersession_audited
from .supersession import supersede_by_reverted_commit
from .models import (
    ChatRequest,
    TraceResponse,
    TraceNode,
    TraceEdge,
    ActiveRoomResponse,
    NeighborInfo,
    VectorQueryRequest,
    SearchResult,
    PromotionExecuteRequest,
    PromotionExecuteResponse,
    AuditRecordResponse,
    AuditLogTailResponse,
    AuditVerifyResponse,
    KnowledgeNode,
    KnowledgeListResponse,
    KnowledgeActionRequest,
    SupersedeByCommitRequest,
    SupersedeByCommitResponse,
    ErasureCaseCreateRequest,
    ErasureCaseCreateResponse,
    ErasureReceiptEntry,
    ErasureCaseResponse,
)

app = FastAPI(title="Aeon Cognitive OS Server", version="0.1.0")

# CORS
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# THE FIRST BACKGROUND WORKER THIS SERVER ACTUALLY RUNS. Before this there was no startup
# hook at all -- only the shutdown handler below -- so `DreamingWorker`, the pattern this
# copies, has never been started in production either. Consolidation is what turns the
# semantic layer from a library into a live system: without it, records are only ever
# written by a benchmark harness.
_consolidation_worker = None


@app.on_event("startup")
async def startup_event():
    """Start background consolidation, if the deployment opted into the semantic layer."""
    global _consolidation_worker
    if not DEFAULT_RECORDS_DIR:
        return
    try:
        _consolidation_worker = build_consolidation_worker()
        if _consolidation_worker is not None:
            _consolidation_worker.start()
    except Exception:
        # A server that cannot consolidate must still serve chat. Consolidation is
        # background enrichment, not a request-path dependency.
        logging.getLogger(__name__).exception(
            "consolidation worker failed to start; continuing without it")


@app.on_event("shutdown")
async def shutdown_event():
    """Stop the worker, then flush sessions.

    Ordering matters: the worker writes through the same per-tenant stores that
    `SessionManager.shutdown()` syncs and drops, so stopping it first means no cycle is
    mid-write when the handles go away.
    """
    global _consolidation_worker
    if _consolidation_worker is not None:
        _consolidation_worker.stop()
        _consolidation_worker = None
    mgr = get_session_manager()
    await mgr.shutdown()

@app.get("/health")
async def health_check():
    return {"status": "ok", "component": "AeonServer"}

# --- Chat Endpoint (Streaming) ---

async def make_async_generator(generator: Generator[str, None, None]):
    """
    Wraps a blocking synchronous generator into an async generator
    by running next() calls in a separate thread.
    """
    iterator = iter(generator)
    while True:
        try:
            # Run the blocking next() in a threadpool to avoid freezing the event loop
            token = await anyio.to_thread.run_sync(next, iterator)
            if token:
                yield {"event": "token", "data": token}
        except StopIteration:
            break
        except Exception as e:
            yield {"event": "error", "data": str(e)}
            break
            
    yield {"event": "done", "data": "[DONE]"}

@app.post("/chat")
async def chat_endpoint(
    request: ChatRequest,
    user_id: str = Depends(get_current_user_id),
    mgr: SessionManager = Depends(get_session_manager)
):
    """
    Streams the LLM response for a specific user session.
    Input: {"text": "Hello"}
    Output: SSE Stream of tokens.
    """
    loop = await mgr.get_loop(user_id)

    # Create the synchronous generator
    sync_gen = loop.chat(request.text, session_id=user_id)
    
    # Wrap in async generator for non-blocking stream
    return EventSourceResponse(make_async_generator(sync_gen))


# --- Observability Endpoints (The Glass Box) ---

_TRACE_ROLE_LABEL = {0: "UserNode", 1: "SystemNode", 2: "ConceptNode", 3: "SummaryNode"}

@app.get("/state/trace", response_model=TraceResponse)
async def get_trace_state(
    user_id: str = Depends(get_current_user_id),
    mgr: SessionManager = Depends(get_session_manager)
):
    """
    Returns the requesting user's own episodic trace history as a graph
    (nodes = events, edges = the chronological prev_id chain within their
    session). Previously called TraceGraph.to_viz_json(), a method that
    doesn't exist on the current C++-backed TraceGraph -- and which took
    no user_id, so it would have returned every user's combined history
    with no isolation at all had it existed. Fixed alongside the /chat
    bug (v4-plan.md): same root cause (context.py/loop.py/server.py never
    updated after TraceGraph was rewritten onto the C++ mmap backend).

    Note for consumers (advisor review, v4-plan.md Stage 4 step 3):
    `details["atlas_id"]` is store-discriminated (encode_store_id()) for
    any concept event, same encoding as /state/atlas/active's `room_id` --
    it is NOT added by this endpoint, it's already baked into the value
    Trace stored at write time. A future console reading this field must
    decode_store_id() it before treating it as a raw Atlas node id. Sent
    as a STRING (not a bare JSON number, advisor review, v4-plan.md Stage
    4): once SHARED_STORE_BIT/NODE_ID_DELTA_MASK are set, the value can
    exceed JS's Number.MAX_SAFE_INTEGER, which a browser JSON.parse
    silently rounds -- see models.py's NeighborInfo.id comment for the
    same fix applied there.
    """
    ctx = await mgr.get_context(user_id)

    history = ctx.trace.get_history(user_id, limit=200)

    nodes = [
        TraceNode(
            id=str(ev["id"]),
            label=ev.get("text_preview") or ev.get("text", ""),
            type=_TRACE_ROLE_LABEL.get(ev["role"], "UnknownNode"),
            timestamp=float(ev["timestamp"]),
            details={
                "atlas_id": str(ev["atlas_id"]),
                "session_id": ev["session_id"],
                "full_text": ev.get("text", ""),
            },
        )
        for ev in history
    ]
    edges = [
        TraceEdge(source=str(ev["prev_id"]), target=str(ev["id"]), type="sequence")
        for ev in history
        if ev["prev_id"] != 0
    ]

    return TraceResponse(nodes=nodes, edges=edges)

@app.get("/state/atlas/active", response_model=ActiveRoomResponse)
async def get_active_room(
    user_id: str = Depends(get_current_user_id),
    mgr: SessionManager = Depends(get_session_manager),
    atlas: AeonClient = Depends(get_atlas_client),
    shared_atlas: AeonClient | None = Depends(get_shared_atlas_client),
):
    """
    Returns the "Active Room" for the user.
    """
    ctx = await mgr.get_context(user_id)

    # Logic: Find the most recent "concept" event in this user's own Trace
    # history (role=2, see trace.py's ROLE_CONCEPT). Was ctx.trace.graph
    # .nodes(...) -- a NetworkX attribute the current TraceGraph doesn't
    # have -- silently swallowed by a bare except, so this always fell
    # back to the root default. Fixed alongside the /chat and
    # /state/trace bugs (v4-plan.md), same root cause.
    active_atlas_id = 0  # Root default -- decode_store_id(0) == (0, False),
                         # i.e. "private store root", so this default needs
                         # no special-casing under store-discriminated ids.

    try:
        history = ctx.trace.get_history(user_id, limit=50)  # newest-first
        concept_events = [ev for ev in history if ev.get("role") == 2 and ev.get("atlas_id")]
        if concept_events:
            active_atlas_id = int(concept_events[0]["atlas_id"])  # most recent
    except Exception:
        pass

    # atlas_id is store-discriminated (encode_store_id(), v4-plan.md Stage
    # 4) -- this is exactly the Atlas->Trace->Atlas graph-expansion
    # boundary Stage 2 task 2's scope check was built for, now with a
    # second dimension: which PHYSICAL store to even query, not just which
    # scope within one. Decode before routing the get_children() call.
    raw_room_id, room_is_shared = decode_store_id(active_atlas_id)
    room_atlas = shared_atlas if room_is_shared else atlas
    if room_atlas is None:
        # A shared-store atlas_id was recorded but this deployment has no
        # shared store configured (or it was since unconfigured) -- fail
        # safe to an empty room rather than raising or silently querying
        # the wrong store.
        children_raw = []
    else:
        children_raw = room_atlas.get_children(raw_room_id)

    neighbors = []
    for row in children_raw:
        # Children live in the SAME store as their parent -- re-encode so a
        # caller navigating into one of these ids later doesn't hit the
        # same store ambiguity one hop downstream.
        neighbors.append(NeighborInfo(
            id=str(encode_store_id(int(row['id']), is_shared=room_is_shared)),
            similarity=0.0
        ))

    path = []

    return ActiveRoomResponse(
        room_id=str(active_atlas_id),
        name=f"Room {active_atlas_id}",
        path=path,
        neighbors=neighbors
    )

# Debug endpoint, OFF by default (v4-plan.md Stage 0): this queries the
# entire shared Atlas with no per-user scoping, so it must never be
# reachable in a multi-tenant deployment unless explicitly opted into for
# local debugging. Authenticated on top of the flag, not instead of it --
# an authenticated-but-unscoped global query is still a cross-tenant leak.
AEON_ENABLE_DEBUG_ENDPOINTS = os.environ.get(
    "AEON_ENABLE_DEBUG_ENDPOINTS", "false"
).lower() == "true"

@app.post("/state/atlas/query", response_model=list[SearchResult])
async def debug_atlas_query(
    request: VectorQueryRequest,
    user_id: str = Depends(get_current_user_id),
    atlas: AeonClient = Depends(get_atlas_client)
):
    """
    Debug: Raw vector search against the PRIVATE Atlas store only (Global/
    System Level within that store, NOT scoped to the requesting user).
    Requires AEON_ENABLE_DEBUG_ENDPOINTS=true.

    Deliberately does NOT take a shared_atlas_client dependency (v4-plan.md
    Stage 4): this is a debug-only, off-by-default endpoint with no
    per-user scoping already; adding a second, physically separate store
    to an already-unscoped global query only widens the blast radius of
    something that already needs the explicit opt-in flag to reach at all.
    """
    if not AEON_ENABLE_DEBUG_ENDPOINTS:
        raise HTTPException(status_code=404, detail="Not found")

    import numpy as np

    vec = np.array(request.vector, dtype=np.float32)
    results = atlas.query(vec)

    output = []
    for row in results:
        output.append(SearchResult(
            id=str(encode_store_id(int(row['id']), is_shared=False)),
            similarity=row['similarity'],
            preview=row['preview'].tolist()
        ))
    return output

# --- Admin: promotion execution (v4-plan.md Stage 4 task 7) ---
#
# Deliberately the ONLY admin-authenticated route in this file (task 7's
# other primitives -- creating approval requests, granting approvals,
# listing/browsing governance state -- have no HTTP surface yet; those
# are the console, task 5, not this increment). This endpoint exists
# because it's the first (and so far only) operation that actually NEEDS
# one: promote_fragment() mutates the shared Atlas store, which only this
# retrieval service has a client for (control_plane/app.py has no Atlas
# DI at all, by design -- it's a pure governance-state service).

@app.post("/admin/promotions/{request_id}/execute", response_model=PromotionExecuteResponse)
async def execute_promotion(
    request_id: int,
    body: PromotionExecuteRequest | None = None,
    user_id: str = Depends(get_current_user_id),
    atlas: AeonClient = Depends(get_atlas_client),
    shared_atlas: AeonClient | None = Depends(get_shared_atlas_client),
    admin_db=Depends(get_admin_db),
    governance_db=Depends(get_governance_db),
    audit_log=Depends(get_audit_log),
    corpus=Depends(get_identifier_corpus),
    keystore=Depends(get_keystore),
    require_verification: bool = Depends(get_require_code_verification),
):
    """
    Executes an already-approved promotion request (v4-plan.md Stage 4
    tasks 2/7). The request must already exist (created out-of-band via
    control_plane.promotion.create_promotion_approval_request() -- no
    HTTP endpoint creates requests yet, see module note above) and carry
    enough distinct approvals (control_plane.admin.AdminDB.grant_approval()).

    Defense in depth beyond the four-eyes approval itself: the CALLER
    triggering execution must independently hold the "admin" role over
    the request's own dest_scope (task 7: "admin reads go through the
    same enforcement path as any other read, never a wildcard bypass") --
    N people approving a request doesn't by itself authorize an arbitrary
    caller to be the one who pulls the trigger.

    `body.destination_embedding` (optional): closes task 2's deferred
    destination-conditioned re-embedding via a caller-supplied vector,
    not a new Aeon-owned embedding pipeline (see PromotionExecuteRequest's
    doc comment). Threaded through as `promote_fragment()`'s existing
    `reembed_fn` seam -- a closure ignoring the redacted text and
    returning this vector unmodified. Omitted/None reuses the source
    vector, exactly today's default behavior.

    `body.verification` (optional): task 3's correctness-gated promotion
    for code knowledge. When this deployment has
    AEON_REQUIRE_CODE_VERIFICATION=true, a missing `verification` or one
    whose `status != "passed"` makes promote_fragment() reject the
    fragment (a 200 response with promoted_node_id=None, same shape as a
    classifier rejection -- NOT a 4xx, since the request itself was valid,
    the OUTCOME was a rejection). When the deployment hasn't opted in,
    `verification` is recorded on the audit trail if supplied but never
    gates.
    """
    if admin_db is None:
        raise HTTPException(status_code=404, detail="control plane not configured")
    if shared_atlas is None:
        raise HTTPException(status_code=404, detail="no shared Atlas store configured")

    req = admin_db.get_request(request_id)
    if req is None:
        raise HTTPException(status_code=404, detail="approval request not found")

    try:
        params = json.loads(req["target"])
        dest_scope = int(params["dest_scope"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        raise HTTPException(status_code=500, detail="malformed approval request target")

    if not admin_db.has_role(principal=user_id, scope_mask=dest_scope):
        raise HTTPException(status_code=403, detail="not authorized for this scope")

    if corpus.is_empty():
        # Distinct from an actual classifier rejection (advisor review):
        # an empty IdentifierCorpus rejects EVERY fragment unconditionally
        # (promotion.py's fail-closed default), so returning 200 with
        # promoted_node_id=None here would be indistinguishable from "this
        # specific fragment's content didn't clear the classifier" -- the
        # caller has no signal that the deployment simply isn't configured
        # to promote anything at all. 503, not 200: this is a standing
        # deployment-configuration state, not a per-request outcome, and
        # (as of the mark_executed() fix above) doesn't consume the
        # request's approval either way.
        raise HTTPException(
            status_code=503,
            detail="identifier corpus is not configured -- this deployment "
            "cannot promote any fragment until AEON_IDENTIFIER_CORPUS_PATTERNS "
            "and/or AEON_REDACT_EMAILS/AEON_REDACT_COMMIT_SHAS are set",
        )

    reembed_fn = None
    if body is not None and body.destination_embedding is not None:
        destination_embedding = body.destination_embedding
        reembed_fn = lambda _text, _vec=destination_embedding: _vec

    verification = None
    if body is not None and body.verification is not None:
        from .promotion import VerificationResult

        verification = VerificationResult(
            status=body.verification.status,
            commit_sha=body.verification.commit_sha,
            verified_by=body.verification.verified_by,
        )

    try:
        new_id = execute_approved_promotion(
            admin_db,
            request_id,
            actor=user_id,
            source_atlas=atlas,
            dest_atlas=shared_atlas,
            corpus=corpus,
            audit_log=audit_log,
            reembed_fn=reembed_fn,
            governance_db=governance_db,
            keystore=keystore,
            verification=verification,
            require_verification=require_verification,
        )
    except (ValueError, RuntimeError, PermissionError) as e:
        raise HTTPException(status_code=409, detail=str(e))

    return PromotionExecuteResponse(
        promoted_node_id=(
            str(encode_store_id(new_id, is_shared=True)) if new_id is not None else None
        )
    )


# --- Admin: minimum console (v4-plan.md Stage 4 task 5) ---
#
# API-only, deliberately: this repo has zero frontend infrastructure (no
# StaticFiles mount, no templates) -- the source design docs' "console"
# language assumed a UI, but building one is a separate product surface
# from this increment. Every route below follows the promotion endpoint's
# established shape: caller identity from get_current_user_id() (never a
# request-body field), admin authorization checked against state read
# FROM THE SERVER (a node's own current scope, or an approval request's
# locked-in target), never from a caller-supplied scope/mask parameter --
# "admin reads go through the same enforcement path as any other read,
# never a wildcard bypass" (v4-plan.md Stage 4 task 7).
#
# Scoped to the SHARED atlas store only, same reasoning as the erasure
# workflow's own scoping (erasure.py's module doc comment): the private
# store has no per-owner authorization model to check a console action
# against.


def _require_admin_db(admin_db):
    if admin_db is None:
        raise HTTPException(status_code=404, detail="control plane not configured")


def _require_shared_atlas(shared_atlas):
    if shared_atlas is None:
        raise HTTPException(status_code=404, detail="no shared Atlas store configured")


def _require_scope_containment(admin_db, user_id: str, node_scope: int) -> None:
    """Authorizes an action against (or a read of) a specific node/case
    whose scope_bitmap is `node_scope` -- CONTAINMENT, not overlap
    (advisor review, real bug: a node's scope_bitmap can carry MULTIPLE
    scope bits at once, e.g. 0x1000|0x2000 for a fragment visible to two
    teams. has_role()'s overlap check (`grant & requested != 0`) is
    correct for the PROMOTION endpoint, where dest_scope is a single
    target being written into -- it is NOT correct here: overlap would
    let a caller granted only 0x1000 tombstone/erase a node ALSO in
    0x2000, which they have no grant over. erasure's combined_scope
    (the OR across every target node) makes this worse -- one grant
    overlapping ANY target authorizes the WHOLE case, including nodes in
    scopes the caller never had.

    Containment: every bit of node_scope must be covered by the caller's
    own effective_scope_mask. `caller_mask == 0` is checked explicitly
    (not left to fall out of the bitwise math) because an UNSCOPED node
    (node_scope == 0, e.g. a promotion delta-diversion orphan) would
    otherwise trivially pass containment against a caller holding NO
    admin grants at all -- 0 & anything == 0. An unscoped node is
    reachable by ANY admin grant (deliberate: it's an anomaly, not a
    node that "belongs" to a scope the caller lacks), but only once the
    caller is confirmed to hold at least one.
    """
    if not _caller_scope_containment_ok(admin_db, user_id, node_scope):
        raise HTTPException(status_code=403, detail="not authorized for this scope")


def _caller_scope_containment_ok(admin_db, user_id: str, node_scope: int) -> bool:
    """Non-raising containment check -- the boolean core of
    _require_scope_containment(), factored out so a per-node batch
    operation (supersede_by_commit()) can filter candidates individually
    instead of failing the whole request on the first unauthorized node."""
    caller_mask = admin_db.effective_scope_mask(principal=user_id)
    return caller_mask != 0 and (node_scope & ~caller_mask & ALL_SCOPES_VISIBLE) == 0


def _decode_shared_node_id(encoded: str) -> int:
    """Decodes a caller-supplied store-encoded node id, rejecting anything
    that doesn't unambiguously name a SHARED-store node -- the console is
    shared-store-only (see module note above), so a private-store id here
    is a caller error, not a value to silently reinterpret."""
    try:
        raw_id, is_shared = decode_store_id(int(encoded))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail=f"malformed node id {encoded!r}")
    if not is_shared:
        raise HTTPException(
            status_code=400,
            detail=f"node id {encoded!r} is not a shared-store id -- the "
            "console only operates on the shared tier",
        )
    return raw_id


# ── (a) Audit log ───────────────────────────────────────────────────────

@app.get("/admin/audit-log", response_model=AuditLogTailResponse)
async def get_audit_log_tail(
    since_seq: int = 0,
    limit: int = 100,
    user_id: str = Depends(get_current_user_id),
    admin_db=Depends(get_admin_db),
    audit_log=Depends(get_audit_log),
):
    _require_admin_db(admin_db)
    # No per-scope filtering of audit records (advisor-reviewable known
    # gap, stated explicitly rather than left silent): ADMIN_ROLE_VALUES
    # has only one role today ("admin"), with no scope-filtered "auditor"
    # variant this endpoint could check per record. Any non-expired admin
    # grant, over ANY scope, is enough to read the whole log.
    if admin_db.effective_scope_mask(principal=user_id) == 0:
        raise HTTPException(status_code=403, detail="not authorized to read the audit log")

    try:
        records = audit_log.tail(since_seq=since_seq, limit=limit)
    except AuditLogError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return AuditLogTailResponse(
        records=[AuditRecordResponse(**r.__dict__) for r in records],
        next_since_seq=records[-1].seq if records else None,
    )


@app.get("/admin/audit-log/verify", response_model=AuditVerifyResponse)
async def verify_audit_log(
    user_id: str = Depends(get_current_user_id),
    admin_db=Depends(get_admin_db),
    audit_log=Depends(get_audit_log),
):
    _require_admin_db(admin_db)
    if admin_db.effective_scope_mask(principal=user_id) == 0:
        raise HTTPException(status_code=403, detail="not authorized to read the audit log")

    try:
        audit_log.verify()
    except AuditLogError as e:
        return AuditVerifyResponse(valid=False, error=str(e))
    return AuditVerifyResponse(valid=True)


@app.get("/admin/audit-log/export")
async def export_audit_log(
    user_id: str = Depends(get_current_user_id),
    admin_db=Depends(get_admin_db),
    audit_log=Depends(get_audit_log),
    export_key: bytes | None = Depends(get_audit_log_export_key),
):
    _require_admin_db(admin_db)
    if admin_db.effective_scope_mask(principal=user_id) == 0:
        raise HTTPException(status_code=403, detail="not authorized to read the audit log")
    if export_key is None:
        raise HTTPException(
            status_code=503,
            detail="audit log export is not configured -- set "
            "AEON_AUDIT_LOG_EXPORT_KEY_HEX to enable signed export",
        )

    from fastapi import Response

    exported = audit_log.export_signed(export_key)
    return Response(content=exported, media_type="application/json")


# ── (b) Knowledge browser ───────────────────────────────────────────────

@app.get("/admin/knowledge", response_model=KnowledgeListResponse)
async def list_knowledge(
    reason: str,
    offset: int = 0,
    limit: int = 100,
    user_id: str = Depends(get_current_user_id),
    shared_atlas: AeonClient | None = Depends(get_shared_atlas_client),
    admin_db=Depends(get_admin_db),
    governance_db=Depends(get_governance_db),
    keystore=Depends(get_keystore),
    audit_log=Depends(get_audit_log),
):
    """
    v4-plan.md Stage 4 task 7's "mandatory read-reason prompts in the audit
    entry" applies to THIS route specifically -- it's the only admin route
    that returns subject content (decrypted fragment text), not just
    governance metadata or a receipt. `reason` is required (not merely
    accepted) and one audit record is appended per REQUEST, not per node
    returned -- a paginated browse of 100 nodes gets one record naming the
    reason/offset/limit/returned_count, never the nodes' own text
    (governance.py's AuditLog.append() docstring: payloads must never carry
    what a classifier redacted, and unredacted browsed content is exactly
    that category).
    """
    _require_admin_db(admin_db)
    _require_shared_atlas(shared_atlas)
    if not reason or not reason.strip():
        raise HTTPException(
            status_code=400,
            detail="list_knowledge: reason is mandatory (v4-plan.md Stage 4 "
            "task 7: 'mandatory read-reason prompts') -- an admin browsing "
            "shared-tier subject content must state why",
        )

    # The caller's OWN granted scope, derived server-side -- never a
    # caller-supplied scope_mask (advisor review: list_nodes_by_scope(
    # ALL_SCOPES_VISIBLE) returns EVERY live node including unscoped ones,
    # so passing a caller-supplied mask through verbatim would hand a
    # narrowly-scoped admin the entire shared store).
    #
    # Deliberate read/write asymmetry (advisor review): this LISTING uses
    # list_nodes_by_scope()'s own OVERLAP semantics (a multi-scope node
    # shows up if the caller has a grant over ANY of its scope bits), but
    # _require_scope_containment() (used by every WRITE/action route
    # below, and by the erasure routes) requires the caller's grants to
    # fully CONTAIN a node's scope_bitmap before they can act on it. This
    # means a caller can legitimately SEE a multi-scope node here that
    # they cannot supersede/tombstone/include-in-an-erasure-case -- "see
    # broadly, act narrowly" is the intended admin UX, not an oversight.
    caller_mask = admin_db.effective_scope_mask(principal=user_id)
    if caller_mask == 0:
        raise HTTPException(status_code=403, detail="not authorized to browse the shared store")

    # Paginated (advisor review): list_nodes_by_scope() itself has no
    # pagination, and each listed node costs FOUR separate EBR-guarded
    # C++ reads below -- unbounded, that's O(scope size) round trips in
    # one request. Sliced in Python after the scope scan, not pushed
    # into the C++ layer -- list_nodes_by_scope() is already documented
    # as a flat, cheap scan (same one tombstone_count()/compact_mmap()
    # use), so the cost this bounds is the per-node metadata/scope/
    # superseded/governance-id reads, not the scan itself.
    all_node_ids = shared_atlas.atlas.list_nodes_by_scope(caller_mask)
    node_ids = all_node_ids[offset : offset + limit]

    def _read_metadata(raw_id: int) -> str:
        """Decrypts a node's metadata (v4-plan.md Stage 4 task 6 Phase B)
        if it's marked encrypted and this deployment can resolve its DEK.
        Falls back to returning the raw stored value (plaintext for a
        legacy/unencrypted node, or still-marker-prefixed ciphertext if
        the key can't be resolved -- e.g. governance_db/keystore not
        configured, or the record/key genuinely doesn't exist) rather
        than raising -- a browsing endpoint degrading to showing
        ciphertext is far better than 500ing the whole listing over one
        node.
        """
        raw_text = shared_atlas.atlas.get_node_metadata(raw_id)
        if keystore is None or governance_db is None:
            return raw_text
        from .crypto import is_encrypted_metadata, decrypt_metadata

        if not is_encrypted_metadata(raw_text):
            return raw_text
        gov_id = shared_atlas.atlas.get_node_governance_id(raw_id)
        subject_id = governance_db.get_subject_id(gov_id) if gov_id else None
        if subject_id is None:
            return raw_text
        node_scope = shared_atlas.atlas.get_node_scope(raw_id)
        dek = keystore.get_dek(subject_id, node_scope)
        if dek is None:
            # No key for this (subject_id, scope) -- a DESIGNED, reachable
            # outcome, not just a theoretical edge case: erasure.py's
            # execute_approved_erasure() destroys one DEK per
            # (subject_id, scope), and that key covers EVERY node sharing
            # the pair. If an erasure case targets only SOME of a
            # subject's fragments in a scope (a caller choice, not a bug --
            # see execute_approved_erasure()'s own collateral-effect
            # comment), the un-erased survivors are still listed here
            # (only the erased ones are tombstoned/excluded) but their
            # shared key is gone -- this branch is exactly how that
            # surfaces: marker-prefixed ciphertext returned to the caller
            # rather than a crash. Pinned by
            # test_console_endpoint.py's partial-erasure-collateral test.
            return raw_text
        return decrypt_metadata(dek, raw_text)

    nodes = []
    for raw_id in node_ids:
        nodes.append(
            KnowledgeNode(
                id=str(encode_store_id(raw_id, is_shared=True)),
                metadata=_read_metadata(raw_id),
                scope_mask=str(shared_atlas.atlas.get_node_scope(raw_id)),
                superseded=shared_atlas.atlas.is_node_superseded(raw_id),
                governance_record_id=str(shared_atlas.atlas.get_node_governance_id(raw_id)),
            )
        )

    # One record per REQUEST, not per node (advisor review) -- a 100-node
    # page must not become 100 audit entries, and the payload is
    # counts/params only, never the nodes' own (possibly just-decrypted)
    # text -- AuditLog.append()'s own docstring: payloads must never carry
    # what a classifier redacted, and unredacted browsed content is exactly
    # that category. audit_log is optional infra (same as everywhere else
    # in this file) -- an unconfigured deployment still browses, it just
    # can't prove it did so for a stated reason.
    if audit_log is not None:
        audit_log.append(
            action="knowledge_read",
            actor=user_id,
            payload={
                "reason": reason,
                "offset": offset,
                "limit": limit,
                "returned_count": len(nodes),
                "caller_scope_mask": str(caller_mask),
            },
        )

    return KnowledgeListResponse(nodes=nodes, total=len(all_node_ids))


@app.post("/admin/knowledge/{node_id}")
async def act_on_knowledge_node(
    node_id: str,
    body: KnowledgeActionRequest,
    user_id: str = Depends(get_current_user_id),
    shared_atlas: AeonClient | None = Depends(get_shared_atlas_client),
    admin_db=Depends(get_admin_db),
    audit_log=Depends(get_audit_log),
    governance_db=Depends(get_governance_db),
):
    """Supersede / revoke-supersede / tombstone a single shared-store node
    -- deliberately NOT behind four-eyes approval (unlike erasure): each
    of these is a single-node action, not the "bulk operation" task 7's
    four-eyes requirement targets, and each is independently reversible
    except tombstone (which is why the erasure workflow, not this route,
    is the one wrapped in four-eyes -- an erasure case is exactly "many
    tombstones, gated"; this route exists for the ordinary single-node
    correction case a knowledge-browser operator does routinely).

    "supersede"/"revoke_supersede" now go through supersession.py's
    audited primitives (v4-plan.md Stage 5 task 2 retrofit) -- found while
    building outcome-verified supersession: this route previously called
    Atlas.supersede_node()/revoke_node_supersede() directly with NO audit
    trail at all, unlike every other governance-mutating path in this
    codebase (promotion, erasure). A blank `reason` is rejected (400) for
    these two actions -- an audit record with no reason defeats the point.

    "tombstone" is a KNOWN, DELIBERATELY UNCHANGED gap: it still calls
    Atlas.tombstone_node() directly with no audit record. Not retrofitted
    here -- out of this task's scope (Stage 5 task 2 is about supersession
    specifically), left as a flagged, not-yet-fixed item rather than
    silently expanded into.
    """
    _require_admin_db(admin_db)
    _require_shared_atlas(shared_atlas)
    raw_id = _decode_shared_node_id(node_id)

    try:
        node_scope = shared_atlas.atlas.get_node_scope(raw_id)
    except RuntimeError as e:
        raise HTTPException(status_code=404, detail=str(e))

    _require_scope_containment(admin_db, user_id, node_scope)

    try:
        if body.action == "supersede":
            if not body.reason or not body.reason.strip():
                raise HTTPException(
                    status_code=400,
                    detail="reason must be non-blank for action=supersede",
                )
            _supersede_node_audited(
                shared_atlas, raw_id, audit_log, actor=user_id,
                reason=body.reason, governance_db=governance_db,
            )
        elif body.action == "revoke_supersede":
            if not body.reason or not body.reason.strip():
                raise HTTPException(
                    status_code=400,
                    detail="reason must be non-blank for action=revoke_supersede",
                )
            _revoke_node_supersession_audited(
                shared_atlas, raw_id, audit_log, actor=user_id,
                reason=body.reason, governance_db=governance_db,
            )
        elif body.action == "tombstone":
            shared_atlas.atlas.tombstone_node(raw_id)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"unknown action {body.action!r} -- must be one of "
                "supersede, revoke_supersede, tombstone",
            )
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=409, detail=str(e))

    return {"ok": True}


@app.post("/admin/supersede-by-commit", response_model=SupersedeByCommitResponse)
async def supersede_by_commit(
    body: SupersedeByCommitRequest,
    user_id: str = Depends(get_current_user_id),
    shared_atlas: AeonClient | None = Depends(get_shared_atlas_client),
    admin_db=Depends(get_admin_db),
    audit_log=Depends(get_audit_log),
    governance_db=Depends(get_governance_db),
):
    """v4-plan.md Stage 5 task 2's HTTP entry point: outcome-verified
    supersession. An external caller (the org's own CI/commit-graph
    integration -- Aeon does not poll or integrate with any VCS/CI
    provider itself, same trust boundary as task 3's VerificationResult)
    tells Aeon that `body.commit_sha` was reverted or replaced. Every
    promoted node whose VerificationResult cited this commit is
    superseded, PROVIDED the caller holds admin scope containment over
    that specific node -- a caller authorized for some but not all cited
    nodes gets a partial result (this node's own outcome is independently
    recorded either way), not a blanket rejection.

    Not four-eyes gated, same reasoning as act_on_knowledge_node():
    supersession is reversible and every outcome is independently
    audit-recorded, unlike erasure's irreversible tombstoning.

    A caller holding NO admin scope at all is rejected outright (403)
    before any lookup runs -- returning a per-node "not authorized"
    breakdown to someone with zero legitimate access would itself leak
    which nodes exist for this commit.
    """
    _require_admin_db(admin_db)
    _require_shared_atlas(shared_atlas)

    caller_mask = admin_db.effective_scope_mask(principal=user_id)
    if caller_mask == 0:
        raise HTTPException(status_code=403, detail="not authorized for any scope")

    def _authorize(node_id: int) -> bool:
        try:
            node_scope = shared_atlas.atlas.get_node_scope(node_id)
        except RuntimeError:
            # A raw id from an old audit record no longer resolving (e.g.
            # a compaction reclaimed/shifted ids since promotion -- see
            # supersession.py's module doc comment) is NOT an
            # authorization failure -- let it proceed so the real error
            # surfaces from supersede_node() itself with an accurate
            # message, rather than being misreported as "not authorized"
            # here.
            return True
        return _caller_scope_containment_ok(admin_db, user_id, node_scope)

    receipt = supersede_by_reverted_commit(
        shared_atlas,
        audit_log,
        body.commit_sha,
        actor=user_id,
        governance_db=governance_db,
        authorize=_authorize,
    )

    return SupersedeByCommitResponse(
        superseded=[
            str(encode_store_id(n, is_shared=True)) for n in receipt["superseded"]
        ],
        could_not_supersede=[
            {**entry, "node_id": str(encode_store_id(entry["node_id"], is_shared=True))}
            for entry in receipt["could_not_supersede"]
        ],
    )


# ── (c) Erasure workflow ────────────────────────────────────────────────

@app.post("/admin/erasure", response_model=ErasureCaseCreateResponse)
async def create_erasure(
    body: ErasureCaseCreateRequest,
    user_id: str = Depends(get_current_user_id),
    shared_atlas: AeonClient | None = Depends(get_shared_atlas_client),
    admin_db=Depends(get_admin_db),
    erasure_db=Depends(get_erasure_db),
):
    _require_admin_db(admin_db)
    _require_shared_atlas(shared_atlas)
    if erasure_db is None:
        raise HTTPException(status_code=404, detail="control plane not configured")
    if not body.node_ids:
        # Checked here, not left to fall through to create_erasure_case()'s
        # own ValueError: an empty list would otherwise compute
        # combined_scope=0 and fail has_role()'s check below with a
        # misleading 403 instead of the actual problem.
        raise HTTPException(status_code=400, detail="node_ids must be non-empty")

    raw_ids = [_decode_shared_node_id(n) for n in body.node_ids]

    # Same defense-in-depth as the promotion endpoint: the CALLER creating
    # the case must independently hold admin over every target node's
    # current scope -- four-eyes approval later authorizes EXECUTION, not
    # who gets to file a case in the first place.
    combined_scope = 0
    try:
        for raw_id in raw_ids:
            combined_scope |= shared_atlas.atlas.get_node_scope(raw_id)
    except RuntimeError as e:
        raise HTTPException(status_code=404, detail=str(e))
    # Containment against the UNION of every target's scope, not overlap
    # (advisor review, same bug as act_on_knowledge_node's fix): a caller
    # granted only 0x1000 must not be able to file a case naming nodes in
    # 0x1000, 0x2000, AND 0x4000 just because one target overlaps their
    # one grant -- every bit of combined_scope must be covered.
    _require_scope_containment(admin_db, user_id, combined_scope)

    try:
        case_id = create_erasure_case(
            admin_db,
            erasure_db,
            shared_atlas=shared_atlas,
            node_ids=raw_ids,
            reason=body.reason,
            requested_by=user_id,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=body.expires_in_seconds),
            required_approvals=body.required_approvals,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return ErasureCaseCreateResponse(case_id=case_id)


def _erasure_case_response(case_id: int, case: dict) -> ErasureCaseResponse:
    if case["completed_at"] is None:
        return ErasureCaseResponse(case_id=case_id, completed=False)
    receipt = json.loads(case["receipt"])
    return ErasureCaseResponse(
        case_id=case_id,
        completed=True,
        erased=[str(encode_store_id(n, is_shared=True)) for n in receipt["erased"]],
        could_not_erase=[
            ErasureReceiptEntry(
                node_id=str(encode_store_id(e["node_id"], is_shared=True)),
                reason=e["reason"],
            )
            for e in receipt["could_not_erase"]
        ],
    )


@app.get("/admin/erasure/{case_id}", response_model=ErasureCaseResponse)
async def get_erasure(
    case_id: int,
    user_id: str = Depends(get_current_user_id),
    admin_db=Depends(get_admin_db),
    erasure_db=Depends(get_erasure_db),
):
    _require_admin_db(admin_db)
    if erasure_db is None:
        raise HTTPException(status_code=404, detail="control plane not configured")

    case = erasure_db.get_case(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="erasure case not found")
    req = admin_db.get_request(case["approval_request_id"])
    scope_mask = int(json.loads(req["target"])["scope_mask"])
    _require_scope_containment(admin_db, user_id, scope_mask)

    return _erasure_case_response(case_id, case)


@app.post("/admin/erasure/{case_id}/execute", response_model=ErasureCaseResponse)
async def execute_erasure(
    case_id: int,
    user_id: str = Depends(get_current_user_id),
    shared_atlas: AeonClient | None = Depends(get_shared_atlas_client),
    admin_db=Depends(get_admin_db),
    erasure_db=Depends(get_erasure_db),
    governance_db=Depends(get_governance_db),
    audit_log=Depends(get_audit_log),
    keystore=Depends(get_keystore),
    mgr: SessionManager = Depends(get_session_manager),
):
    _require_admin_db(admin_db)
    _require_shared_atlas(shared_atlas)
    if erasure_db is None:
        raise HTTPException(status_code=404, detail="control plane not configured")

    case = erasure_db.get_case(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="erasure case not found")
    req = admin_db.get_request(case["approval_request_id"])
    if req is None:
        raise HTTPException(status_code=500, detail="erasure case's approval request is missing")
    scope_mask = int(json.loads(req["target"])["scope_mask"])

    _require_scope_containment(admin_db, user_id, scope_mask)

    try:
        execute_approved_erasure(
            admin_db,
            erasure_db,
            case_id,
            actor=user_id,
            shared_atlas=shared_atlas,
            audit_log=audit_log,
            governance_db=governance_db,
            keystore=keystore,
            # THE DERIVED-RECORD CASCADE, which was dead in production until now.
            # `execute_approved_erasure` has always accepted `record_store` and this call
            # site never passed it, so an approved erasure tombstoned the Atlas nodes and
            # left every record extracted from those sessions in place -- records are PII
            # DERIVED from conversation, and `records_for_session()` is documented as the
            # cascade index. The store is the erasure subject's own per-tenant file.
            record_store=mgr.get_store(user_id),
        )
    except ErasureTransientFailure as e:
        # Distinct from the generic 409 below: the case is NOT completed
        # and the approval is NOT consumed (see the exception's own doc
        # comment) -- 503 signals "retry shortly", not "this request is
        # invalid".
        raise HTTPException(status_code=503, detail=str(e))
    except (ValueError, RuntimeError, PermissionError) as e:
        raise HTTPException(status_code=409, detail=str(e))

    return _erasure_case_response(case_id, erasure_db.get_case(case_id))
