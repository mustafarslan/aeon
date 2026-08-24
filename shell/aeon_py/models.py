from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class ChatRequest(BaseModel):
    text: str

# --- Trace / Graph Visualization Models ---

class TraceNode(BaseModel):
    id: str
    label: str
    type: str
    timestamp: float 
    details: Dict[str, Any]

class TraceEdge(BaseModel):
    source: str
    target: str
    type: str

class TraceResponse(BaseModel):
    nodes: List[TraceNode]
    edges: List[TraceEdge]

# --- Atlas / Room Visualization Models ---

class NeighborInfo(BaseModel):
    # str, not int (advisor review, v4-plan.md Stage 4): this id is
    # store-discriminated (encode_store_id(), client.py) once Stage 4's
    # SHARED_STORE_BIT (1<<62) or NODE_ID_DELTA_MASK (1<<63) is set, which
    # exceeds JS's Number.MAX_SAFE_INTEGER (2**53) -- a browser JSON.parse
    # on a bare large integer literal silently rounds it, corrupting the
    # id a future console (task 5) would try to navigate into. FastAPI/
    # Pydantic serialize a JSON number exactly (proven against Postgres's
    # NUMERIC(20,0) columns, control_plane/, for the same class of value)
    # -- the risk is entirely on the JS-consumer side, so the fix is
    # sending these as strings, not a serialization bug on the Python side.
    id: str
    similarity: float
    # We could add 'preview' vector or truncated text if available

class ActiveRoomResponse(BaseModel):
    room_id: str  # store-discriminated -- see NeighborInfo.id's comment
    name: str # e.g. "Room 123" or metadata if available
    path: List[NeighborInfo]
    neighbors: List[NeighborInfo]

# --- Debug Models ---

class VectorQueryRequest(BaseModel):
    vector: List[float]

class SearchResult(BaseModel):
    id: str  # store-discriminated -- see NeighborInfo.id's comment
    similarity: float
    preview: List[float]

# --- Admin / Promotion Models (v4-plan.md Stage 4 task 7) ---

class PromotionExecuteResponse(BaseModel):
    # str, not int -- store-discriminated, same JS-precision reasoning as
    # NeighborInfo.id. None if the fail-closed classifier rejected the
    # fragment (still a 200: the REQUEST was validly executed, the
    # OUTCOME was a rejection -- see server.py's execute_promotion()).
    promoted_node_id: Optional[str] = None

class VerificationResultRequest(BaseModel):
    # HTTP-level mirror of promotion.py's VerificationResult -- v4-plan.md
    # Stage 4 task 3's correctness-gated promotion for code knowledge.
    # Aeon doesn't integrate with any VCS/CI provider itself; the caller's
    # own CI/test runner already produced this outcome and hands it back
    # here. `status` must be exactly "passed" to clear a gated promotion
    # (see promote_fragment()'s VerificationResult doc comment).
    status: str
    commit_sha: Optional[str] = None
    verified_by: Optional[str] = None

class PromotionExecuteRequest(BaseModel):
    # v4-plan.md Stage 4 task 2's deferred destination-conditioned
    # re-embedding, closed as a caller-supplied vector rather than a new
    # Aeon-owned embedding pipeline (no embedding model lives in this repo
    # to hook into one) -- the caller has already computed this against
    # the DESTINATION scope's corpus using whatever embedder they run, and
    # Aeon just substitutes it for the source vector at insert time (see
    # server.py's execute_promotion()). Its dimension is validated by the
    # destination Atlas's own insert() (aeon_c_api's existing dim check,
    # bindings.cpp) -- no redundant check here. None (the default) keeps
    # today's behavior: reuse the source vector unmodified.
    destination_embedding: Optional[List[float]] = None
    # v4-plan.md Stage 4 task 3. None means "no verification result
    # supplied for this attempt" -- if this deployment has
    # AEON_REQUIRE_CODE_VERIFICATION enabled, that fails the gate closed
    # (see server.py's execute_promotion()); if not enabled, promotion
    # proceeds exactly as before.
    verification: Optional[VerificationResultRequest] = None

# --- Admin / Console Models (v4-plan.md Stage 4 task 5) ---

class AuditRecordResponse(BaseModel):
    seq: int
    prev_hash: str
    action: str
    actor: str
    payload: Dict[str, Any]
    entry_hash: str

class AuditLogTailResponse(BaseModel):
    records: List[AuditRecordResponse]
    # The highest seq returned, or None if `records` is empty -- callers
    # paginate by passing this back as the next call's since_seq.
    next_since_seq: Optional[int] = None

class AuditVerifyResponse(BaseModel):
    valid: bool
    # Set only when valid=False -- the AuditLogError message naming the
    # first record where the hash chain doesn't reconcile.
    error: Optional[str] = None

class KnowledgeNode(BaseModel):
    id: str  # store-discriminated (always shared-store here -- see NeighborInfo.id's doc comment)
    metadata: str
    scope_mask: str  # NUMERIC(20,0)-range value -- str for the same JS-precision reasoning as id
    superseded: bool
    governance_record_id: str

class KnowledgeListResponse(BaseModel):
    nodes: List[KnowledgeNode]
    total: int  # count of ALL nodes within the caller's scope, pre-pagination

class KnowledgeActionRequest(BaseModel):
    action: str  # "supersede" | "revoke_supersede" | "tombstone"
    # Required for "supersede"/"revoke_supersede" (server.py rejects a
    # blank reason for those two -- v4-plan.md Stage 5 task 2 retrofit:
    # both are now recorded in the audit log, and an audit record with no
    # reason defeats the point). Ignored for "tombstone", which is not yet
    # audited by this route (a separate, explicitly-flagged gap -- see
    # server.py's act_on_knowledge_node()).
    reason: str = ""

class SupersedeByCommitRequest(BaseModel):
    # v4-plan.md Stage 5 task 2: an external caller (the org's own CI/
    # commit-graph integration -- Aeon does not poll or integrate with any
    # VCS/CI provider itself) has determined this commit was reverted or
    # replaced. Every promoted node whose VerificationResult.commit_sha
    # (promotion.py, task 3) matches is superseded.
    commit_sha: str

class SupersedeByCommitResponse(BaseModel):
    # Store-discriminated (encode_store_id()) -- same JS-precision
    # reasoning as NeighborInfo.id.
    superseded: List[str]
    could_not_supersede: List[Dict[str, Any]]

class ErasureCaseCreateRequest(BaseModel):
    node_ids: List[str]  # store-encoded ids, as returned by the knowledge browser
    reason: str
    expires_in_seconds: int = 3600
    required_approvals: int = 2

class ErasureCaseCreateResponse(BaseModel):
    case_id: int

class ErasureReceiptEntry(BaseModel):
    node_id: str
    reason: str

class ErasureCaseResponse(BaseModel):
    case_id: int
    completed: bool
    erased: List[str] = []
    could_not_erase: List[ErasureReceiptEntry] = []
