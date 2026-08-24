from functools import lru_cache
from pathlib import Path
from typing import Optional
import os
from .auth import AuthError, InsecureDevAuthProvider, get_auth_provider
from .client import AeonClient
from .llm import MockProvider, OllamaProvider, LLMProvider
from .session import SessionManager
from .trace import TraceGraph
from fastapi import Depends, Header, HTTPException

# Default Paths (can be overridden by Env Vars)
DEFAULT_ATLAS_PATH = os.environ.get("AEON_ATLAS_PATH", "./data/atlas.aeon")
DEFAULT_TRACE_DIR = os.environ.get("AEON_TRACE_DIR", "./data/traces")
DEFAULT_TRACE_PATH = os.environ.get(
    "AEON_TRACE_PATH", str(Path(DEFAULT_TRACE_DIR) / "trace.bin")
)
# V4 Stage 4: the shared/org-wide tier's Atlas store, PHYSICALLY separate
# from the private one above (v4-plan.md Stage 4 architectural decision --
# not the same file filtered by scope_mask). Unset by default
# (AEON_SHARED_ATLAS_PATH) -- the shared tier is optional infrastructure a
# deployment opts into, not a standing requirement; get_shared_atlas_client()
# returns None when unset, and every caller threading it through
# (ContextManager, SessionManager, server.py routes) must handle that.
DEFAULT_SHARED_ATLAS_PATH = os.environ.get("AEON_SHARED_ATLAS_PATH")

# v4-plan.md Stage 4 task 6 Phase B: the shared store's metadata field
# holds an encrypted (nonce + base64) payload once crypto.py's keystore is
# configured, not raw text -- 256 usable bytes yields only ~170 after that
# overhead, tighter than dreamer.py's existing ~250-char convention for
# node text. 512 yields ~370 usable plaintext bytes, LARGER than today's
# effective budget. Only affects NEW shared-store files (metadata_size is
# read from an existing file's own header, same as dim, and does NOT
# retroactively grow an existing store -- enabling crypto-erase on a
# store already created at 256 leaves the budget at ~156 bytes, not
# ~370, with no migration path) -- see AtlasOptions::metadata_size's doc
# comment (atlas.hpp).
#
# Parsed defensively, unlike a bare int(): every other AEON_* constant in
# this file is a string or a `.lower() == "true"` compare and can't throw
# at import time -- a non-numeric value here must not take down `import
# aeon_py` for every caller, including ones that never touch the shared
# store.
try:
    DEFAULT_SHARED_ATLAS_METADATA_SIZE = int(
        os.environ.get("AEON_SHARED_ATLAS_METADATA_SIZE", "512")
    )
except ValueError:
    DEFAULT_SHARED_ATLAS_METADATA_SIZE = 512

# V4 Stage 4 task 2/7: promotion + its control-plane/approval backing.
# All unset by default -- promotion (like the shared tier itself) is
# optional infrastructure a deployment opts into, not a standing
# requirement. AEON_CONTROL_PLANE_DATABASE_URL unset means
# get_governance_db()/get_admin_db() return None, and any route
# depending on them (the promotion-execute endpoint, server.py) must
# handle that the same way every other optional-infra dependency here
# does.
#
# Read once, at import time -- NOT re-read per-request. This env var (and
# the AEON_* ones below it) must be set in the process environment
# BEFORE aeon_py is first imported. Setting it later (e.g. inside a test
# fixture, after import) has no effect: this module-level constant has
# already been captured, and get_control_plane_engine()'s @lru_cache()
# means even a subsequent call sees the same (None) result. Symptom if
# this is missed: every admin/promotion route silently 404s
# ("control plane not configured") with no indication why.
DEFAULT_CONTROL_PLANE_DATABASE_URL = os.environ.get("AEON_CONTROL_PLANE_DATABASE_URL")
DEFAULT_AUDIT_LOG_PATH = os.environ.get("AEON_AUDIT_LOG_PATH", "./data/governance/audit.jsonl")
# Comma-separated adopter regex patterns (promotion.py's IdentifierCorpus
# -- "this is CONFIGURATION, not code"), plus the two generic redactors.
# Empty/unset patterns with both generic redactors off is the classifier's
# own fail-closed default (IdentifierCorpus.is_empty()) -- deliberately
# NOT special-cased here, so an adopter who deploys with zero corpus
# config gets a promotion pipeline that rejects everything, not one that
# silently passes raw content through.
DEFAULT_IDENTIFIER_CORPUS_PATTERNS = [
    p for p in os.environ.get("AEON_IDENTIFIER_CORPUS_PATTERNS", "").split(",") if p
]
DEFAULT_REDACT_EMAILS = os.environ.get("AEON_REDACT_EMAILS", "false").lower() == "true"
DEFAULT_REDACT_COMMIT_SHAS = os.environ.get("AEON_REDACT_COMMIT_SHAS", "false").lower() == "true"

# V4 Stage 4 task 5(a): the console's signed audit-log export endpoint
# (AuditLog.export_signed(), governance.py) needs an HMAC key -- read once
# at import, same as every other AEON_* constant in this file. Hex-encoded
# in the environment (arbitrary key bytes, including NUL, wouldn't survive
# an env var otherwise) -- unset means the export endpoint must 503, NOT
# sign with a default/empty key (that would make "signed" meaningless).
DEFAULT_AUDIT_LOG_EXPORT_KEY_HEX = os.environ.get("AEON_AUDIT_LOG_EXPORT_KEY_HEX")

# v4-plan.md Stage 4 task 6 Phase B: crypto.py's Keystore wraps every
# per-subject-per-scope DEK under this single deployment-wide KEK
# (AES-256-GCM, so 32 raw bytes / 64 hex chars). Same fail-closed,
# hex-in-env, read-once-at-import pattern as
# AEON_AUDIT_LOG_EXPORT_KEY_HEX -- unset means get_keystore() returns
# None and any route needing it must 503, never encrypt/decrypt with a
# default key.
DEFAULT_CRYPTO_ERASE_KEK_HEX = os.environ.get("AEON_CRYPTO_ERASE_KEK_HEX")

# v4-plan.md Stage 4 task 3: correctness-gated promotion for code
# knowledge. Off by default (promote_fragment()'s own
# require_verification=False default) -- an adopter must affirmatively
# opt in, same convention as AEON_REDACT_EMAILS/AEON_REDACT_COMMIT_SHAS,
# before promotions of code fragments start failing closed on a missing/
# failed VerificationResult.
DEFAULT_REQUIRE_CODE_VERIFICATION = (
    os.environ.get("AEON_REQUIRE_CODE_VERIFICATION", "false").lower() == "true"
)

@lru_cache()
def get_atlas_client() -> AeonClient:
    """Singleton Atlas Client (the caller's PRIVATE store)."""
    path = Path(DEFAULT_ATLAS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    return AeonClient(path)

@lru_cache()
def get_shared_atlas_client() -> Optional[AeonClient]:
    """Singleton Atlas Client for the shared/org-wide tier (v4-plan.md
    Stage 4), or None if AEON_SHARED_ATLAS_PATH is unset -- see that env
    var's doc comment above."""
    if not DEFAULT_SHARED_ATLAS_PATH:
        return None
    path = Path(DEFAULT_SHARED_ATLAS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    return AeonClient(path, metadata_size=DEFAULT_SHARED_ATLAS_METADATA_SIZE)

@lru_cache()
def get_trace_manager() -> TraceGraph:
    """
    Singleton, mmap-backed TraceGraph -- ONE shared file for all users,
    exactly like get_atlas_client() (v4-plan.md). Events are scoped by
    `session_id` (the authenticated user_id) rather than one file per
    user: TraceEvent already carries session_id natively, and this mirrors
    the same isolation pattern already built for Atlas's SLB cache in
    Stage 0. This replaces an earlier per-user-JSON-file design that
    predated the C++ mmap TraceManager rewrite and was never actually
    wired up correctly (see context.py/session.py history).
    """
    path = Path(DEFAULT_TRACE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    return TraceGraph(path)

@lru_cache()
def get_llm_provider() -> LLMProvider:
    """Singleton LLM Provider."""
    if os.environ.get("AEON_USE_OLLAMA", "false").lower() == "true":
        return OllamaProvider()
    return MockProvider()

@lru_cache()
def get_control_plane_engine():
    """Singleton sync SQLAlchemy Engine, shared between get_governance_db()
    and get_admin_db() (v4-plan.md Stage 4 task 7 advisor review: two
    independent connection pools per process was wasteful and made it
    impossible to ever put a governance write and an admin check in one
    transaction). None if AEON_CONTROL_PLANE_DATABASE_URL is unset.
    Imports sqlalchemy lazily -- this module must import cleanly with
    zero `db` extras installed, same as every optional-infra dependency
    here."""
    if not DEFAULT_CONTROL_PLANE_DATABASE_URL:
        return None
    from sqlalchemy import create_engine
    return create_engine(DEFAULT_CONTROL_PLANE_DATABASE_URL, pool_pre_ping=True)


@lru_cache()
def get_governance_db():
    """See control_plane/db.py's GovernanceDB. None if no control-plane
    database is configured -- promote_fragment()'s governance_db param
    already handles None by falling back to the JSONL log's own seq."""
    engine = get_control_plane_engine()
    if engine is None:
        return None
    from .control_plane.db import GovernanceDB
    return GovernanceDB(engine)


@lru_cache()
def get_admin_db():
    """See control_plane/admin.py's AdminDB. None if no control-plane
    database is configured -- any route depending on this must 404 (not
    silently allow) when it's None, since there is no meaningful
    "unauthenticated" fallback for an admin/approval check."""
    engine = get_control_plane_engine()
    if engine is None:
        return None
    from .control_plane.admin import AdminDB
    return AdminDB(engine)


@lru_cache()
def get_erasure_db():
    """See control_plane/erasure_db.py's ErasureDB. None if no control-plane
    database is configured -- same optionality pattern as get_admin_db()/
    get_governance_db(), sharing the same engine singleton."""
    engine = get_control_plane_engine()
    if engine is None:
        return None
    from .control_plane.erasure_db import ErasureDB
    return ErasureDB(engine)


def get_crypto_erase_kek() -> Optional[bytes]:
    """The deployment-wide KEK for crypto.py's Keystore. None if
    AEON_CRYPTO_ERASE_KEK_HEX is unset -- same fail-closed reasoning as
    get_audit_log_export_key().

    Also None (with a logged warning, not a raised exception) if the hex
    string is malformed or the wrong length. Requires exactly 32 raw bytes
    (256-bit), not merely any AESGCM-valid length (16/24/32) -- crypto.py's
    DEK_SIZE_BYTES is hardcoded to 32 (AES-256), and a 16- or 24-byte KEK
    would wrap those 256-bit keys at AES-128/192 strength, silently
    weakening the wrapping below the strength of what it protects (caught
    on advisor review: an adopter-facing document was about to describe
    16/24/32 as equally acceptable without noting the weaker choice
    undersells its own DEKs). Validated HERE so a typo'd or under-length
    env var degrades this deployment to "crypto-erase not configured" (the
    same shape as leaving it unset -- every caller already handles
    get_keystore() returning None) rather than surfacing as an unhandled
    500 on the first request that touches get_keystore(), an
    @lru_cache()'d dependency that would otherwise re-raise on every
    subsequent call too.
    """
    if not DEFAULT_CRYPTO_ERASE_KEK_HEX:
        return None
    try:
        kek = bytes.fromhex(DEFAULT_CRYPTO_ERASE_KEK_HEX)
    except ValueError:
        import logging
        logging.getLogger("aeon.crypto").warning(
            "AEON_CRYPTO_ERASE_KEK_HEX is not valid hex -- crypto-erase "
            "is disabled for this deployment until it's fixed"
        )
        return None
    if len(kek) != 32:
        import logging
        logging.getLogger("aeon.crypto").warning(
            "AEON_CRYPTO_ERASE_KEK_HEX decodes to %d bytes -- exactly 32 "
            "(256-bit) is required to match the strength of the DEKs it "
            "wraps -- crypto-erase is disabled for this deployment until "
            "it's fixed",
            len(kek),
        )
        return None
    return kek


@lru_cache()
def get_keystore():
    """See crypto.py's Keystore. None if no control-plane database is
    configured OR no KEK is configured (fail-closed on either gap) --
    shares the same engine singleton as get_governance_db()/get_admin_db()/
    get_erasure_db()."""
    engine = get_control_plane_engine()
    kek = get_crypto_erase_kek()
    if engine is None or kek is None:
        return None
    from .crypto import Keystore
    return Keystore(engine, kek)


@lru_cache()
def get_audit_log():
    """Singleton AuditLog (governance.py) -- local, always available
    regardless of whether a control plane is configured; promote_fragment()
    requires one unconditionally (it's the tamper-evident record of
    intent, Postgres is only ever a queryable index over it)."""
    from .governance import AuditLog
    path = Path(DEFAULT_AUDIT_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    return AuditLog(path)


@lru_cache()
def get_identifier_corpus():
    """See promotion.py's IdentifierCorpus -- adopter configuration, not
    code. Zero patterns and both generic redactors off (the defaults
    here) is the classifier's own documented fail-closed state: a
    deployment that hasn't configured anything gets a promotion pipeline
    that rejects every fragment, not one that silently passes raw content
    through."""
    from .promotion import IdentifierCorpus
    return IdentifierCorpus(
        patterns=DEFAULT_IDENTIFIER_CORPUS_PATTERNS,
        redact_emails=DEFAULT_REDACT_EMAILS,
        redact_commit_shas=DEFAULT_REDACT_COMMIT_SHAS,
    )


def get_require_code_verification() -> bool:
    """Whether this deployment gates promotion on a passed VerificationResult
    (v4-plan.md Stage 4 task 3). Not @lru_cache()'d -- unlike the other
    DI functions here, this returns a plain bool with nothing to
    construct/cache, and FastAPI's Depends() already reuses simple
    callables cheaply per request."""
    return DEFAULT_REQUIRE_CODE_VERIFICATION


@lru_cache()
def get_audit_log_export_key() -> Optional[bytes]:
    """Singleton HMAC key for AuditLog.export_signed()/verify_export_signature()
    (v4-plan.md Stage 4 task 5(a)). None if AEON_AUDIT_LOG_EXPORT_KEY_HEX is
    unset -- the export endpoint must 503 rather than sign with a default
    key, same fail-closed reasoning as get_identifier_corpus()'s empty
    default."""
    if not DEFAULT_AUDIT_LOG_EXPORT_KEY_HEX:
        return None
    return bytes.fromhex(DEFAULT_AUDIT_LOG_EXPORT_KEY_HEX)


@lru_cache()
def get_session_manager() -> SessionManager:
    """Singleton Session Manager."""
    atlas = get_atlas_client()
    trace = get_trace_manager()
    llm = get_llm_provider()
    shared_atlas = get_shared_atlas_client()
    return SessionManager(atlas, trace, llm, shared_atlas_client=shared_atlas)

async def get_current_user_id(
    authorization: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
) -> str:
    """
    Dependency that returns the caller's VERIFIED identity (v4-plan.md
    Stage 0). Real auth (AuthProvider.get_auth_provider() == jwt mode)
    verifies the `Authorization: Bearer <token>` header's signature and
    returns its `sub` claim -- X-User-ID is ignored in this mode, since an
    unsigned header must never be trusted as identity. Only in explicit
    AEON_AUTH_MODE=insecure_dev_no_verify does X-User-ID get trusted
    verbatim, matching the server's pre-Stage-0 behavior for local dev.
    """
    provider = get_auth_provider()
    try:
        if isinstance(provider, InsecureDevAuthProvider):
            return await provider.verify_x_user_id_header(x_user_id)
        return await provider.verify(authorization)
    except AuthError as e:
        raise HTTPException(status_code=401, detail=str(e))

async def get_cognition(
    user_id: str = Depends(get_current_user_id),
):
    """Dependency helper (optional)."""
    mgr = get_session_manager()
    return await mgr.get_loop(user_id)
