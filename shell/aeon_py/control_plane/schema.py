"""
Control-plane schema (v4-plan.md Stage 4 task 1) — SQLAlchemy Core, not
the ORM: two tables, both plain CRUD, no relationship-mapping complexity
that would justify declarative models yet. Alembic's autogenerate reads
`metadata` directly (see ../../../alembic/env.py).
"""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID

from ..governance import GOVERNANCE_RECORD_ACTIONS

metadata = MetaData()

# v4-plan.md Stage 4 task 7: known `role` values for admin_roles. A single
# value today -- "admin" -- deliberately not over-designed with role
# variants (e.g. a read-only "auditor") nothing consumes yet; break-glass
# is NOT a separate role, it's an "admin" grant with a non-NULL
# expires_at (see admin_roles' doc comment) -- one mechanism, not two.
ADMIN_ROLE_VALUES = ("admin",)

# advisor review: `action` was a free-text String(64) written from string
# literals at three call sites in promotion.py -- a typo there would
# produce a row nothing queries. GOVERNANCE_RECORD_ACTIONS (governance.py,
# no sqlalchemy dependency) is the single source of truth both this CHECK
# constraint and promotion.py's writers derive from -- add a new value
# there, not here or in promotion.py directly. CHECK constraint, not a
# Postgres native ENUM type, so adding a value later is a plain migration,
# not an ALTER TYPE dance.

# One row per AuditLog instance (governance.py) -- an AuditLog generates a
# stable UUID on first creation (see AuditLog.instance_id) and this table
# lets a Postgres-assigned governance_records.id resolve back to exactly
# which physical JSONL file + seq it names, even if that file is later
# rotated or relocated. `path` is operator-facing context, not load-bearing
# (the UUID is the actual identity) -- a log can move without breaking
# anything already written here, since nothing in this schema stores a
# filesystem path as a foreign key.
governance_log_instances = Table(
    "governance_log_instances",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column("path", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
)

# One row per governance-affecting event promote_fragment (promotion.py)
# records. `id` is what NodeHeader.governance_record_id (schema.hpp) gets
# set to when a control plane is configured -- a stable, globally-unique
# Postgres primary key, unlike the JSONL AuditLog's own `seq`, which is
# only unique within one log file. The JSONL log remains the authoritative
# TAMPER-EVIDENT record (its hash chain, not this table, is what verify()
# checks) -- this table is a queryable, relationally-stable INDEX over it,
# not a replacement for it. See promotion.py's promote_fragment() for the
# write path and control_plane/db.py for the write itself.
governance_records = Table(
    "governance_records",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column(
        "log_instance_id",
        UUID(as_uuid=True),
        ForeignKey("governance_log_instances.id"),
        nullable=False,
    ),
    Column("log_seq", BigInteger, nullable=False),
    Column("action", String(64), nullable=False),
    Column("actor", String(256), nullable=False),
    # v4-plan.md Stage 4 task 6 Phase A: identity of whichever private-
    # store owner the promoted fragment's content derives from -- NOT the
    # same thing as erasure_cases' free-text `reason` (that's a human
    # audit narrative; this is a structured attribute a future crypto-
    # erase DEK lookup resolves through governance_record_id, i.e.
    # NodeHeader.governance_record_id -> this row -> (subject_id,
    # dest_scope) -> DEK). Required on every row, not just promotions: a
    # rejection/anomaly still names which subject's content was involved.
    # A NEW mandatory column, not backfilled -- this repo carries no
    # pre-existing governance_records rows to migrate (pre-GA).
    Column("subject_id", String(256), nullable=False),
    # NUMERIC(20,0), not BigInteger: Atlas node ids and scope_bitmap are
    # C++ uint64_t (unsigned, full 0..2**64-1 range) -- e.g.
    # NODE_ID_DELTA_MASK (client.py) sets bit 63, producing values above
    # BIGINT's signed max (2**63-1) whenever an id came from the delta
    # buffer. Found by an actual overflow (psycopg.errors.
    # NumericValueOutOfRange) while testing the delta-diversion anomaly
    # path against live Postgres, not spotted by inspection -- BigInteger
    # silently "worked" for every earlier test because none of them
    # happened to insert a delta-masked id. NUMERIC(20,0) exactly holds
    # any uint64_t value (max ~1.8e19, 20 digits) with full comparison/
    # equality semantics, unlike storing the value as TEXT.
    Column("source_node_id", Numeric(20, 0), nullable=True),
    Column("dest_node_id", Numeric(20, 0), nullable=True),
    Column("dest_scope", Numeric(20, 0), nullable=True),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    UniqueConstraint(
        "log_instance_id", "log_seq", name="uq_governance_records_log_instance_seq"
    ),
    CheckConstraint(
        "action IN (" + ", ".join(f"'{a}'" for a in GOVERNANCE_RECORD_ACTIONS) + ")",
        name="ck_governance_records_action",
    ),
)

# ═══════════════════════════════════════════════════════════════════════
# v4-plan.md Stage 4 task 7: admin roles + four-eyes approval.
#
# Order settled by advisor review: roles first (four-eyes needs to know
# who counts as a distinct approver; read-reason needs a role to
# attribute a read to), then approvals (with lazy expiry), then
# break-glass (falls out of roles' own expires_at column almost free).
#
# Expiry is LAZY everywhere in this section, by explicit design: this
# codebase has no scheduled-job mechanism (DreamingWorker is a background
# thread, not a scheduler) and a lazy `expires_at` compared against
# `now()` at read time needs no sweeper, survives process restarts for
# free, and can answer "was this valid at time T" retroactively -- which
# an audit trail wants anyway. A `status` column mutated by a sweeper
# cannot answer that and adds a failure mode (sweeper down -> stale rows
# read as live). `status`-shaped columns here (approval_requests'
# executed_at/revoked_at) hold ONLY terminal facts a human/caller caused,
# never a computed validity state.
# ═══════════════════════════════════════════════════════════════════════

# One row per (principal, scope_mask, role) grant. `expires_at` is NULL
# for a permanent grant, non-NULL for a time-boxed one -- break-glass
# access (plan: "time-boxed break-glass access") is exactly an "admin"
# row with a short expires_at, not a distinct mechanism. Validity is
# always a query (`expires_at IS NULL OR expires_at > now()`), never a
# stored flag -- see the section doc comment above for why.
admin_roles = Table(
    "admin_roles",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("principal", String(256), nullable=False),
    # NUMERIC(20,0), not BigInteger -- scope_mask is a uint64_t bitmap
    # (same reasoning as governance_records' node-id columns above).
    # ALL_SCOPES_VISIBLE (2**64-1) is the exact value an over-broad grant
    # would carry, and it alone overflows signed BIGINT, so this had to
    # be NUMERIC from this table's first migration, not discovered later.
    Column("scope_mask", Numeric(20, 0), nullable=False),
    Column("role", String(32), nullable=False),
    Column("expires_at", DateTime(timezone=True), nullable=True),
    Column("granted_by", String(256), nullable=False),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    CheckConstraint(
        "role IN (" + ", ".join(f"'{r}'" for r in ADMIN_ROLE_VALUES) + ")",
        name="ck_admin_roles_role",
    ),
)

# The operation being approved. `required_approvals` -- how many DISTINCT
# approvers (approval_grants, below) are needed; four-eyes is
# required_approvals=2. `expires_at` is mandatory (not nullable): an
# approval request that can sit pending forever is exactly the
# "time-boxed" property task 7 asks for everywhere else, applied here too.
# `reason` is mandatory (advisor review -- task 7: "mandatory read-reason
# prompts in the audit entry"; a `governance_records` column would only
# cover promotions that already executed, but the reason for REQUESTING a
# privileged operation needs to exist even for a request that's rejected
# or expires unactioned, so it lives here, not on the outcome record).
# `executed_at`/`revoked_at` are terminal facts a caller sets once the
# approved operation actually ran, or once a human explicitly revokes the
# request -- NOT computed from approval count (that's still a query, see
# is_approved() in admin.py).
approval_requests = Table(
    "approval_requests",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("action", String(64), nullable=False),
    Column("target", Text, nullable=False),
    Column("reason", Text, nullable=False),
    Column("requested_by", String(256), nullable=False),
    Column("required_approvals", Integer, nullable=False),
    Column("expires_at", DateTime(timezone=True), nullable=False),
    Column("executed_at", DateTime(timezone=True), nullable=True),
    Column("revoked_at", DateTime(timezone=True), nullable=True),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    CheckConstraint("required_approvals > 0", name="ck_approval_requests_required_approvals"),
)

# One row per APPROVER, not per request -- four-eyes means N distinct
# people, and the UNIQUE(request_id, approver) constraint below IS that
# requirement: without it, one person approving twice would satisfy a
# required_approvals=2 count on their own. is_approved() (admin.py) counts
# DISTINCT rows here against approval_requests.required_approvals.
approval_grants = Table(
    "approval_grants",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column(
        "request_id",
        BigInteger,
        ForeignKey("approval_requests.id"),
        nullable=False,
    ),
    Column("approver", String(256), nullable=False),
    Column("granted_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    UniqueConstraint("request_id", "approver", name="uq_approval_grants_request_approver"),
)

# ═══════════════════════════════════════════════════════════════════════
# v4-plan.md Stage 4 task 5(c): erasure workflow -- tracked cases with a
# completion receipt and an explicit "could not be erased" section.
#
# Reuses admin_roles/approval_requests/approval_grants wholesale rather
# than inventing a parallel approval mechanism: erasure of a subject's
# shared-tier fragments is exactly the "bulk operation" task 7 requires
# four-eyes approval for, and approval_requests' generic (action, target,
# reason) shape already fits -- action="erasure", target=JSON-encoded
# {"node_ids": [...], "scope_mask": N} locking in the EXACT node ids and
# the scope authorization computed at request time (erasure.py's
# create_erasure_case()), same replay-safety reasoning as promotion.py's
# target JSON. `reason` (approval_requests, mandatory) carries the
# case's human-readable justification (e.g. "GDPR Art. 17 request, ticket
# #456") -- no separate `subject` column here, to avoid a second field
# drifting from the one `reason` already serves for promotion.
#
# This table exists only to hold what approval_requests has no room for:
# the completion outcome. `completed_at` is the same terminal-fact
# pattern as approval_requests.executed_at -- set exactly once, by
# erasure.execute_approved_erasure(), regardless of whether every target
# node id actually got erased (a partial outcome is still a legitimate,
# auditable completion, not a reason to leave the case dangling). Scoped
# to the SHARED atlas store only (v4-plan.md Stage 4 deferred-items
# writeup): private-store nodes have no per-owner authorization model in
# this codebase (Stage 0 gave private-store ISOLATION at the SLB
# cache/session level, not per-node ownership tagging in the mmap file
# itself), so there is nothing today for an erasure endpoint to check
# authorization against for a private-store node id -- building that is a
# separate increment, not a corner cut here.
erasure_cases = Table(
    "erasure_cases",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column(
        "approval_request_id",
        BigInteger,
        ForeignKey("approval_requests.id"),
        nullable=False,
        unique=True,
    ),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column("completed_at", DateTime(timezone=True), nullable=True),
    # JSON text: {"erased": [node_id, ...], "could_not_erase": [{"node_id":
    # ..., "reason": ...}, ...]} -- NULL until completed_at is set, set
    # together with it in the same erasure_db.complete_case() call.
    Column("receipt", Text, nullable=True),
)

# ═══════════════════════════════════════════════════════════════════════
# v4-plan.md Stage 4 task 6 Phase B: per-subject-per-scope encryption
# keys for the shared store's metadata field (crypto.py's Keystore).
#
# One row per (subject_id, scope) pair -- a random 256-bit DEK, generated
# on first use and wrapped (AES-256-GCM) under a single deployment-wide
# KEK read from AEON_CRYPTO_ERASE_KEK_HEX (crypto.py), NEVER stored in
# plaintext here. DELETING a row is the actual erasure primitive: unlike
# governance_records/erasure_cases (which exist to be queried forever),
# this table's rows are meant to be destroyed -- a subject's key for a
# scope must stop existing anywhere once erased, not be soft-flagged
# (a soft flag would leave the wrapped DEK bytes sitting in the table,
# defeating the entire point). Deliberately NOT derived from the KEK via
# HKDF: a derived key can only be revoked by destroying the KEK itself,
# which would destroy every OTHER subject's key too -- real per-subject
# independence requires each key to be its own stored, deletable unit.
#
# Same DELETE-vs-WAL/PITR/backups caveat as everything else in this
# schema (the task 6 decision record, v4-plan.md): deleting this row does
# not by itself guarantee the wrapped bytes are gone from Postgres WAL,
# replication, or backups. The KEK-wrapping is what makes that survive:
# even a recovered wrapped-DEK row is unusable without the KEK, which
# lives only in the deployment's environment, never in this table.
subject_scope_keys = Table(
    "subject_scope_keys",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("subject_id", String(256), nullable=False),
    # NUMERIC(20,0), not BigInteger -- scope is a uint64_t bitmap, same
    # reasoning as every other scope_mask/node-id column in this schema.
    Column("scope", Numeric(20, 0), nullable=False),
    # Base64 text (nonce || ciphertext || GCM tag, AESGCM's combined
    # output) -- Text, not a raw bytea, mirroring AuditLog's own
    # hex/base64-in-text convention for key material elsewhere in this
    # codebase (AEON_AUDIT_LOG_EXPORT_KEY_HEX).
    Column("wrapped_dek", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    UniqueConstraint("subject_id", "scope", name="uq_subject_scope_keys_subject_scope"),
)
