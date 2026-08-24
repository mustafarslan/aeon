"""
Sync Postgres client for admin roles + four-eyes approval (v4-plan.md
Stage 4 task 7). Same in-process, synchronous pattern as db.py's
GovernanceDB, for the same reason: callers gating a privileged operation
need a fast, ordered, near-certain-to-succeed check before proceeding,
not a network hop through control_plane/app.py's own HTTP API.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional, Union

from sqlalchemy import create_engine, func, select
from sqlalchemy.engine import Engine

from ..client import ALL_SCOPES_VISIBLE
from .schema import ADMIN_ROLE_VALUES, admin_roles, approval_grants, approval_requests


class WildcardScopeError(ValueError):
    """Raised by grant_role() when scope_mask == ALL_SCOPES_VISIBLE and
    allow_wildcard was not explicitly set -- v4-plan.md Stage 4 task 7's
    "scope-scoped admin roles by default" / "never a wildcard bypass"
    means a role grant covering every scope can't be the path of least
    resistance, the way query_stores(shared_scope_mask=ALL_SCOPES_VISIBLE)
    already is elsewhere. A real need for a wildcard grant (e.g.
    bootstrapping the first admin) still exists -- allow_wildcard=True is
    the deliberate, explicit way to ask for it."""


class DuplicateApprovalError(ValueError):
    """Raised by grant_approval() when `approver` has already approved
    this request -- the UNIQUE(request_id, approver) constraint
    (schema.py) is what actually enforces four-eyes (N DISTINCT
    approvers); this just turns the resulting IntegrityError into a
    clear, typed error instead of a raw driver exception."""


class AdminDB:
    def __init__(self, database_url_or_engine: Union[str, Engine]):
        """See db.py's GovernanceDB.__init__ for why this accepts either
        a URL or an existing Engine -- same reasoning, same pattern."""
        if isinstance(database_url_or_engine, str):
            self._engine: Engine = create_engine(database_url_or_engine, pool_pre_ping=True)
            self._owns_engine = True
        else:
            self._engine = database_url_or_engine
            self._owns_engine = False

    # ── Roles ────────────────────────────────────────────────────────

    def grant_role(
        self,
        *,
        principal: str,
        scope_mask: int,
        granted_by: str,
        role: str = "admin",
        expires_at: Optional[datetime] = None,
        allow_wildcard: bool = False,
    ) -> int:
        """Grants `principal` `role` over `scope_mask`, returning the new
        admin_roles row id. `expires_at=None` is a permanent grant;
        setting it makes this a time-boxed break-glass grant (schema.py's
        admin_roles doc comment -- same mechanism, not a separate one).
        """
        if role not in ADMIN_ROLE_VALUES:
            raise ValueError(f"grant_role: unknown role {role!r}, must be one of {ADMIN_ROLE_VALUES}")
        if scope_mask == ALL_SCOPES_VISIBLE and not allow_wildcard:
            raise WildcardScopeError(
                "grant_role: scope_mask=ALL_SCOPES_VISIBLE is a wildcard "
                "admin grant, forbidden by default (v4-plan.md Stage 4 "
                "task 7: 'scope-scoped admin roles by default'). Pass "
                "allow_wildcard=True if this is truly intended."
            )
        with self._engine.begin() as conn:
            result = conn.execute(
                admin_roles.insert()
                .values(
                    principal=principal,
                    scope_mask=scope_mask,
                    role=role,
                    expires_at=expires_at,
                    granted_by=granted_by,
                )
                .returning(admin_roles.c.id)
            )
            return result.scalar_one()

    def has_role(self, *, principal: str, scope_mask: int, role: str = "admin") -> bool:
        """True if `principal` currently holds a non-expired `role` grant
        whose scope_mask overlaps `scope_mask` (bitwise AND != 0).

        The overlap check runs in PYTHON, not SQL: scope_mask is
        NUMERIC(20,0) (must hold the full uint64_t range, including
        values with bit 63 set -- see governance_records' node-id columns
        for the same reasoning), and Postgres's `&` bitwise operator only
        works on integer types, not numeric. This control plane is
        documented as low-QPS (v4-plan.md Stage 4 task 1), and a
        principal realistically holds a handful of role grants at most --
        fetching the non-expired candidates and filtering in Python is
        the honest choice here, not a premature-optimization tradeoff.
        """
        candidates = self._non_expired_roles(principal=principal, role=role)
        return any((int(row.scope_mask) & scope_mask) != 0 for row in candidates)

    def effective_scope_mask(self, *, principal: str, role: str = "admin") -> int:
        """The OR of every non-expired scope_mask `principal` currently
        holds `role` over (v4-plan.md Stage 4 task 5(b), advisor review):
        the console's knowledge-browser LISTING route must derive the
        caller's own visibility from this, never from a caller-supplied
        scope_mask query parameter -- list_nodes_by_scope(mask)'s own doc
        comment says ALL_SCOPES_VISIBLE returns EVERY live node including
        unscoped ones, so passing a caller-supplied mask through verbatim
        would hand a scope-0x4 admin the entire shared store. Returns 0
        (matches nothing -- 0 & anything == 0, and list_nodes_by_scope(0)
        returns no results) if `principal` holds no grants at all, rather
        than ALL_SCOPES_VISIBLE or raising -- "no grants" must fail
        closed to "sees nothing", the same fail-closed default
        IdentifierCorpus.is_empty() uses for promotion.
        """
        candidates = self._non_expired_roles(principal=principal, role=role)
        mask = 0
        for row in candidates:
            mask |= int(row.scope_mask)
        return mask

    def _non_expired_roles(self, *, principal: str, role: str):
        with self._engine.connect() as conn:
            return conn.execute(
                select(admin_roles).where(
                    admin_roles.c.principal == principal,
                    admin_roles.c.role == role,
                    (admin_roles.c.expires_at.is_(None))
                    | (admin_roles.c.expires_at > func.now()),
                )
            ).fetchall()

    # ── Four-eyes approval ──────────────────────────────────────────

    def create_approval_request(
        self,
        *,
        action: str,
        target: str,
        reason: str,
        requested_by: str,
        expires_at: datetime,
        required_approvals: int = 2,
    ) -> int:
        if not reason or not reason.strip():
            raise ValueError(
                "create_approval_request: reason is mandatory (v4-plan.md "
                "Stage 4 task 7: 'mandatory read-reason prompts') -- an "
                "empty or whitespace-only string doesn't satisfy that"
            )
        with self._engine.begin() as conn:
            result = conn.execute(
                approval_requests.insert()
                .values(
                    action=action,
                    target=target,
                    reason=reason,
                    requested_by=requested_by,
                    required_approvals=required_approvals,
                    expires_at=expires_at,
                )
                .returning(approval_requests.c.id)
            )
            return result.scalar_one()

    def grant_approval(self, *, request_id: int, approver: str) -> int:
        """Records one approver's grant. Raises DuplicateApprovalError if
        `approver` already approved this request -- the SAME person
        approving twice must never satisfy required_approvals=2 on their
        own; that's the entire point of four-eyes."""
        from sqlalchemy.exc import IntegrityError

        try:
            with self._engine.begin() as conn:
                result = conn.execute(
                    approval_grants.insert()
                    .values(request_id=request_id, approver=approver)
                    .returning(approval_grants.c.id)
                )
                return result.scalar_one()
        except IntegrityError as e:
            raise DuplicateApprovalError(
                f"grant_approval: {approver!r} has already approved "
                f"request {request_id}"
            ) from e

    def is_approved(self, request_id: int) -> bool:
        """True iff: the request hasn't expired (lazy -- expires_at
        compared against now() at read time, no sweeper, see schema.py's
        section doc comment), hasn't been revoked, and has at least
        required_approvals DISTINCT approvers (approval_grants' UNIQUE
        constraint already guarantees "distinct"; this just counts rows).
        Does NOT consider whether the request was already executed --
        callers checking "should I execute this" should also check
        get_request(request_id).executed_at is None.
        """
        with self._engine.connect() as conn:
            req = conn.execute(
                select(approval_requests).where(approval_requests.c.id == request_id)
            ).mappings().first()
            if req is None:
                return False
            if req["revoked_at"] is not None:
                return False
            if req["expires_at"] <= datetime.now(timezone.utc):
                return False
            grant_count = conn.execute(
                select(func.count()).select_from(approval_grants).where(
                    approval_grants.c.request_id == request_id
                )
            ).scalar_one()
            return grant_count >= req["required_approvals"]

    def get_request(self, request_id: int) -> Optional[dict]:
        with self._engine.connect() as conn:
            row = conn.execute(
                select(approval_requests).where(approval_requests.c.id == request_id)
            ).mappings().first()
            return dict(row) if row is not None else None

    def mark_executed(self, request_id: int) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                approval_requests.update()
                .where(approval_requests.c.id == request_id)
                .values(executed_at=func.now())
            )

    def revoke_request(self, request_id: int) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                approval_requests.update()
                .where(approval_requests.c.id == request_id)
                .values(revoked_at=func.now())
            )

    def dispose(self) -> None:
        if self._owns_engine:
            self._engine.dispose()
