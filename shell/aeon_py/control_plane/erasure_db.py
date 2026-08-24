"""
Sync Postgres client for erasure cases (v4-plan.md Stage 4 task 5(c)).
Same in-process, synchronous, URL-or-shared-Engine pattern as db.py's
GovernanceDB and admin.py's AdminDB -- see either's docstring for why.

This class is deliberately thin: it only tracks a case's completion
outcome (erasure_cases has no room for anything else -- see schema.py's
doc comment). The actual four-eyes approval lifecycle (create/approve/
revoke/is_approved) is entirely AdminDB's -- an erasure case always has
exactly one approval_requests row behind it (schema.py's UNIQUE
constraint on approval_request_id), and erasure.py's business logic
functions take both an AdminDB and an ErasureDB together rather than this
class reaching into admin_roles/approval_requests itself.
"""

from __future__ import annotations

from typing import Optional, Union

from sqlalchemy import create_engine, select
from sqlalchemy.engine import Engine

from .schema import erasure_cases


class ErasureDB:
    def __init__(self, database_url_or_engine: Union[str, Engine]):
        if isinstance(database_url_or_engine, str):
            self._engine: Engine = create_engine(database_url_or_engine, pool_pre_ping=True)
            self._owns_engine = True
        else:
            self._engine = database_url_or_engine
            self._owns_engine = False

    def create_case(self, *, approval_request_id: int) -> int:
        with self._engine.begin() as conn:
            result = conn.execute(
                erasure_cases.insert()
                .values(approval_request_id=approval_request_id)
                .returning(erasure_cases.c.id)
            )
            return result.scalar_one()

    def get_case(self, case_id: int) -> Optional[dict]:
        with self._engine.connect() as conn:
            row = conn.execute(
                select(erasure_cases).where(erasure_cases.c.id == case_id)
            ).mappings().first()
            return dict(row) if row is not None else None

    def complete_case(self, case_id: int, *, receipt: str) -> None:
        """Sets completed_at (terminal fact, same convention as
        approval_requests.executed_at) and stores the receipt JSON --
        both in the same update, since they're produced together by the
        same erasure.execute_approved_erasure() call. Idempotent from the
        caller's point of view is NOT guaranteed here at the DB layer
        (calling this twice just overwrites the receipt) -- the actual
        replay guard is execute_approved_erasure()'s own
        `completed_at is not None` check before this is ever called a
        second time, same shape as promotion's mark_executed() guard.
        """
        from sqlalchemy import func

        with self._engine.begin() as conn:
            conn.execute(
                erasure_cases.update()
                .where(erasure_cases.c.id == case_id)
                .values(completed_at=func.now(), receipt=receipt)
            )

    def dispose(self) -> None:
        if self._owns_engine:
            self._engine.dispose()
