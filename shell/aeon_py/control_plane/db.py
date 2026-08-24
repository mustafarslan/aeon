"""
Sync Postgres write client for promote_fragment (promotion.py).

Deliberately NOT the async control-plane FastAPI app (app.py) -- see
GovernanceDB's docstring for why the write path stays in-process and
synchronous rather than going through that app's own HTTP API.
"""

from __future__ import annotations

import uuid
from typing import Optional, Union

from sqlalchemy import create_engine, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine

from .schema import governance_log_instances, governance_records


class GovernanceDB:
    """Sync, in-process client for the write path promote_fragment
    (promotion.py) uses. This talks directly to Postgres over the sync
    psycopg3 driver -- not through control_plane/app.py's HTTP API --
    because promote_fragment's anomaly-handling design (see its own doc
    comment) depends on this write landing, ordered, BEFORE the Atlas
    scope/governance mutation, and a network hop through a separate
    service would both add latency there isn't a reason to pay and
    introduce a failure mode ("the write succeeded server-side but the
    response was lost") the existing anomaly-recording code doesn't
    cover. app.py's async engine reads from the SAME database -- one
    database, two engines (sync here, async there), no HTTP between
    them.
    """

    def __init__(self, database_url_or_engine: Union[str, Engine]):
        """Accepts either a database URL (constructs and owns its own
        connection pool -- the standalone-usage case, e.g. tests) or an
        existing Engine (shares it with whatever else the caller already
        built one for -- e.g. AdminDB, when both are used together by the
        same request handler). advisor review: a caller needing both a
        governance write and an admin-role/approval check in the same
        request otherwise ends up with two independent connection pools
        per process for no reason, and can never put the two Postgres-side
        writes in one transaction even when that would help. dispose()
        only closes a pool THIS instance created -- never a shared one a
        caller still owns."""
        if isinstance(database_url_or_engine, str):
            self._engine: Engine = create_engine(database_url_or_engine, pool_pre_ping=True)
            self._owns_engine = True
        else:
            self._engine = database_url_or_engine
            self._owns_engine = False

    def record(
        self,
        *,
        log_instance_id: uuid.UUID,
        log_instance_path: str,
        log_seq: int,
        action: str,
        actor: str,
        subject_id: str,
        source_node_id: Optional[int] = None,
        dest_node_id: Optional[int] = None,
        dest_scope: Optional[int] = None,
    ) -> int:
        """Inserts one governance_records row (ensuring its parent
        governance_log_instances row exists first, idempotently) and
        returns the new row's Postgres-assigned id -- the value
        NodeHeader.governance_record_id gets set to when a control plane
        is configured.

        Both inserts happen in one transaction; whatever the driver
        raises on connection failure or constraint violation propagates
        to the caller. promote_fragment is responsible for recording an
        anomaly and re-raising around this call, same as it already does
        for set_node_scope()/set_node_governance_id() -- this method
        itself does not swallow or retry anything.
        """
        with self._engine.begin() as conn:
            conn.execute(
                pg_insert(governance_log_instances)
                .values(id=log_instance_id, path=log_instance_path)
                .on_conflict_do_nothing(index_elements=["id"])
            )
            result = conn.execute(
                governance_records.insert()
                .values(
                    log_instance_id=log_instance_id,
                    log_seq=log_seq,
                    action=action,
                    actor=actor,
                    subject_id=subject_id,
                    source_node_id=source_node_id,
                    dest_node_id=dest_node_id,
                    dest_scope=dest_scope,
                )
                .returning(governance_records.c.id)
            )
            return result.scalar_one()

    def get_subject_id(self, record_id: int) -> Optional[str]:
        """Reads a governance_records row's subject_id (v4-plan.md Stage 4
        task 6 Phase B) -- the resolution step a shared-store metadata
        reader (the knowledge browser) or the erasure workflow uses to go
        from a node's governance_record_id to the (subject_id, scope) DEK
        lookup key. None if the record doesn't exist."""
        with self._engine.begin() as conn:
            row = conn.execute(
                select(governance_records.c.subject_id).where(
                    governance_records.c.id == record_id
                )
            ).mappings().first()
            return row["subject_id"] if row is not None else None

    def dispose(self) -> None:
        if self._owns_engine:
            self._engine.dispose()
