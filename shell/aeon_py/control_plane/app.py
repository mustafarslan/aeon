"""
Aeon Control Plane — separate FastAPI app (v4-plan.md Stage 4 task 1).

Deliberately its own `FastAPI()` instance, its own dependency module
(dependencies.py), not a router mounted onto shell/aeon_py/server.py --
see dependencies.py's module docstring for why.

This increment: read-only query surface over governance_records, proving
the separation and queryability the plan calls for. Deliberately does NOT
include approvals or roles yet (task 7) -- those need a state machine
designed on its own, not bolted onto this pass. Run with:

    uvicorn aeon_py.control_plane.app:app --port 8001
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncConnection

from .dependencies import get_connection
from .schema import governance_records

app = FastAPI(title="Aeon Control Plane", version="0.1.0")


@app.get("/health")
async def health_check():
    return {"status": "ok", "component": "AeonControlPlane"}


@app.get("/governance-records/{record_id}")
async def get_governance_record(
    record_id: int, conn: AsyncConnection = Depends(get_connection)
):
    result = await conn.execute(
        select(governance_records).where(governance_records.c.id == record_id)
    )
    row = result.mappings().first()
    if row is None:
        raise HTTPException(status_code=404, detail="governance record not found")
    return dict(row)
