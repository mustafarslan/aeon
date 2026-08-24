"""
DI wiring for the control-plane app (app.py) -- deliberately separate
from shell/aeon_py/dependencies.py (the retrieval-service app's own DI).
Two apps, two dependency modules: per v4-plan.md Stage 4 task 7 ("admin
reads go through the same enforcement path as any other read, never a
wildcard bypass"), keeping this app's wiring in its own module makes that
auditable -- a set of admin routes bolted onto the retrieval service's
existing FastAPI() instance would make a bypass one Depends override
away.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

DATABASE_URL = os.environ.get("AEON_CONTROL_PLANE_DATABASE_URL")


@lru_cache()
def get_engine() -> AsyncEngine:
    if not DATABASE_URL:
        raise RuntimeError(
            "control_plane/app.py requires AEON_CONTROL_PLANE_DATABASE_URL "
            "to be set (a postgresql+psycopg:// URL) -- there is no "
            "in-memory/stub fallback for a service whose entire purpose is "
            "being the durable, queryable governance record store."
        )
    return create_async_engine(DATABASE_URL, pool_pre_ping=True)


async def get_connection() -> AsyncIterator[AsyncConnection]:
    engine = get_engine()
    async with engine.connect() as conn:
        yield conn
