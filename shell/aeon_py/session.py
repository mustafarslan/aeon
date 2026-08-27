import logging
import asyncio
import re
from collections import OrderedDict
from typing import Dict, Optional

from .context import ContextManager
from .loop import CognitiveLoop
from .client import AeonClient
from .llm import LLMProvider
from .trace import TraceGraph

logger = logging.getLogger(__name__)

class SessionManager:
    """
    Manages multi-tenant user sessions with in-memory LRU caching.

    Isolation is by `session_id` (== user_id) against ONE shared Atlas and
    ONE shared TraceGraph -- both mmap-backed and already durable on their
    own -- not by per-user files. There is nothing to explicitly save or
    load here: evicting a user from the LRU cache only drops the
    lightweight ContextManager/CognitiveLoop wrapper objects; their data
    already lives in the shared, persistent Atlas/Trace files (v4-plan.md
    Stage 0). This replaces an earlier per-user-JSON-snapshot design that
    predated the C++ mmap TraceManager rewrite and called TraceGraph
    methods (`save`/`load`) that don't exist on the current class.

    `shared_atlas_client` (v4-plan.md Stage 4) is a SEPARATE, PHYSICALLY
    distinct Atlas store for org-wide shared knowledge -- not a scope
    filter over the same private Atlas above. None (the default) means no
    shared tier is configured for this deployment; every ContextManager
    this class creates gets the same shared client reference (or None),
    same sharing pattern as the private atlas/trace above.
    """

    def __init__(self, atlas_client: AeonClient, trace: TraceGraph,
                 llm_provider: LLMProvider, max_sessions: int = 100,
                 shared_atlas_client: Optional[AeonClient] = None,
                 records_dir: Optional[str] = None):
        self.atlas = atlas_client
        self.trace = trace
        self.llm = llm_provider
        self.shared_atlas = shared_atlas_client

        # Thread-safe session storage
        # OrderedDict used as LRU: End = Most Recent, Start = Oldest
        self._active_sessions: OrderedDict[str, ContextManager] = OrderedDict()
        self._loops: Dict[str, CognitiveLoop] = {}
        # PER-TENANT RECORD STORES, and the LRU that bounds them is not optional.
        #
        # Unlike Atlas and Trace -- one shared mmap file each, isolated by session_id -- a
        # record store CANNOT be shared: `all_records()` is a whole-file scan with no tenant
        # argument, so one file would put every tenant's records into every tenant's prompt.
        # See `records.store_path_for` for the demonstration and the rejected alternative.
        #
        # The cost of that boundary is one live `core.Atlas` mmap handle per active tenant,
        # which is exactly why this dict is evicted alongside the others in `_evict_oldest`.
        # A stores dict added WITHOUT that would leak a handle per user, forever.
        self._stores: Dict[str, "RecordStore"] = {}
        self._records_dir = records_dir
        # ONE queue shared by the ingest path and the background worker. It lives here
        # rather than in dependencies because the manager is what both sides already reach:
        # ContextManager marks a session dirty on a turn, ConsolidationWorker drains it.
        from .consolidator import DirtyQueue
        self.dirty_queue = DirtyQueue()

        self.max_sessions = max_sessions
        self._lock = asyncio.Lock()

        # Pre-compile regex for security validation
        self._user_id_pattern = re.compile(r"^[a-zA-Z0-9_-]+$")

    def _validate_user_id(self, user_id: str) -> None:
        """Security check: session_id is used as a Trace key and (via
        Atlas's session_id_to_u64 hashing) a cache shard key -- neither
        needs filesystem-path safety anymore, but rejecting anything
        outside a conservative charset is still cheap insurance against a
        malformed/hostile identity claim reaching storage."""
        if not self._user_id_pattern.match(user_id):
            raise ValueError(f"Invalid user_id: {user_id}. Must be alphanumeric, _, or -.")

    async def get_context(self, user_id: str) -> ContextManager:
        """
        Retrieves or creates an isolated ContextManager for the user.
        Thread-safe and manages LRU eviction.
        """
        self._validate_user_id(user_id)

        async with self._lock:
            # 1. Cache Hit
            if user_id in self._active_sessions:
                # Move to end (Most Recently Used)
                self._active_sessions.move_to_end(user_id)
                return self._active_sessions[user_id]

            # 2. Check Capacity -> Evict if full
            if len(self._active_sessions) >= self.max_sessions:
                self._evict_oldest()

            # 3. Create wrapper over the shared Atlas/Trace, then warm the
            # SLB cache if this user has prior history in the shared trace.
            ctx = ContextManager(self.atlas, self.trace,
                                 shared_atlas_client=self.shared_atlas,
                                 dirty_queue=self.dirty_queue)
            if self.trace.has_session(user_id):
                ctx.warm_start(user_id)
                logger.info(f"Resumed session for user {user_id}")
            else:
                logger.info(f"Created new session for user {user_id}")

            self._active_sessions[user_id] = ctx
            return ctx

    async def get_loop(self, user_id: str) -> CognitiveLoop:
        """Gets or creates the CognitiveLoop for the user."""
        # Ensure context is loaded
        ctx = await self.get_context(user_id)

        if user_id not in self._loops:
            self._loops[user_id] = CognitiveLoop(ctx, self.llm,
                                                 record_store=self.get_store(user_id))

        return self._loops[user_id]

    def get_store(self, user_id: str):
        """The tenant's record store, opened on demand.

        Re-opening is not free -- `RecordStore.__init__` rebuilds its bucket index with an
        O(n) metadata scan -- so an evicted-then-returning user pays a rescan. That is the
        price of bounding mmap handles, and it is bounded work rather than unbounded memory.
        """
        if self._records_dir is None:
            return None
        store = self._stores.get(user_id)
        if store is None:
            from .records import PRODUCTION_DIM, RecordStore, store_path_for
            path = store_path_for(user_id, self._records_dir)
            path.parent.mkdir(parents=True, exist_ok=True)
            store = RecordStore(path, dim=PRODUCTION_DIM, session_id=user_id)
            self._stores[user_id] = store
        return store

    def _evict_oldest(self) -> None:
        """Drops the least-recently-used session's in-memory wrappers.

        The Atlas/Trace data is already durable, but a record store owns an mmap HANDLE, so
        it is synced and dropped here rather than left to the garbage collector -- an
        unevicted store dict is a per-user handle leak with no upper bound.
        """
        user_id, _ctx = self._active_sessions.popitem(last=False)
        logger.info(f"Evicting session for user {user_id}")
        self._loops.pop(user_id, None)
        store = self._stores.pop(user_id, None)
        if store is not None:
            try:
                store.sync()
            except Exception:            # a failing sync must not block eviction
                logger.exception("record store sync failed on eviction for %s", user_id)

    async def shutdown(self) -> None:
        """Drops all in-memory session wrappers. The underlying Atlas and
        Trace data is already durable (mmap-backed); there is nothing
        additional to flush here beyond what Atlas.sync() already covers
        at the storage layer."""
        logger.info("Shutting down SessionManager...")
        async with self._lock:
            for user_id, store in self._stores.items():
                try:
                    store.sync()
                except Exception:
                    logger.exception("record store sync failed on shutdown for %s", user_id)
            self._active_sessions.clear()
            self._loops.clear()
            self._stores.clear()
