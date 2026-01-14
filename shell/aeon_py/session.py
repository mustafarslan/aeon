import logging
import asyncio
import re
import time
import shutil
from pathlib import Path
from collections import OrderedDict
from typing import Optional, Dict

from .context import ContextManager
from .loop import CognitiveLoop
from .client import AeonClient
from .llm import LLMProvider

logger = logging.getLogger(__name__)

class SessionManager:
    """
    Manages multi-tenant user sessions with in-memory LRU caching.
    Ensures isolation between users and handles persistence.
    """
    
    def __init__(self, atlas_client: AeonClient, llm_provider: LLMProvider, 
                 storage_dir: str = "./data/traces", max_sessions: int = 100):
        self.atlas = atlas_client
        self.llm = llm_provider
        
        # Thread-safe session storage
        # OrderedDict used as LRU: End = Most Recent, Start = Oldest
        self._active_sessions: OrderedDict[str, ContextManager] = OrderedDict()
        self._loops: Dict[str, CognitiveLoop] = {}
        
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_sessions = max_sessions
        self._lock = asyncio.Lock()
        
        # Pre-compile regex for security validation
        self._user_id_pattern = re.compile(r"^[a-zA-Z0-9_-]+$")

    def _validate_user_id(self, user_id: str) -> None:
        """Security check to prevent directory traversal."""
        if not self._user_id_pattern.match(user_id):
            raise ValueError(f"Invalid user_id: {user_id}. Must be alphanumeric, _, or -.")

    def _get_trace_path(self, user_id: str) -> Path:
        self._validate_user_id(user_id)
        return self.storage_dir / f"{user_id}.json"

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
                await self._evict_oldest()
                
            # 3. Load or Create
            ctx = ContextManager(self.atlas)
            path = self._get_trace_path(user_id)
            
            if path.exists():
                try:
                    ctx.load_session(path)
                    logger.info(f"Loaded session for user {user_id}")
                except Exception as e:
                    logger.error(f"Failed to load session for {user_id}: {e}")
                    # Start fresh if corrupt
            else:
                logger.info(f"Created new session for user {user_id}")
                
            self._active_sessions[user_id] = ctx
            return ctx

    async def get_loop(self, user_id: str) -> CognitiveLoop:
        """Gets or creates the CognitiveLoop for the user."""
        # Ensure context is loaded
        ctx = await self.get_context(user_id)
        
        if user_id not in self._loops:
            self._loops[user_id] = CognitiveLoop(ctx, self.llm)
            
        return self._loops[user_id]

    async def _evict_oldest(self) -> None:
        """Removes the least recently used session and persists it."""
        # popitem(last=False) returns (key, value) from the *beginning* (Oldest)
        user_id, ctx = self._active_sessions.popitem(last=False)
        logger.info(f"Evicting session for user {user_id}")
        await self._save_session(user_id, ctx)
        
        # Cleanup loop
        if user_id in self._loops:
            del self._loops[user_id]

    async def _save_session(self, user_id: str, ctx: ContextManager) -> None:
        """Persists session to disk using atomic write."""
        path = self._get_trace_path(user_id)
        tmp_path = path.with_suffix(".tmp")
        
        try:
            # Check if event loop is running to decide sync/async wrapper?
            # ContextManager.save_session is synchronous (uses json dump)
            # For thread safety in async context, we should ideally run in executor
            # But json dump is fast enough for MVP unless huge.
            
            # Atomic Write Pattern
            ctx.save_session(tmp_path)
            
            # Atomic Move
            tmp_path.replace(path)
            logger.debug(f"Saved session for {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to save session for {user_id}: {e}")
            if tmp_path.exists():
                tmp_path.unlink()

    async def shutdown(self) -> None:
        """Gracefully saves all active sessions."""
        logger.info("Shutting down SessionManager...")
        async with self._lock:
            # Snapshot keys to avoid runtime modification issues during iteration
            users = list(self._active_sessions.keys())
            for user_id in users:
                ctx = self._active_sessions[user_id]
                await self._save_session(user_id, ctx)
            self._active_sessions.clear()
            self._loops.clear()
