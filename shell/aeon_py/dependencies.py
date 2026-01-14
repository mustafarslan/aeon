from functools import lru_cache
from pathlib import Path
import os
from .client import AeonClient
from .llm import MockProvider, OllamaProvider, LLMProvider
from .session import SessionManager
from fastapi import Header, HTTPException

# Default Paths (can be overridden by Env Vars)
DEFAULT_ATLAS_PATH = os.environ.get("AEON_ATLAS_PATH", "./data/atlas.aeon")
DEFAULT_TRACE_DIR = os.environ.get("AEON_TRACE_DIR", "./data/traces")

@lru_cache()
def get_atlas_client() -> AeonClient:
    """Singleton Atlas Client."""
    path = Path(DEFAULT_ATLAS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    return AeonClient(path)

@lru_cache()
def get_llm_provider() -> LLMProvider:
    """Singleton LLM Provider."""
    if os.environ.get("AEON_USE_OLLAMA", "false").lower() == "true":
        return OllamaProvider()
    return MockProvider()

@lru_cache()
def get_session_manager() -> SessionManager:
    """Singleton Session Manager."""
    atlas = get_atlas_client()
    llm = get_llm_provider()
    return SessionManager(atlas, llm, storage_dir=DEFAULT_TRACE_DIR)

async def get_current_user_id(x_user_id: str = Header(..., alias="X-User-ID")) -> str:
    """Dependency to extract and validate User ID."""
    if not x_user_id:
        raise HTTPException(status_code=400, detail="X-User-ID header is required")
    return x_user_id

async def get_cognition(
    user_id: str = Header(..., alias="X-User-ID"),
):
    """Dependency helper (optional)."""
    mgr = get_session_manager()
    return await mgr.get_loop(user_id)
