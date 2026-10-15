from . import core
from .trace import TraceGraph, EdgeType
from .context import ContextManager
from .client import AeonClient
from .llm import LLMProvider, MockProvider, OllamaProvider
from .prompt import PromptEngine
from .loop import CognitiveLoop

# The names imported above. Before, a missing comma joined "CognitiveLoop"
# and "PromptEngine" into one name, two were listed twice, and the server's
# modules were listed though never imported: `from aeon_py import *` then
# pulled in FastAPI, an extra (aeon_py[server]) since the base install was
# slimmed. Import aeon_py.server and the like by name.
__all__ = [
    "core",
    "TraceGraph",
    "ContextManager",
    "EdgeType",
    "AeonClient",
    "LLMProvider",
    "MockProvider",
    "OllamaProvider",
    "PromptEngine",
    "CognitiveLoop",
]
