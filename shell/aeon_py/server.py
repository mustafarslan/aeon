import asyncio
import anyio
from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from typing import Generator, Any

from .dependencies import (
    get_session_manager,
    get_current_user_id,
    get_atlas_client,
    AeonClient,
    SessionManager
)
from .loop import CognitiveLoop
from .context import ContextManager
from .models import (
    ChatRequest, 
    TraceResponse, 
    ActiveRoomResponse, 
    NeighborInfo,
    VectorQueryRequest,
    SearchResult
)

app = FastAPI(title="Aeon Cognitive OS Server", version="0.1.0")

# CORS
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_event():
    """Ensure all sessions are flushed to disk."""
    mgr = get_session_manager()
    await mgr.shutdown()

@app.get("/health")
async def health_check():
    return {"status": "ok", "component": "AeonServer"}

# --- Chat Endpoint (Streaming) ---

async def make_async_generator(generator: Generator[str, None, None]):
    """
    Wraps a blocking synchronous generator into an async generator
    by running next() calls in a separate thread.
    """
    iterator = iter(generator)
    while True:
        try:
            # Run the blocking next() in a threadpool to avoid freezing the event loop
            token = await anyio.to_thread.run_sync(next, iterator)
            if token:
                yield {"event": "token", "data": token}
        except StopIteration:
            break
        except Exception as e:
            yield {"event": "error", "data": str(e)}
            break
            
    yield {"event": "done", "data": "[DONE]"}

@app.post("/chat")
async def chat_endpoint(
    request: ChatRequest,
    user_id: str = Depends(get_current_user_id),
    mgr: SessionManager = Depends(get_session_manager)
):
    """
    Streams the LLM response for a specific user session.
    Input: {"text": "Hello"}
    Output: SSE Stream of tokens.
    """
    loop = await mgr.get_loop(user_id)
    
    # Create the synchronous generator
    sync_gen = loop.chat(request.text)
    
    # Wrap in async generator for non-blocking stream
    return EventSourceResponse(make_async_generator(sync_gen))


# --- Observability Endpoints (The Glass Box) ---

@app.get("/state/trace", response_model=TraceResponse)
async def get_trace_state(
    user_id: str = Depends(get_current_user_id),
    mgr: SessionManager = Depends(get_session_manager)
):
    """
    Returns the full episodic trace graph for the requesting user.
    """
    ctx = await mgr.get_context(user_id)
    
    # to_viz_json returns dict with 'nodes' and 'edges' lists
    data = ctx.trace.to_viz_json()
    return TraceResponse(**data)

@app.get("/state/atlas/active", response_model=ActiveRoomResponse)
async def get_active_room(
    user_id: str = Depends(get_current_user_id),
    mgr: SessionManager = Depends(get_session_manager),
    atlas: AeonClient = Depends(get_atlas_client)
):
    """
    Returns the "Active Room" for the user.
    """
    ctx = await mgr.get_context(user_id)
    
    # Logic: Find the last "ConceptNode" in the Trace
    active_atlas_id = 0 # Root default
    
    try:
        nodes = list(ctx.trace.graph.nodes(data=True))
        concept_nodes = [
            d for n, d in nodes 
            if d.get('type') == 'ConceptNode'
        ]
        
        if concept_nodes:
            last = concept_nodes[-1]
            active_atlas_id = last.get('atlas_id', 0)
            
    except Exception:
        pass
        
    # 2. Get Children (Neighbors)
    children_raw = atlas.get_children(active_atlas_id)
    
    neighbors = []
    for row in children_raw:
        neighbors.append(NeighborInfo(
            id=row['id'],
            similarity=0.0
        ))
        
    path = []
    
    return ActiveRoomResponse(
        room_id=active_atlas_id,
        name=f"Room {active_atlas_id}",
        path=path,
        neighbors=neighbors
    )

@app.post("/state/atlas/query", response_model=list[SearchResult])
async def debug_atlas_query(
    request: VectorQueryRequest,
    atlas: AeonClient = Depends(get_atlas_client)
):
    """
    Debug: Raw vector search against Atlas (Global/System Level).
    """
    import numpy as np
    
    vec = np.array(request.vector, dtype=np.float32)
    results = atlas.query(vec)
    
    output = []
    for row in results:
        output.append(SearchResult(
            id=row['id'],
            similarity=row['similarity'],
            preview=row['preview'].tolist()
        ))
    return output
