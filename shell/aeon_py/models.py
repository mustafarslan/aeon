from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class ChatRequest(BaseModel):
    text: str

# --- Trace / Graph Visualization Models ---

class TraceNode(BaseModel):
    id: str
    label: str
    type: str
    timestamp: float 
    details: Dict[str, Any]

class TraceEdge(BaseModel):
    source: str
    target: str
    type: str

class TraceResponse(BaseModel):
    nodes: List[TraceNode]
    edges: List[TraceEdge]

# --- Atlas / Room Visualization Models ---

class NeighborInfo(BaseModel):
    id: int
    similarity: float
    # We could add 'preview' vector or truncated text if available
    
class ActiveRoomResponse(BaseModel):
    room_id: int
    name: str # e.g. "Room 123" or metadata if available
    path: List[NeighborInfo] 
    neighbors: List[NeighborInfo] 

# --- Debug Models ---

class VectorQueryRequest(BaseModel):
    vector: List[float]

class SearchResult(BaseModel):
    id: int
    similarity: float
    preview: List[float]
