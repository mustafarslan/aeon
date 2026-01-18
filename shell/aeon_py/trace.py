import networkx as nx
import numpy as np
import json
import uuid
import time
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Dict, List, Union

class EdgeType(str, Enum):
    CAUSAL = "CAUSAL"       # User -> Concept
    NEXT = "NEXT"           # User -> System / System -> User (Time flow)
    REFERS_TO = "REFERS_TO" # System -> Concept

@dataclass
class UserNode:
    text: str
    timestamp: float
    vector: Optional[List[float]] = None
    id: str = field(default_factory=lambda: f"u_{uuid.uuid4().hex[:8]}")
    type: str = "UserNode"

@dataclass
class SystemNode:
    text: str
    timestamp: float
    vector: Optional[List[float]] = None
    id: str = field(default_factory=lambda: f"s_{uuid.uuid4().hex[:8]}")
    type: str = "SystemNode"

@dataclass
class ConceptNode:
    atlas_id: int
    similarity: float
    id: str = field(init=False)
    type: str = "ConceptNode"

    def __post_init__(self):
        # Deterministic ID for idempotency: Concept 123 is always 'c_123'
        self.id = f"c_{self.atlas_id}"

class TraceEncoder(json.JSONEncoder):
    """Custom encoder for Dataclasses and NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class TraceGraph:
    """
    Episodic memory graph (DAG) tracking conversation history.
    Stores User queries, System responses, and Atlas Concepts.
    """
    
    __slots__ = ['graph', 'cursor']

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self.cursor: Optional[str] = None # ID of the active/latest node

    def add_user_event(self, text: str, vector: Union[List[float], np.ndarray, None] = None) -> str:
        """
        Records a user query.
        Links to previous node with NEXT if cursor exists.
        """
        # Convert numpy to list for storage if needed
        vec_list = None
        if vector is not None:
             vec_list = vector.tolist() if isinstance(vector, np.ndarray) else vector

        node = UserNode(
            text=text,
            timestamp=time.time(),
            vector=vec_list
        )
        
        # Add to graph
        self.graph.add_node(node.id, **asdict(node))
        
        # Link temporal flow
        if self.cursor:
            self.link(self.cursor, node.id, EdgeType.NEXT)
            
        self.cursor = node.id
        return node.id

    def add_system_event(self, text: str, vector: Union[List[float], np.ndarray, None] = None) -> str:
        """
        Records a system response.
        Links to previous node (UserNode) with NEXT.
        """
        vec_list = None
        if vector is not None:
             vec_list = vector.tolist() if isinstance(vector, np.ndarray) else vector

        node = SystemNode(
            text=text,
            timestamp=time.time(),
            vector=vec_list
        )
        
        self.graph.add_node(node.id, **asdict(node))
        
        if self.cursor:
            self.link(self.cursor, node.id, EdgeType.NEXT)
            
        self.cursor = node.id
        return node.id

    def add_concept(self, atlas_id: int, similarity: float) -> str:
        """
        Adds a Concept node. Idempotent.
        Does NOT update cursor (Concept is a leaf/side reference).
        """
        node = ConceptNode(atlas_id=atlas_id, similarity=float(similarity))
        
        if node.id not in self.graph:
            self.graph.add_node(node.id, **asdict(node))
        
        # We assume concepts are immutable singleton-like entities in the graph
        # We don't update attributes if it exists.
        
        return node.id

    def link(self, source_id: str, target_id: str, relation: EdgeType) -> None:
        """Create a typed edge between nodes."""
        if source_id in self.graph and target_id in self.graph:
            self.graph.add_edge(source_id, target_id, relation=relation.value)

    def prune(self, max_nodes: int = 10000) -> None:
        """
        Simple safety valve to prevent unbounded growth.
        Removes oldest nodes if excessive.
        """
        if self.graph.number_of_nodes() > max_nodes:
            # This is a naive implementation. 
            # In a real system, we'd archive old chains. 
            # flexible warning or logic here.
            pass

    def prune_tail(self, n_events: int) -> int:
        """
        Removes the last N 'spine' events (User/System nodes) from the graph active tail.
        Used for rollback synchronization with the LookaheadBuffer.
        
        Args:
             n_events: Number of User/System steps to backtrack.
             
        Returns:
             Number of spine nodes actually removed.
        """
        if not self.cursor or n_events <= 0:
            return 0
            
        removed_count = 0
        current_id = self.cursor
        
        # Traverse backwards n_events steps
        for _ in range(n_events):
            if not current_id:
                break
                
            # Identify predecessor in the spine (EdgeType.NEXT points TO current)
            preds = list(self.graph.predecessors(current_id))
            prev_node = None
            for p in preds:
                edge_data = self.graph.get_edge_data(p, current_id)
                if edge_data.get('relation') == EdgeType.NEXT.value:
                    prev_node = p
                    break
            
            # Identify any side-chains (Concepts) linked strictly to this node?
            # If we remove a UserNode, we might leave orphaned Concept nodes if they have no other parents.
            # But ConceptNodes are shared/idempotent. We shouldn't delete them unless we strictly track refcounts.
            # For this implementation, we leave Concepts (they are harmless knowledge references).
            # We ONLY remove the conversation spine node.
            
            self.graph.remove_node(current_id)
            removed_count += 1
            current_id = prev_node
            
        self.cursor = current_id
        return removed_count

    def save(self, path: Union[str, Path]) -> None:
        """Serialize graph to JSON."""
        self.graph.graph['cursor'] = self.cursor
        data = nx.node_link_data(self.graph)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'w') as f:
            json.dump(data, f, cls=TraceEncoder, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'TraceGraph':
        """Load graph from JSON."""
        p = Path(path)
        if not p.exists():
            return cls()
            
        with open(p, 'r') as f:
            data = json.load(f)
            
        instance = cls()
        instance.graph = nx.node_link_graph(data)
        
        # Restore cursor: find the last added NEXT node? 
        # Or just manually set if we tracked it in metadata.
        # For now, let's look for a node with out_degree=0 in NEXT edges?
        # Or simplistic: last node in list? No order guarantee.
        # Better: Save cursor in graph graph-level attributes.
        
        # If the loaded graph has 'graph' attributes (nx supports this)
        if 'cursor' in data.get('graph', {}): # type: ignore
             instance.cursor = data['graph']['cursor'] # type: ignore
        else:
             # Fallback: find a node with no outgoing NEXT edges? Too complex for now.
             pass
             
        return instance

    def to_viz_json(self) -> Dict[str, Any]:
        """
        Returns simplified JSON for Frontend Visualization (React Flow / D3).
        """
        nodes = []
        for n_id, attrs in self.graph.nodes(data=True):
            # Extract label based on type
            label = "Unknown"
            if attrs.get('type') == "UserNode":
                label = attrs.get('text', '')[:30]
            elif attrs.get('type') == "SystemNode":
                label = attrs.get('text', '')[:30]
            elif attrs.get('type') == "ConceptNode":
                label = f"Room {attrs.get('atlas_id')}"
                
            nodes.append({
                "id": n_id,
                "label": label,
                "type": attrs.get('type'),
                "details": attrs # include full details for tooltip
            })
            
        edges = []
        for u, v, attrs in self.graph.edges(data=True):
            edges.append({
                "source": u,
                "target": v,
                "type": attrs.get('relation')
            })
            
        return {"nodes": nodes, "edges": edges}

    def get_recent_concept_ids(self, limit: int = 64) -> List[int]:
        """
        Retrieves IDs of recently accessed Atlas concepts.
        Traverses the conversation spine backwards from the cursor.
        """
        if not self.cursor or self.cursor not in self.graph:
            return []

        concept_ids = []
        visited_concepts = set()
        
        # Start at cursor
        current_id = self.cursor
        
        # Safety counter
        iterations = 0
        max_iterations = 1000

        while current_id and len(concept_ids) < limit and iterations < max_iterations:
            iterations += 1
            
            # 1. Get concepts linked from current node
            if current_id in self.graph:
                for neighbor in self.graph.successors(current_id):
                    edge_data = self.graph.get_edge_data(current_id, neighbor)
                    relation = edge_data.get('relation')
                    
                    if relation in [EdgeType.CAUSAL.value, EdgeType.REFERS_TO.value]:
                        node_attrs = self.graph.nodes[neighbor]
                        if node_attrs.get('type') == "ConceptNode":
                            c_id = node_attrs.get('atlas_id')
                            if c_id is not None and c_id not in visited_concepts:
                                visited_concepts.add(c_id)
                                concept_ids.append(c_id)
            
            # 2. Move to previous spine node (NEXT edge points TO current, so we look for predecessor)
            preds = list(self.graph.predecessors(current_id))
            prev_node = None
            for p in preds:
                edge_data = self.graph.get_edge_data(p, current_id)
                if edge_data.get('relation') == EdgeType.NEXT.value:
                    prev_node = p
                    break
            
            current_id = prev_node
            
        return concept_ids
