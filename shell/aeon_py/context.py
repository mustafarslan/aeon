import numpy as np
from typing import List, Any, Union, Optional
from pathlib import Path
from .client import AeonClient
from .trace import TraceGraph, EdgeType

class ContextManager:
    """
    Orchestrates the interaction between the spatial Atlas (Long-term memory)
    and the episodic Trace (Short-term/Context memory).
    """
    def __init__(self, atlas_client: AeonClient) -> None:
        self.atlas = atlas_client
        self.trace = TraceGraph()

    def process_turn(self, user_query: str, query_vector: Union[List[float], np.ndarray], access_level: str = "public") -> np.ndarray:
        """
        Process a single user interaction turn.
        1. Records user event in Trace.
        2. Queries Atlas for concept associations.
        3. Links top concepts to the user event in Trace.
        4. Filters concepts based on access_level.
        
        Args:
            user_query: Raw text of the user's input.
            query_vector: 768-dim float vector (list or ndarray).
            access_level: Security clearance ("public", "admin", etc).
            
        Returns:
            Structured numpy array of Atlas search results.
        """
        # Ensure vector is in correct format
        if isinstance(query_vector, list):
            q_vec = np.array(query_vector, dtype=np.float32)
        else:
            q_vec = query_vector.astype(np.float32)

        # 1. Trace: Add User Event
        user_node_id = self.trace.add_user_event(user_query, q_vec)
        
        # 2. Atlas: Navigate
        # returns structured array ['id', 'similarity', 'preview']
        results = self.atlas.query(q_vec)
        
        # TODO: Implement metadata-based filtering once Atlas supports returning metadata
        # For now, we simulate filter by ID range or assume all are public
        # allowed_results = [r for r in results if r['level'] <= access_level]
        
        # 3. Trace: Graft Concepts (Top 3)
        # We limit to top 3 to keep graph clean ("High Activation" links only)
        top_k = results[:3]
        for row in top_k:
            # Add concept (idempotent)
            concept_id = self.trace.add_concept(row['id'], row['similarity']) 
            
            # Link CAUSAL: query -> concept
            self.trace.link(user_node_id, concept_id, EdgeType.CAUSAL)
            
        return results

    def add_response(self, text: str) -> str:
        """
        Record the system's textual response to close the turn loop.
        """
        return self.trace.add_system_event(text)
        
    def save_session(self, path: Union[str, Path]) -> None:
        """Persist current trace state."""
        self.trace.save(path)
        
    def load_session(self, path: Union[str, Path]) -> None:
        """Restore trace state and warm the SLB cache."""
        self.trace = TraceGraph.load(path)
        
        # Phase 9: Warm Start the SLB
        recent_ids = self.trace.get_recent_concept_ids(limit=64)
        if recent_ids:
            try:
                # self.atlas is AeonClient, which wraps the C++ Atlas bindings
                # We assume AeonClient exposes load_context or wraps it.
                # Checking client.py is necessary if it's a wrapper.
                # Assuming direct binding access or method on client. 
                # Let's check client.py? 
                # If AeonClient just delegates `query` to `navigate`, 
                # we might need to add `load_context` to `AeonClient` too.
                # For now, I'll assume I need to ADD it to AeonClient in `client.py` 
                # OR access the binding directly.
                # Let's look at `client.py` next.
                # But I will write the call here assuming it exists on self.atlas.
                if hasattr(self.atlas, 'load_context'):
                     self.atlas.load_context(recent_ids)
                elif hasattr(self.atlas, 'atlas') and hasattr(self.atlas.atlas, 'load_context'):
                     # Direct access if it wraps the binding object
                     self.atlas.atlas.load_context(recent_ids)
            except Exception as e:
                # Non-critical failure
                print(f"Warning: Failed to warm SLB: {e}")
