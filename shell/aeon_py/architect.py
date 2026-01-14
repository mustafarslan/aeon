from typing import List, Dict, Any
from .client import AeonClient

class Architect:
    """
    Manages the lifecycle of new knowledge in the Aeon Cognitive OS.
    Handles 'Short-Term' Delta admission and future persistence strategies.
    """
    
    def __init__(self, atlas_client: AeonClient):
        self.atlas = atlas_client
        self.pending_docs: List[Dict[str, Any]] = []

    def ingest(self, text: str, vector: List[float]) -> int:
        """
        Adds new knowledge to the "Short-Term" Delta layer.
        Immediate availability for retrieval via Hybrid Search.
        
        Args:
            text: The raw text content
            vector: 768-dimensional embedding vector
            
        Returns:
            int: The temporary node ID (MSB set) from the Delta Layer.
        """
        # Call the new C++ hybrid insert method
        # Note: client.atlas is the raw C++ binding object
        # vector must be a list or numpy array
        new_id = self.atlas.atlas.insert_delta(vector, text)
        
        self.pending_docs.append({
            "id": new_id, 
            "text": text,
            "vector_preview": vector[:3] # Store preview for debug
        })
        
        return new_id

    def persist(self):
        """
        [Phase 7 Stub]
        In the future, this will trigger a background 'Merge' 
        to write the in-memory Delta Buffer to the immutable mmap file.
        """
        pass
