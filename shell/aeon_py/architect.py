from typing import List, Dict, Any, Tuple
import numpy as np
from .client import AeonClient

# V4 Stage 2 task 4: admission-time near-duplicate detection. Deliberately
# much higher than SLB_HIT_THRESHOLD (0.85, schema.hpp) -- that threshold
# means "similar enough to reuse a cached ANSWER", this one means "close
# enough to BE the same fragment". Reuses the existing navigate() path
# (itself built on math_kernel.hpp's cosine_similarity()) rather than a new
# similarity computation.
NEAR_DUPLICATE_THRESHOLD = 0.97


class Architect:
    """
    Manages the lifecycle of new knowledge in the Aeon Cognitive OS.
    Handles 'Short-Term' Delta admission and future persistence strategies.
    """

    def __init__(self, atlas_client: AeonClient):
        self.atlas = atlas_client
        self.pending_docs: List[Dict[str, Any]] = []

    def ingest(self, text: str, vector: List[float]) -> Tuple[int, bool]:
        """
        Adds new knowledge to the "Short-Term" Delta layer -- unless it's a
        near-duplicate of already-admitted content (v4-plan.md Stage 2 task
        4: addendum Q2's growth-pipeline admission check), in which case no
        new row is inserted and the EXISTING node's id is returned instead.
        This also doubles as the first poisoning checkpoint the addendum's
        TrustMem citation calls for: repeated near-identical content can't
        silently flood the index one row at a time.

        Args:
            text: The raw text content
            vector: 768-dimensional embedding vector

        Returns:
            (node_id, is_duplicate). node_id is either the newly-inserted
            delta node's id, or an EXISTING node's id if this was judged a
            near-duplicate. The caller (ContextManager) decides how to
            record the duplicate case in Trace -- a Refines edge (Stage 1's
            edge_type/supersedes_id), not a second copy of the same
            content.
        """
        q_vec = np.asarray(vector, dtype=np.float32)
        existing = self.atlas.query(q_vec)
        if len(existing) > 0 and float(existing[0]["similarity"]) >= NEAR_DUPLICATE_THRESHOLD:
            return int(existing[0]["id"]), True

        # Call the new C++ hybrid insert method
        # Note: client.atlas is the raw C++ binding object
        # vector must be a list or numpy array
        new_id = self.atlas.atlas.insert_delta(vector, text)

        self.pending_docs.append({
            "id": new_id,
            "text": text,
            "vector_preview": vector[:3] # Store preview for debug
        })

        return new_id, False

    def persist(self):
        """
        [Phase 7 Stub]
        In the future, this will trigger a background 'Merge'
        to write the in-memory Delta Buffer to the immutable mmap file.
        """
        pass
