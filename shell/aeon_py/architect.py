from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from .client import AeonClient

# V4 Stage 2 task 4: admission-time near-duplicate detection. Deliberately
# much higher than SLB_HIT_THRESHOLD (0.85, schema.hpp) -- that threshold
# means "similar enough to reuse a cached ANSWER", this one means "close
# enough to BE the same fragment". Reuses the existing navigate() path
# (itself built on math_kernel.hpp's cosine_similarity()) rather than a new
# similarity computation.
NEAR_DUPLICATE_THRESHOLD = 0.97


@dataclass(frozen=True)
class SituatingContext:
    """Where a fragment came from, prepended before embedding (v4-plan.md Stage 4 task 4).

    A bug-fix note reading "reverted the retry, the deadlock went away" embeds almost
    identically whether it came from the auth service or the billing pipeline, so a shared
    tier that mixes both retrieves the wrong one. Prepending the situating fields moves that
    distinction into the vector, where retrieval can act on it, instead of leaving it in the
    reader's head.

    Every field is optional. An instance with no fields set renders empty and is treated
    exactly as no context at all, so a caller that has only a subsystem is not forced to
    invent a task id.
    """

    task_id: Optional[str] = None
    subsystem: Optional[str] = None
    error_class: Optional[str] = None

    def render(self) -> str:
        """`[task=AE-14 subsystem=auth error=DeadlockDetected] ` or empty.

        Rendered as a bracketed prefix rather than merged into the sentence: the fragment's
        own text stays byte-identical inside the stored string, so provenance display and
        any later substring matching against the original still work.
        """
        parts = [f"{k}={v}" for k, v in (("task", self.task_id),
                                         ("subsystem", self.subsystem),
                                         ("error", self.error_class)) if v]
        return f"[{' '.join(parts)}] " if parts else ""

    def __bool__(self) -> bool:
        return bool(self.render())


class Architect:
    """
    Manages the lifecycle of new knowledge in the Aeon Cognitive OS.
    Handles 'Short-Term' Delta admission and future persistence strategies.
    """

    def __init__(self, atlas_client: AeonClient):
        self.atlas = atlas_client
        self.pending_docs: List[Dict[str, Any]] = []

    def ingest(self, text: str, vector: List[float],
               context: Optional[SituatingContext] = None,
               embed: Optional[Callable[[str], List[float]]] = None
               ) -> Tuple[int, bool]:
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
            vector: 768-dimensional embedding vector, computed from `text` as given.
                When `context` is supplied this vector is DISCARDED and recomputed
                from the contextualised string, since the two must agree.
            context: Optional `SituatingContext` prepended before embedding (v4-plan.md
                Stage 4 task 4). Defaults to None, which reproduces the previous
                behaviour byte for byte.
            embed: Required when `context` is given; maps the contextualised text to a
                vector. Injected rather than imported so this module keeps no encoder
                dependency.

        Note on the near-duplicate check: it runs on the CONTEXTUALISED vector when
        context is supplied. Two identical fragments from different subsystems are then
        correctly no longer near-duplicates, which is the point of the feature. The
        corollary is that a corpus ingested partly with and partly without context is
        comparing across two different spaces, so a deployment should adopt context for
        a given store consistently rather than per call.

        Returns:
            (node_id, is_duplicate). node_id is either the newly-inserted
            delta node's id, or an EXISTING node's id if this was judged a
            near-duplicate. The caller (ContextManager) decides how to
            record the duplicate case in Trace -- a Refines edge (Stage 1's
            edge_type/supersedes_id), not a second copy of the same
            content.
        """
        # CONTEXTUALISE BEFORE EMBEDDING, OR NOT AT ALL. The caller already computed
        # `vector` from the bare `text`, so prepending context to the stored string without
        # re-embedding would leave the text and its vector describing different things --
        # retrieval would rank on one and display the other. `embed` is therefore required
        # whenever `context` is supplied, and injected rather than imported so this stays
        # testable without an encoder, matching `extract_session(generate=...)`.
        if context:
            if embed is None:
                raise ValueError(
                    "ingest(context=...) needs `embed` as well: the supplied vector was "
                    "computed from the bare text, and storing contextualised text beside "
                    "it would make the record and its embedding disagree."
                )
            text = context.render() + text
            vector = embed(text)

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
        """Not implemented, and it says so rather than returning silently.

        This was a `pass`-only body whose docstring promised to "trigger a
        background Merge to write the in-memory Delta Buffer to the immutable
        mmap file". A caller who read that and called this got silence and no
        durability, which is the same silent-success shape that
        `aeon_atlas_drop_session()` and `aeon_atlas_release_pages()` carried
        before they were connected.

        Delta persistence is real, but it belongs to the kernel and is reached
        through `Atlas.compact_mmap()`, which promotes frozen delta nodes into
        the mmap generation. There is no separate Python-side merge to perform,
        so this raises rather than pretending to do one.
        """
        raise NotImplementedError(
            "Architect.persist() is not implemented. Delta nodes are promoted "
            "to the mmap generation by Atlas.compact_mmap(); call that instead."
        )
