"""Aeon Core C++23 High-Performance Backend"""

from collections.abc import Sequence
import os
from typing import Annotated, Final, overload

import numpy
from numpy.typing import NDArray


class EpochGuard:
    def __init__(self, /, *args, **kwargs):
        """Initialize self.  See help(type(self)) for accurate signature."""

    def __enter__(self) -> object: ...

    def __exit__(self, *args) -> None: ...

    def release(self) -> None:
        """Explicitly release the epoch guard (idempotent)"""

    def is_active(self) -> bool:
        """Check if the guard is still protecting memory"""

def version() -> str:
    """Get the library version"""

class BuildInfo:
    def __init__(self, /, *args, **kwargs):
        """Initialize self.  See help(type(self)) for accurate signature."""

    @property
    def compiler(self) -> str: ...

    @property
    def architecture(self) -> str: ...

    @property
    def simd_level(self) -> str: ...

    @property
    def standard(self) -> str: ...

    @property
    def repr(self) -> str: ...

def get_build_info() -> BuildInfo:
    """Get build environment details"""

def get_result_node_size() -> int:
    """Return size of ResultNode struct for schema validation"""

class Atlas:
    def __init__(self, path: str | os.PathLike, dim: int = 0, quantization_type: int = 0, metadata_size: int = 0) -> None:
        """
        dim/quantization_type/metadata_size are new-file-only (0 = default); an existing file's on-disk values are always authoritative. metadata_size (v4-plan.md Stage 4 task 6 Phase B) lets a store be opened with a larger metadata field, e.g. for the shared Atlas store to absorb encrypted-payload overhead.
        """

    def size(self) -> int: ...

    @property
    def metadata_size(self) -> int:
        """
        Metadata field size (bytes) of this Atlas instance -- callers writing an encoded payload into the metadata field must length-check against this BEFORE calling insert(), since insert() truncates silently rather than raising on overflow.
        """

    def insert(self, parent_id: int, vector: Sequence[float], metadata: str, session_id: str | None = None) -> int: ...

    def insert_delta(self, vector: Sequence[float], metadata: str) -> int:
        """Insert into in-memory Delta Buffer (immediate availability)"""

    def prune_delta_tail(self, n: int) -> int:
        """Remove last N nodes from delta buffer (for rollback)"""

    def navigate_raw(self, query: Sequence[float], beam_width: int = 1, apply_csls: bool = False, session_id: str | None = None, scope_mask: int | None = 18446744073709551615) -> Annotated[NDArray[numpy.uint8], dict(shape=(None,), writable=False)]:
        """
        Beam search navigate. beam_width=1 is greedy. apply_csls=True applies hub penalty. session_id scopes the SLB cache lookup to the caller's session (v4-plan.md Stage 0) -- omit for the shared default session. scope_mask (v4-plan.md Stage 2) filters results to nodes matching scope_mask & node.scope_bitmap != 0; omit (ALL_SCOPES_VISIBLE) for unfiltered pre-Stage-2 behavior.
        """

    def drop_session(self, session_id: str) -> bool:
        """
        Remove a session's SLB L1 cache entry and free its memory (prevents unbounded growth across many short-lived sessions)
        """

    def sync(self) -> None:
        """
        Explicitly flush pending mmap writes to disk (see Atlas::insert doc comment for the durability model this closes the gap on)
        """

    def set_node_scope(self, node_id: int, scope_bitmap: int) -> None:
        """
        Sets a node's scope_bitmap in place (v4-plan.md Stage 1/2). Mmap nodes only -- raises for a delta-arena id or if compaction is in progress.
        """

    def get_node_scope(self, node_id: int) -> int:
        """Reads a node's current scope_bitmap."""

    def supersede_node(self, node_id: int) -> None:
        """
        Reversibly excludes a node from beam search results (v4-plan.md Stage 2), branchless like tombstoning but reversible via revoke_node_supersede(). Mmap nodes only.
        """

    def revoke_node_supersede(self, node_id: int) -> None:
        """Reverses a prior supersede_node() call. Mmap nodes only."""

    def is_node_superseded(self, node_id: int) -> bool:
        """Reads whether a node currently has NODE_FLAG_SUPERSEDED set."""

    def set_node_governance_id(self, node_id: int, governance_record_id: int) -> None:
        """
        Sets a node's governance_record_id in place (v4-plan.md Stage 4) -- an opaque link into the control plane. Mmap nodes only.
        """

    def get_node_governance_id(self, node_id: int) -> int:
        """Reads a node's current governance_record_id."""

    def get_node_metadata(self, node_id: int) -> str:
        """
        Reads a node's metadata string back out (v4-plan.md Stage 4 task 2, promotion). Works for both mmap and delta-arena ids.
        """

    def get_node_centroid(self, node_id: int) -> list[float]:
        """
        Reads a node's full centroid vector back out, dequantized to FP32 if this Atlas is INT8-quantized (v4-plan.md Stage 4 task 2, promotion). Works for both mmap and delta-arena ids.
        """

    def list_nodes_by_scope(self, scope_mask: int) -> list[int]:
        """
        Lists live (non-tombstoned) node ids whose scope_bitmap overlaps scope_mask (v4-plan.md Stage 4 console primitive). Superseded nodes ARE included; tombstoned nodes are not.
        """

    def bulk_set_node_scope(self, updates: Sequence[tuple[int, int]]) -> None:
        """
        Applies many (node_id, scope_bitmap) updates under a single lock/WAL-flush pass (v4-plan.md Stage 4 bulk bit remap). All-or-nothing: every entry is validated before any node is mutated.
        """

    def tombstone_node(self, node_id: int) -> None:
        """
        Logically deletes a single mmap node by id (v4-plan.md Stage 4 console/erasure-workflow primitive). WAL-protected, idempotent, TERMINAL (no revoke) -- see Atlas::tombstone_node's doc comment for the physical-vs-logical deletion distinction the erasure workflow must account for.
        """

    def get_children_raw(self, parent_id: int, scope_mask: int = 18446744073709551615) -> Annotated[NDArray[numpy.uint8], dict(shape=(None,), writable=False)]:
        """
        Returns byte array of child nodes (view as structured in Python). scope_mask (v4-plan.md Stage 2 task 2): the Atlas->Trace->Atlas graph-expansion-boundary enforcement point -- see Atlas::get_children()'s doc comment (atlas.hpp).
        """

    def load_context(self, node_ids: Sequence[int], session_id: str | None = None) -> None:
        """
        Pre-fill SLB cache with node IDs for warm start, scoped to session_id (v4-plan.md Stage 0)
        """

    def consolidate_subgraph(self, old_node_ids: Sequence[int], summary_vector: Sequence[float], summary_metadata: str) -> int:
        """
        Atomically: insert summary → re-wire children → tombstone old nodes. Returns the new summary node ID.
        """

    def compact_mmap(self) -> None:
        """
        Shadow compaction: defragment Atlas file with generational naming (stutter-free, no path needed).
        """

    def tombstone_count(self) -> int:
        """Returns count of tombstoned (dead) nodes for compaction triggers."""

    def acquire_read_guard(self) -> EpochGuard:
        """Acquire EBR read guard for safe zero-copy memory access"""

class HierarchicalSLB:
    def __init__(self, dim: int = 768) -> None:
        """
        Create a session-aware cache for the given embedding dimension (must match the owning Atlas's dim)
        """

    @property
    def dim(self) -> int: ...

    def find_nearest(self, session_id: int, query: Sequence[float], threshold: float = 0.8500000238418579) -> object:
        """
        Hierarchical L1/L2 lookup: session cache then global cache. Returns a dict with node_id/similarity/centroid_preview, or None on a cache miss.
        """

    def insert(self, session_id: int, node_id: int, centroid: Sequence[float]) -> None:
        """Insert into session L1 cache and global L2 cache"""

    def drop_session(self, session_id: int) -> bool:
        """Remove session and free its L1 cache (prevents OOM leaks)"""

    def active_session_count(self) -> int:
        """Count of active sessions across all shards (diagnostic)"""

    shard_count: Final[int] = ...
    """Number of lock-striped shards (64)"""

class TraceManager:
    @overload
    def __init__(self) -> None:
        """Create in-memory-only trace manager"""

    @overload
    def __init__(self, path: str | os.PathLike) -> None:
        """Create or open mmap-backed trace file"""

    def size(self) -> int:
        """Total event count (mmap + delta)"""

    def mmap_event_count(self) -> int:
        """Event count in mmap file"""

    def delta_event_count(self) -> int:
        """Event count in delta buffer"""

    def append_event(self, session_id: str, role: int, text: str, atlas_id: int = 0, embedding: Sequence[float] = [], edge_type: int = 0, supersedes_id: int = 0, reason_code: int = 0, event_time: int = 0) -> int:
        """
        Append an episodic event. Returns the new event ID. embedding (v4-plan.md Stage 2 task 3): optional, empty by default = not embedded (excluded from semantic_search()). The FIRST non-empty embedding ever appended to this trace file fixes its embedding_dim; a later mismatched size raises ValueError. edge_type/supersedes_id/reason_code (v4-plan.md Stage 1/2 task 4): EdgeType/ReasonCode enum values (see schema.hpp) for a version/admission edge this event carries -- 0/None (default) means no edge. supersedes_id is in practice always a store-encoded Atlas node id, not a TraceEvent id -- see schema.hpp's TraceEvent doc comment (V4 STAGE 4 note).
        """

    def semantic_search(self, query: Sequence[float], top_k: int = 10) -> list:
        """
        Semantic search over embedded trace events (v4-plan.md Stage 2 task 3), via TraceBlockIndex's two-phase O(|V|/1024 + K*1024) search. Only events appended with a non-empty embedding are indexed. Returns an empty list if no embedding has ever been appended to this file, or if query's length doesn't match embedding_dim.
        """

    @property
    def embedding_dim(self) -> int:
        """
        Dimensionality of indexed embeddings, or 0 if none have been appended to this trace file yet.
        """

    def get_history(self, session_id: str, limit: int = 100) -> list:
        """Retrieve session history (newest first). Returns list of dicts."""

    def compact(self) -> None:
        """Shadow compaction: defragment trace file."""

    def has_session(self, session_id: str) -> bool:
        """Check if a session has any events"""

    def drop_session(self, session_id: str) -> bool:
        """Drop session tail pointer (session cleanup)"""
