import numpy as np
from typing import List, Any, Union, Optional
from .architect import Architect
from .client import ALL_SCOPES_VISIBLE, AeonClient, decode_store_id, encode_store_id
from .trace import EdgeType, TraceGraph

import logging

logger = logging.getLogger(__name__)

# TraceEvent role used for "this event represents a retrieved Atlas concept,
# adjacent in the episodic log to the user query that surfaced it" -- see
# process_turn()'s docstring for why this replaces an explicit graph edge.
_CONCEPT_ROLE = "concept"


class ContextManager:
    """
    Orchestrates the interaction between the spatial Atlas (Long-term memory)
    and the episodic Trace (Short-term/Context memory).

    `trace` is a SHARED TraceGraph instance (one mmap file for all users,
    like AeonClient/Atlas), not private per-ContextManager state -- events
    are isolated by `session_id`, not by file. See
    dependencies.get_trace_manager()'s docstring for why (v4-plan.md).

    `shared_atlas_client` (v4-plan.md Stage 4) is a SEPARATE, physically
    distinct Atlas store for org-wide shared knowledge -- None (the
    default) if this deployment has no shared tier configured. Every
    TraceEvent.atlas_id this class writes is store-discriminated via
    encode_store_id() (client.py) -- see process_turn()/warm_start() --
    since physical separation means a bare node id no longer says which
    store it names.
    """

    def __init__(self, atlas_client: AeonClient, trace: TraceGraph,
                 shared_atlas_client: Optional[AeonClient] = None,
                 dirty_queue=None) -> None:
        self.atlas = atlas_client
        self.trace = trace
        self.shared_atlas = shared_atlas_client
        # Architect.ingest() was previously fully unwired -- nothing in
        # shell/aeon_py called it, so the delta (short-term) admission path
        # never ran. Wired in process_turn() below: each turn's own query
        # becomes a new, immediately-searchable delta concept, symmetric
        # with how retrieved concepts are already recorded into Trace.
        # Architect only ever writes to the PRIVATE store -- Stage 4's
        # promotion pipeline (not built yet, see v4-plan.md) is the only
        # sanctioned path content reaches the shared tier through; nothing
        # is ever admitted there directly.
        self.architect = Architect(atlas_client)
        # The write-side hook for background consolidation. Optional: a deployment without
        # the semantic layer passes None and this class behaves exactly as before.
        self._dirty_queue = dirty_queue

    def process_turn(
        self,
        user_query: str,
        query_vector: Union[List[float], np.ndarray],
        access_level: str = "public",
        session_id: Optional[str] = None,
    ) -> np.ndarray:
        """
        Process a single user interaction turn.
        1. Records user event in Trace, embedded with query_vector so it's
           findable via TraceGraph.semantic_search() (v4-plan.md Stage 2
           task 3) -- reuses the vector already computed for step 2, no
           extra embedding-model call.
        2. Queries Atlas for concept associations.
        3. Records the top matching concepts in Trace, adjacent to the user
           event (episodic adjacency is the "link" -- see note below).
        4. Admits the query itself as a new delta concept (Architect.ingest())
           so it's immediately searchable by future turns -- unless it's a
           near-duplicate of already-admitted content (Stage 2 task 4), in
           which case a Refines edge is recorded instead of a redundant row.

        access_level is accepted but not yet enforced -- metadata-based
        filtering needs Atlas to return metadata with results, which it
        doesn't yet; this parameter is a placeholder for that, not a live
        security boundary today.

        Args:
            user_query: Raw text of the user's input.
            query_vector: 768-dim float vector (list or ndarray).
            access_level: Security clearance ("public", "admin", etc).
            session_id: Caller's authenticated identity (pass the verified
                user_id from server.py's get_current_user_id). Required to
                actually isolate one user's trace history from another's --
                falls back to a shared "default" session if omitted, which
                is safe only for genuinely single-tenant use.

        Returns:
            Structured numpy array of Atlas search results.

        On the "link" redesign: the current C++-backed TraceGraph is a flat,
        per-session chronological event log (TraceEvent.prev_id chains
        within a session), not an arbitrary graph -- there is no `link()`
        edge-creation primitive to call. Each event already carries a
        single `atlas_id`, so a retrieved concept's relationship to the
        query that surfaced it is expressed as: record a "concept" event
        (atlas_id=<the Atlas node>), which chains immediately after the
        "user" event in the same session's history via prev_id. That
        chronological adjacency IS the causal link in this data model.
        A richer typed-edge model (supersedes/refines/contradicts/...) is
        exactly what v4-plan.md Stage 1/2 add on top of this.
        """
        # Ensure vector is in correct format
        if isinstance(query_vector, list):
            q_vec = np.array(query_vector, dtype=np.float32)
        else:
            q_vec = query_vector.astype(np.float32)

        sid = session_id or "default"

        # 1. Trace: Add User Event, embedded with the SAME vector already
        # computed for the Atlas query below (v4-plan.md Stage 2 task 3) --
        # no extra embedding-model call, and it's what makes this event
        # (and the Architect.ingest() admission a few lines down) findable
        # via TraceGraph.semantic_search().
        self.trace.add_event(sid, "user", user_query, embedding=q_vec.tolist())

        # 2. Atlas: Navigate
        # returns structured array ['id', 'similarity', 'preview']
        results = self.atlas.query(q_vec, session_id=session_id)

        # TODO: Implement metadata-based filtering once Atlas supports returning metadata
        # For now, we simulate filter by ID range or assume all are public
        # allowed_results = [r for r in results if r['level'] <= access_level]

        # 3. Trace: Record top concepts, chained after the user event
        # We limit to top 3 to keep the session history readable ("High
        # Activation" results only).
        top_k = results[:3]
        for row in top_k:
            # This turn's own Atlas query only ever touches the PRIVATE
            # store (process_turn() doesn't route to the shared tier --
            # see ContextManager.query_stores() for that capability,
            # v4-plan.md Stage 4), so is_shared=False always here.
            self.trace.add_event(
                sid,
                _CONCEPT_ROLE,
                f"[Concept {int(row['id'])}] similarity={float(row['similarity']):.3f}",
                atlas_id=encode_store_id(int(row['id']), is_shared=False),
            )

        # 4. Architect: admit this turn's own query as a new delta concept --
        # unless it's a near-duplicate of already-admitted content
        # (v4-plan.md Stage 2 task 4), in which case Architect returns the
        # EXISTING node's id instead of inserting a redundant one, and the
        # Trace event records that relationship as a Refines edge (Stage
        # 1's edge_type/supersedes_id) rather than a second copy of the
        # same content. Either way this is immediately searchable by
        # future navigate() calls (delta-buffer scan) without waiting for
        # a compaction to promote it -- the "short-term memory" Architect
        # was built for but that nothing called until Stage 2 task 3.
        # Architect.ingest() always writes to the PRIVATE store (see
        # __init__'s doc comment) -- is_shared=False always here too.
        node_id, is_duplicate = self.architect.ingest(user_query, q_vec.tolist())
        encoded_id = encode_store_id(node_id, is_shared=False)
        if is_duplicate:
            self.trace.add_event(
                sid,
                _CONCEPT_ROLE,
                f"[Refines {node_id}] {user_query[:80]}",
                atlas_id=encoded_id,
                edge_type=EdgeType.REFINES,
                supersedes_id=encoded_id,
            )
        else:
            self.trace.add_event(
                sid,
                _CONCEPT_ROLE,
                f"[Ingested {node_id}] {user_query[:80]}",
                atlas_id=encoded_id,
            )

        return results

    def query_stores(
        self,
        query_vector: Union[List[float], np.ndarray],
        mode: str = "private",
        session_id: Optional[str] = None,
        top_k: int = 10,
        shared_scope_mask: int = ALL_SCOPES_VISIBLE,
    ) -> List[dict]:
        """
        Query the private and/or shared Atlas stores (v4-plan.md Stage 4),
        returning a unified list of dicts with store-discriminated ids
        (encode_store_id(), client.py) rather than a numpy structured
        array -- merging two independent stores' raw node ids is exactly
        the ambiguity that encoding exists to prevent (private node 5 and
        shared node 5 are different nodes; deduplicating by raw id would
        silently conflate them).

        This is a separate capability from process_turn(), which
        deliberately keeps querying the private store only -- rewiring
        process_turn()'s own routing is out of scope for this increment
        (v4-plan.md Stage 4 task 1).

        Args:
            mode: "private" (default) queries only self.atlas. "shared"
                queries only self.shared_atlas, raising RuntimeError if
                none is configured (an explicit ask for something that
                doesn't exist should fail loudly, not silently degrade).
                "merged" queries both and combines results -- if no shared
                store is configured, "merged" degrades to private-only
                rather than erroring, since an absent shared tier is a
                valid, common deployment state, not a caller mistake.
            top_k: Cap on returned results, applied AFTER merging so a
                "merged" query doesn't return up to 2x a "private" query's
                count.
            shared_scope_mask: V4 Stage 4 task 2 -- forwarded as-is to the
                shared Atlas's query() (Stage 2's existing
                `(node.scope_bitmap & scope_mask) != 0` enforcement, see
                client.py's query() docstring). Ignored for "private" mode
                (the private store has no team-scoping concept; a caller's
                own data is unconditionally theirs). Defaults to
                ALL_SCOPES_VISIBLE (no filtering) for callers that haven't
                been updated to pass a caller's team scope yet -- promotion
                (task 2) is what starts writing real, scope-bearing content
                into the shared tier, so this parameter has no effect until
                a caller actually threads a real mask through.

        Returns:
            List of {"id": <store-discriminated int>, "similarity": float,
            "store": "private"|"shared"} dicts, sorted by similarity
            descending, deduplicated by store-discriminated id (mirrors
            TieredClient._merge_results()'s existing local/cloud merge
            logic in client.py, generalized to be store-id-aware).
        """
        if isinstance(query_vector, list):
            q_vec = np.array(query_vector, dtype=np.float32)
        else:
            q_vec = query_vector.astype(np.float32)

        if mode not in ("private", "shared", "merged"):
            raise ValueError(f"query_stores: unknown mode {mode!r}")

        rows: List[dict] = []
        if mode in ("private", "merged"):
            for r in self.atlas.query(q_vec, session_id=session_id):
                rows.append({
                    "id": encode_store_id(int(r["id"]), is_shared=False),
                    "similarity": float(r["similarity"]),
                    "store": "private",
                })

        if mode in ("shared", "merged"):
            if self.shared_atlas is None:
                if mode == "shared":
                    raise RuntimeError(
                        "query_stores: mode='shared' requested but no "
                        "shared Atlas store is configured for this "
                        "deployment (AEON_SHARED_ATLAS_PATH unset)"
                    )
                # mode == "merged" with no shared store: degrade to
                # private-only, see docstring.
            else:
                for r in self.shared_atlas.query(
                    q_vec, session_id=session_id, scope_mask=shared_scope_mask
                ):
                    rows.append({
                        "id": encode_store_id(int(r["id"]), is_shared=True),
                        "similarity": float(r["similarity"]),
                        "store": "shared",
                    })

        best: dict[int, dict] = {}
        for row in rows:
            existing = best.get(row["id"])
            if existing is None or row["similarity"] > existing["similarity"]:
                best[row["id"]] = row

        merged = sorted(best.values(), key=lambda r: -r["similarity"])
        return merged[:top_k]

    def _mark_dirty(self, session_id: Optional[str]) -> None:
        """Enqueue this session for background consolidation. O(1), no I/O.

        Marked on `add_response`, not `process_turn`: a session becomes worth extracting once
        the turn is COMPLETE (user question plus assistant answer), which is what
        `extract_session` expects to read. Marking on the user turn alone would enqueue a
        half-written conversation.

        The queue is a set, so a session written to ten times before the worker wakes costs
        one consolidation.
        """
        if self._dirty_queue is None:
            return
        try:
            self._dirty_queue.mark_dirty(session_id or "default")
        except Exception:
            # Enqueueing is background enrichment. It must never fail a live turn.
            logger.exception("failed to mark session dirty")

    def add_response(self, text: str, session_id: Optional[str] = None) -> int:
        """
        Record the system's textual response to close the turn loop.

        Returns the new event's ID.
        """
        event_id = self.trace.add_event(session_id or "default", "system", text)
        self._mark_dirty(session_id)
        return event_id

    def recall_episodic(
        self, query_vector: Union[List[float], np.ndarray], unit: str = "window_5",
        generate_fn=None,
    ) -> list[dict]:
        """
        Cross-session semantic episodic recall (v4-plan.md Stage 7) --
        the capability LongMemEval's Stage 6 benchmark measures via
        `TraceGraph.semantic_search()` directly, now callable from real
        code instead of only a benchmark harness.

        NOT wired into `CognitiveLoop.chat()` or any other live call site
        yet, and intentionally so: Stage 7 task 1's expansion-unit
        experiment (`scripts/longmemeval/expansion_unit_experiment.py`,
        not yet run as of this method's addition) is what determines
        which `unit` actually closes the most oracle-vs-real gap at the
        lowest retrieved-token cost -- picking a default here before that
        runs would be exactly the kind of assumption this project has
        corrected repeatedly during Stage 6. This method exists so that
        once task 1's winner is known, wiring it into a real call site is
        a config choice, not new code to write under time pressure.

        Args:
            query_vector: The already-computed query embedding (same one
                a caller would pass to `process_turn()` -- no extra
                embedding-model call).
            unit: One of "full_session", "window_3", "window_5",
                "window_10", "summary" -- see
                `shell/aeon_py/session_expansion.py` for what each means.
                Default "window_5" is a placeholder for "some non-trivial
                unit is available," NOT Stage 7's recommended default --
                do not treat this default as the task 1 decision; that
                decision does not exist until the experiment has run.
            generate_fn: Required only for `unit="summary"` -- a callable
                `(prompt: str) -> str`, e.g. a thin wrapper around
                whatever `LLMProvider` this deployment already holds.
                `session_expansion.py` takes this as an injected
                dependency rather than importing a provider directly, so
                this method does too, for the same reason.

        Returns:
            Chronological (oldest-first) list of event dicts for
            `unit != "summary"`; a single summary string is NOT returned
            in this shape -- callers using `unit="summary"` get back
            `[{"role": 3, "text": <summary>, "session_id": ...}]` (role 3
            = `TraceGraph.ROLE_SUMMARY`) so the return type is uniform
            across all five units regardless of which one is selected.
            Empty list if nothing is indexed yet (no prior `add_event()`
            call in this trace supplied an embedding).
        """
        from .session_expansion import build_expanded_context

        # Bug found running Stage 7 task 1's first live experiment
        # (v4-plan.md): anchoring to a single `find_top_hit()` session and
        # REPLACING the retrieved context with that session's expansion
        # collapsed accuracy on every question type needing more than one
        # gold session (knowledge-update always needs 2, multi-session
        # 3-4) -- to the point of scoring below plain top_k retrieval.
        # `build_expanded_context()` anchors to the top-N distinct
        # sessions among a top_k=30 semantic_search and merges each
        # session's expansion ADDITIVELY onto those hits, so this method
        # can no longer retrieve less than plain top_k would.
        q_vec = query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector)
        return build_expanded_context(self.trace, q_vec, unit, generate_fn=generate_fn)

    def warm_start(self, session_id: str, limit: int = 64) -> None:
        """
        Pre-fill the Atlas SLB cache with this session's recently-relevant
        concepts, so the first query after resuming a conversation isn't a
        cold cache. Call when a session with existing history becomes
        active again (see SessionManager.get_context()).

        Replaces the old load_session()'s warm-start step, which read from
        a private per-user snapshot file that no longer exists in this
        shared-trace design -- history is already in the shared trace, so
        there's nothing to "load," only to warm the cache from.
        """
        if not self.trace.has_session(session_id):
            return
        try:
            history = self.trace.get_history(session_id, limit=limit)
            # atlas_id is store-discriminated (encode_store_id(), v4-plan.md
            # Stage 4) -- decode and route each id to the store it actually
            # came from, rather than sending every id to self.atlas
            # regardless of origin.
            private_ids: List[int] = []
            shared_ids: List[int] = []
            for ev in history:
                if ev.get("role") != 2 or not ev.get("atlas_id", 0):  # ROLE_CONCEPT
                    continue
                raw_id, is_shared = decode_store_id(int(ev["atlas_id"]))
                (shared_ids if is_shared else private_ids).append(raw_id)

            if private_ids:
                self.atlas.load_context(private_ids, session_id=session_id)
            if shared_ids and self.shared_atlas is not None:
                self.shared_atlas.load_context(shared_ids, session_id=session_id)
        except Exception as e:
            # Non-critical failure -- a cold cache is a latency cost, not a
            # correctness issue.
            print(f"Warning: Failed to warm SLB for session {session_id}: {e}")
