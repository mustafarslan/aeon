import numpy as np
import warnings
from typing import Generator
from .context import ContextManager
from .llm import LLMProvider
from .prompt import PromptEngine

def _question_block(user_input: str) -> str:
    """Render the question with today's date, matching the benchmark harness's format.

    The date line is not decoration: this project measured the reference date as worth ~19
    questions when it was absent, and relative anchors ("last Saturday", "a week ago") are
    unresolvable without it.
    """
    from datetime import datetime
    now = datetime.now()
    return (f"Today's date is {now.strftime('%Y/%m/%d')} ({now.strftime('%a')}).\n"
            f"Question: {user_input}")


class CognitiveLoop:
    """
    The Main Loop of the Cognitive OS.
    Orchestrates the flow between User Input -> Memory (Context) -> LLM -> Response.
    """
    
    def __init__(self, context_manager: ContextManager, llm_provider: LLMProvider,
                 record_store=None):
        self.ctx = context_manager
        self.llm = llm_provider
        self.prompt_engine = PromptEngine()
        self._encoder = None # Lazy load
        # When present, the semantic layer becomes the authoritative context builder and
        # PromptEngine is bypassed entirely -- the two produce mutually exclusive prompt
        # formats (compose() emits its own headers and answer tail; PromptEngine emits a
        # four-section template), so running both would give the model two competing layouts.
        # None keeps the pre-existing path exactly, which is what every current test asserts.
        self.record_store = record_store

    # 'all-mpnet-base-v2' is a native 768-dim model, matching Atlas's
    # EMBEDDING_DIM_DEFAULT (schema.hpp) exactly -- no padding/projection
    # needed. The previous 'all-MiniLM-L6-v2' (384-dim) required zero-padding
    # to fit Atlas's schema, which produces a real embedding, then discards
    # half its information content by construction; this was flagged as an
    # unresolved bug in this method's own comments rather than fixed.
    _ENCODER_MODEL_NAME = "all-mpnet-base-v2"

    def _get_encoder(self):
        """The process-wide encoder (v4.1). This used to build a SentenceTransformer PER
        LOOP, i.e. per user -- at max_sessions=100 that is ~100 copies of a ~420 MB model.
        `dependencies.get_encoder()` is an lru_cache singleton; the local attribute is kept
        only as a per-instance memo of it."""
        if self._encoder is None:
            try:
                from .dependencies import get_encoder
                self._encoder = get_encoder()
            except ImportError:
                warnings.warn(
                    "sentence-transformers not installed -- semantic memory "
                    "is NON-FUNCTIONAL: falling back to hash-seeded random "
                    "vectors with no semantic meaning. Install with "
                    "`pip install sentence-transformers` (already a "
                    "declared dependency in pyproject.toml) before relying "
                    "on Atlas retrieval quality."
                )
                self._encoder = "MOCK"
            except Exception as e:
                warnings.warn(
                    f"Failed to load sentence-transformers model "
                    f"'{self._ENCODER_MODEL_NAME}': {e}. Semantic memory is "
                    f"NON-FUNCTIONAL: falling back to hash-seeded random "
                    f"vectors with no semantic meaning."
                )
                self._encoder = "MOCK"
        return self._encoder

    def _vectorize(self, text: str) -> np.ndarray:
        """
        Embeds text into a 768-dimensional vector via
        sentence-transformers/all-mpnet-base-v2 (native 768-dim, matching
        Atlas's EMBEDDING_DIM_DEFAULT -- no padding/projection needed).
        """
        encoder = self._get_encoder()

        if encoder == "MOCK":
            # Fallback: deterministic random-ish vector based on hash, for
            # dev environments without the real dependency installed.
            # Retrieval over these vectors is NOT semantically meaningful --
            # see _get_encoder()'s warning.
            np.random.seed(hash(text) % 2**32)
            return np.random.rand(768).astype(np.float32)

        vec = encoder.encode(text)
        assert vec.shape[0] == 768, (
            f"Encoder '{self._ENCODER_MODEL_NAME}' produced a "
            f"{vec.shape[0]}-dim vector, expected 768 -- this indicates the "
            f"model itself changed dimensionality, not a normal runtime "
            f"condition to silently work around."
        )
        return vec.astype(np.float32)

    def chat(self, user_input: str, session_id: str = None,
             event_time: int = 0) -> Generator[str, None, None]:
        """
        Process a full conversation turn.

        1. Vectorize Input
        2. Update Memory (Atlas + Trace)
        3. Build Prompt from Context
        4. Generate Response (Stream)
        5. Close Loop (Record Response)

        Args:
            user_input: The user's message text.
            session_id: Caller's authenticated identity (the verified
                user_id from server.py), scoping the Atlas SLB cache lookup
                to this session (v4-plan.md Stage 0).
        """

        # 1. Vectorize
        vec = self._vectorize(user_input)

        # 2. Update Memory & Retrieve Knowledge
        # context_manager.process_turn writes to Trace and searches Atlas
        knowledge_results = self.ctx.process_turn(user_input, vec, session_id=session_id,
                                                  event_time=event_time)
        
        # 3. Gather Context State for Prompt
        # Get recent history from the shared Trace, scoped to this session.
        # get_history() returns newest-first; PromptEngine expects
        # chronological (oldest-first, with the just-recorded current user
        # turn as the LAST item) -- see PromptEngine._format_history.
        _ROLE_TO_TYPE = {0: "UserNode", 1: "SystemNode"}  # exclude concept/summary
        history_nodes = []
        try:
            sid = session_id or "default"
            # Bug found and fixed while building Stage 7 (v4-plan.md): each
            # process_turn() call above appends up to 5 Trace events per
            # turn (1 user + up to 3 Atlas-concept + 1 ingestion-concept),
            # of which only the single "user" event passes this filter --
            # add_response() below adds one more "system" event per turn
            # that does. So a full turn contributes up to 6 raw events but
            # only 2 that pass. The previous `limit=12` therefore could
            # only ever see ~2 turns' worth of raw events -- capping
            # `history_nodes` at ~4 passing entries, never the 6 the
            # `[-6:]` slice below intends (3 turns), and shrinking further
            # if a future turn adds more per-turn bookkeeping events.
            # Silently starves the prompt of conversational history as a
            # session grows. Fixed by requesting generously more raw
            # events than the worst-case per-turn filter ratio requires --
            # a single get_history() call is a cheap prev_id-chain walk
            # (O(limit), not O(session length)), so this costs no extra
            # round trip and stays well within Aeon's latency budget.
            raw_history = self.ctx.trace.get_history(sid, limit=48)
            for ev in reversed(raw_history):  # oldest first
                node_type = _ROLE_TO_TYPE.get(ev.get("role"))
                if node_type is None:
                    continue
                history_nodes.append({"type": node_type, "text": ev.get("text", "")})
            history_nodes = history_nodes[-6:]  # Last 6 items
        except Exception as e:
            warnings.warn(f"Failed to fetch trace history: {e}")
            history_nodes = []
            
        active_room = {"metadata": "General Context"} # Atlas metadata not fully implemented in mock
        # If knowledge_results (structured array) has valid data, we could map it.
        
        # Convert structured array results to list of dicts for PromptEngine
        # results dtype = [('id', 'u8'), ('similarity', 'f4'), ('preview', 'f4', (3,))]
        # We don't have text content in Atlas results yet (just vector preview).
        # So PromptEngine will show previews.
        knowledge_list = []
        for row in knowledge_results:
            knowledge_list.append({
                "id": row['id'],
                "similarity": row['similarity'],
                "content": f"[Result ID {row['id']} Sim {row['similarity']:.2f}]"
            })
        
        # 4. Build Prompt -- semantic layer if wired, PromptEngine otherwise.
        if self.record_store is not None:
            from .compose import compose_from_store
            # The question block carries the reference date with the question, which this
            # project measured as worth ~19 questions when it was missing.
            question_block = _question_block(user_input)
            final_prompt = compose_from_store(
                self.record_store, self.ctx.trace, question_block)["prompt"]
        else:
            final_prompt = self.prompt_engine.build(history_nodes, active_room, knowledge_list)
        
        # 5. Generate Response
        full_response = ""
        
        # We also pass system prompt if we had one.
        system_instr = self.prompt_engine.build_system_prompt() if hasattr(self.prompt_engine, 'build_system_prompt') else "You are Aeon."
        
        response_stream = self.llm.generate(final_prompt, system_prompt=system_instr)
        
        for token in response_stream:
            full_response += token
            yield token
            
        # 6. Record Response (Close Loop)
        self.ctx.add_response(full_response, session_id=session_id, event_time=event_time)
