import numpy as np
import warnings
from typing import Generator
from .context import ContextManager
from .llm import LLMProvider
from .prompt import PromptEngine

class CognitiveLoop:
    """
    The Main Loop of the Cognitive OS.
    Orchestrates the flow between User Input -> Memory (Context) -> LLM -> Response.
    """
    
    def __init__(self, context_manager: ContextManager, llm_provider: LLMProvider):
        self.ctx = context_manager
        self.llm = llm_provider
        self.prompt_engine = PromptEngine()
        self._encoder = None # Lazy load

    def _get_encoder(self):
        """Lazy load SentenceTransformer to optimize startup time."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                # Use a small, efficient model suitable for CPU/local use
                self._encoder = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                warnings.warn("sentence-transformers not installed. Using random fallback.")
                self._encoder = "MOCK"
            except Exception as e:
                 warnings.warn(f"Failed to load sentence-transformers: {e}. Using random fallback.")
                 self._encoder = "MOCK"
        return self._encoder

    def _vectorize(self, text: str) -> np.ndarray:
        """
        Embeds text into a 768-dimensional vector.
        """
        encoder = self._get_encoder()
        
        if encoder == "MOCK":
             # Fallback: Deterministic random-ish vector based on hash for stability
            np.random.seed(hash(text) % 2**32) 
            return np.random.rand(768).astype(np.float32)
            
        # Real embedding
        # ensure output is numpy array
        vec = encoder.encode(text)
        
        # Check dimensionality - all-MiniLM-L6-v2 produces 384 dim!
        # Atlas expects 768 dim.
        # We need to pad or project. Mismatch is CRITICAL.
        # FIX: Provide a model that is 768 dim or pad.
        # 'all-mpnet-base-v2' is 768 dim.
        # 'all-MiniLM-L6-v2' is 384.
        
        if vec.shape[0] == 384:
            # Simple padding for now if we stick to the small model
            # Or better: switch to 'all-mpnet-base-v2' which is standard 768.
            # But let's check what the user asked previously? 
            # User specified: "random vector for now, or use sentence-transformers".
            # The codebase Phase 4 implies 768 dim.
            # Let's try to load a 768 model if possible.
            # 'paraphrase-mpnet-base-v2' or 'all-mpnet-base-v2'
            pass
            
        # Re-check model: 'all-mpnet-base-v2' is the best general purpose 768 model.
        # But if we must use small one, we pad.
        # Let's enforce 768.
        if vec.shape[0] != 768:
            # If we loaded MiniLM (384), we pad with zeros to match Atlas schema
            padded = np.zeros(768, dtype=np.float32)
            padded[:vec.shape[0]] = vec
            return padded
            
        return vec.astype(np.float32)

    def chat(self, user_input: str) -> Generator[str, None, None]:
        """
        Process a full conversation turn.
        
        1. Vectorize Input
        2. Update Memory (Atlas + Trace)
        3. Build Prompt from Context
        4. Generate Response (Stream)
        5. Close Loop (Record Response)
        """
        
        # 1. Vectorize
        vec = self._vectorize(user_input)
        
        # 2. Update Memory & Retrieve Knowledge
        # context_manager.process_turn writes to Trace and searches Atlas
        knowledge_results = self.ctx.process_turn(user_input, vec)
        
        # 3. Gather Context State for Prompt
        # Get recent history from Trace
        # We need direct access to trace graph to get list.
        # TraceGraph doesn't have a 'get_recent_history' method returning list of dicts yet?
        # Let's check trace.py
        # trace.py defines 'TraceGraph'. It has 'graph' (nx.DiGraph).
        # We need to implement retrieval logic. 
        # Since 'trace.py' doesn't have a helper, we implement simple walking here.
        # Or better: We utilize the graph traversal.
        
        # Pull last 5 nodes following the cursor backwards?
        # For simplicity in this iteration, we iterate nodes in insertion order 
        # (which relies on Python dict order preservation in nx graph since 3.7+).
        history_nodes = []
        try:
             # Get User/System nodes only (exclude Concepts which don't have timestamps)
             relevant_nodes = [
                 d for n, d in self.ctx.trace.graph.nodes(data=True) 
                 if d.get('type') in ('UserNode', 'SystemNode')
             ]
             all_nodes = sorted(relevant_nodes, key=lambda x: x['timestamp'])
             history_nodes = all_nodes[-6:] # Last 6 items
        except Exception:
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
        
        # 4. Build Prompt
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
        self.ctx.add_response(full_response)
