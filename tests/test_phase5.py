import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from aeon_py.llm import MockProvider
from aeon_py.prompt import PromptEngine
from aeon_py.loop import CognitiveLoop
from aeon_py.context import ContextManager
from aeon_py.trace import TraceGraph

class TestPhase5:
    
    def test_prompt_engine_formatting(self):
        engine = PromptEngine()
        
        # Mock Data
        history = [
             {'type': 'UserNode', 'text': 'Who is Aeon?', 'timestamp': 100},
             {'type': 'SystemNode', 'text': 'I am the shell.', 'timestamp': 101},
             {'type': 'UserNode', 'text': 'What is your core?', 'timestamp': 102} 
        ]
        
        # Remember: PromptEngine Logic might slice the last user node?
        # Let's check implementation. 
        # _format_history: `lines[:-1] if lines and lines[-1]['type'] == 'UserNode'`
        # So "What is your core?" should be in USER INPUT section, not HISTORY section.
        
        context = {'metadata': 'Test Room'}
        knowledge = [{'id': 1, 'content': 'Aeon Core is Written in C++'}]
        
        prompt = engine.build(history, context, knowledge)
        
        print("\nGenerated Prompt:\n", prompt)
        
        assert "Test Room" in prompt
        assert "Aeon Core is Written in C++" in prompt
        assert "User: Who is Aeon?" in prompt
        assert "Aeon: I am the shell." in prompt
        # The last user input in history should NOT be in Conversation History section
        assert "User: What is your core?" not in prompt 
        # But should be in User Input section
        assert "### USER INPUT\nWhat is your core?" in prompt

    def test_mock_provider_streaming(self):
        provider = MockProvider()
        stream = provider.generate("hello")
        
        parts = list(stream)
        text = "".join(parts)
        assert "[Mock Response]" in text
        assert "hello" in text

    @patch('aeon_py.loop.CognitiveLoop._vectorize')
    def test_cognitive_loop_flow(self, mock_vectorize):
        # Setup Mocks
        mock_vectorize.return_value = np.zeros(768, dtype=np.float32)

        mock_atlas = MagicMock()
        mock_atlas.query.return_value = np.array(
            [(1, 0.9, (0,0,0))],
            dtype=[('id', 'u8'), ('similarity', 'f4'), ('preview', 'f4', (3,))]
        )
        # Architect.ingest() (called by ContextManager.process_turn() step 4)
        # goes through mock_atlas.atlas.insert_delta() -- give it a concrete
        # int return so it round-trips through the REAL TraceGraph.add_event()
        # below (a bare MagicMock isn't a valid atlas_id for the C++ binding).
        mock_atlas.atlas.insert_delta.return_value = 42

        # Real, in-memory (no path) TraceGraph -- v4-plan.md Stage 2's shared-
        # trace design made `trace` a required ContextManager constructor arg
        # (this test previously called ContextManager(mock_atlas) with only
        # one argument, predating that change). Using a real TraceGraph
        # rather than a MagicMock here so the assertions below check genuine
        # C++-backed behavior instead of trivially passing against mock
        # objects.
        trace = TraceGraph()
        ctx = ContextManager(mock_atlas, trace)
        llm = MockProvider()

        loop = CognitiveLoop(ctx, llm)

        # Verify Fallback/Mock behavior of encoder implicitly (or mocked)
        # We mocked _vectorize directly to avoid loading weights in test

        # Chat
        response = []
        for token in loop.chat("Open the pod bay doors"):
            response.append(token)

        full_resp = "".join(response)

        # Assertions
        assert "[Mock Response]" in full_resp
        assert "Open the pod bay doors" in full_resp  # Mock echoes input

        # Verify Trace was updated: process_turn() records 1 user event, 1
        # concept event (top_k=3, but the mock only returns 1 result), and 1
        # admission event (Architect.ingest() -- Ingested, since the mocked
        # 0.9 similarity is below NEAR_DUPLICATE_THRESHOLD's 0.97); then
        # add_response() records 1 system event. 4 total.
        assert trace.size == 4

        # Check the most recent event is the system response.
        # get_history() returns reverse-chronological order. Text is
        # truncated to a max preview length in C++ (add_event()'s doc
        # comment), so compare as a prefix, not an exact match.
        last_event = trace.get_history("default", limit=10)[0]
        assert last_event['role'] == TraceGraph.ROLE_SYSTEM
        assert full_resp.startswith(last_event['text'])

    def test_real_embedding_loading(self):
        # Only run if sentence-transformers is installed (it should be)
        try:
            import sentence_transformers
        except ImportError:
            pytest.skip("sentence-transformers not installed")

        # We want to test that _get_encoder loads the model (on first call)
        # But we don't want to actually download heavy weights if not present.
        # So we might mock SentenceTransformer constructor in `aeon_py.loop`.
        
        with patch('sentence_transformers.SentenceTransformer') as MockST:
            mock_model = MagicMock()
            # v4-plan.md Stage 2 task 3 replaced the old 384-dim MiniLM +
            # zero-padding-to-768 hack with all-mpnet-base-v2 (native
            # 768-dim, no padding) -- see loop.py's _vectorize(). Simulate
            # the CURRENT model's real output shape, not the removed one.
            mock_model.encode.return_value = np.ones(768, dtype=np.float32)
            MockST.return_value = mock_model

            ctx = MagicMock()
            loop = CognitiveLoop(ctx, MockProvider())

            # This should trigger lazy load
            vec = loop._vectorize("test")

            assert MockST.called
            assert vec.shape == (768,)
            assert vec[0] == 1.0
