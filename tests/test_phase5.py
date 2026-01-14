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
        
        ctx = ContextManager(mock_atlas)
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
        
        # Verify Trace was updated
        # 1 User Node + 1 System Node + 1 Concept (from mock items)
        # ContextManager logic: 1 user, 1 concept (top k=3, we returned 1), 1 system response.
        assert ctx.trace.graph.number_of_nodes() >= 3
        
        # Check last node is system response
        last_id = ctx.trace.cursor
        assert ctx.trace.graph.nodes[last_id]['type'] == 'SystemNode'
        assert ctx.trace.graph.nodes[last_id]['text'] == full_resp

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
            mock_model.encode.return_value = np.ones(384, dtype=np.float32) # Simulating MiniLM
            MockST.return_value = mock_model
            
            ctx = MagicMock()
            loop = CognitiveLoop(ctx, MockProvider())
            
            # This should trigger lazy load
            vec = loop._vectorize("test")
            
            assert MockST.called
            # Should have padded 384 -> 768
            assert vec.shape == (768,)
            assert vec[0] == 1.0
            assert vec[384] == 0.0 # Zero padding start
