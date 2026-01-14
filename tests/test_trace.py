import pytest
import numpy as np
import json
import networkx as nx
from pathlib import Path
from aeon_py.trace import TraceGraph, EdgeType, UserNode, SystemNode, ConceptNode
from aeon_py.context import ContextManager
from unittest.mock import MagicMock

class TestTraceGraph:
    def test_node_creation_and_types(self):
        trace = TraceGraph()
        
        # 1. User Event
        vec = np.random.rand(768).astype(np.float32)
        uid = trace.add_user_event("Hello Trace", vec)
        assert uid.startswith("u_")
        assert trace.graph.nodes[uid]['text'] == "Hello Trace"
        assert trace.graph.nodes[uid]['type'] == "UserNode"
        assert trace.graph.nodes[uid]['vector'] == vec.tolist()
        
        # 2. System Event
        sid = trace.add_system_event("Hello Human")
        assert sid.startswith("s_")
        assert trace.graph.nodes[sid]['text'] == "Hello Human"
        
        # 3. Concept Event
        cid = trace.add_concept(101, 0.95)
        assert cid == "c_101"
        assert trace.graph.nodes[cid]['atlas_id'] == 101
        assert trace.graph.nodes[cid]['similarity'] == 0.95

    def test_linking_logic(self):
        trace = TraceGraph()
        
        u1 = trace.add_user_event("q1")
        s1 = trace.add_system_event("a1")
        
        # Check automatic NEXT link
        assert trace.graph.has_edge(u1, s1)
        assert trace.graph.edges[u1, s1]['relation'] == EdgeType.NEXT.value
        
        # Check manual CAUSAL link
        c1 = trace.add_concept(500, 0.88)
        trace.link(u1, c1, EdgeType.CAUSAL)
        
        assert trace.graph.has_edge(u1, c1)
        assert trace.graph.edges[u1, c1]['relation'] == EdgeType.CAUSAL.value

    def test_concept_idempotency(self):
        trace = TraceGraph()
        
        id1 = trace.add_concept(123, 0.5)
        id2 = trace.add_concept(123, 0.6) # Different sim, same ID
        
        assert id1 == id2 == "c_123"
        assert trace.graph.number_of_nodes() == 1
        # Should retain first one or ignore? Implementation says: if not in self.graph: add
        # So it ignores the second one's attributes.
        assert trace.graph.nodes[id1]['similarity'] == 0.5 

    def test_serialization(self, tmp_path):
        trace = TraceGraph()
        vec = np.array([0.1, 0.2], dtype=np.float32)
        
        u1 = trace.add_user_event("persistent", vec)
        s1 = trace.add_system_event("reply")
        
        save_path = tmp_path / "trace.json"
        trace.save(save_path)
        
        assert save_path.exists()
        
        # Load back
        loaded = TraceGraph.load(save_path)
        assert len(loaded.graph) == 2
        assert loaded.graph.has_edge(u1, s1)
        
        # Check node data recovery
        # Note: JSON converts numpy arrays to lists
        loaded_vec = loaded.graph.nodes[u1]['vector']
        assert loaded_vec == pytest.approx([0.1, 0.2], abs=1e-5)
        assert loaded.cursor == s1 # Last added node should be cursor concept? No, save/load logic needs verification.
        # Wait, implementation of load:
        # if 'cursor' in data['graph']: instance.cursor = ...
        # networkx node_link_data puts graph attributes in 'graph' key.
        # Does TraceGraph store cursor in graph attributes?
        # Looking at implementation: It does NOT explicitly put cursor in graph attrs before saving.
        # FIX: The implementation of save() in TraceGraph used `nx.node_link_data(self.graph)`.
        # `self.graph` is a DiGraph. If I didn't set `self.graph.graph['cursor'] = self.cursor`, it won't be saved.
        # I should probably fix that in the implementation if I want it to persist. 
        # But for now, let's see if the test fails or passes (I suspect it fails on cursor).

    def test_viz_json(self):
        trace = TraceGraph()
        u1 = trace.add_user_event("viz me")
        c1 = trace.add_concept(99, 1.0)
        trace.link(u1, c1, EdgeType.CAUSAL)
        
        viz = trace.to_viz_json()
        assert "nodes" in viz
        assert "edges" in viz
        assert len(viz['nodes']) == 2
        assert len(viz['edges']) == 1
        assert viz['nodes'][0]['label'] == "viz me"

class TestContextManager:
    def test_process_turn(self):
        # Mock Atlas
        mock_atlas = MagicMock()
        # Return structured array as per client.py
        # dtype = [('id', 'u8'), ('similarity', 'f4'), ('preview', 'f4', (3,))]
        dt = np.dtype([('id', 'u8'), ('similarity', 'f4'), ('preview', 'f4', (3,))])
        results = np.zeros(5, dtype=dt)
        results['id'] = [1, 2, 3, 4, 5]
        results['similarity'] = [0.9, 0.8, 0.7, 0.6, 0.5]
        mock_atlas.query.return_value = results
        
        ctx = ContextManager(mock_atlas)
        
        q_vec = np.random.rand(768).astype(np.float32)
        ctx.process_turn("Search query", q_vec)
        
        # Check Trace
        # Should have 1 UserNode + 3 ConceptNodes (Top 3)
        assert ctx.trace.graph.number_of_nodes() == 4 
        
        # Find User Node
        user_nodes = [n for n, d in ctx.trace.graph.nodes(data=True) if d['type'] == 'UserNode']
        assert len(user_nodes) == 1
        uid = user_nodes[0]
        
        # Check Edges (3 CAUSAL edges)
        assert ctx.trace.graph.out_degree(uid) == 3
        
        # Check edge types
        for _, neighbor, data in ctx.trace.graph.out_edges(uid, data=True):
            assert data['relation'] == "CAUSAL"
