
import pytest
import numpy as np
import shutil
from pathlib import Path
from aeon_py.client import AeonClient
from aeon_py.architect import Architect

@pytest.fixture
def atlas_env(tmp_path):
    atlas_path = tmp_path / "test.atlas"
    return atlas_path

def test_delta_ingestion_and_search(atlas_env):
    """
    End-to-End verification of Phase 4:
    1. Start with fresh Atlas
    2. Zero results for "Secret Password"
    3. Architect.ingest("Secret Password")
    4. Query should now find it (Hybrid Search)
    """
    client = AeonClient(str(atlas_env))
    architect = Architect(client)
    
    # 1. New Atlas should be empty (or near empty if we consider root)
    # Let's create a vector for "Secret Password"
    # (In real life we'd embed, but here random stable vector)
    np.random.seed(42)
    secret_vector = np.random.rand(768).astype(np.float32)
    
    # 2. Query - Expect no close match (or just root)
    # Since Atlas is empty, navigate returns empty or just root.
    initial_results = client.query(secret_vector)
    # It might return root if it exists, but similarity is likely low
    print("\nInitial Results:", initial_results)

    # 3. Ingest via Architect
    secret_text = "The Eagle has landed at Midnight"
    new_id = architect.ingest(secret_text, secret_vector.tolist())
    
    print(f"Ingested ID: {new_id} (Hex: {new_id:X})")
    
    # Verify ID format (MSB set)
    assert (new_id & 0x8000000000000000) != 0, "New ID must have MSB set (Delta Flag)"
    
    # 4. Hybrid Search
    # Query with the SAME vector
    results = client.query(secret_vector)
    print("Post-Ingest Results:", results)
    
    found = False
    for res in results:
        # Check if ID matches
        if res['id'] == new_id:
            found = True
            # Expect high similarity (near 1.0 for exact same vector)
            assert res['similarity'] > 0.99, f"Similarity should be ~1.0, got {res['similarity']}"
            break
            
    assert found, "Did not find the ingested 'Secret Password' in search results!"

def test_delta_isolation(atlas_env):
    """
    Verify that delta node doesn't persist if we reload (since persist() is stub).
    (Phase 4 is in-memory only)
    """
    atlas_path = str(atlas_env)
    
    # Session 1
    client1 = AeonClient(atlas_path)
    architect1 = Architect(client1)
    
    vec = np.ones(768, dtype=np.float32)
    architect1.ingest("Temporary Info", vec.tolist())
    
    res1 = client1.query(vec)
    assert any(r['similarity'] > 0.99 for r in res1)
    
    # Session 2 (Simulate restart by new client)
    # Since delta_buffer is in C++ memory of the *instance*, and we don't persist...
    # Wait, the C++ object `core.Atlas` is bound to the FILE.
    # But `delta_buffer_` is a `std::vector` in the C++ CLASS INSTANCE.
    # So if we maintain the same python object `client1`, it persists.
    # If we create `client2`, it loads FROM FILE. The file was NOT written to.
    
    client2 = AeonClient(atlas_path)
    # Delta buffer in client2 should be empty
    res2 = client2.query(vec)
    
    # Should NOT find the high-result match
    has_high_match = any(r['similarity'] > 0.99 for r in res2)
    assert not has_high_match, "Delta buffer leaked to new instance! It should be volatile."

if __name__ == "__main__":
    test_delta_ingestion_and_search(Path("/tmp/aeon_phase4_check"))
