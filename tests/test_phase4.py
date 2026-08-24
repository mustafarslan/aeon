
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
    # Architect.ingest() returns (node_id, is_duplicate) as of v4-plan.md
    # Stage 2 task 4 (admission-time near-duplicate detection) -- a fresh
    # Atlas has nothing to duplicate against, so is_duplicate is False here.
    secret_text = "The Eagle has landed at Midnight"
    new_id, is_duplicate = architect.ingest(secret_text, secret_vector.tolist())
    assert not is_duplicate

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
    Verify that a delta-buffer insert survives a new client instance on the
    same file, via WAL replay -- NOT the reverse. This test originally
    asserted the opposite ("delta buffer is volatile, doesn't survive a new
    instance"), written when delta_buffer_ was assumed to be a purely
    in-process std::vector with no persistence. That assumption predates
    Atlas::insert_delta() writing a WAL_RECORD_ATLAS record for every delta
    insert (core/src/atlas.cpp) specifically for crash-recovery durability
    (see CLAUDE.md's WAL lock-ordering note, which already names
    delta_mutex_ in the same chain as wal_mutex_). Atlas::replay_wal()
    reconstructs WAL_RECORD_ATLAS payloads back into delta_buffer_ (not
    mmap storage), re-assigning a fresh NODE_ID_DELTA_MASK id on replay --
    confirmed by reading replay_wal() directly, not assumed. So a second
    AeonClient opening the same path is expected to see the delta insert:
    that's the durability guarantee working as designed, not a leak.
    """
    atlas_path = str(atlas_env)

    # Session 1
    client1 = AeonClient(atlas_path)
    architect1 = Architect(client1)

    vec = np.ones(768, dtype=np.float32)
    architect1.ingest("Temporary Info", vec.tolist())

    res1 = client1.query(vec)
    assert any(r['similarity'] > 0.99 for r in res1)

    # Session 2 (simulate restart by opening a new client on the same path).
    # WAL replay on open() reconstructs session 1's delta insert into
    # client2's own delta_buffer_ -- this is crash-recovery durability, not
    # cross-instance shared state (client2 never talks to client1's process
    # memory; it only ever reads client1's already-flushed WAL file).
    client2 = AeonClient(atlas_path)
    res2 = client2.query(vec)

    has_high_match = any(r['similarity'] > 0.99 for r in res2)
    assert has_high_match, "WAL-backed delta insert did not survive replay on a new instance"

if __name__ == "__main__":
    test_delta_ingestion_and_search(Path("/tmp/aeon_phase4_check"))
