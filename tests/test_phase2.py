import pytest
import numpy as np
import psutil
import os
import gc
from pathlib import Path
from aeon_py.client import AeonClient

def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def test_memory_pressure(tmp_path):
    """
    Verify Zero-Copy by querying Atlas 10,000 times.
    Memory usage should remain relatively flat (no leaks from copying).
    """
    # 1. Setup Atlas
    atlas_path = tmp_path / "test.atlas"
    client = AeonClient(str(atlas_path))
    
    # Create a dummy node to query against
    # Use direct core access or rely on empty atlas behavior handling 
    # (Our navigate returns empty list if empty, so it's safe)
    # But to test properly we need data. 
    # Let's insert one node via the raw core object if possible, or just query empty.
    # Querying empty path is fast but still exercises the binding alloc/dealloc logic?
    # No, if path is empty, we alloc 0 bytes. We need a path.
    
    # Let's insert via client.atlas (raw access)
    vector = np.random.rand(768).astype(np.float32)
    meta = "test_node"
    
    # We need to use the C++ binding's insert.
    # client.atlas is the core.Atlas object.
    # core.Atlas.insert(parent, vector, metadata)
    
    # Root
    client.atlas.insert(0, vector, meta)
    # Child 1
    client.atlas.insert(0, vector, "child1") 
    
    # 2. Warmup
    query_vec = np.random.rand(768).astype(np.float32)
    for _ in range(100):
        client.query(query_vec)
        
    gc.collect()
    start_mem = get_process_memory()
    print(f"Start Memory: {start_mem:.2f} MB")
    
    # 3. Stress Loop
    params = np.array(query_vec) # keep same query
    
    # Run 10k iterations
    for i in range(10000):
        res = client.query(params)
        # Touch the data to ensure it was accessed
        _ = res['similarity']
        
    gc.collect()
    end_mem = get_process_memory()
    print(f"End Memory: {end_mem:.2f} MB")
    
    growth = end_mem - start_mem
    # We allow some small growth due to python internal fragmentation or jit
    # But it shouldn't be huge (like 10k * 768 * 4 bytes = 30MB copy)
    # If we were copying dicts, it would be HUGE (overhead of dicts).
    
    assert growth < 50.0, f"Memory grew by {growth:.2f} MB, likely leaking or copying excessively."
    
if __name__ == "__main__":
    test_memory_pressure(Path("/tmp/aeon_test"))
