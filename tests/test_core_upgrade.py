import pytest
import numpy as np
import shutil
from pathlib import Path
from aeon_py.client import AeonClient

@pytest.fixture
def temp_atlas(tmp_path):
    atlas_path = tmp_path / "test.atlas"
    client = AeonClient(atlas_path)
    return client

def test_get_children(temp_atlas):
    # 1. Create Root (ID 0)
    root_vec = np.zeros(768, dtype=np.float32)
    root_id = temp_atlas.atlas.insert(0, root_vec, "Root")
    
    # 2. Add Children to Root
    child1_vec = np.zeros(768, dtype=np.float32)
    child1_vec[0] = 1.0
    id1 = temp_atlas.atlas.insert(root_id, child1_vec, "Child 1")
    
    child2_vec = np.zeros(768, dtype=np.float32)
    child2_vec[1] = 1.0
    id2 = temp_atlas.atlas.insert(root_id, child2_vec, "Child 2")
    
    # 3. Retrieve Children
    children = temp_atlas.get_children(root_id)
    
    # 4. Verify
    assert len(children) == 2
    ids = sorted([row['id'] for row in children])
    assert ids == [id1, id2]
    
    # Verify centroids (previews)
    # Child 1: [1.0, 0.0, 0.0]
    # Child 2: [0.0, 1.0, 0.0]
    
    # Find child 1 in results
    c1 = next(r for r in children if r['id'] == id1)
    assert np.allclose(c1['preview'], [1.0, 0.0, 0.0])
    
    c2 = next(r for r in children if r['id'] == id2)
    assert np.allclose(c2['preview'], [0.0, 1.0, 0.0])

def test_get_children_empty(temp_atlas):
    # Root has no children
    root_vec = np.zeros(768, dtype=np.float32)
    root_id = temp_atlas.atlas.insert(0, root_vec, "Root")
    
    children = temp_atlas.get_children(root_id)
    assert len(children) == 0

def test_get_children_invalid_id(temp_atlas):
    children = temp_atlas.get_children(99999)
    assert len(children) == 0
