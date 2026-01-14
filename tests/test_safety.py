import pytest
import numpy as np
from aeon_py import core
from aeon_py.client import AeonClient, RESULT_DTYPE

def test_schema_validation_pass(tmp_path):
    """Verify that normally, the schema matches and no error is raised."""
    # This should pass without error
    client = AeonClient(tmp_path / "safety.atlas")
    assert client is not None

def test_schema_mismatch_detection(tmp_path, monkeypatch):
    """Verify that if C++ reports a different size, we crash."""
    
    # Mock core.get_result_node_size to return a wrong value
    original_size = core.get_result_node_size()
    fake_size = original_size + 1 
    
    # We can't easily mock C++ extension functions with standard monkeypatch 
    # if they are bound directly. But core is a module.
    # We need to wrap or patch it.
    
    # Since we can't easily patch the C++ function in the extension module directly
    # in a way that 'client.py' sees if it imports 'core' directly...
    # client.py does `from . import core`.
    
    # Let's use `unittest.mock.patch` on `aeon_py.client.core`
    from unittest.mock import patch
    
    with patch('aeon_py.client.core') as mock_core:
        mock_core.get_result_node_size.return_value = fake_size
        
        # We also need to mock Atlas construction because validation happens first
        # But validation happens BEFORE Atlas construction if we look at Code?
        # __init__: self._validate_schema(); self.atlas = ...
        # wait, self.atlas is lazy loaded. 
        # But _validate_schema calls core.get_result_node_size().
        
        with pytest.raises(RuntimeError) as excinfo:
            AeonClient(tmp_path / "unsafe.atlas")
            
        assert "CRITICAL ARCHITECTURE FAIL" in str(excinfo.value)
        assert f"C++ ResultNode: {fake_size} bytes" in str(excinfo.value)
