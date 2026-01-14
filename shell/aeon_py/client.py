import numpy as np
from pathlib import Path
from typing import Optional
from . import core

# Define the schema strictly matching C++ ResultNode
# Layout: id(8), sim(4), preview(12) = 24 bytes (packed? or aligned?)
# C++ Struct: 
#   uint64_t id; (8)
#   float similarity; (4)
#   float centroid_preview[3]; (12)
# Total = 24 bytes. 
# Alignment: uint64 needs 8-byte alignment.
# 8 (id) + 4 (sim) + 12 (preview) = 24.
# 24 is divisible by 8. So sizeof should be 24.
RESULT_DTYPE = np.dtype([
    ('id', 'u8'), 
    ('similarity', 'f4'), 
    ('preview', 'f4', (3,))
])

class AeonClient:
    """
    High-level Python interface to the Atlas spatial memory engine.
    Wraps the C++ core with lazy loading and type safety.
    """
    
    def __init__(self, atlas_path: Path | str) -> None:
        self._path = Path(atlas_path)
        self._atlas: Optional[core.Atlas] = None
        
        # Verify schema immediately
        self._validate_schema()
        
    def _validate_schema(self):
        """Ensures C++ binary layout matches Python NumPy definition."""
        cpp_size = core.get_result_node_size()
        py_size = RESULT_DTYPE.itemsize
        
        if cpp_size != py_size:
            raise RuntimeError(
                f"CRITICAL ARCHITECTURE FAIL: Memory Layout Mismatch.\n"
                f"C++ ResultNode: {cpp_size} bytes\n"
                f"Python Dtype:   {py_size} bytes\n"
                "Check struct padding/alignment in schema.hpp vs client.py."
            )

    @property
    def atlas(self) -> core.Atlas:
        """Lazy initialization of the C++ Atlas."""
        if self._atlas is None:
            self._atlas = core.Atlas(str(self._path))
        return self._atlas

    def query(self, embedding: np.ndarray) -> np.ndarray:
        """
        Navigate the Atlas using a query vector.
        
        Args:
            embedding: (768,) float32 numpy array
            
        Returns:
            Structured numpy array with columns ['id', 'similarity', 'preview']
        """
        if embedding.shape != (768,) or embedding.dtype != np.float32:
            if embedding.size == 768:
                embedding = embedding.astype(np.float32).reshape(768)
            else:
                raise ValueError("Embedding must be shape (768,) and float32")
                
        # Call C++ binding which returns nb::ndarray<uint8_t>
        byte_view = self.atlas.navigate_raw(embedding)
        
        # Use simple cast to established global dtype
        dtype = RESULT_DTYPE
        
        # Zero-Copy view from the capsule
        # Note: frombuffer expects a buffer-like object. Capsule interacts with binding?
        # Actually standard python ctypes/buffer interface. 
        # CAUTION: 'nb::capsule' in Python doesn't directly support buffer protocol unless specified?
        # Nanobind capsules are opaque.
        # But we can use ctypes? Or did I assume too much?
        
        # WAIT: Nanobind capsule might not work with frombuffer directly.
        # But `nanobind` ndarray does.
        # If I return `nb::tuple(capsule, count)`, I still have the problem of accessing the pointer in Python.
        
        # CORRECTION:
        # I should have returned `nb::ndarray` of BYTES from C++!
        # My previous attempt `byte_array` WAS correct C++ side, but the `.view(dtype)` failed or built failed?
        # The build failure was seemingly `nb::dtype` related.
        # If I simply return `nb::ndarray<uint8_t>` (bytes) from C++, 
        # then in Python I do `.view(dtype)`.
        
        # Let's revert the "capsule" return idea which is unsafe in Python (pointer access).
        # Let's return `nb::ndarray` (uint8) from C++.
        # And do the casting in Python.
        # NOTE: I am editing CLIENT.PY now. 
        # I should expect `navigate_raw` to return a `np.ndarray` of bytes (uint8).
        
        # Cast to structured array
        result = byte_view.view(dtype)
        result.flags.writeable = False # Enforce read-only explicitly
        
        return result

    def get_children(self, node_id: int) -> np.ndarray:
        """
        Returns child nodes (siblings of next step) for visualization.
        
        Args:
            node_id: uint64 ID of the parent node.
            
        Returns:
            Structured numpy array of child nodes.
        """
        # Call C++ binding which returns nb::ndarray<uint8_t>
        byte_view = self.atlas.get_children_raw(node_id)
        
        # Cast to structured array
        result = byte_view.view(RESULT_DTYPE)
        result.flags.writeable = False # Enforce read-only explicitly
        
        return result

    def warmup(self) -> None:
        """
        Runs a dummy query to touch memory pages and warm up the OS page cache for the mmap.
        """
        dummy = np.zeros(768, dtype=np.float32)
        try:
            self.query(dummy)
        except Exception:
            # Ignore errors on empty atlas
            pass
