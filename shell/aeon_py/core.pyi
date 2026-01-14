from collections.abc import Sequence
import os
from typing import Annotated

import numpy
from numpy.typing import NDArray


def version() -> str:
    """Get the library version"""

class BuildInfo:
    @property
    def compiler(self) -> str: ...

    @property
    def architecture(self) -> str: ...

    @property
    def simd_level(self) -> str: ...

    @property
    def standard(self) -> str: ...

    @property
    def repr(self) -> str: ...

def get_build_info() -> BuildInfo:
    """Get build environment details"""

def get_result_node_size() -> int:
    """Return size of ResultNode struct for schema validation"""

class Atlas:
    def __init__(self, path: str | os.PathLike) -> None: ...

    def size(self) -> int: ...

    def insert(self, parent_id: int, vector: Sequence[float], metadata: str) -> int: ...

    def navigate_raw(self, query: Sequence[float]) -> Annotated[NDArray[numpy.uint8], dict(shape=(None,))]:
        """Returns byte array of results (view as structured in Python)"""
