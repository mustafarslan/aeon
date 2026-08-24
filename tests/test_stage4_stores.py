"""
V4 Stage 4 task 1: physically separate private/shared Atlas stores.

Covers the store-discrimination encoding (client.py) and
ContextManager.query_stores()/warm_start()'s routing between two
physically distinct Atlas instances -- the isolation guarantee this
increment exists to build (v4-plan.md).
"""
import numpy as np
import pytest
from unittest.mock import MagicMock

from aeon_py.client import AeonClient, SHARED_STORE_BIT, encode_store_id, decode_store_id
from aeon_py.context import ContextManager
from aeon_py.trace import TraceGraph


def _vec(seed: float) -> np.ndarray:
    return np.full(768, seed, dtype=np.float32)


class TestStoreIdEncoding:
    def test_roundtrip_private(self):
        assert decode_store_id(encode_store_id(0, is_shared=False)) == (0, False)
        assert decode_store_id(encode_store_id(12345, is_shared=False)) == (12345, False)

    def test_roundtrip_shared(self):
        assert decode_store_id(encode_store_id(0, is_shared=True)) == (0, True)
        assert decode_store_id(encode_store_id(12345, is_shared=True)) == (12345, True)

    def test_private_encoding_is_byte_identical_to_raw_id(self):
        # The common case (no shared tier configured) must round-trip
        # exactly as before Stage 4 -- no representation change for
        # existing single-store deployments.
        assert encode_store_id(999, is_shared=False) == 999

    def test_shared_and_private_same_raw_id_are_distinct(self):
        priv = encode_store_id(7, is_shared=False)
        shared = encode_store_id(7, is_shared=True)
        assert priv != shared
        assert shared == (7 | SHARED_STORE_BIT)

    def test_rejects_double_encoding(self):
        once = encode_store_id(5, is_shared=True)
        with pytest.raises(ValueError):
            encode_store_id(once, is_shared=True)
        with pytest.raises(ValueError):
            encode_store_id(once, is_shared=False)


@pytest.fixture
def private_atlas(tmp_path):
    return AeonClient(tmp_path / "private.atlas")


@pytest.fixture
def shared_atlas(tmp_path):
    return AeonClient(tmp_path / "shared.atlas")


class TestQueryStores:
    def test_private_mode_returns_encoded_private_ids(self, private_atlas):
        private_atlas.atlas.insert(0, _vec(1.0).tolist(), "root")
        ctx = ContextManager(private_atlas, TraceGraph())

        results = ctx.query_stores(_vec(1.0), mode="private")

        assert len(results) == 1
        assert results[0]["store"] == "private"
        assert results[0]["id"] == 0  # unencoded -- private ids round-trip raw

    def test_shared_mode_raises_without_configured_shared_store(self, private_atlas):
        ctx = ContextManager(private_atlas, TraceGraph())  # shared_atlas_client=None
        with pytest.raises(RuntimeError):
            ctx.query_stores(_vec(1.0), mode="shared")

    def test_merged_mode_degrades_to_private_only_without_shared_store(self, private_atlas):
        private_atlas.atlas.insert(0, _vec(1.0).tolist(), "root")
        ctx = ContextManager(private_atlas, TraceGraph())  # shared_atlas_client=None

        results = ctx.query_stores(_vec(1.0), mode="merged")

        assert len(results) == 1
        assert results[0]["store"] == "private"

    def test_merged_mode_distinguishes_colliding_raw_ids_across_stores(
        self, private_atlas, shared_atlas
    ):
        # Both stores independently assign node id 0 to their first insert
        # -- exactly the collision physical separation introduces. Give
        # them DIFFERENT vectors so a query can't accidentally match one
        # and not the other, then confirm both survive the merge as
        # distinct entries rather than being deduplicated into one.
        private_atlas.atlas.insert(0, _vec(1.0).tolist(), "private root")
        shared_atlas.atlas.insert(0, _vec(-1.0).tolist(), "shared root")

        ctx = ContextManager(private_atlas, TraceGraph(), shared_atlas_client=shared_atlas)

        # A neutral query with some similarity to both.
        results = ctx.query_stores(_vec(0.0), mode="merged")

        assert len(results) == 2
        ids = {r["id"] for r in results}
        stores = {r["store"] for r in results}
        assert stores == {"private", "shared"}
        # Distinct encoded ids despite both being raw node 0.
        assert 0 in ids  # private node 0, unencoded
        assert (0 | SHARED_STORE_BIT) in ids  # shared node 0, encoded
        assert len(ids) == 2  # NOT deduplicated into one

    def test_unknown_mode_raises(self, private_atlas):
        ctx = ContextManager(private_atlas, TraceGraph())
        with pytest.raises(ValueError):
            ctx.query_stores(_vec(1.0), mode="bogus")


class TestWarmStartRouting:
    def test_routes_private_and_shared_ids_to_correct_client(self):
        mock_private = MagicMock()
        mock_shared = MagicMock()
        trace = TraceGraph()  # real, in-memory

        ctx = ContextManager(mock_private, trace, shared_atlas_client=mock_shared)

        # Manually record concept events with pre-encoded atlas_ids mixing
        # both stores, bypassing process_turn() to isolate warm_start()'s
        # own decode/route logic.
        trace.add_event(
            "s1", "concept", "priv", atlas_id=encode_store_id(10, is_shared=False)
        )
        trace.add_event(
            "s1", "concept", "shared", atlas_id=encode_store_id(20, is_shared=True)
        )

        ctx.warm_start("s1")

        mock_private.load_context.assert_called_once()
        assert mock_private.load_context.call_args.args[0] == [10]
        mock_shared.load_context.assert_called_once()
        assert mock_shared.load_context.call_args.args[0] == [20]

    def test_shared_ids_skipped_when_no_shared_store_configured(self):
        mock_private = MagicMock()
        trace = TraceGraph()

        ctx = ContextManager(mock_private, trace)  # shared_atlas_client=None

        trace.add_event(
            "s1", "concept", "priv", atlas_id=encode_store_id(10, is_shared=False)
        )
        trace.add_event(
            "s1", "concept", "shared", atlas_id=encode_store_id(20, is_shared=True)
        )

        ctx.warm_start("s1")  # must not raise despite an unroutable shared id

        mock_private.load_context.assert_called_once()
        assert mock_private.load_context.call_args.args[0] == [10]

    def test_delta_arena_id_survives_store_decode_untouched(self):
        # advisor review (v4-plan.md Stage 4 step 3): a delta-arena node id
        # (NODE_ID_DELTA_MASK, bit 63) can reach process_turn()'s write site
        # on the very first turn after a fresh insert_delta(), before that
        # node is ever compacted into the mmap generation. decode_store_id()
        # must only touch bit 62 (SHARED_STORE_BIT) and leave bit 63 intact
        # -- warm_start() then hands that id, delta bit and all, to
        # load_context(), which already no-ops for delta ids on its own
        # (Atlas::load_context(), atlas.cpp) rather than raising. This test
        # pins that the shell layer doesn't corrupt or strip the delta bit
        # anywhere in the encode/decode/route path.
        NODE_ID_DELTA_MASK = 1 << 63
        delta_id = NODE_ID_DELTA_MASK | 7

        encoded = encode_store_id(delta_id, is_shared=False)
        assert encoded == delta_id  # private encoding is a pure passthrough

        raw, is_shared = decode_store_id(encoded)
        assert raw == delta_id  # delta bit preserved, not stripped
        assert is_shared is False

        mock_private = MagicMock()
        trace = TraceGraph()
        ctx = ContextManager(mock_private, trace)
        trace.add_event("s1", "concept", "priv", atlas_id=encoded)

        ctx.warm_start("s1")  # must not raise

        mock_private.load_context.assert_called_once()
        assert mock_private.load_context.call_args.args[0] == [delta_id]
