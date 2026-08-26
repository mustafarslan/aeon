"""Tests for `aeon_py.parallel`.

The properties tested here are the ones whose failure would be SILENT: out-of-order results
producing unreproducible record sets, shared provider state racing and mis-attributing
context sizes, and one bad item aborting a 23,000-call pass.
"""

import threading
import time

import pytest

from aeon_py.parallel import (
    DEFAULT_MAX_WORKERS, ThreadLocalResource, benchmark_concurrency, parallel_map,
)


def test_preserves_input_order_regardless_of_completion_order():
    """Consolidation assembles records into a set; if order depended on which worker finished
    first, two runs over identical input would produce different record files and every
    downstream comparison would be unreproducible."""
    def slow_then_fast(i):
        time.sleep(0.02 if i == 0 else 0.0)   # first item finishes last
        return i * 2

    assert parallel_map(slow_then_fast, list(range(8)), max_workers=4) == [i * 2 for i in range(8)]


def test_actually_runs_concurrently():
    barrier = threading.Barrier(4, timeout=5)

    def wait(_):
        barrier.wait()          # only completes if 4 run at once
        return True

    assert parallel_map(wait, list(range(4)), max_workers=4) == [True] * 4


def test_respects_the_worker_cap():
    seen, lock = [], threading.Lock()
    live = 0

    def track(_):
        nonlocal live
        with lock:
            live += 1
            seen.append(live)
        time.sleep(0.01)
        with lock:
            live -= 1
        return None

    parallel_map(track, list(range(12)), max_workers=3)
    assert max(seen) <= 3


def test_empty_input():
    assert parallel_map(lambda x: x, []) == []


def test_single_worker_path_matches_parallel_path():
    items = list(range(6))
    assert parallel_map(lambda x: x + 1, items, max_workers=1) == \
           parallel_map(lambda x: x + 1, items, max_workers=4)


def test_workers_never_exceed_item_count():
    assert parallel_map(lambda x: x, [1], max_workers=16) == [1]


# --- failure isolation -------------------------------------------------------

def test_one_bad_item_does_not_abort_the_pass():
    """A single malformed session must not lose the other 22,999."""
    def maybe_raise(i):
        if i == 3:
            raise ValueError("bad session")
        return i

    out = parallel_map(maybe_raise, list(range(6)), max_workers=4,
                       on_error=lambda item, exc: -1)
    assert out == [0, 1, 2, -1, 4, 5]


def test_exception_propagates_when_no_handler_given():
    with pytest.raises(ValueError):
        parallel_map(lambda i: (_ for _ in ()).throw(ValueError("x")) if i == 2 else i,
                     list(range(4)), max_workers=2)


def test_error_handler_sees_the_item_and_exception():
    captured = {}

    def handler(item, exc):
        captured["item"], captured["exc"] = item, type(exc)
        return None

    parallel_map(lambda i: 1 / 0, ["session-7"], max_workers=1, on_error=handler)
    assert captured["item"] == "session-7" and captured["exc"] is ZeroDivisionError


# --- per-thread resources ----------------------------------------------------

def test_thread_local_resource_is_one_instance_per_thread():
    """Providers hold per-call state (`last_num_ctx`) that this project records per question.
    Sharing one across threads mis-attributes it silently."""
    made = []
    lock = threading.Lock()

    def factory():
        obj = object()
        with lock:
            made.append(obj)
        return obj

    res = ThreadLocalResource(factory)
    ids = parallel_map(lambda _: id(res.get()), list(range(16)), max_workers=4)
    assert len(set(ids)) <= 4                    # at most one per worker thread
    assert len(made) == len(set(id(m) for m in made))


def test_thread_local_resource_reuses_within_a_thread():
    res = ThreadLocalResource(object)
    assert id(res.get()) == id(res.get())


# --- progress + benchmark ----------------------------------------------------

def test_progress_is_reported_for_every_item():
    seen = []
    lock = threading.Lock()

    def progress(done, total):
        with lock:
            seen.append((done, total))

    parallel_map(lambda x: x, list(range(10)), max_workers=4, progress=progress)
    assert len(seen) == 10 and all(t == 10 for _, t in seen)
    assert max(d for d, _ in seen) == 10


def test_default_cap_is_the_measured_knee():
    """4 is measured on the REAL extraction workload (74.6s -> 36.2s, 2.06x), not on a toy
    prompt -- the synthetic probe said 3.2x and overstated it, because longer generations
    contend harder. No reliable gain past 4."""
    assert DEFAULT_MAX_WORKERS == 4


def test_benchmark_concurrency_returns_a_level_per_input():
    out = benchmark_concurrency(lambda: time.sleep(0.001), levels=(1, 2), samples=2)
    assert set(out) == {1, 2} and all(v > 0 for v in out.values())
