"""
Bounded parallel execution for LLM-bound shell work.

Consolidation is embarrassingly parallel -- each session is extracted independently -- and it
was entirely sequential, which is the dominant cost of the semantic layer. Measured on this
project's own workload: a full n=500 consolidation pass is ~23,000 extraction calls, roughly
9 hours sequential.

CONCURRENCY IS CAPPED AT 4, measured on the REAL workload rather than a synthetic one. A
short-output probe suggested 3.2x, but a real session-extraction pass (46 sessions, ~200-token
outputs) measured:

    workers=1    74.6 s      --
    workers=4    36.2 s      2.06x
    workers=8    52.9 s      1.41x
    workers=12   36.6 s      2.04x

Two things worth carrying forward. First, the synthetic figure (3.2x) OVERSTATED the real one
(2.06x): longer generations contend harder, so a concurrency number measured on toy prompts
does not transfer to production traffic. Second, the 8-worker result being worse than both 4
and 12 is measurement noise on a shared endpoint, not a knee -- these are single runs, and
reading structure into them would be exactly the kind of unvalidated constant this project has
already been burned by.

The supportable claim is therefore narrow: **~2x at 4 workers, with no reliable gain past 4.**
4 is kept because it captures that gain without pushing rate limits on a 23,000-call pass.
Re-measure with `benchmark_concurrency()` on a different endpoint rather than inheriting this.

PER-THREAD RESOURCES: `OllamaProvider` carries mutable per-call state (`last_num_ctx`, which
this project's result files record per question). Sharing one provider across threads makes
that state race and silently mis-attributes context sizes. `thread_local_factory` gives each
worker its own instance.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, Optional, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")

# Measured knee on this project's endpoint (see module docstring). Not a guess.
DEFAULT_MAX_WORKERS = 4


class ThreadLocalResource:
    """One instance per worker thread, created on first use.

    Exists because provider objects hold per-call state that must not be shared across
    threads -- the failure mode is silent (wrong `last_num_ctx` recorded against a question),
    not a crash, which is the kind that survives into published numbers.
    """

    def __init__(self, factory: Callable[[], Any]) -> None:
        self._factory = factory
        self._local = threading.local()

    def get(self) -> Any:
        obj = getattr(self._local, "obj", None)
        if obj is None:
            obj = self._factory()
            self._local.obj = obj
        return obj


def parallel_map(
    fn: Callable[[T], R],
    items: Sequence[T],
    *,
    max_workers: int = DEFAULT_MAX_WORKERS,
    on_error: Optional[Callable[[T, BaseException], R]] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> list[R]:
    """Apply `fn` across `items` with bounded concurrency, preserving INPUT ORDER.

    Order matters: consolidation assembles records into a set whose contents must not depend
    on which worker happened to finish first, or two runs of the same input produce different
    record files and every downstream comparison becomes unreproducible.

    A raised exception is confined to its item -- `on_error` supplies a replacement, or the
    exception is re-raised if no handler is given. One bad session must not abort a pass over
    the other 22,999.
    """
    n = len(items)
    if n == 0:
        return []
    workers = max(1, min(int(max_workers), n))
    if workers == 1:
        out: list[R] = []
        for i, item in enumerate(items, 1):
            out.append(_run_one(fn, item, on_error))
            if progress:
                progress(i, n)
        return out

    done = 0
    lock = threading.Lock()

    def wrapped(item: T) -> R:
        nonlocal done
        result = _run_one(fn, item, on_error)
        if progress:
            with lock:
                done += 1
                current = done
            progress(current, n)
        return result

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(wrapped, items))      # ex.map preserves input order


def _run_one(fn: Callable[[T], R], item: T,
             on_error: Optional[Callable[[T, BaseException], R]]) -> R:
    try:
        return fn(item)
    except BaseException as exc:                  # noqa: BLE001 - deliberately broad
        if on_error is None:
            raise
        return on_error(item, exc)


def benchmark_concurrency(call: Callable[[], Any], *, levels: Iterable[int] = (1, 2, 4, 8),
                          samples: int = 8) -> dict[int, float]:
    """Measure effective seconds-per-call at each concurrency level.

    Provided so the cap above is re-derived rather than inherited: a different model, endpoint
    or hardware moves the knee, and this project already learned the hard way that carrying an
    unvalidated constant forward produces confident wrong numbers.
    """
    out: dict[int, float] = {}
    for w in levels:
        t0 = time.perf_counter()
        parallel_map(lambda _: call(), list(range(samples)), max_workers=w)
        out[int(w)] = (time.perf_counter() - t0) / samples
    return out
