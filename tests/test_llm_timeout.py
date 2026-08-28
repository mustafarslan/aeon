"""Total wall-clock budget on one streamed generation (`OllamaProvider`).

WHY THIS EXISTS. `requests`' `timeout=` is a CONNECT and PER-READ timeout, never a total one.
A server that emits one byte inside every read window -- or half-closes without signalling
end-of-stream -- keeps `iter_lines()` blocked indefinitely and the per-read timeout never
fires. There is then no way for a caller to bound a single call.

That is not hypothetical here: during the v4.2 chronology cohort a single question consumed
17 minutes while printing nothing, and was diagnosed as a hung process before `lsof` showed
otherwise. These tests pin the guard and, more importantly, pin that an overrun is reported
as an ERROR rather than as a short answer -- a truncated generation scored as a wrong answer
is exactly the defect §8c.5 records costing 4 questions of an n=500 arm.
"""

import os
from unittest import mock

import pytest

from aeon_py.llm import OllamaProvider


class _FakeResponse:
    """A stream that never ends -- the failure `timeout=` cannot catch."""

    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        return None

    def json(self):
        # `_compute_num_ctx()` probes /api/show through the same `requests.post` this
        # test patches; an empty payload sends it down its documented fallback path.
        return {}

    def iter_lines(self):
        yield from self._lines


def _endless():
    while True:
        yield b'{"response": "tick "}'


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setenv("AEON_LLM_MODEL", "test-model")
    # Pin num_ctx so sizing does not depend on a live /api/show probe.
    monkeypatch.setenv("AEON_LLM_NUM_CTX", "8192")
    return OllamaProvider()


def test_total_budget_defaults_and_is_separate_from_the_read_timeout(provider):
    """Two different numbers with two different jobs; conflating them is the bug."""
    assert provider.total_timeout_seconds == 600.0
    assert provider.timeout_seconds == 120.0


def test_total_budget_is_configurable(monkeypatch):
    monkeypatch.setenv("AEON_LLM_TOTAL_TIMEOUT_SECONDS", "42")
    assert OllamaProvider().total_timeout_seconds == 42.0


def test_an_endless_stream_terminates_instead_of_hanging(provider, monkeypatch):
    monkeypatch.setenv("AEON_LLM_TOTAL_TIMEOUT_SECONDS", "0")
    provider.total_timeout_seconds = 0.0
    with mock.patch("requests.post", return_value=_FakeResponse(_endless())):
        out = "".join(provider.generate("hello"))
    assert "[System Error:" in out, "an overrun must be MARKED, never returned as a short answer"
    assert "total budget" in out


def test_a_normal_stream_is_untouched_by_the_guard(provider):
    """The guard must not truncate a generation that completes inside its budget."""
    lines = [b'{"response": "hello "}', b'{"response": "world"}', b'{"done": true}']
    with mock.patch("requests.post", return_value=_FakeResponse(iter(lines))):
        out = "".join(provider.generate("hi"))
    assert out == "hello world"
    assert "[System Error:" not in out
