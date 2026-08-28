"""Contextual distillation at ingestion (v4-plan.md Stage 4 task 4).

A bug-fix note reading "reverted the retry, the deadlock went away" embeds almost identically
whether it came from the auth service or the billing pipeline. On a shared tier that mixes
both, retrieval cannot tell them apart. Prepending the situating fields before embedding moves
that distinction into the vector, where retrieval can act on it.

The invariant these tests exist to protect is that the stored text and its vector describe the
SAME string. Contextualising the text without re-embedding would rank on one and display the
other, which is a silent correctness failure rather than a visible one.
"""

import numpy as np
import pytest

from aeon_py.architect import Architect, SituatingContext


class _FakeAtlasClient:
    """Records what reached the kernel. No encoder, no mmap."""

    def __init__(self, existing_similarity=0.0):
        self.inserted = []
        self._sim = existing_similarity
        self.atlas = self

    def query(self, vec, **kw):
        if self._sim <= 0:
            return []
        return np.array([(1, self._sim)], dtype=[("id", "<u8"), ("similarity", "<f4")])

    def insert_delta(self, vector, text):
        self.inserted.append((list(vector), text))
        return 42


def _embed(text: str):
    """Deterministic, content-sensitive stand-in: distinct text yields a distinct vector."""
    return [float(len(text)), float(sum(map(ord, text)) % 1000), 0.5]


# --- rendering -------------------------------------------------------------------

def test_render_emits_only_the_fields_that_are_set():
    assert SituatingContext(task_id="AE-14").render() == "[task=AE-14] "
    assert SituatingContext(subsystem="auth", error_class="Deadlock").render() == \
        "[subsystem=auth error=Deadlock] "


def test_an_empty_context_is_falsy_and_renders_nothing():
    """A caller holding only a subsystem must not be forced to invent a task id, and a
    fully-empty context must behave as no context at all rather than prepending '[] '."""
    assert SituatingContext().render() == ""
    assert not SituatingContext()


# --- the text/vector agreement invariant -----------------------------------------

def test_context_is_prepended_to_the_stored_text():
    client = _FakeAtlasClient()
    Architect(client).ingest("deadlock on retry", [0.0, 0.0, 0.0],
                             context=SituatingContext(subsystem="auth"), embed=_embed)
    _vector, text = client.inserted[0]
    assert text == "[subsystem=auth] deadlock on retry"


def test_the_stored_vector_is_recomputed_from_the_contextualised_text():
    """THE CENTRAL INVARIANT. The caller's vector was computed from the bare text; storing it
    beside contextualised text would make the record and its embedding disagree."""
    client = _FakeAtlasClient()
    stale = [9.9, 9.9, 9.9]
    Architect(client).ingest("deadlock on retry", stale,
                             context=SituatingContext(subsystem="auth"), embed=_embed)
    vector, text = client.inserted[0]
    assert vector != stale, "the pre-context vector was stored against contextualised text"
    assert vector == _embed(text)


def test_context_without_an_encoder_is_refused_rather_than_silently_skipped():
    """Failing closed: silently ignoring the context would produce an uncontextualised
    corpus that the operator believes is contextualised."""
    with pytest.raises(ValueError, match="embed"):
        Architect(_FakeAtlasClient()).ingest("x", [0.0], context=SituatingContext(task_id="A"))


# --- unchanged default -----------------------------------------------------------

def test_no_context_reproduces_the_previous_behaviour_exactly():
    client = _FakeAtlasClient()
    Architect(client).ingest("deadlock on retry", [1.0, 2.0, 3.0])
    assert client.inserted[0] == ([1.0, 2.0, 3.0], "deadlock on retry")


# --- interaction with near-duplicate admission -----------------------------------

def test_near_duplicate_admission_still_fires_on_the_contextualised_vector():
    """Admission is checked against the vector actually being stored, not the caller's."""
    client = _FakeAtlasClient(existing_similarity=0.99)
    node_id, is_dup = Architect(client).ingest(
        "deadlock on retry", [0.0, 0.0, 0.0],
        context=SituatingContext(subsystem="auth"), embed=_embed)
    assert is_dup is True
    assert node_id == 1
    assert client.inserted == [], "a near-duplicate must not be inserted"
