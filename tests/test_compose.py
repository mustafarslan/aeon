"""Tests for the composite read path (`aeon_py.compose`)."""

import pytest

from aeon_py.compose import (
    compose, order_records, render_records, select_records,
)
from aeon_py.records import Provenance, Record


def item(text, bucket="MEDIA", subtype="film", **kw):
    return Record(kind="ITEM", text=text, bucket=bucket, subtype=subtype, **kw)


def test_countable_members_are_grouped_together():
    """Counting is the failure this layer targets. A model counting ITEM lines scattered
    through 250 unordered records is doing the same multi-hop scan on a smaller haystack."""
    recs = [item("Dune"), Record(kind="FACT", text="owns a bike"),
            item("Happier Than Ever", "ACQUISITION", "music album"),
            Record(kind="PREF", text="likes jazz"),
            item("Midnight Sky", "ACQUISITION", "music album")]
    out = order_records(recs)
    albums = [i for i, r in enumerate(out) if r.subtype == "music album"]
    assert albums == list(range(min(albums), min(albums) + len(albums)))  # contiguous


def test_items_come_before_prose_records():
    out = order_records([Record(kind="FACT", text="f"), item("Dune")])
    assert out[0].kind == "ITEM"


def test_updates_are_kept_ahead_of_other_prose():
    out = order_records([Record(kind="PREF", text="p"), Record(kind="UPDATE", text="u")])
    kinds = [r.kind for r in out]
    assert kinds.index("UPDATE") < kinds.index("PREF")


# --- the scaling path is a category SCAN, not a similarity search ------------

def test_select_records_returns_every_member_of_a_category():
    """Counting requires completeness. A top-k would return a subset and turn a complete
    answer into a plausible wrong one."""
    recs = [item(f"album{i}", "ACQUISITION", "music album") for i in range(5)]
    recs.append(item("Dune", "MEDIA", "film"))
    got = select_records(recs, buckets=["ACQUISITION"])
    assert len(got) == 5


def test_select_records_filters_by_subtype():
    recs = [item("a", "MEDIA", "music album"), item("b", "MEDIA", "film")]
    assert len(select_records(recs, subtype_contains="album")) == 1


def test_select_records_is_case_insensitive():
    assert len(select_records([item("a", "MEDIA", "Music Album")],
                              buckets=["media"], subtype_contains="ALBUM")) == 1


def test_select_records_unfiltered_returns_all():
    recs = [item("a"), Record(kind="FACT", text="f")]
    assert len(select_records(recs)) == 2


# --- prompt assembly --------------------------------------------------------

def test_compose_includes_both_sources():
    p = compose([item("Dune")], ["- [user] I watched Dune"], "Question: what did I watch?")
    assert "Long-term memory records" in p and "Relevant conversation excerpts" in p
    assert "Dune" in p and "I watched Dune" in p


def test_compose_omits_the_episodic_section_when_empty():
    p = compose([item("Dune")], [], "Question: x")
    assert "Relevant conversation excerpts" not in p


def test_compose_carries_the_question_block_verbatim():
    """The reference date travels inside question_block -- a field this project measured as
    worth ~19 questions when it was missing."""
    qb = "Today's date is 2023/05/06.\nQuestion: how many weeks ago?"
    assert qb in compose([item("x")], [], qb)


def test_counting_hint_is_present_by_default_and_suppressible():
    assert "COUNT" in compose([item("x")], [], "Q")
    assert "COUNT" not in compose([item("x")], [], "Q", counting_hint=False)


def test_render_records_handles_an_empty_store():
    assert render_records([]) == "(no records)"
    assert "(no records)" in compose([], [], "Question: x")


def test_records_render_with_their_category_and_supersession():
    p = render_records([Record(kind="ITEM", text="$400,000", bucket="FINANCE",
                               subtype="pre-approval", supersedes="$350,000")])
    assert "FINANCE/pre-approval" in p and "supersedes $350,000" in p
