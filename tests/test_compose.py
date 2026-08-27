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


def test_premise_guard_is_off_by_default_and_opt_in():
    """Tried and reverted: it fixed abstention (22 -> 30/30) and cost 28 questions overall,
    19 of them single-session-preference advice questions the guard refused to answer."""
    assert "not available" not in compose([item("x")], [], "Q")
    assert "not available" in compose([item("x")], [], "Q", premise_guard=True)


def test_premise_guard_precedes_the_counting_hint_when_enabled():
    """Ordering was the substance of the attempt: with zero matching records, "COUNT the
    matching records" is what produced the measured "Answer: 0" on an unanswerable question."""
    p = compose([item("x")], [], "Q", premise_guard=True)
    assert p.index("not available") < p.index("COUNT")


# --- co-referent collapse (v4.1 Stage 2a) -------------------------------------------

def test_one_entity_in_two_buckets_renders_as_one_line():
    """The measured `gpt4_15e38248` case: the same table filed under ACQUISITION and
    POSSESSION, counted twice, answered 5 against a gold of 4."""
    recs = [item("coffee table", "ACQUISITION", "furniture"),
            item("coffee table", "POSSESSION", "furniture")]
    out = render_records(recs)
    assert out.count("coffee table") == 1
    assert "ACQUISITION/furniture" in out and "POSSESSION/furniture" in out


def test_render_is_byte_identical_when_there_are_no_coreferents():
    """The equivalence guarantee. Collapsing must be invisible to a corpus that has
    nothing to collapse -- otherwise this change silently moves the measured 429 on the
    ~13 of 500 questions that have no duplicates at all."""
    recs = [item("Dune", "MEDIA", "film"),
            item("Kind of Blue", "ACQUISITION", "music album"),
            Record(kind="FACT", text="owns a bike")]
    expected = "\n".join(r.display() for r in order_records(recs))
    assert render_records(recs) == expected


def test_within_bucket_exact_duplicates_collapse():
    """25% of the corpus's duplicates are same-bucket, not cross-bucket."""
    recs = [item("Dune", "MEDIA", "film"), item("dune!", "MEDIA", "film")]
    assert render_records(recs).count("\n") == 0        # one line


def test_collapse_keeps_the_longest_form_as_the_visible_text():
    recs = [item("coffee table", "ACQUISITION", "furniture"),
            item("Wooden coffee table with metal legs", "ACQUISITION", "furniture")]
    assert "Wooden coffee table with metal legs" in render_records(recs)


def test_deduped_render_does_not_lose_a_bucket_from_the_head():
    """Findability is the whole reason multi-bucket filing exists; a collapse that drops
    a bucket would trade a counting fix for a retrieval regression."""
    recs = [item("Kind of Blue", "ACQUISITION", "music album"),
            item("Kind of Blue", "MEDIA", "music album"),
            item("Kind of Blue", "POSSESSION", "music album")]
    out = render_records(recs)
    for b in ("ACQUISITION", "MEDIA", "POSSESSION"):
        assert b in out


def test_select_records_still_sees_every_bucket_copy():
    """FEATURE-PRESERVATION. Dedup is render-only: filtering and rehydration still
    operate on the raw record list, so every provenance link survives."""
    recs = [item("Kind of Blue", "ACQUISITION", "music album"),
            item("Kind of Blue", "MEDIA", "music album")]
    assert len(select_records(recs, buckets=["MEDIA"])) == 1
    assert len(select_records(recs, buckets=["ACQUISITION"])) == 1
    assert len(recs) == 2


def test_collapse_preserves_date_and_supersession_markers():
    recs = [item("salary", "FINANCE", "pay", date="2023/05/18", supersedes="$350,000"),
            item("salary", "EDUCATION_WORK", "pay")]
    out = render_records(recs)
    assert "[2023/05/18]" in out and "[supersedes $350,000]" in out


def test_prose_records_keep_their_order_after_collapse():
    recs = [item("Dune"), Record(kind="FACT", text="alpha"), Record(kind="PREF", text="beta")]
    out = render_records(recs).splitlines()
    assert out[-2].startswith("FACT:") and out[-1].startswith("PREF:")


def test_empty_records_still_render_the_placeholder():
    assert render_records([]) == "(no records)"
