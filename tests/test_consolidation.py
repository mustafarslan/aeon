"""Tests for the semantic layer's write path (`aeon_py.consolidation`).

Parsing is tested hard and without a model. These lines are produced by a query-blind prompt
across hundreds of sessions, so the parser is the component most exposed to malformed input,
and a single bad line must never abort a consolidation pass.
"""

import pytest

from aeon_py.consolidation import (
    consolidate, extract_session, number_turns, parse_consolidated, parse_record_line,
    parse_records, render_for_consolidation,
)
from aeon_py.records import Provenance, Record


def p(line, session="s1", **kw):
    return parse_record_line(line, session, **kw)


# --- the shapes the model actually emits ------------------------------------

def test_item_with_bucket_subtype_date_and_citation():
    r = p('ITEM(ACQUISITION/music album): "Happier Than Ever" [2023/05/20] #3')
    assert r.kind == "ITEM" and r.bucket == "ACQUISITION" and r.subtype == "music album"
    assert r.date.startswith("2023/05/20")
    assert r.text == '"Happier Than Ever"'
    assert r.provenance.turn_indices == (3,)


@pytest.mark.parametrize("cite,expected", [
    ("#3", (3,)), ("#3,4", (3, 4)), ("#3, 4", (3, 4)), ("#2-4", (2, 3, 4)), ("# 5", (5,)),
])
def test_turn_citation_formats(cite, expected):
    assert p(f"FACT: the user owns a bike {cite}").provenance.turn_indices == expected


@pytest.mark.parametrize("kind", ["FACT", "UPDATE", "PREF", "TASK"])
def test_simple_kinds(kind):
    r = p(f"{kind}: something durable #1")
    assert r.kind == kind and r.text == "something durable"


def test_event_with_bracketed_date():
    r = p("EVENT [2023/05/20 (Sat) 10:00]: ran a 5K #2")
    assert r.kind == "EVENT" and r.date.startswith("2023/05/20") and r.text == "ran a 5K"


def test_supersedes_is_extracted_not_left_in_text():
    r = p("ITEM(FINANCE/pre-approval): $400,000 [supersedes $350,000] #7")
    assert r.supersedes == "$350,000" and "supersedes" not in r.text and r.text == "$400,000"


def test_leading_bullets_are_tolerated():
    assert p("- ITEM(MEDIA/film): Dune #1").bucket == "MEDIA"
    assert p("* FACT: owns a bike #1").kind == "FACT"


# --- robustness: a bad line must not poison a pass ---------------------------

@pytest.mark.parametrize("junk", [
    "", "   ", "(none)", "none", "Records:", "here are the records",
    "ITEM(): #1", "FACT:   #1", "ITEM(MEDIA/film):    ",
])
def test_non_records_return_none(junk):
    assert p(junk) is None


def test_invented_bucket_degrades_to_fact_rather_than_minting_a_category():
    """An invented bucket is the exact failure the closed vocabulary prevents: it would never
    accumulate with anything. Keep the content, drop the bogus category."""
    r = p("ITEM(SNACKS/crisps): salt and vinegar #2")
    assert r is not None and r.kind == "FACT" and r.bucket == ""
    assert "salt and vinegar" in r.text


def test_out_of_range_citation_falls_back_not_lost():
    r = p("FACT: owns a bike #99", fallback_indices=(0, 1), n_turns=5)
    assert r.provenance.turn_indices == (0, 1)


def test_malformed_huge_range_is_ignored():
    r = p("FACT: x #1-99999", fallback_indices=(2,), n_turns=10)
    assert r.provenance.turn_indices == (2,)


def test_parse_records_skips_junk_and_keeps_the_rest():
    out = parse_records(
        "\n".join([
            "Records:",
            "ITEM(MEDIA/film): Dune #0",
            "this line is prose the model added",
            "FACT: the user owns a bike #1",
            "",
        ]), "s1", n_turns=4)
    assert [r.kind for r in out] == ["ITEM", "FACT"]


def test_uncited_records_get_session_level_provenance():
    """A missing citation degrades the neighbourhood; it must not lose the link entirely."""
    (r,) = parse_records("FACT: owns a bike", "s1", n_turns=3)
    assert r.provenance.session_id == "s1" and r.provenance.turn_indices == (0, 1, 2)


# --- prompt construction -----------------------------------------------------

def test_turns_are_numbered_for_citation():
    txt = number_turns([{"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "hello"}])
    assert txt.splitlines()[0].startswith("[0] user:")
    assert txt.splitlines()[1].startswith("[1] assistant:")


def test_extract_session_is_query_blind():
    """The prompt must never contain a question -- the whole risk of write-time extraction is
    that it cannot see one."""
    seen = {}

    def fake(prompt, system_prompt="", temperature=0.0):
        seen["prompt"] = prompt
        return "ITEM(MEDIA/film): Dune #0"

    out = extract_session([{"role": "user", "content": "watched Dune"}], "s1",
                          "2023/05/20", fake)
    assert len(out) == 1 and out[0].bucket == "MEDIA"
    assert "?" not in seen["prompt"].split("Records:")[-1]
    assert "Question" not in seen["prompt"]


def test_extract_session_survives_a_transport_error():
    assert extract_session([{"role": "user", "content": "x"}], "s1", "d",
                           lambda *a, **k: "[System Error: nope]") == []


def test_extract_session_on_empty_input():
    assert extract_session([], "s1", "d", lambda *a, **k: "should not be called") == []


# --- the merge pass ----------------------------------------------------------

def test_provenance_survives_a_consolidation_rewrite():
    """The merge pass rewrites lines; without the @prov tag every record would come back
    origin-less and the neighbourhood could never be rehydrated."""
    recs = [Record(kind="ITEM", text="Dune", bucket="MEDIA", subtype="film",
                   provenance=Provenance("s1", (2, 3)))]
    rendered = render_for_consolidation(recs)
    assert "@prov:s1:2-3" in rendered
    (back,) = parse_consolidated(rendered)
    assert back.provenance.session_id == "s1" and back.provenance.turn_indices == (2, 3)


def test_consolidate_returns_input_when_model_errors():
    recs = [Record(kind="FACT", text=f"f{i}", provenance=Provenance("s1", (i,)))
            for i in range(5)]
    assert consolidate(recs, lambda *a, **k: "[System Error: down]") == recs


def test_consolidate_rejects_a_collapsing_pass():
    """Consolidation is normalisation, not summarisation. A pass that collapses the record
    set is a failure, and silently accepting it would delete memory."""
    recs = [Record(kind="FACT", text=f"f{i}", provenance=Provenance("s1", (i,)))
            for i in range(10)]
    out = consolidate(recs, lambda *a, **k: "FACT: everything @prov:s1:0")
    assert len(out) == 10


def test_consolidate_accepts_a_genuine_merge():
    recs = [Record(kind="FACT", text=f"f{i}", provenance=Provenance("s1", (i,)))
            for i in range(4)]
    merged = "\n".join(f"ITEM(MEDIA/film): m{i} @prov:s1:{i}" for i in range(4))
    out = consolidate(recs, lambda *a, **k: merged)
    assert len(out) == 4 and all(r.bucket == "MEDIA" for r in out)
    assert out[0].provenance.turn_indices == (0,)


# --- shapes found in REAL model output, not invented -------------------------
# The hand-written cases above cover the documented schema. These cover what the model
# actually emitted across 4,504 record lines in the consolidation probe. Rejecting the
# bare-bucket shorthand discarded 149 records (3.3%), including every HEALTH and
# EVENT_ATTENDED record, so these are regression guards on real data.

@pytest.mark.parametrize("line,bucket,subtype", [
    ("HEALTH: User averages 7 hours of sleep per night.", "HEALTH", ""),
    ("OBLIGATION/decision: Choose an external hard drive.", "OBLIGATION", "decision"),
    ("OBLIGATION/social: Catch up with friend Alex.", "OBLIGATION", "social"),
    ("EVENT_ATTENDED [2023/08/14]: Auto racking event.", "EVENT_ATTENDED", ""),
])
def test_bare_bucket_shorthand_is_accepted(line, bucket, subtype):
    r = p(line)
    assert r is not None and r.kind == "ITEM"
    assert r.bucket == bucket and r.subtype == subtype
    assert r.text


def test_bare_bucket_shorthand_keeps_its_date():
    r = p("EVENT_ATTENDED [2023/08/14]: Auto racking event.")
    assert r.date.startswith("2023/08/14")


def test_a_bare_word_that_is_not_a_bucket_is_not_an_item():
    """The shorthand is only safe because the bucket vocabulary is closed."""
    assert p("NOTE: this is just prose") is None


def test_consolidated_header_is_not_a_record():
    assert p("Consolidated records:") is None
