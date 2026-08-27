"""Tests for entity identity (`aeon_py.entities`).

`canonical_key` is FROZEN -- tuning it against which benchmark questions flip would be
fitting the test set. Several tests below exist specifically to make a future "improvement"
fail loudly rather than silently change what counts as one entity.
"""

from aeon_py.entities import EntityGroup, canonical_key, duplicate_count, group_entities
from aeon_py.records import Provenance, Record


def item(text, bucket="ACQUISITION", subtype="thing", date="", session="s1"):
    return Record(kind="ITEM", text=text, bucket=bucket, subtype=subtype, date=date,
                  provenance=Provenance(session))


def fact(text):
    return Record(kind="FACT", text=text)


# --- canonical_key ------------------------------------------------------------------

def test_key_ignores_case_and_punctuation():
    assert canonical_key("Kind of Blue") == canonical_key("kind of blue!")
    assert canonical_key('"Happier Than Ever"') == canonical_key("Happier Than Ever")


def test_key_ignores_bracketed_spans():
    """Dates and supersession markers are metadata, not identity."""
    assert canonical_key("coffee table [2023/05/18]") == canonical_key("coffee table")
    assert canonical_key("salary [supersedes $350,000]") == canonical_key("salary")


def test_key_strips_leading_articles_only():
    assert canonical_key("The Power of Now") == canonical_key("Power of Now")
    # Interior words are load-bearing: these are two different things.
    assert canonical_key("oat milk") != canonical_key("milk")


def test_key_is_order_sensitive():
    """Pins the frozen decision. An order-insensitive token-set key catches ~594 more
    duplicate lines across the corpus and would collide these two."""
    assert canonical_key("Alice called Bob") != canonical_key("Bob called Alice")


def test_key_normalises_unicode_width_and_whitespace():
    assert canonical_key("Ｃanon  rangefinder") == canonical_key("canon rangefinder")


def test_key_of_only_articles_is_empty_not_a_crash():
    assert canonical_key("the") == ""
    assert canonical_key("[2023/05/18]") == ""


# --- grouping -----------------------------------------------------------------------

def test_the_measured_coffee_table_case_groups():
    """`gpt4_15e38248`: the same table filed under two buckets, counted twice, answered
    5 against a gold of 4."""
    recs = [item("coffee table", bucket="ACQUISITION"),
            item("coffee table", bucket="POSSESSION")]
    groups, others = group_entities(recs)
    assert len(groups) == 1
    assert groups[0].buckets == ["POSSESSION", "ACQUISITION"]   # taxonomy order
    assert others == []


def test_non_item_records_pass_through_untouched():
    """FACT/PREF prose mentioning an entity must not collapse into it -- that would
    change what the record asserts, not merely how often it appears."""
    recs = [item("coffee table"), fact("The user owns a coffee table"), item("coffee table")]
    groups, others = group_entities(recs)
    assert len(groups) == 1
    assert len(others) == 1 and others[0].kind == "FACT"


def test_distinct_entities_do_not_group():
    groups, _ = group_entities([item("Kind of Blue"), item("Bitches Brew")])
    assert len(groups) == 2


def test_group_order_is_first_appearance():
    groups, _ = group_entities([item("zebra"), item("apple"), item("zebra")])
    assert [g.representative.text for g in groups] == ["zebra", "apple"]


def test_grouping_is_deterministic_under_input_permutation():
    """Same set in any order -> same groups with the same members."""
    recs = [item("coffee table", bucket="ACQUISITION"),
            item("Coffee Table", bucket="POSSESSION"),
            item("bookshelf", bucket="POSSESSION")]
    a, _ = group_entities(recs)
    b, _ = group_entities(list(reversed(recs)))
    assert {g.key for g in a} == {g.key for g in b}
    assert {g.key: len(g.records) for g in a} == {g.key: len(g.records) for g in b}


def test_representative_is_the_longest_text():
    """The longest form is the most specific one the extractor produced."""
    g = EntityGroup(key="k", records=[item("coffee table"),
                                      item("Wooden coffee table with metal legs")])
    assert g.representative.text == "Wooden coffee table with metal legs"


def test_representative_ties_break_deterministically():
    g = EntityGroup(key="k", records=[item("bbb"), item("aaa")])
    assert g.representative.text == "bbb"        # (len, text) max -> lexicographically last


def test_group_keeps_every_constituent_record():
    """Lineage must survive. A Record carries one Provenance, and cross-bucket
    co-referents come from different sessions -- so the group holds them all rather than
    synthesising one merged record that would discard all but one session link."""
    recs = [item("coffee table", bucket="ACQUISITION", session="s1"),
            item("coffee table", bucket="POSSESSION", session="s9")]
    groups, _ = group_entities(recs)
    assert {r.provenance.session_id for r in groups[0].records} == {"s1", "s9"}


def test_group_never_synthesises_a_record():
    groups, _ = group_entities([item("coffee table"), item("coffee table")])
    assert all(isinstance(r, Record) for r in groups[0].records)
    assert groups[0].representative in groups[0].records


# --- bucket ordering ----------------------------------------------------------------

def test_primary_bucket_prefers_acquisition_over_possession():
    """Not cosmetic: plain taxonomy order files the coffee table under POSSESSION, while
    the question that gets it wrong is 'how many did I BUY'."""
    groups, _ = group_entities([item("coffee table", bucket="POSSESSION"),
                                item("coffee table", bucket="ACQUISITION")])
    assert groups[0].primary_bucket == "ACQUISITION"


def test_categories_list_every_filing_deduped():
    groups, _ = group_entities([
        item("Kind of Blue", bucket="ACQUISITION", subtype="music album"),
        item("Kind of Blue", bucket="MEDIA", subtype="music album"),
        item("Kind of Blue", bucket="ACQUISITION", subtype="music album"),
    ])
    assert groups[0].categories == ["ACQUISITION/music album", "MEDIA/music album"]


def test_date_is_the_first_non_empty_in_group_order():
    """Chronological selection needs date parsing, which is a separate pre-registered
    change; this must not quietly anticipate it."""
    groups, _ = group_entities([item("x", date=""), item("x", date="2023/05/18"),
                                item("x", date="2023/01/01")])
    assert groups[0].date == "2023/05/18"


# --- the offline diff helper --------------------------------------------------------

def test_duplicate_count_counts_redundant_lines_only():
    recs = [item("a"), item("a"), item("a"), item("b")]
    assert duplicate_count(recs) == 2            # 4 lines -> 2 entities


def test_duplicate_count_is_zero_without_coreferents():
    assert duplicate_count([item("a"), item("b"), fact("c")]) == 0
