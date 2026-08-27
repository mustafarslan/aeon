"""
Composite context assembly — the semantic layer's read path.

One LLM call over two DIFFERENT information sources:

    consolidated records   (semantic: what the conversation means, accumulated)
  + budgeted episodic turns (episodic: what was actually said, verbatim)

This is a composite, NOT a router, and that distinction is empirical. Routing between arms
was measured three separate times in this project and never beat always-on: the type-label
oracle ceiling sat at or inside the noise band every time (+1.4, +7, +12 questions against
thresholds of ~11.5), and the one trained classifier scored *below* its own oracle. Those
arms all carried the SAME information, so choosing between them could only lose. Records and
raw turns do not: each answers questions the other cannot, and records cost only a few
hundred characters, so both are always included.

Why each source is needed, from measurements rather than intuition:

  * Records alone fail the "what brand?" class. Compression strips pragmatic licensing --
    "picked it up at Trader Joe's" answers "what brand?" only by conversational implicature --
    which cost 12 questions when extract-then-compute compressed to bare facts.
  * Episodic alone fails the counting class. An ORACLE handed the gold turns directly still
    answered "how many albums have I purchased?" with 2 against a gold of 3, because counting
    scattered mentions is multi-hop. A maintained ITEM set answers it by enumeration.
  * In the probe, records+episodic converted more of the hard cohort than records alone
    (5 vs 4 genuine), and every case the composite won and records-alone lost needed a number
    that lived in the raw turn ("save ~$50, as a taxi is around $60").

ALL RECORDS ARE INCLUDED, not a vector top-k over them. Top-k retrieval over records would
fetch 3 of 5 `ITEM(ACQUISITION/music album)` lines and reintroduce partial recall on exactly
the counting questions this layer exists to fix -- the old failure in new clothes. Records are
small (~21k chars for 46 sessions of history), so completeness is affordable. For histories
long enough that it stops being affordable, the scaling path is a STRUCTURED CATEGORY SCAN --
all ITEMs in a bucket/subtype, which is a cheap filter over small records -- not vector
retrieval. `select_records()` implements that filter; it is not yet needed at benchmark scale.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional, Sequence

from .entities import EntityGroup, group_entities
from .records import Record

RECORDS_HEADER = "Long-term memory records about the user:"
EPISODIC_HEADER = "Relevant conversation excerpts:"
TIMELINE_HEADER = "How these changed over time:"

# MEASURED HARMFUL (2026-08-27) -- do NOT wire this into the read path as written.
#
# Every composite number in v4-plan.md (429/500, the abstention breach, the 401) was
# produced with `system_prompt=""`. This constant existed but was referenced nowhere, so
# it had never been measured. Measured now, on the dev half (n=252, records frozen,
# n_errors=0): sending it scores **207 against the empty prompt's 220 -- -13, McNemar
# +3/-16, p=0.0044**. Extrapolated, roughly -26 at n=500.
#
# The cause is visible in the outputs, not inferred: median answer length collapses from
# 172 characters to 49. The clause "Answer as concisely as possible -- a short phrase or
# sentence, not a paragraph" truncates exactly the two types that need room --
# single-session-preference -5 (its judge is a RUBRIC that rewards reflecting the user's
# stated preferences, which a short phrase cannot do) and temporal-reasoning -4 (which
# needs the arithmetic shown). Abstention, ss-user and ss-assistant are unchanged.
#
# Kept rather than deleted, with the measurement attached, because a production read path
# will eventually want SOME system prompt and this records what a revised one must avoid.
# Anyone wiring `compose_from_store()` into `loop.py` should treat this as a live trap:
# the library ships a 26-question regression that only fires once someone sends it.
COMPOSE_SYSTEM = (
    "You are answering a question using a long-term memory record about the user, plus "
    "excerpts of the original conversation. If the information genuinely is not there, say so "
    "plainly instead of guessing. Answer as concisely as possible -- a short phrase or "
    "sentence, not a paragraph."
)

# Reconciliation-aware alternative to _COUNTING_HINT. Under test, not the default.
#
# The hypothesis: _COUNTING_HINT's "COUNT the matching records" is what drives the
# supersession sums (`4b24c848` -> 3+5=8 against a gold of 5). Implicated but NOT
# convicted -- counting two matching records literally yields 2, not 8; the model is
# summing quantities *inside* the records, which the hint does not command.
#
# The free ablation (`scripts/longmemeval/counting_hint_probe.py`) could not settle it:
# targets went 2/6 current, 3/6 no-hint, 4/6 reconcile, but 2 of 9 re-run cells disagreed
# with their own dev label under identical config -- 22% self-disagreement at n=9 against
# 3.0% at scale, so that swing is inside the noise. What the probe DID show is that on
# `4b24c848` the directive is read and still loses: the model answers "a total of 8 tops
# (three on 2023/08/11 and five on 2023/09/30)", classifying them as distinct additions
# rather than a revised total. That is exactly the judgement bullet one asks it to make.
#
# Scoped to counting questions on purpose. `_PREMISE_GUARD` failed because it was
# UNSCOPED and fired on advice questions, costing 19 preference questions; question-shape
# scoping is the separation that post-mortem demanded.
#
# MEASURED AND REVERTED TO OPT-IN (2026-08-27). On dev (n=252, records frozen,
# n_errors=0): **220 against the 221 baseline -- -1, McNemar +6/-7, p=1.000**, against a
# bar of >=227. knowledge-update, the type it targets, moved +2 (32->34) -- but at n=36
# that type's own 2x sd is ~2.1, so +2 is at the edge and does not clear it.
#
# The target check failed for the third consecutive intervention, and identically:
# `4b24c848` answers "a total of 8 tops from H&M (three tops on 2023/08/11 and five tops
# on 2023/09/30)" -- the same shape as under the original hint and under chronological
# ordering. The directive is present, read, and loses. The model classifies the two
# records as distinct additions rather than a revised total, which is exactly the
# judgement the directive asks it to make.
#
# That is now four nulls from four directions -- co-referent collapse, chronological
# ordering, the shipped system prompt, and this. Instructions do not change how the
# reader CLASSIFIES; that is the `_PREMISE_GUARD` lesson arriving again.
_RECONCILE_HINT = (
    "If the question asks for a count or total quantity: if several records give updated "
    "totals or revised statuses for the same item or activity on different dates, use the "
    "latest figure rather than summing the historical ones; if the records describe "
    "separate, distinct additions or events, sum them; and if the item or activity is "
    "never mentioned at all, say there is no record of it rather than answering zero."
)

_COUNTING_HINT = (
    "If the question asks how many, COUNT the matching records and show the count."
)

# Measured at n=500 (v4-plan.md): the composite lost 6 abstention questions to ETC, 0 won,
# p=0.031. Reading all six, five identify the unsupported premise and then override themselves
# with a committed answer -- "Answer: 0", "Total: 10 days", "Total: 10 years", a computed
# "4 years and 9 months", "you completed fixing the fence first". The cause is structural, not
# formatting: ETC abstains because its extract stage found nothing to compute with, while the
# composite always has ~270 records in context and so always has SOMETHING to compute over.
# _COUNTING_HINT makes it worse on exactly this class -- three of the six are count/total
# questions, and "COUNT the matching records" over zero matching records yields a confident 0.
# So the guard is stated BEFORE the counting hint, which is thereby conditional on the premise
# holding. It names the three override forms actually observed rather than any specific question.
#
# TRIED AND REVERTED (2026-08-27) -- kept, defaulted OFF, as a recorded negative result, and
# `premise_guard=True` still enables it. It did exactly what it was designed to do and cost more
# than it bought: abstention 22 -> 30 of 30 (+8, 8 wins to 0 losses, p=0.008), and the run overall
# 429 -> 401 (-28, p=0.0018), failing the pre-registered primary bar of >=425. One attempt, no
# wording iteration, per the pre-committed protocol.
#
# The damage is concentrated in single-session-preference: 28 -> 9, ZERO wins to 19 losses.
# Those questions are ADVICE questions ("what should I meal-prep next week?", "tips for keeping
# the kitchen clean?"), graded on whether the response reflects the user's stated preferences.
# The right behaviour is to SYNTHESISE advice grounded in the records -- but the records never
# contain "recipe suggestions for next week" verbatim, so the guard fires and the model refuses:
# "The provided records and excerpts do not mention specific new recipe suggestions."
#
# The lesson is a distinction the guard cannot draw: "the question asks for a fact the records
# do not contain" (abstain) versus "the question asks for advice the records inform"
# (synthesise) look identical to a literal premise check. Any future attempt must separate those
# two, not phrase the same check more carefully.
#
# Note also that the pre-registered RISK was named for the wrong cohort. The bar guarded the
# known-miss 27, on the theory that suppressing commitment would suppress it where the fact is
# present; that cohort HELD at 22/27. Over-abstention was the right prediction, the location was
# wrong -- it landed where the answer is not a stored fact at all.
_PREMISE_GUARD = (
    "First check whether the records and excerpts actually mention the specific thing the "
    "question asks about. If they do not, say that the information is not available and stop -- "
    "do not answer with a count of zero, do not total up the parts that are present, and do not "
    "substitute a related fact."
)


# --- chronology (v4.1 Stage 2b) -----------------------------------------------------
#
# `Record.date` has existed since the schema was written, is populated on 31% of the
# cached corpus, and until now was read by NOTHING -- not this module's sort key, not
# `select_records()`, not any date parsing anywhere in the semantic layer.
#
# That is measurably load-bearing. The knowledge-update failures are not missing
# information: `4b24c848` holds "three tops from H&M" [2023/08/11] AND "five tops from
# H&M" [2023/09/30] and answers 8 against a gold of 5; `5831f84d` holds "finished 10
# Crash Course videos" [2023/08/11] and "watched 15 Crash Course videos" [2023/09/30]
# and answers 10 + 12 + 15 = 37 against a gold of 15. Everything needed is present and
# dated. The reader sums a sequence of running totals because nothing tells it they ARE
# a sequence.
#
# So chronology is made legible rather than adjudicated. Deliberately NOT done here:
# suppressing the earlier record. A mechanical "later value supersedes earlier" rule was
# prototyped against all 500 corpora before writing any of this -- it fires on 3
# questions, suppresses 4 records, and at least one firing is a plain false positive
# ("10-gallon tank" and "20-gallon tank" are two tanks, not a revision). Deleting a
# record on a coin flip converts a knowledge-update error into an undercount, and there
# are already more undercounts than overcounts.

_ISO_DATE = re.compile(r"(\d{4})[/-](\d{2})(?:[/-](\d{2}))?")
_YEAR_ONLY = re.compile(r"\b(\d{4})\b")
_MONTHS = {m: i for i, m in enumerate(
    ("january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"), start=1)}
_SEASONS = {"winter": 1, "spring": 3, "summer": 6, "autumn": 9, "fall": 9}


def _date_key(date: str) -> tuple:
    """Sortable key for a record date. Returns `(sortable, y, m, d)` where `sortable` is
    0 for a parseable date and 1 otherwise, so undated records sort LAST and -- because
    every sort here is stable -- keep their existing relative order among themselves.

    Coarse dates sort at the start of their period ("2023" before "2023/05/20"), which is
    the only defensible reading: a record that says only "2023" cannot be placed later
    than one that names a day.

    Prose dates are parsed here rather than rewritten on write. "spring 2023" stays
    verbatim in the record; guessing `2023/03/01` at write time would invent precision the
    user never stated and destroy the original string, whereas being wrong HERE costs an
    ordering, not a fact.
    """
    if not date:
        return (1, 0, 0, 0)
    d = date.strip().lower()
    m = _ISO_DATE.search(d)
    if m:
        return (0, int(m.group(1)), int(m.group(2)), int(m.group(3) or 0))
    y = _YEAR_ONLY.search(d)
    if y:
        year = int(y.group(1))
        for name, mon in _MONTHS.items():
            if name in d:
                return (0, year, mon, 0)
        for name, mon in _SEASONS.items():
            if name in d:
                return (0, year, mon, 0)
        return (0, year, 0, 0)
    return (1, 0, 0, 0)


def order_records(records: Iterable[Record]) -> list[Record]:
    """Group records so countable members of one category are adjacent.

    Counting is the failure mode this layer targets, and a model counting `ITEM` lines
    scattered through 250 unordered lines is doing the same multi-hop scan over a smaller
    haystack. Grouping by bucket/subtype turns it into reading one contiguous list.
    """
    recs = list(records)
    items = [r for r in recs if r.kind == "ITEM"]
    others = [r for r in recs if r.kind != "ITEM"]
    items.sort(key=lambda r: (r.bucket, r.subtype.lower(), r.text.lower()))
    # UPDATE records carry supersession and are cheap; keep them where the model will read
    # them before falling back to the raw episodic text.
    updates = [r for r in others if r.kind == "UPDATE"]
    rest = [r for r in others if r.kind != "UPDATE"]
    # Chronological order for the kinds that assert something happened AT a time. ITEM
    # ordering is untouched -- it is the counting-critical bucket grouping, and reordering
    # it by date would scatter the contiguous category runs that grouping exists to build.
    updates.sort(key=lambda r: _date_key(r.date))
    events = [r for r in rest if r.kind == "EVENT"]
    non_events = [r for r in rest if r.kind != "EVENT"]
    events.sort(key=lambda r: _date_key(r.date))
    return items + updates + events + non_events


def select_records(records: Iterable[Record], *, buckets: Optional[Sequence[str]] = None,
                   subtype_contains: str = "") -> list[Record]:
    """Structured filter over records -- the scaling path when all-records stops fitting.

    A category scan, not a similarity search: it returns EVERY member of a matching category,
    which is what counting requires. Vector top-k would silently return a subset and turn a
    complete answer into a plausible wrong one.
    """
    out = []
    want = {b.upper() for b in buckets} if buckets else None
    needle = subtype_contains.lower().strip()
    for r in records:
        if want is not None and r.bucket.upper() not in want:
            continue
        if needle and needle not in r.subtype.lower():
            continue
        out.append(r)
    return out


def _render_group(group: EntityGroup) -> str:
    """One real entity, one line, listing every category it is filed under.

    For a group of one this is byte-identical to `Record.display()` -- deliberately, so
    that a corpus with no co-referents renders exactly as it did before this change
    existed (pinned by `test_tree_refactor_does_not_change_rendered_context`).
    """
    rep = group.representative
    cats = group.categories
    head = f"ITEM({' | '.join(cats)})" if cats else "ITEM"
    date = f" [{group.date}]" if group.date else ""
    sup = f" [supersedes {group.supersedes}]" if group.supersedes else ""
    return f"{head}: {rep.text}{date}{sup}"


def _group_sort_key(group: EntityGroup) -> tuple:
    """Deliberately the SAME shape as `order_records()`'s ITEM key, and deliberately a
    string comparison on the bucket rather than a taxonomy rank.

    That is what makes the equivalence exact rather than lucky: for an entity filed under
    a single bucket, `primary_bucket` IS its bucket and `representative` IS its record, so
    the key is byte-for-byte the old one and the ordering cannot move. The rank in
    `entities._PRIMARY_RANK` only chooses WHICH bucket represents a multi-bucket entity --
    it never reorders anything that was not duplicated in the first place.
    """
    rep = group.representative
    return (group.primary_bucket, rep.subtype.lower(), rep.text.lower())


def render_records(records: Iterable[Record]) -> str:
    """Renders the record block, collapsing co-referent ITEMs to one line each.

    Cross-bucket duplication is a FEATURE of the taxonomy -- an album the user downloaded
    is genuinely both an ACQUISITION and a MEDIA item, and that is how both questions find
    it. It is only a defect at COUNT time, where the reader sees one object twice. So the
    collapse happens here, at render, and nowhere else: `select_records()` and
    `compose_from_store()`'s rehydration still see every individual record, so filtering
    and provenance are untouched.
    """
    recs = order_records(records)
    groups, others = group_entities(recs)
    if not groups and not others:
        return "(no records)"
    groups.sort(key=_group_sort_key)
    lines = [_render_group(g) for g in groups]
    lines += [r.display() for r in others]
    return "\n".join(lines)


def compose(records: Iterable[Record], episodic_lines: Sequence[str], question_block: str,
            *, counting_hint: bool = True, premise_guard: bool = False,
            reconcile_hint: bool = False, timeline: bool = False,
            retired: Sequence[Record] = ()) -> str:
    """Assemble the single-call prompt. `question_block` is pre-rendered by the caller so the
    reference date travels with the question -- a field this project measured as worth ~19
    questions when it was missing."""
    recs = list(records)
    parts = [RECORDS_HEADER, render_records(recs), ""]
    # ACTIVE STATE IS THE DEFAULT AND STAYS THE DEFAULT. `timeline=False` must produce output
    # byte-identical to the measured renderer -- pinned by
    # `test_timeline_flag_off_is_byte_identical`. A factual or counting question therefore
    # never sees a historical value: the stale record is absent from `records` (removed by
    # the merge, excluded by `_decode()`), not merely deprecated within it, which is the
    # whole reason 852ce960 stopped answering $350,000.
    #
    # Timeline is opt-in because it is the fifth reader-side legibility intervention in a
    # family with four nulls. It shows the model *both* values on purpose, which is exactly
    # what active-state projection exists to avoid -- so it is for "how did X change over
    # time", and for nothing else.
    if timeline:
        from .timeline import build_chains, render_timeline
        chains, _ = build_chains(recs, retired)
        if chains:
            parts += [TIMELINE_HEADER, render_timeline(chains), ""]
    if episodic_lines:
        parts += [EPISODIC_HEADER, "\n".join(episodic_lines), ""]
    parts.append(question_block)
    parts.append("")
    tail = "Answer using the records and excerpts above."
    if premise_guard:
        tail += " " + _PREMISE_GUARD
    if reconcile_hint:
        tail += " " + _RECONCILE_HINT
    elif counting_hint:
        tail += " " + _COUNTING_HINT
    parts.append(tail)
    parts.append("\nAnswer:")
    return "\n".join(parts)


def compose_from_store(store, trace, question_block: str, *,
                       neighbours: int = 1, max_turns: int = 12,
                       buckets: Optional[Sequence[str]] = None) -> dict:
    """Production-shaped assembly: all records (optionally category-filtered), with the
    episodic component rehydrated from those records' own provenance.

    Sourcing episodic context from provenance rather than from an independent retrieval pass
    means the excerpts are guaranteed to be about the records in play, and it reuses the link
    the write path already stored instead of paying for a second search.

    `query_embedding` was a parameter of this function and was never read -- removed rather
    than left as a signature that implies a vector search this path does not do.
    """
    # The bucket filter stays a Python scan, and that is now a DELIBERATE choice rather than
    # an oversight. `records.py`'s docstring calls the category scan "a kernel subtree walk,
    # not a Python filter" and names it the scaling path -- but the walk is unsound for any
    # realistic write order. Atlas requires a node's children to be PHYSICALLY ADJACENT in
    # the file, and `insert()` appends at the tail, so a bucket's child block can only grow
    # while that bucket is the last thing written. Measured: write MEDIA, POSSESSION, MEDIA
    # and `records_in_bucket("MEDIA")` returns ONE of the two -- while `all_records()`
    # returns all three. Extraction emits records across many buckets per session, so
    # interleaving is the normal case, not the exception.
    #
    # A silent subset is the one failure this layer must never have: counting is what it
    # exists for, and a partial category scan turns a complete answer into a plausible wrong
    # one. So the O(n) filter is CORRECT and the O(subtree) walk is BROKEN, and correctness
    # wins until the kernel can hold a non-contiguous child set.
    records = store.all_records()
    if buckets:
        records = select_records(records, buckets=buckets)
    episodic = store.rehydrate(records, trace, neighbours=neighbours, max_turns=max_turns)
    prompt = compose(records, episodic, question_block)
    return {"prompt": prompt, "records": records, "episodic": episodic,
            "record_count": len(records), "chars": len(prompt)}
