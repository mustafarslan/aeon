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

from typing import Iterable, Optional, Sequence

from .records import Record

RECORDS_HEADER = "Long-term memory records about the user:"
EPISODIC_HEADER = "Relevant conversation excerpts:"

COMPOSE_SYSTEM = (
    "You are answering a question using a long-term memory record about the user, plus "
    "excerpts of the original conversation. If the information genuinely is not there, say so "
    "plainly instead of guessing. Answer as concisely as possible -- a short phrase or "
    "sentence, not a paragraph."
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
_PREMISE_GUARD = (
    "First check whether the records and excerpts actually mention the specific thing the "
    "question asks about. If they do not, say that the information is not available and stop -- "
    "do not answer with a count of zero, do not total up the parts that are present, and do not "
    "substitute a related fact."
)


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
    return items + updates + rest


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


def render_records(records: Iterable[Record]) -> str:
    recs = order_records(records)
    return "\n".join(r.display() for r in recs) if recs else "(no records)"


def compose(records: Iterable[Record], episodic_lines: Sequence[str], question_block: str,
            *, counting_hint: bool = True, premise_guard: bool = True) -> str:
    """Assemble the single-call prompt. `question_block` is pre-rendered by the caller so the
    reference date travels with the question -- a field this project measured as worth ~19
    questions when it was missing."""
    parts = [RECORDS_HEADER, render_records(records), ""]
    if episodic_lines:
        parts += [EPISODIC_HEADER, "\n".join(episodic_lines), ""]
    parts.append(question_block)
    parts.append("")
    tail = "Answer using the records and excerpts above."
    if premise_guard:
        tail += " " + _PREMISE_GUARD
    if counting_hint:
        tail += " " + _COUNTING_HINT
    parts.append(tail)
    parts.append("\nAnswer:")
    return "\n".join(parts)


def compose_from_store(store, trace, query_embedding, question_block: str, *,
                       neighbours: int = 1, max_turns: int = 12,
                       buckets: Optional[Sequence[str]] = None) -> dict:
    """Production-shaped assembly: all records (optionally category-filtered), with the
    episodic component rehydrated from those records' own provenance.

    Sourcing episodic context from provenance rather than from an independent retrieval pass
    means the excerpts are guaranteed to be about the records in play, and it reuses the link
    the write path already stored instead of paying for a second search.
    """
    records = store.all_records()
    if buckets:
        records = select_records(records, buckets=buckets)
    episodic = store.rehydrate(records, trace, neighbours=neighbours, max_turns=max_turns)
    prompt = compose(records, episodic, question_block)
    return {"prompt": prompt, "records": records, "episodic": episodic,
            "record_count": len(records), "chars": len(prompt)}
