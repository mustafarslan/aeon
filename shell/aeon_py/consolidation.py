"""
Aeon consolidation — the write path of the semantic layer (v4-plan.md, PRODUCT DIRECTION).

Turns conversation into durable records at ingest time, so answering later is a lookup
rather than a multi-hop scan over scattered raw turns. Deliberately separate from
`records.py`: that module is pure storage with no LLM dependency (its tests run in
milliseconds); this one owns the model-facing prompts and parsing.

Two stages, matching what the probe validated:

  1. `extract_session()` -- per-session, QUERY-BLIND. Runs once per session as it arrives,
     the way a real consolidator works. It never sees a future question, which is the whole
     risk of the approach: extract-then-compute was query-*conditioned* ("facts relevant to
     this question") and won +23 overall, whereas this has to anticipate.

  2. `consolidate()` -- global merge across accumulated records. Per-session extraction is
     necessarily local; accumulation is global. This is the concrete job `dreamer.py` was
     designed for and currently stubs out with `StubSummarizer`.

Why the closed bucket vocabulary is non-negotiable here: a free-form category field was
tried first and multi-session conversion was 1/8. The evidence was present but did not
accumulate -- independent per-session calls invented their own category names, so of three
albums one became an `ITEM`, one a `PREF` and one an `EVENT`, and counting `ITEM`s returned
1 against a gold of 3. With the closed taxonomy the same question answers 3.

TURN CITATIONS: the extractor is shown numbered turns and asked to cite them, because
provenance at session granularity would be useless -- a session is ~10 turns, and
rehydrating a whole session plus neighbours is just the uncompressed context again. A record
whose citation is missing or unparseable still gets session-level provenance rather than
none, so bad citations degrade the neighbourhood rather than losing the link.
"""

from __future__ import annotations

import re
from typing import Callable, Optional, Sequence

from .records import BUCKETS, KINDS, Provenance, Record

_BUCKET_SET = set(BUCKETS)

BUCKET_DESCRIPTIONS: tuple[tuple[str, str], ...] = (
    ("POSSESSION", "something the user owns or has"),
    ("ACQUISITION", "something acquired: bought, downloaded, received, adopted, won"),
    ("MEDIA", "a book, film, show, album, article or game consumed"),
    ("PERSON", "a person in the user's life, and their relationship"),
    ("EVENT_ATTENDED", "something the user attended, participated in, or completed"),
    ("OBLIGATION", "something still to do, return, pick up, finish or decide"),
    ("EDUCATION_WORK", "schooling, degrees, jobs, roles, and their durations"),
    ("HEALTH", "symptoms, treatments, appointments, habits"),
    ("TRAVEL", "trips, destinations, transport, accommodation"),
    ("PROJECT", "an ongoing project, repair, or home/garden improvement"),
    ("FINANCE", "amounts, prices, budgets, savings, approvals"),
    ("CONSUMABLE", "food, drink, recipes, supplies"),
)
BUCKET_BLOCK = "\n".join(f"    {b} -- {d}" for b, d in BUCKET_DESCRIPTIONS)

EXTRACT_SYSTEM = (
    "You maintain a durable long-term memory record about a user. You are shown ONE "
    "conversation session at a time and you do NOT know what will be asked later, so you "
    "must record anything that could matter. Be exhaustive and literal."
)

EXTRACT_PROMPT = (
    "Session date: {date}\n\n"
    "Conversation (turns are numbered):\n{session}\n\n"
    "Write memory records about the USER, one per line, each ending with the turn numbers "
    "it came from, like  #3  or  #3,4\n\n"
    "COUNTABLE MEMBERS use one of these EXACT bucket names -- never invent a bucket:\n"
    "{buckets}\n\n"
    "  ITEM(<BUCKET>/<short subtype>): <the member> [<date if known>] #<turns>\n\n"
    "Other record types:\n"
    "  FACT: <a durable attribute or relationship> #<turns>\n"
    "  EVENT [{date}]: <something that happened> #<turns>\n"
    "  UPDATE: <a statement revising an earlier one; say what it replaces> #<turns>\n"
    "  PREF: <a stated preference> #<turns>\n"
    "  TASK: <something the user still needs to do, return, pick up or decide> #<turns>\n\n"
    "Rules:\n"
    "- Record only what the user states or clearly implies about themselves.\n"
    "- EMIT AN ITEM LINE FOR EVERY COUNTABLE THING, even when you also record it as a FACT, "
    "EVENT or PREF. One real fact often needs several records: an album the user downloaded "
    "is BOTH ITEM(ACQUISITION/music album) AND ITEM(MEDIA/music album). Never let a countable "
    "thing exist only inside a PREF or EVENT line.\n"
    "- Use the bucket matching HOW the user relates to the thing, not what it is: something "
    "still to collect is OBLIGATION as well as whatever it physically is.\n"
    "- Always include dates when stated or implied, and always cite turn numbers.\n"
    "- If the session contains nothing durable about the user, output exactly: (none)\n\n"
    "Records:"
)

CONSOLIDATE_SYSTEM = (
    "You are consolidating a long-term memory record. You merge and normalise existing "
    "records. You never invent facts that are not present in the input."
)

CONSOLIDATE_PROMPT = (
    "Below are memory records about a user, accumulated from many separate sessions and "
    "therefore inconsistent.\n\n{records}\n\n"
    "Rewrite them into a consolidated record set, preserving each line's trailing @prov tag "
    "exactly:\n"
    "1. PROMOTE: if a FACT, EVENT or PREF line mentions a countable thing that has no ITEM "
    "line, add the missing ITEM line using the exact buckets below, copying its @prov tag.\n"
    "2. MERGE: give near-duplicate subtypes inside a bucket one consistent name "
    "(\"music album/EP\", \"album\", \"vinyl\" -> one subtype); drop exact duplicates.\n"
    "3. RESOLVE: if the same real thing appears under several names, keep one line.\n"
    "4. SUPERSEDE: where an UPDATE revises an earlier record, keep the current value and "
    "mark it, e.g. 'ITEM(FINANCE/pre-approval): $400,000 [supersedes $350,000]'.\n\n"
    "Buckets:\n{buckets}\n\n"
    "Preserve every distinct fact -- this is normalisation, not summarisation. Output the "
    "consolidated records only, one per line.\n\nConsolidated records:"
)

# ITEM(BUCKET/subtype): text   |   FACT: text   |   EVENT [date]: text
_ITEM_RE = re.compile(r"^ITEM\s*\(\s*([A-Z_]+)\s*(?:/\s*([^)]*))?\)\s*:\s*(.*)$", re.S)
# The model frequently drops the ITEM(...) wrapper and writes the bucket as the kind:
# "HEALTH: user averages 7 hours of sleep", "OBLIGATION/decision: choose a drive",
# "EVENT_ATTENDED [2023/08/14]: auto racking event". Measured on 4,504 real record lines
# from the probe, rejecting this shorthand discarded 149 records (3.3%) including entire
# HEALTH and EVENT_ATTENDED categories. It is unambiguous precisely because the bucket
# vocabulary is closed, so it is accepted rather than fought.
_BARE_BUCKET_RE = re.compile(
    r"^([A-Z_]+)\s*(?:/\s*([^:\[]*))?\s*(?:\[([^\]]*)\])?\s*:\s*(.*)$", re.S)
_KIND_RE = re.compile(r"^(FACT|EVENT|UPDATE|PREF|TASK)\s*(?:\[([^\]]*)\])?\s*:\s*(.*)$", re.S)
_TURNS_RE = re.compile(r"#\s*([0-9]+(?:\s*[,\-]\s*[0-9]+)*)\s*$")
_DATE_RE = re.compile(r"\[([0-9]{4}/[0-9]{2}/[0-9]{2}[^\]]*)\]")
_SUPERSEDES_RE = re.compile(r"\[supersedes\s+([^\]]+)\]", re.I)
_PROV_RE = re.compile(r"\s*@prov:([^\s]+)\s*$")


def number_turns(turns: Sequence[dict]) -> str:
    """Render a session with explicit turn numbers so the extractor can cite them."""
    return "\n".join(
        f"[{i}] {t.get('role', 'user')}: {t.get('content', '')}" for i, t in enumerate(turns)
    )


def _parse_turn_citation(text: str) -> tuple[str, tuple[int, ...]]:
    """Strip a trailing `#3` / `#3,4` / `#3-5` citation and return (text, indices)."""
    m = _TURNS_RE.search(text)
    if not m:
        return text.strip(), ()
    body = text[: m.start()].strip()
    idx: list[int] = []
    for part in re.split(r"[,\s]+", m.group(1)):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, _, b = part.partition("-")
            try:
                lo, hi = int(a), int(b)
                if hi - lo <= 64:          # guard against a malformed huge range
                    idx.extend(range(lo, hi + 1))
            except ValueError:
                continue
        else:
            try:
                idx.append(int(part))
            except ValueError:
                continue
    return body, tuple(sorted(set(idx)))


def parse_record_line(line: str, session_id: str, *,
                      fallback_indices: tuple[int, ...] = (),
                      n_turns: Optional[int] = None) -> Optional[Record]:
    """Parse one model-emitted line into a `Record`, or None if it is not a record.

    Tolerant by design: the model produces these under a query-blind prompt across hundreds
    of sessions, and a single malformed line must never abort a consolidation pass. Anything
    unrecognised returns None; a record with a missing or out-of-range citation still gets
    `fallback_indices` so its provenance link degrades rather than disappears.
    """
    line = (line or "").strip().lstrip("-*• ").strip()
    if not line or line.lower() in {"(none)", "none", "records:", "consolidated records:"}:
        return None

    body, turns = _parse_turn_citation(line)
    if n_turns is not None and turns:
        turns = tuple(i for i in turns if 0 <= i < n_turns)
    if not turns:
        turns = fallback_indices

    supersedes = ""
    ms = _SUPERSEDES_RE.search(body)
    if ms:
        supersedes = ms.group(1).strip()
        body = _SUPERSEDES_RE.sub("", body).strip()

    m = _ITEM_RE.match(body)
    if m:
        bucket = m.group(1).strip().upper()
        if bucket not in _BUCKET_SET:
            # An invented bucket is exactly the failure the closed vocabulary exists to
            # prevent; keep the content as a FACT rather than minting a category that will
            # never accumulate with anything.
            return _finish("FACT", body[body.find(":") + 1:].strip(), "", "",
                           session_id, turns, supersedes)
        return _finish("ITEM", m.group(3).strip(), bucket,
                       (m.group(2) or "").strip(), session_id, turns, supersedes)

    m = _KIND_RE.match(body)
    if m:
        kind = m.group(1).upper()
        inline_date = (m.group(2) or "").strip()
        text = m.group(3).strip()
        return _finish(kind, text, "", "", session_id, turns, supersedes,
                       explicit_date=inline_date)

    # Bare-bucket shorthand, e.g. "HEALTH: ..." / "OBLIGATION/decision: ..."
    m = _BARE_BUCKET_RE.match(body)
    if m and m.group(1).strip().upper() in _BUCKET_SET:
        return _finish("ITEM", m.group(4).strip(), m.group(1).strip().upper(),
                       (m.group(2) or "").strip(), session_id, turns, supersedes,
                       explicit_date=(m.group(3) or "").strip())
    return None


def _finish(kind: str, text: str, bucket: str, subtype: str, session_id: str,
            turns: tuple[int, ...], supersedes: str,
            explicit_date: str = "") -> Optional[Record]:
    date = explicit_date
    if not date:
        md = _DATE_RE.search(text)
        if md:
            date = md.group(1).strip()
            text = _DATE_RE.sub("", text).strip()
    else:
        text = _DATE_RE.sub("", text).strip()
    text = text.strip(" -–—")
    if not text or kind not in KINDS:
        return None
    return Record(kind=kind, text=text, bucket=bucket, subtype=subtype, date=date,
                  provenance=Provenance(session_id, turns), supersedes=supersedes)


def parse_records(output: str, session_id: str, *,
                  n_turns: Optional[int] = None) -> list[Record]:
    """Parse a full extraction response. Records citing no turn fall back to whole-session
    provenance so the link survives even when the citation does not."""
    fallback = tuple(range(n_turns)) if n_turns else ()
    out: list[Record] = []
    for line in (output or "").splitlines():
        rec = parse_record_line(line, session_id, fallback_indices=fallback,
                               n_turns=n_turns)
        if rec is not None:
            out.append(rec)
    return out


def extract_session(turns: Sequence[dict], session_id: str, date: str,
                    generate: Callable[..., str]) -> list[Record]:
    """Query-blind extraction of one session into records.

    `generate(prompt, system_prompt=..., temperature=...)` is injected rather than imported
    so this is testable without a model and works with any provider.
    """
    if not turns:
        return []
    prompt = EXTRACT_PROMPT.format(date=date, session=number_turns(turns),
                                   buckets=BUCKET_BLOCK)
    out = generate(prompt, system_prompt=EXTRACT_SYSTEM, temperature=0.0)
    if not out or "[System Error:" in out:
        return []
    return parse_records(out, session_id, n_turns=len(turns))


def render_for_consolidation(records: Sequence[Record]) -> str:
    """Render records for the merge pass with a machine-readable provenance tag, so the
    model can carry provenance across a rewrite it would otherwise drop."""
    return "\n".join(f"{r.display()} @prov:{r.provenance.encode()}" for r in records)


def parse_consolidated(output: str, fallback_session: str = "") -> list[Record]:
    """Parse the merge pass's output, restoring provenance from the @prov tags."""
    out: list[Record] = []
    for line in (output or "").splitlines():
        line = line.rstrip()
        prov = Provenance(fallback_session)
        m = _PROV_RE.search(line)
        if m:
            prov = Provenance.decode(m.group(1))
            line = _PROV_RE.sub("", line)
        rec = parse_record_line(line, prov.session_id)
        if rec is not None:
            rec.provenance = prov
            out.append(rec)
    return out


def consolidate(records: Sequence[Record], generate: Callable[..., str]) -> list[Record]:
    """Global merge across accumulated records -- the Dreamer's real job.

    Returns the input unchanged if the model errors or returns a suspiciously short result:
    consolidation is normalisation, not summarisation, and a pass that collapses the record
    set is a failure, not a compression win.
    """
    if not records:
        return []
    rendered = render_for_consolidation(records)
    out = generate(CONSOLIDATE_PROMPT.format(records=rendered, buckets=BUCKET_BLOCK),
                   system_prompt=CONSOLIDATE_SYSTEM, temperature=0.0)
    if not out or "[System Error:" in out:
        return list(records)
    merged = parse_consolidated(out)
    if len(merged) < max(1, int(len(records) * 0.3)):
        return list(records)
    return merged
