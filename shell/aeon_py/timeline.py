"""Temporal property graph overlay: supersession made traversable.

WHAT THIS IS. One relation -- supersession -- expressed today in three disconnected places
that never meet:

  * `Record.supersedes` -- free-text prose ("three tops from H&M"), no id, no link;
  * `Atlas.supersede_node()` -- a flag and a float, recording **no target**, so nothing can
    answer "what replaced this?";
  * `TraceEvent.edge_type` / `supersedes_id` -- a real durable edge with five writers and,
    until now, **zero readers**.

This module joins them. It is a DERIVED OVERLAY: no kernel change, no WAL record type, no
`NodeHeader` field, and no eighth `Record` field -- `decode()` does `text=_SEP.join(parts[6:])`,
so an appended field would be silently swallowed into `text` by every existing reader. The
precedent is `entities.canonical_key`: derive what is a pure function of what is already
stored, and store only what cannot be derived.

WHAT IT BUYS, MEASURED, AND WHAT IT DOES NOT.

  * Active-state projection -- ALREADY DELIVERED by `consolidate()` + `run_merge()`, which
    remove the stale record rather than deprecating it in place. This module adds only
    *reversibility*: you can now enumerate what was retired and `revoke_node_supersede()` it.
    Do not claim more.
  * Timeline projection -- NEW, and off by default. It is the fifth reader-side legibility
    intervention in a family with four nulls, so it carries no accuracy claim.
  * "What superseded this?" -- NEW. Today that question needs a full audit-log scan.
  * Erasure lineage -- constituent records keep their own provenance, because a synthesized
    merged record carries ONE `Provenance` and merging N records would destroy N-1 lineage
    links, which is the right-to-erasure cascade index.

THE RESOLUTION RULE IS FROZEN, and its hit rate is stated rather than assumed -- measured on
the full merged corpus available (`records_merged_dev.json`, 55 dev questions), not on the
20-question sample the design was drafted against:

    markers   39
    exact      7      linked: 12 of 39  =  31%
    partial    5
    dangling   5      asserted but not executed by the merge
    unresolved 22

**The number got WORSE with more data** -- the first estimate on 20 questions was 4 of 9 (44%).
The larger measurement supersedes it, and is recorded here rather than the flattering one.

The three failure modes, from the corpus: **bare dates** as markers ("2023/05/11", "October
2023") have no key to match on, since `canonical_key` strips bracketed spans; **paraphrases**
("$350,000 pre-approval" against a record reading "$350,000 from Wells Fargo") miss the
substring test; and **dangling** markers name a record the merge kept.

31% is what the derived resolver is worth, and it is the reason `run_merge()` writes a durable
edge at the one moment the pairing is known exactly -- the edge log is not a nicety on top of
the resolver, it is the answer to the resolver being weak. Tuning the rule to raise 31%
against these questions would be fitting the test set: the same refusal `entities.py` records,
with the ETC v3 precedent behind it.

CHAINS COME FROM ASSERTED MARKERS ONLY. The same merge retired 593 records while asserting 9
supersessions -- 384 of those retirements reappear in the after-set as pure dedup. Building
chains from the retired set would fabricate ~584 false chains per 20 questions. And mechanical
chain *discovery* by date-and-quantity heuristics was already prototyped against all 500
corpora and rejected: it fires on 3 questions, and one firing is a plain false positive
("10-gallon tank" and "20-gallon tank" are two tanks, not a revision).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from .entities import canonical_key
from .records import Record
from .trace import EdgeType, ReasonCode, event_time_of

# Edges live in their own tenant-namespaced session. That is what solves the "five writers,
# zero readers" problem without a kernel change: `get_history()` is the only exposed way to
# read `edge_type`/`supersedes_id`, it requires a session id, and it has no filter -- so
# giving the edges a session containing nothing else turns "linear-scan every session" into
# one bounded call.
TPG_SESSION_PREFIX = "__tpg__:"


def edge_session(tenant: str) -> str:
    """The session id holding one tenant's supersession edges.

    Tenant-namespaced so edges are dropped with their tenant: `drop_session()` on teardown
    must include this session, or a deleted tenant's lineage outlives it.
    """
    return f"{TPG_SESSION_PREFIX}{tenant}"


@dataclass(frozen=True)
class Link:
    """One asserted supersession, resolved to records where possible."""

    successor: Record
    predecessor: Optional[Record]
    marker: str
    confidence: str          # "exact" | "partial" | "unresolved"
    dangling: bool = False   # the target is still LIVE -- asserted but not executed


@dataclass
class Chain:
    """One entity's versions, oldest to newest."""

    versions: list[Record] = field(default_factory=list)

    @property
    def head(self) -> Record:
        return self.versions[-1]


def _date_sort_key(record: Record):
    # Imported lazily: `compose` imports this module's renderer, and a module-level import
    # the other way would be a cycle.
    from .compose import _date_key
    return _date_key(record.date)


def resolve_links(current: Sequence[Record],
                  retired: Sequence[Record] = ()) -> list[Link]:
    """Resolve each `[supersedes X]` marker to the record it names.

    THE RULE IS FROZEN -- three steps, and an ambiguous match is left unresolved rather than
    guessed:

      1. `canonical_key(marker)` equals `canonical_key(candidate.text)` -> "exact".
      2. the marker key is a substring of EXACTLY ONE candidate key -> "partial".
      3. otherwise "unresolved", and the marker is still rendered verbatim, exactly as today.

    A marker resolving to a record in `current` rather than `retired` is `dangling`: the merge
    asserted a supersession it did not execute (1 of 9 measured). **Surfaced, never
    suppressed** -- suppressing on that signal is precisely the mechanical rule that was
    prototyped and rejected, and a mis-resolution there deletes a fact.
    """
    candidates = [(canonical_key(r.text), r, True) for r in retired]
    candidates += [(canonical_key(r.text), r, False) for r in current]
    links: list[Link] = []
    for rec in current:
        if not rec.supersedes:
            continue
        key = canonical_key(rec.supersedes)
        if not key:
            links.append(Link(rec, None, rec.supersedes, "unresolved"))
            continue
        hits_exact = [(r, ret) for k, r, ret in candidates if k == key and r is not rec]
        if hits_exact:
            r, was_retired = hits_exact[0]
            links.append(Link(rec, r, rec.supersedes, "exact", dangling=not was_retired))
            continue
        hits_sub = [(r, ret) for k, r, ret in candidates
                    if r is not rec and k and (key in k or k in key)]
        if len(hits_sub) == 1:
            r, was_retired = hits_sub[0]
            links.append(Link(rec, r, rec.supersedes, "partial", dangling=not was_retired))
            continue
        links.append(Link(rec, None, rec.supersedes, "unresolved"))
    return links


def build_chains(current: Sequence[Record],
                 retired: Sequence[Record] = ()) -> tuple[list[Chain], list[Link]]:
    """Group resolved links into version chains, oldest to newest.

    Returns `(chains, links)` so a caller can report unresolved and dangling markers rather
    than silently seeing fewer chains than the corpus asserted.
    """
    links = resolve_links(current, retired)
    chains: list[Chain] = []
    for link in links:
        if link.predecessor is None or link.dangling:
            continue
        versions = sorted([link.predecessor, link.successor], key=_date_sort_key)
        chains.append(Chain(versions=versions))
    return chains, links


def active(current: Sequence[Record],
           retired: Sequence[Record] = ()) -> list[Record]:
    """Currently-valid records only.

    This is the DEFAULT view and it is already what callers get: `RecordStore._decode()`
    excludes superseded and tombstoned nodes, and `run_merge()` removes the stale record
    outright. The function exists so active-state projection is nameable and testable rather
    than merely emergent, and so a caller holding a mixed list can ask for it explicitly.
    """
    retired_ids = {id(r) for r in retired}
    return [r for r in current if id(r) not in retired_ids and not r.superseded]


def successor_of(record: Record, current: Sequence[Record],
                 retired: Sequence[Record] = ()) -> Optional[Record]:
    """What replaced `record`, or None. The derived half of "what superseded this?"."""
    for link in resolve_links(current, retired):
        if link.predecessor is record and not link.dangling:
            return link.successor
    return None


def render_timeline(chains: Sequence[Chain]) -> str:
    """Render version chains for the prompt. Off by default -- see `compose(timeline=True)`."""
    lines = []
    for chain in chains:
        parts = []
        for i, v in enumerate(chain.versions):
            date = f"[{v.date}] " if v.date else ""
            tail = " (current)" if i == len(chain.versions) - 1 else ""
            parts.append(f"{date}{v.text}{tail}")
        head = chain.head
        label = head.subtype or head.kind
        lines.append(f"  {label}: " + " -> ".join(parts))
    return "\n".join(lines)


# --- the durable edge log -----------------------------------------------------------

# --- consolidation watermarks (durable dirty state) ---------------------------------

WATERMARK_SESSION_PREFIX = "__wm__:"
_WATERMARK_MARKER = "consolidated-through"


def watermark_session(tenant: str) -> str:
    """The session id holding one tenant's consolidation watermarks.

    Same namespacing trick as `edge_session()`, for the same reason and with one extra: a
    watermark written into the SESSION'S OWN id would advance that session's tail and leave
    it permanently dirty. It must live somewhere else, and a namespaced session is the only
    durable store here that is dropped with its tenant.
    """
    return f"{WATERMARK_SESSION_PREFIX}{tenant}"


def write_watermark(trace, *, tenant: str, session_id: str, event_id: int) -> int:
    """Record that `session_id` is consolidated through Trace event `event_id`.

    `atlas_id` carries the event id and the text carries the session id -- no new field, the
    same "an id in an existing slot" convention `write_supersession_edge` already uses.

    The event id is the right watermark and this was checked rather than assumed: ids are
    monotonic (`next_event_id_++`), persisted in the file header, and **survive compaction**
    -- `trace.cpp` copies events verbatim and carries the counter forward, so ids go sparse
    but are never reassigned. Atlas node ids, by contrast, shift after `compact_mmap()`,
    which is why this is keyed on Trace and not on the record store.
    """
    return trace.add_event(watermark_session(tenant), "system",
                           f"{_WATERMARK_MARKER} {session_id}", atlas_id=int(event_id))


def read_watermarks(trace, *, tenant: str, limit: int = 5000) -> dict[str, int]:
    """`{session_id: last_consolidated_event_id}` for a tenant.

    `get_history` is newest-first, so the FIRST entry seen for a session is the latest
    watermark and later ones are superseded history -- hence `setdefault`. The session is
    append-only, so `limit` bounds how far back this sees; a tenant with more than `limit`
    watermarks would lose the oldest, which costs a redundant re-consolidation rather than
    lost data.
    """
    out: dict[str, int] = {}
    try:
        history = trace.get_history(watermark_session(tenant), limit)
    except Exception:
        return out
    for ev in history:
        text = ev.get("text") or ev.get("text_preview") or ""
        if not text.startswith(_WATERMARK_MARKER):
            continue
        session_id = text[len(_WATERMARK_MARKER):].strip()
        if session_id:
            out.setdefault(session_id, int(ev.get("atlas_id", 0)))
    return out


def session_tail(trace, session_id: str) -> int:
    """The id of the newest event in a session, or 0. One bounded call, not a scan --
    `get_history` is newest-first, so `limit=1` is the tail."""
    try:
        history = trace.get_history(session_id, 1)
    except Exception:
        return 0
    return int(history[0]["id"]) if history else 0


def write_supersession_edge(trace, *, tenant: str, survivor_node_id: int,
                            retired_node_id: int, text: str, event_time: int = 0,
                            reason_code: ReasonCode = ReasonCode.CONSOLIDATED_BY_DREAMING
                            ) -> int:
    """Record `survivor SUPERSEDES retired` durably, using fields that already exist.

    `atlas_id` = the survivor, `supersedes_id` = the retired node. That is an Atlas node id
    inside a Trace field, which is the convention BOTH existing writers already use
    (`supersession.py`, `promotion.py`) and which `schema.hpp` documents as opaque to the
    kernel -- so this adds no new convention, it reads one that was already write-only.

    Inherited limitations, stated rather than solved (same as `supersession.py`/`erasure.py`):
    node ids shift after `compact_mmap()`, and `get_history`'s `limit` bounds how far back a
    reader sees.
    """
    return trace.add_event(
        edge_session(tenant), "system", text,
        atlas_id=survivor_node_id,
        edge_type=EdgeType.SUPERSEDES,
        supersedes_id=retired_node_id,
        reason_code=reason_code,
        event_time=event_time,
    )


def read_supersession_edges(trace, *, tenant: str, limit: int = 1000) -> list[dict]:
    """Every supersession edge for a tenant, newest first.

    One `get_history()` call rather than a scan of every session, which is the whole reason
    the edges get their own session id. Non-edge events in that session are ignored rather
    than assumed absent.
    """
    out = []
    for ev in trace.get_history(edge_session(tenant), limit):
        if int(ev.get("edge_type", 0)) == int(EdgeType.SUPERSEDES):
            out.append(ev)
    return out


def superseded_by(trace, node_id: int, *, tenant: str, limit: int = 1000) -> list[int]:
    """Which node(s) replaced `node_id`. The durable half of "what superseded this?"."""
    return [int(e["atlas_id"]) for e in read_supersession_edges(trace, tenant=tenant,
                                                                limit=limit)
            if int(e.get("supersedes_id", 0)) == node_id]


def supersedes(trace, node_id: int, *, tenant: str, limit: int = 1000) -> list[int]:
    """Which node(s) `node_id` replaced."""
    return [int(e["supersedes_id"]) for e in read_supersession_edges(trace, tenant=tenant,
                                                                     limit=limit)
            if int(e.get("atlas_id", 0)) == node_id]


# --- point-in-time projection (v4.2 Track A) ----------------------------------------
#
# WHAT THIS IS, AND WHAT IT DELIBERATELY IS NOT. `as_of` answers "what did this memory
# hold on 2023/06/01?" without a schema change, a kernel change, or a new field --
# the same derived-overlay discipline the rest of this module follows.
#
# THE DESIGN THE SPEC ASKED FOR DOES NOT EXIST. `spatiotemporal.md` proposed filtering
# records on a per-record `timestamp`. There is none: records are Atlas nodes, `NodeHeader`
# is full at 64 bytes (`schema.hpp` says so in as many words -- "no further per-node field
# fits without a side structure"), and `Record.encode()` writes exactly seven \x1f fields
# while `decode()` does `text=_SEP.join(parts[6:])`, so an eighth is silently swallowed
# into the TEXT of every record by every existing reader.
#
# So validity is DERIVED from provenance instead, which is better rather than merely
# cheaper: `Provenance` already names the session and the turns a record came from, and
# those turns carry `event_time`. That makes a record's validity the time its EVIDENCE was
# said -- valid-time -- rather than the time its row happened to be written. "What did I
# know in March" is a question about the conversation, not about the consolidator's
# schedule.
#
# NO ACCURACY CLAIM IS ATTACHED. LongMemEval asks every question at T_now and its corpora
# predate the question date, so this cannot move a benchmark number and is not gated on one.
# It is a product capability with correctness tests, and that is the whole of its warrant.
#
# THE FIDELITY BOUND, STATED RATHER THAN HIDDEN. `consolidate()` + `run_merge()` REMOVE a
# stale record rather than superseding it (see this module's header). So `as_of`
# reconstructs what supersession RETIRED; it cannot reconstruct what the merge DROPPED.
# Making the merge supersede-not-drop would close that gap and is a real decision with its
# own measurement -- deliberately not smuggled in here.

def session_event_times(trace, session_id: str, *, history_limit: int = 200) -> list[int]:
    """Valid-times of one session's turns, in the order `Provenance.turn_indices` numbers
    them against.

    `get_history()` is newest-first and `RecordStore.rehydrate()` reverses it before
    indexing. That reversal is repeated here rather than reinvented: a provenance index
    that meant one thing in rehydration and another here would be a silent
    off-by-everything, and the two must move together.
    """
    try:
        hist = trace.get_history(session_id, history_limit)
    except Exception:
        return []
    return [event_time_of(ev) for ev in reversed(hist)]


def validity_start(record: Record, times: Sequence[int]) -> Optional[int]:
    """When `record`'s evidence became complete: the LATEST valid-time among its source
    turns.

    Latest, not earliest -- a record derived from turns 3 and 4 is not assertable until
    turn 4 has been said. A record whose citation was lost carries whole-session provenance
    instead (`parse_records()` does that on write), and the session's last turn is then the
    honest answer rather than a guess at which turn it meant.

    Returns None when the session is unknown to this Trace. The caller decides what an
    unresolvable record means, because the answer differs between an audit view and a read
    path -- see `visible_at(unresolved=...)`.
    """
    if not times:
        return None
    idx = [i for i in record.provenance.turn_indices if 0 <= i < len(times)]
    return max(times[i] for i in idx) if idx else max(times)


def edges_before(trace, *, tenant: str, as_of: int, limit: int = 1000) -> list[dict]:
    """Supersession edges asserted at or before `as_of`, newest first.

    `<=` rather than `<`: `as_of` names a moment, and a supersession recorded AT that
    moment had already happened by it. The boundary is stated because an off-by-one here
    silently returns the wrong version of a fact.
    """
    return [e for e in read_supersession_edges(trace, tenant=tenant, limit=limit)
            if event_time_of(e) <= as_of]


def visible_at(records: Sequence[Record], trace, as_of: Optional[int], *,
               tenant: str = "", unresolved: str = "include",
               history_limit: int = 200, edge_limit: int = 1000) -> list[Record]:
    """The records that were live at `as_of`.

    `as_of=None` returns the input unchanged, so threading this through a read path costs
    exactly nothing until someone asks a temporal question.

    Two filters, and they are not the same filter:

      1. **Not yet true.** A record whose evidence was said after `as_of` is dropped.
      2. **No longer true.** A record superseded at or before `as_of` is dropped -- and,
         symmetrically, one superseded AFTER `as_of` is KEPT even though today's default
         view excludes it. That second half is why a caller wants
         `all_records(include_superseded=True)` here: the retired records are the whole
         point of a historical view, and `_decode()` filters them out of the live one.

    `tenant` is what makes supersession datable, since the edge log lives in that tenant's
    `edge_session()`. **Without it the retirement times are unknown**, so this falls back to
    today's default view -- already-superseded records are dropped rather than dated. That
    is a degradation to present-day semantics, never to a wrong historical answer.

    `unresolved` governs records whose session is not in this Trace: `"include"` (default)
    matches `RecordStore._decode()`'s posture, which degrades toward showing a record rather
    than vanishing it from a user's memory. An audit caller that must not show a fact it
    cannot date passes `"exclude"`.
    """
    if as_of is None:
        return list(records)
    if unresolved not in ("include", "exclude"):
        raise ValueError(f"unresolved must be 'include' or 'exclude', got {unresolved!r}")

    retired_ids: Optional[set[int]] = None
    if tenant:
        retired_ids = {int(e.get("supersedes_id", 0))
                       for e in edges_before(trace, tenant=tenant, as_of=as_of,
                                             limit=edge_limit)}
        retired_ids.discard(0)

    times: dict[str, list[int]] = {}
    out: list[Record] = []
    for r in records:
        sid = r.provenance.session_id
        if sid and sid not in times:
            times[sid] = session_event_times(trace, sid, history_limit=history_limit)
        start = validity_start(r, times.get(sid, ())) if sid else None
        if start is None:
            if unresolved == "include":
                out.append(r)
            continue
        if start > as_of:
            continue
        if retired_ids is None:
            if r.superseded:
                continue
        elif r.node_id is not None and int(r.node_id) in retired_ids:
            continue
        out.append(r)
    return out
