"""
Aeon Record Store — the persistent semantic layer (v4-plan.md, PRODUCT DIRECTION).

Aeon's episodic half (Trace) stores what was said. This stores what it *means*: durable,
accumulating records derived from conversation at write time, so answering a question
becomes a lookup instead of a multi-hop scan over scattered raw turns.

Why this exists, empirically rather than architecturally: 28 benchmark questions are wrong
under an ORACLE that is handed the gold evidence directly, under extract-then-compute, and
under single-shot -- perfect evidence, wrong answer, dominated by counting ("how many albums
have I purchased?" gold 3, oracle answered 2). A maintained `ITEM(ACQUISITION/music album)`
set answers those by enumeration. The consolidation probe converted 5/18 of exactly those
questions, ~4x the measured noise floor, which is what justifies building this.

RECORD SCHEMA (locked; see v4-plan.md "SCHEMA v2 RESULT" for why it is locked on mechanism
grounds rather than aggregate ones):

    kind    ITEM | FACT | EVENT | UPDATE | PREF | TASK
    bucket  one of the 12 closed buckets, for ITEM records
    subtype free-form, beneath the bucket

The closed bucket set is load-bearing. A free-form category vocabulary was tried first and
failed: independent, query-blind, per-session extraction calls invent their own category
names, so members of one real category scatter and never accumulate -- of three albums one
became an ITEM, one a PREF, one an EVENT, and counting ITEMs returned 1 against a gold of 3.

PROVENANCE (built first, deliberately, because it is the part no probe exercised):

Records are lossy by construction, and this project measured what that costs: compressing to
bare facts destroys pragmatic licensing -- "picked it up at Trader Joe's" only answers "what
brand?" through conversational implicature -- and cost 12 questions when extract-then-compute
did it. The fix that measurably works is keeping one conversational turn either side of the
evidence (oracle scored 63/64 on that type with +/-1 neighbours, versus 55 without).
So every record carries a link back to the turns it came from, and the read path can rehydrate
that neighbourhood on demand.

Provenance is stored as `session_id` + turn indices, NOT as raw event/node ids. That is a
correctness decision, not a convenience: this codebase already documents that a raw Atlas node
id can shift after `compact_mmap()` reclaims tombstoned slots (see supersession.py's known
limitation), and Trace exposes history per session rather than by id. Session id plus turn
index survives compaction and matches the API that exists.

Provenance is also the **erasure cascade index**. Records are PII derived from conversation,
and `erasure.py` tombstones Atlas nodes but nothing previously cascaded to derived records.
Because every record carries the session it came from, honouring a deletion request is a
filter on `provenance.session_id` -- the field built for pragmatic licensing turns out to be
the right-to-be-forgotten index too. See `records_for_session()`.

USES THE KERNEL RATHER THAN REIMPLEMENTING IT. Three capabilities that already exist are used
instead of Python equivalents:

  * **Atlas is a tree**, not a flat list (`insert(parent_id, ...)` / `get_children(parent_id)`).
    The closed bucket taxonomy IS a tree, so each bucket is a node and its records are its
    children, and that parent link is what the beam-search hierarchy descends.

    **CORRECTED (v4.1).** This bullet used to continue: "A category scan ... is therefore
    `get_children()`, a kernel subtree walk, not a Python filter over every record." That was
    false in production and `records_in_bucket()` is now a filter. Atlas registers a child only
    when its byte offset is the next slot after the parent's existing children, and `insert()`
    appends at the tail -- so a bucket's child block can only grow while that bucket is the
    last thing written, and every later child is invisible to `get_children()` forever.
    Extraction emits records across many buckets per session, so interleaving is the normal
    case: insert MEDIA, POSSESSION, MEDIA and the walk returns one of the two MEDIA records.
    The subset failure this layer must never have is exactly the one the walk produced.
  * **`session_id` scoping** on `insert()`, with `drop_session()` for teardown. The first
    version passed neither, which is invisible in a benchmark (one store per question) and
    fatal in a multi-tenant product.
  * **`supersede_node()` / `revoke_node_supersede()`** -- reversible, branchless exclusion from
    search results. A superseded record can be *removed from retrieval* rather than shown to
    the model alongside its replacement and hoped about.

Not reimplemented and deliberately not used here: `event_time` is a `TraceEvent` field, and
records are Atlas nodes whose only payload is the metadata string, so dates live in the record
text. That is the available option, not an oversight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from . import core
from .session_expansion import format_events

# Closed bucket set. Deliberately overlapping (an album purchase is both ACQUISITION and
# MEDIA) because one real fact often needs several records -- type-assignment was the third
# measured failure mode of the free-form schema.
BUCKETS: tuple[str, ...] = (
    "POSSESSION", "ACQUISITION", "MEDIA", "PERSON", "EVENT_ATTENDED", "OBLIGATION",
    "EDUCATION_WORK", "HEALTH", "TRAVEL", "PROJECT", "FINANCE", "CONSUMABLE",
)
KINDS: tuple[str, ...] = ("ITEM", "FACT", "EVENT", "UPDATE", "PREF", "TASK")

_SEP = "\x1f"          # field separator, cannot occur in model-generated text
_RANGE_SEP = ","
_DEFAULT_METADATA = 1024
_ALL_SCOPES_VISIBLE = 0xFFFFFFFFFFFFFFFF   # every scope bit; unscoped nodes are included
_BUCKET_MARKER = "BUCKET"      # metadata prefix marking a taxonomy node, not a record


@dataclass
class Provenance:
    """Where a record came from. `turn_indices` are positions within the session's
    chronological history, so they stay valid across compaction."""
    session_id: str
    turn_indices: tuple[int, ...] = ()

    def encode(self) -> str:
        return f"{self.session_id}:{_compress_indices(self.turn_indices)}"

    @staticmethod
    def decode(s: str) -> "Provenance":
        if not s:
            return Provenance("")
        sid, _, idx = s.rpartition(":")
        return Provenance(sid or s, _expand_indices(idx) if sid else ())


def _compress_indices(idx: Sequence[int]) -> str:
    """Contiguous turn indices collapse to ranges. Provenance shares a fixed-size metadata
    field with the record text, and `Atlas.insert()` truncates silently on overflow, so
    compactness here directly buys record-text headroom."""
    if not idx:
        return ""
    s = sorted(set(int(i) for i in idx))
    out, start, prev = [], s[0], s[0]
    for i in s[1:]:
        if i == prev + 1:
            prev = i
            continue
        out.append(f"{start}-{prev}" if prev > start else f"{start}")
        start = prev = i
    out.append(f"{start}-{prev}" if prev > start else f"{start}")
    return _RANGE_SEP.join(out)


def _expand_indices(s: str) -> tuple[int, ...]:
    out: list[int] = []
    for part in s.split(_RANGE_SEP):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, _, b = part.partition("-")
            try:
                out.extend(range(int(a), int(b) + 1))
            except ValueError:
                continue
        else:
            try:
                out.append(int(part))
            except ValueError:
                continue
    return tuple(out)


@dataclass
class Record:
    kind: str
    text: str
    bucket: str = ""
    subtype: str = ""
    date: str = ""
    provenance: Provenance = field(default_factory=lambda: Provenance(""))
    supersedes: str = ""
    node_id: Optional[int] = None

    def encode(self) -> str:
        """Provenance and structure are encoded BEFORE the free text. Overflow truncates the
        tail, so this ordering means a too-long record loses the end of its text rather than
        its provenance link -- a record you can still trace back is worth more than one whose
        text is complete but whose origin is gone."""
        return _SEP.join((self.kind, self.bucket, self.subtype, self.date,
                          self.provenance.encode(), self.supersedes, self.text))

    @staticmethod
    def decode(s: str, node_id: Optional[int] = None) -> "Record":
        parts = s.split(_SEP)
        while len(parts) < 7:
            parts.append("")
        return Record(kind=parts[0], bucket=parts[1], subtype=parts[2], date=parts[3],
                      provenance=Provenance.decode(parts[4]), supersedes=parts[5],
                      text=_SEP.join(parts[6:]), node_id=node_id)

    @property
    def category(self) -> str:
        return f"{self.bucket}/{self.subtype}" if self.bucket else self.kind

    def display(self) -> str:
        head = f"{self.kind}({self.category})" if self.kind == "ITEM" else self.kind
        date = f" [{self.date}]" if self.date else ""
        sup = f" [supersedes {self.supersedes}]" if self.supersedes else ""
        return f"{head}: {self.text}{date}{sup}"


class RecordStore:
    """Atlas-backed store of consolidated records, each linked to its source turns.

    Records live in Atlas rather than Trace because they are semantic, not episodic: they
    accumulate and are revised, whereas Trace is an append-only log of what was said.
    """

    ROOT = 0

    def __init__(self, atlas_path: str | Path, dim: int,
                 metadata_size: int = _DEFAULT_METADATA,
                 session_id: Optional[str] = None) -> None:
        self.path = Path(atlas_path)
        self.atlas = core.Atlas(str(self.path), dim=dim, metadata_size=metadata_size)
        self.dim = dim
        self.session_id = session_id
        # Rebuilt from disk, not started empty. This index was in-memory only until v4.1,
        # which made the taxonomy silently wrong across a process restart -- see
        # `_load_bucket_nodes()`.
        self._bucket_nodes: dict[str, int] = {}
        # insert() truncates silently rather than raising (documented in core.pyi), so every
        # write is length-checked here instead. -1 leaves room for the null terminator.
        self.capacity = _as_int(self.atlas.metadata_size) - 1
        self._load_bucket_nodes()

    def _load_bucket_nodes(self) -> None:
        """Rebuild the bucket index by scanning for `BUCKET\x1f<name>` markers.

        BUG FIX (v4.1). `_bucket_nodes` was populated only by `_bucket_node()` during a
        write, so a `RecordStore` opened over an existing file started with an empty index.
        Two consequences, neither of which any test caught -- every existing test scanned the
        same instance that had written:

          * `records_in_bucket()` returned `[]` on a re-opened store. The bucket nodes and
            their children were intact on disk; the dict naming them was gone. The only real
            subtree walk in the record layer was dead in production.
          * The first `add()` after a re-open minted a SECOND `BUCKET\x1f<name>` node under
            ROOT, so `records_in_bucket()` then returned only the records written since the
            re-open -- a SILENT SUBSET. That is exactly the failure `compose.py`'s docstring
            says the structured category scan exists to prevent ("Vector top-k would silently
            return a subset and turn a complete answer into a plausible wrong one"), and it is
            the worst available failure for a layer whose flagship capability is counting.

        Reproduced before fixing: write "Dune" to MEDIA, re-open, scan -> `[]`; one further
        add grows the store by two nodes and the scan then returns only `["Arrival"]`.

        The scan is O(n) once per open, against `all_records()` being O(n) per call, so this
        costs nothing that was not already being paid. First marker wins, so a store that
        already contains duplicate bucket nodes from before this fix keeps using the original
        and degrades to the old behaviour for the stragglers rather than picking arbitrarily.
        """
        for nid in range(_as_int(self.atlas.size)):
            meta = self._read_metadata(nid)
            if not meta or not meta.startswith(_BUCKET_MARKER + _SEP):
                continue
            name = meta.split(_SEP, 1)[1]
            self._bucket_nodes.setdefault(name, nid)

    def _bucket_node(self, bucket: str, embedding: Sequence[float]) -> int:
        """The Atlas node standing for a bucket. Created on first use, so the taxonomy
        materialises as a subtree without a separate schema-init step."""
        if not bucket:
            return self.ROOT
        node = self._bucket_nodes.get(bucket)
        if node is None:
            node = self.atlas.insert(self.ROOT, list(embedding),
                                     _SEP.join((_BUCKET_MARKER, bucket)), self.session_id)
            self._bucket_nodes[bucket] = node
        return node

    def add(self, record: Record, embedding: Sequence[float]) -> int:
        payload = record.encode()
        if len(payload.encode("utf-8")) > self.capacity:
            record = self._fit(record)
            payload = record.encode()
        parent = self._bucket_node(record.bucket, embedding)
        node_id = self.atlas.insert(parent, list(embedding), payload, self.session_id)
        record.node_id = node_id
        return node_id

    def records_in_bucket(self, bucket: str) -> list[Record]:
        """Every record in a bucket.

        REIMPLEMENTED AS A FILTER (v4.1), and the docstrings that claimed otherwise are
        corrected with it. `records.py` used to say a category scan "is therefore
        `get_children()`, a kernel subtree walk, not a Python filter over every record."
        **That is false in production.** `atlas.cpp` registers a child only when its byte
        offset equals `first_child_offset + child_count * stride`; `insert()` appends at the
        tail, so a bucket's child block can only grow while that bucket is the last thing
        written, and any other child is invisible to `get_children()` forever.

        Extraction emits records across many buckets per session, so interleaving is the
        normal case. Measured: insert MEDIA, POSSESSION, MEDIA and the subtree walk returns
        ONE of the two MEDIA records while `all_records()` returns all three.

        A silent subset is the one failure this layer must never have -- counting is what it
        exists for, and `compose.py` says so in as many words about vector top-k. So the O(n)
        filter is correct and the subtree walk is not, at a record-store scale (a few hundred
        records per user) where `all_records()` was already being paid on every read.

        The bucket nodes stay on disk: they are the physical parent link the beam-search
        hierarchy uses. Only the enumeration claim is withdrawn.
        """
        return [r for r in self.all_records() if r.bucket == bucket]

    def supersede(self, record: Record) -> bool:
        """Reversibly exclude a superseded record from retrieval, using the kernel's own
        supersession rather than a text marker the model has to interpret.

        Inherits the documented limitation that a raw node id can shift after
        `compact_mmap()` reclaims tombstoned slots.
        """
        if record.node_id is None:
            return False
        try:
            self.atlas.supersede_node(int(record.node_id))
            return True
        except Exception:
            return False

    def records_for_session(self, session_id: str,
                            include_superseded: bool = True) -> list[Record]:
        """Records derived from one session -- the erasure cascade index.

        `erasure.py` tombstones Atlas nodes but nothing cascaded to records derived from an
        erased conversation. Provenance makes that a filter rather than a new mechanism.
        """
        # DEFAULTS TO include_superseded=True, unlike every other read path, and that
        # difference is the point: this is the right-to-erasure index, not the
        # prompt-assembly path. A superseded record is excluded from PROMPTS, not from
        # EXISTENCE -- it is still PII derived from the session -- so a cascade using the
        # read-path default would tombstone the live records and silently leave the
        # retired ones on disk. Caught in review before it shipped.
        return [r for r in self.all_records(include_superseded=include_superseded)
                if r.provenance.session_id == session_id]

    def _fit(self, record: Record) -> Record:
        """Trim the record's TEXT until the encoded payload fits, preserving every structural
        field. Structure and provenance are what make a record queryable and traceable; the
        tail of a long sentence is the cheapest thing to lose.

        Truncation snaps back to a UTF-8 character boundary rather than a raw byte count.
        The first version of this did not, and appended a 3-byte ellipsis on top of an
        already-full budget -- the store then wrote a payload whose last character was cut
        mid-sequence and every subsequent read raised UnicodeDecodeError. The C++ side hit
        and fixed the identical bug on `TraceEvent.text_preview` (`safe_utf8_truncate_length`
        in trace.cpp); this is the Python-side counterpart.
        """
        fixed = len(_SEP.join((record.kind, record.bucket, record.subtype, record.date,
                               record.provenance.encode(), record.supersedes, "")
                              ).encode("utf-8"))
        ellipsis = "..."                      # ASCII: costs 3 bytes, never splits
        room = max(0, self.capacity - fixed - len(ellipsis))
        text = _utf8_truncate(record.text, room)
        return Record(record.kind, (text + ellipsis) if text else "", record.bucket,
                      record.subtype, record.date, record.provenance, record.supersedes,
                      record.node_id)

    def _decode(self, node_ids: Iterable[int],
                include_superseded: bool = False) -> list[Record]:
        """Decode node ids into records, skipping the taxonomy's own bucket nodes and
        (by default) superseded ones.

        Shared by every read path so the "is this a record or a bucket marker?" rule exists
        once rather than three times.

        THE SUPERSESSION SKIP IS A BUG FIX, not an optimisation. `supersede_node()` marks a
        node by setting `hub_penalty = TOMBSTONE_PENALTY`, which excludes it from BEAM
        SEARCH -- and nothing else. Enumeration never consulted it: `all_records()` walks
        `range(atlas.size)` and `records_in_bucket()` walks a subtree, so a record could be
        superseded and still render into every prompt, indefinitely. The kernel's own
        `list_nodes_by_scope` documents the same behaviour ("Superseded nodes ARE
        included"). `test_supersede_uses_the_kernel_primitive` passed while asserting only
        that the flag was set, so this was invisible.

        Fixed at the choke point rather than in three callers. The check degrades to
        INCLUDING the record on any binding error, matching `_read_metadata()`'s defensive
        posture: a change in the extension must fall back to today's behaviour, never to
        records silently vanishing from a user's memory.
        """
        out: list[Record] = []
        for nid in node_ids:
            meta = self._read_metadata(int(nid))
            if not meta or meta.startswith(_BUCKET_MARKER):
                continue
            if not include_superseded:
                try:
                    if self.atlas.is_node_superseded(int(nid)):
                        continue
                except Exception:
                    pass
            out.append(Record.decode(meta, node_id=int(nid)))
        return out

    def query(self, embedding: Sequence[float], top_k: int = 20) -> list[Record]:
        raw = self.atlas.navigate_raw(list(embedding), beam_width=max(4, top_k))
        return self._decode(_node_ids_from_raw(raw, top_k))

    def all_records(self, include_superseded: bool = False) -> list[Record]:
        """Every LIVE record. `include_superseded=True` is for audit and admin paths that
        must see what was retired, not for the read path.

        Enumerates via `list_nodes_by_scope()` rather than `range(atlas.size)`, and that is a
        BUG FIX, not a refactor. `tombstone_node()` sets `NODE_FLAG_TOMBSTONE` and there is no
        Python-visible per-node tombstone predicate -- only `tombstone_count()` (an aggregate)
        and this call, whose contract is "live (non-tombstoned) node ids". Walking the raw id
        range therefore could not distinguish an erased node from a live one, so **records
        tombstoned by the right-to-erasure workflow still rendered into every prompt.** That
        is the supersession bug's twin, and worse: an erased record reaching the model is a
        compliance failure, not an accuracy one.

        `ALL_SCOPES_VISIBLE` is the correct mask here and was verified rather than assumed:
        records are inserted unscoped (`scope_bitmap == 0`), and an all-ones mask returns them
        while still excluding tombstones.
        """
        live = self.atlas.list_nodes_by_scope(_ALL_SCOPES_VISIBLE)
        return self._decode(live, include_superseded)

    def _read_metadata(self, node_id: int) -> str:
        """Reads defensively. A store may outlive the code that wrote it, and one bad row --
        e.g. written by a version with the UTF-8 truncation bug above -- must not make the
        whole record set unreadable. Undecodable bytes are replaced, never raised."""
        try:
            return self.atlas.get_node_metadata(node_id)
        except UnicodeDecodeError:
            try:
                raw = self.atlas.get_node_metadata_bytes(node_id)
            except AttributeError:
                return ""
            return bytes(raw).decode("utf-8", errors="replace").rstrip("\x00")
        except Exception:
            return ""

    def sync(self) -> None:
        self.atlas.sync()

    # ---- provenance resolution ------------------------------------------------

    @staticmethod
    def rehydrate(records: Iterable[Record], trace, *, neighbours: int = 1,
                  max_turns: int = 12, history_limit: int = 200) -> list[str]:
        """Return the source turns behind these records, with `neighbours` turns of
        conversational context on each side.

        The neighbourhood is the point. Bare facts lose pragmatic licensing -- the measured
        cost was 12 questions when compression dropped it -- and one turn either side was
        enough to recover it (63/64 versus 55 on the affected type). `max_turns` bounds the
        rehydration so provenance cannot silently reintroduce the 100k-char context this
        layer exists to avoid.
        """
        wanted: dict[str, set[int]] = {}
        for r in records:
            sid = r.provenance.session_id
            if not sid or not r.provenance.turn_indices:
                continue
            keep = wanted.setdefault(sid, set())
            for i in r.provenance.turn_indices:
                keep.update(range(max(0, i - neighbours), i + neighbours + 1))

        picked: list[dict] = []
        for sid, idxs in wanted.items():
            hist = trace.get_history(sid, history_limit)
            if not hist:
                continue
            hist = list(reversed(hist))          # get_history is reverse-chronological
            for i in sorted(idxs):
                if 0 <= i < len(hist):
                    picked.append(hist[i])
                    if len(picked) >= max_turns:
                        break
            if len(picked) >= max_turns:
                break
        # Rendered through session_expansion.format_events rather than a local formatter, so
        # rehydrated provenance is byte-identical to every other context this repo builds --
        # `role` comes back from get_history as an INT, and a local f-string printed "[0]"
        # instead of "[user]". Sharing the formatter makes that class of drift impossible.
        return format_events(picked).splitlines() if picked else []


def _utf8_truncate(text: str, max_bytes: int) -> str:
    """Longest prefix of `text` whose UTF-8 encoding fits in `max_bytes`, never splitting a
    character."""
    if max_bytes <= 0:
        return ""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    cut = encoded[:max_bytes]
    while cut and (cut[-1] & 0xC0) == 0x80:      # inside a continuation sequence
        cut = cut[:-1]
    return cut.decode("utf-8", errors="ignore")


def _as_int(attr) -> int:
    """nanobind exposes some Atlas accessors as properties and others as bound methods;
    tolerate both so a binding change cannot silently break record enumeration."""
    return int(attr() if callable(attr) else attr)


def _node_ids_from_raw(raw, top_k: int) -> list[int]:
    """`navigate_raw` returns a packed byte view; decode defensively so a layout change
    degrades to 'no results' rather than to wrong records."""
    try:
        arr = np.frombuffer(bytes(raw), dtype=np.uint64)
    except (ValueError, TypeError):
        return []
    return [int(x) for x in arr[:top_k]]
