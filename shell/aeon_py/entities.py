"""Entity identity for the composite read path.

WHY THIS EXISTS -- the measured cause, not a hypothesis. The 12-bucket taxonomy is
deliberately relational and overlapping: the extraction prompt states outright that "an
album the user downloaded is BOTH ITEM(ACQUISITION/music album) AND ITEM(MEDIA/music
album)". That is a feature -- it is how a question about things acquired and a question
about media consumed both find the album. But `compose.order_records()` sorts ITEMs by
bucket, so the copies of one real entity land far apart in a 26k-char prompt, and the
reader counts them twice.

Measured on the 500-question cached corpus against the 429/500 run:

  * exact-normalized duplicate ITEM lines: ~5/question, 487 of 500 questions affected,
    75% of them cross-bucket;
  * globally NON-discriminative -- 5.35 duplicates/question on correct answers vs 5.66
    on wrong, which is no signal at all;
  * but on the 9 identifiable OVERCOUNT errors: 9.9/question, 1.8x the corpus median,
    while the 8 identifiable UNDERCOUNT errors sit at 5.6, i.e. at the median.

That conditional split is the whole justification. It is also the honest ceiling: this
targets the 16 overcounts and possibly the 4 right-number-wrong-enumeration errors, and
does nothing for the 13 undercounts, 13 temporal, 8 abstention or 13 lookup errors.

Confirmed in the model's own output rather than inferred: on `gpt4_15e38248` it lists
"Wooden coffee table with metal legs" and "Coffee table" as separate members and answers
5 against a gold of 4; on `gpt4_59c863d7`, "172 scale b29 bomber" is filed under
ACQUISITION + POSSESSION + PROJECT and it answers 7 against 5.

WHY DERIVED AND NOT STORED. The key is a pure function of `text`, so a stored copy can
only go stale; the record schema is declared locked in `records.py` and `_fit()` trims
TEXT only, so a new field would tax text headroom on every record forever; and decisively
the measured benchmark path never constructs a `RecordStore` at all
(`composite_arm_experiment.py` goes `parse_records()` -> `compose()`), so a stored key
would be invisible to the instrument that produces the 429.

WHY GROUPS HOLD RECORDS AND NEVER SYNTHESISE ONE. A `Record` carries exactly one
`Provenance` with one `session_id`. In production, cross-bucket co-referents come from
DIFFERENT sessions, so a synthesised merged record would have to discard all but one
provenance link -- and that link is both the rehydration key and the right-to-erasure
cascade index (`records_for_session()`). Grouping is a rendering concern; identity must
not destroy lineage.

MEASURED RESULT (2026-08-27), recorded here so the code carries its own verdict.
On the dev half (n=252, records frozen, `n_errors=0`): **220 against v1's 219 -- +1, McNemar
+3/-2, p=1.000**, against a noise floor of ~2.7. Every question type flat but multi-session
+1. **All five identified overcount questions present in dev were wrong before and are wrong
after.**

The motivating evidence still holds -- 7 of 12 parseable overcounts contain a self-duplicate
in the model's own enumeration, and the overcount cohort carries 1.8x the median duplicate
load. **The causal step is what failed.** The duplicate was a co-occurring symptom, not the
binding constraint: over-inclusion is driven by predicate-boundary judgement ("is a
rearranged sofa something I *bought*?") and by supersession ("Three tops" then "Five tops"
summed to 8 against a gold of 5), neither of which is a co-reference problem. `gpt4_59c863d7`
is the clean demonstration -- the duplicate collapsed, the count moved 7->6, still wrong
against a gold of 5.

KEPT ANYWAY, WITH NO BENCHMARK CLAIM. The full n=500 confirmation was deliberately skipped
(a recorded deviation from the pre-registered ladder, taken by the user, on the grounds that
dev is a random half of the same instrument and already puts >=437 out of reach). This code
stays because rendering one real entity once is more nearly correct than rendering it twice,
and it costs a median 212 characters and 6 lines less prompt per question. Neither is worth
claiming as an accuracy result, and neither should be cited as one.

THE KEY IS FROZEN. Tuning `canonical_key` against which benchmark questions flip is
fitting the test set, and this repo has a written precedent for refusing that (the ETC v3
COMPUTE prompt, reverted on an exact tie rather than iterated). Any change here needs its
own pre-registration.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .records import BUCKETS, Record

# Leading articles only. Interior stopwords are load-bearing -- "the power of now" and
# "power of now" are one book, but "milk" and "oat milk" are two things.
_ARTICLES = frozenset(("a", "an", "the"))

_BRACKETED = re.compile(r"\[[^\]]*\]")      # dates, [supersedes X]
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_BUCKET_RANK = {b: i for i, b in enumerate(BUCKETS)}

# Which bucket an entity is FILED under when it belongs to several. This is not
# cosmetic and not alphabetical: plain BUCKETS order puts POSSESSION first, so the
# coffee table (ACQUISITION + POSSESSION) would file under POSSESSION -- while the
# question that gets it wrong is "how many pieces of furniture did I BUY". The measured
# counting cohort is acquisition- and attendance-shaped, so the entity should live in
# the group those questions read. Findability for the other buckets is preserved because
# every category still appears on the rendered line.
_PRIMARY_FIRST = ("ACQUISITION", "EVENT_ATTENDED", "MEDIA")
_PRIMARY_RANK = {b: i for i, b in enumerate(
    _PRIMARY_FIRST + tuple(b for b in BUCKETS if b not in _PRIMARY_FIRST))}


def canonical_key(text: str) -> str:
    """The identity function. FROZEN -- see the module docstring.

    Order-preserving by design. An order-insensitive token-set key was measured to catch
    594 more duplicate lines (+1.2/question), and buys that by making "Alice called Bob"
    and "Bob called Alice" the same entity. Not worth it.
    """
    s = unicodedata.normalize("NFKC", text)
    s = _BRACKETED.sub(" ", s)
    s = _NON_ALNUM.sub(" ", s.casefold())
    words = s.split()
    while words and words[0] in _ARTICLES:
        words.pop(0)
    return " ".join(words)


@dataclass
class EntityGroup:
    """One real entity and every record that mentions it."""

    key: str
    records: list[Record] = field(default_factory=list)

    @property
    def representative(self) -> Record:
        """Longest text wins, ties broken by text ascending -- deterministic, and the
        longest form is the most specific one the extractor produced ("Wooden coffee
        table with metal legs" over "coffee table")."""
        return max(self.records, key=lambda r: (len(r.text), r.text))

    @property
    def buckets(self) -> list[str]:
        """Every bucket this entity is filed under, in taxonomy order."""
        seen = {r.bucket for r in self.records if r.bucket}
        return sorted(seen, key=lambda b: _BUCKET_RANK.get(b, len(BUCKETS)))

    @property
    def primary_bucket(self) -> str:
        """The bucket the entity sorts under. See `_PRIMARY_RANK`."""
        bs = self.buckets
        return min(bs, key=lambda b: _PRIMARY_RANK.get(b, len(BUCKETS))) if bs else ""

    @property
    def categories(self) -> list[str]:
        """`bucket/subtype` for every filing, deduped, taxonomy order then subtype."""
        seen = {(r.bucket, r.subtype) for r in self.records if r.bucket}
        return [f"{b}/{s}" for b, s in
                sorted(seen, key=lambda bs: (_PRIMARY_RANK.get(bs[0], len(BUCKETS)),
                                             bs[1].lower()))]

    @property
    def date(self) -> str:
        """First non-empty date in group order. Chronological selection needs date
        parsing, which is a separate change (`compose._date_key`) with its own
        pre-registration -- deliberately not smuggled in here."""
        for r in self.records:
            if r.date:
                return r.date
        return ""

    @property
    def supersedes(self) -> str:
        for r in self.records:
            if r.supersedes:
                return r.supersedes
        return ""


def group_entities(records: Iterable[Record]) -> tuple[list[EntityGroup], list[Record]]:
    """Splits records into ITEM entity groups and everything else, untouched.

    Only ITEM records group. The verified mechanism is ITEM-ITEM double counting; FACT or
    PREF prose that happens to mention the same entity is a residual risk, recorded and
    deliberately not addressed here -- collapsing a FACT into an ITEM would change what
    the record asserts, not merely how often it appears.

    Group order is first appearance in the input, so callers control ordering by sorting
    their input rather than by anything hidden in here.
    """
    groups: dict[str, EntityGroup] = {}
    order: list[str] = []
    others: list[Record] = []
    for r in records:
        if r.kind != "ITEM":
            others.append(r)
            continue
        key = canonical_key(r.text)
        if key not in groups:
            groups[key] = EntityGroup(key=key)
            order.append(key)
        groups[key].records.append(r)
    return [groups[k] for k in order], others


def duplicate_count(records: Sequence[Record]) -> int:
    """Redundant ITEM lines -- how many would collapse. Used by the offline render diff
    that runs before any LLM budget is spent."""
    items = [r for r in records if r.kind == "ITEM"]
    groups, _ = group_entities(items)
    return len(items) - len(groups)
