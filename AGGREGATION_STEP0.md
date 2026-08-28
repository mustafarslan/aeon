# Step 0 — reading the 3+ aggregation losses, before building anything

**Verdict: the deterministic roster header is killed. It would convert at most 2 of 17 losses,
print a wrong number on 2, and have no defined value on 7.**

Reproduce: `.venv/bin/python scripts/longmemeval/aggregation_failure_read.py`
(zero LLM calls; writes `reproducibility_benchmarks/longmemeval/aggregation_failure_read.json`).

---

## 1. What was checked and why

`spatiotemporal.md` proposes pre-aggregated cluster records to close the 3+-evidence deficit
that `CROSS_MODEL_SESSION_REPORT.md §7.4` localised. The cheap, LLM-free version of that idea
is a per-category roster header computed over `EntityGroup`s at render time:

```
ITEM(ACQUISITION/music album) — 3 items:
```

That is worth building only if the header would print the **gold** number. A header that
prints a wrong number is worse than no header, because it hands the reader a computed value
to be confident about — and this project has already measured what confident-and-wrong costs
(`_PREMISE_GUARD`, −28).

The partition reproduces §7.4 exactly before anything downstream is trusted:
**n=68, composite 51, oracle 61.**

## 2. The losses are not enumerations

| shape | n | what the question asks |
|---|---|---|
| **windowed** | 5 | count a category *inside a date window* — "in the past two weeks", "in March", "in this year", "before I made an offer" |
| **order** | 3 | rank by date — "what is the order of the three trips" |
| **predicate** | 3 | count members satisfying a property the taxonomy does not encode — "projects I *led*", "kitchen items I *replaced or fixed*" |
| **argmax** | 2 | extremum of a per-entity quantity — "which store did I spend the most at" |
| **enumerate** | 3 | plain category count — the roster's design target |
| **arithmetic** | 1 | sum quantities stated inside records |

**13 of 17 apply a filter the question supplies at read time.** That is the finding, and it
refutes the premise of Track B as written: *you cannot pre-aggregate over a predicate you do
not know at write time.* Write-time clustering can only produce the unfiltered total, which is
the number the reader already gets wrong in the other direction.

One further correction to the denominator: `gpt4_7f6b06db` is an **empty generation** — one of
the four §5 harness artefacts. It is a missing measurement, not an aggregation loss. The real
figure is 16.

## 3. What a roster header would actually print

| verdict | n | |
|---|---|---|
| NOT-A-COUNT | 6 | order / argmax / arithmetic — a count header is not the shape of the answer |
| AMBIGUOUS-FRAGMENTED-RUNS | 6 | the target category splits into runs of differing sizes, so there is no single number to print |
| WOULD-PRINT-WRONG | 2 | |
| WOULD-PRINT-GOLD | 1 | `1a8a66a6` — magazines, `ACQUISITION/magazine=2` and `MEDIA/magazine=2`, gold 2 |
| GOLD-ONLY-IF-MERGED-ACROSS-BUCKETS | 1 | `bf659f65` — albums, `ACQUISITION/music album=2` + `POSSESSION/music album=1` = **3 = gold** |
| NO-SINGLE-RUN | 1 | |

The 6 ambiguous cases are **not neutral**. Several competing headers carrying non-gold numbers
still anchor the reader; they simply do so without a single number to attribute the error to.
So the harm floor is "≥2, plausibly more", which only sharpens the verdict.

The two that fail loudest are worth naming, because they are the harm case rather than a null:

- `gpt4_a56e767c` — "how many movie festivals": `EVENT_ATTENDED/film festival=3`, model answered
  **3**, gold **4**. The header would have *confirmed* the wrong answer.
- `9d25d4e0` — "jewelry acquired in the last two months": `POSSESSION/jewelry=4`, model answered
  **4**, gold **3**. Same shape — the window is what makes gold 3, and the header cannot see it.

## 4. The one real signal

`bf659f65` is the canonical case `records.py`'s own docstring is built on ("how many albums have
I purchased?", gold 3, every arm answers 2). The three albums **are all present** — they are
split `ACQUISITION/music album` ×2 and `POSSESSION/music album` ×1. Counting *entities sharing a
subtype across buckets* yields 3. Counting the primary-bucket run yields 2.

So the open design question in the plan resolves, and it resolves against the shipped keying:
**per-run counting is the wrong axis; subtype-across-buckets is the right one.** But it converts
2 questions, not 10 — which is not a track, it is a footnote.

## 5. A methodological correction made during this check

The first version of this script scored convertibility as *"some run in the corpus has exactly
the gold size"* and reported **7 of 17**. That was coincidence matching: on `88432d0a` (gold 4,
"how many times did I bake in the past two weeks") it matched `TRAVEL/trip=4`, and on `6d550036`
(gold 2) fifteen distinct runs had size 2 in a corpus of 121 groups. Replaced by a table that
names the target category per question — a judgement, stated and inspectable — with the
arithmetic over it mechanical.

A second flattering reading was removed after that: scoring the largest matching run as the
"printed" value turned `6d550036` into a conversion, when its target fragments into
`PROJECT/=2, PROJECT/event=2, PROJECT/academic=1, PROJECT/art series=1` and no single number
would be printed at all. Those cases are now `AMBIGUOUS-FRAGMENTED-RUNS`, which moved the
headline from 2 conversions to 1 plus 1 conditional.

## 6. Where this leaves v4.2

Rung 1 is not redesignable into the same shape — the losses are filtered aggregations, and a
static write-time or render-time total is the one thing that cannot serve a read-time filter.
Rung 2 (LLM-written cluster records) inherits the same defect *and* the lineage blocker, so it
does not survive either.

What Step 0 does establish, and what it cost nothing to learn:

- The 3+ deficit is a **filtered-aggregation** problem, mostly **temporal** (8 of 17 are windowed
  or ordering questions requiring date arithmetic over ITEMs).
- `order_records()` deliberately does **not** sort ITEMs chronologically — that is the
  counting-critical bucket grouping, and the measured chronology null explicitly left ITEM
  ordering untouched (`v4-plan.md`: *"ITEM ordering is untouched"*). So the interaction between
  category grouping and date order has **never been measured**, and the losses sit exactly on it.
- That is a hypothesis, not a plan. It is also a reader-side rendering change, which is the
  family with five nulls — so it needs the same one-shot pre-registered treatment, and it should
  not be started on the strength of this document alone.
