# Aeon v4.2: Cross-Model Position, the Localised Residual, and the Named Successor Problem

*Revised 2026-08-28, after the tracks in the original draft were checked against measurement.
Track B was killed by its own pre-registered Step 0 (§4); Track A shipped with no accuracy claim
(§5); Track C was withdrawn against five prior nulls (§6). The claims in §1–2 have been corrected
to the arms and comparators they actually rest on.*

---

## 1. Where Aeon stands

Every figure is regenerated from a committed result file by
`scripts/longmemeval/cross_model_table.py`. LongMemEval-S, n=500, **judge held fixed at
`gemma4:31b-cloud` in every cell**, so the reader is the only variable across model columns.

| arm | correct | acc | median ctx chars | median gen s |
|---|---|---|---|---|
| composite @ `gemma4:31b` *(v1 config — the §8b headline)* | 429 | 85.8% | 26,056 | 0.85 |
| **composite @ `gemma4:31b`** *(config-parity control — use this for cross-model claims)* | **428** | 85.6% | 25,822 | 0.93 |
| oracle-precision @ `gemma4:31b` | 419 | 83.8% | 5,654 | 0.51 |
| **composite @ `glm-5.2`** | **448** | **89.6%** | 25,822 | 4.17 |
| oracle-precision @ `glm-5.2` | 457 | 91.4% | 5,654 | 0.78 |

**429 and 428 come from different `compose()` configs and must never appear in one sentence
without saying which arm each is.** The oracle is handed the gold `has_answer` turns; the
composite uses no gold annotations at all.

### What can and cannot be claimed from this

- **Against its own gold-evidence ceiling:** +9 (p=0.391) on `gemma4`, −9 (p=0.314) on `glm-5.2`.
  Both are **failures to reject, not demonstrations of equivalence** — with discordant pairs in
  the 80s the interval admits a real difference of several questions. The defensible sentence is:
  *the composite reaches oracle accuracy from production-available inputs, and no significant
  difference is detectable at n=500.* "Beats the oracle" and "establishes parity" are both wrong.
- **448 as-scored, +1 repaired.** Four questions returned an empty generation that carried no error
  marker and was scored as a wrong answer (`§8c.5`). Re-run under the fixed harness at identical
  config with prompt parity enforced — all four prompts matched byte-length exactly — all four now
  answer substantively and **1 is correct**. Written as a repair, never as a replacement: 448 is the
  artefact every contrast is computed against, and the other 3 are ordinary wrong answers.
- **The stronger claim, and the one to lead with:** on the **411 of 500 questions (82.2%)** needing
  one or two evidence turns, the composite **beats gold-evidence selection under both readers** —
  92.9% vs 90.7% at one turn, 93.0% vs 91.7% at two. That is sharper and more defensible than any
  parity statement, and it is a per-stratum result rather than a pooled one.
- **Context efficiency is against extract-then-compute, not the oracle.** The 3.9× figure belongs
  to that comparator. Against the oracle the composite uses **4.6× more** context (25,822 vs
  5,654). Detaching that comparator is the exact attribution error an external evaluation of this
  work already made (`§6.11`).

### Where Aeon is genuinely differentiated

| dimension | Aeon v4 |
|---|---|
| write-path latency | 163 ns WAL / 2.2 µs mmap insert — no LLM on the write path |
| consolidation | asynchronous C++23 Dreamer, zero query-time penalty |
| prompt context | ~25.8k chars (~6k tokens) |
| multi-tenant isolation | hard per-tenant kernel files (`{user_id}.atlas`), not a metadata filter |
| state supersession | durable `EdgeType.SUPERSEDES` edges in Trace |
| erasure | automatic lineage cascade keyed on `Provenance.session_id` |
| point-in-time recall | `as_of` traversal derived from provenance (§5) |

## 2. The competitor claims, checked against their sources

*Every figure below was verified on **2026-08-28** against the publishing source, and every quote was
then re-checked against the raw page rather than a summariser's paraphrase. The table in the first
draft of this document was wrong in three ways, and all three flattered the comparison.*

| system | headline | what the number is | reader | judge | source (accessed 2026-08-28) |
|---|---|---|---|---|---|
| MemPalace | 96.6% | **R@5 retrieval recall** — explicitly *not* QA accuracy | n/a | n/a | [BENCHMARKS.md](https://github.com/MemPalace/mempalace/blob/main/benchmarks/BENCHMARKS.md), Mar 2026 |
| Exabase M-1 | 96.4% | QA at top-50 retrieval | Gemini 3 Flash | **Gemini 3 Flash — self-judged** | [exabase.io](https://exabase.io/research/exabase-achieves-state-of-the-art-on-longmemeval-benchmark), Aug 7 2026 |
| OMEGA | 95.4% | **task-averaged**; raw **466/500 = 93.2%** | GPT-4.1 | **GPT-4.1 — self-judged** | [omegamax.co](https://omegamax.co/benchmarks), Feb 2026 |
| Mastra | 94.87% | **unweighted mean over 6 categories**, `longmemeval_s` | `gpt-5-mini` | `gpt-4o` | [mastra.ai](https://mastra.ai/research/observational-memory), Feb 9 2026 |
| Mem0 | 94.4% | overall accuracy | not disclosed | GPT-4o *(per their blog, not a methods section)* | [mem0.ai/research](https://mem0.ai/research), Aug 27 2026 |
| **Aeon** | **89.6%** | **raw micro over 500, abstention included** | `glm-5.2` | `gemma4:31b` | `composite_glm52_n500.json` |

### The three defects, with the receipts

**1. A retrieval metric was being read as a QA score — and that error is entirely ours.**
MemPalace's own document flags it in bold, above the table:

> "**Important caveat — read before quoting this table.** MemPal's `R@5` in this table is **retrieval
> recall**: is the labelled session for this question inside the top-5 retrieved candidates? Several
> of the other systems below publish **end-to-end QA accuracy** — a different metric that scores
> whether the system's generated answer is correct. **Retrieval recall and QA accuracy are not
> comparable; a system can have 100% retrieval recall and 40% QA accuracy, and vice versa.**"

**MemPalace reported honestly and warned against precisely the misuse this document committed.** No
criticism attaches to them; the defect was in the first draft of §1 here, which lifted 96.6% into a
column of QA accuracies. Recorded as our error, not theirs — and it is the same error §6.11 records
an external evaluation making about *this* work, in the opposite direction.

**2. Two of the three leaders are self-judged.** Verbatim:

> Exabase M-1: "We used a smaller model: Gemini 3 Flash, **for both answering questions and judging
> correctness**." — and "M-1 achieves state-of-the-art results across all recall depths, reaching
> 96.4% at top-50 using Gemini 3 Flash as the answering model."

> OMEGA: "The test environment: OMEGA v1.0.0, **GPT-4.1 as the generation and grading LLM**,
> bge-small-en-v1.5 ONNX embeddings, running on an M1 MacBook Pro with 16GB RAM."

§8c.1 of the findings spends a full section on why that is disqualifying — a self-judged arm is
comparable to nothing — and this project re-judged 80 hypotheses under an explicit external judge to
establish that its own arms were not self-judged. Neither of those two reports an equivalent check.
Mastra and Mem0 both use an external `gpt-4o` judge and are not implicated.

**3. Two headline figures are macro averages; 89.6% is a micro average.** Verbatim:

> Mastra: "gpt-4o is used as the judge model. Each question is evaluated as correct/incorrect
> (binary). **Overall accuracy is the unweighted average across all 6 categories.**"

> OMEGA: "OMEGA scores 95.4% **(466/500)**" — its own raw count, which is **93.2%**.

### The comparison at matched metrics

Aeon's composite @ `glm-5.2`, recomputed on each competitor's own metric from the committed
`composite_glm52_n500.json`:

| metric | Aeon | comparator |
|---|---|---|
| raw micro, all 500 *(Aeon's headline — the strictest available)* | **89.6%** (448/500) | OMEGA raw **93.2%** (466/500) |
| macro over OMEGA's 5 categories | **90.9%** | OMEGA **95.4%** |
| macro over 6 categories, abstention excluded (Mastra's metric) | **92.1%** | Mastra **94.87%** |
| raw micro, abstention excluded | **91.3%** (429/470) | — |

**The gap is real but roughly half what the original table implied**: against Mastra on Mastra's own
metric it is **2.8 points**, not 5.3; against OMEGA on raw counts it is **18 questions**, not the 29
that 89.6 vs 95.4 suggests.

**None of this is a new headline.** 89.6% stays the number this project reports. Micro accuracy over
all 500 with abstention included is the strictest of the available definitions, and switching metrics
after seeing them is precisely the manoeuvre this section criticises. The recomputation exists to
size the gap honestly, not to shrink it by choosing.

**One confound that is NOT resolved.** Every competitor is graded by a frontier judge — `gpt-4o`,
GPT-4.1, Gemini 3 Flash — and LongMemEval specifies GPT-4o. Aeon is graded by `gemma4:31b`. Whether a
31B judge is systematically stricter than GPT-4o on this task is **unmeasured**, so an unknown part
of the residual gap is judge severity rather than system quality. Every *internal* contrast in this
project is judge-invariant (the same judge scores both sides of every paired test), so the confound
lives only in this external comparison — but it lives here undismissed, and it is cheap to test.

### On reader capability

Mastra's own 10.6-point collapse under a reader swap (94.87% on `gpt-5-mini`, 84.23% on `gpt-4o`) is
external evidence that these numbers are reader-sensitive. This project measures the same thing
internally: `gemma4:31b` → `glm-5.2` moves the composite **+20** and the gold-evidence ceiling **+38**.
**Consolidation caps the headroom a stronger reader can exploit** — a record is a lossy summary fixed
at write time, while raw turns retain everything a better reader might newly extract.

**But the reader lever saturates, and this was tested rather than assumed (§8c.9).** A second
frontier-class reader, `deepseek-v4-pro:cloud`, run on the dev split at **byte-identical prompts**
with the judge held fixed and recorded, lands at **223/252 against `glm-5.2`'s 224 — net −1,
p=1.000**, with only 13 of 252 pairs discordant. One rung of the ladder does not predict the next.

So *"the gap to the 94–96% systems is reader capability"* is **too coarse as stated**, and this
document previously overstated it. What is established is that a **weak** reader binds. What is
**not** established is that further reader capability is a lever, and one measurement says it is not.
With §2's findings alongside it, the remaining candidates for the difference are the memory pipeline,
the metric definition, and the judge.

## 3. The residual, localised

`§8c.7`: the composite leads at one and two answer turns on both readers. Its **entire** deficit is
the 68 questions requiring three or more.

| answer turns | n | composite @ `glm-5.2` | oracle @ `glm-5.2` | composite @ `gemma4` |
|---|---|---|---|---|
| 1 | 182 | **92.9%** | 90.7% | 91.2% |
| 2 | 229 | **93.0%** | 91.7% | 89.5% |
| **3+** | **68** | **75.0%** | **89.7%** | **55.9%** |

> Consolidation is not losing evidence. It is losing multi-fact aggregation over evidence it holds.
> More context cannot fix that — it is the thing more context makes worse (`§8c.7`).

Episodic budget is ruled out on an experiment this project already ran to completion: deeper
retrieval raised coverage 56.9% → 94.4% and delivered **+3 questions of 500 at 2.69× context and
1.8× latency**, because it improves findability and degrades aggregation.

## 4. Track B — killed by its own Step 0

The original draft proposed pre-aggregated cluster records written by the Dreamer. A
pre-registered kill-check read all 17 losses individually first
(`AGGREGATION_STEP0.md`, `scripts/longmemeval/aggregation_failure_read.py`, zero LLM calls).

**The losses are not enumerations.** 5 are windowed ("in the past two weeks", "in March"), 3 are
orderings, 3 are ad-hoc predicates ("projects I *led*"), 2 are argmax, 1 is arithmetic — and only
**3 of 17** are the plain category counts a cluster record could serve.

**13 of 17 apply a filter the question supplies at read time.** That refutes the premise, not the
implementation: *you cannot pre-aggregate over a predicate you do not know at write time.* A
deterministic roster header would print gold on 1 of 17, a wrong number on 2, and nothing defined
on 7 — reaching at most 53 from a 51/68 baseline against a bar of 58. It cannot clear its own gate
at zero losses, and on two questions it would have *confirmed* the model's wrong answer. Not built.

An LLM-written variant inherits that defect and adds one: `Provenance` holds a single `session_id`,
so a cluster record spanning three sessions destroys two lineage links — and those links are the
right-to-erasure cascade index.

### The named successor problem

**Dynamic read-time filtered aggregation**: computing a count, an order or an extremum over a
subset selected by a predicate — most often a relative date window — that exists only in the
question. Every mechanism tried so far sits on the wrong side of that boundary. Write-time
consolidation cannot see the predicate; render-time totals cannot condition on it; reader-side
instruction has returned five nulls at asking the model to do it unaided.

One unmeasured interaction is recorded as a hypothesis, not a plan: 8 of the 17 losses need date
arithmetic over ITEM records, and `order_records()` deliberately does not sort ITEMs
chronologically — it groups them for counting, and the measured chronology null left ITEM ordering
untouched by design. That interaction has never been tested and the losses sit on it.

## 5. Track A — shipped, as a product capability with no accuracy claim

`as_of` point-in-time traversal is implemented (`timeline.visible_at()`, `edges_before()`,
`compose_from_store(as_of=…)`, `tests/test_time_travel.py`).

**The design in the original draft does not exist and cannot.** It proposed filtering records on a
per-record `timestamp`; there is none. `NodeHeader` is full at 64 bytes, and `Record.encode()`
writes exactly seven fields while `decode()` folds an eighth into `text`. Validity is therefore
**derived from provenance** — the latest `event_time` among the turns a record came from — which is
also the better semantics: valid time, not transaction time. "What did I know in March" is a
question about the conversation, not about when the consolidator ran.

Both halves of the assembly are cut, because a record block filtered to March beside excerpts
quoting September is a leak that reads as a historical view. `as_of=None` is byte-identical to the
measured read path.

**No accuracy claim.** LongMemEval asks every question at T_now and its corpora predate the question
date, so this cannot move a benchmark number and is not gated on one. Fidelity bound: `run_merge()`
removes a stale record rather than superseding it, so `as_of` reconstructs what supersession
retired, not what the merge dropped.

## 6. Track C — withdrawn

The draft proposed an abstention prompt guard. Its wording is a near-rewrite of `_PREMISE_GUARD`,
which was measured at **429 → 401 (−28, p=0.0018)**, with single-session-preference collapsing 28 → 9
because advice questions ("what should I meal-prep?") look identical to unanswerable ones under a
literal premise check. It would be the **sixth** reader-side instruction in a family with five
nulls, and the standing conclusion is that instructions do not change how the reader *classifies*.
Abstention is also the smallest bucket (+11 maximum). If revisited it needs a question-shape-scoped
**mechanism**, not a prompt.

## 7. Honest ceiling arithmetic

From 448 on `glm-5.2`: aggregation ~+10 (now known not to be reachable by write-time clustering),
abstention +11 maximum with the naive fix already measured at −28, empty-generation repair **closed
at +1**. **The ceiling of the memory layer's own tracks is roughly 458–462 (91.6–92.4%).**

The earlier version of this section said crossing 470 "requires a stronger reader". **That no longer
survives §8c.9**: a second frontier-class reader at identical prompts moved the result by −1. The
honest statement is that **no currently identified lever reaches 470** — not the memory layer's own
tracks, and not, on the one datum available, a further reader upgrade.

Two of the three things §2 leaves open sit outside that arithmetic entirely, and both are cheap:
**judge severity** (unmeasured, and every competitor is graded by a frontier judge while this is
graded by a 31B one) and **metric definition** (at Mastra's own metric the gap is 2.8 points, not
5.3). Neither is a lever on the system; both change what the gap is understood to mean. The
substantive open research question remains dynamic read-time filtered aggregation (§4).

## 8. Reproducing anything here

```bash
python3 scripts/longmemeval/cross_model_table.py              # §1 arms and contrasts
python3 scripts/longmemeval/metric_variants.py                # §2, Aeon on each competitor's metric
#   composite_deepseek_dev.json  vs  composite_glm52_n500.json  # §2, the reader-saturation probe
python3 scripts/longmemeval/residual_error_profile.py         # residual partition
.venv/bin/python scripts/longmemeval/aggregation_failure_read.py   # §4, aborts unless 68/51/61
.venv/bin/python -m pytest tests/test_time_travel.py -q            # §5
```
