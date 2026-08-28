# Aeon V4 — Empirical Findings for the Paper Revision

Working record for the next revision of the Aeon preprint (current: arXiv:2601.15311, which covers
the V3/V4.1 kernel). **This document is organised by claim, not chronologically** — `v4-plan.md` holds
the dated experimental log with full context, including every decision that was reversed. Everything
here is reproducible from committed scripts and result files; nothing is quoted from memory.

**Scope note.** The existing preprint's contribution is a memory *kernel* (WAL, INT8 quantisation,
blob arena, EBR, shadow compaction). The contribution below is different in kind: an **empirical study
of what actually determines end-to-end quality and latency when a memory engine feeds an LLM agent**,
and a resulting design thesis. The two are complementary — the second explains what the first should
optimise for.

---

## 1. Headline thesis

> For an LLM-agent memory system, the binding constraint is **precision per token delivered**, not
> retrieval recall and not kernel latency. Accuracy is **flat between the measured endpoints of ~5.6k
> and ~101k characters** (and *falls* at 271k), while latency and cost scale with context throughout.
> A memory engine's job is to send the *smallest sufficient* context, not the largest affordable one.
>
> *Sampling caveat*: the curve is characterised by three measured points (5.6k / 101k / 271k);
> intermediate sizes are not sampled, so "flat" is a statement about those endpoints, not a
> demonstration that no interior structure exists.

Supporting measurements, all n=500, same model/judge/seed (§2):

| configuration | accuracy | median context | LLM calls | median generation | **correct / 1k chars** |
|---|---|---|---|---|---|
| single-shot, session-expanded | 77.6% | 100,889 | 1 | 1.51 s | 3.85 |
| extract-then-compute (ETC) | 82.6% | 100,889 | 2 | 2.46 s | 4.09 |
| ETC, deep retrieval (`top_k`=200, `max_sessions`=20) | 83.2% | 270,972 | 2 | 4.47 s | 1.54 |
| **oracle-precision (answer turns ±1)** | **83.8%** | **5,654** | **1** | **0.51 s** | **74.10** |
| **composite — records + episodic (this work)** | **85.8%** | **26,056** | **1** | **0.85 s** | **32.93** |

The oracle row (§6.1) is not a system: it is handed the gold `has_answer` turns. Its significance is
that the same accuracy is reachable at **1/18th the context, half the LLM calls, and 1/5th the
generation latency** — 1/18th being 5,654 against the 100,889 of the two full-context arms.

**The composite is the first arm to reach that accuracy without gold annotations** (§8b, §10). Paired
on all 500: **+16 vs ETC (McNemar +59/−43, p=0.137)**, **+41 vs single-shot (+77/−36, p<0.001)**, and
**+10 vs the oracle (+54/−44, p=0.363) — statistically tied with the ceiling, nominally above it.**

*Two attributions to keep straight, because an external summary of this work conflated them.* The
1/18th-context and 1/5th-latency figures belong to the **oracle**, measured against the 100,889-char
arms. The composite's own efficiency claim is **against ETC: 3.9× less context and 2.9× faster
generation**. Measured against the *oracle* the composite uses **4.6× more** context (26,056 vs 5,654)
and is **1.7× slower** (0.85 s vs 0.51 s) — it buys its independence from gold annotations with
context, and that is the honest trade.

---

## 2. Experimental setup

- **Benchmark**: LongMemEval-S (cleaned), n=500, stratified across 6 question types plus the
  30-question unanswerable ("abstention") augmentation.
- **Model**: `gemma4:31b-cloud` via Ollama, `temperature=0.0`, `num_ctx` auto-sized from actual prompt
  length (a fixed 8192 default was found to be silently truncating ~36% of real prompts; fixed).
- **Judge**: same model, LongMemEval's official `anscheck` prompt, abstention variant for `_abs` ids.
- **Retrieval**: Aeon Trace + `build_expanded_context(unit="full_session")`, `base_top_k=30`,
  `max_sessions=10` unless stated.
- **Embeddings**: mpnet; turn text embedded *bare*, date prefix stored but not embedded (a shared
  ~25-char prefix on every document measurably compresses the angular spread of the embedding space).
- **Reproducibility**: every arm's per-question record (hypothesis, judge verdict, context size,
  per-stage timings) is committed under `reproducibility_benchmarks/longmemeval/`.

### 2.1 Measured noise floor — required to read any result below

Re-running an **identical configuration** on the same questions flips a non-trivial fraction of
verdicts, because the cloud-served model is not deterministic at `temperature=0`:

| measurement | flips | rate |
|---|---|---|
| opportunistic (n=50 sample nested in an n=500 run, identical config) | 4/50 | 8.0% |
| deliberate (n=100 nested repeat, pre-registered) | 6/100 | 6.0% |
| **pooled** | **10/150** | **6.7%** |

Attribution: ~5/6 generation nondeterminism, ~1/6 judge nondeterminism (byte-identical answer, opposite
verdict). Extraction output reproduces verbatim on only **31/50** repeats. **Retrieval is byte-stable
(479/479 contexts identical on rebuild)** — the nondeterminism is entirely LLM-side.

Consequence: at n=500 the net-delta standard deviation is **≈6 questions (≈1.2 points)**. Any reported
delta below ~12 questions is not distinguishable from noise. **Two prompt-engineering changes earlier
in this project were reverted on deltas inside this band**, i.e. on noise — which is why the noise
floor is reported as a primary result rather than a footnote.

---

## 3. Methodological contributions

These are reusable beyond Aeon and are arguably the most transferable part of the work.

1. **Measure the noise floor before interpreting any delta.** Nested-sample repeats make this nearly
   free: an n=100 stratified sample with the same seed is a subset of the n=500 sample, so the
   overlapping questions constitute a controlled repeat.
2. **Pre-register bars, sized at ≥2× the measured noise sd, before running.** Bars were committed to
   version control *before* each run in this study; one bar was later found mis-calibrated
   (unpassable by construction) and that error is recorded rather than quietly repaired.
3. **Free → cheap → expensive experiment ordering.** For a retrieval change: (a) rebuild contexts
   offline and measure evidence *coverage* (0 LLM calls); (b) run only the affected question cohort
   (~12 min); (c) only then run the full paired n=500 (~4 h). Each stage can kill the hypothesis.
4. **Cohorts prove mechanism; only aggregates decide worth.** Demonstrated concretely in §5.2: a
   cohort predicted 17/25 conversions and the aggregate delivered exactly 17/25 — while the *net*
   effect on the full benchmark was +3 questions, i.e. nothing.
5. **Answer-turn attribution.** LongMemEval annotates which turns contain the answer (`has_answer`).
   Rebuilding each context and checking which annotated turns arrived splits errors into
   *retrieval-side* vs *downstream* mechanically, over all 500 questions, with no LLM calls.
6. **Validate proxy metrics against known-good cases before trusting them** (§6.2).

---

## 4. Error attribution (n=500, ETC configuration)

Using `has_answer` annotations; 21 questions have no answer-bearing turn by construction and are
excluded.

| | correct | retrieval miss | retrieved-but-wrong |
|---|---|---|---|
| all types | 369 | **27 (25% of errors)** | 83 (75% of errors) |

Causal check:

| | mean answer-turn recall | all answer turns present |
|---|---|---|
| correct answers (n=369) | 99.5% | 98.9% |
| wrong answers (n=110) | 85.9% | 75.5% |

**Only ~29% of remaining error is attributable to the memory engine**; ~71% is the generator failing
with the evidence in hand. The 98.9% figure also bounds the matcher's own false-miss rate at ~1%.

The 27 retrieval misses decompose into two mechanisms requiring different fixes:

- **20 partial recall** — question needs several answer turns, retrieval delivers some.
- **7 complete miss** — 0 of N turns retrieved. All seven share one cause: *the answer is a passing
  aside inside a turn whose dominant topic is something else* (e.g. a request for BBQ-sauce recipes
  ending "By the way, I just got a smoker today", for the question "what kitchen appliance did I buy").
  Whole-turn embedding is dominated by the turn's main topic, so the turn ranks far below where its
  answer-bearing content warrants. **This is the direct empirical motivation for sub-turn chunking.**

---

## 5. Principal findings

### 5.1 A harness defect worth reporting for benchmark hygiene

LongMemEval ships a `question_date` field — the reference "now" that makes relative-time questions
answerable. It was never passed to the model in any arm. Consequences were measurable and large:
**21 questions (all temporal-reasoning, all wrong, 17 in the wrong-under-every-arm hard core)**
explicitly complained about or *hallucinated* a current date (one output: *"The current date is
2023/01/15"*, invented). Passing the field:

| arm | temporal pre → post | McNemar |
|---|---|---|
| ETC | 84 → 103 (**+19**) | +23/−4 |
| single-shot baseline | 67 → 83 (**+16**) | +21/−5 |

19 of the 21 named cohort questions (90%) became correct. Drift was excluded as an explanation: the
gain concentrates in the named cohort, the outputs show arithmetic that was previously impossible, and
six of seven types moved only within noise. **Any LongMemEval temporal-reasoning number published
without this field is measuring a broken configuration** — including this project's own earlier
numbers, which are superseded.

### 5.2 More retrieval does not help — it trades findability against aggregation

Deep retrieval (`top_k`=200, `max_sessions`=20) was validated through the full free→cheap→expensive
ladder, and still failed:

- **Coverage (free)**: answer-turn coverage of the 27 misses rises 56.9% → 94.4%; `base_top_k`, not
  `max_sessions`, is the binding constraint.
- **Cohort (12 min)**: 17/25 previously-failing questions become correct — ~10× the expected noise.
- **Aggregate (4 h)**: **+3 questions overall (413 → 416)**, i.e. nothing, at 2.69× context and 1.8×
  generation latency. The cohort reproduced *exactly* (17/25) — the gain was real and was cancelled.

Decomposing by how many answer turns a question needs explains why:

| question shape | deep helps | deep hurts | net |
|---|---|---|---|
| 1 answer turn | 13 | 4 | **+9** |
| 2 answer turns | 7 | 12 | **−5** |
| 3+ answer turns | 9 | 11 | **−2** |

**Deep retrieval improves findability and degrades aggregation.** Multi-session accuracy fell 96 → 90
(+7/−13) — a larger haystack makes the model miscount. For multi-evidence questions, coverage improved
while accuracy fell.

### 5.3 Compression preserves accuracy — if it preserves conversational neighbourhood

The oracle-precision arm (context = annotated answer turns ±1 neighbouring turn, single-shot) reached
**83.8% at 5,654 median chars, one LLM call, 0.51 s generation** (§1). Paired against ETC the accuracy
delta is +6 (+49/−43) — **noise; this is parity, not superiority.** The result is about cost.

The mechanistically important row is **single-session-user: 63/64**, a *real* +8 over ETC's 55 and
above the uncompressed baseline's 60. Context: ETC's own extract step is a compression layer, and it
**broke** this question type (60 → 55) through loss of pragmatic licensing — e.g. "picked it up at
Trader Joe's" answers "what brand?" only under conversational implicature that a bare fact list
destroys. Oracle compression **with one neighbouring turn on each side does not break it.**

> **Design constraint (transferable):** compress aggressively, but never below the conversational
> neighbourhood. One turn of context on each side of an evidence turn was sufficient here to preserve
> the implicature that bare-fact compression destroys.

---

## 6. Negative results and corrections (reported deliberately)

Credibility of §5 depends on these being visible.

### 6.1 Limits of the oracle
The oracle-precision arm uses gold `has_answer` annotations no production system has. It bounds the
**ceiling** of a perfect compressor; it does not demonstrate that a real reranker reaches it. That gap
is the direction's principal engineering risk. Additionally, multi-evidence types trend slightly
negative under compression (temporal −3, multi-session −1, both within noise, both with heavy two-way
churn), so perfect precision is **not strictly dominant**.

### 6.2 A proxy metric that failed validation, and was withdrawn
An automated extraction-vs-compute split (content-word overlap between answer turns and extracted
facts) produced "extraction loss = 36 (33% of errors)". Validated against questions answered
**correctly** with all answer turns retrieved — where extraction demonstrably succeeded — it flagged
**50%** of them as "extraction incomplete" (91% at a stricter threshold), a *higher* false-positive
rate than on wrong answers. It measures turn verbosity, not extraction fidelity. **The number is
withdrawn**; the script retains a printed warning so it cannot be re-quoted.

### 6.3 A keyword-grep method that systematically misattributed
An earlier locus analysis grepped assembled context for a *topic keyword* and concluded 3 of 5
aggregation failures were extraction losses. Checking the annotated *answer turns* individually
reversed 2 of 3: aggregation questions repeat their topic word across many sessions, so keyword
presence does not imply the answer-bearing turn was retrieved. **The method under-reports retrieval
misses on exactly the question shape it was built to diagnose.**

### 6.4 Prompt engineering: three failures on one target
Three independent attempts to fix a single-session regression via prompting — a supersession
instruction, a literalism relaxation, and a system-prompt frame correction — produced, respectively:
noise-level churn (reverted on noise), *zero* movement on the targeted questions (identical refusals
word-for-word), and 0/8 conversions in a controlled 6-cell grid. The failure was correctly diagnosed
only when attribution showed the missing ingredient was never in the model's input.

### 6.5 Query routing: three measurements, never worth building
Routing between configurations was evaluated three times (ETC vs single-shot; post-date-fix; deep vs
shallow retrieval). In every case the **type-label oracle ceiling** sat at or within noise of the
better always-on configuration (+1.4, +7, +12 questions against thresholds of ~11.5), and the one
trained classifier scored *below* its oracle (77.6% out-of-fold). Reported as a consistent negative:
**routing's value here is latency, never accuracy.**

### 6.6 A mis-calibrated acceptance bar
One pre-registered bar set floors above the technique's own previously-achieved numbers, making it
unpassable independent of whether the change worked. Recorded rather than silently repaired.

---

### 6.7 Four reader-side interventions, four nulls — and the sequence is the result

Each was pre-registered with bars before running, on a held-out-gated dev split (n=252, frozen records,
measured answer-stage noise 3.0% → sd 2.7).

| intervention | Δ | McNemar | verdict |
|---|---|---|---|
| co-referent collapse (dedup entities across buckets) | +1 | +3/−2, p=1.000 | null |
| chronological ordering of dated records | +1 | +3/−2, p=1.000 | null |
| reconciliation-aware counting directive | −1 | +6/−7, p=1.000 | null |
| the library's own (never-sent) system prompt | **−13** | +3/−16, **p=0.0044** | harmful |

**No single row is the finding; the sequence is.** After chronological ordering the model *cites both
dates in its own answer and sums them anyway* — `"Three tops (bought 2023/08/11), Five tops (bought
2023/09/30) — Total: 8"` against a gold of 5. Under the reconciliation directive it answers `"a total
of 8 tops (three tops on 2023/08/11 and five tops on 2023/09/30)"`. The directive is present, read, and
loses. **The reader makes a semantic judgement — "revised total or second purchase?" — that
instruction, ordering and deduplication do not change.** Only removing the stale record from the
context does (§10). That is what justified moving the work from the read path to the write path, and it
is a transferable result about prompt-level intervention rather than about this benchmark.

Each null also falsified its own motivating mechanism, which is why they were worth running:
co-referent duplicates *are* 1.8× denser on overcount errors than the corpus median, and collapsing
them still fixed nothing — `gpt4_59c863d7`'s duplicate collapsed, its count moved 7→6, and it stayed
wrong against a gold of 5. **The duplicate was a co-occurring symptom, not the binding constraint.**

### 6.8 A latent regression shipped in the library and never fired

`COMPOSE_SYSTEM` was defined in the read path and **referenced nowhere** — not in the package, not in
the harness, not in tests — so every number in this document was produced with an empty system prompt.
Measured for the first time: sending it scores **207 against 220 (n=252, p=0.0044)**. The cause is
visible in the outputs rather than inferred — **median answer length collapses from 172 characters to
49**, because *"Answer as concisely as possible — a short phrase or sentence"* truncates exactly the two
types that need room: single-session-preference (−5), whose judge is a **rubric** rewarding a response
that reflects the user's stated preferences, and temporal-reasoning (−4), which needs the arithmetic
shown. It is retained, annotated with this measurement, rather than deleted — a production read path
will want *some* system prompt, and this records what a revised one must avoid.

### 6.9 An acceptance bar that was unpassable by construction — the third instance

The supersession work was pre-registered with a primary bar of ≥229 on a 252-question split. That bar
could not be cleared by a *perfect* fix: **knowledge-update carries 4 errors on the entire
500-question benchmark**, against a detection threshold of 11.6. The bar was withdrawn mid-flight,
before any result existed.

This is the third occurrence in this project's record — after the ETC-v3 floors (§6.6) and a set of
collateral guards placed at the baseline value with no noise band — **and the first written by the
author of this document.** The standing rule was "bars at ≥2× the measured noise sd"; ≥2σ alone did not
prevent any of the three. **The rule is amended to "≥2σ *and* checked against the available headroom
for the mechanism under test."**

### 6.10 A measurement that got worse with more data, and the worse number is the one recorded

The derived resolver that links a supersession marker back to the record it retires was first measured
at **4 of 9 (44%)** on 20 questions. Across all 55 available: **12 of 39 (31%)**, with 5 dangling and 22
unresolved. The failure modes are bare-date markers, paraphrases, and assertions the merge did not
execute. The larger, less flattering figure supersedes the first and is the one carried in the code.
31% is also the argument for the durable edge log in §10: it is not an enhancement layered on the
resolver, it is the answer to the resolver being weak.

### 6.11 External claims about this work, checked and rejected

An external summary of these results circulated with figures that do not survive checking. Recorded
because the corrections are more instructive than the claims:

| claim | what the repo actually measures |
|---|---|
| "matches the oracle at **1/18th the context, 2.9× faster**" | 1/18th and the latency ratio are the **oracle's** properties vs the 100,889-char arms. The composite uses **4.6× more** context than the oracle and is **1.7× slower**. Its true efficiency claim is vs ETC (3.9×, 2.9×). |
| "ingest enqueues in **163 ns into mmap C++ WAL**" | 163 ns is a **Python `set.add` under a lock** (`DirtyQueue.mark_dirty`), explicitly *not* crash-safe — its own docstring says process death loses every pending entry. The C++ insert path is 2.23 µs. No committed benchmark reproduces the 163 ns. |
| "`TraceRole::Action`, SIMD linear scans (RAT), architected for action-observation-reward trajectories" | `TraceRole` is exactly `{User, System, Concept, Summary}`. Nothing named **RAT** exists. `TraceBlockIndex` is SIMD but **block-pruned, not linear**. The design is an mmap episodic conversation log with session isolation and supersession. |
| "**MemPalace 96.6%** beats this work" | That figure is **retrieval recall @ top-5, not QA accuracy** — its own benchmark document warns the two are not comparable. The comparable figure here is **94.6%** (evidence reached the model on 473 of 500). |
| "V2: even gold oracles average ~37.5%" | One **n=40** bypass-oracle run on a single model with `MAX_ORACLE_STATES=10` — not a field average. The V2 paper reports a coding-agent memory at **72.5%**. |

## 7. Latency: where the time actually goes

| configuration | Aeon retrieval | LLM generation | Aeon's share |
|---|---|---|---|
| single-shot | 12.6 ms | 1.51 s | **0.83%** |
| ETC | 12.2 ms | 2.46 s | **0.49%** |
| ETC deep | 13.0 ms | 4.47 s | **0.29%** |
| oracle-precision | 12.6 ms* | 0.51 s | **~2.4%** |

\* retrieval cost of a real precision layer is not yet measured; the oracle arm has no retrieval step.

**The kernel's microbenchmarks (2.23 µs insert, 3.09 µs navigate, 4.70 ns SDOT) do not move
end-to-end latency at all.** They are table stakes. System latency is ~99% LLM, and LLM latency scales
with context — demonstrated directly: tripling context took generation from 2.46 s to 4.47 s, while
reducing it 18× took generation to 0.51 s. **The only lever a memory engine has on user-visible
latency is how many tokens it sends.** This is the strongest argument for the precision thesis and
should lead the paper's systems section.

---

## 8. Reproducibility index

| artifact | path |
|---|---|
| Experimental log with full context | `v4-plan.md` |
| Answer-turn attribution | `scripts/longmemeval/answer_turn_attribution.py` |
| Retrieval coverage sweep | `scripts/longmemeval/retrieval_coverage_sweep.py` |
| Oracle-precision arm | `scripts/longmemeval/oracle_precision_experiment.py` |
| Extract-then-compute | `scripts/longmemeval/extract_then_compute_experiment.py` |
| Failure inventory | `scripts/longmemeval/failure_inventory.py` |
| System-prompt probe (negative) | `scripts/longmemeval/system_prompt_probe.py` |
| All per-question results | `reproducibility_benchmarks/longmemeval/*.json` |

---

## 8b. The semantic layer exceeds the raw-retrieval ceiling (headline result)

On 85 questions (60 stratified + all 27 known retrieval misses), one LLM call over consolidated
records plus provenance-linked episodic turns:

| arm | all 85 | normal 58 | known-miss 27 | calls | context | generation |
|---|---|---|---|---|---|---|
| single-shot @top_k=30 | 49 | 46 | 3 | 1 | 100,889 | 1.51 s |
| extract-then-compute | 54 | 52 | 2 | 2 | 100,889 | 2.46 s |
| oracle-precision *(prior ceiling)* | 70 | 49 | 21 | 1 | 5,654 | 0.51 s |
| **composite (this work)** | **72** | **50** | **22** | **1** | **25,557** | **0.93 s** |

Paired: **+23 vs single-shot** (McNemar +28/−5) and **+18 vs ETC** (+25/−7), both far outside the
±4.8 noise band; statistically tied with the oracle (+2, +11/−9). **The oracle uses gold `has_answer`
annotations; the composite uses none** — so §6.1's "ceiling" was the ceiling of *perfect raw-turn
selection*, not of a memory system, and a system that derives answers from accumulated records
matches it from production-available inputs. 22 of 27 questions where retrieval had provably failed
are answered, versus ETC's 2.

Efficiency: **33.14 accuracy-points per 10k chars vs ETC's 8.19 — 4×** — at half the LLM calls and 2.6×
faster generation, with consolidation paid once at write time (ingest enqueues in 163 ns).

*Limits*: the 84.7% is not comparable to n=500 figures (this sample is enriched with hard cases; only
the paired comparison is valid); on the normal 58-question slice the composite is 50 vs ETC's 52 —
inside noise but not an improvement, so **the gain is concentrated on hard retrieval and temporal
questions, not uniform**; and abstention −1.

**CONFIRMED AT n=500 (2026-08-27). `n_errors=0`. 429/500 = 85.8%**, and the reading is deliberately
two-sided:

| bar, pre-registered before the run | result |
|---|---|
| PRIMARY ≥ 425 | **PASS — 429** |
| COLLATERAL GUARD (no type > 2× its own sd below ETC) | **BREACH — abstention 22 vs floor 22.4** |
| KNOWN-MISS FLOOR ≥ 15 of 27 | **PASS — 22/27**, against ETC's 2 |

**The primary bar passes and the paired test does not confirm it.** +16 over ETC carries **p=0.137**:
the bar was set against a repeat-run sd, McNemar tests discordant pairs, and here the two instruments
disagree. Both are reported rather than whichever is more flattering. Only two per-type effects are
real: **single-session-preference +15** (15 wins, 0 losses, p<0.001) and **abstention −6** (0 wins, 6
losses, p=0.031). Excluding abstention the composite is +22 at p=0.032 — post-hoc, and labelled as
such, but it means one 30-question slice is what keeps the headline from being significant.

*The abstention breach is a design tension, not a bug.* Read in full, five of the six lost questions
**identify the unsupported premise and then override themselves with a committed answer** — `Answer: 0`,
`Total: 10 days`, a computed "4 years and 9 months". ETC abstains there because its extract stage found
nothing to compute with; the composite holds ~270 records and therefore always has *something* to
compute with. **The same density that recovers 22 of 27 known retrieval misses is what induces
over-answering on unsupported premises.** The obvious fix was measured and rejected (§6.9).

## 8c. Cross-model evaluation: what the consolidated layer does as the reader gets stronger

*All five arms are n=500, judged by the official `anscheck` prompt under `gemma4:31b-cloud`, and every
cell is regenerated from a committed result file by `scripts/longmemeval/cross_model_table.py` — no
figure in this section is transcribed.*

| arm | correct | acc | median ctx chars | median gen s |
|---|---|---|---|---|
| composite @ `gemma4:31b` *(v1 config, §8b headline)* | 429 | 85.8% | 26,056 | 0.85 |
| **composite @ `gemma4:31b`** *(config-parity control)* | **428** | 85.6% | 25,822 | 0.93 |
| oracle-precision @ `gemma4:31b` | 419 | 83.8% | 5,654 | 0.51 |
| **composite @ `glm-5.2`** | **448** | **89.6%** | 25,822 | 4.17 |
| oracle-precision @ `glm-5.2` | 457 | 91.4% | 5,654 | 0.78 |

### 8c.1 Two parity checks the table depends on — one of which failed silently

*A cross-model comparison is worth exactly the parity of everything that is not the model.* Both were
checked against the artefacts rather than assumed, and the second is the methodological contribution.

**Judge parity — passes, after verification.** `--judge-model` **defaults to `--model`** in both
harnesses and the result files recorded only the reader, so a held-fixed judge was indistinguishable
from a self-judge — and a self-judged arm is comparable to nothing. 80 hypotheses from
`oracle_glm52_n500.json` were re-judged under an explicit `gemma4:31b-cloud` judge, sampling **all 40
of its `no` verdicts** (where a self-judge would diverge) plus 40 of its `yes` verdicts: **80/80
agreement.** Both harnesses now write `judge_model` into the result metadata.

**Config parity — failed, and the flags could not show it.** Both composite arms logged identical
flags. But `n_records` was identical for all 500 while `prompt_chars` differed on **495 of 500**, by a
median **−212 chars**. The cause was `render_records()`, which had gained an **unconditional**
`group_entities()` call — no flag to log — after the v1 arm ran. **A library moved underneath matching
flags.** The composite-vs-composite contrast was therefore a reader swap *plus two render changes*.

*The fix generalises past this incident.* `--assert-prompt-parity` takes a prior result file and
aborts the run on the first prompt whose length differs, before spending an LLM call on it. Parity
becomes an **enforced invariant of the artefact** rather than a claim made about it in a log, and it
was verified in both directions before use — passing against the glm arm, aborting on question 1
against v1 with a −355 char delta. *A gate that never fires proves nothing.*

**The control was pre-registered with both branches fixed in advance** (v4-plan.md, 2026-08-28): inside
the standing ~12-question band → render changes null at scale; outside → a new finding with its own
section. Either way it is the comparison column and the committed 429 stays the §8b headline. It
landed at **428, −1, McNemar +11/−12, p=1.000** — inside the band, and the dev nulls confirmed at
n=500. **The contaminated contrast was +19; the clean one is +20**, so the confound had *deflated* the
effect rather than inflating it.

### 8c.2 The result

| contrast | net | McNemar | p |
|---|---|---|---|
| composite vs its own oracle, `gemma4` *(parity control; §8b's +10 is the v1 arm)* | +9 | +48/−39 | 0.391 |
| composite vs its own oracle, `glm-5.2` | −9 | +27/−36 | 0.314 |
| **reader swap, composite arms** (clean) | **+20** | +32/−12 | **0.0037** |
| **reader swap, oracle arms** | **+38** | +49/−11 | **7.6 × 10⁻⁷** |

**The consolidated layer reaches gold-evidence accuracy from production-available inputs on both
reader tiers, and no significant difference is detectable at n=500** — nominally ahead on the 31B
open-weights reader, nominally behind on the frontier reader, neither significant, using **no gold
annotations**. The dev split predicted this: it put the composite 8 behind its own ceiling and the
full run put it 9 behind.

⚠️ **That is a failure to reject, not a demonstration of equivalence.** With discordant pairs in the
80s the interval comfortably admits a real difference of several questions, and an equivalence test
would be required to claim parity. The sentence above is the defensible one and "establishes parity"
is not — a distinction §8c.8 makes moot in any case, since the per-stratum result below is a strictly
stronger claim than parity.

**The stronger reader helps gold evidence roughly twice as much as it helps consolidated records**
(+38 vs +20). Both effects are significant and the ratio's confidence interval is wide, so this is
reported as a **direction, not a factor**: a record is a lossy summary fixed at write time, and a
better reader cannot recover from it what was never written down, whereas raw turns retain everything
a better reader might newly extract. **Consolidation caps the headroom a stronger reader can exploit.**

### 8c.3 Where the two representations differ — and they differ by kind, not by rank

| type | `gemma4` composite v oracle | `glm-5.2` composite v oracle |
|---|---|---|
| single-session-preference | **28 v 15 (+13), p=2.4×10⁻⁴** | **26 v 18 (+8), p=0.022** |
| knowledge-update | 68 v 65 (+3), p=0.51 | 68 v 65 (+3), p=0.55 |
| multi-session + temporal | 190 v 195 (−5), p=0.59 | **217 v 229 (−12), p=0.029** |
| abstention | 25 v 28 (−3), p=0.25 | **19 v 29 (−10), p=0.0020** |

**Preference synthesis is where compression wins, and it wins on both readers** — but *the multiple is
reader-specific*: **1.87×** gold raw evidence on `gemma4`, **1.44×** on `glm-5.2`. Quoting a single
multiple across tiers is not supported.

**Multi-hop implicature is where raw evidence wins — but only once the reader is strong enough to
exploit it.** On `gemma4` the difference is −5 at p=0.59; on `glm-5.2` it is −12 at p=0.029. This is
an advantage that *emerges with reader capability*, not a standing property of raw turns, and an
earlier draft of this claim asserted the standing version.

**Knowledge-update is a tie with a nominal composite edge on both readers** (+3, p≈0.5). Two opposite
claims were made about this cell and neither survives: it is not evidence that compression is *optimal*
for state tracking, and an earlier rejection of that claim rested on a 36-question dev cell that the
full run does not reproduce.

**Abstention is the one significant composite deficit, and it worsens on the better reader** (−3 →
−10). §8b's diagnosis holds and sharpens: the density that recovers hard retrieval misses is what
induces over-answering on unsupported premises, and a more capable reader is *more* willing to commit.

### 8c.4 Latency does not transfer, and the earlier claim was wrong

| arm | median | mean | p90 |
|---|---|---|---|
| composite @ `gemma4` | 0.93 s | 1.15 s | 1.77 s |
| composite @ `glm-5.2` | **4.17 s** | **9.21 s** | **14.37 s** |

At an identical 25,822-char prompt the frontier reader is **4.5× slower at the median and 8× at the
mean**, with a heavy tail. This is a property of the reader, not of contention or context: generation
latency is flat across both run-order quintiles (3.92 / 3.89 / 4.68 / 4.20 / 3.98 s) and prompt-size
quintiles, the arm's extract leg was 0.00 s for all 500 (fully cached), and the oracle harness contains
no concurrency primitives. **The 3.9× context efficiency over extract-then-compute holds on both tiers
by construction; the sub-second generation claim does not survive the reader swap** and a draft
sentence quoting "1.65 s" was carried from a dev split rather than from these files.

### 8c.5 A harness defect the cross-model run exposed

`composite_glm52_n500.json` reports `n_errors=0` while containing **4 questions whose hypothesis is the
empty string**. An empty generation carried no error marker, so it returned on the first pass, went to
the judge, came back `no`, and was recorded with `is_error=False` — **indistinguishable from a
substantive wrong answer.** That arm's 448 is a **lower bound by up to 4**, and its −9 gap to its own
oracle is correspondingly an upper bound.

Only the long-context frontier arm produced them: **0 in either `gemma4` arm and 0 in the `glm-5.2`
oracle**, whose prompts are ~4× shorter — consistent with the visible answer being crowded out at long
context on a reasoning-style reader. Empty generations are now retried and, if they persist, marked so
they are *counted* rather than silently scored. The fix landed after the parity control launched and
deliberately was not backported into it, so both composite columns retain identical retry semantics.

**REPAIRED, AND REPORTED AS A REPAIR RATHER THAN A REPLACEMENT** (2026-08-28,
`composite_glm52_empty_repair.json`). The four questions were re-run under the fixed harness at
identical config, with `--assert-prompt-parity` enforced against the committed arm: **all four prompts
matched byte-length exactly (delta 0)**, so the retry semantics are the only thing that changed.
All four now produce substantive answers where they produced nothing — the empties were transient, not
a property of those prompts — and **1 of the 4 is correct**.

So the figure is **448 as-scored, +1 repaired**. It is written that way deliberately and 449 does not
replace 448 anywhere: the committed arm is the artefact every contrast in this section is computed
against, and silently improving it after the fact would invalidate the parity control it is paired
with. The remaining 3 are ordinary wrong answers, not missing measurements — which also **retires the
"lower bound by up to 4" caveat at k=1** rather than leaving it open.

One consequence for §8c.8: the repaired question, `gpt4_7f6b06db`, is in the 3+ stratum, so that
bucket reads **51/68 as-scored, 52/68 repaired**, and one of the 17 losses read there was a missing
measurement rather than an aggregation failure. The shape decomposition is unaffected — it was already
recorded as 16 real losses on exactly that ground.

### 8c.6 What the residual is made of, and what it implies

Crossing the composite against the oracle **under the same reader and judge** partitions the residual
into buckets that imply different work. Abstention is held out: its gold answer is a prose statement of
non-answerability that can never appear in a record, so a containment test files every abstention error
as evidence-absent and doubles the actionable bucket.

| bucket | `glm-5.2` (52 errors) | `gemma4` (72 errors) |
|---|---|---|
| benchmark/reader floor — oracle fails too | 15 (29%) | 31 (43%) |
| read-side — evidence in records, unused | 16 (31%) | 27 (38%) |
| **write-side — evidence absent from records** | **10 (19%)** | **9 (13%)** |
| abstention — calibration | 11 (21%) | 5 (7%) |

**The write-side bucket is ~10 of 500 on both readers while every other bucket moves with reader
capability.** It is also an *upper* bound twice over: the containment test counts a paraphrase as
absent, and reading all ten cases individually — rather than trusting the classifier — shows most are
counting or aggregation failures over evidence that *is* present (five items reported as four, four
properties as five), with only about three being true extraction loss. **Record extraction is close to
saturated; the residual is overwhelmingly reader-bound.**

**The two representations are complementary rather than ranked.** Under `glm-5.2` the composite wins
**27** questions the oracle loses and the oracle wins 36 the composite loses; their **union is 484
(96.8%)**, against 457 for the oracle alone. Across all five arms the union is **489 (97.8%)** — the parity control adds nothing the other four do not already cover — and only
**11 questions of 500 (2.2%) are failed by every arm**. *This headroom is real but its mechanism is
not transferable*: the oracle's advantage comes entirely from gold `has_answer` annotations that do not
exist in production. It bounds what any downstream work could reach; it is not itself a result.

### 8c.7 Localising the residual: it is aggregation, not retrieval — and §5.2 already tested the fix

*The obvious reading of §8c.6 is that the oracle's 36 wins are a retrieval gap, and the obvious fix is
a bigger episodic budget. Both were tested. The reading is wrong and the fix has already failed once.*

**Coverage confirms a retrieval gap exists.** On the 36 questions the oracle wins and the composite
loses, answer-turn coverage of the composite's own episodic slice is far below that of questions it
answers correctly. The control group is what makes this readable — a low number on the losses alone
would not distinguish signal from a too-crude containment test:

| | losses (n=30) | control, both correct (n=59) | Fisher, two-sided |
|---|---|---|---|
| mean answer-turn coverage | 51.6% | 79.9% | — |
| all answer turns present | 30% | 63% | **p = 0.0066** |
| **no answer turn present** | **20%** | **3%** | **p = 0.016** |

**The missing turns are reachable, but only deep in the ranking.** Re-selecting the *same* index at
larger budgets (one encoder pass per question, no LLM calls):

| episodic budget | losses: mean cov | losses: none present | control: mean cov |
|---|---|---|---|
| 6,000 *(shipped)* | 51.6% | 20% | 86.3% |
| 12,000 | 64.3% | 17% | 93.6% |
| 24,000 | 76.1% | 7% | 98.0% |
| 48,000 | 86.0% | 3% | 99.3% |

Coverage climbs steeply and has not plateaued at 8× the budget, while the control saturates by 24,000.
So the answer-bearing turns *are* ranked by the existing selector — there is simply no room for them at
6,000 chars.

**And this is the free rung of a ladder §5.2 already climbed to the top, where it returned a null.**
That experiment raised retrieval depth, measured coverage of the known misses rising **56.9% → 94.4%**
(the same shape as the curve above), confirmed a **17/25 cohort gain**, and then found the aggregate
effect was **+3 questions of 500 at 2.69× context and 1.8× latency** — because deep retrieval improves
findability and *degrades aggregation*.

**The loss profile predicts the same outcome here.** §5.2 decomposed the effect by how many answer
turns a question requires: deep retrieval helped 1-turn questions (+9) and hurt 2-turn (−5) and 3+-turn
(−2) questions. The composite's losses are concentrated in exactly the shapes it hurts:

| answer turns required | losses | of 500 | loss rate | §5.2 verdict |
|---|---|---|---|---|
| 1 | 8 | 182 | 4.4% | deep helps |
| 2 | 9 | 229 | 3.9% | deep hurts |
| **3+** | **13** | **68** | **19.1%** | deep hurts |

**61% of the losses need 2+ answer turns.** Raising the budget is therefore predicted to be net-neutral
at best, and this project has already paid four hours to learn that.

**The decisive number is the accuracy gradient, which localises the entire gap to one bucket:**

| answer turns | n | composite @ `glm-5.2` | oracle @ `glm-5.2` | composite @ `gemma4` |
|---|---|---|---|---|
| 1 | 182 | **92.9%** | 90.7% | 91.2% |
| 2 | 229 | **93.0%** | 91.7% | 89.5% |
| **3+** | **68** | **75.0%** | **89.7%** | **55.9%** |

**The composite beats gold-evidence selection at one and two answer turns, on both readers.** Those
two strata are **411 of 500 questions — 82.2% of the benchmark** — and the composite leads on both of
them under each reader while holding no gold annotations at all. Its entire deficit is the 68
questions requiring three or more: 51/68 against the oracle's 61/68 — a gap of 10, which is the whole
−9 net difference between the arms. *Consolidation is not losing evidence; it is
losing multi-fact aggregation over evidence it holds.*

That is also where the reader swap lands: the 3+ bucket moves **55.9% → 75.0%** on the reader change
alone, against 89.7% for the same reader given gold turns. It is the same conclusion §8c.6 reached from
the residual partition, reached independently from the evidence side: **the remaining work is
aggregation capability, not memory capacity, and retrieval volume is not a lever on it.**

*One check on the abstention comparison, since it would otherwise be an artefact.* 21 of the 30
abstention questions have no `has_answer` turn at all, so the oracle cannot be receiving gold evidence
for them. It is not: the harness falls back to ordinary top-8 retrieval (`ABSTENTION_EVENTS = 8`,
`unit="none"`), landing at a median 6,164 chars. **The oracle's abstention advantage is earned, not
free** — and the mechanism is the one §8b named: it abstains on ~6k chars of retrieved text where the
composite holds ~270 records and 26k chars, and therefore always has something to compute with.

### 8c.8 What the 3+ deficit is made of — and why write-time clustering cannot reach it

*§8c.7 localises the entire residual to 68 questions and names it "aggregation". That name is too
coarse to design against, so the 17 losses were read individually before any fix was built.*
Zero LLM calls; reproduced by `scripts/longmemeval/aggregation_failure_read.py`, which re-derives the
stratum and **aborts unless it reproduces n=68 / composite 51 / oracle 61** before reporting anything.

The obvious fix — and the one an external design note proposed — is a pre-aggregated cluster record
written at consolidation time, whose cheap deterministic form is a per-category roster header
(`ITEM(ACQUISITION/music album) — 3 items:`) computed over `EntityGroup`s at render time. It is worth
building only if the header would print the **gold** number, because a header that prints a wrong one
is worse than none: it hands the reader a computed value to be confident about, and §6.4/§6.7 already
measure what confident-and-wrong costs here.

**The losses are not enumerations.**

| shape | n | what the question actually asks |
|---|---|---|
| **windowed** | 5 | count a category *inside a date window* — "in the past two weeks", "in March", "before I made an offer" |
| **order** | 3 | rank by date — "what is the order of the three trips" |
| **predicate** | 3 | count members satisfying a property the taxonomy does not encode — "projects I *led*", "kitchen items I *replaced or fixed*" |
| **argmax** | 2 | extremum of a per-entity quantity — "which store did I spend the most at" |
| **enumerate** | 3 | plain category count — the roster's design target |
| **arithmetic** | 1 | sum quantities stated inside records |

**13 of 17 apply a filter the question supplies at read time**, and that is the finding: *you cannot
pre-aggregate over a predicate you do not know at write time.* Write-time clustering can only produce
the unfiltered total — which is the number the reader is already getting wrong, in the other direction.

**What the header would print:** gold on 1 of 17; gold only if entities are merged across buckets on 1;
a wrong number on 2; and **no defined value at all on 7** (6 where the target category fragments into
runs of differing sizes, 1 with no matching run). At most 2 conversions from a 51/68 baseline reaches
53 against a bar of 58 — **the intervention cannot clear a gate set at the oracle's own level even at
zero losses**, so it was not built. Two cases are active harm rather than nulls: on `gpt4_a56e767c` the
header prints 3, the model answered 3, gold is 4; on `9d25d4e0` it prints 4, the model answered 4, gold
is 3. In both the header would have *confirmed* the wrong answer.

**Two flattering readings were removed from the check itself, both moving the number the unfavourable
way.** Scoring convertibility as "some run in the corpus has exactly the gold size" gave **7 of 17**
and was coincidence matching — on `88432d0a`, a baking question with gold 4, it matched `TRAVEL/trip=4`,
and on `6d550036` fifteen distinct runs had size 2 in a corpus of 121 groups. Replaced by a table that
names the target category per question, so the judgement is explicit and the arithmetic over it
mechanical. Then scoring the largest matching run as "printed" turned `6d550036` into a conversion when
its target fragments into `PROJECT/=2, PROJECT/event=2, PROJECT/academic=1, PROJECT/art series=1` and
no single number would be printed at all. That moved the headline from 2 conversions to 1.

**One real signal, recorded as a footnote rather than a track.** `bf659f65` is the canonical case
`records.py` is built on ("how many albums have I purchased?", gold 3, every arm answers 2). All three
albums *are* present — split `ACQUISITION/music album` ×2 and `POSSESSION/music album` ×1. Counting
entities by **subtype across buckets** yields 3; counting the primary-bucket run yields 2. The keying
question therefore resolves against the shipped axis, and it is worth ~2 questions.

**The successor problem, named precisely.** The open direction is not "aggregation" but **dynamic
read-time filtered aggregation**: computing a count, an order or an extremum over a subset selected by
a predicate — most often a relative date window — that exists only in the question. Every mechanism
this project has tried lives on the wrong side of that boundary. Write-time consolidation cannot see
the predicate; render-time totals cannot condition on it; and reader-side instruction has now returned
five nulls at asking the model to do it unaided (§6.7). A sixth prompt is not the answer, and neither
is more context — §8c.7 shows more context is what makes this stratum worse.

*One structural observation, offered as a hypothesis and explicitly not as a result.* 8 of the 17
losses are windowed or ordering questions requiring date arithmetic over ITEM records, and
`order_records()` deliberately does **not** sort ITEMs chronologically — it groups them by
bucket/subtype, and the measured chronology intervention (§6.7) left ITEM ordering untouched by
design, because that grouping is what counting depends on. So the interaction between category
grouping and date order has never been measured, and the losses sit precisely on it. That is a
pre-registration waiting to be written, not a finding.

## 9. The write path: cross-session supersession

**The operation per-session extraction structurally cannot perform.** Extraction runs per session and
in parallel, so the worker reading a September session has never seen the August one and cannot know
that "five tops from H&M" revises "three tops from H&M". That — not a weak prompt — is why the
`supersedes` field was populated on **4 of 133,902 records** and on **0 of 72** knowledge-update
corpora. Accumulation is global; extraction is local.

A **global merge pass** sees all of a user's records at once and can. With a reconciliation step added
to the merge prompt, it produced **39 supersession markers across 55 questions**, where the entire
prior 500-question corpus held 4. The records go from two live lines to one:

> `ITEM(FINANCE/loan pre-approval): $400,000 from Wells Fargo [2023/11/30] [supersedes $350,000 from Wells Fargo]`

**Two named target conversions**, on the merged instrument: `852ce960` stopped answering the stale
$350,000 against a gold of $400,000, and `4b24c848` went from `3 + 5 = 8` to 5. A third apparent
conversion (`5831f84d`) is **discounted**: it answered correctly in all three cells of an earlier
ablation including the unmodified one, so it is a variance case and counting it would inflate two
into three.

**The mechanism works and no aggregate claim is available — those are separate statements, and the
second was established before the run.** Per §6.9, knowledge-update carries 4 errors on the whole
benchmark against a threshold of 11.6.

| dev split, n=252 | result |
|---|---|
| aggregate | 223 vs 221 baseline — **+2, p=0.754** |
| knowledge-update *(the targeted type)* | **35/36, +3** — clears that type's own 2σ of ~2.1 |
| corpus-damage guard *(losses among 221 previously correct)* | **4**, against a threshold of 8.2 |

**35/36 is not a benchmark win.** It clears the *type* threshold while contributing nothing detectable
to the aggregate — the shape of this project's standing rule that cohorts prove mechanism and only
aggregates decide worth. The corpus-damage guard is the result that licenses shipping the merge: it
rewrites 2.4% of records and does not destroy facts.

**Supersession is made traversable rather than merely applied.** It had been expressed in three places
that never met — a free-text `supersedes` marker with no id, an Atlas flag recording *no target*, and a
durable `TraceEvent` edge with five writers and **zero readers**. A derived overlay joins them, giving
edges their own tenant-namespaced session so `get_history()` — the only exposed reader, requiring a
session id and offering no filter — becomes one bounded call. Active-state projection is the default
and byte-identical to the measured renderer; timeline projection is opt-in. No kernel change, no WAL
record type, no schema change.

## 10. Production defects found while wiring the layer into a server

Not paper results. They are recorded because the contribution claims a *production* memory system, and
each was reproduced before being fixed.

| defect | reproduction |
|---|---|
| **Cross-tenant record leak** | Two stores on one file: tenant B's `all_records()` returned `['tenant A salary is $400,000', 'tenant B salary is $90,000']`. Atlas and Trace share one file and isolate by `session_id`; that convention does **not** transfer, because the record read path is a whole-file scan with no tenant argument. Fixed with per-tenant files. |
| **Category scan returned a silent subset** | Insert MEDIA, POSSESSION, MEDIA — the kernel subtree walk returns **1 of 2**. Atlas requires children to be physically adjacent and inserts append at the tail, so interleaved writes (the normal case) make later children invisible. The module docstring claiming "a kernel subtree walk, not a Python filter" was false in production and is corrected. |
| **Erased and superseded records still reached the prompt** | `supersede_node()`/`tombstone_node()` set flags that exclude a node from **beam search and nothing else**; enumeration never consulted them. With the flag confirmed set, the old path returned `['current', 'stale']`. |
| **Erasure cascade aimed at the wrong party** | The endpoint's `user_id` is the **admin executing the request**, not the data subject — the cascade would have deleted the operator's records and left the subject's intact. Now resolved per tenant. The same fix revived the cascade at all: the parameter existed and was never passed. |
| **A 26-question regression waiting in the library** | §6.8. |
| **Unbounded mmap handle leak** | Per-tenant stores mean one live handle per tenant; the session evictor popped only two dictionaries. |
| **~42 GB of duplicated model weights** | Each session built its own SentenceTransformer — at 100 sessions, ~100 copies of a ~420 MB model. Replaced with a process singleton, which also supplies the background consolidator's embedding function. |

## 11. Open items before publication

- [x] **Real reranker vs the 74.10 correct-per-1k-chars ceiling** — FIRST ANSWER (tier 2, n=85):
      a sub-turn selector reaches **5.52 correct/1k chars = 7.4% of the oracle ceiling**, which is
      1.35× ETC's 4.09. It **ties single-shot accuracy at 1/11.6 the context and 1/2.8 the generation
      latency**, but does not reach ETC's accuracy. **The binding constraint has moved from context
      size to ranking quality**: the selector reaches ~79% answer-turn coverage vs the oracle's 100%,
      and captures only 4 of the 21 hard misses the oracle recovers. Pending confirmation at n=500.
- [x] Sub-turn chunking implementation + its effect on the buried-aside misses — implemented;
      de-dilution confirmed at the mechanism level (turn-rank 32/501 → chunk-rank 0/4838 on one case,
      71/500 → 53/4827 on another), but embedding similarity does not bridge hypernym gaps
      ("smoker" ↔ "kitchen appliance") at any granularity, so chunking alone recovers only 4/27.
- [x] Retrieval-side cost of the precision layer — measured: **~10.2 s/question to build the chunk
      index** (≈2× turn-level ingest, amortised across queries) and **15.7 ms query-time selection**.
- [x] **Does query-blind write-time consolidation exceed the raw-retrieval ceiling on its cohort?**
      **YES — first evidence.** On 18 questions wrong under oracle AND ETC AND single-shot,
      query-blind consolidation converts **4/18 (records only)** and **5/18 (composite, after
      discounting one judge false-positive)** — ~4× the ~1.2 expected noise flips. `ITEM(category)`
      enumeration fixed a count the oracle got wrong with perfect evidence (gold 5, oracle 6, records
      5); `UPDATE` produced explicit supersession ("$400,000, previously $350,000"). Records are
      **22.6× smaller than the raw haystack** (21,584 vs ~487,000 chars) and complete rather than
      truncated, at ~1 min/question **one-time at write time** — ETC's extract step moved off the
      query path. Temporal converted 4/9; **multi-session only 1/8, the main open problem.**
      Remaining: does consolidation *cost* accuracy on already-passing questions (composite n=500)?
- [x] **Why did aggregation convert worst (1/8) when the requirements table predicted it best?**
      Diagnosed: not extraction recall — the evidence is present but does not *accumulate*. Free-form
      per-session categories scatter members across labels and record types (of three albums, one was
      an `ITEM`, one a `PREF`, one an `EVENT`), and independent query-blind calls sharing no
      vocabulary cannot name categories consistently. Fixed with a **closed 12-bucket taxonomy +
      global merge pass**: the two diagnosed cases now answer correctly (albums 1→3, clothing 2→3).
      **The aggregate did not move** (R 4→5, R+E 5→5 genuine; multi-session +1, temporal −1) — all
      inside the n=18 noise floor. Schema locked on mechanism grounds; aggregate claim deferred to
      the composite n=500. A negative-result reminder that mechanism fixes verified on diagnosed
      cases do not automatically show at cohort scale.
- [x] **Does consolidation cost accuracy on already-passing questions? (composite n=500)** — ANSWERED
      (§8b): no. 429/500, primary bar passed, and the one collateral breach is abstention, whose cause
      is now understood (§8b) and whose obvious fix was measured and rejected (§6.9).
- [ ] **Confirm findings hold on ≥1 additional model — now the highest-value open item, and no longer
      hygiene.** Published LongMemEval QA-accuracy results sit at 94–96% on stacks with frontier
      readers and the benchmark's official GPT-4o judge; every number here is single-model, with one
      31B model serving as **both generator and judge**, and §2 already states this makes absolute
      numbers non-comparable. The diagnosis that transfers is **reader-bound**: with the gold
      answer-bearing turns handed to it, this generator reaches only 83.8%, and the composite is
      statistically tied with that ceiling. A generator-swap probe (oracle arm, judge held fixed) is
      the direct test and is pre-registered in `v4-plan.md`.
- [x] **`event_time` is now written through ingestion** (chat request → `CognitiveLoop.chat()` →
      `ContextManager` → `trace.add_event()`), with 0 kept as the documented "unset" sentinel so live
      chat is unchanged. Before this it was always 0 and session dates fell back to insertion
      wall-clock — correct for live chat, wrong for imported history, and `Record.date` reads it.
- [x] **The consolidation dirty-queue is now recoverable** from a tenant-namespaced Trace watermark,
      so dirty state is *derivable* rather than held only in memory. `mark_dirty` keeps its O(1)
      no-I/O fast path; recovery reconciles against the watermarks at start rather than trusting the
      in-memory set, which only covered in-process failure.
- [x] **"429 may be this instrument's ceiling" is DISPROVED, and the disproof is §8c.** It was the
      ceiling of *this reader*, not of the instrument. Swapping the reader alone moves the composite
      to **448 (+20, p=0.0037)** and the gold-evidence ceiling to **457 (+38, p=7.6×10⁻⁷)**, and the
      gains land precisely on the multi-session and temporal-reasoning errors that six interventions
      could not move — they were **reader-capability errors**. What survives from the original entry
      is its narrower claim: those errors are not reachable *by record hygiene or by instruction*.
      §8c.6 sharpens it — the write-side residual is ~10 of 500 on both readers and is an upper bound
      twice over, so extraction is near-saturated and the remainder is reader-bound.
- [x] **"Aggregation" is now named precisely, and one proposed fix is closed out before being built.**
      §8c.8 reads all 17 losses in the 3+ stratum: **13 of 17 apply a filter the question supplies at
      read time** (5 windowed, 3 ordering, 3 ad-hoc predicates, 2 argmax) and only 3 are the plain
      category enumerations a pre-aggregated cluster record could serve. A deterministic roster header
      would print gold on 1 of 17, a wrong number on 2, and nothing defined on 7 — reaching at most 53
      from a 51/68 baseline, so it cannot clear a bar set at the oracle's own 61 even at zero losses.
      **Not built.** The successor problem is **dynamic read-time filtered aggregation**, and it is
      genuinely open: write-time consolidation cannot see the predicate, render-time totals cannot
      condition on it, and reader-side instruction has now returned five nulls.
- [ ] **Chronological ordering of ITEM records has never been measured**, and 8 of the 17 losses are
      windowed or ordering questions needing date arithmetic over exactly those records. §6.7's
      chronology null covered EVENT and UPDATE and left ITEM ordering untouched by design, since
      bucket grouping is what counting depends on. The interaction is unmeasured and the losses sit on
      it. This is a hypothesis awaiting its own one-shot pre-registration, not a finding — and it
      belongs to the same reader-side family that has returned five nulls, so it inherits that
      protocol.
- [x] **Point-in-time (`as_of`) traversal ships as a product capability with no accuracy claim.**
      Derived from provenance rather than stored: there is no per-record timestamp and there cannot be
      one (`NodeHeader` is full at 64 bytes; `Record.decode()` folds an eighth field into `text`), so a
      record's validity is the latest `event_time` among the turns it came from — valid time, not
      transaction time, which is what "as of March" means. Both halves of the assembly are cut, since
      a record block filtered to March beside excerpts quoting September is a leak that reads as a
      historical view. LongMemEval asks every question at T_now, so this cannot move a benchmark number
      and is not gated on one. Fidelity bound: `run_merge()` *removes* a stale record rather than
      superseding it, so `as_of` reconstructs what supersession retired, not what the merge dropped.
- [ ] (superseded framing) 28 questions are wrong under oracle AND ETC AND single-shot — perfect evidence,
      wrong answer — and they are dominated by counting/aggregation ("how many albums": gold 3,
      oracle 2). 83.8% bounds *perfect raw-turn selection*, not a memory system that derives answers.
      Risk to test: ETC's extraction was query-conditioned; write-time extraction is query-blind.
- [ ] LongMemEval v2 is a *different* benchmark (static/dynamic-environment, procedure,
      errors-gotchas; 451 questions), and its recorded oracle is 37.5% at 95% haystack coverage —
      i.e. generator-bound, not retrieval-bound. v1 retrieval tuning does not transfer; v2's
      categories are close to a specification for entity-state + procedure memory.
- [~] Confirm the tier-2 selector result at n=500 — PARKED as superseded by the consolidation pivot;
      the expected result is already known at n=85 and would not change what to build next.
- [ ] Close the ranking-quality gap (hybrid lexical+semantic scoring, or query expansion) — now the
      highest-value open problem, since context size is no longer binding.
- [ ] Confirm findings hold on ≥1 additional model — every number here is single-model, and the noise
      floor is a property of this cloud-served deployment.
- [ ] Consider reporting correct-per-1k-chars as a standard axis for memory-system papers; current
      benchmarks report accuracy alone and are therefore blind to the tradeoff in §5.2.
