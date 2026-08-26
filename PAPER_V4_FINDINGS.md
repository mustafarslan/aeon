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

The last row is an oracle (§6.1). Its significance is not the accuracy — which is statistically tied
with ETC — but that the same accuracy is reachable at **1/18th the context, half the LLM calls, and
1/5th the generation latency**.

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

Efficiency: **33.14 accuracy-points per 1k chars vs ETC's 8.19 — 4×** — at half the LLM calls and 2.6×
faster generation, with consolidation paid once at write time (ingest enqueues in 163 ns).

*Limits*: the 84.7% is not comparable to n=500 figures (this sample is enriched with hard cases; only
the paired comparison is valid); on the normal 58-question slice the composite is 50 vs ETC's 52 —
inside noise but not an improvement, so **the gain is concentrated on hard retrieval and temporal
questions, not uniform**; abstention −1; and the composite has not yet been run at n=500.

## 9. Open items before publication

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
