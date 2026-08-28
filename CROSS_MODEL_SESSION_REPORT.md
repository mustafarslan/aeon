# Cross-Model Evaluation — Complete Session Record

**Scope.** Everything from the launch of the full n=500 composite arm under `glm-5.2:cloud` through
the diagnostics that followed. Benchmark: LongMemEval-S, 500 questions. **Judge held fixed at
`gemma4:31b-cloud` in every cell**, so the reader is the only variable between model columns.

Every number here is regenerated from a committed result file by
`scripts/longmemeval/cross_model_table.py`. Nothing is transcribed from memory. Where a figure
contradicts something stated earlier in the session, the contradiction is recorded rather than
quietly replaced.

---

## 1. What was run, in order

| # | run | file | outcome |
|---|---|---|---|
| 1 | composite @ `glm-5.2`, n=500 | `composite_glm52_n500.json` | **448 / 500 = 89.6%** |
| 2 | judge-parity verification, 80 re-judged | *(scratch)* | **80/80 agreement** |
| 3 | prompt-parity gate, 6-question smoke, both directions | *(scratch)* | passes vs glm arm, aborts vs v1 |
| 4 | composite @ `gemma4:31b`, config-parity control, n=500 | `composite_gemma4_parity_n500.json` | **428 / 500 = 85.6%** |
| 5 | residual partition, both readers | `residual_error_profile.py` | see §6 |
| 6 | episodic coverage diagnostic + control | `episodic_coverage_gap.py` | see §7 |
| 7 | episodic budget sweep, 4 budgets | `episodic_budget_sweep.py` | see §7 |

Runs 2–4 were not planned at the start. They exist because a parity check failed (§3).

---

## 2. The headline result

| arm | correct | acc | median ctx chars | median gen s |
|---|---|---|---|---|
| composite @ `gemma4:31b` *(v1 config, prior headline)* | 429 | 85.8% | 26,056 | 0.85 |
| composite @ `gemma4:31b` *(config-parity control)* | **428** | 85.6% | 25,822 | 0.93 |
| oracle-precision @ `gemma4:31b` | 419 | 83.8% | 5,654 | 0.51 |
| **composite @ `glm-5.2`** | **448** | **89.6%** | 25,822 | 4.17 |
| oracle-precision @ `glm-5.2` | 457 | 91.4% | 5,654 | 0.78 |

*The oracle is handed the gold `has_answer` annotated turns. The composite uses no gold annotations —
only inputs that exist in production.*

| paired contrast | net | McNemar | p |
|---|---|---|---|
| composite vs its own oracle, `gemma4` *(parity arm)* | +9 | +48/−39 | 0.391 |
| composite vs its own oracle, `glm-5.2` | −9 | +27/−36 | 0.314 |
| **reader swap, composite arms** *(clean)* | **+20** | +32/−12 | **0.0037** |
| **reader swap, oracle arms** | **+38** | +49/−11 | **7.6 × 10⁻⁷** |
| render changes at n=500 (parity arm vs v1) | −1 | +11/−12 | 1.000 |

**Reading.** The consolidated layer reaches gold-oracle accuracy without gold annotations on both
reader tiers, with no significant difference detectable at n=500 in either direction.

⚠️ **This is a failure to reject, not a positive finding.** With discordant pairs in the 80s the
interval comfortably admits a real difference of several questions. "Firmly establishes parity"
overstates it; an equivalence test would be needed for that claim. The defensible sentence is:
*the composite reaches oracle accuracy from production-available inputs, and no significant difference
is detectable at n=500.*

**Secondary finding.** The stronger reader buys the oracle **+38** but the composite only **+20**. A
record is a lossy summary fixed at write time; raw turns retain everything a better reader might newly
extract. **Consolidation caps the headroom a stronger reader can exploit.** Reported as a direction,
not a factor — the ratio's interval is too wide to quote.

---

## 3. Two parity checks — one failed silently

*A cross-model table is worth exactly the parity of everything that is not the model.*

### 3.1 Judge parity — passed, but only after verification

`--judge-model` **defaults to `--model`** in both harnesses, and the result files recorded only the
reader. So "judge held fixed" was an intent in the log, not a fact recoverable from the artefact — and
a self-judged arm is comparable to nothing.

80 hypotheses from `oracle_glm52_n500.json` were re-judged under an explicit `gemma4:31b-cloud` judge,
sampling **all 40 of its `no` verdicts** (where a self-judge would diverge) plus 40 `yes` verdicts:
**80/80 agreement.** The arm was judged by `gemma4`.

*Fixed permanently:* both harnesses now write `judge_model` into result metadata.

### 3.2 Config parity — FAILED, and the flags could not show it

Both composite arms logged identical flags. But:

- `n_records` **identical for all 500** (same record cache)
- `prompt_chars` **differed on 495 of 500**, median **−212 chars**, always in the same direction

Cause: `render_records()` had gained an **unconditional** `group_entities()` call — *no flag to log* —
after the v1 arm ran. **A library moved underneath matching flags.** The composite-vs-composite
contrast was therefore a reader swap *plus two render changes*, not a reader swap.

*Fixed generally:* `--assert-prompt-parity` takes a prior result file and aborts the run on the first
prompt whose length differs, **before spending an LLM call on it**. Parity becomes an enforced
invariant of the artefact rather than a claim about it. Verified in both directions before use —
passes against the glm arm, aborts on question 1 against v1 with a −355 char delta. *A gate that never
fires proves nothing.*

### 3.3 The control arm, pre-registered with both branches fixed in advance

Committed to `v4-plan.md` **before** the run: inside the standing ~12-question band → render changes
null at scale; outside → a new finding with its own section. Either way it is the comparison column,
and the committed 429 stays the prior headline. *That is what separates a control from a re-roll.*

It landed at **428 (−1, p=1.000)** — inside the band. Both render changes confirmed null at n=500,
matching their dev readings. **The contaminated contrast was +19; the clean one is +20** — the confound
had *deflated* the effect, not inflated it.

---

## 4. Claims corrected against these files

Four statements were drafted on the dev split and do not survive the full run.

| claim as drafted | status | corrected |
|---|---|---|
| "generation latency 0.96 s and 1.65 s" | ✗ wrong | **0.93 s and 4.17 s** (median); mean 9.21 s, p90 14.4 s |
| "records deliver ~1.9× gold evidence on preference" | ✗ tier-specific | **1.87× on `gemma4`, 1.44× on `glm-5.2`** |
| "raw rehydration *remains essential* for multi-hop" | ✗ overstated | the advantage **emerges with reader capability**: −5 (p=0.59) → −12 (p=0.029) |
| *(my own)* "compression optimal for state tracking is FALSE" | ✗ dev artefact | knowledge-update is **+3 on both readers, p≈0.5** — a tie; neither the claim nor my rejection survives |

Also to keep attached to their comparators:

- **3.9× context efficiency is against extract-then-compute, not the oracle.** Against the oracle the
  composite uses **4.6× more** context (25,822 vs 5,654). Detaching that comparator is the exact
  attribution error found in an external evaluation of this work.
- **429 and 448 come from different `compose()` configs.** For cross-model claims the `gemma4` figure
  is the parity control's **428**; 429 is the v1 arm. One question apart, so nothing rests on it — but
  they should not appear in one sentence without saying which arm each is.

### Per-type detail

| type | `gemma4` composite v oracle | `glm-5.2` composite v oracle |
|---|---|---|
| single-session-preference | **28 v 15 (+13), p=2.4×10⁻⁴** | **26 v 18 (+8), p=0.022** |
| knowledge-update | 68 v 65 (+3), p=0.51 | 68 v 65 (+3), p=0.55 |
| multi-session + temporal | 190 v 195 (−5), p=0.59 | **217 v 229 (−12), p=0.029** |
| abstention | 25 v 28 (−3), p=0.25 | **19 v 29 (−10), p=0.0020** |

### Latency does not transfer

| arm | median | mean | p90 |
|---|---|---|---|
| composite @ `gemma4` | 0.93 s | 1.15 s | 1.77 s |
| composite @ `glm-5.2` | **4.17 s** | **9.21 s** | **14.37 s** |

At an identical 25,822-char prompt the frontier reader is 4.5× slower at the median, 8× at the mean.
**Verified as a reader property, not contention:** latency is flat across run-order quintiles
(3.92 / 3.89 / 4.68 / 4.20 / 3.98 s) and across prompt-size quintiles; the arm's extract leg was 0.00 s
for all 500 (fully cached); the oracle harness contains no concurrency primitives; and `--workers`
gates only the extraction pass, with generation a single serial call in every arm.

---

## 5. A harness defect the run exposed

`composite_glm52_n500.json` reports `n_errors=0` while containing **4 questions whose hypothesis is the
empty string**. An empty generation carried no error marker, so it returned on the first pass, went to
the judge, came back `no`, and was recorded with `is_error=False` — **indistinguishable from a
substantive wrong answer**.

- **448 is a lower bound by up to 4**, and the −9 gap to its own oracle is correspondingly an upper bound.
- Only the long-context frontier arm produced them: **0 in either `gemma4` arm, 0 in the `glm-5.2`
  oracle** (prompts ~4× shorter) — consistent with the visible answer being crowded out at long context
  on a reasoning-style reader.
- Fixed: empty generations are now retried and, if they persist, returned as a `[System Error:]` marker
  so they are **counted** rather than silently scored.
- The fix landed *after* the parity control launched and was deliberately not backported, so both
  composite columns retain identical retry semantics.

*If these 4 are ever repaired with the fixed harness, it must be reported as "448 as-scored, +k
repaired" — never as a replacement number.*

---

## 6. What the residual is made of

Crossing the composite against the oracle **under the same reader and judge** partitions the residual
into buckets implying different work. Abstention is held out: its gold answer is a prose statement of
non-answerability that can never appear in a record, so a containment test files every abstention error
as evidence-absent — which had **doubled** the apparent actionable bucket before it was caught.

| bucket | `glm-5.2` (52 errors) | `gemma4` (72 errors) |
|---|---|---|
| benchmark/reader floor — oracle fails too | 15 (29%) | 31 (43%) |
| read-side — evidence in records, unused | 16 (31%) | 27 (38%) |
| **write-side — evidence absent from records** | **10 (19%)** | **9 (13%)** |
| abstention — calibration | 11 (21%) | 5 (7%) |

**The write-side bucket is ~10 of 500 on both readers while every other bucket scales with reader
capability.** It is an upper bound twice over: the containment test counts paraphrase as absent, and
reading all ten cases individually — rather than trusting the classifier — shows most are counting or
aggregation failures over evidence that *is* present (five items reported as four, four properties as
five). Only about three are true extraction loss. **Record extraction is close to saturated.**

**Complementarity.** Under `glm-5.2` the composite wins **27** questions the oracle loses; the oracle
wins 36 the composite loses. Their **union is 484 (96.8%)** against 457 for the oracle alone; across all
five arms the union is **489 (97.8%)** and only **11 of 500 (2.2%)** are failed by every arm. *This
bounds what downstream work could reach but prescribes nothing* — the oracle's advantage comes entirely
from gold annotations that do not exist in production.

---

## 7. Localising the residual — and why episodic selection is not the answer

### 7.1 A retrieval gap exists and is significant

On the 36 questions the oracle wins and the composite loses, answer-turn coverage of the composite's
own episodic slice is far below that of questions it answers correctly. **The control group is what
makes this readable** — a low number on the losses alone could not be distinguished from a too-crude
containment test.

| | losses (n=30) | control, both correct (n=59) | Fisher, two-sided |
|---|---|---|---|
| mean answer-turn coverage | 51.6% | 79.9% | — |
| all answer turns present | 30% | 63% | **p = 0.0066** |
| **no answer turn present** | **20%** | **3%** | **p = 0.016** |

### 7.2 The missing turns are reachable — it is a room problem, not a ranking failure

Re-selecting the **same** index at larger budgets (one encoder pass per question, zero LLM calls):

| episodic budget | losses: mean cov | losses: none present | control: mean cov |
|---|---|---|---|
| 6,000 *(shipped)* | 51.6% | 20% | 86.3% |
| 12,000 | 64.3% | 17% | 93.6% |
| 24,000 | 76.1% | 7% | 98.0% |
| 48,000 | 86.0% | 3% | 99.3% |

Coverage climbs steeply and has **not plateaued at 8× budget**, while the control saturates by 24,000.
The existing selector *does* rank the answer-bearing turns; there is simply no room at 6,000 chars.

### 7.3 …but this is the free rung of a ladder already climbed to the top

§5.2 of the findings ran exactly this experiment: raised retrieval depth, measured coverage of the
known misses rising **56.9% → 94.4%** (the same shape as the curve above), confirmed a **17/25 cohort
gain** that reproduced exactly, and then measured the aggregate at **+3 questions of 500, at 2.69×
context and 1.8× latency**. Deep retrieval improves findability and **degrades aggregation**.

The loss profile predicts the same ending here. §5.2 decomposed the effect by evidence count: deep
retrieval helped 1-turn questions (+9), hurt 2-turn (−5) and 3+-turn (−2).

| answer turns required | losses | of 500 | loss rate | §5.2 verdict |
|---|---|---|---|---|
| 1 | 8 | 182 | 4.4% | deep helps |
| 2 | 9 | 229 | 3.9% | deep hurts |
| **3+** | **13** | **68** | **19.1%** | deep hurts |

**61% of the losses need 2+ answer turns** — the shape deep retrieval makes worse.

### 7.4 The decisive number

| answer turns | n | composite @ `glm-5.2` | oracle @ `glm-5.2` | composite @ `gemma4` |
|---|---|---|---|---|
| 1 | 182 | **92.9%** | 90.7% | 91.2% |
| 2 | 229 | **93.0%** | 91.7% | 89.5% |
| **3+** | **68** | **75.0%** | **89.7%** | **55.9%** |

**The composite beats gold-evidence selection at one and two answer turns, on both readers.** Its
entire deficit is the 68 questions requiring three or more: **51/68 against the oracle's 61/68 — a gap
of 10, which is the whole −9 net difference between the arms.**

> **Consolidation is not losing evidence. It is losing multi-fact aggregation over evidence it holds.**
> More context cannot fix that — it is the thing more context makes worse.

The reader swap moves the 3+ bucket **55.9% → 75.0%**, against 89.7% for the same reader given gold
turns. So roughly **10 questions of headroom in that bucket are not reader-bound.**

### 7.5 One artefact ruled out

21 of the 30 abstention questions have **no `has_answer` turn at all**, so the oracle could not be
receiving gold evidence for them. Checked: the harness falls back to ordinary top-8 retrieval
(`ABSTENTION_EVENTS = 8`, `unit="none"`), landing at a median 6,164 chars. **The oracle's abstention
advantage is earned, not free.** The mechanism is density — it abstains on ~6k chars where the
composite holds ~270 records and 26k chars, and therefore always has something to compute with.

---

## 8. Corrections made to my own statements during this session

Recorded because the reasoning behind a number matters as much as the number.

1. **Mid-run interim reading was wrong.** At 251/500 I reported the dev −8 ceiling gap "is NOT
   reproducing at full scale." The final is **−9** — it reproduced almost exactly. A partial prefix is
   not evidence and will not be reported as one.
2. **The +19 I showed carried a hidden confound** (§3.2). The clean figure is **+20**.
3. **My residual classifier misfiled all 11 abstention errors** as evidence-absent, doubling the
   apparent actionable write-side bucket. Held out now.
4. **My write-side bucket was inflated even after that fix** — reading all ten cases showed most are
   counting failures over present evidence, not evidence loss.
5. **My verdict on "compression optimal for state tracking"** rested on a 36-question dev cell the full
   run does not reproduce.

---

## 9. Where this leaves the decision

**Episodic selection is ruled out** — not on judgement, but on an experiment this project already ran
to completion (§7.3) plus a loss profile that predicts the same null (§7.4).

| option | measured ceiling | assessment |
|---|---|---|
| **Publish on the current result** | — | Write path saturated (~3 true losses in 500); residual localised and explained |
| **Aggregation (3+ evidence questions)** | ~10 questions, not reader-bound | The only substantive open direction, and a genuinely new one |
| Abstention calibration | +11 max | Smallest bucket; §6.9 already measured and rejected the naive fix — needs a mechanism, not a prompt |
| Reader upgrade | +20, measured | 4.5× median / 8× mean latency. This is the honest answer to the 94–96% leaderboard gap |
| Repair the 4 empty generations | ≤ +4 | Declared measurement repair only; reported as "448 as-scored, +k repaired" |
| ~~Episodic budget / selection~~ | ~~—~~ | **Ruled out by §7.3–7.4** |

**Recommendation: publish on the current result, and scope aggregation as the named successor
problem.** §8c is a stronger claim than the prior §8b: the layer *beats* gold-evidence selection on
82% of the benchmark by evidence count and loses only on multi-fact aggregation. That is sharper and
more defensible than "statistical parity."

On the leaderboard question — those systems run frontier readers. The gold-evidence ceiling here at
`glm-5.2` is **91.4%**, so the gap is reader capability, which §8c now demonstrates directly rather
than infers.

---

## 10. Reproducing any number in this document

```bash
python3 scripts/longmemeval/cross_model_table.py            # §2 tables and contrasts
python3 scripts/longmemeval/residual_error_profile.py       # §6, glm-5.2
python3 scripts/longmemeval/residual_error_profile.py \
    --composite composite_gemma4_parity_n500.json \
    --oracle oracle_precision_n500_results.json             # §6, gemma4
.venv/bin/python scripts/longmemeval/episodic_coverage_gap.py    # §7.1
.venv/bin/python scripts/longmemeval/episodic_budget_sweep.py    # §7.2
```

Commits: `0e6329b` (pre-registration) · `5d4bec7` (parity gate) · `b187ea5` (empty-generation fix)
· `eb13ffb` (control arm) · `e4d01b6` (§8c) · `374fa09` (consistency sweep) · `02a66b3` (§8c.7)
