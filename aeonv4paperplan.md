# Aeon v4 Paper: Multi-Phase Improvement Plan

**Deliverable:** this document is to be written to `aeonv4paperplan.md` at the repository root,
for execution by a separate agent. It must be self-contained: the executing agent will not have
this session's context.

---

## Part I. Context and Session Record

### What the paper is now

`paper/paper_final_v4.tex` (650 lines) plus `paper/v4_semantic_layer.tex` (306 lines, included
before the Conclusion). It compiles to 17 pages, two-column, 0 undefined references.

v1 through v3 were a kernel-performance paper: Atlas, INT8 quantization, WAL, Blob Arena, SLB.
v4 adds a second half covering the semantic layer and a memory-quality evaluation on
LongMemEval-S. The two halves currently argue different theses and read as two papers bound
together. The target quality bar is IEEE / JMLR / JAIR. This is a quality bar, not a venue
retarget; no template change is assumed.

### What was completed in the session preceding this plan

Research and correctness work, all committed. The executing agent should treat these results as
settled inputs, not as open questions.

| commit | outcome |
|---|---|
| `27656a9` | Step 0 kill-check. 13 of 17 aggregation losses apply a read-time filter, so write-time clustering cannot reach them. A proposed fix was retired by analysis before implementation. |
| `44558ea` | Track A: `as_of` point-in-time traversal, derived from provenance. 17 tests. No accuracy claim. |
| `86402ad` | Paper section 8c.8 added; two overstated claims corrected. |
| `972c968` | Four empty generations repaired. Reported as "448 as-scored, +1 repaired". |
| `f1257e4` | Track D: a second frontier reader moves the result by 1 question. The reader lever saturates. |
| `0399564`, `abc1560` | Competitor claims verified against publishing sources, quotes checked against raw pages, one misattribution corrected in our favour and recorded against us. |
| `8701fc7` | Judge panel across three judges. Two prior expectations failed, one against us. |
| `7b7fd9b` | Chronological ITEM ordering: sixth reader-side null, with a control arm. |
| `5b8e73b` | Harness: total wall-clock timeout, retry logging. |
| `91b44b5` | Kernel: `get_children()` returned a silent subset under interleaved writes. |
| `1b12e2b` | Four defects: reassigned node ids after compaction, an armed prompt regression, an unenforced security parameter, a silent Win32 no-op. |
| `7607428` | `claims_inventory.py`: regenerates every evaluation figure from committed result files. |
| `dffde79` | v4 manuscript created with six new sections. |
| `5cdadab` | Benchmark sweep repaired (`BUILD_TESTING`), `master_metrics.txt` regenerated, kernel numbers updated. |
| `a55c6fb` | Prefetch widening measured across three variants and rejected. |
| `7faf3eb` | `BM_Navigate_SLBWarm` added; SLB contribution to navigate measured at 2.4x; a number wrongly deleted in `5cdadab` restored with correct labelling. |

Two facts from that work bear directly on this plan:

1. `reproducibility_benchmarks/run_all_benchmarks.sh` had been unable to build any benchmark
   (targets sit inside `if(BUILD_TESTING)`, which the script never set). Every kernel number in
   the paper was therefore unverifiable until it was fixed and the sweep re-run. Regenerated
   figures now live in `reproducibility_benchmarks/master_metrics.txt`.
2. Sections covering Scalability, SLB performance, Beam Search, Trace GC and Zero-Copy were
   **never audited** against the regenerated metrics. Beam-search P99 figures in the paper
   (47.2 and 99.1 microseconds) are stale. Phase 1 must close this.

   **Do not transcribe a target value from this plan.** `master_metrics.txt` has been
   regenerated twice since the paper was last touched, and these figures move 2 to 4 percent
   between runs on the same machine and the same code: beam-search P99 read 43.0 and 88.8 in
   one sweep and 42.4 and 92.9 in the next. Read the committed file at execution time.

   That variance is itself a finding the paper should carry. It currently reports kernel
   latencies as bare point values with no run-to-run spread, which a reviewer at this bar
   will question. Google Benchmark already emits `_cv` alongside every median; Phase 4 should
   add a sentence to Experimental Methodology stating the observed variance and the number of
   repetitions, rather than implying a precision the instrument does not have.

   A fourth benchmark now exists, `bench_scope_filter`, measuring scope-filtered navigate.
   It is registered in the sweep and reports that filtering costs 3.9x on a 100K-node tree,
   almost all of it from an unconditional beam escalation. If the paper claims multi-tenant
   retrieval, that cost belongs in it.

### User decisions taken

- **Disclosure scope: strip process narrative only.** Remove drafting history and
  self-corrections. Negative results stay, restated impersonally as findings with data. The
  competitor comparison stays.
- **Dash policy: zero em-dashes in rendered output.**
- Tone: academic throughout.

---

## Part II. Non-Negotiable Invariants

These survive every rewrite. A professional-tone pass is exactly the operation that smooths
careful hedges into overclaims, so each is stated with its permitted phrasing.

| # | Invariant | Required form | Never |
|---|---|---|---|
| 1 | Composite vs its own oracle | "reaches gold-evidence accuracy from production-available inputs; no significant difference is detectable at n=500" | "parity", "beats the oracle" |
| 2 | Lead claim | per-stratum: leads gold-evidence selection on 411 of 500 (82.2%), both readers | a pooled single number |
| 3 | gemma4 arms | 428 for cross-model claims; 429 is the v1-config arm; always labelled | the two in one sentence unlabelled |
| 4 | Empty generations | "448 as-scored, +1 repaired" | 449 anywhere |
| 5 | Headline accuracy | 89.6% micro over 500, abstention included | adopting 90.9 / 92.1 / 91.3, which are labelled comparability figures only |
| 6 | Competitors | "unverifiable without an external check" | "inflated" |
| 7 | Context efficiency | 3.9x is against extract-then-compute; against the oracle the composite uses 4.6x more | 3.9x unqualified |
| 8 | Navigate latency | two paths, labelled: SLB-warm 3.10 us, cold traversal 7.30 us | either alone as "traversal" |
| 9 | Reader saturation | scoped: one additional reader, n=252 dev | "saturation" as a general property |
| 10 | Judge panel | scoped: one arm, containment-style judge prompt | "self-judging is safe" |

**Erratum floor.** v1 through v3 circulated "3.09 us tree traversal (3.4x over FP32)". v4
relabels that as SLB-warm and adds the cold figure. A brief erratum note stays in every version
of this paper. Dropping it would silently redefine a published number. The same applies to the
428/429 labelling and to invariant 4.

---

## Part III. Per-Section Review

### Front matter

| Section | Defects | Action |
|---|---|---|
| **Title** | Describes the kernel half only; the paper is now half evaluation. | Propose two or three alternatives at execution time and let the author choose. Do not change unilaterally. |
| **Abstract** | 409 words against a 150 to 250 norm. One paragraph. Says "three advances" while the paper lists eight contributions. Carries both theses without subordinating one. | Rewrite to 200 to 250 words, single paragraph, leading with the unifying thesis (Part IV). State the kernel result in one sentence, the evaluation result in two, the erratum in one clause. |
| **Keywords** | Absent. | Add: long-term memory, retrieval-augmented generation, vector index, quantization, benchmark evaluation. |

### Inherited sections

| Section | Defects | Action |
|---|---|---|
| **1. Introduction** | Marketing register: "Cognitive Operating System", "Memory Palace", "Vector Haze", "Ring 0", "paradigm shift" (9 instances). Contributions list has 8 items after v4 additions. No formal problem statement. | Retain the OS analogy as an explanatory device, introduced once and not repeated as branding. Consolidate contributions to 4 or 5. Add a short problem statement with notation for the memory, the query, and the two retrieval paths. |
| **2. Related Work** | Six one-paragraph subsections. **No coverage of the agent-memory systems the paper now compares against.** 18 references total, thin for this bar. | Add a subsection on long-horizon agent memory and its evaluation, covering LongMemEval and the systems in the comparison section. Grow references to roughly 40; verified sources are already listed in `spatiotemporal.md` section 2. Merge Crash Recovery and EBR into one systems-background subsection. |
| **3. System Architecture** | Sound and detailed. Register issues only. Some subsections read as implementation notes rather than design rationale. | Light edit. Preserve technical content. Convert "Ring 0" and similar to neutral terms on first and every use. |
| **4. Semantic Lookaside Buffer** | Under-credited relative to its measured importance. The SLB is worth 2.4x on the INT8 read path, the largest read-path effect in the system, and this is stated only in the evaluation. | Add one forward reference from this section to the measured contribution. This is the paper's strongest systems result and currently reads as a design description. |
| **5. Experimental Methodology** | No statistical methodology. No multiple-comparisons policy despite dozens of paired tests. Hardware described; software environment and seeds not fully stated. | Add a statistics paragraph: McNemar exact two-sided for paired arms, Fisher for independent proportions, pre-registered primary endpoints with everything else reported descriptively. This converts an anticipated reviewer objection into stated method. Add a reproducibility paragraph naming `run_all_benchmarks.sh`, `master_metrics.txt`, `claims_inventory.py`. |
| **6. Evaluation (kernel)** | **Subsections on Scalability, SLB performance, Beam Search, Trace GC, Zero-Copy were never audited against regenerated metrics. Beam-search P99 47.2 and 99.1 us are stale; current values are 43.0 and 88.8.** Ten subsections is fragmented. | Audit every number against `master_metrics.txt`. Consolidate to about six subsections. Keep the traversal subsection's two-path table and erratum note as written. |
| **13. Conclusion** | Restates contributions rather than synthesising. Future Work is three items of uneven weight. | Rewrite to argue the unifying thesis. Lead Future Work with dynamic read-time filtered aggregation. |

### New sections (v4_semantic_layer.tex)

| Section | Defects | Action |
|---|---|---|
| **The Semantic Layer** | Strongest new section. Motivation via the oracle result is well-framed. Some em-dashes. | Dash and register pass only. |
| **Memory-Quality Evaluation** | Lead claim is correct and well-hedged. **No figure**: the per-stratum result is the paper's headline and appears only as a table. | Add a figure for accuracy by evidence-turn stratum. Load the `dataviz` skill before writing any chart code. |
| **Cross-Model Evaluation** | Sound. Saturation correctly scoped. Judge panel subsection contains "two prior expectations failed", which is process narrative. | Restate as findings: report the panel result and its scope without the expectation framing. |
| **Evaluation Discipline and Negative Results** | The most valuable and most at-risk section. Contains the largest concentration of process narrative and bold editorial phrases. | Keep every result. Convert to impersonal register throughout. Retain the six-null table. Remove "a gate that never fires proves nothing" style aphorisms. |
| **A Fix Retired Before It Was Built** | Contains "two flattering readings removed from our own analysis" (line 240). Pure drafting history. | Delete that paragraph. State the final classification method in Methodology instead. The result stands on the shape table. |
| **Comparison to Published Systems** | Contains "an earlier draft of this work made exactly that error" (line 273). Quoted vendor methodology is retained per the user's decision. | Delete the drafting-history clause; keep the substantive point that retrieval recall and QA accuracy are not comparable, attributed to the source that states it. Keep the matched-metric table. |
| **Localising the Residual** | Sound. Successor problem well named. | Dash and register pass only. |

### Whole-paper issues

| Issue | Measurement | Action |
|---|---|---|
| **Em-dashes** | 44 in rendered prose (14 main, 30 include). The 414 source count is LaTeX comment separators, which do not render. | Reduce to zero by sentence-level rewrite. Do not use `sed`; it will mangle grammar. |
| **En-dashes** | 2, both numeric ranges (94--96, 1--3). | Keep. These are correct typography. Flagged for author veto. |
| **Literal Unicode dashes** | 3 U+2014 characters in the main file. | Remove. Verification must run on rendered output, not source. |
| **Bold density** | 183 `\textbf` spans. Mid-paragraph bolding of editorial phrases. | Reduce to roughly 40. Retain bold for table headers and defined terms only. |
| **Register** | "honest", "the honest number", aphoristic asides. | Remove throughout. |
| **Two theses** | Kernel half and evaluation half argue separately. | Resolve via Part IV. |
| **Figures** | 8 in the kernel half, 0 in the evaluation half. | Add 2 to 3. `paper/generate_figures.py` exists. |

---

## Part IV. The Unifying Thesis

The paper needs one argument, and the material already supports a strong one:

> Retrieval latency was never the binding constraint on long-horizon memory quality. An oracle
> handed the gold answer-bearing turns, with nothing to retrieve, still fails 81 of 500
> benchmark questions. A memory system must therefore be evaluated on two axes that are not
> reducible to each other: how fast it retrieves, and what it retains. This paper contributes a
> kernel that answers the first and a semantic layer that answers the second, and shows that the
> second is where the remaining difficulty lies.

This sentence is already latent in the current Conclusion. Promoting it to the Introduction and
Abstract converts two stapled papers into one argument, and it is the single highest-value
structural change in this plan.

---

## Part V. Phases

Each phase ends with the Part VII verification suite. Do not batch phases before verifying.

**Phase 0. Baseline.** Compile, record page count, run the full verification suite to capture
starting values. Commit nothing yet.

**Phase 1. Number audit (correctness first).** Audit Scalability, SLB, Beam Search, Trace GC and
Zero-Copy subsections against `master_metrics.txt`. Extend `scripts/longmemeval/claims_inventory.py`
with a kernel-numbers mode so both halves verify from one script. This phase changes facts and
must precede all prose work.

**Phase 2. Mechanical.** Em-dashes to zero. Literal U+2014 removed. Bold density reduced.
Abstract rewritten to 250 words. Register scrub ("honest", aphorisms, marketing terms).

**Phase 3. Structural.** Unifying thesis into Abstract, Introduction and Conclusion.
Contributions consolidated to 4 or 5. Related Work extended with agent-memory literature.
References grown to roughly 40.

**Phase 4. Journal-mandatory additions.** Limitations and Threats to Validity section. Statistics
paragraph with multiple-comparisons policy. Reproducibility statement. Two to three evaluation
figures.

**Phase 5. Disclosure and tone.** Per the user's decision: strip process narrative, keep negative
results as impersonal findings, keep the competitor comparison. Preserve the erratum floor.

**Phase 6. Final pass.** Full verification, page-count check, read end to end for argument flow.

---

## Part VI. Files

- `paper/paper_final_v4.tex` (650 lines) and `paper/v4_semantic_layer.tex` (306 lines): the manuscript.
- `paper/references.bib`: 18 entries, to grow to roughly 40.
- `paper/generate_figures.py`: existing figure pipeline.
- `reproducibility_benchmarks/master_metrics.txt`: authoritative kernel numbers. Never transcribe
  from the v3 source; regenerate with `reproducibility_benchmarks/run_all_benchmarks.sh`.
- `scripts/longmemeval/claims_inventory.py`: authoritative evaluation numbers.
- `spatiotemporal.md` section 2: verified competitor quotes, URLs and access dates.

---

## Part VII. Verification Suite

Run after every phase.

```bash
# 1. Compiles clean
cd paper && latexmk -pdf -interaction=nonstopmode paper_final_v4.tex
grep -c "undefined" paper_final_v4.log          # must be 0

# 2. Dashes, on RENDERED output (source grep is not sufficient)
pdftotext paper_final_v4.pdf - | grep -c '—'    # em-dash, must be 0
pdftotext paper_final_v4.pdf - | grep -c '–'    # en-dash, expect 2 numeric ranges

# 3. Evaluation figures still match committed result files
.venv/bin/python scripts/longmemeval/claims_inventory.py

# 4. Retired phrasings absent
pdftotext paper_final_v4.pdf - | grep -icE "beats the oracle|statistical parity|449/500|establishes parity|inflated"

# 5. Word and density counts
sed -n '/begin{abstract}/,/end{abstract}/p' paper_final_v4.tex | wc -w   # target 200-250
grep -c "textbf" paper_final_v4.tex paper/v4_semantic_layer.tex          # target ~40 combined
```

**Acceptance:** compiles clean; zero rendered em-dashes; every figure reproduced by
`claims_inventory.py` and `master_metrics.txt`; all ten Part II invariants intact; erratum note
present; Limitations, statistics and reproducibility sections present.
