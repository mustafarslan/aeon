#!/usr/bin/env python3
"""Four-way failure attribution using LongMemEval's `has_answer` turn flags.

ZERO LLM CALLS. Reuses the `extracted_facts` already stored in the n=500
extract-then-compute run and rebuilds each question's assembled context
deterministically, so this is pure re-analysis -- no benchmark rerun.

The question this settles (v4-plan.md failure inventory): the multi-session
aggregation bucket was hand-checked on 5 cases and split 2 retrieval-miss /
3 extraction-loss. Five cases can't size a bucket. LongMemEval flags the
specific turns that contain the answer (`has_answer: true`, 896 of them
across the 500 questions), which turns the split into a mechanical
measurement over every question:

  retrieval miss     -- an answer-bearing turn never reached the assembled
                        context at all (Aeon-side recall failure)
  extraction loss    -- the answer turn WAS in context, but EXTRACT's output
                        doesn't carry its content (EXTRACT-side failure)
                        *** WITHDRAWN -- DO NOT QUOTE THIS AXIS. Validated
                        against questions answered CORRECTLY with all answer
                        turns retrieved (where extraction demonstrably worked):
                        the content-word metric flags 50% of them (generous
                        threshold) and 91% (strict) as "extraction incomplete"
                        -- a HIGHER false-positive rate than it produces on
                        wrong answers. It measures turn verbosity, not
                        extraction fidelity, and has no discriminating power.
                        Retained only so the negative result is reproducible.
                        See v4-plan.md. Splitting extraction from compute needs
                        an LLM-judged pass. ***
  compute/judge      -- answer turn retrieved AND extracted, still wrong
                        (COMPUTE reasoning, or judge nondeterminism)
  correct            -- answered correctly

Matching, and its asymmetry (stated up front so the numbers aren't
over-read):
  * Retrieval side is near-exact -- normalized-substring containment of two
    windows of the turn text against the assembled context. Calibrated
    against the 5 hand-verified cases from `aggregation_locus_check.py`.
  * Extraction side is APPROXIMATE -- extraction paraphrases, so substring
    matching fails and would wildly overcount extraction loss. Uses
    content-word recall (stopwords stripped) at two thresholds, reported as a
    RANGE (strict/generous), never a point estimate.

Known limitation, matters for the temporal rows: `has_answer` marks the turns
the benchmark's authors deemed sufficient. A multi-hop temporal question may
legitimately need additional un-flagged turns (date anchors), so "all answer
turns present but still wrong" on temporal-reasoning is NOT automatically a
compute failure -- part of it is the `question_date` gap (fixed in code but
not yet re-run; the stored extractions this reads are all pre-fix).
Multi-session rows are therefore cleaner than temporal rows.

Usage:
    python scripts/longmemeval/answer_turn_attribution.py --dataset ... \
        --results reproducibility_benchmarks/longmemeval/extract_then_compute_n500_results.json \
        --out reproducibility_benchmarks/longmemeval/answer_turn_attribution.json
    # calibration only (the 5 hand-verified cases, no full run):
    python scripts/longmemeval/answer_turn_attribution.py ... --calibrate
"""

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_benchmark import _get_encoder, _ingest_haystack  # noqa: E402

import numpy as np  # noqa: E402
from aeon_py.session_expansion import build_expanded_context, format_events  # noqa: E402
from aeon_py.trace import TraceGraph  # noqa: E402

BASE_TOP_K = 30
MAX_SESSIONS = 10

# Ground truth for the retrieval matcher, established by inspecting each
# `has_answer` turn individually against the rebuilt context.
#
# CORRECTION (2026-08-25): three of these were initially recorded from
# `aggregation_locus_check.py`, which grepped the context for a KEYWORD
# ("subscri", "graduation") rather than for the answer-bearing TURN. That
# conflates "the word appears somewhere in 120k chars" with "the turn holding
# the answer was retrieved" -- the keyword hits were in other, non-answer turns
# (assistant echoes, unrelated mentions). Checked per-turn, `1a8a66a6` and
# `81507db6` each have THREE answer turns of which only ONE reached the
# context: they are retrieval misses, not the extraction losses the keyword
# grep reported. The keyword method systematically under-reports retrieval
# misses on exactly the aggregation questions it was built to diagnose, since
# an aggregation question repeats its topic word in every session. Only
# `gpt4_ab202e7f` survives as a verified extraction loss (5/5 answer turns
# present, extraction surfaced 4).
CALIBRATION = {
    "8e91e7d9": "retrieval_miss",       # 1/2 answer turns in context
    "ba358f49": "retrieval_miss",       # 1/2 answer turns in context
    "1a8a66a6": "retrieval_miss",       # 1/3 -- was mis-called "retrieved" by keyword grep
    "gpt4_ab202e7f": "retrieved",       # 5/5 answer turns in context (true extraction loss)
    "81507db6": "retrieval_miss",       # 1/3 -- was mis-called "retrieved" by keyword grep
}

_STOP = set("""a an the and or but if then than that this these those there here of in on at to for
with without from by as is are was were be been being am do does did doing have has had having i me
my mine you your yours he him his she her hers it its we us our ours they them their theirs what
which who whom when where why how all any both each few more most other some such no nor not only own
same so too very can will just should now about into over after before again once it's i'm i've don't
would could may might must shall get got go going went said say says also really been""".split())


def _norm(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace -- applied identically
    to the turn text and the assembled context so containment is meaningful."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()


def _content_words(text: str) -> set:
    return {w for w in _norm(text).split() if w not in _STOP and len(w) > 2}


def _turn_in_context(turn_text: str, norm_ctx: str) -> bool:
    """Near-exact: does this answer-bearing turn appear in the assembled
    context? Uses two windows (head + middle) rather than the whole turn,
    because format_events may rewrap or truncate long turns -- requiring a
    full-text match would score formatting differences as retrieval misses."""
    nt = _norm(turn_text)
    if not nt:
        return False
    if len(nt) <= 120:
        return nt in norm_ctx
    head = nt[:200]
    mid_start = max(0, len(nt) // 2 - 100)
    mid = nt[mid_start:mid_start + 200]
    return head in norm_ctx or mid in norm_ctx


def _turn_in_extraction(turn_text: str, facts: str, threshold: float) -> bool:
    """Approximate: extraction paraphrases, so this measures what fraction of
    the answer turn's content words survive into the extracted facts."""
    tw = _content_words(turn_text)
    if not tw:
        return False
    fw = _content_words(facts)
    return len(tw & fw) / len(tw) >= threshold


def _assembled_context(question, encoder, tmp_dir: Path) -> str:
    qid = question["question_id"]
    tp = tmp_dir / f"{qid}.trace"
    for sfx in ("", ".blobs", ".wal"):
        Path(str(tp) + sfx).unlink(missing_ok=True)
    trace = TraceGraph(path=str(tp))
    _ingest_haystack(trace, encoder, question)
    qv = np.asarray(encoder.encode(question["question"]), dtype=np.float32).tolist()
    events = build_expanded_context(
        trace, qv, "full_session", base_top_k=BASE_TOP_K, max_sessions=MAX_SESSIONS,
    )
    ctx = format_events(events)
    for sfx in ("", ".blobs", ".wal"):
        Path(str(tp) + sfx).unlink(missing_ok=True)
    return ctx


def _answer_turns(question) -> list:
    return [t for sess in question["haystack_sessions"] for t in sess if t.get("has_answer")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--tmp-dir", default="/tmp/aeon_attr")
    ap.add_argument("--strict", type=float, default=0.50)
    ap.add_argument("--generous", type=float, default=0.25)
    args = ap.parse_args()

    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    dataset = {q["question_id"]: q for q in json.load(open(args.dataset))}
    stored = {r["question_id"]: r for r in json.load(open(args.results))["results"]}
    encoder = _get_encoder()

    if args.calibrate:
        print("CALIBRATION against hand-verified locus checks:")
        ok = True
        for qid, expected in CALIBRATION.items():
            q = dataset[qid]
            ctx = _assembled_context(q, encoder, tmp_dir)
            nctx = _norm(ctx)
            turns = _answer_turns(q)
            present = sum(1 for t in turns if _turn_in_context(t["content"], nctx))
            got = "retrieved" if present == len(turns) and turns else (
                "retrieval_miss" if present < len(turns) else "no_answer_turns")
            verdict = "OK " if got == expected else "MISMATCH"
            if got != expected:
                ok = False
            print(f"  {verdict} {qid:<16} expected={expected:<15} got={got:<15} "
                  f"({present}/{len(turns)} answer turns in context)")
        print("\nCalibration", "PASSED" if ok else "FAILED -- matcher needs work")
        return

    ids = [q for q in stored if q in dataset]
    print(f"Attributing {len(ids)} questions (zero LLM calls)...", file=sys.stderr)

    rows = []
    t_start = time.perf_counter()
    for i, qid in enumerate(ids, 1):
        q, rec = dataset[qid], stored[qid]
        turns = _answer_turns(q)
        if not turns:
            rows.append({"question_id": qid, "report_type": rec["report_type"],
                         "correct": rec["correct"], "category": "no_answer_turns",
                         "n_answer_turns": 0})
            continue

        ctx = _assembled_context(q, encoder, tmp_dir)
        nctx = _norm(ctx)
        facts = rec["extracted_facts"]

        retrieved = [t for t in turns if _turn_in_context(t["content"], nctx)]
        n_ret = len(retrieved)
        ext_strict = sum(1 for t in retrieved if _turn_in_extraction(t["content"], facts, args.strict))
        ext_gen = sum(1 for t in retrieved if _turn_in_extraction(t["content"], facts, args.generous))

        if rec["correct"]:
            cat = "correct"
        elif n_ret < len(turns):
            cat = "retrieval_miss"
        elif ext_gen < len(turns):
            cat = "extraction_loss"
        else:
            cat = "compute_or_judge"

        rows.append({
            "question_id": qid, "report_type": rec["report_type"], "correct": rec["correct"],
            "category": cat, "n_answer_turns": len(turns), "n_retrieved": n_ret,
            "n_extracted_strict": ext_strict, "n_extracted_generous": ext_gen,
            "assistant_answer_turns": sum(1 for t in turns if t.get("role") == "assistant"),
            "assistant_turns_retrieved": sum(1 for t in retrieved if t.get("role") == "assistant"),
            "context_chars": len(ctx),
            "context_chars_recorded": rec.get("context_chars"),
        })
        if i % 25 == 0:
            el = time.perf_counter() - t_start
            print(f"  {i}/{len(ids)}  ({el:.0f}s, ~{el/i*(len(ids)-i):.0f}s left)", file=sys.stderr)

    # ---- report ----
    real = [r for r in rows if r["category"] != "no_answer_turns"]
    print("\n" + "=" * 86)
    print("FOUR-WAY ATTRIBUTION (excludes questions with no answer-bearing turns)")
    print("=" * 86)
    types = ["temporal-reasoning", "multi-session", "single-session-preference",
             "single-session-user", "knowledge-update", "single-session-assistant", "abstention"]
    hdr = f"{'type':<27}{'n':>5}{'correct':>9}{'retr.miss':>11}{'extr.loss':>11}{'compute':>9}"
    print(hdr)
    for t in types:
        sub = [r for r in real if r["report_type"] == t]
        if not sub:
            continue
        c = Counter(r["category"] for r in sub)
        print(f"{t:<27}{len(sub):>5}{c['correct']:>9}{c['retrieval_miss']:>11}"
              f"{c['extraction_loss']:>11}{c['compute_or_judge']:>9}")
    c = Counter(r["category"] for r in real)
    print(f"{'TOTAL':<27}{len(real):>5}{c['correct']:>9}{c['retrieval_miss']:>11}"
          f"{c['extraction_loss']:>11}{c['compute_or_judge']:>9}")

    errs = [r for r in real if not r["correct"]]
    ce = Counter(r["category"] for r in errs)
    print(f"\nOf {len(errs)} errors: retrieval_miss={ce['retrieval_miss']} "
          f"({ce['retrieval_miss']/max(1,len(errs))*100:.0f}%), "
          f"extraction_loss={ce['extraction_loss']} ({ce['extraction_loss']/max(1,len(errs))*100:.0f}%), "
          f"compute_or_judge={ce['compute_or_judge']} ({ce['compute_or_judge']/max(1,len(errs))*100:.0f}%)")

    print("\n--- CAUSAL CHECK: answer-turn recall, correct vs wrong ---")
    for label, grp in (("correct", [r for r in real if r["correct"]]),
                       ("wrong", errs)):
        if not grp:
            continue
        frac = sum(r["n_retrieved"] / r["n_answer_turns"] for r in grp) / len(grp)
        allp = sum(1 for r in grp if r["n_retrieved"] == r["n_answer_turns"]) / len(grp)
        print(f"  {label:<8} n={len(grp):<4} mean answer-turn recall={frac*100:.1f}%  "
              f"all-present rate={allp*100:.1f}%")

    print("\n*** WARNING: the extraction_loss column above is WITHDRAWN as invalid --")
    print("*** it flags ~50% of CORRECTLY-answered questions as 'extraction incomplete'.")
    print("*** Treat extraction_loss + compute_or_judge as one 'retrieved-but-wrong' bucket.")
    print("\n--- EXTRACTION THRESHOLD SENSITIVITY (the approximate axis) ---")
    ret_ok = [r for r in real if r["n_retrieved"] == r["n_answer_turns"]]
    for name, key in (("strict %.2f" % args.strict, "n_extracted_strict"),
                      ("generous %.2f" % args.generous, "n_extracted_generous")):
        full = sum(1 for r in ret_ok if r[key] == r["n_answer_turns"])
        print(f"  {name}: {full}/{len(ret_ok)} fully-retrieved questions also fully extracted "
              f"({full/max(1,len(ret_ok))*100:.0f}%)")

    print("\n--- ASSISTANT-ROLE ANSWER TURNS ---")
    aa = [r for r in real if r.get("assistant_answer_turns")]
    print(f"  {len(aa)} questions have assistant-role answer turns; "
          f"retrieved {sum(r['assistant_turns_retrieved'] for r in aa)}/"
          f"{sum(r['assistant_answer_turns'] for r in aa)}")

    mismatch = [r for r in real if r.get("context_chars_recorded") and
                r["context_chars"] != r["context_chars_recorded"]]
    print(f"\n--- INTEGRITY: rebuilt context differs from recorded on {len(mismatch)}/{len(real)} "
          f"questions (0 expected -- retrieval is deterministic) ---")

    if args.out:
        json.dump({"strict": args.strict, "generous": args.generous, "results": rows},
                  open(args.out, "w"), indent=2)
        print(f"\nWrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
