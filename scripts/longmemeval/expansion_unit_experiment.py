#!/usr/bin/env python3
"""
Stage 7 task 1 -- expansion-unit experiment (v4-plan.md Stage 7).

Determines which retrieval-EXPANSION UNIT (not which top_k) closes most of
LongMemEval-S multi-session's oracle-vs-real gap, before any kernel/shell
API is built around a guess. Stage 6's two oracle controls each handed
over a different unit and each beat real top_k retrieval: `oracle_run.py`
uses the whole gold session's raw text; the V2 oracle (scripts/longmemeval-v2/
oracle_run.py) uses a window centered on matched content. Nothing run so
far isolates which unit is doing the work.

Exactly five arms, fixed -- not open-ended (Stage 7's own scoping
requirement, after this project relitigated an unscoped finding three
times during Stage 6):

  1. full_session  -- every event in the top hit's session (ceiling)
  2. window_3      -- +-3 events around the top hit
  3. window_5      -- +-5 events around the top hit
  4. window_10     -- +-10 events around the top hit
  5. summary       -- LLM-produced summary of the top hit's session
                      (most expensive; only worth running per the
                      decision rule below if no window arm qualifies)

All five build on `shell/aeon_py/session_expansion.py`'s primitives
(`find_top_hit`, `expand_full_session`, `expand_window`, `expand_summary`)
rather than reimplementing retrieval logic here, so this experiment tests
the exact same code Stage 7 task 2 would wire into a real capability --
not a separate, potentially-drifted reimplementation.

DECISION RULE (fixed before any arm runs, per Stage 7's own text): pick
the cheapest arm (by mean retrieved-token/char count) that lands within
10 accuracy points of the full_session ceiling. If more than one clears
that bar, the smallest token count wins outright. If none clears it,
full_session wins by default and the gap to the cheapest window arm is
task 2's known, stated cost/quality tradeoff. `_apply_decision_rule()`
below computes this mechanically from the run's own results -- the point
is to make the call impossible to relitigate with hindsight once numbers
are in hand.

CONTAMINATION GUARD (required before trusting any arm's number, per
Stage 7's own text -- this project hit both bug classes during Stage 6):
every arm ingests into a freshly-deleted scratch trace path, every call
runs at `temperature=0.0`, and `num_errors: 0` is a precondition for
reading an arm's accuracy at all -- a nonzero count means re-run that arm,
not average around it.

THIS FILE IS NOT RUN AS PART OF WRITING THIS STAGE. Written and
statically checked (syntax, imports) only -- no Ollama calls, no dataset
scoring -- per explicit instruction to hold all execution for a later
session.

Usage (once ready to actually run it):
    python scripts/longmemeval/expansion_unit_experiment.py \\
        --dataset /path/to/longmemeval_s_cleaned.json \\
        --num-questions 50 --seed 42 --model qwen3.8:27b-mlx \\
        --out reproducibility_benchmarks/longmemeval/expansion_unit_results.json

    # Skip the expensive summary arm until the window arms have been seen:
    python scripts/longmemeval/expansion_unit_experiment.py ... \\
        --arms full_session window_3 window_5 window_10
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_prompts import get_anscheck_prompt  # noqa: E402
from run_benchmark import (  # noqa: E402
    SYSTEM_PROMPT, _generate_with_retry, _get_encoder, _ingest_haystack,
    _stratified_sample,
)

import numpy as np  # noqa: E402
from aeon_py.llm import OllamaProvider  # noqa: E402
from aeon_py.session_expansion import build_expanded_context, format_events  # noqa: E402
from aeon_py.trace import TraceGraph  # noqa: E402

ARMS = ("full_session", "window_3", "window_5", "window_10", "summary")

# Stage 7's decision-rule band: an arm within this many accuracy points of
# full_session's ceiling is "close enough" to prefer on cost instead.
DECISION_RULE_BAND_POINTS = 10.0

# Originally 5 (this project's dataset sample's max observed gold-session
# count, 4, plus margin). Raised to 10 per v4-plan.md Stage 7's gold-session
# recall@N diagnostic (LLM-free, scripts/longmemeval/session_recall_analysis.py):
# all-golds-present recall for multi-session was only 63.6% at N=5, but
# 90.9% at N=10 -- matching the full top_k=30 ceiling exactly. The gold
# sessions were reachable, just ranked 6th-10th by ordinary embedding rank
# noise, not excluded by relevance -- this is a data-justified retrieval-
# width correction, not a parameter tuned until a score moved. See
# session_expansion.py's multi-session section for the earlier bug this
# constant already survived: every arm originally anchored to ONE session
# and REPLACED the retrieved context with it, collapsing knowledge-update
# (always 2 gold sessions) and multi-session (3-4) to near-zero.
MAX_SESSIONS = 10
BASE_TOP_K = 30


def _fresh_trace_path(tmp_dir: Path, question_id: str, arm: str) -> Path:
    """Same contamination-guard pattern as run_benchmark.py's per-question
    trace handling and the V2 harness's `fresh_trace_path()` -- delete any
    existing trace + sidecars before opening, every time. This project hit
    the identical stale-scratch-file bug twice during Stage 6; a fixed
    per-(question, arm) path reused across a re-run without this guard
    would reproduce it a third time."""
    trace_path = tmp_dir / f"{question_id}_{arm}.trace"
    for suffix in ("", ".blobs", ".wal"):
        Path(str(trace_path) + suffix).unlink(missing_ok=True)
    return trace_path


def _build_context(arm: str, trace: TraceGraph, q_vec: list[float], llm: OllamaProvider) -> str:
    """Dispatches to the expansion unit named by `arm`, using
    `session_expansion.py`'s `build_expanded_context()` exclusively --
    multi-session-anchored (top `MAX_SESSIONS` distinct sessions among a
    top_k=`BASE_TOP_K` search, not a single top hit) and additive (merged
    onto those top_k hits, never replacing them) -- this function's job is
    picking which unit, not implementing any of them."""
    generate_fn = (
        (lambda p: _generate_with_retry(llm, p, system_prompt="", temperature=0.0))
        if arm == "summary" else None
    )
    events = build_expanded_context(
        trace, q_vec, arm, base_top_k=BASE_TOP_K, max_sessions=MAX_SESSIONS, generate_fn=generate_fn,
    )
    return format_events(events)


def _run_one(question: dict, arm: str, encoder, llm: OllamaProvider, tmp_dir: Path) -> dict:
    trace_path = _fresh_trace_path(tmp_dir, question["question_id"], arm)
    trace = TraceGraph(path=str(trace_path))
    ingest_seconds = _ingest_haystack(trace, encoder, question)

    t0 = time.perf_counter()
    q_vec = np.asarray(encoder.encode(question["question"]), dtype=np.float32).tolist()
    context_block = _build_context(arm, trace, q_vec, llm)
    context_seconds = time.perf_counter() - t0

    user_prompt = f"Retrieved memories:\n{context_block}\n\nQuestion: {question['question']}\n\nAnswer:"

    t0 = time.perf_counter()
    response = _generate_with_retry(llm, user_prompt, system_prompt=SYSTEM_PROMPT, temperature=0.0)
    generation_seconds = time.perf_counter() - t0
    # Recorded per-question, not just measured once by hand -- a fixed
    # num_ctx=8192 default silently truncated this exact experiment's
    # larger arms (v4-plan.md Stage 7: `full_session` needed ~12.8K tokens
    # against that 8192 cap). `OllamaProvider` now auto-sizes num_ctx from
    # the actual prompt; recording what it resolved to here means a future
    # truncation regression is visible in the result file, not something
    # that has to be independently re-measured.
    context_num_ctx = llm.last_num_ctx

    is_abstention = "_abs" in question["question_id"]
    is_error = "[System Error:" in response

    if is_error:
        judge_response = ""
        judge_seconds = 0.0
        correct = False
    else:
        t0 = time.perf_counter()
        judge_prompt = get_anscheck_prompt(
            question["question_type"], question["question"], question["answer"], response,
            abstention=is_abstention,
        )
        judge_response = _generate_with_retry(llm, judge_prompt, system_prompt="", temperature=0.0)
        judge_seconds = time.perf_counter() - t0
        correct = "yes" in judge_response.lower()

    # Scratch state, not meant to persist past this question -- same
    # cleanup run_benchmark.py's per-question harness already does.
    try:
        for suffix in ("", ".blobs", ".wal"):
            Path(str(trace_path) + suffix).unlink(missing_ok=True)
    except OSError:
        pass

    return {
        "question_id": question["question_id"],
        "question_type": question["question_type"],
        "report_type": "abstention" if is_abstention else question["question_type"],
        "arm": arm,
        "reference_answer": question["answer"],
        "hypothesis": response,
        "judge_raw": judge_response,
        "correct": correct,
        "is_error": is_error,
        "retrieved_chars": len(context_block),
        "context_num_ctx": context_num_ctx,
        "timing_seconds": {
            "ingest": ingest_seconds,
            "context_build": context_seconds,
            "generation": generation_seconds,
            "judge": judge_seconds,
        },
    }


def _summarize_arm(results: list[dict], arm: str) -> dict:
    rows = [r for r in results if r["arm"] == arm]
    by_type: dict[str, list[dict]] = {}
    for r in rows:
        by_type.setdefault(r["report_type"], []).append(r)

    def acc(rr):
        scored = [r for r in rr if not r["is_error"]]
        return sum(1 for r in scored if r["correct"]) / len(scored) if scored else 0.0

    return {
        "arm": arm,
        "n": len(rows),
        "n_errors": sum(1 for r in rows if r["is_error"]),
        "accuracy": acc(rows),
        "mean_retrieved_chars": (sum(r["retrieved_chars"] for r in rows) / len(rows)) if rows else 0.0,
        "context_num_ctx": {
            "min": min((r["context_num_ctx"] for r in rows), default=None),
            "mean": (sum(r["context_num_ctx"] for r in rows) / len(rows)) if rows else None,
            "max": max((r["context_num_ctx"] for r in rows), default=None),
        },
        "per_question_type": {t: {"n": len(rr), "accuracy": acc(rr)} for t, rr in sorted(by_type.items())},
    }


def _load_baseline_accuracy(baseline_path: Path) -> tuple[float, dict[str, float]]:
    """Loads the recorded, deterministic top_k=30 baseline this repo
    already has on record (reproducibility_benchmarks/longmemeval/
    topk_sweep_30_temp0_gemma.json) -- overall and per-question-type
    accuracy. Not hardcoded, so a re-run of that baseline automatically
    updates what this guard checks against."""
    with open(baseline_path) as f:
        summary = json.load(f)["summary"]
    overall = summary["overall_accuracy"]
    per_type = {t: row["accuracy"] for t, row in summary["per_question_type"].items()}
    return overall, per_type


def _report_baseline_deltas(
    arm_summaries: dict[str, dict], baseline_overall: float, baseline_per_type: dict[str, float],
    tolerance_points: float = 2.0,
) -> list[str]:
    """INFORMATIONAL ONLY -- does not block `_apply_decision_rule()`.

    This function used to be a blocking contamination guard on the theory
    that, since `build_expanded_context()` merges every unit ADDITIVELY
    onto the same top_k=30 hits (retrieval is a strict superset of the
    baseline's), no arm's accuracy should ever score below that recorded
    baseline. Advisor-caught error in that reasoning, confirmed by this
    experiment's own v3 results (v4-plan.md): the additive merge only
    guarantees a superset of RETRIEVED events, not a superset of DOWNSTREAM
    ACCURACY -- an LLM's answer quality is not monotonic in how much
    correct-but-irrelevant context it's handed. `full_session` (which
    merges the MOST extra content into the exact same unsorted list) beat
    the baseline outright (70.0% vs 68.0%); `window_3`/`window_5`/`summary`
    (which add distractor turns from a session without reliably adding the
    one relevant fact) scored a few points below it on multi-session and
    temporal-reasoning. That is a real result about which units help
    LongMemEval-S's question mix, not a bug -- treating it as contamination
    would have discarded the exact signal Stage 7 task 1 exists to produce.

    Returns human-readable delta strings (both above and below baseline)
    for the printed report and the output JSON -- reporting only, never
    used to gate the decision rule."""
    deltas = []
    for arm, s in arm_summaries.items():
        overall_gap = (s["accuracy"] - baseline_overall) * 100
        if abs(overall_gap) > tolerance_points:
            sign = "above" if overall_gap > 0 else "below"
            deltas.append(
                f"Arm '{arm}': {s['accuracy']*100:.1f}% overall, "
                f"{abs(overall_gap):.1f} points {sign} the top_k=30 baseline ({baseline_overall*100:.1f}%)."
            )
        for qtype, base_acc in baseline_per_type.items():
            row = s["per_question_type"].get(qtype)
            if row is None:
                continue
            type_gap = (row["accuracy"] - base_acc) * 100
            if abs(type_gap) > tolerance_points:
                sign = "above" if type_gap > 0 else "below"
                deltas.append(
                    f"Arm '{arm}', type '{qtype}': {row['accuracy']*100:.1f}% vs "
                    f"baseline {base_acc*100:.1f}% ({abs(type_gap):.1f} points {sign})."
                )
    return deltas


def _apply_decision_rule(arm_summaries: dict[str, dict]) -> dict:
    """Mechanical, pre-committed application of Stage 7's decision rule --
    see this module's docstring. Returns {"winner": arm, "reason": str}.
    Raises if `full_session` is missing (it's the ceiling every other arm
    is measured against) or if any included arm has `n_errors > 0` (the
    contamination guard: don't decide anything from a contaminated arm)."""
    if "full_session" not in arm_summaries:
        raise ValueError("Decision rule requires the 'full_session' arm as the ceiling reference")
    for arm, s in arm_summaries.items():
        if s["n_errors"] > 0:
            raise ValueError(
                f"Arm '{arm}' has {s['n_errors']} transport error(s) -- re-run before deciding, "
                f"per Stage 7's contamination guard. Do not average around it."
            )

    ceiling = arm_summaries["full_session"]["accuracy"] * 100
    band_floor = ceiling - DECISION_RULE_BAND_POINTS

    candidates = [
        (arm, s) for arm, s in arm_summaries.items()
        if arm != "full_session" and s["accuracy"] * 100 >= band_floor
    ]
    if not candidates:
        return {
            "winner": "full_session",
            "reason": (
                f"No arm reached within {DECISION_RULE_BAND_POINTS} points of the "
                f"full_session ceiling ({ceiling:.1f}%); full_session wins by default."
            ),
        }
    winner_arm, winner_summary = min(candidates, key=lambda kv: kv[1]["mean_retrieved_chars"])
    return {
        "winner": winner_arm,
        "reason": (
            f"'{winner_arm}' reached {winner_summary['accuracy']*100:.1f}% "
            f"(ceiling {ceiling:.1f}%, band floor {band_floor:.1f}%) at "
            f"{winner_summary['mean_retrieved_chars']:.0f} mean retrieved chars -- "
            f"cheapest arm clearing the band."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Path to longmemeval_s_cleaned.json")
    parser.add_argument("--num-questions", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=None, help="Overrides AEON_LLM_MODEL for this run")
    parser.add_argument(
        "--arms", nargs="+", default=list(ARMS), choices=list(ARMS),
        help="Restrict to a subset of the 5 fixed arms (default: all 5). "
             "'full_session' must be included for the decision rule to run -- "
             "it is the ceiling every other arm is measured against.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--tmp-dir", default=None)
    parser.add_argument(
        "--baseline",
        default="reproducibility_benchmarks/longmemeval/topk_sweep_30_temp0_gemma.json",
        help="Recorded top_k=30 baseline to regression-check every arm against "
             "(since expansion is additive onto the same top_k hits, no arm should "
             "score below it -- see _check_baseline_regression).",
    )
    args = parser.parse_args()

    import os
    import tempfile
    if args.model:
        os.environ["AEON_LLM_MODEL"] = args.model

    with open(args.dataset) as f:
        all_questions = json.load(f)
    sample = _stratified_sample(all_questions, args.num_questions, args.seed)
    print(f"Sampled {len(sample)} questions (seed={args.seed}) from {len(all_questions)} total")
    print(f"Arms: {args.arms}")

    encoder = _get_encoder()
    llm = OllamaProvider()
    print(f"Using Ollama model: {llm.model}")
    print("Warming up model (forces full load before timing/scoring begins)...")
    warm = _generate_with_retry(llm, "Say OK.", retries=5)
    print(f"  warm-up response: {warm[:80]!r}")

    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else Path(tempfile.mkdtemp(prefix="expansion_unit_"))
    tmp_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for arm in args.arms:
        for i, q in enumerate(sample):
            r = _run_one(q, arm, encoder, llm, tmp_dir)
            all_results.append(r)
            status = "TRANSPORT_ERR" if r["is_error"] else ("OK" if r["correct"] else "WRONG")
            print(
                f"[{arm}][{i+1}/{len(sample)}] {q['question_id']} ({q['question_type']}) "
                f"-> {status} ({r['retrieved_chars']} chars)",
                flush=True,
            )

    arm_summaries = {arm: _summarize_arm(all_results, arm) for arm in args.arms}

    baseline_overall, baseline_per_type = _load_baseline_accuracy(Path(args.baseline))
    baseline_deltas = _report_baseline_deltas(arm_summaries, baseline_overall, baseline_per_type)
    if baseline_deltas:
        print("\n=== Accuracy deltas vs. top_k=30 baseline (informational, not a contamination guard) ===")
        for d in baseline_deltas:
            print(f"  - {d}")

    decision = None
    if "full_session" in args.arms:
        try:
            decision = _apply_decision_rule(arm_summaries)
        except ValueError as e:
            decision = {"winner": None, "reason": f"Decision rule could not run: {e}"}

    out = {
        "model": llm.model,
        "seed": args.seed,
        "num_questions": len(sample),
        "arms_run": args.arms,
        "decision_rule_band_points": DECISION_RULE_BAND_POINTS,
        "max_sessions": MAX_SESSIONS,
        "base_top_k": BASE_TOP_K,
        "baseline_path": str(args.baseline),
        "baseline_deltas": baseline_deltas,
        "arm_summaries": arm_summaries,
        "decision": decision,
        "results": all_results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== Arm summaries ===")
    print(json.dumps(arm_summaries, indent=2))
    if decision:
        print("\n=== Decision rule ===")
        print(json.dumps(decision, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
