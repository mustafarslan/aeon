#!/usr/bin/env python3
"""
Extract-then-compute prompt experiment (v4-plan.md Stage 7, post-Track-1
literal-vs-computed diagnostic).

Stage 7 task 1's corrected result (`full_session` arm, `MAX_SESSIONS=10`,
untruncated, same-model): multi-session 45.5%, temporal-reasoning 46.2%,
against a same-model oracle ceiling of 72.7%/53.8%. The literal-vs-computed
diagnostic (`answer_in_context.py --unit full_session`) showed WHY: 11 of 13
sampled multi-session answers are computed values (sums, counts, day-count
differences), and temporal-reasoning splits cleanly into literal answers
(57.1% answer-in-context) vs. computed date-arithmetic answers (0%, expected
-- the model must compute a difference between two separately-stated dates,
never recall a stated day-count). Session recall is already ~90-100% at this
point (gold-session recall@N diagnostic) -- the model has the right facts in
front of it and still gets these wrong more than half the time.

This experiment tests one candidate fix, cheap and requiring NO kernel or
retrieval changes: does splitting generation into two explicit steps --
(1) extract every fact relevant to the question, with its source
date/session, from the assembled context, THEN (2) compute/answer using
only the extracted facts -- outperform the existing single-shot prompt on
exactly the question types the diagnostic identified as aggregation/
arithmetic-heavy? This is a prompt-scaffold change, not an architecture
change -- if it doesn't help, the bottleneck is more likely genuine model
reasoning capacity than prompt structure.

Reuses the SAME retrieval as the `full_session` arm (build_expanded_context,
MAX_SESSIONS=10, base_top_k=30) -- this experiment isolates the prompting
strategy, not retrieval, so it deliberately does not re-litigate Task 1's
retrieval-unit question.

CONTAMINATION GUARD (same discipline as expansion_unit_experiment.py): fresh
scratch trace paths, temperature=0.0 throughout, num_ctx auto-sized (fixed
this session -- see llm.py), context_num_ctx recorded per-question so a
future truncation regression is visible in the data, n_errors=0 required
before trusting any accuracy number.

Usage:
    python scripts/longmemeval/extract_then_compute_experiment.py \\
        --dataset /path/to/longmemeval_s_cleaned.json \\
        --num-questions 50 --seed 42 --model gemma4:31b-cloud \\
        --out reproducibility_benchmarks/longmemeval/extract_then_compute_results.json

    # Restrict to just the two question types this technique targets:
    python scripts/longmemeval/extract_then_compute_experiment.py ... \\
        --question-types multi-session temporal-reasoning
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
    _stratified_sample, format_question_with_date,
)

import numpy as np  # noqa: E402
from aeon_py.llm import OllamaProvider  # noqa: E402
from aeon_py.session_expansion import build_expanded_context, format_events  # noqa: E402
from aeon_py.trace import TraceGraph  # noqa: E402

BASE_TOP_K = 30
MAX_SESSIONS = 10  # matches the corrected full_session arm this compares against

EXTRACT_PROMPT_TEMPLATE = (
    "You are extracting facts relevant to a question from retrieved memories. "
    "Do NOT answer the question yet.\n\n"
    "Retrieved memories:\n{context}\n\n"
    "{question_block}\n\n"
    "List every fact from the retrieved memories that is relevant to "
    "answering this question, one per line. Include each fact's date or "
    "session context if the memories state one. Be exhaustive -- include a "
    "fact even if you are not yet sure how it relates to the answer. If no "
    "relevant facts are present, say so explicitly.\n\n"
    "Relevant facts:"
)

COMPUTE_PROMPT_TEMPLATE = (
    "You previously extracted these facts from retrieved memories:\n"
    "{extracted_facts}\n\n"
    "{question_block}\n\n"
    "Using ONLY the facts above, determine the answer. If the question "
    "requires combining multiple facts (a sum, a count, a date difference), "
    "show the calculation briefly. Give your final answer on its own line, "
    "prefixed exactly with 'Answer:'.\n\nAnswer:"
)

# v3 (2026-08-25), TRIED AND REVERTED -- kept as a recorded negative result.
# Targeted relaxation of an over-constraint diagnosed from reading all 10
# single-session-user/preference losses in the n=500 paired run
# (v4-plan.md): EXTRACT always had the right fact; COMPUTE refused to use
# it because "use ONLY the facts, determine the answer" framed every
# question as an arithmetic problem. Pre-registered with an acceptance bar
# before running (n=500, ETC arm only, `extract_then_compute_n500_v3_results.json`).
# Result: single-session-user accuracy was EXACTLY unchanged (55/64 both
# v1 and v3) -- re-reading the same 5 diagnosed cases showed the model
# produced near-identical "not mentioned" hedges word-for-word despite the
# new instructions explicitly telling it not to. single-session-preference
# genuinely improved (11/30 -> 17/30), but knowledge-update (-4) and
# temporal-reasoning (-3) regressed, for an exact overall tie (390/500
# both times). Reverted per the pre-committed one-attempt protocol -- no
# second iteration.
# COMPUTE_PROMPT_TEMPLATE = (
#     "You previously extracted these facts from retrieved memories:\n"
#     "{extracted_facts}\n\n"
#     "Question: {question}\n\n"
#     "Using ONLY the facts above, determine the answer.\n"
#     "- If a fact directly states or clearly implies the information asked "
#     "for, that fact IS the answer -- do not say it is 'not mentioned' or "
#     "'not specified' just because the fact is worded differently than the "
#     "question or omits a category label the question happens to use.\n"
#     "- If the question asks for a recommendation or suggestion, synthesize "
#     "one directly from the facts above -- do not refuse just because no "
#     "single fact is itself phrased as a recommendation.\n"
#     "- If the question requires combining multiple facts (a sum, a count, "
#     "a date difference), show the calculation briefly.\n"
#     "- Only say the facts are insufficient if, after checking carefully, "
#     "the specific information asked for is genuinely absent above.\n"
#     "Give your final answer on its own line, prefixed exactly with "
#     "'Answer:'.\n\nAnswer:"
# )

# v2 attempt (2026-08-24), tried and reverted -- kept here as a recorded
# negative result, not a silently-dropped experiment. Added a
# supersession-handling instruction to fix a real knowledge-update
# regression (see v4-plan.md); re-running showed it traded +1
# knowledge-update question for -1 multi-session and -1 temporal-reasoning
# (net overall 82.0% -> 80.0%). Advisor-reviewed: only one of the two new
# regressions was actually about supersession-vs-repeat confusion (the
# diagnosed root cause); the other two were extraction-completeness and
# general hedging side effects unrelated to it -- and all three deltas are
# one question each on an n=8-13 sample, indistinguishable from noise at
# this size. A v3 tuned against the same 50 questions would be fitting to
# noise, not fixing a diagnosed cause. Reverted to v1 above; the
# knowledge-update regression (1 question, cause understood: extraction
# surfaces both an old and a superseding fact, compute hedges instead of
# picking the current one) remains open, pending a re-test at a sample
# size that can actually distinguish a real fix from noise (n=500, not
# n=50 -- same standard already applied to the retrieval-unit arm
# comparison).


def _fresh_trace_path(tmp_dir: Path, question_id: str) -> Path:
    trace_path = tmp_dir / f"{question_id}_etc.trace"
    for suffix in ("", ".blobs", ".wal"):
        Path(str(trace_path) + suffix).unlink(missing_ok=True)
    return trace_path


def _run_one(question: dict, encoder, llm: OllamaProvider, tmp_dir: Path) -> dict:
    trace_path = _fresh_trace_path(tmp_dir, question["question_id"])
    trace = TraceGraph(path=str(trace_path))
    ingest_seconds = _ingest_haystack(trace, encoder, question)

    t0 = time.perf_counter()
    q_vec = np.asarray(encoder.encode(question["question"]), dtype=np.float32).tolist()
    events = build_expanded_context(
        trace, q_vec, "full_session", base_top_k=BASE_TOP_K, max_sessions=MAX_SESSIONS,
    )
    context_block = format_events(events)
    context_seconds = time.perf_counter() - t0

    # Step 1: extract
    t0 = time.perf_counter()
    extract_prompt = EXTRACT_PROMPT_TEMPLATE.format(
        context=context_block, question_block=format_question_with_date(question),
    )
    extracted_facts = _generate_with_retry(llm, extract_prompt, system_prompt=SYSTEM_PROMPT, temperature=0.0)
    extract_seconds = time.perf_counter() - t0
    extract_num_ctx = llm.last_num_ctx
    is_error_extract = "[System Error:" in extracted_facts

    # Step 2: compute
    response = ""
    compute_seconds = 0.0
    compute_num_ctx = None
    is_error_compute = False
    if not is_error_extract:
        t0 = time.perf_counter()
        compute_prompt = COMPUTE_PROMPT_TEMPLATE.format(
            extracted_facts=extracted_facts,
            question_block=format_question_with_date(question),
        )
        response = _generate_with_retry(llm, compute_prompt, system_prompt=SYSTEM_PROMPT, temperature=0.0)
        compute_seconds = time.perf_counter() - t0
        compute_num_ctx = llm.last_num_ctx
        is_error_compute = "[System Error:" in response

    is_error = is_error_extract or is_error_compute
    is_abstention = "_abs" in question["question_id"]

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

    try:
        for suffix in ("", ".blobs", ".wal"):
            Path(str(trace_path) + suffix).unlink(missing_ok=True)
    except OSError:
        pass

    return {
        "question_id": question["question_id"],
        "question_type": question["question_type"],
        "report_type": "abstention" if is_abstention else question["question_type"],
        "reference_answer": question["answer"],
        "extracted_facts": extracted_facts,
        "hypothesis": response,
        "judge_raw": judge_response,
        "correct": correct,
        "is_error": is_error,
        "context_chars": len(context_block),
        "context_num_ctx": {"extract": extract_num_ctx, "compute": compute_num_ctx},
        "timing_seconds": {
            "ingest": ingest_seconds,
            "context_build": context_seconds,
            "extract": extract_seconds,
            "compute": compute_seconds,
            "judge": judge_seconds,
        },
    }


def _summarize(results: list[dict]) -> dict:
    by_type: dict[str, list[dict]] = {}
    for r in results:
        by_type.setdefault(r["report_type"], []).append(r)

    def acc(rr):
        scored = [r for r in rr if not r["is_error"]]
        return sum(1 for r in scored if r["correct"]) / len(scored) if scored else 0.0

    return {
        "n": len(results),
        "n_errors": sum(1 for r in results if r["is_error"]),
        "accuracy": acc(results),
        "per_question_type": {t: {"n": len(rr), "accuracy": acc(rr)} for t, rr in sorted(by_type.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num-questions", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--question-types", nargs="+", default=None,
        help="Restrict to these question_type values (e.g. multi-session temporal-reasoning). "
             "Default: run the full stratified sample (all types), same as expansion_unit_experiment.py, "
             "for a like-for-like comparison against full_session's recorded numbers.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--tmp-dir", default=None)
    args = parser.parse_args()

    import os
    import tempfile
    if args.model:
        os.environ["AEON_LLM_MODEL"] = args.model

    with open(args.dataset) as f:
        all_questions = json.load(f)
    sample = _stratified_sample(all_questions, args.num_questions, args.seed)
    if args.question_types:
        sample = [q for q in sample if q["question_type"] in args.question_types]
    print(f"Sampled {len(sample)} questions (seed={args.seed}) from {len(all_questions)} total")

    encoder = _get_encoder()
    llm = OllamaProvider()
    print(f"Using Ollama model: {llm.model}")
    print("Warming up model (forces full load before timing/scoring begins)...")
    warm = _generate_with_retry(llm, "Say OK.", retries=5)
    print(f"  warm-up response: {warm[:80]!r}")

    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else Path(tempfile.mkdtemp(prefix="extract_then_compute_"))
    tmp_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for i, q in enumerate(sample):
        r = _run_one(q, encoder, llm, tmp_dir)
        all_results.append(r)
        status = "TRANSPORT_ERR" if r["is_error"] else ("OK" if r["correct"] else "WRONG")
        print(
            f"[{i+1}/{len(sample)}] {q['question_id']} ({q['question_type']}) -> {status} "
            f"({r['context_chars']} chars)",
            flush=True,
        )

    summary = _summarize(all_results)
    out = {
        "model": llm.model,
        "seed": args.seed,
        "num_questions": len(sample),
        "base_top_k": BASE_TOP_K,
        "max_sessions": MAX_SESSIONS,
        "summary": summary,
        "results": all_results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== Extract-then-compute summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
