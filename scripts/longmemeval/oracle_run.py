#!/usr/bin/env python3
"""
Oracle-context control run (v4-plan.md Stage 6, advisor-prompted).

Isolates the generator+judge model's own reasoning ceiling from Aeon's
retrieval by REMOVING Aeon from the pipeline entirely: instead of
`trace.semantic_search()`, the context handed to the LLM is built directly
from the full text of the question's own `answer_session_ids` sessions --
the exact right content, with no retrieval step of any kind in between.

Purpose: pilot_50_results_v2.json measured 98% gold-session recall@10 but
only 58% QA accuracy. That gap could mean either (a) Aeon retrieves the
right SESSION but not the specific right TURN within it (a real Aeon-side
retrieval-granularity problem), or (b) the generator+judge model
(qwen3.8:27b-mlx) simply cannot reliably answer/score LongMemEval questions
even when handed exactly the right content (a model-capability ceiling
that no amount of Aeon work would move). This script measures (b) directly:
if oracle accuracy lands near 58% too, retrieval is not the bottleneck. If
it's much higher, there's a real gap between "found the session" and
"found the fact" worth closing on Aeon's side (see answer_in_context.py
for the complementary, LLM-free measurement of exactly that gap).

Same 50 questions, same seed, same judge prompts as run_benchmark.py --
only the context-construction step differs.

Usage:
    python scripts/longmemeval/oracle_run.py \\
        --dataset /path/to/longmemeval_s_cleaned.json \\
        --num-questions 50 --seed 42 --model qwen3.8:27b-mlx \\
        --out reproducibility_benchmarks/longmemeval/oracle_results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_prompts import get_anscheck_prompt  # noqa: E402
from run_benchmark import SYSTEM_PROMPT, _generate_with_retry, _stratified_sample  # noqa: E402

from aeon_py.llm import OllamaProvider  # noqa: E402


def _oracle_context(question: dict) -> str:
    """Full text of every turn in the question's own answer_session_ids
    sessions -- no retrieval, no truncation, no ranking. The exact right
    content, formatted identically to run_benchmark.py's stored text
    (`[<date>] <role>: <content>`) so the comparison isolates the model,
    not a formatting difference."""
    gold_ids = set(question["answer_session_ids"])
    lines = []
    for date, sess_id, turns in zip(
        question["haystack_dates"], question["haystack_session_ids"], question["haystack_sessions"]
    ):
        if sess_id not in gold_ids:
            continue
        for t in turns:
            lines.append(f"[{date}] {t['role']}: {t['content']}")
    return "\n".join(f"- {line}" for line in lines) or "(gold session not found in haystack)"


def _run_one(question: dict, llm: OllamaProvider) -> dict:
    context_block = _oracle_context(question)
    user_prompt = f"Retrieved memories:\n{context_block}\n\nQuestion: {question['question']}\n\nAnswer:"

    t0 = time.perf_counter()
    response = _generate_with_retry(llm, user_prompt, system_prompt=SYSTEM_PROMPT)
    generation_seconds = time.perf_counter() - t0
    # `llm.last_num_ctx` is the num_ctx OllamaProvider actually computed and
    # sent for the answer-generation call above (the one carrying the large
    # oracle context) -- recorded per-question so a truncation-affected
    # result is visible in the data itself, not something that has to be
    # independently measured after the fact (v4-plan.md Stage 7: this
    # project already lost real time re-deriving this for a prior run that
    # didn't record it).
    context_num_ctx = llm.last_num_ctx

    is_abstention = "_abs" in question["question_id"]
    t0 = time.perf_counter()
    judge_prompt = get_anscheck_prompt(
        question["question_type"], question["question"], question["answer"], response,
        abstention=is_abstention,
    )
    judge_response = _generate_with_retry(llm, judge_prompt, system_prompt="", temperature=0.0)
    judge_seconds = time.perf_counter() - t0
    correct = "yes" in judge_response.lower()

    return {
        "question_id": question["question_id"],
        "question_type": question["question_type"],
        "report_type": "abstention" if is_abstention else question["question_type"],
        "question": question["question"],
        "reference_answer": question["answer"],
        "hypothesis": response,
        "judge_raw": judge_response,
        "correct": correct,
        "context_num_ctx": context_num_ctx,
        "context_chars": len(context_block),
        "timing_seconds": {"generation": generation_seconds, "judge": judge_seconds},
    }


def _summarize(results: list[dict], model: str, seed: int) -> dict:
    by_type: dict[str, list[dict]] = {}
    for r in results:
        by_type.setdefault(r["report_type"], []).append(r)

    def acc(rows):
        return sum(1 for r in rows if r["correct"]) / len(rows) if rows else 0.0

    num_ctx_values = [r["context_num_ctx"] for r in results]
    return {
        "model": model,
        "seed": seed,
        "mode": "oracle-context (no retrieval)",
        "num_questions": len(results),
        "overall_accuracy": acc(results),
        "per_question_type": {
            t: {"n": len(rows), "accuracy": acc(rows)} for t, rows in sorted(by_type.items())
        },
        "context_num_ctx": {
            "min": min(num_ctx_values) if num_ctx_values else None,
            "mean": sum(num_ctx_values) / len(num_ctx_values) if num_ctx_values else None,
            "max": max(num_ctx_values) if num_ctx_values else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num-questions", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import os
    if args.model:
        os.environ["AEON_LLM_MODEL"] = args.model

    with open(args.dataset) as f:
        all_questions = json.load(f)
    sample = _stratified_sample(all_questions, args.num_questions, args.seed)

    llm = OllamaProvider()
    print(f"Using Ollama model: {llm.model}")
    print("Warming up model...")
    warm = _generate_with_retry(llm, "Say OK.", retries=5)
    print(f"  warm-up response: {warm[:80]!r}")

    results = []
    for i, q in enumerate(sample):
        t0 = time.perf_counter()
        r = _run_one(q, llm)
        results.append(r)
        elapsed = time.perf_counter() - t0
        status = "OK" if r["correct"] else "WRONG"
        print(
            f"[{i+1}/{len(sample)}] {q['question_id']} ({q['question_type']}) -> {status} ({elapsed:.1f}s)",
            flush=True,
        )

    summary = _summarize(results, llm.model, args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print("\n=== Oracle-context summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
