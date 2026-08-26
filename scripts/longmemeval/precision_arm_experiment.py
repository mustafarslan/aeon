#!/usr/bin/env python3
"""TIER 2/3: the real (non-oracle) precision selector, end to end.

Single-shot generation over a sub-turn-selected, budgeted context. This is the
production-shaped counterpart to `oracle_precision_experiment.py`: same one-call shape,
same prompt, same judge, but the context is chosen by `precision_selector.select()` using
only the query embedding -- `has_answer` never enters the selection path.

Reference points (n=500, date fix, same model/judge):
    single-shot @top_k=30   77.6%   100,889 chars   1 call   1.51s   3.85 correct/1k
    ETC        @top_k=30    82.6%   100,889 chars   2 calls  2.46s   4.09 correct/1k
    oracle-precision        83.8%     5,654 chars   1 call   0.51s  74.10 correct/1k  <- ceiling

Tier-1 coverage (free, already run) established that the selector reaches ~79% answer-turn
coverage at ~8.7k chars against production's 100% at ~101k. Whether that coverage loss is
paid back by the smaller context is exactly what this measures -- the oracle arm showed the
two effects run in opposite directions, so it cannot be predicted from coverage alone.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_prompts import get_anscheck_prompt  # noqa: E402
from precision_selector import build_index, select  # noqa: E402
from run_benchmark import (  # noqa: E402
    SYSTEM_PROMPT, _generate_with_retry, _get_encoder, _stratified_sample,
    format_question_with_date,
)

import numpy as np  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="gemma4:31b-cloud")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-questions", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--question-ids", nargs="+", default=None)
    ap.add_argument("--extra-ids", nargs="+", default=None,
                    help="Union these ids into the stratified sample (e.g. the known "
                         "retrieval-miss cohort), so one cheap run measures both "
                         "conversion on hard cases and collateral on normal ones.")
    ap.add_argument("--budget-chars", type=int, default=9000)
    ap.add_argument("--stitch", type=int, default=1)
    ap.add_argument("--stitch-mode", default="post", choices=["post", "inline"])
    ap.add_argument("--design", default="B", choices=["A", "B"])
    ap.add_argument("--chunk-chars", type=int, default=400)
    args = ap.parse_args()

    import os
    os.environ["AEON_LLM_MODEL"] = args.model
    from aeon_py.llm import OllamaProvider
    llm = OllamaProvider()

    ds = json.load(open(args.dataset))
    by_id = {q["question_id"]: q for q in ds}
    if args.question_ids:
        sample = [by_id[q] for q in args.question_ids]
    else:
        sample = _stratified_sample(ds, args.num_questions, args.seed)
        if args.extra_ids:
            have = {q["question_id"] for q in sample}
            sample += [by_id[q] for q in args.extra_ids if q not in have]

    print(f"Precision arm: {len(sample)} questions | design {args.design} "
          f"budget {args.budget_chars} stitch {args.stitch}/{args.stitch_mode}", flush=True)
    enc = _get_encoder()
    warm = _generate_with_retry(llm, "Say OK.", retries=5)
    print(f"  warm-up: {warm[:40]!r}", flush=True)

    results = []
    for i, q in enumerate(sample, 1):
        qid = q["question_id"]
        t0 = time.perf_counter()
        idx = build_index(q, enc, max_chunk_chars=args.chunk_chars)
        index_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        qv = np.asarray(enc.encode(q["question"]), dtype=np.float32)
        sel = select(q, idx, qv, design=args.design, budget_chars=args.budget_chars,
                     stitch=args.stitch, stitch_mode=args.stitch_mode)
        select_s = time.perf_counter() - t0

        prompt = (f"Retrieved memories:\n{sel['context'] or '(nothing retrieved)'}\n\n"
                  f"{format_question_with_date(q)}\n\nAnswer:")
        t0 = time.perf_counter()
        hyp = _generate_with_retry(llm, prompt, system_prompt=SYSTEM_PROMPT, temperature=0.0)
        gen_s = time.perf_counter() - t0
        is_err = "[System Error:" in hyp
        is_abs = "_abs" in qid
        correct, judge_raw = False, ""
        if not is_err:
            jp = get_anscheck_prompt(q["question_type"], q["question"], q["answer"], hyp,
                                     abstention=is_abs)
            judge_raw = _generate_with_retry(llm, jp, system_prompt="", temperature=0.0)
            correct = judge_raw.strip().lower().startswith("yes")

        results.append({
            "question_id": qid, "question_type": q["question_type"],
            "report_type": "abstention" if is_abs else q["question_type"],
            "reference_answer": q["answer"], "hypothesis": hyp, "judge_raw": judge_raw,
            "correct": correct, "is_error": is_err,
            "context_chars": sel["chars"], "n_turns": sel["n_turns"],
            "n_chunks_indexed": sel["n_chunks_indexed"],
            "context_num_ctx": llm.last_num_ctx,
            "timing_seconds": {"index": index_s, "select": select_s, "generation": gen_s},
        })
        print(f"[{i}/{len(sample)}] {qid} ({q['question_type']}) -> "
              f"{'TRANSPORT_ERR' if is_err else ('OK' if correct else 'WRONG')} "
              f"({sel['chars']} chars, {sel['n_turns']} turns)", flush=True)

    acc = sum(r["correct"] for r in results) / len(results)
    summary = {
        "n": len(results), "n_errors": sum(r["is_error"] for r in results), "accuracy": acc,
        "design": args.design, "budget_chars": args.budget_chars,
        "stitch": args.stitch, "stitch_mode": args.stitch_mode,
        "median_context_chars": statistics.median([r["context_chars"] for r in results]),
        "median_generation_seconds": statistics.median(
            [r["timing_seconds"]["generation"] for r in results]),
        "median_select_seconds": statistics.median(
            [r["timing_seconds"]["select"] for r in results]),
        "median_index_seconds": statistics.median(
            [r["timing_seconds"]["index"] for r in results]),
        "per_question_type": {},
    }
    for t in sorted({r["report_type"] for r in results}):
        sub = [r for r in results if r["report_type"] == t]
        summary["per_question_type"][t] = {
            "n": len(sub), "correct": sum(r["correct"] for r in sub),
            "accuracy": sum(r["correct"] for r in sub) / len(sub)}
    json.dump({"model": args.model, "mode": "precision-selector", "summary": summary,
               "results": results}, open(args.out, "w"), indent=2)
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
