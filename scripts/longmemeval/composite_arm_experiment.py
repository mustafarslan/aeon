#!/usr/bin/env python3
"""Composite arm -- the semantic layer measured as a system.

Query-blind consolidation at write time, then ONE answer call over
`compose()`'s context: all records (category-ordered) + episodic turns rehydrated from
those records' own provenance.

This is the first arm that can show a REGRESSION. Every consolidation result so far was
measured on questions selected for being wrong, so it could only improve; this runs a
stratified sample alongside the hard cohort, so collateral on ordinary questions is visible.

Uses `aeon_py.parallel` for the extraction pass (measured ~2x at 4 workers on this workload)
and caches records to disk, so later read-path variations on the same sample cost minutes
rather than re-paying the extraction.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from consolidation_probe import (  # noqa: E402
    BUCKET_BLOCK, CONSOLIDATE_PROMPT, CONSOLIDATE_SYSTEM, EXTRACT_PROMPT_V2, EXTRACT_SYSTEM,
    session_text,
)
from judge_prompts import get_anscheck_prompt  # noqa: E402
from precision_selector import build_index, select  # noqa: E402
from splits import load_miss_ids  # noqa: E402
from splits import select as select_split  # noqa: E402
from run_benchmark import (  # noqa: E402
    _generate_with_retry, _get_encoder, _stratified_sample, format_question_with_date,
)

import numpy as np  # noqa: E402
from aeon_py.compose import compose  # noqa: E402
from aeon_py.consolidation import parse_records  # noqa: E402
from aeon_py.parallel import ThreadLocalResource, parallel_map  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="gemma4:31b-cloud")
    ap.add_argument("--out", required=True)
    ap.add_argument("--records-cache", required=True)
    ap.add_argument("--num-questions", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--extra-ids", nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--episodic-budget", type=int, default=6000)
    ap.add_argument("--consolidate", action="store_true")
    # Held-out gating (v4-plan.md). "all" is the identity, so every invocation that
    # predates this flag reproduces exactly.
    ap.add_argument("--split", choices=("dev", "heldout", "all"), default="all")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--attribution",
                    default="reproducibility_benchmarks/longmemeval/answer_turn_attribution.json")
    ap.add_argument("--counting-hint", choices=("current", "none", "reconcile"),
                    default="current",
                    help="current: COUNT the matching records (every measured number to "
                         "date). reconcile: the reconciliation-aware variant under test.")
    ap.add_argument("--system-prompt", default="",
                    help="System prompt for the ANSWER call. Every composite number to date "
                         "was measured with this empty -- compose.COMPOSE_SYSTEM was never sent.")
    args = ap.parse_args()

    import os
    os.environ["AEON_LLM_MODEL"] = args.model
    from aeon_py.llm import OllamaProvider
    llm = OllamaProvider()
    providers = ThreadLocalResource(OllamaProvider)

    ds = json.load(open(args.dataset))
    by_id = {q["question_id"]: q for q in ds}
    sample = _stratified_sample(ds, args.num_questions, args.seed)
    if args.extra_ids:
        have = {q["question_id"] for q in sample}
        sample += [by_id[q] for q in args.extra_ids if q not in have]
    if args.split != "all":
        # Split the SAMPLE, not the dataset, so --split composes with --num-questions.
        misses = load_miss_ids(args.attribution) if Path(args.attribution).exists() else None
        sample = select_split(sample, args.split, args.split_seed, misses)

    cache_path = Path(args.records_cache)
    cache = json.load(open(cache_path)) if cache_path.exists() else {}
    print(f"composite arm: {len(sample)} questions (split={args.split}) | "
          f"cached records for {len(cache)} | workers={args.workers} | "
          f"system_prompt={'set' if args.system_prompt else 'empty'} | "
          f"counting_hint={args.counting_hint}", flush=True)
    enc = _get_encoder()
    print(f"warm-up: {_generate_with_retry(llm, 'Say OK.', retries=5)[:20]!r}", flush=True)

    results = []
    for qi, q in enumerate(sample, 1):
        qid = q["question_id"]

        if qid in cache:
            record_text, t_extract = cache[qid], 0.0
        else:
            t0 = time.perf_counter()

            def _extract(pair):
                date, turns = pair
                return _generate_with_retry(
                    providers.get(),
                    EXTRACT_PROMPT_V2.format(date=date, session=session_text(turns),
                                             buckets=BUCKET_BLOCK),
                    system_prompt=EXTRACT_SYSTEM, temperature=0.0)

            outs = parallel_map(
                _extract, list(zip(q["haystack_dates"], q["haystack_sessions"])),
                max_workers=args.workers,
                on_error=lambda item, exc: "",          # one bad session must not lose the rest
            )
            lines = [ln.strip() for out in outs if out and "[System Error:" not in out
                     for ln in out.splitlines()
                     if ln.strip() and ln.strip() != "(none)"
                     and not ln.strip().lower().startswith("records:")]
            record_text = "\n".join(lines)
            if args.consolidate and record_text:
                merged = _generate_with_retry(
                    llm, CONSOLIDATE_PROMPT.format(records=record_text, buckets=BUCKET_BLOCK),
                    system_prompt=CONSOLIDATE_SYSTEM, temperature=0.0)
                # A collapsing merge is a failure, not a compression win.
                if "[System Error:" not in merged and len(merged) > len(record_text) * 0.3:
                    record_text = merged
            t_extract = time.perf_counter() - t0
            cache[qid] = record_text
            json.dump(cache, open(cache_path, "w"))

        records = parse_records(record_text, qid)

        # Episodic component. The probe's independent selection is kept here rather than
        # provenance-rehydration, because these records were extracted without turn citations
        # (the cached corpus predates them), so their provenance is session-level only.
        idx = build_index(q, enc, max_chunk_chars=400)
        qv = np.asarray(enc.encode(q["question"]), dtype=np.float32)
        epi = select(q, idx, qv, design="B", budget_chars=args.episodic_budget,
                     stitch=1, stitch_mode="post")

        prompt = compose(records, epi["context"].splitlines(),
                         format_question_with_date(q),
                         counting_hint=args.counting_hint != "none",
                         reconcile_hint=args.counting_hint == "reconcile")
        t0 = time.perf_counter()
        hyp = _generate_with_retry(llm, prompt, system_prompt=args.system_prompt,
                                   temperature=0.0)
        gen = time.perf_counter() - t0
        err = "[System Error:" in hyp
        correct, judge = False, ""
        if not err:
            jp = get_anscheck_prompt(q["question_type"], q["question"], q["answer"], hyp,
                                     abstention="_abs" in qid)
            judge = _generate_with_retry(llm, jp, system_prompt="", temperature=0.0)
            correct = judge.strip().lower().startswith("yes")

        results.append({
            "question_id": qid, "question_type": q["question_type"],
            "report_type": "abstention" if "_abs" in qid else q["question_type"],
            "reference_answer": q["answer"], "hypothesis": hyp, "judge_raw": judge,
            "correct": correct, "is_error": err,
            "n_records": len(records), "record_chars": len(record_text),
            "episodic_chars": epi["chars"], "prompt_chars": len(prompt),
            "timing_seconds": {"extract": t_extract, "generation": gen},
        })
        print(f"[{qi}/{len(sample)}] {qid} ({q['question_type']}) -> "
              f"{'ERR' if err else ('OK' if correct else 'WRONG')} "
              f"({len(records)} records, {len(prompt)} chars)", flush=True)

    summary = {
        "n": len(results), "n_errors": sum(r["is_error"] for r in results),
        "correct": sum(r["correct"] for r in results),
        "accuracy": sum(r["correct"] for r in results) / len(results),
        "median_records": statistics.median([r["n_records"] for r in results]),
        "median_prompt_chars": statistics.median([r["prompt_chars"] for r in results]),
        "median_generation_seconds": statistics.median(
            [r["timing_seconds"]["generation"] for r in results]),
    }
    json.dump({"model": args.model, "mode": "composite (records + episodic, 1 call)",
               "split": args.split, "split_seed": args.split_seed,
               "counting_hint": args.counting_hint,
               "system_prompt": args.system_prompt,
               "summary": summary, "results": results}, open(args.out, "w"), indent=2)
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
