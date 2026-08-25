#!/usr/bin/env python3
"""
Offline COMPUTE-stage probe (v4-plan.md, post-v3-revert diagnostic).

v3 relaxed the COMPUTE *user* prompt to stop refusing direct-lookup and
recommendation answers, and it had EXACTLY ZERO effect on the 5 diagnosed
single-session-user losses (word-for-word identical "not mentioned" hedges).
Reading the harness code (not just results) found a candidate cause never
touched by v2 or v3: every LLM call in this benchmark -- extract, compute,
AND the single-shot baseline -- reuses the SAME `SYSTEM_PROMPT`
(run_benchmark.py), which is framed for the single-shot arm ("answering a
question using ONLY the retrieved memory snippets below... If the snippets
don't contain enough information to answer, say so plainly instead of
guessing"). For COMPUTE, the input is no longer "retrieved memory snippets"
-- it's `extracted_facts`, and the compute user-prompt says so. A system-
level "say so plainly instead of guessing" instruction plausibly dominates
a user-turn relaxation, which would explain v3's zero-effect result.

This script re-runs ONLY the compute step (no retrieval, no extraction --
reuses the extracted_facts already on disk from the n=500 v1 run) across a
small grid, to see whether the system prompt is actually the blocking
factor before proposing anything at n=500:

    {v1 system, step-appropriate system, empty system} x {v1 compute, v3 compute}

on 8 known losses (5 single-session-user, 3 single-session-preference) plus
5 correct-abstention canaries (regression risk: relaxing "say so plainly"
could cost ETC's 96.7% abstention rate). The (v1 system, v1 compute) cell is
already on disk (all 8 losses, 8/8 wrong) and is not re-run.

Usage:
    python scripts/longmemeval/system_prompt_probe.py \\
        --dataset /path/to/longmemeval_s_cleaned.json \\
        --results reproducibility_benchmarks/longmemeval/extract_then_compute_n500_results.json \\
        --model gemma4:31b-cloud \\
        --out reproducibility_benchmarks/longmemeval/system_prompt_probe_results.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_prompts import get_anscheck_prompt  # noqa: E402
from run_benchmark import SYSTEM_PROMPT, _generate_with_retry  # noqa: E402

from aeon_py.llm import OllamaProvider  # noqa: E402

LOSS_IDS = [
    "ec81a493", "311778f1", "c14c00dd", "8a137a7f", "b86304ba",  # single-session-user
    "06878be2", "06f04340", "a89d7624",  # single-session-preference
]

ABSTENTION_CANARY_IDS = [
    "f685340e_abs", "88432d0a_abs", "80ec1f4f_abs", "2698e78f_abs", "09ba9854_abs",
]

COMPUTE_V1 = (
    "You previously extracted these facts from retrieved memories:\n"
    "{extracted_facts}\n\n"
    "Question: {question}\n\n"
    "Using ONLY the facts above, determine the answer. If the question "
    "requires combining multiple facts (a sum, a count, a date difference), "
    "show the calculation briefly. Give your final answer on its own line, "
    "prefixed exactly with 'Answer:'.\n\nAnswer:"
)

COMPUTE_V3 = (
    "You previously extracted these facts from retrieved memories:\n"
    "{extracted_facts}\n\n"
    "Question: {question}\n\n"
    "Using ONLY the facts above, determine the answer.\n"
    "- If a fact directly states or clearly implies the information asked "
    "for, that fact IS the answer -- do not say it is 'not mentioned' or "
    "'not specified' just because the fact is worded differently than the "
    "question or omits a category label the question happens to use.\n"
    "- If the question asks for a recommendation or suggestion, synthesize "
    "one directly from the facts above -- do not refuse just because no "
    "single fact is itself phrased as a recommendation.\n"
    "- If the question requires combining multiple facts (a sum, a count, "
    "a date difference), show the calculation briefly.\n"
    "- Only say the facts are insufficient if, after checking carefully, "
    "the specific information asked for is genuinely absent above.\n"
    "Give your final answer on its own line, prefixed exactly with "
    "'Answer:'.\n\nAnswer:"
)

SYSTEM_STEP_APPROPRIATE = (
    "You are a careful assistant answering a question using facts "
    "previously extracted from memory. If the facts genuinely do not "
    "contain the information asked for, say so plainly instead of "
    "guessing. Answer as concisely as possible -- a short phrase or "
    "sentence, not a paragraph."
)

SYSTEM_EMPTY = ""

SYSTEM_CONDITIONS = {
    "v1_system": SYSTEM_PROMPT,
    "step_appropriate_system": SYSTEM_STEP_APPROPRIATE,
    "empty_system": SYSTEM_EMPTY,
}

COMPUTE_CONDITIONS = {
    "v1_compute": COMPUTE_V1,
    "v3_compute": COMPUTE_V3,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--model", default="gemma4:31b-cloud")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.environ["AEON_LLM_MODEL"] = args.model
    llm = OllamaProvider()

    with open(args.dataset) as f:
        dataset = json.load(f)
    question_by_id = {q["question_id"]: q for q in dataset}

    with open(args.results) as f:
        etc_results = json.load(f)["results"]
    result_by_id = {r["question_id"]: r for r in etc_results}

    target_ids = LOSS_IDS + ABSTENTION_CANARY_IDS
    print(f"Probing {len(target_ids)} cases x {len(SYSTEM_CONDITIONS)} system x "
          f"{len(COMPUTE_CONDITIONS)} compute conditions "
          f"({len(target_ids) * len(SYSTEM_CONDITIONS) * len(COMPUTE_CONDITIONS)} calls, "
          f"skipping the known v1_system x v1_compute cell already on disk)...",
          file=sys.stderr)

    probe_results = []
    for qid in target_ids:
        base_id = qid.replace("_abs", "")
        r = result_by_id.get(qid) or result_by_id.get(base_id)
        q = question_by_id.get(qid) or question_by_id.get(base_id)
        if r is None or q is None:
            print(f"WARNING: {qid} not found in results/dataset, skipping", file=sys.stderr)
            continue

        extracted_facts = r["extracted_facts"]
        question_text = q["question"]
        reference_answer = q["answer"]
        question_type = r["question_type"]
        is_abstention = "_abs" in qid

        for sys_name, sys_prompt in SYSTEM_CONDITIONS.items():
            for comp_name, comp_template in COMPUTE_CONDITIONS.items():
                if sys_name == "v1_system" and comp_name == "v1_compute":
                    continue  # already on disk -- known result

                compute_prompt = comp_template.format(
                    extracted_facts=extracted_facts, question=question_text,
                )
                hypothesis = _generate_with_retry(
                    llm, compute_prompt, system_prompt=sys_prompt, temperature=0.0,
                )

                judge_prompt = get_anscheck_prompt(
                    question_type, question_text, reference_answer, hypothesis,
                    abstention=is_abstention,
                )
                judge_raw = _generate_with_retry(
                    llm, judge_prompt, system_prompt="", temperature=0.0,
                )
                correct = judge_raw.strip().lower().startswith("yes")

                row = {
                    "question_id": qid,
                    "question_type": question_type,
                    "is_abstention": is_abstention,
                    "system_condition": sys_name,
                    "compute_condition": comp_name,
                    "reference_answer": reference_answer,
                    "hypothesis": hypothesis,
                    "judge_raw": judge_raw,
                    "correct": correct,
                }
                probe_results.append(row)
                print(
                    f"{qid} [{sys_name}/{comp_name}] correct={correct}",
                    file=sys.stderr,
                )

    known_cell = []
    for qid in target_ids:
        base_id = qid.replace("_abs", "")
        r = result_by_id.get(qid) or result_by_id.get(base_id)
        if r is None:
            continue
        known_cell.append({
            "question_id": qid,
            "question_type": r["question_type"],
            "is_abstention": "_abs" in qid,
            "system_condition": "v1_system",
            "compute_condition": "v1_compute",
            "reference_answer": question_by_id[base_id]["answer"],
            "hypothesis": r["hypothesis"],
            "judge_raw": r["judge_raw"],
            "correct": r["correct"],
        })

    all_results = known_cell + probe_results

    summary = {}
    for sys_name in SYSTEM_CONDITIONS:
        for comp_name in COMPUTE_CONDITIONS:
            cell = [
                row for row in all_results
                if row["system_condition"] == sys_name and row["compute_condition"] == comp_name
            ]
            losses = [row for row in cell if not row["is_abstention"]]
            canaries = [row for row in cell if row["is_abstention"]]
            summary[f"{sys_name}/{comp_name}"] = {
                "losses_flipped": sum(1 for row in losses if row["correct"]),
                "losses_total": len(losses),
                "abstention_canaries_held": sum(1 for row in canaries if row["correct"]),
                "abstention_canaries_total": len(canaries),
            }

    out = {"model": args.model, "summary": summary, "results": all_results}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
