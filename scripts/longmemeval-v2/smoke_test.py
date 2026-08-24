#!/usr/bin/env python3
"""
LongMemEval-V2 smoke test for Aeon (v4-plan.md Stage 6 follow-on).

LongMemEval-V2 (github.com/xiaowu0162/LongMemEval-V2, arxiv:2605.12493) is a
DIFFERENT benchmark from LongMemEval-S/M (the one run_benchmark.py targets):
it tests whether a memory system helps an LLM agent recall facts about long
histories of WEB/ENTERPRISE-AGENT TRAJECTORIES (browser/ServiceNow action
sequences with accessibility-tree observations and screenshots), not chat
conversations. This script is a from-scratch, minimal harness -- it does
NOT plug into the official repo's `evaluation/harness.py` (Python 3.11,
CUDA-pinned torch, its own Memory ABC/registry) since that stack doesn't
install on this machine (Apple Silicon, no CUDA) and isn't needed for a
correctness-focused harness. Instead (see common.py):

  - `qa_eval_metrics.py` (this directory) is vendored VERBATIM from the
    official repo (pure Python, no torch) -- the four deterministic
    evaluators (norm_phrase_set_match[_ordered], mc_choice_match[_set])
    cover 295/451 questions with zero judge-model noise; the two LLM
    evaluators (llm_abstention_checker, llm_gotchas_checker) are pointed
    at a LOCAL Ollama model via its OpenAI-compatible /v1 endpoint instead
    of the official default (hosted GPT-5.2) -- same "verbatim prompts,
    local judge" pattern as run_benchmark.py's judge_prompts.py, with the
    same caveat: absolute numbers aren't comparable to any published
    leaderboard entry using the real judge.
  - The domain system prompts and the "### Memory context: / ### Question
    to answer:" message format are ported verbatim from
    evaluation/harness.py's build_messages()/DOMAIN_SYSTEM_PROMPTS, so the
    reader is asked for a \\boxed{} answer exactly as the official prompt
    expects it -- required for mc_choice_match to have any chance of
    matching (get this wrong and every multiple-choice question fails for
    a formatting reason, not a memory reason).

KEY STRUCTURAL FACT (different from LongMemEval-S): the haystack is
SHARED across all questions within a domain (100 trajectories per domain
for the "small" tier, per SCHEMA.md) -- so this script ingests ONCE per
domain, not once per question.

KEY OPEN QUESTION this harness exists to answer: each trajectory `state`
carries a large, highly repetitive `accessibility_tree` (raw UI tree dump,
often 5-25KB/state) alongside a short, compact, natural-language `thought`
+ `action` + `url`. `all-mpnet-base-v2` truncates at 384 tokens
(~1500 chars) -- so naive full-tree embedding would encode mostly
boilerplate (nav menus, skip links) shared across every state in the
whole domain, not the task-specific content. `--state-repr` (common.py's
`state_text()`) makes the embedded representation an explicit, switchable
choice: `compact` (Goal/URL/Thought/Action, fits mpnet's window) or `tree`
(raw accessibility_tree, truncated). See repr_ab_test.py for the actual
compact-vs-tree comparison -- this script is a mechanics smoke test only.

THIS SCRIPT is a mechanics check: a small, deterministic subsample of
trajectories and questions per domain, meant to validate the full
pipeline (streaming trajectory load, ingestion, retrieval, \\boxed{}
parsing, both deterministic and local-judge scoring) end to end before
committing to any full-scale run. A low accuracy number here is NOT
meaningful -- with only --num-trajectories out of each domain's real
100-trajectory haystack, most questions' answers are structurally absent
from context (confirmed empirically: 5/6 "wrong" answers in the first
smoke run were the model correctly saying UNKNOWN, not a wrong guess).

Usage:
    python scripts/longmemeval-v2/smoke_test.py \\
        --questions questions.jsonl --haystack lme_v2_small.json \\
        --trajectories trajectories.jsonl \\
        --num-trajectories 10 --num-questions 3 --state-repr compact \\
        --model qwen3.8:27b-mlx \\
        --out reproducibility_benchmarks/longmemeval-v2/smoke_results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DOMAINS, LLM_EVAL_FUNCTIONS, fresh_trace_path, get_encoder,
    ingest_domain, load_domain_trajectory_ids, load_wanted_trajectories,
    run_generation, score_deterministic, score_with_judge,
)

from aeon_py.trace import TraceGraph  # noqa: E402
from aeon_py.llm import OllamaProvider  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--haystack", required=True)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--num-trajectories", type=int, default=10, help="Per domain, deterministic first-N subsample of the 100-trajectory small haystack")
    parser.add_argument("--num-questions", type=int, default=3, help="Per domain, non-image questions only")
    parser.add_argument("--state-repr", choices=["compact", "tree"], default="compact")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--model", default=None)
    parser.add_argument("--judge-base-url", default="http://localhost:11434/v1")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import os
    if args.model:
        os.environ["AEON_LLM_MODEL"] = args.model

    with open(args.questions) as f:
        all_questions = [json.loads(line) for line in f]
    with open(args.haystack) as f:
        haystack = json.load(f)

    llm = OllamaProvider()
    judge_model = args.model or llm.model
    print(f"Model: {llm.model} | judge: {judge_model} @ {args.judge_base_url} | state-repr: {args.state_repr}")

    domain_traj_ids_full = load_domain_trajectory_ids(all_questions, haystack)
    domain_traj_ids = {d: ids[: args.num_trajectories] for d, ids in domain_traj_ids_full.items()}

    wanted_ids = set(domain_traj_ids["web"]) | set(domain_traj_ids["enterprise"])
    print(f"Loading {len(wanted_ids)} trajectories from {args.trajectories} (streaming)...")
    t0 = time.perf_counter()
    trajectories_by_id = load_wanted_trajectories(args.trajectories, wanted_ids)
    print(f"  loaded {len(trajectories_by_id)} in {time.perf_counter()-t0:.1f}s")

    encoder = get_encoder()

    results = []
    for domain in DOMAINS:
        traj_ids = domain_traj_ids[domain]
        trajectories = [trajectories_by_id[t] for t in traj_ids if t in trajectories_by_id]
        trace = TraceGraph(path=fresh_trace_path(f"/tmp/lmev2_smoke_{domain}.trace"))
        print(f"\n=== {domain}: ingesting {len(trajectories)} trajectories ({args.state_repr}) ===")
        stats = ingest_domain(trace, encoder, trajectories, args.state_repr)
        print(f"  {stats}")

        # non-image questions only (this smoke harness has no image/vision path)
        domain_questions = [q for q in all_questions if q["domain"] == domain and not q.get("image")]
        sample = domain_questions[: args.num_questions]

        for q in sample:
            t0 = time.perf_counter()
            r = run_generation(q, trace, encoder, llm, args.top_k)
            if r["eval_function"] in LLM_EVAL_FUNCTIONS:
                r = score_with_judge(r, judge_model, args.judge_base_url)
            else:
                r = score_deterministic(r)
            results.append(r)
            elapsed = time.perf_counter() - t0
            status = "OK" if r["correct"] else ("ERR" if r["eval_error"] else "WRONG")
            print(
                f"[{domain}] {q['id']} ({q['question_type']}, {r['eval_function']}) -> {status} ({elapsed:.1f}s)",
                flush=True,
            )
            if r["eval_error"]:
                print(f"    eval_error: {r['eval_error']}")

    n = len(results)
    n_correct = sum(1 for r in results if r["correct"])
    n_errors = sum(1 for r in results if r["eval_error"])
    summary = {
        "state_repr": args.state_repr,
        "num_trajectories_per_domain": args.num_trajectories,
        "top_k": args.top_k,
        "model": judge_model,
        "num_questions": n,
        "num_eval_errors": n_errors,
        "accuracy": n_correct / n if n else 0.0,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print("\n=== Smoke test summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
