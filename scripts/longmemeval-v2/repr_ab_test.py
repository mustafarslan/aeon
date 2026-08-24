#!/usr/bin/env python3
"""
Compact-vs-tree state-representation A/B for LongMemEval-V2 (v4-plan.md
Stage 6 follow-on, advisor-directed phase 1 of 2).

Answers the open question from smoke_test.py's docstring: does embedding
each trajectory state's raw `accessibility_tree` (truncated by mpnet's
384-token window) retrieve worse than embedding a compact "Goal/URL/
Thought/Action" text -- and if so, by how much -- BEFORE committing to a
multi-hour full run with one representation baked in?

Scoped to the `norm_phrase_set_match`/`norm_phrase_set_match_ordered`
question types ONLY (225/451 non-image questions) -- these score with pure
regex (qa_eval_metrics.py, vendored verbatim), so this phase needs
generation calls but ZERO judge calls, roughly halving the LLM cost of a
naive "run everything twice" plan. `mc_choice_match`/llm-judge question
types are excluded here deliberately: their answers (a letter, an
explanation sentence) aren't the kind of thing retrieval quality moves in
a way this quick gate is trying to isolate -- see full_run.py for where
every question type gets scored properly, once this has picked a
representation.

An earlier draft of this script tried to reuse run_benchmark.py's
"answer-in-context substring" diagnostic here too. Advisor review caught
that it doesn't transfer: LongMemEval-S answers are distinctive prose
phrases ("Business Administration"); LongMemEval-V2 answers are short UI
strings ("300", "Reports;Problems") drawn from the same vocabulary as the
haystack's own accessibility trees and nav bars -- a substring check would
false-positive against unrelated numbers/menu items throughout the corpus
and read as ~100% for both representations regardless of real retrieval
quality. Scoring actual generations with the actual deterministic
evaluator is the real signal; there is no LLM-free shortcut here.

A first live run of this script (embedding AND prompting with the same
"tree" text) contaminated 15/22 tree-pass results with Ollama transport
errors -- some trajectory states' raw accessibility_tree runs past 300KB,
and top_k=10 of those in one prompt produced either an outright 400
rejection or a request slow enough to exceed the retry timeout. Advisor
review caught the actual bug: the script conflated two independent
questions -- "does embedding the raw tree rank retrieval better?" vs "does
showing the LLM the raw tree at generation time help?" -- into one axis.
Fixed by decoupling them (see common.py's ingest_domain): this script now
always embeds whichever representation the outer loop is testing, but
ALWAYS prompts with `--prompt-repr` (default "compact"), so both passes
have the same prompt-size profile and the comparison isolates ranking
quality, not context-window survival. The "does the LLM benefit from
seeing the raw tree" question is real but separate, and needs its own
harness with explicit per-state prompt truncation -- not attempted here.

Ingests the FULL 100-trajectory-per-domain "small" haystack (not a
subsample -- this is the real corpus, just a stratified question subset)
under both representations, so results here transfer directly to
full_run.py's choice of representation.

Usage:
    python scripts/longmemeval-v2/repr_ab_test.py \\
        --questions questions.jsonl --haystack lme_v2_small.json \\
        --trajectories trajectories.jsonl \\
        --num-questions-per-domain 20 --model qwen3.8:27b-mlx \\
        --out reproducibility_benchmarks/longmemeval-v2/repr_ab_results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DOMAINS, fresh_trace_path, get_encoder, ingest_domain,
    load_domain_trajectory_ids, load_wanted_trajectories, run_generation,
    score_deterministic,
)

from aeon_py.trace import TraceGraph  # noqa: E402
from aeon_py.llm import OllamaProvider  # noqa: E402

DETERMINISTIC_TYPES = {"norm_phrase_set_match", "norm_phrase_set_match_ordered"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--haystack", required=True)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--num-questions-per-domain", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--prompt-repr", choices=["compact", "tree"], default="compact",
        help=(
            "What gets stored/shown to the LLM once a state is retrieved, "
            "independent of which representation ('compact' or 'tree') was "
            "embedded for ranking. Defaults to 'compact' for BOTH embed "
            "passes so this A/B isolates retrieval-ranking quality without "
            "the raw accessibility_tree's prompt-size blowing past the "
            "model's context window (see common.py's ingest_domain docstring)."
        ),
    )
    parser.add_argument("--model", default=None)
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
    print(f"Model: {llm.model}")

    domain_traj_ids = load_domain_trajectory_ids(all_questions, haystack)
    wanted_ids = set(domain_traj_ids["web"]) | set(domain_traj_ids["enterprise"])
    print(f"Loading {len(wanted_ids)} trajectories (full small-tier haystack, both domains)...")
    t0 = time.perf_counter()
    trajectories_by_id = load_wanted_trajectories(args.trajectories, wanted_ids)
    print(f"  loaded {len(trajectories_by_id)} in {time.perf_counter()-t0:.1f}s")

    # Fixed, deterministic question subset -- SAME questions scored under
    # both representations, so the comparison is paired, not two
    # independent samples.
    eval_questions = []
    for domain in DOMAINS:
        pool = [
            q for q in all_questions
            if q["domain"] == domain and not q.get("image")
            and q["eval_function"].split("|")[0] in DETERMINISTIC_TYPES
        ]
        eval_questions.extend(pool[: args.num_questions_per_domain])
    print(f"Evaluating {len(eval_questions)} questions per representation ({args.num_questions_per_domain}/domain)")

    encoder = get_encoder()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_path.with_suffix(".jsonl")
    all_results = []

    with open(jsonl_path, "w") as jf:
        for repr_mode in ("compact", "tree"):
            traces = {}
            for domain in DOMAINS:
                trajectories = [trajectories_by_id[t] for t in domain_traj_ids[domain] if t in trajectories_by_id]
                trace_path = fresh_trace_path(f"/tmp/lmev2_ab_{repr_mode}_{domain}.trace")
                trace = TraceGraph(path=trace_path)
                print(
                    f"\n=== [embed={repr_mode} prompt={args.prompt_repr}] {domain}: "
                    f"ingesting {len(trajectories)} trajectories ==="
                )
                stats = ingest_domain(trace, encoder, trajectories, repr_mode, args.prompt_repr)
                print(f"  {stats}")
                traces[domain] = trace

            for i, q in enumerate(eval_questions):
                t0 = time.perf_counter()
                r = run_generation(q, traces[q["domain"]], encoder, llm, args.top_k)
                r = score_deterministic(r)
                r["state_repr"] = repr_mode
                all_results.append(r)
                jf.write(json.dumps(r) + "\n")
                jf.flush()
                elapsed = time.perf_counter() - t0
                status = (
                    "TRANSPORT_ERR" if r.get("is_error")
                    else "OK" if r["correct"]
                    else ("ERR" if r["eval_error"] else "WRONG")
                )
                print(
                    f"[{repr_mode}][{i+1}/{len(eval_questions)}] {q['id']} ({q['domain']}, {r['eval_function']}) "
                    f"-> {status} ({elapsed:.1f}s)",
                    flush=True,
                )

    def acc(rows):
        # Transport-error rows excluded from both numerator and denominator
        # -- see common.py's score_deterministic docstring.
        scored = [r for r in rows if not r.get("is_error")]
        return sum(1 for r in scored if r["correct"]) / len(scored) if scored else 0.0

    def n_errors(rows):
        return sum(1 for r in rows if r.get("is_error"))

    summary = {}
    for repr_mode in ("compact", "tree"):
        rows = [r for r in all_results if r["state_repr"] == repr_mode]
        by_domain = {d: acc([r for r in rows if r["domain"] == d]) for d in DOMAINS}
        summary[repr_mode] = {
            "model": llm.model,
            "n": len(rows),
            "n_errors": n_errors(rows),
            "accuracy": acc(rows),
            "by_domain": by_domain,
            "prompt_repr": args.prompt_repr,
        }

    with open(out_path, "w") as f:
        json.dump({"summary": summary, "results": all_results}, f, indent=2)

    print("\n=== Representation A/B summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path} (incremental log at {jsonl_path})")


if __name__ == "__main__":
    main()
