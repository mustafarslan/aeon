#!/usr/bin/env python3
"""
Answer-in-context diagnostic (v4-plan.md Stage 6, advisor-prompted).

`gold_session_hit_at_k` (run_benchmark.py) is a SESSION-level recall metric:
it only checks whether any retrieved event's session_id is one of the
question's answer_session_ids. A session has ~10 turns; top_k=10 retrieves
individual EVENTS, not whole sessions -- so "gold session hit" can be true
while the one specific turn that actually states the answer never entered
the LLM's prompt. This script measures the finer-grained thing directly:
does the reference answer string literally appear in the retrieved text
Aeon actually handed the LLM?

Only meaningful for question types where `answer` is a literal factual
string expected to appear near-verbatim in the source turn --
single-session-user/assistant, multi-session, temporal-reasoning,
knowledge-update. Skipped for single-session-preference (answer is a
rubric, not a fact) and abstention (answer is an explanation sentence).

No LLM calls -- same cost profile as --retrieval-only.

Usage:
    python scripts/longmemeval/answer_in_context.py \\
        --dataset /path/to/longmemeval_s_cleaned.json \\
        --num-questions 50 --seed 42 --top-k 10
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_benchmark import _stratified_sample, _get_encoder, _ingest_haystack  # noqa: E402

from aeon_py.session_expansion import build_expanded_context  # noqa: E402
from aeon_py.trace import TraceGraph  # noqa: E402

APPLICABLE_TYPES = {
    "single-session-user", "single-session-assistant", "multi-session",
    "temporal-reasoning", "knowledge-update",
}


def _normalize(s) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num-questions", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--tmp-dir", default=None)
    parser.add_argument(
        "--unit", default=None,
        help="If set (e.g. 'full_session', 'window_10'), checks answer-in-context "
             "against the ACTUAL context that Stage 7's expansion_unit_experiment.py "
             "assembles for this unit (via session_expansion.py's "
             "build_expanded_context(), same base_top_k/max_sessions as that "
             "experiment) instead of plain semantic_search(top_k). Discriminates "
             "'the fact isn't retrieved at all' from 'the fact IS in the assembled "
             "context and the model/assembly still can't use it' -- advisor-prompted, "
             "v4-plan.md Stage 7, after full_session/window_5 (45.5%% multi-session) "
             "and window_10 (36.4%%) diverged 9 points on near-identical content, "
             "which pointed at context assembly/ordering, not raw retrieval, as a "
             "candidate explanation for the residual gap below the oracle ceiling.",
    )
    parser.add_argument("--max-sessions", type=int, default=10,
                         help="Only used with --unit; must match the experiment being diagnosed.")
    parser.add_argument("--base-top-k", type=int, default=30,
                         help="Only used with --unit; must match the experiment being diagnosed.")
    args = parser.parse_args()

    import tempfile
    with open(args.dataset) as f:
        all_questions = json.load(f)

    sample = _stratified_sample(all_questions, args.num_questions, args.seed)
    encoder = _get_encoder()
    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else Path(tempfile.mkdtemp(prefix="longmemeval_aic_"))
    tmp_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, q in enumerate(sample):
        applicable = q["question_type"] in APPLICABLE_TYPES
        trace_path = tmp_dir / f"{q['question_id']}.trace"
        for suffix in ("", ".blobs", ".wal"):
            Path(str(trace_path) + suffix).unlink(missing_ok=True)
        trace = TraceGraph(path=str(trace_path))
        _ingest_haystack(trace, encoder, q)

        q_vec = np.asarray(encoder.encode(q["question"]), dtype=np.float32).tolist()
        if args.unit:
            retrieved = build_expanded_context(
                trace, q_vec, args.unit, base_top_k=args.base_top_k, max_sessions=args.max_sessions,
            )
        else:
            retrieved = trace.semantic_search(q_vec, top_k=args.top_k)
        gold_sessions = set(q["answer_session_ids"])
        session_hit = bool(gold_sessions & {ev["session_id"] for ev in retrieved})

        answer_in_context = None
        if applicable:
            answer_norm = _normalize(q["answer"])
            blob_norm = _normalize(" ".join(ev["text"] for ev in retrieved))
            answer_in_context = bool(answer_norm) and answer_norm in blob_norm

        results.append({
            "question_id": q["question_id"],
            "question_type": q["question_type"],
            "applicable": applicable,
            "gold_session_hit_at_k": session_hit,
            "answer_in_context": answer_in_context,
        })
        print(
            f"[{i+1}/{len(sample)}] {q['question_id']} ({q['question_type']}) "
            f"session_hit={session_hit} answer_in_context={answer_in_context}",
            flush=True,
        )

        try:
            trace_path.unlink(missing_ok=True)
        except OSError:
            pass

    applicable_results = [r for r in results if r["applicable"]]
    n = len(applicable_results)
    session_hit_rate = sum(1 for r in applicable_results if r["gold_session_hit_at_k"]) / n
    answer_in_context_rate = sum(1 for r in applicable_results if r["answer_in_context"]) / n

    by_type = Counter()
    by_type_hit = Counter()
    by_type_aic = Counter()
    for r in applicable_results:
        t = r["question_type"]
        by_type[t] += 1
        by_type_hit[t] += r["gold_session_hit_at_k"]
        by_type_aic[t] += r["answer_in_context"]

    print("\n=== Answer-in-context diagnostic ===")
    print(f"Applicable questions: {n}/{len(results)}")
    print(f"Session-level recall@{args.top_k} (applicable subset): {session_hit_rate*100:.1f}%")
    print(f"Answer-in-context rate (applicable subset):            {answer_in_context_rate*100:.1f}%")
    print("\nBy type:")
    for t in sorted(by_type):
        print(
            f"  {t:<28} n={by_type[t]:<3} session_hit={by_type_hit[t]/by_type[t]*100:5.1f}%  "
            f"answer_in_context={by_type_aic[t]/by_type[t]*100:5.1f}%"
        )

    out = {
        "unit": args.unit,
        "top_k": args.top_k if not args.unit else None,
        "base_top_k": args.base_top_k if args.unit else None,
        "max_sessions": args.max_sessions if args.unit else None,
        "n_applicable": n,
        "session_hit_rate": session_hit_rate,
        "answer_in_context_rate": answer_in_context_rate,
        "by_type": {
            t: {
                "n": by_type[t],
                "session_hit_rate": by_type_hit[t] / by_type[t],
                "answer_in_context_rate": by_type_aic[t] / by_type[t],
            }
            for t in sorted(by_type)
        },
        "results": results,
    }
    out_name = f"answer_in_context_{args.unit}.json" if args.unit else "answer_in_context.json"
    out_path = Path(f"reproducibility_benchmarks/longmemeval/{out_name}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
