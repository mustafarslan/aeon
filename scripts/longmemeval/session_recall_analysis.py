#!/usr/bin/env python3
"""
Gold-session recall@N analysis (v4-plan.md Stage 7 diagnostic, post-gate STOP).

LLM-FREE: embeds and ranks only, no generation/judge calls, no Ollama
dependency at all -- this is the single measurement advisor identified as
decisive before writing a next-steps plan for the multi-session track.
`full_session` expansion (all 5 arms, task 1) anchors to the top `N`
DISTINCT session_ids among a `semantic_search(top_k=30)` result. Multi-session
accuracy plateaued at 45.5% even at its best (full_session, N=5) against a
90.9% oracle (all gold sessions handed directly) -- this script isolates
whether that's because the gold sessions aren't showing up in the ranked
hits at all, or because they show up but rank beyond N=5, which are two
completely different fixes:

  - golds present in top_k=30 but ranked past N=5 distinct sessions ->
    raising `max_sessions` is a real, trivial, cheap fix.
  - golds not present in top_k=30 hits at all -> no expansion parameter can
    help; Aeon needs a genuinely better multi-fact retrieval capability
    (session-level vectors, query decomposition, etc).

For each question: run semantic_search(top_k=30), take the ranked list of
DISTINCT session_ids (same `distinct_session_ids()` primitive
expansion_unit_experiment.py already uses), and for N=1..10 report whether
ALL of the question's `answer_session_ids` are covered by the top-N distinct
sessions ("all-golds-present@N" -- the metric that predicts multi-session
accuracy, since these questions need every gold session's fact to answer
correctly) and the per-question fraction of golds covered (partial recall).
Breaks down by question_type; multi-session/knowledge-update/
temporal-reasoning are the types with >1 gold session and are the ones this
diagnostic is actually about.

Usage:
    python scripts/longmemeval/session_recall_analysis.py \\
        --dataset /path/to/longmemeval_s_cleaned.json \\
        --num-questions 50 --seed 42 \\
        --out reproducibility_benchmarks/longmemeval/session_recall_results.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_benchmark import _get_encoder, _ingest_haystack, _stratified_sample  # noqa: E402

import numpy as np  # noqa: E402
from aeon_py.session_expansion import distinct_session_ids, find_top_hits  # noqa: E402
from aeon_py.trace import TraceGraph  # noqa: E402

BASE_TOP_K = 30
MAX_N = 10


def _fresh_trace_path(tmp_dir: Path, question_id: str) -> Path:
    trace_path = tmp_dir / f"{question_id}_recall.trace"
    for suffix in ("", ".blobs", ".wal"):
        Path(str(trace_path) + suffix).unlink(missing_ok=True)
    return trace_path


def _analyze_one(question: dict, encoder, tmp_dir: Path) -> dict:
    trace_path = _fresh_trace_path(tmp_dir, question["question_id"])
    trace = TraceGraph(path=str(trace_path))
    _ingest_haystack(trace, encoder, question)

    q_vec = np.asarray(encoder.encode(question["question"]), dtype=np.float32).tolist()
    hits = find_top_hits(trace, q_vec, top_k=BASE_TOP_K)
    ranked_sessions = distinct_session_ids(hits, max_sessions=MAX_N)

    gold_sessions = set(question["answer_session_ids"])
    n_gold = len(gold_sessions)

    per_n = {}
    for n in range(1, MAX_N + 1):
        covered = gold_sessions & set(ranked_sessions[:n])
        per_n[n] = {
            "all_present": len(covered) == n_gold,
            "fraction_present": (len(covered) / n_gold) if n_gold else 1.0,
        }

    # Not present anywhere in the ranked top_k=30 distinct sessions at all
    # (i.e. beyond MAX_N=10 wouldn't have helped either) -- the "no
    # expansion parameter can fix this" signal.
    all_ranked = distinct_session_ids(hits, max_sessions=len(hits))
    covered_in_full_topk = gold_sessions & set(all_ranked)

    try:
        for suffix in ("", ".blobs", ".wal"):
            Path(str(trace_path) + suffix).unlink(missing_ok=True)
    except OSError:
        pass

    return {
        "question_id": question["question_id"],
        "question_type": question["question_type"],
        "report_type": "abstention" if "_abs" in question["question_id"] else question["question_type"],
        "n_gold_sessions": n_gold,
        "n_distinct_sessions_in_top_k": len(all_ranked),
        "fraction_golds_in_top_k30": (len(covered_in_full_topk) / n_gold) if n_gold else 1.0,
        "all_golds_in_top_k30": len(covered_in_full_topk) == n_gold,
        "per_n": per_n,
    }


def _summarize(results: list[dict]) -> dict:
    by_type: dict[str, list[dict]] = {}
    for r in results:
        by_type.setdefault(r["report_type"], []).append(r)

    def agg(rr: list[dict]) -> dict:
        n = len(rr)
        out = {
            "n": n,
            "mean_fraction_golds_in_top_k30": sum(r["fraction_golds_in_top_k30"] for r in rr) / n if n else 0.0,
            "all_golds_in_top_k30_rate": sum(1 for r in rr if r["all_golds_in_top_k30"]) / n if n else 0.0,
            "recall_at_n": {},
        }
        for nn in range(1, MAX_N + 1):
            out["recall_at_n"][nn] = {
                "all_present_rate": sum(1 for r in rr if r["per_n"][nn]["all_present"]) / n if n else 0.0,
                "mean_fraction_present": sum(r["per_n"][nn]["fraction_present"] for r in rr) / n if n else 0.0,
            }
        return out

    return {
        "overall": agg(results),
        "per_question_type": {t: agg(rr) for t, rr in sorted(by_type.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num-questions", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tmp-dir", default=None)
    args = parser.parse_args()

    import tempfile
    with open(args.dataset) as f:
        all_questions = json.load(f)
    sample = _stratified_sample(all_questions, args.num_questions, args.seed)
    print(f"Sampled {len(sample)} questions (seed={args.seed}) from {len(all_questions)} total")

    encoder = _get_encoder()
    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else Path(tempfile.mkdtemp(prefix="session_recall_"))
    tmp_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, q in enumerate(sample):
        r = _analyze_one(q, encoder, tmp_dir)
        results.append(r)
        print(
            f"[{i+1}/{len(sample)}] {q['question_id']} ({q['question_type']}) "
            f"n_gold={r['n_gold_sessions']} all_in_top30={r['all_golds_in_top_k30']} "
            f"frac_in_top30={r['fraction_golds_in_top_k30']:.2f}",
            flush=True,
        )

    summary = _summarize(results)
    out = {"seed": args.seed, "num_questions": len(sample), "base_top_k": BASE_TOP_K, "max_n": MAX_N,
           "summary": summary, "results": results}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
