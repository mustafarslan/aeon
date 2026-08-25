#!/usr/bin/env python3
"""
Offline query router evaluation for the extract-then-compute rollout decision
(v4-plan.md, post n=500 paired re-run).

The n=500 paired run showed extract-then-compute (ETC) is a confirmed win on
multi-session/temporal-reasoning, a wash on knowledge-update, and a real
one-directional regression on single-session-user/preference (ETC's compute
step hedges on direct-lookup/recommendation questions it wasn't built for).
The user picked "build a query classifier to route" over always-on or
do-nothing. A production system only ever has the question TEXT at routing
time -- no `question_type` label -- so the classifier has to predict routing
from the question alone.

Critically, this entire evaluation runs with ZERO new LLM calls: both arms'
per-question correctness for all 500 questions is already on disk
(`full_session_n500_results.json`, `extract_then_compute_n500_results.json`).
Routed accuracy for any routing function is just "look up the stored outcome
from whichever arm that function picks for each question" -- so the
classifier-quality question and the accuracy-ceiling question both reduce to
lookups over stored data plus one local (non-LLM) embedding pass.

Two numbers this script produces, in order:

1. **Type-based oracle ceiling**: route {multi-session, temporal-reasoning}
   to ETC, everything else to single-shot (knowledge-update goes to
   single-shot too -- it's an accuracy wash between arms, so route it by
   latency, not accuracy). This is an upper bound assuming a hypothetical
   classifier with perfect access to the true label -- sets the honest
   ceiling before judging any real classifier's quality.

2. **Real classifier, stratified 5-fold CV, out-of-fold routed accuracy**:
   input is the question's mpnet embedding (already computed once per query
   in the real pipeline for retrieval -- so a classifier over it costs zero
   marginal LLM/embedding calls at inference time), target is the binary
   {ETC, single-shot} route implied by (1)'s rule. Routed accuracy is
   computed ONLY from out-of-fold predictions -- an in-sample number would
   repeat the exact overfitting mistake flagged earlier this stage (see the
   knowledge-update prompt-tuning revert). Per-type misroute rates are
   reported too, since a misrouted single-session-user question costs the
   known regression specifically.

Caveat that stays a caveat, not a result: a classifier trained on
LongMemEval's formulaic question phrasing will likely score very well here.
That is not evidence it generalizes to less formulaic production queries --
keep the model simple for exactly this reason, and treat this number as a
lower bound on the work needed, not a final validation.

Usage:
    python scripts/longmemeval/router_experiment.py \\
        --dataset /path/to/longmemeval_s_cleaned.json \\
        --baseline reproducibility_benchmarks/longmemeval/full_session_n500_results.json \\
        --etc reproducibility_benchmarks/longmemeval/extract_then_compute_n500_results.json \\
        --out reproducibility_benchmarks/longmemeval/router_results.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_benchmark import _get_encoder  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

# Types routed to extract-then-compute under the type-based rule. Everything
# else (including knowledge-update, an accuracy wash) routes to single-shot.
ETC_TYPES = {"multi-session", "temporal-reasoning"}


def _load_results(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    return {r["question_id"]: r for r in data["results"]}


def _oracle_ceiling(question_ids, types_by_id, baseline_by_id, etc_by_id) -> dict:
    correct = 0
    per_type = {}
    for qid in question_ids:
        t = types_by_id[qid]
        route = "etc" if t in ETC_TYPES else "single_shot"
        row = (etc_by_id if route == "etc" else baseline_by_id)[qid]
        is_correct = bool(row["correct"])
        correct += is_correct
        bucket = per_type.setdefault(t, {"n": 0, "correct": 0})
        bucket["n"] += 1
        bucket["correct"] += is_correct
    return {
        "accuracy": correct / len(question_ids),
        "per_type": {t: b["correct"] / b["n"] for t, b in sorted(per_type.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--etc", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.dataset) as f:
        all_questions = json.load(f)
    q_by_id = {q["question_id"]: q for q in all_questions}

    baseline_by_id = _load_results(args.baseline)
    etc_by_id = _load_results(args.etc)
    question_ids = sorted(set(baseline_by_id) & set(etc_by_id) & set(q_by_id))
    print(f"{len(question_ids)} questions common to dataset + both result files")

    types_by_id = {qid: q_by_id[qid]["question_type"] for qid in question_ids}

    # --- 1. Type-based oracle ceiling (no classifier, just the routing rule) ---
    ceiling = _oracle_ceiling(question_ids, types_by_id, baseline_by_id, etc_by_id)
    print("\n=== Type-based oracle ceiling (hypothetical perfect-label router) ===")
    print(json.dumps(ceiling, indent=2))

    always_etc_acc = sum(etc_by_id[qid]["correct"] for qid in question_ids) / len(question_ids)
    always_base_acc = sum(baseline_by_id[qid]["correct"] for qid in question_ids) / len(question_ids)
    print(f"\nFor reference: always single-shot = {always_base_acc:.3f}, always ETC = {always_etc_acc:.3f}")

    # --- 2. Real classifier over question embeddings, stratified CV ---
    print("\nEmbedding all questions (local mpnet encoder, no LLM calls)...")
    encoder = _get_encoder()
    texts = [q_by_id[qid]["question"] for qid in question_ids]
    X = np.asarray(encoder.encode(texts, show_progress_bar=False), dtype=np.float64)
    y = np.array([1 if types_by_id[qid] in ETC_TYPES else 0 for qid in question_ids])  # 1 = route to ETC

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof_pred = np.full(len(question_ids), -1, dtype=int)
    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(X[train_idx], y[train_idx])
        oof_pred[test_idx] = clf.predict(X[test_idx])
    assert (oof_pred >= 0).all()

    routed_correct = 0
    per_type_routed = {}
    per_type_misroute = {}
    for i, qid in enumerate(question_ids):
        t = types_by_id[qid]
        predicted_route = "etc" if oof_pred[i] == 1 else "single_shot"
        true_route = "etc" if y[i] == 1 else "single_shot"
        row = (etc_by_id if predicted_route == "etc" else baseline_by_id)[qid]
        is_correct = bool(row["correct"])
        routed_correct += is_correct

        b = per_type_routed.setdefault(t, {"n": 0, "correct": 0})
        b["n"] += 1
        b["correct"] += is_correct
        m = per_type_misroute.setdefault(t, {"n": 0, "misrouted": 0})
        m["n"] += 1
        m["misrouted"] += int(predicted_route != true_route)

    classifier_result = {
        "folds": args.folds,
        "n": len(question_ids),
        "routed_accuracy": routed_correct / len(question_ids),
        "per_type_routed_accuracy": {t: b["correct"] / b["n"] for t, b in sorted(per_type_routed.items())},
        "per_type_misroute_rate": {t: m["misrouted"] / m["n"] for t, m in sorted(per_type_misroute.items())},
    }
    print("\n=== Real classifier, out-of-fold routed accuracy (5-fold CV, question-embedding input) ===")
    print(json.dumps(classifier_result, indent=2))

    out = {
        "always_single_shot_accuracy": always_base_acc,
        "always_etc_accuracy": always_etc_acc,
        "type_based_oracle_ceiling": ceiling,
        "classifier_cv_result": classifier_result,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
