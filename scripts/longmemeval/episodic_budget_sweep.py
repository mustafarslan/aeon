"""Is the episodic gap a BUDGET problem or a SELECTOR problem? They imply different work.

`episodic_coverage_gap.py` established that questions the composite loses to the oracle have
significantly worse answer-turn coverage in their episodic slice (51.6% vs 79.9% on questions
it wins; 20% vs 3% with NO answer turn present, p=0.016). That is correlational and does not
say what to build.

This separates the two candidate causes by re-selecting the SAME index at several budgets:

  BUDGET-BOUND   coverage climbs steeply toward 100% as the budget grows -> the right turns
                 are ranked highly enough, there was simply no room. The fix is a number, and
                 the cost is context (the composite's episodic slice is ~6k of a ~26k prompt).
  SELECTOR-BOUND coverage plateaus well short of 100% even at several times the budget -> the
                 answer-bearing turns are not ranked highly enough for any budget to reach.
                 The fix is ranking (hybrid lexical+semantic, query expansion), which is real
                 work and is already section 11's "highest-value open problem".

The index is built ONCE per question and re-selected at each budget, so this costs one
encoder pass rather than one per budget, and no LLM calls at all. The control group is
carried through for the same reason as before: a coverage curve on the losses alone cannot
be read without knowing the curve on the questions that already succeed.
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from composite_arm_experiment import _get_encoder, build_index, select
from episodic_coverage_gap import answer_turns

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "..", "reproducibility_benchmarks", "longmemeval")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/Volumes/AI_SSD/hf-cache/longmemeval/longmemeval_s_cleaned.json")
    ap.add_argument("--composite", default="composite_glm52_n500.json")
    ap.add_argument("--oracle", default="oracle_glm52_n500.json")
    ap.add_argument("--budgets", default="6000,12000,24000,48000")
    ap.add_argument("--controls", type=int, default=30)
    args = ap.parse_args()
    budgets = [int(b) for b in args.budgets.split(",")]

    C = {r["question_id"]: r for r in json.load(open(os.path.join(RESULTS, args.composite)))["results"]}
    O = {r["question_id"]: r for r in json.load(open(os.path.join(RESULTS, args.oracle)))["results"]}
    qs = {q["question_id"]: q for q in json.load(open(args.dataset))}

    losses = [q for q in C if q in O and O[q]["correct"] and not C[q]["correct"]]
    wins = [q for q in C if q in O and O[q]["correct"] and C[q]["correct"]]
    controls = random.Random(42).sample(wins, min(args.controls, len(wins)))

    enc = _get_encoder()

    def curve(ids, label):
        acc = {b: [] for b in budgets}
        for i, qid in enumerate(ids, 1):
            q = qs[qid]
            golds = answer_turns(q)
            if not golds:
                continue
            idx = build_index(q, enc, max_chunk_chars=400)          # built once, reused below
            qv = np.asarray(enc.encode(q["question"]), dtype=np.float32)
            for b in budgets:
                epi = select(q, idx, qv, design="B", budget_chars=b, stitch=1, stitch_mode="post")
                ctx = epi["context"]
                hits = sum(1 for g in golds if g[:120].strip() and g[:120].strip() in ctx)
                acc[b].append(hits / len(golds))
            if i % 10 == 0:
                print(f"  {label}: {i}/{len(ids)}", flush=True)
        return acc

    print(f"losses n={len(losses)}  controls n={len(controls)}  budgets {budgets}\n", flush=True)
    for label, ids in (("LOSSES", losses), ("CONTROL", controls)):
        acc = curve(ids, label)
        n = len(acc[budgets[0]])
        print(f"\n{label} (n={n})")
        print(f"   {'budget':>8}{'mean cov':>11}{'all present':>14}{'none present':>14}")
        for b in budgets:
            v = acc[b]
            full = sum(1 for c in v if c >= 0.999)
            none = sum(1 for c in v if c <= 0.001)
            print(f"   {b:>8}{sum(v)/len(v):>10.1%}{full/len(v):>13.0%}{none/len(v):>13.0%}")


if __name__ == "__main__":
    main()
