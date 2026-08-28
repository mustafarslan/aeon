"""Why does gold retrieval win 36 questions the composite loses, and is that reachable?

THE DECISION THIS INFORMS. composite@glm-5.2 scores 448 and oracle@glm-5.2 scores 457,
but their UNION is 484 -- they fail on different questions. The oracle's advantage comes
entirely from LongMemEval's `has_answer` turn annotations, which do not exist in
production, so that headroom is real but its mechanism is not transferable. The question
that decides whether any of it is reachable is narrower and answerable:

  On the questions the oracle wins and the composite loses, is the answer-bearing turn
  ABSENT from the composite's own episodic slice, or PRESENT and unused?

ABSENT  -> a retrieval/budget problem. The composite selects its episodic context by
           embedding similarity under a 6,000-char budget; if the load-bearing turn does
           not survive that cut, widening or retargeting the budget is a cheap, entirely
           production-available intervention with a measurable ceiling.
PRESENT -> a reader problem. The evidence was in the prompt and the model did not use it,
           which is where six prior interventions already produced zero net gain.

The composite's episodic selection is rebuilt exactly as the arm ran it (design B,
6,000-char budget, stitch=1/post), so this measures the shipped selector, not a proxy.
No LLM calls: this is a containment question about the prompt, not about the answer.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from composite_arm_experiment import _get_encoder, build_index, select

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "..", "reproducibility_benchmarks", "longmemeval")


def answer_turns(question):
    """The turns LongMemEval marks as load-bearing -- the oracle's entire advantage."""
    return [t["content"] for turns in question["haystack_sessions"] for t in turns
            if t.get("has_answer")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/Volumes/AI_SSD/hf-cache/longmemeval/longmemeval_s_cleaned.json")
    ap.add_argument("--composite", default="composite_glm52_n500.json")
    ap.add_argument("--oracle", default="oracle_glm52_n500.json")
    ap.add_argument("--budget", type=int, default=6000)
    args = ap.parse_args()

    C = {r["question_id"]: r for r in json.load(open(os.path.join(RESULTS, args.composite)))["results"]}
    O = {r["question_id"]: r for r in json.load(open(os.path.join(RESULTS, args.oracle)))["results"]}
    qs = {q["question_id"]: q for q in json.load(open(args.dataset))}

    targets = [q for q in C if q in O and O[q]["correct"] and not C[q]["correct"]]
    controls = [q for q in C if q in O and O[q]["correct"] and C[q]["correct"]]
    print(f"oracle wins / composite loses: {len(targets)}   both correct (control): {len(controls)}\n")

    enc = _get_encoder()

    def coverage(qid):
        """Fraction of answer-bearing turns whose text survives into the episodic slice."""
        q = qs[qid]
        idx = build_index(q, enc, max_chunk_chars=400)
        qv = np.asarray(enc.encode(q["question"]), dtype=np.float32)
        epi = select(q, idx, qv, design="B", budget_chars=args.budget, stitch=1, stitch_mode="post")
        ctx = epi["context"]
        golds = answer_turns(q)
        if not golds:
            return None
        # Substring on a 120-char window: the selector chunks turns, so a long turn can be
        # present in part. A window that survives the cut is what the reader actually sees.
        hits = sum(1 for g in golds if g[:120].strip() and g[:120].strip() in ctx)
        return hits / len(golds)

    # The control group is what makes this interpretable: if coverage were low everywhere,
    # low coverage on the losses would say nothing about why they were lost.
    import random
    ctrl_sample = random.Random(42).sample(controls, min(60, len(controls)))
    for label, ids in (("LOSSES (oracle wins)", targets), ("CONTROL (both correct)", ctrl_sample)):
        covs = []
        for i, qid in enumerate(ids, 1):
            c = coverage(qid)
            if c is not None:
                covs.append(c)
            if i % 20 == 0:
                print(f"  {label}: {i}/{len(ids)}", flush=True)
        if not covs:
            continue
        full = sum(1 for c in covs if c >= 0.999)
        none = sum(1 for c in covs if c <= 0.001)
        print(f"\n{label}  n={len(covs)}")
        print(f"   mean answer-turn coverage : {sum(covs)/len(covs):.1%}")
        print(f"   ALL answer turns present  : {full:>3} ({full/len(covs):.0%})")
        print(f"   NO answer turn present    : {none:>3} ({none/len(covs):.0%})\n")


if __name__ == "__main__":
    main()
