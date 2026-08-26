#!/usr/bin/env python3
"""TIER 1 (free): does the sub-turn selector actually put the evidence in a small context?

ZERO LLM CALLS. Measures answer-turn coverage at a fixed char budget across selector
configurations, reusing the CALIBRATED `_turn_in_context` matcher from
`answer_turn_attribution.py` rather than writing a new one -- an earlier keyword-based
matcher in this project systematically misattributed retrieval misses, and that lesson
is not worth re-learning.

Coverage is necessary but not sufficient (the 200x20 run proved coverage can rise while
accuracy falls), so this tier only picks which configurations are worth spending LLM
calls on. Reference points:
    turn-level top_k=30  (production today) -- the baseline to beat
    oracle answer turns +/-1                -- 100% coverage by construction, 5,654 chars

`has_answer` is used ONLY to score coverage here; it never enters the selector.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from answer_turn_attribution import _norm, _turn_in_context  # calibrated matcher
from precision_selector import build_index, select
from run_benchmark import _get_encoder, _stratified_sample

import numpy as np  # noqa: E402

# Round 2. Round 1 findings folded in: MMR is a confirmed NEGATIVE (43.3% vs 49.7% turn
# coverage -- the near-duplicate top chunks were carrying real signal, so pushing them
# apart lost evidence), and design A == design B on coverage, so the two-stage pool adds
# nothing. Both dropped. The live question is now the stitch tension: inline stitching
# cost 21 points of coverage at fixed budget, so "post" ordering is tested against it.
GRID = [
    # (design, chunk_chars, budget, stitch, mmr, stitch_mode)
    ("B", 400, 9000, 0, 0.0, "inline"),    # coverage ceiling, no licensing
    ("B", 400, 9000, 1, 0.0, "inline"),    # round-1 naive stitching
    ("B", 400, 9000, 1, 0.0, "post"),      # coverage first, then buy licensing
    ("B", 400, 12000, 1, 0.0, "post"),
    ("B", 400, 15000, 1, 0.0, "post"),
    ("B", 400, 15000, 0, 0.0, "inline"),
]


def answer_turns(q):
    return [t for s in q["haystack_sessions"] for t in s if t.get("has_answer")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--attribution", required=True)
    ap.add_argument("--sample", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out")
    args = ap.parse_args()

    ds = json.load(open(args.dataset))
    by_id = {q["question_id"]: q for q in ds}
    attr = json.load(open(args.attribution))["results"]
    miss_ids = [r["question_id"] for r in attr if r["category"] == "retrieval_miss"]

    sample = _stratified_sample(ds, args.sample, args.seed)
    qs = {q["question_id"]: q for q in sample}
    for mid in miss_ids:                       # always include the known-hard cases
        qs.setdefault(mid, by_id[mid])
    qs = {k: v for k, v in qs.items() if answer_turns(v)}
    print(f"Sweeping {len(qs)} questions ({len(miss_ids)} known retrieval misses included) "
          f"x {len(GRID)} configs, zero LLM calls", file=sys.stderr)

    enc = _get_encoder()
    rows = []
    t0 = time.perf_counter()
    for i, (qid, q) in enumerate(qs.items(), 1):
        idx400 = build_index(q, enc, max_chunk_chars=400)
        idx = idx400   # round 2 grid is all c400; skip the second encode pass
        qv = np.asarray(enc.encode(q["question"]), dtype=np.float32)
        ats = answer_turns(q)
        rec = {"question_id": qid, "is_known_miss": qid in miss_ids,
               "report_type": ("abstention" if "_abs" in qid else q["question_type"]),
               "n_answer_turns": len(ats), "n_chunks": idx400.n_chunks, "configs": {}}
        for design, cc, budget, stitch, mmr, smode in GRID:
            index = idx if cc == 250 else idx400
            r = select(q, index, qv, design=design, budget_chars=budget, stitch=stitch,
                       mmr=mmr, stitch_mode=smode)
            nctx = _norm(r["context"])
            got = sum(1 for t in ats if _turn_in_context(t["content"], nctx))
            rec["configs"][f"{design}/b{budget}/s{stitch}/{smode}"] = {
                "retrieved": got, "total": len(ats), "chars": r["chars"], "turns": r["n_turns"]}
        rows.append(rec)
        if i % 10 == 0:
            el = time.perf_counter() - t0
            print(f"  {i}/{len(qs)} ({el:.0f}s, ~{el/i*(len(qs)-i):.0f}s left)", file=sys.stderr)

    print("\n" + "=" * 96)
    print("TIER 1 COVERAGE (answer turns landing in a small budgeted context; 0 LLM calls)")
    print("=" * 96)
    hdr = (f"{'config':<26}{'full cov':>11}{'turn cov':>10}{'misses fixed':>14}"
           f"{'med chars':>11}{'med turns':>10}")
    print(hdr)
    keys = [f"{d}/b{b}/s{s}/{sm}" for d, c, b, s, m, sm in GRID]
    misses = [r for r in rows if r["is_known_miss"]]
    for k in keys:
        full = sum(1 for r in rows if r["configs"][k]["retrieved"] == r["n_answer_turns"])
        cov = (sum(r["configs"][k]["retrieved"] for r in rows)
               / sum(r["n_answer_turns"] for r in rows))
        mf = sum(1 for r in misses if r["configs"][k]["retrieved"] == r["n_answer_turns"])
        ch = statistics.median([r["configs"][k]["chars"] for r in rows])
        tn = statistics.median([r["configs"][k]["turns"] for r in rows])
        print(f"{k:<26}{full:>7}/{len(rows):<3}{cov*100:>9.1f}%{mf:>9}/{len(misses):<4}"
              f"{ch:>11,.0f}{tn:>10.0f}")
    print(f"\n  reference: production turn-level top_k=30 delivers ~100,889 chars")
    print(f"  reference: oracle (answer turns +/-1) = 100% coverage at 5,654 chars")

    if args.out:
        json.dump({"grid": keys, "results": rows}, open(args.out, "w"), indent=2)
        print(f"\nWrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
