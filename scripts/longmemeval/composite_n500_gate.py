#!/usr/bin/env python3
"""Scores the composite n=500 run against the bars pre-registered in v4-plan.md
(2026-08-26, before the run) -- so the verdict is mechanical rather than read off
the numbers after the fact.

    python scripts/longmemeval/composite_n500_gate.py \
        --composite reproducibility_benchmarks/longmemeval/composite_arm_n500.json

Bars, verbatim from the pre-registration:
  PRIMARY          >= 425 correct (ETC's 413 + 2x the ~6-question noise sd).
                   414-424 is an improvement this instrument cannot distinguish
                   from noise and is reported as such rather than claimed.
  COLLATERAL GUARD no question type more than 2x its own noise sd below ETC.
  KNOWN-MISS FLOOR >= 15 of the 27 verified retrieval misses.
"""

import argparse
import json
import math
import statistics
from pathlib import Path

BASE = Path(__file__).resolve().parents[2] / "reproducibility_benchmarks" / "longmemeval"

PRIMARY_BAR = 425
KNOWN_MISS_FLOOR = 15

# Per-type noise sd, measured on the n=100 date-fix repeat and committed at pre-registration.
TYPE_SD = {
    "temporal-reasoning": 5.8, "multi-session": 5.7, "knowledge-update": 4.4,
    "single-session-user": 4.1, "single-session-assistant": 3.9,
    "single-session-preference": 2.8, "abstention": 2.8,
}

REFERENCE = {
    "single-shot": "full_session_n500_datefix_results.json",
    "ETC": "extract_then_compute_n500_datefix_results.json",
    "ETC deep-retrieval": "extract_then_compute_n500_topk200_results.json",
}


def load(path):
    rs = json.load(open(path))["results"]
    return {r["question_id"]: r for r in rs}


def mcnemar(a, b):
    """Exact two-sided binomial McNemar over the discordant pairs (a wins, b wins)."""
    both = set(a) & set(b)
    win = sum(1 for q in both if a[q]["correct"] and not b[q]["correct"])
    loss = sum(1 for q in both if b[q]["correct"] and not a[q]["correct"])
    n = win + loss
    if n == 0:
        return win, loss, 1.0
    k = min(win, loss)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return win, loss, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--composite", default=str(BASE / "composite_arm_n500.json"))
    ap.add_argument("--attribution", default=str(BASE / "answer_turn_attribution.json"))
    args = ap.parse_args()

    comp = load(args.composite)
    refs = {k: load(BASE / v) for k, v in REFERENCE.items()}
    etc = refs["ETC"]
    miss_ids = [r["question_id"] for r in json.load(open(args.attribution))["results"]
                if r["category"] == "retrieval_miss"]

    n = len(comp)
    correct = sum(r["correct"] for r in comp.values())
    errors = sum(r["is_error"] for r in comp.values())
    print(f"composite n={n}  correct={correct}  accuracy={correct / n:.3%}  n_errors={errors}")
    if n != 500:
        print(f"  !! partial run ({n}/500) -- bars below are NOT the pre-registered verdict")
    print()

    print("=== arm comparison (paired, same 500 questions) ===")
    print(f"{'arm':22s}{'correct':>9}{'delta':>8}{'wins':>7}{'losses':>8}{'p':>10}")
    for name, ref in refs.items():
        w, l, p = mcnemar(comp, ref)
        rc = sum(r["correct"] for r in ref.values())
        print(f"{name:22s}{rc:>9}{correct - rc:>+8}{w:>7}{l:>8}{p:>10.2e}")
    print(f"{'COMPOSITE':22s}{correct:>9}")
    print()

    print("=== BAR 1 -- PRIMARY (>= 425 correct) ===")
    if correct >= PRIMARY_BAR:
        primary = "PASS"
        print(f"  {correct} >= {PRIMARY_BAR}  PASS -- beats ETC's 413 by more than 2x the noise sd")
    elif correct > 413:
        primary = "INDISTINGUISHABLE"
        print(f"  {correct} is in the 414-424 band: better than ETC's 413, but by less than 2x the "
              f"noise sd. Pre-registered reading: an improvement this instrument CANNOT "
              f"distinguish from noise. Reported as such, not claimed.")
    else:
        primary = "FAIL"
        print(f"  {correct} <= 413  FAIL -- does not beat ETC")
    print()

    print("=== BAR 2 -- COLLATERAL GUARD (no type > 2x its own sd below ETC) ===")
    print(f"{'type':30s}{'n':>5}{'ETC':>6}{'comp':>6}{'delta':>8}{'floor':>8}  verdict")
    collateral, breaches = "PASS", []
    for t in sorted(TYPE_SD):
        ids = [q for q, r in comp.items() if r["report_type"] == t]
        if not ids:
            continue
        ec = sum(etc[q]["correct"] for q in ids if q in etc)
        cc = sum(comp[q]["correct"] for q in ids)
        floor = ec - 2 * TYPE_SD[t]
        ok = cc >= floor
        if not ok:
            collateral, _ = "BREACH", breaches.append(t)
        print(f"{t:30s}{len(ids):>5}{ec:>6}{cc:>6}{cc - ec:>+8}{floor:>8.1f}  "
              f"{'ok' if ok else 'BREACH'}")
    print(f"  -> {collateral}" + (f" on {', '.join(breaches)}" if breaches else ""))
    print()

    print(f"=== BAR 3 -- KNOWN-MISS FLOOR (>= {KNOWN_MISS_FLOOR} of 27) ===")
    have = [q for q in miss_ids if q in comp]
    got = sum(comp[q]["correct"] for q in have)
    etc_got = sum(etc[q]["correct"] for q in have if q in etc)
    floor_v = "PASS" if got >= KNOWN_MISS_FLOOR else "FAIL"
    print(f"  composite {got}/{len(have)}   ETC {etc_got}/{len(have)}   -> {floor_v}")
    print()

    print("=== cost (the product argument) ===")
    med = lambda xs: statistics.median(xs)
    cc = med([r["prompt_chars"] for r in comp.values()])
    cg = med([r["timing_seconds"]["generation"] for r in comp.values()])
    print(f"{'arm':22s}{'accuracy':>10}{'context':>10}{'calls':>7}{'gen s':>8}{'pts/10k ch':>12}")
    def _chars(r):
        return r.get("prompt_chars") or r.get("context_chars") or r.get("retrieved_chars") or 0

    def _gen(r):
        """Answer-side seconds: one call for single-shot/composite, extract+compute for ETC."""
        t = r["timing_seconds"]
        return t["generation"] if "generation" in t else t.get("extract", 0) + t.get("compute", 0)

    for name, ref, calls in (("single-shot", refs["single-shot"], 1), ("ETC", etc, 2)):
        rc = sum(r["correct"] for r in ref.values()) / len(ref) * 100
        rch = med([_chars(r) for r in ref.values()])
        rg = med([_gen(r) for r in ref.values()])
        print(f"{name:22s}{rc:>9.1f}%{rch:>10.0f}{calls:>7}{rg:>8.2f}"
              f"{(rc / rch * 10000 if rch else 0):>12.2f}")
    print(f"{'COMPOSITE':22s}{correct / n * 100:>9.1f}%{cc:>10.0f}{1:>7}{cg:>8.2f}"
          f"{correct / n * 100 / cc * 10000:>12.2f}")
    print()
    print(f"VERDICT  primary={primary}  collateral={collateral}  known-miss={floor_v}")


if __name__ == "__main__":
    main()
