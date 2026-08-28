#!/usr/bin/env python3
"""Aeon's accuracy under each competitor's own metric definition. Zero LLM calls.

WHY THIS EXISTS. Published LongMemEval numbers are not all the same measurement, and reading
them as though they were is how a 5.3-point gap gets reported where a 2.8-point one exists.
Verified against the publishing sources on 2026-08-28 (see `spatiotemporal.md` §2):

  * Mastra  -- "Overall accuracy is the unweighted average across all 6 categories."
  * OMEGA   -- "95.4% (466/500)", i.e. task-averaged over 5 categories; the raw count is 93.2%.
  * MemPal  -- 96.6% is R@5 RETRIEVAL RECALL, and their own document warns in bold that it is
               "not comparable" to QA accuracy. Not a QA number and not reproduced here.

THIS DOES NOT CHANGE AEON'S HEADLINE. Micro accuracy over all 500 with abstention included is
the strictest definition available, and switching to a friendlier one after seeing the numbers
is the exact manoeuvre §2 criticises. This sizes the gap; it does not restate it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BENCH = Path("reproducibility_benchmarks/longmemeval")

# OMEGA reports five categories, merging the two single-session recall types.
OMEGA_CATEGORIES = {
    "Single-Session Recall": ("single-session-user", "single-session-assistant"),
    "Preference Application": ("single-session-preference",),
    "Multi-Session Reasoning": ("multi-session",),
    "Knowledge Updates": ("knowledge-update",),
    "Temporal Reasoning": ("temporal-reasoning",),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=str(BENCH / "composite_glm52_n500.json"))
    args = ap.parse_args()

    results = json.load(open(args.arm))["results"]
    cats: dict[str, list[bool]] = {}
    for r in results:
        cats.setdefault(r["report_type"], []).append(bool(r["correct"]))

    def micro(keys) -> float:
        hit = sum(sum(cats[k]) for k in keys)
        n = sum(len(cats[k]) for k in keys)
        return hit / n if n else 0.0

    correct, n = sum(sum(v) for v in cats.values()), sum(len(v) for v in cats.values())
    no_abstain = [k for k in cats if k != "abstention"]

    print(f"arm: {Path(args.arm).name}\n")
    print(f"  raw micro, all {n} (THE HEADLINE)          {correct}/{n} = {correct/n*100:.1f}%")
    print(f"  macro over all {len(cats)} categories             "
          f"{sum(sum(v)/len(v) for v in cats.values())/len(cats)*100:.1f}%")
    print(f"  macro over 6 categories (Mastra's metric)  "
          f"{sum(micro([k]) for k in no_abstain)/len(no_abstain)*100:.1f}%   vs Mastra 94.87%")
    print(f"  macro over OMEGA's 5 categories            "
          f"{sum(micro(v) for v in OMEGA_CATEGORIES.values())/len(OMEGA_CATEGORIES)*100:.1f}%"
          f"   vs OMEGA 95.4%")
    print(f"  raw micro, abstention excluded             "
          f"{sum(sum(cats[k]) for k in no_abstain)}/{sum(len(cats[k]) for k in no_abstain)}"
          f" = {micro(no_abstain)*100:.1f}%")
    print(f"\n  apples-to-apples on RAW counts: Aeon {correct}/500 vs OMEGA 466/500"
          f"  -> {correct-466:+d} questions")
    print("\n  per category:")
    for k in sorted(cats):
        print(f"    {k:<28} n={len(cats[k]):<4} {sum(cats[k]):>3}  {micro([k])*100:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
