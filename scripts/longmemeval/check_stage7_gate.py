#!/usr/bin/env python3
"""
Stage 7 gate checker (v4-plan.md Stage 7's "Gate" section).

Applies the gate's own checkable criterion mechanically, so it isn't
relitigated with hindsight once a live result is in hand (the same
concern that motivated task 1's `_apply_decision_rule()`):

    "the live re-run must close at least half of the 54.5-point gap
    (multi-session accuracy >= ~63.7%) without regressing other question
    types' accuracy or blowing the CI latency-regression budget"

Reads the two baselines this repo already has on record rather than
hardcoding the numbers (so a re-run of either baseline automatically
updates what this script checks against, instead of silently going stale):
  - the oracle ceiling: reproducibility_benchmarks/longmemeval/oracle_results.json
  - the real, deterministic top_k=30 baseline:
    reproducibility_benchmarks/longmemeval/topk_sweep_30_temp0_gemma.json

Accepts the live gate re-run's result in either shape this project already
produces:
  - `expansion_unit_experiment.py`'s `{"arm_summaries": {<arm>: {...}}}`
    (pass --arm to pick which one -- this is what task 1's own run already
    produces for the winning arm, so a separate gate-only re-run is only
    needed if the live integration point built in task 2 differs from
    what task 1's harness measured)
  - `run_benchmark.py`'s `{"summary": {"per_question_type": {...}}}`

THIS FILE IS NOT RUN AS PART OF WRITING STAGE 7 -- syntax-checked only,
per explicit instruction to hold all execution for a later session.

Usage (once there is a live result to check):
    python scripts/longmemeval/check_stage7_gate.py \\
        --live-result reproducibility_benchmarks/longmemeval/expansion_unit_results.json \\
        --arm window_5
"""

import argparse
import json
from pathlib import Path

DEFAULT_ORACLE_PATH = "reproducibility_benchmarks/longmemeval/oracle_results.json"
DEFAULT_BASELINE_PATH = "reproducibility_benchmarks/longmemeval/topk_sweep_30_temp0_gemma.json"
GAP_CLOSURE_FRACTION_REQUIRED = 0.5  # "at least half of the gap" -- Stage 7's own text


def _multi_session_accuracy(summary_per_type: dict) -> float:
    row = summary_per_type.get("multi-session")
    if row is None:
        raise ValueError("No 'multi-session' row in this result's per_question_type breakdown")
    return row["accuracy"]


def _load_live_result(path: Path, arm: str | None) -> dict:
    """Returns {"per_question_type": {...}, "n_errors": int} regardless of
    which of the two known result shapes `path` holds."""
    with open(path) as f:
        data = json.load(f)

    if "arm_summaries" in data:
        if arm is None:
            raise ValueError(
                "This is an expansion_unit_experiment.py-shaped result (multiple arms) -- "
                "pass --arm to pick which one to check against the gate."
            )
        if arm not in data["arm_summaries"]:
            raise ValueError(f"Arm {arm!r} not found in {path} (available: {list(data['arm_summaries'])})")
        s = data["arm_summaries"][arm]
        return {"per_question_type": s["per_question_type"], "n_errors": s["n_errors"]}

    if "summary" in data:
        s = data["summary"]
        return {"per_question_type": s["per_question_type"], "n_errors": s.get("num_errors", 0)}

    raise ValueError(f"{path} doesn't match either known result shape (arm_summaries or summary)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live-result", required=True, help="Path to the live gate re-run's result JSON")
    parser.add_argument("--arm", default=None, help="Required if --live-result is an expansion_unit_experiment.py-shaped file with multiple arms")
    parser.add_argument("--oracle", default=DEFAULT_ORACLE_PATH)
    parser.add_argument("--baseline", default=DEFAULT_BASELINE_PATH)
    args = parser.parse_args()

    oracle_path = Path(args.oracle)
    baseline_path = Path(args.baseline)
    live_path = Path(args.live_result)

    with open(oracle_path) as f:
        oracle_summary = json.load(f)["summary"]
    with open(baseline_path) as f:
        baseline_summary = json.load(f)["summary"]
    live = _load_live_result(live_path, args.arm)

    ceiling = _multi_session_accuracy(oracle_summary["per_question_type"])
    baseline = _multi_session_accuracy(baseline_summary["per_question_type"])
    live_acc = _multi_session_accuracy(live["per_question_type"])

    gap = ceiling - baseline
    required_acc = baseline + GAP_CLOSURE_FRACTION_REQUIRED * gap
    gap_closed_fraction = (live_acc - baseline) / gap if gap > 0 else float("nan")

    # Contamination guard: don't judge a contaminated live run against the gate.
    if live["n_errors"] > 0:
        print(
            f"BLOCKED: live result has {live['n_errors']} transport error(s) -- "
            f"re-run before checking the gate, per Stage 7's contamination guard."
        )
        raise SystemExit(1)

    passed = live_acc >= required_acc

    print(f"Oracle ceiling (multi-session):     {ceiling*100:.1f}%  ({oracle_path})")
    print(f"Real baseline (multi-session, k=30): {baseline*100:.1f}%  ({baseline_path})")
    print(f"Gap to close:                        {gap*100:.1f} points")
    print(f"Required (>= half the gap closed):   {required_acc*100:.1f}%")
    print(f"Live result (multi-session):          {live_acc*100:.1f}%  ({live_path}"
          f"{f', arm={args.arm}' if args.arm else ''})")
    print(f"Fraction of gap closed:               {gap_closed_fraction*100:.1f}%")
    print()
    if passed:
        print("GATE: PASS -- closes at least half the multi-session gap. Stage 7's success criterion is met "
              "for this metric (still check: no regression on other question types, and the CI "
              "latency-regression budget, before calling the stage done).")
    else:
        print("GATE: STOP, not escalate -- falls short of closing half the gap. Per Stage 7's own text, this "
              "means the retrieval unit was not (solely) the bottleneck. Next step is characterizing what "
              "else is (e.g. re-checking answer_in_context against this strategy's actual retrieved "
              "context), not reflexively trying a bigger/more expensive unit.")


if __name__ == "__main__":
    main()
