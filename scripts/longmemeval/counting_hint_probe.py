#!/usr/bin/env python3
"""Three-cell ablation of `_COUNTING_HINT`, run BEFORE pre-registering a replacement.

The hypothesis under test is that `_COUNTING_HINT` -- "If the question asks how many,
COUNT the matching records and show the count." -- is what drives the supersession sums
(`4b24c848` answers 3+5=8 against a gold of 5; `5831f84d` answers 10+12+15=37 against 15).

It is IMPLICATED, not convicted: "count the matching records" over two matching records
literally yields 2, not 8. The model is summing quantities *inside* the records, which the
hint does not literally command. So this discriminates rather than confirms:

  * hint removed fixes the sums  -> the fix is REMOVAL or narrowing, and no new instruction
    surface is needed. Given that the last instruction-mediated read-path change cost 28
    questions, less surface is the safer bet.
  * hint removed does not fix them -> a reconciliation directive's semantic clause is doing
    the real work, and we know exactly what is being bet on.

Canaries matter as much as targets: removing the hint could cost the counting questions that
are currently RIGHT, and this layer exists to make counting work.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_prompts import get_anscheck_prompt  # noqa: E402
from precision_selector import build_index, select  # noqa: E402
from run_benchmark import _generate_with_retry, _get_encoder, format_question_with_date  # noqa: E402

import numpy as np  # noqa: E402
from aeon_py import compose as C  # noqa: E402
from aeon_py.consolidation import parse_records  # noqa: E402

RECONCILE_HINT = (
    "If the question asks for a count or total quantity:\n"
    "- If multiple records provide updated totals or revised statuses for the same item or "
    "activity across dates, use the latest updated figure rather than summing historical "
    "values.\n"
    "- If records describe separate, distinct additions or events, sum them.\n"
    "- If the item or activity is never mentioned at all, state that there is no record of "
    "it (do not answer 0)."
)

# Supersession sums + other confirmed overcounts + currently-correct counting canaries.
TARGETS = ["4b24c848", "5831f84d", "gpt4_213fd887", "d23cf73b", "gpt4_15e38248",
           "gpt4_59c863d7"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--records-cache", required=True)
    ap.add_argument("--baseline", required=True, help="run whose correct/wrong labels pick canaries")
    ap.add_argument("--model", default="gemma4:31b-cloud")
    ap.add_argument("--canaries", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.environ["AEON_LLM_MODEL"] = args.model
    from aeon_py.llm import OllamaProvider
    llm = OllamaProvider()

    ds = {q["question_id"]: q for q in json.load(open(args.dataset))}
    cache = json.load(open(args.records_cache))
    base = {r["question_id"]: r for r in json.load(open(args.baseline))["results"]}
    cohort = set(json.load(open(
        Path(args.baseline).parent / "counting_cohort.json")))

    canaries = [q for q in sorted(cohort) if base.get(q, {}).get("correct")][:args.canaries]
    ids = [q for q in TARGETS if q in cache] + canaries
    print(f"probe: {len(ids)} questions ({len(ids) - len(canaries)} targets, "
          f"{len(canaries)} currently-correct canaries)", flush=True)
    enc = _get_encoder()
    print(f"warm-up: {_generate_with_retry(llm, 'Say OK.', retries=5)[:12]!r}", flush=True)

    cells = {"current": dict(counting_hint=True),
             "no_hint": dict(counting_hint=False),
             "reconcile": dict(counting_hint=False)}
    rows = []
    for qid in ids:
        q = ds[qid]
        records = parse_records(cache[qid], qid)
        idx = build_index(q, enc, max_chunk_chars=400)
        qv = np.asarray(enc.encode(q["question"]), dtype=np.float32)
        epi = select(q, idx, qv, design="B", budget_chars=6000, stitch=1, stitch_mode="post")
        row = {"question_id": qid, "gold": str(q["answer"])[:80],
               "is_canary": qid in canaries, "cells": {}}
        for name, kw in cells.items():
            prompt = C.compose(records, epi["context"].splitlines(),
                               format_question_with_date(q), **kw)
            if name == "reconcile":
                prompt = prompt.replace("\nAnswer:", f" {RECONCILE_HINT}\n\nAnswer:")
            hyp = _generate_with_retry(llm, prompt, system_prompt="", temperature=0.0)
            jp = get_anscheck_prompt(q["question_type"], q["question"], q["answer"], hyp,
                                     abstention="_abs" in qid)
            # An infrastructure failure is not a wrong answer. This benchmark's own harness
            # was bitten by exactly this (run_benchmark.py:79-83, a transient 503 during
            # model load silently scored as two wrong answers), and this probe reproduced
            # the bug on its first run before the guard was added.
            err = "[System Error:" in hyp
            judge = "" if err else _generate_with_retry(llm, jp, system_prompt="",
                                                        temperature=0.0)
            ok = (not err) and judge.strip().lower().startswith("yes")
            row["cells"][name] = {"correct": ok, "is_error": err, "tail": hyp.strip()[-110:]}
        rows.append(row)
        marks = "  ".join(f"{n}={'OK ' if row['cells'][n]['correct'] else 'WRG'}" for n in cells)
        print(f"[{qid}]{' (canary)' if row['is_canary'] else ''}  {marks}", flush=True)

    json.dump({"reconcile_hint": RECONCILE_HINT, "results": rows}, open(args.out, "w"), indent=2)
    print("\n=== totals ===")
    for name in cells:
        t = [r for r in rows if not r["is_canary"]]
        c = [r for r in rows if r["is_canary"]]
        print(f"  {name:10s} targets {sum(r['cells'][name]['correct'] for r in t)}/{len(t)}"
              f"   canaries {sum(r['cells'][name]['correct'] for r in c)}/{len(c)}")


if __name__ == "__main__":
    main()
