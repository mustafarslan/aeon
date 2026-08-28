#!/usr/bin/env python3
"""Step 0 kill-check for the v4.2 aggregation track. ZERO LLM CALLS.

Reads the 17 losses in the 3+-answer-turn stratum under `glm-5.2` individually, because
the report's own precedent (CROSS_MODEL_SESSION_REPORT.md section 6 note 4) is that a
classifier over this bucket was wrong twice and reading the cases corrected it both times.

THE QUESTION THIS ANSWERS. `spatiotemporal.md` proposes pre-aggregated cluster records,
and the cheap deterministic version of that is a per-category roster header --
"ITEM(ACQUISITION/music album) -- N items:" -- computed over EntityGroups at render time.
The check is not "is this an aggregation question" (all 17 are, by construction). It is:

    WOULD THE HEADER PRINT THE GOLD NUMBER?

A header that prints the wrong number is worse than no header, because it hands the reader
a computed value to be confident about. So each case names the category a roster would key
on and the run size is computed mechanically against gold.

A FIRST VERSION OF THIS SCRIPT SCORED CONVERTIBILITY AS "some run in the corpus has
exactly the gold size" AND REPORTED 7/17. That was coincidence matching, not
convertibility: on `88432d0a` (gold 4, "how many times did I bake in the past two weeks")
it matched `TRAVEL/trip=4`, and on `6d550036` (gold 2) fifteen different runs had size 2
in a corpus of 121 groups. The classifier is replaced by the table below -- the target
category is a judgement, stated explicitly and inspectable, and the arithmetic over it is
mechanical.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from aeon_py.compose import order_records
from aeon_py.consolidation import parse_record_line
from aeon_py.entities import group_entities

DEFAULT_DS = "/Volumes/AI_SSD/hf-cache/longmemeval/longmemeval_s_cleaned.json"
BENCH = Path("reproducibility_benchmarks/longmemeval")

# shape: what the question actually asks for, read from the question text.
#   enumerate  -- count the members of a category. The roster's design target.
#   windowed   -- count the members of a category falling inside a date window.
#   predicate  -- count the members satisfying a property the taxonomy does not encode
#                 ("led", "replaced or fixed", "including the one for my friend's kid").
#   order      -- rank by date. Not a count at all.
#   argmax     -- pick the extremum of a per-entity quantity.
#   arithmetic -- sum quantities stated inside records.
# target: regex over "BUCKET/subtype" naming the run a roster would key on, or "" when
#         no single category holds the answer set.
# gold_n: the integer the gold answer asserts, including when stated in prose.
CASES: dict[str, tuple[str, str, int | None]] = {
    "bf659f65":      ("enumerate", r"music album",              3),
    "1a8a66a6":      ("enumerate", r"magazine",                 2),
    "gpt4_a56e767c": ("enumerate", r"film festival",            4),
    "88432d0a":      ("windowed",  r"CONSUMABLE/(recipe|dish)", 4),
    "9d25d4e0":      ("windowed",  r"jewelry",                  3),
    "a9f6b44c":      ("windowed",  r"bicycle|bike",             2),
    "gpt4_2f8be40d": ("windowed",  r"wedding",                  3),
    "gpt4_7fce9456": ("windowed",  r"townhouse|home|condo",     4),
    "6d550036":      ("predicate", r"PROJECT/",                 2),
    "46a3abf7":      ("predicate", r"aquarium|tank",            3),
    "gpt4_ab202e7f": ("predicate", r"",                         5),
    "gpt4_7f6b06db": ("order",     r"TRAVEL/trip",           None),
    "gpt4_d6585ce8": ("order",     r"",                      None),
    "gpt4_f420262c": ("order",     r"TRAVEL/flight",         None),
    "gpt4_2f56ae70": ("argmax",    r"",                      None),
    "gpt4_2ba83207": ("argmax",    r"",                      None),
    "aae3761f":      ("arithmetic", r"TRAVEL/trip",          None),
}


def answer_turn_count(q: dict) -> int:
    return sum(1 for s in q["haystack_sessions"] for t in s if t.get("has_answer"))


def build_runs(block: str):
    """(primary_bucket, subtype) -> [EntityGroup], as `render_records()` would build them.

    Routed through the production functions rather than reimplemented, so a change to the
    taxonomy or to co-referent collapse moves this check with it.
    """
    recs = [r for r in (parse_record_line(l, "") for l in block.splitlines()) if r]
    groups, _ = group_entities(order_records(recs))
    runs: dict[tuple[str, str], list] = {}
    for g in groups:
        runs.setdefault((g.primary_bucket, g.representative.subtype.lower()), []).append(g)
    return runs, len(recs), len(groups)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DS)
    ap.add_argument("--composite", default=str(BENCH / "composite_glm52_n500.json"))
    ap.add_argument("--oracle", default=str(BENCH / "oracle_glm52_n500.json"))
    ap.add_argument("--records-cache", default=str(BENCH / "records_composite_partial.json"))
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    ds = {q["question_id"]: q for q in json.load(open(args.dataset))}
    comp = {r["question_id"]: r for r in json.load(open(args.composite))["results"]}
    orac = {r["question_id"]: r for r in json.load(open(args.oracle))["results"]}
    cache = json.load(open(args.records_cache))

    bucket3 = [i for i, q in ds.items() if answer_turn_count(q) >= 3]
    cc = sum(comp[i]["correct"] for i in bucket3)
    oc = sum(orac[i]["correct"] for i in bucket3)
    print(f"3+ stratum: n={len(bucket3)} composite={cc} oracle={oc}"
          f"   (report 7.4: n=68 composite=51 oracle=61)")
    if (len(bucket3), cc, oc) != (68, 51, 61):
        print("PARTITION DOES NOT REPRODUCE -- stop.")
        return 1
    losses = sorted(i for i in bucket3 if not comp[i]["correct"])
    missing = set(losses) ^ set(CASES)
    if missing:
        print(f"CASES table out of sync with the loss set: {sorted(missing)}")
        return 1
    print()

    shapes, verdicts, rows = Counter(), Counter(), []
    for qid in sorted(losses, key=lambda i: (CASES[i][0], i)):
        shape, target, gold_n = CASES[qid]
        q = ds[qid]
        runs, n_recs, n_groups = build_runs(cache.get(qid, ""))
        hits = ([(k, v) for k, v in runs.items() if re.search(target, f"{k[0]}/{k[1]}", re.I)]
                if target else [])
        printed = max((len(v) for _k, v in hits), default=None)   # the run a header names
        merged = len({g.key for _k, v in hits for g in v}) or None  # same subtype, any bucket

        # When the target category fragments into runs of DIFFERENT sizes there is no
        # single number a header would print -- the reader is handed several competing
        # ones and must still choose. Scoring that as a hit by taking the max is exactly
        # the flattering reading this script exists to avoid: `6d550036` fragments into
        # PROJECT/=2, PROJECT/event=2, PROJECT/academic=1, PROJECT/art series=1 and would
        # otherwise score as a conversion on the strength of `max` landing on gold.
        fragmented = len(hits) > 1 and len({len(v) for _k, v in hits}) > 1
        if gold_n is None:
            verdict = "NOT-A-COUNT"
        elif printed is None:
            verdict = "NO-SINGLE-RUN"
        elif fragmented:
            verdict = ("AMBIGUOUS-FRAGMENTED-RUNS" if merged != gold_n
                       else "GOLD-ONLY-IF-MERGED-ACROSS-BUCKETS")
        elif printed == gold_n:
            verdict = "WOULD-PRINT-GOLD"
        elif merged == gold_n:
            verdict = "GOLD-ONLY-IF-MERGED-ACROSS-BUCKETS"
        else:
            verdict = "WOULD-PRINT-WRONG"
        shapes[shape] += 1
        verdicts[verdict] += 1

        print(f"--- {qid}  [{shape}]  oracle={'OK' if orac[qid]['correct'] else 'ALSO WRONG'}"
              f"  records={n_recs} groups={n_groups}")
        print(f"  Q     : {q['question'][:130]}")
        print(f"  GOLD  : {str(q['answer'])[:110]}"
              + (f"   -> {gold_n}" if gold_n is not None else ""))
        print(f"  GOT   : {(comp[qid]['hypothesis'] or '(EMPTY GENERATION)')[:130]}".replace("\n", " "))
        if target:
            runs_desc = ", ".join(f"{k[0]}/{k[1]}={len(v)}"
                                  for k, v in sorted(hits, key=lambda kv: -len(kv[1]))[:4])
            print(f"  runs  : {runs_desc or '(no run matches the target category)'}"
                  + (f"   [distinct entities across buckets: {merged}]"
                     if merged and len(hits) > 1 else ""))
        print(f"  ROSTER: {verdict}"
              + (f"  (header would print {printed}, gold {gold_n})"
                 if printed is not None and gold_n is not None else ""))
        print()
        rows.append({"question_id": qid, "shape": shape, "gold_n": gold_n,
                     "roster_would_print": printed, "merged_across_buckets": merged,
                     "verdict": verdict, "oracle_correct": orac[qid]["correct"]})

    print("=" * 74)
    print(f"{len(losses)} losses in the 3+ stratum\n")
    print("  by question shape:")
    for k, v in shapes.most_common():
        print(f"    {k:<12} {v}")
    print("\n  by what a deterministic roster header would print:")
    for k, v in verdicts.most_common():
        print(f"    {k:<36} {v}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"stratum": {"n": len(bucket3), "composite": cc, "oracle": oc},
             "shapes": dict(shapes), "verdicts": dict(verdicts), "losses": rows}, indent=1))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
