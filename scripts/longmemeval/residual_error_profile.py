"""Partition the composite arm's residual errors into buckets that imply different work.

THE QUESTION THIS ANSWERS. "We are at 448/500, what now?" is unanswerable from an
accuracy number, because the remaining 52 are not one problem. Handed the SAME reader
and the SAME judge, the oracle arm gets gold answer-bearing turns while the composite
gets consolidated records. Crossing the two partitions the residual:

  BENCHMARK/READER FLOOR  both arms fail. Perfect evidence does not fix it, so no
                          memory-side work can. This is the ceiling of the instrument.
  READ-SIDE              oracle succeeds, and the gold answer is present in the
                          composite's records. The evidence reached the model and the
                          model did not use it. Six interventions already failed here.
  WRITE-SIDE             oracle succeeds and the gold answer is ABSENT from the records.
                          Extraction lost it. This is the only bucket where record
                          hygiene can help, and it is the one worth sizing.
  ABSTENTION             held out of the split entirely. Its gold answer is a PROSE
                          STATEMENT OF NON-ANSWERABILITY ("The information provided is not
                          enough..."), which by construction never appears in a record, so
                          the containment test files every abstention error as write-side.
                          It is a calibration failure -- the model committing to an answer
                          it should have declined -- not an evidence failure, and mixing it
                          in overstated the actionable write-side bucket by 2x.
  COMPOSITE-ONLY WIN     composite succeeds where oracle fails -- reported because it
                          sizes what consolidation buys that gold retrieval cannot.

The record check is a normalised substring test, so it is a LOWER BOUND on read-side and
an UPPER BOUND on write-side: a paraphrased-but-present answer counts as absent. It is
reported as a heuristic and not as a measurement, and the split is printed per type so a
type whose answers are numeric (where the test is reliable) can be read separately from
one whose answers are prose (where it is not).
"""

import argparse
import json
import os
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "..", "reproducibility_benchmarks", "longmemeval")


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def answer_in_records(answer, record_text):
    """Normalised containment, plus an all-tokens fallback for multi-part answers."""
    a, r = norm(answer), norm(record_text)
    if not a:
        return False
    if a in r:
        return True
    toks = [t for t in a.split() if len(t) > 2]
    return bool(toks) and all(t in r for t in toks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--composite", default="composite_glm52_n500.json")
    ap.add_argument("--oracle", default="oracle_glm52_n500.json")
    ap.add_argument("--cache", default="records_composite_partial.json")
    ap.add_argument("--show", type=int, default=0, help="print N write-side examples")
    args = ap.parse_args()

    C = {r["question_id"]: r for r in json.load(open(os.path.join(RESULTS, args.composite)))["results"]}
    O = {r["question_id"]: r for r in json.load(open(os.path.join(RESULTS, args.oracle)))["results"]}
    cache = json.load(open(os.path.join(RESULTS, args.cache)))

    buckets = defaultdict(list)
    for qid, c in C.items():
        o = O.get(qid)
        if o is None:
            continue
        if c["correct"] and not o["correct"]:
            buckets["composite_only_win"].append(qid)
        elif c["correct"]:
            continue
        elif c["report_type"] == "abstention":
            buckets["abstention"].append(qid)
        elif not o["correct"]:
            buckets["floor"].append(qid)
        elif answer_in_records(c["reference_answer"], cache.get(qid, "")):
            buckets["read_side"].append(qid)
        else:
            buckets["write_side"].append(qid)

    total_err = sum(1 for r in C.values() if not r["correct"])
    print(f"{args.composite}: {len(C)-total_err}/{len(C)} correct, {total_err} errors\n")
    order = ["floor", "read_side", "write_side", "abstention", "composite_only_win"]
    label = {"floor": "BENCHMARK/READER FLOOR (oracle fails too)",
             "read_side": "READ-SIDE  (evidence in records, unused)",
             "write_side": "WRITE-SIDE (evidence absent from records)",
             "abstention": "ABSTENTION (calibration, held out of the split)",
             "composite_only_win": "composite wins where oracle fails"}
    for b in order:
        n = len(buckets[b])
        share = f"{n/total_err:5.1%} of errors" if b != "composite_only_win" else "not an error"
        print(f"  {label[b]:<45} {n:>4}   {share}")

    print("\nBY TYPE (floor / read-side / write-side)")
    types = sorted({r["report_type"] for r in C.values()})
    idx = {q: b for b in order for q in buckets[b]}
    print(f"{'report_type':<28}{'floor':>8}{'read':>8}{'write':>8}{'absten':>8}{'total err':>11}")
    for t in types:
        qs = [q for q in C if C[q]["report_type"] == t and not C[q]["correct"]]
        f = sum(1 for q in qs if idx.get(q) == "floor")
        rd = sum(1 for q in qs if idx.get(q) == "read_side")
        w = sum(1 for q in qs if idx.get(q) == "write_side")
        ab = sum(1 for q in qs if idx.get(q) == "abstention")
        print(f"{t:<28}{f:>8}{rd:>8}{w:>8}{ab:>8}{len(qs):>11}")

    for qid in buckets["write_side"][: args.show]:
        print(f"\n--- WRITE-SIDE {qid} ({C[qid]['report_type']})")
        print(f"    gold: {C[qid]['reference_answer']!r}")
        print(f"    said: {C[qid]['hypothesis'][:200]!r}")


if __name__ == "__main__":
    main()
