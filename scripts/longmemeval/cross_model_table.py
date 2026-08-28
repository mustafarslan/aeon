#!/usr/bin/env python3
"""Recompute the cross-model matrix from committed result files.

WHY THIS IS A SCRIPT AND NOT A SPREADSHEET. Every number that enters
PAPER_V4_FINDINGS.md must be derivable from a committed result file by the same
command that prints it -- that is the standard §1 already holds itself to, and
this sprint produced at least one figure (429 vs 401) where two same-day files
differ by a reverted intervention. Quoting from memory is how the wrong one gets
cited.

The four arms are a 2x2: {consolidated records, gold raw evidence} x
{gemma4:31b, glm-5.2}. The oracle arms are handed the answer-bearing turns via
the dataset's `has_answer` annotation; the composite arms use only inputs that
exist in production. Judging is the official `anscheck` prompt under
gemma4:31b-cloud in every cell, so the reader swap is the only variable between
columns.
"""

import argparse
import json
import math
import os
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "..", "reproducibility_benchmarks", "longmemeval")

# (label, reader, arm, filename). composite_arm_n500.json is v1 -- `_v2` is the
# reverted abstention fix (commit 7539192), not a better measurement of the same thing.
ARMS = [
    ("composite@gemma4", "gemma4:31b", "composite", "composite_arm_n500.json"),
    ("oracle@gemma4",    "gemma4:31b", "oracle",    "oracle_precision_n500_results.json"),
    ("composite@glm5.2", "glm-5.2",    "composite", "composite_glm52_n500.json"),
    ("oracle@glm5.2",    "glm-5.2",    "oracle",    "oracle_glm52_n500.json"),
]

# Paired contrasts worth reporting, as (a, b): does a differ from b?
CONTRASTS = [
    ("composite@gemma4", "oracle@gemma4"),    # can records reach the gold ceiling?
    ("composite@glm5.2", "oracle@glm5.2"),    # ...does that survive a stronger reader?
    ("composite@glm5.2", "composite@gemma4"), # what does the reader swap alone buy?
    ("oracle@glm5.2",    "oracle@gemma4"),    # ...and how much of that is reader-only?
]


def load(fname):
    path = os.path.join(RESULTS, fname)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        doc = json.load(fh)
    return {r["question_id"]: r for r in doc["results"]}, doc


def mcnemar(a, b):
    """Two-sided exact binomial on the discordant pairs.

    The exact test rather than the chi-square approximation: several contrasts here
    have discordant counts in the tens, where the approximation is not trustworthy,
    and a p-value that flips with the choice of test is not one to build a claim on.
    """
    common = sorted(set(a) & set(b))
    wins = sum(1 for q in common if a[q]["correct"] and not b[q]["correct"])
    loss = sum(1 for q in common if b[q]["correct"] and not a[q]["correct"])
    n = wins + loss
    if n == 0:
        return wins, loss, 1.0, len(common)
    k = min(wins, loss)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return wins, loss, min(1.0, 2 * tail), len(common)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true",
                    help="score whatever arms exist, on their intersection, "
                         "rather than refusing an incomplete matrix")
    args = ap.parse_args()

    loaded, docs = {}, {}
    for label, reader, arm, fname in ARMS:
        got = load(fname)
        if got is None:
            print(f"  (missing: {fname} -> {label})")
            continue
        loaded[label], docs[label] = got

    missing = [a[0] for a in ARMS if a[0] not in loaded]
    if missing and not args.partial:
        raise SystemExit(f"incomplete matrix, missing {missing}; pass --partial to score anyway")

    print("=" * 78)
    print("CROSS-MODEL MATRIX -- LongMemEval-S, n=500, judge gemma4:31b-cloud")
    print("=" * 78)
    print(f"{'arm':<20}{'n':>6}{'correct':>9}{'acc':>8}{'ctx chars':>12}{'gen s':>9}{'err':>6}")
    for label, reader, arm, fname in ARMS:
        if label not in loaded:
            continue
        rows = list(loaded[label].values())
        ok = sum(1 for r in rows if r["correct"])
        ctx = [r.get("prompt_chars") or r.get("context_chars") or 0 for r in rows]
        # `timing_seconds` is a dict; the composite also carries an `extract` leg, but
        # only `generation` is comparable across arms -- the oracle has no extract phase.
        gen = [r["timing_seconds"]["generation"] for r in rows
               if isinstance(r.get("timing_seconds"), dict)
               and r["timing_seconds"].get("generation")]
        errs = sum(1 for r in rows if r.get("is_error"))
        print(f"{label:<20}{len(rows):>6}{ok:>9}{ok/len(rows):>8.3f}"
              f"{statistics.median(ctx):>12,.0f}{statistics.median(gen):>9.2f}{errs:>6}")

    print()
    print("PER-TYPE (correct / n)")
    types = sorted({r["report_type"] for rows in loaded.values() for r in rows.values()})
    present = [a[0] for a in ARMS if a[0] in loaded]
    print(f"{'report_type':<28}" + "".join(f"{p:>19}" for p in present))
    for t in types:
        cells = []
        for p in present:
            rows = [r for r in loaded[p].values() if r["report_type"] == t]
            ok = sum(1 for r in rows if r["correct"])
            cells.append(f"{ok:>3}/{len(rows):<3}")
        print(f"{t:<28}" + "".join(f"{c:>19}" for c in cells))

    print()
    print("PAIRED CONTRASTS (McNemar, exact two-sided)")
    for x, y in CONTRASTS:
        if x not in loaded or y not in loaded:
            print(f"  {x} vs {y}: skipped (arm missing)")
            continue
        w, l, p, n = mcnemar(loaded[x], loaded[y])
        nx = sum(1 for r in loaded[x].values() if r["correct"])
        ny = sum(1 for r in loaded[y].values() if r["correct"])
        verdict = "significant" if p < 0.05 else "not significant"
        print(f"  {x} vs {y}: {nx} vs {ny} (net {nx-ny:+d}) "
              f"| +{w}/-{l} discordant | p={p:.3g} [{verdict}] | paired on {n}")


if __name__ == "__main__":
    main()
