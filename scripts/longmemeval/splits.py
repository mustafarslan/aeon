#!/usr/bin/env python3
"""Dev / held-out splits for the LongMemEval-S composite work.

WHY THIS EXISTS. Until now every read-path change was gated by "one attempt, no
iteration" with bars on all 500 questions. That is honest but it burns an idea per
attempt, and it makes refinement indistinguishable from fitting the test set -- which
is exactly what stalled the abstention work (v4-plan.md: the premise guard fixed its
slice 22->30 and cost 28 questions, and the protocol forbade a second try).

The replacement, following Regimes (arXiv:2606.10241), is seeded held-out gating:
iterate freely on `dev`, and let `heldout` decide. Anything published is still a full
500 re-answer, so the headline number never comes from a half.

STRATIFICATION is on `(report_type, is_known_miss)`, not on `question_type` alone as
`run_benchmark._stratified_sample()` does. Two reasons, both learned from measurements
already in the plan:

  * `report_type` splits `abstention` out of its base type. Abstention is 30 of 500 and
    carries the standing collateral breach; a split that put 25 of them in one half
    would make that guard unmeasurable on the other.
  * `is_known_miss` keeps the 27 verified retrieval misses -- the cohort the semantic
    layer exists to convert, currently 22/27 -- balanced across halves.

DETERMINISM. Strata are sorted by question_id before shuffling, so the split depends on
the seed alone and not on the order the dataset happened to load in. Assignment inside
a stratum alternates after the shuffle, which keeps each stratum split within one
question rather than binomially wobbly.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, Optional, Sequence

DEFAULT_SEED = 42


def _report_type(q: dict) -> str:
    """Abstention questions report under their own bucket, per the benchmark's own
    convention (`_abs` in the id), not under the type they were derived from."""
    return "abstention" if "_abs" in q["question_id"] else q["question_type"]


def load_miss_ids(attribution_path: str | Path) -> list[str]:
    """The 27 verified retrieval misses, from the committed attribution run."""
    rows = json.load(open(attribution_path))["results"]
    return [r["question_id"] for r in rows if r["category"] == "retrieval_miss"]


def dev_heldout(questions: Sequence[dict], seed: int = DEFAULT_SEED,
                miss_ids: Optional[Iterable[str]] = None) -> tuple[list[dict], list[dict]]:
    """Splits `questions` into two stratified halves. Returns `(dev, heldout)`."""
    misses = set(miss_ids or ())
    strata: dict[tuple[str, bool], list[dict]] = {}
    for q in questions:
        key = (_report_type(q), q["question_id"] in misses)
        strata.setdefault(key, []).append(q)

    rng = random.Random(seed)
    dev: list[dict] = []
    heldout: list[dict] = []
    for key in sorted(strata):
        bucket = sorted(strata[key], key=lambda q: q["question_id"])
        rng.shuffle(bucket)
        # Alternate rather than slice in half: a stratum of odd size then differs by
        # exactly one, and which half gets the extra rotates with the stratum.
        for i, q in enumerate(bucket):
            (dev if i % 2 == 0 else heldout).append(q)
    return dev, heldout


def select(questions: Sequence[dict], split: str, seed: int = DEFAULT_SEED,
           miss_ids: Optional[Iterable[str]] = None) -> list[dict]:
    """`split` is one of `dev`, `heldout`, `all`. `all` is the identity, so existing
    callers that never pass a split keep their exact behaviour."""
    if split == "all":
        return list(questions)
    dev, heldout = dev_heldout(questions, seed, miss_ids)
    if split == "dev":
        return dev
    if split == "heldout":
        return heldout
    raise ValueError(f"unknown split {split!r} (expected dev, heldout or all)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Write the dev/held-out split manifest.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--attribution",
                    default="reproducibility_benchmarks/longmemeval/answer_turn_attribution.json")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", default="reproducibility_benchmarks/longmemeval/splits.json")
    args = ap.parse_args()

    ds = json.load(open(args.dataset))
    misses = load_miss_ids(args.attribution)
    dev, heldout = dev_heldout(ds, args.seed, misses)

    def _breakdown(qs):
        out: dict[str, int] = {}
        for q in qs:
            out[_report_type(q)] = out.get(_report_type(q), 0) + 1
        return dict(sorted(out.items()))

    manifest = {
        "seed": args.seed,
        "n_total": len(ds),
        "dev": {"n": len(dev), "by_type": _breakdown(dev),
                "known_miss": sum(1 for q in dev if q["question_id"] in set(misses)),
                "question_ids": sorted(q["question_id"] for q in dev)},
        "heldout": {"n": len(heldout), "by_type": _breakdown(heldout),
                    "known_miss": sum(1 for q in heldout if q["question_id"] in set(misses)),
                    "question_ids": sorted(q["question_id"] for q in heldout)},
    }
    Path(args.out).write_text(json.dumps(manifest, indent=2))

    print(f"seed={args.seed}  total={len(ds)}  dev={len(dev)}  heldout={len(heldout)}")
    print(f"{'type':30s}{'dev':>6}{'heldout':>9}")
    for t in sorted(set(manifest['dev']['by_type']) | set(manifest['heldout']['by_type'])):
        print(f"{t:30s}{manifest['dev']['by_type'].get(t, 0):>6}"
              f"{manifest['heldout']['by_type'].get(t, 0):>9}")
    print(f"{'known-miss cohort':30s}{manifest['dev']['known_miss']:>6}"
          f"{manifest['heldout']['known_miss']:>9}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
