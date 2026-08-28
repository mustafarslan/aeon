#!/usr/bin/env python3
"""Stage 4 gate: false-negative and false-positive rates for the identifier detector.

v4-plan.md's Stage 4 gate requires a "measured false-negative rate for the identifier-corpus
detector on a labelled corpus of real distilled fragments, published alongside the feature".
v4-plan.md:2168 records that this had not been done, because a labelled corpus of REAL
fragments does not exist for this project.

THIS CORPUS IS SYNTHETIC, AND EVERY REPORT OF IT SAYS SO. A synthetic measurement that is
honestly labelled is worth more than an unpublished gate; it is worth less than the real
thing, and substituting one for the other silently is exactly the misattribution this
project's evaluation sections criticise elsewhere. The real-corpus measurement stays
deferred alongside the Stage 3 outcome experiment, which needs a pilot org for the same
underlying reason: neither can be manufactured from code.

What the numbers mean here:

  false negative  an identifier the detector should have redacted and did not. This is the
                  gate's subject and the only one with a confidentiality consequence: a
                  missed identifier reaches the shared tier.
  false positive  a fragment with no identifier that the detector redacted anyway. Costs
                  utility, not confidentiality. Reported because a detector tuned to zero
                  false negatives by redacting everything would pass the gate and be useless.

Run:  .venv/bin/python scripts/governance/detector_fnr.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shell"))

from aeon_py.promotion import IdentifierCorpus, classify_and_redact  # noqa: E402

# The adopter-supplied half of the corpus. A real deployment supplies its own directory
# names, ticket formats and hostnames; these stand in for that shape.
ADOPTER_PATTERNS = [r"PROJ-\d+", r"INC\d{4,}", r"\bsvc-[a-z0-9-]+\.internal\b"]


@dataclass(frozen=True)
class Case:
    text: str
    identifiers: tuple[str, ...]   # substrings that must NOT survive; empty = clean
    note: str = ""


POSITIVES: list[Case] = [
    Case("deadlock in PROJ-1234 after the retry change", ("PROJ-1234",)),
    Case("see INC90210 for the postmortem", ("INC90210",)),
    Case("reported by alice@example.com on Tuesday", ("alice@example.com",)),
    Case("reverted in a1b2c3d4e5f6 and the leak stopped", ("a1b2c3d4e5f6",)),
    Case("svc-billing-api.internal timed out", ("svc-billing-api.internal",)),
    Case("PROJ-77 and bob@corp.co.uk both", ("PROJ-77", "bob@corp.co.uk"),
         "two identifiers, two categories, in one fragment"),
    Case("cc dana+ops@example.io about INC12345", ("dana+ops@example.io", "INC12345"),
         "plus-addressing is a common miss"),
    Case("merged deadbeef1234567 into main", ("deadbeef1234567",)),
    Case("PROJ-1 is the oldest ticket", ("PROJ-1",), "boundary: shortest ticket id"),
    Case("host svc-a.internal is flaky", ("svc-a.internal",), "boundary: shortest hostname"),

    # ADVERSARIAL. These are chosen to find the detector's limits rather than to
    # confirm it. A corpus that only contains cases the detector already handles
    # measures nothing. Failures here are the informative result, not a defect in
    # the corpus: they bound what deterministic shape-matching can be trusted to do.
    Case("Alice Chen owns the billing service", ("Alice Chen",),
         "a person's name matches no shape; the class docstring names this limit"),
    Case("fixed in abc123f, see the diff", ("abc123f",),
         "7 hex chars is exactly the sha threshold"),
    Case("fixed in abc123, see the diff", ("abc123",),
         "6 hex chars falls BELOW the 7-char sha threshold"),
    Case("reach me at alice dot chen at example dot com", ("alice dot chen",),
         "obfuscated address defeats the email shape"),
    Case("the box at 10.0.14.22 was unreachable", ("10.0.14.22",),
         "an IP is identifying and no pattern targets it"),
]

NEGATIVES: list[Case] = [
    Case("the retry loop deadlocks under load", ()),
    Case("increasing the pool size fixed the timeout", ()),
    Case("this is a project-wide change", (), "'project' must not trip the PROJ- pattern"),
    Case("we saw 1234 timeouts in an hour", (), "a bare number is not a ticket id"),
    Case("the incident was resolved by a rollback", (), "'incident' is not INC\\d+"),
    Case("email the team before deploying", (), "the word email is not an address"),
    Case("the internal API was slow", (), "'internal' alone is not a hostname"),
    Case("reverted the change and it recovered", ()),

    # ADVERSARIAL negatives: ordinary words that happen to be valid hex. The sha
    # pattern is shape-based, so these are the cost side of enabling it.
    Case("the cafebabe magic number in the header", (),
         "8 hex chars, but a well-known constant rather than a commit"),
    Case("deadbeef is the classic fill pattern", (),
         "8 hex chars, ordinary prose"),
]


def main() -> int:
    corpus = IdentifierCorpus(patterns=ADOPTER_PATTERNS,
                             redact_emails=True, redact_commit_shas=True)

    false_negatives, false_positives = [], []

    for case in POSITIVES:
        result = classify_and_redact(case.text, corpus)
        leaked = [i for i in case.identifiers if i in result.redacted_text]
        if leaked or not result.passed:
            # A rejected fragment is not a leak: it never reaches the shared tier. Only a
            # PASSED fragment still carrying an identifier is a false negative.
            if result.passed and leaked:
                false_negatives.append((case, leaked))

    for case in NEGATIVES:
        result = classify_and_redact(case.text, corpus)
        if "[REDACTED" in result.redacted_text:
            false_positives.append((case, result.redacted_text))

    n_pos, n_neg = len(POSITIVES), len(NEGATIVES)
    fnr = len(false_negatives) / n_pos if n_pos else 0.0
    fpr = len(false_positives) / n_neg if n_neg else 0.0

    print("Identifier-detector detection rates  **SYNTHETIC CORPUS**")
    print("=" * 62)
    print(f"  positives (contain an identifier) : {n_pos}")
    print(f"  negatives (clean)                 : {n_neg}")
    print()
    print(f"  false negatives (identifier survived a PASSED fragment): "
          f"{len(false_negatives)}/{n_pos} = {fnr*100:.1f}%")
    print(f"  false positives (clean fragment redacted)              : "
          f"{len(false_positives)}/{n_neg} = {fpr*100:.1f}%")

    for case, leaked in false_negatives:
        print(f"    FN: {case.text!r} leaked {leaked}")
    for case, got in false_positives:
        print(f"    FP: {case.text!r} -> {got!r}")

    print()
    print("This corpus is SYNTHETIC. The Stage 4 gate asks for real distilled fragments,")
    print("which do not exist for this project; that measurement stays deferred alongside")
    print("the Stage 3 outcome experiment. Report these figures as synthetic or not at all.")

    Path("reproducibility_benchmarks/detector_fnr_synthetic.json").write_text(json.dumps({
        "corpus": "synthetic",
        "n_positive": n_pos, "n_negative": n_neg,
        "false_negatives": len(false_negatives), "false_positives": len(false_positives),
        "fnr": fnr, "fpr": fpr,
        "fn_cases": [c.text for c, _ in false_negatives],
        "fp_cases": [c.text for c, _ in false_positives],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
