"""Tests for the dev/held-out split (`scripts/longmemeval/splits.py`).

The split is the instrument that makes iteration legitimate, so its properties are
pinned here rather than eyeballed once: it must be deterministic, order-independent,
a true partition, balanced per stratum, and -- the one that matters most --
outcome-blind, so a split can never be chosen to flatter a result.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "longmemeval"))

from splits import dev_heldout, select  # noqa: E402

TYPES = ("multi-session", "temporal-reasoning", "knowledge-update",
         "single-session-user", "single-session-assistant", "single-session-preference")


def _dataset(n=200):
    """Synthetic questions covering every type, with an abstention variant every 7th."""
    out = []
    for i in range(n):
        t = TYPES[i % len(TYPES)]
        qid = f"q{i:04d}" + ("_abs" if i % 7 == 0 else "")
        out.append({"question_id": qid, "question_type": t})
    return out


def test_split_is_a_partition():
    ds = _dataset()
    dev, held = dev_heldout(ds)
    ids_dev = {q["question_id"] for q in dev}
    ids_held = {q["question_id"] for q in held}
    assert ids_dev & ids_held == set()
    assert ids_dev | ids_held == {q["question_id"] for q in ds}


def test_split_is_deterministic_for_a_seed():
    ds = _dataset()
    a, _ = dev_heldout(ds, seed=42)
    b, _ = dev_heldout(ds, seed=42)
    assert [q["question_id"] for q in a] == [q["question_id"] for q in b]


def test_split_does_not_depend_on_input_order():
    """Strata are sorted by question_id before shuffling, so a dataset that loads in a
    different order must still split identically -- otherwise the 'same seed, same
    split' guarantee silently depends on file ordering."""
    ds = _dataset()
    a, _ = dev_heldout(ds, seed=42)
    b, _ = dev_heldout(list(reversed(ds)), seed=42)
    assert sorted(q["question_id"] for q in a) == sorted(q["question_id"] for q in b)


def test_different_seeds_give_different_splits():
    ds = _dataset()
    a, _ = dev_heldout(ds, seed=42)
    b, _ = dev_heldout(ds, seed=7)
    assert {q["question_id"] for q in a} != {q["question_id"] for q in b}


def test_abstention_is_balanced_across_halves():
    """Abstention carries the standing collateral breach and is only 30 of 500. A split
    that piled them into one half would make that guard unmeasurable on the other."""
    ds = _dataset()
    dev, held = dev_heldout(ds)
    n_dev = sum(1 for q in dev if "_abs" in q["question_id"])
    n_held = sum(1 for q in held if "_abs" in q["question_id"])
    assert abs(n_dev - n_held) <= 1


def test_known_miss_cohort_is_balanced_across_halves():
    ds = _dataset()
    misses = [q["question_id"] for q in ds[:27]]
    dev, held = dev_heldout(ds, miss_ids=misses)
    n_dev = sum(1 for q in dev if q["question_id"] in set(misses))
    assert abs(n_dev - (len(misses) - n_dev)) <= len(TYPES) * 2


def test_every_stratum_is_balanced_to_within_one():
    """The guarantee is per STRATUM -- (report_type, is_known_miss) -- because that is
    what the alternating assignment operates on."""
    ds = _dataset()
    dev, held = dev_heldout(ds)

    def strata(qs):
        out = {}
        for q in qs:
            key = "abstention" if "_abs" in q["question_id"] else q["question_type"]
            out[key] = out.get(key, 0) + 1
        return out

    a, b = strata(dev), strata(held)
    for key in set(a) | set(b):
        assert abs(a.get(key, 0) - b.get(key, 0)) <= 1, key


def test_question_type_is_balanced_to_within_the_strata_it_spans():
    """A question_type spans TWO strata -- its own, plus the abstention stratum its
    `_abs` variants were split into -- so it can differ by 2, not 1. Asserting <=1 here
    would be asserting a guarantee the design does not make (on the real 500 this shows
    up as single-session-preference 16 vs 14)."""
    ds = _dataset()
    dev, held = dev_heldout(ds)
    for t in TYPES:
        a = sum(1 for q in dev if q["question_type"] == t)
        b = sum(1 for q in held if q["question_type"] == t)
        assert abs(a - b) <= 2, t


def test_split_is_outcome_blind():
    """dev_heldout() takes only ids and types. It cannot see whether a question was
    answered correctly, which is what stops a split from being chosen to flatter a
    result. Pinned as a test because it is a methodology guarantee, not an
    implementation detail."""
    ds = _dataset()
    a, _ = dev_heldout(ds)
    for q in ds:                      # an outcome field must not change the split
        q["correct"] = q["question_id"].endswith("2")
    b, _ = dev_heldout(ds)
    assert [q["question_id"] for q in a] == [q["question_id"] for q in b]


def test_select_all_is_the_identity():
    """Every invocation predating --split must reproduce exactly."""
    ds = _dataset()
    assert [q["question_id"] for q in select(ds, "all")] == [q["question_id"] for q in ds]


def test_select_rejects_an_unknown_split():
    with pytest.raises(ValueError):
        select(_dataset(), "test")
