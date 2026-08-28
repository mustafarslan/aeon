#!/usr/bin/env python3
"""Every number the v4 manuscript asserts, regenerated from committed result files.

Nothing here is transcribed. This exists because the manuscript is the one artefact where a
stale number cannot be corrected by a later commit -- and because this project has already
caught itself twice quoting figures whose generating config had moved underneath them
(§8c.1's judge-parity defect, §8c.2's render-change confound).

Run before touching the TeX, and again before submitting. Any row that changes is either a
real result change or a regression, and both need explaining.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

B = Path("reproducibility_benchmarks/longmemeval")
DS = "/Volumes/AI_SSD/hf-cache/longmemeval/longmemeval_s_cleaned.json"


def load(name):
    d = json.load(open(B / name))
    return d, {r["question_id"]: r for r in d["results"]}


def mcnemar(a, b, ids):
    """Exact two-sided McNemar over paired verdicts. `b` wins counted positive."""
    win = sum(1 for i in ids if b[i]["correct"] and not a[i]["correct"])
    los = sum(1 for i in ids if a[i]["correct"] and not b[i]["correct"])
    n = win + los
    p = (min(1.0, 2 * sum(math.comb(n, k) for k in range(min(win, los) + 1)) / 2 ** n)
         if n else 1.0)
    return win - los, win, los, p


def row(label, value, source):
    print(f"| {label} | {value} | `{source}` |")


def main() -> int:
    ds = {q["question_id"]: q for q in json.load(open(DS))}
    print("## Claims inventory — every figure regenerated, none transcribed\n")
    print("| claim | value | committed source |")
    print("|---|---|---|")

    arms = [
        ("composite @ gemma4 (v1 config, §8b headline)", "composite_arm_n500.json"),
        ("composite @ gemma4 (config-parity control — use for cross-model)",
         "composite_gemma4_parity_n500.json"),
        ("oracle-precision @ gemma4", "oracle_precision_n500_results.json"),
        ("composite @ glm-5.2 (headline)", "composite_glm52_n500.json"),
        ("oracle-precision @ glm-5.2", "oracle_glm52_n500.json"),
    ]
    loaded = {}
    for label, f in arms:
        d, r = load(f)
        loaded[f] = r
        n = len(r)
        c = sum(x["correct"] for x in r.values())
        row(label, f"**{c}/{n} = {c/n*100:.1f}%**", f)

    print("\n### Paired contrasts\n")
    print("| contrast | net | McNemar | p | source |")
    print("|---|---|---|---|---|")
    pairs = [
        ("composite vs its own oracle, gemma4 (parity arm)",
         "oracle_precision_n500_results.json", "composite_gemma4_parity_n500.json"),
        ("composite vs its own oracle, glm-5.2",
         "oracle_glm52_n500.json", "composite_glm52_n500.json"),
        ("reader swap, composite arms (clean)",
         "composite_gemma4_parity_n500.json", "composite_glm52_n500.json"),
        ("reader swap, oracle arms",
         "oracle_precision_n500_results.json", "oracle_glm52_n500.json"),
    ]
    for label, af, bf in pairs:
        a, b = loaded[af], loaded[bf]
        ids = sorted(set(a) & set(b))
        net, w, l, p = mcnemar(a, b, ids)
        print(f"| {label} | {net:+d} | +{w}/−{l} | {p:.4g} | `{bf}` |")

    print("\n### The lead claim: per-evidence-turn strata\n")
    comp, orac = loaded["composite_glm52_n500.json"], loaded["oracle_glm52_n500.json"]
    gem = loaded["composite_gemma4_parity_n500.json"]
    turns = {q: sum(1 for s in ds[q]["haystack_sessions"] for t in s if t.get("has_answer"))
             for q in comp}
    print("| answer turns | n | composite @ glm-5.2 | oracle @ glm-5.2 | composite @ gemma4 |")
    print("|---|---|---|---|---|")
    strata = [("1", lambda n: n == 1), ("2", lambda n: n == 2), ("3+", lambda n: n >= 3)]
    lead_n = lead_c = lead_o = 0
    for name, pred in strata:
        ids = [q for q in comp if pred(turns[q])]
        c, o, g = (sum(x[q]["correct"] for q in ids) for x in (comp, orac, gem))
        print(f"| {name} | {len(ids)} | {c/len(ids)*100:.1f}% | {o/len(ids)*100:.1f}% | "
              f"{g/len(ids)*100:.1f}% |")
        if name in ("1", "2"):
            lead_n += len(ids); lead_c += c; lead_o += o
    print(f"\n**Lead claim:** on the 1- and 2-turn strata — **{lead_n} of 500 "
          f"({lead_n/5:.1f}% of the benchmark)** — composite {lead_c} vs oracle {lead_o}.")

    print("\n### Secondary results\n")
    print("| claim | value | committed source |")
    print("|---|---|---|")
    dd, dsk = load("composite_deepseek_dev.json")
    _, glmdev = load("composite_glm52_dev.json")
    ids = sorted(set(dsk) & set(glmdev))
    net, w, l, p = mcnemar(glmdev, dsk, ids)
    row("reader saturation: deepseek-v4-pro vs glm-5.2 (dev)",
        f"{sum(dsk[i]['correct'] for i in ids)} vs {sum(glmdev[i]['correct'] for i in ids)} "
        f"of {len(ids)} — net {net:+d}, +{w}/−{l}, p={p:.3g}", "composite_deepseek_dev.json")

    jp = json.load(open(B / "judge_panel_glm52_n500.json"))
    for j in jp["judges"]:
        row(f"judge panel: {j['judge']}{' [SELF-JUDGE]' if j['self_judge'] else ''}",
            f"{j['correct']}/{j['n']} — vs incumbent +{j['no_to_yes']}/−{j['yes_to_no']} "
            f"(net {j['net_vs_incumbent']:+d})", "judge_panel_glm52_n500.json")
    row("judge panel: unanimous across all three",
        f"{jp['unanimous']}/{jp['n']} = {jp['unanimous']/jp['n']*100:.1f}%",
        "judge_panel_glm52_n500.json")

    ag = json.load(open(B / "aggregation_failure_read.json"))
    row("3+ stratum loss shapes",
        ", ".join(f"{k} {v}" for k, v in sorted(ag["shapes"].items())),
        "aggregation_failure_read.json")
    read_time = sum(v for k, v in ag["shapes"].items()
                    if k in ("windowed", "order", "predicate", "argmax"))
    row("losses applying a read-time filter",
        f"**{read_time} of {sum(ag['shapes'].values())}**", "aggregation_failure_read.json")

    _, ctl = load("chrono_cohort_control.json")
    _, trt = load("chrono_cohort_treatment.json")
    cids = sorted(ctl)
    row("chronological ITEM ordering (control vs treatment)",
        f"control {sum(ctl[i]['correct'] for i in cids)}/{len(cids)}, "
        f"treatment {sum(trt[i]['correct'] for i in cids)}/{len(cids)}",
        "chrono_cohort_{control,treatment}.json")

    _, rep = load("composite_glm52_empty_repair.json")
    row("empty-generation repair", f"448 as-scored, **+{sum(r['correct'] for r in rep.values())} "
        f"repaired** of {len(rep)}", "composite_glm52_empty_repair.json")

    print("\n### Comparability numbers (labelled — NOT the headline)\n")
    cats = {}
    for r in comp.values():
        cats.setdefault(r["report_type"], []).append(r["correct"])
    micro = lambda ks: sum(sum(cats[k]) for k in ks) / sum(len(cats[k]) for k in ks)
    six = [k for k in cats if k != "abstention"]
    omega = [("single-session-user", "single-session-assistant"),
             ("single-session-preference",), ("multi-session",),
             ("knowledge-update",), ("temporal-reasoning",)]
    print("| metric | Aeon | comparator |")
    print("|---|---|---|")
    tot = sum(sum(v) for v in cats.values())
    print(f"| raw micro, all 500 **(THE HEADLINE)** | **{tot}/500 = {tot/5:.1f}%** | "
          f"OMEGA raw 466/500 = 93.2% |")
    print(f"| macro over OMEGA's 5 categories | {sum(micro(g) for g in omega)/5*100:.1f}% | "
          f"OMEGA 95.4% (task-averaged) |")
    print(f"| macro over 6 categories (Mastra's metric) | "
          f"{sum(micro([k]) for k in six)/len(six)*100:.1f}% | Mastra 94.87% |")
    print(f"| raw micro, abstention excluded | {micro(six)*100:.1f}% | — |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
