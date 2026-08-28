#!/usr/bin/env python3
"""Re-judge a committed arm's hypotheses under several judges. No generation, judging only.

WHY. Every number this project publishes is graded by `gemma4:31b-cloud`, while every published
LongMemEval result it is compared against is graded by a frontier judge -- `gpt-4o` (Mastra,
Mem0, and the benchmark's own specification), GPT-4.1 (OMEGA), Gemini 3 Flash (Exabase). If a
31B judge is systematically stricter, some unknown part of the reported gap is judge severity
rather than system quality. That confound is stated in `spatiotemporal.md` §2 and, until this
runs, undismissed.

WHAT THIS CAN AND CANNOT DO. No GPT-4o is reachable here, so this CANNOT convert the headline
into a leaderboard-comparable number. What it can do is measure **dispersion**: if judges from
three unrelated model families agree within a question or two, the confound is small and
everyone's leaderboard numbers are robust to it. If they disagree by fifteen, the confound is
large -- and that is a finding about the benchmark, not just about Aeon.

THE SELF-JUDGE CELL IS THE POINT, NOT A FLAW. Judging an arm with the model that generated it
is exactly the configuration §8c.1 calls disqualifying, and which Exabase M-1 and OMEGA ship.
Running it deliberately, beside third-party judges on the same hypotheses, is what turns
"self-judging is bad" from an assertion into a measurement. Those cells are flagged SELF in
every output.

THE HEADLINE DOES NOT MOVE. 89.6% is and stays the `gemma4`-judged figure. A friendlier judge
discovered by a check proposed after seeing the gap is metric-shopping with extra steps; the
directional flips are reported both ways precisely so that swapping cannot be done quietly.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from judge_prompts import get_anscheck_prompt              # noqa: E402
from run_benchmark import _generate_with_retry             # noqa: E402
from aeon_py.llm import OllamaProvider                     # noqa: E402
from aeon_py.parallel import ThreadLocalResource, parallel_map  # noqa: E402

DEFAULT_DS = "/Volumes/AI_SSD/hf-cache/longmemeval/longmemeval_s_cleaned.json"


def judge_one(judge_model: str, providers: ThreadLocalResource, q: dict, hyp: str) -> bool:
    """One verdict, using the official `get_anscheck_prompt` unchanged.

    An error marker counts as NOT correct rather than being dropped: a judge that failed to
    answer is a missing measurement, and silently excluding it would shrink the denominator
    for whichever judge happened to be flakiest -- the §8c.5 lesson, one layer up.
    """
    llm = providers.get()
    llm.model = judge_model
    prompt = get_anscheck_prompt(q["question_type"], q["question"], q["answer"], hyp,
                                 abstention="_abs" in q["question_id"])
    out = _generate_with_retry(llm, prompt, system_prompt="", temperature=0.0)
    return out.strip().lower().startswith("yes")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="committed result file to re-judge")
    ap.add_argument("--judges", nargs="+", required=True)
    ap.add_argument("--dataset", default=DEFAULT_DS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    arm = json.load(open(args.arm))
    reader = arm.get("model", "?")
    incumbent = arm.get("judge_model") or "gemma4:31b-cloud (assumed; not recorded in file)"
    ds = {q["question_id"]: q for q in json.load(open(args.dataset))}

    rows = [r for r in arm["results"] if not r.get("is_error")]
    if args.limit:
        rows = rows[:args.limit]
    print(f"arm      : {Path(args.arm).name}")
    print(f"reader   : {reader}")
    print(f"incumbent judge: {incumbent}")
    print(f"questions: {len(rows)}   judges: {', '.join(args.judges)}\n", flush=True)

    providers = ThreadLocalResource(OllamaProvider)
    verdicts: dict[str, dict[str, bool]] = {}
    for jm in args.judges:
        tag = "  <-- SELF-JUDGE" if jm.split(":")[0] in reader else ""
        print(f"judging under {jm}{tag}", flush=True)
        out = parallel_map(
            lambda r: judge_one(jm, providers, ds[r["question_id"]], r["hypothesis"] or ""),
            rows,
            max_workers=args.workers,
            on_error=lambda r, e: False,
            progress=lambda done, total: (
                print(f"    {done}/{total}", flush=True) if done % 50 == 0 else None))
        verdicts[jm] = {r["question_id"]: bool(v) for r, v in zip(rows, out)}
        n = sum(verdicts[jm].values())
        print(f"  -> {n}/{len(rows)} = {n/len(rows)*100:.1f}%\n", flush=True)

    base = {r["question_id"]: bool(r["correct"]) for r in rows}
    ids = [r["question_id"] for r in rows]

    print("=" * 74)
    print(f"{'judge':<28} {'correct':>9} {'acc':>7}   vs incumbent (no->yes / yes->no)")
    print(f"{incumbent[:28]:<28} {sum(base.values()):>9} "
          f"{sum(base.values())/len(ids)*100:>6.1f}%   (incumbent)")
    summary = []
    for jm in args.judges:
        v = verdicts[jm]
        lenient = sum(1 for i in ids if v[i] and not base[i])
        strict = sum(1 for i in ids if base[i] and not v[i])
        selfj = jm.split(":")[0] in reader
        print(f"{jm:<28} {sum(v.values()):>9} {sum(v.values())/len(ids)*100:>6.1f}%   "
              f"+{lenient} / -{strict}  net {lenient-strict:+d}"
              f"{'   [SELF-JUDGE]' if selfj else ''}")
        summary.append({"judge": jm, "correct": sum(v.values()), "n": len(ids),
                        "no_to_yes": lenient, "yes_to_no": strict,
                        "net_vs_incumbent": lenient - strict, "self_judge": selfj})

    print("\npairwise agreement (all judges incl. incumbent):")
    allj = {incumbent: base} | verdicts
    names = list(allj)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            agree = sum(1 for q in ids if allj[a][q] == allj[b][q])
            print(f"  {a[:24]:<24} vs {b[:24]:<24} {agree}/{len(ids)} = {agree/len(ids)*100:.1f}%")

    unanimous = sum(1 for q in ids if len({allj[j][q] for j in names}) == 1)
    print(f"\nunanimous across all {len(names)} judges: {unanimous}/{len(ids)} "
          f"= {unanimous/len(ids)*100:.1f}%   (contested: {len(ids)-unanimous})")
    spread = [sum(allj[j][q] for j in names) for q in ids]
    print("questions by number of judges calling them correct: "
          + ", ".join(f"{k}:{v}" for k, v in sorted(Counter(spread).items())))

    Path(args.out).write_text(json.dumps(
        {"arm": str(args.arm), "reader": reader, "incumbent_judge": incumbent,
         "n": len(ids), "incumbent_correct": sum(base.values()), "judges": summary,
         "unanimous": unanimous,
         "per_question": {q: {"incumbent": base[q],
                              **{j: verdicts[j][q] for j in args.judges}} for q in ids}},
        indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
