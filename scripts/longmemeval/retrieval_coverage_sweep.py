#!/usr/bin/env python3
"""Coverage sweep over retrieval params for the 27 verified retrieval misses.

ZERO LLM CALLS. Answers, before spending any run: would raising `base_top_k`
and/or `max_sessions` actually put the missing answer-bearing turns into the
assembled context? If a setting doesn't improve *coverage*, it cannot improve
accuracy, and there is no point running it.

Ingests each question ONCE, then rebuilds the context at every (top_k,
max_sessions) combination -- ingest is the slow part (~5s), context assembly is
~20ms, so the whole grid costs about one ingest per question.

Also reports context size at each setting, since the cost of more retrieval is
prompt size (num_ctx pressure, latency, and dilution risk on other types).

Discriminates the two sub-mechanisms found in v4-plan.md:
  * partial recall (20 questions) -- turns exist in the candidate set but are
    truncated away; MORE retrieval should recover them
  * semantic dilution (7 questions) -- the answer is a passing aside in a turn
    whose dominant topic is different, so the turn never ranks near the query;
    NO value of top_k should recover them, and if the sweep shows that, it is
    direct evidence the fix has to be embedding/chunking-side
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from answer_turn_attribution import _answer_turns, _norm, _turn_in_context  # noqa: E402
from run_benchmark import _get_encoder, _ingest_haystack  # noqa: E402

import numpy as np  # noqa: E402
from aeon_py.session_expansion import build_expanded_context, format_events  # noqa: E402
from aeon_py.trace import TraceGraph  # noqa: E402

GRID_TOP_K = [30, 60, 100, 200]
GRID_MAX_SESSIONS = [10, 15, 20]

# The 7 complete misses (0 of N answer turns retrieved), diagnosed as buried
# asides -- tracked separately because the prediction is that they do NOT
# improve with more retrieval.
DILUTION_IDS = {
    "726462e0", "5d3d2817", "gpt4_8279ba03", "gpt4_68e94288",
    "gpt4_468eb064", "gpt4_1e4a8aec", "gpt4_468eb063",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--attribution", required=True)
    ap.add_argument("--tmp-dir", default="/tmp/aeon_sweep")
    ap.add_argument("--out")
    args = ap.parse_args()

    tmp = Path(args.tmp_dir)
    tmp.mkdir(parents=True, exist_ok=True)
    ds = {q["question_id"]: q for q in json.load(open(args.dataset))}
    attr = json.load(open(args.attribution))["results"]
    miss = [r["question_id"] for r in attr if r["category"] == "retrieval_miss"]
    print(f"Sweeping {len(miss)} retrieval-miss questions over "
          f"{len(GRID_TOP_K)}x{len(GRID_MAX_SESSIONS)} settings (zero LLM calls)...",
          file=sys.stderr)

    enc = _get_encoder()
    rows = []
    t0 = time.perf_counter()
    for i, qid in enumerate(miss, 1):
        q = ds[qid]
        tp = tmp / f"{qid}.trace"
        for sfx in ("", ".blobs", ".wal"):
            Path(str(tp) + sfx).unlink(missing_ok=True)
        trace = TraceGraph(path=str(tp))
        _ingest_haystack(trace, enc, q)
        qv = np.asarray(enc.encode(q["question"]), dtype=np.float32).tolist()
        turns = _answer_turns(q)

        per_setting = {}
        for tk in GRID_TOP_K:
            for ms in GRID_MAX_SESSIONS:
                ev = build_expanded_context(trace, qv, "full_session",
                                            base_top_k=tk, max_sessions=ms)
                ctx = format_events(ev)
                nctx = _norm(ctx)
                got = sum(1 for t in turns if _turn_in_context(t["content"], nctx))
                per_setting[f"{tk}x{ms}"] = {
                    "retrieved": got, "total": len(turns), "chars": len(ctx),
                }
        rows.append({"question_id": qid, "dilution": qid in DILUTION_IDS,
                     "n_answer_turns": len(turns), "settings": per_setting})
        for sfx in ("", ".blobs", ".wal"):
            Path(str(tp) + sfx).unlink(missing_ok=True)
        if i % 5 == 0:
            el = time.perf_counter() - t0
            print(f"  {i}/{len(miss)} ({el:.0f}s)", file=sys.stderr)

    print("\n" + "=" * 90)
    print("ANSWER-TURN COVERAGE BY SETTING  (of the 27 verified retrieval misses)")
    print("=" * 90)
    partial = [r for r in rows if not r["dilution"]]
    dil = [r for r in rows if r["dilution"]]
    hdr = f"{'setting':<12}{'fully covered':>15}{'turn coverage':>15}{'partial(20)':>13}{'dilution(7)':>13}{'med chars':>11}"
    print(hdr)
    for tk in GRID_TOP_K:
        for ms in GRID_MAX_SESSIONS:
            k = f"{tk}x{ms}"
            full = sum(1 for r in rows if r["settings"][k]["retrieved"] == r["n_answer_turns"])
            cov = sum(r["settings"][k]["retrieved"] for r in rows) / sum(r["n_answer_turns"] for r in rows)
            pf = sum(1 for r in partial if r["settings"][k]["retrieved"] == r["n_answer_turns"])
            df = sum(1 for r in dil if r["settings"][k]["retrieved"] == r["n_answer_turns"])
            chars = sorted(r["settings"][k]["chars"] for r in rows)
            med = chars[len(chars) // 2]
            print(f"{k:<12}{full:>10}/{len(rows):<4}{cov*100:>14.1f}%{pf:>8}/{len(partial):<4}"
                  f"{df:>8}/{len(dil):<4}{med:>11,}")

    print("\n--- PER-QUESTION, baseline 30x10 vs best setting ---")
    best = f"{GRID_TOP_K[-1]}x{GRID_MAX_SESSIONS[-1]}"
    for r in sorted(rows, key=lambda x: (x["dilution"], x["question_id"])):
        b = r["settings"]["30x10"]; e = r["settings"][best]
        tag = "DILUTION" if r["dilution"] else "partial "
        flag = "  <-- recovered" if e["retrieved"] == r["n_answer_turns"] and b["retrieved"] < r["n_answer_turns"] else ""
        print(f"  {tag} {r['question_id']:<16} {b['retrieved']}/{r['n_answer_turns']} -> "
              f"{e['retrieved']}/{r['n_answer_turns']} @{best}{flag}")

    if args.out:
        json.dump({"grid_top_k": GRID_TOP_K, "grid_max_sessions": GRID_MAX_SESSIONS,
                   "results": rows}, open(args.out, "w"), indent=2)
        print(f"\nWrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
