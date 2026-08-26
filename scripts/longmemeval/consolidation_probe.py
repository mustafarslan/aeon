#!/usr/bin/env python3
"""Consolidation probe -- schema-validation step for Aeon's semantic half.

Pre-registered in v4-plan.md before running. Tests whether QUERY-BLIND write-time
consolidation answers questions that perfect raw-turn retrieval measurably cannot.

Cohort: the 18 questions wrong under the oracle AND extract-then-compute AND single-shot.
The oracle had the gold evidence in hand and still failed, so no retrieval improvement can
fix them -- any conversion here is attributable to consolidation, not to better search.

The extractor NEVER sees the question. That is the point, and the named risk: ETC's
extraction was query-conditioned ("facts relevant to this question") and won +23; a real
write-time consolidator has to anticipate instead. The mitigating hypothesis is that the
dominant failures are entity-state, which a generic schema covers.

Schema (from the failure->requirement table in v4-plan.md):
    FACT:              durable attribute / possession / relationship
    EVENT [date]:      something that happened, dated       -> temporal arithmetic
    ITEM(category):    one member of a countable collection -> aggregation miscounts
    UPDATE:            a statement revising an earlier one  -> knowledge-update
    PREF:              stated preference
    TASK:              an open obligation / pending action   -> dynamic state

TASK was added AFTER inspecting the first smoke case, and that is disclosed rather than
quietly folded in. Justification for why it is schema design and not cohort overfitting:
"things the user still has to do" is a category every assistant memory needs independently
of this benchmark, and it maps directly onto LongMemEval-v2's `dynamic-environment` type.
The case that motivated it ("how many items of clothing do I need to pick up or return")
also exposed a real limit: ITEM(clothing) captured clothing the user OWNS, while the
question asked about a pending-obligation category -- query-blind extraction cannot
anticipate every category a question will slice by, which is the named risk made concrete.

Also observed on that case and worth recording: the model answered "2 items (blazer +
boots)" against a gold of 3, where the three annotated evidence turns describe two physical
items and three obligations. Some oracle-failed questions are granularity-ambiguous rather
than solvable, so this cohort is not fully winnable by any architecture.

ITEM(category) is load-bearing. The oracle's counting failures ("how many albums have I
purchased or downloaded?" gold 3, oracle answered 2) happen because counting is a multi-hop
operation over scattered raw mentions. One ITEM line per member turns counting into
enumeration of a list that already exists.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_prompts import get_anscheck_prompt  # noqa: E402
from precision_selector import build_index, select  # noqa: E402
from run_benchmark import (  # noqa: E402
    _generate_with_retry, _get_encoder, format_question_with_date,
)

import numpy as np  # noqa: E402

EXTRACT_SYSTEM = (
    "You maintain a durable long-term memory record about a user. You are shown ONE "
    "conversation session at a time and you do NOT know what will be asked later, so you "
    "must record anything that could matter. Be exhaustive and literal."
)

EXTRACT_PROMPT = (
    "Session date: {date}\n\n"
    "Conversation:\n{session}\n\n"
    "Write memory records about the USER, one per line, using exactly these prefixes:\n"
    "  FACT: <a durable attribute, possession, or relationship>\n"
    "  EVENT [{date}]: <something that happened, with its date>\n"
    "  ITEM(<category>): <one member of a countable collection>\n"
    "  UPDATE: <a statement that revises something stated earlier, say what it replaces>\n"
    "  PREF: <a stated preference>\n"
    "  TASK: <something the user still needs to do, return, pick up, finish, or decide>\n\n"
    "Rules:\n"
    "- Record only what the user states or clearly implies about themselves or their life.\n"
    "- If the user mentions several members of a countable category (things bought, fixed, "
    "watched, owned, attended, people in a group), emit ONE ITEM line for EACH member, and "
    "name the category the same way every time so members accumulate across sessions.\n"
    "- Always include dates when they are stated or implied.\n"
    "- Prefer many short specific records over few general ones.\n"
    "- Record an obligation as BOTH a TASK line and, when it belongs to a countable group, "
    "an ITEM line -- a question may count either what the user owns or what they still owe.\n"
    "- If the session contains nothing durable about the user, output exactly: (none)\n\n"
    "Records:"
)

ANSWER_SYSTEM = (
    "You are answering a question using a long-term memory record about the user. If the "
    "records genuinely do not contain the information, say so plainly instead of guessing. "
    "Answer as concisely as possible -- a short phrase or sentence, not a paragraph."
)

ANSWER_RECORDS = (
    "Long-term memory records about the user:\n{records}\n\n"
    "{question_block}\n\n"
    "Answer using the records above. If the question asks how many, COUNT the relevant "
    "records and show the count.\nAnswer:"
)

ANSWER_COMPOSITE = (
    "Long-term memory records about the user:\n{records}\n\n"
    "Relevant conversation excerpts:\n{episodic}\n\n"
    "{question_block}\n\n"
    "Answer using the records and excerpts above. If the question asks how many, COUNT the "
    "relevant records and show the count.\nAnswer:"
)


def session_text(turns: list[dict]) -> str:
    return "\n".join(f"{t['role']}: {t['content']}" for t in turns)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--model", default="gemma4:31b-cloud")
    ap.add_argument("--out", required=True)
    ap.add_argument("--records-cache", default=None,
                    help="Reuse/persist extracted records so re-running the ANSWER arms "
                         "does not repeat the expensive query-blind extraction pass.")
    ap.add_argument("--episodic-budget", type=int, default=6000)
    args = ap.parse_args()

    import os
    os.environ["AEON_LLM_MODEL"] = args.model
    from aeon_py.llm import OllamaProvider
    llm = OllamaProvider()

    ds = {q["question_id"]: q for q in json.load(open(args.dataset))}
    cohort = json.load(open(args.cohort))
    cache = {}
    if args.records_cache and Path(args.records_cache).exists():
        cache = json.load(open(args.records_cache))
        print(f"loaded cached records for {len(cache)} questions", flush=True)

    enc = _get_encoder()
    warm = _generate_with_retry(llm, "Say OK.", retries=5)
    print(f"warm-up: {warm[:30]!r} | cohort {len(cohort)}", flush=True)

    results = []
    for qi, qid in enumerate(cohort, 1):
        q = ds[qid]
        # ---- query-blind consolidation pass ----
        if qid in cache:
            records = cache[qid]
            t_extract = 0.0
        else:
            t0 = time.perf_counter()
            lines = []
            for si, (date, turns) in enumerate(zip(q["haystack_dates"],
                                                   q["haystack_sessions"])):
                prompt = EXTRACT_PROMPT.format(date=date, session=session_text(turns))
                out = _generate_with_retry(llm, prompt, system_prompt=EXTRACT_SYSTEM,
                                           temperature=0.0)
                if "[System Error:" in out:
                    continue
                for ln in out.splitlines():
                    ln = ln.strip()
                    if ln and ln != "(none)" and not ln.lower().startswith("records:"):
                        lines.append(ln)
                if (si + 1) % 15 == 0:
                    print(f"    [{qi}/{len(cohort)}] {qid} extracted {si+1}/"
                          f"{len(q['haystack_sessions'])} sessions", flush=True)
            records = "\n".join(lines)
            t_extract = time.perf_counter() - t0
            cache[qid] = records
            if args.records_cache:
                json.dump(cache, open(args.records_cache, "w"))

        # ---- episodic component for the composite arm ----
        idx = build_index(q, enc, max_chunk_chars=400)
        qv = np.asarray(enc.encode(q["question"]), dtype=np.float32)
        epi = select(q, idx, qv, design="B", budget_chars=args.episodic_budget,
                     stitch=1, stitch_mode="post")

        row = {"question_id": qid, "question_type": q["question_type"],
               "reference_answer": q["answer"], "record_chars": len(records),
               "record_lines": records.count("\n") + 1 if records else 0,
               "episodic_chars": epi["chars"],
               "extract_seconds": t_extract, "arms": {}}

        for arm, tmpl in (("R", ANSWER_RECORDS), ("R+E", ANSWER_COMPOSITE)):
            prompt = (tmpl.format(records=records or "(none)",
                                  question_block=format_question_with_date(q))
                      if arm == "R" else
                      tmpl.format(records=records or "(none)", episodic=epi["context"],
                                  question_block=format_question_with_date(q)))
            t0 = time.perf_counter()
            hyp = _generate_with_retry(llm, prompt, system_prompt=ANSWER_SYSTEM,
                                       temperature=0.0)
            gen = time.perf_counter() - t0
            err = "[System Error:" in hyp
            correct, judge = False, ""
            if not err:
                jp = get_anscheck_prompt(q["question_type"], q["question"], q["answer"],
                                         hyp, abstention="_abs" in qid)
                judge = _generate_with_retry(llm, jp, system_prompt="", temperature=0.0)
                correct = judge.strip().lower().startswith("yes")
            row["arms"][arm] = {"hypothesis": hyp, "judge_raw": judge, "correct": correct,
                                "is_error": err, "prompt_chars": len(prompt),
                                "generation_seconds": gen}
            print(f"  [{qi}/{len(cohort)}] {qid} ({q['question_type']}) arm {arm}: "
                  f"{'ERR' if err else ('OK' if correct else 'WRONG')}", flush=True)
        results.append(row)

    summary = {
        "cohort_size": len(results),
        "baseline": "all wrong under oracle AND ETC AND single-shot",
        "bar": ">=4 conversions in either arm",
        "conversions": {a: sum(r["arms"][a]["correct"] for r in results)
                        for a in ("R", "R+E")},
        "n_errors": sum(r["arms"][a]["is_error"] for r in results for a in ("R", "R+E")),
        "median_record_chars": sorted(r["record_chars"] for r in results)[len(results) // 2],
        "median_record_lines": sorted(r["record_lines"] for r in results)[len(results) // 2],
    }
    json.dump({"model": args.model, "summary": summary, "results": results},
              open(args.out, "w"), indent=2)
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
