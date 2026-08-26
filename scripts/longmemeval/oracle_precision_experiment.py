#!/usr/bin/env python3
"""Oracle-precision arm: the ceiling of perfect compression (v4-plan.md, PRODUCT DIRECTION).

Measures what accuracy is reachable if the retrieval layer delivered ONLY the
load-bearing evidence instead of ~100k chars of session text. Context is built
from LongMemEval's `has_answer` turn annotations plus one neighbouring turn on
each side, single-shot (ONE LLM call), n=500.

Why this is the decisive experiment for the product direction: measured on the
existing runs, answer-bearing turns are a median 292 chars each / 563 chars per
question against 100,831 chars actually delivered -- 99.2% of what Aeon ships is
padding. Meanwhile Aeon is 0.3-0.8% of end-to-end latency and the LLM is the rest,
with LLM latency context-bound. So the only real latency lever Aeon has is sending
fewer tokens, and the open question is what that costs in accuracy.

The +/-1 neighbouring turn is deliberate, not padding: ETC's extract step is
already a compression layer and it costs -12 single-session questions by stripping
the conversational licensing that mode (ii) failures depend on (the answer
"Trader Joe's" is licensed as a *brand* only by the surrounding chat). Including
neighbours tests whether that licensing survives compression. If single-session
still tanks WITH neighbours, that is the most useful possible negative result: it
measures precisely what a compressor must preserve.

Comparison points, all n=500 with the date fix, same judge, same model:
    single-shot @30x10  388/500 = 77.6%   (~100,889 chars, 1 LLM call, 1.5s)
    ETC        @30x10   413/500 = 82.6%   (~100,889 chars, 2 LLM calls, 2.5s)
    ETC        @200x20  416/500 = 83.2%   (~270,972 chars, 2 LLM calls, 4.5s)

ABSTENTION HANDLING (matters -- read before trusting the headline): 21 of the 500
questions have NO answer-bearing turn by construction (the unanswerable
augmentation). Handing them an empty context would make them trivially correct and
silently inflate the overall number. They instead get real Aeon-retrieved context
trimmed to comparable size, so abstention is still earned. Results are reported
both overall and on the 479-question answer-bearing subset (the primary, paired
comparison).
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_prompts import get_anscheck_prompt  # noqa: E402
from run_benchmark import (  # noqa: E402
    SYSTEM_PROMPT, _generate_with_retry, _get_encoder, _ingest_haystack,
    format_question_with_date,
)

import numpy as np  # noqa: E402
from aeon_py.session_expansion import build_expanded_context  # noqa: E402
from aeon_py.trace import TraceGraph  # noqa: E402

NEIGHBOURS = 1
ABSTENTION_EVENTS = 8  # ~ the size the oracle contexts land at, so abstention is earned not free
_ROLE = {"user": "user", "assistant": "system"}


def _line(date: str, role: str, content: str) -> str:
    """Reproduces exactly what the model sees in every other arm: `format_events`
    wraps each stored event as `- [<role_name>] <stored text>`, and ingest stores
    `[<date>] <role>: <content>`."""
    return f"- [{_ROLE.get(role, role)}] [{date}] {role}: {content}"


def _oracle_context(question: dict) -> tuple[str, int]:
    """Answer-bearing turns plus NEIGHBOURS turns either side, chronological."""
    lines, n_answer = [], 0
    for date, turns in zip(question["haystack_dates"], question["haystack_sessions"]):
        keep = set()
        for i, t in enumerate(turns):
            if t.get("has_answer"):
                n_answer += 1
                for j in range(max(0, i - NEIGHBOURS), min(len(turns), i + NEIGHBOURS + 1)):
                    keep.add(j)
        for i in sorted(keep):
            lines.append(_line(date, turns[i]["role"], turns[i]["content"]))
    return "\n".join(lines), n_answer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="gemma4:31b-cloud")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp-dir", default="/tmp/aeon_oracle")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import os
    os.environ["AEON_LLM_MODEL"] = args.model
    from aeon_py.llm import OllamaProvider
    llm = OllamaProvider()

    questions = json.load(open(args.dataset))
    if args.limit:
        questions = questions[: args.limit]
    tmp = Path(args.tmp_dir)
    tmp.mkdir(parents=True, exist_ok=True)
    encoder = _get_encoder()

    print(f"Oracle-precision arm: {len(questions)} questions, single-shot, "
          f"has_answer turns +/-{NEIGHBOURS}", flush=True)
    print(f"Model: {llm.model}", flush=True)
    warm = _generate_with_retry(llm, "Say OK.", retries=5)
    print(f"  warm-up: {warm[:40]!r}", flush=True)

    results = []
    for i, q in enumerate(questions, 1):
        qid = q["question_id"]
        ctx, n_answer = _oracle_context(q)
        mode = "oracle"
        if n_answer == 0:
            # No answer turn exists (unanswerable augmentation) -- give real
            # retrieved context so abstention is earned, never free.
            mode = "retrieved-abstention"
            tp = tmp / f"{qid}.trace"
            for sfx in ("", ".blobs", ".wal"):
                Path(str(tp) + sfx).unlink(missing_ok=True)
            trace = TraceGraph(path=str(tp))
            _ingest_haystack(trace, encoder, q)
            qv = np.asarray(encoder.encode(q["question"]), dtype=np.float32).tolist()
            # unit="none" = raw top-k hits with no session expansion, which is
            # the closest analogue to what a precision-oriented retriever would
            # surface for a question whose answer does not exist.
            ev = build_expanded_context(trace, qv, "none", base_top_k=ABSTENTION_EVENTS,
                                        max_sessions=1)
            ctx = "\n".join(f"- [{e.get('role', 'user')}] {e.get('text', '')}"
                            for e in ev[:ABSTENTION_EVENTS]) or "(nothing retrieved)"
            for sfx in ("", ".blobs", ".wal"):
                Path(str(tp) + sfx).unlink(missing_ok=True)

        prompt = (f"Retrieved memories:\n{ctx}\n\n"
                  f"{format_question_with_date(q)}\n\nAnswer:")
        t0 = time.perf_counter()
        hyp = _generate_with_retry(llm, prompt, system_prompt=SYSTEM_PROMPT, temperature=0.0)
        gen_s = time.perf_counter() - t0
        is_err = "[System Error:" in hyp

        is_abs = "_abs" in qid
        correct, judge_raw = False, ""
        if not is_err:
            jp = get_anscheck_prompt(q["question_type"], q["question"], q["answer"], hyp,
                                     abstention=is_abs)
            judge_raw = _generate_with_retry(llm, jp, system_prompt="", temperature=0.0)
            correct = judge_raw.strip().lower().startswith("yes")

        results.append({
            "question_id": qid,
            "question_type": q["question_type"],
            "report_type": "abstention" if is_abs else q["question_type"],
            "context_mode": mode, "n_answer_turns": n_answer,
            "reference_answer": q["answer"], "hypothesis": hyp, "judge_raw": judge_raw,
            "correct": correct, "is_error": is_err,
            "context_chars": len(ctx), "context_num_ctx": llm.last_num_ctx,
            "timing_seconds": {"generation": gen_s},
        })
        print(f"[{i}/{len(questions)}] {qid} ({q['question_type']}) -> "
              f"{'TRANSPORT_ERR' if is_err else ('OK' if correct else 'WRONG')} "
              f"({len(ctx)} chars, {mode})", flush=True)

    n_err = sum(r["is_error"] for r in results)
    oracle_rows = [r for r in results if r["context_mode"] == "oracle"]
    acc = sum(r["correct"] for r in results) / len(results)
    summary = {
        "n": len(results), "n_errors": n_err, "accuracy": acc,
        "neighbours": NEIGHBOURS,
        "median_context_chars": statistics.median([r["context_chars"] for r in results]),
        "median_oracle_context_chars": statistics.median(
            [r["context_chars"] for r in oracle_rows]) if oracle_rows else None,
        "median_generation_seconds": statistics.median(
            [r["timing_seconds"]["generation"] for r in results]),
        "answer_bearing_subset": {
            "n": len(oracle_rows),
            "correct": sum(r["correct"] for r in oracle_rows),
        },
        "per_question_type": {},
    }
    for t in sorted({r["report_type"] for r in results}):
        sub = [r for r in results if r["report_type"] == t]
        summary["per_question_type"][t] = {
            "n": len(sub), "correct": sum(r["correct"] for r in sub),
            "accuracy": sum(r["correct"] for r in sub) / len(sub),
        }
    json.dump({"model": args.model, "mode": "oracle-precision (has_answer turns +/-1)",
               "summary": summary, "results": results}, open(args.out, "w"), indent=2)
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
