#!/usr/bin/env python3
"""
LongMemEval benchmark harness for Aeon (v4-plan.md Stage 6).

Measures how well Aeon's episodic Trace (shell/aeon_py/trace.py --
TraceGraph.semantic_search(), backed by the C++ TraceBlockIndex) surfaces
the right facts from a long conversation history, and how well a local LLM
answers LongMemEval questions given only what Aeon retrieves.

Dataset: xiaowu0162/longmemeval-cleaned on Hugging Face
(https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned), the
maintained replacement for the original LongMemEval release
(arxiv:2410.10813). Each question ships with its own independent
"haystack" -- typically ~48 chat sessions / ~490 turns / ~120k tokens of
distractor + relevant history -- which this harness ingests into a fresh,
temporary Aeon Trace file before asking the question, so no ingestion or
retrieval work is shared across questions (matching the benchmark's own
per-question isolation contract).

Judge prompts are ported verbatim from the official evaluation harness
(judge_prompts.py, src/evaluation/evaluate_qa.py in
github.com/xiaowu0162/LongMemEval) so scores stay comparable in *shape* to
published numbers -- though see v4-plan.md Stage 6 for why the judge MODEL
here (a local Ollama model, not GPT-4o) makes absolute numbers
non-comparable to the paper's own reported baselines.

What this harness deliberately does NOT exercise: Atlas (the spatial/
concept index), Architect's admission-time dedup, Dreaming/consolidation,
or ContextManager.process_turn()'s full orchestration -- see v4-plan.md
Stage 6 "scope" section for the reasoning. This is a Trace-retrieval-and-
answer-generation benchmark, not a full CognitiveLoop benchmark.

Usage:
    python scripts/longmemeval/run_benchmark.py \\
        --dataset /path/to/longmemeval_s_cleaned.json \\
        --num-questions 50 --seed 42 --top-k 10 \\
        --model qwen3.8:27b-mlx \\
        --out reproducibility_benchmarks/longmemeval/pilot_results.json

Requires: `pip install -e .` (aeon_py importable), `sentence-transformers`,
and a running Ollama daemon with the requested model already pulled.
"""

import argparse
import json
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, median

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_prompts import get_anscheck_prompt  # noqa: E402

from aeon_py.trace import TraceGraph  # noqa: E402
from aeon_py.llm import OllamaProvider  # noqa: E402

QUESTION_TYPES = [
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
]

SYSTEM_PROMPT = (
    "You are a careful assistant answering a question using ONLY the "
    "retrieved memory snippets below. Each snippet is tagged with the date "
    "it was recorded and who said it. If the snippets don't contain enough "
    "information to answer, say so plainly instead of guessing. Answer as "
    "concisely as possible -- a short phrase or sentence, not a paragraph."
)

# Found empirically in this benchmark's own first pilot run: the FIRST two
# questions both got "[System Error: Could not connect ...]" hypotheses/
# judge_raw values (RemoteDisconnected, then a transient 404) while Ollama
# was still loading the ~32GB model into memory for the first request --
# silently scored as wrong answers rather than infrastructure failures,
# which would have deflated overall_accuracy by ~4 points on a 50-question
# run for a reason having nothing to do with Aeon or the model's actual
# capability. Fixed two ways: an explicit warm-up call in main() before the
# loop starts, and this retry wrapper as defense-in-depth against any other
# transient connection hiccup during the run.
_TRANSIENT_ERROR_MARKER = "[System Error:"

# Substrings that identify a DETERMINISTIC rejection (the provider parsed the
# request and refused it) rather than a transient transport hiccup (dropped
# connection, timeout, cold-start). Found via the LongMemEval-V2 tree-repr
# A/B run: a handful of oversized prompts got "400 Client Error: Bad
# Request" back in ~15s, and _generate_with_retry burned 3 attempts x
# backoff retrying a rejection that was never going to succeed -- the same
# request, unchanged, fails the same way every time. Retry only for errors
# that plausibly resolve on their own.
_NON_RETRYABLE_MARKERS = (
    "400 Client Error", "401 Client Error", "403 Client Error",
    "404 Client Error", "413 Client Error", "422 Client Error",
)

# A rate-limit response is not "the provider is briefly unreachable" (which
# a dropped connection or cold start resolves within a few seconds) -- it
# means "you are over quota for this window," which typically needs several
# times as long to clear. Found via Stage 7 task 1's summary arm (v4-plan.md):
# up to 5 extra LLM calls per question for session-level summarization
# produced a sustained burst that outlasted the plain transient-retry budget
# (3 attempts, 5s/10s backoff) and left 26/50 questions as unrecoverable
# transport errors. Given its own retry budget so a real rate limit doesn't
# get treated the same as a one-off hiccup.
_RATE_LIMIT_MARKER = "429 Client Error"
_MAX_RATE_LIMIT_RETRIES = 6
_RATE_LIMIT_BACKOFF_SECONDS = 15


def format_question_with_date(question: dict) -> str:
    """Renders the question with LongMemEval's `question_date` as an explicit
    reference "now".

    BUG FIX (2026-08-25, v4-plan.md failure inventory): `question_date` ships
    on all 500 questions in the dataset and was referenced in ZERO lines of
    code -- it was never passed to any prompt, in any arm. Relative-time
    questions ("What did I buy 10 days ago?", "How many weeks ago did I start
    using Ibotta?") are therefore unanswerable by construction: the model has
    no "now" to subtract from. This is visible verbatim in the stored n=500
    outputs -- `gpt4_e072b769` answered "The provided text does not contain the
    current date, so the number of weeks cannot be calculated", and
    `gpt4_59149c78` HALLUCINATED "The current date is 2023/01/15" and reasoned
    from the invented date. 21 questions (all temporal-reasoning, all wrong, 17
    of them in the 74-question wrong-under-every-arm hard core) explicitly
    complain about or invent a current date -- 49% of all temporal-reasoning
    errors and 19% of every error in the run, and a strict lower bound, since a
    question that just answers "I don't know" without naming the date isn't
    counted by that scan.

    Consequence for existing numbers: every temporal-reasoning result recorded
    before this fix (single-shot baseline, extract-then-compute v1 and v3, and
    the +17-question ETC "win" on that type) compares two configurations that
    were both broken the same way. Those numbers are not wrong about which arm
    scored higher, but they cannot be read as measuring temporal-reasoning
    capability, and any post-fix number is not comparable to them -- the
    baseline has to be re-run alongside whatever else changes.

    Kept as a single shared helper (rather than inlined per arm) so all three
    prompt-building call sites -- this file's single-shot arm,
    `expansion_unit_experiment.py`'s full_session arm, and
    `extract_then_compute_experiment.py`'s EXTRACT and COMPUTE steps -- stay in
    sync; a fix applied to only some arms would silently confound the next
    comparison between them.
    """
    date = question.get("question_date")
    if not date:
        return f"Question: {question['question']}"
    return (
        f"Today's date is {date}.\n"
        f"Question: {question['question']}"
    )


def _generate_with_retry(
    llm: OllamaProvider, prompt: str, system_prompt: str = "",
    temperature: float | None = None, retries: int = 3,
) -> str:
    last = ""
    attempt = 0
    rate_limit_attempt = 0
    while True:
        last = "".join(
            llm.generate(prompt, system_prompt=system_prompt, temperature=temperature)
        ).strip()
        if _TRANSIENT_ERROR_MARKER not in last:
            return last
        if any(marker in last for marker in _NON_RETRYABLE_MARKERS):
            return last
        if _RATE_LIMIT_MARKER in last:
            if rate_limit_attempt >= _MAX_RATE_LIMIT_RETRIES:
                return last
            rate_limit_attempt += 1
            time.sleep(_RATE_LIMIT_BACKOFF_SECONDS * rate_limit_attempt)
            continue
        attempt += 1
        if attempt >= retries:
            return last
        time.sleep(5 * attempt)


def _stratified_sample(questions: list[dict], n: int, seed: int) -> list[dict]:
    """Proportionally samples `n` questions across LongMemEval's 6 question
    types, preserving the full dataset's type distribution (v4-plan.md
    Stage 6) rather than uniform random sampling, so a 50-question pilot
    isn't accidentally skewed away from e.g. temporal-reasoning (26.6% of
    the full 500) just by chance."""
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = {}
    for q in questions:
        by_type.setdefault(q["question_type"], []).append(q)

    total = len(questions)
    sample: list[dict] = []
    remaining = n
    types = sorted(by_type.keys())
    for i, t in enumerate(types):
        bucket = by_type[t]
        if i == len(types) - 1:
            take = remaining
        else:
            take = round(n * len(bucket) / total)
            take = min(take, remaining)
        rng.shuffle(bucket)
        sample.extend(bucket[:take])
        remaining -= take
    rng.shuffle(sample)
    return sample[:n]


def _get_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-mpnet-base-v2")


# LongMemEval-S's own haystack_dates format, e.g. "2023/05/20 (Sat) 02:21" --
# the "(Sat)" weekday token isn't part of a strptime format, stripped before
# parsing.
_HAYSTACK_DATE_FORMAT = "%Y/%m/%d %H:%M"


def _parse_haystack_date_micros(date_str: str) -> int:
    """Parses LongMemEval's haystack_dates string into epoch microseconds,
    for `event_time` (v4-plan.md Stage 7 Track 2). Returns 0 (== "unset")
    on any parse failure rather than raising -- a date the harness can't
    parse should silently fall back to Aeon's own insertion `timestamp`
    for ordering, not abort ingestion."""
    try:
        cleaned = date_str.split(" (", 1)[0] + " " + date_str.rsplit(") ", 1)[1]
        dt = datetime.strptime(cleaned, _HAYSTACK_DATE_FORMAT)
        return int(dt.timestamp() * 1_000_000)
    except (ValueError, IndexError):
        return 0


def _ingest_haystack(trace: TraceGraph, encoder, question: dict) -> float:
    """Ingests one question's full haystack into `trace`. The STORED text
    is prefixed with its session's real-world date (`[<date>] <role>:
    <content>`) for human/LLM readability once a turn is retrieved in
    isolation. Each event's `event_time` (V4 Stage 7 Track 2) is ALSO set
    to the same parsed date -- Aeon's own `TraceEvent.timestamp` is always
    insertion wall-clock, never caller-supplied, so before `event_time`
    existed this benchmark's synthetic historical dates had nowhere to
    live except inside the text itself, unusable for programmatic
    chronological ordering across sessions
    (`session_expansion.py`'s `merge_expanded_context()` now sorts by
    `event_time`). The EMBEDDED text is deliberately the bare turn
    content, NOT the date-prefixed string: every one of ~490 documents
    would otherwise carry ~25 identical prefix characters that the query
    text never has, which measurably biases a cosine-similarity embedding
    space (shifts every document vector toward a shared direction,
    compressing the angular spread retrieval depends on) -- an
    advisor-caught issue, verified by the harness's own retrieval-only A/B
    (see v4-plan.md Stage 6). `trace.add_event(text=..., embedding=...)`
    already accepts these as two independent arguments, so this is a pure
    decoupling, not a trade-off. Returns ingestion wall-clock seconds.
    """
    t0 = time.perf_counter()
    for sess_idx, (date, sess_id, turns) in enumerate(
        zip(
            question["haystack_dates"],
            question["haystack_session_ids"],
            question["haystack_sessions"],
        )
    ):
        if not turns:
            continue
        event_time = _parse_haystack_date_micros(date)
        contents = [t["content"] for t in turns]
        vecs = encoder.encode(contents)
        for turn, content, vec in zip(turns, contents, vecs):
            role = "user" if turn["role"] == "user" else "system"
            stored_text = f"[{date}] {turn['role']}: {content}"
            trace.add_event(
                sess_id, role, stored_text,
                embedding=np.asarray(vec, dtype=np.float32).tolist(),
                event_time=event_time,
            )
    return time.perf_counter() - t0


def _run_one_question(
    question: dict, encoder, llm: OllamaProvider, top_k: int, tmp_dir: Path,
    retrieval_only: bool = False, temperature: float | None = None,
) -> dict:
    trace_path = tmp_dir / f"{question['question_id']}.trace"
    # TraceManager mmap-opens an existing file rather than truncating it --
    # a stale file left by a prior crashed/killed run (before the
    # end-of-function cleanup below ran) would otherwise get silently
    # reused and double-ingested. See the V2 harness's identical bug
    # (common.py's fresh_trace_path docstring) for how this bit that run.
    for suffix in ("", ".blobs", ".wal"):
        Path(str(trace_path) + suffix).unlink(missing_ok=True)
    trace = TraceGraph(path=str(trace_path))

    ingest_seconds = _ingest_haystack(trace, encoder, question)

    t0 = time.perf_counter()
    q_vec = encoder.encode(question["question"])
    query_encode_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    retrieved = trace.semantic_search(
        np.asarray(q_vec, dtype=np.float32).tolist(), top_k=top_k
    )
    # This is the only timer in this script that measures Aeon's own C++
    # kernel work (TraceBlockIndex, via TraceManager.semantic_search) --
    # every other "seconds" field here times Python-side sentence-transformers
    # encoding or the local LLM, not Aeon itself. Kept as its own variable
    # (not folded into query_encode_seconds above) specifically so a v4-docs
    # chart can't misattribute encoder latency to the kernel -- an
    # advisor-caught issue in an earlier version of this script, where a
    # single timer spanned both and the resulting number (dominated by
    # mpnet inference) was ~1000x Aeon's own measured navigate() latency.
    search_seconds = time.perf_counter() - t0

    gold_sessions = set(question["answer_session_ids"])
    retrieved_sessions = {ev["session_id"] for ev in retrieved}
    gold_session_hit = bool(gold_sessions & retrieved_sessions)

    is_abstention = "_abs" in question["question_id"]
    result = {
        "question_id": question["question_id"],
        "question_type": question["question_type"],
        "is_abstention": is_abstention,
        # LongMemEval's 30/500 (6%) abstention-augmented questions carry
        # one of the 6 base question_types (they're each a real question
        # paired with a haystack where the relevant session was removed --
        # see the official repo), NOT a distinct type of their own.
        # Reporting them under their base type would silently blend two
        # different skills ("recall the fact" vs. "correctly say you don't
        # know") into one accuracy number, so this is the bucket used for
        # per-type reporting below -- "abstention" regardless of origin
        # type, matching how the literature usually presents this slice.
        "report_type": "abstention" if is_abstention else question["question_type"],
        "question": question["question"],
        "reference_answer": question["answer"],
        "num_haystack_sessions": len(question["haystack_sessions"]),
        "num_haystack_turns": sum(len(s) for s in question["haystack_sessions"]),
        "num_retrieved": len(retrieved),
        "gold_session_hit_at_k": gold_session_hit,
        "timing_seconds": {
            "ingest": ingest_seconds,
            "query_encode": query_encode_seconds,
            "search": search_seconds,
        },
    }

    if not retrieval_only:
        context_block = "\n".join(f"- {ev['text']}" for ev in retrieved) or "(nothing retrieved)"
        user_prompt = (
            f"Retrieved memories:\n{context_block}\n\n"
            f"{format_question_with_date(question)}\n\nAnswer:"
        )

        t0 = time.perf_counter()
        response = _generate_with_retry(llm, user_prompt, system_prompt=SYSTEM_PROMPT, temperature=temperature)
        generation_seconds = time.perf_counter() - t0

        # A response that's itself a transport-error marker isn't a wrong
        # answer -- it's infrastructure noise. Skip the judge call (there's
        # nothing meaningful to judge) and flag it separately so summaries
        # can exclude it from the accuracy denominator instead of silently
        # counting it as a model failure.
        is_error = _TRANSIENT_ERROR_MARKER in response
        if is_error:
            judge_response = ""
            judge_seconds = 0.0
            correct = False
        else:
            t0 = time.perf_counter()
            judge_prompt = get_anscheck_prompt(
                question["question_type"], question["question"], question["answer"], response,
                abstention=is_abstention,
            )
            judge_response = _generate_with_retry(llm, judge_prompt, system_prompt="", temperature=0.0)
            judge_seconds = time.perf_counter() - t0
            correct = "yes" in judge_response.lower()

        result["hypothesis"] = response
        result["judge_raw"] = judge_response
        result["correct"] = correct
        result["is_error"] = is_error
        result["timing_seconds"]["generation"] = generation_seconds
        result["timing_seconds"]["judge"] = judge_seconds

    try:
        trace_path.unlink(missing_ok=True)
    except OSError:
        pass

    return result


def _summarize(
    results: list[dict], top_k: int, model: str, seed: int, dataset_path: str,
    encoder_name: str, retrieval_only: bool, temperature: float | None = None,
) -> dict:
    by_type: dict[str, list[dict]] = {}
    for r in results:
        by_type.setdefault(r["report_type"], []).append(r)

    def n_errors(rows):
        return sum(1 for r in rows if r.get("is_error"))

    def acc(rows):
        # Transport-error rows (Ollama unreachable/rejected the request)
        # are infrastructure noise, not the model getting the question
        # wrong -- excluded from both the numerator and denominator so a
        # run with a handful of hiccups doesn't read as lower accuracy
        # than the model actually achieved on the questions it was asked.
        if retrieval_only:
            return None
        scored = [r for r in rows if not r.get("is_error")]
        return sum(1 for r in scored if r["correct"]) / len(scored) if scored else 0.0

    def recall(rows):
        return sum(1 for r in rows if r["gold_session_hit_at_k"]) / len(rows) if rows else 0.0

    per_type = {
        t: {
            "n": len(rows),
            "n_errors": n_errors(rows),
            "accuracy": acc(rows),
            "gold_session_recall_at_k": recall(rows),
        }
        for t, rows in sorted(by_type.items())
    }

    phases = ["ingest", "query_encode", "search"]
    if not retrieval_only:
        phases += ["generation", "judge"]
    all_timings = {
        phase: [r["timing_seconds"][phase] for r in results]
        for phase in phases
    }

    def pctl(values, p):
        if not values:
            return 0.0
        s = sorted(values)
        idx = min(len(s) - 1, int(round(p * (len(s) - 1))))
        return s[idx]

    return {
        # None in retrieval-only mode: no LLM call is ever made, so
        # recording a model name here would misleadingly imply one was
        # used (an advisor-caught issue -- OllamaProvider() is still
        # constructed for API uniformity, but its .model is never invoked).
        "model": None if retrieval_only else model,
        "top_k": top_k,
        "temperature": temperature,
        "seed": seed,
        "dataset_path": str(dataset_path),
        "encoder": encoder_name,
        "retrieval_only": retrieval_only,
        "num_questions": len(results),
        "num_errors": n_errors(results),
        "overall_accuracy": acc(results),
        "overall_gold_session_recall_at_k": recall(results),
        "per_question_type": per_type,
        "latency_seconds": {
            phase: {
                "mean": mean(values) if values else 0.0,
                "median": median(values) if values else 0.0,
                "p95": pctl(values, 0.95),
            }
            for phase, values in all_timings.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Path to longmemeval_s_cleaned.json (or _oracle/_m variant)")
    parser.add_argument("--num-questions", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Passed through to the answer-generation call (the judge call "
             "already hardcodes 0.0). Default None uses the model's own "
             "default temperature, which is non-zero -- fine for a single "
             "run, but it means two runs of the SAME questions/top_k can "
             "disagree on a few answers by sampling alone (observed: 3/40 "
             "questions flipped between two otherwise-identical compact-repr "
             "passes in the V2 harness). Pass 0.0 when comparing two "
             "conditions (e.g. a top_k sweep) so a result difference can't "
             "be sampling noise.",
    )
    parser.add_argument("--model", default=None, help="Overrides AEON_LLM_MODEL for this run")
    parser.add_argument("--out", required=True, help="Path to write results JSON")
    parser.add_argument("--tmp-dir", default=None, help="Directory for per-question scratch Trace files (default: system tmp)")
    parser.add_argument(
        "--retrieval-only", action="store_true",
        help="Skip LLM generation+judging entirely -- ingest, retrieve, and "
             "compute gold_session_hit_at_k only. No Ollama calls, no model "
             "needed. For cheap retrieval-quality A/B tests (v4-plan.md Stage 6).",
    )
    args = parser.parse_args()

    import os
    import tempfile
    if args.model:
        os.environ["AEON_LLM_MODEL"] = args.model

    with open(args.dataset) as f:
        all_questions = json.load(f)

    sample = _stratified_sample(all_questions, args.num_questions, args.seed)
    print(f"Sampled {len(sample)} questions (seed={args.seed}) from {len(all_questions)} total:")
    print(f"  {dict(Counter(q['question_type'] for q in sample))}")

    encoder_name = "all-mpnet-base-v2"
    encoder = _get_encoder()
    llm = OllamaProvider()
    if args.retrieval_only:
        print("--retrieval-only: skipping LLM generation+judging entirely")
    else:
        print(f"Using Ollama model: {llm.model}")
        print("Warming up model (forces full load before timing/scoring begins)...")
        warm = _generate_with_retry(llm, "Say OK.", retries=5)
        print(f"  warm-up response: {warm[:80]!r}")

    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else Path(tempfile.mkdtemp(prefix="longmemeval_"))
    tmp_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, q in enumerate(sample):
        t0 = time.perf_counter()
        r = _run_one_question(
            q, encoder, llm, args.top_k, tmp_dir,
            retrieval_only=args.retrieval_only, temperature=args.temperature,
        )
        results.append(r)
        elapsed = time.perf_counter() - t0
        if args.retrieval_only:
            status = "HIT" if r["gold_session_hit_at_k"] else "MISS"
        elif r.get("is_error"):
            status = "TRANSPORT_ERR"
        else:
            status = "OK" if r["correct"] else "WRONG"
        print(
            f"[{i+1}/{len(sample)}] {q['question_id']} ({q['question_type']}) "
            f"-> {status} ({elapsed:.1f}s, {r['num_haystack_turns']} turns ingested)",
            flush=True,
        )

    summary = _summarize(
        results, args.top_k, llm.model, args.seed, args.dataset, encoder_name,
        args.retrieval_only, args.temperature,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
