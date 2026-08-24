"""
Shared helpers for the LongMemEval-V2 scripts in this directory (smoke_test.py,
repr_ab_test.py, full_run.py). See smoke_test.py's module docstring for the
full context on what LongMemEval-V2 is and why this harness is bespoke
rather than a wrapper around the official evaluation/harness.py.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np


def fresh_trace_path(path) -> Path:
    """TraceManager mmap-opens an existing file rather than truncating it --
    correct for Aeon's real persistent-storage use case, wrong for a scratch
    scaffold reused across script invocations by a literal fixed path. A
    killed repr_ab_test.py run once left 171-268MB of stale
    accessibility-tree blobs at one of these paths; the next run silently
    inherited them via the sidecar .blobs/.wal files and reproduced the
    exact same timeout even after the actual bug (unbounded prompt size)
    was fixed elsewhere. Delete any existing trace + its sidecars before
    opening -- every script in this directory uses fixed /tmp paths as
    transient per-run scratch state, never meant to persist across runs."""
    path = Path(path)
    for suffix in ("", ".blobs", ".wal"):
        Path(str(path) + suffix).unlink(missing_ok=True)
    return path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qa_eval_metrics import eval_from_spec  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "longmemeval"))
from run_benchmark import _generate_with_retry, _TRANSIENT_ERROR_MARKER  # noqa: E402

from aeon_py.trace import TraceGraph  # noqa: E402
from aeon_py.llm import OllamaProvider  # noqa: E402

DOMAINS = ("web", "enterprise")

# Ported verbatim from evaluation/harness.py's DOMAIN_SYSTEM_PROMPTS.
DOMAIN_SYSTEM_PROMPTS = {
    "web": (
        "You are an experienced colleague in a web browsing environment that has "
        "a customized magento-based shopping website, a customized magento-based "
        "shopping admin cms website, as well as a customized forum website based "
        "on reddit/postmill. Answer based on your memory of the environment. "
        "If you do not know the answer, output exactly \\boxed{UNKNOWN}. "
        "Do not guess. Never attempt to guess an answer if you are not sure. "
        "If you believe the question's construction/premise is wrong, provide an "
        "explanation in \\boxed{} explaining why the question is flawed."
    ),
    "enterprise": (
        "You are an experienced colleague working in a customized ServiceNow "
        "environment. Answer based on your memory of the environment. "
        "If you do not know the answer, output exactly \\boxed{UNKNOWN}. "
        "Do not guess. Never attempt to guess an answer if you are not sure. "
        "If you believe the question's construction/premise is wrong, provide an "
        "explanation in \\boxed{} explaining why the question is flawed."
    ),
}

LLM_EVAL_FUNCTIONS = {"llm_abstention_checker", "llm_gotchas_checker"}


def extract_boxed_answer(text: str) -> str:
    """Ported verbatim from qa_eval_metrics.py (duplicated here so this
    module has no import-order dependency on it beyond eval_from_spec)."""
    marker = "\\boxed{"
    idx = text.rfind(marker)
    if idx == -1:
        return text.strip()
    i = idx + len(marker)
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
            out.append(ch)
        else:
            out.append(ch)
        i += 1
    parsed = "".join(out).strip()
    return parsed if parsed else text.strip()


def get_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-mpnet-base-v2")


def state_text(traj: dict, state: dict, repr_mode: str) -> str:
    if repr_mode == "tree":
        return state.get("accessibility_tree") or ""
    # compact
    return (
        f"Goal: {traj['goal']}\n"
        f"URL: {state.get('url') or ''}\n"
        f"Thought: {state.get('thought') or ''}\n"
        f"Action: {state.get('action') or 'none'}"
    )


# Some trajectory states' raw accessibility_tree runs past 300KB (not the
# "5-25KB" assumed when repr_ab_test.py was first written) -- top_k=10 of
# those in one reader prompt is well past what even a 262144-token model
# context tolerates, and Ollama rejects the request outright (400) rather
# than silently truncating. Embedding representation and prompt
# representation are independent choices: state_text()'s caller decides
# what gets encoded for retrieval ranking, and separately what gets shown
# to the LLM. ingest_domain() keeps that decoupled so a "does raw-tree
# embedding rank better" experiment doesn't also blow up the prompt.


def load_domain_trajectory_ids(questions: list, haystack: dict) -> dict:
    """One representative question per domain to read that domain's
    shared trajectory-id list (haystack files map EVERY question in a
    domain to the same list for the 'small' tier, per SCHEMA.md)."""
    out = {}
    for domain in DOMAINS:
        rep_q = next(q for q in questions if q["domain"] == domain)
        out[domain] = haystack[rep_q["id"]]
    return out


def load_wanted_trajectories(trajectories_path: str, wanted_ids: set) -> dict:
    """Streams trajectories.jsonl line by line, keeping only wanted ids --
    the file is >1GB with individual records up to ~1.3MB (raw
    accessibility trees), so parsing and discarding unwanted lines avoids
    holding the whole file's parsed form in memory at once."""
    out = {}
    remaining = set(wanted_ids)
    with open(trajectories_path, "r", encoding="utf-8") as f:
        for line in f:
            if not remaining:
                break
            line = line.strip()
            if not line:
                continue
            # Cheap pre-check before the full json.loads: every record's
            # own id appears as `"id":"<tid>"` in the line's first ~40
            # chars, so skip the (comparatively expensive) JSON parse for
            # any line -- some over 1MB of raw accessibility-tree text --
            # whose id isn't one we still want.
            head = line[:40]
            if not any(f'"id":"{tid}"' in head for tid in remaining):
                continue
            obj = json.loads(line)
            tid = obj.get("id")
            if tid in remaining:
                out[tid] = obj
                remaining.discard(tid)
    if remaining:
        print(f"WARNING: {len(remaining)} wanted trajectory ids not found: {sorted(remaining)[:5]}...")
    return out


def ingest_domain(
    trace: TraceGraph, encoder, trajectories: list, embed_repr: str,
    prompt_repr: str | None = None,
) -> dict:
    """embed_repr controls what gets encoded (and therefore how retrieval
    ranks states); prompt_repr controls what gets stored as the event text
    (and therefore what a retrieved hit actually hands the LLM). Defaults
    to embed_repr when unset, matching the original single-representation
    behavior smoke_test.py still relies on."""
    if prompt_repr is None:
        prompt_repr = embed_repr
    n_states = 0
    for traj in trajectories:
        states = traj.get("states") or []
        if not states:
            continue
        embed_texts = [state_text(traj, s, embed_repr) for s in states]
        prompt_texts = (
            embed_texts if prompt_repr == embed_repr
            else [state_text(traj, s, prompt_repr) for s in states]
        )
        vecs = encoder.encode(embed_texts)
        for prompt_text, vec in zip(prompt_texts, vecs):
            trace.add_event(
                traj["id"], "system", prompt_text,
                embedding=np.asarray(vec, dtype=np.float32).tolist(),
            )
            n_states += 1
    return {"n_trajectories": len(trajectories), "n_states": n_states}


def build_reader_prompt(q: dict, retrieved: list) -> str:
    """Ported from evaluation/harness.py's build_messages() text-only path
    (no image support in this harness -- see smoke_test.py docstring)."""
    intro = "### Memory context:\n"
    if not retrieved:
        intro += "(empty)"
    context_text = intro + "\n".join(f"- [{ev['session_id']}] {ev['text']}" for ev in retrieved)
    return f"{context_text}\n\n### Question to answer:\n{q['question']}"


def run_generation(
    q: dict, trace: TraceGraph, encoder, llm: OllamaProvider, top_k: int,
) -> dict:
    """Retrieval + generation only -- no judge call. Returns everything
    needed to score the response later (deterministically or via judge)."""
    q_vec = encoder.encode(q["question"])
    retrieved = trace.semantic_search(np.asarray(q_vec, dtype=np.float32).tolist(), top_k=top_k)
    user_text = build_reader_prompt(q, retrieved)

    t0 = time.perf_counter()
    response = _generate_with_retry(llm, user_text, system_prompt=DOMAIN_SYSTEM_PROMPTS[q["domain"]])
    gen_seconds = time.perf_counter() - t0
    parsed = extract_boxed_answer(response)

    return {
        "question_id": q["id"],
        "domain": q["domain"],
        "question_type": q["question_type"],
        "eval_function": q["eval_function"].split("|", 1)[0],
        "question": q["question"],
        "reference_answer": q["answer"],
        "response_raw": response,
        "response_parsed_boxed": parsed,
        "num_retrieved": len(retrieved),
        "generation_seconds": gen_seconds,
    }


def score_deterministic(result: dict) -> dict:
    """Scores a run_generation() result with a non-LLM eval_function.
    Raises if the question's eval_function is an LLM-judge type -- callers
    must route those through score_with_judge() instead."""
    if result["eval_function"] in LLM_EVAL_FUNCTIONS:
        raise ValueError(f"{result['eval_function']} requires a judge call, not score_deterministic()")
    # A transport-error response ("[System Error: ...") isn't a wrong
    # answer, it's the request never actually reaching the model -- flag it
    # so callers can exclude it from the accuracy denominator instead of
    # counting a connection failure as the model getting the question
    # wrong (same contamination class the LongMemEval-S harness already
    # guards against in run_benchmark.py's _run_one_question).
    if _TRANSIENT_ERROR_MARKER in result["response_raw"]:
        result["correct"] = False
        result["is_error"] = True
        result["eval_error"] = None
        return result
    spec = result["eval_function"]  # bare name is a valid spec (no options)
    try:
        correct = bool(eval_from_spec(spec, result["response_parsed_boxed"], result["reference_answer"]))
        error = None
    except Exception as e:  # noqa: BLE001 -- surfaced in results, not fatal to the run
        correct = False
        error = str(e)
    result["correct"] = correct
    result["is_error"] = False
    result["eval_error"] = error
    return result


def score_with_judge(result: dict, judge_model: str, judge_base_url: str) -> dict:
    """Scores a run_generation() result with an LLM-judge eval_function
    (llm_abstention_checker / llm_gotchas_checker), pointed at a local
    Ollama model via its OpenAI-compatible endpoint instead of the
    official default (hosted GPT-5.2) -- see module docstring."""
    if _TRANSIENT_ERROR_MARKER in result["response_raw"]:
        result["correct"] = False
        result["is_error"] = True
        result["eval_error"] = None
        return result
    eval_kwargs = {
        "question_item": {"question": result["question"]},
        "parsed_prediction": result["response_parsed_boxed"],
        "model_response": result["response_raw"],
        "evaluator_model": judge_model,
        "evaluator_base_url": judge_base_url,
        "evaluator_api_key": "ollama",
    }
    try:
        correct = bool(eval_from_spec(
            result["eval_function"], result["response_parsed_boxed"], result["reference_answer"], **eval_kwargs
        ))
        error = None
    except Exception as e:  # noqa: BLE001
        correct = False
        error = str(e)
    result["correct"] = correct
    result["is_error"] = False
    result["eval_error"] = error
    return result
