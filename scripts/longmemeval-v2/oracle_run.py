#!/usr/bin/env python3
"""
LongMemEval-V2 oracle-context control (advisor-directed follow-up to
repr_ab_test.py's compact-vs-tree A/B, v4-plan.md Stage 6).

LongMemEval-S has an explicit `answer_session_ids` field per question, so its
oracle control (scripts/longmemeval/oracle_run.py) is simple: hand the reader
the named session's full text directly, no retrieval involved, and see how
far real-retrieval accuracy falls short of that ceiling. LongMemEval-V2's
public question/haystack files have NO equivalent "gold trajectory" field --
`questions.jsonl` only carries {id, domain, question_type, question, answer,
eval_function}, and the "static-environment"/"dynamic-environment" question
types this A/B is scoped to (norm_phrase_set_match[_ordered]) ask about facts
of the shared environment itself (e.g. "which Incidents-list filter labels
contain the substring 'Incident'") that may be observable from more than one
trajectory, not one designated episode.

So this script builds an ADAPTED oracle: split the reference answer into its
scored phrases (qa_eval_metrics.split_phrases, the same normalization the
real evaluator uses) and grep the ENTIRE domain's raw `accessibility_tree`
text (not the compact Goal/URL/Thought/Action summary -- see the finding
below) for any state containing one of those phrases verbatim. This directly
answers a question repr_ab_test.py's compact-vs-tree A/B could not: that A/B
always prompted with `prompt_repr=compact` (deliberately, to avoid the
prompt-size blowup documented in common.py), which means neither of its two
conditions ever showed the model the literal UI text a fact like a filter
dropdown's option labels would actually appear in -- the compact summary
(the agent's Goal/URL/Thought/Action) has no reason to ever restate a
menu's full option list. If this oracle (fed the actual matching raw-tree
snippets) scores far above the compact-embed/compact-prompt run's 15%, that
confirms the ~15% ceiling isn't (only) a ranking-quality problem -- it's that
the fact-bearing text was structurally excluded from the prompt in every
condition tested so far, regardless of how good retrieval ranking gets.

Also reports the fraction of questions where NO state in the entire
100-trajectory/domain haystack contains any answer phrase at all -- if that
fraction is high, the fact isn't reliably re-derivable from these
trajectories' raw text either (a dataset/environment-coverage ceiling, not an
Aeon problem), which matters for interpreting whatever accuracy comes back.

Usage:
    python scripts/longmemeval-v2/oracle_run.py \\
        --questions questions.jsonl --haystack lme_v2_small.json \\
        --trajectories trajectories.jsonl \\
        --num-questions-per-domain 20 --model gemma4:31b-cloud \\
        --out reproducibility_benchmarks/longmemeval-v2/oracle_results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DOMAIN_SYSTEM_PROMPTS, DOMAINS, build_reader_prompt, extract_boxed_answer,
    load_domain_trajectory_ids, load_wanted_trajectories, score_deterministic,
    state_text,
)
from qa_eval_metrics import split_phrases  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "longmemeval"))
from run_benchmark import _generate_with_retry  # noqa: E402

from aeon_py.llm import OllamaProvider  # noqa: E402

DETERMINISTIC_TYPES = {"norm_phrase_set_match", "norm_phrase_set_match_ordered"}

# Cap on how many matching states get included in the oracle prompt and how
# much of each one's raw text is kept -- a real oracle should still be a
# bounded, plausible amount of context (not "paste the whole domain"), and
# this keeps prompt size in the same safe territory as the fixed prompt-size
# bug from repr_ab_test.py's first (contaminated) run.
MAX_ORACLE_STATES = 10
MAX_STATE_CHARS = 3000


_WINDOW_RADIUS = MAX_STATE_CHARS // 2


def _oracle_context(question: dict, states_by_traj: dict) -> tuple[list[dict], int]:
    """Returns (matched states as {session_id, text} dicts, total match count
    before capping) by grepping every state's raw tree text for any of the
    answer's scored phrases.

    Two bugs found by eyeballing the first run's actual prompts (advisor-
    directed check) and fixed here, both of which silently hid the very
    evidence this oracle exists to test:
    1. Truncating each state to `tree_text[:MAX_STATE_CHARS]` kept only the
       page's boilerplate header -- the actual match (e.g. "Source Code" at
       character 3551 of a 4714-char state, immediately answering "which
       column is right of Quantity") was past the cutoff and never reached
       the prompt. Now extracts a window CENTERED on the first match
       position instead of always taking the start of the string.
    2. Consecutive states within one trajectory are often near-identical
       (one page, several small actions) -- a question with only a couple
       of matching trajectories still burned its whole `MAX_ORACLE_STATES`
       budget on 10 near-duplicate copies of the same page. Now dedupes by
       exact matched-window text before capping.
    """
    phrases = [p for p in split_phrases(question["answer"], separators=[",", ";"]) if len(p) >= 3]
    if not phrases:
        return [], 0

    matches = []
    seen_windows = set()
    for traj_id, traj in states_by_traj.items():
        for s in traj.get("states") or []:
            tree_text = state_text(traj, s, "tree")
            hay = tree_text.lower()
            idx = min((hay.find(p) for p in phrases if p in hay), default=-1)
            if idx == -1:
                continue
            start = max(0, idx - _WINDOW_RADIUS)
            end = min(len(tree_text), idx + _WINDOW_RADIUS)
            window = tree_text[start:end]
            matches.append({"session_id": traj_id, "text": window, "_window": window})

    total = len(matches)
    deduped = []
    for m in matches:
        if m["_window"] in seen_windows:
            continue
        seen_windows.add(m["_window"])
        deduped.append({"session_id": m["session_id"], "text": m["text"]})

    return deduped[:MAX_ORACLE_STATES], total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--haystack", required=True)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--num-questions-per-domain", type=int, default=20)
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import os
    if args.model:
        os.environ["AEON_LLM_MODEL"] = args.model

    with open(args.questions) as f:
        all_questions = [json.loads(line) for line in f]
    with open(args.haystack) as f:
        haystack = json.load(f)

    llm = OllamaProvider()
    print(f"Model: {llm.model}")

    domain_traj_ids = load_domain_trajectory_ids(all_questions, haystack)
    wanted_ids = set(domain_traj_ids["web"]) | set(domain_traj_ids["enterprise"])
    print(f"Loading {len(wanted_ids)} trajectories (full small-tier haystack, both domains)...")
    t0 = time.perf_counter()
    trajectories_by_id = load_wanted_trajectories(args.trajectories, wanted_ids)
    print(f"  loaded {len(trajectories_by_id)} in {time.perf_counter()-t0:.1f}s")

    # Same fixed, deterministic question subset as repr_ab_test.py, so this
    # oracle is directly comparable to that run's compact/tree results.
    eval_questions = []
    for domain in DOMAINS:
        pool = [
            q for q in all_questions
            if q["domain"] == domain and not q.get("image")
            and q["eval_function"].split("|")[0] in DETERMINISTIC_TYPES
        ]
        eval_questions.extend(pool[: args.num_questions_per_domain])
    print(f"Evaluating {len(eval_questions)} questions ({args.num_questions_per_domain}/domain)")

    states_by_domain = {
        domain: {tid: trajectories_by_id[tid] for tid in domain_traj_ids[domain] if tid in trajectories_by_id}
        for domain in DOMAINS
    }

    results = []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_path.with_suffix(".jsonl")

    with open(jsonl_path, "w") as jf:
        for i, q in enumerate(eval_questions):
            t0 = time.perf_counter()
            retrieved, total_matches = _oracle_context(q, states_by_domain[q["domain"]])
            user_text = build_reader_prompt(q, retrieved)
            response = _generate_with_retry(
                llm, user_text, system_prompt=DOMAIN_SYSTEM_PROMPTS[q["domain"]],
                temperature=args.temperature,
            )
            parsed = extract_boxed_answer(response)
            r = {
                "question_id": q["id"], "domain": q["domain"],
                "question_type": q["question_type"],
                "eval_function": q["eval_function"].split("|", 1)[0],
                "question": q["question"], "reference_answer": q["answer"],
                "response_raw": response, "response_parsed_boxed": parsed,
                "num_oracle_states_matched_total": total_matches,
                "num_oracle_states_used": len(retrieved),
                "phrase_found_in_haystack": total_matches > 0,
            }
            r = score_deterministic(r)
            results.append(r)
            jf.write(json.dumps(r) + "\n")
            jf.flush()
            elapsed = time.perf_counter() - t0
            status = (
                "TRANSPORT_ERR" if r.get("is_error")
                else "OK" if r["correct"]
                else ("ERR" if r["eval_error"] else "WRONG")
            )
            print(
                f"[{i+1}/{len(eval_questions)}] {q['id']} ({q['domain']}, {r['eval_function']}) "
                f"-> {status} (matches={total_matches}, {elapsed:.1f}s)",
                flush=True,
            )

    def acc(rows):
        scored = [r for r in rows if not r.get("is_error")]
        return sum(1 for r in scored if r["correct"]) / len(scored) if scored else 0.0

    found = [r for r in results if r["phrase_found_in_haystack"]]
    not_found = [r for r in results if not r["phrase_found_in_haystack"]]

    summary = {
        "model": llm.model,
        "temperature": args.temperature,
        "num_questions": len(results),
        "num_errors": sum(1 for r in results if r.get("is_error")),
        "haystack_coverage": len(found) / len(results) if results else 0.0,
        "oracle_accuracy_overall": acc(results),
        "oracle_accuracy_when_phrase_found": acc(found),
        "n_phrase_not_found": len(not_found),
        "by_domain": {
            d: {
                "n": len([r for r in results if r["domain"] == d]),
                "haystack_coverage": (
                    len([r for r in found if r["domain"] == d]) /
                    len([r for r in results if r["domain"] == d])
                    if [r for r in results if r["domain"] == d] else 0.0
                ),
                "oracle_accuracy_when_phrase_found": acc([r for r in found if r["domain"] == d]),
            }
            for d in DOMAINS
        },
    }

    with open(out_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print("\n=== Oracle summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path} (incremental log at {jsonl_path})")


if __name__ == "__main__":
    main()
