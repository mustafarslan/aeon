"""
Session-level retrieval expansion (v4-plan.md Stage 7).

Both LongMemEval-S and LongMemEval-V2 (Stage 6) converged on the same gap:
`TraceGraph.semantic_search()` ranks individual events, but many real
questions need several related events from one session/trajectory to
answer at all -- a single top-ranked event is often necessary but not
sufficient. Stage 6's oracle controls (handing over a whole gold session,
or a window of raw text around a matched fact) each beat real `top_k`
retrieval by ~50 points on the type of question that needs this. Stage 7
task 1 is the experiment that determines which EXPANSION UNIT (full
session / +-N-turn window / session summary) captures most of that gap at
the lowest retrieved-token cost; this module is the reusable primitive
layer both that experiment and any eventual production integration build
on, so the two don't drift into separate implementations.

IMPORTANT, stated explicitly rather than left to be discovered later:
as of Stage 7, `TraceGraph.semantic_search()` is not called from any
production code path. `ContextManager.process_turn()` only queries Atlas
for concept associations and records events into Trace; `CognitiveLoop.
chat()` only pulls the current session's last 12 events by recency
(`get_history(sid, limit=12)`), not by relevance, and never across
sessions. Stage 7 task 2's "wire the winning unit into a real capability"
therefore has no existing call site to extend -- choosing where semantic,
cross-session episodic recall first enters the live serving path is
still an open design decision for task 2, not something this module
presupposes. This module works standalone against any `TraceGraph` +
`session_id`, so it's usable the moment that decision is made.

All three functions below build on exactly two existing primitives --
`TraceGraph.semantic_search()` (already used to find the top hit) and
`TraceGraph.get_history()` -- no kernel changes. Matches Stage 6's own
scope decision (Trace only, not Atlas/SLB) and must not touch
`HierarchicalSLB`'s FP32-only cache invariant (CLAUDE.md).
"""

from __future__ import annotations

from typing import Optional

from .trace import TraceGraph

_ROLE_NAMES = {0: "user", 1: "system", 2: "concept", 3: "summary"}

# Generous default: large enough to hold any LongMemEval-S haystack session
# (observed max ~560 turns in Stage 6's pilot) without truncating silently.
# A real deployment's sessions could run longer -- callers with a known
# larger bound should pass their own `limit`.
_DEFAULT_HISTORY_LIMIT = 2000


def _role_name(role: int) -> str:
    return _ROLE_NAMES.get(role, "system")


def format_events(events: list[dict]) -> str:
    """Renders a chronological event list into the `- [role] text` lines
    every Stage 6 harness prompt already uses, so output from any
    expansion unit below drops into an existing prompt-building path
    unchanged."""
    if not events:
        return "(no events)"
    return "\n".join(f"- [{_role_name(ev.get('role', 1))}] {ev.get('text', '')}" for ev in events)


def expand_full_session(
    trace: TraceGraph, session_id: str, limit: int = _DEFAULT_HISTORY_LIMIT,
) -> list[dict]:
    """The 'full session' unit -- every event in `session_id`, chronological
    (oldest first). Unlike Stage 6's `oracle_run.py` (which builds this
    from the benchmark's own gold-label haystack structure, bypassing
    Aeon entirely to isolate the model's reasoning ceiling), this version
    goes through `get_history()` -- the primitive a real caller actually
    has, since a real caller knows a `session_id` from a `semantic_search()`
    hit, never a gold label. This is what Stage 7 task 1's 'full_session'
    arm measures: the ceiling AS Aeon can actually reach it, not the
    benchmark's oracle-only ceiling.
    """
    history = trace.get_history(session_id, limit=limit)  # newest-first
    return list(reversed(history))


def expand_window(
    trace: TraceGraph, session_id: str, hit_event_id: int, window_n: int,
    limit: int = _DEFAULT_HISTORY_LIMIT,
) -> list[dict]:
    """The '+-N-turn window' unit -- `window_n` events on each side of the
    event that `semantic_search()` actually ranked top, within its own
    session. Finds the hit's position by matching `id` (both
    `semantic_search()` and `get_history()` return real event ids for any
    real TraceGraph -- no text-matching workaround needed, unlike a
    synthetic benchmark haystack that doesn't have Aeon ids until
    ingested). Returns an empty list if the hit isn't found within
    `limit` events -- callers should treat that as "widen `limit`", not
    "no window exists".

    Latency note, stated plainly against Aeon's own ultra-low-latency bar:
    this pulls the WHOLE session (up to `limit`) via `get_history()` and
    slices in Python -- O(session length), not O(window_n). `get_history()`
    is a prev_id-chain walk, so this is cheap for the session lengths this
    benchmark and near-term production traffic exercise (hundreds of
    events), and is the right tradeoff for task 1's experiment (no kernel
    changes needed to test whether this unit is even worth building). If
    task 1 picks this unit AND profiling on real session lengths ever
    shows this mattering (very long-running sessions, thousands of
    events), the real fix is a targeted kernel-side primitive -- "events
    near anchor id X, +-N" -- rather than widening `limit` further; that
    is `core/` work, out of scope for this stage, and should be scoped
    separately if it becomes necessary.
    """
    chronological = expand_full_session(trace, session_id, limit=limit)
    hit_idx = next((i for i, ev in enumerate(chronological) if ev["id"] == hit_event_id), None)
    if hit_idx is None:
        return []
    lo = max(0, hit_idx - window_n)
    hi = min(len(chronological), hit_idx + window_n + 1)
    return chronological[lo:hi]


# Kept deliberately generic (not LongMemEval-specific): asks for a dense,
# fact-preserving summary rather than a narrative gloss, since a summary
# that reads well but drops the one date/number/name a question needs is
# worse than raw turns for this use case.
SUMMARY_PROMPT_TEMPLATE = (
    "Summarize the following conversation session into a dense paragraph "
    "that preserves every concrete fact, date, name, number, and stated "
    "preference. Do not omit details for brevity -- someone must be able "
    "to answer specific factual questions about the session from the "
    "summary alone, without seeing the raw text.\n\nSession:\n{session_text}\n\nSummary:"
)


def expand_summary(
    trace: TraceGraph, session_id: str, generate_fn, limit: int = _DEFAULT_HISTORY_LIMIT,
) -> str:
    """The 'session summary' unit -- an LLM-produced synopsis of the full
    session, substituted for its raw turns. `generate_fn` is a callable
    `(prompt: str) -> str` (e.g. `lambda p: _generate_with_retry(llm, p,
    temperature=0.0)` in a benchmark harness, or a thin wrapper around
    whatever LLM call a real caller already has) -- kept as an injected
    dependency rather than importing an LLM provider directly, so this
    module has no hard dependency on Ollama or any specific provider.
    Per Stage 7's task 1 decision rule, this is the most expensive unit
    (an extra LLM call per question) and is only worth using if no window
    arm lands within 10 points of the full-session ceiling.
    """
    events = expand_full_session(trace, session_id, limit=limit)
    session_text = format_events(events)
    prompt = SUMMARY_PROMPT_TEMPLATE.format(session_text=session_text)
    return generate_fn(prompt)


def find_top_hit(trace: TraceGraph, query_embedding: list[float]) -> Optional[dict]:
    """The single best-ranked event for `query_embedding`, or None if
    nothing is indexed. `window_n`/`summary` units both expand from this
    hit's session -- keeping the "which event ranked top" decision in one
    place means task 1's arms differ only in expansion strategy, not in
    how the anchor hit is chosen."""
    hits = trace.semantic_search(query_embedding, top_k=1)
    return hits[0] if hits else None


# --- Multi-session, additive expansion --------------------------------
#
# Bug found running Stage 7 task 1's first live experiment (v4-plan.md):
# every arm above anchors to a SINGLE session (whichever `find_top_hit()`
# landed in) and REPLACES the retrieved context with that session's
# expansion. Checked directly against this project's own dataset sample:
# knowledge-update questions always have exactly 2 gold `answer_session_ids`,
# temporal-reasoning 1-3, multi-session 3-4 -- so a single-session unit
# structurally cannot reach the answer for most of these questions no
# matter how large that one session's window gets. Worse, "replace" meant
# an arm could (and did) score BELOW the plain top_k=30 baseline that
# already ships in run_benchmark.py (knowledge-update: 37.5% here vs 100%
# at top_k=30) -- discarding hits from other sessions that plain top-k
# retrieval was already surfacing correctly.
#
# The functions below fix both: anchor to the top-N DISTINCT sessions
# among a top_k semantic_search (not top-1), and merge each session's
# expansion ADDITIVELY onto the full top_k hit list (hits from sessions
# that weren't selected for expansion are kept, never dropped) -- so an
# expansion arm is structurally a superset of what top_k alone retrieves,
# never a lossy substitute for it.

# Originally 5, sized off this project's LongMemEval-S sample's max observed
# gold-session count (4). Measured directly to be too narrow (v4-plan.md
# Stage 7's gold-session recall@N diagnostic, LLM-free): all-golds-present
# recall for multi-session was only 63.6% at N=5, climbing to 90.9% -- the
# full top_k=30 ceiling -- at N=10. The gold sessions were reachable, just
# ranked 6th-10th by ordinary embedding noise, not excluded by relevance.
# Raised to 10 to match the measured recall ceiling; no data yet supports
# going higher.
_DEFAULT_MAX_SESSIONS = 10


def find_top_hits(trace: TraceGraph, query_embedding: list[float], top_k: int = 30) -> list[dict]:
    """The full ranked top_k events -- the same retrieval this project's
    existing top_k pipeline already uses (the 58%/68%/78% baselines in
    v4-plan.md). This is the base retrieval set every expansion arm below
    augments, never replaces."""
    return trace.semantic_search(query_embedding, top_k=top_k)


def distinct_session_ids(hits: list[dict], max_sessions: int = _DEFAULT_MAX_SESSIONS) -> list[str]:
    """Up to `max_sessions` distinct session_ids among `hits`, in the
    order their best (first, since `hits` is already rank-sorted) hit
    appears."""
    seen: list[str] = []
    for h in hits:
        sid = h["session_id"]
        if sid not in seen:
            seen.append(sid)
        if len(seen) >= max_sessions:
            break
    return seen


def expand_full_sessions(
    trace: TraceGraph, session_ids: list[str], limit: int = _DEFAULT_HISTORY_LIMIT,
) -> dict[str, list[dict]]:
    """`expand_full_session()` applied to each of several sessions."""
    return {sid: expand_full_session(trace, sid, limit=limit) for sid in session_ids}


def expand_windows(
    trace: TraceGraph, hits: list[dict], session_ids: list[str], window_n: int,
    limit: int = _DEFAULT_HISTORY_LIMIT,
) -> dict[str, list[dict]]:
    """`expand_window()` applied around each session's own best-ranked hit
    (the first hit in `hits`, which is rank-sorted, belonging to that
    session) -- not just the single overall top hit."""
    best_hit_per_session: dict[str, dict] = {}
    for h in hits:
        if h["session_id"] in session_ids and h["session_id"] not in best_hit_per_session:
            best_hit_per_session[h["session_id"]] = h
    return {
        sid: expand_window(trace, sid, best_hit_per_session[sid]["id"], window_n, limit=limit)
        for sid in session_ids if sid in best_hit_per_session
    }


def _event_sort_key(ev: dict) -> int:
    """V4 Stage 7 Track 2: order by `event_time` (caller-supplied "when
    this happened") when set, falling back to `timestamp` (Aeon's own
    insertion wall-clock) when it isn't -- `event_time == 0` means unset,
    not "epoch zero," so it must not be preferred over a real timestamp."""
    return ev.get("event_time") or ev.get("timestamp", 0)


def expand_summaries(
    trace: TraceGraph, session_ids: list[str], generate_fn, limit: int = _DEFAULT_HISTORY_LIMIT,
) -> dict[str, list[dict]]:
    """`expand_summary()` applied to each of several sessions, each
    wrapped as a single `role=ROLE_SUMMARY`-shaped event dict so it merges
    into `merge_expanded_context()` the same way the other units' events
    do. Tagged with a representative `event_time` (the earliest event in
    the session being summarized) so a summary lands in roughly the right
    place when multiple sessions' content is sorted chronologically --
    without this, a synthetic summary dict (no `event_time`/`timestamp` of
    its own) would sort as if it happened at epoch zero, ahead of
    everything else, regardless of which session it actually summarizes."""
    result = {}
    for sid in session_ids:
        events = expand_full_session(trace, sid, limit=limit)
        representative_time = min((_event_sort_key(ev) for ev in events), default=0)
        summary_text = expand_summary(trace, sid, generate_fn, limit=limit)
        result[sid] = [{
            "role": 3, "text": summary_text, "session_id": sid,
            "event_time": representative_time,
        }]
    return result


def merge_expanded_context(
    base_hits: list[dict], expansion_by_session: dict[str, list[dict]],
) -> list[dict]:
    """Additive merge: EVERY base hit is kept, full stop, then expansion
    content is unioned on top (deduplicated by event id).

    Bug found in this function's first version, live-diagnosed against
    Stage 7 task 1's real data (v4-plan.md): it dropped ALL of a session's
    base hits once that session was selected for expansion, on the
    assumption the expansion was always a superset. True for
    `full_session` (which pulls literally everything), false for
    `window_N` with a small N -- a session can have several distinct
    relevant hits in the top_k=30 list spread further apart than the
    window radius around just the single best-ranked hit in that session,
    and those extra hits were silently discarded. Measured effect:
    window_3/window_5 still scored 9-27 points below the recorded top_k=30
    baseline on knowledge-update/multi-session even after the multi-session
    anchoring fix. Keeping every base hit unconditionally removes the
    assumption entirely -- an expansion unit can only ever ADD context
    Aeon didn't already have, never remove it.

    V4 Stage 7 Track 2: the merged result is sorted chronologically (by
    `event_time`, falling back to `timestamp` when unset) before being
    returned, rather than left in the base-hits-by-similarity-then-
    per-session-groups order construction produces. Real gap this fixes:
    a multi-session question's assembled context previously interleaved
    unrelated sessions in similarity-rank order, destroying any "what
    happened before what" signal across sessions -- plausibly costly
    specifically for temporal-reasoning questions, whose entire premise
    depends on that ordering."""
    merged: list[dict] = []
    seen_ids: set = set()

    def _add(ev: dict, sid: str) -> None:
        key = ev.get("id", (sid, ev.get("text")))
        if key in seen_ids:
            return
        seen_ids.add(key)
        merged.append(ev)

    for h in base_hits:
        _add(h, h["session_id"])
    for sid, events in expansion_by_session.items():
        for ev in events:
            _add(ev, sid)

    merged.sort(key=_event_sort_key)
    return merged


def build_expanded_context(
    trace: TraceGraph, query_embedding: list[float], unit: str,
    base_top_k: int = 30, max_sessions: int = _DEFAULT_MAX_SESSIONS,
    generate_fn=None, limit: int = _DEFAULT_HISTORY_LIMIT,
) -> list[dict]:
    """Top-level orchestrator: ranked top_k hits, plus multi-session
    expansion of the top `max_sessions` distinct sessions among them,
    merged additively. `unit="none"` returns the plain top_k hits
    unchanged -- useful as a same-process regression check against the
    recorded top_k baseline before trusting any other unit's number."""
    hits = find_top_hits(trace, query_embedding, top_k=base_top_k)
    if not hits:
        return []
    if unit == "none":
        return hits

    session_ids = distinct_session_ids(hits, max_sessions=max_sessions)
    if unit == "full_session":
        expansion = expand_full_sessions(trace, session_ids, limit=limit)
    elif unit.startswith("window_"):
        n = int(unit.split("_", 1)[1])
        expansion = expand_windows(trace, hits, session_ids, n, limit=limit)
    elif unit == "summary":
        if generate_fn is None:
            raise ValueError("unit='summary' requires a generate_fn callable")
        expansion = expand_summaries(trace, session_ids, generate_fn, limit=limit)
    else:
        raise ValueError(
            f"Unknown unit: {unit!r} (must be one of none, full_session, "
            f"window_3, window_5, window_10, summary)"
        )
    return merge_expanded_context(hits, expansion)
