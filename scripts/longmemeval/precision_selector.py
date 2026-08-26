#!/usr/bin/env python3
"""Sub-turn precision selector -- the real (non-oracle) candidate for the product direction.

The oracle-precision arm established the ceiling: 83.8% at 5,654 median chars, one LLM call,
0.51s generation, 74.10 correct per 1k chars. That arm used gold `has_answer` annotations. THIS
module is the production-shaped approximation of it, and the open question in
PAPER_V4_FINDINGS.md §9 is what fraction of 74.10 it reaches.

IMPORTANT: `has_answer` is used ONLY for evaluation in this file, never inside the selection path.
Nothing in `select()` may read it.

Two designs, both offline-measurable for zero LLM calls:

  A  chunk-index    -- split every turn into sentence-level chunks, embed each chunk bare, retrieve
                       chunks directly. A buried aside gets its own vector instead of being averaged
                       into its turn's dominant topic, which is the diagnosed cause of all 7
                       complete-miss retrieval failures (v4-plan.md).
  B  two-stage      -- turn-level candidates at a wide top_k (the 200x20 run measured 94.4%
                       answer-turn coverage there), then chunk-level scoring picks a small set of
                       turns out of that pool. This is the 200x20 finding turned into architecture:
                       deep retrieval already HAD the coverage; the context bloat is what killed it,
                       and deleting the bloat while keeping the coverage is exactly a reranker's job.

Both designs share the same tail, and the stitching step is not optional:
    chunk hit -> parent turn -> stitch +/-1 neighbouring turns -> dedupe -> stop at char budget

The +/-1 stitch is the one design constraint this project has *proven* rather than assumed. ETC's
extract step compresses to bare facts and BROKE single-session-user (60 -> 55) by destroying
pragmatic licensing ("picked it up at Trader Joe's" answers "what brand?" only under conversational
implicature). The oracle arm compressed just as hard but kept one turn either side and scored 63/64.

Sentence splitting treats newlines and list markers as boundaries: assistant turns are frequently
markdown bullet lists, and splitting on `.` alone would produce one enormous chunk and defeat the
entire point of sub-turn granularity.
"""

import re
from dataclasses import dataclass, field

import numpy as np

# (?<![0-9]) guards against splitting inside numbered list markers: without it
# "Here are tips:\n1. Moisturize" splits after the "1." and produces the useless
# chunk "Here are tips: 1.", which then absorbs its neighbour via MIN_CHUNK_CHARS.
_SENT_SPLIT = re.compile(r"(?<![0-9])(?<=[.!?])\s+|\n+|(?=^\s*[-*\u2022]\s)|(?=^\s*\d+\.\s)", re.M)
# Deliberately small. An early value of 40 merged any shorter fragment BACKWARD into
# its predecessor, which glued the standalone sentence "By the way, I just got a smoker
# today." (38 chars) onto an unrelated chunk -- re-creating, inside the chunker, exactly
# the topic-dilution that sub-turn chunking exists to remove. Short *content* sentences
# are frequently the whole answer; only orphaned list markers deserve merging, and they
# merge FORWARD into the item they label.
MIN_CHUNK_CHARS = 25
_BARE_MARKER = re.compile(r"^(?:\d+[.)]|[-*\u2022])$")


def split_chunks(text: str, max_chars: int = 400) -> list[str]:
    """Sentence/line-level split, list-marker aware.

    Assistant turns are frequently markdown lists, so newlines and list markers are
    boundaries; and a numbered marker ("1.") must not be read as a sentence terminator.
    """
    pieces = []
    for piece in _SENT_SPLIT.split(text):
        if not piece:
            continue
        piece = piece.strip()
        if not piece:
            continue
        while len(piece) > max_chars:
            cut = piece.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            pieces.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        if piece:
            pieces.append(piece)

    # Merge FORWARD: an orphan marker or very short fragment belongs with what follows,
    # never with what precedes it.
    out: list[str] = []
    pending = ""
    for p in pieces:
        cand = f"{pending} {p}".strip() if pending else p
        if _BARE_MARKER.match(p) or len(cand) < MIN_CHUNK_CHARS:
            pending = cand
            continue
        out.append(cand)
        pending = ""
    if pending:
        if out:
            out[-1] = f"{out[-1]} {pending}"
        else:
            out.append(pending)
    return out


@dataclass
class TurnRef:
    session_idx: int
    turn_idx: int
    date: str
    role: str
    content: str


@dataclass
class ChunkIndex:
    turns: list[TurnRef] = field(default_factory=list)
    chunk_text: list[str] = field(default_factory=list)
    chunk_owner: list[int] = field(default_factory=list)   # index into .turns
    embeddings: np.ndarray | None = None
    turn_embeddings: np.ndarray | None = None

    @property
    def n_chunks(self) -> int:
        return len(self.chunk_text)


def build_index(question: dict, encoder, max_chunk_chars: int = 400,
                embed_turns: bool = True) -> ChunkIndex:
    idx = ChunkIndex()
    for s_i, (date, turns) in enumerate(zip(question["haystack_dates"],
                                            question["haystack_sessions"])):
        for t_i, t in enumerate(turns):
            ref = TurnRef(s_i, t_i, date, t["role"], t["content"])
            owner = len(idx.turns)
            idx.turns.append(ref)
            for c in split_chunks(t["content"], max_chunk_chars):
                idx.chunk_text.append(c)
                idx.chunk_owner.append(owner)
    idx.embeddings = _encode(encoder, idx.chunk_text)
    if embed_turns:
        idx.turn_embeddings = _encode(encoder, [t.content for t in idx.turns])
    return idx


def _encode(encoder, texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 768), dtype=np.float32)
    v = np.asarray(encoder.encode(texts), dtype=np.float32)
    if v.ndim == 1:
        v = v[None, :]
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.clip(n, 1e-9, None)


def _fmt(t: TurnRef) -> str:
    role_name = "system" if t.role == "assistant" else t.role
    return f"- [{role_name}] [{t.date}] {t.role}: {t.content}"


def select(question: dict, index: ChunkIndex, q_vec: np.ndarray, *,
           design: str = "B", budget_chars: int = 9000, stitch: int = 1,
           stage1_turns: int = 200, per_turn_cap: int = 1, mmr: float = 0.0,
           stitch_mode: str = "inline", core_frac: float = 0.7) -> dict:
    """Returns the assembled context plus diagnostics. Never reads `has_answer`.

    per_turn_cap limits how many chunk hits from the SAME turn can each pull in a
    fresh neighbourhood -- without it a single verbose turn monopolises the budget.

    stitch_mode resolves a tension the tier-1 coverage sweep MEASURED rather than predicted:
    stitching +/-1 inline costs budget that would otherwise hold more evidence, and at a fixed
    9k budget it dropped answer-turn coverage from 83.8% to 62.6%. But stitching is the one
    constraint this project proved necessary (it is what keeps single-session-user at 63/64
    instead of 55). "post" resolves it by ordering the spend: fill `core_frac` of the budget
    with evidence turns alone (maximising coverage), then spend whatever remains buying
    neighbourhoods around the highest-scoring hits (buying licensing). "inline" is the naive
    interleaved version, kept for comparison.

    mmr (0..1) trades query-similarity against diversity, Maximal Marginal Relevance
    style: score = (1-mmr)*sim(q,c) - mmr*max_sim(c, already_selected). Added on direct
    evidence, not speculation -- inspecting the top-10 chunks for a failing question
    showed them to be near-duplicates of each other (all variations on "kitchen gifts"),
    so a tight char budget was being spent several times on the same information while
    the answer-bearing chunk sat at rank 47. Diversity is how a small budget covers more
    ground. mmr=0 disables it and reproduces pure similarity ranking.
    """
    qv = q_vec / max(float(np.linalg.norm(q_vec)), 1e-9)
    chunk_scores = index.embeddings @ qv

    if design == "B":
        # stage 1: wide turn-level candidate pool, then score only chunks inside it
        turn_scores = index.turn_embeddings @ qv
        pool = set(np.argsort(-turn_scores)[:stage1_turns].tolist())
        mask = np.array([o in pool for o in index.chunk_owner], dtype=bool)
        chunk_scores = np.where(mask, chunk_scores, -np.inf)

    order = np.argsort(-chunk_scores)
    if mmr > 0.0:
        # greedy MMR over a candidate prefix (full pairwise over all chunks is wasteful;
        # anything below the prefix could never fit the budget anyway)
        cand = [int(c) for c in order[:400] if np.isfinite(chunk_scores[c])]
        picked: list[int] = []
        picked_vecs = np.zeros((0, index.embeddings.shape[1]), dtype=np.float32)
        while cand and len(picked) < 400:
            cv = index.embeddings[cand]
            rel = chunk_scores[cand]
            if len(picked):
                red = (cv @ picked_vecs.T).max(axis=1)
            else:
                red = np.zeros(len(cand), dtype=np.float32)
            mscore = (1.0 - mmr) * rel - mmr * red
            best = int(np.argmax(mscore))
            ci = cand.pop(best)
            picked.append(ci)
            picked_vecs = np.vstack([picked_vecs, index.embeddings[ci][None, :]])
        order = np.array(picked, dtype=int)

    chosen: dict[int, float] = {}
    seen_owner_hits: dict[int, int] = {}
    total = 0
    budget_core = int(budget_chars * core_frac) if stitch_mode == "post" else budget_chars
    for ci in order:
        if not np.isfinite(chunk_scores[ci]):
            break
        owner = index.chunk_owner[ci]
        if seen_owner_hits.get(owner, 0) >= per_turn_cap:
            continue
        seen_owner_hits[owner] = seen_owner_hits.get(owner, 0) + 1
        ref = index.turns[owner]
        eff_stitch = 0 if stitch_mode == "post" else stitch
        group = []
        for other in range(owner - eff_stitch, owner + eff_stitch + 1):
            if 0 <= other < len(index.turns) and index.turns[other].session_idx == ref.session_idx:
                group.append(other)
        add = [g for g in group if g not in chosen]
        cost = sum(len(_fmt(index.turns[g])) + 1 for g in add)
        if total + cost > budget_core and chosen:
            continue
        for g in add:
            chosen[g] = float(chunk_scores[ci])
        total += cost
        if total >= budget_core:
            break

    if stitch_mode == "post" and stitch > 0:
        # second pass: spend the remainder on neighbourhoods, best-scoring hits first
        for owner in sorted(chosen, key=lambda o: -chosen[o]):
            ref = index.turns[owner]
            for other in range(owner - stitch, owner + stitch + 1):
                if other in chosen or not (0 <= other < len(index.turns)):
                    continue
                if index.turns[other].session_idx != ref.session_idx:
                    continue
                cost = len(_fmt(index.turns[other])) + 1
                if total + cost > budget_chars:
                    continue
                chosen[other] = chosen[owner] - 1e-6
                total += cost

    ordered = sorted(chosen)  # chronological: turns are appended in session/turn order
    ctx = "\n".join(_fmt(index.turns[g]) for g in ordered)
    return {
        "context": ctx,
        "n_turns": len(ordered),
        "chars": len(ctx),
        "selected_turn_indices": ordered,
        "n_chunks_indexed": index.n_chunks,
    }
