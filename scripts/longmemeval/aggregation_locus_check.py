#!/usr/bin/env python3
"""Retrieval-only discriminating check for the multi-session aggregation bucket.

Zero LLM calls. Rebuilds the EXACT assembled context each arm saw, then greps
it for the gold mentions extraction failed to surface. Splits the bucket into:
  - retrieval miss  (mention NOT in assembled context)  -> raise top_k/max_sessions
  - extraction loss (mention IS in context, dropped)    -> EXTRACT-side fix
"""
import json, sys, re
from pathlib import Path
sys.path.insert(0, 'scripts/longmemeval')
from run_benchmark import _get_encoder, _ingest_haystack
import numpy as np
from aeon_py.session_expansion import build_expanded_context, format_events
from aeon_py.trace import TraceGraph

DS = '/private/tmp/claude-501/-Volumes-AI-SSD-Projects-aeon/3ae98779-a31a-4698-97f5-d3e4b2ae3bf3/scratchpad/longmemeval/longmemeval_s_cleaned.json'
TMP = Path('/private/tmp/claude-501/-Volumes-AI-SSD-Projects-aeon/bc5f36fc-bec5-4147-9c0d-300491cffab0/scratchpad/agg')
TMP.mkdir(parents=True, exist_ok=True)

# question -> regex probes for the gold mentions that extraction did NOT surface
CASES = {
    '8e91e7d9':      [r'\bsister\b', r'\bbrothers\b', r'\bsiblings\b'],
    '1a8a66a6':      [r'subscri', r'magazine'],
    'gpt4_ab202e7f': [r'coffee maker', r'faucet', r'toaster', r'mat\b', r'shel'],
    'ba358f49':      [r'\bage\b', r'\bturn(ed|ing)? \d\d', r'\bI.m \d\d', r'birth'],
    '81507db6':      [r'graduation', r'ceremony', r'ceremonies'],
}

ds = {q['question_id']: q for q in json.load(open(DS))}
v1 = {r['question_id']: r for r in json.load(open(
    'reproducibility_benchmarks/longmemeval/extract_then_compute_n500_results.json'))['results']}
enc = _get_encoder()

for qid, probes in CASES.items():
    q = ds[qid]
    tp = TMP / f'{qid}.trace'
    for sfx in ('', '.blobs', '.wal'):
        Path(str(tp) + sfx).unlink(missing_ok=True)
    trace = TraceGraph(path=str(tp))
    _ingest_haystack(trace, enc, q)
    qv = np.asarray(enc.encode(q['question']), dtype=np.float32).tolist()
    ev = build_expanded_context(trace, qv, 'full_session', base_top_k=30, max_sessions=10)
    ctx = format_events(ev)

    # sanity: context we rebuilt should match what the run recorded
    match = 'MATCH' if len(ctx) == v1[qid]['context_chars'] else f"DIFF({len(ctx)} vs {v1[qid]['context_chars']})"
    print(f"\n{'='*72}\n{qid}  [{v1[qid]['report_type']}]  ctx={len(ctx)} chars  {match}")
    print(f"  Q: {q['question'][:100]}")
    print(f"  GOLD: {str(q['answer'])[:100]}")
    print(f"  EXTRACTED: {v1[qid]['extracted_facts'][:150]}".replace('\n', ' | '))
    for p in probes:
        hits = [m.start() for m in re.finditer(p, ctx, re.I)]
        verdict = f"IN CONTEXT x{len(hits)}" if hits else "ABSENT from context"
        print(f"    probe {p!r:<22} -> {verdict}")
        for h in hits[:2]:
            print(f"        ...{ctx[max(0,h-90):h+90]}...".replace('\n', ' '))
