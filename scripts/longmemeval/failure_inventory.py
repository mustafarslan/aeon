#!/usr/bin/env python3
"""Complete failure inventory across the three n=500 arms already on disk."""
import json, collections

B = 'reproducibility_benchmarks/longmemeval/'
base = json.load(open(B+'full_session_n500_results.json'))
v1   = json.load(open(B+'extract_then_compute_n500_results.json'))
v3   = json.load(open(B+'extract_then_compute_n500_v3_results.json'))

bd = {r['question_id']: r for r in base['results']}
v1d = {r['question_id']: r for r in v1['results']}
v3d = {r['question_id']: r for r in v3['results']}

ids = [r['question_id'] for r in v1['results']]
rt = {r['question_id']: r['report_type'] for r in v1['results']}

print("="*78)
print("1. HEADLINE: accuracy by arm and type")
print("="*78)
types = ['multi-session','temporal-reasoning','knowledge-update','single-session-user',
         'single-session-assistant','single-session-preference','abstention']
print(f"{'type':<28}{'n':>5}{'base':>9}{'ETCv1':>9}{'ETCv3':>9}   {'v1-base':>8}")
tot = collections.Counter()
for t in types:
    sub = [q for q in ids if rt[q]==t]
    b = sum(bd[q]['correct'] for q in sub)
    a = sum(v1d[q]['correct'] for q in sub)
    c = sum(v3d[q]['correct'] for q in sub)
    tot['b']+=b; tot['a']+=a; tot['c']+=c
    print(f"{t:<28}{len(sub):>5}{b:>9}{a:>9}{c:>9}   {a-b:>+8}")
print(f"{'TOTAL':<28}{len(ids):>5}{tot['b']:>9}{tot['a']:>9}{tot['c']:>9}   {tot['a']-tot['b']:>+8}")

print()
print("="*78)
print("2. WHERE THE ERRORS ACTUALLY ARE (ETC v1 = the candidate to ship)")
print("="*78)
errs = [(t, sum(1 for q in ids if rt[q]==t and not v1d[q]['correct'])) for t in types]
errs.sort(key=lambda x:-x[1])
total_err = sum(e for _,e in errs)
cum=0
for t,e in errs:
    cum+=e
    print(f"  {t:<28} {e:>4} wrong  ({e/total_err*100:>5.1f}% of all errors, cum {cum/total_err*100:>5.1f}%)")
print(f"  {'TOTAL':<28} {total_err:>4} wrong")

print()
print("="*78)
print("3. FLOW baseline -> ETCv1, per type  (fixed = ETC gained, broke = ETC lost)")
print("="*78)
print(f"{'type':<28}{'both ok':>9}{'fixed':>8}{'BROKE':>8}{'both bad':>10}")
for t in types:
    sub=[q for q in ids if rt[q]==t]
    both = sum(1 for q in sub if bd[q]['correct'] and v1d[q]['correct'])
    fixed= sum(1 for q in sub if not bd[q]['correct'] and v1d[q]['correct'])
    broke= sum(1 for q in sub if bd[q]['correct'] and not v1d[q]['correct'])
    bad  = sum(1 for q in sub if not bd[q]['correct'] and not v1d[q]['correct'])
    print(f"{t:<28}{both:>9}{fixed:>+8}{-broke:>+8}{bad:>10}")

print()
print("="*78)
print("4. THE HARD CORE: wrong in ALL THREE arms (nothing tried so far touches these)")
print("="*78)
hard = [q for q in ids if not bd[q]['correct'] and not v1d[q]['correct'] and not v3d[q]['correct']]
hc = collections.Counter(rt[q] for q in hard)
print(f"  {len(hard)} of 500 questions ({len(hard)/5:.1f}%) are wrong under every arm tested")
for t,c in hc.most_common():
    n = sum(1 for q in ids if rt[q]==t)
    print(f"    {t:<28} {c:>4} / {n:<4} ({c/n*100:>5.1f}% of that type is unfixed by anything)")

print()
print("="*78)
print("5. UNION CEILING: correct in AT LEAST ONE arm")
print("="*78)
union = sum(1 for q in ids if bd[q]['correct'] or v1d[q]['correct'] or v3d[q]['correct'])
print(f"  best-of-3-arms oracle = {union}/500 = {union/5:.1f}%")
print(f"  vs best single arm    = {max(tot['b'],tot['a'],tot['c'])}/500 = {max(tot['b'],tot['a'],tot['c'])/5:.1f}%")
print(f"  => a PERFECT router over these 3 arms could gain at most {union-max(tot['b'],tot['a'],tot['c'])} questions")

print()
print("="*78)
print("6. v1 -> v3 CHURN (v3 was an exact 390/390 tie: was it really a wash?)")
print("="*78)
gain = [q for q in ids if not v1d[q]['correct'] and v3d[q]['correct']]
loss = [q for q in ids if v1d[q]['correct'] and not v3d[q]['correct']]
print(f"  v3 gained {len(gain)}, lost {len(loss)} -- net 0 but {len(gain)+len(loss)} questions CHANGED")
print("  gains by type:", dict(collections.Counter(rt[q] for q in gain)))
print("  losses by type:", dict(collections.Counter(rt[q] for q in loss)))
print("  => the 'exact tie' hides real churn; prompt changes move ~%d questions of noise+signal" % (len(gain)+len(loss)))
