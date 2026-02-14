#!/usr/bin/env python3
"""Verification script for the CEUR-WS discussion paper."""
import re, sys

tex = open("main.tex").read()
bib = open("references.bib").read()

passed = 0
total = 0

# CHECK 1: Banned terms outside footnote
total += 1
banned = re.compile(r'C\+\+|AVX-512|AVX2|SIMDe|nanobind|zero.copy|cache.line', re.I)
in_fn = False
fails1 = []
for i, line in enumerate(tex.split('\n'), 1):
    s = line.strip()
    if s.startswith('%'): continue
    if '\\footnote{' in line: in_fn = True
    if in_fn:
        if '}' in line: in_fn = False
        continue
    if banned.search(line):
        fails1.append(f"  L{i}: {s[:80]}")
if fails1:
    print("FAIL - Banned terms outside footnote:")
    for f in fails1: print(f)
else:
    print("PASS - No banned terms outside footnote")
    passed += 1

# CHECK 2: Citation keys
total += 1
keys = set()
for m in re.finditer(r'\\cite\{([^}]+)\}', tex):
    for k in m.group(1).split(','):
        keys.add(k.strip())
missing = [k for k in keys if k not in bib]
if missing:
    print(f"FAIL - Missing citation keys: {missing}")
else:
    print(f"PASS - All {len(keys)} citation keys found in references.bib")
    passed += 1

print(f"\n{passed}/{total} checks passed")
sys.exit(0 if passed == total else 1)
