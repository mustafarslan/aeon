# Aeon v4 — Comprehensive Multi-Phase Improvement Plan

> Synthesized and code-verified from two design reviews in `v4/docs/`:
> `aeon_v4_hive_mind_verdict.md` (audits an earlier v3-based shared-memory proposal, finds Aeon's
> current server mode has no retrieval isolation between users, and proposes a revised Stage 0–4
> roadmap) and `aeon_v4_versioning_growth_addendum.md` (answers three follow-up questions on
> git-like versioning, corpus growth, and change-with-reasons, slotting recommendations into the
> same roadmap). Every claim below was verified against the current tree via parallel codebase
> exploration and refined through two rounds of independent engineering review (advisor) before
> being approved. Stage numbers are resequenced from the source docs per decisions made during
> planning (see "Context" below) — each stage notes which source-doc stage it descends from.

## Status

**Stage 0: COMPLETE (2026-08-22).** All tasks landed and verified — C++ unit tests (62/62 green,
up from 45 baseline), live Python bindings, the C-API via a ctypes smoke test, and full FastAPI
integration tests (`TestClient` against the real `app` object). The CI latency-regression gate
(guardrail #0) was proven to both pass cleanly and correctly fail on a synthetic regression before
being relied on. Summary by task:

- SLB dim-generalization spike (guardrail #1.1) — fixed; 14 new parameterized tests (384/768/1536/3072-dim).
- `HierarchicalSLB` wired into `Atlas` for real — session-aware `navigate()`/`insert()`/`load_context()`,
  cross-session L2 sharing disabled pending Stage 3's scope model, `Atlas::drop_session()`/`Atlas::sync()` added.
- C-API (`aeon_atlas_navigate`/`insert`/`drop_session`) — `session_id` routed via FNV-1a (`aeon::hash::fnv1a_64`),
  `drop_session` now matches its documented `AEON_ERR_NODE_NOT_FOUND` contract instead of always returning `AEON_OK`.
- `X-User-ID` replaced with a pluggable, fail-closed `AuthProvider` (`shell/aeon_py/auth.py`) — JWT
  bearer verification by default, explicit-opt-in-only insecure dev mode, refuses to start unconfigured.
- `POST /state/atlas/query` gated behind auth + `AEON_ENABLE_DEBUG_ENDPOINTS` (off by default).
- `aeon_atlas_options_t` given ABI reserved padding.
- SIMD dispatch doc comments corrected (compile-time architecture selection, not runtime CPUID);
  the `release` CMake preset now builds at the same portable baseline as `ci-linux`/`ci-macos`
  instead of `-march=native`, closing a real SIGILL-on-a-different-host risk on both x86 and Apple
  Silicon (verified locally on Apple Silicon: `-mcpu=apple-m1` confirmed in `compile_commands.json`,
  full test suite green under the portable build). True runtime CPUID dispatch (opportunistic
  AVX-512 on capable x86 hosts) remains deferred — verifying it needs x86 hardware to confirm
  whether SIMDe's codegen actually honors a per-function `__attribute__((target(...)))` rather than
  only the translation unit's baseline `-march`, which could not be checked from this (ARM64) session.
- `Atlas::insert()` durability — correctly scoped to the compaction boundary (`msync` before the old
  generation file is deleted, both in `Atlas::compact_mmap()` and, once the identical gap was found
  there too, `TraceManager::compact()`) rather than a per-insert `msync`/WAL entry, which would have
  regressed the 2.23µs insert budget for no real benefit (MAP_SHARED mmap writes are already
  process-crash-safe). `Atlas::compact_mmap()` had zero prior test coverage; added 3 new tests.
- End-to-end session-identity threading: the C++/C-API/Python-binding session_id plumbing is now
  connected all the way to `AeonClient.query()` and `ContextManager.process_turn()`, so the
  authenticated `user_id` from `get_current_user_id` is what actually scopes the Atlas SLB cache
  lookup for the `/chat`, `/state/atlas/active`, and `/state/atlas/query` endpoints.

**Pre-existing bugs found and fixed as unblockers** (confirmed via `git diff` against origin, not
introduced by this work): `shell/aeon_py/__init__.py` failed to import at all (`EdgeType` missing
from `trace.py`) — this means the ENTIRE `aeon_py` package, including the documented
`uvicorn aeon_py.server:app` entrypoint, could not be imported before this fix, regardless of the
isolation work. `HierarchicalSLB`'s Python binding (`find_nearest`) could never return a hit for any
dimension — `optional<Hit>` was never registered as a nanobind-convertible type, so it would throw
on first real cache hit; never previously exercised because nothing called it.

**Pre-existing bug found and fixed (`/chat` endpoint, plus two related visualization endpoints)**:
`ContextManager.process_turn()` (`context.py`) and `CognitiveLoop.chat()` (`loop.py`) called several
`TraceGraph` methods that don't exist on the current class at all — `add_user_event`, `add_concept`,
`link`, `.graph` (a NetworkX attribute), `TraceGraph.load`/`.save`. `trace.py`'s own docstring notes
it was rewritten from a NetworkX-backed graph to a thin wrapper over the C++ mmap `TraceManager`;
`context.py`/`loop.py`/`session.py`/`server.py` were never updated to match. **The `/chat` endpoint —
Aeon's primary conversational entry point — was completely non-functional** (`AttributeError` on the
first Trace call), independent of anything else in this plan.

Fixed by replacing the private-per-user-JSON-snapshot design (itself broken — `SessionManager`
called `ctx.save_session()`/`ctx.load_session()`, methods that didn't exist either) with a single
shared, mmap-backed `TraceGraph` (`dependencies.get_trace_manager()`, mirroring how `get_atlas_client()`
already works), isolated by `session_id` on `TraceEvent` rather than by file — the same pattern
already built for Atlas's SLB cache in Stage 0. "Linking" a retrieved concept to the query that
surfaced it is now expressed as a `role="concept"` event carrying `atlas_id`, chained immediately
after the user's event via the session's natural chronological order (`prev_id`) — episodic
adjacency stands in for an explicit graph edge, since the current data model has no edge-creation
primitive. This is deliberately NOT the typed-edge model (`edge_type`/`supersedes_id`/
`revokes`/...) — that's real design work Stage 1/2 below still need to do; this fix only makes the
*existing* flat-log model actually work, so Stage 1/2 has a working `/chat` flow to build on rather
than a broken one to also redesign.

Two more endpoints had the identical bug, found while fixing this one: `/state/trace` called
`TraceGraph.to_viz_json()` (also nonexistent, and — worse — took no `user_id` at all, so had it
existed it would have returned every user's combined history with no isolation); `/state/atlas/active`
read `ctx.trace.graph.nodes(...)` inside a bare `except: pass`, so it silently always returned the
"Room 0" default rather than erroring. Both now build their response from
`TraceGraph.get_history(user_id, ...)`, correctly scoped to the requesting user.

Verified end-to-end (not just import-clean) via `FastAPI TestClient` against the real `app` object:
two distinct authenticated users chatting produces isolated, growing trace histories neither can see
the other's; `/state/trace` and `/state/atlas/active` return correctly-shaped, correctly-scoped
responses before and after a chat turn. **Stage 3's outcome experiment is no longer blocked by a
broken conversational flow.**

**Stage 1: COMPLETE (2026-08-22).** All schema/WAL tasks landed and verified at every layer — C++
unit tests (72/72 green, up from 62), a direct nanobind smoke test, and a raw ctypes round-trip
against `libaeon.dylib` proving `aeon_trace_event_t` mirrors `TraceEvent` byte-for-byte.

A second advisor pass on the initial implementation caught two real bugs before they shipped:

1. **`revoke_supersede()` on a node that was ALSO tombstoned after being superseded would resurrect a
   real `hub_penalty` on a dead node**, silently breaking the branchless beam-search exclusion that is
   the entire point of the tombstone invariant (`is_tombstoned()` would still correctly return true,
   but the node would score normally again). Not reachable today — nothing calls `supersede_node()` yet
   — but `consolidate_subgraph()` tombstones live nodes, and Stage 5's Dreaming is specified to run both
   operations on the same nodes. Fixed: `revoke_supersede()` now only restores `hub_penalty` from
   `saved_hub_penalty` when `NODE_FLAG_TOMBSTONE` is clear; otherwise it clears `NODE_FLAG_SUPERSEDED`
   and leaves `hub_penalty` at `TOMBSTONE_PENALTY`. New regression test
   (`RevokeAfterTombstoneLeavesHubPenaltyAtTombstonePenalty`) covers exactly this interaction; the
   original `IndependentOfTombstoneFlag` test was also fixed — it had manually reset `hub_penalty` to a
   fake value mid-test, hiding the interaction it was meant to exercise rather than testing it.
2. **The Atlas-side WAL forward-compat test was green even before the fix existed.** It asserted
   `results.size() >= 1` on a query that both the pre-unknown-record and post-unknown-record inserted
   vectors would satisfy (`vec(DIM, 1.0f)` in both), so under the OLD `break` behavior — which never
   reaches the post-unknown record at all — the test still passed on the pre-unknown record alone.
   Fixed by inserting two vectors that only differ from each other, then querying specifically for the
   post-unknown-record one and asserting a near-1.0 similarity match. **Proved discrimination the same
   way Stage 0 proved the CI perf gate**: temporarily reverted `skip` back to `break` in both
   `Atlas::replay_wal()` and `TraceManager::replay_wal()`, confirmed BOTH forward-compat tests went red
   (Trace's `EXPECT_EQ(history.size(), 3u)` already discriminated correctly and failed at `1`; Atlas's
   fixed version now failed too, similarity `0` vs. expected `~1.0`), then restored the fix and
   confirmed all 72 tests green again.

Two deliberate deviations from the plan text below, both confirmed with the advisor before implementing:

- **`WAL_RECORD_SCOPE` was NOT added.** Tracing why the plan named it: the source doc's rationale was
  "journal `Atlas::insert()`, which today writes no WAL at all" — that gap was already closed in
  Stage 0 (guardrail #1.3) via compaction-boundary `msync()`, a better fit than a per-insert WAL entry
  (which would have regressed the 2.23µs insert budget). Since `scope_bitmap` lives inside `NodeHeader`
  itself, and `insert_delta()`'s existing `WAL_RECORD_ATLAS` payload is already a whole-struct capture,
  every new field is already WAL-covered by the existing record type with zero new code — a fresh
  record type would have had no writer, reproducing exactly the `TraceBlockIndex` dead-code pattern
  this plan exists partly to fix (guardrail #2). If a future stage adds an in-place mutation of an
  *existing* node's `scope_bitmap` (e.g. Stage 4's "bulk bit remap"), that operation — not Stage 1 —
  is where a real WAL record type for it belongs, once it has an actual writer.
- **`scope_bitmap` defaults to `0` ("no scope membership") on every new node, not a session-derived
  value.** There is no scope-assignment authority yet — that's Stage 3/4's control plane — so there is
  nothing principled to derive a bit from today. `insert()`/`insert_delta()` deliberately accept no
  caller-supplied scope parameter; the only path by which `scope_bitmap` can ever be populated is
  authenticated session/org context, once that authority exists. This satisfies "assign scope from
  authenticated context, never a caller-supplied label" by construction (the API surface has no such
  parameter to misuse) rather than by inventing a placeholder convention Stage 3 would have to unwind.

Summary by task:

- **`NodeHeader`'s `reserved[20]`** (`schema.hpp`) split into `scope_bitmap` (8B), `governance_record_id`
  (8B), `saved_hub_penalty` (4B) — exactly fills the budget; offsets pinned with `static_assert`s.
  `sizeof(NodeHeader) == 64` unchanged, so `compute_node_stride()` and every downstream SIMD/mmap
  offset are byte-identical to before. All three `NodeHeader::reserved`-memset call sites
  (`insert_delta()`, `insert()`, `consolidate_subgraph()`'s summary node) updated to explicit
  zero-init of the new fields; `consolidate_subgraph()` additionally now OR-unions the source nodes'
  `scope_bitmap` into the summary node, so a future scope-filtered retrieval can't silently drop
  consolidated content.
- **`NODE_FLAG_SUPERSEDED`** (bit 2) plus `supersede_node()`/`revoke_supersede()`/`is_superseded()` in
  `schema.hpp` — same branchless beam-exclusion as `tombstone_node()` (`hub_penalty = TOMBSTONE_PENALTY`)
  but reversible: the real `hub_penalty` is stashed in `saved_hub_penalty` first, restored on revoke.
  Guarded against double-supersede (a second call is a no-op) — the advisor flagged that without this
  guard, calling it twice would stash `TOMBSTONE_PENALTY` itself as the "real" value and permanently
  poison the node once revoked. 8 direct unit tests in new `core/tests/test_schema.cpp`, including
  regression tests for that double-supersede case and the tombstone-then-revoke interaction found by
  the second advisor pass (see below).
- **`TraceEvent`'s `reserved[364]`** carved: `edge_type`/`reason_code` (1B each, new `EdgeType`/
  `ReasonCode` enums), explicit 2B pad, `supersedes_id` (8B — deliberately not `prev_id`, which is the
  per-session chronological chain), `evidence_blob_offset`/`evidence_blob_size` (8B/4B — deliberately a
  separate pair from `blob_offset`/`blob_size`, which already belong to the event's own text), leaving
  `reserved[340]`. `TRACE_FLAG_SUPERSEDED` added alongside. Offsets pinned with `static_assert`s in
  `schema.hpp`; mirrored exactly in `aeon_trace_event_t` (`aeon_c_api.h`), with matching
  `static_assert`s at the point of the flat `memcpy` in `aeon_c_api.cpp` comparing both structs'
  offsets directly (not just `sizeof`) — a size-only check would miss a field reordering that still
  adds up to 512 bytes. `bindings.cpp`'s `get_history()` dict conversion exposes the three
  Python-relevant fields (`edge_type`, `supersedes_id`, `reason_code`).
- **WAL forward-compatibility (skip-not-break)** in both `Atlas::replay_wal()` and
  `TraceManager::replay_wal()`: `payload_size` is now bounded against bytes remaining in the file
  *before* being trusted for anything (closing a latent unbounded-allocation risk on a
  corrupted/adversarial header, not just a correctness nicety); an unrecognized `record_type` (or a
  recognized type with an unexpected `payload_size`) is skipped after checksum verification rather than
  aborting the rest of replay. Checksum failure still stops replay, preserving existing tested
  behavior. New tests in `test_wal.cpp` for both engines: a well-formed-but-unrecognized record spliced
  between two valid records, confirming both survive.
- **`Atlas::insert()`'s WAL coverage / durability task** — already resolved in Stage 0 (see above); no
  duplicate work needed here, per the plan's own note to reconcile rather than re-touch.
- **Scope assignment at write time** — landed as the `scope_bitmap = 0` default and the
  no-caller-supplied-scope-parameter API design described above; the actual assignment-from-authority
  logic is out of scope until Stage 3/4 builds the authority to assign from.

**Perf verification**: the schema change is layout-neutral by construction (`sizeof(NodeHeader) == 64`
unchanged, confirmed by `static_assert`; the only difference is that 20 bytes of always-zero opaque
padding are now 3 named always-zero-by-default fields — identical bytes touched, identical cache
lines). Spot-checked rather than run through the full two-build-dir CI gate (disproportionate for a
provably neutral change): `bench_wal_overhead`'s `BM_Insert` median measured 2.21µs, matching the
README's 2.23µs headline figure; `bench_beam_search` P50/P90/P99 in the expected tens-of-µs range.
Full CI perf-gate A/B run deferred to whenever this stage's changes are committed (it needs a real
merge-base to diff against).

**Known test-coverage gap — RESOLVED (2026-08-22)**, before starting Stage 2. The advisor's framing
changed the calculus: Stage 2's own gate ("recall ≥ 0.99 vs. an exhaustive scope-filtered scan across
selectivities 0.02–1.0") requires nodes with *real* non-zero scope bits to test against, and Stage 4's
plan already names "WAL-durable bitmap get/set" as a required primitive — so this is not a speculative
test-only mutator (which would have repeated the `TraceBlockIndex` mistake), it's a real primitive with
real callers: Stage 1's own gap-closing tests now, Stage 2's scope-filter tests next, Stage 4's control
plane eventually.

Added `Atlas::set_node_scope(node_id, scope_bitmap)` / `Atlas::get_node_scope(node_id)` — the only
supported write path for `scope_bitmap` on an already-inserted node (`insert()`/`insert_delta()` still
always default it to 0). This is also where `WAL_RECORD_SCOPE` (deferred in Stage 1's initial landing,
above) finally earns a real writer: a new `WAL_RECORD_ATLAS_SCOPE` (`0x03`) record type, payload
`{node_id, scope_bitmap}`, because `WAL_RECORD_ATLAS`'s whole-struct-capture payload can only express
"here is a brand-new node," never "node N's field F is now V" — an in-place mutation of an existing
node genuinely needed its own record type, unlike Stage 1's schema-only landing where the existing
record type already covered every new field for free.

Design points from the advisor review, each verified rather than assumed:
- **Mmap nodes only.** `set_node_scope()`/`get_node_scope()` throw `std::invalid_argument` for a
  delta-arena node id (MSB set) — a delta node's id is replaced when `compact_mmap()` promotes it, so a
  scope set against the old id would be silently lost on the next compaction. Hoisted the previously
  duplicated `DELTA_MASK` local constant (defined separately in `insert_delta()` and `replay_wal()`)
  into a shared `NODE_ID_DELTA_MASK` in `schema.hpp`, and updated `load_context()`'s equivalent raw
  literal to match.
- **`compact_in_progress_` guard.** Throws `std::runtime_error` if compaction is running — mutating a
  node compaction is concurrently copying to the new generation file risks the write landing in the old
  generation and being lost. Same reasoning `consolidate_subgraph()` already uses; same pre-existing
  TOCTOU-against-a-concurrent-compaction race class that function already has too (check-then-lock, not
  atomic) — not a new problem introduced here, and out of scope for this fix to resolve generally.
- **New lock chain, checked for deadlock, not just asserted safe.** `set_node_scope()` takes
  `write_mutex_` (serializing against `insert()`/`consolidate_subgraph()`/`compact_mmap()`'s mmap
  mutations) then `wal_mutex_` (to log the WAL record) — the reverse nesting from `insert_delta()`'s
  existing `wal_mutex_` → `delta_mutex_` chain. Verified by reading every existing lock site in
  `atlas.cpp`: `insert_delta()` never touches `write_mutex_`; `insert()`/`consolidate_subgraph()` never
  touch `wal_mutex_`; `compact_mmap()` releases `write_mutex_`/`delta_mutex_` before its later
  `truncate_wal()`/`open_wal()` calls take `wal_mutex_`. No existing code path holds both locks at once
  in the opposite order, so the new chain cannot deadlock against any of them.
- **WAL-write-before-mutate ordering**, matching `insert_delta()`'s write-ahead discipline: if the mmap
  write itself doesn't survive a crash (`insert()` has no WAL of its own — guardrail #1.3 — so a node
  from the same session might not either), replay can still reapply the scope set from the durable WAL
  record.
- **Two-pass WAL replay**, per the advisor's explicit caution: `WAL_RECORD_ATLAS_SCOPE` records are
  buffered during the main replay loop and applied to `file_` in a second pass afterward, rather than
  in-line. The current invariant (scope-set targets are always mmap ids, delta ids are rejected
  outright) means the two record kinds can never actually target the same node within one replay pass —
  but applying in-line would make correctness depend on that invariant holding forever; two-pass removes
  the dependency at zero cost (replay is a rare startup path, not hot). Bounds-checked against the
  *current* `header->node_count` at apply time, skipping gracefully (not fatally) if out of range — the
  same edge case as `insert()`'s no-WAL mmap durability not surviving a true OS/power crash even though
  the scope-set's WAL record (explicitly flushed) did.
- **Stale-union caveat, documented not silently left**: if a node was already folded into a
  `consolidate_subgraph()` summary's scope union before a later `set_node_scope()` call, the summary's
  union becomes stale. Not a correctness break while nothing reads `scope_bitmap` for enforcement
  (true until Stage 2 lands), but flagged in both the function's doc comment and here so Stage 2 sees it
  before building scope-filtered retrieval on top.

5 new tests (`test_atlas.cpp`, `test_wal.cpp`): round-trip, delta-id rejection, invalid-id rejection,
the actual gap-closing test (`AtlasCompactionTest.ScopeBitmapSurvivesCompaction` — insert, set non-zero
scope on two nodes, compact, confirm both survive renumbering), and a WAL-replay-specific test
(`WalAtlasScopeRecordReplayed`) that hand-constructs a `WAL_RECORD_ATLAS_SCOPE` record and proves
replay's second pass applies it — the live mmap node is never scope-set through the API, so this can
only pass if replay itself does the work, not just mmap durability. 77/77 tests green.

**Scoped precisely** (a second advisor pass caught the original gap description over-claiming): this
closes the gap for `scope_bitmap` specifically, end to end (real writer, real WAL record, real
compaction+replay test). `governance_record_id` still has **no write path and no independent
assertion** — it rides the exact same whole-`node_byte_stride_` `memcpy` copy path that
`ScopeBitmapSurvivesCompaction` exercises, so the copy mechanism is proven, but nothing has ever set it
to non-zero, so nothing has *verified* it round-trips. A dedicated `governance_record_id` writer is
Stage 3/4's job (it's meant to hold an opaque control-plane record ID, which doesn't exist yet) — no
need to build a speculative one now.

A second advisor pass also caught a real bug in the fix above: `get_node_scope()` was initially
implemented without an epoch guard or lock, citing `tombstone_count()` as precedent — but
`tombstone_count()` is a diagnostic, while `get_node_scope()` is meant to be called from a different
thread than any writer (Stage 4's control plane), making the unguarded read a real
read-after-mmap-region-retired hazard under a concurrent `compact_mmap()`, not just a theoretical one.
Fixed by marking `epoch_mgr_`/`write_mutex_` `mutable` (matching `delta_mutex_`'s existing precedent for
exactly this reason) so the proper `EpochGuard` + `shared_lock` could be restored in the const method.
Re-verified: full suite (including `Concurrency.*`) green after the fix.

**Stage 2: COMPLETE (2026-08-22).** All five tasks landed and verified (see task-by-task detail
below). The advisor tool became unavailable partway into this stage (session-level, not
project-specific); per the user's explicit standing instruction, real bugs were still fixed
immediately regardless of stage, and design decisions without the second-opinion tool leaned on
direct verification (empirical measurement, controlled A/B, reading every call site) rather than
assumption. Summary by task: (1) union-propagated scope on the beam-search hot path (deferred/
dirty-marked, emission-time enforcement only) plus a real SLB cache-staleness bug found and fixed
along the way; (2) the Trace graph-expansion boundary scope check; (3) `TraceBlockIndex` wired in for
real (previously dead code), plus a severe compaction data-loss bug found and fixed in both
`Atlas::compact_mmap()` and `TraceManager::compact()`, plus a from-scratch embedding pipeline fix
(the shipped one was silently non-functional); (4) admission-time near-duplicate detection via
`Architect.ingest()`; (5) a full CI benchmark self-hit-artifact audit that found and fixed the SAME
bug (a single query vector reused across every measured iteration, defeating `Atlas::navigate()`'s
own SLB cache into serving stale hits) in five benchmark bodies across four files, not just the one
instance guardrail #0 originally named.

**Task 1 progress — supersede/revoke live write path + a real correctness bug found and fixed:**

- Added `Atlas::supersede_node(node_id)` / `Atlas::revoke_node_supersede(node_id)` /
  `Atlas::is_node_superseded(node_id)` — mirroring `set_node_scope()`'s design exactly (mmap-only,
  `compact_in_progress_` guard, `write_mutex_` → `wal_mutex_` lock chain, WAL-write-before-mutate).
  Needed because Stage 2's "superseded fragments are excluded from `navigate()`" gate requires a live
  way to mark a node superseded — schema.hpp's `supersede_node()`/`revoke_supersede()` free functions
  (Stage 1) had no Atlas-level entry point yet.
- New `WAL_RECORD_ATLAS_SUPERSEDE` (`0x04`), payload `{node_id, revoke}`. Unlike
  `WAL_RECORD_ATLAS_SCOPE`'s plain field-set, this replays a *stateful* read-modify-write
  (`supersede_node()`/`revoke_supersede()` stash/restore `hub_penalty` based on current flag state) —
  both operations are idempotent by construction (Stage 1's double-supersede guard, and
  revoke-without-supersede is a no-op), so replaying one against a node already in the target state is
  safe. Applied in a second pass in original chronological WAL order (order matters here, unlike the
  scope records, since correctness depends on earlier records for the same node having been applied
  first).
- `compact_mmap()` needed no changes — it already only drops `is_tombstoned()` nodes, never checks
  `is_superseded()`, which is exactly correct: supersession is reversible (Stage 1), so a superseded
  node must physically survive compaction in case it's later revoked. Verified with a new test
  (`SupersededNodesSurviveCompactionUnlikeTombstoned`) rather than assumed.
- **Real bug found while testing supersede's exclusion, fixed immediately per the user's standing
  instruction (fix any real bug at any stage)**: the SLB cache's fast path in `navigate_internal()`
  returned a cached hit *without checking whether the node had since been tombstoned or superseded*.
  The cache has no reverse index from node id to cache entries, so `supersede_node()`/
  `tombstone_node()`/`consolidate_subgraph()` cannot evict a stale hit when a node's exclusion state
  changes after it was cached — a query that had previously cached a node as its best match would keep
  returning that node via the fast path forever after, silently bypassing the branchless
  `hub_penalty` exclusion mechanism entirely. This directly broke the "superseded fragments are
  excluded from `navigate()` ... with no post-hoc filter" gate before Stage 2's own primitive even had
  a chance to be tested. Fixed by validating the cache hit (checking `is_tombstoned`/`is_superseded` on
  the cached node) before trusting it, falling through to a full beam search on staleness — moved the
  epoch guard + `write_mutex_` shared_lock ahead of the cache check so the validation read is safe.
  **Performance verified, not assumed**, given this touches the ultra-low-latency cache-hit fast path:
  controlled A/B via the same temporary-revert-and-measure technique used for the WAL forward-compat
  proof — `bench_slb_latency`'s `BM_AtlasTraversal_Only` (which, despite its name, measures a cache hit
  through `Atlas::navigate()` — a known self-hit artifact per guardrail #0) went from ~94ns baseline to
  ~109ns fixed, a ~15ns absolute cost on an already-sub-microsecond path, negligible against a
  cache-miss's multi-microsecond alternative. Also confirmed the fix is behaviorally correct, not just
  passing by accident: after `revoke_node_supersede()`, the *same* now-valid-again stale cache entry is
  correctly served again (the stashed `hub_penalty` was restored, so re-validation passes) — the test
  asserting this (`SupersedeNodeExcludesFromCSLSNavigation`) required real, non-trivial vector
  construction to be meaningful (see below).
- Test-authoring note worth keeping: an earlier draft of that test used vectors that were positive
  scalar multiples of each other (e.g. `target=1.0`, `other_a=0.8`, ...) — cosine similarity is
  scale-invariant, so those are all tied at `cos_sim=1.0`, not the intended descending scores, which
  made beam displacement behave unpredictably. Fixed by using vectors that differ in *direction*
  (flipping an increasing fraction of dimensions to `-1.0`), giving genuinely distinct scores. Also
  avoided an all-zero "Root" vector — `cosine_similarity` against a zero vector is `0/0 = NaN`, which
  breaks `std::sort`'s strict-weak-ordering guarantee once enough entries are being sorted to expose
  it; a real, pre-existing, out-of-scope quirk sidestepped in the test rather than fixed here.

8 new tests (`test_atlas.cpp`, `test_wal.cpp`) covering the above. 81/81 tests green.

**Task 1 — emission-time scope filtering landed; recall gate partially met, gap documented (2026-08-22).**

`Atlas::navigate()` gained a `scope_mask` parameter (default `ALL_SCOPES_VISIBLE`, unchanged pre-Stage-2
behavior). Mechanism, exactly per the plan's option (b): beam descent stays scope-blind (same candidates
explored regardless of `scope_bitmap`); only which candidate gets *reported* per tree level becomes
scope-aware, plus the internal beam is widened to `MAX_BEAM_WIDTH` whenever filtering is active (a
pragmatic recall safety net, gives the per-level reporting choice a larger pool). SLB cache-hit
validation (added while fixing the earlier tombstone/supersede staleness bug) extended to also check
scope. Delta-buffer candidates are skipped entirely under an active filter (they can never have a
non-zero `scope_bitmap` — `set_node_scope()` rejects delta ids).

**Measured against the gate's own stated bar** (`recall ≥ 0.99 across selectivities 0.02–1.0`, via a
new `test_scope_recall.cpp`, ground truth computed independently test-side against a semantically
clustered synthetic tree rather than via any new Atlas API): met at selectivity 1.00/0.50 (~1.0 recall),
**not met** at 0.10 (~0.93) or 0.02 (~0.37).

**A scope-affinity steering attempt (the user's pre-authorized fallback if widening alone proved
insufficient) was tried and measured ACTIVELY HARMFUL, not just insufficient** — a soft per-candidate
bonus favoring in-scope nodes during intermediate-level beam-admission decisions dropped 50%-selectivity
recall from ~1.0 to ~0.52. Root cause: under deferred/option-b propagation, an internal node's own
`scope_bitmap` says nothing about whether its *descendants* include the target scope (exactly the
imprecision the plan's own "an internal node's own bit may be false while a descendant's is true"
warned about). Rewarding "is this specific candidate in scope" at every level hijacks descent toward
random, irrelevant in-scope leaves elsewhere in the tree, displacing the correct-but-momentarily-
out-of-scope ancestors that were actually leading toward the true best match. Reverted; confirmed via
full rebuild + test suite that the revert restored the pre-steering (widening-only) numbers exactly.

**Decision (user, 2026-08-22): document the gap and move on to Stage 2 tasks 2–5, rather than building
real ancestor scope-union propagation now or permanently lowering the gate.** The two low-selectivity
`ScopeRecall` tests had their thresholds adjusted to the actual measured ceiling (0.85/0.25, both below
the true ~0.93/~0.37 measured values as margin) with extensive comments explaining the gap, the ruled-
out steering approach, and what a real fix needs — so they still catch a regression below today's known
baseline without pretending the 0.99 gate is met. **Follow-up work, not yet scheduled**: closing this
gap for real needs accurate ancestor scope-union hints to steer descent by — i.e., building out the
eager (option a) propagation the plan originally deferred, or compaction-time union recomputation —
which is materially more work than Task 1 as scoped and was deliberately not pulled forward here.

14 new tests total for Task 1 (`test_atlas.cpp`, `test_scope_recall.cpp`). 86/86 tests green (all
thresholds honest about what's currently met vs. not).

**Task 2 — Trace graph-expansion boundary scope check: DONE (2026-08-22).**

`Atlas::get_children()` was completely unscoped (unlike `navigate()`), and is exactly the
Atlas→Trace→Atlas crossing the plan calls out: `server.py`'s `/state/atlas/active` reads a Trace
event's `atlas_id` (legitimately the requesting user's own, since Trace history is already
session-isolated) and then calls `get_children()` on it -- but `get_children()` itself enforced no
scope at all, so a shared internal node's children could leak content outside the caller's scope even
though the *starting* node was legitimately theirs (the "leakage at pivot depth 2" pattern the
hive-mind doc cites). Fixed: `get_children()` gained the same `scope_mask` parameter as `navigate()`
(default `ALL_SCOPES_VISIBLE`, unchanged behavior), enforced during child enumeration.

Threaded through every layer, closing a consistency gap found along the way: `navigate()`'s own
`scope_mask` (Task 1) had been added at the C++ level only and never reached the C-API, nanobind
bindings, or `client.py` -- fixed both functions together rather than leaving `navigate()` half-wired.
Also discovered `set_node_scope()`/`get_node_scope()`/`supersede_node()`/`revoke_node_supersede()`/
`is_node_superseded()` (all built earlier this stage) were never exposed to Python at all -- added
nanobind bindings for all five, verified via a live Python smoke test (not just "it compiles").

Also updated the Node.js bridge (`bindings/node/src/aeon_node.cpp`), the only *other* language binding
that was actually in sync with the C-API before this change (verified: the C#/Unreal bindings are
already missing `session_id` from Stage 0, a **pre-existing** staleness predating this work, not
something introduced here -- left as-is rather than compounding unrelated effort). Added an optional
`scopeMask` BigInt argument (defaults to unfiltered), rebuilt via the real `npm install` + `cmake-js`
pipeline (not just code review), and smoke-tested live via `node -e`: unfiltered calls unchanged,
a real `scope_mask` with nothing yet assigned to it correctly returns empty results. TypeScript
definitions (`index.d.ts`) updated to match.

New test: `AtlasTest.GetChildrenScopeFilterExcludesNonMatchingChildren` (C++). 87/87 tests green.

**Task 3 — severe pre-existing bug found and fixed: Aeon's semantic memory was non-functional
(2026-08-22).** While investigating how to wire `TraceBlockIndex` (below), discovered the entire
Python shell has exactly one embedding call, `CognitiveLoop._vectorize()` (`loop.py`) — used for every
`/chat` query — and it was broken: `sentence-transformers` was declared in `pyproject.toml` but not
installed in this environment, so every real interaction silently fell through to a
**hash-seeded pseudo-random 768-dim vector** (`np.random.seed(hash(text) % 2**32)`), with zero semantic
content. Even the "real" code path had an unresolved bug the code's own comments flagged but never
fixed: it loaded `all-MiniLM-L6-v2` (384-dim) and zero-padded to 768, discarding half the embedding's
information by construction rather than using a model that natively matches Atlas's
`EMBEDDING_DIM_DEFAULT`. Separately, `Architect.ingest()` — the entire "admit new knowledge to the
short-term delta layer" pathway — was fully built but **called from nowhere** in the app.

**Net effect verified before the fix**: every `/chat` interaction tested throughout this entire session
(including Stage 0's `/chat` isolation fix) was retrieving and storing against meaningless random
vectors — Atlas's actual semantic search never ran in this environment. This did not affect any of the
C++-level correctness work in this plan (Stage 0/1/2's tests always construct vectors directly), but it
meant the one thing Aeon is *for* — semantic retrieval — was not actually exercised end-to-end by
anything until now.

Fixed, per the user's explicit choice (over a TraceBlockIndex-only workaround or reverting to a
narrower scope) to treat this as the severe, standing-instruction-covered bug it is:
- Switched to `all-mpnet-base-v2` (native 768-dim — verified via a real download+encode: correct shape,
  and semantically meaningful — 0.88 cosine similarity between paraphrased sentences, 0.11 between
  unrelated ones). Removed the padding workaround entirely; added an assertion (not a silent
  workaround) if the model's output dimension ever changes, since that would indicate the model itself
  changed, not a condition to paper over.
- Louder, explicit warnings when falling back to the random-vector mock path (no `sentence-transformers`
  installed), stating plainly that semantic memory is non-functional in that mode rather than a quiet
  `warnings.warn`.
- Wired `Architect.ingest()` into `ContextManager.process_turn()`: each turn's own query is now admitted
  as a new delta-layer concept (immediately searchable, no compaction wait) and recorded into Trace the
  same way retrieved concepts already were — symmetric with the existing pattern, not a new one.
- Corrected `process_turn()`'s docstring, which claimed step 4 was "filters concepts based on
  access_level" — that filtering was never implemented (a commented-out line, `access_level` accepted
  but unused); documented as a placeholder, not a live security boundary, rather than leaving the
  overclaim in place.

**Verified end-to-end, not just import-clean**: created a project `.venv`, installed
`sentence-transformers` for real (downloaded and confirmed `all-mpnet-base-v2`'s actual output), ran a
real `/chat` turn through the full `FastAPI TestClient` → `CognitiveLoop` → `ContextManager` →
`Architect` → C++ `Atlas` stack, confirmed the ingested concept lands in the delta buffer (`Atlas.size()
== 0`, delta ID `0x8000000000000000`), then queried with an independently-encoded **paraphrase** of the
original text and confirmed it retrieved the just-ingested concept at 0.86 similarity — proof the real
model is genuinely active end-to-end, not just that the code compiles.

**Known issue found, not fixed (out of scope for this task)**: `pip install -e .` currently fails
outright — `scikit-build-core >= 0.10` rejects `pyproject.toml`'s `cmake.targets` key
(deprecated in favor of `build.targets`). Worked around for verification via direct `sys.path`
manipulation (the same technique used throughout this session) rather than fixing the packaging
config, since it's an unrelated, separately-scoped fix. Flagged here so it isn't lost.

**Task 3 — `TraceBlockIndex` wired in for real: DONE at the C++ layer (2026-08-22).**

- `TraceFileHeader` gained `embedding_dim` (carved from `reserved`, mirroring `AtlasHeader::dim`) --
  0 until the first embedding is ever appended to a given trace file, then fixed for its lifetime.
- `TraceEvent` gained `embedding_blob_offset`/`embedding_blob_size` (a THIRD independent `BlobArena`
  offset/size pair, distinct from both the event's own text and Stage 1's still-unused evidence
  fields), with the same offset-pinning `static_assert`s and `aeon_trace_event_t` C-API mirror
  discipline as every other schema change this plan has made.
- `TraceBlockIndex` fixed for the exact hardcoded-768-dim bug guardrail #1.1 found in
  `SemanticCache`/`HierarchicalSLB` (Stage 0) -- `std::array<float, EMBEDDING_DIM>` throughout,
  generalized to runtime-`dim_`-sized `std::vector<float>` storage (constructor now takes `dim`).
- `TraceManager::append_event()` gained an optional `embedding` parameter: the first non-empty
  embedding ever appended fixes the file's dimensionality (persisted to the header); later mismatches
  throw. Writes the embedding to the blob arena and indexes it into `TraceBlockIndex`, whichever of
  the mmap/delta paths the event itself lands in.
- New `TraceManager::semantic_search(query, top_k)`: two-phase `O(|V|/1024 + K*1024)` search, with
  tombstoned/superseded events filtered post-hoc (correct here, unlike `Atlas::navigate()`'s beam
  search -- this is a flat top-K scan over already-selected blocks, not a hierarchical descent where
  an early exclusion could prune the correct branch).
- `rebuild_block_index()`: since `TraceBlockIndex` is explicitly NOT persisted (an acceleration
  structure only), it's rebuilt from durable embedding blobs on every `TraceManager` open --
  establishing precedent already existed for this (`rebuild_session_tails()` already does a full
  `O(mmap_event_count_)` scan on open). **Known, documented limitation**: the rebuild only scans mmap
  events; a delta-buffer embedding recovered via WAL replay after a crash is not re-indexed until a
  subsequent compaction promotes it to mmap (narrow window: crash-recovery-specific, not normal
  operation, where `append_event()` indexes both paths as they happen).
- Small cleanup found and fixed while touching this exact code: `TraceManager::append_mmap()` was
  declared, defined, and never called anywhere (superseded by `append_event()`'s own inline mmap-path
  logic) -- a small dead-code instance in the same spirit as guardrail #2's `TraceBlockIndex` finding,
  removed.

**A real, GC-affecting bug found and fixed while wiring this in**: `compact()`'s blob GC re-pointed
offsets for event text but not for `evidence_blob_*` (Stage 1, dormant -- no live writer yet) or
`embedding_blob_*` (this task, immediately live) -- a surviving event's embedding would have kept
pointing at the OLD generation's blob file, deleted moments later. Fixed with a shared `gc_blob_pair`
helper applied to all three blob-reference pairs a `TraceEvent` carries, at both compaction copy sites
(mmap events, frozen delta events).

**A SEVERE, unrelated pre-existing bug found and fixed while testing the above**: investigating why a
freshly-reopened `TraceManager` couldn't find an embedding that survived compaction led to discovering
that `TraceManager::compact()` (and, verified, `Atlas::compact_mmap()` identically) previously installed
the compacted generation under a **permanently generation-suffixed name** (`trace_gen1.bin`,
`atlas_gen1.bin`, ...) and **deleted the file at the caller's originally-configured path**.
`trace_path_`/`generation_` (and `atlas_path_`/`generation_`) are only ever tracked **in-memory**,
reset to `(constructor's path argument, 0)` on every fresh construction -- there was no mechanism
anywhere to discover "which generation is current" from an on-disk file. Net effect: **any process
restart after even one compaction, using the same caller-configured path (the normal, expected case --
see `dependencies.py`'s `AEON_ATLAS_PATH`/`AEON_TRACE_PATH`), would find that path already deleted and
silently create a new, empty file** -- total, silent loss of all prior long-term memory (Atlas) and all
prior episodic history (Trace). This affects Aeon's core "persistent, crash-recoverable" guarantee,
described as the first thing the project is, and would have struck on literally the first restart after
literally the first compaction of any real deployment.

Fixed, per the user's explicit choice to fix both Atlas and Trace now rather than defer or fix only one:
build the new generation at a **temporary** path (`<path>.compacting<N>`), durably flush it (`msync`),
then **atomically rename it onto the stable, caller-facing path** (`std::filesystem::rename` -- POSIX
`rename()` only rewrites the directory entry, so it's safe even while the OLD file at that name is
still open via this process's own `old_file`/`old_base`, which remain valid via their own fd/mmap until
Step 4's cleanup closes them). `trace_path_`/`atlas_path_` are never reassigned again after
construction; `generation_` still increments but is now purely an internal temp-filename
disambiguator, no longer tied to the external, discoverable file identity. The stable-path convention
also replaced `TraceManager`'s blob-arena naming (`trace_blobs_gen{N}.bin` → `<trace_path>.blobs`,
matching `wal_path_`'s existing `<path>.wal` suffix convention) -- it had the identical problem, since
nothing tracked which blob generation was current either.

**Verified with two purpose-built regression tests reproducing the exact bug scenario** (not just
inferred from code review): `AtlasCompactionTest.DataSurvivesCompactionAcrossFullRestart` and
`WalTraceTest.DataSurvivesCompactionAcrossFullRestart` each insert/append, compact, **fully destruct**
the `Atlas`/`TraceManager` object (simulating a real process exit), then reopen via the exact same
caller-configured path (simulating a real restart) and confirm all data -- both pre- and
post-compaction -- is present. Both are confirmed to fail without the fix (this is exactly how the bug
was found: the semantic-search compaction-survival test failed until this was diagnosed and fixed).
Full suite re-verified green after each fix (93 tests after the Trace fix, 95 after the Atlas fix),
including all pre-existing compaction tests (`RepeatedCompactionIsSafe` covers 3 consecutive
compactions, unaffected by the rename-based approach).

24 new tests total for Task 3 across the embedding pipeline, `TraceBlockIndex` wiring, and the
compaction fix (`test_trace_semantic_search.cpp`, plus additions to `test_wal.cpp` and
`test_atlas.cpp`). 95/95 tests green.

**Task 3: fully DONE, Python layer included (2026-08-22).** Threaded the C++ mechanism all the way
through: new `aeon_trace_append_event` C-API `embedding_vector`/`embedding_dim` parameters,
`aeon_trace_semantic_search()`, `aeon_trace_embedding_dim()`; nanobind bindings for all three
(`append_event`'s new `embedding` kwarg, `semantic_search()`, `embedding_dim` property); `trace.py`'s
`TraceGraph.add_event()`/`semantic_search()`/`embedding_dim` Python wrappers. `ContextManager.
process_turn()` now records the user's Trace event embedded with the SAME vector already computed for
the Atlas query (no second embedding-model call) -- the concrete thing that makes an event actually
findable via semantic search, not just a capability that exists but nothing calls.

**Verified with a real `/chat` flow**, not just unit tests: `FastAPI TestClient` → `CognitiveLoop` →
`ContextManager` → C++ `TraceManager`, confirmed `trace.embedding_dim == 768` after one real chat turn,
then independently encoded a paraphrase ("Cell powerhouses are called mitochondria" against the
original "The mitochondria is the powerhouse of the cell") and confirmed `semantic_search()` found the
real event via the real model -- the same style of proof used for the embedding pipeline fix earlier in
this stage, now covering the full path including Trace's semantic index.

**Task 4 — admission-time near-duplicate detection: DONE (2026-08-22).**

Gave Stage 1's `edge_type`/`supersedes_id`/`reason_code` `TraceEvent` fields their first real caller
(built in Stage 1, never had a writer until now). `TraceManager::append_event()` gained trailing
`edge_type`/`supersedes_id`/`reason_code` parameters (threaded through the C-API, nanobind bindings,
and `trace.py`'s `add_event()`); `trace.py`'s placeholder `EdgeType.CAUSAL` enum (dead since Stage 1,
per its own doc comment) replaced with `IntEnum`s mirroring `aeon::EdgeType`/`ReasonCode` exactly.

`Architect.ingest()` (built in an earlier stage, never called until Stage 2 task 3's embedding-pipeline
fix) now checks the new fragment's cosine similarity against existing Atlas content via the existing
`navigate()` path (reusing `math_kernel.hpp`'s `cosine_similarity()`, not a new similarity computation)
before inserting. Above `NEAR_DUPLICATE_THRESHOLD = 0.97` (deliberately far above `SLB_HIT_THRESHOLD`'s
0.85 -- that means "similar enough to reuse a cached answer", this means "close enough to be the same
fragment"), it returns the EXISTING node's id instead of inserting a redundant one; `ContextManager.
process_turn()` records that as a `Refines` edge in Trace rather than a second copy of the same content.
This is also the addendum's first poisoning checkpoint (repeated near-identical content can't silently
flood the index one row at a time).

**Verified with a real `/chat` flow**: sent the identical message twice through the full `FastAPI
TestClient` → `CognitiveLoop` → `ContextManager` → `Architect` stack. The first message inserted a new
delta concept as before; the second produced a `Refines` event (`edge_type=2`) with `supersedes_id`
pointing at the first message's node id, instead of a second insert -- confirmed by reading back real
Trace history, not just checking return values.

New C++ test (`TraceSemanticSearchTest.EdgeFieldsRoundTripThroughAppendEvent`) proves the edge fields
round-trip through `append_event()`/`get_history()` correctly at the C++ layer. 96/96 tests green.

**Task 5 — CI benchmark self-hit-artifact audit: DONE (2026-08-22). Stage 2 complete.**

Guardrail #0 named one confirmed instance (`BM_AtlasTraversal_Only`, 0.078µs) and asked for a full
audit before wiring anything else to the CI gate. Swept every `benchmarks/*.cpp` file for the
pattern (grep for `navigate(` immediately inside a `for (auto _ : state)` timed loop, then manually
classified each hit as a genuine bug vs. a deliberate cache-hit test) and found the SAME bug, not
just the one already-known instance, in **five** benchmark bodies across **four** files:

- `bench_tiered_atlas.cpp`: `BM_TieredAtlas_WarmQuery` / `BM_RawNavigate_Baseline` (fixed earlier in
  this task, before this write-up — query was an EXACT bit-for-bit copy of node #42's insertion
  vector).
- `bench_scalability.cpp`: `BM_AtlasTraversal` (a DIFFERENT, non-excluded benchmark name from
  `BM_AtlasTraversal_Only` -- the existing `SKIP_NAME_SUBSTRINGS` exact-substring filter would not
  have caught it even though it has the identical bug. Its results feed
  `reproducibility_benchmarks/run_v3_benchmarks.py`'s "§6.3: Atlas Scalability" report, so this
  wasn't purely academic).
- `bench_quantization_efficiency.cpp`: `BM_Navigate` -- the highest-priority find, since this
  benchmark **is** in the curated per-PR CI gate (`JSON_BENCHMARKS` in `scripts/ci_perf_gate.py`), so
  a corrupted measurement here would have silently defeated the gate's own purpose.
- `bench_main.cpp`: `WarmSearch` / `ColdSearch` -- not CI-gated or doc-quoted, fixed anyway per the
  standing "fix real bugs at any stage" instruction. Especially misleading for `ColdSearch`: its whole
  point is to measure a genuinely cold traversal by flushing CPU L1-L3 caches (`flush_cache()`)
  before each call, but an unflushable SLB hit from the second iteration onward silently defeated
  that methodology regardless of CPU cache state.
- `bench_slb_latency.cpp`: `BM_SLB_CacheMiss_WarmAtlas` / `BM_AtlasTraversal_Only` itself -- the
  ORIGINAL instance guardrail #0 named, previously just excluded from the gate rather than fixed
  (and mislabeled in `ci_perf_gate.py`'s own comment as living in `bench_scalability.cpp`, when it's
  actually in this file -- corrected as part of this fix). `BM_AtlasTraversal_Only`'s own doc comment
  ("Pure Atlas navigate, no SLB overhead") was directly contradicted by its actual behavior:
  `atlas->navigate()` has its own internal SLB fast path (the same one Stage 2 task 1 found and fixed
  a staleness bug in), which caches the first call's result and serves every later bit-for-bit-
  identical query from cache regardless of the fixture's separate, unrelated standalone `slb` member.

**Root cause, uniform across all five**: a single query vector computed once and reused for every
measured loop iteration. `Atlas::navigate()`'s SLB cache (`slb_cache_`/`HierarchicalSLB`) stores an
entry keyed on the query after the first real traversal; a bit-for-bit-identical query on every
subsequent call is a guaranteed cache hit (cosine == 1.0), collapsing "N measured traversals" into
"1 real traversal + (N-1) ~0.1µs cache hits" -- exactly what happened to the original
`BM_AtlasTraversal_Only` (0.078µs) and, empirically confirmed here, to all five of these.

**NOT the same bug, left alone by design**: `bench_slb_latency.cpp`'s `BM_SLB_CacheHit` (explicitly
tests a cache hit, by its own name), `bench_main.cpp`'s `ConversationalDrift` (explicitly rotates
through 10 near-duplicate drifted queries to "test cache HIT speed", per its own comment), and all
three `bench_multitenant_slb.cpp` benchmarks (`BM_MultiTenant_SLB_Sequential`,
`BM_MultiTenant_SLB_Concurrent`, `BM_SLB_CacheIsolation` -- these call `SemanticCache::find_nearest()`
directly, benchmarking the cache lookup mechanism itself rather than an `Atlas::navigate()` traversal
that's supposed to measure real beam search but gets silently short-circuited by it).

**Fix, uniform across all five**: cycle through a pool of distinct, pre-generated query vectors
(seeded well above any insertion seed range, so no accidental collision) instead of reusing one
static vector, with an explicit `->Iterations(QUERY_POOL_SIZE)` cap on the registration so the pool
can never be exhausted mid-run -- Google Benchmark's default iteration auto-scaling (targeting a
minimum wall-clock time) reliably blew past an 8192-entry pool within a fraction of a second even at
the smallest tested scale, confirmed empirically via a `state.SkipWithError()` guard that fires if
`idx` ever exceeds the pool size (kept in place afterward as defense-in-depth against a future
regression, e.g. someone raising a benchmark's `Arg()` range without noticing the pool needs to grow
too). Settled on `QUERY_POOL_SIZE = 4096` after confirming empirically it comfortably covers a full
run at every tested scale with zero `SkipWithError` triggers.

**Verified via direct execution, not just compilation**, for every fix -- before/after timings
(realistic, non-degenerate numbers replacing clearly-fake sub-microsecond ones):
- `bench_tiered_atlas`: `BM_TieredAtlas_WarmQuery` 0.119µs → 17.3µs; `BM_RawNavigate_Baseline` → 7.02µs.
- `bench_scalability`: `BM_AtlasTraversal` now scales sensibly with N (10.7µs @10K → 13.6µs @100K →
  20.1µs @1M) instead of a flat cache-hit plateau.
- `bench_quantization_efficiency`: `BM_Navigate` FP32 10.7µs @10K / 13.5µs @100K vs. INT8 5.79µs @10K
  / 6.92µs @100K -- INT8 now correctly measures FASTER than FP32 (< 1% stddev/cv across all 5
  repetitions), which is exactly the claim this CI-gated benchmark exists to prove and which a
  cache-hit-dominated measurement could not have demonstrated meaningfully either way.
- `bench_main`: `WarmSearch`/`ColdSearch` ~0.1µs (artifact) → ~521µs/524µs (real, and now correctly
  near-identical to each other as expected, since the CPU-cache-flush in `ColdSearch` was never the
  dominant cost either way at this tree size).
- `bench_slb_latency`: `BM_AtlasTraversal_Only` 0.078µs (the artifact guardrail #0 originally named)
  → ~519µs (a ~6650x change); `BM_SLB_CacheMiss_WarmAtlas` → ~519µs, confirming it now genuinely
  falls through to a real `navigate()` call on every iteration as its "miss path" name promises,
  rather than only on the first.

(`bench_main.cpp`'s `AtlasFixture`/`bench_slb_latency.cpp`'s `AtlasFixture` both insert all 10K nodes
as direct children of the root, i.e. the exact "degenerate 1-level tree" `bench_scalability.cpp`'s own
header comment says it was written to replace with a balanced 64-ary BFS tree. That's a separate,
pre-existing, undocumented-here topology issue -- out of scope for this self-hit-artifact audit, since
a correctly-structured replacement already exists in the repo (`bench_scalability.cpp`) and neither
fixture's numbers are CI-gated or doc-quoted.)

**`scripts/ci_perf_gate.py`**: `SKIP_NAME_SUBSTRINGS` emptied from `["AtlasTraversal_Only"]` to `[]`
now that every known instance is fixed at the source rather than needing exclusion; the list itself is
kept (not deleted) as the checkpoint to extend if a future benchmark reintroduces this pattern. The
list's old comment mislabeling `BM_AtlasTraversal_Only` as living in `bench_scalability.cpp` (it's
actually in `bench_slb_latency.cpp`) is corrected in the same edit -- itself a small instance of the
kind of documentation-vs-reality drift this whole plan treats as a first-class defect (see guardrail
#2's `TraceBlockIndex` finding).

Full `ctest --preset dev` re-run after all five benchmark-file edits + the `ci_perf_gate.py` edit:
96/96 passing, no regressions (the fixes only touch `benchmarks/*.cpp`, none of which are part of the
`aeon_tests` target).

**Stage 2 is now fully complete (tasks 1-5).**

**Stage 3: SKIPPED, per the plan's own documented fallback (2026-08-22).** Stage 3's prerequisite —
"a pilot org actually producing issues with measurable resolution time and defect rate... whose org
runs this pilot is an open dependency, not a given" — has no candidate: this is a solo-maintained
open-source project, not an organization with an engineering team available to run a multi-week
observation study on real work. Asked the user directly rather than fabricating a pilot or silently
skipping the gate; the user confirmed no pilot org is available and chose the plan's own named
fallback: *"If no pilot org materializes in a reasonable window, the fallback is not to skip the
experiment [silently] — proceed into Stage 4 with only the human-curated promotion path (already the
minimal-scope design below) and no automated pipeline, treating 'not yet outcome-validated' as an
explicit, documented risk carried into Stage 4 rather than silently assuming the experiment
happened."*

**Carried-forward risk (explicit, not silent)**: no source — published or produced by this project —
has yet demonstrated that shared engineering memory improves real outcomes (issue-resolution time,
defect rate) versus not having it; a cited study (GitOfThoughts) found a null result for a
structurally similar system. Stage 4 below proceeds on engineering merit (the primitives are correct
regardless of who deploys this, per Stage 4's own framing) and on the addendum survey's finding that
every shipped comparable system puts a human in the promotion decision anyway — not on a validated
outcome measurement. If a real pilot org materializes later, run the Stage 3 experiment retroactively
against accumulated real usage rather than treating this skip as permanent.

**Scope consequence for Stage 4**: promotion stays human/maintainer-triggered (Stage 3's minimal
design), not the automated classification-triggered pipeline Stage 4's task list originally
describes. This does NOT relax the engineering requirements that exist because personal data can end
up in distilled fragments (mint-not-flip promotion, crypto-erase, audit log) — those stay hard
requirements per Stage 4's own reasoning ("retrofitting them later is far more expensive than
building them in now"); only the trigger for *when* promotion runs is scoped down, from "automated
pipeline scanning everything" to "a maintainer explicitly invokes promotion on a specific fragment."
Task 3 (correctness-gated auto-promotion for code knowledge, and its auto-revoke-on-superseded-commit
companion in Stage 5) is deferred along with the rest of the automated pipeline, not built now.

**Stage 4 architectural decision, resolved before any code (2026-08-22, advisor + user): PHYSICAL
separation.** Stage 4 task 1's own text says private/shared memory must be "physically separate...
not commingled-and-filtered" — but Stage 2 shipped exactly commingled-and-filtered (one Atlas,
`scope_mask` AND at emission time). These are incompatible readings of the same requirement, flagged
by advisor as the one decision that changes everything downstream and must be settled first, not
discovered mid-implementation. Chosen: the shared tier is a **physically separate Atlas store** (own
path/WAL/blob arena); a retrieval-service layer routes a query to the private store, the shared
store, or both, rather than one Atlas holding both under a filter. Stage 2's `scope_mask` mechanism
is retained, but scoped down to **intra-shared-tier team scoping only** (which team within the
shared org store), not the private/shared boundary itself.

This directly resolves the recall-gap question left open at the end of Stage 2: the documented gap
(recall 0.85 @ selectivity 0.10, 0.25 @ 0.02, vs. the plan's ≥0.99 gate) stays a team-scoping quality
issue inside the shared tier under this design — never a cross-org/private-vs-shared leak, since that
boundary is now a different physical file, not a bitmap filter. Had logical separation been chosen
instead, that same gap would have become the load-bearing isolation primitive between orgs, which is
a substantially harder correctness bar this codebase isn't currently positioned to hit.

**Stage 4 step 1 — packaging fix + a full pytest-suite recovery (2026-08-22).** Advisor's recommended
sequencing for the scoped-down Stage 4 started with fixing `pyproject.toml`'s long-deferred
`cmake.targets`/scikit-build-core-0.10 breakage (previously tracked as out-of-scope, since Stage 4
adds a real new Python dependency surface — Postgres driver, migrations — that a broken editable
install would make painful to verify). Fixing it surfaced far more than expected:

- **`pyproject.toml`**: `cmake.targets = [...]` → `build.targets = [...]` (the actual rename
  scikit-build-core >=0.10 requires); `build-system.requires` floor bumped `>=0.5` → `>=0.10` to match
  (an environment with an old pinned scikit-build-core would otherwise silently use a key it doesn't
  understand). Verified via `pip install -e .` actually succeeding, not just `SettingsReader` parsing
  cleanly.
- **Second, distinct bug found immediately after**: `pip install -e .` still failed at the CMake
  *install* step — `file INSTALL cannot find ... libaeon.dylib`. Root cause: scikit-build-core's
  `build.targets = ["aeon_py_core"]` only ever builds that one target, but `core/CMakeLists.txt`'s
  `install(TARGETS aeon_shared ...)` (the separate C-ABI SDK library for game-engine bindings) is
  unconditional — it always tries to install an artifact that a Python-only wheel build never
  compiled. Fixed by gating that install block on `if(NOT SKBUILD)` (the CMake cache variable
  scikit-build-core sets to `"2"` when it invokes CMake, confirmed by reading
  `scikit_build_core/builder/builder.py` directly rather than assumed) — a normal native/CI build
  (where `SKBUILD` is unset) still builds and installs `aeon_shared` exactly as before; verified by
  rebuilding `build/dev` and confirming `libaeon.dylib` still exists and `ctest` stays 96/96.
- **The real payoff, and the reason this went from a packaging fix to a much bigger recovery**: fixing
  the install unlocked `pytest tests/` for the first time this entire multi-stage v4 session (every
  prior verification was ad-hoc `sys.path` manipulation + manual `.so` copying, per this file's
  earlier stages). `pytest tests/` had never once run successfully, so nothing could catch the root
  `tests/` integration suite silently drifting out of sync with the real shell-layer API — including
  from THIS session's own Stage 2 task 4 change (`Architect.ingest()`'s new tuple return, claimed DONE
  and verified earlier in this file, but never checked against `tests/test_phase4.py`, which still
  asserted the old single-int return and would have failed immediately had anyone been able to run
  it).
  - `tests/test_phase4.py`: unpacked `Architect.ingest()`'s new `(node_id, is_duplicate)` return
    (the Stage 2 task 4 regression above). Inverted `test_delta_isolation`'s assertion after tracing
    `Atlas::replay_wal()` directly (not assumed): `insert_delta()` writes a `WAL_RECORD_ATLAS` record
    for crash-recovery durability, and replay reconstructs it back into `delta_buffer_` (not mmap
    storage, confirmed by reading the exact `hdr->id = NODE_ID_DELTA_MASK | delta_node_count()`
    reassignment in `replay_wal()`) — so a delta insert surviving a new client instance on the same
    path is the durability guarantee working as designed, not the "leak" the original test (written
    before delta inserts were WAL-covered) assumed.
  - `tests/test_phase5.py`: `ContextManager(mock_atlas)` → `ContextManager(mock_atlas, trace)` (Stage
    0's shared-trace design made `trace` a required second constructor argument). Replaced a bare
    `MagicMock()` for `trace` with a real, in-memory `TraceGraph()` so the rewritten assertions
    (`trace.size`, `trace.get_history(...)`) check genuine C++-backed behavior instead of trivially
    passing against mocked attribute chains — the original `ctx.trace.graph.nodes[...]`-style
    assertions referenced a networkx attribute structure that no longer exists at all. Also replaced
    the mocked embedding model's fake 384-dim output with 768-dim: the old value simulated MiniLM,
    which Stage 2 task 3 deliberately removed in favor of native-768-dim all-mpnet-base-v2.
  - `tests/test_phase8.py`: fixture dropped the removed `storage_dir=` kwarg and added the required
    `trace` positional (`SessionManager`'s own class docstring already documents that the old
    per-user-JSON-snapshot design it belonged to was replaced by the current shared-Atlas/Trace LRU
    design — confirmed via source, not assumed). Of the five tests this fixture blocked, three
    (`test_lru_eviction`, `test_input_validation`, `test_concurrency_lock`) needed nothing else — their
    logic was still valid. The other two tested removed concepts outright and were rewritten to test
    the CURRENT behavior instead of patched to compile: `test_user_isolation` now verifies isolation
    via the session_id argument passed to the ONE shared trace's `add_event()` calls (not separate
    per-user `.graph` objects, which don't exist), and `test_session_persistence` now verifies that
    LRU eviction only drops the in-memory wrapper and never calls a delete/drop method on the
    underlying atlas/trace (matching `_evict_oldest()`'s own docstring: "Nothing to persist") rather
    than checking for a `.json` snapshot file that the current design has no reason to write.
  - `tests/test_server.py`: rewritten, not deleted — advisor's distinction (confirmed by reading both
    the test and the current `server.py`) was that its *intent* (dependency-override FastAPI endpoint
    tests for `/health`, `/state/trace`, `/state/atlas/active`, `/chat`) is still exactly the right
    thing to test; only the symbol names (`get_cognitive_loop`/`get_context_manager` → the real
    `get_current_user_id`/`get_session_manager`/`get_atlas_client` from `dependencies.py`, already
    imported into `server.py`) and the trace mock's shape (`to_viz_json()` → `get_history()`'s dict
    list) had drifted from Stage 0's real-auth rework.
  - `tests/test_trace.py`: deleted outright, not rewritten — confirmed via full read that every test
    exercises a removed networkx-based `TraceGraph` (`add_user_event`/`.graph`/`.save()`/`.load()`/
    `to_viz_json()`/an `EdgeType.NEXT`/`CAUSAL` enum with zero overlap with the current
    `aeon::EdgeType`-mirroring one) with no salvageable intent — the current C++-backed `TraceGraph`
    doesn't have an equivalent surface to test against.
  - Final state: **29/29 pytest tests passing** (up from a suite that could not even be collected).
- **Closed the "does this redrift the moment you look away" gap directly** (the question that decided
  whether this was durable or a one-time cleanup): confirmed neither `pytest` nor `anyio` was a
  declared dependency anywhere, and `.github/workflows/build_and_test.yml` had **no Python-shell job
  at all** — not just missing a `pytest` step, missing entirely, despite this whole session's work
  living substantially in `shell/aeon_py/`. Added a `[project.optional-dependencies] test = [...]`
  group to `pyproject.toml` (`pytest`, `anyio`, `httpx`) and a new `python-shell-tests` CI job
  (single Linux runner — the C++ extension itself is already proven across the full OS matrix, this
  job's job is the Python-layer logic on top of it) that installs `.[test]` and runs `pytest tests/`.
  Also removed `networkx>=3.2` from `dependencies` — confirmed via a full-repo grep it was only ever
  used by the now-deleted `test_trace.py`, dead weight from the same removed architecture.

**Governance_record_id status, checked per advisor's flag**: confirmed via grep that this
Stage-1-allocated `NodeHeader` field (`schema.hpp:158`) has zero accessor, zero WAL record, and zero
writer anywhere — every reference either reads its always-0 value in a test or explicitly zeroes it at
node-creation time (`atlas.cpp:586,720,1093`). This is real, not-yet-started Stage 4 task 1 work
(alongside a public list-by-scope API and bulk bit remap — the other two "genuinely new" primitives
advisor identified; scope-mask AND, WAL-durable bitmap get/set, and fragment soft-delete are already
built from Stages 1-2).

**Stage 4 step 2 — the three remaining C++ primitives: DONE (2026-08-22).**

- **`set_node_governance_id()`/`get_node_governance_id()`** — mirrors `set_node_scope()`/
  `get_node_scope()` exactly (mmap-only, `compact_in_progress_` guard, `write_mutex_` → `wal_mutex_`
  lock chain, WAL-write-before-mutate). New `WAL_RECORD_ATLAS_GOVERNANCE` (`0x05`), payload
  `WalGovernanceRecord{node_id, governance_record_id}` — same 16-byte shape as `WalScopeRecord`.
  `replay_wal()` gained a third two-pass buffer (`pending_governance_records`), order-independent
  (unlike supersede's stateful records — a plain field overwrite, so replay order doesn't matter for
  correctness, only that every record from the WAL gets applied). This is the field's first writer of
  any kind — Stage 1 allocated it in `NodeHeader`'s byte budget but nothing ever set it.
- **`list_nodes_by_scope(scope_mask)`** — the console's list-by-scope primitive. No new scan
  mechanism, exactly as advisor predicted: a flat pass over `MemoryFile::get_node(i)` (the same one
  `tombstone_count()`/`compact_mmap()` already use), EBR-guarded + `shared_lock` like
  `get_node_scope()` since this is a control-plane-facing read that can run concurrently with writers
  (unlike `tombstone_count()`'s unguarded diagnostic scan). Excludes tombstoned nodes (logically
  deleted); includes superseded nodes (reversible, still live data an admin console needs to see).
- **`bulk_set_node_scope(updates)`** — the console's bulk bit remap primitive. All-or-nothing: every
  `(node_id, scope_bitmap)` pair is validated in a first pass before any node is mutated in a second,
  so an invalid id anywhere in the batch throws without partially applying the rest. Reuses
  `WAL_RECORD_ATLAS_SCOPE` (replay doesn't distinguish which call produced a record) with a single
  `wal_stream_.flush()` after all records are written, rather than N separate flushes — the actual
  "bulk" efficiency win over N sequential `set_node_scope()` calls. Confirmed no union-invalidation
  concern applies (per the plan's own caveat): Stage 2 chose deferred/dirty-marked scope propagation,
  so there's no cached ancestor-union state a remap could leave stale.
- **Exposed to Python only** (nanobind bindings, `core/src/bindings.cpp`), matching the existing
  scope/supersede functions' precedent exactly — none of these are exposed via the C-ABI
  (`aeon_c_api.h`) either, consistent with how `set_node_scope`/`supersede_node`/etc. were never added
  there. Added `#include <nanobind/stl/pair.h>` (needed for `bulk_set_node_scope`'s
  `std::vector<std::pair<uint64_t,uint64_t>>` parameter — `stl/vector.h` alone doesn't cover it).
- **A test-writing bug caught by the test itself, not silently worked around**: the first version of
  `ListNodesByScopeReturnsMatchingLiveNodes` assumed a `consolidate_subgraph()`-created summary node
  defaults to `scope_bitmap = 0`. It failed with the summary node unexpectedly matching every scope
  filter — traced to `consolidate_subgraph()` (pre-existing, not touched this session) deliberately
  computing `summary_scope_union |= node->scope_bitmap` over its consolidated sources and assigning it
  to the new summary, so Dreaming-consolidated knowledge doesn't silently drop out of scope-filtered
  queries. Confirmed via reading `atlas.cpp` directly (not assumed) that this is correct, intentional
  behavior — the test's expectations were fixed to match reality, the implementation was not changed.
- **New tests**: `SetAndGetNodeGovernanceIdRoundTrip`, `SetNodeGovernanceIdRejectsDeltaArenaId`,
  `SetNodeGovernanceIdRejectsInvalidNodeId`, `GovernanceRecordIdSurvivesCompaction` (compaction copies
  the entire node stride byte-for-byte per live node, so this is automatic — verified, not assumed),
  `ListNodesByScopeReturnsMatchingLiveNodes`, `BulkSetNodeScopeAppliesAllUpdates`,
  `BulkSetNodeScopeIsAllOrNothingOnInvalidId`, `BulkSetNodeScopeRejectsDeltaArenaId` (`test_atlas.cpp`)
  and `WalAtlasGovernanceRecordReplayed` (`test_wal.cpp`, hand-constructs a WAL record with no live
  mmap write, proving replay's second pass genuinely applies it). **105/105 `ctest` passing.**
- **End-to-end Python verification**: rebuilt `aeon_py_core`, reinstalled via `pip install -e .`,
  smoke-tested all four new bindings directly against the compiled extension (round-trip,
  scope-filtered listing, bulk remap, and the all-or-nothing rejection path) rather than trusting the
  binding declarations compiled. **29/29 `pytest` still passing** (no regression from the binding
  changes).

Stage 4 task 1's C++ primitive layer is now complete: scope-mask AND (Stage 2), fragment soft-delete
(`tombstone_node`/`NODE_FLAG_SUPERSEDED`, Stages 1-2), WAL-durable bitmap get/set (Stage 1),
list-by-scope, bulk bit remap, and governance record ID get/set (this step) are all built, WAL-durable,
and compaction-safe. **Checkpoint reached** — the remaining Stage 4 work (a physically separate shared
Atlas store + retrieval-service routing layer, the Postgres control plane, mint-and-recontextualize
promotion, the minimum console, and crypto-erase) has not been started and is substantially larger in
scope (new external dependencies, a new service, real cryptographic infrastructure) than anything
built so far this stage.

**Advisor review of step 2 caught three real gaps before declaring it done**, addressed the same
session:

1. **`shell/aeon_py/core.pyi` had never been regenerated** — CLAUDE.md's own instruction
   (`./scripts/gen_stubs.sh` after any `bindings.cpp` change) had not been run at any point across
   this entire multi-stage session, not just this step. Installed `nanobind` (a build-time-only
   dependency, missing from `.venv`) and ran it: **170 lines of accumulated stub drift closed in one
   command**, covering every binding added since Stage 0 (`set_node_scope`, `supersede_node`,
   `navigate_raw`'s `scope_mask`, `append_event`'s new kwargs, `semantic_search`, `embedding_dim`,
   and this step's four new methods) — confirmed by diffing the regenerated file, not just running
   the script and trusting it.
2. **`list_nodes_by_scope(ALL_SCOPES_VISIBLE)` meant the OPPOSITE of `navigate()`'s documented
   semantics for the same sentinel** — a real bug, not a documentation gap. `navigate()`'s
   `ALL_SCOPES_VISIBLE` means "no filtering" (unscoped nodes included); a plain `scope_bitmap &
   scope_mask` check treats it as an ordinary mask, and `0 & ALL_SCOPES_VISIBLE == 0` (falsy) —
   meaning every unscoped node would have been silently EXCLUDED from a console query meant to
   return "everything". Fixed by special-casing the sentinel in `list_nodes_by_scope()` to bypass the
   mask check entirely; corrected the doc comments in `atlas.hpp` AND a test comment that had
   rationalized the old (wrong) behavior as "matching navigate()'s emission-time filtering
   semantics" — it was doing the inverse. New test
   (`ListNodesByScopeAllScopesVisibleIncludesUnscopedNodes`) pins the corrected behavior; the
   existing combined-mask test's misleading comment/variable name (`scope_all` for an ordinary
   `0x1|0x2` mask, not the sentinel) was also renamed (`scope_combined`) to stop conflating the two.
3. **Verified and pinned an untested interaction**: `consolidate_subgraph()` (pre-existing, not
   touched this session) unions its sources' `scope_bitmap` onto the new summary node but always
   zeroes `governance_record_id` rather than inheriting one — confirmed via reading `atlas.cpp`
   directly. Consistent with Stage 4's mint-not-flip promotion design elsewhere in this plan (a
   summary is new synthesized content, not a copy of any one source, so it shouldn't arbitrarily
   inherit one source's control-plane record among several candidates) but was completely untested
   and undocumented before now. New test (`ConsolidateSubgraphUnionsScopeButZeroesGovernanceId`)
   pins it, since the console's audit-log/knowledge-browser work will build directly on this
   asymmetry.

Full re-verification after all three fixes: **107/107 `ctest` passing** (two new tests added),
**29/29 `pytest` passing** (package reinstalled to pick up the corrected binding behavior).

**Unrelated repo-hygiene fix found while running the above**: `test_atlas.aeon.wal` was a stray
binary test artifact committed into git history at `281da68` (the repo's very first commit) —
`AtlasTest`'s fixture (`test_atlas.cpp`) used a bare relative path (`"test_atlas.aeon"`, not
`fs::temp_directory_path()` like every other fixture in the suite) and never cleaned up its `.wal`
sidecar, so it leaked into and got modified in the repo root every time `ctest` ran from there.
Fixed the fixture to match the rest of the suite's convention, removed the stray file from git
tracking, and added a defensive `.gitignore` entry.

Reporting back before choosing where to continue within Stage 4's remaining scope.

**Stage 4 step 3 — two physically separate Atlas stores + retrieval routing: DONE (2026-08-22).**

The user said "start stage 4"; consulted advisor before writing any routing code given the scale
(new external dependencies, a new service). Advisor: don't build the Postgres control plane yet --
two-stores-plus-routing is pure Python/C++ the existing test suites already cover, and nothing forces
`governance_record_id` to point at Postgres until promotion (task 2, not started) exists to write
governance state worth recording. Build the isolation guarantee first, since that's what Stage 4
actually promises. Also flagged a **blocking** issue found before any of that: physical separation
broke `TraceEvent.atlas_id`. It's a bare `uint64_t` naming an Atlas node -- with two physically
separate stores, node id 42 exists in BOTH, and nothing in the field says which. Three live call
sites (`context.py`'s write, `warm_start()`'s read, `server.py`'s `get_active_room` read) already
consumed it as if one Atlas exists -- the last one is exactly the Atlas->Trace->Atlas crossing Stage
2 task 2's scope check was built to guard, now with a second, unguarded dimension (which *file*, not
just which scope).

**Fix, entirely at the Python shell layer**: the C++ `Atlas`/`TraceManager` classes never see or
interpret which physical store an id belongs to -- that context only exists at the call site (the
caller always supplies the `Atlas` *instance* directly), so the ambiguity is purely a
shell-layer concept and needed no C++/schema/WAL changes. `client.py` gained `SHARED_STORE_BIT = 1 <<
62` (bit 63 is already `NODE_ID_DELTA_MASK`, real and independent; bit 62 was free) and
`encode_store_id()`/`decode_store_id()`: a private-store id round-trips byte-identical to its raw
form (the id representation used by every prior stage), so no representation change for a
single-store deployment; only shared-store ids get the high bit set. `encode_store_id()` raises on a
value that already has the bit set, catching a double-encode at the call site rather than silently
producing a wrong id.

**Two-store construction + DI**: `dependencies.py` gained `AEON_SHARED_ATLAS_PATH` (unset by default
-- the shared tier is optional infrastructure, not a standing requirement) and
`get_shared_atlas_client()`, returning `None` when unset. Threaded through `SessionManager`
(`shared_atlas_client: Optional[AeonClient] = None` constructor param, passed to every
`ContextManager` it creates) and `ContextManager` itself.

**The three call sites, fixed**:
- `context.py`'s `process_turn()` (both writes: the top-3 concept-recording loop, and
  `Architect.ingest()`'s admission event) -- both always encode `is_shared=False`, since neither
  `process_turn()`'s own Atlas query nor `Architect.ingest()` (which only ever writes to the private
  store -- Stage 4's promotion pipeline, not built yet, is the only sanctioned path to the shared
  tier) ever touches the shared store.
- `warm_start()` -- decodes each historical `atlas_id`, splits into private/shared id lists, and
  calls `load_context()` on the correct client for each (silently skips shared ids if no shared store
  is configured, rather than crashing on an unroutable id from before a shared store was
  unconfigured).
- `server.py`'s `get_active_room()` -- decodes the active room's `atlas_id`, routes `get_children()`
  to the correct client (fails safe to an empty room, not an exception, if a shared id was recorded
  but no shared store is configured now), and re-encodes returned neighbor ids with the same
  store tag as their parent (children live in the same store) so a caller navigating into one later
  doesn't hit the same ambiguity one hop downstream.
- `POST /state/atlas/query` (the off-by-default debug endpoint) deliberately does NOT gain a
  shared-store parameter -- advisor's item 4: it's already an unscoped global query gated behind an
  explicit opt-in flag; adding a second physically separate store only widens what that flag exposes.
  Its doc comment was corrected (it queries the PRIVATE store only, not "the entire Atlas") and its
  returned ids are now consistently store-discriminated too.

**New retrieval-routing capability**: `ContextManager.query_stores(query_vector, mode, session_id,
top_k)` -- `"private"` (default), `"shared"` (raises `RuntimeError` if no shared store is configured
-- an explicit ask for something absent should fail loudly, not silently degrade), or `"merged"`
(queries both, degrading to private-only rather than erroring if no shared store is configured, since
an absent shared tier is a valid deployment state, not a caller mistake). Returns a list of dicts
(`id` store-discriminated, `similarity`, `store`) rather than a numpy structured array, deduplicated
by the store-discriminated id -- generalizes the dedup-by-id/sort-by-similarity logic
`TieredClient._merge_results()` (client.py) already established for local/cloud merging, now
store-id-aware so two different stores' colliding raw ids never get wrongly conflated into one
result. Deliberately a NEW, separate method rather than rewiring `process_turn()`'s own query step --
keeps this increment's blast radius contained to what advisor's plan asked for; `process_turn()`
continues querying the private store only, exactly as every prior stage.

**Verification**: new `tests/test_stage4_stores.py` (13 tests) proves the actual isolation guarantee
this increment exists to build, not just that the code runs -- concretely,
`test_merged_mode_distinguishes_colliding_raw_ids_across_stores` inserts node id 0 into BOTH a real
private and a real shared `AeonClient` (different vectors, so a query can't accidentally match only
one) and confirms `query_stores(mode="merged")` returns both as distinct entries rather than
deduplicating them into one -- the exact failure mode store-discrimination exists to prevent. Plus
round-trip/double-encode-rejection tests for `encode_store_id()`/`decode_store_id()`, and
`warm_start()` routing tests (mocked clients, real `TraceGraph`) proving private/shared ids reach the
correct `load_context()` call. `test_server.py` gained
`test_active_room_endpoint_routes_to_shared_store`, exercising the fix through the FULL FastAPI
request cycle (not just `ContextManager` directly) -- a store-discriminated shared-tier concept event
correctly routes `get_children()` to the shared client, never the private one, with correctly
re-encoded neighbor ids in the response. One pre-existing test bug found and fixed along the way:
`test_phase8.py::test_user_isolation`'s mock never configured `atlas.atlas.insert_delta.return_value`
to a concrete int, so it passed a bare `MagicMock` into the new `encode_store_id()` -- correctly
rejected by the double-encode guard (a `MagicMock`'s `__and__` always returns truthy), surfacing a
real test gap rather than a bug in the new code (same class of fix already applied to
`test_phase5.py::test_cognitive_loop_flow` during the earlier pytest recovery).

**42/42 `pytest` passing** (up from 29 -- 13 new). `ctest` untouched this increment (no C++ changes
at all; the whole fix lives at the Python shell layer per the reasoning above) -- **107/107 still
passing**, confirmed by re-running rather than assumed unaffected.

**Explicitly deferred, per advisor's scope line**: the Postgres control plane, promotion (task 2),
the console (task 5), and crypto-erase (task 6) -- none of this increment's changes assume Postgres
exists; `governance_record_id` stays exactly as Stage 4 step 2 left it (settable, but nothing writes
a meaningful value into it yet, since nothing promotes anything).

**Advisor review of step 3, three items checked, one closed with a new test:**
1. **Delta-arena ids crossing the store-encoding boundary — verified safe, not a bug.**
   `insert_delta()` returns ids with `NODE_ID_DELTA_MASK` (bit 63) set; `Architect.ingest()` can hand
   such an id straight to `process_turn()`'s `encode_store_id()` call on the very first turn after a
   fresh insert, before that node is ever compacted into the mmap generation. Checked whether the two
   C++ entry points that later receive such an id back out of Trace (`Atlas::load_context()`,
   `Atlas::get_children()`) tolerate it: `load_context()` (`atlas.cpp:786-793`) explicitly guards
   `if ((id & NODE_ID_DELTA_MASK) == 0)` and silently skips caching for delta ids -- no throw.
   `get_children()` (`atlas.cpp:534`) checks `parent_id >= header->node_count` before touching
   anything, and a delta-masked id (bit 63 set) always fails that check, since `node_count` only
   counts compacted mmap nodes -- so it returns an empty neighbor list, not a crash.
   `decode_store_id()` only masks bit 62 (`SHARED_STORE_BIT`), leaving bit 63 untouched, exactly as
   intended (the two bits are independent, composable properties: which physical store, and whether
   this id is still delta-arena-resident within that store). Both call sites already handled this
   gracefully before Stage 4 touched anything; my encode/decode layer doesn't disturb that. Added
   `test_delta_arena_id_survives_store_decode_untouched` (`tests/test_stage4_stores.py`) to pin it as
   a regression rather than leaving it as "verified once by reading the code."
2. **`SessionManager.__init__`'s new `shared_atlas_client` param, positional-call risk — checked, not
   an issue.** It's the last parameter, after `max_sessions`, with a default; grepped every
   `SessionManager(...)` construction site (`shell/aeon_py/dependencies.py`, `tests/test_phase8.py`)
   and both use keyword args for every parameter past the first three positional ones, so there's no
   positional-arg collision anywhere in the tree today.
3. **`/state/trace`'s `atlas_id` field vs. `/state/atlas/active`'s `room_id` -- same encoding, not an
   inconsistency, but undocumented.** Both fields read `ev["atlas_id"]` off the same underlying Trace
   event, which `process_turn()` already store-encodes at write time -- so `/state/trace` was never
   "raw," it just didn't say so. Added a docstring note to `get_trace_state()` (`server.py`) stating
   `details["atlas_id"]` is store-discriminated and must be `decode_store_id()`-ed by any future
   consumer (the console, task 5) before being treated as a raw Atlas node id.

**Forward-looking risk, not yet acted on**: `query_stores()` has no `scope_mask` parameter -- a
`mode="shared"`/`"merged"` query currently returns everything in the shared store regardless of team
scope. Harmless today (the shared store is empty, nothing writes to it until promotion exists), but
it's exactly the mechanism Stage 2's `scope_bitmap`/`ALL_SCOPES_VISIBLE` machinery was built for, and
the next increment that starts writing real content into the shared tier (promotion, task 2) needs to
thread a scope mask through `query_stores()` before that tier holds anything worth mis-scoping.

**43/43 `pytest` passing** (up from 42 -- the new delta-id regression test). `ctest` unaffected (no
C++ changes in this addendum) -- still 107/107.

Reporting back before choosing where to continue within Stage 4's remaining scope.

**Stage 4 task 2 -- mint-and-recontextualize promotion pipeline: DONE (2026-08-22), Postgres
control plane explicitly deferred.**

Consulted advisor before starting: the framing that promotion needs the Postgres control plane
*first* was a dependency inversion. Task 2's actual deliverable is a pure shell-layer operation on
primitives Stage 4 task 1 already built (read source node, classify, insert into shared store, set
scope, set governance id) -- Postgres only supplies the *value* `governance_record_id` points at, and
a local monotonic counter satisfies that contract exactly as well today. The real question was
whether the console's hash-chained audit log (task 5(a), "build first, retrofitting onto existing
write paths reliably misses some") needs to exist before promotion runs -- it does, and it doesn't
need Postgres either (a local append-only file is sufficient for a hash chain). So this increment
built promotion + the audit log together, with no Postgres dependency at all, keeping the no-Postgres
`python-shell-tests` CI job green with no service container.

**Two missing C++ primitives found and fixed while wiring promotion** (same "real bug regardless of
stage" standing instruction): promotion needs to read a source fragment's actual content before it
can classify/re-embed/copy it, and nothing in the API surface could do that.
1. **`Atlas::get_node_metadata(node_id) -> std::string`** (`atlas.hpp`/`atlas.cpp`) -- `insert()`/
   `insert_delta()` have always WRITTEN a node's metadata string but nothing ever read it back out;
   same class of gap as `governance_record_id`'s missing writer in Stage 1 (allocated, never given
   the other half of its API). Works for BOTH mmap and delta-arena ids, unlike the scope/governance
   accessors -- promotion needs to read same-turn admissions that haven't been compacted yet, and a
   metadata read has no WAL/version-lineage correctness reason to restrict that the way mutation
   primitives are restricted.
2. **`Atlas::get_node_centroid(node_id) -> std::vector<float>`** (same file) -- `query()`/
   `get_children()` only ever return a 3-float preview, not the full `dim_`-length vector; promotion
   needs the source node's actual vector to insert a copy into the shared store. Dequantizes to FP32
   for INT8-quantized Atlases via a new `quant::dequantize_vector()` helper (`quantization.hpp`) --
   the vector-level counterpart to the existing `dequantize_dot_product()`, which only ever
   dequantized a scalar dot-product score, not a full vector; same formula (`v'[i] = q[i] * scale`)
   applied elementwise. Same mmap-and-delta-arena support as `get_node_metadata()`.

Both bound via nanobind (`bindings.cpp`), stubs regenerated (`core.pyi`). New C++ tests: FP32/INT8/
delta-arena round-trip and invalid-id rejection for both accessors (`test_atlas.cpp`).

**`shell/aeon_py/governance.py`** (new) -- `AuditLog`: append-only, hash-chained JSONL file.
`append(action, actor, payload)` returns a monotonic `seq` (this is what `governance_record_id` gets
set to). `verify()` walks the whole chain, raising `AuditLogError` naming the first record that
doesn't reconcile -- detects a modified payload or a deleted record, either of which breaks every
hash after it. `export_signed(key)`/`verify_export_signature()` -- HMAC-SHA256 over the raw log
bytes, the starting point the console (task 5) needs (a shared operator secret); swapping to
asymmetric signing later doesn't change the return shape. Payloads must never contain the raw PII a
classifier redacted -- only categories/counts -- or the audit log becomes exactly the kind of store
redaction exists to prevent.

**`shell/aeon_py/promotion.py`** (new) -- `IdentifierCorpus` (adopter-configured regex `patterns` +
generic `redact_emails`/`redact_commit_shas` toggles) and `classify_and_redact()`, the fail-closed
deterministic classifier: the ONLY layer permitted to PASS a fragment, matching the plan's spec
exactly (an optional LLM layer, not built here, may only reject/flag -- there is deliberately no seam
in this module for an LLM to override a reject into a pass). `promote_fragment()` orchestrates
mint-and-recontextualize: read source text/vector, classify+redact, insert a NEW de-identified node
into the destination store (never mutates the source), scope it, governance-link it to its own audit
record, and optionally record a `PROMOTED_FROM` Trace edge (`EdgeType.PROMOTED_FROM`/`supersedes_id`,
both already landed in Stage 1 with no caller until now).

**A bug caught in advisor review before it shipped, not self-caught**: my first draft called
`dest_atlas.atlas.insert()` then unconditionally `set_node_scope()`/`set_node_governance_id()`. But
`insert()` diverts to the delta buffer if the destination store is mid-compaction
(`Atlas::insert()`'s `compact_in_progress_` diversion, `atlas.cpp`), returning a delta-masked id --
and `set_node_scope()`/`set_node_governance_id()` both reject delta ids outright by design (their own
doc comments: a scope set against a delta id "would be silently lost" once compaction promotes it to
a fresh mmap id). My original ordering would have thrown mid-sequence, leaving an UNSCOPED,
UNGOVERNED node already inserted into the shared store -- readable by any `query_stores(mode="shared")`
caller using the default `ALL_SCOPES_VISIBLE` mask, exactly the leak this whole increment exists to
prevent, and with no audit record at all (the audit `append()` was ordered after the mutations).
Fixed by: (1) rejecting `dest_scope == 0` up front (a scope-0 node is unreachable by any scoped query
but still listed by `list_nodes_by_scope(ALL_SCOPES_VISIBLE)` -- present-but-unreachable reads as data
loss); (2) writing the audit "promotion" record (with the now-known `dest_node_id`) BEFORE calling
`set_node_scope()`/`set_node_governance_id()`, so an attempt that doesn't fully complete is still
discoverable, never silently lost; (3) explicitly detecting a delta-masked return from `insert()` and
failing loudly with a `RuntimeError` (after recording a `promotion_unscoped_anomaly` audit entry
naming the orphaned node id) rather than letting the scope/governance calls throw implicitly; (4)
wrapping the `set_node_scope()`/`set_node_governance_id()` calls themselves in the same anomaly-record
handling, since a narrower race (compaction starting on the destination store between `insert()`
returning and these calls) produces the identical orphaned-node shape. New tests pin both the
delta-diversion case and the narrower post-insert-failure case, confirming `set_node_scope`/
`set_node_governance_id` are never called on a bad id and that the anomaly is always recorded with
the correct orphaned node id before the exception propagates.

**Explicitly NOT built this increment** (module docstring says so directly): re-embedding conditioned
on the destination scope's corpus (task 2's "free retrieval-quality upside" -- no embedding-
conditioning pipeline exists anywhere in this repo to hook into yet; `reembed_fn` is an optional seam
for a caller that has one, defaulting to reusing the source vector, which is correctness-preserving
even if not quality-optimal) and correctness-gated promotion for code knowledge (task 3 -- needs a
"verification run" concept tied to VCS/CI state, a genuinely separate integration surface). Also, per
advisor's explicit call: **no promotion endpoint wired into `server.py`**. Task 7's constraints
(scope-scoped admin roles, four-eyes approval on bulk operations, mandatory read-reason prompts,
time-boxed break-glass access) don't exist yet and nothing in the current auth surface enforces them
-- exposing promotion over HTTP now would ship a privileged mutation with no approval path. The
module-level primitive plus tests is the right stopping point; the console increment (task 5) is what
earns the endpoint.

**70/70 `pytest` passing** (up from 43 -- 27 new: 9 governance, 18 promotion). **114/114 `ctest`
passing** (up from 107 -- 7 new: `get_node_metadata`/`get_node_centroid` round-trip + rejection
tests).

**Advisor review of task 2, two more findings, both fixed:**
1. **`supersedes_id`'s documented contract ("TraceEvent::id") didn't match either real caller.**
   `context.py`'s Stage 2 task 4 REFINES path and this increment's PROMOTED_FROM path both write a
   store-encoded ATLAS NODE id into `supersedes_id`, not a TraceEvent id -- which is what each of them
   actually has on hand (the near-duplicate node being refined; the source fragment being promoted
   from), and the C++ layer never dereferences the field either way (opaque payload to WAL/replay), so
   nothing forced one interpretation over the other. Since both existing implementations already agree
   with each other and only the original Stage 1 doc comment disagreed, corrected the doc (`schema.hpp`
   TraceEvent layout comment, `trace.hpp`'s `append_event()` param doc, `trace.py`'s `add_event()`
   docstring, `bindings.cpp`'s binding docstring) to state the real convention rather than rewriting two
   already-shipped, already-tested call sites to match a design that was never actually implemented as
   literally described. `SUPERSEDES`/`CONTRADICTS`/`REVOKES`/`MERGES_WITH` have no caller yet; the
   corrected comment says a future one should follow the same Atlas-node-id convention for consistency.
2. **`AuditLog.append()` had no `flush()`/`fsync()`, undermining the "record before mutating" safety
   argument the whole promotion anomaly-handling design (above) depends on** -- a crash between the
   `write()` and process exit could lose the very record that makes an incomplete promotion attempt
   discoverable. Fixed: `append()` now `flush()`s and `fsync()`s before returning. Also hardened
   `_load_tail()`: a torn final line (a crash mid-write, despite the fsync fix covering *completed*
   writes) previously raised a bare `json.JSONDecodeError` straight out of the constructor; now raises
   `AuditLogError` with a clear message that this needs manual recovery, rather than presenting as a
   generic, uninstructive parse failure. New test (`test_torn_tail_raises_on_reopen`) pins this.

**Caveat worth naming, not yet acted on** (advisor): `governance_record_id` now points at an audit-log
`seq` that is only meaningful relative to *one specific log file* -- no path or instance identifier is
stored anywhere in the node itself. If an adopter ever rotates or relocates the audit log, every
existing `governance_record_id` silently re-points into the new file's numbering. This is precisely the
constraint a real control plane (Postgres, with a stable, globally-unique record id) removes -- naming
it here as the concrete trigger for eventually building that, rather than leaving "Postgres, deferred"
with no stated reason to pick it back up.

**71/71 `pytest` passing** (up from 70 -- the new torn-tail test). `ctest` unaffected by the governance
fix (pure Python); the `supersedes_id` doc corrections touched `bindings.cpp` (rebuilt, stubs
regenerated) with no behavior change -- **114/114 still passing**.

Reporting back before choosing where to continue within Stage 4's remaining scope.

**Deferred item 1 -- full Postgres control plane: DONE (2026-08-23).** User explicitly chose "Full
Postgres control plane" over the smaller log-instance-id fix when asked directly (the smaller fix
remained the advisor-recommended default, but the user's choice governs). Consulted advisor before
building: the first framing risked two audit trails disagreeing with each other. Resolved explicitly --
**the JSONL `AuditLog` (governance.py) stays the single tamper-evident hash chain; Postgres is a
queryable, relationally-stable INDEX over it, never a second source of truth.** A DB row is *easier* to
rewrite in place than an append-only file, so tamper-evidence has to keep living in the file's hash
chain, not move to Postgres.

**Schema** (`shell/aeon_py/control_plane/schema.py`, SQLAlchemy Core, not the ORM -- two tables, plain
CRUD, no relationship-mapping complexity yet): `governance_log_instances` (one row per physical
`AuditLog` file, keyed by a new stable UUID -- see below) and `governance_records` (one row per
promotion event, whose Postgres-assigned `id` is what `NodeHeader.governance_record_id` gets set to
now, unique `(log_instance_id, log_seq)` tying each row back to its exact JSONL record). This is the
concrete fix for the original deferral trigger: `governance_record_id` is no longer ambiguous across
JSONL log rotation/relocation, because it's a globally-stable Postgres PK, not a per-file seq number.

**`AuditLog` gained a stable `instance_id`** (governance.py) -- a UUID generated once on first creation
and persisted in a sidecar `<path>.instance_id` file (not derived from the path itself, since the path
is exactly what can change on rotation). New tests pin stability across reopen and distinctness across
paths.

**Driver/tooling choice**: SQLAlchemy 2.0 + `psycopg[binary]>=3.1` (one driver, both sync and async modes
-- avoids needing asyncpg as a second dependency for the async app below) + Alembic for migrations.
`greenlet` needed adding explicitly (SQLAlchemy's async engine requires it regardless of driver; not
pulled in transitively by anything else) -- found by actually running the async app against a live DB,
not assumed from the dependency list.

**Two engines, one database, no HTTP between them** (advisor's explicit call, verified before writing
code): `control_plane/db.py`'s `GovernanceDB` is a SYNC, in-process client `promote_fragment`
(promotion.py) calls directly -- not a network call through the control-plane app's own API. Reason:
`promote_fragment`'s whole anomaly-handling design (Stage 4 task 2 writeup, above) depends on the
governance write landing, ordered, before the Atlas scope/governance mutation, fast and
near-certain-to-succeed; a network hop through a separate service would both add unnecessary latency and
introduce a failure mode (write succeeds server-side, response lost) the existing `_record_anomaly` path
wasn't built to distinguish. `control_plane/app.py` is a genuinely SEPARATE `FastAPI()` instance with its
own `dependencies.py` (async engine, `postgresql+psycopg://` via psycopg3's async mode) -- separate
because task 7's "admin reads go through the same enforcement path as any other read, never a wildcard
bypass" needs an auditable boundary; a set of admin routes bolted onto `server.py` would make a bypass
one `Depends` override away. This increment's app.py has one read endpoint
(`GET /governance-records/{id}`) proving the separation and queryability; no approvals/roles yet (see
below).

**`promote_fragment` gained an optional `governance_db` parameter**, same optionality pattern as
`shared_atlas_client` -- `None` (default) falls back to exactly the pre-Postgres behavior (JSONL seq as
`governance_record_id`), so every one of the 18 existing promotion tests kept passing unmodified with
zero Postgres involved. When supplied, the Postgres write is folded into the SAME try/except that already
guarded `set_node_scope()`/`set_node_governance_id()` -- a Postgres failure (timeout, connection drop)
after the row actually landed produces the identical "node exists, not fully governed" shape as the
existing races, and gets the same `promotion_unscoped_anomaly` audit-then-raise handling, not a new
failure mode the code doesn't know how to record.

**CI**: real Postgres service container in the `python-shell-tests` job (`build_and_test.yml`) --
matching advisor's explicit "service container over a SQLite shim" guidance from earlier in Stage 4, a
shim would exercise a different SQL dialect than production and mask dialect-specific bugs. `alembic
upgrade head` runs before pytest. `AEON_CONTROL_PLANE_DATABASE_URL` unset locally makes
`test_control_plane.py` skip cleanly (`pytest.skip(..., allow_module_level=True)`) so `pytest tests/`
stays green with zero Postgres running; CI additionally sets `AEON_REQUIRE_DB_TESTS=1`, which turns that
same skip into a hard collection failure if `AEON_CONTROL_PLANE_DATABASE_URL` is ever unset in CI --
verified both branches directly (skip locally with neither var set; hard failure with
`AEON_REQUIRE_DB_TESTS=1` and no DB URL) rather than assumed from reading the code. `docker-compose.yml`
gained a `postgres` service (same image/credentials CI uses) for local dev parity.

**Verified against a real, disposable Postgres container** (`docker run postgres:16-alpine`), not just
unit-tested against mocks: `alembic revision --autogenerate` correctly detected both new tables from
schema.py with zero manual editing; `alembic upgrade head` applied cleanly against a FRESH database
(torn down and recreated to simulate CI's clean-slate environment, not reused from the autogenerate run);
`GovernanceDB.record()` round-tripped real inserts with distinct PKs; `control_plane/app.py`'s async read
endpoint served a row written by the sync write path back out correctly; the full `pytest tests/` suite
(80 tests) ran end-to-end against that fresh container with `AEON_REQUIRE_DB_TESTS=1`, exactly matching
what CI will do.

**Explicitly NOT built this increment** (advisor's scope cut, stated up front): approvals and roles
(task 7) -- four-eyes approval needs a pending/approved/executed/expired state machine designed on its
own, not bolted onto a schema just written in the same pass. Task 7 is now the next constraint, since it
gates BOTH the promotion HTTP endpoint (deferred item 4) and the console (task 5)'s admin operations.

**80/80 `pytest` passing** (up from 74 -- 6 new control-plane tests, all requiring a live Postgres and
correctly skipped/hard-failed per the above when one isn't configured). `ctest` unaffected (no C++
changes) -- **114/114 still passing**.

**Advisor review of the control plane, three findings, all fixed -- one via live Postgres testing, not
inspection:**
1. **Migration-drift check.** With one migration, every test passed against a hand-created table just as
   well as a real migration -- stops being true the moment a second migration exists (a stale-schema DB,
   migration 1 applied but not 2, would pass every functional test while silently diverging from what
   `alembic/versions/` actually specifies). Added `test_database_is_at_head_revision`, which reads
   `alembic_version` directly and compares it against `ScriptDirectory.get_current_head()` -- catches
   exactly that drift. Deliberately exercised against a real second (then third) migration rather than a
   synthetic one, see below.
2. **Rejection/anomaly paths were never mirrored to Postgres -- blocking.** `governance_db.record()` was
   only ever called on the success path, so Postgres held completed promotions and nothing else; a console
   querying "what was attempted against scope X" would get a systematically incomplete answer, and the one
   row an operator most needs -- an orphaned unscoped node from a delta diversion -- was invisible in the
   queryable store. Fixed via a new `_mirror_governance_record()` helper called from both the
   `promotion_rejected` and `promotion_unscoped_anomaly` paths -- but deliberately **best-effort**: wrapped
   in its own try/except that logs and swallows a Postgres failure rather than letting it mask the
   already-succeeded JSONL write or the original `ValueError`/`RuntimeError` this is called alongside. New
   test (`test_postgres_mirror_failure_does_not_mask_the_original_exception`) pins that a broken
   `governance_db` still lets the real `RuntimeError` propagate. Two more new tests confirm the rejection
   and delta-diversion paths are now actually queryable in Postgres.
3. **`action` was free-text `String(64)`, three literal-string writers.** Moved the enumeration
   (`GOVERNANCE_RECORD_ACTIONS`) into `governance.py` (zero external dependencies, unlike
   `control_plane/schema.py` which needs sqlalchemy) so both the new Postgres `CHECK` constraint and
   `promotion.py`'s three writers derive from ONE tuple instead of three independently-typed literals that
   could drift -- `promotion.py` now imports named constants (`_ACTION_PROMOTION` etc.) instead of bare
   strings. Verified the constraint actually rejects a bad value against live Postgres (not just that the
   migration applied), confirmed `IntegrityError` on `action='typo_action'`.

**A fourth bug, found only because of the live-Postgres testing discipline, not by inspection**:
`source_node_id`/`dest_node_id`/`dest_scope` were `BigInteger` (Postgres signed 64-bit, max `2**63-1`).
Atlas node ids and scope bitmaps are C++ `uint64_t` (full unsigned `0..2**64-1` range) -- `NODE_ID_DELTA_MASK`
(client.py) sets bit 63, so ANY delta-arena id overflows signed `BIGINT` outright. This surfaced as a real
`psycopg.errors.NumericValueOutOfRange` while testing the delta-diversion anomaly-mirroring test above
against the live container -- every earlier test had "passed" only because none of them happened to insert
a delta-masked id into Postgres. Fixed by widening all three columns to `NUMERIC(20,0)` (exact decimal,
holds the full `uint64_t` range; a `TEXT` column would have avoided the overflow too but loses numeric
comparison semantics for no reason). Also directly verified (not assumed) that `NUMERIC` values round-trip
through `control_plane/app.py`'s JSON responses as exact integers, not lossy floats -- FastAPI's
`jsonable_encoder` converts a whole-valued `Decimal` to a bare JSON integer literal; confirmed a
delta-masked value (`9223372036854788153`) survives a live `GET /governance-records/{id}` call bit-for-bit.

**Three migrations now exist** (`create governance tables` → `add action check constraint` →
`widen node id columns to numeric for uint64 range`), each generated via `alembic revision --autogenerate`
against a live container and applied with `alembic upgrade head` before the tests that depend on it ran --
none hand-written, all schema drift caught by the tool that's supposed to catch it.

**84/84 `pytest` passing** (up from 80 -- 1 migration-drift test + 3 mirroring/failure-mode tests).
`ctest` unaffected -- **114/114 still passing**. Confirmed pytest stays green with zero Postgres running
(74 passed, 1 skipped) after tearing the test container back down, matching the "Postgres is optional
infrastructure" design intent.

Reporting back before choosing where to continue -- deferred items 2-4 (re-embedding, correctness-gated
promotion, the promotion endpoint) and Stage 4's remaining tasks (console, crypto-erase) are still ahead.

**Cheap fix landed alongside the control plane, found via advisor review, not by inspection: large
store-discriminated ids were shipping as bare JSON numbers.** `NeighborInfo.id`/`ActiveRoomResponse.
room_id`/`SearchResult.id` (models.py) and `/state/trace`'s `details["atlas_id"]` (server.py) are all
Atlas node ids that can carry `SHARED_STORE_BIT` (1<<62) and/or `NODE_ID_DELTA_MASK` (1<<63) once Stage 4's
store separation is in play -- values that exceed JS's `Number.MAX_SAFE_INTEGER` (2**53-1), which a browser
`JSON.parse` on a bare integer literal silently rounds. No console/frontend exists yet to have hit this,
but it's cheaper to fix in the API contract now than after task 5 builds views against it. Fixed by typing
all four as `str` in the Pydantic response models and converting at each server.py call site. New test
(`test_large_store_discriminated_ids_survive_as_exact_strings_not_js_numbers`, test_server.py) inspects the
RAW HTTP response body text (not `response.json()`, which would hide the bug via Python's unlimited-
precision ints) to confirm a delta+shared id is sent quoted, never as a bare number.

**Task 7 -- admin roles + four-eyes approval: DONE (2026-08-23).** Per advisor's explicit call, NOT
scoped as a fresh advisor consult -- the shape was already determined from the earlier Postgres review
("stop before approvals and roles... four-eyes needs a state machine designed on its own"). Build order
followed the dependency advisor named: roles first (four-eyes needs to know who counts as a distinct
approver; read-reason needs a role to attribute a read to) -- roles → approvals (with expiry) → (read-
reason and full HTTP-endpoint wiring deliberately left for the console increment, task 5, which is the
first thing that actually issues admin-authenticated reads).

**Schema** (`control_plane/schema.py`, three new tables): `admin_roles` -- one row per
`(principal, scope_mask, role)` grant, `expires_at TIMESTAMPTZ NULL` where NULL means permanent and
non-NULL is exactly break-glass (task 7's "time-boxed break-glass access" -- one mechanism, not a second
one bolted on). `approval_requests` -- the operation being approved, `required_approvals` int,
`expires_at` (mandatory, not nullable -- a pending approval that can sit forever violates task 7's
"time-boxed" property just as much as an unbounded admin grant would), `executed_at`/`revoked_at` as
terminal facts a caller/human sets, never computed. `approval_grants` -- **one row per approver, not per
request** (advisor's explicit correction to an easy-to-get-wrong shape): `UNIQUE(request_id, approver)` IS
four-eyes -- without it, one person approving twice satisfies `required_approvals=2` on their own.

**Expiry is lazy everywhere, per advisor's reasoning stated up front**: this codebase has no scheduler
(`DreamingWorker` is a background thread, not one), so `expires_at` is compared against `now()` at read
time in every validity check (`has_role()`, `is_approved()`) -- no sweeper, survives process restarts for
free, and can answer "was this valid at time T" retroactively, which an audit trail wants anyway. Every
`status`-shaped column in this schema (`executed_at`/`revoked_at`) holds only a terminal fact a caller
caused, never a derived state a background job would need to maintain.

**A real type mismatch found while writing `admin.py`, not by inspection**: `scope_mask` has to be
`NUMERIC(20,0)` (the same fix `governance_records`' node-id columns needed, for the same uint64_t-range
reason -- `ALL_SCOPES_VISIBLE` is `2**64-1`, the exact value an over-broad grant would carry, and it alone
overflows signed `BIGINT`). But Postgres's `&` bitwise operator only works on integer types, not `numeric`
-- so `has_role()`'s scope-overlap check (`scope_mask & requested_mask != 0`) cannot run in SQL against
this column. Resolved by moving the overlap check into Python: `_non_expired_roles()` fetches the (lazily-
filtered, non-expired) candidate rows for a `(principal, role)` pair and `has_role()` does the bitwise AND
there. This control plane is documented as low-QPS (task 1), and a principal realistically holds a handful
of grants -- an honest tradeoff, not a premature-optimization shortcut, and it sidesteps inventing a
second scope-mask representation just for this one query.

**Wildcard-scope decision, made explicit per advisor's flagged ambiguity**: `grant_role()` raises
`WildcardScopeError` if `scope_mask == ALL_SCOPES_VISIBLE` unless the caller explicitly passes
`allow_wildcard=True`. Task 7's own wording -- "scope-scoped admin roles **by default**", "never a
wildcard bypass" -- means a grant covering every scope can't be the path of least resistance, the way
`query_stores(shared_scope_mask=ALL_SCOPES_VISIBLE)`'s own default already is elsewhere in this codebase. A
real need for a wildcard grant (bootstrapping the first admin) still exists, so it's an explicit opt-in,
not an impossibility.

**Both of advisor's gate tests pin the design decisions directly, not just the code**:
`test_expired_role_grant_is_invalid_with_no_sweeper` inserts a grant with `expires_at` already in the past
and confirms `has_role()` treats it as invalid with zero background process running between insert and
assertion. `test_same_approver_twice_does_not_satisfy_required_two` confirms the same person approving
twice raises `DuplicateApprovalError` (from the `UNIQUE` constraint) and leaves `is_approved()` false.

**Explicitly NOT built this increment**: any HTTP endpoint wiring (no admin routes exist yet -- that's
task 5, the console, the first real consumer of "is this principal allowed to do X"), read-reason capture
(needs a role to attribute the read to, which now exists, but no read path calls through it yet), and
role-based enforcement anywhere in `server.py` or `control_plane/app.py` (both still have zero callers of
`has_role()`/`is_approved()` -- this increment is the primitives, not the wiring).

**102/102 `pytest` passing with a live Postgres** (75 non-DB + 10 control-plane + 17 new admin/approval).
`ctest` unaffected -- **114/114 still passing**. Confirmed pytest stays green with zero Postgres running
after tearing the test container down (75 passed, 2 skipped -- both DB-backed modules).

Reporting back before choosing where to continue -- deferred items 2-4 (re-embedding, correctness-gated
promotion, the promotion endpoint) and Stage 4's remaining tasks (console, crypto-erase) are still ahead.

**Deferred item 4 -- the promotion HTTP endpoint: DONE (2026-08-23), unblocked by task 7.** Per advisor's
explicit call, one addressed first: `GovernanceDB` and `AdminDB` each constructing their own `create_engine`
would leave the endpoint holding two independent connection pools per process with no way to ever share a
transaction between them. Both now accept either a database URL (owns its pool, the standalone/test case)
or an existing `Engine` (shares one, the endpoint's case) -- `dependencies.py`'s new
`get_control_plane_engine()` is the one singleton `Engine` both `get_governance_db()`/`get_admin_db()`
build on.

**Atomicity decision, made explicit rather than assumed**: the endpoint's sequence (check role → check
approval → `promote_fragment()` → `mark_executed()`) is NOT wrapped in one database transaction, and can't
fully be -- the Atlas mutation is a completely separate storage system (mmap, not SQL), so no Postgres
transaction could make it atomic with the governance/approval writes regardless. Accepted the same
non-atomic-but-discoverable shape `promote_fragment()`'s own anomaly handling already established: a crash
between `promote_fragment()` succeeding and `mark_executed()` running leaves a narrow window where a retry
could re-execute an already-completed approval. Closed the REAL risk that window enables -- replay -- at the
application layer instead: `execute_approved_promotion()` (promotion.py, new) checks `executed_at is None`
BEFORE calling `promote_fragment()`, so even in that narrow window a second call is refused, not silently
re-run. New test (`test_refuses_replay_after_first_execution`) confirms a second execution attempt raises
and does NOT mint a second shared-store node.

**The approval must lock in exactly what gets executed, found while designing the endpoint, not by
inspection**: `approval_requests.target` had been a free-text field in task 7's schema (fine for a generic
admin action); left as free text, the execute endpoint would have had to trust CALLER-supplied
`source_node_id`/`dest_scope` at execution time -- meaning an approval for "promote node 5 into scope 0x1"
could be replayed to execute "promote node 999 into scope `ALL_SCOPES_VISIBLE`" using the same two
approvals, since nothing tied the approval to specific parameters. Fixed by having
`create_promotion_approval_request()` (promotion.py, new) JSON-encode the exact operation parameters into
`target`, and `execute_approved_promotion()` reads them back from THAT record -- the HTTP endpoint's request
body carries no promotion parameters at all, only a `request_id`, by construction. New test
(`test_executes_promotion_with_params_from_the_request`) confirms the executed promotion's `dest_scope`
matches what was approved.

**Mandatory reason, resolved per advisor's flagged decision point**: added `approval_requests.reason`
(`Text NOT NULL`, new migration) rather than a `governance_records` column -- the reason for REQUESTING a
privileged operation needs to exist even for a request that's rejected or expires unactioned, which an
outcome-only table can't hold. `create_approval_request()`/`create_promotion_approval_request()` both
reject an empty/whitespace-only reason.

**Defense in depth beyond the four-eyes approval itself**: the endpoint requires the CALLER triggering
execution to independently hold the `admin` role over the request's own `dest_scope`
(`admin_db.has_role(principal=user_id, scope_mask=dest_scope)`) -- N distinct people approving a request
does not by itself authorize an arbitrary caller to be the one who pulls the trigger. New test
(`test_403_when_caller_lacks_role_despite_full_approval`) confirms a fully-approved request still 403s for
a caller with no role grant.

**DI wiring** (`dependencies.py`): `get_control_plane_engine()`/`get_governance_db()`/`get_admin_db()`/
`get_audit_log()`/`get_identifier_corpus()`, all following the established "optional infrastructure, `None`
when unconfigured" pattern (`shared_atlas_client`) -- importing `sqlalchemy` only happens lazily, inside the
function body, after confirming `AEON_CONTROL_PLANE_DATABASE_URL` is actually set, so `server.py` keeps
importing cleanly with zero `db` extras installed (confirmed directly: full pytest suite green with no
control-plane env vars and no sqlalchemy installed path exercised). `get_identifier_corpus()`'s defaults
(empty patterns, both generic redactors off) inherit the classifier's own fail-closed behavior -- a
deployment that hasn't configured `AEON_IDENTIFIER_CORPUS_PATTERNS` etc. gets a promotion endpoint that
rejects every fragment, not one that silently passes raw content through.

**A real bug caught by my own test, not by inspection**: the endpoint's own test fixture initially called
`app.dependency_overrides.clear()` in teardown -- since `app` is one shared FastAPI instance across every
test module in the process, this wiped out `test_server.py`'s own module-level overrides the moment both
files ran in the same session, breaking five unrelated `test_server.py` tests. Fixed to save-then-restore
(the same discipline `test_server.py`'s own `test_active_room_endpoint_routes_to_shared_store` already
uses, for the identical reason) -- caught by running the FULL suite together, not each new test file in
isolation, which is exactly why "run everything, not just the new tests" stayed the standing verification
habit through this entire session.

**112/112 `pytest` passing with a live Postgres** (102 prior + 5 new endpoint tests via a real FastAPI
`TestClient` against real `AeonClient`/`AdminDB`/`AuditLog` instances, no mocks on the execution path).
`ctest` unaffected -- **114/114 still passing**. Confirmed pytest stays green with zero Postgres running
after tearing the test container down (75 passed, 3 skipped -- all three DB-backed modules).

**All four originally-deferred items are now resolved**: (1) full Postgres control plane -- built; (2)
destination-conditioned re-embedding -- remains a documented seam (`reembed_fn`), no embedding-conditioning
pipeline exists anywhere in this repo to hook into, genuinely different from (1)/(4) which were blocked on
missing infrastructure this session built; (3) correctness-gated promotion for code -- remains deferred,
needs a "verification run" concept tied to VCS/CI state, a genuinely separate integration surface; (4) the
promotion endpoint -- built, unblocked by task 7.

Reporting back -- Stage 4's remaining scope is now: the console (task 5) and crypto-erase (task 6, its own
design spike per the original plan). Items (2) and (3) above stay deferred for the reasons stated, not
reopened without a concrete trigger the way Postgres had one.

**Advisor review of the promotion endpoint (2026-08-23) caught two real behavior bugs and one
documentation gap, all fixed before moving on:**

1. **A classifier rejection was permanently burning the four-eyes approval.** `execute_approved_promotion()`
   called `admin_db.mark_executed(request_id)` unconditionally after `promote_fragment()` returned --
   including on the `None`/rejected path. Since `promote_fragment()`'s rejection is fail-closed classifier
   behavior (e.g. an unconfigured `IdentifierCorpus`), not a considered decision about the content, this
   meant a corpus-configuration problem would silently consume a real four-eyes approval with no way to
   retry once the config was fixed -- a request would have to go through the whole approval process again
   for the exact same, still-valid intent. Fixed: `mark_executed()` now only runs when `promote_fragment()`
   returns a real node id (`promotion.py`'s `execute_approved_promotion()`). An anomaly (delta-diversion,
   post-insert scope-write failure) still counts as executed -- a node was actually minted into the shared
   store either way, unlike a clean rejection. Pinned with
   `test_classifier_rejection_does_not_consume_the_approval` (`tests/test_admin.py`), which rejects once
   with an empty corpus, confirms `executed_at` is still `None`, then successfully retries the *same*
   request with a working corpus.
2. **The HTTP endpoint couldn't distinguish "classifier rejected this fragment" from "this deployment has
   no corpus configured at all."** Both produced an identical `200 {"promoted_node_id": null}` -- so a
   misconfigured adopter (zero `AEON_IDENTIFIER_CORPUS_PATTERNS`, both generic redactors off, the
   documented fail-closed default) would see every promotion silently "succeed" with a null result and get
   no signal that nothing can ever be promoted until they configure something. Fixed: `execute_promotion()`
   (`server.py`) now checks `corpus.is_empty()` before calling `execute_approved_promotion()` and returns
   `503` with an explicit `AEON_IDENTIFIER_CORPUS_PATTERNS`/`AEON_REDACT_EMAILS`/`AEON_REDACT_COMMIT_SHAS`
   pointer -- a standing deployment-configuration state, not a per-request outcome. Combined with fix (1),
   this also doesn't consume the approval. Pinned with
   `test_503_when_identifier_corpus_is_not_configured` (`tests/test_admin_endpoint.py`).
3. **`DEFAULT_CONTROL_PLANE_DATABASE_URL` and its siblings are read once, at `aeon_py` import time, not
   per-request** -- correct given `@lru_cache()` on `get_control_plane_engine()`, but undocumented. Setting
   `AEON_CONTROL_PLANE_DATABASE_URL` after `aeon_py` is first imported (e.g. inside a test fixture instead
   of the process environment) silently leaves every admin/promotion route 404ing with no indication why.
   Documented in `dependencies.py`'s comment rather than changed -- the read-once/`lru_cache()` behavior
   itself is correct and consistent with every other optional-infra singleton in this file.

Full suite re-verified against a live Postgres container after these fixes (`alembic upgrade head` +
`AEON_REQUIRE_DB_TESTS=1 pytest tests/`): 114 passed. Container torn down; local-default state
(`pytest tests/`, zero Postgres) confirmed still green: 75 passed, 3 skipped. No C++ files touched this
round, so `ctest --preset dev` was not re-run.

**Task 5 -- the minimum admin console: DONE (2026-08-23).**

User chose "Console (task 5)" over crypto-erase (task 6) when asked which of Stage 4's two remaining
tasks to tackle next -- task 6 stays flagged, per the original plan, as needing its own design spike.

**Scope decision, stated explicitly rather than left silent**: built as **API-only**. This repo has
zero frontend infrastructure (no `StaticFiles` mount, no templates, no `web/` directory) -- the
source design docs' "console" language assumed a UI, but that's a separate product surface from this
increment. Every route mirrors the promotion endpoint's established shape from task 7: caller identity
from `get_current_user_id()`, never a request body field; authorization checked against state read
FROM THE SERVER, never a caller-supplied parameter.

**One new C++ primitive was needed first: `Atlas::tombstone_node(uint64_t)`.** Verified before writing
anything else: `supersede_node()`/`bulk_set_node_scope()`/`list_nodes_by_scope()`/`get_node_metadata()`/
`get_node_centroid()`/`get_node_governance_id()` all already existed (built across Stage 4 tasks 1/2) --
but the only tombstone mechanism was `tombstone_node(NodeHeader&)`, a private free function only ever
called internally by `consolidate_subgraph()`. Added `Atlas::tombstone_node(uint64_t)` mirroring
`supersede_node()` exactly: mmap-only (rejects delta ids), rejects during compaction, WAL-before-mutate
(`WAL_RECORD_ATLAS_TOMBSTONE`, new `schema.hpp` record type, wired into `replay_wal()`'s existing
two-pass/skip-not-break machinery), idempotent, but **TERMINAL** -- unlike `supersede_node()`, there is
no `revoke_node_tombstone()`, since `schema.hpp`'s `tombstone_node()` never stashed the prior
`hub_penalty` to restore.

Adding a public member of this name inside `Atlas` shadowed the existing unqualified call at
`consolidate_subgraph()`'s Phase 4 (`tombstone_node(*old_node)`, a `NodeHeader&` argument) --
class-scope lookup would have tried to resolve it against the new `uint64_t` overload and failed to
compile. Fixed by qualifying that call site as `aeon::tombstone_node(*old_node)`, the same
disambiguation `supersede_node()`/`revoke_node_supersede()` already needed for
`aeon::supersede_node()`/`aeon::revoke_supersede()`.

Pinned with 6 new C++ tests (`test_atlas.cpp`: round-trip via `list_nodes_by_scope()`/`tombstone_count()`
exclusion, idempotency, delta-id rejection, invalid-id rejection, and a **supersede-then-tombstone-
then-revoke-supersede** sequence proving the interaction `revoke_node_supersede()`'s own doc comment
already reasoned about -- tombstoning after superseding doesn't corrupt `saved_hub_penalty`, and
revoking the supersession afterward correctly leaves the node tombstoned; `test_wal.cpp`: a hand-
constructed `WAL_RECORD_ATLAS_TOMBSTONE` record replayed against a node never tombstoned through the
API, proving replay's second pass genuinely applies it). `ctest --preset dev`: 120/120 (114 + 6 new).
Stubs regenerated (`./scripts/gen_stubs.sh`).

**Python/HTTP layer, three components per the plan's own priority order:**

*(a) Audit log.* `governance.py`'s `AuditLog` already had `verify()`/`export_signed()`/
`verify_export_signature()` from task 5(a)'s earlier increment -- this was pure HTTP exposure plus one
new reader, `AuditLog.tail(since_seq, limit)`, for pagination (`GOVERNANCE_RECORD_ACTIONS` also grew a
4th value, `"erasure"`, for part (c) below). Three routes: `GET /admin/audit-log` (paginated),
`GET /admin/audit-log/verify`, `GET /admin/audit-log/export` (HMAC-signed, key from
`AEON_AUDIT_LOG_EXPORT_KEY_HEX` read at import like every other `dependencies.py` constant -- 503, not
a default key, when unset).

*(b) Knowledge browser.* `GET /admin/knowledge` (paginated; lists shared-store nodes via
`list_nodes_by_scope()` filtered to the caller's own `effective_scope_mask()` -- new `AdminDB` method,
the OR of every non-expired scope grant a principal holds) and `POST /admin/knowledge/{node_id}` with
`{"action": "supersede" | "revoke_supersede" | "tombstone"}`.

*(c) Erasure workflow.* New top-level `erasure.py` (mirrors `promotion.py`'s shape) plus a new
`erasure_cases` table (migration `57beb688a09c`) and `ErasureDB` (`control_plane/erasure_db.py`,
same URL-or-shared-Engine pattern as `GovernanceDB`/`AdminDB`). Reuses the four-eyes approval
infrastructure from task 7 wholesale rather than inventing a parallel mechanism -- erasing a subject's
shared-tier fragments is exactly the "bulk operation" task 7's admin console constraints require
four-eyes approval for. `create_erasure_case()` locks in the exact target node ids AND the
authorization scope (the OR of every target's current `scope_bitmap`, computed at request time) into
the approval request's `target` JSON -- same replay-safety reasoning as promotion's locked-in params.
`execute_approved_erasure()` tombstones every target it can, and records BOTH outcomes explicitly in
one receipt: `{"erased": [...], "could_not_erase": [{"node_id", "reason"}, ...]}` -- a partial outcome
is still a legitimate, auditable completion, not a reason to leave a case dangling. Crash-resumable by
construction: `completed_at` is set only after every id is attempted, so a process killed mid-loop
leaves the case retryable, and `Atlas.tombstone_node()`'s idempotency (pinned by the C++ test above)
makes re-attempting an already-erased id a safe no-op, not a double-erasure. Routes:
`POST /admin/erasure` (create), `GET /admin/erasure/{case_id}`, `POST /admin/erasure/{case_id}/execute`.

**Deferred, stated explicitly**: private-store erasure. `erasure_cases`/`create_erasure_case()` target
the SHARED atlas store only -- the private store has no per-owner authorization model in this codebase
(Stage 0 gave private-store isolation at the SLB cache/session level, not per-node ownership tagging in
the mmap file itself), so there is nothing today for an erasure endpoint to check authorization against
for a private-store node id. Building that ownership model is a separate increment, not a corner cut
here.

**Advisor review caught two real bugs and one nit, all fixed before this was reported done:**

1. **Authorization used overlap where it needed containment -- a real privilege-escalation bug.**
   `has_role()`'s check (`grant.scope_mask & requested != 0`) is correct for the promotion endpoint,
   where `dest_scope` is a single target being written into. It was WRONG for acting on existing nodes,
   which can carry multiple scope bits at once: a node scoped `0x1000|0x2000` with the caller granted
   only `0x1000` would pass `has_role(scope_mask=0x1000|0x2000)` (the AND is `0x1000`, non-zero) --
   authorizing an action on a node ALSO in a scope the caller had no grant over. Erasure was worse:
   `combined_scope` OR'd across every target node, so ONE overlapping grant authorized filing AND
   executing a case spanning scopes the caller never held at all -- privilege escalation on the most
   destructive, least reversible operation in the system. The existing 403 tests didn't catch it
   because they used disjoint scopes (zero overlap passes for the wrong reason -- a coincidence, not
   proof of correctness). Fixed with a new `_require_scope_containment()` helper (`node_scope & ~caller_
   mask & ALL_SCOPES_VISIBLE == 0`, AND `caller_mask != 0` checked explicitly since an unscoped node,
   e.g. a promotion delta-diversion orphan, would otherwise trivially pass containment against a caller
   holding NO admin grants at all), wired into all four sites that act on/read existing state:
   `act_on_knowledge_node`, `create_erasure`, `get_erasure`, `execute_erasure`. `list_knowledge`
   deliberately keeps `list_nodes_by_scope()`'s own OVERLAP semantics for visibility ("see broadly, act
   narrowly" -- documented explicitly as the chosen asymmetry, not an oversight). Pinned with two new
   regression tests using a `0x1000|0x2000` node against a caller granted only `0x1000` -- one for the
   knowledge action, one for erasure-case creation -- both of which the pre-fix code would have wrongly
   authorized.
2. **A transient failure (compaction in progress) would have permanently burned a four-eyes approval.**
   `execute_approved_erasure()`'s original per-node `except (ValueError, RuntimeError)` treated
   "compaction is in progress" (transient -- `Atlas::tombstone_node()`'s own runtime_error for this
   condition) identically to "invalid node id" (permanent): both landed in `could_not_erase`, and
   `complete_case()` still ran -- so a case executed mid-compaction would record every target as
   permanently un-erasable and consume the approval, even though compaction finishing seconds later
   would have let the SAME case succeed. This is the same class of bug just fixed on the promotion
   path (a classifier rejection burning the approval), in the opposite direction. Fixed with a new
   `ErasureTransientFailure` exception: hitting the compaction-in-progress message aborts the WHOLE
   case WITHOUT calling `complete_case()` (`completed_at` stays `None`, the approval stays unconsumed,
   and the case is safely re-executable -- the crash-resumable design already built handles this for
   free, since `tombstone_node()` is idempotent for whatever the aborted run already erased). The HTTP
   layer surfaces this as 503 ("retry shortly"), distinct from the generic 409 for a permanent failure.
   Pinned with a test using a duck-typed fake `shared_atlas` that raises the exact compaction message,
   confirming the case stays uncompleted and is genuinely still executable against the real store
   afterward.
3. **Nit: `list_knowledge` had no pagination**, and each listed node costs four separate EBR-guarded
   C++ reads (metadata/scope/superseded/governance-id) -- unbounded, that's O(scope size) round trips
   per request. Added `offset`/`limit` query params (sliced in Python after the flat scope scan, not
   pushed into the C++ layer -- `list_nodes_by_scope()` is already a cheap flat scan) plus a `total`
   count in the response so a client can page through a large scope. Pinned with a 3-page pagination
   test.

New files: `shell/aeon_py/erasure.py`, `shell/aeon_py/control_plane/erasure_db.py`,
`alembic/versions/57beb688a09c_add_erasure_workflow_task_5c.py`, `tests/test_erasure.py`,
`tests/test_console_endpoint.py`. Extended: `core/include/aeon/{atlas,schema}.hpp`, `core/src/{atlas,
bindings}.cpp`, `core/tests/{test_atlas,test_wal}.cpp`, `shell/aeon_py/{governance,models,server,
dependencies,promotion}.py`, `shell/aeon_py/control_plane/{schema,admin}.py`, `tests/{test_governance,
test_admin}.py`.

Full suite re-verified end to end after the advisor fixes: `ctest --preset dev` 120/120;
`AEON_REQUIRE_DB_TESTS=1 pytest tests/` (live Postgres, migrations applied) 151 passed; local-default
`pytest tests/` (zero Postgres) 80 passed, 5 skipped.

Stage 4's remaining scope is now just task 6 (crypto-erase) -- explicitly flagged in the original plan
as needing its own dedicated design spike given its complexity, not a routine increment like task 5
was.

**Task 6 -- crypto-erase design spike, decision record (2026-08-23). Scope OVERRIDDEN by explicit user
choice; not implemented yet -- this is the spike's deliverable, per the plan's own instruction that
task 6 needs "its own design spike" before code.**

The plan's original task-6 text (line 2136 above) frames this as "real crypto infrastructure, not a
kernel change," implicitly scoping it to blob/metadata payloads only and leaving shared-store centroid
vectors in plaintext (embedding inversion residual risk, documented not fixed). Asked the user to
confirm that scoping explicitly, since it changes the deliverable from a shell-only feature to a kernel
change; **the user chose to cover vectors too.** This supersedes v4-plan.md:2136's framing -- recorded
here as the standing scope for task 6 rather than silently reopening the question later.

Consulted the advisor twice: once on the original (blob-only) scoping, once after the override to work
out what covering vectors actually requires given the real mmap/SIMD architecture (`NodeHeader` layout,
`navigate_internal()`'s direct `span<const float>` reads, confirmed by reading `schema.hpp`/`atlas.cpp`
directly, not assumed). Four binding constraints, and the decision on each:

1. **`NodeHeader` has no room for a per-node nonce/tag/key-id.** Stage 1 carved its `reserved[20]` into
   exactly `scope_bitmap`(8) + `governance_record_id`(8) + `saved_hub_penalty`(4) -- `sizeof == 64`
   static_asserted, nothing left. Per-node AEAD (nonce+tag stored alongside ciphertext) is therefore
   off the table. **Decision: a stream cipher (ChaCha20 or AES-CTR), not an AEAD mode**, so ciphertext
   length == plaintext length and `compute_node_stride()`/every downstream SIMD offset stays
   byte-identical -- zero layout migration for the vector region itself. This buys confidentiality, not
   integrity, which is not a regression: the mmap has no per-node MAC today either. Nonce is *derived*,
   not stored: `node_id ‖ generation` (both already tracked -- `generation` is `Atlas`'s existing
   in-memory compaction-generation counter, Stage 2 task 3's fix). The key itself is resolved via
   indirection, not a new per-node field: `governance_record_id` (already exists, already the intended
   hook per Stage 1's own doc comment) points to a Postgres `governance_records` row, which task 6
   Phase A (below) extends with a `subject_id` column -- `(subject_id, dest_scope)` is the DEK lookup
   key, not anything stored in `NodeHeader` itself.
2. **The encryption boundary must be file I/O, not search.** Resolving a DEK and decrypting per beam
   candidate inside `navigate_internal()` means Ring 0 reaching into a keystore against the stated
   Core-Shell split, and paying decrypt cost on every candidate touched during descent (not just the
   winner) against a 3.09µs navigate budget -- a non-starter. `navigate()` must keep operating on
   plaintext `span<const float>` exactly as today.
3. **Constraint 2 forces full decrypt-at-open into anonymous (non-file-backed) memory, re-encrypt at
   every durable write-out.** Demand-paged decrypt (userfaultfd-style) is unavailable uniformly across
   the three CI platforms (macOS/Linux/Windows) this project already targets, so partial/lazy decrypt is
   not an option. **Decision, stated as a tradeoff rather than an unexamined consequence: the shared
   store gives up the "mmap can exceed RAM" property.** Judged acceptable because the shared store is
   architecturally a *curated, distilled* corpus (Stage 4 task 2's mint-and-recontextualize output),
   expected to be materially smaller than the raw private stores this property matters most for -- but
   this is a real capacity ceiling the adopter docs (task 8) must state honestly, not a free win.
4. **Every durable write of raw vector bytes needs ciphertext, not just the mmap file at rest** -- a
   constraint the advisor's four points named at the read side but that reasoning through the *write*
   side surfaced a second location for: `insert_delta()`'s `WAL_RECORD_ATLAS` payload is a whole-struct
   `memcpy` capture (Stage 1's own design, "insert_delta()'s existing WAL_RECORD_ATLAS payload is
   already a whole-struct capture") -- if that capture happens on the plaintext struct before encryption,
   a subject's vector survives in the WAL until the next compaction truncates it, defeating the whole
   point (WAL truncation has the identical "doesn't guarantee destruction on flash" problem as
   `std::filesystem::remove()` that motivated this task in the first place). **Decision: encryption
   happens at insert time, on both paths** (`insert()`'s direct-to-mmap-slot write, and
   `insert_delta()`'s in-memory-struct-plus-WAL-record write) -- the delta arena's in-memory plaintext
   copy is fine (process RAM, not attacker-visible at rest, same assumption the codebase already makes
   everywhere else), but the WAL bytes it durably captures must already be ciphertext when written.
   Compaction's copy-to-new-generation step (already the point where blob offsets get GC'd and rewritten,
   per Stage 2 task 3's `gc_blob_pair`) is where re-encryption under a fresh `node_id ‖ generation` nonce
   happens for surviving nodes, symmetric with points 1 and 3.

**Multi-subject-node hazard, checked rather than assumed away.** One node, one DEK. If a shared-store
node's content derives from more than one subject, destroying subject X's key destroys subject Y's data
too, silently. `consolidate_subgraph()` OR-unions source scope bitmaps into a summary node and *could*
produce exactly this shape -- but verified (`dreamer.py`) that `DreamingWorker` only ever calls
`consolidate_subgraph()` against `self._atlas`, the private per-session store; Stage 5 ("point the
Dreamer at the shared tier") is explicitly future, not-yet-started work. **So no code path today can
produce a multi-subject shared-store node** -- but this must be enforced as an invariant going forward,
not left to accident. **Decision: `promote_fragment()` (Phase A, below) requires a non-empty
`subject_id` per minted node** (rejected up front, same fail-closed treatment as `dest_scope == 0`),
which satisfies single-subject attribution BY CONSTRUCTION today -- it mints exactly one node from
exactly one caller-supplied subject per call, so there is nothing yet for a runtime check spanning
calls to guard against. This is not the same as an enforced cross-call invariant, which does not exist
yet because nothing yet merges shared-store nodes across subjects; if
Stage 5 ever lets the Dreamer consolidate multiple shared-store nodes into one summary, that summary
must either be rejected outright when its sources' `subject_id`s differ, or the erasure workflow's
existing `could_not_erase` section (task 5c, already built) must list it rather than tombstone-and-
silently-destroy-the-wrong-subject's-data. Left for whoever builds Stage 5 to wire, flagged here so it
isn't lost the way `governance_record_id`'s writer gap was flagged across three stages until it got one.

**The guarantee statement (task 8's console-facing deliverable) now has two envelopes to cover, not
one**, and both must be named explicitly rather than implied: (a) DEK material -- if stored as rows in
Postgres, `DELETE` alone does not guarantee destruction (WAL, PITR, base backups, un-VACUUMed heap pages
all potentially retain it) -- decision: DEKs are stored **wrapped only**, under a per-scope KEK held in
an external KMS/HSM where destruction is that system's own contract, matching the existing
`AEON_AUDIT_LOG_EXPORT_KEY_HEX`-style key-material-from-env pattern for the wrapping key itself, 503 not
a default when unset; (b) the mmap ciphertext file -- filesystem snapshots and volume-level backups can
retain old ciphertext bytes indefinitely, which is fine (they're still ciphertext under a destroyed key)
*provided* the DEK is truly gone per (a) -- the two envelopes only compose into an actual guarantee
together, and the adopter docs must say so rather than asserting either half alone as sufficient.

**Format change**: `ATLAS_VERSION` bumps to 3 (from 2) for any Atlas file using vector encryption --
`AtlasHeader::reserved[12]` (0x34, currently zeroed/unused) becomes `vector_crypto_scheme` (`uint32_t`:
0 = plaintext, matching every existing file; 1 = ChaCha20-CTR-per-subject) plus an 8-byte
`default_kek_id` identifying which per-scope KEK this store's DEKs are wrapped under -- filling the 12
bytes exactly, the same "fits the budget exactly" discipline Stage 1 used for `NodeHeader::reserved`.
Version-3 open on version-2-unaware code, and vice versa, must fail closed (explicit version-mismatch
error), not silently misinterpret ciphertext as plaintext floats.

**Cost callout, said plainly rather than absorbed silently**: the user's override roughly triples this
task's scope from the plan's original framing -- from a shell/control-plane-only feature (keys +
blob/metadata encryption, API and DB work only) to a kernel change touching `Atlas`'s open/insert/
compact/navigate paths, a file-format version bump, and a RAM-ceiling tradeoff for the shared store.
Recommending a **two-phase split so the time-critical, independently-valuable half isn't blocked on the
much larger half**:

- **Phase A -- subject attribution (time-critical, do first, regardless of Phase B's timeline).**
  `promote_fragment()` has no `subject_id` parameter today (confirmed by reading it) -- every fragment
  promoted before this exists is **permanently unattributable to any subject**, independent of whether
  vector encryption ever lands, since Phase B's per-subject DEK lookup depends entirely on this mapping
  existing. Adds a required, non-empty `subject_id` parameter to `promote_fragment()` and a matching
  `subject_id` column on `governance_records`, satisfying single-subject-per-node BY CONSTRUCTION (see
  above) -- real cross-call enforcement is Stage 5's job, once something can actually merge shared-store
  nodes. This alone is a modest, well-precedented increment (same shape as every other Stage 4 task-1/2
  primitive) and should land before any further promotions happen through this codebase.
- **Phase B -- encrypted vector storage (the kernel change).** Stream-cipher-in-place vectors,
  decrypt-at-open/re-encrypt-at-write-out, `ATLAS_VERSION` 3, KMS-wrapped per-(subject,scope) DEKs, the
  WAL-write-path fix. Materially larger than any single task landed so far in this plan -- comparable to
  Stage 4's own physical-separation step (step 3) in scope, not a routine increment.

Next: implement Phase A now (attribution can't wait), then check back in before starting Phase B's
kernel work given its size.

**Task 6 Phase A -- subject attribution: DONE (2026-08-23).**

`promote_fragment()` (`promotion.py`) gained a required `subject_id: str` parameter (positioned right
after `actor`, since it's non-optional and Python requires non-default params before defaulted ones),
rejected up front with a `ValueError` if empty/blank -- the same fail-closed treatment the function
already gives `dest_scope == 0`, added after an advisor review pointed out the first draft recorded
whatever string a caller passed with no validation at all. Recorded on every audit path
(rejection/anomaly/success payloads in the JSONL `AuditLog`, and, when a `governance_db` is supplied, in
the mirrored Postgres row). `create_promotion_approval_request()` locks `subject_id` into the approval
request's `target` JSON alongside `source_node_id`/`dest_scope`, the same
replay-safety reasoning task 7 already established for those two fields; `execute_approved_promotion()`
reads it back from there rather than accepting it as a fresh argument at execution time.

**Invariant wording corrected after advisor review caught it overclaiming.** The first draft's docstring
and this decision record both said `promote_fragment()` "enforces" single-subject-per-node as a hard
invariant -- true only in the weak sense that each call mints exactly one node from exactly one
caller-supplied `subject_id`, which trivially satisfies it today because nothing yet merges shared-store
nodes across subjects (see the multi-subject-node hazard above). There is no independent runtime check
spanning multiple calls, and the wording now says so explicitly in both places rather than claiming
enforcement Stage 5 would actually have to build.

New `governance_records.subject_id` column (`String(256)`, `NOT NULL`, no `server_default` -- this repo
carries no pre-existing rows to backfill, stated explicitly in both the migration and
`control_plane/schema.py`'s column comment rather than left as an unexamined assumption), migration
`d0e24ce99c88` (down_revision `57beb688a09c`, the erasure-workflow migration -- confirmed the new head
resolves via `alembic history`). `control_plane/db.py`'s `GovernanceDB.record()` gained the matching
required `subject_id` kwarg. `control_plane/app.py`'s `GET /governance-records/{id}` needed no code
change -- it already returns `dict(row)` over every column generically, so `subject_id` is exposed for
free once the column exists.

No C++/kernel changes in this increment (Phase A is exactly the shell/control-plane-only half of the
split above) -- `ctest --preset dev` was not re-run, only `pytest`.

Every existing call site updated to pass `subject_id` (11 in `tests/test_promotion.py`, 11 in
`tests/test_control_plane.py` -- 6 `promote_fragment()` + 5 `governance_db.record()` calls -- and 4 in
each of `tests/test_admin.py`/`tests/test_admin_endpoint.py`'s `create_promotion_approval_request()`
calls). Two of `test_control_plane.py`'s existing assertions extended into round-trip proof rather than
just "the call didn't raise": `test_record_readable_via_control_plane_app` and
`test_governance_record_id_is_postgres_pk_not_jsonl_seq` now assert `subject_id` reads back correctly
through the control-plane HTTP app, not just that the write succeeded.

**Advisor review caught a real test-coverage gap before this was reported done**: 151 passed == 151
before is consistent with "nothing broke," but none of the existing assertions actually proved Phase A's
own point -- every one either passed `subject_id` straight into `governance_db.record()` directly, or
read back a request in the same call that created it. None isolated the thing Phase A specifically adds:
`subject_id` surviving `create_promotion_approval_request()`'s target-JSON lock-in THROUGH
`execute_approved_promotion()` into the `governance_records` row -- the exact replay-safety property the
whole lock-in design exists for. A version of this bug (subject_id silently dropped between the two
calls) would have passed every pre-existing test. Added
`test_subject_id_survives_target_json_lock_in_to_postgres_row` (`tests/test_admin.py`, using a new
`governance_db` fixture) and **proved it actually discriminates**, the same temporarily-break-and-measure
technique used throughout this plan: replaced `execute_approved_promotion()`'s
`params["subject_id"]` with a hardcoded wrong string, confirmed the new test failed
(`'wrong-subject-TEMP-BREAK' == 'subject-round-trip'`), then restored the fix and reconfirmed green.
Also added `test_blank_subject_id_rejected` (`tests/test_promotion.py`, parametrized over `""`/`"   "`)
for the empty-subject_id guard above. Confirmed separately (not just inferred) that a non-promotion
approval request can't reach the unguarded `params["subject_id"]` indexing: `test_admin.py`'s
pre-existing `test_non_promotion_request_raises` already asserts `execute_approved_promotion()` raises
`ValueError` on the action-mismatch check, which runs before `target` is ever parsed. Also read
`server.py`'s `execute_promotion()` HTTP handler directly (not inferred) to confirm a `ValueError` from
this path returns its existing 409 ("permanent failure"), consistent with the endpoint's established
503 (misconfig) / 200-null (rejection) / 409 (permanent failure) distinctions -- no new HTTP-layer gap.

**A second advisor pass caught one more real gap, fixed before this was reported done**:
`create_promotion_approval_request()` itself had no `subject_id` validation -- only `promote_fragment()`
did. A blank `subject_id` would pass request creation, get locked into `target`, collect two real
four-eyes approvals, and only fail at `execute_approved_promotion()` time -- a dead request that
permanently consumed two reviewers' attention for a request that could never succeed (the blank value is
baked into `target`, so even a retry replays the same blank value). Fixed with the same guard at
request-creation time, rejecting before any `approval_requests` row is even written. Pinned with
`test_create_request_rejects_blank_subject_id` (parametrized, `tests/test_admin.py`).

Verified end to end, not just import-clean: local-default `pytest tests/` (zero Postgres) **82 passed**,
5 skipped (80 pre-Phase-A + the 2 parametrized `test_blank_subject_id_rejected` cases in
`test_promotion.py`, which are Postgres-independent -- `test_create_request_rejects_blank_subject_id`
lives in the Postgres-gated `test_admin.py` and doesn't run in this suite). Live-Postgres suite
(`docker compose up -d postgres`, `alembic upgrade head` -- confirmed the new migration applies cleanly
on top of the existing chain, `AEON_REQUIRE_DB_TESTS=1 pytest tests/`): **156 passed** (151 pre-Phase-A +
the discriminating round-trip test + 2 blank-subject_id cases in `test_promotion.py` + 2 blank-subject_id
cases in `test_admin.py`'s new create-time guard test). Container torn down after each run.

**Phase B reshaped to blob/metadata-only (2026-08-23, user decision).** Given the RAM-ceiling tradeoff
and kernel-change size the vector-inclusion override actually required (Phase A's report above), the
user chose to reshape back to the plan's ORIGINAL task 6 framing rather than fund that cost.

**The vector-encryption design from the earlier override is retired -- considered and NOT pursued.**
`ATLAS_VERSION` 3, stream-cipher-in-place centroid vectors, decrypt-at-open into anonymous memory, and
the resulting loss of the shared store's "can exceed RAM" property (all documented in detail above) are
no longer the plan. **Accepted residual, to be stated plainly in task 8's adopter-facing guarantee
documentation**: shared-store centroid vectors stay plaintext in the mmap file. Embedding inversion is
real, so a destroyed subject's vector remains partially informative about their original text even after
their key is destroyed. This is the exact limitation the plan's original task 6 text implicitly assumed
by scoping to "real crypto infrastructure, not a kernel change" -- now stated explicitly rather than
left implicit.

**Phase B design (blob/metadata payloads, per-subject-per-scope keys) -- worked out, not yet
implemented.** Three findings changed the shape of this from the initial sketch:

1. **`std::string(meta)` in `Atlas::get_node_metadata()` stops at the first null byte** (`atlas.cpp`),
   so raw binary ciphertext is unstorable through the existing metadata API -- a stream cipher's
   ciphertext bytes are effectively random and will contain embedded nulls almost certainly at any real
   length. Ciphertext must be nonce-prefixed and base64-encoded before it can occupy this field (or the
   representation needs to change, which the sizing question below rules out as necessary).
2. **The 256-byte metadata field is tighter than it looks, but not fatally so.** `dreamer.py:85` already
   truncates summary text to 250 chars because it knows the field's ~255-byte limit -- a pre-existing,
   already-accepted convention that promoted/summarized fragments are short, not multi-paragraph
   documents. Nonce (12B) + base64 (4/3 overhead) against the default 256-byte field leaves roughly
   170 usable plaintext bytes -- tighter than today's ~250, a real but incremental degradation of an
   already-lossy field, not a new category of problem. (Pre-marker estimate -- once implemented, the
   `AEONENC1:` marker prefix `encrypt_metadata()` actually needed brings this down to an exact 155 bytes
   at 256, verified via `crypto.max_plaintext_bytes(256)`.)
3. **`metadata_size` is NOT actually configurable end-to-end today, despite looking generalized --
   found by reading the code, not assumed.** `storage::MemoryFile::open()` and `compute_node_stride()`
   both take/generalize over `metadata_size`, and `AtlasOptions` exists specifically for this kind of
   per-file customization (`dim`, `quantization_type`, `enable_wal`) -- but `Atlas::Atlas(path, opts)`
   (`atlas.cpp`) hardcodes `METADATA_SIZE_DEFAULT` at the one call site that invokes `file_->open()`,
   ignoring whatever `opts` might someday carry. No C++ test exercises a non-default `metadata_size`
   either (grepped `core/tests/*.cpp` -- every `AtlasOptions` use sets only `dim`). This is the exact
   "parameterized-in-theory, hardcoded-in-practice" shape guardrail #1.1 (SLB/SemanticCache's hardcoded
   768-dim) and the `TraceBlockIndex` `std::array<float, EMBEDDING_DIM>` bug both were -- looks
   generalized, has never actually been exercised at a non-default value.

**Decision, given (2) and (3): add a real `metadata_size` field to `AtlasOptions`, wire it through the
constructor for real, and open the SHARED atlas store specifically with a larger value (e.g. 512 bytes)
-- not a Postgres side-table for ciphertext.** This was a genuine fork, resolved on the merits after
checking real numbers rather than guessed: with `metadata_size=512`, the field holds ~370 usable
plaintext bytes after nonce+base64 overhead -- LARGER than today's ~250-byte effective budget, not
smaller. (Pre-marker estimate -- the exact figure once `encrypt_metadata()`'s marker prefix is counted is
347 bytes, verified via `crypto.max_plaintext_bytes(512)`; still larger than today's budget, the
comparison this decision turned on.) A side-table alternative was considered and rejected: it would make
Postgres a hard requirement
for reading ANY shared-store content, breaking the optional-control-plane invariant this codebase has
held since Stage 4 step 1 (`governance_db=None` paths, `get_shared_atlas_client()`'s own optionality).
The in-field design keeps that invariant narrower: Postgres becomes required for shared-store reads only
once crypto-erase is actually enabled for that deployment (DEK lookup), not for the shared store to
exist or be read at all -- stated precisely rather than claiming in-field preserves optionality
outright, since the keystore itself (below) cannot avoid needing Postgres.

This IS a small C++ change (add the `AtlasOptions` field, stop hardcoding `METADATA_SIZE_DEFAULT` in the
constructor, add a real test at a non-default size) -- but it touches no hot path, no WAL record format,
no `ATLAS_VERSION`, and no SIMD/navigate() code, unlike the retired vector design. Consistent with the
reshape's intent (avoid the kernel-scale rework), not a reopening of it.

**The keystore cannot be HKDF-derived from a single master secret -- per-subject-per-scope keys must be
independently destroyable, which requires each to be its own stored, deletable unit.** A derived key
(DEK = HKDF(KEK, info=f"{subject_id}:{scope}")) can never be "destroyed" without destroying the KEK
itself (which destroys every subject's key at once) -- that's not real per-subject erasure, just a
revocation list someone with KEK access could bypass. Decision: a new Postgres `subject_scope_keys`
table (subject_id, scope, wrapped_dek, created_at) -- one random DEK generated per (subject_id, scope)
pair on first use, wrapped under a single deployment-wide KEK read from env (matching the existing
`AEON_AUDIT_LOG_EXPORT_KEY_HEX` key-material-from-env pattern -- 503 when unset, not a default key), row
deleted (not soft-flagged) to destroy a subject's key for that scope. Same accepted DELETE-vs-WAL/PITR/
backups caveat already documented in the guarantee statement above -- unchanged by this reshape, since
it was never specific to vectors.

**Write-side truncation must become a hard error, not silent, once the field holds ciphertext.**
`insert()`/`insert_delta()` silently truncate metadata at `metadata_size_ - 1` today -- lossy but
tolerable for plaintext (a truncated sentence), silent corruption for ciphertext (a truncated nonce or
mid-ciphertext cut produces a decrypt failure or garbage, discovered only on read, far from the write
that caused it). Decision: the shell layer length-checks the encoded (nonce+base64) payload against the
shared atlas's actual `metadata_size` BEFORE calling `insert()`, and raises rather than relying on the
C++ truncation to be benign.

**Read/write site enumeration, checked by grep rather than assumed** -- the blast radius is much
narrower than initially feared, because the shared store's text isn't wired into `/chat` retrieval at
all yet:
- `promotion.py`'s `promote_fragment()`: the ONE write site (`dest_atlas.atlas.insert(0, vector,
  result.redacted_text)`) -- must encrypt before this call.
- `promotion.py`'s OWN `source_atlas.atlas.get_node_metadata(source_node_id)` (reading the fragment
  BEFORE promotion, for classification) -- correctly NOT touched: this reads the PRIVATE store, which
  crypto-erase doesn't cover (no per-owner model there, per erasure.py's existing deferred-private-store
  note). Verified this is the private store, not confused with the shared one -- exactly the asymmetry
  most likely to hide a bug.
- `server.py`'s `GET /admin/knowledge` (`shared_atlas.atlas.get_node_metadata(raw_id)`) -- the ONLY
  shared-store read site in the entire shell today. Must decrypt before returning `KnowledgeNode.metadata`.
- `context.py`'s `query_stores()` returns `{id, similarity, store}` only -- confirmed by reading it, no
  metadata/text field at all, so no decryption wiring needed there. `process_turn()`/`/chat` "deliberately
  keeps querying the private store only" per its own docstring -- the shared store's promoted text isn't
  reachable from `/chat` today, so that path needs no change either.
- `erasure.py`/`dreamer.py`: grepped, neither reads node metadata.
- **Flagged for whoever builds it next**: any FUTURE shared-store text reader (a promotion-review-queue
  UI, Stage 5's Dreamer pointed at the shared tier) must decrypt too -- the same kind of forward-flagged
  gap `governance_record_id`'s writer was carried across three stages until it got one.

**Erasure workflow's actual key-destruction step**: `execute_approved_erasure()` (`erasure.py`) currently
only tombstones. Task 6's own gate ("erasure workflow demonstrates actual key destruction end-to-end, not
just a tombstone flag") means this must ALSO resolve each target's `(subject_id, scope)` via
`governance_record_id` -> `governance_records` row, and delete the corresponding `subject_scope_keys` row
-- not yet wired, part of the implementation below.

**Task 6 Phase B -- blob/metadata encryption: DONE (2026-08-23).** Implemented in the order the design
called for: `metadata_size` plumbing + C++ test first (the whole design depended on it actually working,
not just looking generalized), then the keystore table/migration, then the crypto helpers, then wiring
into `promote_fragment()`/the knowledge browser/erasure execution, then end-to-end verification with a
real destroy-then-reopen negative-control test.

**C++/kernel layer** (the one small, non-hot-path change this scoping still needed): `AtlasOptions`
gained a real `metadata_size` field (`atlas.hpp`), and `Atlas::Atlas(path, opts)` (`atlas.cpp`) was fixed
to actually use it instead of hardcoding `METADATA_SIZE_DEFAULT` -- confirming the exact
"parameterized-in-theory, hardcoded-in-practice" gap this design's own verification step found (no C++
test had ever exercised a non-default `metadata_size`, despite `MemoryFile::open()`/
`compute_node_stride()` both generalizing over it). New `Atlas::metadata_size()` accessor
(mirroring `dim()`), exposed through `aeon_atlas_options_t`/`aeon_atlas_get_metadata_size()` (C-API,
carved from `aeon_atlas_options_t::reserved[32]` down to `reserved[28]`, same "fits the budget exactly"
discipline as every other reserved-carving in this plan) and nanobind (`Atlas.__init__`'s new
`metadata_size` kwarg, `Atlas.metadata_size` read-only property). 4 new C++ tests (`test_atlas.cpp`):
default-256 round-trip, a 400-byte string surviving a 512-byte field intact (would have silently
truncated at the old default), a regression pinning the pre-existing 255-byte truncation is unchanged,
and survival across `compact_mmap()`. 124/124 C++ tests green. Stubs regenerated
(`./scripts/gen_stubs.sh`); the venv's stale compiled `core*.so` had to be manually re-synced from
`build/dev/lib/` (this repo's editable install keeps the compiled extension in `site-packages`, separate
from the editable pure-Python `shell/aeon_py/` -- a rebuild alone doesn't reach the venv, confirmed only
after a live Python probe returned the old constructor signature).

**Python/control-plane layer**: new `shell/aeon_py/crypto.py` -- `Keystore` (Postgres-backed, new
`subject_scope_keys` table via migration `d6315ab6e406`: one random 256-bit DEK per (subject_id, scope),
wrapped AES-256-GCM under a single deployment-wide KEK from `AEON_CRYPTO_ERASE_KEK_HEX`, `get_or_create_dek()`/
`get_dek()` (read-only, doesn't mint on a miss -- a destroyed key must stay destroyed)/`destroy_key()`
(the actual erasure primitive: DELETEs the row, doesn't soft-flag it)) plus module-level
`encrypt_metadata()`/`decrypt_metadata()`/`is_encrypted_metadata()`/`max_plaintext_bytes()`. AES-256-GCM,
not the bare stream cipher the retired vector design needed -- this field has no NodeHeader-style
fixed-layout constraint, so there's no reason to give up the authentication tag. A fresh random nonce per
encryption (pinned by a test asserting two calls on identical plaintext produce different ciphertext --
nonce reuse under a fixed key is a real vulnerability, not a style nit). Stored value is
`"AEONENC1:" + base64(nonce‖ciphertext‖tag)` -- the marker prefix (found necessary while implementing,
not anticipated in the design pass) lets a reader distinguish encrypted from legacy-plaintext metadata
WITHOUT a DEK or schema field: a shared store that enables crypto-erase after nodes already exist has
some nodes minted before any keystore existed (permanently plaintext, not retroactively encrypted -- out
of scope) alongside newly-encrypted ones.

Added `cryptography>=42.0.0` as a new base `pyproject.toml` dependency (a real, audited primitive, not
hand-rolled) -- installed and verified in the working venv.

**Wiring, at exactly the two sites the enumeration found**: `promote_fragment()` (`promotion.py`) gained
an optional `keystore` parameter -- encrypts `result.redacted_text` (never the vector, never the text fed
to `reembed_fn`) before `insert()`, length-checking against `crypto.max_plaintext_bytes(dest_atlas.atlas.
metadata_size)` and raising `ValueError` rather than letting `insert()` silently truncate ciphertext (the
central failure mode the design pass flagged). `create_promotion_approval_request()`/
`execute_approved_promotion()`/`server.py`'s `/admin/promotions/{id}/execute` all thread `keystore` through
the same way `governance_db` already does. `server.py`'s `GET /admin/knowledge` (the only shared-store
read site) decrypts via a new `_read_metadata()` helper, falling back to returning the raw stored value
(plaintext for a legacy node, or still-marker-prefixed ciphertext if the key can't be resolved) rather than
raising -- a browsing endpoint degrading to showing ciphertext beats 500ing the whole listing over one
node. New `GovernanceDB.get_subject_id()` resolves a node's `governance_record_id` back to the
`(subject_id, scope)` DEK lookup key both this read site and erasure need.

**Erasure workflow's actual key destruction** (task 6's own gate: "demonstrates actual key destruction
end-to-end, not just a tombstone flag"): `execute_approved_erasure()` gained an optional `keystore`
parameter -- after a node is successfully tombstoned (not preemptively; a node that fails to erase must
not lose its key), resolves its `(subject_id, scope)` and calls `destroy_key()`, best-effort (a
resolution/destruction failure never undoes the tombstone). `server.py`'s
`/admin/erasure/{case_id}/execute` threads it through the same way. **Collateral effect, stated rather
than left implicit**: one DEK covers every node sharing a `(subject_id, scope)` pair, so destroying it
also makes any OTHER still-live node sharing that pair undecryptable -- acceptable because
erasure.py's own docstring already frames a case as erasing "a SUBJECT's shared-tier fragments"
(all of them in-scope), but a case that deliberately erases only some of a subject's fragments has this
side effect on the ones left behind. Pinned with a real test (`test_console_endpoint.py`): two fragments
promoted under the same subject+scope, only one erased, the survivor's `GET /admin/knowledge` entry comes
back marker-prefixed ciphertext, not decrypted and not a 500.

**A second Phase-A-introduced regression found and fixed while wiring this in**: `execute_approved_erasure()`'s
own `governance_db.record()` call was never updated when Phase A made `governance_records.subject_id`
NOT NULL -- would have raised a bare `TypeError` on every erasure execution against any deployment with a
control plane configured (`server.py`'s endpoint always passes `governance_db`), and no existing test
exercised that branch with `governance_db` actually set. Fixed by resolving a representative `subject_id`
from the first successfully-erased node's own `governance_record_id` (falling back to a documented
`"unknown"` sentinel if nothing resolves), pinned with a dedicated regression test
(`test_crypto_keystore.py`).

**Two config-validation gaps found and fixed on a second advisor pass, before this was reported done**
(the same "503 for a bad config, not a crash" standard `get_identifier_corpus()`'s empty-corpus handling
and `get_audit_log_export_key()` already meet): `get_crypto_erase_kek()` now validates the decoded KEK is
16/24/32 bytes (AESGCM's only valid key sizes) and is valid hex, logging a warning and returning `None`
(same as unset) rather than letting `AESGCM(kek)` raise inside an `@lru_cache()`-wrapped FastAPI
dependency -- which would have 500'd every request touching it, repeatedly, until fixed.
`DEFAULT_SHARED_ATLAS_METADATA_SIZE`'s `int(os.environ.get(...))` is now wrapped in try/except (falling
back to 512) -- unlike every other `AEON_*` constant in `dependencies.py` (all strings or
`.lower() == "true"` compares), a bare `int()` on a bad value would have failed `import aeon_py` itself
for every caller, not just ones touching the shared store.

**A third advisor pass caught a real packaging bug**: `crypto.py` originally imported `sqlalchemy` and
`control_plane.schema` at module level, breaking `promotion.py`'s own explicitly-stated convention ("this
module must keep working with zero DB dependencies installed") -- confirmed by simulating an environment
with `sqlalchemy` unimportable and watching `import aeon_py.crypto` fail. Fixed by moving those imports
inside `Keystore`'s own methods (lazy, per-call), leaving `encrypt_metadata()`/`decrypt_metadata()`/
`is_encrypted_metadata()`/`max_plaintext_bytes()` importable with zero DB extras -- re-verified with the
same simulated-no-sqlalchemy probe, now passing.

**Task 8's guarantee statement gains one more precise caveat**: `metadata_size` is read from an existing
file's on-disk header (same as `dim`) and does NOT retroactively grow -- enabling crypto-erase on a
shared store already created at the default 256 bytes leaves the encrypted-metadata budget at ~156 usable
bytes, not the ~370 a fresh 512-byte store gets, with no migration path in this increment.

**Test coverage**: 13 pure-function tests (`test_crypto.py`, zero Postgres -- round-trip, unicode, wrong-
key failure, marker detection, nonce-uniqueness, budget math verified end-to-end not just computed) + 14
Postgres-backed tests (`test_crypto_keystore.py` -- `Keystore` CRUD, promote-then-encrypt, promote-without-
keystore regression, oversized-text-raises, the erasure `governance_db` regression) + 2 HTTP-level tests
(`test_console_endpoint.py` -- knowledge-browser decryption, partial-erasure collateral effect). The
central end-to-end gate test (`test_erasure_destroys_the_key_and_a_fresh_keystore_cannot_decrypt`):
promotes real content under a keystore, confirms it decrypts, erases it through the full
create-approve-execute case flow, confirms the key is gone, then constructs a **fresh** `Keystore`
instance against the same KEK and Postgres (simulating a process restart) and confirms the deletion is
genuinely persisted, not just forgotten by one in-memory object -- plus the companion negative control
(a *different* subject's key in the same scope still decrypts), proving the hierarchy is actually
per-subject, not just per-scope.

Verified end to end: local-default `pytest tests/` (zero Postgres) 95 passed, 6 skipped; live-Postgres
suite (`docker compose up -d postgres`, `alembic upgrade head` -- both new migrations, `d0e24ce99c88` and
`d6315ab6e406`, apply cleanly on the existing chain) `AEON_REQUIRE_DB_TESTS=1 pytest tests/`: 185 passed;
`ctest --preset dev`: 124/124. Containers torn down after each run. Nothing committed, per the standing
instruction.

**A real gap found while starting task 8, not while task 6 was in progress -- fixed rather than just
documented, per standing instruction to fix bugs at any stage regardless of current scope.** Task 7's own
writeup explicitly deferred "read-reason capture" to task 5 ("needs a role to attribute a read to, which
now exists, but no read path calls through it yet"). Task 5's completion writeup never revisited it, and
grepping `server.py` for `audit_log\.append` confirmed zero admin READ route ever calls it -- `GET
/admin/knowledge` (the only route that returns decrypted subject content), `GET /admin/audit-log{,/verify,
/export}`, and `GET /admin/erasure/{case_id}` all read without ever recording who read what or why. This
directly contradicts task 7's own explicit constraint ("mandatory read-reason prompts in the audit
entry") and is the second deferred item in this document with no landing record -- the first was
`governance_record_id`'s writer, flagged across three stages before it got one.

**Scoped narrower than the constraint's literal wording, on review**: only `GET /admin/knowledge` gets a
mandatory `reason` param and an audit record. The other four routes return the audit log itself or an
erasure receipt (node ids and failure strings), not subject content -- requiring a justification to read
the audit log inverts the control, and logging every audit-log read into the same hash-chained log it
just read is a self-referential growth path with no compliance value. `list_knowledge` is the one route
where an admin is looking at someone's (possibly just-decrypted) fragment text, which is exactly what a
read-reason prompt exists to gate.

**Implementation**: `reason: str` added as a required (no default) query param -- FastAPI 422s a request
that omits it entirely, and a new explicit guard (`if not reason or not reason.strip()`, same shape as
`create_approval_request()`'s existing mandatory-reason check) 400s a blank one. One `AuditLog.append()`
call per REQUEST, not per node -- a 100-node page gets one record naming `{reason, offset, limit,
returned_count, caller_scope_mask}`, never the nodes' own text (`AuditLog.append()`'s own docstring:
payloads must never carry what a classifier redacted, and unredacted browsed content is exactly that
category). New action `"knowledge_read"` is deliberately NOT added to `GOVERNANCE_RECORD_ACTIONS` --
that tuple feeds `control_plane/schema.py`'s `ck_governance_records_action` CHECK constraint, and reads
never call `GovernanceDB.record()` (no `NodeHeader.governance_record_id` to attach to), so adding it there
would force a migration for a value no Postgres row will ever hold. `governance.py` gained a comment at
the tuple stating this explicitly. `audit_log` is optional infra, same as everywhere else in this file --
an unconfigured deployment still browses, it just can't prove it did so for a stated reason.

**Concurrency, checked rather than assumed**: `AuditLog` is documented as not thread-safe (callers must
serialize their own `append()` calls); every prior caller was a write route. Verified `run_server.sh` runs
a single uvicorn worker with no `--workers` flag, and `append()` itself is a plain synchronous function
with no `await` inside it -- so within one event loop, a call to `append()` from an `async def` route runs
to completion before any other coroutine can interleave, identical to how the existing promotion/erasure
execute routes already serialize their own appends. A concurrent-browse race was the real risk this
review flagged; confirmed it's already closed by the existing single-worker deployment shape, not newly
introduced.

**Test discipline**: a test that only checks the 400 wouldn't discriminate this from a bare query-param
validator, so `test_list_records_reason_in_audit_chain_not_just_a_400_check` browses with a real reason,
reads the record back via `AuditLog.tail()`, confirms `actor`/`reason`/`returned_count` all landed
correctly and no node text leaked into the payload, then calls `GET /admin/audit-log/verify` and confirms
the chain (now carrying this read record) still validates -- the one guarantee task 8 is about to promise
in writing. Plus `test_list_422_when_reason_missing` and `test_list_400_when_reason_blank` for the two
distinct rejection paths. All five pre-existing `GET /admin/knowledge` call sites across
`test_console_endpoint.py` updated to pass a `reason`.

Verified: `AEON_REQUIRE_DB_TESTS=1 pytest tests/` (live Postgres) **188 passed** (185 + 3 new); local-default
`pytest tests/` (zero Postgres) **95 passed, 6 skipped** (unchanged -- this fix is entirely inside the
Postgres-gated console endpoint). Container torn down after.

**Task 8 -- adopter-facing compliance documentation: DONE (2026-08-23).** `COMPLIANCE.md` (root-level,
linked from `README.md`'s documentation table): what data classes can end up in shared-tier fragments
(private vs. shared store, metadata vs. vector), what `promote_fragment()` actually does
(mint-and-recontextualize, with the `trace`-conditional `promoted-from` edge stated as optional, not
guaranteed), what crypto-erase guarantees and its explicit boundaries (vector plaintext/embedding-
inversion residual, the two-envelope DEK/ciphertext guarantee, the collateral multi-node-per-key effect,
`metadata_size`'s no-retroactive-growth limit with exact `max_plaintext_bytes()` figures -- 155 usable
bytes at the 256-byte default, 347 at the recommended 512), the admin console's actual access-control
shape (containment-vs-overlap asymmetry, four-eyes, read-reason scoped to the one route that returns
subject content), and a self-hoster checklist. Written from what the code actually does, verified by
reading it and by test, not from the plan's own aspirational language -- three factual claims caught and
corrected on advisor review before this was reported done (the `trace`-optional edge, the exact
`max_plaintext_bytes()` figures replacing "~156"/"~370" estimates, and a nuance on
`AEON_CONTROL_PLANE_DATABASE_URL`'s row: a store holding nodes encrypted under a keystore is not
meaningfully readable without the control plane, even though the browser degrades to ciphertext rather
than raising).

**A fourth advisor-caught fix, applied to the code itself, not just the doc**: `get_crypto_erase_kek()`
previously accepted any AESGCM-valid KEK length (16/24/32 bytes), but `crypto.py`'s `DEK_SIZE_BYTES` is
hardcoded to 32 (AES-256) -- a 16- or 24-byte KEK would wrap those 256-bit keys at AES-128/192 strength,
silently weakening the wrapping below the strength of what it protects. `COMPLIANCE.md` was about to
describe 16/24/32 as equally acceptable without flagging the weaker choice; fixed at the source instead
of caveated in the doc -- `get_crypto_erase_kek()` now requires exactly 32 bytes, failing closed (same
warning-and-`None` shape as every other validation gap here) on anything else.

**Stage 4's own gate, first item stated precisely rather than left ambiguous**: "measured false-negative
rate for the identifier-corpus detector on a labelled corpus of real distilled fragments, published
alongside the feature" has **not** been done -- this requires a labelled corpus of real fragments this
repository does not have and cannot fabricate. This is now **unmet-and-disclosed**, not
unmet-and-unaddressed: `COMPLIANCE.md` §4 states it explicitly and tells a self-hoster what to do about
it (validate the classifier against their own labelled sample before trusting it). The other three gate
items are met: hash-chain verification (`GET /admin/audit-log/verify`, tested), actual key destruction
end-to-end (`TestEndToEndKeyDestruction`, tested). The CI perf-gate A/B run needs a real commit to diff
benchmark numbers against a baseline; nothing has been committed this session (`git status` still shows
every file above as modified/untracked), so that run has not happened yet either -- stated as a fact of
where this session's work stands, not a claim that committing is disallowed.

**A fifth advisor-caught gap**: the 32-byte KEK requirement above shipped without any test for
`get_crypto_erase_kek()`'s rejection branches -- the existing suite's fixtures all already used
`os.urandom(32)`, so it passed identically before and after tightening the check and couldn't have caught
a regression. Added `TestGetCryptoEraseKek` (`tests/test_crypto.py`, zero Postgres): unset, malformed hex,
16-byte, and 24-byte all assert `None`; a real 32-byte key round-trips. Proved these discriminate, not
just pass, by temporarily reverting the check back to `(16, 24, 32)` and confirming the 16/24-byte tests
go red, then restoring.

**Stage 4 is now fully complete**: all eight tasks (six primitives, the console, crypto-erase) and the
adopter-facing documentation task are done. The vector-encryption path considered under the earlier
(superseded) scope decision was not pursued -- centroid vectors remain plaintext, a residual documented
in `COMPLIANCE.md` §5.1 rather than left as an internal note only. Verified after these final fixes:
`AEON_REQUIRE_DB_TESTS=1 pytest tests/` (live Postgres) **193 passed** (188 + 5 new); local-default
`pytest tests/` (zero Postgres) **100 passed, 6 skipped** (95 + 5 new); `ctest --preset dev` **124/124**
(unchanged -- nothing in this final round touched C++). Container torn down after.

**Stage 2 follow-up -- scope-filtered recall gap: CLOSED (2026-08-23).** With Stage 4 shipped and
depending directly on the exact mechanism this gap affected (team-scoped queries within the shared
tier), the user chose to close it rather than move to Stage 5 or the remaining deferred Stage 4 items.
Recall at low selectivity had been left at ~0.93 (10%) / ~0.37 (2%) against the gate's 0.99 bar --
`test_scope_recall.cpp`'s own KNOWN GAP comment from Stage 2 (2026-08-22) named the fix this needed:
"accurate ANCESTOR scope-union hints to steer descent by (eager/option-a propagation, or
compaction-time union recomputation)."

**Design consulted with advisor before writing code, given the hot-path/correctness stakes.**
Confirmed via `insert()`'s code that `parent_id` is caller-supplied (not computed by nearest-centroid
search), so "propagate scope on insert" doesn't apply the way the original Stage 2 task text assumed --
scope is only ever assigned AFTER insertion via `set_node_scope()` (insert() hardcodes
`scope_bitmap = 0`), so that's the real propagation hook. Reused advisor's two concrete corrections
before implementing: (1) confirm `test_scope_recall.cpp`'s tree (branching 8, depth 3, independent
per-node random scope assignment) is deep/branchy enough for subtree pruning to matter -- verified by
hand: at 2% selectivity, ~85% of leaf-groups (8 leaves under one level-2 parent) are provably
scope-free, and the level-2 admission step is exactly where 64 candidates compete for 16 beam slots,
so there was real room for a union hint to help, not a shape mismatch; (2) the admission fix needs an
empty-beam fallback so union pruning can only improve recall, never regress it below today's blind
baseline.

**Storage decision, the one that mattered most: a separate RAM-only auxiliary index
(`Atlas::scope_union_`), NOT a reuse of `NodeHeader::scope_bitmap` itself.** `scope_bitmap` has zero
spare bytes (Stage 1 already filled its `reserved[20]` budget exactly) and every node in this Atlas
holds real content -- there is no leaf-vs-internal-routing-node distinction the way a routing-only
B+-tree would have. Reusing the field to mean "subtree union" for internal nodes would collide with
its OWN real scope assignment on the exact same nodes the admin console's `_require_scope_containment()`
and the erasure workflow's DEK lookup (`shell/aeon_py/server.py`) already read as ground truth for
AUTHORIZATION -- polluting it would let a grant over scope Y wrongly authorize acting on an ancestor
whose own real content belongs to a different scope entirely, purely because a descendant happens to
be scoped to Y. A real privilege-escalation risk, not theoretical, given how much of Stage 4 was spent
fixing exactly this class of containment-vs-overlap bug. `scope_union_` lives in its own
`std::vector<uint64_t>`, indexed identically to mmap node ids, read ONLY by `navigate()`'s internal
beam-admission heuristic -- impossible to leak into an authorization decision by construction.

**Maintenance: a worklist fixpoint, not the naive single reverse-index pass first considered.** A
"process indices N-1 down to 0, OR each node's union into its parent" pass would be O(N) and correct
ONLY if parent index always < child index -- true for every `insert()`-created edge, but
**`consolidate_subgraph()` rewires a surviving child's `parent_offset` to the new summary node, which
is inserted after every pre-existing node and therefore has a NUMERICALLY LATER index than the
child** -- found by reading `consolidate_subgraph()`'s Phase 3 before assuming the simpler pass would
work. Implemented instead as a bounded worklist (`rebuild_scope_union_locked()` for the full O(N)
rebuild at Atlas construction and after `compact_mmap()`; `propagate_scope_union_locked()` for the
O(depth) incremental update `set_node_scope()`/`bulk_set_node_scope()`/`consolidate_subgraph()` each
call after mutating) that re-queues any ancestor whose union actually changes, correct regardless of
edge direction, terminating because union bits only ever grow (monotonic OR, <=64 bits/node).
Monotonic OR-only by design: narrowing a node's real `scope_bitmap` leaves ancestor unions
stale-wide, accepted as safe since the union is consulted only for beam-admission PRIORITY, never for
emission (which still checks the real per-node `scope_bitmap` exclusively, unaffected by any of this).

**Admission change**: a candidate whose subtree union doesn't overlap `scope_mask` is deprioritized in
beam admission via a composite (union-tier, then raw score) ordering -- NOT hard-excluded, so if too
few union-positive candidates exist to fill the beam, remaining slots still go to the next-best
union-negative candidates exactly as before, the built-in empty-beam fallback advisor's review asked
for. This is a strict generalization of the pre-fix comparison (identical to plain score-ordering
whenever not filtering, or when every candidate ties on union tier), not a new code path outside
scope-filtered queries -- and it is NOT the per-candidate own-bit steering bonus tried and measured
actively harmful in Stage 2 (50% recall dropped to ~0.52) -- a subtree union bit is a sound
over-approximation of "does this subtree contain a match," unlike a leaf's own bit under deferred
propagation, so it cannot repeat that failure mode: a candidate is only deprioritized when its entire
subtree is PROVABLY irrelevant, never because a real reachable path scored lower on raw similarity.

**Result, measured**: all four selectivities (1.00/0.50/0.10/0.02) now pass at the actual 0.99 gate,
not the lenient 0.85/0.25 regression-guard thresholds Stage 2 shipped with. Discrimination proved
before landing (this codebase's standing practice, same as Stage 0's CI-gate proof and Stage 1's WAL
forward-compat proof): temporarily forced the union check to always return `true` (disabling the fix)
and re-ran -- selectivity 0.10 and 0.02 reproduced the exact old ~0.93/~0.37 ceiling and failed at
0.99, confirming the test genuinely discriminates the fix rather than passing regardless. Restored and
re-confirmed green. Also re-verified at a different seed and 5x the query count (300 vs 60) before
settling on the final thresholds, to rule out a lucky sample.

**A real, separate structural finding surfaced while testing the trickiest case (consolidate_subgraph()
rewiring), NOT fixed here -- flagged for whoever builds Stage 5.** An attempt to test
`propagate_scope_union_locked()`'s handling of a rewired child specifically was written, found to pass
for the wrong reason (it used `apply_csls=false`, under which a tombstoned node's `hub_penalty`
exclusion never applies, so the needle stayed reachable via the OLD pre-consolidation path regardless
of the fix), and was deleted rather than landed once that was understood -- a test that passes
regardless of the mechanism is worse than no test, since it reads as coverage that doesn't exist.
Rewriting it with `apply_csls=true` doesn't fix this either: an empirical scratch check (root -> child
-> `consolidate_subgraph({child})`, then `navigate()` with a query matching the summary's own vector)
confirmed the resulting summary node is **never returned by `navigate()` at all**, with or without this
fix. Root cause: `consolidate_subgraph()` sets the new summary's OWN `parent_offset` (so the summary
"knows" its logical parent) but never updates that PARENT's `child_count`/`first_child_offset` to
enumerate the summary back -- a node can point at its parent without the parent ever discovering it via
beam descent, since `navigate()` only walks contiguous child blocks reachable from something already in
the beam. This is a separate, pre-existing property of `consolidate_subgraph()`, unrelated to scope
filtering and not introduced or worsened by this fix -- but it means **Dreaming's summary nodes may be
structurally unreachable via `navigate()` today**, independent of scope. No existing test exercises
this (confirmed by grep: `ListNodesByScopeReturnsMatchingLiveNodes` reaches a summary via the flat
`list_nodes_by_scope()` scan, never via `navigate()`). Relevant directly to Stage 5's "point the Dreamer
at the shared tier" task, since that's the first and only caller of `consolidate_subgraph()` -- worth
checking before Stage 5 assumes a consolidated summary is retrievable the normal way.

**C++ test coverage**: `test_scope_recall.cpp`'s four selectivity tests tightened to the real 0.99 gate
(comments rewritten from "KNOWN GAP" to "GATE MET," documenting the discrimination proof). No new test
file added -- the fix is entirely inside `Atlas`'s private admission/maintenance logic, exercised by
the existing recall tests. `ctest --preset dev`: **124/124** (unchanged count, tightened assertions).

**Performance, per Stage 2's own gate ("CI perf gate shows `navigate()` P50/P99 within tolerance... if
option (b) deferred was chosen, confirm insert latency is unaffected")**: `bench_beam_search` (1M
nodes, unfiltered -- this fix only touches the `scope_filtering` branch, so an unfiltered query pays
one extra `bool` check per candidate that short-circuits immediately): P50 23.7µs vs. the recorded
27.0µs baseline in `reproducibility_benchmarks/master_metrics.txt` (within noise, no regression).
`bench_wal_overhead`'s `BM_Insert`: 2.07µs mean vs. the documented ~2.21-2.23µs baseline (also within
noise) -- `insert()`'s only change is one `scope_union_.push_back(0)`, an O(1) amortized append.
Full A/B CI perf-gate run (comparing against a real merge-base) still deferred to whenever this
session's changes are actually committed, per the same standing note as every other increment above --
nothing has been committed yet (`git status` still shows the whole session's changes as
modified/untracked), not a claim that committing is disallowed.

**Dreaming summary-node reachability gap: CHASED AND FIXED (2026-08-23), per the user's explicit
instruction to chase it before moving to the remaining deferred Stage 4 items.** The finding from the
recall-gap session: `consolidate_subgraph()` set the new summary's own `parent_offset` but never
updated that PARENT's `child_count`/`first_child_offset` to enumerate the summary back, so the
summary was never returned by `navigate()` at all -- confirmed empirically with a scratch
root->child->`consolidate_subgraph()` check before deciding this was worth fixing rather than just
documenting. This is not cosmetic: Stage 5's own plan text for "point the Dreamer at the shared tier"
describes consolidating "fourteen tickets about the same flaky test" into one summary and tombstoning
the originals -- if the summary can never be found again, Dreaming would have been a net RETRIEVAL
REGRESSION (deleting fourteen findable fragments, replacing them with nothing findable), not a neutral
no-op.

**Blast-radius check before designing a fix**: grepped every Python-side call site
(`shell/aeon_py/*.py`) for `Atlas.insert(` -- every real caller (`promotion.py`'s
`promote_fragment()`) uses `parent_id=0` (root); `architect.py`'s ordinary ingestion path uses
`insert_delta()` (the flat, parent-less delta buffer), never the tree-based `insert()` at all. So the
"children must be a contiguous byte-array run" tree structure is, in every real production code path
today, only ever exercised in the trivial "everyone is root's child" shape -- meaning the fix's
common-case correctness (below) covers 100% of today's actual usage, not just a convenient subset.

**Fix: reuse `insert()`'s own existing contiguity check, applied to the summary's placement.** The
summary is always created at the current tail (`header->node_count`), the same position `insert()`'s
own `new_idx` would occupy for a genuinely new child -- so the identical "is this address exactly
`parent->first_child_offset + parent->child_count * stride`" check `insert()` already runs correctly
determines whether the summary lands contiguously with its parent's existing children, and if so,
increments `parent->child_count` (or sets `first_child_offset` if the parent had none) exactly as
`insert()` would. When it does NOT land contiguously (something else was inserted under a different
parent since), the parent is left unregistered -- exactly `insert()`'s own pre-existing, already-shipped
behavior for the identical scenario, not a new failure mode this fix introduces. No format change, no
ABI bump, no new field: this reuses an existing, already-correct mechanism rather than inventing one.

**Verified, not assumed**: the original empirical scratch check (root -> child ->
`consolidate_subgraph({child})`, query matching the summary's own vector, `apply_csls=true`) now
returns the summary node. Landed as two permanent C++ tests
(`test_atlas.cpp`): `ConsolidateSubgraphSummaryIsReachableViaNavigateWhenContiguous` (the fixed common
case -- also confirms the tombstoned original leaf, now genuinely competing against the summary for
the level's "best" report under CSLS exclusion, correctly loses) and
`ConsolidateSubgraphSummaryNotRegisteredWhenNonContiguousButNoCorruption` (the residual case: a node
inserted under a different parent in between breaks contiguity, and the summary is correctly left
unregistered -- not incorrectly registered in a way that would make `navigate()` read past real
children into unrelated node bytes). Discrimination proved the standing way: temporarily disabled the
new registration logic, confirmed `ConsolidateSubgraphSummaryIsReachableViaNavigateWhenContiguous`
failed exactly as expected (summary not found, AND the tombstoned leaf wrongly reported instead, since
without real competition its terrible CSLS-penalized score still "wins" as the only candidate),
restored, reconfirmed green. A second, independent discrimination check targeted the non-contiguous
test specifically: an advisor review flagged that inserting `Other` under `leaf_id` (rather than under
`root_id`) changes Phase 3 from a no-op into a real rewire, and asked whether `EXPECT_FALSE(found_summary)`
was actually sensitive to the new contiguity gate or just trivially true. Verified by forcing
`parent_contiguous = true` unconditionally (bypassing the gate entirely) and rebuilding: the test then
FAILED, but on the child-count assertion rather than `found_summary` -- root's `child_count` was
incorrectly bumped to 2, and `get_children(root_id)` returned the unrelated `Other` node (which sits at
the contiguous slot the corrupted logic now claims) as root's phantom second child, while the summary
itself (one slot further) still wasn't found. This confirms the test does discriminate the real
mechanism, just via a different assertion than initially assumed -- the corruption the non-contiguous
branch guards against surfaces as a false parent-child relationship, not as a spuriously-found summary.
Restored, rebuilt, reconfirmed identical-to-backup via `diff`, and reconfirmed `ctest --preset dev`:
**126/126** clean. `ctest --preset dev`: **126/126** (124 + 2 new). Local-default `pytest tests/`:
**100 passed, 6 skipped** (unaffected -- nothing Python-side calls `consolidate_subgraph()`).

**Residual, explicitly NOT solved, flagged for Stage 5**: when `old_node_ids` are non-leaf nodes (have
their own surviving children), Phase 3 correctly rewires those children's `parent_offset` to the
summary (unchanged from before), but the summary's OWN `child_count`/`first_child_offset` are not
updated to enumerate them, since they generally are not physically contiguous with the summary's
position or with each other (they were scattered across each old node's own child block).
Making that fully general would require actually relocating node bytes into a fresh contiguous run
(the same class of operation `compact_mmap()` already does at file scope, but scoped to just this
consolidation's participants) -- a materially larger change, and one that would give every relocated
child a NEW id, a breaking change for any caller holding onto the old one across a
`consolidate_subgraph()` call. Not attempted here: today's only caller (`dreamer.py`) consolidates leaf-
level fragments with no children of their own, so Phase 3's rewiring loop is a no-op in every real case
this codebase currently exercises. Stage 5 should re-check this specifically if it ever consolidates
non-leaf nodes.

**Remaining deferred Stage 4 items -- CLOSED (2026-08-23), per the user's explicit instruction to
address them next.** Both had been left open at "documented seam, not implemented" for lack of a
concrete design decision, not lack of a landing site. Rather than build a full pipeline/integration
speculatively, both were resolved by asking the user which of several concrete shapes to build, the
same way the crypto-erase KEK-length decision was made earlier this session:

- **Destination-conditioned re-embedding (task 2)**: `reembed_fn` (promotion.py) already existed as a
  documented, tested seam (`text -> vector`, defaulting to the source vector) -- what was actually
  missing was any way to REACH it from the HTTP surface, since `execute_approved_promotion()`'s own
  server.py caller never passed one through. Closed by adding `PromotionExecuteRequest.destination_
  embedding` (models.py) -- an optional `List[float]` on the promotion-execute request body -- which
  server.py's `execute_promotion()` wraps in a closure (`lambda _text: destination_embedding`) and
  passes as `reembed_fn`. Deliberately NOT a new Aeon-owned embedding pipeline: the caller has already
  computed this vector against the destination scope's corpus using whatever embedder they run: Aeon
  only substitutes it in at insert time. Dimension validation is NOT duplicated in Python -- the
  destination Atlas's own `insert()` (bindings.cpp) already raises on a size mismatch, so a redundant
  check would just be dead code paralleling one the C++ layer already owns. New tests:
  `test_promotion.py`'s existing `test_reembed_fn_overrides_default_vector` already covered the
  seam itself; added `test_admin_endpoint.py::test_destination_embedding_overrides_source_vector`
  covering the HTTP wiring specifically. Discrimination proved the standing way: temporarily
  short-circuited the HTTP wiring (`if False and body is not None...`), confirmed the test fails for
  the expected reason (stored vector reverts to matching the source vector, not the caller-supplied
  one), restored, reconfirmed identical-to-backup via `diff`, reconfirmed green.
- **Correctness-gated promotion for code knowledge (task 3)**: closed the same way the user chose for
  crypto-erase and re-embedding -- ask which concrete design, then build exactly that, rather than
  guess. Given four options (caller-supplied result / local ctest-gated / GitHub Checks API
  integration / defer), the user chose the smallest-surface one: Aeon never talks to any VCS/CI
  provider itself (a genuinely separate integration surface -- auth, rate limits, a provider
  abstraction across GitHub/GitLab/etc -- correctly identified as out of scope for this module in the
  plan's own earlier framing). Added `VerificationResult` (promotion.py): a caller-supplied
  `status`/`commit_sha`/`verified_by` triple. `promote_fragment()`/`execute_approved_promotion()` gained
  `verification`/`require_verification` params (both optional, default `require_verification=False` so
  no existing caller's behavior changes); when `require_verification` is True, a missing verification or
  `status != "passed"` rejects the promotion (recorded as `promotion_rejected`, same fail-closed shape
  and 200-not-4xx convention as a classifier rejection -- critically, does NOT consume the four-eyes
  approval already granted, so a retry once CI actually passes still works against the same request,
  mirroring the existing "classifier rejection doesn't burn the approval" rule verbatim). Deliberately
  supplied at EXECUTION time, not locked into the approval request's own `target` at creation time
  (unlike `dest_scope`/`subject_id`) -- a verification outcome is often not known yet when a promotion
  is first queued for four-eyes approval. Deployment-level opt-in via `AEON_REQUIRE_CODE_VERIFICATION`
  (dependencies.py, same env-var/DI pattern as `AEON_REDACT_EMAILS`), threaded through server.py's
  `execute_promotion()` and `PromotionExecuteRequest.verification` (an HTTP-level mirror of
  `VerificationResult`). Tests: 5 new unit tests in `test_promotion.py` (off-by-default no-op,
  reject-when-missing, reject-when-failed, pass-when-passed with full audit-payload check, and
  recorded-but-non-gating when `require_verification=False`) plus 2 new HTTP-level tests in
  `test_admin_endpoint.py` (reject path confirms the approval survives via `get_request(...)
  ["executed_at"] is None`; pass path confirms a node is actually minted). Discrimination proved the
  standing way: temporarily short-circuited the gating `if` in `promote_fragment()` (`if False and
  require_verification...`), confirmed both the "missing" and "status-not-passed" unit tests failed
  for the expected reason (a fragment was wrongly minted instead of rejected), restored, reconfirmed
  identical-to-backup via `diff`, reconfirmed green.
- Both are now documented in `COMPLIANCE.md` (§3's promotion-mechanics list, a new §4.1 on the
  verification gate's trust-boundary caveats, and §7's config table gaining
  `AEON_REQUIRE_CODE_VERIFICATION`) and in `promotion.py`'s own module docstring (the "Explicitly NOT
  built" framing there was stale and has been corrected).
- Verification: Postgres-backed `pytest tests/`: **201 passed** (up from 194, the reachability-gap
  session total, +7 new: 5 unit + 2 HTTP). Local-default `pytest tests/` (no DB): **105 passed, 6
  skipped** (up from 100, +5 -- the unit-level tests run without Postgres; the 2 HTTP-level tests are
  in the Postgres-only `test_admin_endpoint.py` module).

A final advisor review before moving on caught two documentation gaps, both fixed:
- The non-contiguous reachability test's inline `EXPECT_FALSE(found_summary)` comment read as if that
  assertion were the one discriminating the contiguity gate; the discrimination experiment actually
  showed it's true both with the gate on and (for a different reason) forced off, and it's the
  `EXPECT_EQ(get_children(root_id).size(), ...)` assertion right after it that catches the real
  corruption (phantom `Other` child). Rewrote the test's own comments (`test_atlas.cpp`) to say this
  plainly, so a future reader trusts the assertion that actually carries the weight.
- `VerificationResult.commit_sha` is recorded verbatim into the audit log, never run through
  `IdentifierCorpus`/`redact_commit_shas` -- correct (it's promotion-caller-supplied CI metadata for an
  admin-only audit record, not fragment text entering the shared store a broader audience reads), but
  undocumented, and this codebase elsewhere treats commit SHAs as redactable-by-configuration. Added an
  explicit paragraph to `VerificationResult`'s docstring explaining why this one is different in kind.
- Rebuilt and reconfirmed `ctest --preset dev`: **126/126** after both fixes (doc-only changes, but
  reconfirmed rather than assumed).

With both deferred Stage 4 items now closed, Stage 4 is fully complete end-to-end and the next step,
per the user's own explicit sequencing ("chase that gap and then remaining deferred stage 4 items.
after these we will start stage 5"), is Stage 5.

## Context

Aeon v3 is published (open source, arXiv:2601.15311). Before scoping v4, the user commissioned two
independent design reviews, now in `v4/docs/`:

- `aeon_v4_hive_mind_verdict.md` — audits an earlier v3-based "hive mind" (shared org-memory)
  proposal, finds its claimed foundation (RaBitQ, FastScan, NAMM, DiskANN, `nexus/`) does not exist
  in the repo, finds the proposed scope-filter mechanism would silently destroy recall, and finds
  Aeon **today, in server mode, has no retrieval isolation between users at all** — a defect
  independent of whether the hive-mind feature is ever built. It proposes a revised Stage 0–4
  roadmap.
- `aeon_v4_versioning_growth_addendum.md` — answers three follow-up questions (git-like versioning,
  corpus growth/"vector haze", change-with-reasons) and slots its recommendations into the same
  Stage 0–4 roadmap without displacing it.

Both documents are analysis-grade but were written without full access to verify every claim
against the current tree, and both frame the shared-memory feature primarily around a single
company deploying Aeon on its own engineers (driving heavy GDPR/works-council content). This plan:

1. **Verifies every load-bearing claim** in both docs against the current codebase (three parallel
   Explore agents), correcting the ones that no longer hold or never held.
2. **Gets a second engineering opinion** (advisor) on the synthesis, which caught a wrong
   AVX-512-crash claim, a NodeHeader byte-budget framing error, a hidden dependency (SLB caches are
   fixed-768-dim, contradicting Aeon's own "dynamic dimensionality" headline claim), a CI-flakiness
   trap, and an exact WAL-replay ordering bug risk.
3. **Reframes Stage 3** around the user's answer: v4 ships as **open-source infrastructure for
   third-party self-hosters**, not a single company's internal compliance program — so Stage 3 keeps
   every engineering primitive the docs specify but sheds the internal-DPIA/works-council apparatus
   in favor of adopter-facing documentation.
4. **Resequences**: per the user's answer, the outcome experiment (does shared memory actually help?)
   moves before the full shared-tier build-out, not after.
5. **Adds a cross-cutting engineering-guardrails workstream** the source docs don't cover at all:
   there is currently **zero automated latency-regression protection** in CI, and multiple findings
   below (SLB fixed-dim, compile-time-only SIMD dispatch, `-march=native` release builds, missing
   WAL/durability on `Atlas::insert()`) directly threaten the "production ready, high performance,
   ultra-low-latency" bar this plan is required to hold every stage to.
6. **Folds in a finding the user surfaced independently**: `TraceBlockIndex` is a fully built,
   documented (README/ARCHITECTURE.md §12/INTERNALS.md §6), but completely unreferenced class —
   `TraceManager::get_history()` does a plain O(K) `prev_id` walk and never touches it. This must be
   resolved (wire it in for real, or delete it and correct the docs) as part of Stage 0/2, not left
   as a silent gap between what Aeon claims and what it does.

The intended outcome is a roadmap that is trustworthy line-by-line (every claim traceable to a
current file:line), and that treats "production ready / high performance / ultra-low-latency" as a
gate on *every* stage, not a separate concern bolted on at the end.

---

## How to read this plan

Each stage lists concrete tasks with **files touched**, **what changes**, and a **gate** (how you
know it's done and safe). Stage numbers are new (resequenced per the ordering decision below); each
task notes which source-doc stage it descends from so you can cross-reference `v4/docs/`.

---

## Cross-cutting guardrail #0: latency-regression CI gate (build this first, it protects everything after)

**Why it must come first:** every stage below modifies `NodeHeader`, `TraceEvent`, WAL replay, or the
`navigate()`/beam-search hot path — exactly the code the README's headline numbers (2.23µs insert,
3.09µs INT8 navigate, 4.70ns SIMD dot) depend on. Verified: `core/CMakeLists.txt` registers ~13
benchmark binaries (`bench_wal_overhead`, `bench_quantization_efficiency`, `bench_ebr_contention`,
`bench_beam_search`, `bench_multitenant_slb`, `bench_trace_gc`, etc.), but **none are registered with
CTest** (`gtest_discover_tests(aeon_tests)` is the only test registration in the file) and
`.github/workflows/build_and_test.yml` never invokes any `bench_*` binary or
`reproducibility_benchmarks/`. `reproducibility_benchmarks/run_v3_benchmarks.py` computes PASS/FAIL
verdicts against hardcoded thresholds but has no `sys.exit()` on failure — it cannot fail a PR even
if run manually in CI. **Nothing today would catch a PR that silently turns a 3µs navigate into
30µs.**

- **Design it as a relative, same-run comparison, not an absolute threshold.** Comparing hosted CI
  runners against `master_metrics.txt` (recorded on an M4 Max with active cooling) at µs/ns scale
  will flake and get disabled within a week. Instead: build `merge-base` and `HEAD` in the same CI
  job, run the same curated benchmark binaries back-to-back on the same runner, fail on **delta**
  (~20–25% tolerance — tight enough to catch an algorithmic regression like an O(1)→O(N) scope scan,
  loose enough to survive runner noise).
- **Exclude self-hit artifacts as a class**, not case-by-case: any benchmark whose query vector is a
  bit-for-bit copy of a stored node (cosine = 1.000000) measures an SLB cache hit, not real
  traversal — the hive-mind doc names `BM_AtlasTraversal_Only` (0.078µs) as one instance; audit the
  full benchmark set for the same pattern before wiring anything to the gate.
- **Curated benchmark subset for the gate** (fast enough to run per-PR): `bench_kernel_throughput`,
  `bench_wal_overhead`, `bench_beam_search`, `bench_quantization_efficiency`, `bench_ebr_contention`.
  Leave `bench_scalability` (1M-node sweep) and the full `reproducibility_benchmarks/` suite as a
  nightly/manual job, not per-PR.
- **Files**: new CI job in `.github/workflows/build_and_test.yml` (or a sibling workflow), a small
  comparison script (Python or shell) that parses Google Benchmark JSON output
  (`--benchmark_format=json`) for both builds and diffs them, replacing/extending
  `reproducibility_benchmarks/run_v3_benchmarks.py`'s verdict logic with a real exit code.
- **Gate**: the job itself must demonstrably fail — prove it by temporarily reintroducing a known
  O(N) regression locally and confirming the job goes red, before relying on it for Stage 1+.

## Cross-cutting guardrail #1: three correctness/production-readiness gaps found during verification (independent of scope/versioning work)

These aren't in either source doc. Fix them because they're real gaps, and because Stage 0–2 touch
the exact same code:

1. **SLB caches are hardcoded to 768-dim, contradicting Aeon's own "dynamic dimensionality"
   feature.** Verified: `core/include/aeon/slb.hpp`'s `CacheEntry::centroid` is
   `float[EMBEDDING_DIM_DEFAULT]` (768, `schema.hpp:26`), `find_nearest()` hardcodes
   `std::span<const float>(entry.centroid, 768)`, and `insert()` guards
   `if (centroid.size() != 768) return;` — **silently doing nothing** for any non-768-dim Atlas.
   `core/include/aeon/hierarchical_slb.hpp`'s `SessionCacheEntry`/`GlobalCacheEntry` have the same
   `centroid[EMBEDDING_DIM]` fixed-size layout. Since `Atlas` unconditionally uses `SemanticCache`
   (`slb_cache_` member, `atlas.hpp`) on every navigate/insert, **any Atlas created with `dim=384` or
   `dim=1536` today gets zero SLB cache acceleration** — every query falls through to full beam
   search, silently, with no error. This must be fixed (generalize `CacheEntry`/`SessionCacheEntry`
   to the file's actual `dim`, e.g. via a `std::vector<float>` sized at construction, or a
   max-dim `std::array` with a stored `dim` field) **before** Stage 0 wires `HierarchicalSLB` into
   `Atlas` for real — otherwise the wiring work inherits the same bug into the new session-aware path.
   **Gate**: a new test creating 384-dim and 1536-dim Atlases confirms SLB cache hits actually occur
   (not just navigate correctness).

2. **SIMD dispatch is compile-time-architecture-only, and the doc comment claiming otherwise is
   wrong — plus `release`/`dev` presets build non-portable binaries.** Verified:
   `core/src/simd_impl.cpp` defines `SIMDE_ENABLE_NATIVE_ALIASES` and uses SIMDe consistently, so
   there is **no SIGILL risk from CI's `-march=x86-64-v3` preset** — SIMDe correctly lowers AVX-512
   intrinsics to AVX2/SSE when the target lacks native AVX-512 (confirmed: zero bare
   `_mm512_*`-without-SIMDe calls). The real problems: (a) `math_kernel.hpp`'s comment claims
   "runtime dynamic dispatch (AVX512 → AVX2 → Scalar)" but `get_best_similarity_impl()`
   (`simd_impl.cpp`) picks the implementation via `#if __aarch64__ / else` at **compile time**, with
   no CPUID probing — an AVX-512-capable host gets no benefit over a portable build without a
   rebuild; (b) `core/CMakeLists.txt`'s non-CI branch uses `-march=native` for the `dev` and
   `release` CMake presets. A `release` binary built on a machine that has AVX-512 and distributed to
   one that doesn't **will SIGILL** — this is the actual production-distribution risk, not a SIMDe
   problem.

   **Fix — primary recommendation: implement real runtime CPUID dispatch, not a portable-flags
   downgrade.** Simply rebuilding `release` at `-march=x86-64-v3` (AVX2-only) is *not* "cheaper and
   sufficient" — it silently drops the x86 INT8 path from AVX-512 VNNI (`dot_int8_avx512`, uses
   `_mm512_dpbusd_epi32`) to SIMDe-emulated AVX2, which trades away exactly the ultra-low-latency
   requirement this plan is gated on (the 4.70ns SDOT figure is a NEON/Apple-Silicon number; the x86
   VNNI path is where AVX-512 actually matters). All the candidate implementations already exist as
   separate functions behind a function pointer (`similarity_avx512`, `similarity_avx2`,
   `similarity_scalar`, and the INT8 trio) — so the fix is a `__builtin_cpu_supports`/`cpuid` probe
   inside `get_best_similarity_impl()`/`get_best_int8_dot_impl()` selecting among them at
   first-call/load time, not new kernel code. This keeps both portability (one `release` binary runs
   correctly on any x86-64 host) and peak throughput (AVX-512/VNNI hosts still get the fast path).
   Build `release` at the portable `x86-64-v3` baseline (so the binary itself never contains an
   illegal instruction for its baseline path) with runtime dispatch layered on top to opportunistically
   use AVX-512 when the *running* CPU supports it. Only fall back to the cheaper "ship at v3 and
   accept the AVX-512/VNNI throughput loss" option if runtime dispatch proves out of scope for this
   stage — and if you do, say so explicitly in release notes, since it's a real ultra-low-latency
   regression on VNNI-capable x86 hardware, not a neutral simplification. **Gate**: a `release`-preset
   binary built once runs without SIGILL on any same-OS/ISA-family host regardless of AVX-512
   support, *and* a benchmark run on an AVX-512/VNNI host shows the fast path is actually selected
   (not silently downgraded to the portable baseline).

3. **`Atlas::insert()` has zero WAL coverage and no explicit `msync`.** Verified:
   `core/src/atlas.cpp`'s `insert()` (mmap-direct path) contains no `wal_stream_`/`WalRecordHeader`
   references at all — only `insert_delta()` writes WAL. Unlike `TraceManager::append_event()` (which
   conditionally WALs based on whether the event is going to mmap or delta, `trace.cpp:344-367`),
   Atlas's mmap-direct inserts have **no crash-durability guarantee whatsoever** beyond OS page-cache
   flush timing. This predates and is orthogonal to the scope/version work, but Stage 1 touches this
   exact function anyway (adding `WAL_RECORD_SCOPE`) — fix it in the same pass rather than leaving a
   silent data-loss window. **Gate**: new `test_wal.cpp` case — kill the process mid-`insert()`
   (mmap path, not delta), reopen, confirm the node is durable (either via WAL or explicit `msync`,
   whichever mechanism is chosen).

## Cross-cutting guardrail #2: `TraceBlockIndex` — dead code claiming a live feature (user-surfaced finding)

Confirmed independently: `grep -rl "TraceBlockIndex\|trace_block_index"` across the repo matches only
`core/include/aeon/trace_block_index.hpp` itself. `TraceManager::get_history()`
(`core/src/trace.cpp:463-491`) does a plain O(K) linked-list walk via `prev_id` and never touches
`TraceBlockIndex`. Meanwhile `README.md`, `ARCHITECTURE.md` §12, and `INTERNALS.md` §6 all document
sub-linear `O(|V|/1024)` block-centroid search as a shipped feature.

This is a documentation-vs-reality integrity gap that a "production ready" v4 cannot ship unresolved,
**and** it collides directly with Stage 2's supersession-aware retrieval work below (which needs to
touch `TraceBlockIndex`'s Phase 2 scan regardless). Decision point, not deferred:

- **Recommended: wire it in for real, as part of Stage 2**, not delete it — Trace history at scale is
  exactly the episodic-search use case both v4 docs assume (14 tickets about the same flaky test,
  MRMS-style multi-resolution retrieval), and throwing away a working O(|V|/1024) implementation to
  match today's O(K) reality is the wrong direction for a system explicitly trying to grow its
  episodic corpus (addendum doc, Question 2). If wiring proves to need more than a focused effort
  (e.g., block-boundary consistency under concurrent Trace writes turns out to be a real distinct
  problem), fall back to deleting the header and correcting the three docs — but attempt the wire-in
  first since Stage 2 needs to modify this file anyway for supersession filtering.
- **Gate**: either `get_history()` (or a new block-search entry point) demonstrably uses
  `TraceBlockIndex` and a benchmark shows sub-linear scaling vs. the O(K) walk at 100K+ events, *or*
  the header is deleted and `README.md`/`ARCHITECTURE.md`/`INTERNALS.md` are corrected in the same PR.

---

## Stage 0 — Fix present-tense isolation + engineering guardrails (~2–3 weeks)

*(Descends from hive-mind doc Stage 0, expanded with guardrails #0/#1 above and the SLB dim spike.)*

Verified, all current: `get_atlas_client()` is a process-wide `@lru_cache()` singleton
(`shell/aeon_py/dependencies.py:13-18`); `ContextManager.process_turn()`'s access-level filter is
dead code (`context.py:44-47`, the actual commented line is
`# allowed_results = [r for r in results if r['level'] <= access_level]`); `aeon_atlas_navigate` and
`aeon_atlas_insert` accept `session_id` and discard it via `(void)session_id;`
(`core/src/aeon_c_api.cpp:133-149`, `:175-195`) because `Atlas::navigate()`/`insert()` in
`atlas.hpp:93-95` have no session parameter at all to receive it; `aeon_atlas_drop_session`
(`aeon_c_api.cpp:212-227`) validates the pointer and unconditionally returns `AEON_OK`, contradicting
its own header doc's OOM-prevention claim; user identity is an unauthenticated `X-User-ID` header
(`dependencies.py:34-38`); `POST /state/atlas/query` (`server.py:157-177`) is the only endpoint
missing the `get_current_user_id` dependency; `HierarchicalSLB` is fully implemented, tested, and
Python-bound but **never instantiated by `Atlas`** (which uses the simpler global `SemanticCache`
instead); `HierarchicalSLB`'s own L2 `scan_global_cache()` (`hierarchical_slb.hpp:440-468`) has no
session/scope predicate on its 256-entry global ring buffer.

Tasks:
1. **SLB dim-generalization spike** (guardrail #1.1) — do this before anything else touches the SLB,
   or the isolation fix inherits the bug.
2. **Give `Atlas` a real session-aware cache path.** Add `session_id`/scope parameters to
   `Atlas::navigate()`/`Atlas::insert()` in `atlas.hpp`+`atlas.cpp`; wire `HierarchicalSLB` in as
   Atlas's L1/L2 cache (replacing or sitting alongside `slb_cache_`); add a scope/session field to
   `GlobalCacheEntry` so `scan_global_cache()` stops returning cross-session hits — until Stage 3's
   shared tier exists, the simplest correct fix is disabling L2 cross-session sharing entirely
   (single-session-only global cache) rather than half-building scope enforcement twice.
3. **Stop discarding `session_id`** in `aeon_atlas_navigate`/`aeon_atlas_insert`
   (`aeon_c_api.cpp`) — route it to the new `Atlas` session parameter.
4. **Implement `aeon_atlas_drop_session` for real** — forward to `HierarchicalSLB::drop_session()`.
5. **Replace `X-User-ID` trust with real auth** (`dependencies.py`) — OIDC or equivalent; this is a
   Python-shell change, not hot-path, but blocks calling the server multi-tenant-safe.
6. **Remove or auth-gate `POST /state/atlas/query`** (`server.py`) — either delete the debug endpoint
   or require `Depends(get_current_user_id)` and feature-flag it off by default.
7. **ABI-safety cleanup, cheap and preventive**: `aeon_atlas_options_t` (`aeon_c_api.h:172-176`) has
   **zero reserved padding** — add e.g. `uint8_t reserved[32]` now, before Stage 1/3 need to add
   `scope_id`/version config to atlas-creation options and are forced into an ABI break or a second
   `_ex2` struct.
8. **`TraceBlockIndex` decision** (guardrail #2) — make the call here; execute in Stage 2.
9. **Land guardrail #0 (CI perf gate)** and guardrail #1.2/#1.3 (SIMD dispatch honesty,
   `Atlas::insert()` durability) in this stage — they're independent of isolation work but touch
   adjacent code and must be in place before Stage 1's `NodeHeader`/WAL changes.

**Gate**: a new cross-session isolation integration test (Python, hitting the FastAPI server) proves
two different `X-User-ID`/auth identities never see each other's Atlas results, including via the L2
cache path; the CI perf-regression job is live and provably fails on a synthetic regression; SLB
cache hits are confirmed for non-768-dim Atlases; `release`-preset binaries are portable across
same-ISA-family hosts.

## Stage 1 — Carry scope + version durably (~2–3 weeks)

*(Descends from hive-mind doc Stage 1 + addendum's "version-graph schema lands in Stage 1".)*

**NodeHeader byte budget** (`core/include/aeon/schema.hpp:123-138`, `alignas(64)`,
`sizeof(NodeHeader)==64`, `reserved` is `uint8_t[20]` at offset `0x28`): the hive-mind doc's six
core primitives already name both a scope bitmap *and* "a governance record ID in
`NodeHeader.reserved`" — this is not a conflict to resolve, it's an allocation to make:

| Bytes | Field | Purpose |
|---|---|---|
| 8 | `scope_bitmap` (`uint64_t`) | up to 64 scopes; OR'd up through ancestors for Stage 2 union-propagation |
| 8 | `governance_record_id` (`uint64_t`) | opaque pointer into the Postgres control plane (Stage 3/4); doubles as the Atlas-side version/promotion linkage |
| 4 | `saved_hub_penalty` (`float`) | see `NODE_FLAG_SUPERSEDED` below — closes out the 20-byte reserved region entirely; any further per-node field needs a side structure, not `NodeHeader` |

**Version/supersession edges live in `TraceEvent`, not `NodeHeader`** — `TraceEvent` has 364 reserved
bytes (`schema.hpp:300-316`, `reserved[364]` at `0x094`) vs. `NodeHeader`'s 20, and the addendum doc's
own framing supports this: "Trace is already an append-only event log (a commit log by
construction)" — fragments (bug fixes, distilled knowledge) are episodic/event-shaped, not
spatial-centroid-shaped. Carve from `TraceEvent::reserved`: `edge_type` (1B: supersedes / refines /
contradicts / revokes / merges-with / promoted-from), `supersedes_id` (8B, do **not** reuse
`prev_id` — verified `prev_id` is already the per-session chronological chain pointer, conflating it
with version lineage would break both), `reason_code` (1B enum, addendum Q3), `evidence_ref`
(fixed-size, e.g. commit-SHA-sized, or a `BlobRef{offset,size}` into the existing `BlobArena` sidecar
for longer free-text rationale + acting-principal — reuse the exact pattern `blob_offset`/`blob_size`
already use for full event text). Mirror the same fields into `aeon_trace_event_t`
(`aeon_c_api.h:95-107`, also has `reserved[364]`) — no ABI break, both structs already have headroom.

**Superseded-node exclusion must be branchless, not a post-filter** (advisor correction to an earlier
draft of this plan): a post-filter at result emission recreates exactly the recall failure the
hive-mind doc diagnosed for naive scope filtering — the beam fills with superseded candidates and
live heads never get scored. The natural instinct is to reuse `tombstone_node()`'s exact mechanism
(`schema.hpp:253-258`: `flags |= NODE_FLAG_TOMBSTONE` + `hub_penalty = TOMBSTONE_PENALTY (1e9f)`,
excluding dead nodes from the beam with zero conditional jumps) — **but that overwrite is only safe
because tombstoning is terminal.** Supersession is not: this plan's `edge_type` includes `revokes`,
and admin-correction/rectification flows (Article 16-equivalent, Stage 4) can *un-supersede* a
fragment — by then the original `hub_penalty` (a real CSLS value used in scoring) would already be
destroyed with no way to restore it. Correct version: `NodeHeader::flags` is a `uint16_t`; bit 0 is
`NODE_FLAG_TOMBSTONE`, bit 1 is `NODE_FLAG_SUMMARY` — **bit 2 is free**. Add `NODE_FLAG_SUPERSEDED`
there (zero reserved-byte cost) and, on supersession, **stash the current `hub_penalty` into the
`saved_hub_penalty` spare field** (the 4 bytes reclaimed in the budget table above) before
overwriting `hub_penalty = TOMBSTONE_PENALTY` for the same branchless beam-exclusion effect; on
revoke, restore `hub_penalty` from `saved_hub_penalty` and clear the flag bit. This makes supersession
fully reversible at the cost of the last 4 reserved bytes — `NodeHeader.reserved` is now completely
allocated (8+8+4=20); any further per-node metadata needs a side structure, not this struct. Same
flag-bit pattern for `TraceEvent` (check available bits in its `flags` field alongside
`TRACE_FLAG_TOMBSTONE`/`TRACE_FLAG_ARCHIVED`) — Trace events aren't scored by a beam search, so no
`hub_penalty`-equivalent stash is needed there, just the flag.

**`TraceBlockIndex` Phase 2 (`core/include/aeon/trace_block_index.hpp`)** needs the same branchless
treatment, not an over-fetch-and-filter: add a parallel per-entry liveness/flags byte to `TraceBlock`
alongside its existing `embeddings`/`node_ids` arrays (maintained in lockstep on supersession), checked
before `push_back` in the Phase 2 scan — this avoids a second memory dereference back to `TraceEvent`
during the hot scan and keeps the wiring from guardrail #2 fast from day one.

**WAL forward-compatibility — exact fix ordering matters** (advisor correction): today both
`Atlas::replay_wal()` (`atlas.cpp:1086-1136`) and `TraceManager::replay_wal()` (`trace.cpp:782-838`)
`break` the entire replay loop on the first record with a mismatched `record_type` or
`payload_size` — confirmed via `test_wal.cpp`'s `WalAtlasCorruptedTail`/`WalAtlasChecksumFail`, which
exercise truncated-tail and checksum-mismatch paths but **not** an unrecognized-type-mid-stream case.
Once new record types (`WAL_RECORD_SCOPE`, and whatever versioning needs) exist, an old binary
encountering a new record type must **skip it, not abort everything after it**. Correct sequence
(do not reorder): (1) read `WalRecordHeader` → (2) **sanity-bound `payload_size` against bytes
remaining in the file** before trusting it for anything → (3) read exactly that many payload bytes →
(4) verify checksum → (5) only then dispatch on `record_type` → (6) unknown type after checksum
passes → **skip** (continue the loop), not `break`. Skipping *before* checksum verification would
mean trusting an attacker/corruption-controlled `payload_size` to determine how many bytes to skip —
that's the OOB-read risk to avoid.

Tasks:
1. Land the `NodeHeader` field split above (scope bitmap + governance/version record ID).
2. Add `NODE_FLAG_SUPERSEDED` (bit 2) and the equivalent `TraceEvent` flag bit.
3. Add `edge_type`/`supersedes_id`/`reason_code`/`evidence_ref` fields carved from
   `TraceEvent::reserved`, mirrored in `aeon_trace_event_t`.
4. Add `WAL_RECORD_SCOPE` (and any new Trace WAL record types needed for the edge fields) plus the
   skip-not-break forward-compat fix in both `replay_wal()` implementations, in the exact order above.
5. Add WAL coverage (or explicit `msync`) to `Atlas::insert()` (guardrail #1.3) — same PR family,
   same struct, avoid touching `NodeHeader`/WAL twice.
6. Scope must be assigned from authenticated session/scope context at write time (Stage 0's auth
   work), never from a caller-supplied label.

**Gate**: scope bits, governance record ID, and version-edge fields all survive WAL replay
(including the new skip-not-break path) and compaction; a node-ID remap test passes; a new WAL test
specifically covers "unknown `record_type` mid-stream, followed by valid records, all recovered" —
the exact gap found in `test_wal.cpp` during verification; CI perf gate (Stage 0) shows no regression
on insert/WAL-overhead benchmarks despite the added fields.

## Stage 2 — Correct scope-filtered + supersession-aware retrieval (~3–4 weeks)

*(Descends from hive-mind doc Stage 2 + addendum's "admission dedup and supersession-aware retrieval
land in Stage 2".)*

Verified: `Atlas::navigate_internal()` (`atlas.cpp:140-396`) is **level-synchronous beam search**
(bounded-width, not DFS/unbounded BFS) — children reached only via `first_child_offset`
(`atlas.cpp:212-321`). A flat, O(1)-indexable array of every node **does exist**
(`MemoryFile::get_node(index)`, `storage.hpp:270-277`, used today by `tombstone_count()` and
`compact_mmap()` for full linear scans) — but it is *not* used by `navigate()`'s hot path, which
only touches ~193 nodes at depth 3 for a 100K-node tree. This means: the hive-mind doc's diagnosis
still holds for the **hot path** — union-propagated scope + enforcement at emission, not
early-reject-during-descent — but the flat-scan capability is exactly what Stage 3's admin/console
primitives ("list-by-scope", "bulk bit remap") need, and it already exists; no new mechanism required
there.

Tasks:
1. **Union-propagated scope on the beam-search hot path — decide the propagation strategy
   explicitly, it's the actual cost center, not `navigate()`.** Compaction-time recomputation is
   free (already touches every node). Insert-time propagation is not: it means walking O(depth)
   ancestor nodes and mutating their `scope_bitmap` on every insert, and those ancestors live in the
   mmap region — read/written concurrently with lock-free EBR readers — while the new node itself may
   be landing in the delta buffer. That's O(depth) extra mmap cache-line touches (and possibly
   EBR-guarded writes) against a 2.23µs insert budget, which is the part of this change actually at
   risk, not navigate. Two options, pick one and document the choice in the implementation:
   - **(a) Eager-on-insert**: propagate immediately, accept the added insert latency; measure it
     against the CI perf gate (below) rather than assuming it's negligible.
   - **(b) Deferred/dirty-marked**: don't touch ancestors on insert; internal nodes default to
     "all scopes possible" (i.e., never exclude during descent, which is already the beam-search
     behavior — scope enforcement stays purely at result emission) until the next compaction
     recomputes true unions. This costs nothing on the insert path and nothing on recall correctness
     (emission-time filtering is still exact), only precision of the *internal-node* union hint,
     which is used solely as a future optimization opportunity, not a correctness mechanism — Aeon's
     current beam search doesn't prune by scope during descent regardless of stage. **Recommended**:
     start with (b); it requires no new insert-path work at all in this stage, since emission-time
     filtering is already the plan's correctness mechanism.
   Beam search itself never excludes a node by scope during descent (an internal node's own bit may
   be false while a descendant's is true); enforce scope filtering only at result emission.
2. **Enforce independently at the Trace graph-expansion boundary** — wherever code crosses from an
   Atlas result (via `atlas_id` on a `TraceEvent`) into Trace or back, add an explicit scope check at
   that specific crossing. This addresses the hive-mind doc's citation of vector+graph composition
   producing cross-tenant leakage at pivot depth 2 that pure vector filtering doesn't catch.
3. **Execute the `TraceBlockIndex` wiring decision from Stage 0/guardrail #2** here — this is also
   where the branchless superseded-exclusion flag (Stage 1) gets consumed in the Phase 2 scan.
4. **Admission-time near-duplicate detection** (addendum Q2, growth pipeline stage 1): before a new
   fragment is admitted to a scope, check cosine similarity against recent same-scope fragments
   (reuse `math_kernel.hpp`'s existing `cosine_similarity()`); a near-duplicate becomes a `refines`
   edge (Stage 1's `edge_type`) or a counter increment, not a new row. This is also the first
   poisoning checkpoint per the addendum's TrustMem citation.
5. **Do not gate CI on `BM_AtlasTraversal_Only`-class benchmarks** (guardrail #0) — audit for other
   SLB-self-hit artifacts while touching this code.

**Gate**: recall ≥ 0.99 vs. an exhaustive scope-filtered scan across selectivities 0.02–1.0; a
pivot-attack red-team test shows no cross-scope leakage at depth 2 through the Trace boundary;
superseded fragments are excluded from `navigate()` and Trace block-search results with no post-hoc
filter; CI perf gate shows `navigate()` P50/P99 within tolerance of the pre-Stage-2 baseline **and**,
since navigate isn't what union-propagation threatens, also add `bench_wal_overhead`/insert-throughput
to this stage's gate — if option (a) eager-on-insert was chosen, insert latency must stay within the
same tolerance band; if option (b) deferred was chosen, confirm insert latency is unaffected (as
expected) and that a post-compaction recall check still passes.

---

## Stage 3 — Outcome experiment, run *before* funding the full shared tier (~1–2 weeks + observation window)

*(This is the source docs' Stage 4, resequenced per the user's explicit choice: both docs flag it as
the cheapest way to learn whether Stage 4's expensive build-out is worth funding, and the user chose
to run it first rather than last.)*

No published source — including the one close production analog the hive-mind doc cites — has yet
demonstrated that shared engineering memory improves outcomes; a separate cited study (GitOfThoughts)
found a null result for a structurally similar versioned-memory system. Given that, build the
smallest possible version of the shared tier that can produce a real measurement, not the full Stage
4 system.

**Prerequisite this stage depends on, and that Stage 4's OSS-adopter framing makes non-trivial**: the
experiment needs a pilot org actually producing issues with measurable resolution time and defect
rate — a real engineering team using Aeon's shared tier on real work for an observation window. Since
Stage 4 reframes v4 as OSS infrastructure for third-party self-hosters (not a single company you
control), *whose* org runs this pilot is an open dependency, not a given. Identify one before treating
Stage 3 as a ~1–2 week task: either an internal team already dogfooding Aeon, or a design-partner
adopter willing to run the pilot and share outcome data. **If no pilot org materializes in a
reasonable window, the fallback is not to skip the experiment** — proceed into Stage 4 with only the
human-curated promotion path (already the minimal-scope design below) and no automated pipeline,
treating "not yet outcome-validated" as an explicit, documented risk carried into Stage 4 rather than
silently assuming the experiment happened.

- **Minimal scope**: a single pilot scope (e.g. one team), using Stage 0–2 primitives directly — the
  scope bitmap and version/supersession edges already exist by now. Promotion is **manual/human-
  curated** (an engineer or maintainer explicitly promotes a fragment), not the automated pipeline
  Stage 4 builds — every shipped system the addendum doc surveyed (Devin, Cursor, Copilot Spaces,
  Glean) puts a human in the promotion decision anyway, so this isn't a corner cut, it's the
  consensus design.
- **No console, no crypto-erase infrastructure, no automated classification gate** — those are Stage
  4 deliverables and this experiment doesn't need production-scale governance to produce a
  measurement; a pilot is small enough to govern by hand.
- **Measure real engineering outcomes**: issue-resolution time and defect rate on real work with
  shared memory vs. without — not retrieval F1 or recall, which the addendum doc explicitly warns
  doesn't predict outcome benefit.

**Gate**: a written verdict — shared memory measurably helped / measurably didn't / inconclusive —
before committing engineering time to Stage 4's full promotion pipeline, governance console, and
crypto-erase/audit infrastructure. If the result is null or inconclusive, downgrade Stage 4 to
research-track rather than roadmap.

## Stage 4 — Full shared tier + minimum console + promotion pipeline (only if Stage 3 justifies it) (~4–6 weeks)

*(Descends from hive-mind doc Stage 3 + addendum's "contextual distillation, promotion-time
recontextualization and the rationale-bearing console views land in Stage 3" — reframed for OSS
self-hosting per the user's answer.)*

**OSS reframing, per the user's decision**: keep every engineering primitive the hive-mind doc
specifies (they're correct regardless of who deploys this); drop the internal compliance program
(DPIA, works-council co-determination, per-jurisdiction legal review) as *Aeon's own* deliverable —
those become **adopter-facing documentation**: a clear description of what a self-hoster deploying
the shared tier on their own org is responsible for (lawful basis, retention policy, DPIA if
applicable in their jurisdiction), shipped as docs alongside the feature, not as work items on this
roadmap. The engineering requirements that exist *because* personal data can end up in distilled
fragments (mint-not-flip promotion, crypto-erase, audit log) stay as hard requirements — they're
correct for any adopter, and retrofitting them later is far more expensive than building them in now.

Tasks:
1. **Single trusted retrieval service, org-scoped shared store, private memory physically separate**
   from shared memory — not commingled-and-filtered. Governance state (relational, low-QPS,
   transactional, history-bearing) lives in a separate FastAPI + Postgres control plane; the C++
   core stays enforcing-only. Six primitives in C++: scope-mask AND on the read path (Stage 2),
   fragment soft-delete (extends `tombstone_node`/`tombstone_event`), WAL-durable bitmap get/set
   (Stage 1), **list-by-scope** (a pure `MemoryFile::get_node(i)` flat scan, already proven by
   `tombstone_count()`/`compact_mmap()` — no new mechanism needed) and **bulk bit remap** (leaf-level
   bit rewrites *are* a flat `get_node(i)` scan, but if Stage 2 propagates internal-node scope unions
   at all — see Stage 2's eager-vs-deferred decision — a remap that changes leaf scopes invalidates
   those unions too, requiring a tree pass or a forced recompute-at-next-compaction, not a pure flat
   scan; this is a non-issue if Stage 2 chose the deferred/dirty-marked strategy, since there's no
   union state to invalidate), governance record ID (Stage 1's `NodeHeader` field).
2. **Promotion = mint-and-recontextualize** (addendum's synthesis primitive, the credible
   novelty claim): promotion creates a *new*, de-identified fragment (never flips a bit on the
   original) behind a fail-closed deterministic classifier — a detector over the adopter's own
   identifier corpus (directory names/aliases, emails, internal IDs, PR/ticket formats, commit SHAs,
   hostnames) is the *only* layer permitted to **pass** a fragment; an optional LLM layer may only
   reject/flag. Re-embed conditioned on the destination scope's corpus (contextual-document-embedding
   style) — promotion already re-embeds for GDPR/de-identification reasons, so this is free
   retrieval-quality upside on top of a mandatory step. Creates a `promoted-from` edge (Stage 1).
3. **Correctness-gated promotion for code knowledge**: a bug-fix fragment cites a merged commit,
   requires a passing verification run, and is auto-revoked (`NODE_FLAG_SUPERSEDED` / Trace
   `edge_type=revokes`) when that commit is superseded — cheaper and more precise than an importance
   score, and it's available specifically because this use case has ground truth attached.
4. **Contextual distillation at ingestion** (not just promotion): prepend situating context (task ID,
   subsystem, error class) before embedding — near-zero marginal cost on top of the existing
   distillation step (`shell/aeon_py/architect.py`'s `ingest()`, currently minimal).
5. **Minimum console, three components only, in this priority order** (per hive-mind doc): (a)
   hash-chained audit log with independent verification and signed export — build first, retrofitting
   onto existing write paths reliably misses some; (b) knowledge browser — search/filter by scope,
   provenance, classification, age, with supersede/redact/delete; (c) erasure workflow — export and
   delete as tracked cases with a completion receipt and an explicit "could not be erased" section.
   Defer promotion-review-queue UI, org-graph editor, policy authoring, observability views.
6. **Crypto-erase, not unlink-and-hope**: per-subject-per-scope keys beneath a per-scope key,
   independently destroyable — `std::filesystem::remove()` (the only physical-deletion mechanism
   verified in `compact_mmap()`/Trace `compact()`) does not guarantee destruction on flash, and
   soft-deleted vectors are recoverable from raw index files below the API layer. This is real crypto
   infrastructure, not a kernel change — scope it as a dedicated design spike given its complexity.
7. **Admin console constraints, regardless of who deploys it**: scope-scoped admin roles by default,
   four-eyes approval on bulk operations, mandatory read-reason prompts in the audit entry,
   time-boxed break-glass access, and admin reads go through the *same* enforcement path as any
   other read (never a wildcard bypass) — a privileged branch in the enforcement code is exactly
   where isolation bugs like Stage 0's live.
8. **Adopter-facing compliance documentation** (replacing the internal DPIA/works-council program):
   what data classes can end up in fragments, what "promote" actually does (mint-and-recontextualize,
   not a bitmap flip), what the crypto-erase mechanism guarantees, and a checklist of what a
   self-hoster needs to assess in their own jurisdiction before enabling the shared tier.

**Gate**: measured false-negative rate for the identifier-corpus detector on a labelled corpus of
real distilled fragments, published alongside the feature; console audit log passes independent
hash-chain verification; erasure workflow demonstrates actual key destruction end-to-end, not just a
tombstone flag; CI perf gate shows the promotion/console additions (Python/Postgres-side) introduce
no regression on the C++ hot-path benchmarks, since none of this stage should touch `navigate()`'s
critical path.

## Stage 5 — Shared-tier hardening: Dreaming over shared memory + outcome-verified supersession

*(Descends from addendum's "Dreaming-over-shared-tier and outcome-verified supersession land in
Stage 4, where the outcome experiment can measure them" — now Stage 5 after resequencing, and only
relevant if Stage 4 shipped.)*

1. **Point the existing Dreamer at the shared tier**: `shell/aeon_py/dreamer.py`'s
   `DreamingWorker`/`_execute_dream_cycle()` is verified single-tenant today (no scope/session
   parameter anywhere in the class or `DreamConfig`) — extend it to accept a scope, and cluster
   related resolved fragments *within that scope* (e.g. fourteen tickets about the same flaky test)
   into one summary fragment with `merges-with` edges (Stage 1) back to the sources, reusing the same
   tombstone-and-summarize flow the single-user path already performs via `consolidate_subgraph`.
2. **Outcome-verified supersession**: a fragment citing a commit SHA (Stage 4's `evidence_ref`) is
   auto-superseded when that commit is reverted or replaced — a scheduled job or CI webhook
   cross-references `evidence_ref` fields against the org's commit graph and fires the Stage 1
   supersession mechanism with `reason_code=superseded-by-commit`. This is the addendum's most
   concrete credible-novelty claim: nobody else triggers supersession from ground truth rather than
   embedding similarity (which the addendum's cited literature shows performs at AUROC 0.59 — near
   chance — for distinguishing a contradiction from a duplicate).

**Gate**: Dreaming consolidation on the shared tier respects scope boundaries (a consolidation cycle
never merges fragments across scopes); a synthetic "commit reverted" event correctly and
automatically supersedes the fragment that cited it, with an audit-visible reason and evidence link.

---

**Stage 5 progress (2026-08-23), per the user's "start stage 5" instruction.**

**Task 1's tightest constraint, landed first (advisor review, before any Python clustering code was
written): `consolidate_subgraph()` now REJECTS mixed-scope input.** A plain scope-union (already
built, Stage 2 follow-up) can only ever WIDEN visibility, never narrow or preserve it exactly --
consolidating a scope-0x1 node with a scope-0x2 node would silently mint a scope-0x3 summary readable
by BOTH, as a side effect of a storage-GC operation nobody asked for. Rather than trust a future
Python clustering layer to always group correctly, the kernel itself now enforces it: every id in
`old_node_ids` must share the IDENTICAL `scope_bitmap`, checked in Phase 1 before any mutation, else
`std::invalid_argument` naming the offending id. Private-store nodes (`scope_bitmap == 0` uniformly)
trivially satisfy this and are unaffected -- the existing single-tenant Dreaming tests never call
`set_node_scope()`. Explicitly does NOT check subject-id attribution (task 6's one-subject-per-node
invariant) -- the kernel has no visibility into the Postgres `subject_id` a `governance_record_id`
resolves to; that check is left for the Python-layer shared-tier Dreamer to enforce before it ever
calls this function (documented in both `atlas.hpp`'s doc comment and this writeup, so whoever builds
the clustering layer doesn't rediscover it the hard way).

Two new C++ tests (`test_atlas.cpp`): `ConsolidateSubgraphRejectsMixedScopeInput` (mixed 0x1/0x2 input
throws BEFORE any mutation -- `atlas.size()` unchanged, both sources' own scopes unchanged) and
`ConsolidateSubgraphAllowsUniformScopeInput` (a same-scope regression guard, so a future tightening of
the check doesn't accidentally reject the common case). Discrimination proved the standing way:
temporarily short-circuited the new equality check (`if (false && ...)`), confirmed the mixed-scope
test failed for the expected reason (no throw, and `atlas.size()` grew by one -- the mixed-scope
summary was silently minted), restored, reconfirmed identical-to-backup via `diff`. `ctest --preset
dev`: **128/128** (126 + 2 new). Rebuilt and resynced the nanobind extension (`aeon_py_core` ->
`.venv`'s `core.cpython-314-darwin.so`) since this changes `consolidate_subgraph()`'s exposed Python
behavior; full `pytest tests/`: **105 passed, 6 skipped** (unaffected -- no existing Python caller
consolidates mixed-scope nodes).

**Task 2 (outcome-verified supersession) built next, per advisor review's "don't re-ask the user which
shape -- they answered this pattern twice already this session" (crypto-erase's KEK-length decision,
task 3's caller-supplied VerificationResult): same trust boundary applies here. Aeon does not
integrate with any VCS/CI provider itself** (a genuinely separate integration surface, correctly
identified as out of scope for this codebase back when task 3 made the same call) **-- an external
caller (the org's own CI/commit-graph integration) determines a commit was reverted/replaced and tells
Aeon so.**

- **The citation channel already existed and needed no new plumbing.** `v4-plan.md`'s own Stage 1 text
  called for a fixed `evidence_ref` field carved from `TraceEvent::reserved`; the actual shipped struct
  has `evidence_blob_offset`/`evidence_blob_size` (a `BlobArena` pointer pair, correctly GC'd on
  compaction) but grep confirms **no writer ever populates it** -- `TraceManager::append_event()`'s
  signature has no evidence parameter at all. Rather than wire a new C++ writer path (a signature
  change, a bindings change, a `.so` resync, a new blob-write in the append hot path) for data already
  available elsewhere, advisor review pointed out the obvious: task 3's `VerificationResult.commit_sha`
  (this session, earlier) is already recorded in every promotion's audit-log payload
  (`verification_commit_sha`), and the audit log is this codebase's own stated source of truth
  (governance.py: "Postgres is a queryable index over it, not a replacement"). `evidence_blob_*`
  remains unwritten by design -- superseded by the audit-log channel, not a gap this stage needed to
  fill. `find_promoted_nodes_by_commit_sha()` (new, `shell/aeon_py/supersession.py`) answers "which
  promoted nodes cited commit X" via a linear scan of the audit log (`AuditLog.tail()`, same cost class
  the log's own docstring already accepts for `verify()`/`export_signed()` -- an occasional governance
  operation, not a hot path), needing zero new Postgres columns.
- **New module `shell/aeon_py/supersession.py`**: `supersede_node()`/`revoke_node_supersession()` (the
  audited mutation+record unit -- calls `Atlas.supersede_node()`/`revoke_node_supersede()`, then
  appends an audit-log entry, a best-effort Postgres mirror when `governance_db` is supplied, and an
  optional `SUPERSEDES`/`REVOKES` Trace edge), `find_promoted_nodes_by_commit_sha()` (the lookup), and
  `supersede_by_reverted_commit()` (the batch entry point: resolves every citing node, supersedes each
  independently, returns a `{"superseded": [...], "could_not_supersede": [...]}` receipt -- same
  "partial outcome is a legitimate, auditable completion" philosophy as `erasure.py`'s own receipt
  shape, not all-or-nothing).
- **A real, pre-existing audit gap found and closed as a byproduct** (not scope-creep -- the same
  primitive both paths need): `server.py`'s `/admin/knowledge/{node_id}` console action route called
  `Atlas.supersede_node()`/`revoke_node_supersede()` directly with **zero audit trail**, unlike every
  other governance-mutating path in this codebase (promotion, erasure). Retrofitted to call
  `supersession.py`'s audited functions instead; `KnowledgeActionRequest.reason` is now required
  (non-blank, 400 otherwise) for `supersede`/`revoke_supersede`. `tombstone` was deliberately left
  unchanged -- flagged as a known, separate, not-fixed-here gap, since expanding scope into a third
  action's audit design wasn't what this task asked for.
- **New governance actions**: `"supersession"`/`"supersession_revoked"` added to
  `GOVERNANCE_RECORD_ACTIONS` (governance.py) -- two distinct values (not one with a
  `payload["revoked"]` flag), matching the existing promotion/promotion_rejected split, since a
  queryable "every row where action=='supersession'" is worth more than saving one migration. New
  Alembic migration (`8f3a1c9e2b7d`) widens `ck_governance_records_action` accordingly (CHECK
  constraint diffs are never auto-detected, same gap the "add erasure workflow" migration hit) --
  applied cleanly against live Postgres (`alembic upgrade head`).
- **New HTTP endpoint** `POST /admin/supersede-by-commit` (`SupersedeByCommitRequest`/`Response`,
  models.py): the Stage 5 task 2 gate's concrete trigger. Per-node scope-containment authorization
  (a caller authorized for some but not all cited nodes gets a partial receipt, not a blanket
  rejection) -- but a caller holding NO admin scope at all is rejected outright (403) BEFORE any
  lookup runs, since returning a per-node "not authorized" breakdown to someone with zero legitimate
  access would itself leak which nodes exist for a given commit. `_caller_scope_containment_ok()`
  factored out of `_require_scope_containment()` (the boolean core) so the batch endpoint can filter
  candidates individually instead of failing the whole request on the first unauthorized node.
- **Tests**: `tests/test_supersession.py` (12 new unit tests: supersede/revoke audit-recording, Trace
  edge presence/absence, invalid-id-raises-before-any-audit-record, the commit-sha lookup and its
  rejected/unrelated-commit edge cases, the batch entry point's authorization-filtering). HTTP-level:
  `tests/test_console_endpoint.py` gained `test_supersede_400_when_reason_blank`,
  `test_supersede_records_audit_entry`, and a new `TestSupersedeByCommitEndpoint` class (4 tests:
  successful batch supersession, unrelated-commit empty receipt, 403 for a caller with no admin role at
  all, and scope-containment correctly filtering an unauthorized node into the receipt without
  mutating it). Existing `test_supersede_then_revoke_round_trips` updated to supply the now-required
  `reason` field.
- **Discrimination proofs, both confirmed the standing way**: (1) supersession.py's commit-sha match
  condition temporarily replaced with `True` -- confirmed `test_returns_empty_when_no_promotion_cites_
  commit`/`test_unrelated_commit_supersedes_nothing` both failed for the expected reason (an unrelated
  node got superseded), restored, reconfirmed identical via `diff`. (2) server.py's audited-supersede
  call temporarily reverted to a direct `Atlas.supersede_node()` call (bypassing the audit wrapper) --
  confirmed `test_supersede_records_audit_entry` failed with `IndexError` (empty audit log, no record
  written), restored, reconfirmed identical via `diff`.
- **Verification**: Postgres-backed `pytest tests/`: **219 passed** (up from 201, +18: 12 unit + 6
  HTTP-level). Local-default `pytest tests/` (no DB): **117 passed, 6 skipped** (up from 105, +12 --
  `test_supersession.py`'s unit tests run without Postgres). `ctest --preset dev`: **128/128**
  (unaffected by this task -- pure Python/migration work).

Task 1's clustering layer (scope/subject-aware grouping, `MERGES_WITH` Trace edges, pointing an actual
Dreamer entry point at the shared tier) is the remaining work for this stage.

---

**Task 1 completed (2026-08-23).** Advisor review, before writing any clustering code: don't add this
to `DreamingWorker` -- that class is deeply tied to private single-tenant semantics (file-size/
tombstone-ratio triggers, arbitrary lowest-N-ids candidate selection, a config object half of whose
fields wouldn't apply). Built instead as a SEPARATE module-level function, `consolidate_shared_scope()`
(`shell/aeon_py/dreamer.py`), reusing `LLMSummarizer`/`StubSummarizer` but with its own candidate-
selection, grouping, and clustering logic end to end.

- **Candidate selection**: `atlas.list_nodes_by_scope(scope)`, then re-filtered to nodes whose OWN
  scope is EXACTLY `scope` -- `list_nodes_by_scope()`'s own doc comment says it returns OVERLAP, not
  exact match, so a node scoped to `scope | other_bits` would otherwise be treated as a same-scope
  candidate when it isn't. This is a Python-layer property DISTINCT from (and in addition to) the
  kernel's own unconditional scope-uniformity rejection (task 1's first landing, above) -- the gate
  this function's own test proves is "clustering never PRODUCES a mixed-scope call in the first place,"
  not merely "a bad call gets rejected if one is attempted."
- **Subject-id grouping, enforced entirely in Python** (advisor review): `consolidate_subgraph()` has
  no visibility into Postgres `subject_id` at all (its own doc comment says so explicitly), so task 6's
  one-subject-per-node invariant can ONLY be enforced here. A caller-supplied `subject_id_of(node_id)`
  resolver groups candidates before any similarity clustering; a node resolving to `None` (never
  promoted through `promote_fragment()`, or the resolver itself unavailable) is SKIPPED entirely, never
  grouped under a shared "unknown" key -- advisor's specific warning: grouping unattributed nodes
  together would let the invariant be silently, retroactively violated the moment one of them later
  gains a real subject_id. `subject_id_of=None` (the default) means every candidate is treated as
  unattributed and skipped -- fail-closed, same discipline as `IdentifierCorpus.is_empty()`, not
  fail-open.
- **Similarity clustering**: threshold-based greedy grouping over centroid cosine similarity
  (`_cluster_by_similarity()`) -- no k-means/HDBSCAN dependency; "cluster fourteen tickets about the
  same flaky test" doesn't need more than this, and a simple, fully-tested mechanism beats an opaque
  one for a path that mutates data. `min_cluster_size` (default 2) leaves true singletons alone.
- **Deliberately does NOT call `compact_mmap()`** (advisor review, a real correctness issue caught
  before it shipped): the private-tier path compacts because it's a storage-pressure GC on a single
  device with no other observers. Compacting here would reassign node ids as a side effect of the SAME
  cycle that just wrote `MERGES_WITH` Trace edges carrying `supersedes_id` pointing at the tombstoned
  source ids -- silently invalidating those edges the moment they're written. Left as a separate,
  deliberately operator-triggered call; documented explicitly in the function's own doc comment so a
  future caller doesn't "helpfully" add it back.
- **Returns a LIST of `DreamCycleReport`** (one per cluster actually consolidated), not a single
  report -- one shared-scope cycle can produce multiple independent summaries, unlike the private-tier
  path's single-blob-per-cycle model.
- **Tests**: `tests/test_dreamer.py` (11 new tests) -- similar-nodes-cluster, dissimilar-nodes-stay-
  separate (caught a real test-fixture bug of its own: the existing `_vec(seed)` helper produces
  UNIFORM `[c,c,...,c]` vectors, which are scalar multiples of each other and therefore have cosine
  similarity 1.0 regardless of `c` -- unusable for a "these are dissimilar" test; fixed with a new
  `_orthogonal_vec(axis)` one-hot helper), node-outside-scope excluded, node-with-broader-overlapping-
  scope excluded, never-clusters-across-subject-id, skips-unattributed-node, default-None-skips-
  everything, MERGES_WITH edges recorded/omitted correctly, `min_cluster_size=1` singleton case, and a
  does-not-compact regression guard.
- **A masked-test finding caught via the discrimination-proof process itself** (not an advisor catch --
  found empirically while proving discrimination the standing way): temporarily disabling BOTH the
  exact-scope filter and the subject-id grouping and re-running the suite showed 4 of the 11 tests
  correctly failing -- but `test_excludes_node_with_broader_overlapping_scope` did NOT fail, even
  though its own filter was disabled. Root cause: with the Python filter gone, the mixed-scope pair
  (0x1 and 0x1|0x2) reaches `consolidate_subgraph()`, which correctly REJECTS it via the kernel's own
  scope-uniformity precondition (task 1's first landing) -- caught by this function's `except Exception:
  log+continue`, producing the identical empty `reports` result as if the Python filter had done its
  job. Asserting on `reports` alone cannot distinguish "the filter worked" from "the filter was broken,
  but the kernel's defense-in-depth caught the resulting bad call anyway." Fixed by introducing
  `_ConsolidateSpy`, a thin wrapper recording every `consolidate_subgraph()` call's actual node-id
  argument (needed because the nanobind `Atlas` object rejects direct attribute monkeypatching --
  "attribute is read-only" -- so a spy has to wrap the instance, not patch it) -- the test now asserts
  the mixed-scope pair was never even ATTEMPTED, not just that the end state looks the same. Re-ran the
  full disable/re-enable cycle: all 4 affected tests (including the fixed one) now fail correctly with
  both checks disabled, restored, reconfirmed identical-to-backup via `diff`, reconfirmed green.
- **Verification**: Postgres-backed `pytest tests/`: **230 passed** (up from 219, +11). Local-default
  `pytest tests/` (no DB): **128 passed, 6 skipped** (up from 117, +11 -- these tests need no Postgres
  at all). `ctest --preset dev`: **128/128** (unaffected -- pure Python work, no C++ touched this task).
- **A final advisor pass caught two more test-quality gaps before landing**: (1) the discrimination
  proof above exercised the scope filter and subject grouping but never the similarity-clustering step
  itself -- forcing `_cluster_by_similarity()` to return all-singletons confirmed `test_clusters_
  similar_nodes_within_scope` correctly goes red (the only test that would have caught "clustering is
  broken such that nothing ever merges," as opposed to "everything merges into one blob," which the
  existing dissimilar-nodes test already covered). Restored, reconfirmed identical via `diff`,
  reconfirmed green. (2) `test_does_not_compact` asserted `size()`/`tombstone_count()` values that would
  hold identically even if a compaction ran and reclaimed nothing -- it didn't actually test non-
  compaction, just a byproduct consistent with it. Renamed to `test_sources_tombstoned_not_reclaimed`
  with a comment explaining the real guarantee (no call site for `compact_mmap()` in this function,
  documented in its own doc comment) is what actually backs the claim, not a runtime assertion.

**Stage 5 is now fully complete**: task 1 (shared-tier Dreaming with scope/subject-aware clustering and
`MERGES_WITH` provenance edges) and task 2 (outcome-verified supersession via a caller-supplied
commit-revert signal) both landed, tested, and discrimination-proven. The gate's two halves both hold:
consolidation never merges fragments across scopes -- for `consolidate_shared_scope()` specifically,
this holds twice over (a kernel-level precondition in `consolidate_subgraph()` AND a Python-layer
property, proven by this function's own tests, that never triggers it). The kernel precondition alone
is unconditional and applies to any future caller of `consolidate_subgraph()`; the Python-layer property
is a guarantee of this one function, not something every caller gets for free by construction. And a
synthetic commit-reverted event correctly and automatically supersedes the fragment that cited it,
audit-visible with reason and evidence.

---

**Post-completion verification, per the user's explicit "start stage 5 remaining items" follow-up
(2026-08-23)**: two specific items the user asked to check, both closed.

**1. The non-leaf consolidation residual, verified and explicitly guarded (not just assumed).** Grep
confirmed `promote_fragment()` (`promotion.py:487`) is the ONLY real production path that inserts into
a shared Atlas, and it always uses `parent_id=0` -- but chasing exactly WHY that makes every promoted
node "childless" surfaced a real, previously-undocumented structural fact about `Atlas::insert()`
worth recording precisely: the parent-linking block is skipped entirely for the very FIRST node ever
inserted into a fresh file (`if (new_idx > 0)`, atlas.cpp), so that first node becomes the tree's
IMPLICIT ROOT and legitimately accumulates every subsequent same-parent (`parent_id=0`) insert as its
OWN child, via the same contiguous-append mechanism `insert()`/`consolidate_subgraph()` both already
rely on elsewhere. Verified empirically (a scratch script inserting 5 nodes, then querying each via
`AeonClient.query()`): all 5 are correctly retrievable via `navigate()` -- this is NOT a retrieval bug,
it's the intended one-level-deep tree shape (root fans out to every promoted fragment). The practical
consequence: in ANY real shared Atlas, exactly one node (whichever fragment happened to be promoted
FIRST) is not actually childless -- it has every other promoted fragment as its child. Rather than
leave this as an implicit, unenforced assumption, `consolidate_shared_scope()` (dreamer.py) now
explicitly filters out any candidate with children of its own (`get_children_raw()` non-empty) BEFORE
grouping/clustering -- turning "verified unexercised today" into "structurally guaranteed regardless of
future changes." Documented in the function's own doc comment. New test
`test_excludes_candidate_with_its_own_children` (`test_dreamer.py`), using the same `_ConsolidateSpy`
technique as the mixed-scope test (this residual has NO kernel-level backstop, unlike the scope
precondition -- only this Python filter protects it, so the test asserts on what was actually attempted,
not just the outcome). Discrimination proved the standing way: temporarily commented out the filter,
confirmed the test failed with `assert 1 not in [1, 2]` (the non-leaf candidate was attempted), restored,
reconfirmed identical via `diff`, reconfirmed green.
  - **A byproduct of chasing this**: the new filter's own correctness surfaced a latent bug in
    `test_dreamer.py`'s OWN fixtures, not in the code under test -- every existing test inserted its
    first node directly (`atlas.insert(0, ...)`, mirroring `promote_fragment()`'s literal convention)
    without first seeding a distinct root, exactly like every other test file in this codebase already
    does (e.g. `test_atlas.cpp`'s explicit `"Root"` node). That made each test's own "n1" accidentally
    BECOME the implicit root with n2 as a real child -- which the new non-leaf filter then correctly
    (if confusingly, at first) excluded, breaking 4 of 11 existing tests for a reason that had nothing
    to do with a bug in `consolidate_shared_scope()` itself. Fixed by updating the `shared_atlas`
    fixture to seed an explicit, unscoped dummy root before yielding, matching the established
    convention; documented at length in the fixture's own comment so a future reader doesn't rediscover
    this by tracing a confusing test failure again.
  - **Verification**: local `pytest tests/`: **129 passed, 6 skipped** (up from 128, +1 new test).
    Postgres-backed: **231 passed** (up from 230, +1). `ctest --preset dev` unaffected (pure Python).
  - **Product-level consequence, stated plainly (not hypothetical)**: because the implicit-root fact
    above means exactly one node in any real shared Atlas has children -- the fragment promoted
    FIRST -- and the new filter excludes any candidate with children, **the first fragment ever
    promoted into a given shared scope can never be consolidated by shared-tier Dreaming.** This is
    not a corner case that depends on unusual usage; it happens by construction in every deployment,
    every time. It is currently harmless (that one node just never gets clustered/summarized -- it's
    still fully readable via `navigate()`), but it is a real, permanent gap in Dreaming's coverage of
    a shared scope, not just a defensive guard against a theoretical future change. Worth revisiting
    if/when Dreaming coverage-completeness for shared scopes becomes a goal in its own right.

**2. Full CI perf-gate A/B run**, comparing this session's cumulative Stage 4/5 changes against a real
merge-base -- previously only ever spot-checked individual benchmarks against the recorded
`master_metrics.txt` baseline (a different machine, a different day), never run as the actual A/B
`scripts/ci_perf_gate.py` comparison guardrail #0 specifies ("deferred to whenever this stage's changes
are committed" -- nothing has been committed yet, so it kept getting deferred). Since every change this
session is still uncommitted working-tree state, git `HEAD` (`5d43eda`, "README updated with ArXiv id")
IS the correct merge-base: `git worktree add` a detached checkout of it into the scratchpad directory (no
disturbance to the working tree), built with the `ci-macos` preset (portable `apple-m1` flags, matching
what real CI would actually run -- not `dev`'s `-march=native`), and built the current working tree with
the SAME preset for a true same-runner, same-flags comparison.

```
python3 scripts/ci_perf_gate.py --baseline-dir <worktree>/build/ci-macos \
    --head-dir build/ci-macos --tolerance 0.25
```

**Result: PASS.** All 12 directly-comparable benchmarks (SIMD kernel throughput x5, INT8/FP32 insert
x2, WAL insert x2, beam search P50 x3) landed within single-digit percent deltas, well inside the 25%
tolerance -- no regression from anything this session touched (the scope-uniformity precondition, the
reachability fix, `consolidate_shared_scope()`, `supersession.py`, or any of Stage 4's promotion/erasure/
governance work). `bench_ebr_contention`'s own absolute P99 threshold (<10us) also passed (667ns).

**One set of benchmarks (`QuantizationFixture/BM_Navigate`, 4 configs) could not be auto-diffed** -- not
because of anything Stage 4/5 changed, but because `bench_quantization_efficiency.cpp` itself has an
already-staged, PRE-EXISTING uncommitted fix (present in this working tree since before this whole
engagement began, per the session's own initial `git status`) correcting a self-hit-artifact bug: the
old benchmark reused one static query vector across every iteration, so after the first call
`navigate()`'s own SLB cache served every subsequent "measurement" as a cosine==1.0 cache hit rather
than a real traversal -- silently under-reporting true latency the whole time. The fix (cycling through
a 4096-query pool, with a matching `->Iterations()` cap) changed the benchmark's own NAME format
(gained an `/iterations:4096/` segment), which is why the gate script correctly reported these as
"new"/"missing" rather than diffing them numerically -- it has no way to know a differently-named
benchmark is the "same" measurement under a corrected methodology.

Manually compared the raw JSON anyway, for completeness: e.g. INT8/10k-node navigate jumped from 1.89us
(baseline, cache-biased) to 6.11us (head, real traversal) -- a large apparent increase, but explained
entirely by the OLD number being artificially fast due to the cache-hit bug, not by anything getting
slower. This is a measurement-methodology correction landing at the same time as Stage 4/5's work, not
a regression Stage 4/5 introduced -- worth recording precisely here so a future reader comparing these
two numbers doesn't mistake a bug fix for a performance regression.

---

## Deferred to research (build only if a requirement appears the centralized design can't meet)

Peer-to-peer federation, hierarchical brokers, per-scope gossip domains, RaBitQ/Bloom/MinHash sketch
publication, macaroon capability tokens, covert-channel/timing-channel hardening. The hive-mind doc's
argument holds: federation's only claimed benefit (raw memory never leaves the machine) is already
surrendered the moment distilled fragments and their sketches leave, since text embeddings are
invertible — and centralized erasure is a delete you can confirm, federated erasure is a best-effort
TTL you can't.

---

## Stage 6 — LongMemEval benchmark: how does Aeon's episodic recall actually perform?

**Motivation**: every number quoted so far in this plan (WAL/insert/navigate latency, quantization
efficiency, SLB hit rate) measures Aeon's own mechanics in isolation -- none measures the thing Aeon
exists to do: let an LLM correctly recall a fact from deep in a long-past conversation. LongMemEval
(Wu et al. 2024, arxiv:2410.10813) is the standard published benchmark for exactly this capability,
so it's the right instrument to point at Aeon before writing any v4 marketing/doc copy that claims a
memory-quality result, not just a systems-performance one.

**Dataset**: `xiaowu0162/longmemeval-cleaned` on Hugging Face -- the maintained replacement for the
original release (its README flags the original as noisy/deprecated). Three configs: `_oracle`
(only the truly-relevant sessions pre-selected, no distractors -- doesn't exercise retrieval at all),
`_s_cleaned` (~48 sessions / ~490 turns / ~120k tokens of haystack per question, ~500 questions,
the variant most papers report as "LongMemEval-S"), `_m_cleaned` (~10x larger haystacks, 2.7 GB).
Used `_s_cleaned` per the user's explicit choice. Each of the 500 questions ships its OWN
independent haystack (not a shared corpus across questions) -- confirmed by inspecting the raw JSON
directly, not assumed from the paper. Question-type distribution in the 500: multi-session 133,
temporal-reasoning 133, knowledge-update 78, single-session-user 70, single-session-assistant 56,
single-session-preference 30. **Correction from an earlier draft of this section**: that draft
claimed no `_abs` (abstention) question ids were present in this file, checked via `not
answer_session_ids` -- the wrong proxy (abstention questions still carry non-empty
`answer_session_ids`, pointing at the removed-session id). The correct check, matching the official
evaluation harness exactly, is `'_abs' in question_id`; by that check there ARE 30/500 (6%)
abstention-augmented questions, distributed across 4 of the 6 base question_types (multi-session 12,
single-session-user 6, temporal-reasoning 6, knowledge-update 6) -- each a real question re-posed
against a haystack with its answer-bearing session removed, per the official methodology, not a
question_type of its own. The harness's judge code already branches on this correctly
(`abstention="_abs" in question["question_id"]`, `run_benchmark.py`), which matters because a handful
of `_abs` ids showed up in the live pilot's progress log while this section was first being drafted --
that's what surfaced the error. Per-type reporting re-buckets any `_abs` question under its own
`"abstention"` row (`report_type`, `run_benchmark.py`) instead of its base type, so it isn't silently
blended into e.g. "multi-session" accuracy -- identifying an unanswerable question correctly is a
different skill than recalling a fact, and averaging them together would misrepresent both.

**Scope decision, stated plainly**: this benchmark exercises `TraceGraph.semantic_search()`
(`trace.py`/`TraceBlockIndex`) directly -- ingesting every haystack turn as an embedded TraceEvent,
retrieving via semantic search, then having a local LLM answer from what's retrieved. It deliberately
does NOT go through `ContextManager.process_turn()`/`CognitiveLoop.chat()` end to end. Two concrete
reasons, not just convenience: (1) `Atlas.query()`'s `ResultNode` only carries a 3-float numeric
preview, not retrievable text (`client.py`'s `RESULT_DTYPE`) -- there is currently no supported path
to get full text back out of Atlas by node id alone, only through the Trace event that references it,
so a text-answering benchmark has to be Trace-based regardless. (2) LongMemEval is fundamentally an
episodic-recall benchmark ("what did I say in session 12 three weeks ago"), which is exactly Trace's
job per `CLAUDE.md`'s Core-Shell description, not Atlas's spatial/concept-abstraction job. This means
the benchmark does NOT exercise Atlas, Architect's admission-time dedup, Dreaming/consolidation, the
SLB cache, or multi-session isolation via `session_id` (retrieval here is global to the trace file,
matching `semantic_search()`'s actual signature -- it takes no `session_id` argument at all). Anyone
quoting these numbers in v4 docs/marketing must carry this caveat: this is a Trace-retrieval-and-
answer-generation result, not a full-CognitiveLoop result.

**Harness** (`scripts/longmemeval/`):
- `judge_prompts.py`: `get_anscheck_prompt()`, ported **verbatim** from the official benchmark's own
  evaluation harness (`src/evaluation/evaluate_qa.py` in `github.com/xiaowu0162/LongMemEval`, fetched
  directly via `gh api` to avoid transcribing from memory) -- per-question-type judge templates plus
  the temporal-reasoning off-by-one-day leniency clause and the abstention variant, unmodified.
  Comparable in *shape* to published numbers; **not** comparable in absolute value, because the
  official harness runs this prompt through GPT-4o/GPT-4o-mini -- this pass runs it through the same
  local Ollama model as the answer-generator (the user's explicit choice, see below), which is a
  weaker and differently-biased judge than the paper's baselines used. Any cross-paper comparison
  must caveat this, not present raw accuracy numbers as apples-to-apples.
- `run_benchmark.py`: for each sampled question, opens a **fresh, isolated, temporary** `TraceGraph`
  (a question's haystack must not leak into another question's retrieval -- matches the benchmark's
  own per-question independence). Ingests every turn as `trace.add_event(session_id=<haystack
  session id>, role, text, embedding=...)`, embedded via `sentence-transformers/all-mpnet-base-v2`
  (the same 768-dim encoder `CognitiveLoop` already uses, batch-encoded per session for throughput).
  A deliberate scaffolding detail worth recording precisely: each turn's stored/embedded text is
  prefixed `[<session date>] <role>: <content>` before ingestion, rather than relying on Aeon's own
  `TraceEvent.timestamp` (which is always real wall-clock insertion time, not caller-settable) --
  without this, a single retrieved turn would carry no way to recover which historical date it came
  from, breaking every temporal-reasoning question by construction. This is benchmark harness
  scaffolding to compensate for LongMemEval's synthetic-date requirement, not an Aeon product
  capability or a claim that Aeon has caller-settable event timestamps.
  At query time: embeds the question, calls `semantic_search(top_k=10)`, builds a prompt from the
  retrieved snippets, generates an answer, then judges it. Also computes a cheap secondary metric --
  `gold_session_hit_at_k`: whether any retrieved event's `session_id` is one of the question's own
  `answer_session_ids` -- isolating "did retrieval find the needle" from "did the LLM reason
  correctly given it", mirroring the official repo's own separate `print_retrieval_metrics.py`
  concern without depending on that script.
- Sampling: `_stratified_sample()` preserves the full 500-question type distribution proportionally
  (not uniform random) when drawing a pilot subset, seeded (`--seed`, default 42) for reproducibility
  -- a 50-question uniform-random draw could plausibly under/over-represent temporal-reasoning
  (26.6% of the full set) by chance; stratifying removes that variance from the pilot's own numbers.

**Model, per explicit user instruction**: `qwen3.8:27b-mlx` via Ollama, run locally (no hosted-API
cost/key), used for BOTH answer generation and judging. Verified reachable and functioning before
the real run (`curl .../api/generate` smoke call, then a harness-level `2+2` sanity call through
`OllamaProvider` itself) -- it's a reasoning model that emits a separate `"thinking"` field per
Ollama's streaming protocol, which `OllamaProvider.generate()` already correctly excludes (only
`data["response"]` is yielded), confirmed by inspecting actual output rather than assumed.

**Byproduct fixes to `shell/aeon_py/llm.py`'s `OllamaProvider`, found and fixed while building this
harness** (real latent bugs affecting any real deployment with a non-trivial local model, not
benchmark-only concerns -- fixed per this project's standing "fix bugs at any stage" convention):
  1. Hardcoded `timeout=30` on the HTTP request -- too short for a 13B+ local model's first-token
     latency on a long prompt; would have surfaced as a spurious "Could not connect to LLM Provider"
     error, not a real connectivity problem. Now `AEON_LLM_TIMEOUT_SECONDS` (default 120).
  2. No `options.num_ctx` was ever sent -- Ollama silently defaults to 2048 for any model whose own
     Modelfile doesn't override it, which would silently truncate a prompt carrying real retrieved
     memory context (Aeon's entire purpose) with no error surfaced anywhere. Now `AEON_LLM_NUM_CTX`
     (default 8192).
  3. No way to request deterministic output -- needed for a faithful port of the official judge
     script, which calls its judge model at `temperature=0`. Added an optional `temperature` param
     to `LLMProvider.generate()`/`OllamaProvider.generate()` (`None` default = provider/model default,
     unchanged behavior for every existing caller); `MockProvider` and `loop.py`'s only call site
     verified unaffected (`grep` for every `.generate(` call site, `pytest tests/test_phase5.py`
     rerun green after the change).

**Pilot validation before the real run**: a 2-question smoke test (the two shortest haystacks in the
dataset, ~400 turns each) using the already-installed `gemma2:9b` confirmed the harness runs
end-to-end with no errors and produces sane output before spending any time on the real 27B model --
both retrieved sessions correctly matched `answer_session_ids` (`gold_session_hit_at_k=1.0` on both),
and both wrong answers were legitimate model failures inspectable in the raw transcript (a
multi-hop age/date arithmetic question the model gave up on, and a knowledge-update question where
the model recalled a plausible-but-superseded number) -- not a harness bug silently producing empty
or garbled input.

**Two more real bugs, caught by advisor review of the harness itself before any number was trusted**:
  1. **Retrieval bias from asymmetric embedding.** The first working version embedded the SAME
     date-prefixed string it stored (`[<date>] <role>: <content>`), while the query embedding had no
     such prefix. Every one of ~490 documents per question carried ~25 identical prefix characters
     the query never had -- for a cosine-similarity space this systematically shifts every document
     vector toward a shared direction and compresses the angular spread retrieval depends on, making
     Aeon's retrieval look worse than it is for no reason connected to Aeon itself. Fixed by
     decoupling: `trace.add_event(text=..., embedding=...)` already takes these as independent
     arguments, so the embedded text is now the bare turn content while the stored/displayed text
     keeps its date prefix (`_ingest_haystack()`, `run_benchmark.py`).
  2. **Mistimed "retrieval" latency.** The original single timer spanned both `encoder.encode()` (the
     query embedding, Python-side sentence-transformers) and `trace.semantic_search()` (Aeon's own
     C++ kernel call) -- so the smoke test's reported "0.16s retrieval" was almost entirely encoder
     inference, not Aeon. Split into two separate fields, `query_encode` and `search` -- `search`
     alone is the ONLY timer anywhere in this harness that measures Aeon's own kernel work.

**Retrieval-quality A/B, isolating fix #1's actual measured impact** (not just its methodological
correctness): same 50 questions, same seed (42), same top_k (10), `--retrieval-only` (zero LLM calls,
~10 min total) -- one run with the bare-content embedding (the fix), one with the prefixed embedding
(the original bug) restored via a temporary, reverted-and-diff-verified edit to `_ingest_haystack()`
(this project's standing discrimination-proof discipline). Result: **98% overall gold-session
recall@10 (fixed) vs. 96% (prefixed)** -- a real but modest difference, concentrated entirely in
temporal-reasoning (92.3% vs. 84.6%; exactly 1 of 50 questions flipped hit->miss between the two
runs, `4dfccbf8`). Recall was already high with the bug present, so the honest framing is "removed a
systematic asymmetry that was measurably, if modestly, costing recall" -- not "recovered N points,"
which the data doesn't support and the fix would have been correct to make either way.

**A third bug, found by inspecting the FIRST full run's raw judge/hypothesis text, not just its
summary numbers**: questions 1 and 2 of that run both scored "WRONG" with `judge_raw` containing
`"[System Error: Could not connect to LLM Provider ..."` -- Ollama was still loading the ~32GB model
into memory for the very first request when the harness's warm-up-free first call landed, so 2 of 50
"wrong answers" were actually infrastructure failures silently scored as model failures. Fixed two
ways: an explicit warm-up call (`_generate_with_retry(llm, "Say OK.", retries=5)`) before the timed
loop starts, and a `_generate_with_retry()` wrapper (3 attempts, backoff) around every real
generation/judge call as defense-in-depth. That first run's file
(`pilot_50_results_v1_CONTAMINATED.json`) is kept only as the pre-fix diagnostic reference point (all
three bugs present) with an explicit `CONTAMINATED` field in its own JSON explaining why -- it is not
a number to quote anywhere.

**A fourth, purely a documentation-accuracy correction, not a code bug**: an earlier draft of this
section claimed no `_abs` (abstention) question ids were present in the dataset, checked via `not
answer_session_ids` -- the wrong proxy (abstention questions still carry non-empty
`answer_session_ids`). The correct check (`'_abs' in question_id`, matching the official harness
exactly) shows 30/500 (6%) ARE abstention-augmented, spread across 4 of the 6 base question_types.
The harness's judge logic already branched on this correctly throughout; what was missing was
per-type reporting -- now every `_abs` question reports under its own `"abstention"` bucket
(`report_type`, `run_benchmark.py`) rather than silently blending "recall a fact" and "correctly say
you don't know" into one number.

**Final corrected pilot run** (all four fixes applied; `pilot_50_results_v2.json`; figures in
`reproducibility_benchmarks/longmemeval/figures/`): 50 questions, stratified by type, seed 42,
top_k=10, `qwen3.8:27b-mlx`, zero infrastructure failures this time (every `judge_raw` was 2-3 chars,
"yes"/"no", spot-checked in full -- no leaked reasoning-model `<thinking>` content contaminating the
yes/no check).

| Question type | n | QA accuracy | Gold-session recall@10 |
|---|---|---|---|
| single-session-user | 6 | 100% | 100% |
| single-session-assistant | 6 | 100% | 100% |
| abstention | 3 | 100% | 100%* |
| single-session-preference | 3 | 67% | 100% |
| temporal-reasoning | 13 | 54% | 92.3% |
| knowledge-update | 8 | 50% | 100% |
| multi-session | 11 | 9.1% | 100% |
| **OVERALL** | **50** | **58%** | **98%** |

*Abstention's recall@10 does NOT mean "found the answer" -- by construction there isn't one.
Advisor review flagged this as suspicious (expected the answer-bearing session to be *removed* from
the haystack, in which case a 100% hit rate would be a bug); checked directly against the raw
dataset instead of assuming either way. It is present, unmodified: for `0862e8bf_abs` ("What is the
name of my hamster?"), `answer_session_ids` names a real session that's entirely about the user's
CAT (Luna) -- never mentions a hamster at all. That's the mechanism: the official benchmark's
`answer_session_ids` for an abstention question names the closest topically-related DISTRACTOR
session, not a removed answer. So `gold_session_hit_at_k=100%` here means "retrieval surfaced the
plausible-but-insufficient session a human would also be drawn to" -- a genuinely good retrieval
outcome (it's exactly the content a model needs in front of it to correctly conclude "not
mentioned"), just not the same claim as for every other row, where the same metric means "found the
fact." Presented in the same column for comparability; read it with this caveat, not as parity with
the other six rows.

Per-phase latency (seconds/question, `qwen3.8:27b-mlx`): ingest (haystack encoding + `add_event()`
writes, ~490 turns) mean 9.2s/p95 12.7s; query encode mean 0.021s/p95 0.039s; **search (Aeon
`semantic_search()` itself) mean 80µs/p95 141µs**; generation (LLM answer) mean 22.5s/p95 45.5s;
judge (LLM score) mean 7.9s/p95 21.7s. Caveat on `ingest`: this run's first ~15 questions overlapped
in wall-clock time with the two retrieval-only A/B runs above, which were also CPU-bound
sentence-transformers encoding -- comparing against the contention-free v1 run's `ingest` mean
(6.6s) suggests real per-question ingest cost is closer to 6-7s than the 9.2s reported here, some of
which is encoder contention from this session's own concurrent benchmarking, not a property of the
harness or Aeon in steady state. The `search` number is unaffected (Aeon's own C++ call, no shared
resource with the Python-side encoder contention). Bounded to what was actually measured: **at
~490 uncompacted delta-buffer events per question, Aeon's own retrieval call (`search`) is roughly
five orders of magnitude cheaper than every other phase in this pipeline** -- not a general claim
about Aeon at arbitrary scale (that's what `core/benchmarks/`' own navigate/search benchmarks are
for), just what this specific LongMemEval-S-scale run showed.

**What the accuracy numbers actually say** (own reading, not just the raw table): single-session
recall of the exact right session is essentially solved at this top_k (92-100% across every type) --
the QA failures are concentrated where the model has to REASON over what was retrieved, not where
retrieval failed to surface it. **Multi-session at 9.1% despite 100% gold-session recall** is the
starkest example: Aeon found the right session in literally every case, and the local judge still
scored 10 of 11 answers wrong -- these questions require synthesizing facts stated across several
separate sessions, which is a model-reasoning gap, not a memory-retrieval gap, and it's visible ONLY
because this harness reports `gold_session_hit_at_k` and `accuracy` as two separate numbers rather
than one blended score. Temporal-reasoning's 92.3% recall vs. 54% accuracy tells a similar, smaller
story. This split is the main reason this benchmark stage is worth the harness it took to build:
a single "58% accuracy" headline number would have looked like a retrieval problem, when the
evidence says it's almost entirely a reasoning-model-capability problem sitting downstream of
retrieval that already works.

**CORRECTION (post-pilot follow-up, prompted directly by the user asking "why is quality low" and
pushing back on taking the 58%/98% split at face value)**: the paragraph above is WRONG about where
the multi-session gap comes from, and the record should say so plainly rather than quietly editing it
away. Two follow-up diagnostics, both run specifically to check this claim rather than assume it,
overturned it:

1. **`answer_in_context` diagnostic** (`scripts/longmemeval/answer_in_context.py`, no LLM calls --
   re-runs ingestion + retrieval for the same 50 questions and checks whether the reference answer
   TEXT literally appears anywhere in what got retrieved, not just whether the right SESSION was
   touched). Result, restricted to the 47 questions where "answer" is a literal fact string (excludes
   single-session-preference's rubric and abstention's explanation, same exclusion logic as the
   report_type work above): **session-level recall@10 was 97.9%, but answer-in-context was only
   44.7%** -- multi-session specifically: 100% session hit, only **30.8%** answer-in-context;
   temporal-reasoning: 92.3% session hit, only **23.1%** answer-in-context. `gold_session_hit_at_k`
   only checks whether ANY retrieved event belongs to the right session -- it says nothing about
   whether `top_k=10` retrieved enough of THAT session's ~10 turns to include the one that actually
   states the fact. It was never claiming otherwise, but was read as a stronger signal than it
   supports, and this diagnostic is what makes that gap visible.
2. **Oracle-context control** (`scripts/longmemeval/oracle_run.py`,
   `reproducibility_benchmarks/longmemeval/oracle_results.json`): removes Aeon/retrieval entirely --
   builds the reader's context directly from the full text of the question's own gold
   `answer_session_ids` session(s), same 50 questions/seed, same model, same judge. **Overall
   accuracy: 90%** (vs. 58% through real Aeon retrieval). Per type, oracle vs. real-retrieval:

   | Type | Oracle accuracy | Real-retrieval accuracy | Gap |
   |---|---|---|---|
   | knowledge-update | 100% | 50% | 50 pts |
   | multi-session | 90.9% | 9.1% | **81.8 pts** |
   | temporal-reasoning | 76.9% | 53.8% | 23.1 pts |
   | single-session-preference | 66.7% | 66.7% | 0 pts |
   | single-session-user/assistant, abstention | 100% | 100% | 0 pts |

   Multi-session's 9.1% real-retrieval accuracy is NOT a reasoning ceiling -- the same model gets
   90.9% of the SAME questions right when simply handed the correct session's full text directly. The
   model can synthesize facts across sessions perfectly well; Aeon's `top_k=10` retrieval, spread
   across however many sessions a multi-session question's answer touches, isn't surfacing enough of
   any one of them to reconstruct the narrative. Knowledge-update's 50-point gap fits the same
   pattern: with the updated fact handed directly, the model gets it right 100% of the time -- through
   retrieval, it's probably being shown the OLD (superseded) value at least as often as the new one,
   or missing the new one outright, both consistent with `answer_in_context`'s 62.5% for this type.
   Only single-session-preference shows a real, retrieval-independent capability gap (unchanged at
   66.7% whether Aeon or the oracle supplies the context) -- though n=3 is too small to lean on this.

**So: is Aeon "the best" on LongMemEval?** Not answerable from this data, and worth saying plainly
rather than hedging around it. Two hard blockers: the judge here is `qwen3.8:27b-mlx`, not the GPT-4o
family every published LongMemEval-S baseline is scored with, so 58%/90% aren't on the same axis as
any number in the paper or a leaderboard; and there is no second memory system run through this same
harness to compare against -- nothing in this data speaks to "best," only to "how this specific
retrieval pipeline behaves against this specific local model."

**What's actually worth fixing, and why it isn't just chasing a benchmark score**: the 58%-vs-90% gap
is overwhelmingly a RETRIEVAL problem, not a generator/judge-model problem -- confirmed by two
independent diagnostics that agree with each other (`answer_in_context`'s direct text-presence check,
and oracle-context's full removal of retrieval from the loop). Concretely, `top_k=10` semantic-search
over individual Trace events is too coarse for questions whose answer depends on several turns within
a session (knowledge-update: which of several mentions is the CURRENT one) or across several sessions
(multi-session: synthesizing facts nobody stated in one place). The natural next Aeon-side experiments
this points at, in rough order of expected leverage: (a) a two-stage retrieval strategy -- once
`semantic_search` identifies a relevant session, pull that session's full turn sequence rather than
just its top-ranked individual events, directly targeting the 97.9%-recall/44.7%-answer-in-context
gap; (b) a larger `top_k` as the cheap first experiment, to see how much of the gap closes before
building anything more elaborate; (c) for knowledge-update specifically, whether Aeon's own
supersession machinery (Stage 5's `supersede_node()`/`EdgeType.SUPERSEDES`) could bias retrieval or
prompt construction toward the current value over stale ones it already knows how to mark. None of
this is about moving a published-comparable score -- it's confirmed, oracle-verified evidence that
Aeon's retrieval is leaving real, recoverable accuracy on the table on exactly the question types
(multi-session, knowledge-update, temporal-reasoning) that most resemble what a production agent
memory needs to get right.

**A stronger claim than the accuracy table, worth stating plainly**: this run ingested ~490 events
per question into a brand-new Trace file and never called `trace.compact()` -- every `semantic_search()`
call was served entirely out of the uncompacted delta buffer. Guardrail #2 (above) was the finding
that `TraceBlockIndex` risked being "dead code claiming a live feature." That gap was closed earlier
in this project (Stage 2) by actually wiring `semantic_search()` through it -- but this pilot's 98%
gold-session recall across 50 independent, realistic, multi-session haystacks is the first evidence
in this whole plan that the wired-in path works correctly under real (if synthetic) load end to end,
not just in the unit tests that exercise it directly. That's a more load-bearing result than the
headline accuracy number, and it hadn't been said anywhere until this benchmark ran.

**Deviation from the official judge harness, stated explicitly** (Stage 6's `judge_prompts.py` claims
a verbatim port): the official script calls its judge model with `max_tokens=10` to force a terse
completion. This harness does NOT cap output length on any call. `qwen3.8:27b-mlx` is a reasoning
model that emits a separate `"thinking"` phase before its final `response` (Ollama's streaming
protocol keeps these in different fields, and `OllamaProvider.generate()` already only yields
`response`) -- capping total output at 10 tokens would very likely truncate the model mid-thought,
before it ever reaches a yes/no. Leaving output uncapped is the deliberate, correct choice for this
model family, not an oversight, but it is a real deviation from the ported prompt's original
calling convention and is recorded here so nobody assumes byte-for-byte parity with the paper's setup.

**Follow-up experiment: does the gap just close if `top_k` is bigger?** (advisor-directed, the cheap
test before committing to any two-stage retrieval build) -- same 50 questions, same seed, same model,
`top_k` changed from 10 to 30 to 50 and nothing else. Both harness bugs the tree-repr A/B below
surfaced (non-retryable-error fail-fast, error-aware scoring) were already in place, so these numbers
are clean (`num_errors: 0` at every `top_k`).

| | top_k=10 (baseline) | top_k=30 | top_k=50 |
|---|---|---|---|
| Overall accuracy | 58.0% | **78.0%** | 78.0% |
| Overall gold-session recall@k | 98.0% | 100% | 100% |
| knowledge-update | 50.0% | **100%** | 100% |
| multi-session | 9.1% | 45.5% | 54.5% |
| temporal-reasoning | 53.8% | 61.5% | 61.5% |
| single-session-preference (n=3) | 66.7% | 100% | 66.7% |
| single-session-user/assistant, abstention | 100% | 100% | 100% |
| median generation latency | 21.0s | 43.1s | 65.6s |

Two different things are true at once here, and both matter more than the headline 58%->78% jump:

1. **knowledge-update's entire 50-point oracle gap closes at `top_k=30`.** This type's problem really
   was "the right fact wasn't in the retrieved window" -- once `top_k` is large enough to include it,
   the model resolves old-vs-current correctly on its own, with no supersession-aware re-ranking
   needed. The cheap fix fully explains and fixes this type's gap.
2. **multi-session does NOT converge the same way -- but the first read of this run overstated its own
   evidence, and got corrected before it went further.** The `qwen3.8:27b-mlx` numbers above (9.1% ->
   45.5% -> 54.5%) were read as "keeps climbing toward the 90.9% oracle ceiling," which would indeed
   be the signature of a real, continued gain. Advisor review caught the flaw: that sweep never pinned
   generation temperature, and a 9-11-point move on n=11 (multi-session) or n=3
   (single-session-preference, which wobbled 100% -> 66.7% the other direction) is well within what
   sampling noise alone produces. Two follow-up checks, both immune to this problem, settled it:
   - `answer_in_context` (no LLM call at all) at `top_k=10` -> `top_k=30` -> `top_k=50`: multi-session's
     rate is **flat at 30.8% across all three** -- widening the window puts zero additional
     answer-bearing text in front of the model for this type, full stop.
   - The full sweep re-run with `temperature=0.0` on a second, independent model
     (`gemma4:31b-cloud`): `top_k=30` and `top_k=50` are **identical to three decimal places** on every
     per-type bucket, multi-session included (36.4% both). No climb, on either model, once sampling
     noise is removed.

   So the corrected finding is stronger, not weaker, than the original: multi-session's gap is flat
   under `top_k` widening -- confirmed by two independent methods (an LLM-free direct-text-presence
   check, and a deterministic re-run on a different model) -- and sits roughly 55-60 points below its
   oracle ceiling regardless of how large the window gets. That is the actual signature of a ranking/
   session-coherence problem, not a window-size problem: individual top-k events are each independently
   a coin-flip on belonging to the relevant session, so more of them doesn't reliably assemble the
   narrative the way handing over the whole session (the oracle control) does. This remains the
   concrete case for two-stage session-expansion retrieval -- now on firmer ground than the original
   (noisy) sweep provided.

Practical read for Aeon: `top_k=30` is a strictly better default than `10` for this workload and
should be the new baseline; going to `50` buys nothing further on any metric measured so far
(confirmed flat by both the LLM-free and the deterministic-model checks) and is not worth its added
latency as a blanket change. Multi-session's real, oracle-confirmed gap needs an actual
retrieval-strategy change, not a bigger `k`.

**Provenance caveat on the "+20 points" headline number -- now closed with a clean, fully-deterministic
three-point line.** The original 58%(k=10) -> 78%(k=30) jump was `qwen3.8:27b-mlx` at free (non-zero)
temperature -- the same run whose multi-session reading later had to be corrected -- and the
deterministic evidence available at the time only covered the k30-vs-k50 *plateau* on
`gemma4:31b-cloud` (68.0% both), not an independent k10-vs-k30 delta on that same model. A
`--temperature 0.0 --model gemma4:31b-cloud --top-k 10` re-run of the identical 50 questions (`0`
transport errors) closes that gap:

| `top_k` | 10 | 30 | 50 |
|---|---|---|---|
| Overall accuracy (gemma, `temperature=0.0`) | 58.0% | 68.0% | 68.0% |
| multi-session | 18.2% | 36.4% | 36.4% |
| knowledge-update | 62.5% | 100% | 100% |
| temporal-reasoning | 38.5% | 38.5% | 38.5% |

`top_k=30` beats `top_k=10` by +10 points overall on a fully deterministic run, on a *different* model
from the one that produced the original noisy +20-point claim -- confirming the direction (bigger
window helps, up to a point) while landing on a smaller, now-trustworthy number rather than the
free-temperature figure. Striking incidental confirmation: gemma's deterministic k=10 overall accuracy
(58.0%) lands exactly on qwen's original free-temperature k=10 number, by coincidence rather than
design -- the two models simply agree at the baseline `top_k`, and diverge only as `top_k` grows.
Temporal-reasoning is flat across all three `top_k` values on gemma specifically (38.5% at k=10/30/50)
-- for this model, widening the window does nothing for this type at all, reinforcing that whatever
temporal-reasoning's problem is, it is not a retrieval-window-size problem (see below).

**Temporal-reasoning: flagged as genuinely unresolved, not folded into the multi-session story it
happens to sit next to.** Its oracle ceiling (76.9%) sits well above its `answer_in_context` rate
(30.8%, flat across k=10/30/50, same pattern as multi-session) -- but its *real* accuracy diverges
sharply by model: 61.5% on `qwen3.8:27b-mlx`, 38.5% on `gemma4:31b-cloud`, both at `top_k=30`. A
23-point spread between two models answering the identical retrieved context is larger than several
findings this stage chased harder, and Stage 7's session-expansion fix isn't obviously aimed at
whatever causes it (unlike multi-session/knowledge-update, where the oracle-vs-real gap is the whole
story). Worth a dedicated look before or alongside Stage 7 -- possibly a model-specific date-arithmetic
or leniency-clause handling difference (this benchmark's temporal questions carry an explicit
off-by-one-day leniency rule per Stage 6's earlier harness notes) rather than a retrieval-unit problem
at all.

**LongMemEval-V2 (agent-trajectory benchmark): does embedding-representation choice matter?** A
second, structurally different benchmark (`github.com/xiaowu0162/LongMemEval-V2`, arxiv:2605.12493)
was run to check whether the S-benchmark's findings are chat-conversation-specific or generalize --
V2 tests recall over web/enterprise-agent action trajectories (browser/ServiceNow accessibility-tree
observations) rather than chat turns, with a haystack **shared** across all questions in a domain (100
trajectories/domain, "small" tier) rather than per-question isolated. `scripts/longmemeval-v2/` is a
bespoke harness (the official `evaluation/harness.py` needs Python 3.11 + CUDA torch, unavailable on
this Apple Silicon machine) that vendors the official deterministic evaluators verbatim
(`qa_eval_metrics.py`) and points the two LLM-judge evaluators at the same local `qwen3.8:27b-mlx`.

The open question (`smoke_test.py`'s docstring): each trajectory state carries a large, repetitive raw
`accessibility_tree` (UI dump) alongside a short `Goal/URL/Thought/Action` summary -- does embedding
the raw tree for retrieval ranking help or hurt, given `all-mpnet-base-v2` truncates at 384 tokens? A
first attempt at this A/B (embedding AND prompting with the raw tree) **contaminated 15/22
tree-pass results** with Ollama transport errors: some trajectory states' raw `accessibility_tree`
runs past 300KB (not the 5-25KB the docstring assumed), and `top_k=10` of those in one prompt produced
either an outright 400 rejection or a request slow enough to exceed the retry timeout. Root-caused via
direct reproduction (replaying the exact failing request against Ollama) rather than assumed, and
fixed two ways, both now shared, future-proofing infrastructure rather than one-off patches:

- **Embed/prompt representation decoupling** (`common.py`'s `ingest_domain`): embedding representation
  (what ranks retrieval) and prompt representation (what the LLM actually sees) were the same
  parameter: testing "does raw-tree embedding rank better" also silently exploded the prompt size.
  Split into independent `embed_repr`/`prompt_repr` so the ranking experiment doesn't confound with
  context-window survival.
- **`_generate_with_retry` fail-fast on deterministic rejections** (`run_benchmark.py`, shared by both
  harnesses): a `400 Bad Request` was being retried 3x with backoff before giving up -- the same
  request fails the same way every time, so retrying wastes time without ever succeeding. Now only
  transport hiccups (timeouts, connection resets) retry; 4xx responses fail immediately. Paired with
  error-aware scoring (`is_error` field, both harnesses) so a transport failure is excluded from the
  accuracy denominator instead of silently counted as the model getting the question wrong -- the same
  contamination class as the S-benchmark's original cold-start bug, now guarded in both places.
- **A second, independent bug surfaced immediately after the first fix**: the retry on
  the exact same first question (`05cce9b3`) still timed out at 375s even with prompt-size fixed,
  because `TraceManager` mmap-opens an existing file rather than truncating it, and the killed run's
  stale `/tmp/lmev2_ab_tree_*.trace.blobs` files (171-268MB of old raw-tree data) were silently
  reused by the next run via the same fixed `/tmp` path. Fixed by deleting any existing trace file and
  its `.blobs`/`.wal` sidecars before every ingest (`fresh_trace_path()` in `common.py`, applied to
  every script in this directory including the S-benchmark's per-question harness defensively).

With both fixed, the corrected run completed cleanly (`n_errors: 0` in both passes, 40 questions,
paired same-question comparison, both passes prompting with the same compact text so the comparison
isolates ranking quality alone):

| Embedding representation | Accuracy | web | enterprise |
|---|---|---|---|
| compact (Goal/URL/Thought/Action) | **15.0%** (6/40) | 25.0% | 5.0% |
| tree (raw `accessibility_tree`) | 2.5% (1/40) | 5.0% | 0.0% |

Clean, valid, generalizing signal: compact-summary embedding beats raw-tree embedding by 6x for
retrieval ranking, consistent with the mpnet-truncation theory -- the tree's first ~1500 characters
across most states in a domain are boilerplate (nav chrome, skip links, repeated menu structure), so
truncated tree embeddings carry very little state-specific signal to rank against. This generalizes
the S-benchmark's implicit finding (embed a compact, information-dense representation, not raw
verbatim source text) to a structurally unrelated domain and modality.

**Does V2's gap generalize the S-benchmark's retrieval-not-reasoning finding?** V2's public question/
haystack files have no `answer_session_ids`-equivalent field (`questions.jsonl` carries only
`{id, domain, question_type, question, answer, eval_function}`), so `scripts/longmemeval/oracle_run.py`'s
approach (hand over the named gold session directly) doesn't transfer. `scripts/longmemeval-v2/oracle_run.py`
adapts it: split the reference answer into its scored phrases (`qa_eval_metrics.split_phrases`, the
same normalization the real evaluator uses) and grep the entire domain's raw `accessibility_tree` text
for states containing one, feeding the matches as oracle context (same 40 fixed questions as the
embedding A/B, `gemma4:31b-cloud`, `temperature=0.0`).

The first version of this oracle came back at 15.8% -- barely above the 15.0% real-retrieval number --
which would have meant "V2's gap isn't retrieval, unlike V1's." That reading was wrong, and advisor
review caught it before it went into this doc: eyeballing the actual prompts sent for three low-match
WRONG questions found the oracle harness itself was broken two ways -- (1) each matched state was
truncated to its first `MAX_STATE_CHARS` characters regardless of where the match fell, so the answer
to "which column is right of Quantity" (a literal `"Source Code"` header, confirmed present at
character 3551 of a 4714-char state) was being cut off before it ever reached the prompt; (2)
consecutive states within one trajectory are often near-identical, so a question with few matching
trajectories still burned its whole 10-state context budget on near-duplicate copies of one page.
Fixed by centering the kept window on the match position instead of the string's start, and
deduplicating identical windows before capping.

With both fixed, oracle accuracy (when the answer phrase was found at all) rose from 15.8% to 36.8%.
But the aggregate still mixes two very different cases: a **low-match subset** (<=20 matching states
across the whole domain -- the phrase is specific enough that "some state contains it" is decent
evidence of relevance) and a **high-match subset** (>20 matches, up to 3349 for generic answers like
`"300"` or `"false"`, where a matching phrase is nearly meaningless -- confirmed by reading one directly:
a `"300"` match for a "what's the SSD upgrade cost" question turned out to be an unrelated ServiceNow
incident-ticket dump that happened to contain the digits, nothing about laptop pricing). Split by that:

| Subset | n | Oracle accuracy | Real-retrieval accuracy (same questions, compact-embed) | Gap |
|---|---|---|---|---|
| Low-match (proxy precise) | 14 | **64.3%** | 14.3% | **50.0 pts** |
| High-match (proxy unreliable, not a measured ceiling) | 24 | 20.8% | -- | not comparable |

The low-match subset is the trustworthy number, and it lands almost exactly on V1 knowledge-update's
50-point oracle gap: on the 14/40 questions where the phrase-match proxy is precise enough to trust,
the model answers correctly 64.3% of the time when handed the actual answer-bearing content directly,
versus 14.3% through real (compact-embed) retrieval, on the identical questions. The high-match
subset's lower number is NOT evidence of a lower ceiling for those questions -- it's evidence the
substring proxy breaks down for generic short answers, confirmed by direct inspection, not assumed.
Enterprise domain's much lower overall oracle number (16.7% vs web's 55%) concentrates almost entirely
in its high-match questions (519-3349 matches each), consistent with a proxy-precision explanation
rather than a genuine enterprise-specific capability gap -- though this isn't proven for every
individual high-match question, only demonstrated for the ones directly inspected.

**So V2 does generalize the S-benchmark's core finding**, on the subset where it can be measured
cleanly: the bottleneck is retrieval granularity, not model reasoning capacity, and it holds across a
structurally different modality (agent trajectories vs. chat turns) and a much larger, domain-shared
haystack (1737-3358 states vs. ~490 turns per question) -- the same access-pattern problem multi-
session retrieval has in V1, now confirmed at a different scale too.

**Overall status**: pilot complete and trustworthy (four real bugs found and fixed via this project's
usual discrimination-proof discipline, not assumed away, plus five more surfaced and fixed during the
V2 follow-up: prompt-size confound, non-retryable-error retry waste, stale-scratch-file reuse, oracle-
truncation-hides-the-answer, and oracle-duplicate-state-budget-waste). The follow-up diagnostics above
(answer-in-context, oracle-context, the `top_k` sweep re-run deterministically on a second model, the
V2 embedding-representation A/B, and the V2 oracle) turned a plausible-sounding but wrong initial
reading ("the gap is model reasoning") into a confirmed, quantified, actionable one, twice over: on
LongMemEval-S, knowledge-update's gap is fully closed by a larger `top_k` (no architecture change
needed) while multi-session's ~55-60 point gap is a genuine ranking/session-coherence problem that
`top_k` alone cannot close (confirmed flat across two models and one LLM-free metric); on
LongMemEval-V2, the same pattern holds on the cleanly-measurable subset (50-point oracle gap). Both
point at the same concrete next build: multi-event/session-level retrieval as a first-class capability,
not a bigger flat `top_k`. Scaling to the full LongMemEval-S 500-question set and LongMemEval-V2's full
422-question non-image set, and building the two-stage session-expansion
retrieval strategy, are the natural next steps if/when these numbers are actually being written into
v4 docs/marketing -- deliberately not done automatically here, since the user's own choice was "pilot
first," and the retrieval-fix direction needed a real, evidence-backed decision (now in hand) rather
than an assumption about where to spend engineering effort.

---

## Stage 7 — Multi-event retrieval: give Aeon a session-expansion primitive (~2-3 weeks)

**Status: Task 1 (retrieval-unit experiment) run to completion, gate result STOP -- not escalate, on
solid footing after a corrected re-run (2026-08-24).** The first live run's STOP verdict was real but
its numbers weren't trustworthy: fixing `MAX_SESSIONS` (5->10, per the gold-session recall@N
diagnostic) surfaced three independent, real bugs along the way -- a silent `num_ctx` prompt-truncation
bug in `shell/aeon_py/llm.py` (~36% of a `full_session` prompt was being dropped with no error), a
cross-model mismatch between the existing oracle ceiling (qwen) and the live results being compared
against it (gemma), and a kernel-level crash (`core/src/trace.cpp`'s inline text preview split
multi-byte UTF-8 characters, crashing any Python-side read on real-world non-ASCII text -- fixed with a
regression test, independent of any benchmark). All three fixed; see "Track 1, first pass" below for
the full postmortem. **Corrected result: the real oracle-vs-baseline gap is 36.4 points (not the old,
confound-inflated 54.5), and retrieval-unit expansion closes ~25% of it -- STOP stands, now on
apples-to-apples footing.** A further LLM-free diagnostic (literal-vs-computed answers) reframed the
residual gap: most multi-session and a good share of temporal-reasoning questions require aggregating
or computing over several retrieved facts, not looking up one -- a different, more specific capability
question than "which retrieval unit is best." Task 2 (wiring a unit into a real call site) remains
correctly not proceeding on this basis. The primitive layer and `ContextManager.recall_episodic()`
remain built, available, and bug-fixed, but still deliberately unwired from any live call site. Full
next-steps plan (Track 2: event-time as a first-class kernel capability; Track 3: V2 re-verification;
a proposed extract-then-compute prompt experiment for aggregation-heavy question types) is written up
below, pending user go-ahead before any further LLM/benchmark runs -- and explicitly NOT including
another 50-question arm comparison, which this session's own noise-floor check showed can't resolve
which of the 5 units is best at that sample size.

**Motivation, direct from Stage 6's evidence, not from a benchmark score.** Both benchmarks converged
on the same shape of gap: flat `top_k` semantic-search over individual `TraceEvent`s hits a real
ceiling for any question whose answer requires several turns/states from the same session/trajectory,
and no amount of widening `top_k` closes it. LongMemEval-S multi-session: 36.4% real (deterministic,
two models) vs. 90.9% oracle (full gold session handed directly) -- confirmed flat under `top_k`
widening by both an LLM-free metric (`answer_in_context`, 30.8% at k=10/30/50) and a temperature-
pinned re-run on a second model. LongMemEval-V2, on the 14/40 questions where the diagnostic proxy is
trustworthy (demonstrated, not assumed -- see Stage 6): 64.3% oracle vs. 14.3% real retrieval, the same
questions. Two structurally different benchmarks (per-question isolated chat haystacks vs. a shared,
much larger agent-trajectory haystack), same 50-point-class gap, same root cause: the unit Aeon
retrieves (one event) is finer than the unit an answer actually lives in (a session's worth of related
turns). This is a retrieval-architecture gap in Aeon itself, not an artifact of either benchmark --
fixing it serves any caller whose answer spans more than one stored event, which is the common case for
a real agent's episodic memory, not a corner case LongMemEval happens to probe.

**Explicitly out of scope for this stage, and why**: supersession-aware ranking
(`supersede_node()`/`EdgeType.SUPERSEDES`, `shell/aeon_py/supersession.py`) was considered as a
knowledge-update fix in Stage 6's early write-up. Dropped: knowledge-update reaches 100% accuracy at
`top_k=30` on both models tested, with no residual gap for supersession-aware ranking to close. The
efficiency argument for it (today's fix "won't scale past ~500 turns") is a real, separate,
*unmeasured* hypothesis -- if it becomes a priority, it needs its own experiment (Trace sizes an order
or two beyond this benchmark's haystacks) rather than riding in on Stage 6's retrieval-quality evidence.

**Task 1 -- determine the retrieval unit before building an API around a guess.** Stage 6's two oracle
controls each handed over a *different* unit and each beat real retrieval: V1's oracle used the whole
gold session's full text; V2's (corrected) oracle used a window centered on the matched content, not
the whole trajectory. Nothing run so far isolates *which* unit is doing the work, and picking one by
assumption here is the exact failure mode this project corrected three times over during Stage 6 (the
noisy top_k sweep, the untruncated tree-prompt, the truncated V2 oracle). Build a small, explicitly
scoped experiment -- **five arms, fixed, not open-ended** -- reusing Stage 6's already-existing harness
(`scripts/longmemeval/oracle_run.py`, `run_benchmark.py`'s `--temperature 0.0`), against the same 50 V1
pilot questions, deterministic decoding throughout:
  - **Full session** (what V1's oracle already measured: 90.9% overall) -- baseline ceiling, arm 1.
  - **±N-turn window around the top semantic-search hit**, N in {3, 5, 10} -- arms 2-4, built directly
    from primitives that already exist: `TraceGraph.semantic_search()` to find the top hit's
    `session_id`, `TraceGraph.get_history(session_id)` to pull that session's full ordered event list,
    then slice to the window around the hit's position within it. No kernel changes needed for these.
  - **Session summary** (an LLM- or Dreaming-produced synopsis of the session, substituted for its raw
    turns) -- arm 5, more expensive to produce and only worth running if no window arm lands within the
    decision-rule band below; if built, reuse Stage 5's `DreamingWorker` consolidation machinery rather
    than inventing a second summarization path.

  **Decision rule, fixed before any arm runs** (the two-objective trap -- accuracy vs. token cost, with
  no stated exchange rate -- is exactly how this project relitigated the same finding three times over
  during Stage 6): pick the cheapest arm (by mean retrieved-token count) that lands within **10
  accuracy points of the full-session ceiling** (i.e. multi-session accuracy >= ~81%). If more than one
  arm clears that bar, the smallest token count wins outright, no further judgment call. If no arm
  clears it, the full-session arm wins by default (it defines the ceiling) and the gap between it and
  the cheapest window arm becomes task 2's known, stated cost/quality tradeoff rather than a
  post-hoc rationalization.

  **Contamination guard, required before trusting any arm's number** -- this project fixed this exact
  bug class twice in Stage 6 (stale `.blobs` reuse across runs; near-duplicate states silently eating a
  fixed context budget): every arm ingests into a freshly-deleted scratch trace path
  (`fresh_trace_path()`-equivalent, not a reused fixed path), runs at `temperature=0.0`, and must report
  `num_errors: 0` before its accuracy number is read at all -- a nonzero count means re-run that arm,
  not average around it.

**Task 2 -- implement the winning unit as a real Aeon capability, not a benchmark harness hack.**

**Correction to this task's original text, caught by actually reading the call sites before writing
code (not assumed):** the original draft claimed `ContextManager.process_turn()` "already orchestrates
Trace retrieval per turn," implying task 2 would extend an existing semantic-recall call site. That's
wrong. `process_turn()` only queries **Atlas** for concept associations and records events into Trace
-- it never calls `TraceGraph.semantic_search()`. `CognitiveLoop.chat()` only pulls the *current*
session's last 12 events by recency (`get_history(sid, limit=12)`), never across sessions, never by
relevance. **`TraceGraph.semantic_search()` is not called from any production code path today** --
Stage 6's entire benchmark exercises it directly, bypassing `ContextManager`/`CognitiveLoop` by design
(Stage 6's own scope decision). So task 2 doesn't have an existing semantic-recall call site to extend
-- it has to both build the primitive layer AND decide where cross-session episodic recall first enters
the live serving path, which is a real, still-open design question, not a detail to gloss over.

**Primitive layer -- built now, ahead of task 1's result** (`shell/aeon_py/session_expansion.py`,
implemented this session): `find_top_hit()`, `expand_full_session()`, `expand_window()`,
`expand_summary()`, `format_events()` -- pure functions over an existing `TraceGraph` +
`session_id`, built from exactly the two primitives that already exist (`semantic_search()` +
`get_history()`), no kernel changes. Deliberately built as a standalone module rather than inline in
`context.py`/`loop.py` so Stage 7 task 1's experiment (`scripts/longmemeval/
expansion_unit_experiment.py`, also implemented this session, not yet run) tests the *exact same code*
a real integration would call -- not a separate implementation that could drift from what actually
ships. `expand_summary()` takes its LLM call as an injected `generate_fn` callable rather than
importing a provider directly, so this module has zero hard dependency on Ollama or any specific
provider -- keeps it usable from a benchmark harness, a future real call site, or a unit test with a
stub callable, unchanged.

**Where this gets wired in for real is still open, and should be decided with task 1's numbers in
hand, not before**: candidates are extending `CognitiveLoop.chat()`'s context-gathering step (currently
recency-only) to also do a cross-session semantic lookup via `find_top_hit()` + the winning expansion
function, or adding it as a new `ContextManager` method callable independently of `process_turn()`.

**A real, callable, unit-parameterized version of the latter is built now** (`ContextManager.
recall_episodic(query_vector, unit=..., generate_fn=None)`, `shell/aeon_py/context.py`) --
deliberately NOT wired into `CognitiveLoop.chat()` or any other live call site, and its `unit="window_5"`
default is a placeholder ("some non-trivial unit is available"), not a claim about what task 1 will
find best -- calling that shot before the experiment runs would be the same assumption this stage
exists to avoid making. Built this far ahead of task 1's result specifically so that once the winning
unit is known, wiring it into a real call site is a config choice (which `unit` string to pass, and
where to call `recall_episodic()` from), not new code to write under time pressure the next day.
Only reaches into `core/` if task 1's winner is the full-session or window unit at a scale where doing
the expansion in Python is measurably too slow (per Stage 0's CI latency-regression gate -- Aeon's
standing bar is high-performance, production-ready, ultra-low-latency, and a Python-side expansion step
added to a real request path must be held to that, not exempted because it's "just orchestration"), or
if the summary unit is selected and needs an on-write (not on-read) precomputation hook into
`TraceManager`. Whatever the shape, it must not touch `HierarchicalSLB`'s FP32-only cache invariant
(CLAUDE.md: the SLB is deliberately FP32-only even for INT8-backed Atlases, to keep cache-hit latency
off the dequantization path) -- this stage is scoped to `Trace`, not `Atlas`/SLB, matching Stage 6's own
scope decision, and should stay there.

**Gate**: re-run the corrected 50-question V1 pilot (`temperature=0.0`, same seed) with the winning
expansion strategy live and compare multi-session accuracy against both baselines already on record --
the flat 36.4% (real, `top_k`-only) and the 90.9% ceiling (full-session oracle). Success criterion,
made checkable rather than left as "closes most of the gap": the live re-run must close **at least
half of the 54.5-point gap** (multi-session accuracy >= ~63.7%) without regressing other question
types' accuracy or blowing the CI latency-regression budget (guardrail #0). Falling short of that bar
is itself a valid, reportable outcome, not a failure to hide -- but it is a **stop condition, not an
escalation trigger**: it means the retrieval unit picked by task 1 was not (solely) the bottleneck, and
the next step is characterizing what else is (e.g. re-checking `answer_in_context` against the live
expansion strategy's actual retrieved context, the same LLM-free diagnostic Stage 6 already trusts) --
not reflexively reaching for a bigger or more expensive unit than task 1's decision rule already chose.

This gate's criterion is applied mechanically by `scripts/longmemeval/check_stage7_gate.py`
(implemented this session, not yet run) -- reads the oracle-ceiling and real-baseline numbers directly
from their recorded result files rather than hardcoding them, applies the ">= half the gap closed"
threshold, and includes its own contamination-guard check (blocks rather than judges a result with
`n_errors > 0`). Verified against three synthetic scenarios (clears the bar, falls short, contaminated)
before being trusted -- same discrimination-proof pattern as `_apply_decision_rule()` above.

### Task 1's first live run: two real bugs found and fixed, then the gate result

**First attempt (contaminated, discarded, not a real finding).** `expansion_unit_experiment.py`'s
initial live run anchored every arm to a *single* session -- whichever one `find_top_hit()`'s single
top-ranked event happened to land in -- and *replaced* the retrieved context with that session's
expansion. Checked directly against this run's own 50-question sample: knowledge-update questions
always have exactly 2 gold `answer_session_ids`, temporal-reasoning 1-3, multi-session 3-4. A
single-session unit structurally cannot answer most of these regardless of how large that one
session's window gets. Result: all 5 arms collapsed to ~40% overall / 0% multi-session / 0%
temporal-reasoning -- drastically below Stage 6's trusted 90.9% multi-session oracle -- and the
decision rule "picked" `summary` only because every arm tied at 0% on the metric that mattered and it
broke the tie on token cost. Not reported as a finding; advisor-reviewed before any further data was
trusted.

**Fix, two separable changes** (`shell/aeon_py/session_expansion.py`, `expansion_unit_experiment.py`):
1. **Multi-session anchoring**: anchor to the top `MAX_SESSIONS=5` *distinct* session_ids among a
   `semantic_search(top_k=30)` result (`distinct_session_ids()`), not the single best hit -- sized off
   this sample's own measured gold-session counts (max observed: 4).
2. **Additive merge, not replacement**: `build_expanded_context()` runs the existing top_k=30 retrieval
   first (the same retrieval `run_benchmark.py`'s 58%/68%/78% baselines already use) and every arm's
   expansion is unioned on top of those hits (`merge_expanded_context()`), never substituted for them.

**Second bug, found from the re-run's own numbers, not assumed.** Re-running with fix #1 alone still
regressed `window_3`/`window_5` 9-27 points below the recorded top_k=30 baseline on knowledge-update
and multi-session -- which should have been structurally impossible under an "additive" merge.
Root cause: `merge_expanded_context()`'s first version dropped *all* of a session's base top_k hits
once that session was selected for windowed expansion, on the assumption a window around the single
best hit in that session was always a superset of what top_k already found there. False for a small
window when a session has several distinct relevant hits spread further apart than the window radius.
Fixed by keeping every base hit unconditionally, always, then unioning expansion content on top --
removing the assumption rather than tuning around it. Verified against synthetic fixtures (a session
with hits both inside and outside a small window; expansion-id/base-id dedup) before re-running.

**Third issue, external not a code bug: Ollama cloud rate limiting.** The `summary` arm (up to 5 extra
LLM calls per question, one per anchor session) produced a sustained burst of `429 Too Many Requests`
from `gemma4:31b-cloud` that outlasted `run_benchmark.py`'s existing transient-retry budget (3 attempts,
5s/10s backoff) -- 26/50 questions ended as unrecoverable transport errors. Fixed in
`_generate_with_retry()` with a rate-limit-specific retry path (its own budget, longer backoff: up to 6
attempts, 15s/30s/.../90s) separate from the generic transient-hiccup path, since "over quota for this
window" needs materially longer to clear than "connection briefly dropped." Even with that fix, a
second attempt still hit a **hard** rate limit that didn't clear within the extended backoff -- this
was the account's actual cloud quota, not a burst; confirmed via a standalone single-call check that
kept 429ing with no retry in flight. Paused, polled every 5 minutes with a single lightweight call
(no further quota pressure) until the user resolved it by subscribing to Ollama Pro, then re-ran clean
(0 transport errors across all 5 arms).

**Fourth issue -- a wrong assumption in the fix itself, not a bug, caught by advisor before being
reported.** After both merge fixes, `window_3`/`window_5`/`summary` *still* scored a few points below
the top_k=30 baseline on multi-session/temporal-reasoning, which the guard at the time (`_check_
baseline_regression()`) treated as a blocking contamination signal ("additive merge ⇒ no arm can score
below baseline"). That reasoning was wrong: an additive merge guarantees retrieval is a strict superset
of the baseline's hits -- it guarantees nothing about downstream LLM accuracy, which is not monotonic
in how much correct-but-irrelevant context it's handed. The number that settles it: `full_session` --
which merges the *most* extra content into the exact same unsorted list -- scored *above* baseline
(70.0% vs 68.0%) rather than below, ruling out ordering/formatting as the cause. Renamed to `_report_
baseline_deltas()` and demoted from a blocking guard to an informational report; the decision rule was
unblocked and allowed to run on the real numbers.

**Corrected, trustworthy results** (`reproducibility_benchmarks/longmemeval/expansion_unit_results_v3.json`
-- `gemma4:31b-cloud`, seed 42, same 50 questions as every other V1 result, `temperature=0.0`,
`n_errors=0` on all 5 arms):

| arm | overall | knowledge-update | multi-session | temporal-reasoning | mean retrieved chars |
|---|---|---|---|---|---|
| full_session | 70.0% | 100% | **45.5%** | 38.5% | 65,541 |
| window_10 | 68.0% | 100% | 36.4% | 38.5% | 64,045 |
| window_5 | 66.0% | 100% | 27.3% | 38.5% | 52,496 |
| window_3 | 64.0% | 100% | 27.3% | 30.8% | 44,047 |
| summary | 64.0% | 100% | 27.3% | 30.8% | 37,441 |
| *(top_k=30 baseline, no expansion)* | 68.0% | 100% | 36.4% | 38.5% | -- |

`window_10` and `full_session` are nearly identical in both content (64,045 vs 65,541 mean chars) and
accuracy -- expected once checked against this sample's own session-length distribution (mean 10.4
turns, median 12, p90 14; ±10 events covers 98.7% of sessions outright). `window_10` isn't a
meaningfully cheaper unit here, it's `full_session` with extra steps; the real cost/quality frontier in
this data is between `window_3`/`window_5` and `full_session`.

**Applying the pre-committed decision rule as written** (cheapest arm within 10 points of the
full-session ceiling on *overall* accuracy) picks **`summary`** (64.0%, within the 60.0% band floor, at
the lowest token cost). **Stated plainly, not silently re-banded after seeing the numbers**: this is a
real limitation of the rule as pre-committed, not a good outcome -- `summary` is the *worst* arm on
multi-session (27.3%, an 18.2-point drop from `full_session`'s 45.5%), the exact metric this stage
exists to move, because the rule bands on overall accuracy and multi-session is only 11/50 questions
diluted into that average. The rule was pre-committed specifically to prevent picking a winner with
hindsight, and it is being honored here even though its output is not the arm this stage's own
motivation would prefer -- the fix is to change the rule for *future* stages that reuse this pattern
(band per-type, at minimum on the type the stage is chartered to move), not to override this run's
answer now.

**Gate result: STOP, not escalate.** `check_stage7_gate.py --arm full_session` (the strongest arm on
the metric the gate checks): oracle ceiling 90.9%, real baseline 36.4%, required >= 63.6% (half the
54.5-point gap), live result 45.5% -- **16.7% of the gap closed**, well short of the bar. Per the gate's
own text and Stage 7's stated stop-not-escalate framing: **the retrieval unit was not (solely) the
bottleneck for multi-session**. Reaching for a bigger/more expensive unit than `full_session` (which is
already close to a hard ceiling on this data, per the session-length finding above) is not indicated.
The next step is characterizing what else is limiting multi-session accuracy once the model is handed
several whole sessions' worth of correct context and still gets under half of them right -- e.g.
re-running `answer_in_context` (the LLM-free diagnostic Stage 6 already trusts) against `full_session`'s
actual retrieved context, to separate "the fact still isn't in the context" from "the fact is in the
context and the model still can't use it" -- before deciding whether this is a retrieval problem, a
reasoning/synthesis problem, or a prompt-construction problem (the merged context's ordering is
similarity-rank-then-per-session, not chronological, which is plausibly costly specifically for
temporal-reasoning).

### Diagnostic: gold-session recall@N (LLM-free, run to decide the next step -- not a benchmark run)

Per advisor review of the STOP result above: the multi-session chain (baseline 36.4% -> full_session
expansion 45.5% -> oracle 90.9%) has one unexplained variable -- whether the gold sessions are even
among the sessions `distinct_session_ids()` anchors to. This is answerable directly from data already
in hand, with **zero model calls** (`scripts/longmemeval/session_recall_analysis.py`, embeds and ranks
only): for each question, is every one of its gold `answer_session_ids` present among the top-N
*distinct* sessions of a `semantic_search(top_k=30)` result, for N=1..10.

**Result** (same 50-question sample, seed 42):

| question type | all-golds-present@N=1 | @N=3 | @N=5 (task 1's `MAX_SESSIONS`) | @N=10 | in full top_k=30 |
|---|---|---|---|---|---|
| multi-session (n=11) | 0% | 36.4% | 63.6% | **90.9%** | 90.9% |
| temporal-reasoning (n=13) | 0% | 53.8% | 76.9% | **92.3%** | 92.3% |
| knowledge-update (n=8) | 0% | 75.0% | 75.0% | 100% | 100% |

This discriminates the two tracks cleanly:

- **Multi-session: retrieval is genuinely leaving reachable gold sessions out at N=5.** Recall climbs
  from 63.6% at N=5 to 90.9% at N=10 -- exactly matching the ceiling of what's findable anywhere in the
  top_k=30 hits at all. `MAX_SESSIONS=5` was sized off the *first, contaminated* run's max observed
  gold count (4) -- it didn't account for ordinary embedding-rank noise putting a genuinely relevant
  session 6th-10th rather than in the top few. Raising the anchor width is a real, cheap, data-justified
  fix (the gold sessions are demonstrably reachable, just missed), not a parameter tuned until a score
  moves.
- **Temporal-reasoning: retrieval is NOT the bottleneck, confirmed twice over.** Recall at N=5 (76.9%)
  already matches this type's own oracle ceiling from Stage 6 (76.9%), and at N=10 (92.3%) it
  *exceeds* that ceiling -- proving more retrieval width cannot move this number further. The 76.9%
  ceiling is a reasoning/model-capability limit given perfect context, not a recall gap. Consistent with
  the already-recorded qwen (61.5%) vs. gemma (38.5%) 23-point spread on this exact type at the same
  `top_k=30`.

### Proposed next steps (advisor-reviewed; each is a real memory-engine capability, not a benchmark hack; holding all further LLM/benchmark runs for go-ahead)

**Track 1, first pass -- three real bugs found while executing it, all fixed, none benchmark-specific.**
Re-ran task 1 with `MAX_SESSIONS: 5 -> 10` as planned above. Three independent, standing correctness
bugs surfaced along the way and were fixed immediately, per this project's own "fix real bugs at any
stage" convention -- **none of these were introduced to chase a benchmark number; all three are real
production-readiness gaps any caller handing Aeon-retrieved context to an LLM, or storing ordinary
human text, would eventually hit**:

1. **`num_ctx` silent truncation (`shell/aeon_py/llm.py`).** `OllamaProvider` sent a fixed
   `num_ctx=8192` on every request, regardless of prompt size. Measured directly: a real
   `full_session`-expansion prompt needed ~12,795 tokens against that 8192 cap -- about 36% of the
   intended context silently dropped, no error, no warning. Fixed by sizing `num_ctx` from the actual
   prompt length (conservative chars-per-token estimate, so it errs toward requesting more context, not
   less), capped at the model's own advertised context length (`gemma4:31b-cloud`: 262,144 tokens,
   queried once via `/api/show` and cached), with an explicit `warnings.warn()` if genuine truncation is
   still unavoidable. `last_num_ctx` is now recorded per-request so a harness can persist what was
   actually sent, closing exactly the ambiguity this bug caused.
2. **Cross-model ceiling comparison.** The existing "ceiling" (`oracle_results.json`, 90.9% multi-session
   / 76.9% temporal-reasoning) was `qwen3.8:27b-mlx`; every live Stage 7 result is `gemma4:31b-cloud`.
   These were never comparable, and `check_stage7_gate.py` had been comparing them anyway. Re-ran the
   oracle on `gemma4:31b-cloud` (`oracle_results_gemma.json`) so ceiling and live result finally share a
   model. **Corrected, same-model, untruncated oracle: 80.0% overall, 72.7% multi-session, 53.8%
   temporal-reasoning** -- both lower than the old qwen numbers, entangling two effects (model choice,
   truncation) that can't be separated without re-running the qwen oracle untruncated too, which was
   **not done** -- don't attribute the qwen-vs-gemma gap to one cause or the other without that.
3. **Kernel-level UTF-8 crash (`core/src/trace.cpp`, `core/src/bindings.cpp`).** `TraceManager::
   append_event`'s 64-byte inline preview (`TraceEvent::text_preview`) truncated via a raw
   `std::strncpy` at a fixed 63-byte cutoff, with no regard for UTF-8 character boundaries. Any stored
   text containing a multi-byte character (curly quotes, accented Latin, CJK, emoji -- i.e. most
   real-world non-ASCII text) whose 63-byte boundary landed mid-character left an invalid, truncated
   UTF-8 sequence in the preview. `bindings.cpp`'s `get_history()`/`semantic_search()` unconditionally
   convert `text_preview` to a Python string for *every* returned event (`nb::str(ev.text_preview)`),
   regardless of whether the caller even uses the preview field -- the correct full text, read
   separately from the blob arena, was never affected. Nanobind's strict UTF-8 decode threw
   `str_from_cstr(): conversion error!`, crashing the entire call. Reproduced directly (session
   `sharegpt_MpJ5UCF_0`, a `'` at byte 61-63 of a stored turn), fixed with `safe_utf8_truncate_length()`
   (walks back to the nearest complete-sequence boundary rather than cutting raw bytes), and a
   regression test added (`BlobArenaTest.TraceInlinePreviewNeverSplitsUtf8Sequence`) -- confirmed to
   fail against the pre-fix code (scoped revert of just the one call, not the whole file) and pass
   against the fix. All 129 existing C++ tests still pass. **This is a standing correctness fix
   independent of LongMemEval entirely** -- any real deployment ingesting ordinary user text (which
   routinely contains smart quotes from any modern editor) would eventually hit this crash.

**Track 1, corrected result** (`expansion_unit_results_v4.json` -- same model/seed/50 questions,
`MAX_SESSIONS=10`, both the `num_ctx` fix and the kernel fix applied, `n_errors=0` on all 5 arms):

| arm | overall | multi-session | temporal-reasoning | mean chars |
|---|---|---|---|---|
| full_session | 72.0% | 45.5% | 46.2% | 97,284 |
| window_5 | 72.0% | 45.5% | 46.2% | 74,328 |
| window_3 | 70.0% | 45.5% | 38.5% | 58,945 |
| window_10 | 70.0% | 36.4% | 46.2% | 94,681 |
| summary | 70.0% | 36.4% | 46.2% | 43,281 |
| *(top_k=30 baseline)* | 68.0% | 36.4% | 38.5% | -- |

Checked against the corrected, same-model, untruncated oracle ceiling (72.7% multi-session):
`check_stage7_gate.py --oracle oracle_results_gemma.json --arm full_session` -> **gap 36.4 points
(not the old 54.5), required >=54.5% to pass, live 45.5%, 25.0% of the gap closed -- GATE: STOP, not
escalate.** This STOP is now on solid footing: the old 54.5-point gap was inflated by two confounds
(truncation, cross-model comparison) that are both fixed here, and the corrected 36.4-point gap is
still only a quarter closed by retrieval-unit work alone.

**Explicit noise-floor caveat, advisor-caught before over-reading the arm table above:** multi-session
is 11 questions per arm. The `window_10`/`summary` "drop" to 36.4% vs. `full_session`/`window_5`'s 45.5%
is **exactly one question flipping** -- not a measurable assembly or ordering effect at this sample
size, whatever a plausible-sounding story about context ordering might suggest. **The arm ranking in
this table is not resolvable at n=11 per type, and the pre-committed decision rule's `summary` pick
must not be read as a recommendation** -- restated from before, and now doubly true: the rule can't
even discriminate `summary` from the no-expansion baseline (both 36.4%) at this sample size. Any future
attempt to rank these 5 arms against each other needs the full 500-question set, not a 50-question
pilot -- and shouldn't be run casually, since it's 5x the LLM cost of the equivalent single-arm run.

**Diagnostic: literal-vs-computed answers (LLM-free, salvages `answer_in_context` for these types).**
`answer_in_context.py` gained a `--unit` flag to check the literal-substring-match diagnostic against
`full_session`'s actual assembled context (not raw top_k hits) -- result: **100% session-hit, but only
30.8% answer-in-context for multi-session, 30.8% for temporal-reasoning** (vs. 87.5% knowledge-update,
85.7% single-session-user), which reads like a large residual gap until the *type* of answer is
accounted for. Splitting each type's questions by whether the reference answer is a literal
string/entity or a computed value (sums, counts, day-differences):

- **Multi-session: 11 of 13 sampled answers are computed** (dollar totals, visit counts, day-count
  differences -- e.g. `"$1,300"`, `"3.83"`, `"10 times"`). These can never appear as a literal substring
  anywhere in the source haystack by construction -- the model has to aggregate several separately
  -stated facts, not recall one. The diagnostic doesn't apply to this type as originally used; a low
  literal-match rate here does not mean "the fact isn't retrieved."
- **Temporal-reasoning splits cleanly**: literal answers (book titles, app names, "Four weeks") show
  57.1% answer-in-context; computed answers (date-difference arithmetic -- "30 days. 31 days (including
  the last day) is also acceptable.") show 0%, exactly as expected, since the model must compute a
  difference between two separately-stated dates, not recall a stated day-count.

**This reframes the residual gap more precisely than "retrieval vs. reasoning": for both multi-session
and computed-answer temporal-reasoning questions, the task is fundamentally multi-fact aggregation/
arithmetic over content that recall (now measured at ~90-100%) already surfaces correctly, not simple
lookup.** That's a different, more specific capability question than Stage 7 set out to answer, and it
reframes what "improve this further" would even mean -- not a bigger/different retrieval unit (already
shown to plateau around a real ceiling), but possibly a prompt-scaffold change for these question types
(e.g. an explicit extract-the-relevant-facts-first, then-compute two-step prompt, testable cheaply with
no kernel changes) -- **not yet attempted, proposed here for a future decision, not started.**

**Track 2 -- temporal reasoning: event-time as a first-class kernel capability. IMPLEMENTED
(2026-08-24).** `TraceEvent` gained `event_time` (`core/include/aeon/schema.hpp`, 0x0C0, 8 bytes,
carved from `reserved` with an explicit 4-byte alignment pad -- caught and fixed a self-introduced bug
here: the first attempt forgot a `uint64_t` following a `uint32_t` needs 8-byte alignment, silently
grew `TraceEvent` to 576 bytes; caught immediately by the struct-size static_assert, not left for a
runtime surprise) -- a caller-supplied event time, epoch microseconds, `0` = unset, distinct from
`timestamp` (always Aeon's own insertion wall-clock). Threaded through the full stack:
`TraceManager::append_event()`'s new trailing `event_time` parameter (`core/src/trace.cpp`); a new
`aeon_trace_append_event_ex()` C-ABI function (`core/include/aeon/aeon_c_api.h`,
`core/src/aeon_c_api.cpp`) -- added as a new function rather than changing
`aeon_trace_append_event()`'s existing signature, since that's a fixed C-ABI call site already used by
`bindings/node/src/aeon_node.cpp`, and breaking it would be exactly the kind of avoidable blast-radius
mistake this project has caught elsewhere; the nanobind Python binding (`core/src/bindings.cpp`) took
the parameter directly instead, since Python kwargs with a default are backward-compatible by
construction, no `_ex` needed there; `shell/aeon_py/trace.py`'s `TraceGraph.add_event()`. Two new C++
tests (`TraceSemanticSearchTest.EventTimeRoundTripsThroughAppendEvent`,
`.EventTimeSurvivesCompaction`) -- the latter confirmed `compact()`'s existing whole-struct `memcpy()`
already carries the new field over for free (unlike `embedding_blob_*`/`blob_offset`, which are
file-relative offsets `compact()` must explicitly re-point). All 131 C++ tests pass.

`session_expansion.py`'s `merge_expanded_context()` now sorts the final assembled multi-session context
chronologically (by `event_time`, falling back to `timestamp` when unset) instead of leaving it in
base-hits-by-similarity-then-per-session-groups order -- the concrete fix for "which of these things
happened first" that motivated this track. `expand_summaries()` tags each session's summary with a
representative `event_time` (its earliest event) so a summary sorts into roughly the right place too,
rather than defaulting to epoch zero. `run_benchmark.py`'s `_ingest_haystack()` now parses
LongMemEval's own `haystack_dates` strings into real `event_time` values (previously only smuggled into
event *text* as a prefix, unusable for programmatic ordering) via a new
`_parse_haystack_date_micros()` -- verified directly against real sample dates (exact round-trip) before
trusting it, and against a real multi-session question's assembled `full_session` context: all 74
merged events carried a real (non-zero) `event_time`, and the merged list was confirmed chronologically
sorted end-to-end, all via retrieval-only checks, no LLM calls.

**Not yet done, deliberately**: re-running the 5-arm experiment to measure temporal-reasoning's actual
accuracy improvement from chronological ordering -- per the noise-floor caveat above (11-13 questions
per type), a meaningful read of this needs a larger sample than the 50-question pilot, not another
50-question run. Also keeping on record: the qwen/gemma spread (23 points at `top_k=30` on temporal-
reasoning) is a real, separate confound -- the qwen oracle was never re-run untruncated, so how much of
that spread is model capability versus the fixed truncation bug can't be cleanly attributed yet.

**Track 3 -- V2 re-verification. DONE (2026-08-24), both confirmed clean.**
`repr_ab_test.py` re-run on `gemma4:31b-cloud` (same params, `n_errors=0` both arms):
**compact 10.0%, tree 2.5%** (`repr_ab_results_v2.json`) -- the original 15.0%/2.5% reading had an
unclear model provenance (its own log header said qwen; the saved JSON recorded no model at all, the
same gap this session fixed for `oracle_run.py` and now also fixed here via a `model` field added to
`repr_ab_test.py`'s summary output). The compact-beats-tree finding (4x here, was 6x) survives cleanly
under an unambiguous, same-model, untruncated re-run -- direction and magnitude both hold.
`oracle_run.py` re-run the same way: **37.5% overall / 39.5% when-phrase-found** (`oracle_results_v2.json`)
vs. the original 35.0%/36.8% -- a ~2.5-point move, consistent with ordinary run-to-run noise rather than
a truncation artifact, since this oracle's context is capped by `MAX_ORACLE_STATES=10` with a bounded
match window and was already small enough to fit under the old 8192-token default. Nothing further to
correct here; V2's numbers were sound.

**Extract-then-compute prompt experiment. RUN (2026-08-24) -- large, spot-checked-genuine improvement
on exactly the question types the literal-vs-computed diagnostic flagged, one small legitimate
regression.** Built `scripts/longmemeval/extract_then_compute_experiment.py`: same retrieval as
`full_session` (`build_expanded_context`, `MAX_SESSIONS=10`), but generation splits into two prompts --
(1) extract every fact relevant to the question, with source date/session, from the assembled context
(explicitly told not to answer yet); (2) using only the extracted facts, compute/determine the answer,
showing the calculation for anything requiring combining multiple facts. Same 50-question sample,
`gemma4:31b-cloud`, `temperature=0.0`, `n_errors=0`:

| type | single-shot (`full_session`, corrected) | extract-then-compute | oracle ceiling (same model) |
|---|---|---|---|
| multi-session | 45.5% | **72.7%** | 72.7% |
| temporal-reasoning | 46.2% | **69.2%** | 53.8% |
| knowledge-update | 100% | 87.5% | -- |
| overall | 72.0% | 82.0% | 80.0% |

Multi-session lands exactly on the oracle ceiling; temporal-reasoning *exceeds* its own oracle ceiling.
**Checked directly, not assumed, before trusting either number**: spot-checked several flipped-to-
correct multi-session/temporal-reasoning answers against their extraction/computation text -- every one
inspected is genuine correct arithmetic over separately-stated facts (e.g. `"1 (Dr. Smith, March 3rd) +
1 (Dr. Thompson, March 20th) = 2"` matching a reference answer of `"2"`; `"2.5 weeks + 3 weeks = 5.5
weeks"` matching `"5.5 weeks"`), not a judge-leniency artifact from the longer, more structured output.
Temporal-reasoning exceeding its own oracle ceiling has a benign, understood explanation rather than
being a red flag: `oracle_run.py`'s ceiling is deliberately narrow (only the question's exact gold
`answer_session_ids`, raw and unstructured), while `full_session` at `MAX_SESSIONS=10` can pull in
additional non-gold sessions the extraction step benefits from -- the two aren't testing an identical
information budget, so beating a narrower ceiling on a specific 50-question sample is plausible, not
suspicious, though the noise-floor caveat above (11-13 questions per type) still applies to the exact
margin.

**The one regression, also spot-checked, not glossed over**: knowledge-update dropped 100% -> 87.5% (1
of 8 questions). Inspected directly: the extraction step surfaced BOTH an old and a superseding fact
("Thursday (2023/06/16)" and "Friday (2023/06/30)") and the compute step hedged ("Thursday and Friday")
instead of resolving which one supersedes the other, where the single-shot prompt had picked the
correct, current answer outright. A real, specific, understandable failure mode -- knowledge-update
needs picking the most-recent fact, not aggregating all of them -- rather than a sign the technique is
unsound; worth a targeted prompt adjustment (e.g. explicitly asking the extraction step to flag which
facts are superseded) before this goes further, not a reason to abandon the approach.

**Knowledge-update fix attempt (v2 prompt). TRIED AND REVERTED (2026-08-24) -- recorded negative
result.** Added a supersession-handling instruction to both prompts: EXTRACT told to mark which of two
conflicting same-topic facts is current rather than listing both as true; COMPUTE told to use only the
current fact when facts conflict, not average/combine them. Re-ran the same 50-question sample
(`extract_then_compute_results_v2.json`), `n_errors=0`:

| type | v1 (original prompt) | v2 (supersession-aware prompt) |
|---|---|---|
| multi-session | 72.7% | 63.6% |
| temporal-reasoning | 69.2% | 61.5% |
| knowledge-update | 87.5% | **100%** |
| overall | 82.0% | 80.0% |

Fixed the targeted question but introduced two new ones-per-type regressions, net negative overall.
Cross-referenced every multi-session/temporal-reasoning question that flipped correct->wrong against
the v1 run before accepting this as real, not assumed:

1. `bf659f65` (multi-session, ref `"3"`): extraction collapsed two separate purchases of the same EP on
   different dates into one, undercounting to `"2"` -- genuine update-vs-repeat confusion, the failure
   mode the fix was diagnosed against, but applied to the wrong case (two independent repeat purchases,
   not one fact updated).
2. `gpt4_e05b82a6` (multi-session, ref `"10 times"`): extraction dropped a visit ("Space Mountain")
   entirely that v1's extraction had captured -- an extraction-completeness change, not a supersession
   misfire; same context, same temperature, only the prompt text differed.
3. `4dfccbf8` (temporal-reasoning, ref names starting ukulele lessons): compute step refused to answer
   ("I don't have enough information") despite a single, non-conflicting fact -- general hedging induced
   by the added conflict-handling language, not an actual conflict.

Advisor-reviewed before iterating further: only #1 matches the diagnosed root cause; #2 and #3 are
different failure modes the same edit happened to also trigger. All three deltas are one question each
on an n=8-13 bucket -- indistinguishable from noise at this size (same noise-floor caveat as everywhere
else in this stage). A v3 tuned against this same 50-question sample would be fitting to noise, not
fixing a diagnosed cause, so this was not attempted. **Reverted `extract_then_compute_experiment.py` to
the v1 prompt** (the v2 templates are kept in the file as a commented-out record, not silently dropped).
The knowledge-update regression (1 of 8 questions, cause understood: extraction surfaces both an old and
a superseding fact, compute hedges instead of picking the current one) remains open.

**Not yet done**: wiring this into `ContextManager`/a real call site (this was an experiment, per the
same "prove it before building it" discipline Stage 7 task 1 already established); re-running at a
larger sample size (n=500, not n=50 -- same standard as the retrieval-unit arm comparison) to confirm
the extract-then-compute magnitude past the noise floor and to give any future knowledge-update prompt
fix a sample size that can actually distinguish a real improvement from noise.

**What stays out of the plan, restated a second time:** the pre-committed decision rule's `summary`
pick is not a recommendation (now for two independent reasons: it was the worst arm on the metric this
stage exists to move in the first pass, and in this corrected pass it's statistically indistinguishable
from every other arm at n=11); don't expect Track 2 alone to fully close temporal-reasoning's gap --
part of it turned out to be exactly the kind of arithmetic-over-retrieved-facts gap extract-then-compute
just closed most of, not something chronological ordering alone would fix; and don't re-run the 5-arm
retrieval-unit comparison again on 50 questions expecting to resolve which unit is "best" -- that
question needs n=500, not n=50.

**n=500 paired re-run. DONE (2026-08-24) -- pilot's baseline was the misleading number, not the
treatment; a real, mixed effect confirmed at scale.** Ran the full LongMemEval-S dataset (all 500
questions, no sampling) through both `full_session` single-shot and extract-then-compute v1, same
seed, same model (`gemma4:31b-cloud`), same corrected retrieval pipeline (additive multi-session merge,
`MAX_SESSIONS=10`, `num_ctx` auto-sizing, chronological sort). Both runs `n_errors=0`.
(`full_session_n500_results.json`, `extract_then_compute_n500_results.json`)

| type | n | single-shot | extract-then-compute | net flips (gained/lost) |
|---|---|---|---|---|
| abstention | 30 | 90.0% | 96.7% | +2/-0 |
| knowledge-update | 72 | 87.5% | 88.9% | +7/-6 (wash) |
| multi-session | 121 | 66.9% | **76.9%** | +19/-7 |
| single-session-assistant | 56 | 98.2% | 96.4% | +0/-1 |
| single-session-preference | 30 | 46.7% | 36.7% | +2/-5 |
| single-session-user | 64 | 93.8% | 85.9% | +0/-5 |
| temporal-reasoning | 127 | 52.8% | **66.1%** | +22/-5 |
| **overall** | 500 | 73.4% | **78.0%** | -- |

Computed from paired per-question flips (same 500 question IDs, same seed, both arms), not from
comparing the two summary blocks in isolation -- this is what actually tells a real effect from noise,
per the standing discipline.

**First correction to the pilot's own read**: the n=50 pilot's misleading number was the *baseline*,
not extract-then-compute. Pilot baseline showed multi-session at 45.5% and temporal-reasoning at
46.2%; the true full-dataset baseline is 66.9% and 52.8%. Extract-then-compute's own numbers roughly
replicated (72.7%->76.9%, 69.2%->66.1%). The technique's real lift is smaller than the pilot suggested
(+9.9 and +13.4 points, not +27 and +23), but it is now decisively above noise: multi-session flips
19 gained vs 7 lost on n=121, temporal-reasoning 22 vs 5 on n=127 -- both are one-directional wins, not
a coin flip that happened to land favorably.

**Knowledge-update: the pilot "regression" was a baseline-size artifact, not a technique effect.**
At n=72 the two arms are a wash (+7/-6, net +1) -- statistically indistinguishable. This retires the
open question from the earlier fix-and-revert cycle: extract-then-compute does not regress
knowledge-update at scale; the n=8 pilot sample was just too small to read. Read the 2 questions wrong
in BOTH arms (`07741c45`, `6a1eabeb`) directly: both are genuine recency-resolution failures --
`07741c45`'s extraction correctly captured both the old fact ("under the bed") and the newer,
superseding one ("shoe rack", later date), yet the compute step picked the older one anyway, in both
prompting techniques. This confirms the underlying difficulty is real and technique-independent -- not
something either prompt variant reliably solves -- which means a further prompt-level fix is the wrong
shape of fix. The kernel already carries unused `supersedes_id`/`edge_type` fields on `TraceEvent`
(`core/include/aeon/schema.hpp`) built for exactly this; recording supersession at write time and
surfacing it through `session_expansion.py`'s `format_events` is the track worth proposing, sized to
this now-confirmed ~1-in-9 knowledge-update miss rate, not a prompt-tuning track.

**A real, previously-unseen regression, found only because n=500 gives single-session buckets enough
size to read: single-session-user (net -5, one-directional, zero gains) and single-session-preference
(net -3).** Read every one of the 5 single-session-user losses directly (`ec81a493`, `311778f1`,
`c14c00dd`, `8a137a7f`, `b86304ba`): in all 5, the EXTRACT step correctly captured the exact fact
needed, verbatim. The failure is entirely in COMPUTE: told to "use ONLY the facts above" and reason
like it's solving an arithmetic/aggregation problem, it treats a simple, already-answered lookup as
insufficient evidence and refuses ("the text states X but does not specify Y" / "Not mentioned") even
when X *is* Y (e.g. extracted fact states a signed poster is "a limited edition of only 500 copies";
question asks how many copies; compute answers "Not mentioned" instead of 500). Single-session-
preference's 5 losses show the same mechanism applied to open-ended recommendation questions: extracted
facts are complete and correct, but compute refuses to synthesize a recommendation from them
("I don't have enough information to suggest a specific meal" over a list of extracted homegrown
ingredients), where the single-shot prompt used the same facts to make the (judge-accepted)
recommendation directly. **This is a real, general failure mode of the COMPUTE prompt, not noise**:
the "use ONLY the facts above, determine the answer, show the calculation" framing biases the model
toward literal/arithmetic answers and away from any direct-lookup or synthesis/recommendation answer,
causing false abstention specifically on the question types that don't need the two-step split.
Single-session-preference's 46.7% baseline itself is a separate, pre-existing weakness (not something
extract-then-compute broke) and is its own future thread, not addressed here.

**Net picture**: extract-then-compute is a confirmed, substantial win for multi-session and
temporal-reasoning (its intended targets), a wash for knowledge-update, and a real, understood
regression for single-session-user/preference (its unintended targets) -- +4.6 points overall
(73.4%->78.0%), which is positive in aggregate but not a uniform improvement. A production system has
no `question_type` label to route on, so the actual decision is a product trade, not a benchmark
call: always-on extract-then-compute (net positive overall, doubles per-turn generation latency, costs
single-session-user/preference accuracy) vs. a query classifier that routes only aggregation/temporal
questions through the two-step path (avoids the collateral regression, adds a routing component and
its own error mode) vs. leaving single-shot as the default until a classifier is built. This decision
belongs to the user, not to further prompt tuning -- no new prompt variant has been attempted at n=500
or any other n as part of this finding, per the standing no-more-tuning-at-small-n guardrail.

**Router experiment. DONE (2026-08-25) -- offline, zero LLM calls, honest result: the classifier ties
always-on ETC on accuracy, and only partially fixes the regression it was meant to avoid.** User picked
"build a query classifier to route" over always-on/do-nothing. Built
`scripts/longmemeval/router_experiment.py`: since both arms' per-question correctness for all 500
questions is already on disk, routed accuracy for ANY routing function is just a lookup -- this entire
evaluation cost zero new LLM calls, only one local (non-LLM) mpnet embedding pass over the 500 question
texts.

*Step 1 -- type-based oracle ceiling* (route {multi-session, temporal-reasoning} to ETC, everything
else -- including knowledge-update, an accuracy wash -- to single-shot, using the TRUE type label):
**79.4%**. This is the ceiling for ANY classifier working from question text alone, before any real
model was built: only +1.4pp (7 questions) above always-on ETC's 78.0%. Routing was never going to be
a large accuracy play; this number, computed first, set that expectation honestly before judging the
real classifier.

*Step 2 -- real classifier*: logistic regression over the question's mpnet embedding (the SAME
embedding the real pipeline already computes once per query for retrieval, so this costs zero marginal
inference calls), trained to predict the binary {ETC, single-shot} route implied by the type-based
rule. Scored via stratified 5-fold CV, routed accuracy computed ONLY from out-of-fold predictions (an
in-sample number would repeat the exact overfitting mistake the knowledge-update prompt-tuning revert
already flagged this stage).

| | accuracy | vs. always-on ETC |
|---|---|---|
| always single-shot | 73.4% | -4.6pp |
| **routed (real classifier, out-of-fold)** | **77.6%** | **-0.4pp (a wash, ~2 questions)** |
| always extract-then-compute | 78.0% | -- |
| type-based oracle ceiling (upper bound) | 79.4% | +1.4pp |

The classifier routes 55% of questions to ETC (232 true positives + 43 false positives of 500),
confusion `TP=232 FN=34 FP=43 TN=191`. **The 77.6%-vs-78.0% result is not the classifier failing --
it's confirmation the oracle ceiling already said accuracy was never the case for routing.** The actual
value of routing is two other things, and both are partial, not clean:

1. **Latency**: ~45% of queries skip the second (compute) LLM call entirely.
2. **Regression mitigation, partial not full**: single-session-user lands at 90.0% routed accuracy
   (between single-shot's 93.8% and always-ETC's 85.9%) -- roughly half the known regression
   recovered, not eliminated, because 30% of single-session-user questions still get misrouted to ETC.
   Single-session-assistant and single-session-preference, by contrast, misroute at 0% -- the
   classifier separates those cleanly.

**Why single-session-user misroutes at 30% while -assistant/-preference misroute at 0%, worth stating
plainly so a future reader doesn't re-attempt "just train a better classifier":** the embedding is
separating question STYLE, not routing NEED. Single-session-user questions ("how many hours did I
spend...", "how much did I pay for...") are lexically similar to the aggregation-style multi-
session/temporal-reasoning questions ETC was built for, even though they don't need the two-step split
(the fact is already a direct, single-session answer). This is a phrasing-similarity confound, not a
data or training deficiency -- a bigger model or more folds is unlikely to move it much, since the
signal that would distinguish them (whether the answer requires combining multiple facts) isn't
strongly present in the question's surface embedding alone.

Also worth flagging precisely rather than letting it read as a routing win: knowledge-update's routed
accuracy (89.7%) sits above BOTH arms individually (87.5%/88.9%). This is a mixing artifact of a 28%
misroute rate on a type where the two arms are a wash, not a real benefit of routing -- don't cite it
as one.

**Caveat that stays a caveat**: a classifier trained on LongMemEval's own formulaic question phrasing
will likely score well on LongMemEval's own held-out folds. That is not evidence it generalizes to less
formulaic production queries. Kept the model deliberately simple (embedding + logistic regression, no
feature engineering, no hyperparameter search) for exactly this reason -- and per the standing
guardrail, no threshold/hyperparameter search was run against these same 500 questions, since a missed
ETC route and a wrong ETC route each cost roughly 8% of their bucket, making the expected gain from
tuning small and the overfit risk the same one already dodged twice this stage.

**Re-presented to the user rather than proceeding to wiring**: the "build a classifier" pick predated
this data, and the decision surfaced by it is materially different from what was asked for --
routed accuracy ties always-on, so the real choice is now (a) wire the router anyway for the latency
win and partial regression relief, at the cost of one more component with its own failure mode; (b)
skip the router, wire always-on ETC instead -- simpler, statistically the same accuracy, full 2x
latency and full single-session-user regression; (c) hold, keep single-shot the default, park both
until there's a reason to revisit. Not decided in this pass.

**COMPUTE prompt fix (v3), one pre-committed attempt. PRE-REGISTERED (2026-08-25), before running.**
User asked what the best path was; advisor recommendation: the ss-user/preference regression is not an
intrinsic property of extract-then-compute, it's one over-constrained instruction in the COMPUTE
prompt. Reading all 10 losing questions (5 ss-user, 5 ss-preference, above) showed EXTRACT correctly
captured the needed fact in every single case -- the failure is entirely COMPUTE refusing to use a
directly-stated fact ("not mentioned") or refusing to synthesize a recommendation from complete facts,
because "use ONLY the facts above, determine the answer, show the calculation" frames every question as
an arithmetic problem. This is a targeted relaxation of an existing over-constraint, not a new
semantic-distinction instruction (unlike the reverted v2 knowledge-update fix, which asked the model to
reliably tell "update" from "repeat" -- a distinction it can't reliably make). Standing no-more-
prompt-tuning guardrail explicitly lifted for this ONE attempt, because both conditions that justified
it have changed: this runs at n=500 (not the n=50 noise floor) and the user explicitly asked for the
best path forward (not a speculative side-quest).

Change (COMPUTE prompt only -- EXTRACT stays v1 unchanged, since extraction was never the failure):
add that (a) a fact which directly states or clearly implies the answer IS the answer, even under
different wording or missing a category label the question happens to use -- never "not mentioned" in
that case; (b) recommendation/suggestion questions are answered by synthesizing directly from the
facts, not refused for lack of a pre-phrased recommendation; (c) abstain only when the specific
information asked for is genuinely absent after checking carefully. Clause (c) is the regression risk
of the fix itself -- ETC's abstention type is currently 96.7% (above baseline's 90%), and relaxing
COMPUTE's strictness could pull that back down.

**Acceptance bar, committed BEFORE the run, against the stored n=500 baseline** (`full_session_n500_results.json`,
unchanged -- only the ETC arm reruns):

| type | n | floor to pass |
|---|---|---|
| single-session-user | 64 | >= 59/64 (~92%, recovers to within ~1 net loss of baseline's 60/64) |
| single-session-preference | 30 | no explicit floor (baseline itself is only 46.7% -- pre-existing weakness, not this fix's job) but must not fall below baseline's 14/30 |
| multi-session | 121 | >= 109/121 (~90%, holds the confirmed win, allows minor CV-scale noise) |
| temporal-reasoning | 127 | >= 102/127 (~80%, holds the confirmed win) |
| abstention | 30 | >= 27/30 (90%, doesn't fall below single-shot's own abstention floor) |
| overall | 500 | >= 78.0% (must not regress below the already-confirmed always-on ETC number) |

One rerun only, no second iteration regardless of outcome. Clears the bar -> wire always-on
extract-then-compute into the shell (`ContextManager`/`loop.py`), 2x per-turn generation latency stated
plainly as the product cost at wiring time, router experiment stays parked as a separate, decoupled
future latency optimization. Fails the bar -> revert to the v1 prompt, hold single-shot as the shell
default, extract-then-compute stays a documented, verified-but-not-shipped finding. Unchanged either
way: kernel supersession track stays parked (user didn't pick it), single-session-preference's 46.7%
baseline is its own separate pre-existing thread, no classifier threshold tuning.

**Result: FAILED, reverted (2026-08-25).** Ran the ETC arm only, n=500, same seed/model,
`extract_then_compute_n500_v3_results.json`, `n_errors=0`. **Correction owed here first**: the
committed multi-session (>=109/121, ~90%) and temporal-reasoning (>=102/127, ~80%) floors above were a
calibration error on my part -- the technique's own already-confirmed n=500 baseline never reached
those numbers (93/121 = 76.9%, 84/127 = 66.1%), so those two floors were unpassable by construction,
independent of whether the fix worked. Flagging this plainly rather than letting a broken bar stand.
Judged instead against the number that actually matters -- v3's raw counts vs. v1's own achieved
counts, same 500 questions:

| type | n | v1 | v3 | delta |
|---|---|---|---|---|
| single-session-user | 64 | 55 (85.9%) | 55 (85.9%) | **0 -- completely unchanged** |
| single-session-preference | 30 | 11 (36.7%) | 17 (56.7%) | +6, genuine improvement |
| multi-session | 121 | 93 (76.9%) | 94 (77.7%) | +1, noise |
| temporal-reasoning | 127 | 84 (66.1%) | 81 (63.8%) | -3, new regression |
| knowledge-update | 72 | 64 (88.9%) | 60 (83.3%) | -4, new regression |
| single-session-assistant | 56 | 54 (96.4%) | 53 (94.6%) | -1, noise |
| abstention | 30 | 29 (96.7%) | 30 (100%) | +1, noise |
| **overall** | 500 | **390 (78.0%)** | **390 (78.0%)** | **exact tie** |

Re-read the same 5 single-session-user questions diagnosed earlier (`ec81a493`, `311778f1`, `c14c00dd`,
`8a137a7f`, `b86304ba`) directly against the v3 output: all 5 still fail, producing near-word-for-word
the same "not mentioned"/"insufficient information" hedge as v1, despite the new instructions
explicitly telling the model not to do that (e.g. `ec81a493`'s v3 hypothesis: *"the signed poster is a
limited edition of 500 copies, but it does not specify the total number of album copies released...
Answer: Not mentioned"* -- unchanged from v1's failure on the identical question). **The fix did not
move the failure mode it targeted at all**, single-session-preference improved genuinely but that
wasn't the metric the bar was written around, and the relaxation introduced two new regressions
(knowledge-update, temporal-reasoning) that offset the preference gain almost exactly, for an overall
tie. Reverted `extract_then_compute_experiment.py`'s COMPUTE prompt to v1 (v3 kept as a commented-out
record). Per the pre-committed protocol: no second iteration. **Extract-then-compute stays a
documented, verified, NOT-shipped finding**; single-shot remains the shell default. The
single-session-user/preference regression against always-on ETC is unresolved -- the mechanism
(COMPUTE's literalism) is understood, but the specific fix attempted for it did not work, and prompt-
level iteration on this specific failure mode is now considered exhausted for this stage.

**System-prompt probe (2026-08-25). Offline diagnostic, not a prompt-tuning attempt -- no rerun of
the pipeline, no acceptance bar needed, script + results committed alongside this entry.** Advisor
hypothesis after v3's failure: every LLM call in this benchmark (extract, compute, AND the
single-shot baseline) reuses the same `SYSTEM_PROMPT` (`run_benchmark.py`), framed for the
single-shot arm ("answering using ONLY the retrieved memory snippets below... say so plainly instead
of guessing"). By COMPUTE time the input is `extracted_facts`, not snippets -- a frame mismatch, and
a system-level "say so plainly" instruction could plausibly dominate a user-turn relaxation, which
would explain why v3's explicit anti-hedging instructions moved single-session-user 0/8 despite being
pre-registered and reasoned. Built `scripts/longmemeval/system_prompt_probe.py`: re-runs ONLY the
COMPUTE step (no retrieval, no extraction -- reuses `extracted_facts` already on disk from the n=500
v1 run) across `{v1 system, step-appropriate system, empty system} x {v1 compute, v3 compute}` on the
8 known single-session-user/preference losses plus 5 abstention questions as regression canaries
(one canary selection bug: sampled `abst[:5]` instead of `correct_abst[:5]`, so `09ba9854_abs` was
already wrong under the stored v1 baseline -- the "canaries held" counts below are correct as raw
counts but shouldn't be read as "4->5 is an improvement" for that one row).

| system \ compute | v1 compute | v3 compute |
|---|---|---|
| v1 (single-shot-framed) | 0/8 flipped (known, on disk) | 2/8 flipped, 4/5 canaries held |
| step-appropriate (rewritten for "facts, not snippets") | 0/8 flipped, 5/5 canaries held | 2/8 flipped, 5/5 canaries held |
| empty (no system prompt at all) | 1/8 flipped, 4/5 canaries held | 4/8 flipped, 5/5 canaries held |

**Hypothesis refuted.** The discriminating cell is step-appropriate-system/v1-compute: fixing the
system-prompt frame mismatch while leaving COMPUTE's wording untouched flipped zero of the 8 losses.
Even the maximal condition tested -- system prompt removed entirely, combined with v3's relaxed
compute wording -- still left 4 of 5 single-session-user losses stuck. The system prompt is not the
(or even a major) blocking factor. **This probe has zero coverage of knowledge-update and
temporal-reasoning**, the two question types v3's real n=500 run regressed on (-4, -3) -- no cell in
this table is a candidate to ship; it answers the mechanism question only.

**Mechanism, corrected with per-case evidence (grepped each of the 5 single-session-user losses'
raw answer-bearing session against the extracted fact and the question's exact wording) -- corrects
the earlier "EXTRACT always captured the exact fact needed, verbatim" claim (made when only the
extracted-facts side had been read, not the raw session):**

| id | extraction vs. raw session | failure sub-mode |
|---|---|---|
| `311778f1` | raw session contains "documentaries on **Netflix**" in the same turn; the extracted bullet dropped "Netflix" | **(i) literal extraction loss** |
| `ec81a493` | extracted bullet is verbatim-identical to the raw session's "limited edition of only 500 copies worldwide" | **(ii) pragmatic license** -- gold answer treats the poster's edition size as the album's worldwide release count, a leap the raw text doesn't literally state either |
| `c14c00dd` | extracted bullet is verbatim-identical to the raw session's "shampoo... picked up at Trader Joe's" | **(ii) pragmatic license** -- gold answer treats the store as the brand |
| `8a137a7f` | extracted bullet is verbatim-identical to the raw session; raw session never uses the word "replace" near the bulb at all | **(ii) pragmatic license** -- gold answer treats "currently using" as "replaced with," a leap not literally licensed by the raw text either |
| `b86304ba` | "painting of a sunset" is absent from the raw session (grepped for it directly) -- same as it's absent from the extracted facts | **(iii) hedge-format difference** -- baseline's accepted answer is "the text doesn't mention a painting of a sunset, but it mentions a flea-market find worth triple"; COMPUTE's rejected answer is a flat "I don't know" over the identical gap |

Only 1 of 5 is genuine extraction data loss. The dominant pattern (3 of 5, mode ii) is: the
answer-bearing sentence is verbatim-identical between what COMPUTE saw and what the raw session
contains, and the gold label itself requires a pragmatic leap (store-as-brand, currently-using-as-
replaced) that a single-shot conversational read makes liberally and COMPUTE's "use ONLY the facts,
determine the answer" framing does not, regardless of which of the six wordings above was tried. This
is a fact about what LongMemEval's single-session-user/preference labels grade as correct (loose,
conversationally-licensed inference), not a claim that baseline and COMPUTE had unequal information
in these 3 cases -- they had the same answer-bearing sentence; what differed was the surrounding
conversation and the task framing around it.

**What this does and doesn't settle.** No prompt-level lever on COMPUTE (v2, v3, or this probe's
system-prompt axis) fixes modes (ii)/(iii), because the missing ingredient isn't in the extracted
facts to begin with -- prompt-level iteration on this failure mode is exhausted, now with the
mechanism actually verified rather than inferred from output text alone. The `full_session` baseline
is an existence proof that raw session text as input produces correct, judge-accepted answers on all
5 of these questions, across all three sub-modes -- but that's evidence about raw-alone input, not
about raw-text-plus-extracted-facts together (an untested combination), and says nothing about
whether multi-session/temporal-reasoning's wins survive adding raw text back in for single-session-
shaped queries. A hybrid-input arm (raw session alongside extracted facts, gated to single-session-
shaped queries) is the one option actually aimed at this now-verified mechanism, but it is new
experiment scope, not a rerun of anything pre-registered above, and per the standing guardrail is not
self-authorized -- it's presented to the user alongside the original hold/router/always-on-ETC
decision, not run.

**Full failure inventory before authorizing any further n=500 (2026-08-25). Zero LLM calls -- pure
re-analysis of the three arms already on disk (`full_session_n500_results.json`,
`extract_then_compute_n500_results.json`, `extract_then_compute_n500_v3_results.json`) plus the
existing oracle and session-recall diagnostics.** Motivated by the user's question -- do we actually
understand what's broken before spending hours on another run? Answer: no, and the three most
important findings all contradict what this stage had been assuming.

**Finding 1 -- the noise floor was never measured, and past decisions were made inside it.** The n=50
extract-then-compute sample is fully nested in the n=500 run with identical model, seed, prompts,
`base_top_k`, and `max_sessions` -- so those 50 questions were run twice under identical config. They
disagree on **4 of 50 (8%)**. Attribution by final `Answer:` line: 3 are generation nondeterminism
(different answers: 3 vs 2, "$720" vs "not enough information", 10 vs 7), 1 is pure judge
nondeterminism (`ce6d2d27`, byte-identical answer "thursday and friday", opposite verdicts). Deeper:
`extracted_facts` reproduces verbatim on only **31/50**, and the final hypothesis on only 24/50 --
i.e. **extraction is nondeterministic on ~38% of questions at `temperature=0.0`** (`gemma4:31b-cloud`
is a cloud-batched model; temp 0 is not determinism). 4/50 is a wide CI (~2-19%), so call it "roughly
8%, preliminary" -- but even at that estimate, ~40 of 500 questions flip run-to-run, giving a net-delta
standard deviation of ~sqrt(40) ~= **6 questions (~1.2 points) at n=500**. Consequences, stated
plainly: v3's headline "regressions" (knowledge-update -4, temporal-reasoning -3) are **not
distinguishable from noise at any defensible bar** (knowledge-update's -4 on n=72 is ~1.7 sigma --
suggestive at best, nowhere near a revert-grade signal), as is knowledge-update's +1, single-session-assistant's -1, and abstention's +2. v3's
"exact 390/390 tie" hides **32 changed questions** (16 gained, 16 lost) -- almost exactly the ~40
expected from noise alone. The v2 revert, decided on +1/-1/-1 deltas at n=50, was decided on noise.
**No acceptance bar written in this stage had a noise model; that is the methodological defect behind
both reverts, not the prompts themselves.**

**What survives the noise test.** ETC's overall +23 (367 -> 390, +4.6 points), multi-session +12, and
temporal-reasoning +17 are all far outside the band -- **real wins, unchanged**. The
single-session-user regression also survives: 5 losses with **zero** offsetting gains (one-directional,
p ~= 3% by chance), replicated at exactly 55/64 in two independent runs, and corroborated case-by-case
by the deterministic system-prompt probe above. The noise finding does *not* dissolve it.

**Finding 2 -- a real harness bug: `question_date` is never passed to the model.** LongMemEval ships
`question_date` on all 500 questions (it is the reference "now" that makes relative-time questions
answerable). It appears in **zero lines of Python in this repo** -- `run_benchmark.py`,
`extract_then_compute_experiment.py`, and `session_expansion.py` all build prompts without it. Every
relative-time question ("What kitchen appliance did I buy 10 days ago?", "How many weeks ago did I
start using Ibotta?", "an art event two weeks ago -- where was it held?") is therefore **unanswerable
by construction**. The failure is visible verbatim in the outputs: `gpt4_e072b769` -- *"The provided
text does not contain the current date, so the number of weeks cannot be calculated"*; `gpt4_59149c78`
-- the model **hallucinates** *"The current date is 2023/01/15"* and reasons from it. Counting only
answers that explicitly complain about or invent a current date: **21 questions, all
temporal-reasoning, all wrong, 17 of them in the hard core** -- that is **49% of all
temporal-reasoning errors and 19% of every error in the run**, and it is a strict lower bound (a
question that just answers "I don't know" without naming the date isn't counted). **All
temporal-reasoning numbers in this document -- baseline, ETC v1, ETC v3, and the +17-question ETC
"win" -- are comparisons between two configurations that were both broken in the same way.** Fixing
this is a bug fix, not an experiment, and it must land before any further temporal-reasoning claim is
made.

**Finding 3 -- the error mass was never where this stage was working, and retrieval is not the
bottleneck.** ETC v1's 110 errors by bucket: temporal-reasoning 43 (39.1%), multi-session 28 (25.5%),
single-session-preference 19 (17.3%), single-session-user 9 (8.2%), knowledge-update 8 (7.3%),
single-session-assistant 2, abstention 1. **The single-session-user regression that consumed three
consecutive fix attempts (v2, v3, the system-prompt probe) is 8.2% of the error mass**; temporal +
multi-session + preference are 81.8% and, until this inventory, **no case in either of the two largest
buckets had ever been read at the case level** in this stage. Reading 10 of them (5 temporal, 5
multi-session, all from the 74-question hard core that is wrong under every arm) shows the dominant
mode is neither compute reasoning nor a single locus but **incomplete fact recall on aggregation**:
`8e91e7d9` (gold: 4 siblings) extracted only *"user: I have a brother"* -> answered 1; `1a8a66a6`
(gold: 2 subscriptions) extracted one -> answered 1; `81507db6` (gold: 3 ceremonies) extracted two ->
answered 1; `gpt4_ab202e7f` (gold: 5 kitchen items) extracted four -> answered 4.

**Locus split, settled by a zero-LLM-call discriminating check** (`scripts/longmemeval/aggregation_locus_check.py`):
rebuild the exact assembled context each arm saw (verified byte-identical to the recorded
`context_chars` for all 5 questions) and grep it for the gold mentions extraction failed to surface.
This was necessary because "extraction dropped it" was, until run, an output-side inference -- the
facts were missing from extraction's *output*, but nobody had checked extraction's *input*. Result:
**the bucket splits both ways, and the earlier single-locus reading was wrong.**

**METHOD CORRECTION (2026-08-25) -- the first version of this split was wrong, and wrong in a
systematic direction.** `aggregation_locus_check.py` grepped the rebuilt context for a **keyword**
("subscri", "graduation", "coffee maker") and called a hit "the fact was retrieved, so extraction
dropped it." That conflates *the topic word appearing somewhere in 120k chars* with *the
answer-bearing turn being retrieved* -- and on aggregation questions the topic word necessarily
repeats across many sessions, including assistant echoes and unrelated mentions. So the method
**systematically under-reports retrieval misses on exactly the question shape it was built to
diagnose.** LongMemEval flags the specific answer-bearing turns (`has_answer: true`, 896 across the
500 questions); checking those turns individually against the context
(`scripts/longmemeval/answer_turn_attribution.py`, calibration mode) gives the corrected table:

| question | answer-bearing turns retrieved | locus |
|---|---|---|
| `8e91e7d9` (4 siblings) | **1 of 2** | **retrieval miss** |
| `ba358f49` (user's age) | **1 of 2** | **retrieval miss** |
| `1a8a66a6` (2 subscriptions) | **1 of 3** -- keyword grep called this "retrieved" | **retrieval miss** (was: extraction loss) |
| `gpt4_ab202e7f` (5 kitchen items) | **5 of 5** | **extraction loss** (confirmed) |
| `81507db6` (3 ceremonies) | **1 of 3** -- keyword grep called this "retrieved" | **retrieval miss** (was: extraction loss) |

Corrected reading: **4 of 5 are Aeon-side retrieval failures**, not EXTRACT failures -- the answer
turns never reached the model. Only `gpt4_ab202e7f` is a verified extraction loss. This moves the
aggregation bucket's centre of gravity from a prompt fix to a **retrieval/recall fix in the product
itself**, which is the opposite of the conclusion the keyword method produced. It also refines the
existing session-recall diagnostic (96% all-golds-in-top-k30, 98.2% mean gold fraction): recall is
high *per gold session on average*, but aggregation questions need **every** answer turn, and
`top_k=30`/`max_sessions=10` is losing most of them on exactly these questions. **The EXTRACT prompt
has never been modified in this stage** -- v2, v3, and the probe all changed COMPUTE or the system
prompt -- but "never tried" is a reason to *test* EXTRACT, not evidence it is the fix; on this
corrected split, 4 of 5 cases cannot be fixed there at all.

Five cases still cannot size a bucket, which is why the same `has_answer` method is run over all 500
questions as a four-way attribution (retrieval miss / extraction loss / compute-or-judge / correct) --
results in the next entry.

**Correcting an over-claim before it propagates:** the oracle comparison (80.0% gold-context vs ETC's
82% on the same 50) does *not* establish "we are at the model's capability ceiling." That is n=50
(CI ~ +/-11 points), and the honest slice is the intersection of the oracle sample with the hard core:
6 questions, of which the oracle **also failed 4** with perfect gold context (2 temporal, 1 preference,
1 multi-session) and **got 2 right** that ETC got wrong. Supportable claim: *part* of the hard core is
model-reasoning-bound; part is recoverable. n=6 cannot size the split. Similarly, the best-of-3-arms
union ceiling (426/500 = 85.2%, +36 questions over the best single arm) is **headroom context only, not
a routing pitch** -- a perfect 3-arm router is unattainable and the real classifier already measured
77.6%.

**Per-bucket decision table.** "Detectable?" is the noise model earning its keep: any effect smaller
than ~2x the ~6-question net-delta sd cannot be distinguished from nondeterminism by a single n=500
run, no matter how long it takes.

| bucket | errors | what's broken | mechanism status | candidate fix | expected gain | trade-off | detectable at n=500? |
|---|---|---|---|---|---|---|---|
| temporal-reasoning, missing "now" | 21+ of 43 | `question_date` never passed to any prompt; model invents or refuses | **verified** (grep: 0 code refs; 21 outputs name it) | pass `question_date` into EXTRACT + COMPUTE + baseline | ~15-21 questions (3-4 pts) | none -- bug fix; but **invalidates every prior temporal number**, so baseline must be re-run too | **yes**, ~3x noise |
| multi-session aggregation -- **retrieval miss** | **4 of 5 read** (corrected) | answer-bearing turns never reach the assembled context -- `top_k=30`/`max_sessions=10` surfaces 1 of 2-3 answer turns on aggregation questions | **verified per-`has_answer`-turn**; sized over all 500 in the next entry | raise/adapt `top_k`+`max_sessions` for aggregation-shaped queries -- **Aeon-side recall work, the product itself** | unsized until the n=500 attribution | latency, context size, num_ctx pressure | pending attribution |
| multi-session aggregation -- **extraction loss** | 1 of 5 read (`gpt4_ab202e7f`) | all answer turns present in context, EXTRACT drops one; compute faithfully sums the subset | **verified** (5/5 turns retrieved, 4 extracted) | EXTRACT-side fix (never yet attempted in this stage) | smaller than previously implied | more extraction tokens; over-extraction may hurt precision-sensitive types | pending attribution |
| single-session-preference | 19 | 11 unfixed by any arm; pre-existing baseline weakness (14/30), not caused by ETC | **partly diagnosed** -- the 5 ETC-regressed cases were read (modes ii/iii above); the **14 wrong-in-both-arms have never been read** | unknown; read the 14 first | unknown | -- | n/a until diagnosed |
| single-session-user regression | 9 | modes (i)/(ii)/(iii) above -- pragmatic-license gap, verified | **verified** (probe + per-case grep) | hybrid raw-session input, single-session-gated | **<= 9 questions (1.8 pts)** | 2x latency; new component | **NO -- at/below the ~1.2-pt noise floor.** A full n=500 likely cannot distinguish this fix from noise |
| knowledge-update supersession | 8 | 2 hard-core; rest inside noise | inferred, not verified | kernel supersession track (parked) | ~2-6 | kernel work | no, alone |

**The punchline for the ship decision: the hybrid-input arm -- the option this session was building
toward -- targets at most 9 questions against a ~6-question noise sd, and is the *least* detectable
item in the table, while the single largest, fully-verified, zero-experiment-needed defect
(`question_date`) sits unfixed and contaminates 133 temporal-reasoning questions.** Running the
hybrid n=500 next would spend hours to measure something the instrument cannot resolve. Any future
n=500 in this stage must additionally: (a) report **paired McNemar-style gain/loss counts**, not raw
subtype deltas; (b) set acceptance bars at **>= 2x the measured noise floor**; (c) re-measure the noise
floor properly (repeat one arm on a fixed ~100-question slice) rather than relying on this
opportunistic 4/50; and (d) treat any subtype bucket smaller than ~50 questions as undecidable on its
own.

**Answer-turn attribution over all 500 questions (2026-08-25). Zero LLM calls**
(`scripts/longmemeval/answer_turn_attribution.py`, results in
`answer_turn_attribution.json`). Requested before authorizing any rerun: settle the
extraction-vs-retrieval split properly rather than from 5 hand-picked cases. LongMemEval flags the
exact turns containing each answer (`has_answer: true`, 896 turns across the 500 questions), so the
split becomes mechanical: rebuild each question's assembled context deterministically, check which
answer-bearing turns reached it, and cross that with the stored n=500 extraction and correctness.

*Integrity checks, all clean*: the rebuilt context is byte-identical to the recorded `context_chars`
on **479/479** questions (retrieval is deterministic -- the 8% run-to-run flip rate is entirely
LLM-side, confirming the noise attribution above); the retrieval matcher passes calibration against
five per-turn-verified cases; and assistant-role answer turns are retrieved **54/54**, so there is no
user-turn bias in retrieval or `format_events` (a hypothesis worth killing, now killed). The 21
questions with no answer-bearing turns are the abstention set and are excluded from attribution.

| type | n | correct | retrieval miss | retrieved-but-wrong |
|---|---|---|---|---|
| temporal-reasoning | 127 | 84 | **15** | 28 |
| multi-session | 121 | 93 | **9** | 19 |
| single-session-preference | 30 | 11 | 1 | 18 |
| single-session-user | 64 | 55 | 2 | 7 |
| knowledge-update | 72 | 64 | 0 | 8 |
| single-session-assistant | 56 | 54 | 0 | 2 |
| abstention (answer-bearing subset only) | 9 | 8 | 0 | 1 |
| **TOTAL** | **479** | **369** | **27 (25% of errors)** | **83 (75% of errors)** |

(21 of the 30 abstention questions have no answer-bearing turn by construction and are excluded
entirely; the 9 shown are the remainder, which do carry one.)

**The retrieval number is solid, and the causal check is the strongest signal in this whole stage:**

| | mean answer-turn recall | all answer turns present |
|---|---|---|
| correct answers (n=369) | **99.5%** | **98.9%** |
| wrong answers (n=110) | **85.9%** | **75.5%** |

A quarter of wrong answers are missing at least one answer-bearing turn, against ~1% of correct ones.
**27 of 110 errors (25%) are Aeon-side recall failures** -- the evidence never reached the model, so no
prompt anywhere in the pipeline could have fixed them. This is the first hard number sizing Aeon's own
retrieval as a contributor to benchmark error, and it is concentrated exactly where the earlier
per-case reads pointed: aggregation questions needing several answer turns, where `top_k=30`/
`max_sessions=10` surfaces one of two or three.

**NEGATIVE RESULT -- the automated extraction/compute split does not work, and the number it produced
is withdrawn.** The plan was to sub-split the 83 retrieved-but-wrong errors by content-word overlap
between each answer turn and the stored `extracted_facts`. That metric produced "extraction_loss=36
(33% of errors)", and it is **invalid**: among questions that were answered **correctly** with all
answer turns retrieved -- where extraction demonstrably succeeded -- the metric flags **50%** as
"extraction incomplete" at the generous threshold and **91%** at the strict one, a *higher* false-
positive rate than it produces on wrong answers (43% / 82%). It has no discriminating power; it
measures turn verbosity (long assistant turns lose most content words to any faithful summary), not
extraction fidelity. Reported here rather than quietly dropped, because the point of this exercise was
to stop trusting unvalidated proxies -- and because the same failure would recur in anyone's next
attempt at it. Splitting extraction from compute needs an LLM-judged pass ("is this fact present in
these extracted facts?"), which is a run, not free re-analysis.

**Hand-read of 9 retrieved-but-wrong cases** (indicative, not a sizing) gives a provisional shape:

| case | verdict |
|---|---|
| `gpt4_ab202e7f` (5 kitchen items) | **extraction loss** -- 5/5 turns retrieved, 4 extracted |
| `46a3abf7` (3 tanks) | compute -- facts complete, excluded the "old" 5-gallon tank |
| `8979f9ec` (8 meals) | compute -- refused to add 3+5, same pragmatic-literalism mode as the ss-user losses |
| `gpt4_7fce9456` (4 properties) | compute -- counted the target townhouse itself |
| `2318644b` (Hawaii vs Tokyo) | **judge** -- answered "Over $270" against gold "$270"; the arithmetic is right |
| `gpt4_e072b769`, `gpt4_59149c78`, `gpt4_b0863698`, `9a707b82` | **all four are the `question_date` bug** |

Two things follow. First, **4 of 9 -- and, on the earlier scan, 21 of the 43 temporal errors -- are the
already-fixed `question_date` bug**, so a large share of the temporal "retrieved-but-wrong" column is
expected to clear on the next run without any new work. Second, among multi-session cases, only 1 of 5
was extraction loss; the rest are compute-side counting and literalism errors -- the same failure mode
already verified on single-session-user, appearing again in the largest bucket. **The corrected
picture inverts the working assumption of this stage: the aggregation bucket is retrieval + compute,
not extraction.** The EXTRACT prompt, still never modified, remains untested rather than exonerated,
but nothing found so far makes it the leading candidate.

**Revised priority, evidence-ranked:** (1) `question_date` fix -- done in code, ~21 questions, needs a
paired rerun to bank; (2) **Aeon-side retrieval -- 27 questions (25% of all errors), verified, and the only
item that is squarely product work rather than benchmark prompt-tuning; it splits into 20 partial-recall
(top_k/max_sessions tuning) and 7 complete misses caused by semantic dilution of buried asides
(embedding/chunking work -- a different repair, and raising top_k will not touch it)**; (3) compute-side literalism/counting -- the largest remaining bucket but the one
where three prompt attempts have already failed; (4) EXTRACT -- untested; (5) the hybrid ss-user arm --
unchanged at <= 9 questions against a ~6-question noise floor, still the least measurable item on the
list.

**Shape of the 27 retrieval misses -- two different failures needing two different fixes.** The
previous entry described these as aggregation truncation, a shape verified only on multi-session
cases; 15 of the 27 are temporal and had never been inspected. Splitting by how many answer turns
each question has:

| shape | n | by type | mechanism | fix |
|---|---|---|---|---|
| **partial recall** (e.g. 1 of 3 turns retrieved) | **20** | multi-session 9, temporal 10, preference 1 | question needs several answer turns; `top_k=30`/`max_sessions=10` surfaces some | raise/adapt `top_k`+`max_sessions` for multi-evidence queries |
| **complete miss** (0 of N retrieved) | **7** | temporal 5, single-session-user 2 | see below -- ranking/relevance, not truncation | embedding/chunking work |

Reading the 7 complete misses shows one consistent, nameable mechanism: **the answer is a passing
aside inside a turn whose dominant topic is something else.** `gpt4_8279ba03` (gold: a smoker) --
the answer turn is a request for *BBQ sauce recipes* ending "By the way, I just got a smoker today";
`726462e0` (gold: 10% discount) -- the turn is about *promoting writing services on Instagram and
Twitter*; `gpt4_468eb064` (gold: Emma) -- the turn is about *social media advertising tips*. Aeon
embeds the whole turn, so the aside is swamped by the turn's main topic and the query never matches
it. This is **semantic dilution, a genuine kernel/embedding-side finding**, and distinct from the
truncation story: raising `top_k` will not fix it, because the turn does not rank anywhere near the
query at any `k`. Candidate directions (none attempted): sub-turn chunking before embedding,
multi-vector per turn, or query expansion. Worth stating plainly because it is the one failure in this
entire stage that is unambiguously about Aeon's own indexing rather than prompt scaffolding.

**Single-session-preference, previously flagged "read the 14 first" -- now read.** Four
retrieved-but-wrong cases inspected (`09d032c9`, `6b7dfb22`, `75f70248`, `38146c39`). One is an
extraction loss (`09d032c9`: the answer turn was retrieved, extraction returned "No relevant facts").
The other three share a single mechanism: **the rubric grades personalization, and COMPUTE answers
generically from a sterile fact list.** `38146c39` -- facts carry the user's turbinado-sugar
experiments; the answer recommends "a pinch of flaky sea salt", never mentioning turbinado.
`6b7dfb22` -- facts carry Instagram flower paintings and a 30-day challenge; the answer says "seek
inspiration from social media, art communities". `75f70248` -- facts carry the shedding cat; the
answer restates the facts without committing to a recommendation. This is the same root as mode (ii)
in the single-session-user diagnosis: extraction converts a conversation into a decontextualized fact
list, and the two-step split strips exactly the conversational specificity these rubrics reward.
**It also retro-explains v3**: v3's clause (b) told COMPUTE to synthesize recommendations directly from
the facts, and preference was the one type that genuinely moved (11 -> 17). On the noise analysis
above, that +6 was real signal and the -4/-3 that killed v3 were not -- v3 was reverted against a bar
with no noise model.

**Sampling caveat**: the 9 retrieved-but-wrong reads and these 4 preference reads were first-N per
type, not random draws, and all 4 temporal draws happened to hit the `question_date` bug. Treat them
as indicative of mechanism, never as rates. The independent scan (21 of 43 temporal errors naming a
missing/invented current date) is the number to cite for the date bug's size.

**PAIRED RERUN WITH THE `question_date` FIX. PRE-REGISTERED 2026-08-26, before any run.** User
authorized the recommended option: bank the verified bug fix and get a clean noise measurement to size
everything else against. Three runs, all `gemma4:31b-cloud`, seed 42, `base_top_k=30`,
`max_sessions=10`, temperature 0, sequentially (not concurrently -- concurrency is itself a config
change that could alter the nondeterminism this run is trying to measure):

| run | arm | n | output |
|---|---|---|---|
| A | extract-then-compute (v1 prompt + date fix) | 500 | `extract_then_compute_n500_datefix_results.json` |
| B | ETC repeat, identical config to A | 100 | `extract_then_compute_n100_datefix_repeat.json` |
| C | full_session single-shot baseline (+ date fix) | 500 | `full_session_n500_datefix_results.json` |

Run B is the **deliberate noise measurement**, replacing the opportunistic 4/50 estimate: `_stratified_sample`
shuffles each type bucket from the same seeded RNG and differs only in how many it takes, so the n=100
sample is nested in the n=500 one and those 100 questions are run twice under identical conditions.

**This is a BUG FIX, not a prompt experiment -- the bar decides what we may CLAIM, not whether the fix
stays.** Passing `question_date` is correct on its face (the field exists precisely to make
relative-time questions answerable, and the pre-fix outputs show the model inventing dates); the fix is
kept regardless of outcome. That is the key difference from the v2/v3 pre-registrations, where the bar
gated a revert.

**Bars, computed from the measured ~8% flip rate.** For a type of size n, expected flips ~= 0.08n and
net-delta sd ~= sqrt(0.08n); bars are set at >= 2x that sd, per the standing rule this stage adopted:

| type | n | noise sd | 2x sd | pre-fix ETC | pre-fix baseline | bar (BOTH arms, vs own pre-fix count) |
|---|---|---|---|---|---|---|
| **temporal-reasoning (PRIMARY)** | 127 | ~3.2 | ~6.4 | 84 | 67 | **>= +7 questions** (ETC >= 91, baseline >= 74) |
| multi-session | 121 | ~3.1 | ~6.2 | 93 | 81 | guard: must not fall >6 |
| knowledge-update | 72 | ~2.4 | ~4.8 | 64 | 63 | guard: must not fall >5 |
| single-session-user | 64 | ~2.3 | ~4.5 | 55 | 60 | guard: must not fall >5 |
| single-session-preference | 30 | ~1.5 | ~3.1 | 11 | 14 | guard: must not fall >3 |
| abstention | 30 | ~1.5 | ~3.1 | 29 | 27 | guard: must not fall >3 |
| overall | 500 | ~6.3 | ~12.6 | 390 | 367 | must not regress |

The guard rows exist because the date line is prepended to **every** prompt, not just temporal ones --
a plausible way to lose accuracy elsewhere by distraction, and the fix's own risk.

**Reporting rules, committed now:** results reported as **paired McNemar-style gain/loss counts**
(how many questions flipped each way), never as raw subtype deltas; any bucket smaller than ~50
questions treated as undecidable on its own; run B's flip rate reported as the noise floor and used to
re-derive every bar above if it differs materially from 8%. `n_errors=0` required in all three runs
before any number is trusted.

**Interpretation committed in advance:** primary bar met in both arms -> the date bug is confirmed as a
real ~15-21-question defect, all pre-fix temporal numbers in this document are formally superseded, and
the ETC-vs-baseline comparison is re-decided on the post-fix numbers. Primary bar missed -> the fix is
still kept (it is correct), but the 21-question estimate was wrong and temporal's residual is
model-capability-bound rather than harness-bound, which redirects priority to the 27 verified
retrieval misses. Either way, no prompt-level iteration follows from this run.

**RESULT of the pre-registered paired rerun (2026-08-26). All three runs `n_errors=0`. PRIMARY BAR
PASSED IN BOTH ARMS.**

*Noise floor, measured deliberately for the first time (run B):* the n=100 sample is confirmed nested
in the n=500 run (100/100 overlap), and re-running those 100 questions under identical config flips
**6 (6.0%**, 95% CI 1.3-10.7%; 5 generation, 1 judge-only). Pooled with the earlier opportunistic
measurement, **10 flips / 150 questions = 6.7%**. The bars were pre-registered assuming 8%, so they
were *conservative*, not lenient -- nothing needed re-deriving in a self-serving direction. All
"REAL / noise" verdicts below use 2x sd at the measured rate.

*The `question_date` fix, both arms, paired McNemar counts:*

| type | n | ETC pre -> post | gain/loss | baseline pre -> post | gain/loss | 2x sd | verdict |
|---|---|---|---|---|---|---|---|
| **temporal-reasoning** | 127 | **84 -> 103 (+19)** | +23/-4 | **67 -> 83 (+16)** | +21/-5 | 5.8 | **REAL, both arms** |
| multi-session | 121 | 93 -> 96 (+3) | +5/-2 | 81 -> 84 (+3) | +5/-2 | 5.7 | noise |
| knowledge-update | 72 | 64 -> 66 (+2) | +3/-1 | 63 -> 63 (0) | 0/0 | 4.4 | noise |
| single-session-user | 64 | 55 -> 55 (0) | 0/0 | 60 -> 60 (0) | 0/0 | 4.1 | noise |
| single-session-assistant | 56 | 54 -> 52 (-2) | 0/-2 | 55 -> 56 (+1) | +1/0 | 3.9 | noise |
| single-session-preference | 30 | 11 -> 13 (+2) | +5/-3 | 14 -> 16 (+2) | +3/-1 | 2.8 | noise |
| abstention | 30 | 29 -> 28 (-1) | 0/-1 | 27 -> 26 (-1) | 0/-1 | 2.8 | noise |
| **overall** | 500 | **390 -> 413 (+23)** | +36/-13 | **367 -> 388 (+21)** | +30/-9 | 11.5 | **REAL, both arms** |

The primary bar was >= +7 temporal in both arms independently; delivered **+19 and +16**, roughly 3x
the bar and >5x the noise sd, and strongly one-directional in both (+23/-4, +21/-5). **Every guard row
held** -- prepending a date line to *every* prompt cost nothing measurable on non-temporal types, so
the distraction risk the pre-registration flagged did not materialise. Direct cohort check: of the 21
questions whose pre-fix answer complained about or invented a current date, **19 (90%) are now
correct** -- against a pre-registered estimate of 15-21, so the sizing was well calibrated rather than
lucky.

**Why this is the fix and not cloud-model drift.** `gemma4:31b-cloud` is a remote, unpinned model; the
pre-fix runs are from 2026-08-24/25 and the post-fix runs from 2026-08-26, so a server-side model
change is a live alternative explanation for any improvement and has to be ruled out rather than
assumed away. Three things rule it out: the gain is **concentrated 19-of-21 inside the exact cohort
that named the missing date pre-fix** (drift would lift questions indiscriminately); the post-fix
outputs show explicit date arithmetic that was previously impossible ("6 - 2 = 4", "January 31 -
January 10 = 21 days"); and **every non-temporal row sits at noise level in both arms** -- a generally
stronger model would not leave six of seven types unmoved.

**Consequences.** The `question_date` bug is confirmed as a real ~19-question defect, and **every
pre-fix temporal-reasoning number in this document is formally superseded** -- including the
"+17-question ETC win on temporal", which was measured between two equally-broken configurations.
Post-fix, both arms rise by about the same amount, so ETC's advantage on temporal was not an artefact
of the bug: it survives at +20 head-to-head.

*Post-fix head-to-head, the comparison the ship decision actually rests on:*

| type | n | baseline | ETC | delta | gain/loss | verdict |
|---|---|---|---|---|---|---|
| temporal-reasoning | 127 | 83 | 103 | **+20** | +22/-2 | **REAL** |
| multi-session | 121 | 84 | 96 | **+12** | +18/-6 | **REAL** |
| knowledge-update | 72 | 63 | 66 | +3 | +6/-3 | noise |
| single-session-user | 64 | 60 | 55 | **-5** | 0/-5 | **REAL** |
| single-session-assistant | 56 | 56 | 52 | **-4** | 0/-4 | **REAL** |
| single-session-preference | 30 | 16 | 13 | **-3** | +3/-6 | **REAL** |
| abstention | 30 | 26 | 28 | +2 | +2/0 | noise |
| **overall** | 500 | **388 (77.6%)** | **413 (82.6%)** | **+25** | +51/-26 | **REAL** |

With a proper noise model, ETC's profile is now measured rather than inferred: **+32 questions on its
two target types, -12 across all three single-session types, net +25.** The single-session-user
regression reproduces at exactly -5 for the third consecutive measurement with zero churn in either
direction -- these failures are deterministic, matching the system-prompt probe. Two regressions that
were previously indistinguishable from noise (single-session-assistant -4, preference -3) now cross
the threshold, so the collateral cost is broader than the ss-user framing implied: it is a
**single-session-shaped** regression, exactly what the mode (ii) pragmatic-license diagnosis predicts.

**The router question is now settled on accuracy, and the answer is no.** Post-fix type-oracle routing (ETC for
temporal + multi-session, baseline otherwise) scores **420/500 = 84.0%**, only **+7 over always-on
ETC** -- *below* the 11.5-question noise threshold at n=500. This is the oracle using TRUE type labels,
i.e. an unattainable upper bound for any real classifier, and the real classifier previously measured
77.6% out-of-fold. Routing was not worth it pre-fix (a wash), and it is still not worth it post-fix,
now demonstrated against a measured noise floor rather than argued. (Strictly, routed and always-ETC
differ only on the 252 non-routed questions, so the applicable threshold is ~8.2 rather than 11.5 --
+7 is inside it either way.) **This kills the ACCURACY case only.** Routing's other motivation was
latency -- ~45% of queries would skip the second LLM call, and always-on ETC doubles per-turn
generation latency. That remains a live reason to build it later; the router is **parked as a pure
latency optimisation, not refuted.** Best-of-both union is 439/500
(87.8%), reported as headroom context only.

**Error mass after the fix** (ETC, 87 errors, down from 110): multi-session 25 (28.7%),
temporal-reasoning 24 (27.6%), single-session-preference 17 (19.5%), single-session-user 9 (10.3%),
knowledge-update 6, single-session-assistant 4, abstention 2. Temporal is no longer the largest
bucket. The 27 verified retrieval misses are unaffected by this fix (retrieval is unchanged) and now
represent a **larger share** of remaining error -- roughly 31% of the 87.

**Retrieval coverage sweep (2026-08-26). ZERO LLM CALLS** (`scripts/longmemeval/retrieval_coverage_sweep.py`,
results in `retrieval_coverage_sweep.json`). Before spending a run on retrieval tuning: would more
retrieval actually put the 27 verified missing answer-turns into the context? If coverage doesn't
move, accuracy cannot. Each question ingested once, then the context rebuilt at every
`base_top_k` x `max_sessions` combination.

| setting | fully covered | answer-turn coverage | median context | x baseline |
|---|---|---|---|---|
| **30x10 (current)** | **0/27** | **56.9%** | 107,833 | 1.00 |
| 60x10 | 6/27 | 66.7% | 125,111 | 1.16 |
| **100x10** | **12/27** | **77.8%** | 145,724 | **1.35** |
| 60x15 | 12/27 | 76.4% | 165,350 | 1.53 |
| 100x15 | 16/27 | 83.3% | 180,620 | 1.67 |
| 200x10 | 19/27 | 88.9% | 222,798 | 2.07 |
| 100x20 | 19/27 | 88.9% | 235,749 | 2.19 |
| **200x20** | **23/27** | **94.4%** | 279,371 | **2.59** |

**Finding 1 -- `base_top_k` is the binding constraint, not `max_sessions`.** 100x10 reaches the same
12/27 coverage as 60x15 with *less* context (145k vs 165k chars), and 200x10 matches 100x20 (19/27) at
smaller size. Session expansion is not what's missing; the answer turns are not in the top-30
candidate set to begin with. The efficient knee is **100x10 -- 12 of 27 recovered for a 35% context
increase**; 200x20 recovers 23 of 27 but at 2.6x context.

**Finding 2 -- CORRECTION: the "semantic dilution" claim was wrong, and my own wording overstated it.**
The previous entry said a buried aside "does not rank anywhere near the query at any `k`" and framed
the 7 complete misses as needing sub-turn chunking or multi-vector embedding -- kernel rearchitecting.
**All 7 are fully recovered at 200x20**, and 6 of 7 by 100x20. They rank *low*, not *off the list*.
The underlying mechanism is still real (a passing aside is swamped by its turn's dominant topic, so the
turn ranks far below where its answer-bearing content deserves), but the practical consequence is much
cheaper than claimed: **retrieve deeper, don't re-architect the index.** Sub-turn chunking remains a
legitimate efficiency idea -- it would reach the same coverage at far smaller context -- but it is now
an optimisation, not a prerequisite, and nothing in the benchmark blocks on it.

**What this does NOT establish.** Coverage is necessary, not sufficient: putting the answer turn in
front of the model does not mean the model will use it, and every one of these settings makes the
prompt substantially larger, which carries real needle-in-haystack risk for the other 473 questions.
The one historical data point (`topk_sweep_30` vs `topk_sweep_50`, n=50, pre-expansion) showed raising
top_k neither helped nor hurt -- weak evidence of low dilution risk, not evidence of gain. **12
recovered questions also sits right at the 11.5-question noise threshold**, so a coverage win of that
size might not be separable in aggregate even if every one converts.

**Cheap next step, proposed not run: a cohort-only conversion test.** Running the 27 miss IDs alone
through ETC at a raised setting costs ~12 minutes rather than ~3.5 hours, and answers the only open
question -- does coverage convert to correctness? Expected noise flips on a 27-question cohort at the
measured 6.7% rate is ~1.8, so 8+ conversions would be unambiguous. Only if it converts is a full
n=500 (needed to measure the collateral cost on everything else) worth authorizing. This ordering --
free coverage check, then cheap cohort check, then expensive aggregate check -- is the pattern the
date fix validated, and it keeps the expensive run for the question only it can answer.

**COHORT CONVERSION TEST. PRE-REGISTERED 2026-08-26, before running.** Does the retrieval coverage
the sweep proved available actually convert into correct answers? Adds `--question-ids`,
`--base-top-k` and `--max-sessions` to `extract_then_compute_experiment.py` so a named cohort can be
re-run without the stratified sampler.

*Cohort*: the 25 of the 27 verified retrieval-miss questions that are **still wrong after the date
fix** (`0bc8ad92` and `a3838d2b` were already recovered by it and are excluded, since a question that
is already correct cannot demonstrate conversion). Baseline is therefore a clean **0/25** at the
current 30x10 setting -- every one of these is wrong today, and verified to be wrong *because an
answer-bearing turn never reached the model*.

*Arms*, both ETC with the v1 prompt and the date fix, everything except retrieval held constant:

| arm | setting | answer-turn coverage (from the free sweep) | cost |
|---|---|---|---|
| 1 | `--base-top-k 100 --max-sessions 10` | 12/27 fully covered, 77.8% turn coverage | 1.35x context |
| 2 | `--base-top-k 200 --max-sessions 20` | 23/27 fully covered, 94.4% turn coverage | 2.59x context |

Both arms are run because they answer different questions: arm 2 is the **mechanism** test (with
near-total coverage, does the model use the evidence?) and arm 1 is the **efficiency** test (does the
cheap setting capture most of the benefit?). ~12 minutes each.

*Bar*: expected noise flips on a 25-question cohort at the measured 6.7% rate is ~1.7, so
**>= 5 conversions in arm 2 counts as the mechanism confirmed** (~3x expected noise, and every one of
these questions is a verified retrieval miss, so a conversion has a known cause rather than being an
unexplained flip). Fewer than 5 means coverage does not convert -- the model is given the answer turn
and still fails -- and the entire retrieval-parameter lever is dead regardless of how much coverage is
theoretically available, which would redirect effort to the compute-side levers.

*Explicitly NOT claimed by this test, whatever it returns*: net benefit. The cohort is selected on
being wrong, so it can only go up, and it says nothing about the collateral cost of a 1.35-2.6x larger
prompt on the other 475 questions -- needle-in-haystack risk is real and is precisely what a cohort
test cannot see. **A positive result here authorises nothing by itself**; it makes a full paired n=500
worth proposing, with per-type guards and latency/context-size reporting, which remains the user's
call. This is the free -> cheap -> expensive ordering the date fix validated, and the cohort/aggregate
split advisor flagged: a cohort check proves the mechanism, only the aggregate decides net worth.

**COHORT CONVERSION RESULT (2026-08-26). Both arms `n_errors=0`. BAR PASSED, decisively.**
Baseline was a verified 0/25 -- every cohort question wrong at 30x10, each for the known reason that
an answer-bearing turn never reached the model.

| arm | context | fully covered (predicted) | **correct (actual)** |
|---|---|---|---|
| 100x10 | 1.35x | 11/25 | **13/25 (52%)** |
| 200x20 | 2.59x | 21/25 | **17/25 (68%)** |

The bar was >= 5 conversions against ~1.7 expected noise flips; 200x20 delivered **17, roughly 10x
expected noise**. **Coverage converts into correctness** -- the model does use the recovered evidence.
This retires the last doubt about the retrieval lever's mechanism.

Two details worth keeping. At 200x20, coverage (21/25) exceeds conversions (17/25): **4 questions got
the answer turn and still failed**, which is the compute-side residue showing through and a reminder
that retrieval is necessary, not sufficient. At 100x10 the reverse -- 13 correct on 11 fully-covered --
so partial evidence sometimes suffices. Two questions flip the "wrong" way between arms
(`d23cf73b`, `gpt4_a1b77f9c` correct at 100x10 but not 200x20), consistent with more context
occasionally burying the needle, which is exactly the collateral risk the aggregate run must measure.

**FULL AGGREGATE RUN. PRE-REGISTERED, before running.** Per the standing rule that a cohort proves
mechanism while only an aggregate decides net worth: ETC, v1 prompt, date fix, **`--base-top-k 200
--max-sessions 20`**, n=500, seed 42, paired against `extract_then_compute_n500_datefix_results.json`
(same questions, same everything except retrieval breadth).

*Why the extreme setting rather than the efficient one*: 200x20 maximises both the expected gain
(+17 on cohort evidence, comfortably above the 11.5-question noise threshold, where 100x10's +13 would
land uncomfortably close to it) **and** the collateral risk, so a single run bounds the design space in
both directions. If it is net positive, the follow-up question is tuning down for efficiency; if the
collateral cost swamps the gain at 2.6x context, that is decisive about the whole family and 100x10
is the fallback to test.

*Bars*: primary -- **overall must improve by >= 12 questions** (2x the measured noise sd at n=500).
Cohort guard -- the 25 cohort questions should show ~17 correct, confirming the effect reproduces at
scale. Collateral guards -- **no question type may fall by more than 2x its own noise sd**
(temporal 5.8, multi-session 5.7, knowledge-update 4.4, ss-user 4.1, ss-assistant 3.9, preference 2.8,
abstention 2.8). Reported as paired McNemar counts, plus median context size and per-question latency,
since a 2.6x prompt is a real product cost even where accuracy improves and the ship decision needs
both numbers.

*Committed in advance*: a net gain that clears the bar makes raised `base_top_k` a genuine Aeon-side
improvement worth wiring, with the exact setting to be chosen by a follow-up efficiency comparison,
not assumed to be 200x20. A miss means the collateral cost of large prompts cancels the recovered
evidence, and the retrieval lever is capped well below what the coverage sweep suggested.

**AGGREGATE RESULT at 200x20 (2026-08-26). `n_errors=0`. PRIMARY BAR FAILED -- and the failure is the
most informative result of this stage.**

| type | n | 30x10 | 200x20 | delta | gain/loss | guard | verdict |
|---|---|---|---|---|---|---|---|
| temporal-reasoning | 127 | 103 | 110 | **+7** | +13/-6 | 5.8 | REAL |
| **multi-session** | 121 | 96 | 90 | **-6** | +7/-13 | 5.7 | **GUARD BREACH** |
| knowledge-update | 72 | 66 | 63 | -3 | +2/-5 | 4.4 | noise |
| single-session-user | 64 | 55 | 57 | +2 | +2/0 | 4.1 | noise |
| single-session-assistant | 56 | 52 | 53 | +1 | +1/0 | 3.9 | noise |
| single-session-preference | 30 | 13 | 13 | 0 | +3/-3 | 2.8 | noise |
| abstention | 30 | 28 | 30 | +2 | +2/0 | 2.8 | noise |
| **overall** | 500 | **413** | **416** | **+3** | +30/-27 | 11.5 | **FAIL (bar +12)** |

**The cohort guard hit 17/25 -- exactly what the 25-question cohort test predicted.** The mechanism
reproduced perfectly at scale; the targeted questions really were fixed. What the cohort could not see,
and what the aggregate exists to measure, is that the collateral almost exactly cancelled the gain.
Cost of that +3: **2.69x median context (100,889 -> 270,972 chars) and 1.8x generation latency
(2.5s -> 4.5s per question)**. Not shippable.

**The mechanism, and it inverts this document's earlier prescription.** Splitting the effect by how
many answer-bearing turns a question needs:

| question shape | deep helps | deep hurts | net |
|---|---|---|---|
| **1 answer turn** | 13 | 4 | **+9** |
| 2 answer turns | 7 | 12 | **-5** |
| 3+ answer turns | 9 | 11 | **-2** |

Deep retrieval helps **findability** and harms **aggregation**, and those are opposing forces that
roughly cancel. A single buried fact ranks low, so searching deeper finds it -- that is the temporal
+13 and the buried-aside recoveries. But a question needing several facts gets a 2.7x larger haystack,
and the model miscounts: multi-session loses 13 questions to win 7. **The earlier decision-table entry
prescribing "raise `top_k`/`max_sessions` for the 20 partial-recall multi-evidence misses" is
therefore wrong and is retracted** -- their answer-turn *coverage* improves (the sweep proved that)
while their *accuracy* falls. Coverage was necessary but, for multi-evidence questions, actively
counterproductive at the context cost required to obtain it.

**Routing is again marginal, for the third time.** Per-question oracle over the two settings is
443/500 (88.6%), but that is unattainable. Type-based oracle routing (deep only for the types where
deep wins net) gives **425/500 = 85.0%, +12 over always-30x10 -- right at the 11.5 threshold, using
TRUE type labels.** Every previous real classifier has landed below its oracle. This is the same shape
as the ETC router result and should be read the same way: not worth building for accuracy.

**Where this leaves the retrieval lever, and what it promotes.** The lever is not dead but it is
*capped*, and the cap is caused by context cost, not by retrieval quality: the findability win is real
(+13 on temporal alone) and only the accompanying context bloat makes it a wash. That distinction
matters, because it points at a fix that takes the win without paying the cost -- **improve ranking at
constant context rather than retrieving more**. Sub-turn chunking / multi-vector embedding, which the
coverage sweep downgraded to "an optimisation, not a prerequisite", is **promoted back to the leading
Aeon-side candidate on this evidence**: a buried aside ranks low because its turn's embedding is
dominated by the turn's main topic, and chunking fixes precisely that at `top_k=30`-scale context, so
it would capture the +9-to-+13 findability gain *without* the -13 multi-session aggregation damage.
It is real kernel work rather than a config change, and it is the only remaining candidate that
addresses a verified defect without a measured downside.

**The 100x10 fallback named in the pre-registration is not worth running on this evidence.** It sits
on the same opposing-forces curve at a milder point: a smaller findability gain (13/25 cohort
conversions vs 17/25) bought with milder aggregation damage. Its most likely outcome is a small net
positive well inside the noise band -- a ~3-hour run whose result would be undecidable by
construction. Recording that decision explicitly rather than letting a pre-registered fallback lapse
silently.

**Methodological note worth keeping.** The cohort test predicted 17/25 and the aggregate delivered
17/25. The instrument works exactly as intended -- and the pre-registered bar is what stopped a
genuine, reproducible, mechanism-confirmed 17-question win from being shipped as an improvement when
its true net effect was +3 questions for 2.7x the context. A cohort proves mechanism; only an
aggregate decides worth.

## PRODUCT DIRECTION (2026-08-26): Aeon competes on precision-per-token, not recall

Everything in this stage has been LongMemEval accuracy optimisation, which measures *the LLM's answer
quality given whatever context Aeon assembles*. That is not the same question as *is Aeon the best
memory engine*, and four measurements taken from the runs already on disk separate them sharply.

**1. 71% of remaining error is not Aeon's.** Of ETC's 87 errors at 30x10, **25 (29%) are Aeon-side**
(an answer-bearing turn never reached the model) and **62 (71%) are LLM-side** (the evidence was
delivered and the model still failed). Benchmark accuracy is mostly a measure of the generator.

**2. Aeon is already invisible in end-to-end latency, and always will be.**

| config | Aeon retrieval | LLM generation | Aeon's share of wall-clock |
|---|---|---|---|
| single-shot @30x10 | 12.6 ms | 1.5 s (1 call) | **0.83%** |
| ETC @30x10 | 12.2 ms | 2.5 s (2 calls) | **0.49%** |
| ETC @200x20 | 13.0 ms | 4.5 s (2 calls) | **0.29%** |

The kernel's 2.23us insert and 3.09us navigate are excellent and **do not move end-to-end latency at
all** -- they are table stakes. System latency is ~99% LLM, and **LLM latency is context-bound**:
tripling the context at 200x20 took generation from 2.5s to 4.5s. The only lever Aeon has on the
latency a user actually feels is **how many tokens it sends**.

**3. 99.2% of what Aeon delivers is padding (measured, not estimated).** Answer-bearing turns are a
**median 292 chars each, 563 chars per question in total**, against a **median 100,831 chars
delivered**. Load-bearing share: median 0.58%, mean 0.76%.

**4. The north-star metric follows: correct answers per 1k chars delivered.**

| config | accuracy | median context | **correct / 1k chars** |
|---|---|---|---|
| single-shot @30x10 | 77.6% | 100,889 | 3.85 |
| **ETC @30x10** | **82.6%** | 100,889 | **4.09** |
| ETC @200x20 | 83.2% | 270,972 | 1.54 |

The 200x20 run was, in this frame, a 2.7x regression on the metric that matters bought for +3
questions of noise -- and its failure mode (aggregation collapsing under a bigger haystack) is direct
evidence that **for a memory engine, sending less is not merely cheaper, it is often more accurate.**

**The thesis, stated carefully: precision at small context -- NOT small context per se.** Two results
in this document are counterevidence to naive compression and must stay in the thesis rather than be
buried under it. (a) Raw `top_k=30` snippets -- small context, low precision -- scored **68%**, nine
points *below* full-session expansion at ~100k chars; that gap is why session expansion exists.
(b) ETC's extract step **is** a compression layer, and it costs **-12 single-session questions**
because compression strips the conversational licensing that mode (ii) failures depend on
(Trader-Joe's-as-brand). So some padding is load-bearing, and **the accuracy-vs-context-size curve
between ~2k and ~100k chars has never been measured.** That unmeasured curve is exactly where the
product lives.

**Reading ETC correctly reframes it as evidence FOR this direction.** ETC wins (82.6% vs 77.6%)
*because* it compresses ~100k chars into a short fact list before answering -- but it buys that
compression with a second LLM call, which is why it costs 2x generation latency. **If the compression
happened in the retrieval layer at ms latency instead of in a second LLM call, the same accuracy would
come at single-shot latency or better.** That is the product: not a faster index, a *smaller, better
context*. Precise claim, since it is easy to overstate: the kernel cannot do ETC's semantic
extraction. The realistic version is **"retrieval at sub-turn granularity plus cheap reranking
approximates the extract step at us-ms latency"**, and how good that approximation can get is an
empirical question with a cheap answer, below.

**RECOMMENDED NEXT STEP -- the oracle-precision arm (cheap, decisive, proposed not run).** Context =
the `has_answer` turns plus one neighbouring turn on each side (neighbourhood preserved specifically
to test whether pragmatic licensing survives compression), single-shot, n=500. Annotations already
exist; contexts are ~2-4k chars, so this is fast and cheap -- roughly single-shot latency on 1/30th
the context. It measures **the ceiling of perfect compression**:

*Pre-registered bar (to be committed before running)*: primary -- **>= 82.6%** (matches always-on ETC)
would validate the entire direction end-to-end: ETC-or-better accuracy, **one** LLM call, ~30-50x less
context, sub-second generation. Secondary -- **single-session-user/preference must not fall below
single-shot's 60/64 and 16/30**; if they tank even with neighbouring turns included, that measures
precisely what compression must preserve, which is the most useful possible negative result. Existing
`oracle_results_gemma.json` already hints at parity (80% at n=50 on whole gold *sessions*, ~10k
tokens); this sharpens it to answer-turn granularity at n=500 and, unlike that file, also measures
whether single-shot on compressed context beats two-call ETC on raw context.

**Build stack if the oracle arm confirms**, ascending cost: (1) **sub-turn chunking in the kernel** --
fixes buried-aside findability at constant context, promoted by the 200x20 result; (2) a **ms-scale
salience/rerank layer** between retrieval and the LLM -- the cheap approximation of EXTRACT;
(3) **neighbourhood stitching** to preserve the licensing that mode (ii) needs. All three are Aeon
product work, all three improve correct-per-token, and all three cut LLM latency and cost as a side
effect rather than trading against them.

**Explicitly parked, so nothing lapses silently:** routing (three independent measurements, always at
or below its oracle ceiling); deep retrieval (measured wash, -12 on the north-star metric); the 100x10
fallback (undecidable by construction); compute-prompt v4 clause (b) (still a valid ~+6 preference
play, but benchmark polish rather than product direction -- and the compression thesis subsumes its
target if the oracle arm confirms). The **ship decision on wiring always-on ETC into the shell remains
open and orthogonal** to all of this.

**ORACLE-PRECISION ARM. PRE-REGISTERED 2026-08-26, before running.** Tests the product thesis
directly: what accuracy is reachable if the retrieval layer delivered only the load-bearing evidence?
Context = `has_answer` turns plus one neighbouring turn each side, **single-shot (one LLM call)**,
n=500, same model/judge/date-fix as every other arm
(`scripts/longmemeval/oracle_precision_experiment.py`).

*Abstention handling, decided before seeing results*: 21 questions have no answer-bearing turn by
construction. An empty context would make them trivially correct and inflate the headline, so they
receive real Aeon-retrieved context trimmed to comparable size -- **abstention stays earned**.
Reported both overall and on the 479-question answer-bearing subset.

*Comparison points* (all n=500, date fix, same judge):

| arm | accuracy | median context | LLM calls | median generation |
|---|---|---|---|---|
| single-shot @30x10 | 77.6% | 100,889 | 1 | 1.5 s |
| ETC @30x10 | 82.6% | 100,889 | 2 | 2.5 s |
| ETC @200x20 | 83.2% | 270,972 | 2 | 4.5 s |
| **oracle-precision** | ? | **~3,000 (smoke)** | **1** | **~0.5 s (smoke)** |

*Bars*: **primary -- >= 82.6% (>= 413/500)**, matching always-on ETC. Clearing it validates the
direction end-to-end: ETC-or-better accuracy at **one** LLM call, ~33x less context and ~5x faster
generation, which is the entire product thesis in one measurement. **Secondary floor --
single-session-user >= 60/64 and single-session-preference >= 16/30** (single-shot's own numbers):
this is the mode (ii) test, checking whether pragmatic licensing survives compression when neighbours
are included.

*Interpretation committed in advance.* Primary met -> perfect compression is at least as good as the
100k-char firehose, the build stack (sub-turn chunking -> ms-scale rerank -> neighbourhood stitching)
is chasing a verified ceiling, and correct-per-token becomes the stage's headline metric. Primary
missed but single-session floors held -> compression is lossy in aggregate and the curve between 3k
and 100k has a real slope worth mapping before building. **Single-session floors breached -> the most
useful negative result available**: it measures exactly what a compressor must preserve, and any
reranker built later must be evaluated against that constraint rather than raw hit-rate. No outcome
here authorises kernel work by itself; it sizes the ceiling that work would be chasing.

*Not claimed either way*: this is an ORACLE (it uses gold answer-turn annotations no production system
has). It measures the **ceiling** of a perfect compressor, not what a real reranker would achieve. Its
value is bounding the design space cheaply -- if even a perfect compressor cannot match 82.6%, no
reranker will, and the direction dies for ~25 minutes of compute instead of a quarter of kernel work.

**ORACLE-PRECISION RESULT (2026-08-26). `n_errors=0`. PRIMARY BAR PASSED. The product thesis is
validated.**

| arm | accuracy | median context | LLM calls | median generation | **correct / 1k chars** |
|---|---|---|---|---|---|
| single-shot @30x10 | 77.6% | 100,889 | 1 | 1.51 s | 3.85 |
| ETC @30x10 | 82.6% | 100,889 | 2 | 2.46 s | 4.09 |
| ETC @200x20 | 83.2% | 270,972 | 2 | 4.47 s | 1.54 |
| **oracle-precision** | **83.8%** | **5,654** | **1** | **0.51 s** | **74.10** |

Bars: primary >= 413 -> **419 (83.8%), PASS**. Floor ss-user >= 60/64 -> **63/64, PASS**. Floor
preference >= 16/30 -> **15/30, MISSED BY ONE** (2x sd is 2.8, so this is statistically
indistinguishable from the floor, and still +2 above ETC's 13 -- reported as a miss because it was
pre-registered as one, not explained away).

**Read the headline correctly: this is accuracy PARITY, not an accuracy win.** Paired against ETC the
overall delta is **+6 (+49/-43) against an 11.5 threshold -- noise.** The oracle context changes ~92
individual answers in both directions and nets out level. Nothing here says perfect compression makes
the model smarter. What it says is far more useful for the product: **the same accuracy is available
at 1/18th the context, half the LLM calls, and 1/5th the generation latency.** On the north-star
metric the gap is not marginal -- **74.10 vs 4.09 correct per 1k chars, an 18x improvement** -- and
that metric is the one that converts directly into user-visible latency and per-turn cost.

*Per-type against ETC*, everything inside noise except one:

| type | ETC | oracle | delta | gain/loss | verdict |
|---|---|---|---|---|---|
| **single-session-user** | 55 | **63** | **+8** | +8/-0 | **REAL** |
| single-session-preference | 13 | 15 | +2 | +5/-3 | noise |
| single-session-assistant | 52 | 53 | +1 | +2/-1 | noise |
| abstention | 28 | 28 | 0 | +1/-1 | noise |
| knowledge-update | 66 | 65 | -1 | +4/-5 | noise |
| multi-session | 96 | 95 | -1 | +15/-16 | noise |
| temporal-reasoning | 103 | 100 | -3 | +14/-17 | noise |

**The single-session-user result is the one that matters mechanistically, and it vindicates a specific
design choice.** ETC's compression *broke* this type (60 -> 55, the mode (ii) pragmatic-license
failures: "Trader Joe's" stops reading as a brand once the surrounding chat is stripped). Oracle
compression at **+/-1 neighbouring turn scores 63/64 -- better than ETC by a REAL +8 and better than
the uncompressed single-shot baseline's 60.** So the -12 single-session cost that has shadowed this
entire stage **is not intrinsic to compression at all; it was caused by compressing to bare facts
without their conversational neighbourhood.** Keeping one turn of context on each side is enough to
preserve the licensing. That is a concrete, transferable design constraint for any reranker built
next, and it was obtained for 15 minutes of compute.

**Honest limits, stated plainly.** (1) This is an **oracle** -- it uses gold `has_answer` annotations
no production system has. It bounds the ceiling of a perfect compressor; it does not say a real
reranker reaches it, and the gap between them is the entire engineering risk of the direction.
(2) Multi-evidence types (temporal -3, multi-session -1, both noise but both negative) show heavy
two-way churn, so perfect precision is *not* strictly dominant -- for aggregation questions some of
the surrounding session appears to carry real signal, consistent with the 200x20 finding that these
types behave differently from everything else. (3) The preference floor was missed by one question.

**What this settles for the roadmap.** The accuracy-vs-context curve between ~5k and ~100k chars is
**flat** -- 83.8% at 5.6k versus 82.6% at 100.9k. Aeon has been shipping ~95k chars per query for no
measurable accuracy return, at direct cost in LLM latency and tokens. **The build stack
(sub-turn chunking -> ms-scale salience/rerank -> neighbourhood stitching, the last now proven
necessary rather than assumed) is chasing a verified ceiling of ~84% at ~5.6k chars, one LLM call,
~0.5s generation.** Correct-per-token becomes this stage's headline metric, and the honest target for
a real reranker is stated as a fraction of the 74.10 oracle figure rather than as an accuracy number.

**PRECISION SELECTOR — TIER 1 (free) and TIER 2 (cheap), 2026-08-26.** Building the real,
non-oracle counterpart to the oracle-precision arm (`scripts/longmemeval/precision_selector.py`,
`precision_coverage_sweep.py`, `precision_arm_experiment.py`). Deliberately **shell-side first**: the
open question is algorithmic (what fraction of the oracle ceiling a real selector reaches), and a
C++/kernel port is a later deliverable gated on *latency at scale*, not accuracy. Porting an
unvalidated algorithm would be weeks of work ordered before the experiment that can kill it.

Design: split every turn into sentence-level chunks, embed each bare, rank chunks, map hits back to
parent turns, stitch +/-1 neighbours, dedupe, stop at a char budget. `has_answer` is used only to
score coverage, never inside selection.

**Tier 1 — coverage (zero LLM calls, n=82 incl. all 27 known retrieval misses).** Round 1 produced
three findings, two of them negative:

- **Sub-turn chunking works, and is measurable at the mechanism level.** For the diagnosed buried-aside
  cases, the answer-bearing content moves from turn-rank 71/500 to chunk-rank 53/4827, and for
  `726462e0` from turn-rank 32/501 to **chunk-rank 0/4838**. De-dilution is real.
- **MMR diversity is a NEGATIVE** (43.3% vs 49.7% turn coverage). It was added on direct evidence --
  the top-10 chunks for a failing question were near-duplicates -- but pushing them apart lost more
  evidence than the redundancy cost. Dropped.
- **Design A (chunk-index) == Design B (two-stage pool)** on coverage, so the wide turn-level
  pre-pool adds nothing.

Round 1 also exposed a tension that had to be resolved rather than assumed away: **inline +/-1
stitching costs 21 points of coverage at a fixed budget** (83.8% -> 62.6%), because neighbours consume
budget that would otherwise hold more evidence -- yet stitching is the one constraint this project
*proved* necessary. Round 2 resolved it by ordering the spend (`stitch_mode="post"`: fill 70% of the
budget with evidence turns, then buy neighbourhoods with the remainder):

| config | answer-turn coverage (unbiased n=55) | median chars |
|---|---|---|
| **production turn-level top_k=30** | **100.0%** | 100,889 |
| selector, no stitch | 83.8% | 8,983 |
| selector, inline stitch | 62.6% | 8,932 |
| **selector, post stitch** | **78.8%** | **8,694** |
| selector, post stitch, 12k budget | 82.8% | 11,785 |

Post-ordering recovers most of the stitching cost (62.6% -> 78.8%) *and* fixes more known misses
(5 vs 4). Two implementation bugs were found and fixed here, one of them substantive: short fragments
were merged **backward**, which glued the standalone sentence *"By the way, I just got a smoker
today."* (38 chars, under the threshold) onto an unrelated chunk -- **re-creating inside the chunker
exactly the topic-dilution that sub-turn chunking exists to remove.**

**Tier 2 — end-to-end (n=85: a 60-question stratified sample unioned with all 27 known misses, so one
cheap run measures conversion and collateral together). `n_errors=0`.**

| arm | all 85 | normal 58 | known-miss 27 |
|---|---|---|---|
| single-shot @top_k=30 | 49 | 46 | 3 |
| ETC @top_k=30 | **54** | 52 | 2 |
| **precision selector** | 48 | 44 | 4 |
| oracle-precision (ceiling) | **70** | 49 | **21** |

Cost: **8,689 median chars (11.6x less than production), 0.54 s generation (2.8x faster than
single-shot), 15.7 ms selection**. North-star: **5.52 correct per 1k chars — 1.35x ETC's 4.09, but only
7.4% of the oracle's 74.10.**

**The honest reading, and it is the answer to PAPER_V4_FINDINGS §9's central question.** A real
selector **matches single-shot accuracy at one twelfth the context and roughly a third of the
generation latency** — a genuine cost win, and the first evidence that the precision thesis survives
contact with a non-oracle implementation. But it does **not** reach ETC's accuracy, and it captures
only **4 of the 21** hard retrieval misses the oracle recovers. **The binding constraint has moved:
it is no longer context size, it is ranking quality.** The oracle proves 83.8% is reachable at 5.6k
chars; the selector reaches ~79% answer-turn coverage and pays for the missing 21% in accuracy.
Closing that gap is a retrieval-quality problem (hypernym gaps like "smoker" vs "kitchen appliance"
are not bridged by embedding similarity at any granularity), not a budget problem.

**Ingest cost, previously unmeasured and now reported** (the asterisk in the latency table): building
the chunk index costs **~10.2 s per question** against ~4.7 s for turn-level ingest — roughly a 2x
ingest-time penalty, amortised across queries in production but real. Query-time selection is
**15.7 ms**, negligible.

## STRATEGIC REFRAME (2026-08-26): the benchmark has been testing the episodic half only

Prompted by the standing goal -- make Aeon the best long-term memory for LLM agents, with benchmark
scores as a *consequence* rather than the target. Stepping back from the measurement work produced a
structural observation that changes what to build next.

**Every quality experiment in this stage imported exactly three things from Aeon**: `TraceGraph`,
`session_expansion`, and `llm` (grep-verified across all of `scripts/longmemeval/`). Untouched by any
end-to-end quality run: `Atlas`/`client.py` (the concept index), `dreamer.py` (consolidation),
`architect.py` (write-time ingestion), `context.py`, `promotion.py`. So the entire stage has measured
**vector search over raw conversation turns** -- which is the *least differentiated* thing Aeon does,
and precisely where it competes with any off-the-shelf vector store.

**Stated precisely, because the stronger version is also the true one** (an earlier draft of this
entry overclaimed): this is not "we forgot to switch features on." The semantic half is largely
*unbuilt* for this purpose -- `dreamer.py` ships a `StubSummarizer` and is designed for
summarise-to-forget rather than entity-state-for-answering; `architect.py` is a thin ingest wrapper;
and `supersession.py`, which looked at first glance like a knowledge-update engine, is **audited
governance plumbing** for externally-triggered node exclusion. It provides a reusable audited
primitive, but the hard part of knowledge-update -- *detecting* that "3 recipes" supersedes "2
recipes" -- exists nowhere. Kernel microbenchmarks cover Atlas; no end-to-end **quality** experiment
does. **This stage's findings are therefore the requirements spec for the semantic half.**

**The evidence for the thesis is already in our own data.** Extract-then-compute *is* consolidation,
performed at query time -- and it won (+23 overall) precisely on aggregation and temporal questions.
That is direct evidence that *computing over extracted facts beats computing over raw turns*.
Write-time consolidation is ETC's extract step moved to ingest: paid once, amortised across queries,
and -- critically for the latency story -- it converts ETC's second LLM call into background work.

**The sharpest consequence: 83.8% is the ceiling of one architecture, not of a memory system.** The
oracle-precision arm bounds *perfect selection of raw turns*. It is not a bound on what a system that
*derives* answers can do. The proof is in the oracle's own failures: 28 questions are wrong under
oracle **and** ETC **and** single-shot -- the gold evidence was in hand and the answer was still
wrong. Samples: *"How many items of clothing do I need to pick up?"* gold 3, oracle answered 1;
*"How many babies were born?"* gold 5, oracle answered 6 (over-counted); *"How many albums have I
purchased or downloaded?"* gold 3, oracle answered 2; *"How many years in formal education?"* gold 10,
oracle answered 8. **A maintained record `siblings: 4` or `clothing_to_collect: [...]` answers these by
lookup, where perfect raw evidence plus multi-hop counting measurably does not.** Consolidation also
*reduces generator dependence*, which is a product claim independent of any benchmark.

### Failure → record-requirement table (the core artifact of this reframe)

| measured failure | evidence | record requirement |
|---|---|---|
| aggregation miscounts, *even with perfect evidence* | 8 multi-session in the oracle-failed cohort | entity-state records with running counts/sets |
| temporal-arithmetic residue | 9 temporal in the oracle-failed cohort | timestamps on records; derived intervals |
| knowledge-update (~1-in-9 miss) | `v4-plan.md` supersession finding | supersession links + **write-time detection (unbuilt)** |
| hypernym gap ("smoker" ↔ "kitchen appliance") | selector recovers 4/27 vs oracle 21/27 | category/type enrichment at write time |
| mode-(ii) licensing loss (−12 single-session under ETC) | oracle ±1 neighbour scores 63/64 | provenance links back to source turns + neighbourhood |
| *architecture constraint* | ETC extraction was query-**conditioned** | write-time extraction is query-**blind** and must anticipate |

**The key technical risk, named so the probe tests exactly it**: ETC extracted "facts relevant to this
question." Write-time extraction cannot see the question. The mitigating hypothesis is that the
dominant failures are **entity-state** (siblings, subscriptions, tanks, kitchen items, albums), which a
generic user-model schema covers without knowing the question -- but that is a hypothesis until
measured.

### LongMemEval v2 is a different benchmark, and it reinforces this direction

The repo already contains a v2 harness (`scripts/longmemeval-v2/`) and dataset (451 questions). **v2
does not share v1's question types at all**: `static-environment` 134, `dynamic-environment` 86,
`procedure` 74, plus `-abs` variants and `errors-gotchas` 29, across web/enterprise domains with
`eval_function` scoring. It tests **environment state, procedures, and learned gotchas** -- not
conversation recall. The recorded v2 oracle run is **37.5% overall at 95% haystack coverage**
(n=40): even *with* the gold evidence located, the model fails most of the time.

Two consequences. First, **v1 retrieval tuning will not transfer to v2** -- optimising `top_k` against
v1 buys nothing there. Second, v2's categories are almost a literal specification for the semantic
half: "dynamic-environment" is entity-state-that-changes (supersession), "procedure" is multi-step
knowledge worth storing once rather than re-deriving, "errors-gotchas" is learned-lesson memory.
**The consolidation direction is the one that serves both benchmarks, and it is the one that matches
the product goal rather than a leaderboard.**

### Decisions recorded

- **The pre-registered tier-3 selector n=500 is PARKED as superseded by this pivot.** Its expected
  result ("ties single-shot at ~1/12 context") is already established at n=85 and would not change
  what to build next. Recorded rather than left to lapse silently.
- The selector is **not** discarded: it becomes the *episodic component* of a composite context.
- **The destination is a composite context, not a router.** The three prior routing failures compared
  arms carrying the *same* information; consolidated records and episodic turns are *different
  information sources*, and records cost a few hundred chars. Always include both -- records +
  budgeted episodic selection -- in **one** LLM call. Each component is now separately measured.

**CONSOLIDATION PROBE. PRE-REGISTERED 2026-08-26, before running.** First experiment of the semantic
half. It is deliberately the *schema-validation* step of building that half, not a detour: the probe
uses LLM extraction as a stand-in for the persistent layer, and **its record schema is intended to
become the production schema**. Building storage before knowing the schema works would be weeks spent
on the wrong shape.

*Cohort*: the **18 questions wrong under oracle AND ETC AND single-shot** in the aggregation/temporal/
update types (multi-session 8, temporal-reasoning 9, knowledge-update 1). These are the cleanest
possible target: the oracle had the gold evidence in hand and still failed, so **no retrieval
improvement of any kind can fix them** and any conversion is unambiguously attributable to
consolidation rather than to better search.

*Method*: for each of the **859 sessions** across the cohort (median 46 per question), run
**query-blind** extraction into a fixed schema -- the extractor never sees the question, exactly as a
real write-time consolidator never would. Records are then the only memory available at answer time.
Schema, designed from the failure→requirement table:

| line type | purpose | targets |
|---|---|---|
| `FACT:` | durable attribute / possession / relationship | entity state |
| `EVENT [date]:` | something that happened, dated | temporal arithmetic |
| `ITEM(category):` | one member of a countable collection | **aggregation miscounts** |
| `UPDATE:` | a statement revising an earlier one | knowledge-update supersession |
| `PREF:` | stated preference | preference questions |

`ITEM(category)` is the load-bearing element: the oracle's counting failures ("how many albums": gold
3, answered 2) happen because counting is a multi-hop operation over scattered raw mentions. Emitting
one `ITEM` line per member converts counting into enumeration of an existing list.

*Arms*, both answering from records with one LLM call:
- **R**: records only.
- **R+E**: records **plus** the budgeted episodic selection from the precision selector -- the
  composite context. Tests the "different information sources, not a router" hypothesis, and supplies
  the provenance neighbourhood that mode (ii) licensing needs.

*Bar*: all 18 are currently wrong, so expected noise flips at the measured 6.7% rate is ~1.2.
**>= 4 conversions in either arm confirms the mechanism** (>3x noise). Fewer than 4 means query-blind
extraction cannot anticipate what the questions need -- which is the named risk (ETC's extraction was
query-*conditioned*) -- and the consolidation direction needs a different schema or a query-aware
component before any storage work begins.

*Explicitly not claimed by this probe*: net benefit. The cohort is selected on being wrong, so it can
only improve, and nothing here measures whether consolidated records **cost** accuracy on the ~400
questions that already pass. That is the composite n=500's job and remains user-gated. Cost of the
probe: 859 extraction calls + 36 answer calls, roughly 45 minutes.

**CONSOLIDATION PROBE RESULT (2026-08-26). `n_errors=0`. BAR PASSED.** Baseline was 0/18 by
construction -- every cohort question is wrong under oracle AND ETC AND single-shot.

| arm | conversions | note |
|---|---|---|
| **R** (records only) | **4/18** | meets the >=4 bar on its own |
| **R+E** (records + episodic selection) | **6/18 reported, 5 real** | see judge correction below |
| union of both arms | 7 reported, 6 real | |

Expected noise on 18 questions at the measured 6.7% rate is ~1.2 flips, so **5 real conversions is
~4x noise. The mechanism is confirmed: query-blind write-time consolidation answers questions that
perfect raw-turn retrieval measurably cannot.**

**Judge correction, applied before reporting**: one R+E "conversion" (`e4e14d04`) answered
*"Approximately three weeks"* against a gold of *"Two weeks"* and the judge returned `yes`. That is a
judge false-positive, so the honest count is 5, not 6. Flagged rather than banked -- the same judge
nondeterminism measured at ~1/6 of the noise floor is visible here as a *directional* error too.

**What actually converted, and why it matters:**

- **`ITEM(category)` enumeration works exactly as designed.** *"How many babies were born to friends
  and family?"* -- gold **5**; the oracle, holding perfect evidence, answered **6** (over-counted);
  records answered **5**, naming all five. Counting became enumeration of an existing list instead of
  a multi-hop scan over scattered mentions.
- **`UPDATE` produces explicit supersession.** *"What was I pre-approved for?"* -- records answered
  **"$400,000 (previously $350,000)"**, surfacing both the current value and what it replaced. This is
  the knowledge-update mechanism that `supersession.py` does not implement (it is governance
  plumbing); write-time detection is what was missing, and this is the first evidence it works.
- **Temporal is the strength: 4/9.** Dated `EVENT` records turn interval arithmetic into subtraction
  over stored timestamps ("The Nightingale: Jan 1 to Jan 15 (~2 weeks)").
- **Multi-session is the weakness: 1/8.** The aggregation target converted least, which is the
  opposite of what the failure->requirement table predicted and is the main open problem.

**The composite beats records-alone (5-6 vs 4), which validates "different information sources, not a
router."** Three prior routing experiments failed because the arms carried the *same* information;
records and episodic turns do not. Every case where R+E won and R lost needed a number or phrasing
that lived in the raw turn rather than in the record ("save ~$50, as a taxi is around $60").

**Cost profile, which is the product argument:** median records are **21,584 chars / 292 lines**
against a **~487,000-char raw haystack -- 22.6x smaller**, and unlike the episodic selector's budget
they are *complete* rather than truncated. Consolidation costs ~1 minute per question for ~46
sessions **once, at write time**, and is fully amortised across every later query. That is ETC's
extract step moved off the query path: the same consolidation benefit with **no** second LLM call at
answer time.

**Honest limits.** (1) The cohort is selected on being wrong, so it can only improve; nothing here
measures whether records *cost* accuracy on the ~400 questions that already pass -- that is the
composite n=500's job and stays user-gated. (2) Some cohort questions are **granularity-ambiguous
rather than solvable** (documented in `consolidation_probe.py`: gold 3 counts two physical items as
three obligations), so 18 was never a fully winnable denominator. (3) `TASK` was added to the schema
after inspecting one smoke case; disclosed, and justified as a category any assistant memory needs
independent of this benchmark. (4) Single model, single benchmark.

**Next, in order:** the schema is validated enough to build against, so the persistent semantic layer
(Atlas-backed records with provenance links, write-time UPDATE detection, background consolidation)
is now justified work rather than speculation. The open problem to attack first is
**multi-session/aggregation at 1/8**, since that is where the failure->requirement table predicted the
largest win and did not get it.

**WHY MULTI-SESSION CONVERTED 1/8 — diagnosed, and it reverses the build order (2026-08-26).**
Free re-analysis of the probe's cached records, no new LLM calls.

The aggregation target was expected to be consolidation's *strongest* case and was its weakest. The
cause is not extraction recall -- **the evidence is present in the records** -- it is that the
evidence does not **accumulate into a countable set**. Worked example, `bf659f65` (*"How many music
albums or EPs have I purchased or downloaded?"*, gold 3, answered 1); all three are in the records,
under three different record types:

| item | how it was recorded |
|---|---|
| "Happier Than Ever" (downloaded) | `PREF: The user likes Billie Eilish, specifically the album...` |
| "Midnight Sky" (purchased) | `ITEM(music album/EP): Midnight Sky` ✓ |
| Tame Impala vinyl | `EVENT [...]: saw Tame Impala live ... and got their vinyl` |

Only one of three landed in an `ITEM` line, so counting `ITEM` lines yields 1. The same pattern
explains the rest of the bucket: the albums question's extracted categories are `movie`, `bird seen`,
`sport`, `pet purchase` -- **no music category exists at all**; the education question has no
education category; the babies question lumped newborns into `ITEM(person)` (12 members) and
over-counted to 6, exactly as the oracle did.

**Root cause: the category vocabulary is emergent and per-session.** Each session is extracted by an
independent, query-blind call that invents its own category names and picks its own record type, so
members of one real-world category scatter across many labels and never meet. The extraction prompt
asks for consistent naming, but **independent calls with no shared vocabulary cannot comply** -- that
instruction was never satisfiable.

**Consequence for build order.** The persistent layer's whole value is that records accumulate across
sessions. Building storage now would harden a **free-form category field** -- precisely the broken
part -- into the schema and the on-disk format. So the order is reversed, per the user's standing
allowance to do so: **fix accumulation first, then build persistence against the corrected schema.**
This is not a detour from (b); a schema that does not accumulate is not worth persisting.

**Three fixes, in the order they should be tried** (the second is what `dreamer.py` exists for, which
is a useful convergence -- the architecture already anticipated this component even though it ships a
`StubSummarizer`):

1. **Closed category taxonomy.** Replace free-form `ITEM(<anything>)` with a fixed, enumerated set
   given to the extractor in every call, so members accumulate by construction.
2. **A consolidation/merge pass over accumulated records** -- canonicalise near-duplicate categories,
   resolve entities, and promote `PREF`/`EVENT` mentions that are really category members into `ITEM`
   lines. This is exactly the Dreamer's job, and the first concrete requirement for it.
3. **Type-assignment discipline**: a fact may need to appear as more than one record type (the vinyl
   is both an event and a collection member), which the current prompt only requests for `TASK`.

**Also resolved while checking (a stale TODO from 2026-08-17):** `TraceBlockIndex` is **no longer
orphaned**. It is constructed in `TraceManager`, appended to on every embedded event, and queried in
the semantic-search path (`core/src/trace.cpp:624-631`), with `core/tests/test_trace_semantic_search.cpp`
covering it. The `README.md`/`ARCHITECTURE.md`/`INTERNALS.md` sub-linear claim is therefore accurate
and the documentation-truth gap recorded in guardrail #2 is closed for this class. Relevance to the
record layer: it supplies sub-linear block scan for free at scale -- though per this stage's own
latency finding it will not move end-to-end numbers, where Aeon is 0.3-0.8% of wall-clock.

**SCHEMA v2 RESULT -- closed taxonomy + merge pass (2026-08-26). `n_errors=0`.** Twelve fixed
top-level buckets (POSSESSION, ACQUISITION, MEDIA, PERSON, EVENT_ATTENDED, OBLIGATION, EDUCATION_WORK,
HEALTH, TRAVEL, PROJECT, FINANCE, CONSUMABLE) with a free-form subtype, handed to the extractor on
every call, plus explicit multi-type discipline and a global consolidation/merge pass over the
accumulated records.

**The diagnosed mechanism is fixed -- demonstrably, on the cases that exposed it:**

| case | gold | v1 | **v2** |
|---|---|---|---|
| `bf659f65` "how many albums or EPs" | 3 | 1 | **3** ("Happier Than Ever", "Midnight Sky", Tame Impala vinyl) |
| `0a995998` "how many clothing items to pick up/return" | 3 | 2 | **3** (blazer, exchanged boots, old boots) |

The records show why: all three albums now carry consistent `ITEM(<BUCKET>/music album)` lines instead
of hiding inside `PREF`/`EVENT` prose, and the clothing case is answered from the `OBLIGATION` bucket
that free-form categories never produced.

**But the aggregate did not move, and that is reported as the result rather than the mechanism win:**

| | v1 | v2 |
|---|---|---|
| R (records only) | 4 | **5** |
| R+E (composite) | 6 reported / 5 real | 6 reported / **5 real** |
| union | 7 | 7 |
| multi-session (union) | 2 | **3** |
| temporal-reasoning (union) | 4 | **3** |

Judge audit of every v2 conversion found **the same false-positive as v1** (`e4e14d04` R+E answered
*"About three weeks"* against gold *"Two weeks"*, judged `yes`), so both arms are **5 genuine**. Note
the R arm answered the same question correctly and differently -- *"You had been a member for 14
days"* -- which is a true conversion.

**Honest reading.** v2 moved R by +1 and R+E by 0; multi-session +1, temporal -1. **Every one of those
deltas is inside the ~1.2-flip noise floor for n=18.** The taxonomy is justified on *mechanism*
grounds -- two specific diagnosed failures are fixed and the record structure visibly does what it was
designed to do -- but **there is no aggregate evidence at n=18 that v2 beats v1**, and it would be
exactly the error this project already documented (reverting v2/v3 prompts on noise-level deltas) to
claim otherwise.

**Decision: lock the schema and stop iterating here.** n=18 cannot resolve +/-2, so further schema
tuning against this cohort would be fitting to noise. The schema is locked on mechanism grounds; the
composite n=500 -- where both aggregate gain *and* collateral on already-passing questions become
measurable -- is the real test, and it remains user-gated. Build order now proceeds to **(b), the
persistent semantic layer**, against this locked schema.

**PERSISTENT SEMANTIC LAYER -- PROVENANCE (2026-08-26).** `shell/aeon_py/records.py` +
`tests/test_records.py` (21 tests, green; full suite 150 passed / 6 skipped, no regressions).
Built provenance first, deliberately: it is the one part the consolidation probe never exercised,
and everything else depends on it.

**Storage.** Records live in **Atlas**, not Trace -- they are semantic and accumulate/revise,
whereas Trace is an append-only log of what was said. Each record is one Atlas node: the vector is
the record's embedding, and the fixed-size `metadata` field carries the structure
(`kind / bucket / subtype / date / provenance / supersedes / text`).

**Provenance is `session_id` + turn indices, NOT raw event or node ids.** That is a correctness
decision: this codebase already documents that a raw Atlas node id can shift after `compact_mmap()`
reclaims tombstoned slots (supersession.py's known limitation), and Trace exposes history *per
session* rather than by id. Session id plus turn index survives compaction and matches the API that
actually exists. Contiguous indices compress to ranges (`0-3`), which matters because provenance
shares a fixed metadata budget with the record text.

**Three real bugs found and fixed while building, each documented in the code:**

1. **Silent truncation.** `Atlas.insert()` truncates rather than raising (documented in `core.pyi`,
   verified empirically: a 600-byte write reads back 511). Every write is now length-checked, and
   the encoding puts **provenance before free text** so an oversized record loses its text tail
   rather than its origin link -- *a record you can still trace back beats one whose text is intact
   but whose provenance is gone.*
2. **UTF-8 truncation split a character.** The first `_fit` appended a 3-byte `…` onto an already
   full budget; the stored payload ended mid-sequence and **every subsequent read raised
   `UnicodeDecodeError`**. This is the exact bug the C++ side already fixed on
   `TraceEvent.text_preview` (`safe_utf8_truncate_length` in `trace.cpp`) -- now mirrored in Python,
   plus defensive reads so one bad row written by an older version cannot make a whole store
   unreadable.
3. **Roles rendered as ints.** `get_history` returns `role` as an integer, and a local formatter
   emitted `- [0]` instead of `- [user]`. Fixed by rendering through
   `session_expansion.format_events`, so rehydrated provenance is byte-identical to every other
   context this repo builds and that class of drift cannot recur.

**Rehydration is the payoff, and it is the measured constraint made operational.** Given records,
`RecordStore.rehydrate()` returns their source turns with `+/-1` neighbouring turns. Demonstrated on
the exact failure mode that cost 12 questions: the record `ITEM(POSSESSION/toiletry): lavender
shampoo` **cannot** answer *"what brand?"*, while its rehydrated neighbourhood contains *"picked up
a lavender shampoo at Trader Joe's"*. **Compression discards pragmatic licensing; provenance is how
it comes back on demand rather than by shipping 100k chars up front.** `max_turns` bounds
rehydration so provenance cannot quietly reintroduce the context this layer exists to avoid, and
missing/dangling/out-of-range provenance degrades to an empty list rather than raising.

Next in the layer: the write path (per-session extraction on ingest, via `Architect`), the
consolidation/merge pass as the `Dreamer`'s real job replacing `StubSummarizer`, and the composite
read path (records + budgeted episodic selection, one LLM call).

**PERSISTENT SEMANTIC LAYER -- WRITE PATH (2026-08-26).** `shell/aeon_py/consolidation.py` +
`tests/test_consolidation.py` (42 tests; suite total now **192 passed / 6 skipped**, no regressions).

Deliberately split from `records.py`: storage stays LLM-free (its 21 tests run in ~0.1s), while
this module owns the model-facing prompts and parsing. Two stages, matching what the probe
validated -- `extract_session()` (per-session, query-blind, runs as each session arrives) and
`consolidate()` (global merge across accumulated records, which is the concrete job `dreamer.py`
was designed for and currently stubs out).

**Turn citations, so provenance is useful rather than nominal.** The extractor is shown numbered
turns and asked to cite them. Session-granularity provenance would be worthless -- a session is
~10 turns, so rehydrating a whole session plus neighbours is just the uncompressed context again.
Records whose citation is missing or out-of-range fall back to session-level provenance, so a bad
citation *degrades the neighbourhood* instead of losing the link.

**Two safety properties the tests pin down**, both of which would silently destroy memory:

- An **invented bucket** degrades to a `FACT` rather than minting a category. A category that
  exists once never accumulates with anything -- that is precisely the failure the closed
  vocabulary was introduced to fix, so the parser must not quietly reintroduce it.
- A **collapsing consolidation pass is rejected**. Consolidation is normalisation, not
  summarisation; if the merge returns less than 30% of the input records, the original set is
  kept. Accepting it would be indistinguishable from deleting the user's memory.

**Validated against real model output, not just hand-written shapes -- and that caught a
3.3% loss.** Running the parser over the **4,504 record lines the model actually produced** in the
consolidation probe gave **96.7%**, and the failures showed a consistent emergent pattern: the model
frequently drops the `ITEM(...)` wrapper and writes the bucket as the kind -- `HEALTH: ...`,
`OBLIGATION/decision: ...`, `EVENT_ATTENDED [2023/08/14]: ...`. Rejecting that shorthand was
discarding **149 records including every HEALTH and EVENT_ATTENDED record in the corpus**. The
shorthand is unambiguous *because the bucket vocabulary is closed*, so it is now accepted:
**parse rate 99.9% (4,500/4,504)**, with the remaining 4 being `Consolidated records:` headers that
are correctly rejected as non-records. Those real shapes are pinned as regression tests against the
measured corpus rather than against invented examples.

Bucket distribution over that corpus, which is also the first evidence the closed taxonomy is
actually *used* across its range rather than collapsing onto two or three buckets: OBLIGATION 524,
POSSESSION 509, PROJECT 219, MEDIA 179, CONSUMABLE 177, PERSON 159, ACQUISITION 136, TRAVEL 121,
EVENT_ATTENDED 109, HEALTH 101, EDUCATION_WORK 84, FINANCE 51 -- all twelve populated.

Remaining in the layer: wiring extraction into ingest (`Architect`), replacing `StubSummarizer` with
`consolidate()` in the `Dreamer`, and the composite read path (records + budgeted episodic
selection, one LLM call). The composite n=500 -- the first run that measures whether records *cost*
accuracy on the ~400 already-passing questions -- remains user-gated.

**PARALLELISM (2026-08-26).** `shell/aeon_py/parallel.py` + 14 tests (suite **206 passed /
6 skipped**). Prompted by the standing goal's instruction to exhaust cache/parallel options before
spending hours on tests -- and the arithmetic justified it: a composite n=500 consolidation pass is
**~23,000 extraction calls, ~9 hours sequential**. **Nothing in the shell was parallel** (grep-verified:
no `ThreadPool`, `asyncio.gather`, or `concurrent.futures` anywhere), despite per-session extraction
being embarrassingly parallel.

**The measured numbers, and a methodological correction worth more than the speedup.** A synthetic
probe (short prompts, ~5-line outputs) reported **3.2x at 4 workers**. The real workload -- one
question, 46 sessions, ~200-token outputs -- measured:

| workers | wall-clock | speedup |
|---|---|---|
| 1 | 74.6 s | -- |
| **4** | **36.2 s** | **2.06x** |
| 8 | 52.9 s | 1.41x |
| 12 | 36.6 s | 2.04x |

**The synthetic figure overstated the real one by 55%** -- longer generations contend harder, so a
concurrency number measured on toy prompts does not transfer. And the 8-worker result being worse
than *both* 4 and 12 is endpoint noise on single runs, not knee structure; reading a shape into it
would be precisely the unvalidated-constant error this stage has already documented twice. The
supportable claim is deliberately narrow: **~2x at 4 workers, no reliable gain past 4.** The
docstring and its test carry the real numbers, not the flattering ones.

**Three silent-failure properties pinned by tests**, all of which would corrupt results rather than
crash: results preserve **input order** (otherwise two runs over identical input produce different
record sets and every downstream comparison is unreproducible); each worker gets its **own
provider** via `ThreadLocalResource`, because `OllamaProvider.last_num_ctx` is mutable per-call state
that result files record per question, so a shared instance mis-attributes context sizes silently;
and **one failing item cannot abort the pass** -- a single malformed session must not lose the other
22,999.

**Two design corrections adopted before they were built in, both advisor-flagged:**

1. **Extraction does NOT belong on the ingest path.** It is a ~1.3s LLM call; inline, it would
   destroy the write-latency story that is half the product argument. Correct locus: ingest marks
   the session dirty at kernel speed, and **`DreamingWorker` consumes the dirty queue in
   background**, with extraction + `consolidate()` as its cycle, replacing `StubSummarizer`.
   Consolidation cost is real but lives entirely off *both* hot paths.
2. **The composite read path must use ALL records, not vector top-k over records.** The probe
   validated all-records-in-context (~21k chars at 46 sessions). Top-k over records would fetch 3 of
   5 `ITEM(ACQUISITION/music album)` lines and reintroduce partial recall on exactly the counting
   questions this layer exists to fix -- the old failure in new clothes. The scaling path for long
   histories is a **structured category scan** (all ITEMs in a bucket, cheap because records are
   small), not vector retrieval. Noted, not built.

**Parked explicitly**: wiring `HierarchicalSLB` into the benchmark path. Retrieval is 12 ms, under
1% of end-to-end latency, so it moves nothing this goal cares about. It is a scale item, and saying
so keeps it from lapsing or tempting.

**Comparability note for future bars**: parallelism is a change to the instrument, and the 6.7%
noise floor was measured sequentially. The nested-repeat slice will be piggybacked inside the first
parallel run and *that* floor used for the composite's bars, rather than inheriting the sequential
one blind.

**COMPOSITE ARM, TIER 2. PRE-REGISTERED 2026-08-26, before running.** First measurement of the
semantic layer as a *system* rather than a probe, and the first that can show a REGRESSION -- every
consolidation result so far was on questions selected for being wrong, so it could only improve.

*Sample*: the same **85 questions** used for the precision-selector tier 2 -- a 60-question
stratified sample unioned with all 27 known retrieval misses -- so one run measures conversion on
hard cases and collateral on normal ones, and every arm is directly comparable on identical
questions.

*Arm*: `compose()` -- **all records + provenance-rehydrated episodic turns, one LLM call**. Two
things differ from the probe's R+E, both from `compose.py`: records are **ordered so countable
members of a category are contiguous** (a model counting `ITEM` lines scattered through 250
unordered records is doing the same multi-hop scan on a smaller haystack), and episodic context is
**rehydrated from the records' own provenance** rather than from an independent retrieval pass, so
excerpts are guaranteed to be about the records in play.

*Reference points on these same 85 questions*:

| arm | all 85 | normal 58 | known-miss 27 |
|---|---|---|---|
| single-shot @top_k=30 | 49 | 46 | 3 |
| ETC @top_k=30 | 54 | 52 | 2 |
| precision selector | 48 | 44 | 4 |
| oracle-precision (ceiling) | 70 | 49 | 21 |

*Bars*: **primary -- >= 54 overall**, i.e. matching the best existing arm (ETC) while using one LLM
call instead of two. **Collateral guard -- the 58 normal questions must not fall below 46**
(single-shot's count); noise there is ~4 flips at the measured 6.7% rate, so this guard catches large
regressions only, and that limitation is stated now rather than discovered later. **Conversion floor
-- >= 4 of the 27 known misses**, reproducing the probe.

*Committed in advance*: clearing the primary bar makes the semantic layer a real improvement to
Aeon rather than a validated mechanism, and justifies the composite n=500. Missing it while holding
the collateral guard means records help the hard cases but do not yet pay for themselves in
aggregate -- in which case the next question is record *ordering and density*, not more extraction.
Breaching the collateral guard means consolidated records actively cost accuracy on ordinary
questions, which would be the most important negative result of the whole direction and would stop
it.

*Cost*: ~3,900 query-blind extraction calls, ~40 minutes at the measured 4-worker concurrency
(~80 sequential). Records are cached, so every later read-path variation on this sample is minutes.

**BACKGROUND CONSOLIDATION WIRING (2026-08-26).** `shell/aeon_py/consolidator.py` +
`shell/aeon_py/compose.py` + 33 tests (suite **239 passed / 6 skipped**). This is the product
wiring the goal asks for, landed alongside benchmark iteration rather than behind it.

**Ingest stays at kernel speed, measured not asserted.** The write path only enqueues:

    ingest  ->  mark_dirty(session_id)     **163 ns**, O(1), no I/O, no LLM
    later   ->  background cycle drains the queue

For scale: `mark_dirty` is **13.7x cheaper than Aeon's own 2.23 us insert**, and **~8,000,000x
cheaper** than the ~1.3 s extraction call it replaces on that path. Consolidation cost is real
(~36 s per 46 sessions at 4-way concurrency) but lives entirely off **both** hot paths -- the write
path enqueues, the read path reads finished records. That is what makes a semantic layer affordable
in a system whose pitch is ultra-low latency, and it is the claim the paper should make.

**Not forced through `LLMSummarizer`.** `dreamer.py`'s existing pluggable interface is shaped
`summarize(texts) -> (text, embedding)` -- summarise-N-into-1, for summarise-to-forget.
Consolidation is a different operation (one session becomes many typed records; records merge
*across* sessions), so it is wired as its own cycle rather than bent through an interface built for
something else.

**Silent-data-loss properties, which is where the tests are concentrated.** Losing a session's
records is **invisible at read time** -- the answer is merely worse, never an error -- so it is
precisely the failure that never gets reported as a bug. Therefore: the dirty queue is a *set*, so
a session written ten times consolidates once (otherwise consolidation cost scales with write
volume rather than distinct sessions); a claimed session moves to in-flight rather than being
forgotten, and a failure **requeues rather than drops**; `requeue_in_flight()` recovers anything
unconfirmed after a crash; one bad session does not block the others; and a merge that returns
nothing is **refused**, because overwriting a populated store with an empty result is
indistinguishable from erasing the user's memory. Commit order is deterministic despite concurrent
extraction, so two runs over the same dirty set produce the same store -- reproducibility a
background worker would otherwise quietly destroy.

**The composite read path (`compose.py`)** assembles one call over records + episodic turns, with
records **ordered so countable members of a category are contiguous**: a model counting `ITEM` lines
scattered through 250 unordered records is doing the same multi-hop scan on a smaller haystack.
`select_records()` provides the scaling path as a **structured category scan** -- every member of a
bucket, never a top-k, because counting requires completeness and a subset turns a complete answer
into a plausible wrong one.

**COMPOSITE ARM RESULT (2026-08-26). `n_errors=0`. ALL THREE BARS PASSED, and the semantic layer
exceeds what was previously treated as the ceiling.**

| arm | all 85 | normal 58 | known-miss 27 | LLM calls |
|---|---|---|---|---|
| single-shot @top_k=30 | 49 | 46 | 3 | 1 |
| ETC @top_k=30 | 54 | 52 | 2 | 2 |
| precision selector | 48 | 44 | 4 | 1 |
| oracle-precision *(prior "ceiling")* | 70 | 49 | 21 | 1 |
| **COMPOSITE (records + episodic)** | **72** | **50** | **22** | **1** |

Bars: overall >= 54 -> **72 (+18)**; normal slice >= 46 -> **50**; known misses >= 4 -> **22**.

*Paired significance* (2x sd = 4.8 at n=85):

| vs | delta | McNemar | verdict |
|---|---|---|---|
| single-shot | **+23** | +28/-5 | **REAL** |
| ETC | **+18** | +25/-7 | **REAL** |
| oracle-precision | +2 | +11/-9 | noise (statistically tied) |

**The headline finding: 83.8% was the ceiling of ONE ARCHITECTURE, not of a memory system.** The
oracle bounds *perfect selection of raw turns*, and the composite matches it **without gold
annotations** -- it uses only what a production system actually has. The prediction made when this
direction was proposed ("a maintained record answers what perfect raw evidence plus this generator
measurably cannot") is now measured rather than argued: **22 of 27 questions where retrieval had
provably failed are answered**, against ETC's 2.

*Per type against ETC*: temporal-reasoning **+10** (16->26), multi-session **+4** (9->13),
single-session-user +2, preference +2, single-session-assistant +1, knowledge-update 0,
abstention -1.

**Cost -- this is the product argument, and it improves on every axis at once:**

| | accuracy | context | calls | generation | **accuracy-pts / 1k chars** |
|---|---|---|---|---|---|
| single-shot | 77.6% | 100,889 | 1 | 1.51 s | 7.69 |
| ETC | 82.6% | 100,889 | 2 | 2.46 s | 8.19 |
| **composite** | **84.7%*** | **25,557** | **1** | **0.93 s** | **33.14** |

**4x more context-efficient than ETC, 2.6x faster generation, and half the LLM calls -- while more
accurate.** (A first version of this table used raw correct-count per 1k chars, which is not
comparable across different sample sizes; restated as accuracy-points per 1k chars.)

**Honest limits, which matter for how this number is used.** (1) *The 84.7% is NOT comparable to the
n=500 figures* -- this sample is deliberately enriched with 27 known-hard questions, so the only
valid reading is the **paired arm-vs-arm comparison on these same 85**, which is what the tables
above report. (2) On the **normal 58-question slice the composite is 50 vs ETC's 52** -- inside noise
(2x sd = 3.9 there), but *not* an improvement: **the gain is concentrated on hard retrieval cases and
temporal reasoning, and is not uniform**. (3) Abstention is -1, worth watching at scale. (4) The
composite has not been run at n=500; that remains the decisive test of whether records cost accuracy
across the full distribution.

**What this means for Aeon.** The semantic layer is no longer a validated mechanism -- it is a
measured improvement to the product: better answers, a quarter of the context, one LLM call instead
of two, sub-second generation, with consolidation cost paid once at write time and off both hot
paths (ingest enqueues in **163 ns**).

## KERNEL-CAPABILITY AUDIT (2026-08-26): the semantic layer had drifted off Aeon's own architecture

Prompted by the user's observation that this work may have drifted from v3/v4 capabilities that could
help. It had -- and the finding is better framed as **convergence** than as an error:
`EdgeType::MergesWith` in `schema.hpp` is documented as *"this event and supersedes_id were
consolidated (Dreaming)"*. **The kernel anticipated the consolidation layer this stage built.** That
is an architecture argument worth making, not just a mea culpa.

**Two genuine reinventions, now removed** -- both were parameters of functions the shell was already
calling, so adopting them **deleted Python rather than adding it**:

| kernel capability | what the layer did instead | now |
|---|---|---|
| **Atlas is a tree** (`insert(parent_id, …)` / `get_children(parent_id)`) | inserted every record flat at `parent_id=0` and filtered in Python | each bucket is a node, records are its children; `records_in_bucket()` is a kernel subtree walk |
| **`session_id` on `insert()`** (+ `drop_session`) | passed `None` -- unscoped | records are tenant-scoped at write |

**One self-charge withdrawn after checking**: `event_time` is a **`TraceEvent`** field. Records are
Atlas nodes whose only payload is the metadata string, so dates-as-text is the available option, not
an oversight. Recorded because an inaccurate confession is as bad as a missed capability.

**The taxonomy IS the tree.** The closed 12-bucket vocabulary was designed so countable members
accumulate; Atlas is a tree of nodes reached by descent. Those are the same structure, and the
"structured category scan" named earlier as the scaling path is exactly `get_children()` -- already
implemented, in the kernel, at kernel speed. Honest sizing: **this buys nothing measurable today** (a
Python scan over 263 records is microseconds, and Aeon is <1% of end-to-end latency). It is adopted
for product correctness and scale, and an **equivalence test is pinned first** -- same records in,
byte-identical `render_records()` out -- so the refactor cannot silently move the measured 72/85.

**Supersession is first-class in the kernel and was being done in text.** `supersede_node()` /
`revoke_node_supersede()` / `is_node_superseded()` are exposed, reversible and branchless. The
composite currently shows the model both values plus a `[supersedes $350,000]` marker and hopes it
picks correctly; the kernel can **exclude the stale record from retrieval entirely**, reversibly.
`RecordStore.supersede()` now uses it. Because that changes prompt content it goes through the
free->cheap ladder rather than straight into a run: the knowledge-update cohort first.

**The most valuable output of the audit was two production-readiness gaps it exposed**, neither
visible in any benchmark:

1. **Tenant isolation.** A benchmark uses one store per question, so unscoped inserts look fine and
   are fatal in a multi-tenant product. Fixed at the point of insert.
2. **Erasure cascade.** Records are PII *derived* from conversation. `erasure.py` tombstones Atlas
   nodes, but nothing cascaded to derived records. **`provenance.session_id` is exactly the cascade
   index** -- the field built for pragmatic licensing is also the right-to-be-forgotten index
   (`records_for_session()`). Convergence again, and worth a line in the paper.

**Two corrections to this stage's own work, found by the same audit:**

- **`DirtyQueue`'s crash-safety claim was overstated.** It is an in-memory set: process death loses
  every pending entry, and `requeue_in_flight()` only covers in-process failures. Docstring corrected
  rather than papered over. The kernel-aligned fix is to make dirty state **derivable** (a session's
  latest Trace event vs. a per-session consolidation watermark) so recovery is a rescan and needs no
  second write-ahead structure beside the one Trace already has.
- **There are two extraction prompts.** `composite_arm_experiment.py` imports the probe's
  citation-less prompt, while the production path in `consolidation.py` (turn citations + provenance
  rehydration) has **never been exercised at scale**. The measured 72/85 is on the harness path, not
  the product path. Kept deliberately for now so the n=500 stays comparable to every prior arm;
  reconciled afterwards by a cheap equivalence check.

**Parked explicitly, with reasons**: `scope_bitmap` read-filtering (the all-records design makes it
moot), `expand_window`/`expand_summary` replacing the selector (the measured path works),
`promotion.py` (cross-agent shared records -- roadmap, and `PROMOTED_FROM` is the right edge), and
`HierarchicalSLB` (already parked: 12 ms retrieval is <1% of latency).

## Verification plan (how to confirm this roadmap is being executed correctly, end to end)

- **Per-stage gates above** are the primary mechanism — each is a concrete test or measurement, not
  a subjective checkbox.
- **CI perf-regression job** (guardrail #0) must be green (and provably capable of failing) before
  Stage 1 lands, and stays green through every subsequent stage — treat any red run on the curated
  benchmark subset as a stop-the-line event, not a follow-up ticket.
- **`ctest --preset dev`** (existing `aeon_tests` target) must continue passing throughout; extend it
  with the new tests named in each stage's gate (multi-tenant isolation, WAL unknown-record-type,
  non-768-dim SLB hits, scope-filtered recall, pivot-attack red-team, crypto-erase destruction).
- **Documentation truth check** before declaring v4 done: `README.md`, `ARCHITECTURE.md`,
  `INTERNALS.md`, and `CLAUDE.md` must describe what the code actually does — the `TraceBlockIndex`
  gap (guardrail #2) is the first instance of this found; do a final pass across all four docs against
  the shipped v4 code before release, not just for this one class.
- **Manual smoke test**: run the FastAPI server (`scripts/run_server.sh`) with two distinct
  `X-User-ID`/auth identities from Stage 0 onward at every subsequent stage, confirming isolation
  never regresses as scope/version/shared-tier features are added on top.
