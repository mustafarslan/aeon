# Aeon v4 Hive Mind: Verdict and Revised Roadmap

## Bottom line

Share organizational knowledge across teams — yes. The idea is sound, the demand is real, and the
use case you named (tasks, task IDs, bugs, bug fixes) is the single best-suited payload for it.
Build it the way the v3 proposal specifies — no.

Three findings drive that split, and each is independently sufficient to force a redesign.

**The proposal's foundation does not exist.** Its roadmap labels Phase 0 and Phase 1 as *delivered*
and plans Phase 2 on top of them. A file-by-file audit of the repository found that twelve of the
twelve named components in those two phases are absent: no RaBitQ, no FastScan, no NAMM importance
scoring, no DiskANN layer, and no `nexus/` directory at all: no sketch, no router-client, no
distill, no SPIFFE, no macaroons, no ReBAC authorization. What exists is INT8 symmetric
quantization (not RaBitQ), a tree-descent Atlas, Trace, the SLB, the WAL, epoch-based reclamation
and the blob arena. `bench_rebac_prefilter.py` does not test Aeon; it imports LanceDB with a DuckDB
fallback, never constructs an `Atlas`, and validates a claim about a Neo4j query planner from the
CEUR-WS paper. This is therefore not a proposal to add sharing to a federated multi-tenant system.
It is a proposal to *build* that system, and the sharing design is the smallest part of the work.

**The one mechanism the proposal specifies precisely is wrong, and measurably so.** The early-reject
pseudocode — `if (scope_bmp[node.id] & mask) == 0: continue` inside `traverse()` — assumes a flat
scan. Atlas is a tree descent that reaches children only through their parent's
`first_child_offset` (`core/src/atlas.cpp:218-233`), so skipping an out-of-scope node discards its
entire subtree. Measured on an 8-ary depth-6 tree of 299,593 nodes, recall loss is 32.8% at 0.90
selectivity, **95.8% at 0.50**, and effectively total below 0.10. The filter does not slow retrieval
down; it silently deletes most of the results. The fix is known (propagate the union of descendant
scopes to internal nodes and enforce at result emission, never during descent), but it is a
different mechanism from the one specified.

**The performance analysis optimizes the wrong quantity, and the layout recommendation is
backwards.** The 3.09 µs and 4.70 ns figures reproduce exactly, but the budget denominator is wrong
by roughly 500×: a 100K-node query descends depth 3 and scores 193 nodes, not 100,000. The correct
gate cost is about 0.4% of query time, so performance is emphatically *not* the reason to hesitate.
Meanwhile the proposal's recommended structure-of-arrays side table measured 2–8% *slower* than
baseline, while putting the bitmap in the existing `NodeHeader.reserved[20]` at offset 0x28 ran 19%
*faster* — and gets carried through compaction for free by the existing `memcpy`. The proposal
reached the right conclusion (8 bytes fit) by the wrong reasoning (there is no tail slack; 1,088 is
exactly 64-aligned) and then recommended the worse of the two options.

Underneath all three sits a fourth fact that changes the priority order entirely. **Aeon in server
mode is already multi-user with no retrieval isolation.** `get_atlas_client()` is an `@lru_cache()`
singleton (`shell/aeon_py/dependencies.py:13-18`) shared by every user's `ContextManager`; the
tenancy filter exists only as a commented-out line (`context.py:47`); the C-API accepts
`session_id` on every call and explicitly discards it (`(void)session_id`,
`aeon_c_api.cpp:147-149`); `aeon_atlas_drop_session` is a stub that returns `AEON_OK` without
acting; identity is whatever the caller puts in an `X-User-ID` header
(`dependencies.py:34-38`); and `POST /state/atlas/query` (`server.py:157-161`) has no user
dependency at all. `HierarchicalSLB` — the 64-shard isolation machinery the proposal plans to build
on — is real but never instantiated by Atlas, and even if wired, its L2 miss path scans a shared
global cache with no session predicate (`hierarchical_slb.hpp:290-296`).

That is a present-tense defect, independent of the hive mind. It also inverts the roadmap: you do
not need to *add* multi-tenancy to Aeon before sharing, you need to *fix* it before anything.

---

## Is the idea logical?

The organizational case is straightforward. An engineer's agent solves a problem; the next engineer
to hit it starts from zero. Bug fixes, task context and internal conventions are exactly the
knowledge that is expensive to rediscover, absent from public training data, and cheap to reuse.
Your instinct is right, and it is right for the specific payload you chose.

The caveat is that nobody has yet shown this improves engineering outcomes. The closest published
system is a 2026 production deployment of very nearly your exact design — capturing task-adjacent
engineering experience, curating it into reusable memories, gating security and privacy risk,
retrieving it for later agents ([Dhanyamraju & Raghav 2026](https://arxiv.org/abs/2608.00122),
preprint). Its own abstract states that effects on retrieval and coding tasks remain under
evaluation. The broader picture is no more encouraging. A survey of twelve memory systems across
five workloads finds operational cost, module trade-offs and robustness under knowledge updates
insufficiently explored ([Wang et al. 2026](https://arxiv.org/abs/2606.24775), preprint); a
long-horizon personalization benchmark finds memory agents deliver marginal improvement with
frequent reuse of invalidated memories ([Uddin et al.
2026](https://arxiv.org/abs/2604.20006), preprint).

So the honest framing is that you are entering an area with strong intuition, an active market, and
no demonstrated outcome benefit. That argues for the cheapest possible first version measured
against real engineering outcomes — issue-resolution time, defect rate — not for a broker tier and
a gossip protocol.

## Novelty versus reinvention

Very little of the sharing concept is new, and that is useful information rather than a criticism.
Tiered promotion of knowledge from personal to team to organization already ships as a product
feature in Devin's knowledge editor; Cursor ships admin-authored Team Rules while deliberately
keeping per-user Memories unshared; GitHub Copilot Spaces is a human-curated bundle with
organization read-access; Glean mirrors source-system ACLs. Two-tier private/shared agent memory
with immutable provenance, time-evolving permissions and redaction at promotion time was published
last year ([Rezazadeh et al. 2025](https://arxiv.org/abs/2505.18279), preprint).

Two observations follow. First, **every shipped system centralizes the shared tier and keeps the
private tier local.** Nobody ships peer-to-peer federation of agent memory. Second, **every shipped
system puts a human in the promotion decision** — an admin, a curator, or the contributor. The
proposal's automated NAMM-importance promotion gate is precisely the design choice that everyone
with deployment experience has declined to make.

The genuinely novel component is per-scope RaBitQ/Bloom/MinHash sketches gossiped in per-scope
broker domains. Novel here means unstudied: there is no published threat analysis to lean on, and
the proposal's cited mitigation for sketch leakage rests on an unverified reading of a preprint's
internals.

## The federation question

The proposal inherits peer-to-peer federation from Aeon's existing per-engineer topology and never
argues for it. Nearly all of its complexity descends from that one inherited assumption: the
hierarchical broker tree, per-scope gossip domains, sketch leakage, broker inference, timing
channels, macaroon capability tokens, peer-cache invalidation, and the offboarding propagation
problem. A centralized org-scoped store deletes every one of them.

Federation's stated benefit is that raw memory never leaves the machine. But distilled fragments
and their sketches *do* leave, and embeddings are invertible. Text embeddings can be inverted to
recover much of their source text ([Morris et al. 2023](https://arxiv.org/abs/2310.06816),
preprint), a result strong enough to have drawn a dedicated peer-reviewed reproducibility study
([10.1145/3705328.3748155](https://doi.org/10.1145/3705328.3748155)) and transferable black-box
variants ([Huang et al. 2024](https://doi.org/10.18653/v1/2024.acl-long.230)). So the benefit is
already surrendered, and it was the only thing federation was buying.

The GDPR argument points the same way, and harder. Centrally, erasure is a delete you can confirm.
In the federated design it is a TTL plus a best-effort revocation list with no completion
guarantee, and an erasure you cannot confirm is an erasure you cannot defend to a supervisory
authority.

**Recommendation: build v4 on a single trusted retrieval service over a shared org-scoped store,
with each engineer's private memory kept separate. Defer peer-to-peer federation until a
requirement appears that the centralized design provably cannot meet.**

## Threat model, reoriented

The proposal's threat model is confidentiality-dominant: it worries about broker inference, sketch
leakage, timing channels, scope escalation. Those are real. But for *your* use case — sharing bug
fixes — the dominant risk class is integrity, and the proposal omits it entirely. There is no
AgentPoison, no corpus poisoning, no memory-borne prompt injection in the threat table.

This matters more than it sounds. Insecure coding preferences stored in long-term memory raise
vulnerable-code generation by 2.7 to 50.3 percentage points across four LLMs and five languages,
open a 5.4–14.0pp gap where warning rates lag vulnerability rates, and are **difficult to overwrite
through normal interactions**, continuing to influence output under rephrased prompts ([Chen et al.
2026](https://arxiv.org/abs/2607.17619), preprint). A bad fix promoted to company scope is not a
stale entry that expires. It is a persistent org-wide defect generator, and TTL cannot cure it
because the harm lives in what agents have already absorbed.

The corollary is that **promotion must be gated on verifiable correctness, not on an importance
score.** A fragment about a bug fix should cite a merged commit, carry a passing verification run,
and be revoked automatically when that commit is superseded. That is a stronger and cheaper gate
than a policy engine, and it is available to you precisely because you chose a use case with ground
truth attached.

Staleness compounds this, and TTL does not address it either. Cosine similarity distinguishes a
*contradicted* fact from a *duplicated* one at AUROC 0.59, near chance, because contradictions are
often more embedding-similar to the original than rephrased duplicates are ([Yadav
2026](https://arxiv.org/abs/2606.26511), preprint). A fix invalidated by last month's refactor
retrieves with essentially the same similarity as the current one. You need explicit supersession
semantics, not an expiry date.

One more, specific to Aeon's architecture and worth taking seriously: composing a vector index with
a graph, which is exactly Atlas plus Trace, creates cross-tenant leakage that does not occur in
vector-only retrieval, with undefended risk up to 0.95, appearing at pivot depth 2, requiring no
adversarial injection because naturally shared entities create the paths organically ([Thornton
2026](https://arxiv.org/abs/2602.08668), preprint). Enforcing authorization at the graph-expansion
boundary drove leakage to near zero. The proposal covers Trace in a single clause: "masked at read
time by the same eligibility mask." That clause needs to become a separately red-teamed enforcement
point.


---

## GDPR exposure

Your framing (share product knowledge, don't share personal information) is the correct
objective. The problem is that it is not achievable by classifying fragments as "product" or
"personal," because the category does not survive contact with real engineering text.

**"Bug fixes" are not automatically non-personal.** A fragment reading *"the auth timeout
regression in PR 412 took three attempts to fix"* is performance data about an identifiable
engineer. It is product knowledge and personal data simultaneously, and free-text distilled
fragments will contain exactly this shape of sentence constantly. The CJEU has confirmed both the
breadth of "relates to" and the contextual nature of identifiability (C-413/23 P, *EDPS v SRB*, 4
September 2025), and EDPB Guidelines 01/2025 confirm pseudonymised data remains personal data where
the controller holds the mapping. So the proposal's central compliance premise, that distilled means
safe, does not hold. Distillation reduces disclosure; it does not create a legal boundary.

Four consequences are load-bearing for the design.

**Promotion must mint a new fragment, not flip a bit.** The proposal specifies this two
incompatible ways: the data layer says set a bit in the scope bitmap, the promotion protocol says
re-distill and redact. Only the latter survives an Article 6(4) compatibility analysis. A bitmap OR
means your organizational corpus *is* the private corpus with a wider audience, and every erasure
request then reaches records the whole organization depends on. Minting a new, de-identified
fragment with its own ID, provenance and retention clock collapses most of the GDPR exposure —
this is the single highest-leverage change in this memo.

**Per-engineer opt-in is not a valid lawful basis.** WP29 Opinion 2/2017 states that workplace
processing cannot rest on employee consent, and EDPB Guidelines 05/2020 permit free employee
consent only where refusal carries no detriment, which is structurally impossible for a tool whose
value is collective. The realistic basis is Article 6(1)(f) legitimate interests, with a
documented, pre-design balancing test and a working Article 21 objection route. Keep opt-in as a
fairness feature; do not rest the lawfulness of the system on it.

**Erasure has no implementation today.** There is no per-node delete anywhere in `aeon_c_api.h`.
Atlas tombstoning happens only as a side effect of `consolidate_subgraph`, sets a flag, overwrites
`hub_penalty` with 1e9f, and leaves the 1,088-byte record byte-identical on disk;
`TraceManager::tombstone_event` only sets a flag; `drop_session` is documented as *not* deleting
events from disk; `BlobArena` has no reclaim path. Compaction is the only physical erasure, and it
finishes with `std::filesystem::remove`, an unlink, which does not guarantee destruction on flash.
This is not a theoretical gap: soft-deleted vectors have been recovered from raw index files
beneath the API and inverted to recover 25.5% of exact person names and 46.4% of locations
([Chakraborttii et al. 2026](https://arxiv.org/abs/2606.18497), preprint). That same paper's fix, encrypt and
destroy the key, drove recovery to zero, which is the mechanism you should adopt:
per-subject-per-scope keys beneath a per-scope key, independently destroyable.

**Article 19 is the duty the promotion protocol quietly creates.** Once peers hold a distilled
fragment they are recipients, so rectification and erasure must propagate to each of them with
per-recipient acknowledgement. The proposal's TTL-plus-revocation-list is a mechanism for this with
no operator surface and no completion evidence. Article 16 rectification is absent from the
document entirely, and it matters: a promoted fragment that falsely asserts an engineer's patch
caused an outage is a durable inaccurate statement about them, inside their employer.

Two process items are not optional. An **Article 35 DPIA is mandatory** — the design hits at least
five WP248 rev.01 criteria (evaluation of individuals, systematic monitoring, employees as
vulnerable data subjects, dataset combination, innovative technology). And in Germany, **BetrVG
§87(1)(6) gives the works council a co-determination right** over systems objectively suitable for
monitoring performance, regardless of intent; a negotiated Betriebsvereinbarung is a hard gate
measured in months, and it belongs on the roadmap as a dated dependency rather than a footnote.

On the EU AI Act: as specified, the system sits *outside* Annex III(4), but only because it computes
no per-person aggregate. Three natural product decisions flip it into the high-risk regime: any
per-engineer console view, any staffing or task-routing use, any competence scoring. Carry an
explicit non-goal enforced in the query layer (reject any query grouped or filtered by contributing
individual) and write the Article 6(4) assessment now, while the answer is defensible. Separately,
Article 5(1)(f)'s prohibition on workplace emotion inference has applied since 2 February 2025 with
top-tier penalties; nothing in the proposal trips it, but an importance score weighted by inferred
emotional salience would. A build-failing test asserting the absence of affect/sentiment/stress
features in the distillation and scoring paths is cheap insurance.

There is one thing the proposal says that you should not let anyone soften: that fragments already
absorbed into a model's context cannot be un-learned. It is the most credible sentence in the
document, and the literature backs it. TOFU found no evaluated unlearning baseline effective
([Maini et al. 2024](https://arxiv.org/abs/2401.06121)), and forgotten data remains linearly
decodable from representations after unlearning ([Goel et al.
2026](https://arxiv.org/abs/2601.15111), preprint). What must change is the inference drawn from
it. The response is preventive gating at write time, not a TTL.

*This is engineering analysis, not legal advice; a DPO and employment counsel review per
jurisdiction is required before shipping.*

## The admin console as launch gate

Your instinct here was right, and it is worth stating why it is stronger than "admins need a UI."

Three GDPR obligations have **no operator-facing mechanism whatsoever** in Aeon today: Article 15
access, Article 17 erasure, Article 30 records, plus Article 5(2) demonstrability. A repository-wide
search for audit, authorization, encryption or key-management code returns nothing outside vendored
test fixtures. While Aeon is single-user this is survivable, since deleting an engineer's instance *is*
erasure. The moment a fragment derived from their session is visible to a hundred engineers across
four departments, deleting their instance erases nothing, and there is no way to answer the
question "what does the organization still know that came from this person?"

There is also independent evidence that the console is a *security* control rather than a
convenience. Enterprise search solved permission mirroring years ago, and the documented consequence
is that faithfully honoring ACLs surfaces everything those ACLs already overshare, which is why
Microsoft runs a dedicated oversharing-remediation programme for M365 Copilot. Correctly enforcing
the scopes you are given is the easy half. The scopes being *wrong* is the hard half, and an admin
surface for finding and fixing them is the only known mitigation.

**Minimum console that must ship with v4** — these three, and no more, are the launch gate:

1. **Hash-chained audit log** with independent verification and signed export. Build this *first*:
   retrofitting audit onto existing write paths reliably misses some.
2. **Knowledge browser** — search and filter shared fragments by scope, provenance, classification
   and age; show the provenance chain, who can see it, and its TTL; single-fragment supersede,
   redact and delete.
3. **Erasure workflow** — Article 15 export and Article 17 erasure as tracked cases with a one-month
   deadline clock, a subject-scoped search, an executed-erasure receipt, and an explicit section for
   what could **not** be erased.

Defer the promotion review queue UI, the org-graph editor, the policy authoring surface and the
observability views to v4.1.

Two design points from the console analysis deserve to survive into whatever you build. First, an
org admin who can read every fragment holds more access than any engineer: the exact concentration
the scope model exists to prevent, re-created inside the governance tool. Constrain it:
scope-scoped admin roles by default, four-eyes approval on bulk operations, mandatory read-reason
prompts stored in the audit entry, and time-boxed break-glass that alerts the DPO. Critically, admin
reads must go through the *same* enforcement path with the admin's own effective scopes, never a
wildcard bypass, because a privileged branch in the enforcement code is where such bugs live.

Second, the promotion review queue is a human-throughput bottleneck bolted to a machine-throughput
pipeline, and this is where the design most likely breaks operationally. At a thousand engineers
producing one promotion-worthy learning per day, the queue receives roughly 5,000 items a week. No
staffing level reads those, so reviewers bulk-approve and the audit trail fills with approvals that
record no judgement. If you build it, build three lanes (auto-approve with sampling audit and an
automatic trip to full review on elevated reversal rates; human review for anything referencing a
person or destined for company scope; visible deny-with-reason) and no bulk-approve button on the
human lane.

Architecturally: keep the C++ core **enforcing** and never **governing**. Governance state is
relational, low-QPS, transactional and history-bearing: the opposite of a fixed-stride mmap with
compile-time size assertions. Six small primitives belong in the core (scope mask AND on the read
path, fragment soft-delete, WAL-durable bitmap get/set, list-by-scope, bulk bit remap, a governance
record ID in `NodeHeader.reserved`); everything else belongs in a separate FastAPI + Postgres
control plane, with the console as a client of *that* and never of the C-API, so there is exactly
one enforcement point and no second unaudited path.

---

## Revised roadmap

The proposal's phases assume infrastructure that is not there, so the sequencing has to reset. Note
that Stage 0 is work you need regardless of whether you ever build the hive mind.

### Stage 0 — Fix present-tense isolation (~1 week)

Wire `HierarchicalSLB` into Atlas; honour `session_id` in the three stubbed C-API functions;
scope-filter or disable the shared global L2 cache; implement `aeon_atlas_drop_session` so it stops
returning `AEON_OK` for a no-op; replace `X-User-ID` header identity with real OIDC; remove or
authorize `POST /state/atlas/query`. **Gate:** a cross-session isolation test exists and passes,
including the L2 miss path. None exists today.

### Stage 1 — Carry scope durably (~1 week)

Scope bits in `NodeHeader.reserved[20]` at 0x28, *not* a parallel array: measured 19% faster than
baseline and carried through compaction free by the existing `memcpy`. Add `WAL_RECORD_SCOPE` plus
unknown-record skip in `replay_wal` (it currently `break()`s, so forward-compat is a prerequisite
for any WAL evolution), and journal `Atlas::insert()`, which today writes no WAL at all. Scope must
be assigned from authenticated session context at write time, never from caller-supplied labels.
**Gate:** scope survives WAL replay and compaction; a node-ID remap test passes.

### Stage 2 — Correct scope-filtered retrieval (2–3 weeks)

Replace early-reject-during-descent with union-propagated internal-node scope and enforcement at
result emission. Enforce independently at the Trace graph-expansion boundary. **Gate:** recall >=
0.99 versus an exhaustive scope-filtered scan across selectivities 0.02–1.0; pivot-attack red-team
shows no cross-scope leakage at depth 2. Do **not** gate on `BM_AtlasTraversal_Only`: at 0.078 µs
it is an SLB self-hit artifact, not a traversal measurement, because the benchmark queries a vector
that is a bit-for-bit copy of a stored node (cosine = 1.000000).

### Stage 3 — Centralized shared tier + minimum console (4–6 weeks)

A single trusted retrieval service over a shared org-scoped store; private memory physically
separate from shared memory rather than commingled and filtered. Promotion mints a new
de-identified fragment behind a fail-closed classification gate, with the deterministic detector
over your own identifier corpus (directory names and aliases, emails, internal IDs, PR and ticket
formats, commit SHAs, hostnames) as the only layer permitted to *pass* a fragment; an LLM layer may
only reject or flag. Correctness gate for bug-fix fragments: cites a merged commit, verification run
passes, auto-revoked on supersession. Plus the three console components above. **Gate:** measured
false-negative rate for the subject-reference detector on a labelled corpus of real distilled
fragments, reported in the DPIA.

### Stage 4 — Outcome experiment, before anything else is funded

Shared-memory versus no-shared-memory on real engineering work, measured on issue-resolution time
and defect rate rather than retrieval F1. This is the cheapest experiment in the roadmap and it
decides whether the rest is worth building. No published source, including the one production
deployment, has demonstrated this benefit.

### Deferred to research

Peer-to-peer federation, hierarchical brokers, per-scope gossip domains, sketch publication,
macaroon capability tokens, covert-channel hardening. Each must justify itself against a requirement
the centralized design cannot meet.

Stages 0–3 are roughly a quarter of focused work. The proposal's "Phase 0/1 multi-tenancy seams,"
presented as small additions to a finished system, *are* the infrastructure.

## What the proposal gets right

Four things should survive into v4 unchanged, and they are not minor.

Scope as *labels* rather than graph structure, so a reorg is a relabel rather than a reindex. This
is the correct call, and it is the difference between Filtered-DiskANN baking labels into edges
([Gollapudi et al. 2023](https://doi.org/10.1145/3543507.3583552)) and a design that survives an org
chart changing quarterly. Authorization off the hot path via a locally cached relation closure,
since a 3 µs retrieval cannot contain a millisecond authorization check. Responder-side re-checking
rather than trusting a broker or querier. And defaulting fragments to the most restrictive scope,
promoting only by policy.

The document's own caveats are also, notably, correct. It concedes that the cache-line impact is
unmeasured and that Aeon's per-engineer topology changes the filtered-ANN calculus. Both concessions
are right. The recommendations simply do not follow from them: having noted that its multi-tenant
citations may not transfer, the document continues to cite Curator and HONEYBEE as validation for
the bitmap design. They do not validate it. Both target one-index-many-tenants at *low* selectivity,
while Aeon's per-engineer instance runs at roughly 100% local selectivity, where a peer-reviewed
selectivity sweep across ten algorithms and four datasets shows all filtered-ANN techniques converge
to unfiltered search plus a cheap predicate test
([10.1145/3769763](https://doi.org/10.1145/3769763)).

The deeper point that reframes the whole design: **on the local path the bitmap protects against
nothing**: the engineer is authorized for their own memories by definition. Its real job is
marking what may *leave* the instance. That is an egress-labelling problem, not a
retrieval-filtering problem. The proposal spends its latency budget, its cache-line analysis and its
Phase 0 gate on the cheap, safe part, while the genuinely hard part (deciding a fragment's scope
correctly *at write time*, which is what actually determines whether personal information leaks
org-wide) gets one clause, no mechanism, no accuracy target and no failure analysis.

That is the work. Everything else is plumbing around it.

## Decisions that are yours

Five, and four of them are legal rather than technical:

1. **Lawful basis**: legitimate interests with a documented balancing test, almost certainly, not
   consent.
2. **Whether the org and company tiers may hold identified personal data at all.** The clean answer
   is no, with indirect provenance tokens, and it is what makes erasure tractable and the DPIA
   defensible. This is the highest-leverage decision on the list.
3. **Retention periods per tier**, noting that the proposal specifies only a security TTL bounding
   post-revocation exposure, and says nothing about the Article 5(1)(e) storage-limitation clock,
   which is a different quantity with a different required magnitude.
4. **Whether opt-in is real**: whether an engineer who declines suffers no detriment.
5. **Whether to run Stage 4 first.** Defensible either way, but running the outcome experiment
   before Stage 3 would tell you whether to build the expensive part at all.
