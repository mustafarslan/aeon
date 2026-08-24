# Aeon v4 Addendum: Versioning, Growth, and Rationale for the Shared Tier

## Bottom line

Your three instincts decompose into one architecture, but not the one you sketched. Versioning is
the right call, and the literature has converged on it in the last twelve months: the design space
is now populated, so the question is placement, not invention. Contextual embeddings are worth
adopting, but they solve retrieval *quality*, not corpus *growth*; growth is handled by write-time
deduplication, supersession pruning, and the Dreaming consolidation machinery Aeon already has.
And rationale capture is not a third feature: it is a required field on the supersession edge, and
it is simultaneously your GDPR Article 16/19 mechanism. One version graph discharges all three
requirements plus two legal duties. That unification, placed inside the memory kernel rather than
in an application layer, is where the novelty you asked for actually lives.

## Question 1: git-like versioning

### What already exists

The idea of version-controlling agent memory went from empty to crowded during 2025–2026, so the
wheel-reinvention risk here is real and specific. ChronoMem is a semantic version-control layer for
agent memory integrated into Google's Agent Development Kit: it commits whole-memory snapshots at
each write, keeps structured version histories, and maps natural-language undo requests onto
historical versions ([Su et al. 2026](https://arxiv.org/abs/2607.27773), preprint). Git Context
Controller manages agent context with explicit COMMIT/BRANCH/MERGE operations
([Wu et al. 2025](https://arxiv.org/abs/2508.00031), preprint). GitOfThoughts stores the reasoning
tree literally as a git repository, every scored thought a commit, outcomes as tags
([Shekar et al. 2026](https://arxiv.org/abs/2606.14470), preprint). Kumiho grounds versioned agent
memory in AGM belief-revision semantics, with immutable revisions, mutable tag pointers and typed
dependency edges ([Park 2026](https://arxiv.org/abs/2603.17244), preprint). WorldDB makes every
memory node content-addressed and immutable, so any edit produces a new hash at the node and every
ancestor, a Merkle-style audit trail ([Ganesan 2026](https://arxiv.org/abs/2604.18478), preprint).

Two of these are close enough to your exact setting that you must position against them by name.
MOOSEDev gives coding agents ontology-grounded project memory whose records carry lifecycle status,
provenance and supersession links, and reports near-perfect answer sets (0.98–1.00) on
supersession queries where a production vector-memory tool fails ([Adam
2026](https://arxiv.org/abs/2608.13662), preprint). And MemClaw is a production multi-tenant shared
memory service that formalizes the fleet-memory problem around exactly your four failure modes —
unauthorized leakage, stale propagation, contradiction persistence, provenance collapse — with
scoped retrieval, temporal supersession, provenance tracking and policy-governed propagation as the
primitives, reporting 100% reconstruction of depth-four derivation chains ([Margalit et al.
2026](https://arxiv.org/abs/2606.24535), preprint).

So: versioned agent memory exists, supersession-with-provenance exists, and a production
multi-tenant governed shared memory exists. None of this was true when your v3 document was
written, and all of it postdates the systems it would have cited.

### What to build, given that

**Do not build git, and do not snapshot the corpus.** ChronoMem's whole-memory snapshots suit a
single agent whose memory is small; at org scale they conflate every engineer's writes into one
history and make Article 17 erasure a history-rewriting problem (deleting one person's fragment
from every snapshot is exactly the operation git is designed to resist). Git's merge machinery is
also solving a problem you do not have: merge exists for concurrent edits to the *same mutable
file*, and distilled fragments should never be mutable in place; the verdict memo already requires
supersede-with-provenance for GDPR reasons.

The right shape is **per-fragment immutable versions with typed edges**, closer to Kumiho and
MOOSEDev than to git:

- Every fragment version is immutable and content-addressed (WorldDB's insight; it also gives you
  tamper-evidence for the audit log almost free).
- Mutations are new versions linked by typed edges: `supersedes`, `refines`, `contradicts`,
  `revokes`, `merges-with`. Retrieval defaults to the head of each chain; history is reachable but
  never retrieved by accident.
- The org tier is not a "branch" in the git sense; it is a scope, and promotion (which the GDPR
  analysis already requires to mint a *new* fragment) is therefore a version-graph edge
  (`promoted-from`) rather than a copy. Your versioning requirement and your compliance requirement
  turn out to be the same mechanism.

**Sync is simpler than you fear, because of a decision already taken.** The verdict memo
recommends the shared tier be centralized with the master as the single write authority. That
kills the hard distributed problem: clients never merge, they *pull* an ordered per-scope
supersession log (git fetch of a linear history, per scope) and *propose* promotions upstream.
No CRDTs, no conflict resolution, no vector clocks. A client pinned at log position N is merely
stale, never divergent, and the per-recipient acknowledgement state that Article 19 requires
falls out of the pull protocol for free: the master knows exactly which client has consumed which
supersession.

**The engine-level opportunity.** Every system above is an application layer over Postgres, a
vector DB, or literal git. Nobody has put version semantics inside the memory engine itself. Aeon
is unusually well positioned to: Trace is already an append-only event log (a commit log by
construction), the WAL gives you durable ordering, epoch-based reclamation already gives readers a
consistent view under concurrent writes (an MVCC seed), and compaction is the natural place for
version-chain garbage collection (fold superseded tails older than the retention clock). "Memory
as a versioned OS resource" extends Aeon's kernel framing where the competition has only shims.

## Question 2: growth and "vector haze"

### The correction first

Contextual embeddings do not address growth. Late chunking embeds all tokens of a long text and
chunks after the transformer, so each chunk carries full document context ([Günther et al.
2024](https://arxiv.org/abs/2409.04701), preprint); contextual document embeddings condition each
document's representation on its corpus neighbors ([Morris & Rush
2024](https://arxiv.org/abs/2410.02525)); Anthropic's contextual retrieval prepends an
LLM-generated situating sentence to each chunk before embedding and indexing, reporting a 49%
retrieval-failure reduction combined with contextual BM25 (vendor engineering report, 2024). All
three make each vector *better*. None makes the corpus *smaller*. Adopt them — the fit is
genuinely good, see below — but as retrieval-quality tools, not as your bloat answer.

Also, a calibration: shared engineering knowledge grows linearly in engineer-days, not
exponentially. What degrades is not storage but retrieval precision as near-duplicates and
contradictions accumulate: the "haze" you named. The literature has localized that problem
precisely, and it is not where embeddings can fix it: cosine similarity separates a *contradicted*
fact from a *duplicated* one at AUROC 0.59, near chance, because a superseded fix is often more
embedding-similar to the current one than a paraphrase is ([Yadav
2026](https://arxiv.org/abs/2606.26511), preprint). The haze is not an embedding-quality problem.
It is the *absence of supersession semantics*, which means your Question 1 mechanism is the
actual answer to your Question 2.

### The growth pipeline

Four stages, three of which reuse machinery Aeon already has:

1. **Admission control at write time.** Near-duplicate detection against the target scope before a
   fragment is admitted; a duplicate becomes a `refines` edge or a counter increment, not a new
   row. This is also your first poisoning checkpoint: consolidation-time verification of memory
   transitions is exactly what TrustMem trains a verifier for ([Yang et al.
   2026](https://arxiv.org/abs/2606.25161), preprint), and clean-looking but non-transferable
   experiences are a demonstrated low-privilege attack on self-evolving agents ([Wang et al.
   2026](https://arxiv.org/abs/2605.18930), preprint).
2. **Supersession pruning.** Retrieval sees chain heads only; superseded tails age out of the hot
   index into archival storage on the retention clock. The active set plateaus even as history
   grows.
3. **Dreaming, pointed at the shared tier.** Aeon already has `consolidate_subgraph` and the
   Dreamer: cluster related resolved fragments (the fourteen tickets about the same flaky auth
   timeout) into one summary fragment with `merges-with` edges back to the sources. This is the
   same tombstone-and-summarize flow the single-user Dreaming process already performs.
4. **Multi-resolution retrieval.** Serve the summary by default and expand to constituent
   fragments on demand — the shape MRMS argues for with its structured/vector/graph axes and
   short/medium/long temporal tiers ([Li & Shi-Nash 2026](https://arxiv.org/abs/2607.04617),
   preprint).

### Where contextual embeddings genuinely fit

A distilled fragment is a chunk ripped from its trace, the exact failure mode contextual
retrieval exists for ("the fix was to raise the timeout": which service? which bug?). Two natural
integration points: at distillation, prepend the situating context (task ID, subsystem, error
class) before embedding, which is Anthropic-style contextual retrieval landing on your existing
distillation step at near-zero marginal cost; and at promotion, since promotion already re-distills
and re-embeds the fragment for its new audience, condition the new embedding on the *target
scope's* corpus (the CDE move). Promotion-time re-embedding was mandatory for GDPR reasons anyway;
conditioning it on the destination corpus makes the compliance step also a retrieval-quality step.

## Question 3: change with reasons

This is the cheapest of the three, because it is a field, not a system. Every supersession edge
carries a required, structured rationale: a typed reason code (`superseded-by-commit`,
`contradicted-by-outcome`, `refactor-invalidated`, `policy-redaction`, `admin-correction`,
`erasure`), a free-text note, an evidence link (commit SHA, ticket ID, CI run), and the acting
principal. Enforce it in the write path: an edge without a rationale is rejected, not defaulted.

Three birds with this stone. Engineering: the org can answer "why did we stop believing X," which
is the question MOOSEDev's decision-rationale records exist for. GDPR: Article 16 rectification
*is* a supersession edge with reason `admin-correction`, and Article 19 recipient notification *is*
the supersession log pull — the compliance workflows stop being separate machinery. Forensics:
after a poisoning incident, the rationale chain plus content-addressing is your incident
reconstruction, subject to one caveat the firewall literature just named — LLM-based consolidation
can launder provenance, rewriting an untrusted observation as apparent user history, so rationale
and provenance fields must be platform-written, never model-written ([Xu et al.
2026](https://arxiv.org/abs/2607.29167), preprint).

## Novelty map

Taken (do not reinvent, cite and build on): git-like memory operations (ChronoMem, GCC,
GitOfThoughts); belief-revision formalization of versioned memory (Kumiho); content-addressed
immutable memory nodes (WorldDB); supersession + provenance + lifecycle for coding-agent knowledge
(MOOSEDev); governed multi-tenant shared memory with your four failure modes (MemClaw); contextual
chunk embeddings (late chunking, CDE, Anthropic contextual retrieval).

Open — the composition no one has shipped, and your credible novelty claim for a v4 paper:

1. **Version semantics in the memory kernel.** WAL-durable typed supersession edges, epoch-based
   consistent reads over version chains, compaction-integrated version GC — versus every existing
   system's application-layer shim. This is Aeon's home turf and extends the OS framing.
2. **Outcome-verified supersession for code knowledge.** Nobody triggers supersession from ground
   truth: fragment cites commit → commit reverted or superseded → fragment auto-superseded with
   reason and evidence. Embedding similarity cannot detect staleness (AUROC 0.59); CI and the
   commit graph can. This turns the integrity gate from the verdict memo into a live, self-updating
   property.
3. **Promotion as mint-and-recontextualize.** One operation that simultaneously satisfies the GDPR
   minting requirement, re-embeds in the destination scope's context (CDE/contextual-retrieval),
   and creates the `promoted-from` version edge with rationale. Three literatures — compliance,
   retrieval, versioning — meet in a single primitive. This is the paper-worthy synthesis.

One sobering result to carry into the outcome experiment: GitOfThoughts, having built exactly the
versioned memory you want, tested whether memory from past problems improves accuracy on new
problems across five memory stores, two benchmarks and pre-registered repeats, and the answer was
no ([Shekar et al. 2026](https://arxiv.org/abs/2606.14470), preprint). That does not doom your use
case (organizational bug-fix knowledge is more reusable than reasoning traces), but it is the
second published null for agent memory payoff, and it hardens the case that Stage 4's outcome
experiment should run before the build-out is funded, not after.

## Build order delta

Slot into the verdict memo's roadmap without displacing anything: version-graph schema and typed
edges land in Stage 1 (they are scope metadata's siblings in the WAL work); admission dedup and
supersession-aware retrieval in Stage 2; contextual distillation, promotion-time recontextualization
and the rationale-bearing console views in Stage 3; Dreaming-over-shared-tier and outcome-verified
supersession in Stage 4, where the outcome experiment can measure them.
