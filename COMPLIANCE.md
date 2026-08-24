# Aeon Shared Tier — Adopter Compliance Guide

> Applies to the Stage 4 shared tier (promotion pipeline, admin console, crypto-erase) described in
> `v4-plan.md`. Read this before enabling `AEON_SHARED_ATLAS_PATH` / `AEON_CONTROL_PLANE_DATABASE_URL`
> in any deployment that will hold real user data.

## 1. What this document is, and isn't

Aeon ships the engineering primitives a shared, cross-user memory tier needs to be operated
responsibly: de-identifying promotion, a tamper-evident audit log, scoped admin access, and per-subject
key destruction. It does **not** ship a compliance program. Whether your deployment needs a DPIA, a
lawful basis assessment, or works-council sign-off depends on your jurisdiction, your users, and what
you configure Aeon to store — that determination is yours to make, and this document exists to give you
the accurate technical facts to make it with, not to make it for you.

Everything below is a description of what the code in this repository actually does, verified by
reading it and by test, not aspirational language carried over from a design doc. Where a stated
guarantee has a real limit, the limit is stated in the same paragraph, not left for you to discover
later.

---

## 2. What data classes can end up in shared-tier fragments

Aeon has two storage tiers, and they are held to different standards:

- **Private store** (per-user/session Atlas): whatever raw text and vectors a user's own conversation
  produces. No de-identification, no cross-user visibility, no crypto-erase — isolation here is
  session/scope-level (Stage 0), not per-owner encryption. If a user in your deployment has a right to
  erasure that must cover their private store too, you need a separate mechanism for it; **the erasure
  workflow described in §6 covers the shared store only** (see §6.4).
- **Shared store** (org-wide Atlas, reachable via `promote_fragment()` only): fragments a maintainer or
  automated caller has explicitly promoted. Two things live here per node:
  - **Metadata** (the redacted fragment text) — encrypted at rest once crypto-erase is configured (§5).
  - **A centroid vector** (the embedding) — **always stored in plaintext**, with or without crypto-erase
    enabled. See §5.1 for why, and what that means for erasure.

A promoted fragment is redacted before it is written (§3), but redaction is pattern-based, not a
semantic understanding of the text. Treat "promoted" as "passed a configured deterministic filter,"
not "verified free of personal data" — see §4 for the filter's own limits.

---

## 3. What "promote" actually does

Promotion is **mint-and-recontextualize**, not a visibility flag flip. `promote_fragment()`:

1. Reads the source fragment's text and vector from the **private** store. The source node is never
   mutated, never deleted, never re-scoped — it stays exactly as it was, in the private store, under
   whatever retention/erasure policy applies there.
2. Runs the fragment through the identifier-corpus classifier (§4), which redacts or rejects it.
   If this deployment has `AEON_REQUIRE_CODE_VERIFICATION` enabled, promotion is also gated on a
   caller-supplied verification-run result (§4.1) — checked first, before the classifier runs.
3. Inserts a **new, distinct** node into the shared store, containing the redacted text, a scope, and a
   link back to an audit record. The vector is the **source vector unmodified** by default; a caller (or,
   at the HTTP layer, the `destination_embedding` field on the promotion-execute request) may instead
   supply a destination-conditioned vector it has already computed — Aeon does not run any embedding model
   itself and does not validate that vector's semantic quality, only its dimension (via the destination
   store's own insert-time check).
4. Records an audit-log entry naming the actor, source, and destination scope — **before** the
   shared-store node is fully scoped/governed, specifically so a promotion that fails partway through is
   still discoverable, never silently lost. If the caller supplies a `trace` argument, a `promoted-from`
   Trace edge is also recorded linking the new node back to its source; this edge is **optional**, not a
   guaranteed provenance link — a call to `promote_fragment()` with no `trace` produces no edge, only the
   audit-log record.
5. Requires a non-blank `subject_id` for every promoted node (fail-closed). This is what makes
   crypto-erase (§5) possible at all — a fragment with no recorded subject cannot later have its key
   destroyed for that subject.

A promoted node's presence in the shared store never implies the source fragment was deleted, redacted
in place, or made unavailable to the original owner. If your compliance obligations require "delete
everywhere this data was copied to," promotion is one of the copies you must separately account for
alongside the private-store original.

---

## 4. The classifier: what it guarantees, and what it doesn't (yet)

`IdentifierCorpus` / `classify_and_redact()` is a **fail-closed, deterministic** filter: regex patterns
plus generic email/commit-SHA redaction toggles, all adopter-configured via
`AEON_IDENTIFIER_CORPUS_PATTERNS` / `AEON_REDACT_EMAILS` / `AEON_REDACT_COMMIT_SHAS`. It is the **only**
layer permitted to pass a fragment through to the shared store. An unconfigured corpus (no patterns, both
redaction toggles off) rejects **every** fragment — the promotion endpoint returns 503, not a silent
pass-through, so "I forgot to configure this" fails safe rather than fails open.

**What is not yet true, stated plainly rather than left implicit**: Stage 4's own roadmap gate requires
"measured false-negative rate for the identifier-corpus detector on a labelled corpus of real distilled
fragments, published alongside the feature." As of this writing **that measurement has not been done** —
there is no labelled corpus of real fragments in this repository to measure against, and doing so
requires data this project does not have. This is not a theoretical caveat: it is a currently open item
against Aeon's own release gate.

**What this means for you**: if you configure the classifier and promote fragments through it, you are
relying on regex/pattern coverage that has not been independently measured for false-negative rate
against real-world text shaped like yours. Before treating promoted content as "safe to share broadly,"
validate the classifier against a labelled sample of your own actual fragments (real internal identifiers,
real PII shapes, real formats your organization uses) and track your own false-negative rate. Do not
assume the shipped defaults (`redact_emails`/`redact_commit_shas`) are sufficient for your data — they
are generic patterns, not a tuned model of your corpus.

An optional LLM-based second layer is explicitly designed to only **reject or flag**, never override a
rejection into a pass — there is deliberately no code path for an LLM layer to widen what the
deterministic classifier already narrowed. No such layer ships today; if you add one, preserve this
asymmetry.

### 4.1 Correctness-gated promotion for code knowledge (optional)

Aeon does not integrate with any VCS/CI provider — there is no code anywhere in this project that talks
to GitHub Checks, GitLab pipelines, or any other CI system. Instead, `AEON_REQUIRE_CODE_VERIFICATION` (off
by default) lets you require a **caller-supplied** verification result before a promotion is allowed to
proceed: your own CI/test runner determines whether a code fragment's associated commit passed your own
checks, and hands that outcome to Aeon at promotion-execution time (a `status`/`commit_sha`/`verified_by`
triple). A `status` other than the exact string `"passed"` — or no verification supplied at all, when this
flag is on — fails the promotion closed (recorded as a rejection, identical in shape to a classifier
rejection: a 200 response with no promoted node, not an error, and the four-eyes approval already granted
on the request is **not** consumed, so a retry once your CI passes still works against the same approval).

**What this does and does not verify**: Aeon trusts whatever `status` the caller supplies. It has no way
to confirm that a "passed" result actually corresponds to the commit named in the same payload, or that
the caller's CI system is not compromised or misconfigured. This is a trust boundary, not an independent
verification — the guarantee is "promotion cannot proceed without *someone* asserting a pass," not "Aeon
has confirmed the code is correct."

---

## 5. Crypto-erase: what it guarantees, and its two explicit boundaries

### 5.1 Scope: metadata and blob payloads only, not vectors

Crypto-erase covers a shared-store node's **metadata field** (the redacted text). It does **not** cover
the centroid **vector** — vectors are always stored as plaintext floats in the mmap file, whether or not
crypto-erase is configured.

This was a deliberate, cost-driven scope decision (see `v4-plan.md`'s task 6 decision record for the
full reasoning), not an oversight: covering vectors would require a file-format version bump, decrypt-
at-open into anonymous memory for the whole store, and loss of the "mmap can exceed RAM" property for
the shared tier — a kernel-scale change roughly comparable to Stage 4's own physical-separation work, for
a residual risk that a smaller mitigation (redaction before embedding) already narrows.

**The residual risk, stated precisely**: embeddings are invertible. A destroyed subject's vector remains
partially informative about the original text even after the corresponding metadata key is destroyed.
Destroying a key removes your ability to read the stored *text* back; it does not erase the fact that a
vector geometrically close to "what that text meant" still sits in the file. If your jurisdiction's
erasure obligation extends to any representation that could be used to reconstruct personal data — not
just to the plaintext itself — vectors of erased subjects are a gap this implementation does not close.
Mitigations available to you: keep the shared tier's promoted content short and low-specificity (the
classifier + short-metadata convention already pushes this direction), and treat vector-inversion risk as
a factor in deciding what's eligible for promotion in the first place, not just what's eligible for
erasure later.

### 5.2 The mechanism: per-(subject, scope) keys, wrapped, destroyed by deletion

Every promoted node's metadata is encrypted (AES-256-GCM) under a key unique to its `(subject_id, scope)`
pair. That key is itself wrapped under a single deployment-wide KEK read from
`AEON_CRYPTO_ERASE_KEK_HEX` (exactly 32-byte/256-bit hex is required — not merely any AESGCM-valid
length — so the wrapping is never weaker than the 256-bit keys it protects; the deployment fails closed —
503, not a default key — if this is unset, malformed, or the wrong length). Erasing a subject's data in a
scope **deletes** the corresponding key row
outright; it is not a soft flag, and a deployment that later tries to read that node's metadata gets
ciphertext back, not plaintext, permanently.

**Two envelopes, both must hold for the guarantee to be real**:

1. **The key material.** A Postgres `DELETE` does not, by itself, guarantee the bytes are gone —
   write-ahead logs, point-in-time-recovery archives, and un-VACUUMed heap pages can all retain a deleted
   row for a period defined by *your* Postgres configuration and backup retention policy, not by this
   code. This is why the key is stored **wrapped**, not raw: as long as your KEK itself is destroyed or
   rotated out of reach on your own schedule, a recovered pre-deletion backup of the `subject_scope_keys`
   table is unusable ciphertext, not a working key. You are responsible for having an actual KEK
   rotation/retirement plan if you want backup-retained key rows to stop mattering on a defined timeline.
2. **The mmap ciphertext file.** Filesystem snapshots and volume-level backups of the shared Atlas file
   can retain old ciphertext indefinitely — this is fine, and expected, **provided** the key from (1) is
   genuinely gone. Neither envelope alone is a guarantee; only the combination is.

### 5.3 Collateral effect of scope-level (not per-node) keys

One key covers **every** node sharing the same `(subject_id, scope)` pair. If an erasure case
deliberately targets only some of a subject's fragments in a scope, destroying that key makes **every**
node sharing the pair unreadable — including ones the case didn't name. This is a designed consequence of
how the key hierarchy is shaped, not a bug: an erasure case is documented as erasing "a subject's
shared-tier fragments" (plural, all-in-scope), and a case that deliberately erases only a subset should
be understood to carry this side effect on what's left behind. If your process requires partial,
node-granular erasure with survivors remaining readable, this mechanism does not support that within a
single scope — you would need one scope per erasable unit, which is a modeling decision on your side, not
a configuration flag here.

### 5.4 `metadata_size` does not retroactively grow

The shared Atlas store's `metadata_size` (the byte budget for encrypted text, set at file-creation time)
does not change after the file exists. A store created before crypto-erase was ever configured, at the
engine's 256-byte default, leaves exactly 155 usable plaintext bytes once encryption overhead is
subtracted (`crypto.max_plaintext_bytes(256)`) — noticeably less headroom than the 347 usable bytes a
store created fresh with the recommended 512-byte size gets (`crypto.max_plaintext_bytes(512)`). If you
plan to enable crypto-erase, set
`AEON_SHARED_ATLAS_METADATA_SIZE=512` (or larger) **before** creating the shared Atlas file — there is no
migration path in this release for growing an existing store's field width.

---

## 6. Admin console: access control and what is (and isn't) reason-gated

### 6.1 Roles and scope

Admin access is granted per `(principal, scope_mask, role)` via `AdminDB.grant_role()`. The only role
defined today is `"admin"`. A grant with a non-null `expires_at` is exactly Aeon's break-glass mechanism
— time-boxed access is the same grant row as permanent access, just with an expiry, not a separate
privileged path. Expiry is checked at read time on every authorization check (no scheduler, no sweeper),
so a grant that has expired is invalid immediately, with no window where a stale grant is still honored.

Granting a role over every scope at once (`scope_mask == ALL_SCOPES_VISIBLE`) is intentionally not the
path of least resistance — it raises unless the caller explicitly opts in
(`allow_wildcard=True`), consistent with "scope-scoped admin roles by default, never a wildcard bypass."

### 6.2 Authorization: containment for actions, overlap for browsing

Acting on a node (supersede/tombstone/include in an erasure case) requires the caller's grants to fully
**contain** every scope bit the target node carries — a caller granted only scope A cannot act on a node
also scoped to B, even though their grant overlaps it. Merely **browsing** the knowledge list uses
overlap semantics instead — a caller sees any node touching a scope they hold a grant over. This asymmetry
("see broadly, act narrowly") is deliberate, not an oversight: it lets an admin discover what exists in
scopes adjacent to their own without granting them the ability to mutate it.

### 6.3 Four-eyes approval

Promotion execution and erasure execution both require `required_approvals` distinct approvers
(`UNIQUE(request_id, approver)` makes the same person approving twice a no-op toward the count, not a
second vote). Every approval request carries a mandatory, non-blank `reason` and a mandatory
`expires_at` — a pending request cannot sit forever, the same "time-boxed" property break-glass access
gets. The exact operation parameters (destination scope, target node ids) are locked into the request at
creation time, not re-supplied at execution time, so a replayed or reused approval cannot be redirected
to a different, unapproved target.

### 6.4 Read-reason: one route, not a blanket property

`GET /admin/knowledge` — the only admin route that returns decrypted subject content — requires a
non-blank `reason` query parameter and appends one audit-log record per request naming the actor, the
reason, and the number of nodes returned (never the nodes' own text). This is deliberately narrower than
"every admin read requires a reason": the audit-log routes (`GET /admin/audit-log`, `.../verify`,
`.../export`) and the erasure-case-status route (`GET /admin/erasure/{case_id}`) return the audit trail
itself or a receipt of ids/failure strings, not subject content, and are authenticated and scope-checked
but not reason-gated. If your own compliance program requires a reason on every privileged read
regardless of what it returns, this is narrower than that — extend it before relying on it for that
purpose.

The audit log also currently applies no per-scope filtering to its own read routes: any non-expired
`admin` grant, over any single scope, is sufficient to read the entire log, across all scopes. There is
no scope-restricted "auditor" role in this release.

### 6.5 The audit log itself

Every governance-affecting write (promotion, promotion rejection, an unscoped-promotion anomaly, erasure,
and now knowledge-browse reads) is appended to a local, hash-chained, append-only JSONL file —
`GET /admin/audit-log/verify` independently walks the chain and reports the first record that fails to
reconcile, catching tampering or deletion anywhere in the log. `GET /admin/audit-log/export` produces an
HMAC-signed export (key from `AEON_AUDIT_LOG_EXPORT_KEY_HEX`; fails closed with 503, not a default key,
if unset) for handing to an external verifier. Audit payloads are required, by convention enforced at
every call site in this codebase, to contain categories and counts, never the raw values a classifier
redacted — an audit log that stored what redaction removed would defeat the reason redaction exists.

### 6.6 Erasure workflow

`POST /admin/erasure` files a case naming target shared-store node ids and a reason; execution is
four-eyes-gated exactly like promotion. A completed case returns one receipt with two explicit sections —
`erased` and `could_not_erase` (with a reason per failed id) — so a partial outcome is a legitimate,
auditable completion, not a case left dangling. Execution is crash-resumable: a process killed mid-run
leaves the case retryable, and re-attempting an already-erased node is a safe no-op.

**Deferred, stated explicitly**: erasure targets the **shared** store only. The private store has no
per-owner authorization model in this codebase (private-store isolation is session/scope-level, not
per-node ownership tagging) — there is nothing today for an erasure endpoint to check authorization
against for a private-store node. If a data-subject request in your deployment must also cover private-
store content, you need a separate process for it; this release does not provide one.

---

## 7. Configuration reference

| Env var | Purpose | Fails closed? |
|---|---|---|
| `AEON_SHARED_ATLAS_PATH` | Enables the shared tier at all | Shared tier absent (404s) if unset |
| `AEON_SHARED_ATLAS_METADATA_SIZE` | Metadata field width for a **newly created** shared store (default 512; see §5.4) | Falls back to 512 on a bad value, does not crash import |
| `AEON_CONTROL_PLANE_DATABASE_URL` | Postgres control plane (governance, admin roles, approvals, erasure cases, crypto-erase keys) | Every control-plane feature 404s/503s without it. A shared store holding **never-encrypted** nodes can still be browsed without it. A store holding nodes promoted **with** a keystore cannot meaningfully be browsed without it — the knowledge browser degrades to returning marker-prefixed ciphertext rather than raising, but that is not the same as the content being readable. |
| `AEON_AUDIT_LOG_PATH` | Local hash-chained audit log file | Defaults to `./data/governance/audit.jsonl` |
| `AEON_AUDIT_LOG_EXPORT_KEY_HEX` | HMAC key for signed audit-log export | 503 if unset, never a default key |
| `AEON_CRYPTO_ERASE_KEK_HEX` | Wraps every per-subject-per-scope DEK | 503 if unset or the wrong length/format, never a default key |
| `AEON_IDENTIFIER_CORPUS_PATTERNS`, `AEON_REDACT_EMAILS`, `AEON_REDACT_COMMIT_SHAS` | Classifier configuration | Promotion rejects everything (503) if the corpus is entirely unconfigured — see §4 |
| `AEON_REQUIRE_CODE_VERIFICATION` | Gates promotion on a caller-supplied CI/test verification result | Off by default (verification, if supplied, is recorded but does not gate); when on, a missing/failed result rejects the promotion (200, no node minted) rather than raising — see §4.1 |

---

## 8. Checklist before enabling the shared tier

This is a checklist of what to go decide, not a set of settings that make the decision for you.

- [ ] **Lawful basis / DPIA**: determine whether promoting any user content into a cross-user shared
      store requires a documented lawful basis or impact assessment in your jurisdiction, independent of
      anything this codebase does.
- [ ] **Classifier validation**: assemble a labelled sample of your own real fragment text and measure
      the identifier-corpus classifier's false-negative rate against it before trusting promoted content
      to be free of the identifier classes you configured it to catch (§4 — this has not been measured
      upstream).
- [ ] **Retention/backup policy vs. erasure**: define how long your Postgres WAL/PITR archives and
      volume-level backups of the shared Atlas file are retained, and whether that retention window is
      acceptable given crypto-erase's two-envelope guarantee (§5.2) — the guarantee is only as strong as
      your own backup lifecycle.
- [ ] **KEK and export-key custody**: decide who holds `AEON_CRYPTO_ERASE_KEK_HEX` and
      `AEON_AUDIT_LOG_EXPORT_KEY_HEX`, how they're rotated, and what happens to already-wrapped DEKs on
      a KEK rotation (this release does not implement KEK rotation — rotating the KEK without
      re-wrapping existing DEKs makes them unreadable).
- [ ] **Vector residual risk (§5.1)**: decide whether embedding-inversion risk on erased subjects'
      vectors is acceptable for your data classes, or whether it disqualifies certain content from
      promotion entirely.
- [ ] **Metadata field sizing (§5.4)**: set `AEON_SHARED_ATLAS_METADATA_SIZE` before first creating the
      shared store if you intend to enable crypto-erase from day one.
- [ ] **Admin role assignment process**: decide who can call `grant_role()`, what scope grants are
      appropriate per role in your org, and your policy for break-glass (non-null `expires_at`) grants —
      this codebase enforces the mechanism, not your assignment policy.
- [ ] **Erasure scope**: confirm your data-subject request process covers the private store separately —
      the shared-tier erasure workflow (§6.6) does not reach it.
- [ ] **Reason-gating scope (§6.4)**: decide whether "reason required only for `GET /admin/knowledge`" is
      sufficient for your own audit requirements, or whether you need to extend reason-gating to the
      other admin read routes.
