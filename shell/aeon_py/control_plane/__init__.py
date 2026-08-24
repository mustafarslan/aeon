"""
Aeon Control Plane — Postgres-backed governance state (v4-plan.md Stage 4
task 1: "Governance state (relational, low-QPS, transactional,
history-bearing) lives in a separate FastAPI + Postgres control plane;
the C++ core stays enforcing-only.").

This increment: schema + migrations + the governance-record write path
`promote_fragment` (promotion.py) uses. Deliberately does NOT include
approvals or roles yet (task 7) -- four-eyes approval needs a
pending/approved/executed/expired state machine, which belongs on top of
a schema that's already settled, not designed in the same pass.

Optional infrastructure, same pattern as `shared_atlas_client`
(client.py): a deployment with no Postgres configured falls back to
`promote_fragment`'s pre-Stage-4-control-plane behavior (the local
hash-chained AuditLog's own seq number as governance_record_id) rather
than requiring Postgres to promote anything at all.
"""
