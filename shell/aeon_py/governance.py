"""
Aeon Governance — hash-chained audit log (v4-plan.md Stage 4 task 5(a)).

Built first, ahead of the rest of the admin console (knowledge browser,
erasure workflow), per the plan's own ordering: "retrofitting onto existing
write paths reliably misses some." Every governance-affecting write
(promotion today; erasure/bulk-scope-remap in later increments) appends one
record here as part of the same operation, not after the fact.

No Postgres dependency (v4-plan.md Stage 4 step 3 addendum decision):
mint-and-recontextualize doesn't need a transactional control-plane lookup
at promotion time, only a durable audit trail after -- a local,
append-only, hash-chained JSONL file satisfies that on its own. A
monotonic `seq` from this log is what `NodeHeader.governance_record_id`
(Stage 1) gets set to; swapping that to a future Postgres row id is a
one-line change at the promotion call site, not a schema change here.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

GENESIS_HASH = "0" * 64

# Known `action` values written by promotion.py's three AuditLog.append()
# call sites, plus erasure.py's one (v4-plan.md Stage 4 task 5(c) -- an
# erasure case's completion is a governance-affecting event exactly like a
# promotion, so it gets the same audit-log-before-Postgres-mirror
# treatment). Defined here (governance.py has zero external dependencies)
# rather than in control_plane/schema.py (which needs sqlalchemy, an
# OPTIONAL dependency promotion.py/erasure.py must keep working without) so
# both modules -- and the Postgres CHECK constraint schema.py derives from
# this same tuple -- share one source of truth instead of independently
# maintained lists that could drift.
GOVERNANCE_RECORD_ACTIONS = (
    "promotion",
    "promotion_rejected",
    "promotion_unscoped_anomaly",
    "erasure",
    # v4-plan.md Stage 5 task 2: outcome-verified supersession
    # (supersession.py). Two distinct values, not one with a
    # payload["revoked"] flag, matching the existing promotion/
    # promotion_rejected split -- a queryable "every row where
    # action=='supersession'" is worth more than saving one migration.
    "supersession",
    "supersession_revoked",
)

# Deliberately does NOT include read-path actions (e.g. server.py's
# list_knowledge -- "knowledge_read", the task 7 "mandatory read-reason
# prompts" record). A read never produces a NodeHeader.governance_record_id
# to attach anything to, so it never calls GovernanceDB.record() -- only
# AuditLog.append() (local, no Postgres CHECK constraint to satisfy). Adding
# a read action here would force a migration onto this tuple's Postgres
# mirror (control_plane/schema.py's ck_governance_records_action) for a
# value no governance_records row will ever hold.


class AuditLogError(Exception):
    """Raised by AuditLog.verify() naming the first record where the hash
    chain doesn't reconcile -- tampering, truncation, or corruption
    anywhere in the log invalidates every record after that point, which
    is the entire point of hash-chaining. Also raised by the constructor
    (via _load_tail()) if the log file's final line is unreadable JSON --
    a torn tail from a crash mid-write, requiring manual recovery before
    the log can be reopened."""


@dataclass(frozen=True)
class AuditRecord:
    seq: int
    prev_hash: str
    action: str
    actor: str
    payload: Dict[str, Any]
    entry_hash: str


class AuditLog:
    """Append-only, hash-chained governance audit log.

    Not thread-safe by itself -- callers sharing one AuditLog across
    threads must serialize their own append() calls (same expectation as
    AeonClient's underlying mmap writers, which the caller is already
    responsible for serializing against).
    """

    def __init__(self, path: "str | Path"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seq, self._last_hash = self._load_tail()
        self.instance_id = self._load_or_create_instance_id()

    def _load_or_create_instance_id(self) -> uuid.UUID:
        """A stable identifier for THIS specific log file, generated once
        on first creation and persisted in a sidecar `.instance_id` file
        next to `path` (not derived from the path itself -- a path is
        exactly the thing that can change on rotation/relocation, which
        is the ambiguity this identifier exists to survive). Used by the
        control plane (control_plane/schema.py's
        governance_log_instances table) to resolve a Postgres-assigned
        governance_records.id back to the specific (log file, seq) it
        names, even after the file has moved."""
        id_path = self.path.with_suffix(self.path.suffix + ".instance_id")
        if id_path.exists():
            return uuid.UUID(id_path.read_text().strip())
        new_id = uuid.uuid4()
        id_path.write_text(str(new_id))
        return new_id

    def _load_tail(self) -> tuple:
        if not self.path.exists():
            return 0, GENESIS_HASH
        seq, last_hash = 0, GENESIS_HASH
        with open(self.path, "r") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    # A torn final line (process killed mid-write) is a
                    # real possibility despite append()'s flush+fsync --
                    # fail loudly rather than silently truncating a
                    # security-relevant audit log out from under a
                    # caller. Recovery is a manual decision (inspect the
                    # file, decide whether to truncate the torn tail and
                    # accept the loss of that one record), not something
                    # this constructor should do on a caller's behalf.
                    raise AuditLogError(
                        f"line {line_no}: unreadable record ({e}) -- the "
                        "log file may have a torn tail from a crash "
                        "mid-write. Manual recovery required (inspect "
                        f"{self.path}, decide whether to truncate the "
                        "torn line) before this log can be reopened."
                    ) from e
                seq = rec["seq"]
                last_hash = rec["entry_hash"]
        return seq, last_hash

    @staticmethod
    def _compute_hash(seq: int, prev_hash: str, action: str, actor: str,
                       payload: Dict[str, Any]) -> str:
        canonical = json.dumps(
            {
                "seq": seq,
                "prev_hash": prev_hash,
                "action": action,
                "actor": actor,
                "payload": payload,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def append(self, action: str, actor: str, payload: Dict[str, Any]) -> int:
        """Appends one record, returns its seq number.

        `payload` must never itself contain PII the classifier redacted
        (client.py/promotion.py) -- callers should log categories/counts,
        not raw matched values, or this log becomes exactly the kind of
        store the redaction step exists to prevent.
        """
        seq = self._seq + 1
        entry_hash = self._compute_hash(seq, self._last_hash, action, actor, payload)
        record = {
            "seq": seq,
            "prev_hash": self._last_hash,
            "action": action,
            "actor": actor,
            "payload": payload,
            "entry_hash": entry_hash,
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
            # promotion.py's whole "record before mutating" safety
            # argument depends on this record surviving a crash between
            # the write and process exit -- a page-cache-only write
            # doesn't guarantee that. flush() pushes it out of Python's
            # buffer; fsync() forces the OS to actually persist it.
            f.flush()
            os.fsync(f.fileno())
        self._seq = seq
        self._last_hash = entry_hash
        return seq

    def tail(self, since_seq: int = 0, limit: int = 100) -> list:
        """Console primitive (v4-plan.md Stage 4 task 5(a)): returns up to
        `limit` records with seq > since_seq, in ascending seq order --
        pagination for an HTTP listing endpoint. A plain linear scan of
        the JSONL file, same cost class as verify()/export_signed() (both
        already read the whole file); this log is not expected to reach a
        size where that matters before a real Postgres-backed listing
        (governance_records, control_plane/db.py) becomes the console's
        primary read path instead. Skips a torn/corrupt tail line the same
        way `_load_tail()`'s constructor scan does NOT -- unlike the
        constructor (which must fail loudly, since silently accepting a
        torn log would corrupt this instance's own hash-chain state), a
        read-only listing call raises AuditLogError immediately on the
        first bad line, since a torn tail here means the log needs the
        same manual recovery `_load_tail()` already documents, not a
        best-effort partial page.
        """
        if not self.path.exists():
            return []
        results = []
        with open(self.path, "r") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    raise AuditLogError(
                        f"line {line_no}: unreadable record ({e}) -- torn "
                        "tail, see AuditLog's constructor doc comment for "
                        "recovery guidance"
                    ) from e
                if rec["seq"] <= since_seq:
                    continue
                results.append(
                    AuditRecord(
                        seq=rec["seq"], prev_hash=rec["prev_hash"],
                        action=rec["action"], actor=rec["actor"],
                        payload=rec["payload"], entry_hash=rec["entry_hash"],
                    )
                )
                if len(results) >= limit:
                    break
        return results

    def verify(self) -> None:
        """Walks the whole log, recomputing and reconciling the hash
        chain. Returns normally if every record checks out; raises
        AuditLogError naming the first record that doesn't."""
        if not self.path.exists():
            return
        prev_hash = GENESIS_HASH
        expected_seq = 1
        with open(self.path, "r") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec["seq"] != expected_seq:
                    raise AuditLogError(
                        f"line {line_no}: expected seq {expected_seq}, "
                        f"got {rec['seq']} -- record missing or reordered"
                    )
                if rec["prev_hash"] != prev_hash:
                    raise AuditLogError(
                        f"line {line_no} (seq {rec['seq']}): prev_hash "
                        "mismatch -- chain broken, log may have been "
                        "tampered with or truncated"
                    )
                recomputed = self._compute_hash(
                    rec["seq"], rec["prev_hash"], rec["action"],
                    rec["actor"], rec["payload"],
                )
                if recomputed != rec["entry_hash"]:
                    raise AuditLogError(
                        f"line {line_no} (seq {rec['seq']}): entry_hash "
                        "mismatch -- record contents were modified after "
                        "being written"
                    )
                prev_hash = rec["entry_hash"]
                expected_seq += 1

    def export_signed(self, key: bytes) -> bytes:
        """Exports the raw log bytes plus an HMAC-SHA256 signature, so a
        recipient holding `key` can confirm the export wasn't altered in
        transit -- independent of, and in addition to, the chain's own
        internal tamper-evidence (verify()). HMAC rather than an
        asymmetric scheme: this is the starting point the console (task
        5) needs today (a shared operator secret); swapping to ed25519
        later doesn't change this method's return shape or any caller's
        usage of it.
        """
        raw = self.path.read_bytes() if self.path.exists() else b""
        signature = hmac.new(key, raw, hashlib.sha256).hexdigest()
        return json.dumps(
            {"log": raw.decode("utf-8"), "signature": signature}
        ).encode("utf-8")

    @staticmethod
    def verify_export_signature(exported: bytes, key: bytes) -> bool:
        data = json.loads(exported)
        raw = data["log"].encode("utf-8")
        expected = hmac.new(key, raw, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, data["signature"])
