"""
V4 Stage 4 task 5(a): hash-chained governance audit log.
"""
import json
import pytest

from aeon_py.governance import AuditLog, AuditLogError, GENESIS_HASH


class TestAuditLogAppendAndReload:
    def test_append_returns_monotonic_seq(self, tmp_path):
        log = AuditLog(tmp_path / "audit.jsonl")
        assert log.append("promotion", "alice", {"x": 1}) == 1
        assert log.append("promotion", "bob", {"x": 2}) == 2
        assert log.append("promotion", "alice", {"x": 3}) == 3

    def test_reload_from_existing_file_continues_chain(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        log1 = AuditLog(path)
        log1.append("promotion", "alice", {"x": 1})
        log1.append("promotion", "bob", {"x": 2})

        log2 = AuditLog(path)  # simulates a new process reopening the log
        seq = log2.append("promotion", "carol", {"x": 3})
        assert seq == 3

    def test_genesis_hash_for_empty_log(self, tmp_path):
        log = AuditLog(tmp_path / "audit.jsonl")
        assert log._last_hash == GENESIS_HASH

    def test_torn_tail_raises_on_reopen(self, tmp_path):
        # advisor review: a crash mid-write can leave a torn final line --
        # must fail loudly on reopen (manual recovery), not silently
        # accept a partial/corrupt record as if it were the real tail.
        path = tmp_path / "audit.jsonl"
        log1 = AuditLog(path)
        log1.append("promotion", "alice", {"n": 1})

        with open(path, "a") as f:
            f.write('{"seq": 2, "prev_hash": "abc"')  # torn, no closing brace

        with pytest.raises(AuditLogError):
            AuditLog(path)


class TestAuditLogTail:
    # V4 Stage 4 task 5(a): the console's audit-log listing endpoint pages
    # through tail() rather than reading the whole file itself.
    def test_tail_returns_records_in_ascending_seq_order(self, tmp_path):
        log = AuditLog(tmp_path / "audit.jsonl")
        for i in range(5):
            log.append("promotion", f"actor{i}", {"n": i})

        records = log.tail()
        assert [r.seq for r in records] == [1, 2, 3, 4, 5]

    def test_tail_since_seq_excludes_earlier_records(self, tmp_path):
        log = AuditLog(tmp_path / "audit.jsonl")
        for i in range(5):
            log.append("promotion", f"actor{i}", {"n": i})

        records = log.tail(since_seq=2)
        assert [r.seq for r in records] == [3, 4, 5]

    def test_tail_respects_limit(self, tmp_path):
        log = AuditLog(tmp_path / "audit.jsonl")
        for i in range(5):
            log.append("promotion", f"actor{i}", {"n": i})

        records = log.tail(limit=2)
        assert [r.seq for r in records] == [1, 2]

    def test_tail_on_empty_log_returns_empty_list(self, tmp_path):
        log = AuditLog(tmp_path / "audit.jsonl")
        assert log.tail() == []

    def test_tail_raises_on_torn_tail(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        log1 = AuditLog(path)
        log1.append("promotion", "alice", {"n": 1})
        with open(path, "a") as f:
            f.write('{"seq": 2, "prev_hash": "abc"')  # torn

        # Reopening a torn log already raises in the constructor
        # (TestAuditLogAppendAndReload's own coverage) -- this pins that
        # tail() ITSELF also fails loudly rather than returning a partial
        # page, for a caller that somehow holds an AuditLog instance whose
        # file was corrupted after construction (e.g. a concurrent
        # process crashed mid-write between two tail() calls).
        with pytest.raises(AuditLogError):
            log1.tail()


class TestAuditLogInstanceId:
    def test_instance_id_stable_across_reopen(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        log1 = AuditLog(path)
        log2 = AuditLog(path)  # simulates a new process reopening the log
        assert log1.instance_id == log2.instance_id

    def test_distinct_paths_get_distinct_instance_ids(self, tmp_path):
        log1 = AuditLog(tmp_path / "a.jsonl")
        log2 = AuditLog(tmp_path / "b.jsonl")
        assert log1.instance_id != log2.instance_id

    def test_instance_id_sidecar_file_created(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        log = AuditLog(path)
        sidecar = path.with_suffix(path.suffix + ".instance_id")
        assert sidecar.exists()
        assert sidecar.read_text().strip() == str(log.instance_id)


class TestAuditLogVerify:
    def test_verify_passes_on_untampered_log(self, tmp_path):
        log = AuditLog(tmp_path / "audit.jsonl")
        for i in range(5):
            log.append("promotion", f"actor{i}", {"n": i})
        log.verify()  # must not raise

    def test_verify_passes_on_empty_log(self, tmp_path):
        log = AuditLog(tmp_path / "audit.jsonl")
        log.verify()  # no file at all -- must not raise

    def test_verify_detects_modified_payload(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        log = AuditLog(path)
        log.append("promotion", "alice", {"amount": 100})
        log.append("promotion", "bob", {"amount": 200})

        # Tamper: rewrite the first record's payload without recomputing
        # entry_hash/prev_hash -- exactly what an attacker editing the
        # file directly would do.
        lines = path.read_text().splitlines()
        rec = json.loads(lines[0])
        rec["payload"]["amount"] = 999999
        lines[0] = json.dumps(rec, sort_keys=True)
        path.write_text("\n".join(lines) + "\n")

        with pytest.raises(AuditLogError):
            log.verify()

    def test_verify_detects_deleted_record(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        log = AuditLog(path)
        log.append("promotion", "alice", {"n": 1})
        log.append("promotion", "bob", {"n": 2})
        log.append("promotion", "carol", {"n": 3})

        lines = path.read_text().splitlines()
        del lines[1]  # remove the middle record
        path.write_text("\n".join(lines) + "\n")

        with pytest.raises(AuditLogError):
            log.verify()


class TestAuditLogSignedExport:
    def test_export_and_verify_roundtrip(self, tmp_path):
        log = AuditLog(tmp_path / "audit.jsonl")
        log.append("promotion", "alice", {"n": 1})
        key = b"shared-operator-secret"

        exported = log.export_signed(key)
        assert AuditLog.verify_export_signature(exported, key) is True

    def test_verify_rejects_wrong_key(self, tmp_path):
        log = AuditLog(tmp_path / "audit.jsonl")
        log.append("promotion", "alice", {"n": 1})

        exported = log.export_signed(b"correct-key")
        assert AuditLog.verify_export_signature(exported, b"wrong-key") is False

    def test_verify_rejects_tampered_export(self, tmp_path):
        log = AuditLog(tmp_path / "audit.jsonl")
        log.append("promotion", "alice", {"n": 1})
        key = b"secret"

        exported = log.export_signed(key)
        data = json.loads(exported)
        data["log"] = data["log"].replace("alice", "mallory")
        tampered = json.dumps(data).encode("utf-8")

        assert AuditLog.verify_export_signature(tampered, key) is False
