"""Cross-tenant isolation for the semantic layer.

THE TEST THAT DID NOT EXIST. `grep -rn "X-User-ID" tests/` returned zero hits before this
file: every server test overrides the identity dependency with a single fixed user, so no
test ever exercised two identities against each other. The one test that *looked* like it
covered this -- `test_records_are_session_scoped_for_multi_tenancy` -- asserts that an
attribute was set and that a count is 1. It never adds a second tenant.

That gap is exactly the shape of the bug it would have caught: `RecordStore.all_records()`
is `list_nodes_by_scope(ALL_SCOPES_VISIBLE)`, a whole-file scan taking no tenant argument,
and `compose_from_store()` calls it. Atlas and Trace share one mmap file and isolate by
`session_id`; following that convention for records would have put every tenant's records
into every tenant's prompt.
"""

import numpy as np
import pytest

from aeon_py.compose import compose_from_store
from aeon_py.records import PRODUCTION_DIM, Provenance, Record, RecordStore, store_path_for

DIM = 8


@pytest.fixture
def vec():
    v = np.zeros(DIM, dtype=np.float32)
    v[0] = 1.0
    return v


class _NoTrace:
    def get_history(self, session_id, limit):
        return []


def _fact(text, session):
    return Record(kind="FACT", text=text, provenance=Provenance(session, (0,)))


def test_a_shared_record_file_leaks_across_tenants(tmp_path, vec):
    """PINS THE REASON per-tenant files are mandatory rather than stylistic.

    This asserts the BROKEN behaviour of the rejected design on purpose. If it ever starts
    failing, `all_records()` gained a tenant filter and the per-tenant-file requirement can
    be revisited -- until then, this is why the boundary is the file."""
    shared = tmp_path / "shared.atlas"
    a = RecordStore(shared, dim=DIM, session_id="tenant-a")
    a.add(_fact("tenant A salary is $400,000", "a"), vec)
    b = RecordStore(shared, dim=DIM, session_id="tenant-b")
    b.add(_fact("tenant B salary is $90,000", "b"), vec)

    leaked = [r.text for r in b.all_records()]
    assert "tenant A salary is $400,000" in leaked          # the leak, deliberately pinned
    assert len(leaked) == 2


def test_per_tenant_files_isolate_records(tmp_path, vec):
    a = RecordStore(store_path_for("tenant-a", tmp_path), dim=DIM, session_id="tenant-a")
    a.add(_fact("tenant A salary is $400,000", "a"), vec)
    b = RecordStore(store_path_for("tenant-b", tmp_path), dim=DIM, session_id="tenant-b")
    b.add(_fact("tenant B salary is $90,000", "b"), vec)

    assert [r.text for r in a.all_records()] == ["tenant A salary is $400,000"]
    assert [r.text for r in b.all_records()] == ["tenant B salary is $90,000"]


def test_one_tenants_prompt_never_contains_anothers_records(tmp_path, vec):
    """The property that actually matters: isolation must hold at the PROMPT, which is what
    reaches the model, not merely at the store API."""
    a = RecordStore(store_path_for("tenant-a", tmp_path), dim=DIM, session_id="tenant-a")
    a.add(_fact("tenant A salary is $400,000", "a"), vec)
    b = RecordStore(store_path_for("tenant-b", tmp_path), dim=DIM, session_id="tenant-b")
    b.add(_fact("tenant B salary is $90,000", "b"), vec)

    prompt_b = compose_from_store(b, _NoTrace(), "Question: what is my salary?")["prompt"]
    assert "$90,000" in prompt_b
    assert "$400,000" not in prompt_b
    assert "tenant A" not in prompt_b


def test_tenant_path_rejects_traversal_and_empty_ids():
    """The tenant string becomes a filename, so it is re-validated here rather than trusted
    from `SessionManager._validate_user_id` -- whose own docstring says filesystem safety is
    no longer why it exists."""
    for bad in ("../../etc/passwd", "a/b", "", "..", "tenant a"):
        with pytest.raises(ValueError):
            store_path_for(bad, "/data/records")
    assert store_path_for("tenant-a", "/data/records").name == "tenant-a.atlas"


def test_production_dim_is_the_encoder_width():
    """A store opened at the wrong width against an existing file corrupts silently rather
    than erroring, so the number lives in one place."""
    assert PRODUCTION_DIM == 768


# --- SessionManager: handle lifecycle ------------------------------------------------

@pytest.fixture
def manager(tmp_path):
    from aeon_py.session import SessionManager
    return SessionManager(atlas_client=None, trace=None, llm_provider=None,
                          max_sessions=2, records_dir=str(tmp_path))


def test_manager_gives_each_tenant_its_own_store(manager):
    a, b = manager.get_store("tenant-a"), manager.get_store("tenant-b")
    assert a.path != b.path
    assert manager.get_store("tenant-a") is a          # memoised, not reopened per call


def test_evicting_a_session_drops_its_store_handle(manager, vec):
    """A stores dict that is not evicted alongside the others is a per-user mmap HANDLE leak
    with no upper bound -- `_evict_oldest` popped only two dicts before this."""
    manager.get_store("tenant-a")
    manager._active_sessions["tenant-a"] = object()
    assert "tenant-a" in manager._stores
    manager._evict_oldest()
    assert "tenant-a" not in manager._stores


def test_store_survives_eviction_on_disk(manager, vec):
    store = manager.get_store("tenant-a")
    store.add(_fact("durable", "a"), np.zeros(PRODUCTION_DIM, dtype=np.float32))
    manager._active_sessions["tenant-a"] = object()
    manager._evict_oldest()
    reopened = manager.get_store("tenant-a")
    assert [r.text for r in reopened.all_records()] == ["durable"]


def test_no_records_dir_means_no_store(tmp_path):
    """A deployment that has not opted into the semantic layer gets None, and `CognitiveLoop`
    then keeps its pre-existing PromptEngine path exactly."""
    from aeon_py.session import SessionManager
    mgr = SessionManager(atlas_client=None, trace=None, llm_provider=None)
    assert mgr.get_store("tenant-a") is None


def test_erasure_cascade_uses_the_subjects_store_not_the_actors(tmp_path, vec):
    """A REAL BUG AVOIDED, pinned so it cannot come back. The erasure endpoint's `user_id` is
    the ADMIN who pressed the button, not the data subject. Passing that admin's store would
    have cascaded against the wrong tenant's records -- deleting the operator's memory and
    leaving the subject's intact. The cascade therefore takes a resolver, tenant -> store."""
    from aeon_py.erasure import cascade_to_derived_records
    subject = RecordStore(store_path_for("subject", tmp_path), dim=DIM, session_id="subject")
    subject.add(_fact("subject PII", "subject"), vec)
    admin = RecordStore(store_path_for("admin", tmp_path), dim=DIM, session_id="admin")
    admin.add(_fact("admin's own note", "admin"), vec)

    stores = {"subject": subject, "admin": admin}
    cascaded, failures = cascade_to_derived_records(stores.get, ["subject"])

    assert len(cascaded) == 1 and failures == []
    assert subject.all_records(include_superseded=True) == []      # erased
    assert [r.text for r in admin.all_records()] == ["admin's own note"]   # untouched


def test_cascade_still_accepts_a_single_store(tmp_path, vec):
    """Backwards-compatible for tests and single-tenant callers."""
    from aeon_py.erasure import cascade_to_derived_records
    store = RecordStore(store_path_for("t", tmp_path), dim=DIM, session_id="t")
    store.add(_fact("x", "s1"), vec)
    cascaded, failures = cascade_to_derived_records(store, ["s1"])
    assert len(cascaded) == 1 and failures == []


def test_cascade_skips_a_tenant_with_no_store(tmp_path):
    from aeon_py.erasure import cascade_to_derived_records
    assert cascade_to_derived_records(lambda _t: None, ["gone"]) == ([], [])
