"""Tests for the temporal property graph overlay (`aeon_py.timeline`).

The resolution rule is FROZEN. Several tests below pin known MISSES deliberately -- a
date-only marker, an ambiguous marker -- so that "improving" the rule against the benchmark
questions fails loudly rather than silently becoming test-set fitting.
"""

import numpy as np
import pytest

from aeon_py.compose import compose
from aeon_py.records import Provenance, Record
from aeon_py.timeline import (
    TPG_SESSION_PREFIX, active, build_chains, edge_session, read_supersession_edges,
    resolve_links, superseded_by, supersedes, write_supersession_edge,
)
from aeon_py.trace import EdgeType


def rec(text, *, supersedes="", date="", bucket="ACQUISITION", subtype="clothing",
        node_id=None, session="s1"):
    return Record(kind="ITEM", text=text, bucket=bucket, subtype=subtype, date=date,
                  provenance=Provenance(session, (0,)), supersedes=supersedes,
                  node_id=node_id)


# --- the frozen resolution rule -----------------------------------------------------

def test_exact_marker_links_to_the_retired_record():
    """The measured `4b24c848` shape."""
    new = rec("five tops from H&M", supersedes="three tops from H&M", date="2023/09/30")
    old = rec("three tops from H&M", date="2023/08/11")
    links = resolve_links([new], [old])
    assert len(links) == 1
    assert links[0].predecessor is old
    assert links[0].confidence == "exact"
    assert links[0].dangling is False


def test_marker_matches_regardless_of_case_and_punctuation():
    new = rec("$400,000 from Wells Fargo", supersedes="$350,000 from Wells Fargo!")
    old = rec("$350,000 from Wells Fargo")
    assert resolve_links([new], [old])[0].confidence == "exact"


def test_paraphrased_marker_links_by_unique_substring():
    new = rec("20 autographed baseballs", supersedes="15 autographed baseballs")
    old = rec("15 autographed baseballs in a display case")
    link = resolve_links([new], [old])[0]
    assert link.predecessor is old and link.confidence == "partial"


def test_ambiguous_marker_is_left_unresolved_not_guessed():
    """Two plausible targets means no link. Guessing here deletes a fact."""
    new = rec("five tops", supersedes="tops")
    a = rec("three tops from H&M")
    b = rec("two tops from Zara")
    link = resolve_links([new], [a, b])[0]
    assert link.predecessor is None and link.confidence == "unresolved"


def test_date_only_marker_does_not_link():
    """PINS A KNOWN MISS -- 2 of the 9 measured markers are bare dates. `canonical_key`
    strips bracketed spans, so a date-only marker has no key to match on. Recorded as a
    limitation rather than special-cased, because special-casing it is rule-tuning."""
    new = rec("updated value", supersedes="[2023/05/11]")
    old = rec("older value", date="2023/05/11")
    assert resolve_links([new], [old])[0].confidence == "unresolved"


def test_marker_whose_target_is_still_live_is_flagged_dangling():
    """1 of 9 measured: the merge asserted a supersession it did not execute. Surfaced, never
    suppressed -- suppressing on this signal is the mechanical rule already rejected."""
    new = rec("five tops", supersedes="three tops")
    still_live = rec("three tops")
    link = resolve_links([new, still_live], [])[0]
    assert link.predecessor is still_live and link.dangling is True


def test_record_without_a_marker_produces_no_link():
    assert resolve_links([rec("plain record")], []) == []


def test_a_record_never_supersedes_itself():
    self_ref = rec("five tops", supersedes="five tops")
    assert resolve_links([self_ref], [])[0].predecessor is None


# --- chains -------------------------------------------------------------------------

def test_chain_orders_versions_oldest_to_newest_by_date():
    new = rec("five tops", supersedes="three tops", date="2023/09/30")
    old = rec("three tops", date="2023/08/11")
    chains, _ = build_chains([new], [old])
    assert [v.text for v in chains[0].versions] == ["three tops", "five tops"]
    assert chains[0].head is new


def test_dangling_links_do_not_become_chains():
    new = rec("five tops", supersedes="three tops")
    live = rec("three tops")
    chains, links = build_chains([new, live], [])
    assert chains == []
    assert links[0].dangling is True          # reported, not silently dropped


def test_unresolved_markers_are_returned_for_reporting():
    new = rec("five tops", supersedes="something never recorded")
    chains, links = build_chains([new], [])
    assert chains == [] and links[0].confidence == "unresolved"


# --- active-state projection --------------------------------------------------------

def test_active_excludes_superseded_records():
    live, retired = rec("five tops"), rec("three tops")
    retired.superseded = True
    assert [r.text for r in active([live, retired])] == ["five tops"]


def test_active_is_the_identity_when_nothing_is_retired():
    recs = [rec("a"), rec("b")]
    assert active(recs) == recs


# --- the render flag ----------------------------------------------------------------

def test_timeline_flag_off_is_byte_identical():
    """NON-NEGOTIABLE. Active state is the default: a factual or counting question must never
    see a historical value, and the default render must not move by one byte."""
    new = rec("five tops from H&M", supersedes="three tops from H&M", date="2023/09/30")
    old = rec("three tops from H&M", date="2023/08/11")
    assert compose([new], [], "Q") == compose([new], [], "Q", timeline=False, retired=[old])


def test_timeline_flag_on_renders_the_chain():
    new = rec("five tops from H&M", supersedes="three tops from H&M", date="2023/09/30")
    old = rec("three tops from H&M", date="2023/08/11")
    out = compose([new], [], "Q", timeline=True, retired=[old])
    assert "How these changed over time:" in out
    assert "three tops from H&M" in out and "(current)" in out


def test_timeline_on_with_no_chains_adds_nothing():
    out_plain = compose([rec("a")], [], "Q")
    assert compose([rec("a")], [], "Q", timeline=True) == out_plain


# --- the durable edge log -----------------------------------------------------------

@pytest.fixture
def trace(tmp_path):
    from aeon_py.trace import TraceGraph
    return TraceGraph(tmp_path / "t.trace")


def test_edge_session_is_tenant_namespaced():
    assert edge_session("acme") == f"{TPG_SESSION_PREFIX}acme"
    assert edge_session("a") != edge_session("b")


def test_supersession_edge_round_trips(trace):
    write_supersession_edge(trace, tenant="acme", survivor_node_id=42,
                            retired_node_id=17, text="five tops from H&M")
    edges = read_supersession_edges(trace, tenant="acme")
    assert len(edges) == 1
    assert int(edges[0]["atlas_id"]) == 42
    assert int(edges[0]["supersedes_id"]) == 17
    assert int(edges[0]["edge_type"]) == int(EdgeType.SUPERSEDES)


def test_reader_ignores_non_edge_events_in_the_edge_session(trace):
    trace.add_event(edge_session("acme"), "system", "not an edge")
    write_supersession_edge(trace, tenant="acme", survivor_node_id=1, retired_node_id=2,
                            text="x")
    assert len(read_supersession_edges(trace, tenant="acme")) == 1


def test_edges_are_isolated_between_tenants(trace):
    write_supersession_edge(trace, tenant="a", survivor_node_id=1, retired_node_id=2, text="x")
    assert read_supersession_edges(trace, tenant="b") == []


def test_what_superseded_this_answers_both_directions(trace):
    write_supersession_edge(trace, tenant="acme", survivor_node_id=42, retired_node_id=17,
                            text="five tops")
    assert superseded_by(trace, 17, tenant="acme") == [42]
    assert supersedes(trace, 42, tenant="acme") == [17]
    assert superseded_by(trace, 999, tenant="acme") == []


# --- compaction-epoch stamping (v4.2) -----------------------------------------------

def _atlas_and_trace(tmp_path):
    import numpy as np
    from aeon_py import core
    from aeon_py.trace import TraceGraph
    atlas = core.Atlas(str(tmp_path / "a.atlas"), dim=8, metadata_size=128)
    return atlas, TraceGraph(path=str(tmp_path / "t.trace"))


def test_an_unstamped_edge_reads_as_unverifiable_not_as_epoch_zero(tmp_path):
    """None and 0 mean different things. An edge written before stamping existed cannot be
    validated at all; calling that "epoch 0" would assert it was current on a fresh file."""
    from aeon_py.timeline import edge_epoch
    assert edge_epoch({"text": "five supersedes three"}) is None
    assert edge_epoch({"text": "five supersedes three @epoch:0"}) == 0


def test_edges_are_stale_once_the_atlas_compacts(tmp_path):
    """The failure this guards is NOT a dangling id -- compact_mmap() reassigns ids, so a
    stale reference resolves cleanly to a DIFFERENT LIVE NODE. Detection is the whole gain."""
    import numpy as np
    from aeon_py.timeline import stale_edges, write_supersession_edge
    atlas, trace = _atlas_and_trace(tmp_path)
    v = (np.ones(8, dtype=np.float32) / np.sqrt(8)).tolist()
    root = atlas.insert(0, v, "ROOT", None)
    doomed = atlas.insert(root, v, "doomed", None)
    survivor = atlas.insert(root, v, "survivor", None)

    write_supersession_edge(trace, tenant="t", survivor_node_id=survivor,
                            retired_node_id=doomed, text="survivor supersedes doomed",
                            compaction_epoch=atlas.compaction_epoch)
    assert stale_edges(trace, atlas, tenant="t") == [], "current edge must not read stale"

    atlas.tombstone_node(doomed)
    atlas.compact_mmap()
    stale = stale_edges(trace, atlas, tenant="t")
    assert len(stale) == 1, "compaction reassigned ids; the edge must now read stale"


def test_an_unstamped_edge_is_always_reported_stale(tmp_path):
    """Unverifiable must fail closed. An edge with no epoch is reported so a caller can
    refuse to act on lineage it cannot trust, rather than assuming it is current."""
    from aeon_py.timeline import stale_edges, write_supersession_edge
    atlas, trace = _atlas_and_trace(tmp_path)
    write_supersession_edge(trace, tenant="t", survivor_node_id=1, retired_node_id=2,
                            text="unstamped")          # no compaction_epoch passed
    assert len(stale_edges(trace, atlas, tenant="t")) == 1
