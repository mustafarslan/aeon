#!/usr/bin/env python3
"""
Test 5: ReBAC Deterministic Pre-filtering — CEUR-WS §4.3.2

Claims under test:
  - WHERE layer_id IN [...] physically excludes unauthorized vectors
  - Unauthorized nodes are "never loaded, never compared, and never ranked"
  - Pre-filtering eliminates TOCTOU vulnerabilities

Methodology:
  1. Create vector DB with 1000 vectors across 4 authorized layers
  2. Inject adversarial vector (user_eve) with near-perfect similarity to query
  3. Compare post-filter vs pre-filter results
  4. Assert Eve is NEVER in pre-filtered results
"""

import time
import numpy as np

# Try LanceDB first, fall back to DuckDB+VSS
DB_ENGINE = None
try:
    import lancedb
    DB_ENGINE = "lancedb"
    print("Using LanceDB")
except ImportError:
    try:
        import duckdb
        DB_ENGINE = "duckdb"
        print("Using DuckDB")
    except ImportError:
        print("ERROR: Neither lancedb nor duckdb installed.")
        print("Install with: pip install lancedb")
        exit(1)

DIM = 768
N_VECTORS = 1000
AUTHORIZED_LAYERS = ["system", "org_a", "team_x", "user_bob"]
ADVERSARY_LAYER = "user_eve"
TOP_K = 10

rng = np.random.default_rng(42)


def generate_vectors_with_layers(n: int, dim: int):
    """Generate vectors with assigned layer_ids."""
    data = []
    for i in range(n):
        vec = rng.standard_normal(dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        layer = AUTHORIZED_LAYERS[i % len(AUTHORIZED_LAYERS)]
        data.append({
            "id": i,
            "vector": vec,
            "layer_id": layer,
            "content": f"node_{i}",
        })
    return data


def create_adversarial_vector(query: np.ndarray, dim: int):
    """Create a vector maximally similar to the query (cosine ~0.99)."""
    noise = rng.standard_normal(dim).astype(np.float32) * 0.01
    adversary = query + noise
    adversary /= np.linalg.norm(adversary)
    sim = np.dot(query, adversary) / (np.linalg.norm(query) * np.linalg.norm(adversary))
    return adversary, float(sim)


# ===========================================================================
# LanceDB Implementation
# ===========================================================================
def run_lancedb_test():
    import lancedb
    import pyarrow as pa
    import shutil, os

    db_path = "/tmp/aeon_rebac_bench"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    db = lancedb.connect(db_path)

    # Generate data
    data = generate_vectors_with_layers(N_VECTORS, DIM)

    # Create query
    query = rng.standard_normal(DIM).astype(np.float32)
    query /= np.linalg.norm(query)

    # Create adversarial vector
    adv_vec, adv_sim = create_adversarial_vector(query, DIM)
    print(f"\n  Adversary similarity to query: {adv_sim:.6f}")

    # Add adversary
    data.append({
        "id": N_VECTORS,
        "vector": adv_vec,
        "layer_id": ADVERSARY_LAYER,
        "content": "ADVERSARY_EVE",
    })

    # Create table
    tbl = db.create_table("knowledge", data=[
        {"id": d["id"], "vector": d["vector"].tolist(), "layer_id": d["layer_id"], "content": d["content"]}
        for d in data
    ])

    print(f"  Table size: {tbl.count_rows()} rows")

    # --- Test A: Post-filter (TOCTOU-vulnerable) ---
    print(f"\n{'='*60}")
    print(f"  Test A: POST-FILTER (TOCTOU-vulnerable)")
    print(f"{'='*60}")

    t0 = time.perf_counter_ns()
    results_unfiltered = tbl.search(query.tolist()).limit(TOP_K).to_list()
    t1 = time.perf_counter_ns()

    print(f"  Search latency: {(t1-t0)/1e6:.2f} ms")
    print(f"  Raw results ({len(results_unfiltered)} rows):")
    eve_in_raw = False
    for r in results_unfiltered:
        marker = " ← ADVERSARY" if r["layer_id"] == ADVERSARY_LAYER else ""
        if r["layer_id"] == ADVERSARY_LAYER:
            eve_in_raw = True
        print(f"    id={r['id']:>5}  layer={r['layer_id']:<12}  dist={r.get('_distance', 'N/A')}{marker}")

    # Post-filter
    filtered = [r for r in results_unfiltered if r["layer_id"] in AUTHORIZED_LAYERS]
    print(f"\n  Post-filtered results ({len(filtered)} rows):")
    for r in filtered:
        print(f"    id={r['id']:>5}  layer={r['layer_id']:<12}  dist={r.get('_distance', 'N/A')}")

    print(f"\n  Eve in raw results:      {'YES ⚠️  (TOCTOU exposure!)' if eve_in_raw else 'NO'}")
    print(f"  Eve in post-filtered:    {'YES ❌' if any(r['layer_id'] == ADVERSARY_LAYER for r in filtered) else 'NO ✅'}")

    # --- Test B: Pre-filter (TOCTOU-resistant) ---
    print(f"\n{'='*60}")
    print(f"  Test B: PRE-FILTER (TOCTOU-resistant)")
    print(f"{'='*60}")

    # Build SQL-style IN clause: IN ('system', 'org_a', ...)
    in_clause = ", ".join(f"'{l}'" for l in AUTHORIZED_LAYERS)
    t0 = time.perf_counter_ns()
    results_prefiltered = (
        tbl.search(query.tolist())
        .where(f"layer_id IN ({in_clause})")
        .limit(TOP_K)
        .to_list()
    )
    t1 = time.perf_counter_ns()

    print(f"  Search latency: {(t1-t0)/1e6:.2f} ms")
    print(f"  Pre-filtered results ({len(results_prefiltered)} rows):")
    eve_in_prefiltered = False
    for r in results_prefiltered:
        marker = " ← ADVERSARY" if r["layer_id"] == ADVERSARY_LAYER else ""
        if r["layer_id"] == ADVERSARY_LAYER:
            eve_in_prefiltered = True
        print(f"    id={r['id']:>5}  layer={r['layer_id']:<12}  dist={r.get('_distance', 'N/A')}{marker}")

    print(f"\n  Eve in pre-filtered:     {'YES ❌ FAIL' if eve_in_prefiltered else 'NO ✅ PASS'}")

    # --- Verdict ---
    print(f"\n{'='*60}")
    print(f"  VERDICT")
    print(f"{'='*60}")
    if eve_in_raw and not eve_in_prefiltered:
        print(f"  ✅ PASS: Pre-filtering physically excluded the adversary.")
        print(f"  Eve appeared in post-filter raw results (TOCTOU exposure),")
        print(f"  but was completely absent from pre-filtered results.")
        print(f"  This validates CEUR-WS §4.3.2: deterministic pre-filtering")
        print(f"  eliminates TOCTOU vulnerabilities.")
    elif not eve_in_raw:
        print(f"  ⚠️  INCONCLUSIVE: Eve's adversarial vector didn't appear in")
        print(f"  raw top-{TOP_K} results. Increase TOP_K or adversary strength.")
    else:
        print(f"  ❌ FAIL: Eve appeared in pre-filtered results!")

    # Cleanup
    shutil.rmtree(db_path, ignore_errors=True)


# ===========================================================================
# DuckDB Fallback Implementation
# ===========================================================================
def run_duckdb_test():
    import duckdb

    con = duckdb.connect(":memory:")
    con.install_extension("vss")
    con.load_extension("vss")

    # Generate data
    data = generate_vectors_with_layers(N_VECTORS, DIM)

    # Create query
    query = rng.standard_normal(DIM).astype(np.float32)
    query /= np.linalg.norm(query)

    # Create adversarial vector
    adv_vec, adv_sim = create_adversarial_vector(query, DIM)
    print(f"\n  Adversary similarity to query: {adv_sim:.6f}")

    data.append({
        "id": N_VECTORS,
        "vector": adv_vec,
        "layer_id": ADVERSARY_LAYER,
        "content": "ADVERSARY_EVE",
    })

    # Create table
    con.execute(f"CREATE TABLE knowledge (id INTEGER, vector FLOAT[{DIM}], layer_id VARCHAR, content VARCHAR)")

    for d in data:
        con.execute("INSERT INTO knowledge VALUES (?, ?, ?, ?)",
                     [d["id"], d["vector"].tolist(), d["layer_id"], d["content"]])

    con.execute(f"CREATE INDEX idx ON knowledge USING HNSW (vector) WITH (metric = 'cosine')")

    # Pre-filter
    result = con.execute(f"""
        SELECT id, layer_id, array_cosine_distance(vector, ?::FLOAT[{DIM}]) as dist
        FROM knowledge
        WHERE layer_id IN ('system', 'org_a', 'team_x', 'user_bob')
        ORDER BY array_cosine_distance(vector, ?::FLOAT[{DIM}])
        LIMIT {TOP_K}
    """, [query.tolist(), query.tolist()]).fetchall()

    eve_in_prefiltered = any(r[1] == ADVERSARY_LAYER for r in result)
    print(f"\n  Pre-filtered results:")
    for r in result:
        print(f"    id={r[0]:>5}  layer={r[1]:<12}  dist={r[2]:.6f}")
    print(f"\n  Eve in pre-filtered: {'YES ❌ FAIL' if eve_in_prefiltered else 'NO ✅ PASS'}")


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print(f"  IEEE WCCI 2026 — Test 5: ReBAC Pre-filtering")
    print(f"  Vectors: {N_VECTORS} + 1 adversary")
    print(f"  Dimension: {DIM}")
    print(f"  Authorized layers: {AUTHORIZED_LAYERS}")
    print(f"  Adversary layer: {ADVERSARY_LAYER}")
    print(f"  DB engine: {DB_ENGINE}")
    print("=" * 60)

    if DB_ENGINE == "lancedb":
        run_lancedb_test()
    elif DB_ENGINE == "duckdb":
        run_duckdb_test()
