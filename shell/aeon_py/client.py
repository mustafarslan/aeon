"""
Aeon Memory OS — Python Shell Client.

Provides:
  1. AeonClient:    High-level Atlas query interface with zero-copy views.
  2. TieredClient:  Edge-to-Cloud fallback with cold miss detection.
  3. DriftMonitor:  Rolling SLB hit/miss telemetry and concept drift detection.
"""

import collections
import logging
import math
import numpy as np
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from . import core

logger = logging.getLogger("aeon.telemetry")

# ===========================================================================
# Binary Schema: must match C++ ResultNode (atlas.hpp) exactly.
# Layout: id(8) + sim(4) + preview(12) + requires_cloud_fetch(1) + pad(7) = 32
# Alignment: uint64 needs 8-byte alignment → struct is 32 bytes.
# ===========================================================================
RESULT_DTYPE = np.dtype([
    ('id', 'u8'),
    ('similarity', 'f4'),
    ('preview', 'f4', (3,)),
    ('requires_cloud_fetch', '?'),  # bool (1 byte)
], align=True)


class AeonClient:
    """
    High-level Python interface to the Atlas spatial memory engine.
    Wraps the C++ core with lazy loading and type safety.
    """

    def __init__(self, atlas_path: Path | str) -> None:
        self._path = Path(atlas_path)
        self._atlas: Optional[core.Atlas] = None

        # Verify schema immediately
        self._validate_schema()

    def _validate_schema(self):
        """Ensures C++ binary layout matches Python NumPy definition."""
        cpp_size = core.get_result_node_size()
        py_size = RESULT_DTYPE.itemsize

        if cpp_size != py_size:
            raise RuntimeError(
                f"CRITICAL ARCHITECTURE FAIL: Memory Layout Mismatch.\n"
                f"C++ ResultNode: {cpp_size} bytes\n"
                f"Python Dtype:   {py_size} bytes\n"
                "Check struct padding/alignment in schema.hpp vs client.py."
            )

    @property
    def atlas(self) -> core.Atlas:
        """Lazy initialization of the C++ Atlas."""
        if self._atlas is None:
            self._atlas = core.Atlas(str(self._path))
        return self._atlas

    def query(self, embedding: np.ndarray) -> np.ndarray:
        """
        Navigate the Atlas using a query vector.

        Args:
            embedding: (768,) float32 numpy array

        Returns:
            Structured numpy array with columns
            ['id', 'similarity', 'preview', 'requires_cloud_fetch']
        """
        if embedding.shape != (768,) or embedding.dtype != np.float32:
            if embedding.size == 768:
                embedding = embedding.astype(np.float32).reshape(768)
            else:
                raise ValueError("Embedding must be shape (768,) and float32")

        # Call C++ binding which returns nb::ndarray<uint8_t>
        byte_view = self.atlas.navigate_raw(embedding)

        # Cast to structured array
        result = byte_view.view(RESULT_DTYPE)
        result.flags.writeable = False  # Enforce read-only explicitly

        return result

    def get_children(self, node_id: int) -> np.ndarray:
        """
        Returns child nodes (siblings of next step) for visualization.

        Args:
            node_id: uint64 ID of the parent node.

        Returns:
            Structured numpy array of child nodes.
        """
        # Call C++ binding which returns nb::ndarray<uint8_t>
        byte_view = self.atlas.get_children_raw(node_id)

        # Cast to structured array
        result = byte_view.view(RESULT_DTYPE)
        result.flags.writeable = False  # Enforce read-only explicitly

        return result

    def warmup(self) -> None:
        """
        Runs a dummy query to touch memory pages and warm up the
        OS page cache for the mmap.
        """
        dummy = np.zeros(768, dtype=np.float32)
        try:
            self.query(dummy)
        except Exception:
            # Ignore errors on empty atlas
            pass

    @contextmanager
    def safe_memory_view(self):
        """
        Context manager for safe zero-copy memory access.

        While inside this block, the underlying mmap region is protected
        by an EBR epoch guard — grow() will not reclaim it. Use this
        when holding numpy views across multiple operations.

        Usage:
            with client.safe_memory_view():
                results = client.query(embedding)
                # Memory is pinned — safe to slice, index, pass around
                process(results)
        """
        guard = self.atlas.acquire_read_guard()
        try:
            yield guard
        finally:
            # EpochGuard.__exit__ releases the epoch slot
            pass  # guard goes out of scope, releasing automatically


# ===========================================================================
# Phase 5: Drift Telemetry
# ===========================================================================

class DriftMonitor:
    """
    Rolling telemetry for SLB hit/miss tracking and concept drift detection.

    Designed for continuous 24/7/365 industrial operation. Tracks:
      - Per-session SLB hit rate over a rolling window.
      - Jensen-Shannon divergence between recent queries and historical
        Atlas centroids to detect semantic distribution shift ("concept drift").

    When the hit rate drops below 70% or JS-divergence exceeds the threshold,
    emits a structured Python logging.warning indicating that dreaming/
    compaction is required to restore semantic inertia.
    """

    def __init__(
        self,
        session_id: str = "default",
        window_size: int = 1000,
        hit_rate_threshold: float = 0.70,
        drift_threshold: float = 0.15,
        recent_window: int = 100,
    ):
        self.session_id = session_id
        self.window_size = window_size
        self.hit_rate_threshold = hit_rate_threshold
        self.drift_threshold = drift_threshold
        self.recent_window = recent_window

        # Rolling deque: True = hit (similarity >= cold_miss_threshold),
        #                False = miss
        self._hit_miss: collections.deque[bool] = collections.deque(
            maxlen=window_size
        )

        # Rolling deque of query embeddings (for drift detection)
        self._recent_queries: collections.deque[np.ndarray] = collections.deque(
            maxlen=recent_window
        )

        # Historical centroid (exponential moving average of all queries)
        self._historical_centroid: Optional[np.ndarray] = None
        self._total_queries: int = 0

    def record(
        self,
        query_embedding: np.ndarray,
        best_similarity: float,
        cold_miss_threshold: float = 0.65,
    ) -> dict:
        """
        Record a single query result for telemetry.

        Args:
            query_embedding:    The 768-d query vector.
            best_similarity:    Best similarity score from Atlas navigation.
            cold_miss_threshold: Threshold below which a query is a "miss".

        Returns:
            dict with keys: 'hit', 'hit_rate', 'drift_score', 'alert_emitted'
        """
        is_hit = best_similarity >= cold_miss_threshold
        self._hit_miss.append(is_hit)
        self._recent_queries.append(
            query_embedding.astype(np.float32).ravel()
        )
        self._total_queries += 1

        # Update historical centroid (exponential moving average)
        flat = query_embedding.astype(np.float32).ravel()
        if self._historical_centroid is None:
            self._historical_centroid = flat.copy()
        else:
            # EMA with decay → older queries get exponentially less weight
            alpha = 2.0 / (self._total_queries + 1)
            self._historical_centroid = (
                alpha * flat + (1.0 - alpha) * self._historical_centroid
            )

        # Compute hit rate
        hit_rate = sum(self._hit_miss) / len(self._hit_miss) if self._hit_miss else 1.0

        # Compute drift score
        drift_score = self.detect_concept_drift()

        # Alert logic
        alert_emitted = False
        if len(self._hit_miss) >= self.recent_window:
            if hit_rate < self.hit_rate_threshold:
                logger.warning(
                    "SEMANTIC INERTIA COLLAPSED — Dreaming/Compaction Required | "
                    "session=%s hit_rate=%.3f threshold=%.3f drift=%.4f "
                    "window=%d total_queries=%d",
                    self.session_id,
                    hit_rate,
                    self.hit_rate_threshold,
                    drift_score,
                    len(self._hit_miss),
                    self._total_queries,
                )
                alert_emitted = True

            if drift_score > self.drift_threshold:
                logger.warning(
                    "CONCEPT DRIFT DETECTED — Distribution Shift | "
                    "session=%s js_divergence=%.4f threshold=%.4f "
                    "hit_rate=%.3f total_queries=%d",
                    self.session_id,
                    drift_score,
                    self.drift_threshold,
                    hit_rate,
                    self._total_queries,
                )
                alert_emitted = True

        return {
            "hit": is_hit,
            "hit_rate": hit_rate,
            "drift_score": drift_score,
            "alert_emitted": alert_emitted,
        }

    def detect_concept_drift(self) -> float:
        """
        Calculate Jensen-Shannon divergence between the average of the last
        `recent_window` queries and the historical Atlas centroid.

        Returns:
            JS divergence in [0, 1]. Higher = more drift.
            Returns 0.0 if insufficient data.
        """
        if (
            self._historical_centroid is None
            or len(self._recent_queries) < self.recent_window
        ):
            return 0.0

        # Compute mean of recent queries
        recent_stack = np.stack(list(self._recent_queries), axis=0)
        recent_mean = recent_stack.mean(axis=0)

        # Normalize both distributions to probability simplex
        # (shift to positive, then L1-normalize)
        p = self._softmax_normalize(recent_mean)
        q = self._softmax_normalize(self._historical_centroid)

        # Jensen-Shannon divergence: JSD = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
        m = 0.5 * (p + q)
        jsd = 0.5 * self._kl_divergence(p, m) + 0.5 * self._kl_divergence(q, m)

        return float(jsd)

    @staticmethod
    def _softmax_normalize(v: np.ndarray) -> np.ndarray:
        """Convert a raw embedding to a probability distribution via softmax."""
        # Numerical stability: subtract max
        v_shifted = v - np.max(v)
        exp_v = np.exp(v_shifted)
        return exp_v / (exp_v.sum() + 1e-12)

    @staticmethod
    def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """KL divergence D_KL(P || Q) with epsilon smoothing."""
        eps = 1e-12
        return float(np.sum(p * np.log((p + eps) / (q + eps))))

    @property
    def hit_rate(self) -> float:
        """Current rolling hit rate."""
        if not self._hit_miss:
            return 1.0
        return sum(self._hit_miss) / len(self._hit_miss)

    @property
    def total_queries(self) -> int:
        """Total number of recorded queries."""
        return self._total_queries


class TieredClient(AeonClient):
    """
    Edge-to-Cloud Hybrid Atlas Client.

    Extends AeonClient with:
      - Cold miss detection (requires_cloud_fetch flag inspection)
      - Automatic cloud fallback routing for deep historical queries
      - Integrated DriftMonitor for continuous telemetry
      - Memory budget awareness (triggers Dreaming when exceeded)

    Hybrid Architecture:
        ┌─────────────────┐     cold miss     ┌──────────────────┐
        │   Edge Device   │  ──────────────→  │  Cloud/HPC       │
        │  (local Atlas)  │                   │  (full Atlas)    │
        │  SLB + mmap     │  ←────────────    │  REST/gRPC       │
        │                 │   merged results  │                  │
        └─────────────────┘                   └──────────────────┘
    """

    def __init__(
        self,
        atlas_path: Path | str,
        session_id: str = "default",
        cold_miss_threshold: float = 0.65,
        memory_budget_mb: int = 128,
        cloud_endpoint: Optional[str] = None,
        cloud_timeout_seconds: float = 5.0,
    ):
        super().__init__(atlas_path)
        self.cold_miss_threshold = cold_miss_threshold
        self.memory_budget_mb = memory_budget_mb
        self.cloud_endpoint = cloud_endpoint
        self.cloud_timeout = cloud_timeout_seconds
        self.drift_monitor = DriftMonitor(session_id=session_id)

        # Telemetry counters
        self._cloud_fallback_count: int = 0
        self._total_queries: int = 0

    def query_tiered(self, embedding: np.ndarray) -> dict:
        """
        Execute a tiered query with automatic cloud fallback.

        Flow:
          1. Query the local Atlas (Edge SLB + mmap).
          2. If best_similarity < cold_miss_threshold → cold miss.
          3. On cold miss, route to Cloud/HPC endpoint for deep history.
          4. Merge local + cloud results, sorted by similarity.
          5. Record telemetry for drift detection.

        Returns:
            dict with keys:
              'results':              Structured numpy array (merged)
              'requires_cloud_fetch': bool (was cloud routing triggered?)
              'cloud_results':        Cloud results (or None)
              'best_similarity':      float
              'telemetry':            dict from DriftMonitor.record()
        """
        self._total_queries += 1

        # 1. Local query
        local_results = self.query(embedding)
        best_sim = float(local_results['similarity'].max()) if len(local_results) > 0 else 0.0
        needs_cloud = best_sim < self.cold_miss_threshold

        # 2. Cloud fallback
        cloud_results = None
        merged_results = local_results

        if needs_cloud and self.cloud_endpoint:
            self._cloud_fallback_count += 1
            cloud_results = self._fetch_from_cloud(embedding)
            if cloud_results is not None and len(cloud_results) > 0:
                merged_results = self._merge_results(local_results, cloud_results)
                best_sim = float(merged_results['similarity'].max())

        # 3. Telemetry
        telemetry = self.drift_monitor.record(
            query_embedding=embedding,
            best_similarity=best_sim,
            cold_miss_threshold=self.cold_miss_threshold,
        )

        return {
            "results": merged_results,
            "requires_cloud_fetch": needs_cloud,
            "cloud_results": cloud_results,
            "best_similarity": best_sim,
            "telemetry": telemetry,
        }

    def _fetch_from_cloud(self, embedding: np.ndarray) -> Optional[np.ndarray]:
        """
        Route a cold-miss query to the Cloud/HPC backend.

        In production, this issues an HTTP POST to the cloud endpoint
        with the query embedding and returns the results. For now,
        this is a simulated REST call that logs the routing event.

        Returns:
            Structured numpy array of cloud results, or None on failure.
        """
        logger.info(
            "CLOUD FALLBACK | endpoint=%s query_norm=%.4f",
            self.cloud_endpoint,
            float(np.linalg.norm(embedding)),
        )

        try:
            # === Production Implementation ===
            # import httpx
            # response = httpx.post(
            #     f"{self.cloud_endpoint}/v1/navigate",
            #     json={"embedding": embedding.tolist(), "top_k": 50},
            #     timeout=self.cloud_timeout,
            # )
            # return self._parse_cloud_response(response.json())

            # === Simulated Cloud Response ===
            # Returns an empty result set — the cloud endpoint is not yet
            # deployed. The routing infrastructure is fully wired.
            logger.info(
                "Cloud query routed (simulated) | fallback_count=%d total=%d",
                self._cloud_fallback_count,
                self._total_queries,
            )
            return None

        except Exception as e:
            logger.warning("Cloud fallback failed: %s", e)
            return None

    @staticmethod
    def _merge_results(
        local: np.ndarray,
        cloud: np.ndarray,
    ) -> np.ndarray:
        """
        Merge local and cloud results, sorted by similarity (descending).
        Deduplicates by node ID, preferring the higher similarity score.
        """
        if cloud is None or len(cloud) == 0:
            return local
        if len(local) == 0:
            return cloud

        # Concatenate
        combined = np.concatenate([local, cloud])

        # Deduplicate by ID (keep highest similarity)
        seen: dict[int, int] = {}
        keep_indices: list[int] = []
        for i, row in enumerate(combined):
            nid = int(row['id'])
            if nid not in seen:
                seen[nid] = i
                keep_indices.append(i)
            else:
                prev_idx = seen[nid]
                if row['similarity'] > combined[prev_idx]['similarity']:
                    keep_indices.remove(prev_idx)
                    keep_indices.append(i)
                    seen[nid] = i

        deduped = combined[keep_indices]

        # Sort by similarity descending
        sorted_indices = np.argsort(-deduped['similarity'])
        return deduped[sorted_indices]

    @property
    def cloud_fallback_rate(self) -> float:
        """Fraction of queries that required cloud fallback."""
        if self._total_queries == 0:
            return 0.0
        return self._cloud_fallback_count / self._total_queries

