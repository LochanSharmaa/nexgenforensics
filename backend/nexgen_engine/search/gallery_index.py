from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..utils import l2_normalize

try:  # pragma: no cover - depends on host packages
    import faiss as _faiss  # noqa: F401

    _FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FAISS_AVAILABLE = False


def faiss_available() -> bool:
    """Whether exhaustive search runs through FAISS or numpy.

    Both are exact, so this affects speed only. Exposed so the status endpoint
    can report which path is live.
    """
    return _FAISS_AVAILABLE


@dataclass(frozen=True)
class MatchResult:
    """One candidate returned by a search."""

    template_id: str
    subject_id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "subject_id": self.subject_id,
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class SearchOutcome:
    """Ranked candidates plus the statistics needed to judge them.

    ``all_scores`` is the full score vector over the searched tenant's gallery.
    The decision layer needs it to compute the runner-up margin and the impostor
    distribution; without it a top score cannot be told apart from a near-tie.
    """

    matches: tuple[MatchResult, ...]
    all_scores: np.ndarray
    gallery_size: int

    @property
    def top_score(self) -> float:
        return float(self.matches[0].score) if self.matches else 0.0

    @property
    def margin(self) -> float:
        """Gap between best and second-best *subject* (not template)."""
        if len(self.matches) < 2:
            return 0.0
        return float(self.matches[0].score - self.matches[1].score)


class _TenantShard:
    """All enrolled templates for a single tenant."""

    __slots__ = (
        "dimensions",
        "template_ids",
        "subject_ids",
        "metadata",
        "vectors",
        "_positions",
        "_index",
        "_index_rows",
    )

    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions
        self.template_ids: list[str] = []
        self.subject_ids: list[str] = []
        self.metadata: list[dict[str, Any]] = []
        self.vectors = np.empty((0, dimensions), dtype=np.float32)
        self._positions: dict[str, int] = {}
        # Rebuilt lazily whenever the row count changes.
        self._index = None
        self._index_rows = -1

    def add(self, template_id: str, subject_id: str, vector: np.ndarray, metadata: dict[str, Any]) -> None:
        self._invalidate()
        if template_id in self._positions:
            self.remove(template_id)
        self._positions[template_id] = len(self.template_ids)
        self.template_ids.append(template_id)
        self.subject_ids.append(subject_id)
        self.metadata.append(metadata)
        self.vectors = np.vstack([self.vectors, vector.reshape(1, -1)]) if self.vectors.size else vector.reshape(1, -1)

    def add_many(self, rows: list[tuple[str, str, np.ndarray, dict[str, Any]]]) -> None:
        """Bulk insert. Rebuilding a gallery row by row is O(n^2) in copies."""
        if not rows:
            return
        self._invalidate()
        stacked = np.vstack([vector.reshape(1, -1) for _, _, vector, _ in rows])
        start = len(self.template_ids)
        for offset, (template_id, subject_id, _, metadata) in enumerate(rows):
            self._positions[template_id] = start + offset
            self.template_ids.append(template_id)
            self.subject_ids.append(subject_id)
            self.metadata.append(metadata)
        self.vectors = np.vstack([self.vectors, stacked]) if self.vectors.size else stacked

    def scores(self, query: np.ndarray) -> np.ndarray:
        """Similarity of ``query`` against every template in this shard.

        Uses FAISS ``IndexFlatIP`` when available and falls back to a numpy
        matmul otherwise. Both compute an exhaustive inner product over
        L2-normalized vectors, so the two paths are numerically equivalent and
        return the same ranking -- FAISS is a speed optimisation, never a
        different answer. That matters here: an approximate index would silently
        drop true candidates, and a missed lead is invisible to the examiner.
        """
        if self.vectors.size == 0:
            return np.empty(0, dtype=np.float32)

        index = self._faiss_index()
        if index is not None:
            total = len(self.template_ids)
            distances, ids = index.search(np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32), total)
            # FAISS returns results ranked by score. Scatter them back into
            # gallery order so callers can index by position, exactly as the
            # numpy path does.
            ordered = np.zeros(total, dtype=np.float32)
            valid = ids[0] >= 0
            ordered[ids[0][valid]] = distances[0][valid]
            return ordered

        return (self.vectors @ query).astype(np.float32)

    def _faiss_index(self):  # noqa: ANN202 - faiss types are optional
        """Build (and cache) a FAISS flat index for this shard.

        DELIBERATE CHOICE, MEASURED -- read before "optimising" this.

        faiss is NOT installed and NOT declared in any requirements file, so
        this path is dormant and every search runs the numpy matmul in
        ``scores()``. That is intentional, not an oversight.

        Measured on an RTX A3000 host (BENCHMARKS.md 7b), brute-force cosine:

            1,000 templates    0.207 ms p50
           10,000 templates    1.087 ms p50
          100,000 templates   15.981 ms p50

        Below ~10k the search is under 8% of the ~14.7 ms it costs to encode
        the probe image, i.e. free. Adding a dependency to optimise 8% of the
        request would be premature.

        Note what this branch would and would not buy: ``IndexFlatIP`` is an
        EXACT inner-product index -- brute force with better SIMD. It is not an
        approximate-nearest-neighbour structure. Enabling faiss here would win a
        constant factor, NOT a change in complexity; the 100k cost would still
        grow linearly. Genuine scaling past ~100k needs an approximate index
        (IVF-PQ or HNSW) and the recall loss that comes with it, which must be
        measured and accepted explicitly rather than assumed.
        """
        if not _FAISS_AVAILABLE:
            return None
        if self._index is not None and self._index_rows == len(self.template_ids):
            return self._index
        import faiss

        index = faiss.IndexFlatIP(self.dimensions)
        index.add(np.ascontiguousarray(self.vectors, dtype=np.float32))
        self._index = index
        self._index_rows = len(self.template_ids)
        return index

    def _invalidate(self) -> None:
        self._index = None
        self._index_rows = -1

    def remove(self, template_id: str) -> bool:
        position = self._positions.pop(template_id, None)
        if position is None:
            return False
        self._invalidate()
        self.template_ids.pop(position)
        self.subject_ids.pop(position)
        self.metadata.pop(position)
        self.vectors = np.delete(self.vectors, position, axis=0)
        # Positions after the removed row all shift down by one.
        for key, value in self._positions.items():
            if value > position:
                self._positions[key] = value - 1
        return True

    def __len__(self) -> int:
        return len(self.template_ids)


class GalleryIndex:
    """In-memory vector gallery, partitioned by tenant.

    Tenant isolation is structural rather than advisory: each tenant's vectors
    live in a separate matrix, and ``search`` takes the tenant id as a required
    argument, so there is no code path that can compare a probe against another
    tenant's templates. A filter applied after a global search would be one
    forgotten predicate away from a cross-tenant biometric leak.

    Cosine similarity on L2-normalized templates reduces to a dot product, so a
    brute-force matmul is exact. That is fast enough well past 10^5 templates per
    tenant on CPU; beyond that, swap in an ANN backend behind this same
    interface and accept the recall/latency trade-off explicitly.

    Every method is guarded by a lock because the API server is multi-threaded
    and numpy array replacement is not atomic.
    """

    def __init__(self, dimensions: int = 512) -> None:
        self.dimensions = dimensions
        self._shards: dict[str, _TenantShard] = {}
        self._lock = threading.RLock()

    # ----------------------------------------------------------- mutation ---

    def add(
        self,
        tenant_id: str,
        template_id: str,
        subject_id: str,
        embedding: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        vector = self._prepare(embedding)
        with self._lock:
            self._shard(tenant_id).add(template_id, subject_id, vector, metadata or {})

    def add_many(
        self,
        tenant_id: str,
        rows: list[tuple[str, str, np.ndarray, dict[str, Any]]],
    ) -> None:
        prepared = [(t_id, s_id, self._prepare(vec), meta) for t_id, s_id, vec, meta in rows]
        with self._lock:
            self._shard(tenant_id).add_many(prepared)

    def remove(self, tenant_id: str, template_id: str) -> bool:
        with self._lock:
            shard = self._shards.get(tenant_id)
            return shard.remove(template_id) if shard else False

    def remove_subject(self, tenant_id: str, subject_id: str) -> int:
        """Delete every template belonging to one subject. Returns the count."""
        with self._lock:
            shard = self._shards.get(tenant_id)
            if shard is None:
                return 0
            doomed = [
                template_id
                for template_id, owner in zip(shard.template_ids, shard.subject_ids)
                if owner == subject_id
            ]
            for template_id in doomed:
                shard.remove(template_id)
            return len(doomed)

    def clear(self, tenant_id: str | None = None) -> None:
        with self._lock:
            if tenant_id is None:
                self._shards.clear()
            else:
                self._shards.pop(tenant_id, None)

    # -------------------------------------------------------------- query ---

    def size(self, tenant_id: str) -> int:
        with self._lock:
            shard = self._shards.get(tenant_id)
            return len(shard) if shard else 0

    def subject_count(self, tenant_id: str) -> int:
        with self._lock:
            shard = self._shards.get(tenant_id)
            return len(set(shard.subject_ids)) if shard else 0

    def tenants(self) -> list[str]:
        with self._lock:
            return sorted(self._shards)

    def search(
        self,
        tenant_id: str,
        embedding: np.ndarray,
        top_k: int = 20,
        min_score: float = -1.0,
        collapse_subjects: bool = True,
    ) -> SearchOutcome:
        """Rank the tenant's gallery against a probe.

        With ``collapse_subjects`` (the default) a subject enrolled from several
        photographs appears once, at its best-scoring template. Otherwise a
        well-enrolled subject would fill the entire candidate list and hide every
        other lead from the examiner.
        """
        query = self._prepare(embedding)
        with self._lock:
            shard = self._shards.get(tenant_id)
            if shard is None or len(shard) == 0:
                return SearchOutcome(matches=(), all_scores=np.empty(0, dtype=np.float32), gallery_size=0)

            scores = shard.scores(query)
            template_ids = list(shard.template_ids)
            subject_ids = list(shard.subject_ids)
            metadata = list(shard.metadata)
            gallery_size = len(shard)

        order = np.argsort(scores)[::-1]
        matches: list[MatchResult] = []
        seen_subjects: set[str] = set()

        for index in order:
            score = float(scores[index])
            if score < min_score:
                break
            subject_id = subject_ids[index]
            if collapse_subjects:
                if subject_id in seen_subjects:
                    continue
                seen_subjects.add(subject_id)
            matches.append(
                MatchResult(
                    template_id=template_ids[index],
                    subject_id=subject_id,
                    score=round(score, 6),
                    metadata=metadata[index],
                )
            )
            if len(matches) >= top_k:
                break

        return SearchOutcome(matches=tuple(matches), all_scores=scores, gallery_size=gallery_size)

    # ----------------------------------------------------------- internal ---

    def _shard(self, tenant_id: str) -> _TenantShard:
        shard = self._shards.get(tenant_id)
        if shard is None:
            shard = _TenantShard(self.dimensions)
            self._shards[tenant_id] = shard
        return shard

    def _prepare(self, embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.dimensions:
            raise ValueError(f"Expected a {self.dimensions}-d template, got {vector.shape[0]}.")
        if not np.all(np.isfinite(vector)):
            raise ValueError("Template contains NaN or infinite values.")
        return l2_normalize(vector)


__all__ = ["GalleryIndex", "MatchResult", "SearchOutcome"]
