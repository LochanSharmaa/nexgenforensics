from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1:1 VERIFICATION DECISION THRESHOLDS
#
# These are cosine-similarity cut-points on the fused embedding. They are
# derived from the 10-fold cross-validated sweep in BENCHMARKS.md, where the
# threshold is fitted on 9 folds and applied to the held-out fold, so they are
# not tuned on the data they are reported against.
#
# The previous values (0.28 / 0.42) were not measured -- they were copied from
# the README. 0.42 in particular sat far above every empirically optimal
# operating point (LFW 0.24-0.28, AgeDB-30 0.20-0.22, CFP-FP 0.18-0.23),
# meaning genuine pairs that the model scored correctly were being reported as
# "inconclusive" or "different_person".
#
# MATCH_THRESHOLD is deliberately set from the hardest clean benchmark
# (AgeDB-30, cross-age) rather than from LFW. LFW is saturated at ~99.8% and
# its optimum is unrepresentative of real casework, where age gap and pose are
# the norm. Tuning on the easy set would produce a threshold that is too high
# for anything harder.
#
# Re-derive after ANY change to the embedding pipeline:
#   python backend/scripts/benchmark_verification.py
# ---------------------------------------------------------------------------
#
# SINGLE SOURCE OF TRUTH: nexgen_engine.config.ThresholdConfig.
# These names are re-exported for backwards compatibility only. Do NOT assign
# literals here -- a third copy of a stale threshold is exactly how this system
# ended up telling users ">=42% = same person" while the engine decided at 0.20.
from ..config import ThresholdConfig as _ThresholdConfig  # noqa: E402

MATCH_THRESHOLD = _ThresholdConfig().match
REVIEW_THRESHOLD = _ThresholdConfig().review

from ..config import EngineConfig
from ..inference import FacialRecognitionPipeline
# NOTE: this module previously imported `VectorSearchIndex`, which does not
# exist in nexgen_engine.search -- the package exports GalleryIndex. That made
# EngineService (and therefore every /biometrics route) fail on import.
from ..search import GalleryIndex
from ..security import AuditLogger
from ..utils import cosine_similarity, stable_id
from .schemas import EngineMatch, EngineSearchResponse


@dataclass(frozen=True)
class VerifyResponse:
    score: float
    label: str          # "same_person" | "inconclusive" | "different_person"
    verified: bool
    quality_ref: float
    quality_probe: float
    liveness_ref: float
    liveness_probe: float
    review_required: bool
    reasons_ref: tuple[str, ...]
    reasons_probe: tuple[str, ...]
    audit_hash: str


class EngineService:
    """Recognition service with durable templates and a queryable audit trail.

    The in-memory VectorSearchIndex is a derived cache. Every enrollment is
    written to SQLite first, and the index is rebuilt from SQLite at startup --
    before this, enrolled identities were lost on every restart, so /identify
    silently searched an empty gallery after any redeploy.
    """

    #: Bumped whenever the embedding pipeline changes in a way that makes old
    #: templates incomparable. Stamped on every stored template and audit row.
    MODEL_VERSION = "ensemble_multi_model/v1"

    def __init__(
        self,
        audit_path: str | Path = "runtime/audit.jsonl",
        config: EngineConfig | None = None,
        store_path: str | Path | None = None,
        tenant_id: str = "default",
    ) -> None:
        self.config = config or EngineConfig()
        self.pipeline = FacialRecognitionPipeline(self.config)
        # EngineConfig exposes embedding_dim; the old `final_embedding_dim`
        # attribute referenced here does not exist on it.
        self.index = GalleryIndex(self.config.embedding_dim)
        self.audit = AuditLogger(audit_path)
        self.tenant_id = tenant_id

        from ..search.persistence import BiometricStore

        resolved = Path(
            store_path
            or os.environ.get("NEXGEN_TEMPLATE_DB")
            or Path(audit_path).parent / "templates.db"
        )
        self.store = BiometricStore(resolved)
        self.restored_count = self._restore_index()

    def _restore_index(self) -> int:
        """Rebuild the in-memory gallery from durable storage at startup."""
        count = 0
        for row in self.store.iter_templates(self.tenant_id):
            self.index.add(
                tenant_id=row.tenant_id,
                template_id=row.template_id,
                subject_id=row.subject_id,
                embedding=row.embedding,
                metadata=row.metadata,
            )
            count += 1
        if count:
            logger.info("restored %d enrolled templates from %s", count, self.store.path)
        return count

    def _record(
        self,
        audit_hash: str,
        operation: str,
        operator_id: str,
        decision: str,
        score: float | None = None,
        subject_id: str | None = None,
        detail: dict | None = None,
    ) -> None:
        """Mirror an audit entry into the queryable durable log.

        The JSONL logger keeps the tamper-evident hash chain; this makes the
        same event retrievable by hash so a hash handed to a caller can
        actually be checked against something.
        """
        try:
            self.store.write_audit(
                audit_hash=audit_hash,
                operation=operation,
                operator_id=operator_id,
                tenant_id=self.tenant_id,
                decision=decision,
                model_version=self.MODEL_VERSION,
                score=score,
                subject_id=subject_id,
                detail=detail or {},
            )
        except Exception:
            # An audit write must never take down the request that produced it,
            # but it must be visible in the logs rather than swallowed.
            logger.exception("failed to persist audit row %s", audit_hash)

    def enroll(self, image_bytes: bytes, identity_id: str, metadata: dict[str, str] | None = None) -> EngineSearchResponse:
        result = self.pipeline.encode_bytes(image_bytes)
        template_id = stable_id("tpl", image_bytes + identity_id.encode())
        self.index.add(
            tenant_id=self.tenant_id,
            template_id=template_id,
            subject_id=identity_id,
            embedding=result.embedding,
            metadata=metadata or {},
        )
        entry = self.audit.append("system", "enroll", "enrolled", 1.0, {"identity_id": identity_id})
        source_sha256 = hashlib.sha256(image_bytes).hexdigest()
        self.store.put_template(
            tenant_id=self.tenant_id,
            template_id=template_id,
            subject_id=identity_id,
            embedding=result.embedding,
            metadata=metadata or {},
            source_sha256=source_sha256,
            model_version=self.MODEL_VERSION,
        )
        self._record(
            entry.entry_hash, "enroll", "system", "enrolled",
            score=1.0, subject_id=identity_id,
            detail={"source_sha256": source_sha256},
        )
        return EngineSearchResponse(
            decision="enrolled",
            quality_score=result.quality.score,
            liveness_score=result.liveness.score,
            review_required=result.review_required,
            reasons=list(result.reasons),
            matches=[],
            audit_hash=entry.entry_hash,
        )

    def identify(self, image_bytes: bytes, operator_id: str = "demo_operator", top_k: int = 5) -> EngineSearchResponse:
        result = self.pipeline.encode_bytes(image_bytes)
        outcome = self.index.search(self.tenant_id, result.embedding, top_k=top_k)
        matches = list(outcome.matches)
        confidence = matches[0].score if matches else 0.0
        decision = "review_required" if result.review_required or confidence < self.config.search.min_match_score else "candidate_match_ready"
        entry = self.audit.append(
            operator_id,
            "identify",
            decision,
            confidence,
            {"probe_id": stable_id("probe", image_bytes), "match_count": len(matches)},
        )
        self._record(
            entry.entry_hash, "identify", operator_id, decision,
            score=confidence,
            subject_id=matches[0].subject_id if matches else None,
            detail={
                "match_count": len(matches),
                "top_k": top_k,
                "probe_sha256": hashlib.sha256(image_bytes).hexdigest(),
                "gallery_size": outcome.gallery_size,
            },
        )
        return EngineSearchResponse(
            decision=decision,
            quality_score=result.quality.score,
            liveness_score=result.liveness.score,
            review_required=decision == "review_required",
            reasons=list(result.reasons),
            matches=[EngineMatch(identity_id=item.subject_id, confidence=item.score, metadata=item.metadata) for item in matches],
            audit_hash=entry.entry_hash,
        )

    def verify(self, ref_bytes: bytes, probe_bytes: bytes, operator_id: str = "demo_operator") -> VerifyResponse:
        """1:1 face comparison — returns cosine similarity with label and audit entry."""
        ref = self.pipeline.encode_bytes(ref_bytes)
        probe = self.pipeline.encode_bytes(probe_bytes)
        score = float(cosine_similarity(ref.embedding, probe.embedding))
        # Empirically calibrated -- see the MATCH_THRESHOLD comment above.
        if score >= MATCH_THRESHOLD:
            label = "same_person"
            verified = True
        elif score >= REVIEW_THRESHOLD:
            label = "inconclusive"
            verified = False
        else:
            label = "different_person"
            verified = False
        review = ref.review_required or probe.review_required or label == "inconclusive"
        entry = self.audit.append(
            operator_id,
            "verify",
            label,
            score,
            {
                "ref_id": stable_id("ref", ref_bytes),
                "probe_id": stable_id("probe", probe_bytes),
            },
        )
        self._record(
            entry.entry_hash, "verify", operator_id, label,
            score=score,
            detail={
                "ref_sha256": hashlib.sha256(ref_bytes).hexdigest(),
                "probe_sha256": hashlib.sha256(probe_bytes).hexdigest(),
                "match_threshold": MATCH_THRESHOLD,
                "review_threshold": REVIEW_THRESHOLD,
            },
        )
        return VerifyResponse(
            score=round(score, 6),
            label=label,
            verified=verified,
            quality_ref=ref.quality.score,
            quality_probe=probe.quality.score,
            liveness_ref=ref.liveness.score,
            liveness_probe=probe.liveness.score,
            review_required=review,
            reasons_ref=ref.reasons,
            reasons_probe=probe.reasons,
            audit_hash=entry.entry_hash,
        )
