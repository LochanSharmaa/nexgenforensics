from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import EngineConfig
from ..inference import FacialRecognitionPipeline
from ..search import VectorSearchIndex
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
    def __init__(self, audit_path: str | Path = "runtime/audit.jsonl", config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self.pipeline = FacialRecognitionPipeline(self.config)
        self.index = VectorSearchIndex(self.config.final_embedding_dim)
        self.audit = AuditLogger(audit_path)

    def enroll(self, image_bytes: bytes, identity_id: str, metadata: dict[str, str] | None = None) -> EngineSearchResponse:
        result = self.pipeline.encode_bytes(image_bytes)
        self.index.add(identity_id, result.embedding, metadata or {})
        entry = self.audit.append("system", "enroll", "enrolled", 1.0, {"identity_id": identity_id})
        return EngineSearchResponse(
            decision="enrolled",
            quality_score=result.quality.score,
            liveness_score=result.liveness_score,
            review_required=result.review_required,
            reasons=list(result.reasons),
            matches=[],
            audit_hash=entry.entry_hash,
        )

    def identify(self, image_bytes: bytes, operator_id: str = "demo_operator", top_k: int = 5) -> EngineSearchResponse:
        result = self.pipeline.encode_bytes(image_bytes)
        matches = self.index.search(result.embedding, top_k)
        confidence = matches[0].score if matches else 0.0
        decision = "review_required" if result.review_required or confidence < self.config.search.min_match_score else "candidate_match_ready"
        entry = self.audit.append(
            operator_id,
            "identify",
            decision,
            confidence,
            {"probe_id": stable_id("probe", image_bytes), "match_count": len(matches)},
        )
        return EngineSearchResponse(
            decision=decision,
            quality_score=result.quality.score,
            liveness_score=result.liveness_score,
            review_required=decision == "review_required",
            reasons=list(result.reasons),
            matches=[EngineMatch(identity_id=item.identity_id, confidence=item.score, metadata=item.metadata) for item in matches],
            audit_hash=entry.entry_hash,
        )

    def verify(self, ref_bytes: bytes, probe_bytes: bytes, operator_id: str = "demo_operator") -> VerifyResponse:
        """1:1 face comparison — returns cosine similarity with label and audit entry."""
        ref = self.pipeline.encode_bytes(ref_bytes)
        probe = self.pipeline.encode_bytes(probe_bytes)
        score = float(cosine_similarity(ref.embedding, probe.embedding))
        # Thresholds from README (0.28 / 0.36 / 0.42)
        if score >= 0.42:
            label = "same_person"
            verified = True
        elif score >= 0.28:
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
        return VerifyResponse(
            score=round(score, 6),
            label=label,
            verified=verified,
            quality_ref=ref.quality.score,
            quality_probe=probe.quality.score,
            liveness_ref=ref.liveness_score,
            liveness_probe=probe.liveness_score,
            review_required=review,
            reasons_ref=ref.reasons,
            reasons_probe=probe.reasons,
            audit_hash=entry.entry_hash,
        )
