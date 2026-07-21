from __future__ import annotations

from pathlib import Path

from ..config import EngineConfig
from ..inference import FacialRecognitionPipeline
from ..search import VectorSearchIndex
from ..security import AuditLogger
from ..utils import stable_id
from .schemas import EngineMatch, EngineSearchResponse


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
