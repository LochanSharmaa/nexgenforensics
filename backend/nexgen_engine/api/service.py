from __future__ import annotations

import json
from pathlib import Path

from ..config import EngineConfig
from ..inference import FacialRecognitionPipeline
from ..search import OptionalFaissIndex
from ..security import AuditLogger
from ..security.template_encryption import TemplateEncryptor
from ..storage import Database
from ..utils import stable_id
from .schemas import EngineMatch, EngineSearchResponse


class EngineService:
    def __init__(
        self,
        audit_path: str | Path = "runtime/audit.jsonl",
        config: EngineConfig | None = None,
        index_path: str | Path | None = None,
        database: Database | None = None,
        template_secret: str = "local-template-secret",
    ) -> None:
        self.config = config or EngineConfig()
        self.pipeline = FacialRecognitionPipeline(self.config)
        self.index = OptionalFaissIndex(self.config.final_embedding_dim, index_path=index_path)
        self.audit = AuditLogger(audit_path)
        self.database = database
        self.template_secret = template_secret
        self.encryptor = TemplateEncryptor(self.config.security.pbkdf2_iterations)

    def enroll(self, image_bytes: bytes, identity_id: str, metadata: dict[str, str] | None = None) -> EngineSearchResponse:
        result = self.pipeline.encode_bytes(image_bytes)
        template_metadata = metadata or {}
        self.index.add(identity_id, result.embedding, template_metadata)
        if self.database is not None:
            tenant_id = template_metadata.get("tenant_id") or template_metadata.get("workspace") or "default"
            encrypted = self.encryptor.encrypt(result.embedding, self.template_secret)
            self.database.upsert_tenant(tenant_id, tenant_id)
            self.database.store_enrolled_identity(identity_id, tenant_id, json.dumps(template_metadata, sort_keys=True))
            self.database.store_template(
                stable_id("template", identity_id.encode("utf-8")),
                identity_id,
                tenant_id,
                encrypted.dimensions,
                json.dumps(encrypted.__dict__, sort_keys=True),
            )
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
