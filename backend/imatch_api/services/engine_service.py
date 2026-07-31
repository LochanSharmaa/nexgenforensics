from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
from sqlmodel import Session, func, select

from nexgen_engine.config import EngineConfig
from nexgen_engine.inference.pipeline import FacialRecognitionPipeline, RecognitionResult
from nexgen_engine.inference.score_fusion import Decision, DecisionEngine, ScoreNormalizer
from nexgen_engine.runtime import EngineRuntime
from nexgen_engine.search.gallery_index import GalleryIndex, SearchOutcome
from nexgen_engine.security.template_encryption import (
    EncryptedTemplate,
    TemplateDecryptionError,
    TemplateEncryptor,
)

from ..core.config import Settings, get_settings
from ..db.models import Subject, Template

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GalleryStats:
    tenant_id: str
    templates: int
    subjects: int
    loaded: bool


class EngineService:
    """Single owner of the loaded models, the gallery index, and the crypto key.

    Built once per process. The gallery lives in memory for search speed and is
    rebuilt from the database on first use per tenant, so a restart loses no
    enrolments and the database stays the source of truth.

    Templates on disk are always encrypted. They are decrypted only here, only
    into the in-memory index, and never returned through the API: a raw template
    is as sensitive as the face it came from and can be used to reconstruct an
    approximation of it.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        config: EngineConfig | None = None,
        runtime: EngineRuntime | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.config = config or self.settings.engine_config()
        # An existing runtime can be injected so several services share one set
        # of loaded models. Loading the pack costs seconds and hundreds of MB,
        # so the test suite builds it once and reuses it across cases.
        self.runtime = runtime or EngineRuntime(self.config)
        self.pipeline = FacialRecognitionPipeline(self.config, self.runtime)
        self.decisions = DecisionEngine(self.config)
        self.index = GalleryIndex(self.config.embedding_dim)
        self.encryptor = TemplateEncryptor.from_base64(self.settings.resolved_template_key())

        self._loaded_tenants: set[str] = set()
        self._lock = threading.RLock()

    # ------------------------------------------------------------- status ---

    @property
    def recognition_capable(self) -> bool:
        return self.runtime.recognition_capable

    def status(self) -> dict[str, Any]:
        return self.runtime.status()

    def warm_up(self) -> None:
        """Force model loading now rather than on the first user request."""
        _ = self.runtime.recognizer

    # ------------------------------------------------------------ gallery ---

    def ensure_loaded(self, session: Session, tenant_id: str) -> None:
        """Populate the in-memory index for a tenant, once."""
        with self._lock:
            if tenant_id in self._loaded_tenants:
                return

            templates = session.exec(
                select(Template).where(Template.tenant_id == tenant_id)
            ).all()

            rows: list[tuple[str, str, np.ndarray, dict[str, Any]]] = []
            failed = 0
            for template in templates:
                try:
                    vector = self.encryptor.decrypt(
                        EncryptedTemplate(
                            nonce=template.nonce,
                            ciphertext=template.ciphertext,
                            dimensions=template.dimensions,
                            tenant_id=template.tenant_id,
                        )
                    )
                except TemplateDecryptionError:
                    # Almost always a changed NEXGEN_TEMPLATE_KEY. Skip the row
                    # rather than crashing the service, and make the count loud.
                    failed += 1
                    continue
                rows.append(
                    (
                        template.id,
                        template.subject_id,
                        vector,
                        {
                            "quality": template.quality_score,
                            "recognizer_pack": template.recognizer_pack,
                            "created_at": template.created_at.isoformat(),
                        },
                    )
                )

            if rows:
                self.index.add_many(tenant_id, rows)
            self._loaded_tenants.add(tenant_id)

            if failed:
                logger.error(
                    "Tenant %s: %d of %d templates could not be decrypted and were excluded from "
                    "the gallery. This usually means NEXGEN_TEMPLATE_KEY changed.",
                    tenant_id,
                    failed,
                    len(templates),
                )
            logger.info("Tenant %s gallery loaded: %d templates.", tenant_id, len(rows))

    def invalidate(self, tenant_id: str) -> None:
        with self._lock:
            self.index.clear(tenant_id)
            self._loaded_tenants.discard(tenant_id)

    def stats(self, session: Session, tenant_id: str) -> GalleryStats:
        self.ensure_loaded(session, tenant_id)
        subject_count = session.exec(
            select(func.count()).select_from(Subject).where(
                Subject.tenant_id == tenant_id, Subject.active == True  # noqa: E712
            )
        ).one()
        return GalleryStats(
            tenant_id=tenant_id,
            templates=self.index.size(tenant_id),
            subjects=int(subject_count),
            loaded=True,
        )

    # ---------------------------------------------------------- enrolment ---

    def seal(self, embedding: np.ndarray, tenant_id: str) -> EncryptedTemplate:
        return self.encryptor.encrypt(embedding, tenant_id)

    def register(
        self,
        tenant_id: str,
        template_id: str,
        subject_id: str,
        embedding: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a freshly enrolled template to the live index."""
        self.index.add(tenant_id, template_id, subject_id, embedding, metadata or {})

    def unregister(self, tenant_id: str, template_id: str) -> bool:
        return self.index.remove(tenant_id, template_id)

    def unregister_subject(self, tenant_id: str, subject_id: str) -> int:
        return self.index.remove_subject(tenant_id, subject_id)

    # ------------------------------------------------------------- search ---

    def encode(self, image_bytes: bytes) -> RecognitionResult:
        """Encode one image, recording its per-stage latency.

        This is the single choke point every biometric operation passes
        through -- search, verify, batch and enrolment all call it -- so
        instrumenting here captures the whole service without touching each
        route. See nexgen_engine/observability.py.
        """
        from nexgen_engine.observability import LATENCY  # noqa: PLC0415

        try:
            result = self.pipeline.encode_bytes(image_bytes)
        except Exception:
            # Rejections are load too: a flood of malformed uploads shows up
            # as an error rate rather than vanishing from the metrics.
            LATENCY.record_error()
            raise
        try:
            LATENCY.record(result.timings.as_dict())
        except Exception:  # pragma: no cover - observability must never break a request
            pass
        return result

    def search(
        self,
        session: Session,
        tenant_id: str,
        result: RecognitionResult,
        top_k: int = 10,
    ) -> tuple[SearchOutcome, Decision, np.ndarray]:
        """Run a probe against one tenant's gallery and decide what it means.

        Returns the ranked outcome, the decision, and the per-candidate
        normalized scores so the UI can show how far each lead stands out from
        the impostor distribution.
        """
        self.ensure_loaded(session, tenant_id)

        outcome = self.index.search(
            tenant_id,
            result.embedding,
            top_k=top_k,
            min_score=self.config.search.min_candidate_score,
        )

        normalized = ScoreNormalizer.z_scores(
            np.asarray([match.score for match in outcome.matches], dtype=np.float64)
        ) if outcome.matches else np.empty(0)

        decision = self.decisions.decide(
            top_score=outcome.top_score,
            recognition_capable=self.recognition_capable,
            probe_accepted=result.quality.accepted,
            probe_reasons=result.reasons,
            gallery_size=outcome.gallery_size,
            margin=outcome.margin,
            candidate_count=len(outcome.matches),
        )
        return outcome, decision, normalized

    def compare(self, reference: np.ndarray, probe: np.ndarray) -> float:
        """Cosine similarity between two templates, both already normalized."""
        return float(np.dot(reference, probe))


_service: EngineService | None = None
_service_lock = threading.Lock()


def get_engine_service() -> EngineService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = EngineService()
    return _service


def set_engine_service(service: EngineService | None) -> None:
    """Override the process-wide service. Used by tests."""
    global _service
    _service = service


__all__ = ["EngineService", "GalleryStats", "get_engine_service", "set_engine_service"]
