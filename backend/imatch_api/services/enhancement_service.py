"""Enhancement as a service: analyse, run, store separately, record.

THE ONE ARCHITECTURAL RULE THIS FILE ENFORCES

``EngineService.encode()`` is described in its own docstring as "the single
choke point every biometric operation passes through -- search, verify, batch
and enrolment all call it". That makes it the tempting place to hook
enhancement in, and the wrong one:

  * it would silently enhance ENROLMENT images, which is never wanted -- an
    enrolment photograph is the reference, and processing it moves the
    reference rather than the probe;
  * it would make enhancement implicit everywhere, so no caller could opt out
    and no reader of the code could tell which images had been processed.

So ``encode()`` is left exactly as it is. Enhancement happens here, explicitly,
and callers that want an A/B comparison call ``encode()`` twice -- once with the
original and once with the enhanced bytes. The original's result stays primary.

Storage is separate too. The original lives under the ``probes`` category where
it always did; the enhanced image is written under ``enhanced`` as a distinct
content-addressed object. Neither can overwrite the other, because both are
named by the SHA-256 of their own bytes.
"""

from __future__ import annotations

import json
import logging
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image
from sqlmodel import Session

from nexgen_engine.enhancement import (
    EnhancementCache,
    EnhancementOutcome,
    OriginalImage,
    analyze,
    available_backends,
    execute,
    quality_metrics,
)
from nexgen_engine.enhancement.planner import plan as build_plan
from nexgen_engine.enhancement.vram import device_report
from nexgen_engine.enhancement.weights import catalogue
from nexgen_engine.inference.pipeline import InvalidImageError, decode_image

from ..core.config import Settings, get_settings
from ..db.models import EnhancementRun
from .storage_service import StorageService

logger = logging.getLogger(__name__)


def _weight_specs() -> dict[str, Any]:
    """Declared checkpoints, gathered from whichever backends actually loaded."""
    specs: dict[str, Any] = {}
    from nexgen_engine.enhancement.registry import _REGISTRY  # noqa: PLC0415

    for name, cls in _REGISTRY.items():
        spec = getattr(cls, "weight_spec", None)
        if spec is not None:
            specs[name] = spec
    return specs


class EnhancementDisabledError(RuntimeError):
    """Raised when enhancement is requested while the feature flag is off."""


class EnhancementService:
    """Owns the cache and the policy. Holds no models -- the runner loads and
    releases them per stage, so this object is cheap and safe to construct."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.cache = EnhancementCache(
            self.settings.enhancement_cache_path,
            enabled=self.settings.enhancement_enabled,
        )

    # ------------------------------------------------------------ status --

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.settings.enhancement_enabled,
            "reconstruction_enabled": self.settings.enhancement_reconstruction_enabled,
            "device": device_report(),
            "requested_device": self.settings.enhancement_device,
            "backends": available_backends(),
            "weights": catalogue(_weight_specs()),
            "cache": self.cache.stats(),
        }

    # ---------------------------------------------------------- analysis --

    @staticmethod
    def decode(image_bytes: bytes) -> np.ndarray:
        """Bytes to uint8 RGB, honouring EXIF orientation.

        Reuses the recognition pipeline's decoder so that an image which the
        recogniser would refuse is refused here too, with the same message. Two
        decoders with different tolerances would let an image be enhanced and
        then rejected, which reads as an enhancement failure.
        """
        return np.asarray(decode_image(image_bytes), dtype=np.uint8)

    def analyze_bytes(self, image_bytes: bytes) -> dict[str, Any]:
        """Measure without producing pixels. Safe to run on anything, changes nothing."""
        pixels = self.decode(image_bytes)
        profile = analyze(pixels)
        preview = build_plan(
            profile,
            allow_reconstruction=self.settings.enhancement_reconstruction_enabled,
        )
        return {
            "profile": profile.as_dict(),
            "metrics": quality_metrics(pixels),
            "recommended_plan": preview.as_dict(),
            "notes": list(profile.notes),
        }

    # ----------------------------------------------------------- execute --

    def enhance_bytes(
        self,
        image_bytes: bytes,
        *,
        allow_reconstruction: bool | None = None,
        overrides: dict[str, str] | None = None,
        disabled: set[str] | None = None,
    ) -> EnhancementOutcome:
        if not self.settings.enhancement_enabled:
            raise EnhancementDisabledError(
                "Image enhancement is disabled on this deployment. Set NEXGEN_ENHANCEMENT_ENABLED=true."
            )

        # Policy is an AND, never an OR: a caller may decline reconstruction but
        # may not grant it to itself when the deployment has it off.
        permitted = self.settings.enhancement_reconstruction_enabled
        wants = permitted if allow_reconstruction is None else bool(allow_reconstruction)
        effective_reconstruction = bool(permitted and wants)

        pixels = self.decode(image_bytes)
        original = OriginalImage.of(pixels, source_sha256="")
        profile = analyze(original.pixels)
        plan = build_plan(
            profile,
            allow_reconstruction=effective_reconstruction,
            overrides=overrides,
            disabled=disabled,
        )
        return execute(
            original,
            plan,
            device=self.settings.enhancement_device,
            cache=self.cache,
        )

    # ------------------------------------------------------------- store --

    @staticmethod
    def encode_png(pixels: np.ndarray) -> bytes:
        """PNG, never JPEG.

        Writing the enhanced image as JPEG would put a fresh compression
        operator on top of the one the pipeline just removed, and would make the
        stored artefact differ from the one that was measured.
        """
        buffer = BytesIO()
        Image.fromarray(np.ascontiguousarray(pixels), mode="RGB").save(buffer, format="PNG", optimize=False)
        return buffer.getvalue()

    def persist(
        self,
        session: Session,
        *,
        tenant_id: str,
        operator_id: str,
        outcome: EnhancementOutcome,
        storage: StorageService,
        original_bytes: bytes,
        case_id: str | None = None,
        lawful_basis: str = "",
    ) -> EnhancementRun:
        """Write both images and the run record. The original is stored unchanged."""
        stored_original = storage.store(tenant_id, original_bytes, category="probes")
        enhanced_bytes = self.encode_png(outcome.output.pixels)
        stored_enhanced = storage.store(tenant_id, enhanced_bytes, category="enhanced")

        run = EnhancementRun(
            tenant_id=tenant_id,
            case_id=case_id,
            operator_id=operator_id,
            original_sha256=stored_original.sha256,
            original_path=stored_original.path,
            enhanced_sha256=stored_enhanced.sha256,
            enhanced_path=stored_enhanced.path,
            track="reconstructed" if outcome.track.value == "reconstruction" else "restored",
            ruleset_version=outcome.plan.ruleset_version,
            plan_cache_key=outcome.plan.cache_key(),
            plan_json=json.dumps(outcome.plan.as_dict(), default=str),
            stages_json=json.dumps([result.as_dict() for result in outcome.results], default=str),
            metrics_before_json=json.dumps(outcome.metrics_before, default=str),
            metrics_after_json=json.dumps(outcome.metrics_after, default=str),
            warnings_json=json.dumps(list(outcome.warnings), default=str),
            device=outcome.device,
            total_ms=int(outcome.total_ms),
            served_from_cache=outcome.cached,
            lawful_basis=lawful_basis,
        )
        session.add(run)
        session.flush()
        return run

    # -------------------------------------------------------------- read --

    @staticmethod
    def hydrate(run: EnhancementRun) -> dict[str, Any]:
        """The stored run as the API returns it, with JSON columns decoded."""

        def load(payload: str, fallback: Any) -> Any:
            try:
                return json.loads(payload)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                return fallback

        return {
            "enhancement_id": run.id,
            "case_id": run.case_id,
            "track": run.track,
            "label": (
                "AI-enhanced preview - not evidentiary, for visual reference only"
                if run.track == "reconstructed"
                else "Processed image - deterministic operations only, no synthesised detail"
            ),
            "original_sha256": run.original_sha256,
            "enhanced_sha256": run.enhanced_sha256,
            "ruleset_version": run.ruleset_version,
            "plan": load(run.plan_json, {}),
            "stages": load(run.stages_json, []),
            "metrics_before": load(run.metrics_before_json, {}),
            "metrics_after": load(run.metrics_after_json, {}),
            "warnings": load(run.warnings_json, []),
            "device": run.device,
            "total_ms": run.total_ms,
            "served_from_cache": run.served_from_cache,
            "created_at": run.created_at.isoformat(),
            "audit_hash": run.audit_hash,
        }


_service: EnhancementService | None = None


def get_enhancement_service() -> EnhancementService:
    global _service
    if _service is None:
        _service = EnhancementService()
    return _service


def set_enhancement_service(service: EnhancementService | None) -> None:
    """Override the process-wide service. Used by tests."""
    global _service
    _service = service


__all__ = [
    "EnhancementDisabledError",
    "EnhancementService",
    "InvalidImageError",
    "get_enhancement_service",
    "set_enhancement_service",
]
