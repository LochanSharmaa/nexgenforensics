"""The type separation that makes the evidential rule unforgeable.

A comment saying "do not put an enhanced image in the report" is a convention,
and conventions are forgotten by the next person to touch the file. A type that
the evidential path structurally cannot accept is not.

The canonical pixel interchange for this whole package is **uint8 RGB, HWC**.
Every backend receives that and returns that. Float internals are a backend's
own business and must not leak, because a float array that silently carries a
different value range is the classic way to produce a black or blown output that
nobody notices until it reaches a report.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class Track(str, Enum):
    """Which side of the evidence/prior line a backend sits on.

    Declaring this is mandatory in the registry. A backend that does not choose
    cannot be registered, which is what stops the separation eroding as models
    are added later.
    """

    MEASUREMENT = "measurement"
    RECONSTRUCTION = "reconstruction"


class Task(str, Enum):
    """What a backend is for. One backend, one task."""

    DEINTERLACE = "deinterlace"
    DEBLOCK = "deblock"
    DENOISE = "denoise"
    TONE = "tone"
    DEBLUR = "deblur"
    UPSCALE = "upscale"
    FACE_RESTORE = "face_restore"
    FUSION = "fusion"


def digest_array(image: np.ndarray) -> str:
    """Content address for an image, including its geometry.

    Shape and dtype are hashed alongside the bytes. Without that, a 2x2 and a
    1x4 array holding the same four values would collide, and the cache would
    hand back the wrong picture.
    """
    h = hashlib.sha256()
    h.update(str(image.shape).encode())
    h.update(str(image.dtype).encode())
    h.update(np.ascontiguousarray(image).tobytes())
    return h.hexdigest()


def canonical_digest(obj: Any) -> str:
    """Digest of a structure, stable across runs and Python versions."""
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _validate(pixels: np.ndarray) -> np.ndarray:
    if pixels.dtype != np.uint8:
        raise TypeError(
            f"Enhancement images must be uint8, got {pixels.dtype}. Convert at the "
            "backend boundary -- a float array leaking out of a backend is how an "
            "image ends up silently black."
        )
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError(f"Enhancement images must be HxWx3 RGB, got shape {pixels.shape}.")
    return np.ascontiguousarray(pixels)


@dataclass(frozen=True)
class OriginalImage:
    """Unmodified evidence. The only type the evidential path accepts.

    ``digest`` is the content address of the pixels as decoded. ``source_sha256``
    is the hash of the file bytes as received, which is what chain of custody is
    written against -- the two differ because decoding is not the identity
    (EXIF rotation, colour conversion), and both are recorded.
    """

    pixels: np.ndarray = field(repr=False)
    digest: str
    source_sha256: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def of(cls, pixels: np.ndarray, source_sha256: str = "", **metadata: Any) -> OriginalImage:
        validated = _validate(pixels)
        return cls(
            pixels=validated,
            digest=digest_array(validated),
            source_sha256=source_sha256,
            metadata=dict(metadata),
        )


@dataclass(frozen=True)
class RestoredImage:
    """Track A output. Contains no detail that was not in the original.

    Produced only by deterministic, classical operations whose effect can be
    described without reference to a learned prior. Safe to describe to a court
    as a processed image; still not the evidence.
    """

    pixels: np.ndarray = field(repr=False)
    digest: str
    parent_digest: str
    stages: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    track: Track = Track.MEASUREMENT

    @classmethod
    def of(
        cls,
        pixels: np.ndarray,
        parent_digest: str,
        stages: tuple[str, ...] = (),
        **metadata: Any,
    ) -> RestoredImage:
        validated = _validate(pixels)
        return cls(
            pixels=validated,
            digest=digest_array(validated),
            parent_digest=parent_digest,
            stages=stages,
            metadata=dict(metadata),
        )


@dataclass(frozen=True)
class ReconstructedImage:
    """Track B output. Contains synthesised detail from a learned prior.

    Displaying this without its original beside it, and without the label, is a
    breach of the project constraint recorded in report_pdf.draw_enhanced_pair.
    The renderer therefore takes this type together with its parent, so the
    pairing cannot be omitted by accident.
    """

    pixels: np.ndarray = field(repr=False)
    digest: str
    parent_digest: str
    stages: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    track: Track = Track.RECONSTRUCTION

    @classmethod
    def of(
        cls,
        pixels: np.ndarray,
        parent_digest: str,
        stages: tuple[str, ...] = (),
        **metadata: Any,
    ) -> ReconstructedImage:
        validated = _validate(pixels)
        return cls(
            pixels=validated,
            digest=digest_array(validated),
            parent_digest=parent_digest,
            stages=stages,
            metadata=dict(metadata),
        )


EnhancedImage = RestoredImage | ReconstructedImage
AnyImage = OriginalImage | RestoredImage | ReconstructedImage


class NotEvidenceError(TypeError):
    """Raised when a processed image is offered where evidence is required."""


def assert_evidential(image: AnyImage) -> OriginalImage:
    """Gate for the likelihood/LR path. Accepts the original type and nothing else.

    This is deliberately a hard failure rather than a warning. An enhanced image
    reaching a likelihood computation produces a number that looks ordinary and
    is not, which is the single worst outcome this package could cause.
    """
    if isinstance(image, OriginalImage):
        return image
    kind = type(image).__name__
    raise NotEvidenceError(
        f"{kind} cannot enter the evidential path. Enhancement output is an analysis "
        "intermediate: it may be shown to an investigator beside the original, and it "
        "may be searched against the gallery for comparison, but it may not be used to "
        "compute a likelihood ratio or be recorded as evidence. Pass the OriginalImage."
    )


@dataclass(frozen=True)
class StageResult:
    """One executed stage: what ran, with what, how long, and what it changed.

    ``changed`` is not decoration. A restorer that silently returns its input --
    the usual symptom of a vendor wrapper failing to detect a face in a
    pre-cropped surveillance crop -- looks exactly like "enhancement had no
    effect" unless something checks. This field is that check, and the runner
    escalates it to a warning on the outcome.
    """

    name: str
    task: Task
    track: Track
    parameters: dict[str, Any]
    rationale: str
    duration_ms: float
    input_digest: str
    output_digest: str
    changed: bool
    mean_abs_delta: float
    vram_peak_mb: float = 0.0
    device: str = "cpu"
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "task": self.task.value,
            "track": self.track.value,
            "parameters": self.parameters,
            "rationale": self.rationale,
            "duration_ms": round(self.duration_ms, 2),
            "input_digest": self.input_digest,
            "output_digest": self.output_digest,
            "changed": self.changed,
            "mean_abs_delta": round(self.mean_abs_delta, 4),
            "vram_peak_mb": round(self.vram_peak_mb, 1),
            "device": self.device,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class PlannedStage:
    """A stage the planner chose, or chose against.

    Skipped stages carry their reason and are reported. "Deblur skipped: blur
    confidence R2 0.31, below the 0.50 trust floor" tells an examiner something
    a silent omission does not.
    """

    name: str
    task: Task
    track: Track
    parameters: dict[str, Any]
    rationale: str
    selected: bool
    skip_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "task": self.task.value,
            "track": self.track.value,
            "parameters": self.parameters,
            "rationale": self.rationale,
            "selected": self.selected,
            "skip_reason": self.skip_reason,
        }


@dataclass(frozen=True)
class EnhancementPlan:
    """The ordered decision, made before any pixel is touched.

    The plan is hashed into the cache key, so changing a parameter invalidates
    the cached result rather than silently reusing output from a different
    pipeline.
    """

    stages: tuple[PlannedStage, ...]
    ruleset_version: str
    profile: dict[str, Any] = field(default_factory=dict)

    @property
    def selected(self) -> tuple[PlannedStage, ...]:
        return tuple(stage for stage in self.stages if stage.selected)

    @property
    def skipped(self) -> tuple[PlannedStage, ...]:
        return tuple(stage for stage in self.stages if not stage.selected)

    @property
    def crosses_into_reconstruction(self) -> bool:
        return any(stage.track is Track.RECONSTRUCTION for stage in self.selected)

    def cache_key(self) -> str:
        """Canonical digest of the executable content of this plan.

        Only the selected stages and their parameters participate. The rationale
        strings and the measured profile do not: two runs that execute the same
        operations should hit the same cache entry even if a confidence value
        moved in the fourth decimal place.
        """
        return canonical_digest(
            {
                "ruleset_version": self.ruleset_version,
                "stages": [
                    {"name": s.name, "task": s.task.value, "parameters": s.parameters}
                    for s in self.selected
                ],
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "ruleset_version": self.ruleset_version,
            "cache_key": self.cache_key(),
            "profile": self.profile,
            "stages": [stage.as_dict() for stage in self.stages],
            "crosses_into_reconstruction": self.crosses_into_reconstruction,
        }


@dataclass(frozen=True)
class EnhancementOutcome:
    """Everything one enhancement produced, ready to persist or return."""

    original: OriginalImage = field(repr=False)
    output: EnhancedImage = field(repr=False)
    plan: EnhancementPlan
    results: tuple[StageResult, ...]
    metrics_before: dict[str, Any]
    metrics_after: dict[str, Any]
    total_ms: float
    device: str
    warnings: tuple[str, ...] = ()
    cached: bool = False

    @property
    def track(self) -> Track:
        return self.output.track

    def as_dict(self) -> dict[str, Any]:
        return {
            "original_digest": self.original.digest,
            "original_sha256": self.original.source_sha256,
            "output_digest": self.output.digest,
            "track": self.track.value,
            "label": (
                "AI-enhanced preview - not evidentiary, for visual reference only"
                if self.track is Track.RECONSTRUCTION
                else "Processed image - deterministic operations only, no synthesised detail"
            ),
            "plan": self.plan.as_dict(),
            "stages": [result.as_dict() for result in self.results],
            "metrics_before": self.metrics_before,
            "metrics_after": self.metrics_after,
            "total_ms": round(self.total_ms, 2),
            "device": self.device,
            "warnings": list(self.warnings),
            "cached": self.cached,
        }


__all__ = [
    "AnyImage",
    "EnhancedImage",
    "EnhancementOutcome",
    "EnhancementPlan",
    "NotEvidenceError",
    "OriginalImage",
    "PlannedStage",
    "ReconstructedImage",
    "RestoredImage",
    "StageResult",
    "Task",
    "Track",
    "assert_evidential",
    "canonical_digest",
    "digest_array",
]
