"""Vision provider contract.

A vision model's job here is narrow and it matters that it stays narrow: report
**what is visible in the image**. Not who someone is, not where the photograph
was taken, not what it means. Those are conclusions, and conclusions require
corroborating evidence from sources — which is the rest of the pipeline's job.

The separation the platform depends on:

* **Observation** — "the sign reads MERIDIAN LOGISTICS". A statement about the
  image, sourced to the image, verifiable by looking at it.
* **Investigation** — that string becomes a search clue.
* **Verification** — a fact only exists once pages corroborate it, with
  citations.

Everything a vision model produces enters as the first kind. It is evidence of
what the image *shows*, never of what is *true* — a sign can be a mock-up, a
document can be forged, and a landmark can be a replica.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from shared.enums import VisionCategory


@dataclass(frozen=True)
class VisionObservation:
    """One thing the model reports seeing."""

    category: VisionCategory
    value: str
    """Verbatim where possible — transcribed text, a logo's wordmark, a
    document title. Paraphrase loses the evidential value."""
    detail: str = ""
    """Where in the image, or what it is attached to. Context for a reviewer
    checking the observation against the picture."""
    confidence: float = 0.0
    """The model's own stated confidence. Recorded, never trusted: it feeds the
    OCR-tier weighting, not a fact's confidence directly."""
    verbatim: bool = True
    """False when the model summarised rather than transcribed. A paraphrased
    sign is weaker evidence than a quoted one, and the difference is preserved."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "category": str(self.category),
            "value": self.value,
            "detail": self.detail,
            "confidence": self.confidence,
            "verbatim": self.verbatim,
        }


@dataclass(frozen=True)
class SearchClue:
    """A query the observations justify.

    Derived from what was seen, never invented. Each clue records which
    observation produced it, so a search that leads somewhere can be traced back
    to the pixels that prompted it.
    """

    query: str
    rationale: str
    source_category: VisionCategory
    priority: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "rationale": self.rationale,
            "source_category": str(self.source_category),
            "priority": self.priority,
        }


@dataclass(frozen=True)
class VisionAnalysis:
    """Everything one vision pass produced."""

    provider: str
    model: str
    available: bool
    observations: tuple[VisionObservation, ...] = field(default=())
    clues: tuple[SearchClue, ...] = field(default=())
    people_present: int = 0
    """A count only. That people are visible is an observation; who they are is
    not something this system will ever assert from an image."""
    rejected: tuple[dict[str, str], ...] = field(default=())
    """Model output the guardrails refused, kept so the refusal is auditable
    rather than silent."""
    error: str = ""
    duration_ms: int = 0
    raw_response: dict[str, Any] | None = None

    def by_category(self, category: VisionCategory) -> tuple[VisionObservation, ...]:
        return tuple(o for o in self.observations if o.category == category)

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "available": self.available,
            "observations": [o.as_dict() for o in self.observations],
            "clues": [c.as_dict() for c in self.clues],
            "people_present": self.people_present,
            "rejected": [dict(r) for r in self.rejected],
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


@runtime_checkable
class VisionProvider(Protocol):
    """Implemented by every vision backend."""

    name: str
    model: str

    def available(self) -> bool: ...

    async def analyse(self, image: bytes, mime_type: str) -> VisionAnalysis: ...


__all__ = [
    "SearchClue",
    "VisionAnalysis",
    "VisionObservation",
    "VisionProvider",
]
