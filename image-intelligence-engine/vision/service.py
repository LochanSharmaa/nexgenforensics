"""Turn vision output into evidence.

The boundary this module defends: a vision model produces **observations**, and
observations are sourced to the image. They are not facts. A fact requires
corroboration from pages, and that happens later in the pipeline.

So every finding written here has `method=VISION` and `image_id` set — meaning
"this is what the image shows, and you can check by looking". Nothing here
writes to `facts`, and nothing here writes an entity. Extraction proposes;
correlation and review dispose.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from database.models import Image
from database.repositories import ObservationRepository
from shared.clock import Clock, SystemClock
from shared.config import Settings
from shared.enums import ObservationMethod, VisionCategory
from shared.logging import get_logger

from .base import VisionAnalysis, VisionProvider
from .guardrails import clue_is_safe

logger = get_logger(__name__)

EXTRACTOR_VERSION = "vision@0.1.0"

# Categories worth turning into search clues on their own. A transcribed company
# name or document reference is searchable; "a blue car" is not, and running it
# would waste a paid provider call on noise.
_SEARCHABLE = frozenset(
    {
        VisionCategory.TEXT,
        VisionCategory.SIGN,
        VisionCategory.LOGO,
        VisionCategory.DOCUMENT,
        VisionCategory.LANDMARK,
    }
)

MIN_CLUE_LENGTH = 4


@dataclass
class AnalysisResult:
    """What one analysis produced, in the shape the report and UI want."""

    analysis: VisionAnalysis
    observation_ids: list[uuid.UUID] = field(default_factory=list)
    clues: list[dict[str, Any]] = field(default_factory=list)

    @property
    def by_category(self) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for observation in self.analysis.observations:
            grouped.setdefault(str(observation.category), []).append(observation.as_dict())
        return grouped

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.analysis.provider,
            "model": self.analysis.model,
            "available": self.analysis.available,
            "error": self.analysis.error,
            "duration_ms": self.analysis.duration_ms,
            "people_present": self.analysis.people_present,
            "observation_count": len(self.analysis.observations),
            "by_category": self.by_category,
            "clues": self.clues,
            # Surfaced, not hidden. A model that repeatedly attempts
            # identification is something an operator should be able to see.
            "rejected": [dict(r) for r in self.analysis.rejected],
        }


class VisionService:
    """Runs a vision provider and stores what it saw as observations."""

    def __init__(
        self,
        session: AsyncSession,
        settings: Settings,
        provider: VisionProvider,
        clock: Clock | None = None,
    ) -> None:
        self.session = session
        self.settings = settings
        self.provider = provider
        self.clock = clock or SystemClock()

    async def analyse(
        self, *, investigation_id: uuid.UUID, image: Image, image_bytes: bytes
    ) -> AnalysisResult:
        analysis = await self.provider.analyse(
            image_bytes, image.mime_type or "image/png"
        )

        result = AnalysisResult(analysis=analysis)
        if not analysis.available or analysis.error:
            return result

        observations = ObservationRepository(self.session, self.clock)

        for seen in analysis.observations:
            record = await observations.record(
                investigation_id=investigation_id,
                image_id=image.id,
                method=ObservationMethod.VISION,
                raw_value=seen.value,
                extractor_version=f"{EXTRACTOR_VERSION}:{analysis.model}",
                # The category and placement travel as context so a reviewer can
                # find the thing in the picture and check it.
                context_snippet=(
                    f"[{seen.category}]"
                    + (f" {seen.detail}" if seen.detail else "")
                    + ("" if seen.verbatim else " (paraphrased, not verbatim)")
                ),
                method_confidence=seen.confidence,
            )
            result.observation_ids.append(record.id)

        result.clues = self._clues(analysis)

        logger.info(
            "vision.analysed",
            investigation_id=str(investigation_id),
            image_id=str(image.id),
            observations=len(result.observation_ids),
            clues=len(result.clues),
            rejected=len(analysis.rejected),
        )
        return result

    def _clues(self, analysis: VisionAnalysis) -> list[dict[str, Any]]:
        """Search clues the observations justify.

        The model's own suggestions come first, then anything searchable it
        transcribed but did not think to propose. Both are re-checked against
        the guardrails here: a clue is the point where an observation would
        leave the system and become a query, so it is the last place to stop an
        identification attempt.
        """
        clues: list[dict[str, Any]] = []
        seen: set[str] = set()

        for clue in analysis.clues:
            key = clue.query.strip().casefold()
            if key in seen or not clue_is_safe(clue.query):
                continue
            seen.add(key)
            clues.append({**clue.as_dict(), "origin": "model"})

        for observation in analysis.observations:
            if observation.category not in _SEARCHABLE:
                continue
            value = observation.value.strip()
            key = value.casefold()
            if len(value) < MIN_CLUE_LENGTH or key in seen or not clue_is_safe(value):
                continue
            seen.add(key)
            clues.append(
                {
                    "query": value,
                    "rationale": f"Transcribed from a {observation.category} in the image.",
                    "source_category": str(observation.category),
                    "priority": 1 if observation.verbatim else 0,
                    "origin": "transcription",
                }
            )

        clues.sort(key=lambda c: -c["priority"])
        return clues


__all__ = ["EXTRACTOR_VERSION", "AnalysisResult", "VisionService"]
