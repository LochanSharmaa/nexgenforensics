"""Gemini vision provider.

Constrained on three sides so the model's helpfulness cannot become a liability:

1. **A response schema with no identity field.** The model is asked for a
   structured object whose shape simply has no place to put "who this is".
2. **A prompt that states the boundary and the reason for it.** Models follow
   rules better when the rule makes sense to them.
3. **Post-generation guardrails** (`vision.guardrails`), because 1 and 2 are
   requests and only 3 is a control.

The model transcribes and describes. It does not conclude. "The sign reads
MERIDIAN LOGISTICS" is wanted; "this is Meridian's head office in Delhi" is not
— that is a claim about the world, and claims need sources.
"""

from __future__ import annotations

import json
import time
from typing import Any

from shared.enums import VisionCategory
from shared.logging import get_logger

from ..base import SearchClue, VisionAnalysis, VisionObservation
from ..guardrails import Rejection, clue_is_safe, filter_observations

logger = get_logger(__name__)

DEFAULT_MODEL = "gemini-flash-latest"

SYSTEM_PROMPT = """\
You are the observation stage of an evidence-first OSINT platform. Your only job \
is to report WHAT IS VISIBLE in the image. You are not the investigator and you \
are not the judge.

Report only what a careful person could confirm by looking at the picture.

DO:
- Transcribe visible text exactly, character for character, including on signs, \
labels, screens, documents, number plates and packaging.
- Name logos and wordmarks you can read.
- Describe objects, vehicles, architecture and setting.
- Note landmarks only if the structure itself is distinctive and identifiable.
- Record any dates, times, reference numbers or document titles that are legible.
- Note environmental cues that bear on place: language and script of signage, \
road markings, plug sockets, licence-plate format, flags, currency, vegetation.

DO NOT:
- Identify any person, by name or otherwise. Not from their face, clothing, \
posture, or context. If people are present, only count them.
- Infer anyone's age, ethnicity, nationality, religion, gender or occupation \
from their appearance.
- State where the photograph was taken as a fact. Report the cues; the platform \
corroborates them against sources.
- Guess. If text is partly illegible, transcribe what is readable and mark it \
not verbatim. An honest partial reading is useful; an invented complete one is \
harmful.

Then propose search queries that follow from what you SAW — a transcribed \
company name, a document reference, a distinctive landmark. Never a query that \
tries to identify a person.
"""

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "observations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": [c.value for c in VisionCategory],
                    },
                    "value": {"type": "string"},
                    "detail": {"type": "string"},
                    "confidence": {"type": "number"},
                    "verbatim": {"type": "boolean"},
                },
                "required": ["category", "value"],
            },
        },
        "search_clues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "rationale": {"type": "string"},
                    "source_category": {
                        "type": "string",
                        "enum": [c.value for c in VisionCategory],
                    },
                    "priority": {"type": "integer"},
                },
                "required": ["query", "rationale"],
            },
        },
        # A count, deliberately. There is no field for who they are, so the
        # well-behaved path cannot express an identification at all.
        "people_present": {"type": "integer"},
    },
    "required": ["observations"],
}


class GeminiVisionProvider:
    name = "gemini"

    def __init__(self, api_key: str = "", model: str = DEFAULT_MODEL, *, timeout: float = 60.0):
        self.api_key = (api_key or "").strip()
        self.model = model
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.api_key)

    async def analyse(self, image: bytes, mime_type: str = "image/png") -> VisionAnalysis:
        if not self.available():
            return VisionAnalysis(
                provider=self.name,
                model=self.model,
                available=False,
                error="IIE_GEMINI_API_KEY is not set.",
            )

        started = time.perf_counter()
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            response = await client.aio.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_bytes(data=image, mime_type=mime_type),
                    "Report what is visible in this image, then propose search clues.",
                ],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=RESPONSE_SCHEMA,
                    # Low but non-zero: transcription should be faithful, and
                    # near-deterministic output makes a re-run reproducible.
                    temperature=0.1,
                ),
            )
            payload = json.loads(response.text)
        except Exception as exc:  # noqa: BLE001 - a failed analysis is a result
            logger.warning("vision.failed", provider=self.name, error=str(exc))
            return VisionAnalysis(
                provider=self.name,
                model=self.model,
                available=True,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        return self._parse(payload, started)

    def _parse(self, payload: dict[str, Any], started: float) -> VisionAnalysis:
        raw: list[VisionObservation] = []
        for entry in payload.get("observations") or []:
            try:
                category = VisionCategory(entry["category"])
            except (KeyError, ValueError):
                # An unknown category is still an observation; filing it as a
                # general visual clue keeps it rather than dropping evidence
                # over a vocabulary mismatch.
                category = VisionCategory.VISUAL_CLUE
            raw.append(
                VisionObservation(
                    category=category,
                    value=str(entry.get("value", "")).strip(),
                    detail=str(entry.get("detail", "")).strip(),
                    confidence=float(entry.get("confidence") or 0.0),
                    verbatim=bool(entry.get("verbatim", True)),
                )
            )

        kept, observation_rejections = filter_observations(raw)
        rejected = list(observation_rejections)

        clues: list[SearchClue] = []
        for entry in payload.get("search_clues") or []:
            query = str(entry.get("query", "")).strip()
            if not query:
                continue
            if not clue_is_safe(query):
                # Recorded, not silently dropped: a model that keeps proposing
                # identification queries is a fact worth knowing about the model.
                rejected.append(
                    Rejection(
                        value=query,
                        rule="unsafe_clue",
                        reason="Search clue would attempt to identify a person.",
                    )
                )
                continue
            try:
                source = VisionCategory(entry.get("source_category", "VISUAL_CLUE"))
            except ValueError:
                source = VisionCategory.VISUAL_CLUE
            clues.append(
                SearchClue(
                    query=query,
                    rationale=str(entry.get("rationale", "")).strip(),
                    source_category=source,
                    priority=int(entry.get("priority") or 0),
                )
            )

        if rejected:
            logger.warning(
                "vision.output_rejected",
                provider=self.name,
                count=len(rejected),
                rules=[r.rule for r in rejected],
            )

        return VisionAnalysis(
            provider=self.name,
            model=self.model,
            available=True,
            observations=kept,
            clues=tuple(clues),
            people_present=int(payload.get("people_present") or 0),
            rejected=tuple(r.as_dict() for r in rejected),
            duration_ms=int((time.perf_counter() - started) * 1000),
            raw_response=payload,
        )


def build(settings) -> GeminiVisionProvider:  # noqa: ANN001
    return GeminiVisionProvider(settings.gemini_api_key, settings.gemini_model)


__all__ = ["DEFAULT_MODEL", "RESPONSE_SCHEMA", "SYSTEM_PROMPT", "GeminiVisionProvider", "build"]
