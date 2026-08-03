"""Guardrails on vision output.

The prohibition — *never identify a person from appearance* — cannot rest on
prompt wording. A prompt is a request; a model under pressure to be helpful will
sometimes answer the question it was told not to answer. So the prompt is the
first layer and this module is the second: every observation is checked after
generation, and anything asserting identity is dropped and recorded as dropped.

Three layers, outermost first:

1. **No slot for it.** The response schema has no person-identity field, so the
   well-behaved path cannot express one.
2. **This module.** Free-text values are scanned for identity assertions,
   appearance-based inference, and demographic profiling.
3. **The evidence model.** A `PERSON` graph node cannot exist without a page
   that named them (DATA_MODEL invariant 2), so even a leaked identity claim
   has nowhere to become a finding.

Dropped output is preserved rather than discarded: a model that keeps trying to
identify people is a fact worth knowing about the model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from shared.enums import VisionCategory

from .base import VisionObservation

# Phrasings that assert who someone is, or infer it from how they look. Matched
# case-insensitively against the observation value and its detail.
_IDENTITY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "identity_assertion",
        re.compile(
            r"\b(this|the)\s+(person|man|woman|individual|subject)\s+(is|appears to be|"
            r"looks like|resembles|seems to be)\b",
            re.I,
        ),
    ),
    (
        "identity_assertion",
        re.compile(r"\b(identified as|identifiable as|known to be|believed to be)\b", re.I),
    ),
    (
        "appearance_inference",
        re.compile(
            r"\b(based on|from|judging by)\s+(his|her|their|the)\s+"
            r"(face|features|appearance|likeness)\b",
            re.I,
        ),
    ),
    (
        "facial_analysis",
        re.compile(
            r"\b(facial (recognition|features|structure|analysis)|face match|"
            r"biometric|likeness match)\b",
            re.I,
        ),
    ),
    (
        "demographic_profiling",
        re.compile(
            r"\b(appears to be|looks|estimated)\s+(around\s+)?"
            r"(\d{1,2}\s*(-|to)?\s*\d{0,2}\s*years old|in (his|her|their) \w+s)\b",
            re.I,
        ),
    ),
    (
        "demographic_profiling",
        re.compile(
            r"\b(ethnicity|race|nationality|religion|sexual orientation)\s*[:=]",
            re.I,
        ),
    ),
)

# Categories that may never carry a personal name even incidentally. TEXT and
# DOCUMENT are excluded on purpose: a name printed on a nameplate or a byline is
# *text visible in the image*, which is exactly the kind of observation this
# system exists to capture. What it must not do is attach that name to a face.
_NAME_SENSITIVE = frozenset({VisionCategory.VISUAL_CLUE, VisionCategory.OBJECT})

_PERSON_NOUN = re.compile(
    r"\b(person|man|woman|boy|girl|child|individual|subject|face)\b", re.I
)
_CAPITALISED_NAME = re.compile(r"\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b")


@dataclass(frozen=True)
class Rejection:
    value: str
    rule: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {"value": self.value[:300], "rule": self.rule, "reason": self.reason}


def inspect(observation: VisionObservation) -> Rejection | None:
    """Why this observation must be dropped, or None if it may stand."""
    haystack = f"{observation.value} {observation.detail}".strip()

    for rule, pattern in _IDENTITY_PATTERNS:
        if pattern.search(haystack):
            return Rejection(
                value=observation.value,
                rule=rule,
                reason=(
                    "Asserts or infers who a depicted person is. Identity may only "
                    "come from a source that publishes it, never from appearance."
                ),
            )

    # A name next to a person-noun in a descriptive category is the model
    # captioning a face rather than transcribing text.
    if (
        observation.category in _NAME_SENSITIVE
        and _PERSON_NOUN.search(haystack)
        and _CAPITALISED_NAME.search(haystack)
    ):
        return Rejection(
            value=observation.value,
            rule="name_attached_to_person",
            reason=(
                "Attaches a personal name to a depicted person. A name read "
                "off a sign belongs in TEXT; a name attached to a face is an "
                "identification."
            ),
        )

    if not observation.value.strip():
        return Rejection(
            value=observation.value, rule="empty", reason="Observation carries no value."
        )

    return None


def filter_observations(
    observations: tuple[VisionObservation, ...] | list[VisionObservation],
) -> tuple[tuple[VisionObservation, ...], tuple[Rejection, ...]]:
    """Split model output into what may be kept and what was refused."""
    kept: list[VisionObservation] = []
    rejected: list[Rejection] = []

    for observation in observations:
        verdict = inspect(observation)
        if verdict is None:
            kept.append(observation)
        else:
            rejected.append(verdict)

    return tuple(kept), tuple(rejected)


def clue_is_safe(query: str) -> bool:
    """Whether a generated search clue may be run.

    A clue built from a face — "who is the man in the red jacket" — would turn
    the search layer into the identification path the vision layer just refused
    to be. Blocked at the point it would leave the system.
    """
    lowered = query.lower()
    if any(pattern.search(query) for _rule, pattern in _IDENTITY_PATTERNS):
        return False
    return not (
        _PERSON_NOUN.search(lowered)
        and re.search(r"\b(who is|who are|identify|name of)\b", lowered)
    )


__all__ = ["Rejection", "clue_is_safe", "filter_observations", "inspect"]
