"""Response types for the engine-facing service layer.

These were referenced by service.py but never committed, so
`nexgen_engine.api.service` raised ModuleNotFoundError on import and every
/biometrics route was dead. Field names are fixed by their existing consumers
in backend/app/api/routes_biometrics.py -- do not rename without updating the
route serializers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EngineMatch:
    """One candidate returned by a 1:N search."""

    identity_id: str
    #: Cosine similarity in [-1, 1]. Named "confidence" for the API contract,
    #: but it is a similarity score, NOT a calibrated probability.
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EngineSearchResponse:
    """Result of an enroll or identify call."""

    decision: str
    quality_score: float
    liveness_score: float
    review_required: bool
    reasons: list[str] = field(default_factory=list)
    matches: list[EngineMatch] = field(default_factory=list)
    audit_hash: str = ""
