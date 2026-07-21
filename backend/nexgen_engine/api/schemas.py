from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EngineMatch(BaseModel):
    identity_id: str
    confidence: float = Field(ge=-1.0, le=1.0)
    metadata: dict[str, Any]


class EngineSearchResponse(BaseModel):
    decision: str
    quality_score: float = Field(ge=0.0, le=1.0)
    liveness_score: float = Field(ge=0.0, le=1.0)
    review_required: bool
    reasons: list[str]
    matches: list[EngineMatch]
    audit_hash: str | None = None
