from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QualityResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    resolution_score: float = Field(ge=0.0, le=1.0)
    brightness_score: float = Field(ge=0.0, le=1.0)
    contrast_score: float = Field(ge=0.0, le=1.0)
    sharpness_score: float = Field(ge=0.0, le=1.0)
    reasons: list[str]


class LivenessResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    signals: list[str]


class MatchCandidate(BaseModel):
    id: str
    identity_id: str
    workspace: str
    score: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any]


class ImatchResponse(BaseModel):
    decision: str
    mode: str
    quality: QualityResult
    liveness: LivenessResult
    score: float = Field(ge=0.0, le=1.0)
    match: dict[str, Any]
    matches: list[MatchCandidate]
    review_required: bool
    reasons: list[str]
    audit: dict[str, Any]
