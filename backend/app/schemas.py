from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

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
    report: dict[str, Any] | None = None


class SourceImageFindings(BaseModel):
    """Evidence metadata retained with a generated comparison report."""

    sha256: str
    quality_score: float = Field(ge=0.0, le=1.0)
    pose_yaw: float = 0.0
    pose_pitch: float = 0.0
    liveness_score: float = Field(ge=0.0, le=1.0)
    thumbnail_base64: str | None = None


class ModelSimilarityScore(BaseModel):
    model_name: str
    score: float = Field(ge=0.0, le=1.0)


class MeasurementFinding(BaseModel):
    name: str
    value: float
    confidence: float = Field(ge=0.0, le=1.0)
    unit: str = "ratio"


class ForensicFindings(BaseModel):
    """Immutable, structured input to deterministic report rendering."""

    case_id: str
    examiner_id: str
    timestamp: datetime
    tenant_id: str
    source_images: list[SourceImageFindings] = Field(min_length=1)
    model_scores: list[ModelSimilarityScore] = Field(min_length=1)
    fused_score: float = Field(ge=0.0, le=1.0)
    calibrated_match_probability: float = Field(ge=0.0, le=100.0)
    decision: Literal["match", "no_match", "inconclusive"]
    threshold_value: float = Field(ge=0.0, le=1.0)
    false_match_rate: float = Field(ge=0.0, le=1.0)
    measurements: list[MeasurementFinding] = []
    audit_hash: str = Field(min_length=1)
    config_version: str = "nexgen-default-v1"
