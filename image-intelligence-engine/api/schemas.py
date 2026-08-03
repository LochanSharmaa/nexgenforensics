"""Request and response models.

Phase 2 surface only: auth, investigations, audit. The evidence-facing schemas
in API.md arrive with the tables that back them in Phase 3.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from shared.enums import (
    ArtifactType,
    FactClassification,
    InvestigationStatus,
    ReviewStatus,
    RunTrigger,
)


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# -- auth ------------------------------------------------------------------


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=200)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"  # noqa: S105 - OAuth token type, not a secret
    expires_in_minutes: int


class UserResponse(ORMModel):
    id: uuid.UUID
    email: str
    display_name: str
    role: str


# -- investigations --------------------------------------------------------


class InvestigationCreate(BaseModel):
    case_id: str = Field(min_length=1, max_length=120)
    title: str = Field(min_length=1, max_length=300)
    lawful_basis: str = Field(
        default="",
        max_length=500,
        description="Why this subject may lawfully be examined. Recorded in the audit chain.",
    )
    purpose: str = Field(default="", max_length=500)
    description: str = Field(default="", max_length=10_000)
    jurisdiction: str = Field(default="IN", max_length=8)
    retention_days: int | None = Field(default=None, ge=1, le=36_500)

    @field_validator("case_id", "title")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()


class InvestigationResponse(ORMModel):
    id: uuid.UUID
    case_id: str
    title: str
    description: str
    lawful_basis: str
    purpose: str
    jurisdiction: str
    status: str
    retention_expires_at: datetime | None
    completed_at: datetime | None
    created_at: datetime
    updated_at: datetime


class StatusTransitionRequest(BaseModel):
    to_status: InvestigationStatus
    reason: str = Field(
        default="",
        max_length=1000,
        description=(
            "Mandatory for backward transitions such as reopening a completed case."
        ),
    )


class StatusEventResponse(ORMModel):
    from_status: str
    to_status: str
    reason: str
    occurred_at: datetime


# -- audit -----------------------------------------------------------------


class AuditEntryResponse(ORMModel):
    id: int
    investigation_id: uuid.UUID | None
    actor_label: str
    action: str
    outcome: str
    lawful_basis: str
    resource_type: str
    resource_id: str
    detail: dict[str, Any]
    previous_hash: str
    entry_hash: str
    created_at: datetime


class ChainVerifyResponse(BaseModel):
    valid: bool
    records: int
    broken_at: int | None
    reason: str


# -- health ----------------------------------------------------------------


class ComponentHealth(BaseModel):
    name: str
    healthy: bool
    detail: str = ""


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    environment: str
    components: list[ComponentHealth]


# -- pipeline --------------------------------------------------------------


class RunStartRequest(BaseModel):
    trigger: RunTrigger = RunTrigger.MANUAL


class StageResponse(ORMModel):
    stage: str
    status: str
    items_total: int
    items_done: int
    items_failed: int
    started_at: datetime | None
    finished_at: datetime | None
    error: str


class RunResponse(ORMModel):
    id: uuid.UUID
    investigation_id: uuid.UUID
    trigger: str
    status: str
    started_at: datetime | None
    finished_at: datetime | None
    error: str


class ResumePlanResponse(BaseModel):
    next_stage: str | None
    completed: list[str]
    remaining: list[str]
    is_complete: bool
    stages_skipped: int = Field(
        description="Stages resume avoids re-running. Surfaced so the saving is "
        "visible rather than assumed — each one may represent a paid API call."
    )


class RunStartResponse(BaseModel):
    run: RunResponse
    resumed: bool
    plan: ResumePlanResponse
    progress: float


class RunDetailResponse(BaseModel):
    run: RunResponse
    stages: list[StageResponse]
    plan: ResumePlanResponse
    progress: float


# -- retention -------------------------------------------------------------


class HoldCreateRequest(BaseModel):
    reason: str = Field(min_length=1, max_length=1000)
    artifact_type: ArtifactType = ArtifactType.INVESTIGATION
    artifact_id: uuid.UUID | None = None


class HoldResponse(ORMModel):
    id: uuid.UUID
    investigation_id: uuid.UUID
    artifact_type: str
    artifact_id: uuid.UUID | None
    reason: str
    placed_at: datetime
    released_at: datetime | None


class RetentionStatusResponse(BaseModel):
    holds_total: int
    holds_active: int
    purge_blocked: bool
    block_reasons: list[str]
    retention_expires_at: datetime | None


# -- review queue ----------------------------------------------------------


class ReviewItemResponse(ORMModel):
    id: uuid.UUID
    kind: str
    subject_type: str
    subject_id: uuid.UUID
    proposal: dict[str, Any]
    rationale: dict[str, Any]
    priority: int
    status: str
    decision_note: str
    decided_at: datetime | None
    created_at: datetime


class ReviewDecisionRequest(BaseModel):
    status: ReviewStatus
    note: str = Field(default="", max_length=2000)

    @field_validator("status")
    @classmethod
    def _not_pending(cls, value: ReviewStatus) -> ReviewStatus:
        if value == ReviewStatus.PENDING:
            raise ValueError("A decision cannot set the item back to PENDING.")
        return value


class ReviewQueueSummary(BaseModel):
    pending: int
    blocks_completion: bool = Field(
        description="A case cannot move to COMPLETED while machine output "
        "sits unreviewed."
    )


# -- images ----------------------------------------------------------------


class ImageResponse(ORMModel):
    id: uuid.UUID
    investigation_id: uuid.UUID
    role: str
    sha256: str
    phash: str
    width: int | None
    height: int | None
    file_size: int | None
    mime_type: str
    exif: dict[str, Any]
    progress_state: str
    retention_state: str
    created_at: datetime


class ImageUploadResponse(BaseModel):
    image: ImageResponse
    deduplicated: bool = Field(
        description="True when identical bytes were already uploaded. The "
        "existing image is returned rather than a duplicate created, which "
        "would double-count it in every later stage."
    )


# -- vision analysis -------------------------------------------------------


class VisionObservationModel(BaseModel):
    category: str
    value: str
    detail: str = ""
    confidence: float = 0.0
    verbatim: bool = Field(
        default=True,
        description="False when the model paraphrased rather than transcribed. "
        "A paraphrased sign is weaker evidence than a quoted one.",
    )


class SearchClueModel(BaseModel):
    query: str
    rationale: str
    source_category: str
    priority: int = 0
    origin: str = Field(
        default="model",
        description="`model` if the vision model proposed it, `transcription` "
        "if derived from text it read.",
    )


class RejectedOutputModel(BaseModel):
    value: str
    rule: str
    reason: str


class VisionAnalysisResponse(BaseModel):
    """What the image shows — observations, never conclusions.

    Every entry here is a statement about the *image*, checkable by looking at
    it. Nothing in this response is a fact about the world; facts require
    corroborating sources and live in the findings endpoint.
    """

    provider: str
    model: str
    available: bool
    error: str = ""
    duration_ms: int = 0
    people_present: int = Field(
        default=0,
        description="A count only. Who they are is never asserted from an image.",
    )
    observation_count: int = 0
    by_category: dict[str, list[VisionObservationModel]] = Field(default_factory=dict)
    clues: list[SearchClueModel] = Field(default_factory=list)
    rejected: list[RejectedOutputModel] = Field(
        default_factory=list,
        description="Model output the guardrails refused, surfaced rather than "
        "silently dropped — a model that repeatedly attempts identification is "
        "worth an operator seeing.",
    )


# -- discovery and findings ------------------------------------------------


class DiscoverRequest(BaseModel):
    image_id: uuid.UUID | None = Field(
        default=None, description="Defaults to the investigation's first probe image."
    )
    urls: list[str] = Field(
        default_factory=list,
        max_length=25,
        description=(
            "Operator-supplied leads for the `manual` provider. Targeted "
            "corroboration of URLs the investigator already holds — never a crawl seed."
        ),
    )


class FindingModel(BaseModel):
    url: str
    site: str
    title: str
    match_kind: str
    match_label: str
    confidence: str
    provider: str
    reported_date: str
    verification: str
    image_url: str
    last_archived: str
    archive_url: str


class FindingsSummary(BaseModel):
    sources_found: int
    distinct_sites: int
    earliest_appearance: str
    latest_appearance: str
    exact_matches: int
    similar_only: int


class ProviderStateModel(BaseModel):
    run: list[str]
    unconfigured: list[dict[str, Any]]
    failed: list[dict[str, str]]


class FindingsResponse(BaseModel):
    searched: bool = Field(
        description="False means no provider has been asked yet — which is a "
        "different fact from having asked and found nothing."
    )
    summary: FindingsSummary
    findings: list[FindingModel]
    entities: list[str] = Field(
        default_factory=list,
        description="Labels a provider guessed for the image. Context, never identification.",
    )
    best_guess_labels: list[str] = Field(default_factory=list)
    providers: ProviderStateModel


class ProviderInfo(BaseModel):
    name: str
    title: str
    capabilities: list[str]
    requires_credentials: bool
    config_keys: list[str]
    cost_per_1k: float | None
    notes: str
    configured: bool
    status: str


# -- evidence --------------------------------------------------------------


class ObservationResponse(ORMModel):
    id: uuid.UUID
    page_id: uuid.UUID | None
    image_id: uuid.UUID | None
    method: str
    raw_value: str
    normalized_value: str
    char_start: int | None
    char_end: int | None
    context_snippet: str
    extractor_version: str
    method_confidence: float | None
    extracted_at: datetime


class FactResponse(ORMModel):
    id: uuid.UUID
    entity_id: uuid.UUID
    attribute: str
    value: str
    status: str
    classification: str
    confidence: str
    confidence_score: float
    confidence_explanation: dict[str, Any]
    independent_source_count: int
    observation_count: int
    conflict_group_id: uuid.UUID | None
    computed_at: datetime


class EvidenceChainResponse(BaseModel):
    """The traceability guarantee, as an API contract.

    `confidence` and `confidence_explanation` always travel together — there is
    no projection that returns one without the other.
    """

    fact: FactResponse
    chain: list[ObservationResponse]
    conflicts: list[FactResponse] = Field(default_factory=list)


class FactClassifyRequest(BaseModel):
    classification: FactClassification
    note: str = Field(default="", max_length=2000)


__all__ = [
    "AuditEntryResponse",
    "ChainVerifyResponse",
    "ComponentHealth",
    "DiscoverRequest",
    "EvidenceChainResponse",
    "FactClassifyRequest",
    "FactResponse",
    "FindingModel",
    "FindingsResponse",
    "FindingsSummary",
    "HealthResponse",
    "HoldCreateRequest",
    "HoldResponse",
    "ImageResponse",
    "ImageUploadResponse",
    "InvestigationCreate",
    "InvestigationResponse",
    "LoginRequest",
    "ObservationResponse",
    "ProviderInfo",
    "ProviderStateModel",
    "RejectedOutputModel",
    "ResumePlanResponse",
    "RetentionStatusResponse",
    "ReviewDecisionRequest",
    "ReviewItemResponse",
    "ReviewQueueSummary",
    "RunDetailResponse",
    "RunResponse",
    "RunStartRequest",
    "RunStartResponse",
    "SearchClueModel",
    "StageResponse",
    "StatusEventResponse",
    "StatusTransitionRequest",
    "TokenResponse",
    "UserResponse",
    "VisionAnalysisResponse",
    "VisionObservationModel",
]
