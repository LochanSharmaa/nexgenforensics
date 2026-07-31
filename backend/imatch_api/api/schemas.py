from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from ..db.models import Adjudication, CaseStatus, Role

# The one line that must accompany every automated match shown to a human.
INVESTIGATIVE_NOTICE = (
    "Automated face recognition returns investigative leads, not identifications. "
    "A qualified examiner must verify any candidate before it is relied upon."
)


# ------------------------------------------------------------------- auth ----


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=256)
    tenant: str = Field(default="", description="Tenant slug. Required when the email exists in several tenants.")


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: "UserResponse"


class RefreshRequest(BaseModel):
    refresh_token: str


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    email: str
    full_name: str
    role: Role
    tenant_id: str
    badge_number: str = ""
    last_login_at: datetime | None = None


class CreateUserRequest(BaseModel):
    email: EmailStr
    full_name: str = Field(default="", max_length=200)
    password: str = Field(min_length=12, max_length=256)
    role: Role = Role.INVESTIGATOR
    badge_number: str = Field(default="", max_length=64)


# ------------------------------------------------------------------ cases ----


class CreateCaseRequest(BaseModel):
    reference: str = Field(min_length=1, max_length=120)
    title: str = Field(min_length=1, max_length=250)
    description: str = Field(default="", max_length=5000)
    lawful_basis: str = Field(default="", max_length=500)


class UpdateCaseRequest(BaseModel):
    title: str | None = Field(default=None, max_length=250)
    description: str | None = Field(default=None, max_length=5000)
    status: CaseStatus | None = None
    lawful_basis: str | None = Field(default=None, max_length=500)


class CaseResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    reference: str
    title: str
    description: str
    status: CaseStatus
    lawful_basis: str
    owner_id: str
    created_at: datetime
    updated_at: datetime
    closed_at: datetime | None = None


class CaseDetailResponse(CaseResponse):
    search_count: int = 0
    subject_count: int = 0
    confirmed_count: int = 0


# --------------------------------------------------------------- subjects ----


class EnrolRequest(BaseModel):
    """Add a person to the searchable gallery.

    Either ``image_base64`` or a multipart upload must be supplied. Enrolment is
    a supervisor action because it decides who the system is capable of finding.
    """

    display_name: str = Field(default="", max_length=200)
    external_ref: str = Field(default="", max_length=120)
    notes: str = Field(default="", max_length=5000)
    case_id: str | None = None
    subject_id: str | None = Field(default=None, description="Add another image to an existing subject.")
    image_base64: str = Field(min_length=1)
    lawful_basis: str = Field(default="", max_length=500)

    @field_validator("image_base64")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("image_base64 must not be empty.")
        return value


class SubjectResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    display_name: str
    external_ref: str
    notes: str
    case_id: str | None
    active: bool
    created_at: datetime


class TemplateResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    subject_id: str
    quality_score: float
    # Named recognizer_*, not model_*: Pydantic v2 reserves the "model_" prefix
    # for its own namespace, and a field using it emits a namespace-conflict
    # warning on every model construction.
    recognizer_pack: str
    detector: str
    image_sha256: str
    created_at: datetime


class EnrolResponse(BaseModel):
    subject: SubjectResponse
    template: TemplateResponse
    quality: dict[str, Any]
    liveness: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)
    audit_hash: str


# ----------------------------------------------------------------- search ----


class SearchRequest(BaseModel):
    """A biometric search against the caller's tenant gallery.

    ``lawful_basis`` is mandatory when NEXGEN_REQUIRE_LAWFUL_BASIS is on. It is
    recorded verbatim in the audit chain: the system cannot judge whether a
    search was lawful, but it can make sure someone had to state a reason and
    that the statement is preserved.
    """

    image_base64: str | None = None
    source_url: str | None = Field(default=None, max_length=2000)
    mode: str = Field(default="single", pattern="^(single|compare|batch|url)$")
    case_id: str | None = None
    top_k: int = Field(default=10, ge=1, le=100)
    lawful_basis: str = Field(default="", max_length=500)
    purpose: str = Field(default="", max_length=500)
    checks: list[str] = Field(default_factory=list)

    @field_validator("source_url")
    @classmethod
    def _https_only(cls, value: str | None) -> str | None:
        if value and not value.startswith("https://"):
            raise ValueError("source_url must be an https:// address.")
        return value


class CandidateResponse(BaseModel):
    id: str
    rank: int
    subject_id: str
    template_id: str
    subject_name: str = ""
    external_ref: str = ""
    score: float
    normalized_score: float
    adjudication: Adjudication
    examiner_notes: str = ""


class ProbeAssessment(BaseModel):
    quality: dict[str, Any]
    liveness: dict[str, Any]
    deepfake_risk: float
    faces_detected: int
    detector: str
    box: dict[str, int]
    pose: dict[str, float]


class SearchResponse(BaseModel):
    search_id: str
    decision: str
    confidence: float
    explanation: str
    review_required: bool
    recognition_capable: bool
    reasons: list[str] = Field(default_factory=list)
    candidates: list[CandidateResponse] = Field(default_factory=list)
    probe: ProbeAssessment
    gallery_size: int
    margin: float
    thresholds: dict[str, float]
    model: dict[str, Any]
    duration_ms: int
    audit_hash: str
    notice: str = INVESTIGATIVE_NOTICE


class VerifyRequest(BaseModel):
    """One-to-one comparison of two images. No gallery involved."""

    reference_image_base64: str = Field(min_length=1)
    probe_image_base64: str = Field(min_length=1)
    case_id: str | None = None
    lawful_basis: str = Field(default="", max_length=500)


class VerifyResponse(BaseModel):
    similarity: float
    verified: bool
    threshold: float
    explanation: str
    recognition_capable: bool
    reference: ProbeAssessment
    probe: ProbeAssessment
    morphing: dict[str, Any]
    audit_hash: str
    notice: str = INVESTIGATIVE_NOTICE


class BatchItem(BaseModel):
    """One unit of work in a batch.

    ``probe_image_base64`` is always required. ``reference_image_base64`` is
    required in pair mode and ignored in gallery mode.
    """

    label: str = Field(default="", max_length=200, description="Operator's name for this item, echoed back")
    probe_image_base64: str = Field(min_length=1)
    reference_image_base64: str | None = None


class BatchRequest(BaseModel):
    """Batch 1:1 comparison or batch 1:N gallery search.

    Three modes, because they answer different questions and an endpoint that
    guessed between them would be worse than making the caller say:

      one_to_many -- THE DEFAULT. One reference image, supplied once at request
                 level, compared against every uploaded probe. This is the
                 common investigative case: "here is my suspect, check these 30
                 stills." The reference is encoded once.
      pair    -- each item carries its OWN reference; every item is an
                 independent 1:1 verification of a different couple. Use when
                 comparing 30 distinct pairs, not one face against 30.
      gallery -- each probe is searched against the caller's tenant gallery.
                 Returns nothing useful until subjects are enrolled.

    A batch is capped rather than unbounded: each item costs two full encodes
    in pair mode, and an operator pasting a folder of 5,000 images should get a
    clear rejection instead of a request that appears to hang.
    """

    mode: str = Field(default="one_to_many", pattern="^(one_to_many|pair|gallery)$")
    items: list[BatchItem] = Field(min_length=1, max_length=50)
    case_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=50, description="gallery mode only")
    lawful_basis: str = Field(default="", max_length=500)

    #: one_to_many mode only: ONE reference compared against every uploaded
    #: probe. Supplied once at request level, not per item, so it is encoded a
    #: single time instead of N times -- for 20 probes that is 21 encodes
    #: rather than 40, roughly halving the work (~15ms per encode, see
    #: BENCHMARKS.md 7b).
    reference_image_base64: str | None = None


class BatchItemResult(BaseModel):
    """Per-item outcome. ``error`` is set instead of results when that one item
    failed; one unreadable image must not void the whole batch."""

    index: int
    label: str
    status: str = Field(description="ok | error")
    error: str = ""
    # pair mode
    similarity: float | None = None
    verified: bool | None = None
    # gallery mode
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    gallery_size: int | None = None
    # both
    probe_quality: float | None = None
    probe_liveness: float | None = None
    probe_deepfake_risk: float | None = None
    audit_hash: str = ""


class BatchResponse(BaseModel):
    mode: str
    threshold: float
    recognition_capable: bool
    submitted: int
    succeeded: int
    failed: int
    results: list[BatchItemResult]
    notice: str = INVESTIGATIVE_NOTICE


class AdjudicateRequest(BaseModel):
    adjudication: Adjudication
    examiner_notes: str = Field(default="", max_length=5000)

    @field_validator("adjudication")
    @classmethod
    def _not_pending(cls, value: Adjudication) -> Adjudication:
        if value == Adjudication.PENDING:
            raise ValueError("Adjudication must be confirmed, eliminated, or inconclusive.")
        return value


class SearchRunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    case_id: str | None
    operator_id: str
    decision: str
    top_score: float
    margin: float
    gallery_size: int
    candidate_count: int
    quality_score: float
    liveness_score: float
    review_required: bool
    recognition_capable: bool
    explanation: str
    probe_sha256: str
    duration_ms: int
    created_at: datetime


# ------------------------------------------------------------------ audit ----


class AuditRecordResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    # Chain position within the tenant, starting at 1. A gap means a record was
    # removed.
    sequence: int
    actor_id: str
    actor_label: str
    action: str
    resource_type: str
    resource_id: str
    outcome: str
    lawful_basis: str
    detail: str
    ip_address: str
    entry_hash: str
    created_at: datetime


class ChainVerificationResponse(BaseModel):
    valid: bool
    records_checked: int
    broken_at: str | None = None
    reason: str = ""


# ------------------------------------------------------------------ admin ----


class CreateApiKeyRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    role: Role = Role.INVESTIGATOR
    expires_in_days: int | None = Field(default=None, ge=1, le=3650)


class ApiKeyResponse(BaseModel):
    id: str
    name: str
    prefix: str
    role: Role
    active: bool
    expires_at: datetime | None
    created_at: datetime


class CreatedApiKeyResponse(ApiKeyResponse):
    api_key: str = Field(description="Shown once. It cannot be retrieved again.")


class EngineStatusResponse(BaseModel):
    recognizer: dict[str, Any]
    detector: dict[str, Any]
    device: dict[str, Any]
    capabilities: dict[str, Any]
    thresholds: dict[str, float]
    embedding_dim: int
    flip_tta: bool
    model_load_seconds: float
    recognition_capable: bool
    gallery: dict[str, int]


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    database: str
    recognition_capable: bool


TokenResponse.model_rebuild()


__all__ = [
    "INVESTIGATIVE_NOTICE",
    "AdjudicateRequest",
    "ApiKeyResponse",
    "AuditRecordResponse",
    "CandidateResponse",
    "CaseDetailResponse",
    "CaseResponse",
    "ChainVerificationResponse",
    "CreateApiKeyRequest",
    "CreateCaseRequest",
    "CreateUserRequest",
    "CreatedApiKeyResponse",
    "EngineStatusResponse",
    "EnrolRequest",
    "EnrolResponse",
    "HealthResponse",
    "LoginRequest",
    "ProbeAssessment",
    "RefreshRequest",
    "SearchRequest",
    "SearchResponse",
    "SearchRunResponse",
    "SubjectResponse",
    "TemplateResponse",
    "TokenResponse",
    "UpdateCaseRequest",
    "UserResponse",
    "VerifyRequest",
    "VerifyResponse",
]
