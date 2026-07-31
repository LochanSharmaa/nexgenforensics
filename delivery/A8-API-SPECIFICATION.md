# A8 — API and Interface Specification

**Generated:** 2026-07-31 19:42 UTC · **Repository state:** `cc96a43f62e1`

The complete HTTP surface: 36 endpoints, their handlers, the request and response schemas, and the authentication and governance rules that apply to them.

---

## Conventions

| Aspect | Rule |
|---|---|
| Base URL | Configured per deployment; `http://127.0.0.1:8443` in development |
| Authentication | `Authorization: Bearer <access token>` or `X-API-Key: <key>` |
| Content type | `application/json`; images as base64 strings, not multipart |
| Errors | `{"detail": "..."}`, with `request_id` on 500 |
| CSRF | Required on state-changing requests that do NOT carry a header credential |
| Rate limiting | Per-principal, and per-IP on unauthenticated auth endpoints |
| Caching | `Cache-Control: no-store` on every response |

**Lawful basis.** Every biometric operation requires a stated lawful basis when
`NEXGEN_REQUIRE_LAWFUL_BASIS` is on. The system does not evaluate whether a
basis is lawful; it ensures one was stated and records it verbatim in the audit
chain.

---

## Endpoint index

| Method | Path | Handler |
|---|---|---|
| GET | `/api/admin/api-keys` | `list_api_keys` |
| POST | `/api/admin/api-keys` | `create_api_key` |
| DELETE | `/api/admin/api-keys/{key_id}` | `revoke_api_key` |
| GET | `/api/audit` | `list_audit_records` |
| GET | `/api/audit/verify` | `verify_chain` |
| GET | `/api/auth/csrf` | `csrf_token` |
| POST | `/api/auth/forgot-password` | `forgot_password` |
| POST | `/api/auth/login` | `login` |
| POST | `/api/auth/logout` | `logout` |
| GET | `/api/auth/me` | `me` |
| POST | `/api/auth/refresh` | `refresh` |
| POST | `/api/auth/register` | `register` |
| POST | `/api/auth/resend-otp` | `resend_otp` |
| POST | `/api/auth/reset-password` | `reset_password` |
| POST | `/api/auth/users` | `create_user` |
| POST | `/api/auth/verify-email` | `verify_email` |
| GET | `/api/cases` | `list_cases` |
| POST | `/api/cases` | `create_case` |
| GET | `/api/cases/{case_id}` | `get_case` |
| PATCH | `/api/cases/{case_id}` | `update_case` |
| GET | `/api/cases/{case_id}/report` | `export_report` |
| GET | `/api/health` | `health` |
| POST | `/api/imatch/batch` | `batch` |
| POST | `/api/imatch/candidates/{candidate_id}/adjudicate` | `adjudicate` |
| GET | `/api/imatch/engine/metrics` | `engine_metrics` |
| GET | `/api/imatch/engine/status` | `engine_status` |
| POST | `/api/imatch/search` | `search` |
| GET | `/api/imatch/searches` | `list_searches` |
| GET | `/api/imatch/searches/{search_id}/candidates` | `list_candidates` |
| POST | `/api/imatch/verify` | `verify` |
| GET | `/api/subjects` | `list_subjects` |
| POST | `/api/subjects` | `enrol` |
| DELETE | `/api/subjects/{subject_id}` | `delete_subject` |
| GET | `/api/subjects/{subject_id}` | `get_subject` |
| GET | `/api/subjects/{subject_id}/templates` | `list_templates` |
| DELETE | `/api/subjects/{subject_id}/templates/{template_id}` | `delete_template` |

---

## Endpoint detail

### GET `/api/admin/api-keys`

**Handler:** `list_api_keys`

*No handler docstring.*

### POST `/api/admin/api-keys`

**Handler:** `create_api_key`

```text
Issue a machine credential.

The plaintext key appears in this response and nowhere else -- only its hash
is stored, so it cannot be recovered or re-displayed. A key can never exceed
the role of the admin who created it, and it is bound to their tenant.
```

### DELETE `/api/admin/api-keys/{key_id}`

**Handler:** `revoke_api_key`

*No handler docstring.*

### GET `/api/audit`

**Handler:** `list_audit_records`

```text
Read the caller's tenant audit trail.

Any authenticated user can read it, including their own entries. An audit
log that only administrators can see is much easier to quietly misuse, and
the operators being logged have a legitimate interest in what was recorded
about them.
```

### GET `/api/audit/verify`

**Handler:** `verify_chain`

```text
Recompute the hash chain and report the first break, if any.

A ``valid: false`` result means records were altered or removed after being
written, which is a security incident rather than a data-quality issue.
```

### GET `/api/auth/csrf`

**Handler:** `csrf_token`

```text
Mint a CSRF token for this browser.

The cookie is deliberately NOT HTTPOnly, unlike the session cookies. That
is the whole mechanism: the page has to be able to read this value in order
to echo it back in a header, and an attacker's page cannot read it because
the same-origin policy stops them reading another origin's cookies. The
value is signed, so it also cannot simply be invented.
```

### POST `/api/auth/forgot-password`

**Handler:** `forgot_password`

*No handler docstring.*

### POST `/api/auth/login`

**Handler:** `login`

*No handler docstring.*

### POST `/api/auth/logout`

**Handler:** `logout`

```text
End the session.

The stored refresh-token hash is cleared, so the refresh token presented
later is rejected and the session cannot be resumed. Access tokens remain
stateless and expire on their own within NEXGEN_ACCESS_TOKEN_MINUTES; the
cookies carrying them are cleared here too.
```

### GET `/api/auth/me`

**Handler:** `me`

*No handler docstring.*

### POST `/api/auth/refresh`

**Handler:** `refresh`

*No handler docstring.*

### POST `/api/auth/register`

**Handler:** `register`

```text
Self-service registration, creating an UNVERIFIED account.

Disabled unless NEXGEN_ALLOW_SELF_REGISTRATION is set. This is a biometric
investigation system: who is able to run a search is a controlled decision,
so an open signup form is something an operator switches on deliberately
rather than inherits as a default.
```

### POST `/api/auth/resend-otp`

**Handler:** `resend_otp`

*No handler docstring.*

### POST `/api/auth/reset-password`

**Handler:** `reset_password`

*No handler docstring.*

### POST `/api/auth/users`

**Handler:** `create_user`

```text
Create a user inside the caller's tenant.

The tenant is taken from the authenticated principal, never from the
request, so an admin cannot create accounts in another tenant.
```

### POST `/api/auth/verify-email`

**Handler:** `verify_email`

*No handler docstring.*

### GET `/api/cases`

**Handler:** `list_cases`

```text
List cases in the caller's tenant.

Investigators see only their own cases; supervisors and admins see the whole
tenant. A case file names people who may never be charged, so read access is
not granted tenant-wide by default.
```

### POST `/api/cases`

**Handler:** `create_case`

*No handler docstring.*

### GET `/api/cases/{case_id}`

**Handler:** `get_case`

*No handler docstring.*

### PATCH `/api/cases/{case_id}`

**Handler:** `update_case`

*No handler docstring.*

### GET `/api/cases/{case_id}/report`

**Handler:** `export_report`

```text
Export a case report as JSON or Markdown.

Exports are themselves audited: a report is a copy of biometric findings
leaving the system, which is exactly the event a later review will ask about.
```

### GET `/api/health`

**Handler:** `health`

```text
Liveness and readiness in one call.

Unauthenticated on purpose so load balancers can reach it. It exposes no
tenant data and no configuration values beyond the environment name.
```

### POST `/api/imatch/batch`

**Handler:** `batch`

```text
Batch 1:1 comparison (``pair``) or batch 1:N gallery search (``gallery``).

Design notes, because batch endpoints attract two common mistakes:

1. **One bad image does not fail the batch.** Each item is isolated; a
   decode failure or a frame with no face is recorded as that item's error
   and the rest continue. An operator processing 40 stills from a scene
   should not lose 39 good results to one corrupt file.
2. **Every item is audited individually.** A batch is not one search, it is
   N searches, and the audit trail has to be able to answer "why was this
   specific person compared" for each one. The lawful basis is recorded
   against every item, not once for the batch.

Processing is sequential. Concurrency here would need request batching at
the ONNX layer to be worth anything, which is not built; see BENCHMARKS.md
section 7b for the measured single-threaded cost (~15 ms per encode, so a
pair item costs ~30 ms).
```

### POST `/api/imatch/candidates/{candidate_id}/adjudicate`

**Handler:** `adjudicate`

```text
Record an examiner's verdict on a candidate.

This is the only place a candidate becomes "confirmed", and only a human can
do it. The engine never writes this field.
```

### GET `/api/imatch/engine/metrics`

**Handler:** `engine_metrics`

```text
Per-stage latency percentiles for the running process.

Authenticated deliberately. Timing data is a side channel: response times
leak gallery size and whether a probe matched, so this sits behind the same
auth as every other biometric endpoint rather than on an open /metrics path.

Percentiles come from a bounded in-process window (see
nexgen_engine/observability.py), so they describe THIS process since it
started -- not a historical series and not a cluster. Restarting the
service resets them.
```

### GET `/api/imatch/engine/status`

**Handler:** `engine_status`

```text
What the engine actually is right now.

``recognizer.recognition_capable`` is the field that matters: when it is
false the service is running the deterministic stub and no score it returns
means anything. The console surfaces this as a banner.
```

### POST `/api/imatch/search`

**Handler:** `search`

```text
Search a probe image against the caller's tenant gallery.

The gallery searched is always the authenticated principal's tenant. There
is no parameter to search another tenant, by design.
```

### GET `/api/imatch/searches`

**Handler:** `list_searches`

*No handler docstring.*

### GET `/api/imatch/searches/{search_id}/candidates`

**Handler:** `list_candidates`

*No handler docstring.*

### POST `/api/imatch/verify`

**Handler:** `verify`

```text
One-to-one comparison of two supplied images. Nothing is enrolled.
```

### GET `/api/subjects`

**Handler:** `list_subjects`

*No handler docstring.*

### POST `/api/subjects`

**Handler:** `enrol`

```text
Add a person, or another image of an existing person, to the gallery.

Supervisor-only: enrolment determines who the system is *able* to find, so it
is a deliberately higher-privilege action than running a search.

Poor enrolment images are the most common cause of downstream false matches,
so a low-quality image is rejected outright rather than quietly enrolled.
```

### DELETE `/api/subjects/{subject_id}`

**Handler:** `delete_subject`

```text
Erase a subject: templates, enrolment images, and gallery entries.

A real deletion, not a flag. Retaining biometric data after an erasure
request is the failure mode that gets these systems shut down. Search
history is preserved, because deleting the record of past searches would
destroy the audit trail; the candidate rows keep the subject id only.
```

### GET `/api/subjects/{subject_id}`

**Handler:** `get_subject`

*No handler docstring.*

### GET `/api/subjects/{subject_id}/templates`

**Handler:** `list_templates`

```text
Template metadata only.

The template vector itself is never returned. An ArcFace embedding is enough
to reconstruct a recognizable approximation of the face, so it is treated as
equivalent to the biometric it was derived from.
```

### DELETE `/api/subjects/{subject_id}/templates/{template_id}`

**Handler:** `delete_template`

*No handler docstring.*

---

## Request and response schemas

Pydantic models defining every documented request and response shape, in full.

```python
from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator

from ..db.models import Adjudication, CaseStatus, Role

# Account e-mail.
#
# This deliberately does NOT use pydantic's EmailStr. EmailStr delegates to
# `email-validator`, which rejects special-use and reserved domains -- `.local`
# among them. That produced a real broken state: `bootstrap_admin.py` writes
# straight to the database and happily creates `investigator@nexgen.local`,
# while the login endpoint then refuses that exact address as malformed. The
# system could mint an account that could never authenticate, and the error
# shown at the door ("not a valid email address") pointed at the wrong thing.
#
# Internal and air-gapped deployments legitimately use domains like
# `.local`, `.internal` and `.lan`, so the rule to drop is deliverability, not
# syntax. What remains is a shape check that still catches ordinary typos
# ("alice", "alice@agency") while accepting anything routable inside an
# organisation.
#
# Applied to BOTH creation and login, because the invariant that matters is
# that every account the system lets you create is an account you can sign
# into.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _normalise_email(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    cleaned = value.strip().lower()
    if not _EMAIL_RE.match(cleaned):
        raise ValueError("must look like name@domain.tld")
    return cleaned


AccountEmail = Annotated[
    str,
    BeforeValidator(_normalise_email),
    Field(min_length=3, max_length=320),
]

# The one line that must accompany every automated match shown to a human.
INVESTIGATIVE_NOTICE = (
    "Automated face recognition returns investigative leads, not identifications. "
    "A qualified examiner must verify any candidate before it is relied upon."
)


# ------------------------------------------------------------------- auth ----


class LoginRequest(BaseModel):
    email: AccountEmail
    password: str = Field(min_length=1, max_length=256)
    remember_me: bool = False
    tenant: str = Field(default="", description="Tenant slug. Required when the email exists in several tenants.")


class RegisterRequest(BaseModel):
    full_name: str = Field(min_length=1, max_length=200)
    email: AccountEmail
    password: str = Field(min_length=1, max_length=256)
    confirm_password: str = Field(min_length=1, max_length=256)
    tenant: str = Field(default="", max_length=120)

    @model_validator(mode="after")
    def _passwords_match(self):
        # Compared here rather than in the route so the mismatch is reported as
        # a field error on confirm_password, next to the input that is wrong.
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match.")
        return self


class VerifyEmailRequest(BaseModel):
    email: AccountEmail
    otp: str = Field(min_length=4, max_length=12)


class ResendOtpRequest(BaseModel):
    email: AccountEmail


class ForgotPasswordRequest(BaseModel):
    email: AccountEmail


class ResetPasswordRequest(BaseModel):
    token: str = Field(min_length=16, max_length=512)
    password: str = Field(min_length=1, max_length=256)
    confirm_password: str = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def _passwords_match(self):
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match.")
        return self


class MessageResponse(BaseModel):
    message: str
    email_verified: bool | None = None


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
    email: AccountEmail
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

```

## Machine-readable specification

A complete OpenAPI 3 document is served by the running service at `/openapi.json`, with interactive documentation at `/docs` when not in production. That document is generated from the same type annotations reproduced above, so the two cannot disagree.
