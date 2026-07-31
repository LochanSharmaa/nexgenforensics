from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from sqlmodel import Field, Index, SQLModel


def _uuid() -> str:
    return uuid.uuid4().hex


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Role(str, Enum):
    """Least privilege by default.

    INVESTIGATOR  - run searches, manage their own cases, adjudicate candidates
    SUPERVISOR    - all of the above, plus enrol subjects and read every case
                    in the tenant
    ADMIN         - user and tenant administration, audit export, purge
    """

    INVESTIGATOR = "investigator"
    SUPERVISOR = "supervisor"
    ADMIN = "admin"


class CaseStatus(str, Enum):
    OPEN = "open"
    PENDING_REVIEW = "pending_review"
    CLOSED = "closed"
    ARCHIVED = "archived"


class Adjudication(str, Enum):
    """An examiner's verdict on a candidate.

    ``CONFIRMED`` records the examiner's professional judgement that the images
    depict the same person. It is never set by the system: the engine only ever
    proposes, because an automated face match is an investigative lead and not
    an identification.
    """

    PENDING = "pending"
    CONFIRMED = "confirmed"
    ELIMINATED = "eliminated"
    INCONCLUSIVE = "inconclusive"


class Tenant(SQLModel, table=True):
    __tablename__ = "tenants"

    id: str = Field(default_factory=_uuid, primary_key=True)
    slug: str = Field(index=True, unique=True)
    name: str
    active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=utcnow)


class User(SQLModel, table=True):
    __tablename__ = "users"
    __table_args__ = (Index("ix_users_tenant_email", "tenant_id", "email", unique=True),)

    id: str = Field(default_factory=_uuid, primary_key=True)
    tenant_id: str = Field(foreign_key="tenants.id", index=True)
    email: str = Field(index=True)
    full_name: str = ""
    password_hash: str
    role: Role = Field(default=Role.INVESTIGATOR)
    active: bool = Field(default=True)
    badge_number: str = ""
    last_login_at: datetime | None = Field(default=None)
    created_at: datetime = Field(default_factory=utcnow)

    # --- e-mail verification -------------------------------------------
    # Accounts created by an administrator (bootstrap_admin.py, or the
    # supervisor-only POST /auth/users) are verified on creation: a human with
    # authority already vouched for the address. Only self-service
    # registration starts unverified, because there nobody has.
    email_verified: bool = Field(default=False)
    # Only the HASH of the one-time code is stored, for the same reason only
    # password hashes are: a database copy must not hand over live credentials.
    otp_hash: str | None = Field(default=None)
    otp_expires_at: datetime | None = Field(default=None)
    otp_attempts: int = Field(default=0)
    otp_sent_count: int = Field(default=0)
    otp_window_started_at: datetime | None = Field(default=None)

    # --- password reset -------------------------------------------------
    reset_token_hash: str | None = Field(default=None)
    reset_token_expires_at: datetime | None = Field(default=None)

    # --- brute-force protection -----------------------------------------
    failed_login_attempts: int = Field(default=0)
    locked_until: datetime | None = Field(default=None)

    # --- session -------------------------------------------------------
    # Hash of the currently-valid refresh token. Storing it is what makes
    # logout actually revoke: before this, refresh tokens stayed usable until
    # they expired and /logout only wrote an audit line.
    refresh_token_hash: str | None = Field(default=None)
    refresh_token_expires_at: datetime | None = Field(default=None)
    # Bumped on password reset. Any access token minted before this instant is
    # rejected, which is how "invalidate previous sessions" reaches tokens that
    # are stateless by design.
    session_epoch: int = Field(default=0)

    last_login_ip: str = ""
    updated_at: datetime = Field(default_factory=utcnow)


class ApiKey(SQLModel, table=True):
    """Machine credential for system-to-system integrations.

    Only the hash is stored. The plaintext key is shown exactly once at creation
    and cannot be recovered afterwards.
    """

    __tablename__ = "api_keys"

    id: str = Field(default_factory=_uuid, primary_key=True)
    tenant_id: str = Field(foreign_key="tenants.id", index=True)
    name: str
    prefix: str = Field(index=True)
    key_hash: str
    role: Role = Field(default=Role.INVESTIGATOR)
    active: bool = Field(default=True)
    created_by: str = Field(foreign_key="users.id")
    last_used_at: datetime | None = Field(default=None)
    expires_at: datetime | None = Field(default=None)
    created_at: datetime = Field(default_factory=utcnow)


class Case(SQLModel, table=True):
    __tablename__ = "cases"
    __table_args__ = (Index("ix_cases_tenant_reference", "tenant_id", "reference", unique=True),)

    id: str = Field(default_factory=_uuid, primary_key=True)
    tenant_id: str = Field(foreign_key="tenants.id", index=True)
    reference: str = Field(index=True)
    title: str
    description: str = ""
    status: CaseStatus = Field(default=CaseStatus.OPEN)
    # The legal ground for processing biometric data on this case. Captured up
    # front so every downstream search inherits a recorded justification.
    lawful_basis: str = ""
    owner_id: str = Field(foreign_key="users.id", index=True)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    closed_at: datetime | None = Field(default=None)


class Subject(SQLModel, table=True):
    """A person enrolled in the gallery."""

    __tablename__ = "subjects"

    id: str = Field(default_factory=_uuid, primary_key=True)
    tenant_id: str = Field(foreign_key="tenants.id", index=True)
    external_ref: str = Field(default="", index=True)
    display_name: str = ""
    notes: str = ""
    case_id: str | None = Field(default=None, foreign_key="cases.id", index=True)
    enrolled_by: str = Field(foreign_key="users.id")
    active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=utcnow)


class Template(SQLModel, table=True):
    """One encrypted biometric template derived from one enrolment image.

    The vector is never stored in the clear: ``ciphertext`` is AES-256-GCM
    sealed under a per-tenant subkey, with the tenant id bound in as additional
    authenticated data. ``image_sha256`` lets an examiner prove which image a
    template came from without keeping the image itself.
    """

    __tablename__ = "templates"

    id: str = Field(default_factory=_uuid, primary_key=True)
    tenant_id: str = Field(foreign_key="tenants.id", index=True)
    subject_id: str = Field(foreign_key="subjects.id", index=True)

    nonce: str
    ciphertext: str
    dimensions: int = 512

    recognizer_backend: str = ""
    recognizer_pack: str = ""
    quality_score: float = 0.0
    detector: str = ""
    image_sha256: str = Field(default="", index=True)
    image_path: str = ""

    created_by: str = Field(foreign_key="users.id")
    created_at: datetime = Field(default_factory=utcnow)


class SearchRun(SQLModel, table=True):
    """One probe searched against the gallery."""

    __tablename__ = "search_runs"

    id: str = Field(default_factory=_uuid, primary_key=True)
    tenant_id: str = Field(foreign_key="tenants.id", index=True)
    case_id: str | None = Field(default=None, foreign_key="cases.id", index=True)
    operator_id: str = Field(foreign_key="users.id", index=True)

    mode: str = "single"
    lawful_basis: str = ""
    purpose: str = ""

    decision: str = ""
    top_score: float = 0.0
    margin: float = 0.0
    gallery_size: int = 0
    candidate_count: int = 0

    quality_score: float = 0.0
    liveness_score: float = 0.0
    deepfake_risk: float = 0.0
    faces_detected: int = 0
    review_required: bool = True
    recognition_capable: bool = False
    reasons: str = ""
    explanation: str = ""

    recognizer_backend: str = ""
    recognizer_pack: str = ""
    match_threshold: float = 0.0
    review_threshold: float = 0.0

    probe_sha256: str = Field(default="", index=True)
    probe_path: str = ""
    duration_ms: int = 0
    audit_hash: str = ""
    created_at: datetime = Field(default_factory=utcnow, index=True)


class Candidate(SQLModel, table=True):
    """One ranked result within a search run, plus its examiner verdict."""

    __tablename__ = "candidates"

    id: str = Field(default_factory=_uuid, primary_key=True)
    tenant_id: str = Field(foreign_key="tenants.id", index=True)
    search_run_id: str = Field(foreign_key="search_runs.id", index=True)
    subject_id: str = Field(foreign_key="subjects.id", index=True)
    template_id: str = Field(foreign_key="templates.id")

    rank: int = 0
    score: float = 0.0
    normalized_score: float = 0.0

    adjudication: Adjudication = Field(default=Adjudication.PENDING)
    adjudicated_by: str | None = Field(default=None, foreign_key="users.id")
    adjudicated_at: datetime | None = Field(default=None)
    examiner_notes: str = ""


class AuditRecord(SQLModel, table=True):
    """Append-only, hash-chained record of every consequential action.

    ``previous_hash``/``entry_hash`` form a chain per tenant, so deleting or
    editing a row breaks verification from that point on. The chain is also
    mirrored to a JSONL file on disk to survive database compromise.
    """

    __tablename__ = "audit_records"
    __table_args__ = (
        Index("ix_audit_tenant_created", "tenant_id", "created_at"),
        # Chain position must be unique per tenant. If two processes ever append
        # concurrently they would compute the same previous_hash and silently
        # fork the chain; this constraint turns that into a loud insert failure.
        Index("ix_audit_tenant_sequence", "tenant_id", "sequence", unique=True),
    )

    id: str = Field(default_factory=_uuid, primary_key=True)
    tenant_id: str = Field(foreign_key="tenants.id", index=True)
    # Explicit chain position, starting at 1 per tenant.
    #
    # Ordering by timestamp is not safe here: created_at has millisecond
    # resolution at best, several records routinely land in the same tick, and
    # the previous tiebreaker was a random UUID. That made chain order
    # non-deterministic, so verification could report tampering on an untouched
    # log -- the single most damaging false alarm this system can raise.
    sequence: int = Field(default=0, index=True)
    actor_id: str = Field(default="", index=True)
    actor_label: str = ""
    action: str = Field(index=True)
    resource_type: str = ""
    resource_id: str = ""
    outcome: str = ""
    lawful_basis: str = ""
    detail: str = ""
    ip_address: str = ""
    user_agent: str = ""
    previous_hash: str = ""
    entry_hash: str = Field(default="", index=True)
    created_at: datetime = Field(default_factory=utcnow, index=True)


__all__ = [
    "Adjudication",
    "ApiKey",
    "AuditRecord",
    "Candidate",
    "Case",
    "CaseStatus",
    "Role",
    "SearchRun",
    "Subject",
    "Template",
    "Tenant",
    "User",
    "utcnow",
]
