"""Controlled vocabularies and their legal transitions.

Every state machine in the platform lives here, in the domain layer, as pure
data. Two consequences, both deliberate:

* Transition tables are testable with zero infrastructure. The rules that decide
  whether an investigation may be completed, or whether evidence may be purged,
  are the rules most likely to be challenged, and they must be verifiable without
  a database.
* Nothing assigns a state directly. Repositories route every change through
  :func:`assert_transition`, so an illegal jump raises instead of persisting.

Stored as ``text`` + ``CHECK`` in PostgreSQL rather than native enums: adding a
value to a PG enum locks the table, and these vocabularies will grow.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Final


class IllegalTransition(ValueError):
    """A state change that the domain rules forbid."""

    def __init__(self, machine: str, current: str, requested: str, allowed: frozenset[str]) -> None:
        self.machine = machine
        self.current = current
        self.requested = requested
        self.allowed = allowed
        permitted = ", ".join(sorted(allowed)) or "(terminal state — no transitions)"
        super().__init__(
            f"{machine}: cannot move from {current!r} to {requested!r}. Permitted: {permitted}"
        )


# --------------------------------------------------------------------------
# Investigation workflow  (REVISION_3 §3)
# --------------------------------------------------------------------------


class InvestigationStatus(StrEnum):
    NEW = "NEW"
    ACTIVE = "ACTIVE"
    UNDER_REVIEW = "UNDER_REVIEW"
    COMPLETED = "COMPLETED"
    ARCHIVED = "ARCHIVED"
    DELETED_PENDING_RETENTION = "DELETED_PENDING_RETENTION"


INVESTIGATION_TRANSITIONS: Final[Mapping[str, frozenset[str]]] = {
    InvestigationStatus.NEW: frozenset(
        {InvestigationStatus.ACTIVE, InvestigationStatus.ARCHIVED}
    ),
    InvestigationStatus.ACTIVE: frozenset(
        {InvestigationStatus.UNDER_REVIEW, InvestigationStatus.ARCHIVED}
    ),
    InvestigationStatus.UNDER_REVIEW: frozenset(
        # Backward to ACTIVE is permitted: review routinely surfaces the need for
        # more collection. It is audited with a mandatory reason.
        {InvestigationStatus.COMPLETED, InvestigationStatus.ACTIVE}
    ),
    InvestigationStatus.COMPLETED: frozenset(
        # Real investigations reopen.
        {InvestigationStatus.ARCHIVED, InvestigationStatus.ACTIVE}
    ),
    InvestigationStatus.ARCHIVED: frozenset(
        {InvestigationStatus.DELETED_PENDING_RETENTION, InvestigationStatus.ACTIVE}
    ),
    # Not terminal: a retention hold placed after the deletion request must be
    # able to pull the case back. Deletion intent is reversible until the
    # retention engine actually purges.
    InvestigationStatus.DELETED_PENDING_RETENTION: frozenset({InvestigationStatus.ARCHIVED}),
}

BACKWARD_INVESTIGATION_TRANSITIONS: Final[frozenset[tuple[str, str]]] = frozenset(
    {
        (InvestigationStatus.UNDER_REVIEW, InvestigationStatus.ACTIVE),
        (InvestigationStatus.COMPLETED, InvestigationStatus.ACTIVE),
        (InvestigationStatus.ARCHIVED, InvestigationStatus.ACTIVE),
        (InvestigationStatus.DELETED_PENDING_RETENTION, InvestigationStatus.ARCHIVED),
    }
)
"""Transitions that require a mandatory reason. Reopening a completed case
without recording why is exactly the gap an opposing examiner looks for."""


# --------------------------------------------------------------------------
# Evidence lifecycle — two independent axes  (REVISION_3 §1)
# --------------------------------------------------------------------------


class ProgressState(StrEnum):
    """How far an artifact has moved through the investigation."""

    DISCOVERED = "DISCOVERED"
    DOWNLOADED = "DOWNLOADED"
    VERIFIED = "VERIFIED"
    REVIEWED = "REVIEWED"
    INCLUDED_IN_REPORT = "INCLUDED_IN_REPORT"


PROGRESS_TRANSITIONS: Final[Mapping[str, frozenset[str]]] = {
    ProgressState.DISCOVERED: frozenset({ProgressState.DOWNLOADED}),
    ProgressState.DOWNLOADED: frozenset({ProgressState.VERIFIED}),
    # Verification can fail and be re-run after a fetcher fix, so VERIFIED may
    # return to DOWNLOADED.
    ProgressState.VERIFIED: frozenset({ProgressState.REVIEWED, ProgressState.DOWNLOADED}),
    ProgressState.REVIEWED: frozenset(
        {ProgressState.INCLUDED_IN_REPORT, ProgressState.VERIFIED}
    ),
    ProgressState.INCLUDED_IN_REPORT: frozenset({ProgressState.REVIEWED}),
}


class RetentionState(StrEnum):
    """What the retention regime says about an artifact.

    Deliberately a separate axis from :class:`ProgressState`. An artifact can be
    ``INCLUDED_IN_REPORT`` *and* ``RETAINED`` under a legal hold simultaneously;
    a single column could not express that, and that combination is precisely the
    one compliance cares about.
    """

    ACTIVE = "ACTIVE"
    ARCHIVED = "ARCHIVED"
    RETAINED = "RETAINED"
    PURGED = "PURGED"


RETENTION_TRANSITIONS: Final[Mapping[str, frozenset[str]]] = {
    RetentionState.ACTIVE: frozenset({RetentionState.ARCHIVED, RetentionState.RETAINED}),
    RetentionState.ARCHIVED: frozenset(
        {RetentionState.RETAINED, RetentionState.PURGED, RetentionState.ACTIVE}
    ),
    RetentionState.RETAINED: frozenset({RetentionState.ARCHIVED, RetentionState.ACTIVE}),
    # Terminal. Content is gone; there is nothing to move back to.
    RetentionState.PURGED: frozenset(),
}

PURGE_REQUIRES_PRECONDITIONS: Final[bool] = True
"""``PURGED`` is reachable only via the retention engine, which additionally
checks for unreleased holds and live report references. The transition table
permits the move; it does not authorise it."""


class LifecycleAxis(StrEnum):
    PROGRESS = "PROGRESS"
    RETENTION = "RETENTION"


# --------------------------------------------------------------------------
# Findings and review  (REVISION_3 §4, §5)
# --------------------------------------------------------------------------


class FactStatus(StrEnum):
    """Evidential status. Machine-authored, derived from evidence alone."""

    COMMON = "COMMON"
    UNIQUE = "UNIQUE"
    CONFLICTED = "CONFLICTED"
    UNKNOWN = "UNKNOWN"


class FactClassification(StrEnum):
    """Investigative status. Human-authored judgement.

    Orthogonal to :class:`FactStatus` by design. A fact may be ``COMMON`` (three
    independent sources agree) and ``DISPUTED`` (the investigator has
    off-platform reason to doubt all three). One column could not hold both
    without silently overwriting either the evidence or the judgement.
    """

    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    DISPUTED = "DISPUTED"
    REJECTED = "REJECTED"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class ReviewStatus(StrEnum):
    PENDING = "PENDING"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"
    DEFERRED = "DEFERRED"


class VerificationState(StrEnum):
    """Whether a human has ruled on a machine-proposed entity."""

    MACHINE_PROPOSED = "MACHINE_PROPOSED"
    HUMAN_CONFIRMED = "HUMAN_CONFIRMED"
    HUMAN_REJECTED = "HUMAN_REJECTED"


class ConfidenceTier(StrEnum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNCERTAIN = "UNCERTAIN"


# --------------------------------------------------------------------------
# Custody  (REVISION_3 §2)
# --------------------------------------------------------------------------


class CustodyAction(StrEnum):
    COLLECTED = "COLLECTED"
    HASHED = "HASHED"
    TRANSFORMED = "TRANSFORMED"
    SCREENSHOT_CAPTURED = "SCREENSHOT_CAPTURED"
    EXPORTED = "EXPORTED"
    INCLUDED_IN_REPORT = "INCLUDED_IN_REPORT"
    MIGRATED = "MIGRATED"


class ActorKind(StrEnum):
    HUMAN = "HUMAN"
    SYSTEM = "SYSTEM"


class ArtifactType(StrEnum):
    IMAGE = "IMAGE"
    PAGE = "PAGE"
    SCREENSHOT = "SCREENSHOT"
    HTML_SNAPSHOT = "HTML_SNAPSHOT"
    REPORT = "REPORT"
    EXPORT_PACKAGE = "EXPORT_PACKAGE"
    INVESTIGATION = "INVESTIGATION"


# --------------------------------------------------------------------------
# Pipeline  (ARCHITECTURE §6)
# --------------------------------------------------------------------------


class PipelineStage(StrEnum):
    INGEST = "INGEST"
    DISCOVER = "DISCOVER"
    VERIFY = "VERIFY"
    CRAWL = "CRAWL"
    PARSE = "PARSE"
    CLASSIFY_DOMAIN = "CLASSIFY_DOMAIN"
    OCR = "OCR"
    EXTRACT = "EXTRACT"
    CLUSTER = "CLUSTER"
    CORRELATE = "CORRELATE"
    SCORE = "SCORE"
    GRAPH = "GRAPH"
    TIMELINE = "TIMELINE"
    SUMMARIZE = "SUMMARIZE"
    REPORT = "REPORT"


PIPELINE_ORDER: Final[tuple[PipelineStage, ...]] = tuple(PipelineStage)
"""Declaration order is execution order. Resume finds the first non-OK stage in
this sequence, which is why SCORE must follow CLUSTER — independence cannot be
computed before duplicate content is collapsed."""


class StageStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    OK = "OK"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class RunStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class RunTrigger(StrEnum):
    MANUAL = "MANUAL"
    MONITOR = "MONITOR"
    RESUME = "RESUME"


# --------------------------------------------------------------------------
# Sources and images
# --------------------------------------------------------------------------


class DomainClassification(StrEnum):
    """Descriptive metadata about a domain. **Never evaluative.**

    ``GOVERNMENT`` means the domain is governmental. It does not mean the page
    is true, and ARCHITECTURE §9.3 bars this value from influencing any
    confidence score — encoding institutional trust into a number presented as
    objective would smuggle in an editorial judgement.
    """

    GOVERNMENT = "GOVERNMENT"
    COMPANY = "COMPANY"
    EDUCATIONAL = "EDUCATIONAL"
    NEWS = "NEWS"
    SOCIAL_MEDIA = "SOCIAL_MEDIA"
    FORUM = "FORUM"
    BLOG = "BLOG"
    ARCHIVE = "ARCHIVE"
    DOCUMENTATION = "DOCUMENTATION"
    UNKNOWN = "UNKNOWN"


class ImageRole(StrEnum):
    PROBE = "PROBE"              # uploaded by the investigator
    DISCOVERED = "DISCOVERED"    # found during discovery


class ImageRelationship(StrEnum):
    """How one image relates to another.

    Every one of these is decided from **file content** — cryptographic hash,
    perceptual hash, dimensions. None inspects faces (ARCHITECTURE §8.1).
    """

    EXACT_COPY = "EXACT_COPY"
    RESIZED_COPY = "RESIZED_COPY"
    CROPPED_COPY = "CROPPED_COPY"
    MIRRORED_COPY = "MIRRORED_COPY"
    COMPRESSED_COPY = "COMPRESSED_COPY"
    NEAR_DUPLICATE = "NEAR_DUPLICATE"
    THUMBNAIL = "THUMBNAIL"
    UNVERIFIED = "UNVERIFIED"    # provider asserted it; local fetch failed
    REJECTED = "REJECTED"        # provider was wrong; excluded from findings


class VerificationResult(StrEnum):
    PENDING = "PENDING"
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"
    UNREACHABLE = "UNREACHABLE"


class CopyRole(StrEnum):
    """A page's role within a duplicate-content cluster (ARCHITECTURE §10.2)."""

    ORIGINAL = "ORIGINAL"
    REPOST = "REPOST"
    MIRROR = "MIRROR"
    TRANSLATION = "TRANSLATION"
    PARTIAL_COPY = "PARTIAL_COPY"
    MODIFIED_COPY = "MODIFIED_COPY"


class ProviderCapability(StrEnum):
    """What a provider plugin declares it can do (ARCHITECTURE §15).

    Wayback is an archive lookup and Reddit is a content source; neither is
    image discovery. One interface would have forced half-implemented adapters.
    """

    IMAGE_DISCOVERY = "IMAGE_DISCOVERY"
    ARCHIVE_LOOKUP = "ARCHIVE_LOOKUP"
    CONTENT_SOURCE = "CONTENT_SOURCE"
    PAGE_METADATA = "PAGE_METADATA"


# --------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------


class ObservationOrigin(StrEnum):
    """Who produced a record. Enforced by CHECK on both sides: `observations`
    accepts only EXTRACTED, `notes` only HUMAN. An investigator's hypothesis
    must never become indistinguishable from a crawled fact."""

    EXTRACTED = "EXTRACTED"
    HUMAN = "HUMAN"


class VisionCategory(StrEnum):
    """What kind of thing a vision model reported seeing.

    Every one of these is a statement about *the image*, not about the world.
    "The sign reads MERIDIAN LOGISTICS" is an observation; "this is the Meridian
    Logistics head office" is a conclusion and requires corroborating evidence
    before it may be stated.

    There is deliberately no PERSON category. A vision model may note that a
    person is present — that is visible — but it may never say who they are, and
    the absence of a slot for that is the first line of the defence.
    """

    TEXT = "TEXT"
    SIGN = "SIGN"
    LOGO = "LOGO"
    OBJECT = "OBJECT"
    LANDMARK = "LANDMARK"
    DOCUMENT = "DOCUMENT"
    DATE = "DATE"
    LOCATION_CUE = "LOCATION_CUE"
    VEHICLE = "VEHICLE"
    VISUAL_CLUE = "VISUAL_CLUE"


class ObservationMethod(StrEnum):
    """How an observation was extracted. Feeds the extraction-method weighting
    in confidence scoring — a schema.org block is a site deliberately publishing
    a machine-readable assertion; a regex hit on body text is a guess."""

    SCHEMA_ORG = "SCHEMA_ORG"
    OPENGRAPH = "OPENGRAPH"
    META = "META"
    NER = "NER"
    REGEX = "REGEX"
    OCR = "OCR"
    VISION = "VISION"
    CAPTION = "CAPTION"
    TITLE = "TITLE"
    PROVIDER = "PROVIDER"


METHOD_STRENGTH: Final[Mapping[str, str]] = {
    ObservationMethod.SCHEMA_ORG: "STRUCTURED",
    ObservationMethod.OPENGRAPH: "STRUCTURED",
    ObservationMethod.META: "DECLARED",
    ObservationMethod.PROVIDER: "DECLARED",
    ObservationMethod.CAPTION: "INFERRED",
    ObservationMethod.TITLE: "INFERRED",
    ObservationMethod.NER: "INFERRED",
    ObservationMethod.REGEX: "INFERRED",
    # A vision model reading a sign is a machine reading, like OCR — better at
    # context, no more authoritative about the world. Rated accordingly: it is
    # evidence of what the image shows, never of what is true.
    ObservationMethod.VISION: "OCR",
    ObservationMethod.OCR: "OCR",
}
"""Method → strength tier, consumed by the confidence engine in Phase 11.
Declared here so scoring stays a pure function over stored data."""


class EntityType(StrEnum):
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    EVENT = "EVENT"
    USERNAME = "USERNAME"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    WEBSITE = "WEBSITE"
    DOCUMENT = "DOCUMENT"
    PRODUCT = "PRODUCT"
    VEHICLE = "VEHICLE"
    LANDMARK = "LANDMARK"


class ReviewKind(StrEnum):
    ENTITY_CANDIDATE = "ENTITY_CANDIDATE"
    FACT_CANDIDATE = "FACT_CANDIDATE"
    DUPLICATE_MERGE = "DUPLICATE_MERGE"
    CONFLICT = "CONFLICT"
    PROVENANCE_CLASS = "PROVENANCE_CLASS"


# --------------------------------------------------------------------------
# Graph and timeline
# --------------------------------------------------------------------------


class NodeType(StrEnum):
    IMAGE = "IMAGE"
    PAGE = "PAGE"
    DOMAIN = "DOMAIN"
    ARTICLE = "ARTICLE"
    ORGANIZATION = "ORGANIZATION"
    PERSON = "PERSON"
    LOCATION = "LOCATION"
    EVENT = "EVENT"
    DOCUMENT = "DOCUMENT"
    SOCIAL_PROFILE = "SOCIAL_PROFILE"
    ARCHIVE = "ARCHIVE"


class EdgeType(StrEnum):
    APPEARS_ON = "APPEARS_ON"
    HOSTED_BY = "HOSTED_BY"
    MENTIONS = "MENTIONS"
    AUTHORED_BY = "AUTHORED_BY"
    LOCATED_IN = "LOCATED_IN"
    ARCHIVED_AS = "ARCHIVED_AS"
    COPY_OF = "COPY_OF"
    LINKS_TO = "LINKS_TO"
    EMPLOYED_BY = "EMPLOYED_BY"


class TimePrecision(StrEnum):
    """Stored so the UI renders "2019" rather than "1 Jan 2019 00:00".
    Rendering an inferred date as exact would be a lie of formatting."""

    EXACT = "EXACT"
    DAY = "DAY"
    MONTH = "MONTH"
    YEAR = "YEAR"
    INFERRED = "INFERRED"


class TimelineKind(StrEnum):
    IMAGE_FIRST_APPEARANCE = "IMAGE_FIRST_APPEARANCE"
    LATEST_APPEARANCE = "LATEST_APPEARANCE"
    ARCHIVE_SNAPSHOT = "ARCHIVE_SNAPSHOT"
    NEWS_PUBLICATION = "NEWS_PUBLICATION"
    SITE_UPDATE = "SITE_UPDATE"
    PAGE_REMOVED = "PAGE_REMOVED"
    INVESTIGATION_ACTION = "INVESTIGATION_ACTION"


# --------------------------------------------------------------------------
# Annotation, monitoring, reporting
# --------------------------------------------------------------------------


class TargetType(StrEnum):
    """What a note or bookmark can point at."""

    FACT = "FACT"
    PAGE = "PAGE"
    IMAGE = "IMAGE"
    ENTITY = "ENTITY"
    TIMELINE_EVENT = "TIMELINE_EVENT"
    OBSERVATION = "OBSERVATION"


class MonitorCadence(StrEnum):
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"


class ChangeType(StrEnum):
    """Detected by a monitoring sweep.

    ``PAGE_REMOVED`` and ``PAGE_UNOBSERVABLE`` are deliberately separate. A
    404/410 is evidence of removal; a 403, 429, timeout or robots.txt change
    means only that we can no longer observe the page. Conflating them would
    manufacture a takedown event out of a transient block.
    """

    NEW_APPEARANCE = "NEW_APPEARANCE"
    PAGE_REMOVED = "PAGE_REMOVED"
    PAGE_UPDATED = "PAGE_UPDATED"
    IMAGE_REPLACED = "IMAGE_REPLACED"
    ARCHIVE_ADDED = "ARCHIVE_ADDED"
    PAGE_UNOBSERVABLE = "PAGE_UNOBSERVABLE"


REMOVAL_CONFIRMING_STATUSES: Final[frozenset[int]] = frozenset({404, 410})
"""Only these prove removal. Everything else that fails is unobservable."""

REMOVAL_CONFIRMATIONS_REQUIRED: Final[int] = 2
"""Consecutive confirming sweeps before a removal is asserted, so one outage
never fabricates a takedown."""


class ReportFormat(StrEnum):
    HTML = "HTML"
    PDF = "PDF"
    JSON = "JSON"
    MARKDOWN = "MARKDOWN"


class CopilotRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class ValidationStatus(StrEnum):
    """Outcome of post-generation citation validation. REJECTED is a normal,
    persisted result — the assistant's failures are auditable too."""

    PENDING = "PENDING"
    PASSED = "PASSED"
    REJECTED = "REJECTED"


class SearchObjectType(StrEnum):
    CASE = "CASE"
    IMAGE = "IMAGE"
    DOMAIN = "DOMAIN"
    PAGE = "PAGE"
    ENTITY = "ENTITY"
    EVENT = "EVENT"
    OCR_TEXT = "OCR_TEXT"
    REPORT = "REPORT"
    NOTE = "NOTE"
    FACT = "FACT"


# --------------------------------------------------------------------------
# Transition checking
# --------------------------------------------------------------------------

_MACHINES: Final[Mapping[str, Mapping[str, frozenset[str]]]] = {
    "investigation_status": INVESTIGATION_TRANSITIONS,
    "progress_state": PROGRESS_TRANSITIONS,
    "retention_state": RETENTION_TRANSITIONS,
}


def allowed_transitions(machine: str, current: str) -> frozenset[str]:
    """States reachable from ``current``. Empty means terminal."""
    try:
        table = _MACHINES[machine]
    except KeyError:
        raise KeyError(f"Unknown state machine {machine!r}. Known: {sorted(_MACHINES)}") from None
    if current not in table:
        raise KeyError(f"{machine}: unknown state {current!r}")
    return table[current]


def can_transition(machine: str, current: str, requested: str) -> bool:
    return requested in allowed_transitions(machine, current)


def assert_transition(machine: str, current: str, requested: str) -> None:
    """Raise :class:`IllegalTransition` unless the move is permitted.

    A no-op self-transition is rejected rather than silently allowed: writing a
    lifecycle event that records no change would pollute the custody history
    with noise.
    """
    permitted = allowed_transitions(machine, current)
    if requested not in permitted:
        raise IllegalTransition(machine, current, requested, permitted)


def requires_reason(machine: str, current: str, requested: str) -> bool:
    """Whether this transition must carry an operator-supplied reason."""
    if machine != "investigation_status":
        return False
    return (current, requested) in BACKWARD_INVESTIGATION_TRANSITIONS


__all__ = [
    "ActorKind",
    "ArtifactType",
    "ChangeType",
    "ConfidenceTier",
    "CopilotRole",
    "CopyRole",
    "CustodyAction",
    "DomainClassification",
    "EdgeType",
    "EntityType",
    "FactClassification",
    "FactStatus",
    "INVESTIGATION_TRANSITIONS",
    "IllegalTransition",
    "ImageRelationship",
    "ImageRole",
    "InvestigationStatus",
    "LifecycleAxis",
    "METHOD_STRENGTH",
    "MonitorCadence",
    "NodeType",
    "ObservationMethod",
    "ObservationOrigin",
    "PIPELINE_ORDER",
    "PROGRESS_TRANSITIONS",
    "PipelineStage",
    "ProgressState",
    "ProviderCapability",
    "REMOVAL_CONFIRMATIONS_REQUIRED",
    "REMOVAL_CONFIRMING_STATUSES",
    "RETENTION_TRANSITIONS",
    "ReportFormat",
    "RetentionState",
    "ReviewKind",
    "ReviewStatus",
    "RunStatus",
    "RunTrigger",
    "SearchObjectType",
    "StageStatus",
    "TargetType",
    "TimePrecision",
    "TimelineKind",
    "ValidationStatus",
    "VerificationResult",
    "VerificationState",
    "VisionCategory",
    "allowed_transitions",
    "assert_transition",
    "can_transition",
    "requires_reason",
]
