"""SQLAlchemy models, grouped by bounded context.

Split across modules because the schema is large, re-exported here so callers
keep writing `from database.models import Page` regardless of which module a
table lives in. Importing this package registers every table on
`Base.metadata`, which is what Alembic autogenerate walks — a model that is not
reachable from here is invisible to migrations.
"""

from __future__ import annotations

from .annotations import (
    Bookmark,
    Checklist,
    ChecklistItem,
    Note,
    NoteAttachment,
    NoteLink,
)
from .evidence import (
    Entity,
    EntityAlias,
    Fact,
    FactEvidence,
    Mention,
    Observation,
    ReviewItem,
)
from .graph import GraphEdge, GraphNode, TimelineEvent
from .operations import (
    CopilotMessage,
    CopilotSession,
    Monitor,
    MonitorChange,
    MonitorRun,
    Report,
    SearchDocument,
)
from .sources import (
    Appearance,
    ContentCluster,
    ContentClusterMember,
    DiscoveryRequest,
    Domain,
    Image,
    ImageRelationshipRow,
    Page,
)
from .workspace import (
    AuditLogEntry,
    ConfigSnapshot,
    CustodyEvent,
    EvidenceLifecycleEvent,
    Investigation,
    InvestigationStatusEvent,
    PipelineRun,
    PipelineStageRow,
    RetentionHold,
    RetentionPolicy,
    User,
)

__all__ = [
    "Appearance",
    "AuditLogEntry",
    "Bookmark",
    "Checklist",
    "ChecklistItem",
    "ConfigSnapshot",
    "ContentCluster",
    "ContentClusterMember",
    "CopilotMessage",
    "CopilotSession",
    "CustodyEvent",
    "DiscoveryRequest",
    "Domain",
    "Entity",
    "EntityAlias",
    "EvidenceLifecycleEvent",
    "Fact",
    "FactEvidence",
    "GraphEdge",
    "GraphNode",
    "Image",
    "ImageRelationshipRow",
    "Investigation",
    "InvestigationStatusEvent",
    "Mention",
    "Monitor",
    "MonitorChange",
    "MonitorRun",
    "Note",
    "NoteAttachment",
    "NoteLink",
    "Observation",
    "Page",
    "PipelineRun",
    "PipelineStageRow",
    "Report",
    "RetentionHold",
    "RetentionPolicy",
    "ReviewItem",
    "SearchDocument",
    "TimelineEvent",
    "User",
]
