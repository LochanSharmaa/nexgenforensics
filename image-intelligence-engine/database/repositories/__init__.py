"""Repository layer — the only place that writes.

Split by bounded context, re-exported here so callers keep writing
`from database.repositories import FactRepository`.

Repositories exist to enforce two things a caller cannot bypass by constructing
a model and adding it to a session:

* **State changes route through the domain transition tables.** Nothing assigns
  `status` or a lifecycle state by direct attribute set.
* **Chained logs serialise their appends.** Audit and custody chains read the
  current tail, hash against it, and insert; a race between two concurrent
  appends would fork the chain.

They also duplicate, at the application layer, the invariants the database
enforces at the storage layer. That duplication is intentional: the constraint
is the guarantee, the repository is the good error message.
"""

from __future__ import annotations

from .evidence import (
    EntityRepository,
    FactRepository,
    GraphRepository,
    ImageRepository,
    ObservationRepository,
    ReviewRepository,
    SourceRepository,
)
from .pipeline import PipelineRepository, RetentionRepository
from .workspace import (
    AuditRepository,
    CustodyRepository,
    InvestigationRepository,
    LifecycleRepository,
    UserRepository,
)

__all__ = [
    "AuditRepository",
    "CustodyRepository",
    "EntityRepository",
    "FactRepository",
    "GraphRepository",
    "ImageRepository",
    "InvestigationRepository",
    "LifecycleRepository",
    "ObservationRepository",
    "PipelineRepository",
    "RetentionRepository",
    "ReviewRepository",
    "SourceRepository",
    "UserRepository",
]
