"""Provider plugin contract.

A provider answers **where does this image already appear on the web?** That is
image matching, not face matching. No provider is asked "who is this person",
none receives a face template, and none could produce one — the request carries
image bytes and the response carries URLs.

Capabilities rather than one interface
--------------------------------------
Wayback is an archive lookup, Reddit is a content source, and neither is image
discovery. Forcing every provider through a single protocol would produce
adapters that implement a third of it and raise `NotImplementedError` for the
rest. Each plugin therefore declares what it can do, and implements only that.

Unconfigured is not the same as empty
-------------------------------------
A provider with no credentials returns `available=False`, never an empty result
set. "No matches" and "no API key" lead an investigator to opposite next steps,
and conflating them puts a false negative into a report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from shared.enums import ProviderCapability

# How a hit relates to the probe, as the *provider* describes it. This is the
# provider's claim; local pHash verification (Phase 8) decides what to believe.
KIND_PAGE = "page"
KIND_FULL_IMAGE = "full_image"
KIND_PARTIAL_IMAGE = "partial_image"
KIND_SIMILAR_IMAGE = "similar_image"

MATCH_KINDS = (KIND_PAGE, KIND_FULL_IMAGE, KIND_PARTIAL_IMAGE, KIND_SIMILAR_IMAGE)


@dataclass(frozen=True)
class ProviderManifest:
    """What a plugin is and what it needs. Validated at registry load."""

    name: str
    title: str
    capabilities: tuple[ProviderCapability, ...]
    requires_credentials: bool = True
    config_keys: tuple[str, ...] = ()
    cost_per_1k: float | None = None
    """Rough unit cost, surfaced so an operator can see what a run will spend
    before starting it. None means free or unmetered."""
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "capabilities": [str(c) for c in self.capabilities],
            "requires_credentials": self.requires_credentials,
            "config_keys": list(self.config_keys),
            "cost_per_1k": self.cost_per_1k,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class Appearance:
    """One place a provider says the image, or a relative of it, was found."""

    url: str
    kind: str
    provider: str
    title: str = ""
    score: float | None = None
    reported_date: str = ""
    """First-seen or publication date *as the provider reports it*. TinEye
    supplies this and Google does not; establishing which appearance came first
    is often the whole question, so it is carried even though it is unverified."""
    image_url: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "kind": self.kind,
            "provider": self.provider,
            "title": self.title,
            "score": self.score,
            "reported_date": self.reported_date,
            "image_url": self.image_url,
        }


@dataclass(frozen=True)
class DiscoveryResult:
    """What one provider returned for one probe."""

    provider: str
    available: bool
    appearances: tuple[Appearance, ...] = field(default=())
    entities: tuple[str, ...] = field(default=())
    """Labels the provider guessed for the image — an organisation, an event.
    Investigative context only. These are guesses, never identifications, and
    the report says so."""
    best_guess_labels: tuple[str, ...] = field(default=())
    error: str = ""
    duration_ms: int = 0
    http_status: int | None = None
    raw_response: dict[str, Any] | None = None
    """Retained so a challenged classification can be re-checked against what
    the provider actually said, rather than against what we concluded."""
    cost_units: float | None = None

    @property
    def failed(self) -> bool:
        return self.available and bool(self.error)

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "available": self.available,
            "appearance_count": len(self.appearances),
            "entities": list(self.entities),
            "best_guess_labels": list(self.best_guess_labels),
            "error": self.error,
            "duration_ms": self.duration_ms,
            "http_status": self.http_status,
        }


@dataclass(frozen=True)
class ArchiveRecord:
    """What an archive knows about one URL.

    `first_seen` is the point of this whole capability. Establishing which
    appearance of a photograph came first is often the entire question in a
    provenance enquiry, and an archive snapshot is independent evidence of it —
    a page's own claimed publication date can be edited; a third-party capture
    on a date cannot.
    """

    url: str
    available: bool
    archived_url: str = ""
    first_seen: str = ""
    last_seen: str = ""
    snapshot_count: int = 0
    error: str = ""
    duration_ms: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "available": self.available,
            "archived_url": self.archived_url,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "snapshot_count": self.snapshot_count,
            "error": self.error,
        }


@runtime_checkable
class ArchiveLookupProvider(Protocol):
    """Implemented by plugins declaring `ARCHIVE_LOOKUP`.

    A different shape from image discovery on purpose: this takes a URL, not
    image bytes. Forcing both through one interface would have produced adapters
    implementing half a contract.
    """

    manifest: ProviderManifest

    def available(self) -> bool: ...

    async def lookup(self, url: str) -> ArchiveRecord: ...


@runtime_checkable
class ImageDiscoveryProvider(Protocol):
    """Implemented by plugins declaring `IMAGE_DISCOVERY`."""

    manifest: ProviderManifest

    def available(self) -> bool:
        """Whether this provider has what it needs to run."""
        ...

    async def discover(
        self, image: bytes, *, max_results: int = 50
    ) -> DiscoveryResult: ...


__all__ = [
    "KIND_FULL_IMAGE",
    "ArchiveLookupProvider",
    "ArchiveRecord",
    "KIND_PAGE",
    "KIND_PARTIAL_IMAGE",
    "KIND_SIMILAR_IMAGE",
    "MATCH_KINDS",
    "Appearance",
    "DiscoveryResult",
    "ImageDiscoveryProvider",
    "ProviderManifest",
]
