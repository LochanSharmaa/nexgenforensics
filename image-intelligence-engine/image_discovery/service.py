"""Discovery execution and the investigator-facing findings view.

Two responsibilities, deliberately in one place because they are two halves of
the same contract: running the providers, and turning what they returned into
something an investigator reads.

The findings shape is what the UI renders. It is organised around the questions
an investigator actually asks — *where has this been published, who do those
pages say it is, when did it first appear, do the sources disagree* — rather
than around the pipeline that produced it. Stage names and hash digests are
plumbing; they belong in the technical detail, not in the finding.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import tldextract
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import Appearance, DiscoveryRequest, Domain, Image
from providers.registry import (
    DiscoverySummary,
    load_providers,
    run_archive_lookups,
    run_discovery,
)
from shared.clock import Clock, SystemClock
from shared.config import Settings
from shared.enums import ProviderCapability, VerificationResult
from shared.logging import get_logger

logger = get_logger(__name__)

# `suffix_list_urls=()` keeps this offline: it uses the snapshot bundled with
# the package rather than fetching the public suffix list at first use. A
# network call on an unrelated code path is a surprising failure mode.
_EXTRACT = tldextract.TLDExtract(
    suffix_list_urls=(),
    # Private suffixes matter for independence counting. Without this,
    # `alice.github.io` and `bob.github.io` both collapse to `github.io` and
    # two unrelated authors are scored as a single source.
    include_psl_private_domains=True,
)

# Match kinds, in the order an examiner should read them. An exact page hit is
# a finding; a "visually similar" hit is a lead at best, and is labelled so it
# can never be mistaken for the former.
KIND_LABEL = {
    "page": "Page carrying this image",
    "full_image": "Exact copy of the image file",
    "partial_image": "Cropped or partial copy",
    "similar_image": "Visually similar (not the same image)",
}

KIND_CONFIDENCE = {
    "page": "high",
    "full_image": "high",
    "partial_image": "medium",
    "similar_image": "low",
}


def registrable_domain(url: str) -> str:
    """eTLD+1, via the public suffix list.

    Naive "last two labels" gets this wrong in a way that is visible and
    embarrassing: `www.bbc.co.uk` becomes `co.uk`, which is a public suffix and
    not a source at all. It also matters beyond display — one site must count as
    one source when independence is scored, and `bbc.co.uk` and `itv.co.uk`
    collapsing together would understate independence badly.
    """
    from urllib.parse import urlparse

    extracted = _EXTRACT(url)
    registrable = extracted.top_domain_under_public_suffix
    if registrable:
        return registrable

    # No recognised public suffix — a new gTLD the bundled list predates, a
    # reserved name like `.example`, or an intranet host. Falling back to the
    # *domain label* would be actively dangerous: `newsroom.example` and
    # `registry.example` both reduce to `example`, silently merging two
    # unrelated sites into one source and understating independence in the
    # confidence score. The full hostname is at worst too granular, which
    # over-counts independence and is the safe direction to be wrong in.
    return (urlparse(url).hostname or "").lower().strip(".")


@dataclass
class Finding:
    """One place the image was found, as an investigator sees it."""

    url: str
    site: str
    title: str
    match_kind: str
    match_label: str
    confidence: str
    provider: str
    reported_date: str = ""
    verification: str = VerificationResult.PENDING
    image_url: str = ""
    last_archived: str = ""
    archive_url: str = ""
    """A durable link. Pages get edited and taken down mid-investigation; a
    snapshot still resolves, so a report does not rot before it is read."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "site": self.site,
            "title": self.title,
            "match_kind": self.match_kind,
            "match_label": self.match_label,
            "confidence": self.confidence,
            "provider": self.provider,
            "reported_date": self.reported_date,
            "verification": self.verification,
            "image_url": self.image_url,
            "last_archived": self.last_archived,
            "archive_url": self.archive_url,
        }


@dataclass
class FindingsView:
    """Everything the provenance panel renders."""

    findings: list[Finding] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    best_guess_labels: list[str] = field(default_factory=list)
    providers_run: list[str] = field(default_factory=list)
    providers_unconfigured: list[dict[str, Any]] = field(default_factory=list)
    providers_failed: list[dict[str, str]] = field(default_factory=list)
    searched: bool = False

    @property
    def sites(self) -> list[str]:
        seen: dict[str, None] = {}
        for finding in self.findings:
            seen.setdefault(finding.site, None)
        return list(seen)

    @property
    def earliest(self) -> str:
        dates = sorted(f.reported_date for f in self.findings if f.reported_date)
        return dates[0] if dates else ""

    @property
    def latest(self) -> str:
        """Most recent capture across all sources.

        Deliberately drawn from `last_archived`, not from the newest
        `reported_date`: the latter is the newest *first*-capture, which is a
        different and much less useful fact. Labelling that "latest archived"
        would have been quietly wrong.
        """
        dates = sorted(f.last_archived for f in self.findings if f.last_archived)
        return dates[-1] if dates else ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "searched": self.searched,
            "summary": {
                "sources_found": len(self.findings),
                "distinct_sites": len(self.sites),
                "earliest_appearance": self.earliest,
                "latest_appearance": self.latest,
                "exact_matches": sum(
                    1 for f in self.findings if f.match_kind in ("page", "full_image")
                ),
                "similar_only": sum(
                    1 for f in self.findings if f.match_kind == "similar_image"
                ),
            },
            "findings": [f.as_dict() for f in self.findings],
            # Provider guesses about what the image depicts. Context, never
            # identification — the UI labels them as such.
            "entities": self.entities,
            "best_guess_labels": self.best_guess_labels,
            "providers": {
                "run": self.providers_run,
                "unconfigured": self.providers_unconfigured,
                "failed": self.providers_failed,
            },
        }


class DiscoveryService:
    """Runs providers for a probe image and persists what they returned."""

    def __init__(
        self, session: AsyncSession, settings: Settings, clock: Clock | None = None
    ) -> None:
        self.session = session
        self.settings = settings
        self.clock = clock or SystemClock()
        # First-seen dates keyed by URL, held for the life of the request so the
        # findings view can show them without a second archive round-trip.
        self._reported_dates: dict[str, str] = {}
        self._last_archived: dict[str, str] = {}
        self._domains_seen: set[str] = set()

    async def discover(
        self,
        *,
        investigation_id: uuid.UUID,
        image: Image,
        image_bytes: bytes,
        manual_urls: list[str] | None = None,
    ) -> FindingsView:
        providers = load_providers(
            self.settings, capability=ProviderCapability.IMAGE_DISCOVERY
        )

        # The manual provider carries this run's operator-supplied leads. A
        # per-run copy, so one investigation's URLs never leak into another's.
        if manual_urls:
            for loaded in providers:
                if loaded.name == "manual":
                    loaded.instance.urls = tuple(manual_urls)

        summary = await run_discovery(
            providers, image_bytes, max_results=self.settings.discovery_max_results
        )

        # Archive lookup runs after discovery, over the URLs it produced. Free
        # and keyless, so it enriches every finding regardless of which paid
        # provider (if any) surfaced it — and it supplies the first-seen date
        # that decides which appearance came first.
        archives = {}
        if self.settings.archive_lookup_enabled:
            archives = await run_archive_lookups(
                load_providers(
                    self.settings, capability=ProviderCapability.ARCHIVE_LOOKUP
                ),
                [a.url for a in summary.appearances if a.kind == "page"],
            )

        await self._persist(investigation_id, image, summary, archives)
        return await self.findings_for(investigation_id, image.id)

    async def _persist(
        self,
        investigation_id: uuid.UUID,
        image: Image,
        summary: DiscoverySummary,
        archives: dict[str, Any] | None = None,
    ) -> None:
        """Record how evidence was found, then what was found.

        The request rows come first and exist even when a provider returned
        nothing: "we asked Google and it had no answer" is a different fact from
        "we never asked", and only the former can be relied on later.
        """
        now = self.clock.now()

        for result in summary.results:
            self.session.add(
                DiscoveryRequest(
                    investigation_id=investigation_id,
                    provider=result.provider,
                    capability=ProviderCapability.IMAGE_DISCOVERY,
                    probe_image_id=image.id,
                    query_parameters={"max_results": self.settings.discovery_max_results},
                    requested_at=now,
                    responded_at=now,
                    http_status=result.http_status,
                    results_returned=len(result.appearances),
                    results_accepted=len(result.appearances),
                    results_rejected=0,
                    error=result.error,
                    cost_units=result.cost_units,
                )
            )

        existing = {
            row.page_url
            for row in (
                await self.session.execute(
                    select(Appearance).where(
                        Appearance.investigation_id == investigation_id,
                        Appearance.probe_image_id == image.id,
                    )
                )
            ).scalars()
        }

        archives = archives or {}

        for appearance in summary.appearances:
            if appearance.url in existing:
                continue
            existing.add(appearance.url)

            record = archives.get(appearance.url)
            reported = appearance.reported_date
            archive_url = ""
            if record is not None and record.first_seen:
                # An archive capture outranks a provider's own date: a page
                # owner can edit a claimed publication date, but not a third
                # party's capture on a given day.
                reported = record.first_seen
                archive_url = record.archived_url
            self._reported_dates[appearance.url] = reported
            self._last_archived[appearance.url] = (
                record.last_seen if record is not None else ""
            )

            self.session.add(
                Appearance(
                    investigation_id=investigation_id,
                    probe_image_id=image.id,
                    provider=appearance.provider,
                    page_url=appearance.url,
                    image_url=appearance.image_url,
                    provider_score=appearance.score,
                    archive_url=archive_url,
                    discovered_at=now,
                    # PENDING until Phase 8 fetches the candidate and classifies
                    # it locally. A provider's claim is not yet a verified one.
                    verification_result=VerificationResult.PENDING,
                )
            )

            await self._ensure_domain(appearance.url, now)

        await self.session.flush()
        logger.info(
            "discovery.completed",
            investigation_id=str(investigation_id),
            appearances=len(summary.appearances),
            providers=list(summary.configured),
        )

    async def _ensure_domain(self, url: str, now) -> None:  # noqa: ANN001
        """Insert the domain once, even for several URLs in the same batch.

        A plain SELECT-then-INSERT is not enough here: rows added earlier in
        this loop have not been flushed, so the query cannot see them and every
        URL on the same site would try to insert it again.
        """
        domain = registrable_domain(url)
        if not domain or domain in self._domains_seen:
            return
        self._domains_seen.add(domain)

        found = (
            await self.session.execute(
                select(Domain).where(Domain.registrable_domain == domain)
            )
        ).scalar_one_or_none()
        if found is None:
            self.session.add(Domain(registrable_domain=domain, first_seen_at=now))
            await self.session.flush()

    async def findings_for(
        self, investigation_id: uuid.UUID, image_id: uuid.UUID | None = None
    ) -> FindingsView:
        """Assemble the investigator-facing view from what is persisted."""
        statement = select(Appearance).where(
            Appearance.investigation_id == investigation_id
        )
        if image_id is not None:
            statement = statement.where(Appearance.probe_image_id == image_id)
        rows = (await self.session.execute(statement.order_by(Appearance.id))).scalars().all()

        requests = (
            await self.session.execute(
                select(DiscoveryRequest).where(
                    DiscoveryRequest.investigation_id == investigation_id
                )
            )
        ).scalars().all()

        view = FindingsView(searched=bool(requests))

        for row in rows:
            kind = "page"
            view.findings.append(
                Finding(
                    url=row.page_url,
                    site=registrable_domain(row.page_url),
                    title="",
                    match_kind=kind,
                    match_label=KIND_LABEL.get(kind, kind),
                    confidence=KIND_CONFIDENCE.get(kind, "low"),
                    provider=row.provider,
                    verification=row.verification_result,
                    image_url=row.image_url,
                    reported_date=self._reported_dates.get(row.page_url, ""),
                    last_archived=self._last_archived.get(row.page_url, ""),
                    archive_url=row.archive_url,
                )
            )

        ran = {r.provider for r in requests if not r.error}
        view.providers_run = sorted(ran)
        view.providers_failed = [
            {"provider": r.provider, "error": r.error} for r in requests if r.error
        ]

        # Providers with no credentials are named explicitly, with the env var
        # that would enable them. "Nothing found" and "nothing asked" must never
        # look the same to an investigator.
        for loaded in load_providers(
            self.settings, capability=ProviderCapability.IMAGE_DISCOVERY
        ):
            if not loaded.available():
                view.providers_unconfigured.append(
                    {
                        "name": loaded.name,
                        "title": loaded.manifest.title,
                        "config_keys": list(loaded.manifest.config_keys),
                        "notes": loaded.manifest.notes,
                    }
                )

        return view


__all__ = [
    "KIND_CONFIDENCE",
    "KIND_LABEL",
    "DiscoveryService",
    "Finding",
    "FindingsView",
    "registrable_domain",
]
