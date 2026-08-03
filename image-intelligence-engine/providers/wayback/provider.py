"""Internet Archive (Wayback Machine) — archive lookup.

**Free, and needs no key or account.** That makes it the one provider besides
`manual` that works the moment the service starts.

It cannot answer "where does this image appear" — it takes a URL, not image
bytes. What it does answer is the question TinEye is otherwise the only source
for: **when was this page first captured?** That matters more than it sounds. A
page's own claimed publication date is editable by whoever owns the page; a
third-party capture on a date is not. So an archive snapshot is *independent*
evidence of when something existed, which is exactly what a provenance enquiry
turns on.

It also gives every finding a durable link. Pages get edited and taken down
mid-investigation; a Wayback URL still resolves, so a report written today does
not rot by the time it is read.

Uses the CDX server, which supports the two queries that matter:
`limit=1` for the earliest capture and `limit=-1` for the most recent.
"""

from __future__ import annotations

import time
from datetime import datetime

import httpx

from providers.base import ArchiveRecord, ProviderManifest
from shared.enums import ProviderCapability
from shared.logging import get_logger

logger = get_logger(__name__)

CDX_ENDPOINT = "https://web.archive.org/cdx/search/cdx"
SNAPSHOT_BASE = "https://web.archive.org/web"

MANIFEST = ProviderManifest(
    name="wayback",
    title="Internet Archive — first-seen dates and snapshots",
    capabilities=(ProviderCapability.ARCHIVE_LOOKUP,),
    requires_credentials=False,
    config_keys=(),
    cost_per_1k=None,
    notes=(
        "Free, no key required. Cannot search by image — it dates and preserves "
        "URLs that discovery has already found."
    ),
)


def _readable(timestamp: str) -> str:
    """CDX timestamps are `YYYYMMDDhhmmss`. Rendered as a date only.

    The capture *time* is an artefact of when the crawler happened to pass, not
    a fact about the page, so showing it to the second would imply a precision
    that does not exist.
    """
    try:
        return datetime.strptime(timestamp[:8], "%Y%m%d").date().isoformat()
    except (ValueError, TypeError):
        return timestamp[:8]


class WaybackProvider:
    manifest = MANIFEST

    def __init__(self, *, timeout: float = 20.0) -> None:
        self.timeout = timeout

    def available(self) -> bool:
        # No credentials to check. Reachability is not asserted here — a network
        # failure surfaces per-lookup as that lookup's error, rather than
        # disabling the provider globally on one bad request.
        return True

    async def _cdx(self, client: httpx.AsyncClient, url: str, limit: str) -> list[list[str]]:
        response = await client.get(
            CDX_ENDPOINT,
            params={
                "url": url,
                "output": "json",
                "limit": limit,
                "fl": "timestamp,original,statuscode",
                # Only successful captures. A snapshot of a 404 proves the URL
                # was crawled, not that the page existed.
                "filter": "statuscode:200",
            },
            follow_redirects=True,
        )
        response.raise_for_status()
        rows = response.json()
        # Row 0 is the header when there are any results at all.
        return rows[1:] if len(rows) > 1 else []

    async def lookup(self, url: str) -> ArchiveRecord:
        """Earliest capture, plus the latest if it comes cheaply.

        The two queries are deliberately independent. `limit=-1` makes the
        archive scan the whole index for that URL, which times out on a busy
        site like a news homepage — and losing the *earliest* date because the
        *latest* was slow would throw away the datum that actually matters.
        Earliest is fetched first and kept; latest is best-effort.
        """
        started = time.perf_counter()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                earliest = await self._cdx(client, url, "1")
        except Exception as exc:  # noqa: BLE001 - a failed lookup is a result
            return ArchiveRecord(
                url=url,
                available=True,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        latest: list[list[str]] = []
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                latest = await self._cdx(client, url, "-1")
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "wayback.latest_unavailable", url=url, error=type(exc).__name__
            )

        if not earliest:
            # Genuinely informative: an unarchived page is a page nobody has
            # preserved, which is itself worth an investigator knowing.
            return ArchiveRecord(
                url=url,
                available=True,
                snapshot_count=0,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        first_stamp = earliest[0][0]
        last_stamp = latest[0][0] if latest else ""

        return ArchiveRecord(
            url=url,
            available=True,
            archived_url=f"{SNAPSHOT_BASE}/{first_stamp}/{url}",
            first_seen=_readable(first_stamp),
            # Absent rather than wrong when the expensive query timed out.
            # Falling back to the first capture would present the *oldest*
            # snapshot as the most recent one — a misdated finding is worse
            # than a missing field.
            last_seen=_readable(last_stamp) if last_stamp else "",
            # CDX does not return a total without paging the whole result set,
            # which is expensive for a busy URL. Two known captures is reported
            # honestly rather than guessed at.
            snapshot_count=2 if last_stamp and last_stamp != first_stamp else 1,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )


def build(settings) -> WaybackProvider:  # noqa: ANN001
    return WaybackProvider(timeout=settings.archive_lookup_timeout_seconds)


__all__ = ["MANIFEST", "WaybackProvider", "build"]
