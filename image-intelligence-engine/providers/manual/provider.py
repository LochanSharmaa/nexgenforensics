"""Investigator-supplied URLs.

Needs no credentials, which makes it the provider that works on day one — and
it is not a fallback. An investigator who already holds a lead ("this photo is
on these three pages") is doing targeted corroboration, which is both the most
defensible use of the platform and often the most productive.

It performs no search of its own; the URLs come from the operator. It exists so
operator-supplied leads flow through exactly the same verification, crawling and
evidence machinery as machine-discovered ones. A finding from a hunch and a
finding from an API must be equally traceable, or the cheap one quietly becomes
the untrustworthy one.
"""

from __future__ import annotations

from urllib.parse import urlparse

from providers.base import KIND_PAGE, Appearance, DiscoveryResult, ProviderManifest
from shared.enums import ProviderCapability

MANIFEST = ProviderManifest(
    name="manual",
    title="Investigator-supplied URLs",
    capabilities=(ProviderCapability.IMAGE_DISCOVERY,),
    requires_credentials=False,
    config_keys=(),
    cost_per_1k=None,
    notes="Always available. Targeted corroboration of leads the operator already holds.",
)


class ManualProvider:
    manifest = MANIFEST

    def __init__(self, urls: tuple[str, ...] = ()) -> None:
        self.urls = urls

    def available(self) -> bool:
        return True

    def with_urls(self, urls: list[str]) -> ManualProvider:
        """A per-run copy carrying this request's URLs.

        Returns a new instance rather than mutating: the registry caches
        providers, and one investigation's leads must never leak into another's.
        """
        return ManualProvider(tuple(urls))

    async def discover(self, image: bytes, *, max_results: int = 50) -> DiscoveryResult:
        found: list[Appearance] = []
        rejected: list[str] = []

        for raw in self.urls[:max_results]:
            candidate = raw.strip()
            parsed = urlparse(candidate)
            # Rejected here rather than at fetch time, so the operator learns
            # which of their URLs was unusable before the run starts instead of
            # finding a silent gap in the results afterwards.
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                rejected.append(candidate)
                continue
            found.append(
                Appearance(url=candidate, kind=KIND_PAGE, provider=self.manifest.name)
            )

        error = ""
        if rejected:
            error = (
                f"{len(rejected)} URL(s) ignored — only http and https are accepted: "
                + ", ".join(rejected[:3])
            )

        return DiscoveryResult(
            provider=self.manifest.name,
            available=True,
            appearances=tuple(found),
            error=error,
        )


def build(settings) -> ManualProvider:  # noqa: ANN001
    return ManualProvider()


__all__ = ["MANIFEST", "ManualProvider", "build"]
