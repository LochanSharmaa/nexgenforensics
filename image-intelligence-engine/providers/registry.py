"""Plugin host: discovery, validation, and running providers.

Plugins are found by scanning this package for sub-packages exposing a
`provider` module with a `MANIFEST` and a `build(settings)` factory. Adding a
provider therefore means adding one directory — nothing outside it changes, and
an architecture test enforces that no module imports a concrete provider by
name.
"""

from __future__ import annotations

import asyncio
import importlib
import pkgutil
from dataclasses import dataclass
from typing import Any

from shared.enums import ProviderCapability
from shared.logging import get_logger

from .base import Appearance, ArchiveRecord, DiscoveryResult, ProviderManifest

logger = get_logger(__name__)

# Exact matches first, then partials, then the merely similar. This is the
# confidence order and an examiner reads top-down; burying an exact match under
# forty "visually similar" hits would hide the finding that matters.
_KIND_PRIORITY = {"page": 0, "full_image": 1, "partial_image": 2, "similar_image": 3}


@dataclass(frozen=True)
class LoadedProvider:
    manifest: ProviderManifest
    instance: Any

    @property
    def name(self) -> str:
        return self.manifest.name

    def available(self) -> bool:
        return bool(self.instance.available())


def _discover_plugins() -> dict[str, Any]:
    """Import every sub-package exposing a `provider` module."""
    modules: dict[str, Any] = {}
    package = importlib.import_module(__package__)

    for info in pkgutil.iter_modules(package.__path__):
        if not info.ispkg or info.name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"{__package__}.{info.name}.provider")
        except ImportError as exc:
            # A broken plugin must not stop the others loading. It is logged
            # loudly rather than swallowed, because a silently missing provider
            # looks identical to one that found nothing.
            logger.error("provider.load_failed", provider=info.name, error=str(exc))
            continue

        manifest = getattr(module, "MANIFEST", None)
        builder = getattr(module, "build", None)
        if manifest is None or builder is None:
            logger.error("provider.invalid", provider=info.name,
                         reason="missing MANIFEST or build()")
            continue
        modules[manifest.name] = module

    return modules


def load_providers(
    settings, *, capability: ProviderCapability | None = None
) -> list[LoadedProvider]:  # noqa: ANN001
    """Instantiate every plugin, optionally filtered by capability."""
    loaded: list[LoadedProvider] = []

    for name, module in sorted(_discover_plugins().items()):
        manifest: ProviderManifest = module.MANIFEST
        if capability is not None and capability not in manifest.capabilities:
            continue
        try:
            loaded.append(LoadedProvider(manifest=manifest, instance=module.build(settings)))
        except Exception as exc:  # noqa: BLE001
            logger.error("provider.build_failed", provider=name, error=str(exc))

    return loaded


@dataclass(frozen=True)
class DiscoverySummary:
    """Merged output across every provider that ran."""

    results: tuple[DiscoveryResult, ...]
    appearances: tuple[Appearance, ...]
    entities: tuple[str, ...]
    best_guess_labels: tuple[str, ...]

    @property
    def configured(self) -> tuple[str, ...]:
        return tuple(r.provider for r in self.results if r.available)

    @property
    def unconfigured(self) -> tuple[str, ...]:
        return tuple(r.provider for r in self.results if not r.available)

    @property
    def failed(self) -> tuple[str, ...]:
        return tuple(r.provider for r in self.results if r.failed)

    def page_urls(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for appearance in self.appearances:
            if appearance.kind == "page":
                seen.setdefault(appearance.url, None)
        return tuple(seen)

    def as_dict(self) -> dict[str, Any]:
        return {
            "providers": [r.as_dict() for r in self.results],
            "configured": list(self.configured),
            "unconfigured": list(self.unconfigured),
            "failed": list(self.failed),
            "appearance_count": len(self.appearances),
            "page_count": len(self.page_urls()),
            "entities": list(self.entities),
            "best_guess_labels": list(self.best_guess_labels),
        }


async def run_discovery(
    providers: list[LoadedProvider], image: bytes, *, max_results: int = 50
) -> DiscoverySummary:
    """Query every provider concurrently and merge the results.

    Concurrent because these are independent network calls and a serial run
    would take as long as their sum. One provider failing never suppresses
    another — a failure is recorded as that provider's result.
    """

    async def _one(loaded: LoadedProvider) -> DiscoveryResult:
        try:
            return await loaded.instance.discover(image, max_results=max_results)
        except Exception as exc:  # noqa: BLE001
            return DiscoveryResult(
                provider=loaded.name, available=True,
                error=f"{type(exc).__name__}: {exc}",
            )

    results = await asyncio.gather(*(_one(p) for p in providers)) if providers else []

    seen: set[tuple[str, str]] = set()
    merged: list[Appearance] = []
    entities: list[str] = []
    labels: list[str] = []

    for result in results:
        for appearance in result.appearances:
            # (url, kind) rather than url alone: a page hit and an image hit at
            # the same URL are two different facts about it.
            key = (appearance.url, appearance.kind)
            if key not in seen:
                seen.add(key)
                merged.append(appearance)
        for entity in result.entities:
            if entity not in entities:
                entities.append(entity)
        for label in result.best_guess_labels:
            if label not in labels:
                labels.append(label)

    merged.sort(key=lambda a: (_KIND_PRIORITY.get(a.kind, 9), -(a.score or 0.0)))

    return DiscoverySummary(
        results=tuple(results),
        appearances=tuple(merged),
        entities=tuple(entities),
        best_guess_labels=tuple(labels),
    )


async def run_archive_lookups(
    providers: list[LoadedProvider], urls: list[str], *, limit: int = 25
) -> dict[str, ArchiveRecord]:
    """Date and preserve a set of URLs.

    Capped, and one lookup per URL rather than per provider-URL pair: the
    archive is a free courtesy service with no paid tier, and hammering it on a
    large result set would be both rude and self-defeating.

    A failure for one URL never affects another — that result carries the error
    and the rest proceed.
    """
    if not providers or not urls:
        return {}

    provider = providers[0].instance
    targets = urls[:limit]

    async def _one(url: str) -> tuple[str, ArchiveRecord]:
        try:
            return url, await provider.lookup(url)
        except Exception as exc:  # noqa: BLE001
            return url, ArchiveRecord(
                url=url, available=True, error=f"{type(exc).__name__}: {exc}"
            )

    return dict(await asyncio.gather(*(_one(u) for u in targets)))


def manifests(settings) -> list[dict[str, Any]]:  # noqa: ANN001
    """Every plugin with its configuration state, for `GET /providers`."""
    return [
        {
            **loaded.manifest.as_dict(),
            "configured": loaded.available(),
            "status": "configured" if loaded.available() else "unconfigured",
        }
        for loaded in load_providers(settings)
    ]


__all__ = [
    "DiscoverySummary",
    "run_archive_lookups",
    "LoadedProvider",
    "load_providers",
    "manifests",
    "run_discovery",
]
