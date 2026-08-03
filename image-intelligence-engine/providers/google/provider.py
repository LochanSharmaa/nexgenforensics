"""Google Cloud Vision WEB_DETECTION.

A licensed, documented API with published terms. That is the point of using it
rather than crawling: someone else has already done the lawful-basis work for
the crawl, so this platform is a customer of an image index rather than the
controller of a scraped corpus.

`visuallySimilarImages` deserves care. Those are *not* face matches — they are
images a generic embedding finds alike, so a different person in a similar pose
against a similar background can score well. They are returned under their own
`kind` so an examiner never sees one presented beside an exact match without the
distinction being visible.
"""

from __future__ import annotations

import base64
import time
from typing import Any

import httpx

from providers.base import (
    KIND_FULL_IMAGE,
    KIND_PAGE,
    KIND_PARTIAL_IMAGE,
    KIND_SIMILAR_IMAGE,
    Appearance,
    DiscoveryResult,
    ProviderManifest,
)
from shared.enums import ProviderCapability

ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"

MANIFEST = ProviderManifest(
    name="google",
    title="Google Cloud Vision — web detection",
    capabilities=(ProviderCapability.IMAGE_DISCOVERY,),
    requires_credentials=True,
    config_keys=("IIE_GOOGLE_VISION_API_KEY",),
    cost_per_1k=3.50,
    notes="Enable the Vision API on the project, then restrict an API key to it.",
)


class GoogleVisionProvider:
    manifest = MANIFEST

    def __init__(self, api_key: str = "", *, timeout: float = 20.0) -> None:
        self.api_key = (api_key or "").strip()
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.api_key)

    async def discover(self, image: bytes, *, max_results: int = 50) -> DiscoveryResult:
        if not self.available():
            return DiscoveryResult(
                provider=self.manifest.name,
                available=False,
                error="IIE_GOOGLE_VISION_API_KEY is not set.",
            )

        started = time.perf_counter()
        payload = {
            "requests": [
                {
                    "image": {"content": base64.b64encode(image).decode("ascii")},
                    "features": [{"type": "WEB_DETECTION", "maxResults": max_results}],
                }
            ]
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    ENDPOINT, params={"key": self.api_key}, json=payload
                )
            status = response.status_code
            body = response.json()
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = ""
            try:
                detail = exc.response.json().get("error", {}).get("message", "")
            except Exception:  # noqa: BLE001
                detail = exc.response.text[:300]
            return DiscoveryResult(
                provider=self.manifest.name,
                available=True,
                error=f"HTTP {exc.response.status_code}: {detail}",
                http_status=exc.response.status_code,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
        except Exception as exc:  # noqa: BLE001 - a broken provider is a result, not a crash
            return DiscoveryResult(
                provider=self.manifest.name,
                available=True,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        first: dict[str, Any] = (body.get("responses") or [{}])[0]
        if "error" in first:
            return DiscoveryResult(
                provider=self.manifest.name,
                available=True,
                error=str(first["error"].get("message", "unknown API error")),
                http_status=status,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        web = first.get("webDetection") or {}
        found: list[Appearance] = []

        for page in web.get("pagesWithMatchingImages") or []:
            if page.get("url"):
                found.append(
                    Appearance(
                        url=page["url"],
                        kind=KIND_PAGE,
                        provider=self.manifest.name,
                        title=(page.get("pageTitle") or "")[:300],
                        score=page.get("score"),
                    )
                )

        for key, kind in (
            ("fullMatchingImages", KIND_FULL_IMAGE),
            ("partialMatchingImages", KIND_PARTIAL_IMAGE),
            ("visuallySimilarImages", KIND_SIMILAR_IMAGE),
        ):
            for item in web.get(key) or []:
                if item.get("url"):
                    found.append(
                        Appearance(
                            url=item["url"],
                            kind=kind,
                            provider=self.manifest.name,
                            score=item.get("score"),
                            image_url=item["url"],
                        )
                    )

        return DiscoveryResult(
            provider=self.manifest.name,
            available=True,
            appearances=tuple(found),
            entities=tuple(
                e["description"]
                for e in (web.get("webEntities") or [])
                if e.get("description")
            ),
            best_guess_labels=tuple(
                b["label"] for b in (web.get("bestGuessLabels") or []) if b.get("label")
            ),
            http_status=status,
            raw_response=body,
            cost_units=1.0,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )


def build(settings) -> GoogleVisionProvider:  # noqa: ANN001
    """Factory the registry calls. Keeps settings knowledge inside the plugin."""
    return GoogleVisionProvider(settings.google_vision_api_key)


__all__ = ["MANIFEST", "GoogleVisionProvider", "build"]
