"""TinEye reverse image search.

Run alongside Google rather than instead of it, for one specific reason:
TinEye reports **first-seen dates** on its backlinks and Google's web detection
does not. Establishing which appearance of a photograph came first is often the
entire question in a provenance enquiry, and no amount of extra page coverage
substitutes for a date.

Auth shape varies by plan. This implements the `x-api-key` header form used by
the current TinEye API; older sandbox accounts sign requests with an HMAC over
public/private keys. If yours is the latter, the signing goes in `_headers` and
nothing else here changes.
"""

from __future__ import annotations

import time

import httpx

from providers.base import (
    KIND_FULL_IMAGE,
    KIND_PAGE,
    Appearance,
    DiscoveryResult,
    ProviderManifest,
)
from shared.enums import ProviderCapability

MANIFEST = ProviderManifest(
    name="tineye",
    title="TinEye — reverse image search with first-seen dates",
    capabilities=(ProviderCapability.IMAGE_DISCOVERY,),
    requires_credentials=True,
    config_keys=("IIE_TINEYE_API_KEY", "IIE_TINEYE_API_BASE"),
    cost_per_1k=None,
    notes="Sold in search bundles. The only source of first-seen dates.",
)


class TinEyeProvider:
    manifest = MANIFEST

    def __init__(
        self,
        api_key: str = "",
        *,
        api_base: str = "https://api.tineye.com/rest",
        timeout: float = 30.0,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self.api_key}

    async def discover(self, image: bytes, *, max_results: int = 50) -> DiscoveryResult:
        if not self.available():
            return DiscoveryResult(
                provider=self.manifest.name,
                available=False,
                error="IIE_TINEYE_API_KEY is not set.",
            )

        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.api_base}/search/",
                    headers=self._headers(),
                    files={"image_upload": ("probe", image, "application/octet-stream")},
                    data={"limit": str(max_results)},
                )
            status = response.status_code
            body = response.json()
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            return DiscoveryResult(
                provider=self.manifest.name,
                available=True,
                error=f"HTTP {exc.response.status_code}: {exc.response.text[:300]}",
                http_status=exc.response.status_code,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
        except Exception as exc:  # noqa: BLE001
            return DiscoveryResult(
                provider=self.manifest.name,
                available=True,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        if body.get("code") not in (200, None):
            messages = body.get("messages") or ["unknown TinEye error"]
            return DiscoveryResult(
                provider=self.manifest.name,
                available=True,
                error="; ".join(str(m) for m in messages)[:300],
                http_status=status,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        found: list[Appearance] = []
        for match in (body.get("results") or {}).get("matches") or []:
            if match.get("image_url"):
                found.append(
                    Appearance(
                        url=match["image_url"],
                        kind=KIND_FULL_IMAGE,
                        provider=self.manifest.name,
                        score=match.get("score"),
                        image_url=match["image_url"],
                    )
                )
            for backlink in match.get("backlinks") or []:
                page_url = backlink.get("backlink") or backlink.get("url")
                if not page_url:
                    continue
                found.append(
                    Appearance(
                        url=page_url,
                        kind=KIND_PAGE,
                        provider=self.manifest.name,
                        score=match.get("score"),
                        # The reason this provider is worth running at all.
                        reported_date=str(backlink.get("crawl_date") or ""),
                    )
                )

        return DiscoveryResult(
            provider=self.manifest.name,
            available=True,
            appearances=tuple(found[:max_results]),
            http_status=status,
            raw_response=body,
            cost_units=1.0,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )


def build(settings) -> TinEyeProvider:  # noqa: ANN001
    return TinEyeProvider(settings.tineye_api_key, api_base=settings.tineye_api_base)


__all__ = ["MANIFEST", "TinEyeProvider", "build"]
