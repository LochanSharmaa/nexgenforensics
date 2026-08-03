"""Vision provider selection.

One place that knows which vision backend is in use, so routers and services
never name a concrete provider. Swapping Gemini for another model is a change
here and nowhere else.
"""

from __future__ import annotations

from shared.config import Settings
from shared.logging import get_logger

from .base import VisionProvider

logger = get_logger(__name__)


def build_provider(settings: Settings) -> VisionProvider:
    """The configured vision provider.

    Only Gemini today. The import is local so adding a second backend does not
    mean importing every one of them at module load.
    """
    from .gemini.provider import build as build_gemini

    return build_gemini(settings)


def provider_status(settings: Settings) -> dict[str, object]:
    """Whether analysis can run, and what would enable it."""
    provider = build_provider(settings)
    return {
        "provider": provider.name,
        "model": provider.model,
        "configured": provider.available(),
        "config_keys": ["IIE_GEMINI_API_KEY"],
        "enabled": settings.vision_enabled,
    }


__all__ = ["build_provider", "provider_status"]
