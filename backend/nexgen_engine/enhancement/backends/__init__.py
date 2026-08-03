"""Backend implementations. Importing this package registers every backend.

Import order is deliberate: classical backends register first so that they are
always present as the fallback for a task, even on a host where no weights have
been downloaded and torch is absent. A demo on a fresh machine must still work.

Learned backends import defensively -- a missing weight file or a missing torch
makes a backend *unavailable*, never a broken import that takes the API down.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

from . import classical  # noqa: F401,E402

for _module in ("nafnet", "realesrgan", "facerestore"):
    try:
        __import__(f"{__name__}.{_module}")
    except Exception as exc:  # pragma: no cover - depends on host packages
        logger.warning("Enhancement backend module %s did not load: %s", _module, exc)

__all__ = ["classical"]
