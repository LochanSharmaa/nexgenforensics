"""Content-addressed cache for enhancement output.

Enhancement costs seconds per image; embedding costs milliseconds. Running the
S0.3 arms over 3,000 pairs across several backends without a cache means hours
of redundant GPU work every time a metric is re-computed or a run is resumed.
So the cache is a correctness-of-workflow feature, not an optimisation.

The key is ``sha256(original_digest + plan_cache_key)``:

  * the original's content address, so different images never collide;
  * the plan's canonical digest, so changing a parameter, reordering stages, or
    swapping a backend produces a different key rather than silently reusing
    output from a pipeline that no longer exists.

Storage is **PNG, always**. Writing the enhanced image as JPEG would put a fresh
compression operator on top of the one the pipeline just spent effort removing,
and would make the cached result differ from the computed one -- which would in
turn break the determinism test in a way that looks like a model bug.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def cache_key(original_digest: str, plan_key: str) -> str:
    return hashlib.sha256(f"{original_digest}:{plan_key}".encode()).hexdigest()


@dataclass(frozen=True)
class CacheHit:
    pixels: np.ndarray
    metadata: dict[str, Any]
    key: str


class EnhancementCache:
    """Two-level fan-out on disk. One PNG plus one JSON sidecar per entry."""

    def __init__(self, root: Path | str, enabled: bool = True) -> None:
        self.root = Path(root)
        self.enabled = enabled
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ io --

    def _paths(self, key: str) -> tuple[Path, Path]:
        directory = self.root / key[:2] / key[2:4]
        return directory / f"{key}.png", directory / f"{key}.json"

    def get(self, original_digest: str, plan_key: str) -> CacheHit | None:
        if not self.enabled:
            return None
        key = cache_key(original_digest, plan_key)
        image_path, meta_path = self._paths(key)
        if not image_path.is_file() or not meta_path.is_file():
            return None
        try:
            with Image.open(image_path) as handle:
                handle.load()
                pixels = np.asarray(handle.convert("RGB"), dtype=np.uint8)
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            # A truncated entry -- interrupted write, full disk -- must not be a
            # hard failure. Drop it and recompute.
            logger.warning("Discarding unreadable cache entry %s: %s", key, exc)
            self._discard(key)
            return None
        return CacheHit(pixels=pixels, metadata=metadata, key=key)

    def put(
        self,
        original_digest: str,
        plan_key: str,
        pixels: np.ndarray,
        metadata: dict[str, Any],
    ) -> str:
        key = cache_key(original_digest, plan_key)
        if not self.enabled:
            return key
        image_path, meta_path = self._paths(key)
        image_path.parent.mkdir(parents=True, exist_ok=True)

        buffer = BytesIO()
        # optimize=False keeps the encoder deterministic across Pillow builds;
        # the cache is not trying to be small, it is trying to be exact.
        Image.fromarray(np.ascontiguousarray(pixels), mode="RGB").save(buffer, format="PNG", optimize=False)

        # Write both files atomically. A crash between the PNG and the sidecar
        # would otherwise leave an entry that get() treats as a miss forever.
        self._atomic_write(image_path, buffer.getvalue())
        self._atomic_write(
            meta_path,
            json.dumps({**metadata, "cache_key": key}, indent=2, sort_keys=True, default=str).encode(),
        )
        return key

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        handle, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(handle, "wb") as fh:
                fh.write(payload)
            os.replace(tmp_name, path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    def _discard(self, key: str) -> None:
        for path in self._paths(key):
            try:
                path.unlink(missing_ok=True)
            except OSError:  # pragma: no cover - best effort
                pass

    # --------------------------------------------------------------- admin --

    def stats(self) -> dict[str, Any]:
        if not self.enabled or not self.root.exists():
            return {"enabled": self.enabled, "entries": 0, "bytes": 0}
        entries = 0
        total = 0
        for path in self.root.rglob("*.png"):
            entries += 1
            try:
                total += path.stat().st_size
            except OSError:  # pragma: no cover
                pass
        return {"enabled": True, "entries": entries, "bytes": total, "root": str(self.root)}

    def clear(self) -> int:
        if not self.enabled or not self.root.exists():
            return 0
        removed = 0
        for path in list(self.root.rglob("*")):
            if path.is_file():
                try:
                    path.unlink()
                    removed += 1
                except OSError:  # pragma: no cover
                    pass
        return removed


__all__ = ["CacheHit", "EnhancementCache", "cache_key"]
