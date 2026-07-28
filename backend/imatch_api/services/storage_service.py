from __future__ import annotations

import base64
import binascii
import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Formats Pillow decodes reliably and that investigators actually receive.
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}

# Magic bytes, checked instead of trusting the declared content type. A file
# claiming image/jpeg that is actually something else should never reach the
# decoder.
_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"BM", "image/bmp"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
)


class UnsupportedImageError(ValueError):
    """Raised when a payload is not an image format we accept."""


class ImageTooLargeError(ValueError):
    """Raised when a payload exceeds the configured upload ceiling."""


@dataclass(frozen=True)
class StoredImage:
    sha256: str
    path: str
    size_bytes: int
    content_type: str


def decode_base64_image(value: str, max_bytes: int) -> bytes:
    """Decode a base64 payload, tolerating a data: URL prefix.

    The size is checked before decoding as well as after: base64 inflates by
    about a third, so a caller could otherwise send a string that only becomes
    oversized in memory.
    """
    payload = value.strip()
    if payload.startswith("data:"):
        _, _, payload = payload.partition(",")

    if len(payload) > max_bytes * 4 // 3 + 1024:
        raise ImageTooLargeError(f"Image exceeds the {max_bytes // (1024 * 1024)} MB limit.")

    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise UnsupportedImageError("image_base64 is not valid base64.") from exc

    if len(raw) > max_bytes:
        raise ImageTooLargeError(f"Image exceeds the {max_bytes // (1024 * 1024)} MB limit.")
    return raw


def sniff_content_type(raw: bytes) -> str:
    """Identify the format from magic bytes. Raises if unrecognised."""
    for signature, content_type in _SIGNATURES:
        if raw.startswith(signature):
            return content_type
    # WebP is RIFF....WEBP, so it needs a two-part check.
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    raise UnsupportedImageError(
        "Unrecognised image format. Accepted: JPEG, PNG, WEBP, BMP, TIFF."
    )


class StorageService:
    """Content-addressed storage for probe and enrolment images.

    Files are named by SHA-256, so the same image submitted twice is stored
    once and any stored record can be tied back to exact bytes. Paths are
    derived entirely from that hash, never from a client-supplied filename,
    which removes path traversal as a concern.
    """

    def __init__(self, root: Path, retention_days: int = 90) -> None:
        self.root = Path(root)
        self.retention_days = retention_days
        self.root.mkdir(parents=True, exist_ok=True)

    def store(self, tenant_id: str, raw: bytes, category: str = "probes") -> StoredImage:
        content_type = sniff_content_type(raw)
        digest = hashlib.sha256(raw).hexdigest()

        # Two-level fan-out: a single flat directory with 10^5 files is painful
        # on most filesystems.
        target = self._directory(tenant_id, category) / digest[:2] / digest[2:4] / f"{digest}{_suffix(content_type)}"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(raw)

        return StoredImage(
            sha256=digest,
            path=str(target.relative_to(self.root)),
            size_bytes=len(raw),
            content_type=content_type,
        )

    def read(self, relative_path: str) -> bytes | None:
        target = self._safe_path(relative_path)
        if target is None or not target.is_file():
            return None
        return target.read_bytes()

    def delete(self, relative_path: str) -> bool:
        target = self._safe_path(relative_path)
        if target is None or not target.is_file():
            return False
        target.unlink()
        return True

    def purge_expired(self, tenant_id: str | None = None) -> int:
        """Delete probe images past the retention window.

        Only probes are purged. Enrolment images belong to a subject record and
        are removed when that subject is deleted, not on a timer.
        """
        if self.retention_days <= 0:
            return 0

        cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        scope = self._directory(tenant_id, "probes") if tenant_id else self.root
        if not scope.exists():
            return 0

        removed = 0
        for path in scope.rglob("*"):
            if not path.is_file():
                continue
            if not tenant_id and "probes" not in path.parts:
                continue
            try:
                if datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc) < cutoff:
                    path.unlink()
                    removed += 1
            except OSError as exc:
                logger.warning("Could not purge %s: %s", path, exc)
        return removed

    def _directory(self, tenant_id: str | None, category: str) -> Path:
        safe_tenant = "".join(char for char in (tenant_id or "shared") if char.isalnum() or char in "-_")
        safe_category = "".join(char for char in category if char.isalnum() or char in "-_")
        return self.root / safe_tenant / (safe_category or "misc")

    def _safe_path(self, relative_path: str) -> Path | None:
        """Resolve a stored path, refusing anything that escapes the root."""
        candidate = (self.root / relative_path).resolve()
        try:
            candidate.relative_to(self.root.resolve())
        except ValueError:
            logger.warning("Rejected storage path outside the root: %s", relative_path)
            return None
        return candidate


def _suffix(content_type: str) -> str:
    return {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/bmp": ".bmp",
        "image/tiff": ".tiff",
    }.get(content_type, ".bin")


__all__ = [
    "ALLOWED_CONTENT_TYPES",
    "ImageTooLargeError",
    "StorageService",
    "StoredImage",
    "UnsupportedImageError",
    "decode_base64_image",
    "sniff_content_type",
]
