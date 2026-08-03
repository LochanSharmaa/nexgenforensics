"""Image ingest: hashing, dimensions, EXIF.

Stage 1 of the pipeline. Everything computed here is a property of the *file* —
cryptographic hash, perceptual hash, dimensions, embedded metadata. Nothing
inspects faces, and nothing here could: no model with facial semantics exists in
the dependency graph (ARCHITECTURE §14).

Provenance *classification* — deciding that image B is a cropped copy of image A
— lands in Phase 8 with its calibrated thresholds. This module only produces the
measurements that classification will later compare.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from typing import Any

import imagehash
from PIL import ExifTags, Image, UnidentifiedImageError

from shared.errors import ValidationError
from shared.logging import get_logger

MAX_IMAGE_BYTES = 20 * 1024 * 1024
HASH_SIZE = 8
"""8×8 grids give the 64-bit hashes the schema stores as 16 hex characters."""

SUPPORTED_FORMATS = frozenset({"JPEG", "PNG", "WEBP", "GIF", "BMP", "TIFF", "AVIF"})

logger = get_logger(__name__)

# EXIF tags that carry location. Split out because they are the most privacy-
# sensitive thing an upload can contain and an operator should see, in the UI,
# that the platform noticed them.
_GPS_TAG = "GPSInfo"


@dataclass(frozen=True)
class IngestedImage:
    """Everything stage 1 learns about an uploaded or downloaded image."""

    sha256: str
    phash: str
    dhash: str
    whash: str
    width: int
    height: int
    file_size: int
    mime_type: str
    image_format: str
    exif: dict[str, Any] = field(default_factory=dict)
    has_gps: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "phash": self.phash,
            "dhash": self.dhash,
            "whash": self.whash,
            "width": self.width,
            "height": self.height,
            "file_size": self.file_size,
            "mime_type": self.mime_type,
            "format": self.image_format,
            "has_gps": self.has_gps,
        }


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hex64(value: imagehash.ImageHash) -> str:
    """Render a 64-bit perceptual hash as 16 hex characters.

    `str(ImageHash)` already produces this for an 8×8 grid, but padding is
    asserted rather than assumed: a short hash would silently break Hamming
    comparison later, and a wrong distance is worse than a missing one.
    """
    text = str(value)
    if len(text) != 16:
        raise ValidationError(
            f"Expected a 64-bit perceptual hash (16 hex chars), got {len(text)}."
        )
    return text


def hamming_distance(left: str, right: str) -> int:
    """Bit difference between two hex-encoded hashes.

    Kept here beside the hashing so the encoding and its comparison cannot drift
    apart. Phase 8 calibrates the thresholds that consume this.
    """
    if len(left) != len(right):
        raise ValidationError("Cannot compare hashes of different lengths.")
    return bin(int(left, 16) ^ int(right, 16)).count("1")


def _extract_exif(image: Image.Image) -> tuple[dict[str, Any], bool]:
    """Readable EXIF plus whether it contains GPS.

    Values are coerced to strings: EXIF is arbitrary vendor data including
    IFDRational and bytes, none of which survives JSON serialisation, and a
    crash here would fail an otherwise-valid upload.
    """
    raw = getattr(image, "_getexif", lambda: None)()
    if not raw:
        return {}, False

    exif: dict[str, Any] = {}
    has_gps = False
    for tag_id, value in raw.items():
        name = ExifTags.TAGS.get(tag_id, str(tag_id))
        if name == _GPS_TAG:
            has_gps = bool(value)
            exif[name] = "present"     # coordinates are not expanded here
            continue
        try:
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            exif[name] = str(value)[:500]
        except Exception as exc:  # noqa: BLE001 - one bad tag must not fail the upload
            logger.debug("exif.tag_unreadable", tag=name, error=type(exc).__name__)
            continue
    return exif, has_gps


def ingest(data: bytes, *, max_bytes: int = MAX_IMAGE_BYTES) -> IngestedImage:
    """Decode an image and compute everything stage 1 records about it."""
    if not data:
        raise ValidationError("The uploaded file is empty.")
    if len(data) > max_bytes:
        raise ValidationError(
            f"Image is {len(data)} bytes, over the {max_bytes} byte limit."
        )

    try:
        with Image.open(io.BytesIO(data)) as image:
            image_format = (image.format or "").upper()
            if image_format not in SUPPORTED_FORMATS:
                raise ValidationError(
                    f"Unsupported image format {image_format or 'unknown'!r}. "
                    f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}."
                )
            width, height = image.size
            exif, has_gps = _extract_exif(image)

            # Perceptual hashes are computed on the RGB rendering so that a
            # palette or alpha difference alone cannot change the hash — two
            # files that look identical must hash identically.
            rgb = image.convert("RGB")
            phash = _hex64(imagehash.phash(rgb, hash_size=HASH_SIZE))
            dhash = _hex64(imagehash.dhash(rgb, hash_size=HASH_SIZE))
            whash = _hex64(imagehash.whash(rgb, hash_size=HASH_SIZE))
    except UnidentifiedImageError as exc:
        raise ValidationError(
            "That file is not a decodable image. The content type is sniffed "
            "from the bytes, not trusted from the request header."
        ) from exc
    except ValidationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(f"Could not read the image: {type(exc).__name__}") from exc

    return IngestedImage(
        sha256=sha256_of(data),
        phash=phash,
        dhash=dhash,
        whash=whash,
        width=width,
        height=height,
        file_size=len(data),
        mime_type=Image.MIME.get(image_format, "application/octet-stream"),
        image_format=image_format,
        exif=exif,
        has_gps=has_gps,
    )


def mirrored_phash(data: bytes) -> str:
    """Perceptual hash of the horizontally flipped image.

    A mirrored repost is a distinct provenance relationship, and a mirror image
    hashes nothing like its original. Comparing the probe's *flipped* hash to a
    candidate's normal hash is what detects it. Purely geometric — no facial
    analysis is involved or possible.
    """
    with Image.open(io.BytesIO(data)) as image:
        flipped = image.convert("RGB").transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return _hex64(imagehash.phash(flipped, hash_size=HASH_SIZE))


__all__ = [
    "HASH_SIZE",
    "MAX_IMAGE_BYTES",
    "SUPPORTED_FORMATS",
    "IngestedImage",
    "hamming_distance",
    "ingest",
    "mirrored_phash",
    "sha256_of",
]
