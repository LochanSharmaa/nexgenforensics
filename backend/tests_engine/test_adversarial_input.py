"""
Item 33 — adversarial and malformed input handling.

WHAT THIS IS CHECKING, AND WHY IT MATTERS
------------------------------------------
Every input here is something a real operator could plausibly submit: a
truncated download, a screenshot with no face, a scanned page, a file that was
renamed to .jpg. None of them should produce a 500.

The distinction this suite enforces is between a TYPED failure and an
UNTYPED one:

  * A typed failure (UnsupportedImageError, ImageTooLargeError,
    InvalidImageError, NoFaceDetectedError) is one the API maps to a 4xx with
    a message the operator can act on.
  * Any other exception escaping the pipeline becomes a 500. In this system a
    500 is worse than a rejection: it tells the operator nothing, it may leak
    a stack trace, and in a batch it can abort work that was otherwise fine.

So the assertion is not merely "it raised" — it is "it raised something the
API layer knows how to turn into a clean answer".

A few cases below deliberately assert that a HARMLESS input SUCCEEDS. A
validator that rejects everything would pass a naive version of this suite
while making the product useless.
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from imatch_api.services.storage_service import (  # noqa: E402
    ImageTooLargeError,
    UnsupportedImageError,
    decode_base64_image,
    sniff_content_type,
)
from nexgen_engine.config import EngineConfig  # noqa: E402
from nexgen_engine.inference.pipeline import (  # noqa: E402
    InvalidImageError,
    NoFaceDetectedError,
)

# Errors the API layer maps to 4xx. Anything else escaping is a 500.
HANDLED = (UnsupportedImageError, ImageTooLargeError, InvalidImageError, NoFaceDetectedError)

MAX_BYTES = 8 * 1024 * 1024


def _png(width: int, height: int, colour=(128, 128, 128)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buf, format="PNG")
    return buf.getvalue()


def _jpeg(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


# --------------------------------------------------------------- decoding ---


@pytest.mark.parametrize(
    "payload,label",
    [
        ("", "empty string"),
        ("!!!!not base64!!!!", "invalid base64 alphabet"),
        ("YWJj", "valid base64, but decodes to 'abc' - not an image"),
        ("=", "lone padding character"),
        ("A", "single char, not a valid base64 quantum"),
    ],
)
def test_malformed_base64_raises_handled_error(payload, label):
    """Garbage in the base64 field must not escape as a raw binascii error."""
    try:
        raw = decode_base64_image(payload, MAX_BYTES)
    except HANDLED:
        return  # rejected at decode, correct
    # Decoded to *something*; it must then be rejected by format sniffing.
    with pytest.raises(HANDLED):
        sniff_content_type(raw)


def test_zero_byte_payload_is_rejected():
    """A zero-byte file is the classic truncated-download case."""
    with pytest.raises(HANDLED):
        sniff_content_type(decode_base64_image(_b64(b""), MAX_BYTES))


def test_oversized_payload_rejected_before_decoding():
    """The size gate must fire on the ENCODED length.

    Checking only after decoding would mean allocating the full payload in
    memory first, which is the denial-of-service the gate exists to prevent.
    """
    huge = "A" * (MAX_BYTES * 4 // 3 + 4096)
    with pytest.raises(ImageTooLargeError):
        decode_base64_image(huge, MAX_BYTES)


def test_data_url_prefix_is_tolerated():
    """Browsers produce data: URLs; stripping the prefix must not be lossy."""
    raw = _png(64, 64)
    assert decode_base64_image(f"data:image/png;base64,{_b64(raw)}", MAX_BYTES) == raw


def test_renamed_non_image_is_rejected_by_content_not_name():
    """A PDF or ZIP renamed to .jpg must fail on magic bytes.

    Trusting a client-supplied filename would be the vulnerability here.
    """
    for raw, what in [
        (b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n", "PDF"),
        (b"PK\x03\x04" + b"\x00" * 32, "ZIP"),
        (b"#!/bin/sh\nrm -rf /\n", "shell script"),
        (b"\x00" * 64, "null bytes"),
    ]:
        with pytest.raises(UnsupportedImageError):
            sniff_content_type(raw)


def test_real_formats_are_accepted():
    """Guard against a validator that rejects everything."""
    assert sniff_content_type(_png(32, 32)) == "image/png"
    assert sniff_content_type(_jpeg(32, 32)) == "image/jpeg"


# --------------------------------------------------------------- pipeline ---


@pytest.fixture(scope="module")
def pipeline():
    from nexgen_engine.inference.pipeline import FacialRecognitionPipeline

    return FacialRecognitionPipeline(EngineConfig())


@pytest.mark.slow
@pytest.mark.parametrize(
    "raw,label",
    [
        (b"", "zero bytes"),
        (b"\xff\xd8\xff\xe0" + b"\x00" * 16, "JPEG magic then garbage"),
        (_jpeg(64, 64)[:40], "truncated JPEG"),
        (b"GIF89a" + b"\x00" * 32, "GIF header, unsupported"),
    ],
)
def test_pipeline_rejects_corrupt_bytes_cleanly(pipeline, raw, label):
    """Corrupt bytes must raise a HANDLED error, never an unmapped exception."""
    with pytest.raises(HANDLED):
        pipeline.encode_bytes(raw)


@pytest.mark.slow
@pytest.mark.parametrize(
    "w,h,label",
    [
        (1, 1, "single pixel"),
        (1, 4000, "1px wide, extreme aspect ratio"),
        (4000, 1, "1px tall, extreme aspect ratio"),
        (8, 8, "smaller than any face"),
    ],
)
def test_pipeline_rejects_degenerate_geometry(pipeline, w, h, label):
    """Extreme aspect ratios must not crash the detector or aligner.

    A 1x4000 strip is a realistic accident (a bad crop), and it exercises
    resize/padding maths where a divide-by-zero or negative dimension is easy
    to introduce.
    """
    with pytest.raises(HANDLED):
        pipeline.encode_bytes(_png(w, h))


@pytest.mark.slow
@pytest.mark.parametrize(
    "raw,label",
    [
        (_png(512, 512, (255, 255, 255)), "blank white image"),
        (_png(512, 512, (0, 0, 0)), "blank black image"),
        (_jpeg(512, 512), "random noise"),
    ],
    ids=["blank-white", "blank-black", "random-noise"],
)
def test_pipeline_reports_no_face_rather_than_crashing(pipeline, raw, label):
    """A valid image with no face is the single most common real rejection.

    It must be NoFaceDetectedError specifically -- the operator needs to know
    the image was fine but contained no face, not that something broke.
    """
    with pytest.raises(NoFaceDetectedError):
        pipeline.encode_bytes(raw)


@pytest.mark.slow
def test_pipeline_survives_large_but_legal_image(pipeline):
    """A 4000x3000 phone photo is normal input, not an attack.

    It must not be rejected for size alone. It contains no face, so the
    expected outcome is a clean NoFaceDetectedError rather than a timeout,
    a memory error, or a hang.
    """
    with pytest.raises(NoFaceDetectedError):
        pipeline.encode_bytes(_jpeg(4000, 3000))


@pytest.mark.slow
def test_real_face_still_succeeds(pipeline):
    """The control. Everything above asserts rejection; this proves the
    pipeline has not simply been made to reject all input."""
    agedb = _BACKEND.parent / "src_extracted/AgeDB/AgeDB"
    faces = sorted(agedb.glob("*.jpg"))
    if not faces:
        pytest.skip("AgeDB imagery not available")
    result = pipeline.encode_bytes(faces[0].read_bytes())
    assert result.embedding.shape[0] == 512
    assert np.isfinite(result.embedding).all()
    assert abs(float(np.linalg.norm(result.embedding)) - 1.0) < 1e-3
