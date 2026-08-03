"""Image ingest and upload."""

from __future__ import annotations

import io
import random

import pytest
from PIL import Image as PILImage

from image_discovery.ingest import hamming_distance, ingest, mirrored_phash
from shared.errors import ValidationError

CASE = {
    "case_id": "IMG-2026-1",
    "title": "Image upload",
    "lawful_basis": "Acceptance run",
}


def _base_image(seed: int = 7, size: int = 256) -> PILImage.Image:
    """A deterministic 16x16 grid of random blocks.

    Rich frequency content on purpose. A flat fill or a fixed-pixel
    checkerboard produces degenerate perceptual hashes — the checkerboard also
    changes *pattern density* when the canvas resizes, so it is a different
    picture at each size rather than the same one scaled, which makes any
    resize assertion meaningless.
    """
    rng = random.Random(seed)  # noqa: S311 - test fixtures, not cryptography
    image = PILImage.new("RGB", (size, size))
    block = size // 16
    for bx in range(16):
        for by in range(16):
            colour = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
            for x in range(bx * block, (bx + 1) * block):
                for y in range(by * block, (by + 1) * block):
                    image.putpixel((x, y), colour)
    return image


def _png(size: int = 96, seed: int = 7) -> bytes:
    """The same picture, rendered at ``size``."""
    image = _base_image(seed).resize((size, size), PILImage.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _jpeg(quality: int = 90, size: int = 96, seed: int = 7) -> bytes:
    image = _base_image(seed).resize((size, size), PILImage.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


# ------------------------------------------------------------------ ingest --


def test_ingest_computes_hashes_and_dimensions():
    result = ingest(_png(120))
    assert len(result.sha256) == 64
    assert len(result.phash) == len(result.dhash) == len(result.whash) == 16
    assert (result.width, result.height) == (120, 120)
    assert result.image_format == "PNG"
    assert result.mime_type == "image/png"


def test_identical_bytes_hash_identically():
    assert ingest(_png()).sha256 == ingest(_png()).sha256


def test_recompression_changes_sha_but_not_perceptual_hash():
    """The whole point of a perceptual hash: a re-encode is the same picture."""
    original = ingest(_jpeg(quality=95))
    recompressed = ingest(_jpeg(quality=40))
    assert original.sha256 != recompressed.sha256
    assert hamming_distance(original.phash, recompressed.phash) <= 8


def test_resize_preserves_perceptual_hash():
    small = ingest(_png(48))
    large = ingest(_png(192))
    assert hamming_distance(small.phash, large.phash) <= 8


def test_unrelated_images_are_perceptually_distant():
    """Different seeds are genuinely different pictures, not variants."""
    first = ingest(_png(seed=7))
    second = ingest(_png(seed=99))
    assert hamming_distance(first.phash, second.phash) > 8


def test_mirrored_hash_differs_from_the_original():
    """A mirrored repost hashes nothing like its source, which is why detecting
    one means comparing against the *flipped* probe."""
    asymmetric = PILImage.new("RGB", (64, 64), (10, 10, 10))
    for x in range(32):
        for y in range(64):
            asymmetric.putpixel((x, y), (240, 240, 240))
    buffer = io.BytesIO()
    asymmetric.save(buffer, format="PNG")
    data = buffer.getvalue()

    assert mirrored_phash(data) != ingest(data).phash


def test_non_image_bytes_are_refused():
    with pytest.raises(ValidationError, match="not a decodable image"):
        ingest(b"%PDF-1.7 this is not an image")


def test_empty_upload_is_refused():
    with pytest.raises(ValidationError, match="empty"):
        ingest(b"")


def test_oversized_upload_is_refused():
    with pytest.raises(ValidationError, match="over the"):
        ingest(_png(), max_bytes=10)


def test_hamming_rejects_mismatched_lengths():
    with pytest.raises(ValidationError, match="different lengths"):
        hamming_distance("abcd", "abcdef")


# ------------------------------------------------------------------ upload --


async def _case(auth_client) -> str:
    response = await auth_client.post("/api/v1/investigations", json=CASE)
    assert response.status_code == 201, response.text
    return response.json()["id"]


async def test_upload_stores_the_image(auth_client):
    case_id = await _case(auth_client)
    response = await auth_client.post(
        f"/api/v1/investigations/{case_id}/images",
        files={"file": ("probe.png", _png(), "image/png")},
    )
    assert response.status_code == 201, response.text
    body = response.json()

    assert body["deduplicated"] is False
    assert body["image"]["role"] == "PROBE"
    assert len(body["image"]["sha256"]) == 64
    assert body["image"]["width"] == 96
    assert body["image"]["progress_state"] == "DISCOVERED"


async def test_reupload_returns_the_existing_image(auth_client):
    """Identical bytes are a mistake, not a second probe — creating another row
    would double-count it downstream."""
    case_id = await _case(auth_client)
    payload = _png()
    first = (
        await auth_client.post(
            f"/api/v1/investigations/{case_id}/images",
            files={"file": ("probe.png", payload, "image/png")},
        )
    ).json()
    second = (
        await auth_client.post(
            f"/api/v1/investigations/{case_id}/images",
            files={"file": ("again.png", payload, "image/png")},
        )
    ).json()

    assert second["deduplicated"] is True
    assert second["image"]["id"] == first["image"]["id"]

    listing = (await auth_client.get(f"/api/v1/investigations/{case_id}/images")).json()
    assert len(listing) == 1


async def test_content_type_is_sniffed_not_trusted(auth_client):
    """A PDF renamed to .jpg with an image content type must still be refused."""
    case_id = await _case(auth_client)
    response = await auth_client.post(
        f"/api/v1/investigations/{case_id}/images",
        files={"file": ("innocent.jpg", b"%PDF-1.7 not an image", "image/jpeg")},
    )
    assert response.status_code == 422
    assert "not a decodable image" in response.json()["detail"]


async def test_refused_upload_is_audited(auth_client):
    case_id = await _case(auth_client)
    await auth_client.post(
        f"/api/v1/investigations/{case_id}/images",
        files={"file": ("bad.jpg", b"nonsense", "image/jpeg")},
    )
    entries = (await auth_client.get(f"/api/v1/audit?investigation_id={case_id}")).json()
    refusals = [e for e in entries if e["action"] == "image.upload" and e["outcome"] == "refused"]
    assert refusals


async def test_upload_opens_a_chain_of_custody(auth_client, session_factory, clock):
    """That first record is the anchor everything later derives from."""
    import uuid

    from database.repositories import CustodyRepository
    from shared.enums import ArtifactType

    case_id = await _case(auth_client)
    image = (
        await auth_client.post(
            f"/api/v1/investigations/{case_id}/images",
            files={"file": ("probe.png", _png(), "image/png")},
        )
    ).json()["image"]

    async with session_factory() as session:
        custody = CustodyRepository(session, clock)
        chain = await custody.chain_for(ArtifactType.IMAGE, uuid.UUID(image["id"]))
        verification = await custody.verify(ArtifactType.IMAGE, uuid.UUID(image["id"]))

    assert len(chain) == 1
    assert chain[0].action == "COLLECTED"
    assert chain[0].actor_kind == "HUMAN"
    assert chain[0].content_hash == image["sha256"]
    assert verification["valid"] is True


async def test_image_bytes_round_trip(auth_client):
    case_id = await _case(auth_client)
    payload = _png()
    image = (
        await auth_client.post(
            f"/api/v1/investigations/{case_id}/images",
            files={"file": ("probe.png", payload, "image/png")},
        )
    ).json()["image"]

    response = await auth_client.get(f"/api/v1/images/{image['id']}/content")
    assert response.status_code == 200
    assert response.content == payload
    assert response.headers["content-type"].startswith("image/png")


async def test_images_require_authentication(client):
    response = await client.get("/api/v1/images/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 401


async def test_gps_exif_is_flagged_in_the_audit_trail(auth_client):
    """Location metadata is the most privacy-sensitive thing an upload can
    carry; the operator should learn it is there."""
    case_id = await _case(auth_client)
    await auth_client.post(
        f"/api/v1/investigations/{case_id}/images",
        files={"file": ("probe.png", _png(), "image/png")},
    )
    entries = (await auth_client.get(f"/api/v1/audit?investigation_id={case_id}")).json()
    stored = [e for e in entries if e["action"] == "image.upload" and e["outcome"] == "stored"]
    assert "has_gps_exif" in stored[-1]["detail"]
