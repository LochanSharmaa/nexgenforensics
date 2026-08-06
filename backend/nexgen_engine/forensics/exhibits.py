"""Deterministic rendering of the plates that go into the examination report.

An exhibit is a picture that will be argued about. The properties that matter
are therefore not visual:

**Reproducible.** Re-rendering from the same inputs and the same params must
produce byte-identical output, so an exhibit can be regenerated on challenge
and shown to be the one that was disclosed. That forces PIL only -- no
matplotlib, no system font lookup -- and pinned colours and geometry. The font
is Pillow's own bundled face, loaded by size rather than by path, because a
font substitution changes the bytes and therefore the hash.

**Traceable.** Every exhibit carries the SHA-256 of its own bytes and of every
source image that fed it, so the chain from stored evidence to printed page is
readable off the document.

**Honest about what it shows.** An annotated plate is a derived image. The
unmodified photograph it came from appears in the description-of-exhibits
section of the same report, and each plate's caption names the transform that
was applied to it.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .morphology import (
    BROW_LEFT,
    BROW_RIGHT,
    EYE_LEFT,
    EYE_RIGHT,
    NASION,
    SUBNASALE,
    FaceMorphology,
    LandmarkSet,
    Measurement,
)

logger = logging.getLogger(__name__)

#: Bumped whenever a change to this module would alter rendered bytes. It is
#: recorded on every exhibit so a hash mismatch can be explained by a version
#: change rather than looking like tampering.
EXHIBIT_RENDERER_VERSION = "1.0.0"

#: One colour per measurement letter, matching the convention of the paper
#: form: the arrow and its letter share a colour so a reader can pair them
#: across a crowded plate without a legend.
MEASUREMENT_COLOURS: dict[str, tuple[int, int, int]] = {
    "a": (222, 38, 38),
    "b": (243, 216, 26),
    "c": (255, 255, 255),
    "d": (74, 85, 104),
    "e": (154, 205, 50),
    "f": (240, 112, 160),
    "g": (245, 152, 24),
    "h": (36, 150, 237),
}

_OUTLINE = (16, 16, 16)
_PLATE_HEIGHT = 1180          #: rendered pixel height of a measurement plate
_UPPER_FACE_HEIGHT = 520      #: rendered pixel height of an upper-face crop
_CROP_PAD = 0.42              #: face-box fraction added around the face


@dataclass(frozen=True)
class Exhibit:
    """One rendered plate, with everything needed to reproduce or audit it."""

    png: bytes
    kind: str
    caption: str
    source_sha256: tuple[str, ...]
    params: dict[str, Any] = field(default_factory=dict)
    renderer_version: str = EXHIBIT_RENDERER_VERSION

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.png).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form. The bytes are referenced by hash, not embedded."""
        return {
            "kind": self.kind,
            "caption": self.caption,
            "sha256": self.sha256,
            "source_sha256": list(self.source_sha256),
            "params": self.params,
            "renderer_version": self.renderer_version,
            "bytes": len(self.png),
        }


def _font(size: int) -> ImageFont.FreeTypeFont:
    """Pillow's bundled scalable face.

    Loaded by size, never by path: a path lookup would resolve to whatever the
    host has installed and would silently change the rendered bytes between
    machines, breaking the reproducibility the hash is supposed to guarantee.
    """
    return ImageFont.load_default(size=size)


def _to_png(image: Image.Image) -> bytes:
    buffer = BytesIO()
    # optimize=False keeps the encoder on a fixed path; the default filter
    # selection is deterministic but not guaranteed stable across Pillow's
    # optimisation heuristics.
    image.save(buffer, format="PNG", optimize=False, compress_level=6)
    return buffer.getvalue()


def _crop_box(
    bbox: tuple[float, float, float, float],
    size: tuple[int, int],
    pad: float = _CROP_PAD,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    width, height = x2 - x1, y2 - y1
    px, py = width * pad, height * pad
    return (
        int(max(0, x1 - px)),
        int(max(0, y1 - py)),
        int(min(size[0], x2 + px)),
        int(min(size[1], y2 + py)),
    )


def _outlined_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    colour: tuple[int, int, int],
    size: int,
) -> None:
    """Text with a dark rim, so a light label survives a light background."""
    font = _font(size)
    x, y = xy
    ring = max(2, size // 14)
    for dx in (-ring, 0, ring):
        for dy in (-ring, 0, ring):
            if dx or dy:
                draw.text((x + dx, y + dy), text, font=font, fill=_OUTLINE, anchor="mm")
    draw.text((x, y), text, font=font, fill=colour, anchor="mm")


def _arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    colour: tuple[int, int, int],
    width: int,
) -> None:
    """Double-headed arrow, rimmed so it reads against skin or background."""
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    spread = 0.42

    def stroke_pass(shade: tuple[int, int, int], line_width: int, head: float) -> None:
        draw.line([start, end], fill=shade, width=line_width)
        for point, direction in ((start, angle), (end, angle + math.pi)):
            draw.polygon(
                [
                    point,
                    (point[0] + head * math.cos(direction - spread),
                     point[1] + head * math.sin(direction - spread)),
                    (point[0] + head * math.cos(direction + spread),
                     point[1] + head * math.sin(direction + spread)),
                ],
                fill=shade,
            )

    head = max(9.0, width * 3.0)
    stroke_pass(_OUTLINE, width + 4, head + 3.0)
    stroke_pass(colour, width, head)


def measurement_plate(
    image: Image.Image,
    morphology: FaceMorphology,
    mark: str,
    source_sha256: str,
) -> Exhibit:
    """The face with each measured distance drawn between the points that produced it.

    The arrows are drawn between the *actual* landmark positions in the source
    image, so what the reader sees is the measurement that was taken, not an
    idealised diagram of it. The reported length of each is the straight-line
    distance between those two points, which is why a tilted head produces a
    tilted arrow and the same number.
    """
    landmarks = morphology.landmarks
    frame = image.convert("RGB")
    box = _crop_box(landmarks.bbox, frame.size)
    crop = frame.crop(box)
    if crop.width < 8 or crop.height < 8:
        raise ValueError("Face crop is too small to render a plate.")

    scale = _PLATE_HEIGHT / crop.height
    plate = crop.resize(
        (max(1, int(round(crop.width * scale))), _PLATE_HEIGHT), Image.LANCZOS
    )
    draw = ImageDraw.Draw(plate)

    def place(index: int) -> tuple[float, float]:
        point = landmarks.points_106[index]
        return ((point[0] - box[0]) * scale, (point[1] - box[1]) * scale)

    stroke = max(4, int(plate.height / 150))
    label_size = max(30, int(plate.height / 19))
    centre = (plate.width / 2.0, plate.height / 2.0)

    # Eight arrows over one face is a crowded plate, and a label that lands on
    # top of another is worse than no label. Each is pushed out past the end of
    # its own arrow, away from the middle of the face, then nudged clear of any
    # label already placed. Deterministic: the nudge order follows the fixed
    # measurement order, so the same face always produces the same layout.
    placed: list[tuple[float, float]] = []

    def anchor_for(start: tuple[float, float], end: tuple[float, float]) -> tuple[float, float]:
        far = max((start, end), key=lambda p: math.dist(p, centre))
        near = start if far is end else end
        angle = math.atan2(far[1] - near[1], far[0] - near[0])
        reach = label_size * 1.15
        x = far[0] + reach * math.cos(angle)
        y = far[1] + reach * math.sin(angle)

        for _ in range(12):
            clash = next(
                (p for p in placed if math.dist((x, y), p) < label_size * 1.35), None
            )
            if clash is None:
                break
            x += reach * 0.62 * math.cos(angle)
            y += reach * 0.62 * math.sin(angle)

        pad = label_size * 0.75
        return (
            min(max(x, pad), plate.width - pad),
            min(max(y, pad), plate.height - pad),
        )

    for measurement in morphology.measurements:
        colour = MEASUREMENT_COLOURS.get(measurement.key, (255, 255, 255))
        start, end = (place(i) for i in measurement.landmark_ids)
        _arrow(draw, start, end, colour, stroke)
        anchor = anchor_for(start, end)
        placed.append(anchor)
        _outlined_text(draw, anchor, measurement.key, colour, label_size)

    caption = (
        f"Exhibit {mark}: measured distances a-h drawn between the landmark points "
        f"that produced them. Cropped to the detected face and scaled for reproduction; "
        f"pixel content otherwise unaltered."
    )
    return Exhibit(
        png=_to_png(plate),
        kind="measurement_plate",
        caption=caption,
        source_sha256=(source_sha256,),
        params={
            "mark": mark,
            "crop_box": list(box),
            "plate_height_px": _PLATE_HEIGHT,
            "crop_pad": _CROP_PAD,
            "stroke_px": stroke,
            "label_px": label_size,
            "measurement_keys": [m.key for m in morphology.measurements],
            "landmark_ids": {m.key: list(m.landmark_ids) for m in morphology.measurements},
        },
    )


def upper_face_plate(
    image: Image.Image,
    landmarks: LandmarkSet,
    mark: str,
    source_sha256: str,
) -> Exhibit:
    """The brow-to-nose-base region, unannotated, for the comparison grids.

    The secondary-identification section compares specific features -- a
    hairline, a mole, a piercing -- and a full-frame photograph makes the reader
    hunt for them. The crop is geometric and stated: it is not an enhancement,
    and no pixel is altered.
    """
    frame = image.convert("RGB")
    points = landmarks.points_106
    region = list(EYE_LEFT) + list(EYE_RIGHT) + list(BROW_LEFT) + list(BROW_RIGHT) + [NASION, SUBNASALE]
    xs = points[region, 0]
    ys = points[region, 1]

    # Widened horizontally to keep the temples and hairline edges in frame,
    # since those carry the features this crop exists to show.
    half_width = (xs.max() - xs.min()) * 0.72
    centre_x = (xs.max() + xs.min()) / 2.0
    top = ys.min() - (ys.max() - ys.min()) * 1.05
    bottom = ys.max() + (ys.max() - ys.min()) * 0.18

    box = (
        int(max(0, centre_x - half_width)),
        int(max(0, top)),
        int(min(frame.width, centre_x + half_width)),
        int(min(frame.height, bottom)),
    )
    crop = frame.crop(box)
    if crop.width < 8 or crop.height < 8:
        raise ValueError("Upper-face crop is too small to render.")

    scale = _UPPER_FACE_HEIGHT / crop.height
    plate = crop.resize(
        (max(1, int(round(crop.width * scale))), _UPPER_FACE_HEIGHT), Image.LANCZOS
    )
    return Exhibit(
        png=_to_png(plate),
        kind="upper_face",
        caption=f"Exhibit {mark}: upper face, cropped and scaled. Pixel content unaltered.",
        source_sha256=(source_sha256,),
        params={
            "mark": mark,
            "crop_box": list(box),
            "plate_height_px": _UPPER_FACE_HEIGHT,
        },
    )


def comparison_plate(
    questioned: tuple[Image.Image, str, str],
    reference: tuple[Image.Image, str, str],
    landmarks: tuple[LandmarkSet | None, LandmarkSet | None] = (None, None),
) -> Exhibit:
    """Questioned left, reference right, height-matched on a common baseline.

    The left/right convention and the equal face height are not decoration:
    they are the presentation facial examiners are trained and tested on, and
    an unequal pair invites a size difference to be read as a feature
    difference when it is an artefact of how far away the camera was.

    Both faces are scaled to a common inter-pupillary distance when landmarks
    are available, and the caption records that this was done -- the reader has
    to know the images were geometrically normalised, and that this is why they
    sit at the same scale.
    """
    gap, margin, label_band = 26, 22, 54
    panels: list[Image.Image] = []
    normalised = False

    target_ipd = None
    if landmarks[0] is not None and landmarks[1] is not None:
        target_ipd = max(landmarks[0].ipd_px, landmarks[1].ipd_px)
        normalised = True

    for (image, _mark, _sha), landmark in zip((questioned, reference), landmarks):
        frame = image.convert("RGB")
        if landmark is not None:
            box = _crop_box(landmark.bbox, frame.size)
            frame = frame.crop(box)
            if target_ipd:
                factor = target_ipd / landmark.ipd_px
                frame = frame.resize(
                    (max(1, int(round(frame.width * factor))),
                     max(1, int(round(frame.height * factor)))),
                    Image.LANCZOS,
                )
        panels.append(frame)

    height = min(_PLATE_HEIGHT, max(p.height for p in panels))
    scaled = [
        p.resize((max(1, int(round(p.width * height / p.height))), height), Image.LANCZOS)
        for p in panels
    ]

    total_width = margin * 2 + gap + sum(p.width for p in scaled)
    canvas = Image.new("RGB", (total_width, height + label_band + margin * 2), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    x = margin
    for panel, (_img, mark, _sha) in zip(scaled, (questioned, reference)):
        canvas.paste(panel, (x, margin))
        draw.rectangle(
            [x - 1, margin - 1, x + panel.width, margin + panel.height],
            outline=(90, 90, 90), width=1,
        )
        draw.text(
            (x + panel.width / 2, margin + height + label_band / 2),
            mark, font=_font(38), fill=(20, 20, 20), anchor="mm",
        )
        x += panel.width + gap

    caption = (
        f"Questioned {questioned[1]} (left) beside specimen {reference[1]} (right), "
        "presented simultaneously at equal face height."
    )
    if normalised:
        caption += (
            " Both faces were geometrically scaled to a common inter-pupillary "
            "distance for display; this is why they appear at the same size."
        )
    return Exhibit(
        png=_to_png(canvas),
        kind="comparison_plate",
        caption=caption,
        source_sha256=(questioned[2], reference[2]),
        params={
            "questioned_mark": questioned[1],
            "reference_mark": reference[1],
            "ipd_normalised": normalised,
            "panel_height_px": height,
        },
    )


def measurement_legend() -> tuple[tuple[str, tuple[int, int, int]], ...]:
    """Letter-to-colour pairs, so the PDF can print a key matching the plates."""
    return tuple((key, MEASUREMENT_COLOURS[key]) for key in sorted(MEASUREMENT_COLOURS))


__all__ = [
    "EXHIBIT_RENDERER_VERSION",
    "MEASUREMENT_COLOURS",
    "Exhibit",
    "comparison_plate",
    "measurement_legend",
    "measurement_plate",
    "upper_face_plate",
]
