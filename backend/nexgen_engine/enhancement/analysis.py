"""What is wrong with this image, measured before anything is done to it.

Builds on ``degradation/estimate.py`` rather than duplicating it: blur sigma,
noise sigma and JPEG quality already have confidence-gated estimators there, and
the confidence gating is the important part. That module's ``trust_components``
conjunction exists because an operator applied with false confidence produces a
confidently wrong result, and the same reasoning governs every threshold here --
nothing is acted on unless its own measurement cleared its own bar.

What is added on top is the part the published estimators do not cover, because
restoration benchmarks are not made of surveillance footage:

    infrared capture      night mode is monochrome; colour is absent, not wrong
    interlacing           analog DVRs, still everywhere; comb teeth are not detail
    true vs stored size   DVRs upscale before storage; extra pixels can be empty
    clipping              backlit doorways; saturated wells hold no information
    motion vs defocus     different physics, different correction, one test

All CPU, all deterministic, no learned weights, no new dependencies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from PIL import Image

from ..data.quality_filter import laplacian_variance, match_scale_sharpness
from ..degradation.bandlimit import effective_cutoff
from ..degradation.estimate import estimate_degradation

# A face below this many pixels on its short side carries little identity
# information; ImageQualityFilter uses 60 as its floor and this mirrors it.
SMALL_FACE_PIXELS = 60


def _gray(pixels: np.ndarray) -> np.ndarray:
    """Luma in 0..1. Rec.601 weights, matching what the detector effectively sees."""
    arr = np.asarray(pixels, dtype=np.float64)
    if arr.ndim == 2:
        return arr / 255.0 if arr.max() > 1.5 else arr
    if arr.max() > 1.5:
        arr = arr / 255.0
    return arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114


def infrared_score(pixels: np.ndarray) -> tuple[float, bool]:
    """How monochrome the image is, and whether that means IR capture.

    A camera in night mode retracts its IR-cut filter and illuminates with IR
    LEDs. The result is effectively single-channel: R, G and B carry the same
    value at every pixel.

    This matters more than it looks. Every face restoration model in the field
    is trained on visible-light RGB faces. Handed an IR frame it will produce
    confident, natural-looking daylight skin tone -- all of it invented, and
    invented in a way that does not look invented. Detecting the condition is
    what lets the pipeline refuse to colourise it.

    Returns ``(monochrome_fraction, is_infrared)``.
    """
    arr = np.asarray(pixels, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 1.0, True
    spread = arr.max(axis=2) - arr.min(axis=2)
    # Per-pixel channel spread below ~2/255 is indistinguishable from grey once
    # 8-bit quantisation and codec chroma rounding are accounted for.
    monochrome = float((spread <= 2.0).mean())
    return monochrome, bool(monochrome > 0.98)


def interlace_score(pixels: np.ndarray) -> tuple[float, bool]:
    """Comb-artifact energy: adjacent-row difference against same-parity difference.

    In an interlaced frame the two fields are captured 1/50 s apart, so on
    motion the odd and even rows disagree while rows of the *same* parity stay
    consistent. Progressive content has no such asymmetry.

    Left uncorrected this is actively harmful: a denoiser or an SR model reads
    comb teeth as genuine high-frequency detail and sharpens them.

    Returns ``(ratio, is_interlaced)`` where ratio is adjacent/same-parity mean
    absolute difference. 1.0 is progressive; well above 1 indicates combing.

    KNOWN LIMIT, measured rather than assumed. The test needs the image to have
    vertical spatial correlation, which real imagery has and pure noise does
    not. On smooth content the separation is clean -- progressive measures ~0.57,
    the same content woven from two fields 6px apart measures ~2.76. On white
    noise both cases measure ~1.0 and the test correctly abstains rather than
    guessing. Extremely noisy night-mode footage therefore sits closer to the
    abstain region, which is the right failure direction: not detecting
    interlacing costs a stage, falsely detecting it discards half the rows.
    """
    gray = _gray(pixels)
    if gray.shape[0] < 8:
        return 1.0, False
    adjacent = float(np.abs(gray[1:] - gray[:-1]).mean())
    same_parity = float(np.abs(gray[2:] - gray[:-2]).mean())

    # A flat image says nothing either way.
    if adjacent <= 1e-9 and same_parity <= 1e-9:
        return 1.0, False

    # same_parity == 0 with adjacent > 0 is not "no information", it is the
    # MAXIMAL combing case: rows two apart agree perfectly while neighbours
    # disagree, which is exactly what a static interlaced pattern looks like.
    # Bailing out here reported the most obviously interlaced input possible as
    # progressive.
    ratio = adjacent / max(same_parity, 1e-6)
    # Progressive content sits near 0.5-0.7 (adjacent rows correlate more than
    # rows two apart). Combing pushes it above 1.0. Capped so a degenerate
    # denominator cannot produce an absurd figure in a report.
    ratio = min(ratio, 100.0)
    return ratio, bool(ratio > 1.05)


def clipping(pixels: np.ndarray) -> tuple[float, float]:
    """Fractions of pixels crushed to black and blown to white.

    Backlit entrances are the common surveillance case: the subject sits at
    10-30 DN while the doorway behind clips at 255. A saturated sensor well
    holds no recoverable information, so anything a model puts back there is
    invention. Reporting the fraction is what stops that being silent.
    """
    arr = np.asarray(pixels)
    if arr.size == 0:
        return 0.0, 0.0
    return float((arr <= 2).mean()), float((arr >= 253).mean())


def blur_orientation(pixels: np.ndarray) -> tuple[float, float]:
    """Spectral anisotropy and the orientation of greatest attenuation.

    Motion blur is a directional box: it attenuates frequencies along the
    direction of travel and leaves the perpendicular direction largely intact,
    so the power spectrum is strongly anisotropic. Defocus is a disc: radially
    symmetric, so the spectrum is close to isotropic.

    One measurement therefore separates two degradations that need different
    corrections. Returns ``(anisotropy_0_to_1, angle_degrees)``.
    """
    gray = _gray(pixels)
    if min(gray.shape) < 16:
        return 0.0, 0.0

    # Hann window: without it the FFT sees the image edges as a step, which
    # injects a cross-shaped artifact along the axes and reads as anisotropy.
    h, w = gray.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray * window)))

    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    dy = (yy - cy) / max(cy, 1)
    dx = (xx - cx) / max(cx, 1)
    radius = np.sqrt(dy**2 + dx**2)
    angle = np.degrees(np.arctan2(dy, dx)) % 180.0

    # Mid-to-high band only. Low frequencies are dominated by scene content and
    # carry no information about the blur kernel.
    band = (radius > 0.35) & (radius < 0.9)
    if band.sum() < 32:
        return 0.0, 0.0

    n_bins = 18
    energy = np.zeros(n_bins)
    for index in range(n_bins):
        lo, hi = index * (180.0 / n_bins), (index + 1) * (180.0 / n_bins)
        mask = band & (angle >= lo) & (angle < hi)
        energy[index] = spectrum[mask].mean() if mask.any() else np.nan

    if np.isnan(energy).any():
        return 0.0, 0.0
    lo_e, hi_e = float(energy.min()), float(energy.max())
    if hi_e + lo_e <= 1e-12:
        return 0.0, 0.0
    anisotropy = (hi_e - lo_e) / (hi_e + lo_e)
    # The minimum-energy orientation is the direction the blur travelled.
    direction = float(np.argmin(energy) * (180.0 / n_bins) + (90.0 / n_bins))
    return float(np.clip(anisotropy, 0.0, 1.0)), direction


@dataclass(frozen=True)
class DegradationProfile:
    """Everything measured about one image, with confidences attached."""

    width: int
    height: int
    short_side: int

    mean_luma: float
    contrast: float
    # Raw Laplacian variance at native resolution: a measured property of the
    # stored pixels, kept for continuity of recorded profiles. NOT comparable
    # across resolutions -- anything that gates on sharpness must use
    # ``sharpness_match_scale``, which is measured at the fixed 112px scale.
    sharpness_laplacian: float
    sharpness_match_scale: float

    blur_sigma: float
    blur_confidence: float
    blur_kind: str
    blur_anisotropy: float
    blur_angle_deg: float

    noise_sigma: float

    jpeg_quality: int | None
    jpeg_confidence: float

    spectral_cutoff: float
    effective_resolution_ratio: float
    upscaled_beyond_content: bool

    monochrome_fraction: float
    infrared: bool

    interlace_ratio: float
    interlaced: bool

    clipped_low: float
    clipped_high: float

    small_face: bool
    estimator_trustworthy: bool
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["notes"] = list(self.notes)
        return payload


def analyze(pixels: np.ndarray) -> DegradationProfile:
    """Measure one image. uint8 RGB HWC in, profile out. Never modifies the input."""
    arr = np.asarray(pixels)
    height, width = arr.shape[:2]
    gray = _gray(arr)

    params, report = estimate_degradation(arr)

    cutoff = float(effective_cutoff(gray))
    # 0.5 cycles/px is Nyquist. Content well below it means the stored pixel
    # grid is finer than anything the image actually contains -- the classic
    # signature of a DVR that upscaled before writing, or of repeated re-encode.
    resolution_ratio = float(np.clip(cutoff / 0.5, 0.0, 1.0))

    anisotropy, angle = blur_orientation(arr)
    monochrome, is_ir = infrared_score(arr)
    comb, is_interlaced = interlace_score(arr)
    low, high = clipping(arr)

    blur_sigma = float(report["blur_sigma"])
    blur_conf = float(report["blur_confidence_r2"])
    if blur_sigma < 0.5 or blur_conf <= 0.5:
        kind = "none"
    elif anisotropy >= 0.35:
        kind = "motion"
    else:
        kind = "defocus"

    notes: list[str] = []
    if is_ir:
        notes.append(
            "Infrared capture: the original contains no colour information. Any colour "
            "in an enhanced image is synthetic."
        )
    if is_interlaced:
        notes.append(
            f"Interlacing detected (comb ratio {comb:.2f}). Fields are 1/50 s apart; "
            "sharpening before deinterlacing amplifies comb artifacts as if they were detail."
        )
    if resolution_ratio < 0.55:
        notes.append(
            f"Stored size {width}x{height} overstates true resolution: content is band-limited "
            f"at {cutoff:.3f} cycles/px ({resolution_ratio * 100:.0f}% of Nyquist). The image was "
            "probably upscaled or re-encoded before storage."
        )
    if high > 0.05:
        notes.append(
            f"{high * 100:.1f}% of pixels are clipped white. Saturated pixels carry no recoverable "
            "information; detail placed there by any model is invented."
        )
    if low > 0.10:
        notes.append(f"{low * 100:.1f}% of pixels are crushed black; shadow detail is not recoverable.")
    if min(height, width) < SMALL_FACE_PIXELS:
        notes.append(
            f"Short side is {min(height, width)}px, below the {SMALL_FACE_PIXELS}px floor at which "
            "recognition accuracy degrades sharply."
        )
    if not report["trustworthy"]:
        failed = [k for k, v in report["trust_components"].items() if not v]
        notes.append(
            "Degradation estimate is not fully trustworthy (" + ", ".join(failed) + "). "
            "Stages depending on the failed components are skipped rather than applied on a guess."
        )

    return DegradationProfile(
        width=int(width),
        height=int(height),
        short_side=int(min(height, width)),
        mean_luma=round(float(gray.mean() * 255.0), 2),
        contrast=round(float(gray.std() * 255.0), 2),
        sharpness_laplacian=round(float(laplacian_variance(gray * 255.0)), 2),
        sharpness_match_scale=round(
            float(
                match_scale_sharpness(
                    Image.fromarray(np.clip(gray * 255.0, 0.0, 255.0).astype(np.uint8))
                )
            ),
            2,
        ),
        blur_sigma=round(blur_sigma, 4),
        blur_confidence=round(blur_conf, 4),
        blur_kind=kind,
        blur_anisotropy=round(anisotropy, 4),
        blur_angle_deg=round(angle, 1),
        noise_sigma=round(float(report["noise_sigma"]), 6),
        jpeg_quality=params.jpeg_quality,
        jpeg_confidence=round(float(report["jpeg_confidence"]), 4),
        spectral_cutoff=round(cutoff, 6),
        effective_resolution_ratio=round(resolution_ratio, 4),
        upscaled_beyond_content=bool(resolution_ratio < 0.55),
        monochrome_fraction=round(monochrome, 4),
        infrared=is_ir,
        interlace_ratio=round(comb, 4),
        interlaced=is_interlaced,
        clipped_low=round(low, 5),
        clipped_high=round(high, 5),
        small_face=bool(min(height, width) < SMALL_FACE_PIXELS),
        estimator_trustworthy=bool(report["trustworthy"]),
        notes=tuple(notes),
    )


def quality_metrics(pixels: np.ndarray) -> dict[str, Any]:
    """Compact before/after scorecard. Same measurements, reported as scores.

    Deliberately reuses the aggregation weights nobody should be inventing twice:
    sharpness is measured at the fixed 112px match scale and normalised at 220
    Laplacian variance, contrast at 64 DN, matching ``ImageQualityFilter`` so a
    number here means the same thing it means there.
    """
    arr = np.asarray(pixels)
    gray = _gray(arr) * 255.0
    profile = analyze(arr)

    def clamp(value: float) -> float:
        return float(min(max(value, 0.0), 1.0))

    sharpness = clamp(profile.sharpness_match_scale / 220.0)
    contrast = clamp(profile.contrast / 64.0)
    brightness = 1.0 - clamp(abs(float(gray.mean()) - 128.0) / 128.0)
    resolution = clamp((profile.short_side - SMALL_FACE_PIXELS) / 160.0)
    noise = clamp(1.0 - profile.noise_sigma / 0.05)
    compression = 1.0 if profile.jpeg_quality is None else clamp(profile.jpeg_quality / 95.0)
    detail = profile.effective_resolution_ratio

    overall = (
        resolution * 0.20
        + sharpness * 0.20
        + detail * 0.20
        + noise * 0.15
        + contrast * 0.10
        + brightness * 0.10
        + compression * 0.05
    )

    return {
        "resolution": round(resolution, 4),
        "sharpness": round(sharpness, 4),
        "detail": round(detail, 4),
        "noise": round(noise, 4),
        "contrast": round(contrast, 4),
        "brightness": round(brightness, 4),
        "compression": round(compression, 4),
        "overall": round(overall, 4),
        "raw": {
            "width": profile.width,
            "height": profile.height,
            "laplacian_variance": profile.sharpness_laplacian,
            "laplacian_variance_match_scale": profile.sharpness_match_scale,
            "noise_sigma": profile.noise_sigma,
            "spectral_cutoff": profile.spectral_cutoff,
            "jpeg_quality": profile.jpeg_quality,
            "mean_luma": profile.mean_luma,
        },
    }


__all__ = [
    "SMALL_FACE_PIXELS",
    "DegradationProfile",
    "analyze",
    "blur_orientation",
    "clipping",
    "infrared_score",
    "interlace_score",
    "quality_metrics",
]
