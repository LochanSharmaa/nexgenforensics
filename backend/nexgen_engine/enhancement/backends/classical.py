"""Deterministic operators with no learned prior and no weights.

These are Track A in the strict sense: every one of them is a fixed arithmetic
operation on the pixels that were captured. None can produce structure that was
not present in the input. That property is what allows their output to be shown
as a "processed image" rather than as a reconstruction, and it is why they
remain the default even when the learned backends are available.

They are also the reason the module works at all on a host with no GPU, no torch
and no downloaded weights -- which is the state of any fresh clone.

One shared discipline: **monochrome in, monochrome out.** When the analysis
stage reports infrared capture, every operator here works on luma and
replicates. An operator that let chroma drift on an IR frame would be inventing
colour that the sensor never recorded, which is exactly the failure the IR
detection exists to prevent.
"""

from __future__ import annotations

import numpy as np

from ...degradation.bandlimit import to_passband
from ..registry import BackendSpec, EnhancementBackend, register
from ..types import Task, Track


def _cv2():
    import cv2  # noqa: PLC0415

    return cv2


def _cv2_available() -> tuple[bool, str]:
    try:
        _cv2()
    except Exception as exc:  # pragma: no cover - host-specific
        return False, f"opencv is not importable: {exc}"
    return True, ""


def _as_float(pixels: np.ndarray) -> np.ndarray:
    return np.asarray(pixels, dtype=np.float32) / 255.0


def _as_uint8(pixels: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(np.asarray(pixels, dtype=np.float64) * 255.0), 0, 255).astype(np.uint8)


def _luma(pixels: np.ndarray) -> np.ndarray:
    arr = np.asarray(pixels, dtype=np.float32)
    return arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114


def _enforce_monochrome(out: np.ndarray, monochrome: bool) -> np.ndarray:
    """Collapse to luma and replicate when the source carried no colour."""
    if not monochrome:
        return out
    gray = _luma(out)
    return np.repeat(gray[:, :, None], 3, axis=2).astype(out.dtype)


# --------------------------------------------------------------------------- #
# Deinterlace
# --------------------------------------------------------------------------- #


@register(
    BackendSpec(
        name="classical_deinterlace",
        track=Track.MEASUREMENT,
        task=Task.DEINTERLACE,
        version="1.0",
        summary="Field separation: keep one field and resample, rather than blending two instants together.",
        deterministic=True,
        default_parameters={"method": "field_separation", "field": "even"},
    )
)
class ClassicalDeinterlace(EnhancementBackend):
    """Separate fields instead of blending them.

    A blend (yadif and friends) averages two moments 1/50 s apart and produces
    one plausible frame. That is fine for viewing and wrong for us: it discards
    a genuine temporal sample that multi-frame fusion could have used, and it
    smears a moving face.

    Keeping one field halves the vertical resolution and then resamples back.
    The resampling adds no information and is honest about that -- unlike a
    blend, which looks sharper while being a composite of two instants.
    """

    def availability(self) -> tuple[bool, str]:
        return _cv2_available()

    def apply(self, pixels: np.ndarray, parameters: dict) -> np.ndarray:
        cv2 = _cv2()
        height = pixels.shape[0]
        if height < 4:
            return pixels
        offset = 1 if parameters.get("field") == "odd" else 0
        field = pixels[offset::2]
        # INTER_LINEAR, not LANCZOS: the field is already the full extent of the
        # available vertical detail, and a sharp kernel here rings on the edges
        # the missing lines left behind.
        restored = cv2.resize(field, (pixels.shape[1], height), interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(restored.astype(np.uint8))


# --------------------------------------------------------------------------- #
# Deblock
# --------------------------------------------------------------------------- #


@register(
    BackendSpec(
        name="classical_deblock",
        track=Track.MEASUREMENT,
        task=Task.DEBLOCK,
        version="1.0",
        summary="Edge-preserving smoothing weighted toward the 8x8 transform grid.",
        deterministic=True,
        default_parameters={"quality": 50, "grid": 8},
    )
)
class ClassicalDeblock(EnhancementBackend):
    """Attack block edges specifically, not the whole image.

    A global smooth removes blocking and the detail with it. Blocking lives on a
    known lattice -- the 8x8 transform grid -- so the correction is applied with
    a weight that peaks on the grid lines and falls to near zero between them.
    Detail away from block boundaries is left alone.

    Strength scales with the *estimated* quality factor. That estimate is
    confidence-gated upstream, so this stage only ever runs when the lattice fit
    was decisive.
    """

    def availability(self) -> tuple[bool, str]:
        return _cv2_available()

    def apply(self, pixels: np.ndarray, parameters: dict) -> np.ndarray:
        cv2 = _cv2()
        quality = float(parameters.get("quality", 50))
        grid = int(parameters.get("grid", 8))

        # q=95 -> ~0.05 strength, q=20 -> ~0.79. Linear in the useful range and
        # clamped so a wild estimate cannot smear the image flat.
        strength = float(np.clip((95.0 - quality) / 95.0, 0.0, 0.85))
        if strength <= 0.01:
            return pixels

        smoothed = cv2.bilateralFilter(pixels, d=5, sigmaColor=float(24 * strength + 8), sigmaSpace=5)

        height, width = pixels.shape[:2]
        # Weight peaks on the lattice lines and decays within the block.
        col = np.minimum(np.arange(width) % grid, (-np.arange(width)) % grid).astype(np.float32)
        row = np.minimum(np.arange(height) % grid, (-np.arange(height)) % grid).astype(np.float32)
        col_w = np.clip(1.0 - col / 2.0, 0.0, 1.0)
        row_w = np.clip(1.0 - row / 2.0, 0.0, 1.0)
        weight = np.maximum(col_w[None, :], row_w[:, None]) * strength

        blended = pixels.astype(np.float32) * (1 - weight[..., None]) + smoothed.astype(np.float32) * weight[..., None]
        out = np.clip(blended, 0, 255).astype(np.uint8)
        return _enforce_monochrome(out, bool(parameters.get("monochrome", False)))


# --------------------------------------------------------------------------- #
# Denoise
# --------------------------------------------------------------------------- #


@register(
    BackendSpec(
        name="classical_denoise",
        track=Track.MEASUREMENT,
        task=Task.DENOISE,
        version="1.0",
        summary="Non-local means, strength driven by the measured noise sigma.",
        deterministic=True,
        default_parameters={"sigma": 0.02},
    )
)
class ClassicalDenoise(EnhancementBackend):
    """Non-local means. Deterministic, classical, and defensible in testimony.

    NLM is kept as the default rather than a learned denoiser for a specific
    reason: when an opposing expert asks what was done to the image, "non-local
    means with filter strength h derived from a measured noise sigma" is a
    complete and checkable answer. A neural denoiser's answer is a checkpoint
    hash.
    """

    def availability(self) -> tuple[bool, str]:
        ok, reason = _cv2_available()
        if not ok:
            return ok, reason
        cv2 = _cv2()
        for attr in ("fastNlMeansDenoisingColored", "fastNlMeansDenoising"):
            if not hasattr(cv2, attr):  # pragma: no cover - opencv build-specific
                return False, f"opencv build lacks {attr} (photo module not compiled in)"
        return True, ""

    def apply(self, pixels: np.ndarray, parameters: dict) -> np.ndarray:
        cv2 = _cv2()
        sigma = float(parameters.get("sigma", 0.02))
        monochrome = bool(parameters.get("monochrome", False))

        # h is the NLM filter strength in DN. Measured sigma is in 0..1 units,
        # so scale to DN and apply a mild factor: over-filtering a face costs
        # exactly the fine texture identity depends on.
        h = float(np.clip(sigma * 255.0 * 1.10, 1.5, 22.0))

        if monochrome:
            gray = np.ascontiguousarray(_luma(pixels).astype(np.uint8))
            out_gray = cv2.fastNlMeansDenoising(gray, None, h, 7, 21)
            return np.repeat(out_gray[:, :, None], 3, axis=2)

        out = cv2.fastNlMeansDenoisingColored(pixels, None, h, h, 7, 21)
        return np.ascontiguousarray(out.astype(np.uint8))


# --------------------------------------------------------------------------- #
# Tone / illumination
# --------------------------------------------------------------------------- #


@register(
    BackendSpec(
        name="classical_tone",
        track=Track.MEASUREMENT,
        task=Task.TONE,
        version="1.0",
        summary="CLAHE on luminance only, with clipped pixels left untouched.",
        deterministic=True,
        default_parameters={"clip_limit": 2.0, "tile": 8, "preserve_clipped": True},
    )
)
class ClassicalTone(EnhancementBackend):
    """Local contrast on the luminance channel, chroma untouched.

    Two properties make this safe to run on evidence:

    * It is a monotone remapping of luminance within each tile. It cannot
      introduce an edge that was not there -- only change the mapping of values
      that were.
    * Clipped pixels are restored to their clipped value afterwards. Stretching
      a saturated well produces texture out of nothing, which is precisely the
      invention the analysis stage flags. Saturation stays visible as saturation.
    """

    def availability(self) -> tuple[bool, str]:
        return _cv2_available()

    def apply(self, pixels: np.ndarray, parameters: dict) -> np.ndarray:
        cv2 = _cv2()
        clip_limit = float(parameters.get("clip_limit", 2.0))
        tile = int(parameters.get("tile", 8))
        monochrome = bool(parameters.get("monochrome", False))
        preserve = bool(parameters.get("preserve_clipped", True))

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))

        if monochrome:
            gray = np.ascontiguousarray(_luma(pixels).astype(np.uint8))
            out = np.repeat(clahe.apply(gray)[:, :, None], 3, axis=2)
        else:
            lab = cv2.cvtColor(pixels, cv2.COLOR_RGB2LAB)
            lab[..., 0] = clahe.apply(lab[..., 0])
            out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        if preserve:
            blown = pixels >= 253
            crushed = pixels <= 2
            out = np.where(blown, pixels, out)
            out = np.where(crushed, pixels, out)
        return np.ascontiguousarray(out.astype(np.uint8))


# --------------------------------------------------------------------------- #
# Deblur
# --------------------------------------------------------------------------- #


@register(
    BackendSpec(
        name="classical_deblur",
        track=Track.MEASUREMENT,
        task=Task.DEBLUR,
        version="1.0",
        summary="Unsharp masking whose correction is band-limited to the measured spectral cut-off.",
        deterministic=True,
        default_parameters={"sigma": 1.0, "max_gain": 2.0, "kind": "defocus"},
    )
)
class ClassicalDeblur(EnhancementBackend):
    """Sharpening that provably cannot exceed the band the sensor recorded.

    This is the stage where a Track A operator could quietly stop being Track A.
    Ordinary unsharp masking boosts *all* frequencies present in the difference
    image, including the ones that are pure noise above the optical cut-off, and
    the result looks like recovered detail.

    So the correction term is projected onto the measured passband before it is
    added, using ``degradation.bandlimit.to_passband`` -- the same function the
    S0.3 arms use. Above ``cutoff`` the correction is identically zero, by
    construction rather than by tuning. Whatever this stage produces above the
    cut-off is nothing, which is the correct amount.
    """

    def availability(self) -> tuple[bool, str]:
        return _cv2_available()

    def apply(self, pixels: np.ndarray, parameters: dict) -> np.ndarray:
        cv2 = _cv2()
        sigma = float(np.clip(parameters.get("sigma", 1.0), 0.3, 4.0))
        max_gain = float(np.clip(parameters.get("max_gain", 2.0), 0.0, 3.0))
        cutoff = float(parameters.get("cutoff", 0.5))
        monochrome = bool(parameters.get("monochrome", False))

        if max_gain <= 0.0:
            return pixels

        work = _as_float(pixels)
        blurred = cv2.GaussianBlur(work, (0, 0), sigmaX=sigma, sigmaY=sigma)
        detail = work - blurred

        # The whole point of this backend. Zero the correction above the band
        # the original actually contained.
        if 0.0 < cutoff < 0.5:
            detail = np.stack(
                [to_passband(detail[..., c].astype(np.float64), cutoff) for c in range(detail.shape[2])],
                axis=-1,
            ).astype(np.float32)

        out = np.clip(work + max_gain * detail, 0.0, 1.0)
        return _enforce_monochrome(_as_uint8(out), monochrome)


# --------------------------------------------------------------------------- #
# Upscale
# --------------------------------------------------------------------------- #


@register(
    BackendSpec(
        name="classical_upscale",
        track=Track.MEASUREMENT,
        task=Task.UPSCALE,
        version="1.0",
        summary="Lanczos resampling. Enlarges without inventing; the fallback when no SR weights are present.",
        deterministic=True,
        default_parameters={"scale": 2},
    )
)
class ClassicalUpscale(EnhancementBackend):
    """Lanczos interpolation.

    Track A, and the distinction is worth being precise about: interpolation
    adds no information, but it also adds no *prior*. The output contains
    exactly the frequencies the input contained, resampled onto a finer grid. A
    learned super-resolver adds plausible frequencies that were never measured.
    Both make the image bigger; only one of them makes something up.
    """

    def availability(self) -> tuple[bool, str]:
        return _cv2_available()

    def scale_factor(self, parameters: dict) -> float:
        return float(parameters.get("scale", 2))

    def apply(self, pixels: np.ndarray, parameters: dict) -> np.ndarray:
        cv2 = _cv2()
        scale = float(np.clip(parameters.get("scale", 2), 1.0, 8.0))
        if scale <= 1.0:
            return pixels
        height, width = pixels.shape[:2]
        out = cv2.resize(
            pixels,
            (int(round(width * scale)), int(round(height * scale))),
            interpolation=cv2.INTER_LANCZOS4,
        )
        return _enforce_monochrome(
            np.ascontiguousarray(out.astype(np.uint8)), bool(parameters.get("monochrome", False))
        )


__all__ = [
    "ClassicalDeblock",
    "ClassicalDeblur",
    "ClassicalDeinterlace",
    "ClassicalDenoise",
    "ClassicalTone",
    "ClassicalUpscale",
]
