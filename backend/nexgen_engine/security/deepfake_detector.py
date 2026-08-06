from __future__ import annotations

import re
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter

from ..detection.types import FaceBox
from ..utils import clamp

# ---------------------------------------------------------------------------
# Metadata markers.
#
# GENERATIVE markers are decisive: a tool wrote its own name (or the IPTC
# trainedAlgorithmicMedia source type) into the file. Nothing else in this
# module can reach that level of certainty, so a hit short-circuits the fused
# score to "high". Matched with word boundaries against metadata text only --
# never against pixel content or filenames -- so a surname or a street name
# cannot fire them.
#
# RETOUCH markers are AI face-editing apps: not proof of a swapped identity,
# but proof the face passed through a generative beautifier, which already
# invalidates the image as a forensic exhibit of "what the camera saw".
#
# EDITOR markers are conventional editing software. Informational: a scan
# cropped in Photoshop is still a genuine photograph.
# ---------------------------------------------------------------------------

_GENERATIVE_MARKERS = (
    "stable diffusion", "stable-diffusion", "sd-webui", "sdxl", "automatic1111",
    "comfyui", "invokeai", "novelai", "midjourney", "dall-e", "dall·e",
    "adobe firefly", "flux.1", "black-forest-labs", "ideogram", "leonardo.ai",
    "recraft", "krea.ai", "runwayml", "luma.ai", "grok imagine",
    "stylegan", "stargan", "gaugan", "thispersondoesnotexist",
    "deepfacelab", "faceswap", "facefusion", "simswap", "inswapper", "roop",
    "wav2lip", "heygen", "d-id.com", "synthesia",
    "trainedalgorithmicmedia",
)

# PNG text chunks that generation front-ends write their prompt/settings into.
# The key itself is the evidence; the value does not need to name a tool.
_GENERATION_CHUNK_KEYS = (
    "parameters", "prompt", "workflow", "sd-metadata", "invokeai_metadata",
    "generation_data", "dream",
)

_AI_RETOUCH_MARKERS = (
    "faceapp", "facetune", "remini", "beautyplus", "meitu", "youcam",
    "gfpgan", "codeformer", "topaz photo",
)

_EDITOR_MARKERS = (
    "photoshop", "gimp", "lightroom", "affinity photo", "paintshop",
    "pixelmator", "picsart", "canva", "snapseed",
)


def _find_marker(corpus: str, markers: tuple[str, ...]) -> str | None:
    for marker in markers:
        if re.search(rf"(?<![a-z0-9]){re.escape(marker)}(?![a-z0-9])", corpus):
            return marker
    return None


@dataclass(frozen=True)
class SignalReading:
    """One forensic signal. ``weight`` 0 means informational only."""

    name: str
    score: float
    weight: float
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "score": self.score,
            "weight": self.weight,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class DeepfakeReport:
    """Structured outcome of the synthetic-media screen.

    ``score`` is the fused 0-1 risk; ``band`` maps it to the words an examiner
    reads (minimal / moderate / elevated / high). ``signals`` keeps every
    individual reading so a flag is always explainable, and ``reasons`` uses
    the same stable snake_case vocabulary the rest of the engine emits.
    """

    score: float
    band: str
    flagged: bool
    review_advised: bool
    signals: tuple[SignalReading, ...]
    reasons: tuple[str, ...]
    analyzed_pixels: int
    method: str = "multi_signal_media_forensics_v2"

    def as_dict(self) -> dict[str, object]:
        return {
            "score": self.score,
            "band": self.band,
            "flagged": self.flagged,
            "review_advised": self.review_advised,
            "signals": [signal.as_dict() for signal in self.signals],
            "reasons": list(self.reasons),
            "analyzed_pixels": self.analyzed_pixels,
            "method": self.method,
            "certified": False,
        }


_REASON_BY_SIGNAL = {
    "spectral_peaks": "periodic_upsampling_artefacts",
    "spectral_slope": "spectral_energy_anomaly",
    "texture": "synthetic_texture_smoothness",
    "noise_floor": "sensor_noise_absent",
    "noise_consistency": "noise_pattern_inconsistent",
    "ela": "ela_region_anomaly",
    "boundary": "face_boundary_blending_artefacts",
}


class DeepfakeDetector:
    """Multi-signal screen for synthetic or manipulated face imagery.

    NOT a trained deepfake classifier and NOT certified media authentication.
    It fuses independent classical forensics signals, each of which a careful
    adversary can defeat individually:

    * provenance metadata -- generator tags, prompt chunks, C2PA / IPTC
      trainedAlgorithmicMedia markers, camera EXIF presence or absence;
    * frequency-domain analysis of the face region at native resolution --
      power-spectrum slope, isolated periodic peaks left by upsamplers,
      mid-band texture collapse;
    * sensor-noise analysis -- a real photograph carries spatially consistent
      sensor noise; generated imagery has none, and a swapped face carries
      different noise from the background it was pasted into;
    * error-level analysis (JPEG sources) -- the face region recompresses
      differently from the rest when it did not share the same JPEG history;
    * boundary analysis -- blending a swapped face leaves an unnaturally
      smooth ring between face and background.

    The fused score prioritises examiner attention. It must never be reported
    as proof of authenticity or forgery: a clean pass means "no indicator
    found", nothing stronger.

    Unlike the v1 screen this runs on the ORIGINAL image around the detected
    face box, not on the aligned 112 px recognition crop -- downscaling to the
    recognition input destroys exactly the high-frequency evidence these
    signals rely on.
    """

    def __init__(self, alert_threshold: float = 0.65, review_threshold: float = 0.45) -> None:
        self.alert_threshold = alert_threshold
        self.review_threshold = min(review_threshold, alert_threshold)

    # ------------------------------------------------------------------ api --

    def analyze(
        self,
        image: Image.Image,
        face_box: FaceBox | None = None,
        raw_bytes: bytes | None = None,
    ) -> DeepfakeReport:
        rgb = image.convert("RGB")
        box = face_box.clipped(rgb.width, rgb.height) if face_box is not None else None
        if box is not None and (box.width < 16 or box.height < 16):
            box = None

        provenance, decisive = self._provenance(image, raw_bytes)
        readings: list[SignalReading] = [provenance]

        # All pixel work happens inside a window around the face, cropped at
        # NATIVE resolution -- never a full-frame conversion. On a 12 MP frame
        # the full-frame path cost ~425 ms; the window path costs ~10 ms on
        # typical uploads with identical readings, because every signal only
        # ever looked at the face and its surrounding background anyway.
        rect, wbox = _window_rect(rgb.width, rgb.height, box)
        window_rgb = rgb.crop(rect)
        window = np.asarray(window_rgb.convert("L"), dtype=np.float64)
        analyzed_pixels = min(box.width, box.height) if box else min(window.shape)

        patch = _center_patch(window, wbox)
        if patch is not None:
            readings.extend(self._spectral(patch))
        readings.extend(self._noise(window, wbox))
        readings.append(self._ela(window_rgb, wbox, raw_bytes))
        readings.append(self._boundary(window, wbox))

        active = [reading for reading in readings if reading.weight > 0]
        if active:
            total_weight = sum(reading.weight for reading in active)
            mean = sum(reading.score * reading.weight for reading in active) / total_weight
            # One strong physical signal must not drown in the average of the
            # quiet ones: a spectral checkerboard alone is a real finding.
            strongest = max(
                (reading.score for reading in active if reading.weight >= 0.5), default=0.0
            )
            fused = max(mean, 0.8 * strongest)
        else:
            fused = 0.0
        if decisive:
            fused = max(fused, 0.96)

        score = round(float(clamp(fused)), 4)
        flagged = score >= self.alert_threshold
        review = score >= self.review_threshold

        reasons: list[str] = []
        if flagged:
            reasons.append("synthetic_media_risk")
        elif review:
            reasons.append("synthetic_media_review_advised")
        if decisive:
            reasons.append("generative_metadata_present")
        elif provenance.name == "provenance" and provenance.score >= 0.6 and provenance.weight > 0:
            reasons.append("ai_retouch_metadata_present")
        for reading in readings:
            if reading.weight > 0 and reading.score >= 0.6:
                reason = _REASON_BY_SIGNAL.get(reading.name)
                if reason:
                    reasons.append(reason)
                if reading.name == "noise_consistency" and "mismatch" in reading.detail:
                    reasons.append("face_background_noise_mismatch")

        return DeepfakeReport(
            score=score,
            band=self._band(score),
            flagged=flagged,
            review_advised=review,
            signals=tuple(readings),
            reasons=tuple(dict.fromkeys(reasons)),
            analyzed_pixels=int(analyzed_pixels),
        )

    def risk_score(self, image: Image.Image) -> float:
        return self.analyze(image).score

    def flagged(self, image: Image.Image) -> bool:
        return self.analyze(image).flagged

    def _band(self, score: float) -> str:
        if score >= self.alert_threshold:
            return "high"
        if score >= self.review_threshold:
            return "elevated"
        if score >= 0.25:
            return "moderate"
        return "minimal"

    # -------------------------------------------------------------- signals --

    def _provenance(
        self, image: Image.Image, raw_bytes: bytes | None
    ) -> tuple[SignalReading, bool]:
        texts: list[str] = []
        info = image.info or {}
        for key, value in info.items():
            if isinstance(value, bytes):
                texts.append(f"{key}={value[:4096].decode('latin-1', 'ignore')}")
            elif isinstance(value, str):
                texts.append(f"{key}={value[:4096]}")

        generation_chunk = next((key for key in _GENERATION_CHUNK_KEYS if key in info), None)

        make = model = ""
        try:
            exif = image.getexif()
            make = str(exif.get(271, "")).strip()
            model = str(exif.get(272, "")).strip()
            for tag in (305, 315, 306):
                value = exif.get(tag)
                if value:
                    texts.append(str(value))
            # Generation front-ends that export JPEG put the prompt in the Exif
            # IFD's UserComment, not in a PNG chunk.
            comment = exif.get_ifd(0x8769).get(37510)
            if isinstance(comment, bytes):
                body = comment[8:] if comment[:8].rstrip(b"\x00") in (b"UNICODE", b"ASCII") else comment
                decoded = body.decode("utf-8", "ignore")
                if decoded.count("\x00") > len(decoded) // 4:
                    decoded = body.decode("utf-16-le", "ignore")
                texts.append(decoded.replace("\x00", ""))
            elif isinstance(comment, str):
                texts.append(comment)
        except Exception:  # noqa: BLE001 - malformed EXIF must never break a screen
            pass

        c2pa_present = False
        trained_media = False
        if raw_bytes:
            head = raw_bytes[:262144].lower()
            c2pa_present = b"c2pa" in head or b"jumbf" in head
            trained_media = b"trainedalgorithmicmedia" in head

        corpus = " ".join(texts).lower()
        generative = _find_marker(corpus, _GENERATIVE_MARKERS)
        retouch = _find_marker(corpus, _AI_RETOUCH_MARKERS)
        editor = _find_marker(corpus, _EDITOR_MARKERS)

        if generative or generation_chunk or trained_media:
            evidence = generative or (
                f"generation chunk '{generation_chunk}'" if generation_chunk else "trainedAlgorithmicMedia"
            )
            detail = f"generator provenance in file metadata: {evidence}"
            if c2pa_present:
                detail += "; C2PA manifest present"
            return SignalReading("provenance", 0.98, 3.0, detail), True
        if retouch:
            return (
                SignalReading(
                    "provenance", 0.62, 1.2, f"AI face-retouch application in metadata: {retouch}"
                ),
                False,
            )
        if editor:
            return (
                SignalReading(
                    "provenance",
                    0.3,
                    0.5,
                    f"editing software in metadata: {editor} (editing, not necessarily generation)",
                ),
                False,
            )
        if make or model:
            return (
                SignalReading(
                    "provenance", 0.05, 0.6, f"camera EXIF present ({make} {model})".strip()
                ),
                False,
            )
        return (
            SignalReading(
                "provenance",
                0.18,
                0.3,
                "no capture metadata (consistent with stripping, screenshots, or generation)",
            ),
            False,
        )

    def _spectral(self, patch: np.ndarray) -> list[SignalReading]:
        side = patch.shape[0]
        # Reduced-resolution faces carry less usable frequency evidence, so the
        # readings keep their scores but lose influence on the fused number.
        reliability = 1.0 if side >= 96 else (0.7 if side >= 64 else 0.35)
        hann = np.hanning(side)
        window = hann[:, None] * hann[None, :]
        spectrum = np.abs(np.fft.fftshift(np.fft.fft2((patch - patch.mean()) * window)))
        power = spectrum**2 + 1e-12
        radius = _radial_distance(power.shape)
        rmax = side // 2 - 1

        # Slope is measured on a median-filtered copy: film grain and high-ISO
        # noise flatten the raw spectrum of a GENUINE photograph the same way
        # hallucinated texture does, and the 3x3 median removes exactly that
        # impulse content while leaving structured fake texture in place.
        despeckled = np.asarray(
            Image.fromarray(patch.astype(np.uint8)).filter(ImageFilter.MedianFilter(3)),
            dtype=np.float64,
        )
        despeckled_power = (
            np.abs(np.fft.fftshift(np.fft.fft2((despeckled - despeckled.mean()) * window))) ** 2
            + 1e-12
        )
        r_int = radius.astype(np.int64).ravel()
        counts = np.bincount(r_int)
        radii = np.arange(2, rmax + 1)
        sums = np.bincount(r_int, weights=despeckled_power.ravel())
        profile = sums[2 : rmax + 1] / np.maximum(counts[2 : rmax + 1], 1)
        slope, _ = np.polyfit(np.log10(radii), np.log10(profile), 1)

        # Natural photographs follow a power law: log-power falls roughly
        # linearly in log-frequency. Bounds measured on AgeDB photographic
        # portraits (despeckled slopes -4.4 .. -3.0). Too flat means broadband
        # texture no lens produced; the steep side is capped at 0.5 because
        # steepness alone is blur -- a quality problem, not proof of synthesis.
        if slope > -2.7:
            deviation = clamp((slope + 2.7) / 0.8)
        elif slope < -4.6:
            deviation = min(0.5, clamp((-4.6 - slope) / 0.8))
        else:
            deviation = 0.0
        slope_reading = SignalReading(
            "spectral_slope",
            round(float(deviation), 4),
            0.8 * reliability,
            f"despeckled power-spectrum slope {slope:.2f} (photographic range ≈ -4.6 to -2.7)",
        )

        # Upsampling grids and transposed convolutions betray themselves in
        # two spectral shapes, each with its own detector:
        #
        # * a DENSE COMB (grid replication of detailed content): periodic
        #   log-spectrum profiles, exposed by autocorrelation. Photographs --
        #   including heavy JPEG recompression -- measured at or below 0.18.
        # * an ISOLATED TONE (checkerboard artefact): a single spectral line
        #   standing far above everything else at its radius, exposed by the
        #   annulus-relative excess. Photographs measured at or below ~4.3
        #   robust sigmas; synthetic checkerboards exceed 8.
        log_spectrum = np.log10(power)
        periodicity = max(
            _profile_periodicity(log_spectrum.mean(axis=0)),
            _profile_periodicity(log_spectrum.mean(axis=1)),
        )
        line_excess = _spectral_line_excess(log_spectrum, radius, rmax)
        peaks_score = max(
            clamp((periodicity - 0.18) / 0.14),
            clamp((line_excess - 4.6) / 2.2),
        )
        peaks_reading = SignalReading(
            "spectral_peaks",
            round(float(peaks_score), 4),
            1.2 * reliability,
            f"spectral periodicity {periodicity:.3f} (photographic ≤ ~0.18), "
            f"strongest isolated tone {line_excess:.1f}σ above its band (photographic ≤ ~4.3σ)",
        )

        amplitude_low = float(spectrum[radius <= rmax * 0.19].sum()) + 1e-9
        amplitude_mid = float(spectrum[(radius > rmax * 0.19) & (radius <= rmax * 0.63)].sum())
        smoothness = clamp(1.0 - (amplitude_mid / amplitude_low) / 0.55)
        texture_reading = SignalReading(
            "texture",
            round(float(smoothness), 4),
            0.9 * reliability,
            f"mid/low band energy ratio {amplitude_mid / amplitude_low:.3f}",
        )
        return [slope_reading, peaks_reading, texture_reading]

    def _noise(
        self, window: np.ndarray, wbox: tuple[int, int, int, int] | None
    ) -> list[SignalReading]:
        block = 16
        rows, cols = window.shape[0] // block, window.shape[1] // block
        if rows < 6 or cols < 6:
            return [SignalReading("noise_floor", 0.0, 0.0, "region too small for noise analysis")]

        trimmed = window[: rows * block, : cols * block]
        median = np.asarray(
            Image.fromarray(trimmed.astype(np.uint8)).filter(ImageFilter.MedianFilter(3)),
            dtype=np.float64,
        )
        residual = trimmed - median
        grad_y, grad_x = np.gradient(trimmed)
        gradient = np.hypot(grad_x, grad_y)

        residual_std = residual.reshape(rows, block, cols, block).std(axis=(1, 3))
        gradient_mean = gradient.reshape(rows, block, cols, block).mean(axis=(1, 3))

        # Noise must be estimated where there is no texture to leak into the
        # residual, so only the flatter half of the blocks is used.
        smooth = gradient_mean <= np.percentile(gradient_mean, 55)
        levels = residual_std[smooth]
        if levels.size < 8:
            return [SignalReading("noise_floor", 0.0, 0.0, "no flat regions to estimate noise from")]

        level = float(np.median(levels))
        # A vanished noise floor also happens under aggressive compression, so
        # this signal is capped below the alert threshold: it can push a score
        # up but never flag an image alone.
        floor_score = min(0.6, float(clamp((1.1 - level) / 1.1)))
        readings = [
            SignalReading(
                "noise_floor",
                round(floor_score, 4),
                0.8,
                f"sensor-noise level {level:.2f} in flat regions (denoised, generated, or heavily compressed when ≈ 0)",
            )
        ]

        # Dispersion is symmetric (it cannot tell which region is the fake), so
        # like the floor it is capped below the alert threshold. Only the
        # directional face/background mismatch below may fully flag.
        q75, q25 = np.percentile(levels, [75, 25])
        dispersion = float((q75 - q25) / (level + 1e-9))
        consistency_score = min(0.7, float(clamp((dispersion - 1.2) / 1.6)))
        detail = f"noise dispersion {dispersion:.2f} across flat blocks"

        if wbox is not None:
            left, top, right, bottom = wbox
            centers_y = (np.arange(rows) + 0.5) * block
            centers_x = (np.arange(cols) + 0.5) * block
            inside = (
                (centers_y[:, None] >= top)
                & (centers_y[:, None] < bottom)
                & (centers_x[None, :] >= left)
                & (centers_x[None, :] < right)
            )
            inside_levels = residual_std[smooth & inside]
            outside_levels = residual_std[smooth & ~inside]
            if inside_levels.size >= 4 and outside_levels.size >= 8:
                face_level = float(np.median(inside_levels)) + 1e-9
                background_level = float(np.median(outside_levels)) + 1e-9
                # DIRECTIONAL by design: a swapped or generated face is the
                # denoised part pasted onto a background that kept its sensor
                # noise. The opposite direction -- face noisier than a flat
                # background -- is ordinary photography (bokeh, studio walls,
                # sky) and must not fire.
                ratio = background_level / face_level
                if background_level >= 0.7:
                    mismatch = clamp((ratio - 1.6) / 1.6)
                    if mismatch > consistency_score:
                        consistency_score = mismatch
                        detail = (
                            f"face/background noise mismatch: {face_level:.2f} inside the face vs "
                            f"{background_level:.2f} outside ({ratio:.1f}x quieter face)"
                        )

        readings.append(
            SignalReading("noise_consistency", round(float(consistency_score), 4), 1.0, detail)
        )
        return readings

    def _ela(
        self,
        window_rgb: Image.Image,
        wbox: tuple[int, int, int, int] | None,
        raw_bytes: bytes | None,
    ) -> SignalReading:
        if raw_bytes is None or not raw_bytes.startswith(b"\xff\xd8\xff"):
            return SignalReading("ela", 0.0, 0.0, "not applicable (source is not a JPEG)")
        if wbox is None:
            return SignalReading("ela", 0.0, 0.0, "not applicable (no face region to compare)")

        buffer = BytesIO()
        window_rgb.save(buffer, "JPEG", quality=88)
        buffer.seek(0)
        original = np.asarray(window_rgb, dtype=np.float64)
        recompressed = np.asarray(Image.open(buffer).convert("RGB"), dtype=np.float64)
        difference = np.abs(original - recompressed).mean(axis=2)

        block = 16
        rows, cols = difference.shape[0] // block, difference.shape[1] // block
        if rows < 4 or cols < 4:
            return SignalReading("ela", 0.0, 0.0, "image too small for error-level analysis")
        block_means = (
            difference[: rows * block, : cols * block]
            .reshape(rows, block, cols, block)
            .mean(axis=(1, 3))
        )

        left, top, right, bottom = wbox
        centers_y = (np.arange(rows) + 0.5) * block
        centers_x = (np.arange(cols) + 0.5) * block
        inside = (
            (centers_y[:, None] >= top)
            & (centers_y[:, None] < bottom)
            & (centers_x[None, :] >= left)
            & (centers_x[None, :] < right)
        )
        if inside.sum() < 4 or (~inside).sum() < 8:
            return SignalReading("ela", 0.0, 0.0, "face fills the frame; no background to compare")

        face_level = float(np.median(block_means[inside])) + 1e-6
        background_level = float(np.median(block_means[~inside])) + 1e-6
        ratio = max(face_level, background_level) / min(face_level, background_level)
        score = clamp((ratio - 1.9) / 2.3)
        return SignalReading(
            "ela",
            round(float(score), 4),
            0.9,
            f"recompression error {face_level:.2f} in face vs {background_level:.2f} in background ({ratio:.1f}x)",
        )

    def _boundary(
        self, window: np.ndarray, wbox: tuple[int, int, int, int] | None
    ) -> SignalReading:
        if wbox is None:
            return SignalReading("boundary", 0.0, 0.0, "not applicable (no face box)")
        left, top, right, bottom = wbox
        width, height = right - left, bottom - top
        if width < 48 or height < 48:
            return SignalReading("boundary", 0.0, 0.0, "face too small for boundary analysis")

        grad_y, grad_x = np.gradient(window)
        gradient = np.hypot(grad_x, grad_y)
        rows, cols = window.shape

        shrink_x, shrink_y = int(width * 0.2), int(height * 0.2)
        inner = gradient[top + shrink_y : bottom - shrink_y, left + shrink_x : right - shrink_x]

        expand_x, expand_y = int(width * 0.22), int(height * 0.22)
        outer_top, outer_bottom = max(0, top - expand_y), min(rows, bottom + expand_y)
        outer_left, outer_right = max(0, left - expand_x), min(cols, right + expand_x)
        ring_mask = np.zeros_like(gradient, dtype=bool)
        ring_mask[outer_top:outer_bottom, outer_left:outer_right] = True
        ring_mask[top:bottom, left:right] = False
        beyond = np.ones_like(ring_mask)
        beyond[outer_top:outer_bottom, outer_left:outer_right] = False

        if inner.size < 256 or ring_mask.sum() < 256 or beyond.sum() < 0.1 * gradient.size:
            return SignalReading("boundary", 0.0, 0.0, "insufficient background around the face")

        inner_energy = float(inner.mean())
        ring_energy = float(gradient[ring_mask].mean())
        outer_energy = float(gradient[beyond].mean())
        reference = min(inner_energy, outer_energy) + 1e-9
        softness = 1.0 - ring_energy / reference
        score = clamp((softness - 0.4) / 0.45)
        return SignalReading(
            "boundary",
            round(float(score), 4),
            0.5,
            f"edge energy — face {inner_energy:.1f}, transition ring {ring_energy:.1f}, background {outer_energy:.1f}",
        )


# ------------------------------------------------------------------ helpers --


def _window_rect(
    width: int, height: int, box: FaceBox | None, cap: int = 1024
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int] | None]:
    """Analysis window around the face at NATIVE resolution, as a crop rect.

    Downscaling a 12 MP frame before analysis would average away the sensor
    noise and spectral evidence, so instead a window of at most ``cap`` px is
    cropped around the face box (with margin for background comparison).
    Returns the PIL-style (left, top, right, bottom) rect plus the face box
    translated into window coordinates.
    """
    if box is None:
        side_x, side_y = min(width, 768), min(height, 768)
        left = (width - side_x) // 2
        top = (height - side_y) // 2
        return (left, top, left + side_x, top + side_y), None

    margin = int(0.8 * max(box.width, box.height))
    top = max(0, box.top - margin)
    bottom = min(height, box.bottom + margin)
    left = max(0, box.left - margin)
    right = min(width, box.right + margin)

    # Cap the window while keeping the face inside it.
    if bottom - top > cap:
        center = (box.top + box.bottom) // 2
        top = max(0, min(center - cap // 2, height - cap))
        bottom = top + cap
    if right - left > cap:
        center = (box.left + box.right) // 2
        left = max(0, min(center - cap // 2, width - cap))
        right = left + cap

    wbox = (
        max(0, box.left - left),
        max(0, box.top - top),
        min(right - left, box.right - left),
        min(bottom - top, box.bottom - top),
    )
    return (left, top, right, bottom), wbox


def _center_patch(
    window: np.ndarray, wbox: tuple[int, int, int, int] | None, cap: int = 256
) -> np.ndarray | None:
    """Square patch over the face itself for frequency analysis.

    A crop, never a resize: resampling both destroys generator artefacts and
    manufactures new spectral content of its own.
    """
    height, width = window.shape
    if wbox is None:
        side = min(height, width, cap)
        center_y, center_x = height // 2, width // 2
    else:
        left, top, right, bottom = wbox
        side = min(right - left, bottom - top, cap)
        center_y, center_x = (top + bottom) // 2, (left + right) // 2
    if side < 32:
        return None
    half = side // 2
    side = half * 2
    top = min(max(center_y - half, 0), height - side)
    left = min(max(center_x - half, 0), width - side)
    return window[top : top + side, left : left + side]


def _profile_periodicity(profile: np.ndarray) -> float:
    """Peak autocorrelation of a detrended log-spectrum profile.

    A periodic comb -- the fingerprint of grid upsampling and transposed
    convolutions -- autocorrelates strongly at the comb period. The lag
    window starts at 6, NOT lower: any smooth spectrum autocorrelates highly
    at tiny lags (measured 0.47 at lag 3 on defocused content), which is
    generic smoothness, not a comb. Within this window photographs -- however
    heavily JPEG-compressed -- measured at or below 0.18; grid upsampling
    reaches 0.2-0.4 and generator checkerboards more.
    """
    size = profile.size
    kernel_width = max(5, size // 16)
    trend = np.convolve(profile, np.ones(kernel_width) / kernel_width, mode="same")
    detrended = profile - trend
    detrended = detrended - detrended.mean()
    denominator = float(np.sum(detrended * detrended)) + 1e-12
    autocorr = np.correlate(detrended, detrended, mode="full")[size - 1 :] / denominator
    low, high = 6, size // 2 - 6
    if high <= low:
        return 0.0
    return float(np.max(autocorr[low:high]))


def _spectral_line_excess(log_spectrum: np.ndarray, radius: np.ndarray, rmax: int) -> float:
    """Strongest isolated spectral tone, in robust sigmas above its annulus.

    A checkerboard artefact concentrates energy into a single frequency pair,
    which no amount of radial averaging will surface -- but compared against
    the median of its own annulus it stands out by definition. Directional
    edges in real photographs form ridges through DC that raise a whole
    annulus sector, not a lone bin, so their excess stays moderate.
    """
    r_int = radius.astype(np.int64)
    best = 0.0
    for r in range(max(4, int(rmax * 0.45)), rmax):
        ring = log_spectrum[r_int == r]
        if ring.size < 12:
            continue
        median = float(np.median(ring))
        mad = float(np.median(np.abs(ring - median))) + 1e-9
        best = max(best, (float(ring.max()) - median) / (1.4826 * mad))
    return best


def _radial_distance(shape: tuple[int, int]) -> np.ndarray:
    rows, cols = shape
    row_index = np.arange(rows) - rows / 2.0
    col_index = np.arange(cols) - cols / 2.0
    return np.sqrt(row_index[:, None] ** 2 + col_index[None, :] ** 2)


__all__ = ["DeepfakeDetector", "DeepfakeReport", "SignalReading"]
