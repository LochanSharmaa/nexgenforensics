from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from ..utils import clamp


@dataclass(frozen=True)
class LivenessReport:
    score: float
    texture_energy: float
    moire_energy: float
    colour_spread: float
    passed: bool
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "score": self.score,
            "texture_energy": self.texture_energy,
            "moire_energy": self.moire_energy,
            "colour_spread": self.colour_spread,
            "passed": self.passed,
            "reasons": list(self.reasons),
            "certified": False,
            "method": "passive_single_frame_heuristic",
        }


class LivenessDetector:
    """Passive single-frame presentation-attack screening.

    IMPORTANT: this is a heuristic screen, NOT certified presentation-attack
    detection. It has not been evaluated against ISO/IEC 30107-3 and will not
    stop a determined attacker. It looks for two cheap but real signals:

    * print and replay attacks lose fine skin texture, so mid-band frequency
      energy collapses;
    * screen replays introduce moire, which appears as isolated periodic spikes
      in the high band of the spectrum.

    Genuine anti-spoofing needs active challenge-response, multi-frame analysis,
    or depth/IR capture. Treat a pass as "nothing obviously wrong" only, never
    as evidence that a live person was present.
    """

    def __init__(self, threshold: float = 0.45) -> None:
        self.threshold = threshold

    def analyze(self, image: Image.Image) -> LivenessReport:
        gray = np.asarray(image.convert("L").resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float64)
        gray = gray - gray.mean()

        spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
        total_energy = float(spectrum.sum()) + 1e-9

        radius = _radial_distance(spectrum.shape)
        mid_band = (radius > 12) & (radius <= 40)
        high_band = radius > 40

        # Live skin retains mid-frequency detail: pores, fine shadow gradients.
        texture_energy = float(spectrum[mid_band].sum()) / total_energy

        # Moire from a screen shows up as isolated spikes far above the local
        # median in the high band, rather than as broadband noise.
        high_values = spectrum[high_band]
        if high_values.size:
            median = float(np.median(high_values)) + 1e-9
            moire_energy = float(np.mean(high_values > median * 8.0))
        else:
            moire_energy = 0.0

        rgb = np.asarray(image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float64)
        colour_spread = float(rgb.reshape(-1, 3).std(axis=0).mean()) / 128.0

        texture_component = clamp(texture_energy / 0.35)
        moire_penalty = clamp(moire_energy / 0.08)
        colour_component = clamp(colour_spread)

        score = clamp(texture_component * 0.60 + colour_component * 0.25 + 0.15 - moire_penalty * 0.45)

        reasons: list[str] = []
        if texture_component < 0.30:
            reasons.append("low_skin_texture_detail")
        if moire_penalty > 0.50:
            reasons.append("possible_screen_replay_moire")
        if colour_component < 0.15:
            reasons.append("flat_colour_distribution")

        return LivenessReport(
            score=round(float(score), 4),
            texture_energy=round(texture_energy, 6),
            moire_energy=round(moire_energy, 6),
            colour_spread=round(colour_spread, 6),
            passed=score >= self.threshold,
            reasons=tuple(reasons),
        )

    def score_image(self, image: Image.Image) -> float:
        return self.analyze(image).score

    def passed(self, image: Image.Image) -> bool:
        return self.analyze(image).passed


def _radial_distance(shape: tuple[int, int]) -> np.ndarray:
    rows, cols = shape
    row_index = np.arange(rows) - rows / 2.0
    col_index = np.arange(cols) - cols / 2.0
    return np.sqrt(row_index[:, None] ** 2 + col_index[None, :] ** 2)


__all__ = ["LivenessDetector", "LivenessReport"]
