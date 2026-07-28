from __future__ import annotations

import numpy as np
from PIL import Image

from ..utils import clamp, cosine_similarity


class MorphingDetector:
    """Screens for face-morphing attacks, where one image blends two identities.

    Single-image mode looks for the ghosting a morph leaves behind: blending two
    faces produces double edges and locally inconsistent sharpness, so the
    variance of local sharpness across the crop rises while overall detail falls.
    This is weak on its own, and a well-made and retouched morph will pass.

    Differential mode (``differential_risk``) is far more reliable and should be
    preferred wherever a trusted live capture exists. A morph must resemble both
    contributors, so it sits at an unusual middle distance from a genuine
    reference -- close enough to pass verification, but never as close as a true
    same-person pair.
    """

    def __init__(self, alert_threshold: float = 0.55) -> None:
        self.alert_threshold = alert_threshold

    def risk_score(self, image: Image.Image) -> float:
        gray = np.asarray(image.convert("L").resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float64)
        if min(gray.shape) < 8:
            return 0.0

        # Gradient magnitude stands in for local edge strength.
        grad_y, grad_x = np.gradient(gray)
        magnitude = np.hypot(grad_x, grad_y)

        # Split into an 8x8 grid and compare block sharpness. A morph blends two
        # faces unevenly, so some regions stay crisp while others smear.
        blocks = magnitude.reshape(8, 16, 8, 16).mean(axis=(1, 3))
        mean_sharpness = float(blocks.mean()) + 1e-9
        sharpness_variation = float(blocks.std()) / mean_sharpness

        inconsistency = clamp(sharpness_variation / 0.75)
        softness = clamp(1.0 - mean_sharpness / 18.0)
        return round(float(clamp(inconsistency * 0.55 + softness * 0.45)), 4)

    def differential_risk(self, suspect_template: np.ndarray, trusted_template: np.ndarray) -> dict[str, float | bool]:
        """Compare a questioned image against a trusted live capture.

        A genuine pair scores high. An impostor scores low. A morph tends to land
        in the band between: similar enough to verify, but well short of the
        similarity a true same-person pair reaches.
        """
        similarity = cosine_similarity(suspect_template, trusted_template)
        in_morph_band = 0.28 <= similarity <= 0.45
        return {
            "similarity": round(float(similarity), 6),
            "in_morph_band": bool(in_morph_band),
            "risk": round(float(clamp(1.0 - abs(similarity - 0.36) / 0.36)) if in_morph_band else 0.0, 4),
        }

    def flagged(self, image: Image.Image) -> bool:
        return self.risk_score(image) >= self.alert_threshold


__all__ = ["MorphingDetector"]
