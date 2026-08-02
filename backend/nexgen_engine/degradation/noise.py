"""Sensor noise: the term that sets how much information a pixel actually carries.

Three physically distinct contributions, and conflating them is a common error:

    photon shot noise  Poisson, variance proportional to signal. Dominates in
                       daylight. Irreducible -- it is a property of light.
    read noise         Gaussian, signal-independent. Dominates in the dark, which
                       is exactly the CCTV night-footage regime.
    fixed pattern      Per-pixel gain/offset. Static, so it does NOT average away
                       across frames, which caps multi-frame gains.

The third is why "just use more frames" has a ceiling. Shot and read noise fall
as 1/sqrt(N); fixed-pattern noise does not fall at all. Any multi-frame claim
must account for it or it will overstate the information recovered.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NoiseModel:
    """Affine noise model in normalised [0, 1] intensity units."""

    read_sigma: float = 0.0
    #: Shot-noise coefficient: variance contribution is shot_gain * signal.
    shot_gain: float = 0.0
    #: Fixed-pattern noise standard deviation (multiplicative, per pixel).
    fpn_sigma: float = 0.0

    def apply(self, image: np.ndarray, seed: int = 0) -> np.ndarray:
        img = np.asarray(image, dtype=np.float64)
        rng = np.random.default_rng(seed)
        out = img.copy()
        if self.fpn_sigma > 0:
            # Deterministic per-pixel gain: same sensor, same pattern every frame.
            gain = np.random.default_rng(12345).normal(1.0, self.fpn_sigma, img.shape)
            out = out * gain
        if self.shot_gain > 0:
            out = out + rng.normal(0.0, np.sqrt(np.clip(out, 0, None) * self.shot_gain))
        if self.read_sigma > 0:
            out = out + rng.normal(0.0, self.read_sigma, img.shape)
        return out

    def variance_at(self, signal: float) -> float:
        return self.read_sigma**2 + self.shot_gain * max(signal, 0.0) + (self.fpn_sigma * signal) ** 2

    def snr_at(self, signal: float) -> float:
        v = self.variance_at(signal)
        return float(signal / np.sqrt(v)) if v > 0 else float("inf")

    def multiframe_variance(self, signal: float, n_frames: int) -> float:
        """Residual variance after averaging N frames.

        Shot and read noise average down; fixed-pattern noise does not. This is
        the ceiling on multi-frame information gain and the reason a hundred
        frames is not a hundred times the evidence.
        """
        n = max(int(n_frames), 1)
        averaging = (self.read_sigma**2 + self.shot_gain * max(signal, 0.0)) / n
        return averaging + (self.fpn_sigma * signal) ** 2


def estimate_noise_sigma(image: np.ndarray) -> float:
    """Blind noise estimate via the robust MAD of a Laplacian residual.

    Immerkaer's method: convolve with a kernel that annihilates locally-linear
    signal, leaving noise. Robust to texture, which a plain std over a flat patch
    is not.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:
        img = img.mean(axis=2)
    if min(img.shape) < 3:
        return 0.0
    k = np.array([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]])
    h, w = img.shape
    acc = np.zeros((h - 2, w - 2))
    for dy in range(3):
        for dx in range(3):
            acc += k[dy, dx] * img[dy : dy + h - 2, dx : dx + w - 2]
    # 1.4826 * MAD is a consistent sigma estimate for Gaussian data; the
    # sqrt(36) normalises for the kernel's L2 norm.
    mad = float(np.median(np.abs(acc - np.median(acc))))
    return float(1.4826 * mad / np.sqrt(36.0))


__all__ = ["NoiseModel", "estimate_noise_sigma"]
