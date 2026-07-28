from __future__ import annotations

import numpy as np
from PIL import Image

from ..utils import clamp


class DeepfakeDetector:
    """Heuristic screen for synthetic or heavily retouched face imagery.

    NOT a trained deepfake classifier. It reports two artefacts that generative
    upsamplers commonly leave behind:

    * unnaturally smooth skin, i.e. a collapse of mid-band frequency energy
      relative to low-band, which GAN and diffusion decoders produce when they
      hallucinate texture;
    * a spectral checkerboard, i.e. energy concentrated at the exact Nyquist
      corners, the fingerprint of transposed-convolution upsampling.

    A modern, well-post-processed synthetic face will defeat both. Use the score
    to prioritise examiner attention, never to conclude that media is authentic.
    """

    def __init__(self, alert_threshold: float = 0.65) -> None:
        self.alert_threshold = alert_threshold

    def risk_score(self, image: Image.Image) -> float:
        gray = np.asarray(image.convert("L").resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float64)
        gray = gray - gray.mean()
        spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray)))

        rows, cols = spectrum.shape
        row_index = np.arange(rows) - rows / 2.0
        col_index = np.arange(cols) - cols / 2.0
        radius = np.sqrt(row_index[:, None] ** 2 + col_index[None, :] ** 2)

        low = float(spectrum[radius <= 12].sum()) + 1e-9
        mid = float(spectrum[(radius > 12) & (radius <= 40)].sum())
        high = float(spectrum[radius > 40].sum())

        # Real photographs keep a healthy mid-band relative to the low band.
        smoothness = clamp(1.0 - (mid / low) / 0.55)

        # Upsampling artefacts pile energy into the spectrum's far corners.
        corner = float(spectrum[radius > min(rows, cols) * 0.45].sum())
        checkerboard = clamp((corner / (high + 1e-9)) / 0.35)

        return round(float(clamp(smoothness * 0.65 + checkerboard * 0.35)), 4)

    def flagged(self, image: Image.Image) -> bool:
        return self.risk_score(image) >= self.alert_threshold


__all__ = ["DeepfakeDetector"]
