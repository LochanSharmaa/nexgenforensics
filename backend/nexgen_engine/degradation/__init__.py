"""Forward imaging models: degradation as a known operator, not unknown corruption.

The organising principle of this whole subsystem:

    ALWAYS MOVE THE HYPOTHESIS TOWARD THE EVIDENCE.
    NEVER MOVE THE EVIDENCE TOWARD THE HYPOTHESIS.

A 20-pixel face is not a corrupted 200-pixel face waiting to be restored. It is
the output of a measurable optical chain -- lens PSF, sensor MTF, sampling,
motion, noise, compression -- applied to a real person. Estimating that chain and
applying it FORWARD to a hypothesis is well-posed. Inverting it to recover the
person is not, and every super-resolution model that appears to do so is
substituting a learned prior for information the sensor never captured. In a
forensic setting that means matching against the generator.

So this package provides operators that go one way only. There is no `restore()`,
no `upsample()`, no `enhance()`. Attempting to add one should be treated as an
architectural error, not a feature request.

  psf.py     point-spread functions and the MTF derived from them
  jpeg.py    quantization-table modelling and blocking artefacts
  noise.py   sensor noise: photon shot, read, and fixed-pattern
  estimate.py blind estimation of the above from a single observation
  bandlimit.py projection of two images onto their common passband

CPU only. No training. No GPU.
"""

from __future__ import annotations

from .bandlimit import common_passband, radial_power_spectrum, to_passband
from .jpeg import JpegModel, estimate_quality, quantization_matrix
from .noise import NoiseModel, estimate_noise_sigma
from .psf import PSF, DegradationParams, apply_forward, gaussian_psf, mtf_from_psf, mtf50

__all__ = [
    "PSF",
    "DegradationParams",
    "JpegModel",
    "NoiseModel",
    "apply_forward",
    "common_passband",
    "estimate_noise_sigma",
    "estimate_quality",
    "gaussian_psf",
    "mtf50",
    "mtf_from_psf",
    "quantization_matrix",
    "radial_power_spectrum",
    "to_passband",
]
