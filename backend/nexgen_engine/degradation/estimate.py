"""Blind estimation of the forward operator from a single observation.

Estimating the operator is well-posed; inverting it is not. This module only ever
does the former. Every function answers "what did the imaging chain do?" and none
answers "what was the original?".

Accuracy expectations, stated plainly so nobody over-trusts these numbers:
blur sigma is recoverable to roughly +/-30% on textured content and much worse on
flat faces; JPEG quality recovery is reliable for single compression and
degrades sharply under recompression; noise sigma is the most robust of the
three. All three are strictly worse than measuring the physical camera, which is
why the capture programme in DATA_REQUIREMENTS.md exists.
"""

from __future__ import annotations

import numpy as np

from .bandlimit import effective_cutoff, radial_power_spectrum
from .jpeg import estimate_quality
from .noise import estimate_noise_sigma
from .psf import DegradationParams, gaussian_psf, mtf50


def estimate_blur_sigma(image: np.ndarray, max_sigma: float = 6.0) -> tuple[float, float]:
    """Recover Gaussian blur sigma by matching the observed spectral falloff.

    A Gaussian PSF of width sigma attenuates frequency f by exp(-2 pi^2 sigma^2 f^2).
    Fitting that decay to the measured radial power spectrum recovers sigma
    without needing a reference image.

    Returns ``(sigma, confidence)`` where confidence is the fit R^2. Below about
    0.5 the estimate should not be trusted -- typically flat or heavily
    compressed content where the spectrum carries no usable slope.
    """
    freq, power = radial_power_spectrum(image)
    keep = (freq > 0.02) & (power > 0)
    if keep.sum() < 8:
        return 0.0, 0.0

    f = freq[keep]
    y = np.log(power[keep] / power[keep].max())
    # log P(f) = c - (2 pi sigma f)^2 ... regress on f^2.
    x = f**2
    A = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope = coef[0]
    resid = y - A @ coef
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = float(1.0 - (resid**2).sum() / ss_tot) if ss_tot > 0 else 0.0

    if slope >= 0:
        return 0.0, max(r2, 0.0)
    sigma = float(np.sqrt(-slope / (4.0 * np.pi**2)))
    return float(np.clip(sigma, 0.0, max_sigma)), float(np.clip(r2, 0.0, 1.0))


def estimate_degradation(image: np.ndarray, assume_jpeg: bool = True) -> tuple[DegradationParams, dict]:
    """Full blind estimate of the forward operator. Arm B2 of experiment S0.3.

    Returns the parameters plus a confidence report. The report is not optional
    garnish: an operator applied with false confidence produces a confidently
    wrong likelihood, which is the most dangerous failure mode in the system.
    """
    raw = np.asarray(image, dtype=np.float64)
    if raw.max() > 1.5:
        raw = raw / 255.0
    img = raw.mean(axis=2) if raw.ndim == 3 else raw

    # JPEG estimation runs on a SINGLE channel, never the channel mean.
    #
    # Quantisation is applied per channel, so each channel carries its own DCT
    # lattice. Averaging them superimposes three lattices and largely cancels the
    # structure the estimator looks for. Measured on 100 degraded LFW faces:
    #
    #     channel mean      76.0% correct,  median confidence 0.126,  31% accepted
    #     green channel    100.0% correct,  median confidence 0.525, 100% accepted
    #
    # Blur and noise are still estimated on the mean, where averaging suppresses
    # noise and helps rather than hurts.
    jpeg_src = raw[:, :, 1] if raw.ndim == 3 else raw

    sigma, sigma_conf = estimate_blur_sigma(img)
    noise = estimate_noise_sigma(img)
    quality, q_conf = (estimate_quality(jpeg_src) if assume_jpeg else (None, 0.0))
    cutoff = effective_cutoff(img)

    # JPEG is applied only when the lattice fit is decisive. The threshold is
    # 0.15, not the old 0.05: a weak margin means the image was probably never
    # JPEG-compressed, or was compressed twice (two lattices, neither fitting).
    jpeg_ok = quality is not None and q_conf > 0.15

    params = DegradationParams(
        blur_sigma=sigma,
        noise_sigma=noise,
        jpeg_quality=quality if jpeg_ok else None,
        origin="estimated",
    )
    # TRUSTWORTHINESS IS A CONJUNCTION, NOT A SINGLE FIT STATISTIC.
    #
    # An earlier version returned trustworthy=True whenever the blur fit had
    # R^2 > 0.5. That flag was True on an estimate whose JPEG quality was 95
    # against a true value of 35 -- because R^2 measures how well a Gaussian
    # falloff describes the spectrum and says nothing whatever about the
    # compression estimate. A confident wrong operator is the most dangerous
    # output this module can produce, so every component must clear its own bar.
    trustworthy = bool(sigma_conf > 0.5 and (jpeg_ok or not assume_jpeg) and min(img.shape) >= 24)

    report = {
        "blur_sigma": round(sigma, 4),
        "blur_confidence_r2": round(sigma_conf, 4),
        "noise_sigma": round(noise, 6),
        "jpeg_quality": quality,
        "jpeg_confidence": round(q_conf, 4),
        "jpeg_applied": params.jpeg_quality is not None,
        "jpeg_method": "DCT lattice fit",
        "spectral_cutoff_cycles_per_pixel": round(cutoff, 6),
        "psf_mtf50": round(mtf50(gaussian_psf(sigma)), 6) if sigma > 0 else 0.5,
        "trustworthy": trustworthy,
        "trust_components": {
            "blur_fit_ok": bool(sigma_conf > 0.5),
            "jpeg_decisive": bool(jpeg_ok),
            "size_ok": bool(min(img.shape) >= 24),
        },
        "caveat": (
            "Estimated, not measured. Blur recovery on flat facial content is "
            "unreliable; prefer a measured camera response when the device is "
            "available."
        ),
    }
    return params, report


__all__ = ["estimate_blur_sigma", "estimate_degradation"]
