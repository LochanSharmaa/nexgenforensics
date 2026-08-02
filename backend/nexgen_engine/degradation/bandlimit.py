"""Common-passband projection: never compare frequencies the channel did not pass.

This is the sharpest implementable form of the whole architecture's thesis, and
it needs no renderer, no training and no GPU.

The argument. A high-resolution gallery portrait contains spatial frequencies out
to Nyquist. A 20-pixel CCTV probe contains frequencies out to *its* far lower
cutoff, set by the lens PSF, the sensor MTF and the sampling. When those two
images are embedded and compared, the network sees detail in one that physically
cannot exist in the other. The resulting mismatch is not a modelling failure to
be trained away -- the information genuinely is not there -- and it is a large
part of why cross-resolution recognition collapses.

The remedy is symmetric and conservative: compute the passband both observations
share, project BOTH onto it, and compare there. Nothing is invented. The
high-resolution side is *reduced* to what the degraded side could have recorded.
Evidence is never lifted; the hypothesis is always lowered to meet it.

This is arm B3 of experiment S0.3. It is the cheapest test of the central thesis
available, and if it does not help, that is real evidence against the direction.
"""

from __future__ import annotations

import numpy as np

from .psf import PSF, mtf50


def radial_power_spectrum(image: np.ndarray, n_bins: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Radially-averaged power spectrum.

    Returns ``(frequency, power)`` with frequency in cycles/pixel on [0, 0.5].
    Used to find where an image's content disappears into its noise floor, which
    is a direct empirical read of its effective cutoff.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:
        img = img.mean(axis=2)
    img = img - img.mean()
    # Hann window: without it, edge discontinuities leak broadband energy and the
    # estimated cutoff comes out far too high.
    wy = np.hanning(img.shape[0])[:, None]
    wx = np.hanning(img.shape[1])[None, :]
    spec = np.abs(np.fft.fftshift(np.fft.fft2(img * wy * wx))) ** 2

    h, w = spec.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((yy - cy) / h) ** 2 + ((xx - cx) / w) ** 2)
    edges = np.linspace(0, 0.5, n_bins + 1)
    idx = np.clip(np.digitize(r.ravel(), edges) - 1, 0, n_bins - 1)
    power = np.bincount(idx, spec.ravel(), minlength=n_bins)
    count = np.bincount(idx, minlength=n_bins)
    return (edges[:-1] + edges[1:]) / 2, power / np.maximum(count, 1)


def effective_cutoff(image: np.ndarray, energy_fraction: float = 0.99) -> float:
    """Empirical cutoff: the frequency below which most spectral energy lies.

    Defined by CUMULATIVE energy rather than by a peak-relative floor. The first
    implementation used "first bin below 1% of the peak", which fails on any real
    image: natural spectra fall off steeply, so that criterion fires in the first
    or second bin regardless of how much high-frequency detail is present, and it
    reported the same cutoff for a sharp image and a heavily blurred one.

    The cumulative criterion is monotone under blurring by construction --
    removing high-frequency energy moves the 99% point down -- which is the
    property this function has to have to be usable at all.

    A blunt instrument even so: content and noise both influence it. Prefer a
    cutoff derived from a measured or estimated PSF when one is available.
    """
    freq, power = radial_power_spectrum(image)
    total = power.sum()
    if total <= 0:
        return 0.5
    cumulative = np.cumsum(power) / total
    reached = np.flatnonzero(cumulative >= energy_fraction)
    return float(freq[reached[0]]) if reached.size else 0.5


def common_passband(
    cutoff_a: float,
    cutoff_b: float,
    scale_a: float = 1.0,
    scale_b: float = 1.0,
) -> float:
    """The frequency band both observations can support, in cycles/pixel.

    ``scale_*`` converts each image's pixel-frequency into a common physical
    frame when the two were captured at different sampling densities -- e.g. a
    face 200 px wide versus one 25 px wide implies scale 8. Without this the two
    cutoffs are not comparable and the projection is meaningless.
    """
    return float(min(cutoff_a * scale_a, cutoff_b * scale_b))


def to_passband(image: np.ndarray, cutoff: float, soft: bool = True) -> np.ndarray:
    """Project onto frequencies at or below ``cutoff`` (cycles/pixel).

    ``soft`` applies a raised-cosine roll-off instead of a hard disc. A hard cut
    produces ringing that is itself a spurious high-frequency signal -- exactly
    the artefact this function exists to avoid introducing.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:
        return np.stack([to_passband(img[..., c], cutoff, soft) for c in range(img.shape[2])], -1)
    if cutoff >= 0.5:
        return img

    h, w = img.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    r = np.sqrt(fy**2 + fx**2)

    if soft:
        lo, hi = cutoff * 0.8, cutoff
        mask = np.clip((hi - r) / max(hi - lo, 1e-9), 0.0, 1.0)
        mask = 0.5 - 0.5 * np.cos(np.pi * mask)  # raised cosine
    else:
        mask = (r <= cutoff).astype(np.float64)
    return np.real(np.fft.ifft2(np.fft.fft2(img) * mask))


def match_passband(
    hi_res: np.ndarray,
    lo_res: np.ndarray,
    psf_lo: PSF | None = None,
    face_width_hi: float | None = None,
    face_width_lo: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Project both images onto their common passband. Arm B3 of S0.3.

    Returns ``(hi_projected, lo_projected, report)``. The report records the
    cutoffs and the scale factor so the operation is auditable rather than a
    silent preprocessing step.
    """
    cut_lo = mtf50(psf_lo) if psf_lo is not None else effective_cutoff(lo_res)
    cut_hi = effective_cutoff(hi_res)

    # Convert to a common physical frame. A cycle per PIXEL means something
    # different on a 200 px face than on a 25 px one, so both cutoffs are lifted
    # into cycles per FACE WIDTH before they can be compared at all. Face width is
    # used when known; image height is the fallback, which is correct whenever
    # both crops are framed alike.
    scale_hi = float(face_width_hi or hi_res.shape[0])
    scale_lo = float(face_width_lo or lo_res.shape[0])

    band = common_passband(cut_hi, cut_lo, scale_hi, scale_lo)
    report = {
        "cutoff_hi_cycles_per_pixel": round(cut_hi, 6),
        "cutoff_lo_cycles_per_pixel": round(cut_lo, 6),
        "scale_hi": scale_hi,
        "scale_lo": scale_lo,
        "common_passband_cycles_per_face": round(band, 6),
        "applied_cutoff_hi": round(band / max(scale_hi, 1e-9), 8),
        "applied_cutoff_lo": round(band / max(scale_lo, 1e-9), 8),
        "psf_origin": psf_lo.origin if psf_lo is not None else "estimated_from_spectrum",
    }
    return to_passband(hi_res, band / max(scale_hi, 1e-9)), to_passband(lo_res, band / max(scale_lo, 1e-9)), report


__all__ = [
    "common_passband",
    "effective_cutoff",
    "match_passband",
    "radial_power_spectrum",
    "to_passband",
]
