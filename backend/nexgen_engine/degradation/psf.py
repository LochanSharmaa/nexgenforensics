"""Point-spread functions, MTF, and the forward degradation operator.

The PSF is what an imaging system does to a point of light. Its Fourier magnitude
is the MTF -- the fraction of contrast the system transmits at each spatial
frequency. MTF50, the frequency at which contrast falls to half, is the standard
single-number summary of how much detail a camera can actually deliver.

Why an identity system should care: **identity information lives at spatial
frequencies, and the MTF says which ones survived.** At 20 pixels across a face,
everything above a few cycles per face-width is gone -- not attenuated, gone.
Comparing two images at frequencies one of them never recorded is comparing noise
to signal, and it is a large part of why cross-resolution matching fails.

Parameters here are deliberately explicit and serialisable. For synthetic
degradation they are ground truth for the operator-supervision loss; for real
imagery they are estimated (see estimate.py) or, best of all, MEASURED from the
physical camera with a slanted-edge target.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass(frozen=True)
class PSF:
    """A sampled point-spread function, normalised to unit sum."""

    kernel: np.ndarray = field(repr=False)
    #: Physical scale if known: micrometres per sample. None for pixel units.
    micron_per_sample: float | None = None
    origin: str = "synthetic"  # synthetic | estimated | measured

    def __post_init__(self) -> None:
        k = np.asarray(self.kernel, dtype=np.float64)
        s = k.sum()
        object.__setattr__(self, "kernel", k / s if s > 0 else k)

    @property
    def support(self) -> int:
        return int(self.kernel.shape[0])


def gaussian_psf(sigma: float, size: int | None = None, origin: str = "synthetic") -> PSF:
    """Isotropic Gaussian PSF -- the standard defocus/diffraction approximation."""
    if sigma <= 0:
        return PSF(np.ones((1, 1)), origin=origin)
    size = size or max(3, int(2 * np.ceil(3 * sigma) + 1))
    if size % 2 == 0:
        size += 1
    ax = np.arange(size) - size // 2
    g = np.exp(-(ax**2) / (2 * sigma**2))
    return PSF(np.outer(g, g), origin=origin)


def motion_psf(length: float, angle_deg: float, origin: str = "synthetic") -> PSF:
    """Linear motion blur -- a line segment through the kernel centre."""
    length = max(float(length), 1.0)
    size = int(2 * np.ceil(length / 2) + 1)
    k = np.zeros((size, size), dtype=np.float64)
    c = size // 2
    theta = np.deg2rad(angle_deg)
    for t in np.linspace(-length / 2, length / 2, int(max(length * 4, 8))):
        y = int(round(c + t * np.sin(theta)))
        x = int(round(c + t * np.cos(theta)))
        if 0 <= y < size and 0 <= x < size:
            k[y, x] += 1.0
    if k.sum() == 0:
        k[c, c] = 1.0
    return PSF(k, origin=origin)


def mtf_from_psf(psf: PSF, n: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Radially-averaged MTF.

    Returns ``(frequency, mtf)`` with frequency in cycles/pixel on [0, 0.5],
    where 0.5 is Nyquist. MTF is normalised to 1 at DC.
    """
    k = np.zeros((n, n), dtype=np.float64)
    s = psf.support
    off = (n - s) // 2
    k[off : off + s, off : off + s] = psf.kernel
    otf = np.fft.fftshift(np.abs(np.fft.fft2(np.fft.ifftshift(k))))
    otf /= max(otf.max(), 1e-12)

    cy, cx = n // 2, n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    nbins = n // 2
    idx = np.clip(r.astype(int), 0, nbins - 1)
    prof = np.bincount(idx.ravel(), otf.ravel(), minlength=nbins)
    cnt = np.bincount(idx.ravel(), minlength=nbins)
    mtf = prof / np.maximum(cnt, 1)
    freq = np.arange(nbins) / n  # cycles per pixel
    return freq, mtf / max(mtf[0], 1e-12)


def mtf50(psf: PSF) -> float:
    """Frequency (cycles/pixel) at which contrast transmission falls to 50%.

    The conventional scalar summary of resolving power. Below this, detail is
    transmitted; above it, mostly not.
    """
    freq, mtf = mtf_from_psf(psf)
    below = np.flatnonzero(mtf < 0.5)
    if below.size == 0:
        return float(freq[-1])
    i = int(below[0])
    if i == 0:
        return 0.0
    # Linear interpolation between the bracketing samples.
    f0, f1 = freq[i - 1], freq[i]
    m0, m1 = mtf[i - 1], mtf[i]
    return float(f0 + (f1 - f0) * (m0 - 0.5) / max(m0 - m1, 1e-12))


@dataclass(frozen=True)
class DegradationParams:
    """The full forward operator, serialisable and auditable.

    For synthetic degradation these are ground truth and supervise the operator
    regression term of the training objective. For casework they are estimated,
    or measured from the seized device -- which is strictly better and is the
    reason forensic access to the camera is an asset nobody else exploits.
    """

    blur_sigma: float = 0.0
    motion_length: float = 0.0
    motion_angle_deg: float = 0.0
    downsample: float = 1.0
    noise_sigma: float = 0.0
    jpeg_quality: int | None = None
    origin: str = "synthetic"

    def psf(self) -> PSF:
        if self.motion_length > 1.0:
            m = motion_psf(self.motion_length, self.motion_angle_deg, self.origin)
            if self.blur_sigma > 0:
                g = gaussian_psf(self.blur_sigma, origin=self.origin)
                return PSF(_convolve2d(m.kernel, g.kernel), origin=self.origin)
            return m
        return gaussian_psf(self.blur_sigma, origin=self.origin)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["mtf50_cycles_per_pixel"] = round(mtf50(self.psf()), 6)
        return d


def _convolve2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Full 2-D convolution via FFT. Small kernels; no scipy dependency."""
    sh = (a.shape[0] + b.shape[0] - 1, a.shape[1] + b.shape[1] - 1)
    fa = np.fft.rfft2(a, sh)
    fb = np.fft.rfft2(b, sh)
    return np.fft.irfft2(fa * fb, sh)


def apply_forward(image: np.ndarray, params: DegradationParams, seed: int = 0) -> np.ndarray:
    """Apply the forward operator. There is no inverse in this package.

    Order follows the physical chain: optical blur, then sampling, then sensor
    noise, then compression. Applying them in a different order produces a
    different image, which is why the order is fixed here rather than left to
    the caller.
    """
    from .jpeg import JpegModel  # local import keeps the module graph acyclic
    from .noise import NoiseModel

    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:
        return np.stack([apply_forward(img[..., c], params, seed + c) for c in range(img.shape[2])], -1)

    k = params.psf().kernel
    if k.size > 1:
        img = _same_convolve(img, k)
    if params.downsample > 1.0:
        img = _decimate(img, params.downsample)
    if params.noise_sigma > 0:
        img = NoiseModel(read_sigma=params.noise_sigma).apply(img, seed=seed)
    if params.jpeg_quality is not None:
        img = JpegModel(quality=params.jpeg_quality).apply(img)
    return img


def _same_convolve(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    full = _convolve2d(img, k)
    oy, ox = (k.shape[0] - 1) // 2, (k.shape[1] - 1) // 2
    return full[oy : oy + img.shape[0], ox : ox + img.shape[1]]


def _decimate(img: np.ndarray, factor: float) -> np.ndarray:
    """Area-average decimation. Aliasing is intentional and physical: real
    undersampled sensors alias, and it is what makes multi-frame reconstruction
    informative rather than decorative."""
    h, w = img.shape
    nh, nw = max(1, int(h / factor)), max(1, int(w / factor))
    ys = (np.arange(nh) * h / nh).astype(int)
    xs = (np.arange(nw) * w / nw).astype(int)
    return img[np.ix_(ys, xs)]


__all__ = [
    "PSF",
    "DegradationParams",
    "apply_forward",
    "gaussian_psf",
    "motion_psf",
    "mtf50",
    "mtf_from_psf",
]
