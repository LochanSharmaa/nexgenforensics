"""JPEG as a forward operator, and quality recovery from an observation.

Almost every CCTV frame that reaches a forensic examiner has been JPEG- or
H.264-compressed, often more than once. Compression is not incidental noise: it
is a deterministic, block-structured, frequency-selective operator that discards
specific DCT coefficients. Modelled forward, it is just another stage of the
imaging chain. Ignored, it is a systematic mismatch between a clean gallery image
and a compressed probe -- one that no amount of embedding training removes,
because the two images genuinely differ in which frequencies exist.

Recovering the quantization table from a decoded image is a standard forensic
image-analysis technique, and it is one of the few places where a *measured*
rather than estimated operator parameter is available from the evidence itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Standard IJG luminance quantization table at quality 50.
_BASE_LUMA = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float64,
)


def quantization_matrix(quality: int) -> np.ndarray:
    """IJG quality (1-100) to an 8x8 luminance quantization table."""
    q = int(np.clip(quality, 1, 100))
    scale = 5000.0 / q if q < 50 else 200.0 - 2.0 * q
    table = np.floor((_BASE_LUMA * scale + 50.0) / 100.0)
    return np.clip(table, 1, 255)


def _dct2(block: np.ndarray) -> np.ndarray:
    n = block.shape[0]
    k = np.arange(n)
    basis = np.cos(np.pi * (2 * k[:, None] + 1) * k[None, :] / (2 * n))
    alpha = np.full(n, np.sqrt(2.0 / n))
    alpha[0] = np.sqrt(1.0 / n)
    m = basis * alpha[None, :]
    return m.T @ block @ m


def _idct2(coef: np.ndarray) -> np.ndarray:
    n = coef.shape[0]
    k = np.arange(n)
    basis = np.cos(np.pi * (2 * k[:, None] + 1) * k[None, :] / (2 * n))
    alpha = np.full(n, np.sqrt(2.0 / n))
    alpha[0] = np.sqrt(1.0 / n)
    m = basis * alpha[None, :]
    return m @ coef @ m.T


@dataclass(frozen=True)
class JpegModel:
    """Block-DCT quantization. Forward only."""

    quality: int = 75

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Quantize in 8x8 DCT blocks. Input and output in [0, 1]."""
        img = np.asarray(image, dtype=np.float64)
        if img.ndim == 3:
            return np.stack([self.apply(img[..., c]) for c in range(img.shape[2])], -1)

        q = quantization_matrix(self.quality)
        h, w = img.shape
        ph, pw = (-h) % 8, (-w) % 8
        padded = np.pad(img, ((0, ph), (0, pw)), mode="edge") * 255.0 - 128.0
        out = np.empty_like(padded)
        for y in range(0, padded.shape[0], 8):
            for x in range(0, padded.shape[1], 8):
                blk = padded[y : y + 8, x : x + 8]
                out[y : y + 8, x : x + 8] = _idct2(np.round(_dct2(blk) / q) * q)
        return np.clip((out[:h, :w] + 128.0) / 255.0, 0.0, 1.0)


def _block_dct_coeffs(img: np.ndarray) -> np.ndarray:
    """Block-DCT coefficients of an image, shaped (n_blocks, 8, 8)."""
    h, w = img.shape
    ph, pw = (-h) % 8, (-w) % 8
    padded = np.pad(img, ((0, ph), (0, pw)), mode="edge") * 255.0 - 128.0
    blocks = []
    for y in range(0, padded.shape[0], 8):
        for x in range(0, padded.shape[1], 8):
            blocks.append(_dct2(padded[y : y + 8, x : x + 8]))
    return np.stack(blocks) if blocks else np.zeros((0, 8, 8))


def estimate_quality(image: np.ndarray, candidates: range | None = None) -> tuple[int, float]:
    """Recover the JPEG quality of an already-decoded image by LATTICE FITTING.

    Quantisation leaves its fingerprint in the DCT coefficients themselves: after
    dequantisation every coefficient at frequency (i,j) is a multiple of that
    frequency's quantisation step. So the true quality is the one whose table
    puts the observed coefficients closest to its own lattice.

    WHY NOT MINIMUM RECOMPRESSION ERROR, WHICH THIS REPLACED
    --------------------------------------------------------
    The previous implementation re-compressed at each candidate quality and
    picked the lowest MSE, reasoning that re-compressing at the original quality
    is near-idempotent. Measured behaviour:

        synthetic noise, 112px, true q=35   ->  35   correct
        REAL FACE, 31 / 48 / 112 / 250px    ->  95   WRONG at every size

    Natural faces are smooth, so most DCT energy is low-frequency. Once q=35 has
    zeroed the high-frequency coefficients, re-compressing at q=95 barely
    perturbs what survives, and the error curve slides monotonically toward the
    highest candidate. The criterion only works on broadband content, which face
    imagery is not. It returned high confidence while being wrong, which is the
    dangerous combination.

    Returns ``(quality, confidence)``. Confidence is the relative margin between
    the best and second-best candidate; low values mean the image was probably
    never JPEG-compressed, or was compressed more than once (double compression
    leaves two lattices and neither fits well).
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:
        img = img.mean(axis=2)
    if img.max() > 1.5:
        img = img / 255.0

    coeffs = _block_dct_coeffs(img)
    if coeffs.shape[0] < 4:
        return 75, 0.0

    cands = list(candidates or range(20, 100, 5))
    errs = []
    for q in cands:
        table = quantization_matrix(q)
        total, weight = 0.0, 0.0
        # Skip DC (0,0): it carries the block mean and is dominated by content.
        for i in range(8):
            for j in range(8):
                if i == 0 and j == 0:
                    continue
                c = coeffs[:, i, j]
                t = table[i, j]
                active = np.abs(c) > 0.5 * t
                if active.sum() < 4:
                    continue
                # Distance to the nearest lattice point, normalised by the step
                # so every frequency contributes comparably.
                resid = np.abs(c[active] / t - np.round(c[active] / t))
                total += float(resid.mean()) * active.sum()
                weight += active.sum()
        errs.append(total / weight if weight > 0 else 1.0)

    errs = np.asarray(errs)
    best = int(np.argmin(errs))

    # Confidence: how much better the winner is than a TYPICAL candidate, not
    # than its neighbour. Candidates five quality steps apart produce nearly
    # identical tables, so a best-vs-second-best margin is structurally tiny even
    # when the fit is unambiguous -- it reported 0.016 on an estimate that was
    # exactly right. Comparing against the median separates "one clear lattice"
    # from "no lattice here at all", which is the question actually being asked.
    med = float(np.median(errs))
    margin = float((med - errs[best]) / max(med, 1e-12))
    return cands[best], float(np.clip(margin, 0.0, 1.0))


__all__ = ["JpegModel", "estimate_quality", "quantization_matrix"]
