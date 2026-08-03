"""S0.3 -- the five arms. The experiment that decides whether Stage 3 is built.

THE QUESTION
------------
Does comparing in OBSERVATION SPACE beat comparing in EMBEDDING SPACE on degraded
imagery? If yes, the forward-model thesis has empirical support and a renderer is
worth building. If no, the generative core is expensive overhead and the whole
programme collapses to an evidence layer on ArcFace.

THE DESIGN CONSTRAINT THAT MAKES THIS CHEAP
-------------------------------------------
An earlier version of this experiment required building a renderer first. That is
two months of work AND -- worse -- a null result would be uninterpretable: did the
hypothesis fail, or was the renderer bad?

So: **a real high-resolution image IS the person model.** No renderer. The only
thing under test is where the comparison happens.

    A   embed LR probe, embed HR gallery, cosine        the current paradigm
    B1  downsample gallery to probe resolution, compare does condition-matching
                                                        alone help?
    B2  estimate THIS probe's operator, apply it to     does MODELLING the
        the gallery, compare                            operator beat matching?
    B3  project both onto their common MTF passband,    is it enough to stop
        compare                                         comparing absent freqs?
    C   likelihood in pixel space under the noise model does leaving embedding
                                                        space help at all?

**B2 - B1 is the number that decides the architecture.** B1 is "match the
resolution", a well-known trick. B2 is "model the physics". If modelling adds
nothing over matching, the physics story is not earning its keep.

GPU BOUNDARY
------------
Every arm is a pure image->image transform, all CPU. The ONLY GPU operation is
embedding the transformed images, which is injected as a callable. Pass
``stub_embedder`` to exercise the whole pipeline on CPU with zero inference --
that is how this framework is tested before any GPU time is spent.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

_BACKEND = Path(__file__).resolve().parents[2] / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from nexgen_engine.degradation.bandlimit import effective_cutoff, to_passband  # noqa: E402
from nexgen_engine.degradation.estimate import estimate_degradation  # noqa: E402
from nexgen_engine.degradation.psf import DegradationParams, apply_forward  # noqa: E402


class Embedder(Protocol):
    """Injected. The only component that needs a GPU."""

    def __call__(self, images: list[np.ndarray]) -> np.ndarray: ...


def stub_embedder(images: list[np.ndarray]) -> np.ndarray:
    """Deterministic CPU stand-in for the recognition model.

    Produces a 512-d vector from cheap image statistics. It is NOT a recogniser
    and its accuracy is meaningless -- its only job is to let the full pipeline,
    metrics, bootstrap and reporting run end to end on CPU so that the GPU run is
    debugging-free.
    """
    out = np.empty((len(images), 512), dtype=np.float64)
    for i, im in enumerate(images):
        x = np.asarray(im, dtype=np.float64)
        if x.ndim == 3:
            x = x.mean(axis=2)
        # Fixed-size descriptor: 16x16 block means + radial spectrum, then padded.
        h, w = x.shape
        ys = np.linspace(0, h, 17).astype(int)
        xs = np.linspace(0, w, 17).astype(int)
        blocks = np.array(
            [x[ys[a] : max(ys[a + 1], ys[a] + 1), xs[b] : max(xs[b + 1], xs[b] + 1)].mean()
             for a in range(16) for b in range(16)]
        )
        spec = np.abs(np.fft.rfft(blocks))[:256]
        v = np.concatenate([blocks, spec])[:512]
        v = np.pad(v, (0, max(0, 512 - v.size)))
        out[i] = v
    out -= out.mean(axis=1, keepdims=True)
    return out / np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-12, None)


# --------------------------------------------------------------------------- #
# Arms. Each returns (gallery_transformed, probe_transformed, report).
# --------------------------------------------------------------------------- #


def arm_A(gallery: np.ndarray, probe: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Baseline: no transform. The paradigm being challenged."""
    return gallery, probe, {"arm": "A", "transform": "none"}


def arm_B1(gallery: np.ndarray, probe: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Resolution match: decimate the gallery to the probe's pixel count.

    The well-known trick. Included precisely so B2 has to beat something real
    rather than an unfair baseline.
    """
    gh, gw = gallery.shape[:2]
    ph, pw = probe.shape[:2]
    factor = max(gh / max(ph, 1), 1.0)
    if factor <= 1.0:
        return gallery, probe, {"arm": "B1", "downsample": 1.0}
    reduced = apply_forward(gallery, DegradationParams(downsample=factor))
    return reduced, probe, {"arm": "B1", "downsample": round(factor, 4)}


def arm_B2(gallery: np.ndarray, probe: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Model the probe's actual operator and apply it forward to the gallery.

    The distinguishing arm. B1 matches pixel count; this matches the whole
    imaging chain -- blur, sampling, noise, compression -- estimated from the
    probe itself.

    ORDER MATTERS, AND GETTING IT WRONG IS SILENT. estimate_degradation()
    measures blur in the PROBE's pixel grid. An earlier version of this function
    passed that sigma to apply_forward() alongside the downsample factor, and
    apply_forward blurs BEFORE it decimates -- so a sigma of 0.6 probe-pixels was
    applied at full gallery resolution and then shrunk 8x, delivering ~0.07
    probe-pixels of blur instead of 0.6. Roughly an eighth of the intended
    operator, with nothing to indicate it.

    So the gallery is brought into the probe's sampling grid FIRST, and the
    estimated operator is applied there, in the units it was measured in.
    """
    params, report = estimate_degradation(probe)
    gh, ph = gallery.shape[0], probe.shape[0]
    factor = max(gh / max(ph, 1), 1.0)

    reduced = apply_forward(gallery, DegradationParams(downsample=factor)) if factor > 1.0 else gallery
    in_probe_space = DegradationParams(
        blur_sigma=params.blur_sigma,
        noise_sigma=params.noise_sigma,
        jpeg_quality=params.jpeg_quality,
        origin="estimated",
    )
    return (
        apply_forward(reduced, in_probe_space),
        probe,
        {
            "arm": "B2",
            "estimated": report,
            "downsample_first": round(factor, 4),
            "applied_in_probe_space": in_probe_space.as_dict(),
        },
    )


def arm_B2r(gallery: np.ndarray, probe: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """RESIDUAL forward operator: apply only what the probe suffered BEYOND the gallery.

    B2 applies the probe's ABSOLUTE operator to the gallery. That is correct only
    when the gallery is pristine. When both sides have already been through a
    similar imaging chain -- which is the surveillance-to-surveillance case, and
    is exactly TinyFace and QMUL -- re-applying the probe's full operator
    double-degrades the gallery and destroys information rather than matching
    condition.

    Measured on real imagery, both sides sit at essentially the same operating
    point (median estimated JPEG q=75 for TinyFace Gallery_Match AND Probe), so
    the residual there is close to nothing and B2 should be close to a no-op.
    That it instead cost 5-8 points is the signature of double application.

    The residual is computed in the natural composition law for each component:

        blur    Gaussian variances add, so sigma_r = sqrt(max(s_p^2 - s_g^2, 0))
        noise   independent noise powers add, same quadrature rule
        JPEG    quantisation is not invertible; applying a COARSER table is
                meaningful, applying a finer one is a no-op. So the residual is
                q_probe only when q_probe < q_gallery.
    """
    p_par, p_rep = estimate_degradation(probe)
    g_par, g_rep = estimate_degradation(gallery)

    gh, ph = gallery.shape[0], probe.shape[0]
    factor = max(gh / max(ph, 1), 1.0)

    # THE GALLERY'S BLUR MUST BE CONVERTED INTO THE PROBE'S PIXEL GRID BEFORE
    # THE QUADRATURE SUBTRACTION, OR THE RESIDUAL IS MEANINGLESS.
    #
    # estimate_degradation reports sigma in the pixel units of the image it was
    # given. A 448px gallery and a 100px probe are different grids, so their
    # sigmas are not comparable and subtracting them directly is the same units
    # error already fixed once in arm_B2.
    #
    # Measured on SCface, where it bit: gallery sigma 0.8482 (448px), probe
    # sigma 0.8028 (100px). Compared raw, the gallery looks BLURRIER than the
    # probe, so max(p^2 - g^2, 0) clamped the residual blur to exactly 0.0 and
    # B2r degenerated into "decimate + JPEG". In the probe's grid the gallery
    # blur is 0.8482 * 100/448 = 0.189, giving a true residual near 0.78.
    #
    # Scaling by 1/factor: a Gaussian of width sigma in a grid decimated by f
    # has width sigma/f in the new grid.
    g_sigma_in_probe_grid = g_par.blur_sigma / factor

    blur_r = float(np.sqrt(max(p_par.blur_sigma**2 - g_sigma_in_probe_grid**2, 0.0)))
    noise_r = float(np.sqrt(max(p_par.noise_sigma**2 - g_par.noise_sigma**2, 0.0)))
    jpeg_r = None
    if p_par.jpeg_quality is not None:
        if g_par.jpeg_quality is None or p_par.jpeg_quality < g_par.jpeg_quality:
            jpeg_r = p_par.jpeg_quality

    reduced = apply_forward(gallery, DegradationParams(downsample=factor)) if factor > 1.0 else gallery
    residual = DegradationParams(
        blur_sigma=blur_r, noise_sigma=noise_r, jpeg_quality=jpeg_r, origin="estimated_residual"
    )
    return (
        apply_forward(reduced, residual),
        probe,
        {
            "arm": "B2r",
            "probe_estimate": p_rep,
            "gallery_estimate": g_rep,
            "downsample_first": round(factor, 4),
            "gallery_blur_in_probe_grid": round(g_sigma_in_probe_grid, 6),
            "residual_applied": residual.as_dict(),
        },
    )


def arm_B3(gallery: np.ndarray, probe: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Common-passband projection. Never compare frequencies the channel lost.

    Delegates to bandlimit.match_passband(), which converts each side's
    cycles-per-PIXEL cutoff into cycles-per-FACE before taking the minimum. An
    earlier version of this function did that conversion itself and got it wrong:
    it took min() of two cutoffs measured in different pixel densities and then
    divided by the scale factor again, yielding an applied cutoff of ~0.0024
    cycles/pixel on the gallery and destroying essentially all content.
    """
    from nexgen_engine.degradation.bandlimit import match_passband

    g, p, report = match_passband(
        gallery, probe, face_width_hi=gallery.shape[0], face_width_lo=probe.shape[0]
    )
    return g, p, {"arm": "B3", **report}


def arm_C_score(gallery: np.ndarray, probe: np.ndarray, noise_sigma: float | None = None) -> tuple[float, dict]:
    """Pixel-space likelihood under the estimated noise model. No embedding.

    Returns a log-likelihood, not a cosine, so it is NOT comparable to the other
    arms on absolute scale -- only via rank-based metrics (AUC, TAR@FAR), which
    is how the runner treats it.
    """
    params, rep = estimate_degradation(probe)
    sigma = float(noise_sigma if noise_sigma is not None else max(params.noise_sigma, 1e-3))
    gh, ph = gallery.shape[0], probe.shape[0]
    rendered = apply_forward(
        gallery,
        DegradationParams(
            blur_sigma=params.blur_sigma,
            downsample=max(gh / max(ph, 1), 1.0),
            jpeg_quality=params.jpeg_quality,
            origin="estimated",
        ),
    )
    a = rendered[: probe.shape[0], : probe.shape[1]]
    b = probe[: a.shape[0], : a.shape[1]]
    if a.ndim == 3:
        a = a.mean(axis=2)
    if b.ndim == 3:
        b = b.mean(axis=2)
    # Contrast-normalise: a global gain/offset difference is a nuisance, not
    # identity evidence, and without this the score measures exposure.
    a = (a - a.mean()) / max(a.std(), 1e-6)
    b = (b - b.mean()) / max(b.std(), 1e-6)
    resid = a - b
    n = resid.size
    ll = float(-0.5 * np.sum(resid**2) / sigma**2 - n * np.log(sigma))
    return ll / n, {"arm": "C", "sigma": sigma, "pixels": int(n), "estimate": rep}


ARMS: dict[str, Callable] = {
    "A": arm_A,
    "B1": arm_B1,
    "B2": arm_B2,
    "B2r": arm_B2r,
    "B3": arm_B3,
}


@dataclass
class ArmResult:
    arm: str
    scores: np.ndarray = field(repr=False)
    labels: np.ndarray = field(repr=False)
    reports: list[dict] = field(default_factory=list, repr=False)


__all__ = [
    "ARMS",
    "ArmResult",
    "Embedder",
    "arm_A",
    "arm_B1",
    "arm_B2",
    "arm_B3",
    "arm_C_score",
    "stub_embedder",
]
