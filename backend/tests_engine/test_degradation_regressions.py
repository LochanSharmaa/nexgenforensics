"""Regression tests for the six degradation-layer defects found on 2026-08-02.

Each test pins a failure that was actually observed, produced a wrong scientific
number, and was fixed the same day. If any of these regress, the S0.3 verdict
and the capacity tables stop being trustworthy, so they are tested at the level
of the *property that broke*, not the implementation detail.

The observed failures, for the record:

  1. estimate_quality returned 95 for true q=35 on EVERY real face size tested
     (recompression-MSE criterion; smooth content drives it to the top candidate).
  2. Its confidence was best-vs-second-best, structurally ~0 even when right.
  3. estimate_degradation averaged colour channels before JPEG estimation,
     superimposing three quantisation lattices: 76% accuracy vs 100% on green.
  4. Its `trustworthy` flag was True while the JPEG estimate was wrong by 60
     quality points (it only checked the blur fit's R^2).
  5. arm_B2 applied a probe-space blur sigma at gallery resolution before
     decimation, delivering ~1/8 of the intended operator.
  6. arm_B2 applied the probe's ABSOLUTE operator to an already-degraded
     gallery (double degradation, -8.0 pts TinyFace); arm_B2r applies the
     residual and measured +0.13 -- a no-op, as physics requires when both
     sides sit at the same operating point.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))
_S03 = _BACKEND.parent / "experiments" / "S0_3"
if str(_S03) not in sys.path:
    sys.path.insert(0, str(_S03))

from nexgen_engine.degradation.estimate import estimate_degradation  # noqa: E402
from nexgen_engine.degradation.jpeg import JpegModel, estimate_quality  # noqa: E402
from nexgen_engine.degradation.psf import DegradationParams, apply_forward  # noqa: E402


def _facelike(size: int, seed: int = 0) -> np.ndarray:
    """Smooth, low-frequency content -- the regime where the old estimator broke.

    White noise is exactly the content the old criterion handled fine, so a
    noise image would pass even with the bug present. The fixture has to be
    smooth to be a regression test at all.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(0.5, 0.3, (8, 8)).clip(0, 1)
    img = np.kron(base, np.ones((size // 8 + 1, size // 8 + 1)))[:size, :size]
    # mild texture so blocks are not exactly constant
    return np.clip(img + rng.normal(0, 0.03, (size, size)), 0, 1)


# --------------------------------------------------------------------------- #
# 1 + 2: lattice-fit estimator on smooth content, and its confidence behaviour
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("true_q", [25, 35, 50, 75])
@pytest.mark.parametrize("size", [32, 64, 112])
def test_jpeg_quality_recovered_on_smooth_content(true_q: int, size: int) -> None:
    img = _facelike(size)
    est, _ = estimate_quality(JpegModel(true_q).apply(img))
    assert abs(est - true_q) <= 5, (
        f"lattice fit returned {est} for true q={true_q} at {size}px -- "
        f"the smooth-content failure mode is back"
    )


def test_jpeg_confidence_separates_compressed_from_raw() -> None:
    img = _facelike(112, seed=3)
    _, conf_compressed = estimate_quality(JpegModel(35).apply(img))
    _, conf_raw = estimate_quality(img)
    assert conf_compressed > 0.15, (
        f"confidence {conf_compressed:.3f} on genuinely compressed content would "
        f"be rejected by the 0.15 gate"
    )
    assert conf_raw < conf_compressed, (
        "never-compressed content scored higher confidence than compressed -- "
        "the metric no longer measures lattice presence"
    )


# --------------------------------------------------------------------------- #
# 3: single-channel JPEG estimation on colour input
# --------------------------------------------------------------------------- #


def test_colour_input_does_not_destroy_lattice() -> None:
    img = np.stack([_facelike(64, seed=s) for s in (1, 2, 3)], axis=-1)
    compressed = JpegModel(35).apply(img)
    params, report = estimate_degradation(compressed, assume_jpeg=True)
    assert report["jpeg_quality"] == 35, (
        f"got q={report['jpeg_quality']} on colour input; channel averaging "
        f"before lattice fitting has returned"
    )
    assert params.jpeg_quality == 35


# --------------------------------------------------------------------------- #
# 4: trustworthy is a conjunction with a per-component breakdown
# --------------------------------------------------------------------------- #


def test_trustworthy_reports_components() -> None:
    img = _facelike(64, seed=5)
    _, report = estimate_degradation(JpegModel(50).apply(img))
    tc = report["trust_components"]
    assert set(tc) == {"blur_fit_ok", "jpeg_decisive", "size_ok"}
    assert report["trustworthy"] == all(tc.values()), (
        "trustworthy is not the conjunction of its own components"
    )


def test_trustworthy_false_below_minimum_size() -> None:
    tiny = _facelike(16, seed=7)
    _, report = estimate_degradation(JpegModel(50).apply(tiny))
    assert report["trust_components"]["size_ok"] is False
    assert report["trustworthy"] is False


# --------------------------------------------------------------------------- #
# 5: arm_B2 operates in probe pixel units (decimate first, then degrade)
# --------------------------------------------------------------------------- #


def test_arm_b2_output_matches_probe_grid_and_blur_scale() -> None:
    from arms import arm_B2

    gallery = _facelike(256, seed=11)
    probe = apply_forward(
        gallery, DegradationParams(blur_sigma=1.2, downsample=8.0, noise_sigma=0.01), seed=0
    )
    out_gal, out_prb, report = arm_B2(gallery, probe)
    assert out_gal.shape == probe.shape, "B2 no longer lands on the probe's grid"
    # The applied operator must be expressed in probe space, after decimation.
    assert "applied_in_probe_space" in report, (
        "B2's report no longer declares probe-space application -- the unit-order "
        "fix may have been reverted"
    )
    assert report["downsample_first"] == pytest.approx(256 / probe.shape[0], rel=0.02)


# --------------------------------------------------------------------------- #
# 6: the residual arm is a near-no-op when both sides share the operator
# --------------------------------------------------------------------------- #


def test_arm_b2r_is_noop_when_conditions_match() -> None:
    from arms import arm_B2r

    op = DegradationParams(blur_sigma=1.0, downsample=8.0, noise_sigma=0.01, jpeg_quality=75)
    src_a = _facelike(256, seed=21)
    src_b = _facelike(256, seed=22)
    gallery = apply_forward(src_a, op, seed=1)
    probe = apply_forward(src_b, op, seed=2)

    out_gal, _, report = arm_B2r(gallery, probe)
    res = report["residual_applied"]
    # Both sides went through the identical chain, so the residual must be ~0:
    # tiny blur, no JPEG re-application, and an output close to the input.
    assert res["blur_sigma"] < 0.5, f"residual blur {res['blur_sigma']} on matched conditions"
    assert res["jpeg_quality"] is None or res["jpeg_quality"] >= 75
    rms = float(np.sqrt(np.mean((out_gal - gallery) ** 2)))
    assert rms < 0.06, (
        f"B2r changed a condition-matched gallery by RMS {rms:.4f} -- "
        f"double degradation is creeping back"
    )


def test_arm_b2r_applies_residual_when_probe_is_worse() -> None:
    from arms import arm_B2r

    src = _facelike(256, seed=31)
    gallery = src  # pristine
    probe = apply_forward(src, DegradationParams(blur_sigma=1.5, downsample=8.0, jpeg_quality=35), seed=3)

    out_gal, _, report = arm_B2r(gallery, probe)
    assert out_gal.shape == probe.shape
    res = report["residual_applied"]
    # Clean gallery vs degraded probe: the residual is essentially the probe's
    # own operator, so *something* real must have been applied.
    assert res["blur_sigma"] > 0.1 or res["jpeg_quality"] is not None, (
        "B2r applied nothing against a pristine gallery and a degraded probe"
    )


# --------------------------------------------------------------------------- #
# 12: B2r must convert gallery blur into the probe's pixel grid
# --------------------------------------------------------------------------- #


def test_arm_b2r_converts_gallery_blur_to_probe_grid() -> None:
    """Defect 12: raw sigma subtraction across different pixel grids.

    estimate_degradation reports sigma in the units of the image it was given.
    An earlier arm_B2r subtracted a gallery sigma measured on a 448px image from
    a probe sigma measured on a 100px image. On SCface that made the gallery
    look BLURRIER than the probe (0.8482 vs 0.8028), so max(p^2 - g^2, 0)
    clamped the residual to exactly 0.0 and the arm silently degenerated into
    "decimate + JPEG" -- it was not testing the residual operator at all.

    The fix scales by 1/factor: a Gaussian of width sigma in a grid decimated by
    f has width sigma/f in the new grid.
    """
    from arms import arm_B2r

    gallery = _facelike(448, seed=41)
    probe = apply_forward(
        _facelike(448, seed=41),
        DegradationParams(blur_sigma=2.0, downsample=4.48, jpeg_quality=75),
        seed=0,
    )
    _, _, report = arm_B2r(gallery, probe)

    assert "gallery_blur_in_probe_grid" in report, (
        "B2r no longer reports the grid conversion -- the units fix may be reverted"
    )
    g_own = report["gallery_estimate"]["blur_sigma"]
    g_probe = report["gallery_blur_in_probe_grid"]
    factor = report["downsample_first"]
    # Tolerance is loose because the two figures are rounded to different
    # precisions in the report (sigma to 4 dp, the converted value to 6), so
    # exact agreement is impossible. 1e-3 still proves the division happened.
    assert g_probe == pytest.approx(g_own / factor, rel=1e-3), (
        "gallery blur is not being divided by the decimation factor"
    )
    assert g_probe < g_own, "conversion must SHRINK the gallery sigma into the smaller grid"


def test_arm_b2r_residual_blur_is_not_degenerate() -> None:
    """A genuinely blurrier probe must yield a non-zero residual blur."""
    from arms import arm_B2r

    src = _facelike(448, seed=43)
    probe = apply_forward(
        src, DegradationParams(blur_sigma=2.5, downsample=4.48, jpeg_quality=60), seed=1
    )
    _, _, report = arm_B2r(src, probe)
    res = report["residual_applied"]
    assert res["blur_sigma"] > 0.05, (
        f"residual blur {res['blur_sigma']} is degenerate against a clearly "
        f"blurrier probe -- the grid-conversion bug is back"
    )
