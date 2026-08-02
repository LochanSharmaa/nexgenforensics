"""Tests for the forward imaging models.

The property this suite exists to protect: **this package goes one way only.**
There is no restore, no upsample, no enhance. A test asserting that absence is
included deliberately, because adding one would silently convert the system from
one that weighs evidence into one that invents it.
"""

from __future__ import annotations

import numpy as np
import pytest

from nexgen_engine import degradation
from nexgen_engine.degradation.bandlimit import (
    effective_cutoff,
    match_passband,
    radial_power_spectrum,
    to_passband,
)
from nexgen_engine.degradation.estimate import estimate_blur_sigma, estimate_degradation
from nexgen_engine.degradation.jpeg import JpegModel, estimate_quality, quantization_matrix
from nexgen_engine.degradation.noise import NoiseModel, estimate_noise_sigma
from nexgen_engine.degradation.psf import (
    DegradationParams,
    apply_forward,
    gaussian_psf,
    motion_psf,
    mtf50,
    mtf_from_psf,
)


def textured(size=96, seed=0):
    """Broadband texture: a flat image has no spectrum to estimate from."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.5, 0.2, (size, size))
    yy, xx = np.mgrid[0:size, 0:size]
    x += 0.2 * np.sin(2 * np.pi * xx / 8) + 0.15 * np.cos(2 * np.pi * yy / 5)
    return np.clip(x, 0, 1)


class TestPSF:
    def test_kernel_is_normalised(self):
        assert gaussian_psf(2.0).kernel.sum() == pytest.approx(1.0)

    def test_zero_sigma_is_identity(self):
        assert gaussian_psf(0.0).kernel.shape == (1, 1)

    def test_more_blur_means_lower_mtf50(self):
        assert mtf50(gaussian_psf(3.0)) < mtf50(gaussian_psf(1.0)) < mtf50(gaussian_psf(0.3))

    def test_mtf_starts_at_unity(self):
        _, mtf = mtf_from_psf(gaussian_psf(1.5))
        assert mtf[0] == pytest.approx(1.0)

    def test_mtf_is_non_increasing_overall(self):
        _, mtf = mtf_from_psf(gaussian_psf(2.0))
        assert mtf[-1] < mtf[0]

    def test_motion_psf_is_directional(self):
        h = motion_psf(9, 0).kernel
        v = motion_psf(9, 90).kernel
        assert h.sum(axis=1).max() > h.sum(axis=0).max()
        assert v.sum(axis=0).max() > v.sum(axis=1).max()


class TestForwardOperator:
    def test_blur_reduces_high_frequency_energy(self):
        img = textured()
        out = apply_forward(img, DegradationParams(blur_sigma=2.0))
        assert effective_cutoff(out) <= effective_cutoff(img) + 1e-9

    def test_downsample_changes_shape(self):
        out = apply_forward(textured(96), DegradationParams(downsample=4.0))
        assert out.shape[0] == pytest.approx(24, abs=1)

    def test_identity_params_preserve_the_image(self):
        img = textured()
        assert np.allclose(apply_forward(img, DegradationParams()), img)

    def test_noise_is_reproducible_for_a_seed(self):
        img = textured()
        p = DegradationParams(noise_sigma=0.05)
        assert np.allclose(apply_forward(img, p, seed=7), apply_forward(img, p, seed=7))

    def test_handles_colour(self):
        img = np.stack([textured(64, s) for s in range(3)], -1)
        assert apply_forward(img, DegradationParams(blur_sigma=1.0)).shape == img.shape

    def test_params_serialise_with_mtf50(self):
        d = DegradationParams(blur_sigma=1.5).as_dict()
        assert "mtf50_cycles_per_pixel" in d and d["blur_sigma"] == 1.5

    def test_package_exposes_no_inverse_operation(self):
        """ARCHITECTURAL INVARIANT. Evidence is never lifted toward a hypothesis.

        If this fails someone has added a restoration path, which would let a
        learned prior contribute to a forensic conclusion.
        """
        forbidden = {"restore", "upsample", "enhance", "deblur", "super_resolve", "sharpen"}
        assert forbidden.isdisjoint(set(dir(degradation)))


class TestNoise:
    def test_estimator_recovers_injected_sigma(self):
        """Measured on SMOOTH content. The Laplacian residual cannot separate
        genuine high-frequency texture from noise, so a textured base inflates the
        estimate -- a real limitation of blind estimation, documented rather than
        hidden by a loose tolerance."""
        yy, xx = np.mgrid[0:128, 0:128]
        img = 0.5 + 0.2 * (yy / 128.0) + 0.1 * (xx / 128.0)
        for sigma in (0.02, 0.05, 0.1):
            noisy = img + np.random.default_rng(0).normal(0, sigma, img.shape)
            assert estimate_noise_sigma(noisy) == pytest.approx(sigma, rel=0.25)

    def test_estimator_is_inflated_by_texture(self):
        """The documented failure mode, asserted so it is not forgotten."""
        flat = np.full((128, 128), 0.5)
        noise = np.random.default_rng(0).normal(0, 0.03, (128, 128))
        assert estimate_noise_sigma(textured(128) + noise) > estimate_noise_sigma(flat + noise)

    def test_clean_image_estimates_near_zero(self):
        assert estimate_noise_sigma(np.full((64, 64), 0.5)) < 0.01

    def test_fixed_pattern_noise_does_not_average_away(self):
        """The ceiling on multi-frame gains. Shot and read noise fall as 1/sqrt(N);
        fixed-pattern noise does not fall at all."""
        m = NoiseModel(read_sigma=0.05, fpn_sigma=0.02)
        v1 = m.multiframe_variance(0.5, 1)
        v100 = m.multiframe_variance(0.5, 100)
        assert v100 < v1
        assert v100 >= (0.02 * 0.5) ** 2 - 1e-12

    def test_snr_rises_with_signal(self):
        m = NoiseModel(read_sigma=0.05)
        assert m.snr_at(0.8) > m.snr_at(0.2)


class TestJpeg:
    def test_quantization_table_scales_with_quality(self):
        assert quantization_matrix(10).mean() > quantization_matrix(90).mean()

    def test_output_stays_in_range(self):
        out = JpegModel(quality=30).apply(textured())
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_lower_quality_costs_more_fidelity(self):
        img = textured()
        e_low = np.mean((JpegModel(20).apply(img) - img) ** 2)
        e_high = np.mean((JpegModel(95).apply(img) - img) ** 2)
        assert e_low > e_high

    def test_quality_recovery_lands_near_the_truth(self):
        img = textured(64)
        recovered, conf = estimate_quality(JpegModel(quality=50).apply(img), range(30, 95, 5))
        assert abs(recovered - 50) <= 15
        assert conf >= 0.0

    def test_shape_is_preserved_for_non_multiples_of_eight(self):
        assert JpegModel(50).apply(textured(70)).shape == (70, 70)


class TestBandlimit:
    def test_passband_removes_high_frequencies(self):
        img = textured()
        assert effective_cutoff(to_passband(img, 0.08)) < effective_cutoff(img)

    def test_full_band_is_a_no_op(self):
        img = textured()
        assert np.allclose(to_passband(img, 0.5), img)

    def test_hard_projection_is_idempotent(self):
        """Only the hard mask is idempotent: a soft roll-off applied twice
        squares the mask, which is why `soft` is a documented parameter and not
        a hidden default behaviour."""
        img = textured()
        once = to_passband(img, 0.1, soft=False)
        assert np.allclose(once, to_passband(once, 0.1, soft=False), atol=1e-8)

    def test_blurred_image_reports_a_lower_cutoff(self):
        img = textured()
        blurred = apply_forward(img, DegradationParams(blur_sigma=3.0))
        assert effective_cutoff(blurred) <= effective_cutoff(img)

    def test_spectrum_bins_are_ordered_and_finite(self):
        f, p = radial_power_spectrum(textured())
        assert np.all(np.diff(f) > 0) and np.all(np.isfinite(p))

    def test_match_passband_reduces_the_clean_side(self):
        """The direction of travel: the hypothesis is lowered to the evidence."""
        hi = textured(96)
        lo = apply_forward(hi, DegradationParams(blur_sigma=2.5, downsample=4.0))
        hi_p, lo_p, rep = match_passband(hi, lo)
        assert hi_p.shape == hi.shape and lo_p.shape == lo.shape
        assert effective_cutoff(hi_p) <= effective_cutoff(hi) + 1e-9
        assert "common_passband_cycles_per_face" in rep


class TestEstimation:
    def test_blur_estimate_is_ordered_in_true_sigma(self):
        img = textured(128)
        s1, _ = estimate_blur_sigma(apply_forward(img, DegradationParams(blur_sigma=1.0)))
        s3, _ = estimate_blur_sigma(apply_forward(img, DegradationParams(blur_sigma=3.0)))
        assert s3 > s1

    def test_flat_image_reports_low_confidence(self):
        _, conf = estimate_blur_sigma(np.full((64, 64), 0.5))
        assert conf < 0.9

    def test_full_estimate_reports_its_own_trustworthiness(self):
        img = textured(128)
        deg = apply_forward(img, DegradationParams(blur_sigma=2.0, noise_sigma=0.02))
        params, report = estimate_degradation(deg)
        assert params.origin == "estimated"
        assert "trustworthy" in report and "caveat" in report
        assert report["blur_confidence_r2"] >= 0.0
