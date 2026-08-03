"""NAFNet denoising. Track A, and the boundary case worth arguing about.

NAFNet is a *regression* network trained to minimise reconstruction error
against a clean reference, not an adversarial generator trained to produce
plausible texture. It removes an additive, largely signal-independent
corruption. That is a materially different operation from synthesising a face
from a codebook, and it is why this sits in the measurement track while
Real-ESRGAN does not.

The honest qualification: it is still a learned model, and a learned denoiser on
severely degraded input can smooth away real fine structure and replace it with
what its training distribution expects. It is not classical BM3D. So
``classical_denoise`` stays the default in the planner's preference order, and
NAFNet is the option an examiner selects deliberately when the classical result
is not good enough -- with the choice recorded in the stage log either way.

Weights: the published SIDD-width32 checkpoint is distributed through the
authors' cloud storage rather than a stable direct URL, so no download URL is
declared. Place the file at the path the availability message names.
"""

from __future__ import annotations

from ..registry import BackendSpec, register
from ..types import Task, Track
from ..weights import WeightSpec
from ._torchbase import TorchBackend


@register(
    BackendSpec(
        name="nafnet_denoise",
        track=Track.MEASUREMENT,
        task=Task.DENOISE,
        version="SIDD-width32",
        summary="NAFNet (width 32) real-noise denoiser. Regression-trained, no adversarial objective.",
        deterministic=True,
        requires_weights=True,
        requires_torch=True,
        vram_estimate_mb=900.0,
        default_parameters={},
    )
)
class NAFNetDenoise(TorchBackend):
    weight_spec = WeightSpec(
        filename="NAFNet-SIDD-width32.pth",
        notes=(
            "Published as NAFNet-SIDD-width32.pth. No stable direct download URL is declared, so "
            "the file must be placed manually. Loading is strict: a mismatched architecture reports "
            "which keys differ rather than half-loading and producing plausible garbage."
        ),
    )
    net_scale = 1
    # Denoising is memory-hungrier per pixel than RRDBNet because the encoder
    # keeps four skip tensors alive. A smaller tile keeps peak under ~1 GB.
    tile = 192
    tile_overlap = 16

    def build_module(self):
        from .arch import NAFNet  # noqa: PLC0415

        return NAFNet(
            img_channel=3,
            width=32,
            middle_blk_num=12,
            enc_blk_nums=(2, 2, 4, 8),
            dec_blk_nums=(2, 2, 2, 2),
        )


__all__ = ["NAFNetDenoise"]
