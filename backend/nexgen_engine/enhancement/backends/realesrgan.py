"""Real-ESRGAN super-resolution. Track B: it synthesises detail from a prior.

Chosen as the primary upscaler because it is the robustness benchmark for
genuinely degraded real-world input rather than for bicubic-downsampled test
sets, and because its weights are small, stable and widely mirrored.

It is Track B and the reason is worth stating precisely rather than gesturing
at: RRDBNet was trained adversarially against a second-order degradation
pipeline. Its output above the input's spectral cut-off is drawn from what faces
in its training distribution look like, not from what this sensor recorded. On a
24-pixel face that is most of the visible result.

That does not make it useless -- it makes the image legible to a human examiner,
which is the milestone-1 objective. It makes it unusable as evidence, which is
why the type system puts it where it does.
"""

from __future__ import annotations

from ..registry import BackendSpec, register
from ..types import Task, Track
from ..weights import WeightSpec
from ._torchbase import TorchBackend

_RELEASE = "https://github.com/xinntao/Real-ESRGAN/releases/download"


@register(
    BackendSpec(
        name="realesrgan_x4",
        track=Track.RECONSTRUCTION,
        task=Task.UPSCALE,
        version="v0.1.0-x4plus",
        summary="Real-ESRGAN x4plus (RRDBNet, 23 blocks). Generative super-resolution for legibility.",
        deterministic=True,
        requires_weights=True,
        requires_torch=True,
        vram_estimate_mb=1500.0,
        default_parameters={"scale": 4},
    )
)
class RealESRGANx4(TorchBackend):
    weight_spec = WeightSpec(
        filename="RealESRGAN_x4plus.pth",
        url=f"{_RELEASE}/v0.1.0/RealESRGAN_x4plus.pth",
        notes=(
            "Checksum is intentionally unpinned until an operator installs the file and records it "
            "with scripts/pin_enhancement_weights.py. A guessed checksum would permanently disable "
            "the backend, which is worse than an unverified one that is flagged as unverified."
        ),
    )
    net_scale = 4
    tile = 256
    tile_overlap = 24

    def build_module(self):
        from .arch import RRDBNet  # noqa: PLC0415

        return RRDBNet(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32)

    def scale_factor(self, parameters: dict) -> float:
        return 4.0


@register(
    BackendSpec(
        name="realesrgan_x2",
        track=Track.RECONSTRUCTION,
        task=Task.UPSCALE,
        version="v0.2.1-x2plus",
        summary="Real-ESRGAN x2plus. Half the magnification, correspondingly less invented detail.",
        deterministic=True,
        requires_weights=True,
        requires_torch=True,
        vram_estimate_mb=1400.0,
        default_parameters={"scale": 2},
    )
)
class RealESRGANx2(TorchBackend):
    weight_spec = WeightSpec(
        filename="RealESRGAN_x2plus.pth",
        url=f"{_RELEASE}/v0.2.1/RealESRGAN_x2plus.pth",
        notes="See RealESRGANx4 on checksum pinning.",
    )
    net_scale = 2
    tile = 256
    tile_overlap = 24

    def build_module(self):
        from .arch import RRDBNet  # noqa: PLC0415

        return RRDBNet(num_in_ch=3, num_out_ch=3, scale=2, num_feat=64, num_block=23, num_grow_ch=32)

    def scale_factor(self, parameters: dict) -> float:
        return 2.0


__all__ = ["RealESRGANx2", "RealESRGANx4"]
