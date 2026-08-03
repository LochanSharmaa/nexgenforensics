"""Shared machinery for weight-backed backends: lifecycle, tiling, precision.

Everything here exists to make a 6 GB card behave predictably.

TILING IS NOT OPTIONAL. A full-frame 1080p pass through RRDBNet at x4 allocates
tens of gigabytes of activations. Tiling bounds peak memory by the tile size
rather than by the image size, which is what makes "any image the investigator
uploads" a safe statement instead of an OOM waiting for a big frame.

Tiles are blended with a raised-cosine window rather than butt-jointed. Hard
tile seams on a face are the single most recognisable artifact of naive tiled
super-resolution, and on a forensic image a seam running down someone's cheek is
worse than the degradation it replaced.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..registry import EnhancementBackend
from ..weights import WeightSpec, check, resolve

logger = logging.getLogger(__name__)


def torch_available() -> tuple[bool, str]:
    try:
        import torch  # noqa: F401,PLC0415
    except Exception as exc:  # pragma: no cover - host-specific
        return False, f"torch is not installed ({exc})"
    return True, ""


def _window(height: int, width: int, feather: int) -> np.ndarray:
    """Raised-cosine blend weights, 1 in the interior, tapering at the edges."""

    def ramp(length: int) -> np.ndarray:
        w = np.ones(length, dtype=np.float32)
        taper = min(feather, length // 2)
        if taper <= 0:
            return w
        edge = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, taper, dtype=np.float32)))
        w[:taper] = edge
        w[-taper:] = edge[::-1]
        return w

    return np.outer(ramp(height), ramp(width)).astype(np.float32)


class TorchBackend(EnhancementBackend):
    """Base for a backend that loads a checkpoint and runs a network.

    Subclasses provide ``build_module()`` and declare ``weight_spec``. The base
    handles availability, load/release, device placement, tiling and the uint8
    boundary contract.
    """

    weight_spec: WeightSpec
    net_scale: int = 1
    # Tile edge in input pixels. 256 keeps RRDBNet x4 under ~1.5 GB, which
    # leaves room for the recogniser on a 6 GB card even if something else is
    # resident.
    tile: int = 256
    tile_overlap: int = 24
    use_half_on_cuda: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._module: Any = None
        self._half = False

    # -- availability ------------------------------------------------------

    def availability(self) -> tuple[bool, str]:
        ok, reason = torch_available()
        if not ok:
            return ok, reason
        return check(self.weight_spec)

    # -- lifecycle ---------------------------------------------------------

    def build_module(self) -> Any:  # pragma: no cover - subclass responsibility
        raise NotImplementedError

    def load(self, device: str = "cpu") -> None:
        import torch  # noqa: PLC0415

        from .arch import load_checkpoint  # noqa: PLC0415

        if self._module is not None and self._device == device:
            return
        self.release()

        path = resolve(self.weight_spec)
        module = load_checkpoint(self.build_module(), path)
        module = module.to(device)

        # fp16 on CUDA halves activation memory and is a real difference on a
        # 6 GB card. Never on CPU: half-precision CPU kernels are slower than
        # fp32 and on some builds are not implemented at all.
        self._half = bool(device == "cuda" and self.use_half_on_cuda)
        if self._half:
            module = module.half()

        self._module = module
        self._device = device
        logger.info("%s loaded on %s (%s)", self.spec.name, device, "fp16" if self._half else "fp32")

    def release(self) -> None:
        if self._module is None:
            return
        device = self._device
        # Order matters: drop the reference first, then empty the cache.
        # empty_cache() cannot reclaim memory a live module still holds.
        self._module = None
        self._half = False
        from ..vram import free_memory  # noqa: PLC0415

        free_memory(device)

    # -- inference ---------------------------------------------------------

    def _forward(self, batch: Any) -> Any:
        import torch  # noqa: PLC0415

        with torch.inference_mode():
            return self._module(batch)

    def _to_tensor(self, pixels: np.ndarray) -> Any:
        import torch  # noqa: PLC0415

        array = np.ascontiguousarray(pixels.transpose(2, 0, 1), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0).to(self._device)
        return tensor.half() if self._half else tensor

    @staticmethod
    def _to_pixels(tensor: Any) -> np.ndarray:
        array = tensor.squeeze(0).float().clamp_(0.0, 1.0).cpu().numpy()
        return np.ascontiguousarray((array.transpose(1, 2, 0) * 255.0).round().astype(np.uint8))

    def run_tiled(self, pixels: np.ndarray) -> np.ndarray:
        """Run the network over the image, tiling when it is large enough to matter."""
        height, width = pixels.shape[:2]
        scale = self.net_scale

        if max(height, width) <= self.tile:
            return self._to_pixels(self._forward(self._to_tensor(pixels)))

        step = self.tile - self.tile_overlap
        out_h, out_w = height * scale, width * scale
        accum = np.zeros((out_h, out_w, 3), dtype=np.float32)
        weight = np.zeros((out_h, out_w, 1), dtype=np.float32)

        for top in range(0, height, step):
            for left in range(0, width, step):
                bottom = min(top + self.tile, height)
                right = min(left + self.tile, width)
                # Pull short edge tiles back so the last tile is full size where
                # possible; a 3px-wide tile is both wasteful and prone to
                # boundary artifacts.
                top_a = max(0, bottom - self.tile)
                left_a = max(0, right - self.tile)

                patch = pixels[top_a:bottom, left_a:right]
                result = self._to_pixels(self._forward(self._to_tensor(patch)))

                ph, pw = result.shape[:2]
                blend = _window(ph, pw, self.tile_overlap * scale)[..., None]
                oy, ox = top_a * scale, left_a * scale
                accum[oy : oy + ph, ox : ox + pw] += result.astype(np.float32) * blend
                weight[oy : oy + ph, ox : ox + pw] += blend

                if bottom >= height:
                    break
            if left >= width:  # pragma: no cover - loop bookkeeping
                pass

        weight[weight <= 1e-6] = 1.0
        return np.clip(accum / weight, 0, 255).round().astype(np.uint8)

    # -- contract ----------------------------------------------------------

    def apply(self, pixels: np.ndarray, parameters: dict) -> np.ndarray:
        if self._module is None:
            raise RuntimeError(f"{self.spec.name}: load(device) must be called before apply().")
        out = self.run_tiled(pixels)
        if parameters.get("monochrome"):
            gray = out[..., 0] * 0.299 + out[..., 1] * 0.587 + out[..., 2] * 0.114
            out = np.repeat(gray[:, :, None].round().astype(np.uint8), 3, axis=2)
        return out


__all__ = ["TorchBackend", "torch_available"]
