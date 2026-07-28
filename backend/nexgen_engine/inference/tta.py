from __future__ import annotations

from PIL import Image, ImageOps


class TTAProcessor:
    """Test-time augmentation for aligned face crops.

    Only the horizontal flip is used. ArcFace is trained with flip augmentation,
    so a face and its mirror land close together in embedding space and
    averaging them cancels some pose-specific noise -- a small but real gain.

    Photometric augmentations (brightness shifts, sharpening, resize round-trips)
    are deliberately NOT applied. They were present in an earlier version, but
    they push the crop away from the distribution the network was trained on, so
    averaging over them drags the template toward the dataset mean and makes
    different identities *more* similar, not less.
    """

    def __init__(self, use_flip: bool = True) -> None:
        self.use_flip = use_flip

    def apply(self, crop: Image.Image) -> list[Image.Image]:
        rgb = crop.convert("RGB")
        return [rgb, ImageOps.mirror(rgb)] if self.use_flip else [rgb]


__all__ = ["TTAProcessor"]
