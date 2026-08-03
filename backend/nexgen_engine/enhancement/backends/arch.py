"""Network definitions, implemented here rather than imported from vendor packages.

WHY THIS FILE EXISTS AT ALL, because "just pip install realesrgan" is the
obvious alternative and it does not work here.

``basicsr`` -- the shared dependency of the ``realesrgan``, ``gfpgan`` and
``codeformer`` distributions -- imports ``torchvision.transforms.functional_tensor``
at module scope. That module was removed in torchvision 0.17. This project pins
``torch==2.5.1+cu121`` / ``torchvision==0.20.1+cu121``, and that pin is not
negotiable: requirements-gpu.txt documents that those wheels supply the CUDA 12.1
and cuDNN 9.1 DLLs which ``onnxruntime-gpu`` loads for the *recognition* engine.
Downgrading torchvision to satisfy basicsr would break face recognition in
exchange for enhancement, which is a straightforwardly bad trade.

So the architectures are defined natively against plain ``torch.nn``. The
published checkpoints load into them unchanged, because the parameter names are
the ones the checkpoints were saved with.

Loading is always ``strict=True``. A checkpoint whose keys do not match must
make the backend unavailable with a readable error -- never load partially and
produce output from a half-initialised network, which looks like a bad model
rather than like a bad install.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


# --------------------------------------------------------------------------- #
# RRDBNet -- Real-ESRGAN
# --------------------------------------------------------------------------- #


class ResidualDenseBlock(nn.Module):
    """Five convolutions with dense connectivity, residual-scaled by 0.2."""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual-in-residual dense block: three RDBs, residual-scaled again."""

    def __init__(self, num_feat: int, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return out * 0.2 + x


def pixel_unshuffle(x: torch.Tensor, scale: int) -> torch.Tensor:
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    h, w = hh // scale, hw // scale
    view = x.view(b, c, h, scale, w, scale)
    return view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


class RRDBNet(nn.Module):
    """Real-ESRGAN's generator.

    ``scale`` 4 is the native architecture. Scales 2 and 1 are expressed by
    pixel-unshuffling the input first, which is how the published x2 checkpoint
    is built -- not by changing the upsampling stack.
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 4,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ) -> None:
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


# --------------------------------------------------------------------------- #
# NAFNet -- denoising and deblurring
# --------------------------------------------------------------------------- #


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm over NCHW, parameter names matching the checkpoints."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + self.eps).sqrt()
        return self.weight.view(1, -1, 1, 1) * y + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    """Split channels in half and multiply. NAFNet's replacement for an activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2) -> None:
        super().__init__()
        dw_channel = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, 1, 0, groups=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, 1, 0, groups=1, bias=True)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, 1, 0, groups=1, bias=True),
        )
        self.sg = SimpleGate()

        ffn_channel = ffn_expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, 1, 0, groups=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, 1, 0, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        y = inp + x * self.beta

        x = self.conv5(self.sg(self.conv4(self.norm2(y))))
        return y + x * self.gamma


class NAFNet(nn.Module):
    """NAFNet. Defaults are the SIDD-width32 denoising configuration."""

    def __init__(
        self,
        img_channel: int = 3,
        width: int = 32,
        middle_blk_num: int = 12,
        enc_blk_nums: tuple[int, ...] = (2, 2, 4, 8),
        dec_blk_nums: tuple[int, ...] = (2, 2, 2, 2),
    ) -> None:
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1, groups=1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3, 1, 1, groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2))
            )
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        _, _, h, w = inp.shape
        inp = self._pad(inp)

        x = self.intro(inp)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        # NAFNet predicts a residual, not the image.
        x = x + inp
        return x[:, :, :h, :w]

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.size()
        mod_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_w = (self.padder_size - w % self.padder_size) % self.padder_size
        if mod_h or mod_w:
            x = F.pad(x, (0, mod_w, 0, mod_h), mode="replicate")
        return x


# --------------------------------------------------------------------------- #
# Checkpoint loading
# --------------------------------------------------------------------------- #


def extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    """Unwrap the several envelope conventions published checkpoints use.

    Real-ESRGAN saves ``{"params_ema": {...}}``, NAFNet saves ``{"params": {...}}``,
    some releases save the bare tensors, and some wrap in ``state_dict``. Guessing
    wrong yields a key mismatch, which strict loading then reports clearly -- but
    handling the known envelopes first means that error only appears for a
    genuinely unexpected file.
    """
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint is a {type(payload).__name__}, expected a dict")
    for key in ("params_ema", "params", "state_dict", "model"):
        inner = payload.get(key)
        if isinstance(inner, dict):
            payload = inner
            break
    # Some releases prefix every key with "module." from DataParallel training.
    if payload and all(str(k).startswith("module.") for k in payload):
        payload = {str(k)[len("module.") :]: v for k, v in payload.items()}
    return payload


def load_checkpoint(module: nn.Module, path: Any) -> nn.Module:
    """Load strictly. A key mismatch is an error, never a partial load."""
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    state = extract_state_dict(payload)
    module.load_state_dict(state, strict=True)
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


__all__ = [
    "NAFNet",
    "NAFBlock",
    "RRDB",
    "RRDBNet",
    "LayerNorm2d",
    "ResidualDenseBlock",
    "SimpleGate",
    "extract_state_dict",
    "load_checkpoint",
    "pixel_unshuffle",
]
