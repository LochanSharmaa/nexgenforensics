"""CVLface ViT-B + KP-RPE + AdaFace backbone (candidate replacement recogniser).

Loads `minchul/cvlface_adaface_vit_base_kprpe_webface12m` from the local HF
snapshot and exposes the same `get_feat(images) -> (N, 512)` interface as the
insightface recogniser, so every existing benchmark script can run it by
swapping one loader.

WHY THIS MODEL: selection record in the conversation of 2026-08-02 and
IMPLEMENTATION-PLAN.md. Published TinyFace R1 76.10 / IJB-C 97.82 (CVLface
model board, uncited caveat noted in MEASUREMENT_RECORD.md §1); KP-RPE makes
attention alignment-aware, which targets exactly the landmark failure mode at
20-30 px; AdaFace's feature norm doubles as the quality signal the
ConditionalCalibrator already consumes.

TWO WINDOWS/PACKAGING LANDMINES, both hit on first load and worked around here:

1. THE UPSTREAM PACKAGE CORRUPTS THE PROCESS CWD. `models/vit_kprpe/RPE/
   __init__.py` shells into `rpe_ops/setup.py` to build an optional CUDA
   extension. On a machine without CUDA_HOME the build fails -- which is fine,
   there is a Python fallback -- but the attempt `os.chdir`s into `rpe_ops` and
   never restores the cwd on the failure path. Every relative path afterwards
   (including the wrapper's own 'pretrained_model/model.pt') then resolves from
   the wrong directory and load fails with a baffling FileNotFoundError. The
   workaround: trigger the import chain first, then force cwd back to the
   snapshot root, then construct the model -- and restore the caller's cwd in a
   finally block so the corruption cannot escape this module.

2. KEYPOINTS ARE REQUIRED, IN [0, 1] UNITS. KP-RPE conditions attention on 5
   landmarks of shape (N, 5, 2), normalised to the unit square -- see the
   ARCFACE_5PTS docstring for the measured cost of getting the units wrong.
   For pre-aligned crops (every protocol pack, TinyFace, QMUL resized crops)
   the canonical template positions are correct by construction -- they are
   what alignment aligned TO. For raw imagery, detector landmarks (divided by
   crop size) must be passed instead; canonical points there would quietly
   disable the mechanism KP-RPE exists for.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_SNAPSHOT = (
    Path(__file__).resolve().parents[3]
    / "runtime/models/hf/models--minchul--cvlface_adaface_vit_base_kprpe_webface12m"
    / "snapshots"
)

#: Canonical ArcFace 5-point template for a 112x112 crop -- left eye, right eye,
#: nose tip, left mouth corner, right mouth corner -- in **[0, 1] normalised
#: coordinates** (pixel positions / 112).
#:
#: THE UNITS ARE LOAD-BEARING. KP-RPE's relative_keypoints.py builds its patch
#: grid on torch.linspace(0, 1, ...), so keypoints must live on the same [0, 1]
#: square. Measured on 1,000 LFW pairs with this exact checkpoint:
#:
#:     pixel units (0..112)   60.40%  -- near chance, silently
#:     [0, 1]                 99.90%  <- correct
#:     [-1, 1]                98.30%  -- close enough to be dangerously wrong
#:
#: The first draft of this module used pixel units and failed the LFW gate at
#: 56.70%. Note the [-1,1] row: a plausible-but-wrong convention costs only 1.6
#: points on easy data, which is exactly the kind of error that survives without
#: a validation gate and quietly caps every downstream measurement.
ARCFACE_5PTS = (
    np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )
    / 112.0
)


def _snapshot_dir() -> Path:
    if not _SNAPSHOT.is_dir():
        raise FileNotFoundError(
            f"CVLface snapshot not found under {_SNAPSHOT}. Download with "
            f"huggingface_hub.snapshot_download('minchul/cvlface_adaface_vit_base_kprpe_webface12m', "
            f"cache_dir='runtime/models/hf')."
        )
    candidates = [d for d in _SNAPSHOT.iterdir() if d.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no snapshot revision under {_SNAPSHOT}")
    return sorted(candidates)[-1]


class LoRALinear:  # noqa: D101 - defined dynamically below to avoid a torch import at module scope
    pass


def _make_lora_cls():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _LoRALinear(nn.Module):
        """y = W0 x + (alpha/r) B A x, with W0 frozen. Mirrors train_lora.py."""

        def __init__(self, base, r=8, alpha=16):
            super().__init__()
            self.base = base
            for p in self.base.parameters():
                p.requires_grad = False
            self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
            self.B = nn.Parameter(torch.zeros(base.out_features, r))
            self.scale = alpha / r

        def forward(self, x):
            return self.base(x) + F.linear(F.linear(x, self.A), self.B) * self.scale

    return _LoRALinear


class CvlfaceViTKprpe:
    """Minimal inference wrapper with the insightface `get_feat` interface.

    ``lora_path`` loads a LoRA adapter trained by scripts/train_lora.py. The
    injection must reproduce train-time module targeting exactly -- the adapter
    is stored by parameter name, so a mismatch surfaces as missing/unexpected
    keys rather than as silently wrong embeddings, which is why the load is
    strict.
    """

    def __init__(self, device: str | None = None, batch_size: int = 64,
                 lora_path: str | Path | None = None):
        import torch
        import torch.nn as nn

        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        snap = _snapshot_dir()
        caller_cwd = os.getcwd()
        try:
            os.chdir(snap)
            if str(snap) not in sys.path:
                sys.path.insert(0, str(snap))
            # Do NOT use the repo's wrapper.py: it loads
            # 'pretrained_model/model.pt' by RELATIVE path, and get_model()'s
            # lazy subpackage import runs the rpe_ops setup.py that strands the
            # process cwd inside rpe_ops on the no-CUDA_HOME failure path. No
            # ordering of chdir calls around the wrapper survives that, because
            # the corruption happens between its config load and its state-dict
            # load. So replicate its five lines with absolute paths instead.
            import yaml  # noqa: PLC0415
            from omegaconf import OmegaConf  # noqa: PLC0415
            from models import get_model  # noqa: PLC0415

            with open(snap / "pretrained_model" / "model.yaml", encoding="utf-8") as fh:
                conf = OmegaConf.create(yaml.safe_load(fh))
            net = get_model(conf)  # may corrupt cwd; nothing after this is relative
            net.load_state_dict_from_path(str(snap / "pretrained_model" / "model.pt"))
            net.eval()
            net.to(self.device)
            self.model = net
        finally:
            os.chdir(caller_cwd)

        self.lora_path = str(lora_path) if lora_path else None
        if lora_path:
            ck = torch.load(lora_path, map_location="cpu")
            LoRA = _make_lora_cls()
            n_inj = 0
            for mod in self.model.modules():
                for name, child in list(mod.named_children()):
                    if isinstance(child, nn.Linear) and name in ("qkv", "proj", "fc1", "fc2"):
                        setattr(mod, name, LoRA(child, r=ck["rank"], alpha=ck["alpha"]))
                        n_inj += 1
            missing, unexpected = self.model.load_state_dict(ck["lora"], strict=False)
            got = [k for k in ck["lora"]]
            loaded = [k for k in got if k not in unexpected]
            if len(loaded) != len(got):
                raise RuntimeError(
                    f"LoRA adapter did not apply cleanly: {len(got)-len(loaded)} of "
                    f"{len(got)} tensors unmatched. Injection targets differ from training."
                )
            self.model.eval().to(self.device)
            self._lora_info = {"layers": n_inj, "tensors": len(got),
                               "rank": ck["rank"], "epoch": ck.get("epoch")}
        else:
            self._lora_info = None

        self._torch = torch
        self._canonical = torch.from_numpy(ARCFACE_5PTS).to(self.device)

    @property
    def provider_label(self) -> str:
        if self._lora_info:
            i = self._lora_info
            return (f"cvlface_vit_kprpe+lora_r{i['rank']}@ep{i['epoch']} "
                    f"({i['layers']} layers, {self.device})")
        return f"cvlface_vit_kprpe ({self.device})"

    def get_feat(self, images: list[np.ndarray], keypoints: np.ndarray | None = None) -> np.ndarray:
        """Embed BGR uint8 112x112 crops (insightface convention) -> (N, 512).

        ``keypoints``: optional (N, 5, 2) landmark array in [0, 1] normalised
        coordinates (landmine 2 -- pixel units silently cost ~40 points).
        Omitted => canonical ArcFace template positions, correct for
        pre-aligned crops and WRONG for raw un-aligned imagery.
        """
        torch = self._torch
        out = np.empty((len(images), 512), dtype=np.float32)
        for i in range(0, len(images), self.batch_size):
            chunk = images[i : i + self.batch_size]
            # BGR uint8 -> RGB float in [-1, 1], NCHW (CVLface convention).
            x = np.stack([im[:, :, ::-1] for im in chunk]).astype(np.float32) / 255.0
            x = (x - 0.5) / 0.5
            t = torch.from_numpy(x.transpose(0, 3, 1, 2)).to(self.device)
            if keypoints is None:
                kp = self._canonical.unsqueeze(0).expand(len(chunk), -1, -1)
            else:
                kp = torch.from_numpy(
                    np.asarray(keypoints[i : i + len(chunk)], dtype=np.float32)
                ).to(self.device)
            with torch.no_grad():
                feat = self.model(t, kp)
            out[i : i + len(chunk)] = feat.float().cpu().numpy()
        return out


__all__ = ["ARCFACE_5PTS", "CvlfaceViTKprpe"]
