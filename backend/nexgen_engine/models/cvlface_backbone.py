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

2. KEYPOINTS ARE REQUIRED, NOT OPTIONAL. KP-RPE conditions attention on 5
   landmarks in 112x112 pixel coordinates, shape (N, 5, 2). For pre-aligned
   crops (every protocol pack, TinyFace, QMUL resized crops) the canonical
   ArcFace template positions are the correct input by construction -- that is
   what alignment aligned TO. For raw imagery, detector landmarks must be
   passed through instead; using canonical points there would quietly disable
   the mechanism KP-RPE exists for.
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

#: Canonical ArcFace 5-point template for a 112x112 crop: left eye, right eye,
#: nose tip, left mouth corner, right mouth corner. The alignment target of
#: every pre-aligned crop in this project.
ARCFACE_5PTS = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
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


class CvlfaceViTKprpe:
    """Minimal inference wrapper with the insightface `get_feat` interface."""

    def __init__(self, device: str | None = None, batch_size: int = 64):
        import torch

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

        self._torch = torch
        self._canonical = torch.from_numpy(ARCFACE_5PTS).to(self.device)

    @property
    def provider_label(self) -> str:
        return f"cvlface_vit_kprpe ({self.device})"

    def get_feat(self, images: list[np.ndarray], keypoints: np.ndarray | None = None) -> np.ndarray:
        """Embed BGR uint8 112x112 crops (insightface convention) -> (N, 512).

        ``keypoints``: optional (N, 5, 2) landmark array in 112x112 pixel space.
        Omitted => canonical ArcFace template positions, which is correct for
        pre-aligned crops and WRONG for raw un-aligned imagery (landmine 2).
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
