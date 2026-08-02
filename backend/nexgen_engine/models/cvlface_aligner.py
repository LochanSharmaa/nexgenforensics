"""CVLface DFA aligner -- landmark detection for the KP-RPE pipeline.

WHY THIS EXISTS, MEASURED
-------------------------
The ViT KP-RPE backbone was first evaluated on TinyFace with plain
resize-to-112 crops and canonical keypoints. Result: TAR@FAR=0.1% 15.19% --
WORSE than the R50 incumbent's 20.59% on the identical protocol. The cause is
the landmine documented in cvlface_backbone.py itself: TinyFace/QMUL crops are
detector crops with margin, NOT ArcFace-aligned, so canonical keypoint
positions are false information, and KP-RPE conditions its attention on them.
The LFW gate could not catch this because .bin pack crops ARE aligned -- the
gate validates tensor plumbing, not the alignment assumption.

The published CVLface evaluation pipeline runs this DFA aligner first:
aligned crop + predicted landmarks -> model(aligned, ldmks). This module
reproduces that path.

Interface note: the aligner returns landmarks in the SAME normalised [0, 1]
frame the recogniser's KP-RPE expects (verified empirically -- see
lfw_validation and the tinyface re-run artifacts). Its input is RGB in [-1, 1]
at 160x160 (config: input_size 160, color_space RGB), output a 112x112 aligned
crop plus 5 landmarks.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_SNAPSHOT = (
    Path(__file__).resolve().parents[3]
    / "runtime/models/hf/models--minchul--cvlface_DFA_mobilenet"
    / "snapshots"
)


def _snapshot_dir() -> Path:
    if not _SNAPSHOT.is_dir():
        raise FileNotFoundError(
            f"DFA aligner snapshot not found under {_SNAPSHOT}. Download with "
            f"huggingface_hub.snapshot_download('minchul/cvlface_DFA_mobilenet', "
            f"cache_dir='runtime/models/hf')."
        )
    dirs = [d for d in _SNAPSHOT.iterdir() if d.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"no snapshot revision under {_SNAPSHOT}")
    return sorted(dirs)[-1]


class DfaAligner:
    """Batch aligner: BGR uint8 crops of any size -> aligned 112 crops + landmarks."""

    def __init__(self, device: str | None = None, batch_size: int = 128):
        import torch

        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        snap = _snapshot_dir()
        caller_cwd = os.getcwd()
        try:
            os.chdir(snap)  # package uses relative paths; same landmine as the backbone
            if str(snap) not in sys.path:
                sys.path.insert(0, str(snap))
            import yaml  # noqa: PLC0415
            from omegaconf import OmegaConf  # noqa: PLC0415
            from aligners import get_aligner  # noqa: PLC0415

            with open(snap / "pretrained_model" / "model.yaml", encoding="utf-8") as fh:
                conf = OmegaConf.create(yaml.safe_load(fh))
            net = get_aligner(conf)
            net.load_state_dict_from_path(str(snap / "pretrained_model" / "model.pt"))
            net.eval()
            net.to(self.device)
            self.model = net
        finally:
            os.chdir(caller_cwd)
        self._torch = torch

    def align(self, images: list[np.ndarray]) -> tuple[list[np.ndarray], np.ndarray]:
        """Align BGR uint8 crops.

        Returns ``(aligned_bgr_112, landmarks)`` where landmarks is (N, 5, 2) in
        the aligned crop's [0, 1] frame -- directly consumable by KP-RPE.
        """
        import cv2

        torch = self._torch
        out_imgs: list[np.ndarray] = []
        out_ldmks: list[np.ndarray] = []
        for i in range(0, len(images), self.batch_size):
            chunk = images[i : i + self.batch_size]
            # BGR uint8 -> RGB [-1, 1] at 160x160 (aligner config).
            x = np.stack([cv2.resize(im[:, :, ::-1], (160, 160)) for im in chunk])
            x = (x.astype(np.float32) / 255.0 - 0.5) / 0.5
            t = torch.from_numpy(x.transpose(0, 3, 1, 2)).to(self.device)
            with torch.no_grad():
                aligned, orig_ldmks, aligned_ldmks, score, thetas, bbox = self.model(t)
            a = aligned.float().cpu().numpy()
            # RGB [-1,1] NCHW -> BGR uint8 HWC
            a = ((a * 0.5 + 0.5).clip(0, 1) * 255.0).astype(np.uint8).transpose(0, 2, 3, 1)[:, :, :, ::-1]
            out_imgs.extend(list(a))
            out_ldmks.append(aligned_ldmks.float().cpu().numpy().reshape(len(chunk), 5, 2))
        return out_imgs, np.concatenate(out_ldmks)


__all__ = ["DfaAligner"]
