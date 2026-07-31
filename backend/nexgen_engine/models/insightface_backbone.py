from __future__ import annotations

import os

import cv2
import numpy as np
from PIL import Image
import insightface
from insightface.app import FaceAnalysis

from typing import TYPE_CHECKING

from ..utils import l2_normalize
from .backbones import BackboneOutput
from .cuda_runtime import (
    assert_face_analysis_providers,
    cuda_expected,
    init_cuda,
    resolve_providers,
)

if TYPE_CHECKING:
    from ..config import EngineConfig


def _get_providers() -> tuple[list[str], int]:
    """Detect GPU availability and return providers + ctx_id.

    Delegates to cuda_runtime, which registers the CUDA 12 / cuDNN 9 DLL
    directories and initializes the CUDA context *before* onnxruntime builds
    any session. See cuda_runtime.py for why the ordering matters.
    """
    init_cuda()
    return resolve_providers()


def _embed(app: FaceAnalysis, cv_img: np.ndarray, require_face: bool = False) -> np.ndarray:
    """Extract embedding from an image using a FaceAnalysis app.
    Raises ValueError if require_face is True and no face is detected."""
    faces = app.get(cv_img)
    if not faces:
        if require_face:
            raise ValueError("No face detected in image")
        rec_model = app.models['recognition']
        resized = cv2.resize(cv_img, (112, 112))
        embedding = rec_model.get_feat(resized).flatten()
    else:
        faces.sort(
            key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
            reverse=True
        )
        embedding = faces[0].embedding
    return l2_normalize(embedding.astype(np.float32))


class InsightFaceArcFaceBackbone:
    """Single-model backbone: buffalo_l / w600k_r50 (R50 trained on WebFace600K)."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.name = "insightface_w600k_r50"
        try:
            providers, ctx_id = _get_providers()
            self.app = FaceAnalysis(name="buffalo_l", providers=providers)
            self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            bound = assert_face_analysis_providers(self.app, "buffalo_l")
            print(f"[NEXGEN_ENGINE] buffalo_l loaded: w600k_r50 / SCRFD  bound={bound}")
        except Exception as e:
            raise RuntimeError(f"FATAL: Failed to load buffalo_l: {e}")

    def encode(self, image: Image.Image, quality_score: float = 1.0, require_face: bool = False) -> BackboneOutput:
        cv_img = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        embedding = _embed(self.app, cv_img, require_face=require_face)
        return BackboneOutput(name=self.name, embedding=embedding, quality_weight=1.0)


class InsightFaceEnsembleBackbone:
    """
    Multi-Model Real Pretrained Ensemble:
    - Model 1: buffalo_l (w600k_r50.onnx - ResNet50 trained on WebFace600K)
    - Model 2: antelopev2 (glintr100.onnx - ResNet100 trained on Glint360K)
    - Model 3: buffalo_s (w600k_mbf.onnx - MobileFaceNet trained on WebFace600K)

    Fusion Strategy: WEIGHTED EMBEDDING SPACE AVERAGING & CONCATENATION
    -------------------------------------------------------------------
    1. Weighted Averaging (512-d):
       Each model outputs a 512-d normalized ArcFace embedding.
       We weight the backbones according to capacity & training set scale:
       w_buffalo_l = 0.45, w_antelopev2 = 0.45, w_buffalo_s = 0.10.
       fused = L2_Normalize(w1 * emb1 + w2 * emb2 + w3 * emb3)

       Why? The averaged embedding vector represents the spherical Fréchet mean
       on the 512-dimensional hypersphere, aligning consensus directions while
       suppressing single-model errors and maintaining 512-d database index compatibility.

    2. Concatenation Fusion (1536-d):
       fused_concat = [emb1 / sqrt(3) | emb2 / sqrt(3) | emb3 / sqrt(3)]
       Cosine similarity of concatenated vectors is mathematically identical to
       the arithmetic mean of individual cosine similarities.
    """

    BUFFALO_DIM = 512
    ANTELOPE_DIM = 512
    BUFFALO_S_DIM = 512

    #: Which backbones each fusion method actually needs. Loading a model the
    #: active method never reads costs ~1.5 GB of VRAM and a second of startup
    #: for nothing.
    _REQUIRED = {
        "single_glintr100": ("antelopev2",),
        "single_r50": ("buffalo_l",),
        "dual_ensemble": ("buffalo_l", "antelopev2"),
        "weighted_avg": ("buffalo_l", "antelopev2", "buffalo_s"),
        "equal_avg": ("buffalo_l", "antelopev2", "buffalo_s"),
        "concat": ("buffalo_l", "antelopev2", "buffalo_s"),
    }

    def __init__(self, config: EngineConfig | None = None, fusion_method: str | None = None) -> None:
        self.name = "ensemble_multi_model"
        # Default comes from the measured benchmark, not from habit. See
        # BENCHMARKS.md section 3: glintr100 alone is better than or equal to
        # the 3-model ensemble on all five verification protocols, at 1/3 the
        # inference cost. Override with NEXGEN_FUSION_METHOD to A/B.
        self.fusion_method = fusion_method or os.environ.get(
            "NEXGEN_FUSION_METHOD", "single_glintr100"
        )
        if self.fusion_method not in self._REQUIRED:
            raise ValueError(
                f"unknown fusion method {self.fusion_method!r}; "
                f"expected one of {sorted(self._REQUIRED)}"
            )
        needed = self._REQUIRED[self.fusion_method]
        self.buffalo = self.antelope = self.buffalo_s = None

        try:
            providers, ctx_id = _get_providers()
            print(
                f"[NEXGEN_ENGINE] fusion={self.fusion_method} loading={list(needed)}  "
                f"providers={providers} cuda_expected={cuda_expected()}"
            )

            specs = [
                ("buffalo_l", "w600k_r50, ResNet-50", "buffalo"),
                ("antelopev2", "glintr100, ResNet-100", "antelope"),
                ("buffalo_s", "w600k_mbf, MobileFaceNet", "buffalo_s"),
            ]
            for pack, desc, attr in specs:
                if pack not in needed:
                    continue
                app = FaceAnalysis(name=pack, providers=providers)
                app.prepare(ctx_id=ctx_id, det_size=(640, 640))
                bound = assert_face_analysis_providers(app, pack)
                setattr(self, attr, app)
                print(f"[NEXGEN_ENGINE] {pack} ({desc}) OK  bound={bound}")

        except Exception as e:
            raise RuntimeError(f"FATAL: Failed to load recognition backbone(s): {e}")

    def extract_all_embeddings(self, cv_img: np.ndarray, require_face: bool = False) -> dict[str, np.ndarray]:
        """Extract 512-d L2-normalized embeddings from the loaded backbones.

        Only the backbones the active fusion method needs are loaded, so this
        returns just those. Keys are absent rather than None for models that
        were never loaded.
        """
        out: dict[str, np.ndarray] = {}
        for attr, key in (("buffalo", "buffalo_l"), ("antelope", "antelopev2"), ("buffalo_s", "buffalo_s")):
            app = getattr(self, attr)
            if app is not None:
                out[key] = _embed(app, cv_img, require_face=require_face)
        return out

    def fuse_embeddings(self, embs: dict[str, np.ndarray], method: str | None = None) -> np.ndarray:
        """Fuse backbone embeddings according to the selected strategy."""
        m = method or self.fusion_method

        # Production default: the single strongest backbone. Measured better
        # than or equal to every ensemble variant on all five protocols.
        if m == "single_glintr100":
            return embs["antelopev2"]
        if m == "single_r50":
            return embs["buffalo_l"]

        missing = [k for k in self._REQUIRED[m] if k not in embs]
        if missing:
            raise ValueError(f"fusion method {m!r} needs {missing}, which were not loaded")

        b, a = embs["buffalo_l"], embs["antelopev2"]
        s = embs.get("buffalo_s")

        if m == "equal_avg":
            # Unweighted Mean
            raw_fused = (b + a + s) / 3.0
            return l2_normalize(raw_fused)

        elif m == "weighted_avg":
            # Weighted Mean based on model capacity: ResNet50 (0.45), ResNet100 (0.45), MobileFaceNet (0.10)
            raw_fused = 0.45 * b + 0.45 * a + 0.10 * s
            return l2_normalize(raw_fused)

        elif m == "dual_ensemble":
            # 2-model ensemble (buffalo_l + antelopev2)
            raw_fused = 0.5 * b + 0.5 * a
            return l2_normalize(raw_fused)

        elif m == "concat":
            # Concatenation Fusion (1536-d)
            return np.concatenate([b, a, s], axis=0) / np.sqrt(3.0)

        else:
            raise ValueError(f"Unknown fusion method: {m}")

    def encode(self, image: Image.Image, quality_score: float = 1.0, require_face: bool = False) -> BackboneOutput:
        cv_img = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        embs = self.extract_all_embeddings(cv_img, require_face=require_face)
        fused = self.fuse_embeddings(embs, self.fusion_method)
        return BackboneOutput(
            name=self.name,
            embedding=fused,
            quality_weight=1.0
        )

