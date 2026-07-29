from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
import insightface
from insightface.app import FaceAnalysis

from ..config import BackboneConfig
from ..utils import l2_normalize
from .backbones import BackboneOutput


def _get_providers() -> tuple[list[str], int]:
    """Detect GPU availability and return providers + ctx_id."""
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.cuda.init()
    except Exception:
        pass
    import onnxruntime as _ort
    if "CUDAExecutionProvider" in _ort.get_available_providers():
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], 0
    return ["CPUExecutionProvider"], -1


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

    def __init__(self, config: BackboneConfig | None = None) -> None:
        self.name = "insightface_w600k_r50"
        try:
            providers, ctx_id = _get_providers()
            self.app = FaceAnalysis(name="buffalo_l", providers=providers)
            self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            print(f"[NEXGEN_ENGINE] buffalo_l loaded: w600k_r50 / SCRFD  providers={providers}")
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

    def __init__(self, config: BackboneConfig | None = None, fusion_method: str = "weighted_avg") -> None:
        self.name = "ensemble_multi_model"
        self.fusion_method = fusion_method
        try:
            providers, ctx_id = _get_providers()
            print(f"[NEXGEN_ENGINE] Loading Multi-Model Ensemble on {providers}...")

            # 1. Load buffalo_l (w600k_r50)
            self.buffalo = FaceAnalysis(name="buffalo_l", providers=providers)
            self.buffalo.prepare(ctx_id=ctx_id, det_size=(640, 640))
            print("[NEXGEN_ENGINE] Model 1: buffalo_l (w600k_r50, ResNet-50) loaded OK")

            # 2. Load antelopev2 (glintr100)
            self.antelope = FaceAnalysis(name="antelopev2", providers=providers)
            self.antelope.prepare(ctx_id=ctx_id, det_size=(640, 640))
            print("[NEXGEN_ENGINE] Model 2: antelopev2 (glintr100, ResNet-100) loaded OK")

            # 3. Load buffalo_s (w600k_mbf)
            self.buffalo_s = FaceAnalysis(name="buffalo_s", providers=providers)
            self.buffalo_s.prepare(ctx_id=ctx_id, det_size=(640, 640))
            print("[NEXGEN_ENGINE] Model 3: buffalo_s (w600k_mbf, MobileFaceNet) loaded OK")

        except Exception as e:
            raise RuntimeError(f"FATAL: Failed to load multi-model ensemble: {e}")

    def extract_all_embeddings(self, cv_img: np.ndarray, require_face: bool = False) -> dict[str, np.ndarray]:
        """Extract individual 512-d L2-normalized embeddings from all 3 backbones."""
        emb_b = _embed(self.buffalo, cv_img, require_face=require_face)    # 512-d, buffalo_l (ResNet50)
        emb_a = _embed(self.antelope, cv_img, require_face=require_face)   # 512-d, antelopev2 (ResNet100)
        emb_s = _embed(self.buffalo_s, cv_img, require_face=require_face)  # 512-d, buffalo_s (MobileFaceNet)
        return {"buffalo_l": emb_b, "antelopev2": emb_a, "buffalo_s": emb_s}

    def fuse_embeddings(self, embs: dict[str, np.ndarray], method: str | None = None) -> np.ndarray:
        """Fuse 3 backbone embeddings according to selected strategy."""
        m = method or self.fusion_method
        b, a, s = embs["buffalo_l"], embs["antelopev2"], embs["buffalo_s"]

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

