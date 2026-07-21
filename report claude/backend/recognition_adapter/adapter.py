"""
recognition_adapter/adapter.py

THIS IS THE ONLY FILE YOU SHOULD NEED TO EDIT TO PLUG IN YOUR REAL MODEL.

Everything else in this system (measurements, visualization, report, PDF)
depends only on the return shape documented below — never on how the
numbers were produced. Do not let any other module import a recognition
model directly.

Implementation note
-------------------
Landmark detection  — MediaPipe FaceMesh (478 landmarks, schema "468")
Embedding extraction — nexgen_engine.FacialRecognitionPipeline (512-d, L2-normalized)

The nexgen_engine is treated as a black-box; we call encode_image() and
return its embedding vector unchanged.  If torch / the engine is not
importable (e.g. CI without GPU), a deterministic SHA-256 fallback is
used for the embedding only, so the scaffold can still be demonstrated.
"""
from __future__ import annotations

import hashlib
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data contracts (do NOT change field names — downstream depends on these)
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    bbox: dict            # {"x":.., "y":.., "w":.., "h":..} in pixel space
    detection_confidence: float
    landmarks: list[dict] # [{"id":0, "x":.. , "y":.. , "confidence":..}, ...]
    landmark_schema: str  # "68" | "106" | "468"


@dataclass
class EmbeddingResult:
    vector: np.ndarray
    model_name: str
    model_version: str
    inference_time_ms: float


# ---------------------------------------------------------------------------
# Lazy singletons — initialised once per process
# ---------------------------------------------------------------------------

_facemesh = None        # mediapipe FaceLandmarker instance
_nexgen_pipeline = None # FacialRecognitionPipeline instance
_nexgen_available = None

# Path to the downloaded FaceLandmarker model file (.task)
_MODEL_PATH = Path(__file__).parent / "face_landmarker.task"
# fallback: look in the backend root too
_MODEL_PATH_ALT = Path(__file__).parent.parent / "face_landmarker.task"


def _model_file() -> Optional[Path]:
    for p in [_MODEL_PATH, _MODEL_PATH_ALT]:
        if p.exists():
            return p
    return None


def _get_facemesh():
    """
    Returns a MediaPipe FaceLandmarker (Tasks API, v0.10+).
    Falls back to None if the model file is not available, which causes
    _mediapipe_detect to use the synthetic fallback.
    """
    global _facemesh
    if _facemesh is not None:
        return _facemesh

    import mediapipe as mp
    model_path = _model_file()
    if model_path is None:
        logger.warning(
            "face_landmarker.task model file not found. "
            "Place it at: %s  (Download from "
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task)",
            _MODEL_PATH,
        )
        return None

    try:
        base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
        )
        _facemesh = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        logger.info("MediaPipe FaceLandmarker (Tasks API) initialised from %s.", model_path)
    except Exception as exc:
        logger.warning("Failed to initialise FaceLandmarker: %s", exc)
        _facemesh = None
    return _facemesh


def _get_nexgen_pipeline():
    global _nexgen_pipeline, _nexgen_available
    if _nexgen_available is not None:
        return _nexgen_pipeline  # already tried

    # Try to import from the real backend sitting one level above report claude/
    nexgen_root = Path(__file__).parent.parent.parent.parent / "backend"
    if str(nexgen_root) not in sys.path:
        sys.path.insert(0, str(nexgen_root))

    try:
        from nexgen_engine.inference.pipeline import FacialRecognitionPipeline
        _nexgen_pipeline = FacialRecognitionPipeline()
        _nexgen_available = True
        logger.info("nexgen_engine.FacialRecognitionPipeline loaded successfully.")
    except Exception as exc:
        _nexgen_available = False
        _nexgen_pipeline = None
        logger.warning(
            "nexgen_engine not available (%s). "
            "Embedding will use deterministic SHA-256 fallback.", exc
        )
    return _nexgen_pipeline


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mediapipe_detect(image_bgr: np.ndarray) -> tuple[Optional[dict], list[dict], float]:
    """
    Runs MediaPipe FaceLandmarker on an already-loaded BGR image.

    Returns:
        bbox            — {"x", "y", "w", "h"} in pixel space, or None
        landmarks_list  — list of {"id", "x", "y", "confidence"} dicts
        confidence      — scalar detection confidence proxy (landmark z-depth mean)
    """
    import mediapipe as mp
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    landmarker = _get_facemesh()
    if landmarker is None:
        # Model file unavailable — will use synthetic fallback
        return None, [], 0.0

    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)
    except Exception as exc:
        logger.warning("FaceLandmarker.detect() raised: %s", exc)
        return None, [], 0.0

    if not result.face_landmarks:
        return None, [], 0.0

    face = result.face_landmarks[0]  # first (and only) face

    # FaceLandmarker returns NormalizedLandmark — convert to pixel coords
    landmarks = []
    xs, ys = [], []
    for idx, lm in enumerate(face):
        px = lm.x * w
        py = lm.y * h
        # Use z-depth as a proxy for confidence; deeper = less visible
        conf = float(np.clip(1.0 - abs(lm.z) * 2.0, 0.80, 1.0))
        landmarks.append({"id": idx, "x": float(px), "y": float(py), "confidence": conf})
        xs.append(px)
        ys.append(py)

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    bbox = {
        "x": float(x_min),
        "y": float(y_min),
        "w": float(x_max - x_min),
        "h": float(y_max - y_min),
    }
    mean_conf = float(np.mean([lm["confidence"] for lm in landmarks]))
    return bbox, landmarks, mean_conf


def _deterministic_fallback_embedding(image_path: str) -> np.ndarray:
    """SHA-256-seeded 512-d unit vector (deterministic, no GPU needed)."""
    digest = hashlib.sha256(Path(image_path).read_bytes()).digest()
    seed = int.from_bytes(digest[:4], "big")
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=512).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-9 else vec


# ---------------------------------------------------------------------------
# Public adapter class
# ---------------------------------------------------------------------------

class RecognitionAdapter:
    """
    Thin wrapper around the real recognition stack.

    detect_and_landmark() — MediaPipe FaceMesh, 468-point schema
    embed()               — nexgen_engine.FacialRecognitionPipeline (512-d)
    """

    def __init__(
        self,
        model_name: str = "nexgen-facemesh-468",
        model_version: str = "1.0.0",
    ):
        self.model_name = model_name
        self.model_version = model_version

    # ------------------------------------------------------------------
    def detect_and_landmark(self, image_path: str) -> DetectionResult:
        """
        Runs MediaPipe FaceMesh on the supplied image.

        Returns a DetectionResult with landmark_schema="468".
        Falls back to a synthetic 68-point layout only if no face is
        detected (preserves pipeline operation for low-quality probe images).
        """
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError(f"Could not read image: {image_path}")

        bbox, landmarks, confidence = _mediapipe_detect(image_bgr)

        if bbox is None or not landmarks:
            logger.warning(
                "No face detected by MediaPipe in '%s'. "
                "Using synthetic fallback landmarks.", image_path
            )
            h, w = image_bgr.shape[:2]
            bbox = {"x": w * 0.2, "y": h * 0.15, "w": w * 0.6, "h": h * 0.7}
            landmarks = _synthetic_68_landmarks(bbox)
            return DetectionResult(
                bbox=bbox,
                detection_confidence=0.10,
                landmarks=landmarks,
                landmark_schema="68",
            )

        return DetectionResult(
            bbox=bbox,
            detection_confidence=confidence,
            landmarks=landmarks,
            landmark_schema="468",
        )

    # ------------------------------------------------------------------
    def embed(self, image_path: str, detection: DetectionResult) -> EmbeddingResult:
        """
        Calls nexgen_engine.FacialRecognitionPipeline to extract a real
        512-dimensional L2-normalised embedding.

        The pipeline is called as a black box — its weights and internals
        are never modified.  Falls back to a deterministic SHA-256 vector
        when the engine is unavailable (CI / no GPU).
        """
        t0 = time.time()

        pipeline = _get_nexgen_pipeline()
        if pipeline is not None:
            try:
                from PIL import Image as PILImage
                with PILImage.open(image_path).convert("RGB") as img:
                    result = pipeline.encode_image(img)
                vector = result.embedding.astype(np.float32)
                version = "nexgen-engine-1.0"
            except Exception as exc:
                logger.warning(
                    "nexgen_engine.encode_image() failed for '%s': %s. "
                    "Using SHA-256 fallback.", image_path, exc
                )
                vector = _deterministic_fallback_embedding(image_path)
                version = "sha256-fallback-1.0"
        else:
            vector = _deterministic_fallback_embedding(image_path)
            version = "sha256-fallback-1.0"

        dt = (time.time() - t0) * 1000.0
        return EmbeddingResult(
            vector=vector,
            model_name=self.model_name,
            model_version=version,
            inference_time_ms=round(dt, 2),
        )


# ---------------------------------------------------------------------------
# Fallback helpers (used only when MediaPipe finds no face)
# ---------------------------------------------------------------------------

def _synthetic_68_landmarks(bbox: dict) -> list[dict]:
    """Generates a plausible 68-point layout inside bbox for demo/testing."""
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    rng = np.random.default_rng(42)
    pts = []
    regions = {
        "jaw":    (range(0, 17),  0.0,  1.0,  0.75, 1.05),
        "brow_r": (range(17, 22), 0.15, 0.4,  0.28, 0.35),
        "brow_l": (range(22, 27), 0.55, 0.85, 0.28, 0.35),
        "nose":   (range(27, 36), 0.42, 0.58, 0.3,  0.65),
        "eye_r":  (range(36, 42), 0.18, 0.42, 0.38, 0.46),
        "eye_l":  (range(42, 48), 0.55, 0.8,  0.38, 0.46),
        "mouth":  (range(48, 68), 0.32, 0.68, 0.72, 0.85),
    }
    for region, (idxs, x0f, x1f, y0f, y1f) in regions.items():
        idxs = list(idxs)
        n = len(idxs)
        for k, i in enumerate(idxs):
            frac = k / max(n - 1, 1)
            px = x + (x0f + (x1f - x0f) * frac) * w + rng.normal(0, 0.5)
            py = y + (y0f + (y1f - y0f) * abs(np.sin(frac * np.pi))) * h + rng.normal(0, 0.5)
            pts.append({"id": i, "x": float(px), "y": float(py), "confidence": 0.5})
    pts.sort(key=lambda p: p["id"])
    return pts
