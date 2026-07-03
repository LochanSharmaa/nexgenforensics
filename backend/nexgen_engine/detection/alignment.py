from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL import Image


@dataclass(frozen=True)
class FaceBox:
    left: int
    top: int
    right: int
    bottom: int
    confidence: float
    landmarks: tuple[tuple[float, float], ...] = ()
    backend: str = "unknown"


class FaceDetectorProtocol(Protocol):
    def detect(self, image: Image.Image) -> list[FaceBox]:
        ...


class CenterCropDetector:
    def detect(self, image: Image.Image) -> list[FaceBox]:
        width, height = image.size
        side = int(min(width, height) * 0.72)
        left = (width - side) // 2
        top = (height - side) // 2
        landmarks = (
            (left + side * 0.32, top + side * 0.38),
            (left + side * 0.68, top + side * 0.38),
            (left + side * 0.50, top + side * 0.52),
            (left + side * 0.38, top + side * 0.70),
            (left + side * 0.62, top + side * 0.70),
        )
        return [FaceBox(left, top, left + side, top + side, 0.99, landmarks, "center_crop")]


class InsightFaceRetinaFaceDetector:
    def __init__(self, min_confidence: float = 0.95) -> None:
        self.min_confidence = min_confidence
        self._app = None
        self._load()

    def detect(self, image: Image.Image) -> list[FaceBox]:
        if self._app is None:
            raise RuntimeError("InsightFace RetinaFace detector requires insightface and onnxruntime. Install them and configure model providers.")
        import numpy as np

        faces = self._app.get(np.asarray(image.convert("RGB"))[:, :, ::-1])
        boxes: list[FaceBox] = []
        for face in faces:
            confidence = float(face.det_score)
            if confidence < self.min_confidence:
                continue
            bbox = face.bbox.astype(int).tolist()
            landmarks = tuple((float(x), float(y)) for x, y in getattr(face, "kps", []))
            boxes.append(FaceBox(bbox[0], bbox[1], bbox[2], bbox[3], confidence, landmarks, "insightface_retinaface"))
        return boxes

    def _load(self) -> None:
        try:
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=(640, 640))
            self._app = app
        except Exception:
            self._app = None


class MtcnnDetector:
    def __init__(self, min_confidence: float = 0.95) -> None:
        self.min_confidence = min_confidence
        self._detector = None
        self._load()

    def detect(self, image: Image.Image) -> list[FaceBox]:
        if self._detector is None:
            raise RuntimeError("MTCNN detector requires facenet-pytorch or compatible MTCNN dependency.")
        boxes, probs, landmarks = self._detector.detect(image.convert("RGB"), landmarks=True)
        results: list[FaceBox] = []
        if boxes is None or probs is None:
            return results
        for box, prob, points in zip(boxes, probs, landmarks or []):
            confidence = float(prob or 0)
            if confidence < self.min_confidence:
                continue
            left, top, right, bottom = [int(value) for value in box]
            results.append(FaceBox(left, top, right, bottom, confidence, tuple((float(x), float(y)) for x, y in points), "mtcnn"))
        return results

    def _load(self) -> None:
        try:
            from facenet_pytorch import MTCNN

            self._detector = MTCNN(keep_all=True, device="cpu")
        except Exception:
            self._detector = None


class FaceAligner:
    def __init__(self, output_size: int = 112, detector: FaceDetectorProtocol | None = None) -> None:
        self.output_size = output_size
        self.detector = detector or CenterCropDetector()

    def align(self, image: Image.Image) -> Image.Image:
        boxes = self.detector.detect(image)
        if not boxes:
            raise ValueError("No face candidate detected.")
        box = max(boxes, key=lambda item: item.confidence)
        crop = image.convert("RGB").crop((box.left, box.top, box.right, box.bottom))
        return crop.resize((self.output_size, self.output_size), Image.Resampling.BICUBIC)
