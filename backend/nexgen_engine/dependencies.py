from __future__ import annotations

import importlib.util
from dataclasses import dataclass


@dataclass(frozen=True)
class DependencyStatus:
    name: str
    available: bool
    required_for: str
    install_hint: str


class DependencyVerifier:
    checks = {
        "torch": ("production model training/inference", "Install the platform-specific torch build from pytorch.org."),
        "torchvision": ("production image transforms", "Install torchvision matching your torch build."),
        "timm": ("ViT/Swin/BEiT/EfficientNet backbones", "pip install timm"),
        "insightface": ("RetinaFace/ResNet100 adapters", "pip install insightface onnxruntime"),
        "faiss": ("FAISS vector search", "Install faiss-cpu locally or faiss-gpu in CUDA images."),
        "onnx": ("ONNX export validation", "pip install onnx"),
        "onnxruntime": ("ONNX runtime validation", "pip install onnxruntime"),
        "tensorrt": ("TensorRT engine conversion", "Install NVIDIA TensorRT inside a CUDA/TensorRT container."),
        "cv2": ("OpenCV detector/quality helpers", "pip install opencv-python-headless"),
        "skimage": ("BRISQUE-style image quality features", "pip install scikit-image"),
    }

    def statuses(self) -> list[DependencyStatus]:
        return [
            DependencyStatus(name, importlib.util.find_spec(name) is not None, required_for, hint)
            for name, (required_for, hint) in self.checks.items()
        ]

    def assert_available(self, names: list[str]) -> None:
        status = {item.name: item for item in self.statuses()}
        missing = [status[name] for name in names if not status.get(name, DependencyStatus(name, False, "", "")).available]
        if missing:
            details = "; ".join(f"{item.name}: {item.install_hint}" for item in missing)
            raise RuntimeError(f"Missing required production dependencies: {details}")
