from __future__ import annotations

import importlib.util
import logging
import threading
from dataclasses import dataclass

from .config import EngineConfig
from .detection.detector import InsightFaceDetector, build_detector
from .models.arcface import (
    MODEL_PACKS,
    ArcFaceRecognizer,
    EngineUnavailableError,
    FaceRecognizer,
)

logger = logging.getLogger(__name__)

CUDA_PROVIDER = "CUDAExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"


@dataclass(frozen=True)
class RuntimeCapabilities:
    insightface: bool
    onnxruntime: bool
    opencv: bool
    faiss: bool
    torch: bool
    cuda_provider: bool
    onnx_providers: tuple[str, ...]

    @property
    def can_recognize(self) -> bool:
        return self.insightface and self.onnxruntime and self.opencv

    def as_dict(self) -> dict[str, object]:
        return {
            "insightface": self.insightface,
            "onnxruntime": self.onnxruntime,
            "opencv": self.opencv,
            "faiss": self.faiss,
            "torch": self.torch,
            "cuda_provider": self.cuda_provider,
            "onnx_providers": list(self.onnx_providers),
            "can_recognize": self.can_recognize,
        }

    def missing(self) -> list[str]:
        required = {
            "insightface": self.insightface,
            "onnxruntime": self.onnxruntime,
            "opencv-python-headless": self.opencv,
        }
        return [name for name, present in required.items() if not present]


def _installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - malformed installs
        return False


def detect_runtime_capabilities() -> RuntimeCapabilities:
    providers: tuple[str, ...] = ()
    if _installed("onnxruntime"):
        try:
            import onnxruntime

            providers = tuple(onnxruntime.get_available_providers())
        except Exception:  # pragma: no cover - host-specific
            providers = ()
    return RuntimeCapabilities(
        insightface=_installed("insightface"),
        onnxruntime=_installed("onnxruntime"),
        opencv=_installed("cv2"),
        faiss=_installed("faiss"),
        torch=_installed("torch"),
        cuda_provider=CUDA_PROVIDER in providers,
        onnx_providers=providers,
    )


def resolve_providers(requested_device: str) -> tuple[list[str], str]:
    """Pick ONNX execution providers, returning ``(providers, effective_device)``.

    ``cuda`` is a request, not a guarantee. The stock ``onnxruntime`` wheel is
    CPU-only, and even ``onnxruntime-gpu`` ships kernels only for the compute
    capabilities it was built against -- Maxwell cards such as the Quadro M1200
    (sm_50) are outside the range of recent builds. Rather than fail or pretend,
    the requested device is checked against the providers actually registered
    and the effective device is reported back and logged.

    This is a device choice, never a model choice: the same ArcFace weights and
    the same arithmetic run either way, so results do not change with the
    device.
    """
    capabilities = detect_runtime_capabilities()

    if requested_device == "cuda":
        if capabilities.cuda_provider:
            return [CUDA_PROVIDER, CPU_PROVIDER], "cuda"
        logger.warning(
            "CUDA was requested but %s is not registered (available: %s). Running on CPU. "
            "Install onnxruntime-gpu built for this GPU's compute capability to enable it.",
            CUDA_PROVIDER,
            ", ".join(capabilities.onnx_providers) or "none",
        )
        return [CPU_PROVIDER], "cpu"

    return [CPU_PROVIDER], "cpu"


class EngineRuntime:
    """Owns the loaded models and hands out a detector and a recognizer.

    Models load once per process and are shared. Loading is lazy behind a lock so
    concurrent first requests do not each try to load the pack.

    There is no fallback. If the model cannot load, ``EngineUnavailableError`` is
    raised and the caller fails. A previous revision degraded to a stub that
    hashed pixels into a vector; it kept the service answering while making every
    score meaningless, which is the worst possible failure mode for this system.
    """

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self._lock = threading.Lock()
        self._loaded = False
        self._detector: InsightFaceDetector | None = None
        self._recognizer: FaceRecognizer | None = None
        self._analysis_app = None
        self._effective_device = "cpu"
        self._providers: tuple[str, ...] = ()
        self._load_seconds = 0.0

    @property
    def detector(self) -> InsightFaceDetector:
        self._ensure_loaded()
        assert self._detector is not None
        return self._detector

    @property
    def recognizer(self) -> FaceRecognizer:
        self._ensure_loaded()
        assert self._recognizer is not None
        return self._recognizer

    @property
    def recognition_capable(self) -> bool:
        """True once the real model is loaded; raises if it cannot load."""
        self._ensure_loaded()
        return True

    @property
    def device(self) -> str:
        self._ensure_loaded()
        return self._effective_device

    def status(self) -> dict[str, object]:
        self._ensure_loaded()
        assert self._recognizer is not None and self._detector is not None
        return {
            "recognizer": self._recognizer.info.as_dict(),
            "detector": {
                "name": self._detector.name,
                "produces_landmarks": self._detector.produces_landmarks,
                "min_confidence": self._detector.min_confidence,
                "pad_retry": True,
            },
            "device": {
                "requested": self.config.device,
                "effective": self._effective_device,
                "providers": list(self._providers),
            },
            "capabilities": detect_runtime_capabilities().as_dict(),
            "thresholds": {
                "match": self.config.thresholds.match,
                "review": self.config.thresholds.review,
                "verify": self.config.thresholds.verify,
            },
            "embedding_dim": self.config.embedding_dim,
            "flip_tta": self.config.use_flip_tta,
            "model_load_seconds": round(self._load_seconds, 2),
            "recognition_capable": True,
        }

    def warm_up(self) -> None:
        """Load models now rather than inside the first user request."""
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._load()
            self._loaded = True

    def _load(self) -> None:
        import time

        capabilities = detect_runtime_capabilities()
        missing = capabilities.missing()
        if missing:
            raise EngineUnavailableError(
                "Face recognition cannot start: missing "
                + ", ".join(missing)
                + ". Install them with:  pip install -r backend/requirements-engine.txt"
            )

        if self.config.model_pack not in MODEL_PACKS:
            raise EngineUnavailableError(
                f"Unknown model pack {self.config.model_pack!r}. "
                f"Supported: {', '.join(sorted(MODEL_PACKS))}."
            )

        providers, device = resolve_providers(self.config.device)
        started = time.perf_counter()

        try:
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(
                name=self.config.model_pack,
                root=self.config.model_root or "~/.insightface",
                # Loading only what is used. The pack also contains landmark and
                # gender/age models that this system never calls; skipping them
                # saves memory and startup time.
                allowed_modules=["detection", "recognition"],
                providers=providers,
            )
            app.prepare(
                ctx_id=0 if device == "cuda" else -1,
                det_thresh=self.config.min_detection_confidence,
                det_size=self.config.detection_size,
            )
        except Exception as exc:
            raise EngineUnavailableError(
                f"Failed to load model pack {self.config.model_pack!r}: {exc}. "
                "The first run downloads roughly 300 MB into ~/.insightface/models; "
                "check network access and disk space."
            ) from exc

        recognition_model = app.models.get("recognition")
        detection_model = app.models.get("detection")
        if recognition_model is None or detection_model is None:
            raise EngineUnavailableError(
                f"Model pack {self.config.model_pack!r} loaded without "
                f"{'recognition' if recognition_model is None else 'detection'}."
            )

        self._load_seconds = time.perf_counter() - started
        self._analysis_app = app
        self._providers = tuple(providers)
        self._effective_device = device
        self._recognizer = ArcFaceRecognizer(recognition_model, self.config.model_pack, device, tuple(providers))
        self._detector = build_detector(app, self.config.min_detection_confidence)

        logger.info(
            "Recognition engine ready: %s (%s) on %s in %.1fs.",
            self.config.model_pack,
            self._recognizer.info.recognition_network,
            device,
            self._load_seconds,
        )


__all__ = [
    "CPU_PROVIDER",
    "CUDA_PROVIDER",
    "EngineRuntime",
    "RuntimeCapabilities",
    "detect_runtime_capabilities",
    "resolve_providers",
]
