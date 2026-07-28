"""NexGen iMATCH facial recognition engine.

Pure recognition: detection, alignment, template extraction, matching, and the
integrity screens around them. It carries no HTTP, database, or auth concerns --
those live in the ``imatch_api`` package -- so the engine can be embedded,
benchmarked, and tested on its own.

Typical use::

    from nexgen_engine import EngineConfig, FacialRecognitionPipeline, GalleryIndex

    pipeline = FacialRecognitionPipeline(EngineConfig(mode="real"))
    result = pipeline.encode_bytes(image_bytes)

    index = GalleryIndex(dimensions=512)
    index.add("tenant-a", "template-1", "subject-1", result.embedding)
    outcome = index.search("tenant-a", probe_embedding, top_k=10)

Check ``pipeline.runtime.recognition_capable`` before trusting any score: when
the recognition weights are missing the engine falls back to a deterministic
stub that cannot recognize anyone.
"""

from .config import EngineConfig, QualityConfig, SearchConfig, SecurityConfig, ThresholdConfig
from .inference.pipeline import (
    FacialRecognitionPipeline,
    InvalidImageError,
    NoFaceDetectedError,
    RecognitionResult,
    StageTimings,
    decode_image,
)
from .inference.score_fusion import Decision, DecisionEngine
from .models.arcface import MODEL_PACKS, EngineUnavailableError, RecognizerInfo
from .runtime import EngineRuntime, detect_runtime_capabilities, resolve_providers
from .search.gallery_index import GalleryIndex, MatchResult, SearchOutcome, faiss_available

__version__ = "1.0.0"

__all__ = [
    "Decision",
    "DecisionEngine",
    "EngineConfig",
    "EngineRuntime",
    "EngineUnavailableError",
    "FacialRecognitionPipeline",
    "GalleryIndex",
    "InvalidImageError",
    "MatchResult",
    "NoFaceDetectedError",
    "QualityConfig",
    "RecognitionResult",
    "RecognizerInfo",
    "SearchConfig",
    "SearchOutcome",
    "SecurityConfig",
    "ThresholdConfig",
    "decode_image",
    "detect_runtime_capabilities",
]
