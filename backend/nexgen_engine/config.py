from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class QualityConfig:
    """Gates applied to a probe before it is allowed to reach the matcher."""

    min_quality_score: float = 0.35
    min_detection_confidence: float = 0.70
    min_face_pixels: int = 60
    max_yaw: float = 35.0
    max_pitch: float = 30.0
    max_roll: float = 25.0
    min_brightness: float = 40.0
    max_brightness: float = 220.0
    min_contrast_score: float = 0.30
    min_sharpness_score: float = 0.30
    keep_top_fraction: float = 0.65


@dataclass(frozen=True)
class ThresholdConfig:
    """Cosine-similarity operating points on L2-normalized ArcFace templates.

    Defaults are the widely used ArcFace/buffalo_l starting points, not
    guarantees. Every deployment must recalibrate against its own imagery --
    see ``scripts/calibrate_threshold.py`` -- because the false-match rate at a
    given threshold depends heavily on gallery size and capture conditions.
    """

    match: float = 0.42
    review: float = 0.32
    verify: float = 0.42


@dataclass(frozen=True)
class SecurityConfig:
    # Passive single-image screens. These are heuristics, not certified
    # presentation-attack detection; see nexgen_engine/security/liveness.py.
    liveness_threshold: float = 0.45
    deepfake_threshold: float = 0.65
    morphing_threshold: float = 0.55
    template_key_bytes: int = 32
    audit_hash_algorithm: str = "sha256"


@dataclass(frozen=True)
class SearchConfig:
    embedding_dim: int = 512
    top_k: int = 20
    # Candidates below this are never returned, regardless of top_k.
    min_candidate_score: float = 0.20


@dataclass(frozen=True)
class EngineConfig:
    """Configuration for the recognition engine.

    There is no "mode" any more. The engine either loads real weights or raises
    ``EngineUnavailableError``; earlier revisions had an ``auto`` mode that fell
    back to hashing pixels into a vector, which kept the API answering while
    making every score meaningless.

    ``device`` is a request. CUDA is used when the CUDA execution provider is
    actually registered, otherwise the engine runs on CPU and says so. The model
    and the arithmetic are identical either way, so results do not depend on it.
    """

    model_pack: str = "buffalo_l"
    model_root: str | None = None
    device: str = "cpu"
    # Upper bound for the detector input. The detector picks the smallest size
    # that fits each image, so this is a ceiling and not a fixed cost.
    detection_size: tuple[int, int] = (640, 640)
    min_detection_confidence: float = 0.5
    embedding_dim: int = 512
    # Average the template over the crop and its mirror. Horizontal flip is the
    # only augmentation that is safe here: brightness and sharpening shifts move
    # the template off the manifold ArcFace was trained on and cost accuracy.
    use_flip_tta: bool = True

    quality: QualityConfig = field(default_factory=QualityConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    search: SearchConfig = field(default_factory=SearchConfig)

    def __post_init__(self) -> None:
        if self.device not in {"cpu", "cuda"}:
            raise ValueError(f"Unknown device {self.device!r}; expected cpu or cuda.")
        if not 0.0 < self.min_detection_confidence <= 1.0:
            raise ValueError("min_detection_confidence must be in (0, 1].")

    def with_overrides(self, **changes: object) -> "EngineConfig":
        return replace(self, **changes)  # type: ignore[arg-type]


__all__ = [
    "EngineConfig",
    "QualityConfig",
    "SearchConfig",
    "SecurityConfig",
    "ThresholdConfig",
]
