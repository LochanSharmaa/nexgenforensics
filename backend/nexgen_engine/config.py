from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BackboneConfig:
    name: str
    provider: str
    model_id: str
    embedding_dim: int = 2048
    weight: float = 0.125
    image_size: int = 224
    specialty: str = "general"


@dataclass(frozen=True)
class SearchConfig:
    embedding_dim: int = 512
    top_k: int = 20
    nlist: int = 65536
    nprobe: int = 512
    pq_segments: int = 64
    min_match_score: float = 0.72


@dataclass(frozen=True)
class QualityConfig:
    min_detector_confidence: float = 0.95
    min_faceqnet_score: float = 0.60
    max_yaw: float = 35.0
    max_pitch: float = 30.0
    max_roll: float = 25.0
    min_resolution: int = 80
    min_laplacian_variance: float = 100.0
    max_occlusion: float = 0.30
    min_brightness: float = 40.0
    max_brightness: float = 220.0
    keep_top_fraction: float = 0.65


@dataclass(frozen=True)
class SecurityConfig:
    liveness_threshold: float = 0.95
    deepfake_threshold: float = 0.85
    morphing_threshold: float = 0.30
    pbkdf2_iterations: int = 600_000
    template_key_bytes: int = 32
    audit_hash_algorithm: str = "sha256"


@dataclass(frozen=True)
class EngineConfig:
    final_embedding_dim: int = 512
    cohort_size: int = 200
    verification_threshold: float = 0.82
    backbones: tuple[BackboneConfig, ...] = field(
        default_factory=lambda: (
            BackboneConfig("vit_h14", "timm", "vit_huge_patch14_224", weight=0.25, image_size=224),
            BackboneConfig("swin_l", "timm", "swin_large_patch4_window12_384", weight=0.18, image_size=384),
            BackboneConfig("beit_l", "timm", "beit_large_patch16_224", weight=0.17, image_size=224),
            BackboneConfig("resnet100", "insightface", "iresnet100", weight=0.12, image_size=112),
            BackboneConfig("efficientnet_xl", "timm", "tf_efficientnetv2_xl", weight=0.10, image_size=384),
            BackboneConfig("age_invariant", "custom", "vit_base_patch16_224", weight=0.10, specialty="aging"),
            BackboneConfig("pose_net", "custom", "swin_base_patch4_window7_224", weight=0.05, specialty="pose"),
            BackboneConfig("ir_face_net", "custom", "resnet50", weight=0.03, specialty="ir"),
        )
    )
    quality: QualityConfig = field(default_factory=QualityConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
