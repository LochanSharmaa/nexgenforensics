from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True)
class Settings:
    env: str
    secret_key: str
    database_url: str
    redis_url: str
    require_production_mode: bool
    allow_local_fallbacks: bool
    enable_prometheus: bool
    enable_cuda: bool
    require_cuda: bool
    enable_tensorrt: bool
    require_tensorrt: bool
    device: str
    precision: str
    otel_exporter_otlp_endpoint: str
    sentry_dsn: str
    model_dir: Path
    runtime_dir: Path
    faiss_index_path: Path
    checkpoint_path: Path
    upload_path: Path
    encryption_key: str
    postgres_pool_min: int
    postgres_pool_max: int
    redis_pool_size: int
    max_upload_size_gb: int
    liveness_threshold: float
    deepfake_threshold: float
    morphing_threshold: float
    quality_threshold: float
    match_threshold: float
    cors_origins: tuple[str, ...]

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            env=os.getenv("NEXGEN_ENV", "local"),
            secret_key=os.getenv("NEXGEN_SECRET_KEY", "change-me-use-a-strong-secret"),
            database_url=os.getenv("NEXGEN_DATABASE_URL", "sqlite:///runtime/nexgen.db"),
            redis_url=os.getenv("NEXGEN_REDIS_URL", "redis://localhost:6379/0"),
            require_production_mode=_bool("NEXGEN_REQUIRE_PRODUCTION_MODE", False),
            allow_local_fallbacks=_bool("NEXGEN_ALLOW_LOCAL_FALLBACKS", True),
            enable_prometheus=_bool("NEXGEN_ENABLE_PROMETHEUS", True),
            enable_cuda=_bool("NEXGEN_ENABLE_CUDA", False),
            require_cuda=_bool("NEXGEN_REQUIRE_CUDA", False),
            enable_tensorrt=_bool("NEXGEN_ENABLE_TENSORRT", False),
            require_tensorrt=_bool("NEXGEN_REQUIRE_TENSORRT", False),
            device=os.getenv("NEXGEN_DEVICE", "cpu"),
            precision=os.getenv("NEXGEN_PRECISION", "fp32"),
            otel_exporter_otlp_endpoint=os.getenv("NEXGEN_OTEL_EXPORTER_OTLP_ENDPOINT", ""),
            sentry_dsn=os.getenv("NEXGEN_SENTRY_DSN", ""),
            model_dir=Path(os.getenv("NEXGEN_MODEL_DIR", "models")),
            runtime_dir=Path(os.getenv("NEXGEN_RUNTIME_DIR", "runtime")),
            faiss_index_path=Path(os.getenv("NEXGEN_FAISS_INDEX_PATH", "runtime/faiss.index")),
            checkpoint_path=Path(os.getenv("NEXGEN_CHECKPOINT_PATH", "runtime/checkpoints")),
            upload_path=Path(os.getenv("NEXGEN_UPLOAD_PATH", "runtime/uploads")),
            encryption_key=os.getenv("NEXGEN_ENCRYPTION_KEY", ""),
            postgres_pool_min=_int("NEXGEN_POSTGRES_POOL_MIN", 5),
            postgres_pool_max=_int("NEXGEN_POSTGRES_POOL_MAX", 20),
            redis_pool_size=_int("NEXGEN_REDIS_POOL_SIZE", 10),
            max_upload_size_gb=_int("NEXGEN_MAX_UPLOAD_SIZE_GB", 50),
            liveness_threshold=_float("NEXGEN_LIVENESS_THRESHOLD", 0.75),
            deepfake_threshold=_float("NEXGEN_DEEPFAKE_THRESHOLD", 0.85),
            morphing_threshold=_float("NEXGEN_MORPHING_THRESHOLD", 0.30),
            quality_threshold=_float("NEXGEN_QUALITY_THRESHOLD", 0.60),
            match_threshold=_float("NEXGEN_MATCH_THRESHOLD", 0.65),
            cors_origins=tuple(item.strip() for item in os.getenv("NEXGEN_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if item.strip()),
        )

    @property
    def database_backend(self) -> str:
        scheme = urlparse(self.database_url).scheme
        if scheme.startswith("postgresql"):
            return "postgres"
        if scheme.startswith("sqlite"):
            return "sqlite"
        return scheme or "unknown"

    @property
    def redis_enabled(self) -> bool:
        return bool(self.redis_url.strip())

    def validate_runtime_mode(self, runtime_mode: str) -> None:
        if self.require_production_mode and runtime_mode != "production_ready_dependencies":
            raise RuntimeError(
                "Production mode requires torch+timm+faiss and configured checkpoints. "
                "Disable NEXGEN_REQUIRE_PRODUCTION_MODE only for local development."
            )
        if self.require_cuda and not self.enable_cuda:
            raise RuntimeError("NEXGEN_REQUIRE_CUDA=true requires NEXGEN_ENABLE_CUDA=true.")
        if self.require_tensorrt and not self.enable_tensorrt:
            raise RuntimeError("NEXGEN_REQUIRE_TENSORRT=true requires NEXGEN_ENABLE_TENSORRT=true.")


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default
