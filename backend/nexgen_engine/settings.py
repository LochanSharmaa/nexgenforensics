from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


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
            cors_origins=tuple(item.strip() for item in os.getenv("NEXGEN_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if item.strip()),
        )

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
