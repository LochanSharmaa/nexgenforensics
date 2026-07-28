from __future__ import annotations

import base64
import logging
import secrets
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from nexgen_engine.config import EngineConfig, QualityConfig, SecurityConfig, ThresholdConfig

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Runtime configuration, read from the environment or a .env file.

    Every field is prefixed ``NEXGEN_`` in the environment. Production-critical
    secrets have no usable default: the service refuses to start in production
    without them rather than silently running on a well-known development key.
    """

    model_config = SettingsConfigDict(
        env_prefix="NEXGEN_",
        env_file=(REPO_ROOT / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        # model_pack and model_root would otherwise collide with Pydantic's
        # reserved "model_" namespace and warn on every instantiation. Renaming
        # them would change the documented NEXGEN_MODEL_PACK / NEXGEN_MODEL_ROOT
        # environment variables, so the namespace guard is disabled instead.
        protected_namespaces=(),
    )

    # ------------------------------------------------------------- service --
    env: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8443
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # ------------------------------------------------------------ database --
    database_url: str = "sqlite:///./runtime/imatch.db"

    # ------------------------------------------------------------ security --
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 60
    refresh_token_days: int = 7
    template_key: str = ""
    rate_limit_per_minute: int = 120
    search_rate_limit_per_minute: int = 30

    # -------------------------------------------------------------- engine --
    model_pack: str = "buffalo_l"
    model_root: str = ""
    engine_device: str = "cpu"
    match_threshold: float = 0.42
    review_threshold: float = 0.32
    verify_threshold: float = 0.42
    min_quality: float = 0.35
    min_detection_confidence: float = 0.70

    # ---------------------------------------------------------- governance --
    require_lawful_basis: bool = True
    probe_retention_days: int = 90
    audit_log_path: str = "./runtime/audit.jsonl"

    # ------------------------------------------------------------- storage --
    storage_root: str = "./runtime/storage"
    max_upload_mb: int = 15

    # ----------------------------------------------------------- bootstrap --
    seed_tenant: str = "nexgen-demo"
    seed_admin_email: str = "admin@example.com"
    seed_admin_password: str = ""

    # ---------------------------------------------------------- validation --

    @field_validator("engine_device")
    @classmethod
    def _check_device(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"cpu", "cuda"}:
            raise ValueError("NEXGEN_ENGINE_DEVICE must be cpu or cuda.")
        return value

    @field_validator("env")
    @classmethod
    def _check_env(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"development", "staging", "production", "test"}:
            raise ValueError("NEXGEN_ENV must be development, staging, production, or test.")
        return value

    @model_validator(mode="after")
    def _check_secrets(self) -> "Settings":
        if self.review_threshold > self.match_threshold:
            raise ValueError(
                "NEXGEN_REVIEW_THRESHOLD must not exceed NEXGEN_MATCH_THRESHOLD; "
                "otherwise the review band is empty and borderline scores are auto-accepted."
            )

        if self.is_production:
            missing = [
                name
                for name, value in (("NEXGEN_JWT_SECRET", self.jwt_secret), ("NEXGEN_TEMPLATE_KEY", self.template_key))
                if not value.strip()
            ]
            if missing:
                raise ValueError(
                    f"{' and '.join(missing)} must be set when NEXGEN_ENV=production. "
                    "Refusing to start with a generated key: tokens would be invalidated on every "
                    "restart and stored biometric templates would become permanently unreadable."
                )
            if len(self.jwt_secret) < 32:
                raise ValueError("NEXGEN_JWT_SECRET must be at least 32 characters in production.")

        if self.template_key.strip():
            try:
                decoded = base64.b64decode(self.template_key, validate=True)
            except Exception as exc:
                raise ValueError("NEXGEN_TEMPLATE_KEY must be valid base64.") from exc
            if len(decoded) != 32:
                raise ValueError(
                    f"NEXGEN_TEMPLATE_KEY must decode to exactly 32 bytes, got {len(decoded)}."
                )

        return self

    # ------------------------------------------------------------ derived ---

    @property
    def is_production(self) -> bool:
        return self.env in {"production", "staging"}

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def storage_path(self) -> Path:
        return self._resolve(self.storage_root)

    @property
    def audit_path(self) -> Path:
        return self._resolve(self.audit_log_path)

    def resolved_jwt_secret(self) -> str:
        """The signing secret, generating an ephemeral one only outside production.

        A per-process random secret means restarting the dev server logs everyone
        out. That is the correct trade: the alternative is a hard-coded default
        that eventually ships to production.
        """
        if self.jwt_secret.strip():
            return self.jwt_secret
        logger.warning(
            "NEXGEN_JWT_SECRET is unset; generating an ephemeral development secret. "
            "All issued tokens become invalid when this process exits."
        )
        return secrets.token_urlsafe(64)

    def resolved_template_key(self) -> str:
        """Base64 master key for template encryption.

        Outside production a random key is generated so tests and demos work out
        of the box; stored templates then cannot be decrypted after a restart,
        which is loudly logged.
        """
        if self.template_key.strip():
            return self.template_key
        logger.warning(
            "NEXGEN_TEMPLATE_KEY is unset; generating an ephemeral key. Templates enrolled in "
            "this process will be UNREADABLE after restart. Set a persistent key before storing "
            "anything you intend to keep."
        )
        return base64.b64encode(secrets.token_bytes(32)).decode("ascii")

    def engine_config(self) -> EngineConfig:
        return EngineConfig(
            model_pack=self.model_pack,
            model_root=self.model_root or None,
            device=self.engine_device,
            min_detection_confidence=self.min_detection_confidence,
            quality=QualityConfig(
                min_quality_score=self.min_quality,
                min_detection_confidence=self.min_detection_confidence,
            ),
            thresholds=ThresholdConfig(
                match=self.match_threshold,
                review=self.review_threshold,
                verify=self.verify_threshold,
            ),
            security=SecurityConfig(),
        )

    def _resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (REPO_ROOT / path).resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


__all__ = ["REPO_ROOT", "Settings", "get_settings"]
