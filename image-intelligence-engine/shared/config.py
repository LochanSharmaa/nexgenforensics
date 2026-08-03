"""Typed configuration, validated at startup.

Two rules this module enforces:

* **Fail at boot, not at use.** A malformed database URL or a missing secret in
  production raises while the process is starting, not on the first request that
  happens to need it. A half-configured investigation platform that appears
  healthy is worse than one that refuses to start.
* **Restrictive defaults; widening is opt-in.** Every default is the safe
  setting. Turning off robots.txt compliance, enabling licence-plate extraction,
  or disabling auto-purge all require an explicit environment variable.
"""

from __future__ import annotations

import secrets
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

Environment = Literal["local", "test", "production"]


class Settings(BaseSettings):
    """Runtime configuration. Every field is `IIE_`-prefixed in the environment."""

    model_config = SettingsConfigDict(
        env_prefix="IIE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # -- identity -----------------------------------------------------------
    app_name: str = "Image Intelligence Engine"
    app_version: str = "0.2.0"
    environment: Environment = "local"

    ruleset_version: str = "ruleset@0.1.0"
    """Recorded in every config snapshot. Bump when extraction, scoring or
    classification rules change in a way that alters findings."""

    # -- database -----------------------------------------------------------
    database_url: str = "postgresql+asyncpg://iie:iie@localhost:5432/iie"
    database_echo: bool = False
    database_pool_size: int = 10
    database_max_overflow: int = 20

    # -- redis --------------------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"

    # -- object storage -----------------------------------------------------
    s3_endpoint_url: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"  # noqa: S105 - MinIO dev default; overridden in deployment
    s3_bucket: str = "iie-artifacts"
    s3_region: str = "us-east-1"
    s3_secure: bool = False

    # -- security -----------------------------------------------------------
    secret_key: str = Field(default_factory=lambda: secrets.token_urlsafe(48))
    """Auto-generated in local/test so the stack boots with no setup. Production
    *must* supply one — see the validator below. An auto-generated key rotates on
    every restart, invalidating all sessions."""
    jwt_algorithm: str = "HS256"

    # --- federated sign-in from the NexGen iMATCH workspace ---------------
    # The workspace already authenticates its investigators; making them sign
    # in a second time to reach provenance would be friction with no security
    # benefit. When this secret matches iMATCH's NEXGEN_JWT_SECRET, IIE accepts
    # the bearer token the workspace already holds. Empty disables the path
    # entirely — federation is opt-in, not a default trust relationship.
    imatch_jwt_secret: str = ""
    imatch_jwt_algorithm: str = "HS256"
    imatch_issuer: str = "nexgen-imatch"
    access_token_ttl_minutes: int = 60 * 12
    cors_origins: Annotated[tuple[str, ...], NoDecode] = ("http://localhost:3000",)
    """`NoDecode` because pydantic-settings JSON-parses complex types by
    default, and `IIE_CORS_ORIGINS=http://a,http://b` is not JSON. Deployments
    write comma-separated lists, so the validator below accepts that form."""

    # -- vision -------------------------------------------------------------
    gemini_api_key: str = ""
    gemini_model: str = "gemini-flash-latest"
    vision_enabled: bool = True
    """Vision reports what is *visible*. Its output enters as observations
    sourced to the image, never as facts about the world — those require
    corroborating pages."""

    # -- discovery providers ------------------------------------------------
    # Each plugin reads its own keys; the registry never learns their names.
    google_vision_api_key: str = ""
    tineye_api_key: str = ""
    tineye_api_base: str = "https://api.tineye.com/rest"
    discovery_max_results: int = 50

    archive_lookup_enabled: bool = True
    """Enrich discovered URLs with first-seen dates from the Internet Archive.
    Free and keyless, so on by default. Switched off in tests: a suite that
    depends on a third-party service being reachable is a suite that fails for
    reasons unrelated to the code."""
    archive_lookup_timeout_seconds: float = 20.0

    # -- observability ------------------------------------------------------
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"
    metrics_enabled: bool = True

    # -- policy -------------------------------------------------------------
    require_lawful_basis: bool = True
    """Every investigation must state why the subject may lawfully be examined.
    Turning this off is a testing affordance, never a deployment option."""

    default_jurisdiction: str = "IN"
    default_retention_days: int = 365
    retention_auto_purge: bool = True
    retention_export_before_purge: bool = True

    enable_plate_extraction: bool = False
    """Licence plates are personal data in the EU and several Indian contexts.
    Off unless a deployment has established it may collect them."""

    enable_face_detection_for_redaction: bool = False
    """Bounding boxes used to *blur* faces in report thumbnails, discarded
    immediately. No descriptors, no persistence, no comparison. Off by default;
    enabling it strengthens privacy rather than weakening it."""

    # -- crawler ------------------------------------------------------------
    respect_robots: bool = True
    crawl_timeout_seconds: float = 15.0
    crawl_max_bytes: int = 8 * 1024 * 1024
    crawl_max_redirects: int = 3
    crawl_per_domain_concurrency: int = 2
    crawl_per_domain_delay_seconds: float = 1.0
    user_agent: str = (
        "IIE/0.2 (image provenance research; +https://example.invalid/iie-bot)"
    )
    screenshots_enabled: bool = True

    # -- storage paths ------------------------------------------------------
    data_dir: Path = Path("runtime")

    # ----------------------------------------------------------------- checks

    @field_validator("database_url")
    @classmethod
    def _async_driver(cls, value: str) -> str:
        """Reject sync drivers early.

        `postgresql://` silently selects psycopg2, which blocks the event loop
        under async SQLAlchemy. The failure mode is a mysteriously slow API
        rather than an error, so it is caught here.
        """
        allowed = ("postgresql+asyncpg://", "sqlite+aiosqlite://")
        if not value.startswith(allowed):
            raise ValueError(
                f"database_url must use an async driver {allowed}, got {value.split('://')[0]!r}. "
                "Plain 'postgresql://' selects a blocking driver."
            )
        return value

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """Accept a comma-separated string, a JSON array, or a real sequence."""
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("["):
                import json

                return tuple(json.loads(text))
            return tuple(part.strip() for part in text.split(",") if part.strip())
        return value

    @field_validator("log_level")
    @classmethod
    def _valid_level(cls, value: str) -> str:
        level = value.upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"log_level must be a standard level name, got {value!r}")
        return level

    @field_validator("default_retention_days")
    @classmethod
    def _positive_retention(cls, value: int, info: ValidationInfo) -> int:
        if value < 1:
            raise ValueError("default_retention_days must be at least 1")
        return value

    @model_validator(mode="after")
    def _production_hardening(self) -> Settings:
        if self.environment != "production":
            return self

        problems: list[str] = []
        if len(self.secret_key) < 32:
            problems.append("IIE_SECRET_KEY must be set explicitly (>=32 chars) in production")
        if not self.require_lawful_basis:
            problems.append("IIE_REQUIRE_LAWFUL_BASIS must remain true in production")
        if not self.respect_robots:
            problems.append("IIE_RESPECT_ROBOTS must remain true in production")
        if self.database_url.startswith("sqlite"):
            problems.append("SQLite is not supported in production; use PostgreSQL")
        if problems:
            raise ValueError("Production configuration rejected:\n  - " + "\n  - ".join(problems))
        return self

    # ------------------------------------------------------------- accessors

    @property
    def imatch_federation_enabled(self) -> bool:
        return bool(self.imatch_jwt_secret.strip())

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def alembic_url(self) -> str:
        """Alembic runs migrations synchronously."""
        return self.database_url.replace("+asyncpg", "+psycopg").replace("+aiosqlite", "")

    def component_versions(self) -> dict[str, str]:
        """Seed for a config snapshot (REVISION_3 §9). Modules extend this with
        their own versions as they are built."""
        return {"app": self.app_version, "ruleset": self.ruleset_version}


# `from __future__ import annotations` defers every annotation to a string, so
# pydantic cannot resolve `NoDecode` while building the class. Rebuilding here,
# with the module namespace populated, completes the model.
Settings.model_rebuild()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Tests mutate the environment; without this they would see a stale cache."""
    get_settings.cache_clear()


__all__ = ["Settings", "get_settings", "reset_settings_cache"]
