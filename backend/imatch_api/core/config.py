from __future__ import annotations

import base64
import logging
import secrets
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Decision thresholds have exactly one home. Importing it here (rather than
# repeating the numbers) is what keeps this file from becoming a stale copy.
from nexgen_engine.config import ThresholdConfig as _EngineThresholds

from nexgen_engine.config import EngineConfig, QualityConfig, SecurityConfig, ThresholdConfig

logger = logging.getLogger(__name__)

#: Cached ephemeral JWT secret for non-production runs. Must persist for the
#: lifetime of the process: regenerating it per call breaks every token the
#: moment it is verified. Never used when NEXGEN_JWT_SECRET is set.
_EPHEMERAL_JWT_SECRET: str | None = None

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Runtime configuration, read from the environment or a .env file.

    Every field is prefixed ``NEXGEN_`` in the environment. Production-critical
    secrets have no usable default: the service refuses to start in production
    without them rather than silently running on a well-known development key.
    """

    model_config = SettingsConfigDict(
        env_prefix="NEXGEN_",
        env_file=(REPO_ROOT / ".env", REPO_ROOT / "backend" / ".env", ".env"),
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

    # ------------------------------------------------------- account auth --
    # Self-service registration. OFF by default and deliberately so: this is a
    # biometric investigation tool, and who can run a search is a controlled
    # decision, not a signup form. Turn it on per deployment with
    # NEXGEN_ALLOW_SELF_REGISTRATION=true, and restrict who may register with
    # NEXGEN_REGISTRATION_ALLOWED_DOMAINS.
    allow_self_registration: bool = False
    registration_allowed_domains: str = ""
    registration_default_role: str = "investigator"

    otp_length: int = 6
    otp_ttl_minutes: int = 10
    otp_max_attempts: int = 5
    otp_resend_max: int = 3
    otp_resend_window_minutes: int = 30

    reset_token_ttl_minutes: int = 15

    max_failed_logins: int = 5
    lockout_minutes: int = 15

    # "Remember me" only extends the REFRESH token. The access token lifetime
    # is unchanged, so a stolen access token is never long-lived.
    remember_me_refresh_days: int = 30

    auth_cookies_enabled: bool = True
    cookie_secure: bool = False          # must be True behind HTTPS
    cookie_samesite: str = "lax"
    cookie_domain: str = ""

    # ---------------------------------------------------------------- mail --
    # Read from RESEND_API_KEY exactly, NOT NEXGEN_RESEND_API_KEY. The
    # validation_alias bypasses this class's NEXGEN_ env_prefix because the
    # variable name is fixed by Resend's own convention and by the deployment
    # brief.
    resend_api_key: str = Field(default="", validation_alias="RESEND_API_KEY")
    mail_from: str = "NexGen Forensics <onboarding@resend.dev>"
    mail_reply_to: str = ""
    app_public_url: str = "http://localhost:5173"
    # When no API key is configured, e-mails are written to this file instead
    # of being sent. That keeps local development and the test-suite working
    # without a network call, and makes "did we send it?" checkable.
    mail_outbox_path: str = "runtime/mail_outbox.jsonl"

    # -------------------------------------------------------------- engine --
    # buffalo_l (w600k_r50) is deployed DELIBERATELY, and it is not the model
    # with the best clean-benchmark score. Measured, 1:1 verification:
    #
    #                     clean mean (5 protocols)   TinyFace    TAR@FAR=0.1%
    #   glintr100                 97.35 %            79.68 %       17.37 %
    #   w600k_r50  (deployed)     97.16 %            82.45 %       33.13 %
    #
    # glintr100 wins the clean sets by 0.19 points and loses degraded
    # surveillance imagery badly -- barely half the true-accept rate at
    # FAR=0.1%, the operating point that matters for casework. Clean-benchmark
    # ranking does not predict degraded-footage ranking.
    #
    # Real investigative footage is degraded, so the degraded number governs.
    # Override with NEXGEN_MODEL_PACK if a deployment is exclusively clean
    # imagery (e.g. passport-to-passport), and re-tune the threshold if you do:
    # the optimum is model-specific (w600k_r50 -> 0.20, glintr100 -> 0.22).
    # See BENCHMARKS.md sections 3 and 4.
    model_pack: str = "buffalo_l"
    model_root: str = ""
    # "auto": use CUDA when it actually binds, else CPU. The old "cpu" default
    # meant this service ran ArcFace on CPU on a correctly configured GPU host
    # and reported itself healthy while doing it. Force with NEXGEN_ENGINE_DEVICE.
    engine_device: str = "auto"
    # 10-fold cross-validated on AgeDB-30 for w600k_r50, the pack this service
    # loads (BENCHMARKS.md section 2). The previous 0.42/0.32 were unmeasured
    # README values sitting well above every observed optimum (0.18-0.29), so
    # genuine matches were reported as non-matches. Override per deployment with
    # NEXGEN_MATCH_THRESHOLD / NEXGEN_VERIFY_THRESHOLD after re-running the
    # benchmark on that deployment's model pack.
    # Defaults are DERIVED from nexgen_engine.config.ThresholdConfig, the single
    # source of truth, so this file can never drift into being a second copy.
    # Override per deployment with NEXGEN_MATCH_THRESHOLD / NEXGEN_REVIEW_THRESHOLD
    # / NEXGEN_VERIFY_THRESHOLD after re-running the benchmark on that
    # deployment's model pack -- the optimum is model-specific
    # (w600k_r50 -> 0.20, glintr100 -> 0.22).
    match_threshold: float = Field(default_factory=lambda: _EngineThresholds().match)
    review_threshold: float = Field(default_factory=lambda: _EngineThresholds().review)
    verify_threshold: float = Field(default_factory=lambda: _EngineThresholds().verify)
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
        if value not in {"cpu", "cuda", "auto"}:
            raise ValueError("NEXGEN_ENGINE_DEVICE must be cpu, cuda, or auto.")
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

        The ephemeral secret is cached at module scope. It previously called
        secrets.token_urlsafe() on EVERY invocation, so a token was signed with
        one secret and verified against a freshly generated different one --
        every login succeeded and every authenticated request then failed with
        "Authentication required." Any deployment without NEXGEN_JWT_SECRET set
        had no working authentication at all.
        """
        if self.jwt_secret.strip():
            return self.jwt_secret

        global _EPHEMERAL_JWT_SECRET
        if _EPHEMERAL_JWT_SECRET is None:
            _EPHEMERAL_JWT_SECRET = secrets.token_urlsafe(64)
            logger.warning(
                "NEXGEN_JWT_SECRET is unset; generated an ephemeral development "
                "secret. All issued tokens become invalid when this process exits."
            )
        return _EPHEMERAL_JWT_SECRET

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
