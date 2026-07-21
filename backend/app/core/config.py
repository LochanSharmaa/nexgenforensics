from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Base Config
    PROJECT_NAME: str = "SIFS Creative Reasoning Engine"
    API_V1_STR: str = "/api"

    # DB & Redis
    DATABASE_URL: str = "sqlite:///./sifs_ie_ai.db"
    REDIS_URL: str = "redis://localhost:6379/0"

    # Auth
    JWT_SECRET: str = "supersecretjwtsecretchangeitinprod"
    CLERK_SECRET_KEY: str = ""
    CLERK_PUBLISHABLE_KEY: str = ""
    CLERK_JWKS_URL: str = ""  # Auto-derived from Clerk dashboard domain

    # CORS
    CORS_ORIGINS_STR: str = "http://localhost:3000"

    @property
    def CORS_ORIGINS(self) -> List[str]:
        return [
            origin.strip()
            for origin in self.CORS_ORIGINS_STR.split(",")
            if origin.strip()
        ]

    # ── LLM Providers ──────────────────────────────────────────────
    LLM_PROVIDER: str = "gemini"  # gemini | mock
    GEMINI_API_KEY: str = ""
    GEMINI_PRO_MODEL: str = "gemini-2.5-pro"
    GEMINI_FLASH_MODEL: str = "gemini-2.5-flash"

    # ── Image Providers ────────────────────────────────────────────
    IMAGE_PROVIDER: str = "gemini_image"  # gemini_image | mock
    GEMINI_IMAGE_MODEL: str = "imagen-3.0-generate-002"

    # ── Embedding Providers ────────────────────────────────────────
    EMBEDDING_PROVIDER: str = "gemini"  # gemini | mock
    EMBEDDING_MODEL: str = "models/text-embedding-004"

    # ── Storage (S3 / Cloudinary) ──────────────────────────────────
    S3_BUCKET: str = ""
    S3_ACCESS_KEY: str = ""
    S3_SECRET_KEY: str = ""
    S3_ENDPOINT_URL: str = ""

    # ── Stripe ─────────────────────────────────────────────────────
    STRIPE_SECRET_KEY: str = ""
    STRIPE_WEBHOOK_SECRET: str = ""
    STRIPE_PRICE_PREMIUM: str = ""  # Stripe Price ID for premium plan
    STRIPE_PRICE_BUSINESS: str = ""  # Stripe Price ID for business plan

    # ── Rate Limiting & Tier Caps ──────────────────────────────────
    RATE_LIMIT_RPM: int = 60  # requests per minute per user
    FREE_TIER_GENERATIONS: int = 3  # per month
    PREMIUM_TIER_GENERATIONS: int = 50
    BUSINESS_TIER_GENERATIONS: int = 500
    GEMINI_CONCURRENT_LIMIT: int = 5  # max parallel Concept Generator calls

    # ── Observability ──────────────────────────────────────────────
    SENTRY_DSN: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
