import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator

class Settings(BaseSettings):
    # Base Config
    PROJECT_NAME: str = "SIFS Imagination Expander AI"
    API_V1_STR: str = "/api"
    
    # DB & Redis
    DATABASE_URL: str = "sqlite:///./sifs_ie_ai.db"
    REDIS_URL: str = "redis://localhost:6379/0"
    JWT_SECRET: str = "supersecretjwtsecretchangeitinprod"
    
    # CORS
    CORS_ORIGINS_STR: str = "http://localhost:3000"

    @property
    def CORS_ORIGINS(self) -> List[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS_STR.split(",") if origin.strip()]

    # LLM Providers Configuration
    LLM_PROVIDER: str = "mock"  # mock | gemini | openai | local_vllm
    GEMINI_API_KEY: str = ""
    GEMINI_TEXT_MODEL: str = "gemini-3.5-flash"
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    LOCAL_LLM_BASE_URL: str = "http://localhost:8000/v1"
    LLM_MODEL: str = "gpt-4o-mini"

    # Image Providers Configuration
    IMAGE_PROVIDER: str = "mock"  # mock | gemini_image | comfyui | diffusers
    GEMINI_IMAGE_MODEL: str = "gemini-3.1-flash-image"
    COMFYUI_URL: str = "http://localhost:8188"

    # Storage (Optional)
    S3_BUCKET: str = ""
    S3_ACCESS_KEY: str = ""
    S3_SECRET_KEY: str = ""

    # Embedding Providers Configuration
    EMBEDDING_PROVIDER: str = "mock"  # mock | sentence_transformers | gemini_embedding
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
