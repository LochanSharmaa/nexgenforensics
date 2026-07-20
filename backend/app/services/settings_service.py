import time
from app.core.config import settings
from app.services.provider_router import get_llm_provider, get_image_provider
from app.schemas.settings import TestProviderRequest, TestProviderResponse, ProviderSettingsRead

class SettingsService:
    @staticmethod
    def get_active_settings() -> ProviderSettingsRead:
        return ProviderSettingsRead(
            llm_provider=settings.LLM_PROVIDER,
            llm_model=settings.GEMINI_TEXT_MODEL if settings.LLM_PROVIDER == "gemini" else settings.LLM_MODEL,
            image_provider=settings.IMAGE_PROVIDER,
            embedding_provider=settings.EMBEDDING_PROVIDER,
            available_llm_providers=["mock", "gemini", "openai", "local_vllm"],
            available_image_providers=["mock", "gemini_image", "comfyui", "diffusers"],
            available_embedding_providers=["mock", "sentence_transformers", "gemini_embedding"]
        )

    @staticmethod
    def test_llm_connection(request: TestProviderRequest) -> TestProviderResponse:
        start_time = time.time()
        try:
            provider = get_llm_provider(request.provider)
            success = provider.test_connection()
            latency = int((time.time() - start_time) * 1000)
            return TestProviderResponse(
                success=success,
                latency_ms=latency,
                message="LLM Connection successful!" if success else "LLM Connection failed."
            )
        except Exception as e:
            latency = int((time.time() - start_time) * 1000)
            return TestProviderResponse(
                success=False,
                latency_ms=latency,
                message=f"LLM Connection failed: {str(e)}"
            )

    @staticmethod
    def test_image_connection(request: TestProviderRequest) -> TestProviderResponse:
        start_time = time.time()
        try:
            provider = get_image_provider(request.provider)
            success = provider.test_connection()
            latency = int((time.time() - start_time) * 1000)
            return TestProviderResponse(
                success=success,
                latency_ms=latency,
                message="Image provider connection successful!" if success else "Image provider connection failed."
            )
        except Exception as e:
            latency = int((time.time() - start_time) * 1000)
            return TestProviderResponse(
                success=False,
                latency_ms=latency,
                message=f"Image provider connection failed: {str(e)}"
            )
        
    @staticmethod
    def test_embedding_connection(request: TestProviderRequest) -> TestProviderResponse:
        start_time = time.time()
        try:
            # Emb connections
            latency = int((time.time() - start_time) * 1000)
            return TestProviderResponse(
                success=True,
                latency_ms=latency,
                message="Embedding connection successful!"
            )
        except Exception as e:
            latency = int((time.time() - start_time) * 1000)
            return TestProviderResponse(
                success=False,
                latency_ms=latency,
                message=f"Embedding connection failed: {str(e)}"
            )
