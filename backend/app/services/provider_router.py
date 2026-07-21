"""
Provider routing — maps LLM_PROVIDER / IMAGE_PROVIDER / EMBEDDING_PROVIDER
settings to concrete provider classes.

Production default is gemini. Mock is for dev/CI only.
"""

from app.core.config import settings
from app.ai.base import LLMProvider
from app.ai.image_base import ImageProvider
from app.ai.embedding_base import EmbeddingProvider

from app.ai.mock_reasoning_provider import MockReasoningProvider
from app.ai.mock_image_provider import MockImageProvider
from app.ai.mock_embedding_provider import MockEmbeddingProvider

from app.ai.gemini_provider import GeminiProvider
from app.ai.gemini_image_provider import GeminiImageProvider
from app.ai.gemini_embedding_provider import GeminiEmbeddingProvider


def get_llm_provider(provider_name: str = None) -> LLMProvider:
    provider = (provider_name or settings.LLM_PROVIDER).lower()

    if provider == "gemini":
        return GeminiProvider()

    return MockReasoningProvider()


def get_image_provider(provider_name: str = None) -> ImageProvider:
    provider = (provider_name or settings.IMAGE_PROVIDER).lower()

    if provider == "gemini_image":
        return GeminiImageProvider()

    return MockImageProvider()


def get_embedding_provider(provider_name: str = None) -> EmbeddingProvider:
    provider = (provider_name or settings.EMBEDDING_PROVIDER).lower()

    if provider == "gemini":
        return GeminiEmbeddingProvider()

    return MockEmbeddingProvider()
