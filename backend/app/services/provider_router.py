from app.core.config import settings
from app.ai.base import LLMProvider
from app.ai.image_base import ImageProvider
from app.ai.embedding_base import EmbeddingProvider

from app.ai.mock_reasoning_provider import MockReasoningProvider
from app.ai.mock_image_provider import MockImageProvider
from app.ai.mock_embedding_provider import MockEmbeddingProvider


# We import real providers dynamically or standardly.
# Let's import them only if they exist to avoid circular/missing import errors in Phase 1.
try:
    from app.ai.gemini_provider import GeminiProvider
    from app.ai.openai_compatible_provider import OpenAICompatibleLLMProvider
    from app.ai.local_llm_provider import LocalVLLMProvider
except ImportError:
    GeminiProvider = None
    OpenAICompatibleLLMProvider = None
    LocalVLLMProvider = None

try:
    from app.ai.gemini_image_provider import GeminiImageProvider
    from app.ai.comfyui_provider import ComfyUIProvider
    from app.ai.diffusers_placeholder_provider import DiffusersPlaceholderProvider
except ImportError:
    GeminiImageProvider = None
    ComfyUIProvider = None
    DiffusersPlaceholderProvider = None

def get_llm_provider(provider_name: str = None) -> LLMProvider:
    provider = provider_name or settings.LLM_PROVIDER
    provider = provider.lower()
    
    if provider == "gemini" and GeminiProvider is not None:
        return GeminiProvider()
    elif provider == "openai" and OpenAICompatibleLLMProvider is not None:
        return OpenAICompatibleLLMProvider()
    elif provider == "local_vllm" and LocalVLLMProvider is not None:
        return LocalVLLMProvider()
    
    return MockReasoningProvider()


def get_image_provider(provider_name: str = None) -> ImageProvider:
    provider = provider_name or settings.IMAGE_PROVIDER
    provider = provider.lower()
    
    if provider == "gemini_image" and GeminiImageProvider is not None:
        return GeminiImageProvider()
    elif provider == "comfyui" and ComfyUIProvider is not None:
        return ComfyUIProvider()
    elif provider == "diffusers" and DiffusersPlaceholderProvider is not None:
        return DiffusersPlaceholderProvider()
        
    return MockImageProvider()

def get_embedding_provider(provider_name: str = None) -> EmbeddingProvider:
    provider = provider_name or settings.EMBEDDING_PROVIDER
    provider = provider.lower()
    
    # Placeholders for future real embeddings
    return MockEmbeddingProvider()
