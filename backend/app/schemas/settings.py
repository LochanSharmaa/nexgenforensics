from typing import List, Optional
from pydantic import BaseModel

class ProviderSettingsRead(BaseModel):
    llm_provider: str
    llm_model: str
    image_provider: str
    embedding_provider: str
    available_llm_providers: List[str]
    available_image_providers: List[str]
    available_embedding_providers: List[str]

class TestProviderRequest(BaseModel):
    provider: str
    model: Optional[str] = None
    api_key: Optional[str] = None

class TestProviderResponse(BaseModel):
    success: bool
    latency_ms: int
    message: str
