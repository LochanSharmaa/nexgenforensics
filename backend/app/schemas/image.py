from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

class ImageGenerateRequest(BaseModel):
    concept_id: int
    style: Optional[str] = None
    size: Optional[str] = "1024x1024"

class ImageGenerateBatchRequest(BaseModel):
    concept_ids: List[int]
    size: Optional[str] = "1024x1024"

class ImageResponse(BaseModel):
    image_url: str
    prompt: str
    provider: str
    reference_only: bool = Field(default=True)
    notice: str = Field(default="Reference only - final artwork belongs to the designer.")

class ImageJobRead(BaseModel):
    id: str
    project_id: int
    concept_id: int
    status: str
    provider: str
    image_url: Optional[str] = None
    error_message_sanitized: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
