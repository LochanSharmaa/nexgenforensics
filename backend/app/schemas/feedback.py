from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

class FeedbackCreate(BaseModel):
    project_id: int
    concept_id: int
    rating: int = Field(..., ge=1, le=5)
    liked: bool = True
    notes: Optional[str] = None

class FeedbackRead(FeedbackCreate):
    id: int
    user_id: Optional[int] = None
    created_at: datetime

    class Config:
        from_attributes = True
