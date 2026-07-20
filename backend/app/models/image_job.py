from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field

class ImageJob(SQLModel, table=True):
    __tablename__ = "image_jobs"
    
    id: Optional[str] = Field(default=None, primary_key=True)  # Using string UUID or custom job ID
    project_id: int = Field(foreign_key="projects.id")
    concept_id: int = Field(foreign_key="concepts.id")
    status: str = Field(default="queued")  # queued, running, completed, failed
    provider: str
    image_url: Optional[str] = None
    error_message_sanitized: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
