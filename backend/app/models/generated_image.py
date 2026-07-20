from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field

class GeneratedImage(SQLModel, table=True):
    __tablename__ = "generated_images"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="projects.id")
    concept_id: int = Field(foreign_key="concepts.id")
    image_url: str
    prompt: str
    provider: str
    reference_only: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
