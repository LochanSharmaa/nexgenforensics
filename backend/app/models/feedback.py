from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field

class Feedback(SQLModel, table=True):
    __tablename__ = "feedback"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[int] = Field(default=None, foreign_key="users.id")
    project_id: int = Field(foreign_key="projects.id")
    concept_id: int = Field(foreign_key="concepts.id")
    rating: int  # e.g. 1-5
    liked: bool = Field(default=True)
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
