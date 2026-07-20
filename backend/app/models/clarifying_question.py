from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field

class ClarifyingQuestion(SQLModel, table=True):
    __tablename__ = "clarifying_questions"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="projects.id")
    question: str
    answer: Optional[str] = None
    skipped: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)
