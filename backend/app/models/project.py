from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field

class Project(SQLModel, table=True):
    __tablename__ = "projects"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[int] = Field(default=None, foreign_key="users.id")
    title: str
    raw_imagination: str
    design_type: Optional[str] = None
    status: str = Field(default="pending")  # pending, brief_extracted, concepts_generated
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
