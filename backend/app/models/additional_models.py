from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field

class ProviderLog(SQLModel, table=True):
    __tablename__ = "provider_logs"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    provider: str
    task_type: str  # extract_brief, generate_questions, generate_concepts, etc.
    status: str  # success, failure
    latency_ms: int
    error_message_sanitized: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Export(SQLModel, table=True):
    __tablename__ = "exports"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="projects.id")
    format: str  # json, markdown
    file_url: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
