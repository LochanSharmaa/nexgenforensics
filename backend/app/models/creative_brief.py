from datetime import datetime
from typing import Optional, List
from sqlmodel import SQLModel, Field, Column
from sqlalchemy import JSON as sa_JSON

class CreativeBrief(SQLModel, table=True):
    __tablename__ = "creative_briefs"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="projects.id")
    main_subject: str
    design_type: str
    target_audience: str
    mood: List[str] = Field(default_factory=list, sa_column=Column(sa_JSON))
    colors: List[str] = Field(default_factory=list, sa_column=Column(sa_JSON))
    fixed_elements: List[str] = Field(default_factory=list, sa_column=Column(sa_JSON))
    flexible_elements: List[str] = Field(default_factory=list, sa_column=Column(sa_JSON))
    avoid_elements: List[str] = Field(default_factory=list, sa_column=Column(sa_JSON))
    tensions: List[str] = Field(default_factory=list, sa_column=Column(sa_JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)

