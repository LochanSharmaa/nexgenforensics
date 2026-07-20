from datetime import datetime
from typing import Optional, List
from sqlmodel import SQLModel, Field, Column
from sqlalchemy import JSON as sa_JSON

class StyleProfile(SQLModel, table=True):
    __tablename__ = "style_profiles"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(unique=True, index=True)
    liked_styles: List[str] = Field(default_factory=list, sa_column=Column(sa_JSON))
    disliked_styles: List[str] = Field(default_factory=list, sa_column=Column(sa_JSON))
    preferred_moods: List[str] = Field(default_factory=list, sa_column=Column(sa_JSON))
    preferred_colors: List[str] = Field(default_factory=list, sa_column=Column(sa_JSON))
    preferred_typography: List[str] = Field(default_factory=list, sa_column=Column(sa_JSON))
    updated_at: datetime = Field(default_factory=datetime.utcnow, sa_column_kwargs={"onupdate": datetime.utcnow})
