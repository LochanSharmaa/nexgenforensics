from datetime import datetime
from typing import Optional
from pydantic import BaseModel

class ProjectBase(BaseModel):
    title: str
    raw_imagination: str
    design_type: Optional[str] = None

class ProjectCreate(ProjectBase):
    user_id: Optional[int] = None

class ProjectUpdate(BaseModel):
    title: Optional[str] = None
    design_type: Optional[str] = None
    status: Optional[str] = None

class ProjectRead(ProjectBase):
    id: int
    user_id: Optional[int] = None
    status: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
