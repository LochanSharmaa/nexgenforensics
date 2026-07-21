from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel

class CreativeBriefBase(BaseModel):
    main_subject: str
    design_type: str
    target_audience: str
    mood: List[str]
    colors: List[str]
    fixed_elements: List[str]
    flexible_elements: List[str]
    avoid_elements: List[str]
    tensions: List[str]


class CreativeBriefCreate(CreativeBriefBase):
    project_id: int

class CreativeBriefRead(CreativeBriefBase):
    id: int
    project_id: int
    created_at: datetime

    class Config:
        from_attributes = True

class ClarifyingQuestionBase(BaseModel):
    question: str
    answer: Optional[str] = None
    skipped: bool = False

class ClarifyingQuestionCreate(ClarifyingQuestionBase):
    project_id: int

class ClarifyingQuestionAnswer(BaseModel):
    question_id: int
    answer: Optional[str] = None
    skipped: bool = False

class ClarifyingQuestionRead(ClarifyingQuestionBase):
    id: int
    project_id: int
    created_at: datetime

    class Config:
        from_attributes = True

class BriefExtractionRequest(BaseModel):
    project_id: int
    user_style_profile_id: Optional[int] = None
