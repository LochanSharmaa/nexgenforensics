from datetime import datetime
from typing import Optional, List
from sqlmodel import SQLModel, Field, Column
from sqlalchemy import JSON as sa_JSON

class Concept(SQLModel, table=True):
    __tablename__ = "concepts"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="projects.id")
    concept_number: int
    title: str
    style_category: str
    main_visual_idea: str
    composition: str
    lighting: str
    background: str
    color_palette: List[str] = Field(default_factory=list, sa_column=Column(sa_JSON))
    typography_direction: str
    creative_twist: str
    designer_execution_notes: str
    reference_image_prompt: str
    reference_only_notice: str = Field(default="Reference only - final artwork belongs to the designer.")
    
    # V2 Creative Reasoning Additions
    metaphor: str = Field(default="")
    rejected_metaphor_1: str = Field(default="")
    rejected_metaphor_2: str = Field(default="")
    camera_language: str = Field(default="")
    lighting_reasoning: str = Field(default="")
    material_reasoning: str = Field(default="")
    anti_pattern: str = Field(default="")

    diversity_score: Optional[float] = None
    status: str = Field(default="active")  # active, saved, rejected

    created_at: datetime = Field(default_factory=datetime.utcnow)
