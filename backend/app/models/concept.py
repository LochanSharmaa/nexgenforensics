from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlmodel import SQLModel, Field, Column
from sqlalchemy import JSON as sa_JSON


class Concept(SQLModel, table=True):
    __tablename__ = "concepts"

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="projects.id", index=True)
    concept_number: int
    title: str
    lens_id: Optional[int] = Field(default=None, foreign_key="lenses.id")
    main_visual_idea: str
    composition: str
    lighting: str
    background: str
    color_palette: List[str] = Field(default_factory=list, sa_column=Column(sa_JSON))
    typography_direction: str
    creative_twist: str
    designer_execution_notes: str
    reference_image_prompt: str
    reference_only_notice: str = Field(
        default="Reference only \u2014 final artwork belongs to the designer."
    )

    # Reasoning & Metaphor fields
    metaphor_primary: str = Field(default="")
    metaphor_rejected: List[Dict[str, Any]] = Field(
        default_factory=list, sa_column=Column(sa_JSON)
    )  # [{metaphor: str, reason: str}]
    anti_pattern: str = Field(default="")
    camera_language: str = Field(default="")
    lighting_reasoning: str = Field(default="")
    material_reasoning: str = Field(default="")

    diversity_score: Optional[float] = None
    status: str = Field(default="draft")  # draft, saved, rejected

    created_at: datetime = Field(default_factory=datetime.utcnow)
