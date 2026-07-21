from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ReasoningChainSchema(BaseModel):
    lens_applied: str
    tension_identified: str
    design_decision: str
    consequence: str
    risk_flag: str

    class Config:
        from_attributes = True


class ConceptScoreSchema(BaseModel):
    novelty_score: float
    feasibility_score: float
    feasibility_reason: Optional[str] = None
    lens_fidelity_score: float

    class Config:
        from_attributes = True


class CriticNoteSchema(BaseModel):
    agent_role: str
    note_text: str
    is_dissent: bool

    class Config:
        from_attributes = True


class MetaphorRejectedSchema(BaseModel):
    metaphor: str
    reason: str


class ConceptBase(BaseModel):
    concept_number: int
    title: str
    main_visual_idea: str
    composition: str
    lighting: str
    background: str
    color_palette: List[str]
    typography_direction: str
    creative_twist: str
    designer_execution_notes: str
    reference_image_prompt: str
    reference_only_notice: str = Field(
        default="Reference only \u2014 final artwork belongs to the designer."
    )

    # Reasoning & Metaphor
    metaphor_primary: str = ""
    metaphor_rejected: List[Dict[str, Any]] = []
    camera_language: str = ""
    lighting_reasoning: str = ""
    material_reasoning: str = ""
    anti_pattern: str = ""


class ConceptCreate(ConceptBase):
    project_id: int
    lens_id: Optional[int] = None
    status: str = "draft"


class ConceptRead(ConceptBase):
    id: int
    project_id: int
    lens_id: Optional[int] = None
    status: str
    diversity_score: Optional[float] = None
    created_at: datetime

    # Nested relations
    reasoning_chain: Optional[ReasoningChainSchema] = None
    scores: Optional[ConceptScoreSchema] = None
    critic_notes: List[CriticNoteSchema] = []

    class Config:
        from_attributes = True


class ConceptBoardResponse(BaseModel):
    project_title: str
    brief_summary: str
    concepts: List[ConceptRead]


class ConceptGenerateRequest(BaseModel):
    project_id: int
    creative_distance: Optional[int] = 3


class ConceptRegenerateRequest(BaseModel):
    project_id: int
    concept_id: int
    instruction: str = ""


class ConceptCombineRequest(BaseModel):
    project_id: int
    concept_a_id: int
    concept_b_id: int
    custom_instruction: Optional[str] = None


class PairwiseWarning(BaseModel):
    concept_a: int
    concept_b: int
    similarity: float
    issue: str
    regeneration_instruction: str


class DiversityCheckResponse(BaseModel):
    overall_diversity_score: float
    pairwise_warnings: List[PairwiseWarning]
