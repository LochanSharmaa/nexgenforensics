from datetime import datetime
from typing import Optional, List
from sqlmodel import SQLModel, Field, Column
from sqlalchemy import JSON as sa_JSON

class Lens(SQLModel, table=True):
    __tablename__ = "lenses"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    description: str
    reasoning_move: str
    risk_level: int  # 1 to 5
    active: bool = Field(default=True)

class ConceptLensMap(SQLModel, table=True):
    __tablename__ = "concept_lens_map"
    
    concept_id: int = Field(primary_key=True, foreign_key="concepts.id")
    lens_id: int = Field(primary_key=True, foreign_key="lenses.id")

class ReasoningChain(SQLModel, table=True):
    __tablename__ = "reasoning_chains"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    concept_id: int = Field(foreign_key="concepts.id")
    brief_link: Optional[str] = None
    lens_applied: str
    tension_identified: str
    design_decision: str
    consequence: str
    risk_flag: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ConceptScore(SQLModel, table=True):
    __tablename__ = "concept_scores"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    concept_id: int = Field(foreign_key="concepts.id")
    novelty_score: float
    feasibility_score: float
    feasibility_reason: Optional[str] = None
    lens_fidelity_score: float
    scored_by_provider: str = Field(default="mock")
    created_at: datetime = Field(default_factory=datetime.utcnow)

class CriticNote(SQLModel, table=True):
    __tablename__ = "critic_notes"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    concept_id: int = Field(foreign_key="concepts.id")
    agent_role: str
    note_text: str
    is_dissent: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ConceptGenealogy(SQLModel, table=True):
    __tablename__ = "concept_genealogy"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    parent_concept_id: int
    child_concept_id: int
    mutation_instruction: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class HardConstraint(SQLModel, table=True):
    __tablename__ = "hard_constraints"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="projects.id")
    constraint_text: str
    applies_to: str  # logo, color, tagline, typography, general
    created_at: datetime = Field(default_factory=datetime.utcnow)

class TrendCorpusEntry(SQLModel, table=True):
    __tablename__ = "trend_corpus_entries"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    embedding: List[float] = Field(sa_column=Column(sa_JSON))
    source: str
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    decay_weight: float = Field(default=1.0)

class MissingOpportunityReport(SQLModel, table=True):
    __tablename__ = "missing_opportunity_reports"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="projects.id")
    report_text: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
