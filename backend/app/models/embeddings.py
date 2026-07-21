from datetime import datetime
from typing import Optional, List
from sqlmodel import SQLModel, Field, Column
from sqlalchemy import JSON as sa_JSON

try:
    from pgvector.sqlalchemy import Vector
    HAS_PGVECTOR = True
except ImportError:
    HAS_PGVECTOR = False

class TrendCorpusEntry(SQLModel, table=True):
    __tablename__ = "trend_corpus_entries"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    # 768-dimension embedding
    embedding: List[float] = Field(
        sa_column=Column(Vector(768)) if HAS_PGVECTOR else Column(sa_JSON)
    )
    source: str
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    decay_weight: float = Field(default=1.0)

class ConceptEmbedding(SQLModel, table=True):
    __tablename__ = "concept_embeddings"
    
    concept_id: int = Field(primary_key=True, foreign_key="concepts.id")
    embedding: List[float] = Field(
        sa_column=Column(Vector(768)) if HAS_PGVECTOR else Column(sa_JSON)
    )
    source_text_hash: str = Field(index=True)
