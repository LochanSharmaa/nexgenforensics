import hashlib
from typing import List, Optional
from sqlmodel import Session, select
from app.models.concept import Concept
from app.models.embeddings import ConceptEmbedding, TrendCorpusEntry
from app.schemas.concept import DiversityCheckResponse, PairwiseWarning
from app.services.provider_router import get_embedding_provider

class DiversityService:
    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
        dot_product = sum(x * y for x, y in zip(v1, v2))
        norm_v1 = sum(x * x for x in v1) ** 0.5
        norm_v2 = sum(x * x for x in v2) ** 0.5
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return dot_product / (norm_v1 * norm_v2)

    @classmethod
    def calculate_pairwise_similarities(cls, concepts: List[Concept], session: Optional[Session] = None) -> DiversityCheckResponse:
        provider = get_embedding_provider()
        
        embeddings = []
        for c in concepts:
            # 1. Compile aesthetic description for the concept
            text = f"{c.title} {c.main_visual_idea} {c.composition} {c.lighting} {c.background} {' '.join(c.color_palette or [])}"
            text_hash = hashlib.md5(text.encode()).hexdigest()
            
            emb = None
            # 2. Check ConceptEmbedding cache in DB
            if session and c.id:
                stmt = select(ConceptEmbedding).where(
                    ConceptEmbedding.concept_id == c.id,
                    ConceptEmbedding.source_text_hash == text_hash
                )
                cached = session.exec(stmt).first()
                if cached:
                    emb = cached.embedding
            
            if emb is None:
                # Cache miss: get embedding from provider
                try:
                    emb = provider.get_embedding(text)
                except Exception:
                    # Fallback to local word hash
                    from app.ai.mock_embedding_provider import MockEmbeddingProvider
                    fallback = MockEmbeddingProvider()
                    emb = fallback.get_embedding(text)
                
                # Save to database cache
                if session and c.id:
                    # Clear any stale embedding for this concept
                    session.exec(select(ConceptEmbedding).where(ConceptEmbedding.concept_id == c.id)).delete()
                    session.commit()
                    
                    db_emb = ConceptEmbedding(
                        concept_id=c.id,
                        embedding=emb,
                        source_text_hash=text_hash
                    )
                    session.add(db_emb)
                    session.commit()
                    
            embeddings.append(emb)

        warnings = []
        similarity_sum = 0.0
        comparisons = 0

        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                sim = cls.cosine_similarity(embeddings[i], embeddings[j])
                similarity_sum += sim
                comparisons += 1

                # Flag high-similarity pairs above the threshold
                if sim > 0.82:
                    warnings.append(
                        PairwiseWarning(
                            concept_a=concepts[i].concept_number,
                            concept_b=concepts[j].concept_number,
                            similarity=round(sim, 2),
                            issue=f"Concepts {concepts[i].concept_number} and {concepts[j].concept_number} share high aesthetic similarity.",
                            regeneration_instruction=f"Regenerate concept {concepts[j].concept_number} with an alternative camera angle, composition metaphor, background environment, and typography."
                        )
                    )

        avg_similarity = similarity_sum / comparisons if comparisons > 0 else 0.0
        overall_score = round(1.0 - avg_similarity, 2)

        return DiversityCheckResponse(
            overall_diversity_score=max(0.0, min(overall_score, 1.0)),
            pairwise_warnings=warnings
        )

    @classmethod
    def calculate_novelty_score(cls, session: Session, concept: Concept) -> float:
        """Calculate concept novelty against trend corpus entries using embeddings."""
        # 1. Fetch concept embedding from cache
        text = f"{concept.title} {concept.main_visual_idea} {concept.composition} {concept.lighting} {concept.background} {' '.join(concept.color_palette or [])}"
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        stmt = select(ConceptEmbedding).where(ConceptEmbedding.concept_id == concept.id, ConceptEmbedding.source_text_hash == text_hash)
        cached = session.exec(stmt).first()
        if cached:
            emb = cached.embedding
        else:
            provider = get_embedding_provider()
            try:
                emb = provider.get_embedding(text)
            except Exception:
                from app.ai.mock_embedding_provider import MockEmbeddingProvider
                emb = MockEmbeddingProvider().get_embedding(text)
                
        # 2. Fetch trends
        trends = session.exec(select(TrendCorpusEntry)).all()
        if not trends:
            return 0.85  # Default novelty if trend corpus is empty
            
        max_similarity = 0.0
        for t in trends:
            sim = cls.cosine_similarity(emb, t.embedding)
            # Factor in decay weight if present
            decayed_sim = sim * getattr(t, "decay_weight", 1.0)
            max_similarity = max(max_similarity, decayed_sim)
            
        novelty = round(max(0.1, min(1.0 - max_similarity, 1.0)), 2)
        return novelty
