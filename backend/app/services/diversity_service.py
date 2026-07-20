from typing import List
from app.models.concept import Concept
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
    def calculate_pairwise_similarities(cls, concepts: List[Concept]) -> DiversityCheckResponse:
        provider = get_embedding_provider()
        
        texts = []
        for c in concepts:
            # Compile key aesthetic metadata to determine visual similarity
            text = f"{c.main_visual_idea} {c.composition} {c.lighting} {c.background} {' '.join(c.color_palette)}"
            texts.append(text)

        try:
            embeddings = provider.get_embeddings(texts)
        except Exception:
            # Robust local word-hash fallback
            from app.ai.mock_embedding_provider import MockEmbeddingProvider
            fallback = MockEmbeddingProvider()
            embeddings = fallback.get_embeddings(texts)

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
        # Diversity score is inversely proportional to similarity
        overall_score = round(1.0 - avg_similarity, 2)

        return DiversityCheckResponse(
            overall_diversity_score=max(0.0, min(overall_score, 1.0)),
            pairwise_warnings=warnings
        )
