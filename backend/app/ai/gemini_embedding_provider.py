import os
import logging
from typing import List
import google.generativeai as genai
from app.core.config import settings
from app.ai.embedding_base import EmbeddingProvider

logger = logging.getLogger(__name__)

class GeminiEmbeddingProvider(EmbeddingProvider):
    def __init__(self):
        api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
        if api_key:
            genai.configure(api_key=api_key)
        self.model = "models/text-embedding-004"

    def get_embedding(self, text: str) -> List[float]:
        try:
            logger.info(f"Generating Gemini embedding for text length: {len(text)}")
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document"
            )
            return result["embedding"]
        except Exception as e:
            logger.error(f"Gemini embedding error: {str(e)}")
            # Fallback to a zero vector if API call fails
            return [0.0] * 768

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            logger.info(f"Generating batch Gemini embeddings for {len(texts)} texts")
            result = genai.embed_content(
                model=self.model,
                content=texts,
                task_type="retrieval_document"
            )
            return result["embedding"]
        except Exception as e:
            logger.error(f"Gemini batch embedding error: {str(e)}")
            return [self.get_embedding(text) for text in texts]
