from typing import List
from app.ai.embedding_base import EmbeddingProvider

class MockEmbeddingProvider(EmbeddingProvider):
    def get_embedding(self, text: str) -> List[float]:
        # Create a deterministic 384-dimension vector from string token hashes
        vector = [0.0] * 384
        words = text.lower().split()
        if not words:
            return vector
            
        for word in words:
            # Basic deterministic string hash
            h = 0
            for char in word:
                h = (h * 31 + ord(char)) & 0xFFFFFFFF
            idx = h % 384
            vector[idx] += 1.0
            
        # Normalize the vector to unit length
        magnitude = sum(x * x for x in vector) ** 0.5
        if magnitude > 0:
            vector = [x / magnitude for x in vector]
            
        return vector

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self.get_embedding(text) for text in texts]
