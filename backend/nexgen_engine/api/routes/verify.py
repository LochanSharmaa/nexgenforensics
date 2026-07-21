from __future__ import annotations

import numpy as np

from ...utils import cosine_similarity


def verify_embeddings(reference: np.ndarray, probe: np.ndarray, threshold: float = 0.82) -> dict[str, float | bool]:
    score = cosine_similarity(reference, probe)
    return {"score": round(score, 6), "verified": score >= threshold}
