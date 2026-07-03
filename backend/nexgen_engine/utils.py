from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable

import numpy as np


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    if not math.isfinite(value):
        return minimum
    return max(minimum, min(maximum, value))


def l2_normalize(vector: np.ndarray, axis: int | None = None, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=axis, keepdims=True)
    return vector / np.maximum(norm, eps)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_n = l2_normalize(np.asarray(left, dtype=np.float32))
    right_n = l2_normalize(np.asarray(right, dtype=np.float32))
    return float(np.dot(left_n, right_n))


def deterministic_vector(seed: bytes | str, dimensions: int) -> np.ndarray:
    seed_bytes = seed.encode("utf-8") if isinstance(seed, str) else seed
    values: list[float] = []
    counter = 0
    while len(values) < dimensions:
        digest = hashlib.sha256(seed_bytes + counter.to_bytes(4, "big")).digest()
        values.extend((byte / 127.5) - 1.0 for byte in digest)
        counter += 1
    return l2_normalize(np.asarray(values[:dimensions], dtype=np.float32))


def weighted_average(vectors: Iterable[np.ndarray], weights: Iterable[float]) -> np.ndarray:
    vector_list = [np.asarray(vector, dtype=np.float32) for vector in vectors]
    weight_array = np.asarray(list(weights), dtype=np.float32)
    if not vector_list:
        raise ValueError("At least one vector is required.")
    if len(vector_list) != len(weight_array):
        raise ValueError("Vector and weight counts must match.")
    weight_array = weight_array / max(float(weight_array.sum()), 1e-12)
    stacked = np.stack(vector_list, axis=0)
    return l2_normalize((stacked * weight_array[:, None]).sum(axis=0))


def stable_id(prefix: str, payload: bytes | str, length: int = 16) -> str:
    data = payload.encode("utf-8") if isinstance(payload, str) else payload
    return f"{prefix}_{hashlib.sha256(data).hexdigest()[:length]}"
