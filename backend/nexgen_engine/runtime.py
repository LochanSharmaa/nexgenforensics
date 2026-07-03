from __future__ import annotations

import importlib.util
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeCapabilities:
    torch: bool
    timm: bool
    insightface: bool
    faiss: bool
    cuda: bool

    @property
    def mode(self) -> str:
        if self.torch and self.timm and self.faiss:
            return "production_ready_dependencies"
        return "local_deterministic_fallback"


def detect_runtime_capabilities() -> RuntimeCapabilities:
    torch_available = importlib.util.find_spec("torch") is not None
    cuda_available = False
    if torch_available:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False
    return RuntimeCapabilities(
        torch=torch_available,
        timm=importlib.util.find_spec("timm") is not None,
        insightface=importlib.util.find_spec("insightface") is not None,
        faiss=importlib.util.find_spec("faiss") is not None or importlib.util.find_spec("faiss_cpu") is not None,
        cuda=cuda_available,
    )
