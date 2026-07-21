from __future__ import annotations

from ...config import EngineConfig
from ...runtime import detect_runtime_capabilities


def engine_status() -> dict[str, int | str]:
    config = EngineConfig()
    return {
        "status": "ok",
        "backbone_count": len(config.backbones),
        "embedding_dim": config.final_embedding_dim,
        "cross_client_search": "disabled_by_default",
        "runtime_mode": detect_runtime_capabilities().mode,
    }
