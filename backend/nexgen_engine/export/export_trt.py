from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import EngineConfig


@dataclass(frozen=True)
class TensorRTExportManifest:
    format: str
    precision: str
    int8_calibrated: bool
    embedding_dim: int
    output_path: str


def export_trt_manifest(output_dir: str | Path, config: EngineConfig | None = None, precision: str = "fp16") -> TensorRTExportManifest:
    cfg = config or EngineConfig()
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest = TensorRTExportManifest(
        format="tensorrt",
        precision=precision,
        int8_calibrated=precision.lower() == "int8",
        embedding_dim=cfg.final_embedding_dim,
        output_path=str(target_dir / "nexgen_facial_engine.plan"),
    )
    (target_dir / "tensorrt_manifest.json").write_text(json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8")
    return manifest
