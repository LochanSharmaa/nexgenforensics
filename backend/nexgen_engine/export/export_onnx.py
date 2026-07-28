from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import EngineConfig


@dataclass(frozen=True)
class OnnxExportManifest:
    format: str
    opset: int
    embedding_dim: int
    model_pack: str
    input_shape: list[int]
    output_path: str


def export_onnx_manifest(output_dir: str | Path, config: EngineConfig | None = None, opset: int = 18) -> OnnxExportManifest:
    cfg = config or EngineConfig()
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest = OnnxExportManifest(
        format="onnx",
        opset=opset,
        embedding_dim=cfg.embedding_dim,
        model_pack=cfg.model_pack,
        input_shape=[1, 3, 112, 112],
        output_path=str(target_dir / "nexgen_facial_engine.onnx"),
    )
    (target_dir / "onnx_manifest.json").write_text(json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8")
    return manifest
