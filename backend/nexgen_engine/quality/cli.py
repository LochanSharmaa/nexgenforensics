from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from .scorer import ProductionQualityScorer, QualityScorerConfig


def quality_check(image_path: str | Path, output_path: str | Path | None = None, faceqnet_checkpoint: str | None = None) -> dict:
    image = Image.open(image_path).convert("RGB")
    report = ProductionQualityScorer(
        QualityScorerConfig(faceqnet_checkpoint=faceqnet_checkpoint, require_faceqnet_checkpoint=bool(faceqnet_checkpoint))
    ).evaluate(image)
    payload = asdict(report)
    if output_path:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
