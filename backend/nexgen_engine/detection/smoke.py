from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from .registry import DetectorConfig, DetectorRegistry


def detector_smoke(image_path: str | Path, backend: str, fallback_allowed: bool, output_path: str | Path) -> Path:
    image = Image.open(image_path).convert("RGB")
    detector = DetectorRegistry().create(DetectorConfig(backend=backend, allow_fallback=fallback_allowed))
    boxes = detector.detect(image)
    payload = {
        "backend": backend,
        "fallback_allowed": fallback_allowed,
        "detections": [asdict(box) for box in boxes],
        "count": len(boxes),
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target
