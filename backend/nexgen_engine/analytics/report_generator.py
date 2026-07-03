from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .metrics import UsageMetrics


class AnalyticsReportGenerator:
    def write(self, metrics: UsageMetrics, output_path: str | Path) -> Path:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"usage": asdict(metrics), "review_rate": metrics.review_rate}, indent=2) + "\n", encoding="utf-8")
        return target
