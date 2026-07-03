from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .metrics import VerificationMetrics


class BenchmarkReportGenerator:
    def write(self, metrics: VerificationMetrics, path: str | Path, dataset_name: str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": dataset_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "metrics": asdict(metrics),
            "note": "Benchmark results must be independently validated before commercial accuracy claims.",
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target
