from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManifestRecord:
    image_path: str
    identity_id: str
    workspace: str
    consent: bool
    lawful_basis: str
    split: str = "train"


class DatasetManifest:
    required_columns = {"image_path", "identity_id", "workspace", "consent", "lawful_basis"}

    def __init__(self, records: list[ManifestRecord]) -> None:
        self.records = records

    @classmethod
    def load(cls, path: str | Path) -> "DatasetManifest":
        source = Path(path)
        if source.suffix.lower() == ".json":
            payload = json.loads(source.read_text(encoding="utf-8"))
            rows = payload.get("records", payload if isinstance(payload, list) else [])
        elif source.suffix.lower() == ".csv":
            with source.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        else:
            raise ValueError("Dataset manifest must be .json or .csv.")
        return cls([cls._record_from_row(row) for row in rows])

    @classmethod
    def write_template(cls, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix.lower() == ".csv":
            with target.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["image_path", "identity_id", "workspace", "consent", "lawful_basis", "split"])
                writer.writeheader()
                writer.writerow(
                    {
                        "image_path": "person_001/img_001.jpg",
                        "identity_id": "person_001",
                        "workspace": "tenant_a",
                        "consent": "true",
                        "lawful_basis": "consent",
                        "split": "train",
                    }
                )
        else:
            payload = {
                "records": [
                    {
                        "image_path": "person_001/img_001.jpg",
                        "identity_id": "person_001",
                        "workspace": "tenant_a",
                        "consent": True,
                        "lawful_basis": "consent",
                        "split": "train",
                    }
                ]
            }
            target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return target

    @staticmethod
    def _record_from_row(row: dict) -> ManifestRecord:
        missing = DatasetManifest.required_columns - set(row)
        if missing:
            raise ValueError(f"Dataset manifest missing columns: {', '.join(sorted(missing))}")
        return ManifestRecord(
            image_path=str(row["image_path"]),
            identity_id=str(row["identity_id"]),
            workspace=str(row["workspace"]),
            consent=str(row["consent"]).strip().lower() in {"1", "true", "yes", "y"},
            lawful_basis=str(row["lawful_basis"]),
            split=str(row.get("split") or "train"),
        )
