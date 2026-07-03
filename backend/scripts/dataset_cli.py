from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from nexgen_engine.data.ingestion_validator import DatasetIngestionValidator
from nexgen_engine.data.manifest import DatasetManifest


def main() -> int:
    parser = argparse.ArgumentParser(description="NexGen dataset preparation tools")
    sub = parser.add_subparsers(dest="command", required=True)
    template = sub.add_parser("template", help="Write a dataset manifest template")
    template.add_argument("path")
    validate = sub.add_parser("validate", help="Validate a dataset folder and manifest")
    validate.add_argument("--root", required=True)
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--report", default="dataset_validation_report.json")
    args = parser.parse_args()

    if args.command == "template":
        path = DatasetManifest.write_template(args.path)
        print(path)
        return 0

    report = DatasetIngestionValidator().validate(args.root, args.manifest)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(asdict(report), indent=2))
    return 0 if report.ready_for_training else 1


if __name__ == "__main__":
    raise SystemExit(main())
