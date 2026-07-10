from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from nexgen_engine.data.ingestion_validator import DatasetIngestionValidator
from nexgen_engine.data.image_archive import ImageArchiveCataloger
from nexgen_engine.data.manifest import DatasetManifest
from nexgen_engine.data.recordio import RecordIOArchiveCataloger


def main() -> int:
    parser = argparse.ArgumentParser(description="NexGen dataset preparation tools")
    sub = parser.add_subparsers(dest="command", required=True)
    template = sub.add_parser("template", help="Write a dataset manifest template")
    template.add_argument("path")
    validate = sub.add_parser("validate", help="Validate a dataset folder and manifest")
    validate.add_argument("--root", required=True)
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--report", default="dataset_validation_report.json")
    recordio = sub.add_parser("catalog-recordio-zip", help="Catalog an MXNet RecordIO zip without extracting it")
    recordio.add_argument("--source", required=True)
    recordio.add_argument("--output-dir", default="runtime/datasets")
    recordio.add_argument("--workspace", default="default")
    recordio.add_argument("--lawful-basis", default="dataset_provider_terms")
    recordio.add_argument("--consent", action="store_true")
    recordio.add_argument("--max-manifest-records", type=int)
    image_zip = sub.add_parser("catalog-image-zip", help="Catalog an image zip/tar without extracting it")
    image_zip.add_argument("--source", required=True)
    image_zip.add_argument("--output-dir", default="runtime/datasets")
    image_zip.add_argument("--workspace", default="default")
    image_zip.add_argument("--task", default="detection_validation")
    image_zip.add_argument("--lawful-basis", default="dataset_provider_terms")
    image_zip.add_argument("--consent", action="store_true")
    image_zip.add_argument("--max-manifest-records", type=int)
    args = parser.parse_args()

    if args.command == "template":
        path = DatasetManifest.write_template(args.path)
        print(path)
        return 0

    if args.command == "catalog-recordio-zip":
        report = RecordIOArchiveCataloger().catalog_zip(
            args.source,
            args.output_dir,
            workspace=args.workspace,
            lawful_basis=args.lawful_basis,
            consent=args.consent,
            max_manifest_records=args.max_manifest_records,
        )
        print(json.dumps(asdict(report), indent=2))
        return 0 if report.ready_for_catalog else 1

    if args.command == "catalog-image-zip":
        report = ImageArchiveCataloger().catalog(
            args.source,
            args.output_dir,
            workspace=args.workspace,
            dataset_task=args.task,
            lawful_basis=args.lawful_basis,
            consent=args.consent,
            max_manifest_records=args.max_manifest_records,
        )
        print(json.dumps(asdict(report), indent=2))
        return 0 if report.ready_for_catalog else 1

    report = DatasetIngestionValidator().validate(args.root, args.manifest)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(asdict(report), indent=2))
    return 0 if report.ready_for_training else 1


if __name__ == "__main__":
    raise SystemExit(main())
