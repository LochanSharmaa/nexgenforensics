from __future__ import annotations

import csv
import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RecordIODatasetReport:
    source_path: str
    dataset_name: str
    archive_root: str
    has_train_rec: bool
    has_train_idx: bool
    has_train_lst: bool
    benchmark_files: list[str]
    listed_records: int
    listed_identities: int
    manifest_path: str
    ready_for_catalog: bool
    notes: list[str]


class RecordIOArchiveCataloger:
    """Catalogs MXNet RecordIO face datasets without extracting multi-GB archives."""

    def catalog_zip(
        self,
        source_zip: str | Path,
        output_dir: str | Path,
        *,
        workspace: str = "default",
        lawful_basis: str = "dataset_provider_terms",
        consent: bool = False,
        max_manifest_records: int | None = None,
    ) -> RecordIODatasetReport:
        source = Path(source_zip)
        if not source.exists():
            raise FileNotFoundError(f"Dataset archive does not exist: {source}")
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(source) as archive:
            names = archive.namelist()
            root = self._common_root(names)
            train_rec = self._find(names, "train.rec")
            train_idx = self._find(names, "train.idx")
            train_lst = self._find(names, "train.lst")
            benchmark_files = sorted(name for name in names if name.lower().endswith(".bin"))
            rows, identities = self._read_lst(
                archive,
                train_lst,
                workspace=workspace,
                lawful_basis=lawful_basis,
                consent=consent,
                max_records=max_manifest_records,
            )

        manifest_path = target / f"{source.stem}_recordio_manifest.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = ["image_path", "identity_id", "workspace", "consent", "lawful_basis", "split", "record_index"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        notes = []
        if not consent:
            notes.append("Cataloged with consent=false; do not train until legal/consent status is confirmed.")
        if not train_lst:
            notes.append("train.lst was not found, so record identities could not be cataloged.")
        if train_rec:
            notes.append("Images remain inside train.rec; convert or stream RecordIO before pixel-level quality validation.")

        report = RecordIODatasetReport(
            source_path=str(source),
            dataset_name=source.stem,
            archive_root=root,
            has_train_rec=train_rec is not None,
            has_train_idx=train_idx is not None,
            has_train_lst=train_lst is not None,
            benchmark_files=[Path(name).name for name in benchmark_files],
            listed_records=len(rows),
            listed_identities=len(identities),
            manifest_path=str(manifest_path),
            ready_for_catalog=bool(train_rec and train_idx and train_lst and rows),
            notes=notes,
        )
        (target / f"{source.stem}_recordio_report.json").write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")
        return report

    def _read_lst(
        self,
        archive: zipfile.ZipFile,
        train_lst: str | None,
        *,
        workspace: str,
        lawful_basis: str,
        consent: bool,
        max_records: int | None,
    ) -> tuple[list[dict[str, object]], set[str]]:
        if not train_lst:
            return [], set()
        rows: list[dict[str, object]] = []
        identities: set[str] = set()
        with archive.open(train_lst) as handle:
            for line_number, raw_line in enumerate(handle):
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    parts = line.split()
                if len(parts) < 3:
                    continue
                record_index, identity_id, image_path = self._parse_lst_parts(parts, line_number)
                identities.add(identity_id)
                rows.append(
                    {
                        "image_path": f"recordio://train.rec/{record_index}/{image_path}",
                        "identity_id": identity_id,
                        "workspace": workspace,
                        "consent": str(consent).lower(),
                        "lawful_basis": lawful_basis,
                        "split": "train",
                        "record_index": record_index,
                    }
                )
                if max_records is not None and len(rows) >= max_records:
                    break
        return rows, identities

    @staticmethod
    def _parse_lst_parts(parts: list[str], line_number: int) -> tuple[str, str, str]:
        image_index = 2
        for index, value in enumerate(parts):
            suffix = Path(value.replace("\\", "/")).suffix.lower()
            if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"} or "/" in value or "\\" in value:
                image_index = index
                break
        image_path = parts[image_index]
        record_index = parts[0] if image_index == 2 else str(line_number)
        label_candidates = [value for index, value in enumerate(parts) if index != image_index]
        if "/" in image_path or "\\" in image_path:
            identity_id = Path(image_path.replace("\\", "/")).parent.name
        else:
            identity_id = label_candidates[-1] if label_candidates else Path(image_path).stem
        return record_index, identity_id, image_path

    @staticmethod
    def _find(names: list[str], filename: str) -> str | None:
        filename = filename.lower()
        for name in names:
            if Path(name).name.lower() == filename:
                return name
        return None

    @staticmethod
    def _common_root(names: list[str]) -> str:
        first_parts = [name.split("/", 1)[0] for name in names if "/" in name]
        if not first_parts:
            return ""
        root = first_parts[0]
        return root if all(part == root for part in first_parts) else ""
