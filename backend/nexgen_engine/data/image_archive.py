from __future__ import annotations

import csv
import json
import tarfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

from .loader import IMAGE_EXTENSIONS


@dataclass(frozen=True)
class ImageArchiveDatasetReport:
    source_path: str
    dataset_name: str
    archive_root: str
    image_records: int
    categories: int
    manifest_path: str
    dataset_task: str
    ready_for_catalog: bool
    notes: list[str]


class ImageArchiveCataloger:
    """Catalogs ordinary zip/tar archives containing images without extracting them."""

    def catalog(
        self,
        source_archive: str | Path,
        output_dir: str | Path,
        *,
        workspace: str = "default",
        dataset_task: str = "detection_validation",
        lawful_basis: str = "dataset_provider_terms",
        consent: bool = True,
        max_manifest_records: int | None = None,
    ) -> ImageArchiveDatasetReport:
        source = Path(source_archive)
        if not source.exists():
            raise FileNotFoundError(f"Dataset archive does not exist: {source}")
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)

        names = self._archive_names(source)
        image_names = [
            name
            for name in names
            if not name.endswith("/") and Path(name).suffix.lower() in IMAGE_EXTENSIONS
        ]
        if max_manifest_records is not None:
            image_names = image_names[:max_manifest_records]
        root = self._common_root(names)

        rows = [self._row(name, workspace, lawful_basis, consent, dataset_task) for name in image_names]
        categories = {row["category"] for row in rows}
        manifest_path = target / f"{source.stem}_image_archive_manifest.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = ["image_path", "identity_id", "workspace", "consent", "lawful_basis", "split", "category", "dataset_task"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        notes = [
            "Images remain inside the archive; extract or stream archive entries before pixel-level quality validation.",
            "This catalog is for detector/quality validation, not identity-recognition training labels.",
        ]
        report = ImageArchiveDatasetReport(
            source_path=str(source),
            dataset_name=source.stem,
            archive_root=root,
            image_records=len(rows),
            categories=len(categories),
            manifest_path=str(manifest_path),
            dataset_task=dataset_task,
            ready_for_catalog=bool(rows),
            notes=notes,
        )
        (target / f"{source.stem}_image_archive_report.json").write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")
        return report

    def catalog_zip(self, *args, **kwargs) -> ImageArchiveDatasetReport:
        return self.catalog(*args, **kwargs)

    @staticmethod
    def _archive_names(source: Path) -> list[str]:
        suffixes = "".join(source.suffixes).lower()
        if suffixes.endswith(".zip"):
            with zipfile.ZipFile(source) as archive:
                return archive.namelist()
        if suffixes.endswith(".tar") or suffixes.endswith(".tar.gz") or suffixes.endswith(".tgz"):
            with tarfile.open(source) as archive:
                return archive.getnames()
        raise ValueError(f"Unsupported image archive type: {source.suffix}")

    @staticmethod
    def _row(name: str, workspace: str, lawful_basis: str, consent: bool, dataset_task: str) -> dict[str, object]:
        parts = name.replace("\\", "/").split("/")
        category = "uncategorized"
        if "images" in parts:
            image_index = parts.index("images")
            if len(parts) > image_index + 1:
                category = parts[image_index + 1]
        elif len(parts) > 1:
            category = parts[-2]
        return {
            "image_path": f"archive://{name}",
            "identity_id": "",
            "workspace": workspace,
            "consent": str(consent).lower(),
            "lawful_basis": lawful_basis,
            "split": "val",
            "category": category,
            "dataset_task": dataset_task,
        }

    @staticmethod
    def _common_root(names: list[str]) -> str:
        first_parts = [name.split("/", 1)[0] for name in names if "/" in name]
        if not first_parts:
            return ""
        root = first_parts[0]
        return root if all(part == root for part in first_parts) else ""
