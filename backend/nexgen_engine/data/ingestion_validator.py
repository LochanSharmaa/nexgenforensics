from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .manifest import DatasetManifest, ManifestRecord
from .quality_filter import ImageQualityFilter


@dataclass(frozen=True)
class DatasetValidationIssue:
    image_path: str
    severity: str
    code: str
    message: str


@dataclass(frozen=True)
class DatasetValidationReport:
    total_records: int
    accepted_records: int
    rejected_records: int
    issues: list[DatasetValidationIssue]

    @property
    def ready_for_training(self) -> bool:
        return self.accepted_records > 0 and not any(issue.severity == "error" for issue in self.issues)


class DatasetIngestionValidator:
    def __init__(self, quality_filter: ImageQualityFilter | None = None) -> None:
        self.quality_filter = quality_filter or ImageQualityFilter()

    def validate(self, dataset_root: str | Path, manifest_path: str | Path) -> DatasetValidationReport:
        root = Path(dataset_root)
        manifest = DatasetManifest.load(manifest_path)
        issues: list[DatasetValidationIssue] = []
        accepted = 0
        for record in manifest.records:
            record_issues = self._validate_record(root, record)
            issues.extend(record_issues)
            if not record_issues:
                accepted += 1
        return DatasetValidationReport(
            total_records=len(manifest.records),
            accepted_records=accepted,
            rejected_records=len(manifest.records) - accepted,
            issues=issues,
        )

    def _validate_record(self, root: Path, record: ManifestRecord) -> list[DatasetValidationIssue]:
        issues: list[DatasetValidationIssue] = []
        if not record.consent:
            issues.append(self._issue(record, "error", "missing_consent", "Record is not consent-approved."))
        if not record.lawful_basis.strip():
            issues.append(self._issue(record, "error", "missing_lawful_basis", "Lawful basis is required."))
        if not record.identity_id.strip():
            issues.append(self._issue(record, "error", "missing_identity_id", "Identity ID is required."))
        path = (root / record.image_path).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError:
            issues.append(self._issue(record, "error", "path_outside_dataset", "Image path escapes dataset root."))
            return issues
        if not path.exists():
            issues.append(self._issue(record, "error", "image_not_found", "Image file does not exist."))
            return issues
        try:
            with Image.open(path) as image:
                quality = self.quality_filter.evaluate_image(image)
        except Exception as exc:
            issues.append(self._issue(record, "error", "image_unreadable", f"Image cannot be opened: {exc}"))
            return issues
        if not quality.accepted:
            issues.append(self._issue(record, "warning", "quality_rejected", "; ".join(quality.reasons) or "Quality below target."))
        return [issue for issue in issues if issue.severity == "error"]

    def _issue(self, record: ManifestRecord, severity: str, code: str, message: str) -> DatasetValidationIssue:
        return DatasetValidationIssue(record.image_path, severity, code, message)
