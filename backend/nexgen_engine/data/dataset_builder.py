from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .loader import FolderDatasetLoader, ImageRecord
from .quality_filter import ImageQualityFilter, QualityReport


@dataclass(frozen=True)
class PreparedRecord:
    source: ImageRecord
    quality: QualityReport


class DatasetBuilder:
    def __init__(self, quality_filter: ImageQualityFilter | None = None) -> None:
        self.quality_filter = quality_filter or ImageQualityFilter()

    def build_from_folder(self, root: str | Path, workspace: str = "default") -> list[PreparedRecord]:
        records = FolderDatasetLoader(root, workspace).records()
        prepared: list[PreparedRecord] = []
        for record in records:
            with Image.open(record.path) as image:
                quality = self.quality_filter.evaluate_image(image)
            if quality.accepted:
                prepared.append(PreparedRecord(record, quality))
        return prepared
