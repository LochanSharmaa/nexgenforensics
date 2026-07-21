from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    identity_id: str
    workspace: str


class FolderDatasetLoader:
    def __init__(self, root: str | Path, workspace: str = "default") -> None:
        self.root = Path(root)
        self.workspace = workspace

    def records(self) -> list[ImageRecord]:
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")
        records: list[ImageRecord] = []
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                identity_id = path.parent.name if path.parent != self.root else path.stem
                records.append(ImageRecord(path=path, identity_id=identity_id, workspace=self.workspace))
        return records
