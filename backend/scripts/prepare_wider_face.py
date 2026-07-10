from __future__ import annotations

import argparse
import csv
import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image


IMAGE_SPLITS = {
    "train": "WIDER_train",
    "val": "WIDER_val",
    "test": "WIDER_test",
}


@dataclass
class WiderSplitStats:
    split: str
    images: int = 0
    annotated_images: int = 0
    faces: int = 0
    invalid_faces_skipped: int = 0
    missing_images: int = 0
    unreadable_images: int = 0


def safe_extract(zip_path: Path, target_dir: Path) -> None:
    target = target_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            destination = (target / member.filename).resolve()
            if target != destination and target not in destination.parents:
                raise ValueError(f"Unsafe zip entry blocked: {member.filename}")
        archive.extractall(target)


def parse_wider_annotations(annotation_path: Path) -> list[tuple[str, list[list[int]]]]:
    records: list[tuple[str, list[list[int]]]] = []
    lines = annotation_path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        image_rel = lines[index].strip()
        index += 1
        if not image_rel:
            continue
        face_count = int(lines[index].strip())
        index += 1
        boxes: list[list[int]] = []
        if face_count == 0 and index < len(lines) and looks_like_box_line(lines[index]):
            index += 1
        for _ in range(face_count):
            parts = [int(float(value)) for value in lines[index].split()]
            index += 1
            boxes.append(parts[:10])
        records.append((image_rel, boxes))
    return records


def looks_like_box_line(line: str) -> bool:
    parts = line.split()
    if len(parts) < 4:
        return False
    try:
        [float(value) for value in parts]
    except ValueError:
        return False
    return True


def image_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as image:
            return image.size
    except OSError:
        return None


def yolo_line(box: list[int], width: int, height: int) -> str | None:
    x, y, w, h = box[:4]
    invalid = box[7] if len(box) > 7 else 0
    if invalid or w <= 0 or h <= 0:
        return None

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(width, x + w)
    y2 = min(height, y + h)
    clipped_w = x2 - x1
    clipped_h = y2 - y1
    if clipped_w <= 0 or clipped_h <= 0:
        return None

    x_center = (x1 + clipped_w / 2) / width
    y_center = (y1 + clipped_h / 2) / height
    norm_w = clipped_w / width
    norm_h = clipped_h / height
    return f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"


def write_labeled_split(
    dataset_root: Path,
    split_root: Path,
    annotations: Path,
    split: str,
    manifest_rows: list[dict[str, str]],
) -> WiderSplitStats:
    stats = WiderSplitStats(split=split)
    labels_root = split_root / "labels"
    labels_root.mkdir(parents=True, exist_ok=True)

    for image_rel, boxes in parse_wider_annotations(annotations):
        image_path = split_root / "images" / image_rel
        label_path = labels_root / Path(image_rel).with_suffix(".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        stats.annotated_images += 1

        size = image_size(image_path)
        if not image_path.exists():
            stats.missing_images += 1
            label_path.write_text("", encoding="utf-8")
            continue
        if size is None:
            stats.unreadable_images += 1
            label_path.write_text("", encoding="utf-8")
            continue

        width, height = size
        lines: list[str] = []
        for box in boxes:
            line = yolo_line(box, width, height)
            if line is None:
                stats.invalid_faces_skipped += 1
                continue
            lines.append(line)
        stats.faces += len(lines)
        write_text_if_changed(label_path, "\n".join(lines) + ("\n" if lines else ""))
        manifest_rows.append(
            {
                "image_path": str(image_path.relative_to(dataset_root)).replace("\\", "/"),
                "label_path": str(label_path.relative_to(dataset_root)).replace("\\", "/"),
                "split": split,
                "faces": str(len(lines)),
                "dataset_task": "face_detection",
            }
        )

    stats.images = count_images(split_root / "images")
    return stats


def write_test_manifest(dataset_root: Path, split_root: Path, filelist: Path, manifest_rows: list[dict[str, str]]) -> WiderSplitStats:
    stats = WiderSplitStats(split="test")
    labels_root = split_root / "labels"
    labels_root.mkdir(parents=True, exist_ok=True)
    for image_rel in filelist.read_text(encoding="utf-8").splitlines():
        image_rel = image_rel.strip()
        if not image_rel:
            continue
        image_path = split_root / "images" / image_rel
        stats.annotated_images += 1
        if not image_path.exists():
            stats.missing_images += 1
            continue
        manifest_rows.append(
            {
                "image_path": str(image_path.relative_to(dataset_root)).replace("\\", "/"),
                "label_path": "",
                "split": "test",
                "faces": "",
                "dataset_task": "face_detection_test",
            }
        )
    stats.images = count_images(split_root / "images")
    return stats


def write_text_if_changed(path: Path, content: str) -> None:
    if path.exists():
        try:
            if path.read_text(encoding="utf-8") == content:
                return
        except OSError:
            pass
    path.write_text(content, encoding="utf-8")


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["image_path", "label_path", "split", "faces", "dataset_task"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_data_yaml(path: Path, dataset_root: Path) -> None:
    content = "\n".join(
        [
            f"path: {dataset_root.as_posix()}",
            "train: WIDER_train/images",
            "val: WIDER_val/images",
            "test: WIDER_test/images",
            "names:",
            "  0: face",
            "",
        ]
    )
    write_text_if_changed(path, content)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract WIDER Face and convert official annotations to YOLO labels.")
    parser.add_argument("--output-root", default="backend/runtime/datasets/wider_face")
    parser.add_argument("--split-zip", required=True)
    parser.add_argument("--train-zip", required=True)
    parser.add_argument("--val-zip", required=True)
    parser.add_argument("--test-zip", required=True)
    parser.add_argument("--split-dir", help="Use an already extracted wider_face_split directory for annotations.")
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    raw_root = output_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_extract:
        for zip_path in [args.split_zip, args.train_zip, args.val_zip, args.test_zip]:
            safe_extract(Path(zip_path), raw_root)

    split_dir = Path(args.split_dir).resolve() if args.split_dir else raw_root / "wider_face_split"
    dataset_root = raw_root
    manifest_rows: list[dict[str, str]] = []
    stats = [
        write_labeled_split(dataset_root, raw_root / IMAGE_SPLITS["train"], split_dir / "wider_face_train_bbx_gt.txt", "train", manifest_rows),
        write_labeled_split(dataset_root, raw_root / IMAGE_SPLITS["val"], split_dir / "wider_face_val_bbx_gt.txt", "val", manifest_rows),
        write_test_manifest(dataset_root, raw_root / IMAGE_SPLITS["test"], split_dir / "wider_face_test_filelist.txt", manifest_rows),
    ]

    prepared_root = output_root / "prepared"
    prepared_root.mkdir(parents=True, exist_ok=True)
    manifest_path = prepared_root / "wider_face_manifest.csv"
    data_yaml_path = prepared_root / "wider_face_yolo.yaml"
    report_path = prepared_root / "wider_face_report.json"
    write_manifest(manifest_path, manifest_rows)
    write_data_yaml(data_yaml_path, dataset_root)

    report = {
        "dataset_root": str(dataset_root),
        "manifest_path": str(manifest_path),
        "data_yaml_path": str(data_yaml_path),
        "splits": [asdict(item) for item in stats],
        "notes": [
            "Train and validation labels are YOLO detection labels with class 0 = face.",
            "WIDER test images do not include public bounding boxes, so the test split is listed without labels.",
            "This dataset is for face detection, not identity-recognition training.",
        ],
    }
    write_text_if_changed(report_path, json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
