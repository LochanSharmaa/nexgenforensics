#!/usr/bin/env python
"""End-to-end demonstration of the NexGen iMATCH recognition engine.

Runs the real pipeline -- detection, landmark alignment, ArcFace embedding,
normalization, similarity search, ranking -- and prints what actually happened,
including per-stage timings. Nothing here is simulated: if the model cannot
load, it fails instead of printing a plausible-looking result.

Usage
-----
Enrol a gallery, then search a probe against it::

    python test_recognition.py --enrol path/to/gallery --probe path/to/probe.jpg

Gallery layout is either one directory per person::

    gallery/
      alice/  a1.jpg  a2.jpg
      bob/    b1.jpg

or flat AgeDB-style files named ``<index>_<Name>_<age>_<sex>.jpg``.

Other modes::

    python test_recognition.py --compare a.jpg b.jpg     # 1:1 comparison
    python test_recognition.py --self-test               # uses bundled AgeDB
    python test_recognition.py --status                  # model/device report
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nexgen_engine.config import EngineConfig  # noqa: E402
from nexgen_engine.inference.pipeline import (  # noqa: E402
    FacialRecognitionPipeline,
    InvalidImageError,
    NoFaceDetectedError,
)
from nexgen_engine.models.arcface import EngineUnavailableError  # noqa: E402
from nexgen_engine.search.gallery_index import GalleryIndex, faiss_available  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
TENANT = "demo"
REPO_ROOT = Path(__file__).resolve().parents[1]
AGEDB = REPO_ROOT / "src_extracted" / "AgeDB" / "AgeDB"

RULE = "=" * 68


def build_pipeline(device: str, pack: str) -> FacialRecognitionPipeline:
    pipeline = FacialRecognitionPipeline(EngineConfig(device=device, model_pack=pack))
    pipeline.runtime.warm_up()
    return pipeline


def print_status(pipeline: FacialRecognitionPipeline) -> None:
    status = pipeline.runtime.status()
    recognizer = status["recognizer"]
    device = status["device"]

    print(RULE)
    print("MODEL STATUS")
    print(RULE)
    print(f"  Model loaded          : YES")
    print(f"  Backend               : {recognizer['backend']}")
    print(f"  Model pack            : {recognizer['model_pack']}")
    print(f"  Recognition network   : {recognizer['recognition_network']}")
    print(f"  Embedding dimensions  : {recognizer['embedding_dim']}")
    print(f"  Detector              : {status['detector']['name']} (landmarks: "
          f"{status['detector']['produces_landmarks']})")
    print(f"  Device requested      : {device['requested']}")
    print(f"  Device in use         : {device['effective']}")
    print(f"  ONNX providers        : {', '.join(device['providers'])}")
    print(f"  Search backend        : {'FAISS IndexFlatIP' if faiss_available() else 'numpy matmul'} (exact)")
    print(f"  Model load time       : {status['model_load_seconds']}s")
    print(f"  Match / review thresh : {status['thresholds']['match']} / {status['thresholds']['review']}")
    print()


def collect_gallery(root: Path) -> dict[str, list[Path]]:
    """Group images by person, supporting both directory and AgeDB layouts."""
    grouped: dict[str, list[Path]] = defaultdict(list)

    directories = [p for p in sorted(root.iterdir()) if p.is_dir()]
    if directories:
        for directory in directories:
            for image in sorted(directory.iterdir()):
                if image.suffix.lower() in IMAGE_SUFFIXES:
                    grouped[directory.name].append(image)
        return dict(grouped)

    for image in sorted(root.iterdir()):
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        parts = image.stem.split("_")
        # AgeDB: <index>_<Name>_<age>_<sex>. Anything else keys on the stem.
        grouped[parts[1] if len(parts) >= 3 else image.stem].append(image)
    return dict(grouped)


def enrol(
    pipeline: FacialRecognitionPipeline,
    index: GalleryIndex,
    people: dict[str, list[Path]],
    max_per_person: int,
) -> tuple[int, int, float]:
    print(RULE)
    print("ENROLMENT")
    print(RULE)

    enrolled = failed = 0
    started = time.perf_counter()

    for person, paths in people.items():
        kept = 0
        for path in paths[:max_per_person]:
            try:
                result = pipeline.encode_bytes(path.read_bytes())
            except (NoFaceDetectedError, InvalidImageError) as exc:
                print(f"  SKIP  {person:<22} {path.name:<34} {exc.__class__.__name__}")
                failed += 1
                continue
            index.add(
                TENANT,
                template_id=f"{person}:{path.name}",
                subject_id=person,
                embedding=result.embedding,
                metadata={"person": person, "file": path.name, "quality": result.quality.score},
            )
            enrolled += 1
            kept += 1
        if kept:
            print(f"  OK    {person:<22} {kept} image(s)")

    elapsed = time.perf_counter() - started
    print(f"\n  Enrolled {enrolled} template(s) across {index.subject_count(TENANT)} subject(s); "
          f"{failed} image(s) unusable.")
    print(f"  Total {elapsed:.1f}s ({elapsed / max(enrolled, 1) * 1000:.0f} ms per image)\n")
    return enrolled, failed, elapsed


def search(pipeline: FacialRecognitionPipeline, index: GalleryIndex, probe_path: Path, top_k: int) -> None:
    print(RULE)
    print(f"SEARCH  {probe_path.name}")
    print(RULE)

    try:
        result = pipeline.encode_bytes(probe_path.read_bytes())
    except (NoFaceDetectedError, InvalidImageError) as exc:
        print(f"  FAILED: {exc}")
        return

    timings = result.timings
    print(f"  Faces detected        : {result.faces_detected}"
          f"{'  (found after padding)' if result.padded_detection else ''}")
    print(f"  Face box              : {result.face.box.left},{result.face.box.top} -> "
          f"{result.face.box.right},{result.face.box.bottom}")
    print(f"  Detection confidence  : {result.face.confidence:.3f}")
    print(f"  Head pose (y/p/r)     : {result.face.pose.yaw:.1f} / {result.face.pose.pitch:.1f} / "
          f"{result.face.pose.roll:.1f} deg")
    print(f"  Embedding generated   : {result.embedding.shape[0]} dimensions, "
          f"L2 norm {float(np.linalg.norm(result.embedding)):.4f}")
    print(f"  Image quality         : {result.quality.score:.3f} "
          f"({'accepted' if result.quality.accepted else 'BELOW GATE'})")
    if result.reasons:
        print(f"  Flags                 : {', '.join(result.reasons)}")

    started = time.perf_counter()
    outcome = index.search(TENANT, result.embedding, top_k=top_k)
    search_ms = (time.perf_counter() - started) * 1000

    print(f"\n  Gallery size          : {outcome.gallery_size} templates, "
          f"{index.subject_count(TENANT)} subjects")

    if not outcome.matches:
        print("\n  No candidates above the minimum similarity.")
    else:
        thresholds = pipeline.config.thresholds
        print(f"\n  {'#':<3} {'SUBJECT':<24} {'SIMILARITY':<12} DECISION")
        print(f"  {'-'*3} {'-'*24} {'-'*12} {'-'*22}")
        for rank, match in enumerate(outcome.matches, start=1):
            if match.score >= thresholds.match:
                verdict = "above match thresh"
            elif match.score >= thresholds.review:
                verdict = "review band"
            else:
                verdict = "below thresholds"
            print(f"  {rank:<3} {match.subject_id:<24} {match.score:<12.4f} {verdict}")
        print(f"\n  Margin over runner-up : {outcome.margin:.4f}")

    print(f"\n  TIMING  decode {timings.decode_ms:.0f} ms | detect {timings.detect_ms:.0f} ms | "
          f"align {timings.align_ms:.0f} ms | embed {timings.embed_ms:.0f} ms | "
          f"search {search_ms:.1f} ms")
    print(f"  Total pipeline        : {timings.total_ms:.0f} ms\n")
    print("  NOTE: similarity is not a probability that two images show the same")
    print("        person. Candidates are investigative leads requiring examiner review.\n")


def compare(pipeline: FacialRecognitionPipeline, left: Path, right: Path) -> int:
    print(RULE)
    print("1:1 COMPARISON")
    print(RULE)
    try:
        a = pipeline.encode_bytes(left.read_bytes())
        b = pipeline.encode_bytes(right.read_bytes())
    except (NoFaceDetectedError, InvalidImageError) as exc:
        print(f"  FAILED: {exc}")
        return 1

    similarity = float(np.dot(a.embedding, b.embedding))
    threshold = pipeline.config.thresholds.verify
    print(f"  {left.name}  quality {a.quality.score:.3f}")
    print(f"  {right.name}  quality {b.quality.score:.3f}")
    print(f"\n  Cosine similarity     : {similarity:.4f}")
    print(f"  Verify threshold      : {threshold:.2f}")
    print(f"  Above threshold       : {'YES' if similarity >= threshold else 'NO'}")
    print(f"  Combined time         : {a.timings.total_ms + b.timings.total_ms:.0f} ms\n")
    return 0


def self_test(pipeline: FacialRecognitionPipeline, index: GalleryIndex) -> int:
    """Enrol one image per person from AgeDB, then search with a held-out image."""
    if not AGEDB.is_dir():
        print(f"Self-test needs AgeDB at {AGEDB}. Use --enrol/--probe with your own images.")
        return 2

    people = {k: v for k, v in collect_gallery(AGEDB).items() if len(v) >= 2}
    chosen = dict(list(people.items())[:25])
    print(f"Self-test on {len(chosen)} AgeDB identities.\n")

    # Enrol the first image of each person; hold back the second as the probe.
    enrol(pipeline, index, {k: v[:1] for k, v in chosen.items()}, max_per_person=1)

    correct = 0
    attempted = 0
    genuine: list[float] = []
    impostor: list[float] = []

    for person, paths in chosen.items():
        try:
            probe = pipeline.encode_bytes(paths[1].read_bytes())
        except (NoFaceDetectedError, InvalidImageError):
            continue
        outcome = index.search(TENANT, probe.embedding, top_k=len(chosen))
        if not outcome.matches:
            continue
        attempted += 1
        if outcome.matches[0].subject_id == person:
            correct += 1
        for match in outcome.matches:
            (genuine if match.subject_id == person else impostor).append(match.score)

    print(RULE)
    print("SELF-TEST RESULT")
    print(RULE)
    if not attempted:
        print("  No probes could be processed.")
        return 1

    g = np.array(genuine)
    i = np.array(impostor)
    print(f"  Rank-1 identification : {correct}/{attempted} = {correct / attempted:.1%}")
    if g.size:
        print(f"  Genuine  pairs        : n={g.size:<5} mean={g.mean():.4f}  min={g.min():.4f}")
    if i.size:
        print(f"  Impostor pairs        : n={i.size:<5} mean={i.mean():.4f}  max={i.max():.4f}")
    if g.size and i.size:
        print(f"  Separation (mean gap) : {g.mean() - i.mean():.4f}")
        print(f"\n  {'THRESHOLD':<12}{'TAR':<10}{'FAR':<10}")
        for t in (0.20, 0.28, 0.36, 0.42, 0.50):
            print(f"  {t:<12.2f}{float((g >= t).mean()):<10.3f}{float((i >= t).mean()):<10.4f}")
        print("\n  These are measured on this dataset only, not an accuracy claim.")
    print()

    # A working engine must separate identities. Anything at or below chance
    # means the pipeline is broken, whatever the individual numbers look like.
    return 0 if g.size and i.size and g.mean() > i.mean() + 0.15 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("probe_positional", nargs="?", type=Path, help="Probe image (shorthand for --probe).")
    parser.add_argument("--enrol", "--enroll", dest="enrol", type=Path, help="Gallery folder to enrol.")
    parser.add_argument("--probe", type=Path, help="Probe image to search.")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("A", "B"), help="Compare two images 1:1.")
    parser.add_argument("--self-test", action="store_true", help="Run the bundled AgeDB self-test.")
    parser.add_argument("--status", action="store_true", help="Report model and device status only.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--pack", default="buffalo_l", help="InsightFace model pack.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-per-person", type=int, default=3)
    args = parser.parse_args()

    probe = args.probe or args.probe_positional

    try:
        pipeline = build_pipeline(args.device, args.pack)
    except EngineUnavailableError as exc:
        print("RECOGNITION ENGINE UNAVAILABLE\n")
        print(f"  {exc}\n")
        print("  No result is produced. The engine does not substitute placeholder")
        print("  embeddings when the model is missing.")
        return 2

    print_status(pipeline)
    if args.status:
        return 0

    if args.compare:
        return compare(pipeline, args.compare[0], args.compare[1])

    index = GalleryIndex(pipeline.config.embedding_dim)

    if args.self_test:
        return self_test(pipeline, index)

    if args.enrol:
        if not args.enrol.is_dir():
            print(f"Not a directory: {args.enrol}")
            return 2
        people = collect_gallery(args.enrol)
        if not people:
            print(f"No images found under {args.enrol}")
            return 2
        enrol(pipeline, index, people, args.max_per_person)

    if probe:
        if not probe.is_file():
            print(f"Not a file: {probe}")
            return 2
        search(pipeline, index, probe, args.top_k)
    elif not args.enrol:
        parser.print_help()
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
