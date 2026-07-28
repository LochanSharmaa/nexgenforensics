"""Calibrate the match threshold against your own imagery.

The shipped defaults (match 0.42, review 0.32) are generic ArcFace operating
points. They are a starting position, not a validated setting: the false-match
rate at a fixed threshold rises with gallery size and degrades with image
quality, so a threshold that is safe for 500 subjects can be badly wrong for
50,000.

This script measures the genuine and impostor score distributions on a labelled
folder and reports the threshold for each target false-match rate.

Layout -- one directory per identity, at least two images each::

    dataset/
      person_0001/  a.jpg  b.jpg
      person_0002/  a.jpg  b.jpg  c.jpg

Usage::

    cd backend
    python scripts/calibrate_threshold.py ../src_extracted/AgeDB --max-identities 500

Requires the real engine (backend/requirements-engine.txt). With the stub loaded
it refuses to run, because calibrating against meaningless scores would produce a
confidently wrong threshold.
"""

from __future__ import annotations

import argparse
import itertools
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nexgen_engine.config import EngineConfig  # noqa: E402
from nexgen_engine.inference.pipeline import (  # noqa: E402
    FacialRecognitionPipeline,
    InvalidImageError,
    NoFaceDetectedError,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def collect_identities(root: Path, max_identities: int, max_per_identity: int) -> dict[str, list[Path]]:
    identities: dict[str, list[Path]] = {}
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        images = [
            path
            for path in sorted(directory.iterdir())
            if path.suffix.lower() in IMAGE_SUFFIXES
        ][:max_per_identity]
        if len(images) >= 2:
            identities[directory.name] = images
        if len(identities) >= max_identities:
            break
    return identities


def embed_all(
    pipeline: FacialRecognitionPipeline, identities: dict[str, list[Path]]
) -> dict[str, list[np.ndarray]]:
    embeddings: dict[str, list[np.ndarray]] = {}
    failures = 0
    total = sum(len(paths) for paths in identities.values())
    processed = 0

    for identity, paths in identities.items():
        vectors: list[np.ndarray] = []
        for path in paths:
            processed += 1
            if processed % 100 == 0:
                print(f"  encoded {processed}/{total}...", flush=True)
            try:
                vectors.append(pipeline.encode_bytes(path.read_bytes()).embedding)
            except (NoFaceDetectedError, InvalidImageError, OSError):
                failures += 1
        if len(vectors) >= 2:
            embeddings[identity] = vectors

    print(f"Encoded {processed - failures} images; {failures} could not be processed.")
    return embeddings


def score_pairs(
    embeddings: dict[str, list[np.ndarray]], impostor_pairs: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    genuine = [
        float(np.dot(left, right))
        for vectors in embeddings.values()
        for left, right in itertools.combinations(vectors, 2)
    ]

    rng = random.Random(seed)
    names = list(embeddings)
    impostor: list[float] = []
    if len(names) >= 2:
        for _ in range(impostor_pairs):
            first, second = rng.sample(names, 2)
            impostor.append(
                float(np.dot(rng.choice(embeddings[first]), rng.choice(embeddings[second])))
            )

    return np.asarray(genuine, dtype=np.float64), np.asarray(impostor, dtype=np.float64)


def report(genuine: np.ndarray, impostor: np.ndarray) -> None:
    if genuine.size == 0 or impostor.size == 0:
        print("Not enough pairs to calibrate. Need at least two identities with two images each.")
        return

    print("")
    print(f"Genuine pairs:  {genuine.size:>7}  mean {genuine.mean():.4f}  sd {genuine.std():.4f}")
    print(f"Impostor pairs: {impostor.size:>7}  mean {impostor.mean():.4f}  sd {impostor.std():.4f}")
    print("")
    print("  Target FMR   Threshold   TMR at that threshold   Missed genuine")
    print("  ----------   ---------   ---------------------   --------------")

    for target_fmr in (0.10, 0.01, 0.001, 0.0001):
        # The threshold that admits at most target_fmr of impostor pairs.
        threshold = float(np.quantile(impostor, 1.0 - target_fmr))
        tmr = float((genuine >= threshold).mean())
        print(
            f"  {target_fmr:>10.4%}   {threshold:>9.4f}   {tmr:>21.2%}   {1.0 - tmr:>14.2%}"
        )

    # Equal error rate: where false accepts and false rejects cross.
    candidates = np.linspace(min(impostor.min(), genuine.min()), max(impostor.max(), genuine.max()), 2000)
    far = np.asarray([(impostor >= value).mean() for value in candidates])
    frr = np.asarray([(genuine < value).mean() for value in candidates])
    eer_index = int(np.argmin(np.abs(far - frr)))

    print("")
    print(f"Equal error rate: {far[eer_index]:.2%} at threshold {candidates[eer_index]:.4f}")
    print("")
    print("Pick a threshold from the FMR that your use of the system can tolerate, not from the EER.")
    print("In an investigative context a false match sends someone to the wrong person, so a")
    print("stricter FMR is usually correct even though it misses more true matches.")
    print("")
    print("Apply it with NEXGEN_MATCH_THRESHOLD, and set NEXGEN_REVIEW_THRESHOLD roughly")
    print("0.08-0.12 lower so borderline scores reach an examiner rather than being dropped.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path, help="Root folder containing one directory per identity.")
    parser.add_argument("--max-identities", type=int, default=300)
    parser.add_argument("--max-per-identity", type=int, default=6)
    parser.add_argument("--impostor-pairs", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()

    if not args.dataset.is_dir():
        print(f"No such directory: {args.dataset}", file=sys.stderr)
        return 1

    pipeline = FacialRecognitionPipeline(EngineConfig(mode="auto"))
    if not pipeline.runtime.recognition_capable:
        print(
            "Refusing to calibrate: no recognition model is loaded, so every score would be "
            "meaningless.\nInstall it first:  pip install -r requirements-engine.txt",
            file=sys.stderr,
        )
        return 2

    print(f"Model: {pipeline.runtime.recognizer.info.model_pack} on {pipeline.runtime.recognizer.info.device}")
    identities = collect_identities(args.dataset, args.max_identities, args.max_per_identity)
    if not identities:
        print(
            f"No usable identities under {args.dataset}. Expected one subdirectory per person "
            "with at least two images each.",
            file=sys.stderr,
        )
        return 1

    print(f"Found {len(identities)} identities. Encoding...")
    embeddings = embed_all(pipeline, identities)
    genuine, impostor = score_pairs(embeddings, args.impostor_pairs, args.seed)
    report(genuine, impostor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
