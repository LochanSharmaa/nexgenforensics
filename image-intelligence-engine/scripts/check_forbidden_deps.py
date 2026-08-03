"""Fail the build if a facial-recognition library enters the dependency graph.

ARCHITECTURE §14: the prohibition on facial recognition is enforced structurally
rather than promised. This script is the outermost ring — it inspects what is
*actually installed*, so a transitive pull-in through an innocuous-looking
package is caught even though no source file imports it.

Deliberately dependency-free and runnable standalone, so it works as a
pre-commit hook and in CI before the project is installed:

    python scripts/check_forbidden_deps.py

Exit codes: 0 clean, 1 forbidden package present, 2 could not inspect.
"""

from __future__ import annotations

import sys
from importlib import metadata

# Packages whose entire purpose is face embedding, face identification, or face
# gallery search. Detection-only libraries are not listed: bounded, descriptor-
# free detection for redaction is permitted (ARCHITECTURE §14.4).
FORBIDDEN_PACKAGES: dict[str, str] = {
    "insightface": "ArcFace embeddings and 1:N face search",
    "face-recognition": "dlib-based face identification",
    "face_recognition": "dlib-based face identification",
    "facenet": "face embedding model",
    "facenet-pytorch": "face embedding model",
    "deepface": "face verification and identification framework",
    "arcface": "face embedding loss/model",
    "dlib": "ships face recognition descriptors",
    "mtcnn": "bundled with face recognition pipelines",
    "retina-face": "face detection bundled for recognition pipelines",
    "keras-facenet": "face embedding model",
    "insightface-paddle": "ArcFace embeddings",
}

# Vector search engines. Not forbidden outright — flagged for review, because a
# face gallery needs one and their presence should always be a deliberate,
# explained choice rather than an accident.
REVIEW_PACKAGES: dict[str, str] = {
    "faiss": "ANN index — a face gallery needs one; confirm the intended use",
    "faiss-cpu": "ANN index — confirm the intended use",
    "faiss-gpu": "ANN index — confirm the intended use",
    "annoy": "ANN index — confirm the intended use",
    "hnswlib": "ANN index — confirm the intended use",
    "pgvector": "vector column type — the schema declares no embeddings",
}


def installed_distributions() -> dict[str, str]:
    found: dict[str, str] = {}
    for dist in metadata.distributions():
        name = (dist.metadata["Name"] or "").strip().lower()
        if name:
            found[name] = dist.version
    return found


def main() -> int:
    try:
        installed = installed_distributions()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: could not enumerate installed packages: {exc}", file=sys.stderr)
        return 2

    violations = [
        (name, installed[name], reason)
        for name, reason in FORBIDDEN_PACKAGES.items()
        if name in installed
    ]
    reviews = [
        (name, installed[name], reason)
        for name, reason in REVIEW_PACKAGES.items()
        if name in installed
    ]

    for name, version, reason in reviews:
        print(f"REVIEW: {name}=={version} — {reason}")

    if violations:
        print("", file=sys.stderr)
        print("FORBIDDEN DEPENDENCY DETECTED", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        for name, version, reason in violations:
            print(f"  {name}=={version}", file=sys.stderr)
            print(f"      {reason}", file=sys.stderr)
        print("", file=sys.stderr)
        print(
            "This platform must never identify people from facial features.\n"
            "See docs/ARCHITECTURE.md §1.1 and §14. If this package arrived\n"
            "transitively, pin the intermediate dependency or vendor the\n"
            "specific functionality you need instead.",
            file=sys.stderr,
        )
        return 1

    print(f"OK: no forbidden face-recognition packages ({len(installed)} distributions checked).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
