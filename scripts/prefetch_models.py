"""Download and trim the InsightFace model pack at BUILD time.

WHY THIS EXISTS
---------------
The Render free instance has 512 MB of RAM. Fetching the model pack inside the
running service put three things in that budget at once:

  1. the ~275 MB zip archive being downloaded,
  2. the ~331 MB of extracted .onnx weights,
  3. the ONNX Runtime sessions built from them, plus Python/FastAPI baseline.

The deploy died with `Out of memory (used over 512Mi)` immediately after the
download reported 281857/281857 KB. Running this at build time instead means
the serving process starts with the weights already on disk and never holds the
archive at all.

WHY IT DELETES FILES FROM THE PACK
----------------------------------
`runtime.py` passes ``allowed_modules=["detection", "recognition"]``, which
reads as "only two models are loaded". That is not what InsightFace does.
``FaceAnalysis.__init__`` globs every ``*.onnx`` in the pack directory and calls
``model_zoo.get_model()`` on each one — which constructs a real
``onnxruntime.InferenceSession``, loading the weights — and only *then* discards
the ones outside ``allowed_modules``.

So the unused 3D landmark model (``1k3d68.onnx``, ~143 MB) was being loaded into
memory and thrown away on every single startup. On a 512 MB box that transient
spike is the difference between booting and not.

Deleting the unused files is therefore a memory fix, not disk hygiene. The two
kept models are exactly the ones the engine calls:

    det_10g.onnx     SCRFD detection
    w600k_r50.onnx   ArcFace recognition — the DEPLOYED model, see BENCHMARKS.md

Nothing about the recognition result changes: the deleted models were never
consulted for a similarity score. If a future feature needs age/gender or dense
landmarks, remove the corresponding entry from DISCARD below and re-measure the
memory ceiling before deploying.

USAGE
-----
    python scripts/prefetch_models.py [--root ./.insightface] [--pack buffalo_l]

The layout produced is the one InsightFace expects, so the service finds it by
setting NEXGEN_MODEL_ROOT to the same --root value:

    <root>/models/<pack>/*.onnx
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

RELEASE_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/{pack}.zip"

# Loaded by the engine. Everything else in the pack is removed — see module docstring.
KEEP = {"det_10g.onnx", "w600k_r50.onnx"}


def _download(url: str, dest: Path) -> int:
    """Stream to disk in chunks so the archive never sits in memory."""
    total = 0
    with urllib.request.urlopen(url) as response, dest.open("wb") as handle:
        while chunk := response.read(1 << 20):
            handle.write(chunk)
            total += len(chunk)
    return total


def prefetch(root: Path, pack: str) -> Path:
    target = root / "models" / pack
    weights = sorted(target.glob("*.onnx")) if target.is_dir() else []

    if {p.name for p in weights} >= KEEP:
        print(f"[prefetch] {target} already populated; skipping download.")
        return target

    target.mkdir(parents=True, exist_ok=True)
    url = RELEASE_URL.format(pack=pack)

    # The archive goes to a temp file that is removed before the service ever
    # starts, so its bytes are never part of the runtime memory budget.
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / f"{pack}.zip"
        print(f"[prefetch] downloading {url}")
        size = _download(url, archive)
        print(f"[prefetch] downloaded {size / 1_048_576:.1f} MB")

        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                if member.is_dir():
                    continue
                # Flatten: some packs nest under a top-level directory. Also
                # guards against path traversal in the archive.
                name = Path(member.filename).name
                if not name:
                    continue
                with bundle.open(member) as src, (target / name).open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1 << 20)
        print(f"[prefetch] extracted to {target}")

    discarded = 0
    for path in sorted(target.glob("*.onnx")):
        if path.name in KEEP:
            continue
        freed = path.stat().st_size
        path.unlink()
        discarded += freed
        print(f"[prefetch] removed {path.name} ({freed / 1_048_576:.1f} MB) — not loaded by this engine")

    kept = sorted(p.name for p in target.glob("*.onnx"))
    missing = KEEP - set(kept)
    if missing:
        raise SystemExit(
            f"[prefetch] FAILED: required weights missing from {pack}: {sorted(missing)}.\n"
            f"           Present: {kept}\n"
            f"           The pack layout may have changed; check {url}"
        )

    print(f"[prefetch] kept {kept}; freed {discarded / 1_048_576:.1f} MB")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="./.insightface", help="model root (NEXGEN_MODEL_ROOT)")
    parser.add_argument("--pack", default="buffalo_l", help="model pack name (NEXGEN_MODEL_PACK)")
    args = parser.parse_args()

    target = prefetch(Path(args.root).expanduser().resolve(), args.pack)
    print(f"[prefetch] ready: {target}")
    print(f"[prefetch] set NEXGEN_MODEL_ROOT={Path(args.root).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
