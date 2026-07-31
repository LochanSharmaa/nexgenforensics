#!/usr/bin/env python
"""
Supported installer for the CUDA build of the NexGen iMATCH engine.

    python scripts/setup_gpu.py            # install + verify
    python scripts/setup_gpu.py --check    # verify only, install nothing

WHY A SCRIPT AND NOT JUST `pip install -r`
-------------------------------------------
`insightface` declares a hard dependency on **`onnxruntime`** (the CPU build).
pip has no concept that `onnxruntime-gpu` provides the same import package, so
a plain `pip install -r requirements-gpu.txt` installs BOTH distributions into
the same `onnxruntime/` namespace.

Whichever one pip unpacks last wins. That ordering is not guaranteed --
it varies with resolver version, cache state, and wheel arrival order. When the
CPU build wins, every model silently runs on CPU: no error, no crash, roughly
20x slower. This is the same class of failure that cost this project its GPU
configuration twice.

There is no way to express "onnxruntime-gpu satisfies onnxruntime" in a
requirements file, so the deconflict step lives here, in version control,
rather than in somebody's shell history.

Steps:
  1. pip install the three requirements files
  2. remove the CPU `onnxruntime` distribution if `onnxruntime-gpu` is present
  3. run scripts/verify_gpu.py and propagate its exit code
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_REQS = [
    _ROOT / "backend" / "requirements.txt",
    _ROOT / "backend" / "requirements-engine.txt",
    _ROOT / "backend" / "requirements-gpu.txt",
]


def run(*args: str) -> int:
    print(f"\n$ {' '.join(args)}", flush=True)
    return subprocess.call(args)


def installed() -> set[str]:
    from importlib.metadata import distributions

    return {(d.metadata["Name"] or "").lower() for d in distributions()}


def gpu_pin() -> str:
    """Read the onnxruntime-gpu pin from requirements-gpu.txt.

    Parsed rather than hardcoded so the pin has exactly one source of truth.
    """
    for line in (_ROOT / "backend" / "requirements-gpu.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if line.lower().startswith("onnxruntime-gpu"):
            return line
    raise RuntimeError("no onnxruntime-gpu pin found in backend/requirements-gpu.txt")


def deconflict() -> bool:
    """Leave exactly one onnxruntime distribution installed: the GPU build.

    IMPORTANT: this uninstalls BOTH distributions and then reinstalls the GPU
    one. Uninstalling only the CPU build does NOT work -- both distributions
    unpack into the same `onnxruntime/` directory, so pip's uninstaller deletes
    shared files that the surviving GPU distribution also owns. The result is a
    half-gutted package whose metadata still says onnxruntime-gpu but which
    raises `module 'onnxruntime' has no attribute 'InferenceSession'` on import.
    (Observed on a clean rebuild -- this is not hypothetical.)
    """
    dists = installed()
    has_cpu = "onnxruntime" in dists
    has_gpu = "onnxruntime-gpu" in dists

    if not has_cpu:
        if not has_gpu:
            print("!! onnxruntime-gpu is not installed; this host cannot use CUDA.")
            return False
        print("ok: onnxruntime-gpu is the only onnxruntime distribution")
        return True

    print(
        "\n!! Both `onnxruntime` and `onnxruntime-gpu` are installed.\n"
        "   insightface pulls in the CPU build transitively, and the two share\n"
        "   one import namespace. Removing both, then reinstalling the GPU build.",
        flush=True,
    )
    if run(sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime", "onnxruntime-gpu") != 0:
        return False
    # --no-deps: onnxruntime-gpu's own metadata is fine, but resolving deps here
    # can drag the CPU build back in through insightface.
    if run(sys.executable, "-m", "pip", "install", "--no-deps", gpu_pin()) != 0:
        return False

    dists = installed()
    ok = "onnxruntime" not in dists and "onnxruntime-gpu" in dists
    print("ok: onnxruntime-gpu is now the only onnxruntime distribution" if ok
          else "!! deconflict failed")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify only; do not install")
    args = ap.parse_args()

    print("=" * 74)
    print("  NexGen iMATCH - GPU environment setup")
    print(f"  interpreter: {sys.executable}")
    print("=" * 74)

    if not args.check:
        missing = [p for p in _REQS if not p.exists()]
        if missing:
            print(f"missing requirements file(s): {missing}")
            return 1
        cmd = [sys.executable, "-m", "pip", "install"]
        for p in _REQS:
            cmd += ["-r", str(p)]
        if run(*cmd) != 0:
            print("\npip install failed")
            return 1

    if not deconflict():
        print("\nenvironment is not in a usable GPU state")
        return 1

    print("\nrunning scripts/verify_gpu.py ...", flush=True)
    return run(sys.executable, str(_ROOT / "scripts" / "verify_gpu.py"))


if __name__ == "__main__":
    raise SystemExit(main())
