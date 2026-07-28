"""Check whether this machine can actually run NexGen iMATCH.

    python scripts/check_environment.py

Reports what is installed and, most importantly, whether the recognition model
can load. A service that starts without it still answers every request -- with
scores that look completely normal and mean nothing -- so this check exists to
make that state visible before you rely on it.

Exit codes: 0 recognition works, 1 service runs but cannot recognize,
2 the service cannot run at all.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REQUIRED = [
    ("fastapi", "HTTP framework"),
    ("uvicorn", "ASGI server"),
    ("sqlmodel", "persistence"),
    ("pydantic", "validation"),
    ("PIL", "image decoding (pillow)"),
    ("numpy", "numerics"),
    ("cryptography", "template encryption"),
    ("argon2", "password hashing (argon2-cffi)"),
    ("jwt", "tokens (PyJWT)"),
]

RECOGNITION = [
    ("insightface", "RetinaFace detection + ArcFace recognition"),
    ("onnxruntime", "ONNX inference"),
    ("cv2", "image ops (opencv-python-headless)"),
]


def check(modules: list[tuple[str, str]]) -> list[str]:
    missing = []
    for name, purpose in modules:
        installed = importlib.util.find_spec(name) is not None
        print(f"  [{'ok' if installed else 'MISSING'}] {name:<16} {purpose}")
        if not installed:
            missing.append(name)
    return missing


def main() -> int:
    print(f"Python {sys.version.split()[0]} at {sys.executable}")
    if sys.version_info < (3, 11):
        print("\nPython 3.11 or later is required.")
        return 2

    print("\nCore service dependencies:")
    missing_core = check(REQUIRED)

    print("\nRecognition dependencies:")
    missing_recognition = check(RECOGNITION)

    if missing_core:
        print("\nThe service cannot start. Install core dependencies:")
        print("  pip install -r backend/requirements.txt")
        return 2

    if missing_recognition:
        print("\n" + "=" * 74)
        print("The service will start, but it CANNOT RECOGNIZE ANYONE.")
        print("It falls back to a deterministic stub that hashes pixels: two photographs")
        print("of the same person produce unrelated templates, and every score it returns")
        print("is meaningless. Install the recognition engine:")
        print("  pip install -r backend/requirements-engine.txt")
        print("=" * 74)
        return 1

    print("\nAll dependencies present. Loading the model to confirm it works...")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

    try:
        from nexgen_engine.config import EngineConfig
        from nexgen_engine.runtime import EngineRuntime

        runtime = EngineRuntime(EngineConfig(mode="real"))
        info = runtime.recognizer.info
        print(f"  Recognizer: {info.backend} ({info.model_pack}) on {info.device}")
        print(f"  Detector:   {runtime.detector.name}, landmarks={runtime.detector.produces_landmarks}")
    except Exception as exc:
        print(f"\nThe model failed to load: {exc}")
        print("The first run downloads ~350 MB into ~/.insightface/models; check network access.")
        return 1

    caps = None
    try:
        import onnxruntime

        caps = onnxruntime.get_available_providers()
    except Exception:
        pass
    if caps:
        print(f"  ONNX providers: {', '.join(caps)}")
        if "CUDAExecutionProvider" not in caps:
            print("  (CPU only. Install onnxruntime-gpu and set NEXGEN_ENGINE_DEVICE=cuda for GPU.)")

    print("\nRecognition is working. Calibrate thresholds before deploying:")
    print("  python backend/scripts/calibrate_threshold.py <dataset>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
