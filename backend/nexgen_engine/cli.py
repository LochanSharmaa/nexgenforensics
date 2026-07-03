from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runtime import detect_runtime_capabilities
from .dependencies import DependencyVerifier


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NexGen facial engine utilities")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("capabilities", help="Print optional production dependency status")
    sub.add_parser("verify-deps", help="Print detailed dependency verification status")
    strict = sub.add_parser("strict-production", help="Fail if production ML dependencies are unavailable")
    strict.add_argument("--require", nargs="*", default=["torch", "timm", "faiss", "onnx", "onnxruntime"])
    detector = sub.add_parser("detector-smoke", help="Run detector smoke test on an image")
    detector.add_argument("--image", required=True)
    detector.add_argument("--backend", default="center_crop")
    detector.add_argument("--fallback-allowed", action="store_true")
    detector.add_argument("--output", required=True)
    quality = sub.add_parser("quality-check", help="Run quality check on an image")
    quality.add_argument("--image", required=True)
    quality.add_argument("--output")
    quality.add_argument("--faceqnet-checkpoint")
    validate = sub.add_parser("validate", help="Run the local validation script")
    validate.add_argument("--script", default=str(Path(__file__).resolve().parents[1] / "scripts" / "validate_system.py"))
    args = parser.parse_args(argv)

    if args.command == "capabilities":
        print(json.dumps(detect_runtime_capabilities().__dict__, indent=2, sort_keys=True))
        return 0
    if args.command == "verify-deps":
        print(json.dumps([item.__dict__ for item in DependencyVerifier().statuses()], indent=2, sort_keys=True))
        return 0
    if args.command == "strict-production":
        DependencyVerifier().assert_available(args.require)
        print(json.dumps({"status": "ok", "required": args.require}, indent=2))
        return 0
    if args.command == "detector-smoke":
        from .detection.smoke import detector_smoke

        print(detector_smoke(args.image, args.backend, args.fallback_allowed, args.output))
        return 0
    if args.command == "quality-check":
        from .quality.cli import quality_check

        print(json.dumps(quality_check(args.image, args.output, args.faceqnet_checkpoint), indent=2, sort_keys=True))
        return 0
    if args.command == "validate":
        namespace: dict[str, object] = {"__name__": "__main__"}
        exec(Path(args.script).read_text(encoding="utf-8"), namespace)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
