from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runtime import detect_runtime_capabilities


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NexGen facial engine utilities")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("capabilities", help="Print optional production dependency status")
    validate = sub.add_parser("validate", help="Run the local validation script")
    validate.add_argument("--script", default=str(Path(__file__).resolve().parents[1] / "scripts" / "validate_system.py"))
    args = parser.parse_args(argv)

    if args.command == "capabilities":
        print(json.dumps(detect_runtime_capabilities().__dict__, indent=2, sort_keys=True))
        return 0
    if args.command == "validate":
        namespace: dict[str, object] = {"__name__": "__main__"}
        exec(Path(args.script).read_text(encoding="utf-8"), namespace)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
