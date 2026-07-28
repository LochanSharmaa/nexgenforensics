from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import EngineConfig
from .inference.pipeline import FacialRecognitionPipeline
from .runtime import detect_runtime_capabilities
from .utils import cosine_similarity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nexgen_engine",
        description="NexGen iMATCH recognition engine utilities.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("capabilities", help="Report which optional dependencies are installed.")

    status = sub.add_parser("status", help="Load the engine and report what is actually running.")
    status.add_argument("--mode", choices=["auto", "real", "stub"], default="auto")

    compare = sub.add_parser("compare", help="Compare two face images and print their similarity.")
    compare.add_argument("reference", type=Path)
    compare.add_argument("probe", type=Path)
    compare.add_argument("--mode", choices=["auto", "real", "stub"], default="auto")

    args = parser.parse_args(argv)

    if args.command == "capabilities":
        print(json.dumps(detect_runtime_capabilities().as_dict(), indent=2, sort_keys=True))
        return 0

    if args.command == "status":
        pipeline = FacialRecognitionPipeline(EngineConfig(mode=args.mode))
        print(json.dumps(pipeline.runtime.status(), indent=2, sort_keys=True))
        # Non-zero when the engine cannot recognize, so CI and deploy scripts can
        # gate on it rather than having to parse the JSON.
        return 0 if pipeline.runtime.recognition_capable else 1

    if args.command == "compare":
        pipeline = FacialRecognitionPipeline(EngineConfig(mode=args.mode))
        if not pipeline.runtime.recognition_capable:
            print(
                "No recognition model is loaded, so this comparison would be meaningless.\n"
                "Install it first:  pip install -r requirements-engine.txt"
            )
            return 2

        reference = pipeline.encode_bytes(args.reference.read_bytes())
        probe = pipeline.encode_bytes(args.probe.read_bytes())
        similarity = cosine_similarity(reference.embedding, probe.embedding)
        threshold = pipeline.config.thresholds.verify

        print(
            json.dumps(
                {
                    "similarity": round(similarity, 6),
                    "threshold": threshold,
                    "above_threshold": similarity >= threshold,
                    "reference_quality": reference.quality.score,
                    "probe_quality": probe.quality.score,
                    "note": (
                        "Similarity is not the probability that these images show the same "
                        "person. An examiner must verify any conclusion."
                    ),
                },
                indent=2,
            )
        )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
