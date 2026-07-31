#!/usr/bin/env python
"""
Generate A5 — Benchmark and Measurement Record.

    python scripts/generate_benchmark_record.py

Assembles every measurement this system has produced into one document: the
protocol under which each was taken, the harness that took it, the result, and
the complete raw artefact.

WHY THE RAW JSON IS INCLUDED IN FULL
------------------------------------
A performance report that presents only summary tables asks the reader to trust
the summarisation. Including the artefact that the summary was computed from
means a reader can recompute it. For a system whose numbers may be challenged,
that difference is the whole point of the document.

Protocols are read from the docstrings of the scripts that implement them, so
the protocol description cannot drift from the code that enforces it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
BENCH = _ROOT / "runtime" / "benchmarks"

# Measurement families: (heading, artefact globs, implementing scripts)
FAMILIES: list[tuple[str, list[str], list[str]]] = [
    ("1:1 verification — clean protocols",
     ["verification_results.json", "fusion_selection.json"],
     ["backend/scripts/benchmark_verification.py",
      "backend/nexgen_engine/benchmarks/verification.py"]),
    ("Degraded-condition verification (TinyFace)",
     ["tinyface.json"],
     ["backend/scripts/benchmark_tinyface.py"]),
    ("Threshold calibration",
     ["threshold_calibration*.json"],
     ["backend/scripts/calibrate_threshold_suite.py"]),
    ("Demographic differentials",
     ["demographics*.json"],
     ["backend/scripts/benchmark_demographics.py"]),
    ("Training/evaluation contamination",
     ["train_eval_overlap*.json", "overlap_casia_deep.json", "exclusion_list.json",
      "qmul_exclusion_list.json", "qmul_overlap_control.json", "qmul_quality.json"],
     ["backend/scripts/audit_train_eval_overlap.py",
      "backend/scripts/build_exclusion_list.py",
      "backend/scripts/audit_qmul_survface.py",
      "backend/scripts/qmul_overlap_control.py"]),
    ("Fine-tuning outcomes",
     ["finetuned*.json"],
     ["backend/scripts/finetune_degraded.py", "backend/scripts/finetune_qmul.py",
      "backend/scripts/eval_finetuned_checkpoint.py"]),
    ("Quality-routed model selection",
     ["routed_engine*.json", "routing_threshold.json"],
     ["backend/scripts/evaluate_routed_engine.py"]),
    ("Latency, throughput and concurrency",
     ["speed.json", "concurrency.json"],
     ["backend/scripts/benchmark_speed.py",
      "backend/scripts/benchmark_concurrency.py"]),
    ("Approximate nearest-neighbour search",
     ["ann_search*.json"],
     ["backend/scripts/benchmark_ann.py"]),
]


def docstring_of(path: str) -> str:
    p = _ROOT / path
    if not p.exists():
        return ""
    t = p.read_text(encoding="utf-8", errors="replace")
    if '"""' not in t:
        return ""
    s = t.index('"""') + 3
    e = t.find('"""', s)
    return t[s:e].strip() if e > s else ""


def flatten(obj, prefix="") -> list[tuple[str, str]]:
    """Depth-limited flatten so nested results become a readable table."""
    rows: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (dict, list)):
                rows.extend(flatten(v, key))
            else:
                rows.append((key, str(v)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:60]):
            key = f"{prefix}[{i}]"
            if isinstance(v, (dict, list)):
                rows.extend(flatten(v, key))
            else:
                rows.append((key, str(v)))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_ROOT / "delivery/A5-BENCHMARK-RECORD.md"))
    args = ap.parse_args()

    if not BENCH.is_dir():
        print(f"no benchmark directory at {BENCH}")
        return 1

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_ROOT,
                            capture_output=True, text=True).stdout.strip()[:12]

    parts: list[str] = [f"""# A5 — Benchmark and Measurement Record

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')} ·
**Repository state:** `{commit}`

Every measurement this system has produced, with the protocol under which it was
taken, the harness that took it, the result, and the complete raw artefact.

The raw JSON is reproduced in full for each measurement. A report that presents
only summary tables asks the reader to trust the summarisation; including the
artefact means the summary can be recomputed. For numbers that may be
challenged, that is the point of the document.

Protocol descriptions are read from the docstrings of the scripts that implement
them, so a protocol cannot drift from the code enforcing it.

---

## Governing protocol

| Item | Value |
|---|---|
| Task | 1:1 verification unless explicitly stated |
| Pair source | Official InsightFace `.bin` packs; counts asserted at load |
| Images | Pre-aligned 112x112 ArcFace crops |
| Augmentation | Horizontal-flip TTA, `L2norm(f(x) + f(flip(x)))` |
| Cross-validation | 10-fold, contiguous folds per published layout |
| **Threshold** | **Fitted on 9 folds, applied to the held-out fold** |
| Similarity | Cosine, range [-1, 1] |

**No accuracy figure in this record is measured at a threshold tuned on the
pairs it is reported against.**

---

## Index of measurements

| Family | Artefacts |
|---|---|
"""]
    for name, globs, _ in FAMILIES:
        files = sorted({p.name for g in globs for p in BENCH.glob(g)})
        parts.append(f"| {name} | {len(files)} |\n")
    parts.append("\n---\n")

    total_files = 0
    for name, globs, scripts in FAMILIES:
        files = sorted({p for g in globs for p in BENCH.glob(g)}, key=lambda p: p.name)
        parts.append(f"\n# {name}\n\n")
        if not files:
            parts.append("*No artefact present in this working tree. "
                         "Regenerate with the scripts below.*\n\n")

        for script in scripts:
            doc = docstring_of(script)
            if doc:
                parts.append(f"## Protocol as implemented — `{script}`\n\n```text\n")
                parts.append(doc)
                parts.append("\n```\n\n")

        for f in files:
            total_files += 1
            print(f"  {f.name}", end="\r", flush=True)
            raw = f.read_text(encoding="utf-8", errors="replace")
            try:
                data = json.loads(raw)
                pretty = json.dumps(data, indent=2)
            except Exception:
                data, pretty = None, raw

            parts.append(f"## Measurement — `runtime/benchmarks/{f.name}`\n\n")

            if data is not None:
                rows = flatten(data)
                scalar = [r for r in rows if len(r[0]) < 90][:400]
                if scalar:
                    parts.append("### Values\n\n| Field | Value |\n|---|---|\n")
                    for k, v in scalar:
                        vv = v.replace("|", "\\|")[:200]
                        parts.append(f"| `{k}` | {vv} |\n")
                    parts.append("\n")

            parts.append("### Raw artefact\n\n```json\n")
            parts.append(pretty)
            parts.append("\n```\n\n")

    text = "".join(parts)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    words = len(text.split())
    print(f"\nwrote {out}")
    print(f"  {total_files} artefacts   {len(text.splitlines()):,} lines   {words:,} words")
    print(f"  ~{words / 450:.0f} pages @450 w/p   ~{words / 250:.0f} pages @250 w/p")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
