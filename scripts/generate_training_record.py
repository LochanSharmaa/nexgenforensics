#!/usr/bin/env python
"""
Generate A4 — Model and Training Record.

    python scripts/generate_training_record.py

Every model artefact this system contains or produced, every training run
attempted, and the outcome of each — including the two that failed.

MODEL INTEGRITY
---------------
SHA-256 is computed for every weight file found. A forensic deployment must be
able to prove the model that produced a result is the model that was validated,
and a checksum computed at delivery is the only way to demonstrate later that
the file has not changed. These digests are the reference values.

WHY FAILED RUNS ARE INCLUDED IN FULL
------------------------------------
Two of the three training runs made the model worse or did nothing. Their
configurations, step histories and evaluations are recorded at the same level of
detail as the successful work. A training record containing only the runs that
worked cannot answer "was this approach tried?", which is the question a record
exists to answer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

WEIGHT_DIRS = [
    (_ROOT / "runtime" / "checkpoints", "Checkpoints produced by this project"),
    (Path.home() / ".insightface" / "models" / "buffalo_l", "Deployed InsightFace pack"),
]

# (id, title, script, history artefact, evaluation artefact)
RUNS: list[tuple[str, str, str, str, str]] = [
    ("R1", "Initial fine-tune attempt (superseded)",
     "backend/nexgen_engine/training/arcface_loss.py", "", ""),
    ("R2", "Synthetic-degradation fine-tune — NEGATIVE RESULT",
     "backend/scripts/finetune_degraded.py", "",
     "runtime/benchmarks/finetuned_v1.json"),
    ("R3", "Real-degraded-data fine-tune (QMUL-SurvFace)",
     "backend/scripts/finetune_qmul.py",
     "runtime/checkpoints/arcface_qmul_v2_history.json",
     "runtime/benchmarks/finetuned_qmul_v2.json"),
    ("R4", "Quality-routed model selection — ADOPTED CANDIDATE",
     "backend/scripts/evaluate_routed_engine.py",
     "runtime/benchmarks/routing_threshold.json",
     "runtime/benchmarks/routed_engine_validated.json"),
]

SUPPORTING = [
    "backend/scripts/build_exclusion_list.py",
    "backend/scripts/audit_qmul_survface.py",
    "backend/scripts/qmul_overlap_control.py",
    "backend/scripts/eval_finetuned_checkpoint.py",
]


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def read(path: str) -> str:
    p = _ROOT / path
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_ROOT / "delivery/A4-MODEL-AND-TRAINING-RECORD.md"))
    ap.add_argument("--no-checksums", action="store_true",
                    help="skip hashing (large files take time)")
    args = ap.parse_args()

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_ROOT,
                            capture_output=True, text=True).stdout.strip()[:12]

    parts: list[str] = [f"""# A4 — Model and Training Record

**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} ·
**Repository state:** `{commit}`

Every model artefact this system contains or produced, every training run
attempted, and the outcome of each.

Two of the training runs recorded here made the model worse or changed nothing.
They appear at the same level of detail as the work that succeeded, because a
training record containing only successful runs cannot answer *"was this
approach tried, and what happened?"* — which is the question the record exists
to answer.

---

## Statement of origin

**The deployed recognition weights were not trained by this project.** They are
stock InsightFace `buffalo_l` weights. The system's clean-image accuracy is the
accuracy of the public state of the art, and its provenance is publicly
verifiable. Everything in the "Checkpoints produced by this project" section
below is experimental; **none of it is the deployed model**, and the one
candidate recommended for use (R4) is a *routing rule* over two models rather
than a new set of weights.

---

# Part I — Model inventory and integrity

SHA-256 digests are the reference values for this delivery. A deployment must be
able to demonstrate that the model which produced a result is the model that was
validated; comparing against these digests is how that is shown.

Verify with:

```bash
sha256sum <file>          # Linux/macOS
certutil -hashfile <file> SHA256   # Windows
```

"""]

    for directory, label in WEIGHT_DIRS:
        parts.append(f"## {label}\n\n`{directory}`\n\n")
        if not directory.is_dir():
            parts.append("*Directory not present in this environment.*\n\n")
            continue
        files = sorted(p for p in directory.iterdir()
                       if p.suffix in {".pt", ".onnx", ".engine", ".pth", ".bin"})
        if not files:
            parts.append("*No weight files found.*\n\n")
            continue
        parts.append("| File | Size | SHA-256 |\n|---|---|---|\n")
        for f in files:
            size = f.stat().st_size
            digest = "*(not computed)*" if args.no_checksums else f"`{sha256(f)}`"
            print(f"  hashed {f.name}", end="\r", flush=True)
            parts.append(f"| `{f.name}` | {size / 1e6:,.1f} MB | {digest} |\n")
        parts.append("\n")

    # ---------------------------------------------------------------- runs --
    parts.append("\n---\n\n# Part II — Training runs\n\n")

    for rid, title, script, history, evaluation in RUNS:
        parts.append(f"\n## {rid} — {title}\n\n")

        src = read(script)
        if src:
            doc_end = src.find('"""', src.find('"""') + 3)
            doc = src[src.find('"""') + 3:doc_end].strip() if doc_end > 0 else ""
            if doc:
                parts.append(f"### Method and rationale, as recorded in "
                             f"`{script}`\n\n```text\n{doc}\n```\n\n")

        if history:
            p = _ROOT / history
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                parts.append(f"### Training history — `{history}`\n\n")
                rows = data.get("history")
                if isinstance(rows, list) and rows:
                    keys = list(rows[0])
                    parts.append("| " + " | ".join(keys) + " |\n")
                    parts.append("|" + "---|" * len(keys) + "\n")
                    for row in rows:
                        parts.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |\n")
                    parts.append("\n")
                scalars = {k: v for k, v in data.items() if not isinstance(v, (list, dict))}
                if scalars:
                    parts.append("| Field | Value |\n|---|---|\n")
                    for k, v in scalars.items():
                        parts.append(f"| `{k}` | {v} |\n")
                    parts.append("\n")
                parts.append(f"#### Raw artefact\n\n```json\n"
                             f"{json.dumps(data, indent=2)}\n```\n\n")

        if evaluation:
            p = _ROOT / evaluation
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                parts.append(f"### Evaluation — `{evaluation}`\n\n```json\n"
                             f"{json.dumps(data, indent=2)}\n```\n\n")
            else:
                parts.append(f"### Evaluation — `{evaluation}`\n\n"
                             f"*Artefact not present in this working tree.*\n\n")

        if src:
            parts.append(f"### Complete implementation — `{script}`\n\n"
                         f"```python\n{src}\n```\n\n")

    # ------------------------------------------------------- supporting ----
    parts.append("\n---\n\n# Part III — Supporting instrumentation\n\n")
    parts.append("Scripts that established dataset integrity and evaluated the "
                 "resulting checkpoints. Included in full because the validity of "
                 "every training outcome above depends on them being correct.\n\n")
    for script in SUPPORTING:
        src = read(script)
        if src:
            parts.append(f"## `{script}`\n\n```python\n{src}\n```\n\n")

    text = "".join(parts)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    words = len(text.split())
    print(f"\nwrote {out}")
    print(f"  {len(text.splitlines()):,} lines   {words:,} words")
    print(f"  ~{words / 450:.0f} pages @450 w/p   ~{words / 250:.0f} pages @250 w/p")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
