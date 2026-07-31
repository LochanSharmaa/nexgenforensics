#!/usr/bin/env python
"""
Generate the evidence appendices for A2 Failure and Recovery Log.

    python scripts/generate_failure_appendices.py

A2's body is analysis: what broke and why. Analysis does not compress into more
words than it contains, so the body is short by design. The evidence for each
finding is bulk material -- the commit that fixed it, the patch, the benchmark
output that proves it -- and belongs in appendices where it can be checked
without interrupting the argument.

This assembles those appendices FROM THE REPOSITORY, so the evidence attached to
each failure is the real artefact rather than a description of one. Every
appendix states the command that regenerates it.

Failures are matched to commits by pattern rather than by pinned hash: a pinned
hash breaks silently on rebase and would leave a failure documented with someone
else's patch attached.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# (id, title, commit-subject patterns, artefact files, scripts whose docstring
#  records the reasoning)
FAILURES: list[tuple[str, str, list[str], list[str], list[str]]] = [
    ("F-01", "GPU silently fell back to CPU",
     ["gpu", "cuda", "provider"],
     [], ["scripts/verify_gpu.py", "scripts/setup_gpu.py",
          "backend/nexgen_engine/models/cuda_runtime.py"]),
    ("F-02", "The same GPU fallback in the API layer",
     ["gpu", "cuda", "device"], [], []),
    ("F-03", "Threshold drift and the demonstrated false match",
     ["threshold", "0.2871", "false match"],
     ["runtime/benchmarks/threshold_calibration.json",
      "runtime/benchmarks/demographics_w600k_r50_thr0287.json"],
     ["backend/scripts/calibrate_threshold_suite.py"]),
    ("F-04", "Validation proxy reported improvement while the model got worse",
     ["fine-tune", "finetune", "negative"],
     ["runtime/benchmarks/finetuned_v1.json"],
     ["backend/scripts/finetune_degraded.py",
      "backend/scripts/eval_finetuned_checkpoint.py"]),
    ("F-05", "Contamination audit produced a 96.9% false positive",
     ["QMUL", "overlap", "audit"],
     ["runtime/benchmarks/qmul_exclusion_list.json",
      "runtime/benchmarks/qmul_overlap_control.json",
      "runtime/benchmarks/qmul_quality.json"],
     ["backend/scripts/audit_qmul_survface.py"]),
    ("F-06", "The control for F-05 was itself wrong",
     ["QMUL", "control"], ["runtime/benchmarks/qmul_overlap_control.json"],
     ["backend/scripts/qmul_overlap_control.py"]),
    ("F-07", "Core source files were never committed",
     ["commit", "untrack", "missing"], [], []),
    ("F-08", "500 on extreme aspect ratios", ["aspect", "geometry", "500"], [], []),
    ("F-09", "500 on any non-image upload", ["500", "upload", "storage"], [], []),
    ("F-10", "Authentication broken by per-call secret regeneration",
     ["jwt", "secret", "auth"], [], []),
    ("F-11", "Circular import at startup", ["circular", "import"], [], []),
    ("F-12", "Accounts that could never authenticate",
     ["log in", "email", "csrf"], [], []),
    ("F-13", "ANN benchmark measured nothing",
     ["ann", "faiss", "index"],
     ["runtime/benchmarks/ann_search.json", "runtime/benchmarks/ann_search_real.json"],
     ["backend/scripts/benchmark_ann.py"]),
    ("F-14", "CFP-FP anomaly was dataset provenance",
     ["cfp", "pack", "provenance"], [], []),
    ("F-15", "Fine-tuning attempt 1 - synthetic degradation",
     ["Phase 6", "fine-tune", "negative"],
     ["runtime/benchmarks/finetuned_v1.json"],
     ["backend/scripts/finetune_degraded.py"]),
    ("F-16", "Fine-tuning attempt 2 - real data, no transfer",
     ["QMUL", "licence", "audit"],
     ["runtime/benchmarks/finetuned_qmul_v2.json",
      "runtime/checkpoints/arcface_qmul_v2_history.json"],
     ["backend/scripts/finetune_qmul.py"]),
    ("F-17", "Recovering value through quality-routed selection",
     ["rout", "threshold", "validate"],
     ["runtime/benchmarks/routed_engine.json",
      "runtime/benchmarks/routed_engine_validated.json",
      "runtime/benchmarks/routing_threshold.json"],
     ["backend/scripts/evaluate_routed_engine.py"]),
    ("F-18", "onnxruntime-gpu wheel unavailable", ["onnxruntime", "gpu"], [], []),
    ("F-19", "Deploy OOM at 512 MB", ["deploy", "trim", "render"], [], []),
    ("F-20", "--no-deps broke FastAPI", ["deps", "requirements"], [], []),
    ("F-21", "RESEND_API_KEY never loaded", ["resend", "registration"], [], []),
    ("F-22", "frontend/dist perpetually modified", ["dist", "untrack"], [], []),
    ("F-23", "CPU/GPU numeric divergence", ["render", "psycopg", "cpu"], [], []),
    ("F-24", "CSRF middleware sat outside CORS", ["csrf", "security header"], [], []),
    ("F-25", "SameSite blocked the cookie", ["csrf", "security header"], [], []),
]


def git(*args: str) -> str:
    r = subprocess.run(["git", *args], cwd=_ROOT, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    return r.stdout if r.returncode == 0 else ""


def matching_commits(patterns: list[str], limit: int = 4) -> list[str]:
    seen: list[str] = []
    for pat in patterns:
        out = git("log", "--all", "-i", f"--grep={pat}", "--format=%H", f"-{limit}")
        for sha in out.splitlines():
            if sha and sha not in seen:
                seen.append(sha)
    return seen[:limit]


def docstring_of(path: str) -> str:
    p = _ROOT / path
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8", errors="replace")
    if '"""' not in text:
        return ""
    start = text.index('"""') + 3
    end = text.find('"""', start)
    return text[start:end].strip() if end > start else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_ROOT / "delivery/A2-APPENDICES.md"))
    ap.add_argument("--max-diff-lines", type=int, default=6000)
    args = ap.parse_args()

    parts: list[str] = [f"""# A2 Appendices — Evidence for the Failure and Recovery Log

Generated from the repository by `scripts/generate_failure_appendices.py`.

Each appendix carries the primary evidence for one entry in A2: the commits that
fixed it with their full patches, the measurement artefacts that prove the fix,
and the reasoning recorded in the relevant scripts at the time.

Nothing here is a description of evidence. It is the evidence.

---

## Contents

| Appendix | Failure |
|---|---|
"""]
    for fid, title, _, _, _ in FAILURES:
        parts.append(f"| {fid} | {title} |\n")
    parts.append("\n---\n")

    for fid, title, patterns, artefacts, scripts in FAILURES:
        print(f"  {fid} ...", end="\r", flush=True)
        parts.append(f"\n# Appendix {fid} — {title}\n\n")

        # ---- reasoning recorded in the code at the time -----------------
        for script in scripts:
            doc = docstring_of(script)
            if doc:
                parts.append(f"## Reasoning recorded in `{script}`\n\n```text\n")
                parts.append(doc)
                parts.append("\n```\n\n")

        # ---- measurement artefacts -------------------------------------
        for artefact in artefacts:
            p = _ROOT / artefact
            if not p.exists():
                parts.append(f"## `{artefact}`\n\n*Not present in this working "
                             f"tree (runtime artefacts are gitignored; regenerate "
                             f"with the scripts named in A2).*\n\n")
                continue
            raw = p.read_text(encoding="utf-8", errors="replace")
            try:
                raw = json.dumps(json.loads(raw), indent=2)
            except Exception:
                pass
            parts.append(f"## Measurement artefact — `{artefact}`\n\n```json\n")
            parts.append(raw)
            parts.append("\n```\n\n")

        # ---- commits ----------------------------------------------------
        shas = matching_commits(patterns)
        if not shas:
            parts.append("*No commit matched the search patterns for this entry; "
                         "see A1 Development History for the full record.*\n\n")
            continue

        for sha in shas:
            subject = git("show", "-s", "--format=%s", sha).strip()
            body = git("show", "-s", "--format=%b", sha).strip()
            day = git("show", "-s", "--format=%ad", "--date=short", sha).strip()
            parts.append(f"## Commit `{sha[:12]}` — {subject}\n\n**Date:** {day}\n\n")
            if body:
                parts.append("### Rationale as recorded at the time\n\n```text\n")
                parts.append(body)
                parts.append("\n```\n\n")
            stat = git("show", sha, "--format=", "--stat=100").strip()
            if stat:
                parts.append("### Files changed\n\n```text\n")
                parts.append(stat)
                parts.append("\n```\n\n")
            diff = git("show", sha, "--format=", "--no-color").splitlines()
            if len(diff) > args.max_diff_lines:
                diff = diff[:args.max_diff_lines] + [
                    f"... [truncated; full patch: git show {sha[:12]}]"]
            if diff:
                parts.append("### Patch\n\n```diff\n")
                parts.append("\n".join(diff))
                parts.append("\n```\n\n")

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
