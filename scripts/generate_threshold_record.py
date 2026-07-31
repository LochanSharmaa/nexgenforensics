#!/usr/bin/env python
"""
Generate A7 — Threshold Calibration Record.

    python scripts/generate_threshold_record.py

The decision threshold is the single most consequential number in this system.
Every match or non-match a user sees is that comparison. This document records
where the current value came from, what was measured at each candidate, what was
rejected and why, and what the value does NOT fix.

WHY THIS IS A SEPARATE DOCUMENT
-------------------------------
A threshold buried in a configuration file looks like a tuning parameter. It is
not. It is the point at which the system asserts two photographs show the same
person, and in a forensic setting it is the number most likely to be challenged.
The reasoning behind it should be readable without reading the source.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
BENCH = _ROOT / "runtime" / "benchmarks"

DEMOGRAPHIC_RUNS = [
    ("0.20", "demographics_w600k_r50_thr020.json",
     "The value in force when a user demonstrated a false match at 0.2405."),
    ("0.2363", "demographics_w600k_r50_thr0236.json",
     "Suite-wide recalibration candidate. REJECTED — see Part III."),
    ("0.2871", "demographics_w600k_r50_thr0287.json",
     "**Adopted.** In force in the delivered system."),
]


def load(name: str):
    p = BENCH / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_ROOT / "delivery/A7-THRESHOLD-CALIBRATION-RECORD.md"))
    args = ap.parse_args()

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_ROOT,
                            capture_output=True, text=True).stdout.strip()[:12]

    parts: list[str] = [f"""# A7 — Threshold Calibration Record

**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} ·
**Repository state:** `{commit}`

| Threshold | Value | Status |
|---|---|---|
| Match | **0.2871** | In force |
| Review band | 0.2153 | In force |
| Routing (degraded specialist) | 0.539 | Measured, not enabled by default |

The decision threshold is the most consequential number in this system: every
match or non-match a user sees is that comparison. A threshold sitting in a
configuration file looks like a tuning parameter, and it is not — it is the
point at which the system asserts two photographs show the same person.

---

# Part I — How the current value was arrived at

## The failure that started it

A user submitted two images of **different people** and the interface reported a
match at similarity **0.2405**.

The cause was not the threshold being wrong in principle. It was that the
threshold existed as **four independent constants in four files**, which had
drifted apart, and the lowest value in the chain was **0.20** — low enough to
admit a genuine false match. The engine, the API, the test suite and the
interface did not agree on what the system's decision rule was.

Recorded in full as A2 / F-03.

## The structural fix

`ThresholdConfig` in `nexgen_engine/config.py` became the single source of
truth. Every other site derives from it rather than restating it:

```python
match: float = 0.2871
review: float = 0.2153
verify: float = 0.2871
```

An API test that had hard-coded `top_score=0.36` was rewritten to derive its
expectation from configuration. That test had been passing against the same
stale constant the code used, so it could not have detected the drift — it was
confirming the bug rather than catching it.

## Why 0.2871

Raising the threshold trades recall for a lower false-match rate. For a system
generating investigative leads about real people that is the correct direction:
a missed lead is recoverable by other means, a false lead attaches a name to a
person who is not in the image.

0.2871 places the demonstrated false pair (0.2405) below threshold with margin,
rather than immediately beneath it.

---

# Part II — Measured effect of each candidate

Same model, same pair sets; only the decision threshold differs. FNMR is the
false-non-match rate (true pairs rejected); FMR the false-match rate (different
people accepted).

"""]

    for value, filename, note in DEMOGRAPHIC_RUNS:
        data = load(filename)
        parts.append(f"\n## Threshold {value}\n\n{note}\n\n")
        if data is None:
            parts.append(f"*Artefact `{filename}` not present in this working tree.*\n\n")
            continue
        groups = data.get("groups", {})
        parts.append("| Cohort | Genuine pairs | Impostor pairs | FNMR | FMR |\n")
        parts.append("|---|---|---|---|---|\n")
        for name, g in groups.items():
            parts.append(
                f"| {name} | {g.get('genuine', ''):,} | {g.get('impostor', ''):,} | "
                f"{g.get('fnmr', 0) * 100:.2f}% | {g.get('fmr', 0) * 100:.4f}% |\n")
        parts.append("\n")
        scalars = {k: v for k, v in data.items() if not isinstance(v, (dict, list))}
        if scalars:
            parts.append("| Field | Value |\n|---|---|\n")
            for k, v in scalars.items():
                parts.append(f"| `{k}` | {v} |\n")
            parts.append("\n")
        parts.append(f"<details><summary>Raw artefact — <code>{filename}</code></summary>\n\n"
                     f"```json\n{json.dumps(data, indent=2)}\n```\n\n</details>\n\n")

    # ------------------------------------------------------- rejection ----
    parts.append("""
---

# Part III — The recalibration that was measured and REJECTED

After 0.2871 was adopted, a suite-wide calibration was run to check the value
against every dataset rather than against the single demonstrated failure. It
proposed **0.2363**.

**It was rejected.** At 0.2363 the originally-reported false pair, at 0.2405,
scores above threshold and is admitted again — the exact failure the change was
made to eliminate.

This is recorded because it is the more instructive outcome of the two. 0.2363
is the value that optimises an aggregate metric across the benchmark suite. It
is also the value that reintroduces a demonstrated, reproducible false match on
real submitted images. **An aggregate optimum is not automatically the correct
operating point**, and a calibration script that returns a number is not
authority to deploy it.

""")

    calib = load("threshold_calibration.json")
    if calib:
        parts.append("## Calibration output that proposed 0.2363\n\n")
        per = calib.get("per_dataset")
        if isinstance(per, dict) and per:
            first = next(iter(per.values()))
            keys = list(first) if isinstance(first, dict) else []
            parts.append("| Dataset | " + " | ".join(keys) + " |\n")
            parts.append("|---|" + "---|" * len(keys) + "\n")
            for ds, row in per.items():
                cells = [f"{row[k]:.5f}" if isinstance(row.get(k), float) else str(row.get(k, ""))
                         for k in keys]
                parts.append(f"| {ds} | " + " | ".join(cells) + " |\n")
            parts.append("\n")
        scalars = {k: v for k, v in calib.items() if not isinstance(v, (dict, list))}
        if scalars:
            parts.append("| Field | Value |\n|---|---|\n")
            for k, v in scalars.items():
                parts.append(f"| `{k}` | {v} |\n")
            parts.append("\n")
        parts.append(f"<details><summary>Raw artefact</summary>\n\n```json\n"
                     f"{json.dumps(calib, indent=2)}\n```\n\n</details>\n\n")

    at2871 = load("threshold_calibration_at_2871.json")
    if at2871:
        parts.append("## Suite behaviour at the adopted value\n\n")
        parts.append(f"```json\n{json.dumps(at2871, indent=2)}\n```\n\n")

    # ------------------------------------------------------ what remains --
    parts.append("""
---

# Part IV — What the threshold does NOT fix

## Demographic differentials persist

Comparing the cohort tables in Part II across the three values: raising the
threshold **relocated** the errors, it did not remove the disparity between
cohorts. Women show approximately **1.7x** the false-non-match rate of men, and
under-25s approximately **3.8x** the 41-55 band.

A single global threshold applies the same decision rule to populations on which
the system does not perform equally. That is a property of the model, and no
choice of threshold corrects it. It is recorded as an unresolved limitation in
the Model Card (L3) rather than presented as managed.

## One threshold cannot serve both conditions

The adopted value is calibrated on clean-imagery benchmarks. On degraded
imagery the same model achieves **33.13% TAR at FAR=0.1%** — roughly one true
match in three. No threshold recovers information the embedding does not carry.

The measured response to that is not a different threshold but a different
model, selected per probe. See Part V.

---

# Part V — The routing threshold (0.539)

A degraded-condition specialist model is available and can be selected per
comparison using the quality score the pipeline already computes.

**The operating point was chosen before the measurement, not after it.** The
threshold was derived from QMUL-SurvFace and CASIA-WebFace quality
distributions, both of which are **disjoint from every reporting benchmark**:

| | median | tail |
|---|---|---|
| QMUL (degraded, n=3,000) | 0.4677 | p90 0.5250 |
| CASIA (clean, n=2,924) | 0.7547 | p10 0.5671 |

Crossover, where the miss rate on degraded equals the false-route rate on clean,
is **0.539** (5.8% and 5.9% respectively).

A threshold sweep had shown 0.50 producing better benchmark numbers. It was
**not** used, because selecting an operating point on the sets it will then be
reported against is fitting to the test set — the same error class as A2 / F-04,
in a new place.

""")
    routing = load("routing_threshold.json")
    if routing:
        parts.append(f"```json\n{json.dumps(routing, indent=2)}\n```\n\n")

    validated = load("routed_engine_validated.json")
    if validated:
        res = validated.get("results", {})
        parts.append("## Measured result at 0.539\n\n")
        parts.append("| Dataset | TAR@FAR0.1% deployed | routed | Delta | % routed |\n")
        parts.append("|---|---|---|---|---|\n")
        for ds, r in res.items():
            d = r.get("deployed_tar_1e3", 0)
            t = r.get("routed_tar_1e3", 0)
            parts.append(f"| {ds} | {d:.2f}% | {t:.2f}% | {t - d:+.2f}pp | "
                         f"{r.get('fraction_routed_to_specialist', 0) * 100:.0f}% |\n")
        parts.append("\n")

    parts.append("""
---

# Part VI — Change control

The threshold is part of the validated configuration. Changing it **invalidates
every accuracy figure in this delivery**, because every one of them was measured
under a protocol that fits a threshold on nine folds and applies it to the
tenth.

Before any change:

1. Re-run `calibrate_threshold_suite.py` and record the proposal.
2. Re-run the full verification suite at the candidate value.
3. Re-run `benchmark_demographics.py` at the candidate value — a change that
   improves the aggregate may worsen a cohort.
4. Confirm the candidate still rejects the known false pair at 0.2405.
5. Re-issue the Model Card with a new version.

Step 4 exists because it is the step that stopped 0.2363.
""")

    text = "".join(parts)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    words = len(text.split())
    print(f"wrote {out}")
    print(f"  {len(text.splitlines()):,} lines   {words:,} words")
    print(f"  ~{words / 450:.0f} pages @450 w/p   ~{words / 250:.0f} pages @250 w/p")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
