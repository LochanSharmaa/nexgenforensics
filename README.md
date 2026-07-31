# NexGen iMATCH

Forensic face recognition with auditable 1:1 verification and 1:N gallery search, built so every accuracy claim is measured and reproducible.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests: 161 passing](https://img.shields.io/badge/tests-161%20passing-brightgreen.svg)](#running-the-tests)
[![LFW 99.78%](https://img.shields.io/badge/LFW%201%3A1-99.78%25-informational.svg)](BENCHMARKS.md)
[![TinyFace 82.45%](https://img.shields.io/badge/TinyFace-82.45%25-orange.svg)](BENCHMARKS.md)

> **On badges:** there is no CI in this repository, so there is deliberately no
> build-status badge. The test and accuracy badges above are static values
> reproduced by the commands in [Running the tests](#running-the-tests) and
> [Benchmarks](#benchmarks); they do not update automatically. Adding GitHub
> Actions would make them live.

---

## Demo

A real 1:1 comparison against the running API. Two photographs of the same
person, taken seven years apart:

```bash
curl -X POST http://127.0.0.1:8443/api/imatch/verify \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
        "reference_image_base64": "...",
        "probe_image_base64": "...",
        "lawful_basis": "Warrant 2026/114"
      }'
```

```jsonc
{
  "similarity": 0.494431,
  "verified": true,
  "threshold": 0.2871,
  "explanation": "Similarity 0.494 meets the verification threshold 0.29. This supports, but does not establish, that the images show the same person.",
  "reference": {
    "quality":  { "score": 0.731, "sharpness": 1.0, "accepted": true },
    "liveness": { "score": 0.5634, "certified": false,
                  "method": "passive_single_frame_heuristic" },
    "deepfake_risk": 0.2771,
    "pose": { "yaw": 9.41, "pitch": -4.56, "roll": 1.48 }
  },
  "audit_hash": "1656a2d54b6a7a428daa0fc9b35dee0b2bc143a4c76b5cca0d640e2bd1ee2ebc",
  "notice": "Automated face recognition returns investigative leads, not identifications. A qualified examiner must verify any candidate before it is relied upon."
}
```

Note `"certified": false` on the liveness block. Every heuristic in this system
says so in its own output.

---

## Why this exists

Face recognition tooling routinely reports a single blended accuracy figure that
does not survive contact with real evidence — clean web photographs and
surveillance stills differ by more than fifteen accuracy points, and averaging
them hides exactly the case that matters.

This project separates those numbers, records the protocol behind each one, and
labels every component that is a heuristic rather than a trained model. It is
built for investigative and research use where a result may have to be explained
to someone who did not write it.

---

## Features

- **1:1 verification** — cosine similarity on L2-normalised ArcFace templates,
  with a calibrated decision threshold and a plain-language explanation.
- **1:N gallery search** — ranked candidates, subject-collapsed so one
  well-enrolled person cannot fill the list.
- **Batch processing** — three modes: one reference against many probes, many
  independent pairs, or many probes against the gallery. One unreadable file
  does not fail the batch.
- **Forensic PDF reports** — case header, per-search findings, chain-of-custody
  hashes, examiner sign-off, and the investigative-lead notice on every page.
- **Hash-chained audit trail** — every search, verification and export writes a
  retrievable record. Audit hashes returned to callers resolve to real rows.
- **Tenant isolation** — enforced structurally: each tenant's vectors live in a
  separate shard and `search()` requires a tenant id, so no code path can
  compare across tenants.
- **Encrypted templates at rest** — AES-256-GCM per row. Authenticated, so a
  wrong key raises rather than decrypting to plausible noise.
- **Image quality, capture and synthetic-media screening** — reported alongside
  every result, labelled as heuristics (see [Limitations](#limitations)).
- **Verified GPU execution** — asserts the provider an ONNX session actually
  bound, not the one the wheel was compiled with.

Deeper detail: [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md).

---

## Quick start

Requires Python 3.11+, Node 18+, and an NVIDIA GPU with CUDA 12.x for
GPU inference (CPU works, more slowly).

```bash
git clone https://github.com/LochanSharmaa/nexgenforensics.git
cd nexgenforensics
python -m venv .venv && .venv/Scripts/activate      # Linux/macOS: source .venv/bin/activate
```

**1. Install.** On a GPU host use the installer — it resolves a dependency
conflict that a plain `pip install` cannot:

```bash
python scripts/setup_gpu.py
```

<details>
<summary>Why a script rather than <code>pip install -r</code></summary>

`insightface` depends on the CPU `onnxruntime`, and pip has no way to know that
`onnxruntime-gpu` provides the same import package. A plain install lands both
in one namespace and whichever unpacks last wins — often the CPU build, silently,
at roughly 20× the latency. `setup_gpu.py` installs, removes the CPU
distribution, and then asserts CUDA actually bound.
</details>

CPU-only hosts:

```bash
pip install -r backend/requirements.txt -r backend/requirements-engine.txt -r backend/requirements-cpu.txt
```

**2. Configure secrets.** Both are read from a `.env` at the repository root:

```bash
python - <<'EOF'
import secrets, base64, pathlib
pathlib.Path(".env").write_text(
    f"NEXGEN_JWT_SECRET={secrets.token_urlsafe(64)}\n"
    f"NEXGEN_TEMPLATE_KEY={base64.b64encode(secrets.token_bytes(32)).decode()}\n"
    "NEXGEN_ENGINE_DEVICE=auto\n")
EOF
```

Without `NEXGEN_JWT_SECRET` the service generates an ephemeral per-process
secret and every token dies on restart. Without `NEXGEN_TEMPLATE_KEY`
biometric templates are stored **unencrypted**, and the service says so loudly
at startup.

**3. Create the first operator.** No account exists by default, and every
biometric endpoint requires one:

```bash
python backend/scripts/bootstrap_admin.py \
  --email 'investigator@your-org.example' \
  --password '<a strong password>' \
  --role admin
```

**4. Run.**

```bash
python -m uvicorn imatch_api.main:app --host 127.0.0.1 --port 8443 --app-dir backend
```

API docs at `http://127.0.0.1:8443/docs`. First start takes 30–60 s while the
InsightFace model pack loads (~350 MB, downloaded once).

**5. Frontend.**

```bash
cd frontend && npm install && npm run dev
```

**Verify the install:**

```bash
python scripts/verify_gpu.py     # 12 checks; fails loudly if inference is silently on CPU
```

---

## Architecture

Two Python packages and a React frontend.

```
React SPA  ──HTTPS/JSON──▶  imatch_api  ──▶  nexgen_engine  ──▶  ONNX Runtime
(Vite)                      :8443            (library)           (CUDA / CPU)
                              │                    │
                              ▼                    ▼
                        SQLite/SQLModel      InsightFace model pack
                        cases · subjects     buffalo_l / w600k_r50
                        templates · audit
```

| Component | Responsibility |
|---|---|
| **`nexgen_engine`** | Recognition library: detection, alignment, embedding, quality/liveness/deepfake screening, gallery index, benchmark harness. No HTTP, no database. |
| **`imatch_api`** | FastAPI service: auth, tenancy, cases, subjects, search, batch, audit, reports. Owns persistence. |
| **`frontend`** | React 19 + Vite. Marketing site plus an authenticated investigator workspace. |

The single source of truth for decision thresholds is
`nexgen_engine/config.py::ThresholdConfig`; the API derives from it, and a
[regression gate](#running-the-tests) fails if a second copy ever reappears.

Full folder map and module reference: [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md).

---

## Benchmarks

1:1 verification, deployed model `w600k_r50`, published pair protocols,
10-fold cross-validation with the threshold fitted on 9 folds and applied to the
held-out fold.

| Dataset | Pairs | Accuracy | What it stresses |
|---|---:|---|---|
| LFW | 6,000 | **99.78%** | Frontal, unconstrained |
| AgeDB-30 | 6,000 | **96.68%** | 30-year age gap |
| CFP-FP | 7,000 | 97.44%¹ | Frontal vs. profile |
| CALFW | 6,000 | 96.07% | Cross-age |
| CPLFW | 6,000 | 94.47% | Cross-pose |
| **TinyFace** | 6,000 | **82.45%** | **Surveillance resolution, median 32×32 px** |

| Operating point | Value |
|---|---|
| Decision threshold | 0.2871 (FMR ≈ 0.1%) |
| TAR @ FAR=0.1%, clean | 96.03% |
| TAR @ FAR=0.1%, TinyFace | **33.13%** |
| Encode latency | 14.72 ms p50, 18.98 ms p99 |
| 1:1 verification | 31.04 ms p50 |

¹ Measured on a `cfp_fp.bin` variant that scores ~1.5 points below two other
copies of the same protocol; see [BENCHMARKS.md](BENCHMARKS.md) §2b.

**These figures were produced on public academic datasets by the same tooling
that built the system, and have not been independently validated.** Full
protocols, per-fold results, demographic breakdown and the threshold decision
record: [BENCHMARKS.md](BENCHMARKS.md).

---

## Limitations

Stated plainly because they change how results should be used.

- **Degraded footage is weak.** At FAR=0.1% on surveillance-resolution imagery
  the system finds roughly **one genuine match in three**. Results from such
  imagery are investigative leads for human review, never identifications.
- **Liveness detection is a heuristic, not anti-spoofing.** A passive
  single-frame texture and moiré check. It has not been evaluated against
  ISO/IEC 30107-3 and will not stop a printed photograph or a replayed screen.
  It reports `"certified": false` in its own output.
- **Synthetic-media screening is a heuristic, not a trained classifier.** An
  FFT artefact check. A well-post-processed synthetic face defeats it.
- **Error rates are not uniform across demographics.** At the deployed
  threshold, women are falsely rejected ~1.7× as often as men and the under-25
  group ~3.8× as often as the 41–55 group. See [BENCHMARKS.md](BENCHMARKS.md) §5.
- **No independent or NIST evaluation.** Not submitted to FRVT.
- **No custom-trained model.** This runs stock InsightFace ArcFace weights.
  Fine-tuning was attempted and abandoned: every available training archive
  overlaps the evaluation sets, so no gain could be trusted.
- **Not currently deployable as configured.** There is no hosted backend, and
  the frontend deploy config's CSP blocks a plain-HTTP API origin.
- **Concurrency is unmeasured.** All latency figures are single-threaded.

The full register — every known issue, fixed or open — is in
[SCORECARD.md](SCORECARD.md), and every product claim is mapped to the test that
backs it in [CLAIMS.md](CLAIMS.md).

---

## Running the tests

```bash
pytest backend/tests_engine/ backend/tests/     # 161 tests
python backend/scripts/regression_check.py      # accuracy + config regression gate
python scripts/verify_gpu.py                    # GPU binding, 12 checks
```

The regression gate compares measured accuracy against a recorded baseline
*and* asserts configuration invariants — deployed model pack, thresholds, and
that the API still derives its thresholds from the engine rather than holding a
copy. It exits non-zero on regression.

---

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

One rule matters more than the rest: **no accuracy claim without a measurement.**
If a change affects the model, thresholds, fusion, or the embedding pipeline,
run `regression_check.py` and include the numbers.

---

## License

[MIT](LICENSE).

The InsightFace model packs this project downloads at runtime carry their own
licences, which are **not** MIT and in some cases restrict commercial use.
Check them before deploying commercially.
