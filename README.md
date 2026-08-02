<div align="center">

# NexGen iMATCH

**Forensic face recognition with auditable 1:1 verification and 1:N gallery search — built so every accuracy claim is measured, separated by operating condition, and reproducible from the command line.**

[![CI](https://github.com/LochanSharmaa/nexgenforensics/actions/workflows/ci.yml/badge.svg)](https://github.com/LochanSharmaa/nexgenforensics/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests 161](https://img.shields.io/badge/tests-161%20passing-brightgreen.svg)](#testing-and-verification)
[![LFW 99.78%](https://img.shields.io/badge/LFW%201%3A1-99.78%25-informational.svg)](BENCHMARKS.md)
[![TinyFace 82.45%](https://img.shields.io/badge/TinyFace-82.45%25-orange.svg)](BENCHMARKS.md)

[Benchmarks](BENCHMARKS.md) · [Claims ↔ Code](CLAIMS.md) · [Scorecard](SCORECARD.md) · [Delivery Package](delivery/) · [Roadmap](ROADMAP.md)

</div>

---

## Status

| | |
|---|---|
| **Version** | 1.0.0 · package issued 2026-08-01 |
| **Maturity** | Working system, locally verified. **Not deployed, not independently validated.** |
| **Deployed model** | `buffalo_l` / `w600k_r50` (stock InsightFace ArcFace weights) |
| **Operating point** | Verification threshold 0.2871 (FMR ≈ 0.1%) |
| **Verification** | 161 tests passing · accuracy + configuration regression gate passing · GPU binding 12/12 |

This project's distinguishing property is not its accuracy — that is the
open-source state of the art, because it *is* the open-source state of the art.
It is that the claims are measured, separated by operating condition, published
with their protocol, and corrected in public when found wrong. Negative results
are reported as negative.

---

## Contents

- [Demo](#demo)
- [Why this exists](#why-this-exists)
- [Capabilities](#capabilities)
- [Quick start](#quick-start)
- [Architecture](#architecture)
- [Benchmarks](#benchmarks)
- [Limitations](#limitations)
- [Responsible use](#responsible-use)
- [Documentation](#documentation)
- [Testing and verification](#testing-and-verification)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

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
does not survive contact with real evidence. Clean web photographs and
surveillance stills differ here by **more than seventeen accuracy points**, and
averaging them hides exactly the case that matters to an investigator.

This project separates those numbers, records the protocol behind each one, and
labels every component that is a heuristic rather than a trained model. It is
built for investigative and research use where a result may have to be explained
to someone who did not write it — and defended by someone who did not build it.

---

## Capabilities

| Capability | Implementation |
|---|---|
| **1:1 verification** | Cosine similarity on L2-normalised ArcFace templates, calibrated threshold, plain-language explanation with every result |
| **1:N gallery search** | Ranked candidates, subject-collapsed so one well-enrolled person cannot fill the list |
| **Batch processing** | Three modes — one reference vs. many probes, many independent pairs, many probes vs. gallery. One unreadable file does not fail the batch |
| **Forensic reporting** | PDF, JSON and Markdown export: case header, per-search findings, chain-of-custody hashes, examiner sign-off, investigative-lead notice on every page |
| **Hash-chained audit trail** | Append-only; each record's hash covers its content plus the previous hash. `GET /api/audit/verify` re-walks and validates the chain |
| **Tenant isolation** | Enforced structurally — each tenant's vectors live in a separate shard and `search()` *requires* a tenant id, so no code path can compare across tenants |
| **Encrypted templates at rest** | AES-256-GCM per row. Authenticated, so a wrong key raises rather than decrypting to plausible noise |
| **Quality / capture screening** | Reported alongside every result, explicitly labelled as heuristics — see [Limitations](#limitations) |
| **Verified GPU execution** | Asserts the provider an ONNX session *actually bound*, not the one the wheel was compiled with. Raises rather than silently running 20× slower on CPU |

---

## Quick start

Requires Python 3.11+, Node 18+, and an NVIDIA GPU with CUDA 12.x for GPU
inference (CPU works, more slowly).

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
in one namespace and whichever unpacks last wins — often the CPU build,
silently, at roughly 20× the latency. `setup_gpu.py` installs, removes the CPU
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
secret and every token dies on restart. Without `NEXGEN_TEMPLATE_KEY` biometric
templates are stored **unencrypted**, and the service says so loudly at startup.

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

**6. Verify the install.**

```bash
python scripts/verify_gpu.py
```

12 checks; fails loudly if inference is silently running on CPU.

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
`nexgen_engine/config.py::ThresholdConfig`. The API derives from it, and a
[regression gate](#testing-and-verification) fails if a second copy ever
reappears — it did once, in four places, and they had drifted.

Computed module reachability and the full folder map:
[A3 — System Architecture](delivery/A3-SYSTEM-ARCHITECTURE.md) ·
[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md).

---

## Benchmarks

1:1 verification, deployed model `w600k_r50`, published pair protocols,
10-fold cross-validation with the threshold fitted on 9 folds and applied to
the held-out fold.

| Dataset | Pairs | Accuracy | What it stresses |
|---|---:|---|---|
| LFW | 6,000 | **99.78%** | Frontal, unconstrained |
| CFP-FF | 7,000 | 99.91% | Frontal–frontal control |
| CFP-FP | 7,000 | 97.44%¹ | Frontal vs. profile |
| AgeDB-30 | 6,000 | **96.68%** | 30-year age gap |
| CALFW | 6,000 | 96.07% | Cross-age |
| CPLFW | 6,000 | 94.47% | Cross-pose |
| **TinyFace** | 6,000 | **82.45%** | **Surveillance resolution, median 32×32 px** |

### Operating point and throughput

| Metric | Value |
|---|---|
| Decision threshold | 0.2871 (FMR ≈ 0.1%) |
| TAR @ FAR=0.1%, clean | 96.03% |
| TAR @ FAR=0.1%, TinyFace | **33.13%** |
| Encode latency | 14.72 ms p50 · 18.98 ms p99 |
| 1:1 verification | 31.04 ms p50 |
| Concurrency, threading | 131.9/s at 4 workers (1.86×); 8 workers gains nothing and degrades p99 2.3× |
| Concurrency, request batching | **551.2/s (2.82×)** without the latency cost |

¹ Measured on a `cfp_fp.bin` variant that scores ~1.5 points below two other
copies of the same protocol — a pack-provenance artefact, not an accuracy
finding. Rankings are unaffected. See [BENCHMARKS.md](BENCHMARKS.md) §2b.

### Measured, not yet shipped

**Quality-routed model selection** (BENCHMARKS.md §6f) routes each comparison to
a degradation specialist when image quality falls below 0.539 — a threshold
derived from QMUL-SurvFace and CASIA-WebFace quality distributions only, both
disjoint from all seven reporting benchmarks, and fixed *before* the measurement
rather than after it.

| Dataset | Deployed TAR@FAR0.1% | Routed | Δ |
|---|---|---|---|
| **TinyFace** | 33.13% | **37.37%** | **+4.23pp** |
| LFW | 99.70% | 99.70% | 0.00 |
| CPLFW (worst clean cost) | 87.40% | 87.27% | −0.13pp |

Slated for 1:1 in the next phase; 1:N stays single-model until embedding-space
compatibility is verified at gallery scale. See [ROADMAP.md](ROADMAP.md) §9.

> **These figures were produced on public academic datasets by the same person
> and tooling that built the system, and have not been independently
> validated.** Full protocols, 350 per-fold entries, demographic breakdown and
> the threshold decision record: [BENCHMARKS.md](BENCHMARKS.md) and
> [A5 — Benchmark Record](delivery/A5-BENCHMARK-RECORD.md).

---

## Limitations

Stated plainly because they change how results should be used.

- **Degraded footage is weak, and measurably worse than earlier drafts of this
  file said.** Re-measured on official protocol splits against protocol-defined
  reference populations (2026-08-02): at FAR=0.1% the system finds **one genuine
  match in five** on TinyFace (20.59%) and **one in forty-three** on
  QMUL-SurvFace (2.31%). The earlier "one in three" came from a self-constructed
  pair list, 39.4% of which paired near-duplicate frames from the same
  surveillance track.
- **On real surveillance imagery it cannot reject strangers.** Against 1,844
  genuinely unenrolled probes from QMUL's official open-set split, true-positive
  identification rate at a 1% false-alarm rate is **0.00%**. This is the number
  that matters operationally, and nothing in the evidence layer improves it —
  calibration is already within 0.6% of optimal. See
  [docs/MEASUREMENT_RECORD.md](docs/MEASUREMENT_RECORD.md).
- **Not independently validated.** Not submitted to NIST FRTE. Every number
  here is internal. This is the single largest credibility gap and no internal
  work closes it — see [ROADMAP.md](ROADMAP.md).
- **Error rates are not uniform across demographics.** At the deployed
  threshold, women are falsely rejected ~1.7× as often as men, and the under-25
  group ~3.8× as often as the 41–55 group. Raising the threshold *relocated*
  these errors without removing the gap. See [BENCHMARKS.md](BENCHMARKS.md) §5.
- **Liveness detection is a heuristic, not anti-spoofing.** A passive
  single-frame texture and moiré check, not evaluated against ISO/IEC 30107-3.
  It will not stop a printed photograph or a replayed screen, and reports
  `"certified": false` in its own output.
- **Synthetic-media screening is a heuristic, not a trained classifier.** An FFT
  artefact check. A well-post-processed synthetic face defeats it.
- **No custom-trained model.** This runs stock InsightFace ArcFace weights.
  Fine-tuning was run end to end, properly: train/eval contamination was
  measured and removed (692 of 10,572 CASIA identities excluded, §6c), then the
  model was fine-tuned from ArcFace weights on degraded data. **It scored worse
  on every benchmark** — worst on TinyFace, 82.45% → 79.38%, with TAR@FAR0.1%
  falling 33.13% → 22.23%. The deployed model is unchanged and the negative
  result is published in full (§6d). The evidence points at synthetic
  degradation not matching real low-resolution capture.
- **Not currently deployable as configured.** There is no hosted backend, and
  the frontend deploy config's CSP blocks a plain-HTTP API origin.
- **Concurrency measured at the engine, not the full HTTP stack.** The figures
  above are engine-level; end-to-end request throughput is still unmeasured.

The complete register — every known issue, fixed or open, including the ones
found *after* the original audit — is in [SCORECARD.md](SCORECARD.md). Every
product claim is mapped to the test that backs it in [CLAIMS.md](CLAIMS.md).

---

## Responsible use

This is biometric identification software. It processes data that is
irrevocable — a person cannot reissue their face.

- **Outputs are investigative leads, not identifications.** Every response
  carries that notice. A qualified examiner must verify any candidate before it
  is relied upon, and the system is designed to make skipping that step visible
  in the audit trail rather than convenient.
- **Lawful basis is a required field**, not an optional one, on verification and
  search requests. It is recorded in the audit chain.
- **Known demographic differentials are published above.** They must be
  disclosed to anyone relying on a result, not discovered later.
- **Degraded-imagery results carry an error rate of roughly two in three
  missed genuine matches** at the defensible operating point. Presenting such a
  result as an identification would misrepresent the system.
- Deployment in a jurisdiction regulating biometric processing (GDPR, BIPA, and
  equivalents) is the deployer's responsibility. The audit chain, retention
  controls and template encryption exist to support compliance; they do not
  constitute it.

---

## Documentation

### Next-generation architecture programme

| Document | What it is |
|---|---|
| [NEXTGEN-ARCHITECTURE.md](NEXTGEN-ARCHITECTURE.md) | The proposed successor architecture and the assumptions it challenges |
| [IMPLEMENTATION-PLAN.md](IMPLEMENTATION-PLAN.md) | Stage-by-stage execution plan with success and failure criteria |
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | **Proven / hypothesis / unknown**, with the commands that reproduce each figure |
| [docs/CAPACITY_VALIDATION.md](docs/CAPACITY_VALIDATION.md) | Which capacity measurements are valid, which are invalid, and why |
| [docs/DATASET_INVENTORY.md](docs/DATASET_INVENTORY.md) | Every dataset held, with protocol compatibility and missing metadata |
| [DATA_REQUIREMENTS.md](DATA_REQUIREMENTS.md) | Datasets requested for download, ranked by what they unblock |
| [GPU_EXECUTION_REQUEST.md](GPU_EXECUTION_REQUEST.md) | GPU tasks awaiting approval, with runtimes and the decisions they enable |


### Core records

| Document | Contents |
|---|---|
| [BENCHMARKS.md](BENCHMARKS.md) | Every measurement, protocol and decision record — including negative results |
| [CLAIMS.md](CLAIMS.md) | Each user-visible claim mapped to the code and test that proves it, or marked unbacked |
| [SCORECARD.md](SCORECARD.md) | Complete issue register: original findings, everything found since, and open limitations |
| [ROADMAP.md](ROADMAP.md) | Phase 8 onward — NIST FRTE readiness, degraded-imagery contribution, product completeness |
| [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) | Folder map, module reference, data flows, entry points |

### Delivery package

Written for external review — [`delivery/`](delivery/):

| | | | |
|---|---|---|---|
| [A1](delivery/A1-DEVELOPMENT-HISTORY.md) Development History | [A2](delivery/A2-FAILURE-AND-RECOVERY-LOG.md) Failure & Recovery Log | [A3](delivery/A3-SYSTEM-ARCHITECTURE.md) System Architecture | [A4](delivery/A4-MODEL-AND-TRAINING-RECORD.md) Model & Training Record |
| [A5](delivery/A5-BENCHMARK-RECORD.md) Benchmark Record | [A6](delivery/A6-DATASET-PROVENANCE.md) Dataset Provenance | [A7](delivery/A7-THRESHOLD-CALIBRATION-RECORD.md) Threshold Calibration | [A8](delivery/A8-API-SPECIFICATION.md) API Specification |
| [A9](delivery/A9-TEST-SUITE-CATALOGUE.md) Test Suite Catalogue | [A10](delivery/A10-SECURITY-ARCHITECTURE.md) Security Architecture | [A11](delivery/A11-DEPLOYMENT-AND-OPERATIONS.md) Deployment & Operations | [04](delivery/04-MODEL-CARD.md) Model Card |

Plus [A2 — Appendices](delivery/A2-APPENDICES.md) and the
[package index](delivery/README.md).

The package's governing principle: *no performance figure appears anywhere in it
that was not measured on this system, on a named dataset, under a stated
protocol.* Sections that cannot yet be written truthfully are marked **NOT YET
DELIVERABLE** with the reason, rather than filled with plausible text.

### Operator guides

[Installation](docs/installation.md) · [Configuration](docs/configuration.md) ·
[Deployment](docs/deployment.md) · [Architecture](docs/architecture.md) ·
[Models](docs/models.md) · [Security](docs/security.md) ·
[Testing](docs/testing.md) · [Troubleshooting](docs/troubleshooting.md)

---

## Testing and verification

```bash
pytest backend/tests_engine/ backend/tests/     # 288 tests
python backend/scripts/regression_check.py      # accuracy + config regression gate
python scripts/verify_gpu.py                    # GPU binding, 12 checks
```

### Forensic evidence layer (CPU only, no model load)

```bash
python backend/scripts/audit_assets.py              # dataset inventory
python backend/scripts/evaluate_baseline.py         # ROC/DET/CMC/Cllr/open-set
python backend/scripts/validate_capacity.py         # which capacity numbers are valid
python experiments/S0_3/run.py --embedder stub      # S0.3 pipeline check
```

These run entirely from cached embeddings — no GPU, no inference. They produce
[docs/METHODOLOGY.md](docs/METHODOLOGY.md)'s measured figures, and
[docs/CAPACITY_VALIDATION.md](docs/CAPACITY_VALIDATION.md) records which
measurements are trustworthy and which the system refuses to produce.

The regression gate compares measured accuracy against a recorded baseline
*and* asserts configuration invariants — deployed model pack, thresholds, and
that the API still derives its thresholds from the engine rather than holding a
copy. It exits non-zero on regression.

**What the CI badge attests.** CI runs the subset of the 288 tests — everything that
does not need the InsightFace model pack — plus the configuration half of the
regression gate. It verifies that the code imports, the logic is sound, and the
decision thresholds have not drifted. It does **not** re-measure accuracy: that
needs a ~350 MB model pack and a ~595 MB embedding cache, neither of which is in
the repository. The accuracy badges are static values reproduced by the commands
in [Benchmarks](#benchmarks). The tests that exercise the real recognition
pipeline must be run locally before release.

Test-by-test inventory: [A9 — Test Suite Catalogue](delivery/A9-TEST-SUITE-CATALOGUE.md).

---

## Roadmap

The next phase is ordered around closing the credibility gap rather than adding
features: CPU-path verification, NIST FRTE readiness, shipping the measured
quality-routing win, and an original contribution on degraded imagery. Product
completeness — video/CCTV ingestion, the digital-forensics evidence layer,
explainable AI — runs in parallel and is not gated on the evaluation cycle.

Full plan, including what was deliberately rejected and why:
[ROADMAP.md](ROADMAP.md).

---

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

One rule matters more than the rest: **no accuracy claim without a
measurement.** If a change affects the model, thresholds, fusion, or the
embedding pipeline, run `regression_check.py`, re-run the CLAIMS ↔ BENCHMARKS
cross-check, and include the numbers.

---

## Citation

```bibtex
@software{nexgen_imatch_2026,
  title  = {NexGen iMATCH: Auditable Forensic Face Recognition},
  author = {Sharma, Lochan},
  year   = {2026},
  url    = {https://github.com/LochanSharmaa/nexgenforensics},
  note   = {Version 1.0.0}
}
```

---

## License

[MIT](LICENSE).

The InsightFace model packs this project downloads at runtime carry their own
licences, which are **not** MIT and in some cases restrict commercial use.
Several evaluation datasets referenced in [BENCHMARKS.md](BENCHMARKS.md) are
research-use-only. Check both before deploying commercially — see
[A6 — Dataset Provenance](delivery/A6-DATASET-PROVENANCE.md).
