# NexGen Forensics — Face Recognition Engine

A commercial-grade facial recognition platform combining a FastAPI backend engine with a React frontend product interface (iMATCH). This document reflects the **verified current state** as of 2026-07-29. Claims not independently verified are explicitly marked.

---

## What is Running in Production Today

### Active Model Pipeline

The production service uses a **3-model InsightFace ensemble** with **embedding-space averaging** fusion.

All three models are loaded by `backend/nexgen_engine/models/insightface_backbone.py` via `BackboneEnsemble` in `backend/nexgen_engine/models/backbones.py`:

| # | Pack Name | Recognition Model | Architecture | Embedding Dim | Detector |
|---|-----------|-------------------|--------------|---------------|----------|
| 1 | `buffalo_l` | `w600k_r50.onnx` | ResNet-50 (ArcFace) | 512-d | SCRFD `det_10g.onnx` |
| 2 | `antelopev2` | `glintr100.onnx` | ResNet-100 (ArcFace) | 512-d | SCRFD `scrfd_10g_bnkps.onnx` |
| 3 | `buffalo_s` | `w600k_mbf.onnx` | MobileFaceNet | 512-d | SCRFD `det_500m.onnx` |

**Fusion method:** Embedding-space L2-normalized averaging over all 3 models' 512-d ArcFace embeddings. The fused embedding is the geometric mean direction in cosine-similarity space. No learned fusion layer is used.

**Runtime:** All three models run on `CPUExecutionProvider` (no GPU/CUDA drivers installed — CUDA DLL load fails gracefully and falls back to CPU).

**What `service.py` actually calls today:** `FacialRecognitionPipeline` → `BackboneEnsemble` → `InsightFaceEnsembleBackbone`. No fine-tuned weights are loaded in production; see Phase 3 note below.

---

## iMATCH API Endpoints

All endpoints live under `/api/biometrics/` — implemented in `backend/app/api/routes_biometrics.py`.

### 1:1 Verify — `POST /api/biometrics/verify`

Compares two face images and returns a cosine similarity score.

**Request:** `multipart/form-data` — fields `reference` (image), `probe` (image), `operator_id` (string, optional).

**Response:**
```json
{
  "status": "success",
  "score": 0.557,
  "label": "same_person",
  "verified": true,
  "review_required": true,
  "quality_ref": 0.6253,
  "quality_probe": 0.5808,
  "liveness_ref": 0.6516,
  "liveness_probe": 0.5875,
  "reasons_ref": ["liveness_below_threshold"],
  "reasons_probe": ["liveness_below_threshold"],
  "audit_hash": "...",
  "thresholds": { "same_person": 0.42, "inconclusive_low": 0.28 }
}
```

**Decision thresholds:**
- `score >= 0.42` → `same_person`
- `0.28 <= score < 0.42` → `inconclusive`
- `score < 0.28` → `different_person`

**Input validation:** Rejects non-image files (HTTP 422), blank/uniform images (pixel std dev < 5.0), and images > 160x160 with no SCRFD-detected face.

### 1:N Identify — `POST /api/biometrics/identify`

Encodes the probe face and searches an in-memory vector index for top-K candidates.

> **Note:** The in-memory index is empty at server start. You must enroll identities first via `/api/biometrics/enroll`. The index is **not persisted** between restarts.

### Batch 1:N Identify — `POST /api/biometrics/batch-identify`

Processes multiple probe images in a single request. Each file is independently validated and encoded.

**Request:** `multipart/form-data` — field `files[]` (multiple images), `top_k` (int, default 5).

### Enroll — `POST /api/biometrics/enroll`

Encodes a face image and stores it in the in-memory vector index under the given `identity_id`.

---

## Verified Benchmark Results

### Phase 1 — Single Model Baseline (`buffalo_l` / `w600k_r50`)

Benchmark script: `backend/scripts/benchmark_agedb_25.py`  
Test set: **AgeDB**, 25 identities, **686 probes**  
Log source: task-117.log (run 2026-07-29, verified exit code 0)

**1:N Closed-Set Identification (Rank-1):**

| Model | Rank-1 Accuracy | Correct / Total |
|-------|----------------|-----------------|
| buffalo_l (w600k_r50, R50) | **87.32%** | 599 / 686 |

**1:1 Verification (TAR @ FAR thresholds):**

| Threshold | TAR | FAR |
|-----------|-----|-----|
| >= 0.28 | 87.60% | 0.00% |
| >= 0.36 | 82.40% | 0.00% |
| >= 0.42 | 73.60% | 0.00% |

### Phase 2 — 3-Model Ensemble

Benchmark script: `backend/scripts/benchmark_ensemble.py`  
Same test set (25 identities, 686 probes on AgeDB).

> **UNVERIFIED:** The Phase 2 ensemble benchmark log did not contain a complete summary table — it ran against a live server that timed out before writing final aggregate stats. Per-probe rows were written but summary totals were not captured. Do not treat Phase 2 as having a confirmed accuracy number.

The live API integration test (2026-07-29) confirms the 3-model ensemble produces correct directional results:
- Same person (Maria Callas): score `0.557` → `same_person` ✓
- Different person (Callas vs Close): score `-0.047` → `different_person` ✓
- Non-face image: HTTP 422, "Blank or uniform image uploaded" ✓

### Phase 3 — Fine-Tuning (ArcFace / PyTorch)

Training script: `backend/nexgen_engine/training/train_pipeline.py`  
Loss: ArcFace (margin=0.5, scale=64)

**Sanity-check training run confirmed the loop runs** (task-200.log, 50 steps, batch=32, LR=1e-4):

| Step | ArcFace Loss | LR | Grad Norm |
|------|-------------|-----|-----------|
| 1 | 37.17 | 1.00e-4 | 99.7 |
| 10 | 37.41 | 1.00e-4 | 96.1 |
| 25 | 36.53 | 9.8e-5 | 90.8 |
| 50 | 36.51 | 9.1e-5 | 88.8 |

Loss was decreasing (37.2 → 35.7 by step 41+). Gradient norms stable at 82–107.

> **UNVERIFIED / NOT IN PRODUCTION:** Full fine-tuning did not complete. The loop crashed at epoch boundary due to a BatchNorm error on the last incomplete batch (batch size 1). No fine-tuned checkpoint was saved and the production service does NOT load fine-tuned weights. It uses the stock pretrained InsightFace ONNX models only.

---

## Model Files Required on Disk

InsightFace downloads model packs automatically to `~/.insightface/models/` on first use.

Expected paths (confirmed present):
```
~/.insightface/models/buffalo_l/
    1k3d68.onnx, 2d106det.onnx, det_10g.onnx, genderage.onnx, w600k_r50.onnx

~/.insightface/models/antelopev2/
    1k3d68.onnx, 2d106det.onnx, genderage.onnx, glintr100.onnx, scrfd_10g_bnkps.onnx

~/.insightface/models/buffalo_s/
    1k3d68.onnx, 2d106det.onnx, det_500m.onnx, genderage.onnx, w600k_mbf.onnx
```

If not present, any endpoint call will trigger automatic download (internet required).

---

## Setup & Run Instructions

### Prerequisites
- Python 3.11+
- Node.js 18+
- Windows (tested) or Linux

### Backend

```powershell
python -m venv .venv
.venv\Scripts\activate

# GPU host (recommended) -- installs and then ASSERTS CUDA actually bound
python scripts/setup_gpu.py

# CPU-only host
pip install -r backend/requirements.txt -r backend/requirements-engine.txt -r backend/requirements-cpu.txt

# Create the first operator account (required -- no account exists by default)
export NEXGEN_JWT_SECRET="$(python -c 'import secrets;print(secrets.token_urlsafe(64))')"
python backend/scripts/bootstrap_admin.py --password '<a strong password>' \
    --email 'investigator@your-org.example' --role admin

# Start the server
python -m uvicorn imatch_api.main:app --host 127.0.0.1 --port 8443 --app-dir backend
```

Server: `http://127.0.0.1:8443`
API docs: `http://127.0.0.1:8443/docs`

> **There is only one backend: `imatch_api`, on port 8443.** This is what
> `backend/deployment/Dockerfile` runs. Earlier revisions of this README
> documented `app.main:app` on port 8000; that package was a leftover from an
> unrelated product, could not start (its `core/config.py` was never
> committed), and has been quarantined to `backend/_deprecated_app/`. Do not
> restore it.

> `NEXGEN_JWT_SECRET` must be set. Without it the service generates an
> ephemeral per-process secret, so every token is invalidated on restart.

> First startup takes ~30–60 seconds while all 3 InsightFace model packs load into memory.

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

iMATCH UI: `http://localhost:5173`  
Face search console: `http://localhost:5173/face-search`

### Environment Variables

Copy `.env.example` to `.env` and fill in values. No API keys are required for local inference — all models run locally.

---

## Files Changed in This Commit

| File | Status | What Changed |
|------|--------|-------------|
| `backend/app/api/routes_biometrics.py` | NEW | Full biometrics API (verify, identify, batch-identify, enroll). Real face detection validation, audit logging. |
| `backend/nexgen_engine/models/insightface_backbone.py` | NEW | Real InsightFace 3-model ensemble. Embedding-space averaging fusion. |
| `backend/nexgen_engine/models/backbones.py` | MODIFIED | Delegates to `InsightFaceEnsembleBackbone` instead of deterministic stubs. |
| `backend/nexgen_engine/api/service.py` | MODIFIED | Real cosine similarity, quality/liveness scoring, threshold decisions. |
| `backend/nexgen_engine/training/train_pipeline.py` | NEW | ArcFace fine-tuning pipeline. Sanity-checked only; not in production. |
| `backend/nexgen_engine/training/arcface_loss.py` | NEW | ArcFace margin loss implementation. |
| `backend/nexgen_engine/training/dataset.py` | NEW | AgeDB/VGGFace data loader for training. |
| `frontend/src/services/imatchApi.js` | MODIFIED | Calls live `/api/biometrics/verify` and batch-identify. No mock computation. |
| `frontend/src/components/sections/FaceSearchExperience.jsx` | MODIFIED | 1:1 compare panel, batch upload panel, real result display. |
| `frontend/src/components/sections/FaceSearchExperience.css` | MODIFIED | New styles for compare/batch panels. |
| `backend/app/main.py` | MODIFIED | Registered new biometrics routes. |
| `backend/app/db/database.py` | MODIFIED | Minor database connection updates. |
| `backend/requirements.txt` | MODIFIED | Added insightface, onnxruntime, opencv-python, torch, torchvision. |
| `backend/scripts/benchmark_*.py` | NEW | Benchmark scripts for Phase 1 (single model), Phase 2 (ensemble), Phase 3 (fine-tuned). |

---

## What Is Not Yet Verified / Still Under Review

- **Phase 2 ensemble aggregate benchmark** — per-probe logs exist but summary stats (Rank-1, TAR) were not captured due to server timeout. Not independently confirmed.
- **Phase 3 fine-tuning** — Training loop runs and loss decreases over 50 steps, but full training did not complete. No fine-tuned checkpoint is used in production.
- **GPU inference** — CUDA provider fails to load on dev machine (no CUDA toolkit/drivers). All inference is CPU-only.
- **Index persistence** — The in-memory vector index is not persisted to disk. A restart clears all enrolled identities. A persistent store (FAISS + disk, Milvus, Qdrant) is needed for production use.
- **Security/privacy review** — Not conducted. No production deployment has been certified.
