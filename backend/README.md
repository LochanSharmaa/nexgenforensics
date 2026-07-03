# NexGen Identity AI Service

This backend powers the first AI layer for the NexGen commercial biometric platform. It exposes an iMatch-compatible endpoint for consent-aware demo analysis:

- image quality scoring
- liveness scoring
- eight-backbone ensemble abstraction with deterministic local fallback
- TTA embedding fusion and cohort normalization
- tenant-scoped enrollment and identification
- template encryption and immutable audit logging
- metric-learning losses for AdaFace, ElasticFace, CosFace, Triplet, and Uniformity
- four-stage curriculum utilities, hard-negative mining, and benchmark reporting
- Docker, Compose, Kubernetes, and hybrid deployment configuration
- configuration files for model, training, inference, security, and deployment settings
- dataset loading, synthetic augmentation, dataset preparation, and face alignment helpers
- index persistence, sharded search, API middleware, and reusable route wiring
- ONNX/TensorRT export manifests, client packaging, analytics, and API route tests
- optional Torch/timm and FAISS integration hooks with runtime capability reporting
- audit metadata for every request
- benchmark-target wording without unvalidated accuracy claims

Run locally:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8443 --reload
```

Health check:

```text
GET http://127.0.0.1:8443/api/v1/health
```

iMatch endpoint used by the frontend:

```text
POST http://127.0.0.1:8443/api/imatch/search
```

Engine endpoints:

```text
POST http://127.0.0.1:8443/api/v1/engine/enroll
POST http://127.0.0.1:8443/api/v1/engine/identify
```

Run the local validation suite:

```powershell
cd backend
$env:PYTHONPATH = (Get-Location).Path
python scripts/validate_system.py
```

Prepare dataset handoff:

```powershell
cd backend
$env:PYTHONPATH = (Get-Location).Path
python scripts/dataset_cli.py template dataset_manifest.csv
python scripts/dataset_cli.py validate --root C:\path\to\dataset --manifest C:\path\to\dataset_manifest.csv
```

Check optional production dependency status:

```powershell
cd backend
$env:PYTHONPATH = (Get-Location).Path
python -m nexgen_engine.cli capabilities
```

This is a responsible demo service. Production deployment still needs real model training, independent benchmark validation, legal review, privacy review, security testing, and customer-specific compliance sign-off.
