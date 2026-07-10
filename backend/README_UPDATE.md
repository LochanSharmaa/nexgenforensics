# NexGen Backend Update

## What This Backend Provides

The backend is a FastAPI service for consent-aware biometric demo workflows and production-oriented facial recognition engine integration. It now keeps the local fallback path runnable while exposing explicit hooks for production storage, Redis caching, encrypted template persistence, index persistence, and operational health reporting.

## Main Entrypoint

Run the API with:

```powershell
cd C:\Users\locha\Desktop\nexgenforensics\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8443 --reload
```

Health:

```text
GET http://127.0.0.1:8443/api/v1/health
```

The health response includes service status, configured database backend, Redis configuration status, and current index count.

## API Surface

```text
GET  /api/v1/health
GET  /metrics
POST /api/imatch/search
POST /api/v1/engine/enroll
POST /api/v1/engine/identify
POST /api/v1/auth/register
POST /api/v1/auth/login
GET  /api/v1/auth/me
POST /api/v1/auth/users
POST /api/v1/jobs
GET  /api/v1/jobs/{job_id}
```

## Configuration

Important environment variables:

```text
NEXGEN_ENV=local
NEXGEN_SECRET_KEY=change-me-use-a-strong-secret
NEXGEN_DATABASE_URL=sqlite:///runtime/nexgen.db
NEXGEN_REDIS_URL=redis://localhost:6379/0
NEXGEN_RUNTIME_DIR=runtime
NEXGEN_FAISS_INDEX_PATH=runtime/faiss.index
NEXGEN_MODEL_DIR=models
NEXGEN_CHECKPOINT_PATH=runtime/checkpoints
NEXGEN_UPLOAD_PATH=runtime/uploads
NEXGEN_ENCRYPTION_KEY=
NEXGEN_DEVICE=cpu
NEXGEN_PRECISION=fp32
NEXGEN_POSTGRES_POOL_MIN=5
NEXGEN_POSTGRES_POOL_MAX=20
NEXGEN_REDIS_POOL_SIZE=10
NEXGEN_MAX_UPLOAD_SIZE_GB=50
NEXGEN_LIVENESS_THRESHOLD=0.75
NEXGEN_DEEPFAKE_THRESHOLD=0.85
NEXGEN_MORPHING_THRESHOLD=0.30
NEXGEN_QUALITY_THRESHOLD=0.60
NEXGEN_MATCH_THRESHOLD=0.65
```

## Storage

Local mode uses SQLite through `Database`. The schema includes tenants, users, enrolled identities, encrypted biometric templates, audit logs, jobs, model versions, index versions, agency partitions, model accuracy history, auto-improve jobs, and dataset uploads.

Production Postgres support is exposed through `AsyncDatabase` for `postgresql+asyncpg://` URLs. Production deployment still needs migration orchestration, secret management, and service-level wiring around the async session lifecycle.

## Search Index

`OptionalFaissIndex` attempts to use FAISS when available. If FAISS is not installed, it uses the NumPy vector index. Both modes support persistence through `NEXGEN_FAISS_INDEX_PATH`.

Local fallback behavior is deliberate: it keeps development and tests runnable without CUDA or FAISS GPU. It should not be represented as GPU IVF-PQ production search.

## Jobs And Redis

The local job queue stores jobs in the database and tracks priority/progress. `RedisClient` provides async Redis cache, hash, pub/sub, and sliding-window rate-limit helpers with an in-memory fallback for local development.

Production Celery/Redis workers can be layered on top of these boundaries, but the local API does not require a running Redis server.

## Model Weights

The current code can run with deterministic fallback embeddings when heavy model dependencies or checkpoints are unavailable. Real production recognition requires:

- validated trained weights for each backbone
- detector and alignment model assets
- FaceQNet/deepfake/morphing/liveness checkpoints
- CUDA-compatible PyTorch runtime
- FAISS CPU or GPU runtime
- independent benchmark validation on approved datasets

Do not claim production biometric accuracy from fallback embeddings.

## Validation

Run:

```powershell
cd C:\Users\locha\Desktop\nexgenforensics\backend
$env:PYTHONPATH = (Get-Location).Path
python scripts/validate_system.py
```

Targeted tests:

```powershell
pytest tests
```

## Production Checklist

Before live biometric use:

- replace fallback embeddings with trained and versioned model checkpoints
- configure Postgres with migrations and backups
- run Redis/Celery workers under supervision
- configure TLS, secrets, retention, deletion, audit review, and incident response
- run independent accuracy, demographic performance, spoofing, and security assessments
- document lawful basis and customer-specific data handling policies

