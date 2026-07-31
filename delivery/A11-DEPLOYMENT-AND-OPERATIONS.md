# A11 — Deployment and Operations

**Generated:** 2026-07-31 20:32 UTC · **Repository state:** `da66fad0d7f1`

How the delivered system is installed, configured, run and monitored, and what has and has not been validated in deployment.

---

## Validation status

**The system has not been validated in a live deployment.** All verification in
this package was performed locally. There is no hosted instance whose behaviour
under real traffic, real concurrency or real failure conditions has been
observed. This is stated first because every operational figure below should be
read with it in mind.

---

## Requirements

| Component | Requirement |
|---|---|
| Python | 3.11 |
| GPU (recommended) | NVIDIA with CUDA 12.x; ~2 GB VRAM for the recognition pack |
| GPU (verified on) | RTX A3000 Laptop / RTX 3060, `CUDAExecutionProvider` |
| CPU fallback | Supported and correct, but roughly an order of magnitude slower |
| Database | SQLite for single-node; PostgreSQL for multi-node |
| Disk | ~350 MB models, plus gallery templates |
| Memory | ~2 GB service, plus model residency |

**CPU and GPU results differ in the last decimal places** (0.494431 vs
0.494213 on a reference pair). This is ordinary floating-point divergence
between execution providers. An earlier claim that results were identical was
incorrect and has been corrected.

---

## Configuration

All configuration is by environment variable, prefixed `NEXGEN_` except where a
third party fixes the name.

| Variable | Purpose | Production requirement |
|---|---|---|
| `NEXGEN_JWT_SECRET` | Token signing | **Must be set.** Absent, an ephemeral key is generated and every restart invalidates all sessions |
| `NEXGEN_TEMPLATE_KEY` | Template encryption at rest | **Must be set**, or templates cannot be decrypted after restart |
| `NEXGEN_DATABASE_URL` | Database | `postgresql+psycopg://` for multi-node |
| `NEXGEN_CORS_ORIGINS` | Allowed browser origins | Never `*` with credentials |
| `NEXGEN_COOKIE_SECURE` | Secure cookie flag | **`true` behind HTTPS** |
| `NEXGEN_ENGINE_DEVICE` | `auto`, `cuda` or `cpu` | `auto` fails loudly if CUDA was expected and not bound |
| `NEXGEN_REQUIRE_LAWFUL_BASIS` | Governance enforcement | `true` |
| `NEXGEN_ALLOW_SELF_REGISTRATION` | Public signup | **`false`** unless deliberately opened |
| `RESEND_API_KEY` | Transactional e-mail | Backend only. Never expose to a browser bundle |

---

## Installation

```bash
python -m venv .venv && .venv/bin/pip install -r backend/requirements.txt
python scripts/setup_gpu.py            # GPU only; deconflicts onnxruntime builds
python scripts/verify_gpu.py           # 12 checks; refuse to proceed if it fails
python backend/scripts/migrate_auth_columns.py
python backend/scripts/bootstrap_admin.py --password '<strong password>'
python -m uvicorn imatch_api.main:app --host 0.0.0.0 --port 8443 --app-dir backend
```

`setup_gpu.py` exists because `insightface` installs a CPU `onnxruntime`
alongside `onnxruntime-gpu`, and the CPU build wins at import. Without it the
service runs on CPU while reporting no error at all — a failure that produced
correct results at a tenth of the speed and went unnoticed until throughput was
questioned.

---

## Operational limits

| Property | Measured value | Source |
|---|---|---|
| Thread concurrency | Saturates at ~4 workers (1.86x) | A5, concurrency |
| Request batching | 2.82x at batch 32 | A5, concurrency |
| Scaling lever | **Batching, not more workers** | The engine runs in-process |

---

## Monitoring

| Signal | Where |
|---|---|
| Liveness | `GET /api/health` |
| Engine status and bound provider | `GET /api/imatch/engine/status` |
| Latency percentiles | `LATENCY` collector, bounded ring buffer |
| Audit chain integrity | `GET /api/audit/verify` (admin) |
| Correlation | `X-Request-ID` on every response |

**Verify the bound execution provider after every deployment.** A service that
has silently fallen back to CPU is the highest-frequency operational failure
this system has had, and it does not announce itself.

---

## Backup and recovery

| Asset | Consequence of loss |
|---|---|
| Database | Gallery, cases and audit chain lost |
| `NEXGEN_TEMPLATE_KEY` | **Every enrolled template is permanently unrecoverable.** Back up separately from the database, or the backup and the key are lost together |
| `NEXGEN_JWT_SECRET` | All sessions invalidated; recoverable by re-authentication |
| Model weights | Re-downloadable; verify against the SHA-256 digests in A4 Part I |

---

## Deployment configuration as delivered

### `render.yaml`

```yaml
# Render blueprint for the NexGen iMATCH backend — FREE TIER (option B).
#
# CPU ONLY. Render offers no GPU instances, and this service does not need one:
# NEXGEN_ENGINE_DEVICE=cpu loads the same weights and applies the same
# algorithm.
#
# Results are NOT bit-identical to the GPU path, and an earlier version of this
# comment wrongly said they were. Measured on the same AgeDB pair: GPU 0.494431
# vs CPU 0.494213, a difference of 2.2e-4. That is ordinary CUDA-vs-CPU kernel
# arithmetic, and it is ~1300x smaller than the margin to the 0.2871 decision
# threshold, so no decision changes. But a similarity quoted in a report to four
# decimal places WILL differ depending on which device produced it, and for a
# forensic record that is worth knowing rather than assuming away.
#
# ---------------------------------------------------------------------------
# WHY THERE IS NO `disk:` BLOCK HERE
# ---------------------------------------------------------------------------
# Render's persistent disks attach only to PAID services; they are not offered
# on the free tier. Rather than pay for one, this configuration keeps all
# durable state in Postgres, which Render does offer free.
#
# That works because imatch_api/db/session.py branches on the URL scheme, so
# SQLite and Postgres are both supported. Two consequences follow, and both are
# real limitations rather than details:
#
#   1. PROBE IMAGES DO NOT SURVIVE A RESTART. StorageService writes uploaded
#      images to the local filesystem, which on a free instance is ephemeral.
#      Database rows survive -- cases, subjects, templates, the audit chain --
#      but the bytes at `probe_path` are gone after a redeploy or a spin-down.
#      A report can still be produced; the original image it references cannot
#      be retrieved. Acceptable for a demo, NOT acceptable for casework.
#
#   2. COLD STARTS ARE SLOW. Free services spin down when idle, and the first
#      request afterwards restarts the process and reloads the weights. Expect
#      50 s or more, which the Render dashboard also warns about.
#
#      It used to be worse: the pack was fetched inside the running service, so
#      a cold start re-downloaded 275 MB before serving anything. That is what
#      exhausted the 512 MB budget and produced
#      `Out of memory (used over 512Mi)` on 2026-07-31. The pack is now fetched
#      and trimmed at BUILD time by scripts/prefetch_models.py and read from
#      NEXGEN_MODEL_ROOT, so the serving process never holds the archive.
#
#      MEMORY, MEASURED FROM THE PACK ON DISK. buffalo_l ships 341.3 MB of
#      weights. InsightFace globs every *.onnx in the directory and builds an
#      InferenceSession for each one BEFORE discarding those outside
#      `allowed_modules`, so the unused 3D-landmark model (1k3d68.onnx,
#      143.6 MB) was being loaded and thrown away on every startup. The build
#      step deletes the three unused models, leaving 191.3 MB:
#
#        det_10g.onnx      16.9 MB  detection      KEPT
#        w600k_r50.onnx   174.4 MB  recognition    KEPT  (the deployed model)
#        1k3d68.onnx      143.6 MB  3D landmark    removed
#        2d106det.onnx      5.0 MB  2D landmark    removed
#        genderage.onnx     1.3 MB  age/gender     removed
#
#      No similarity score changes: the removed models are never consulted for
#      one. 191.3 MB of weights plus the Python/ONNX-Runtime baseline still
#      leaves little headroom in 512 MB. If this OOMs again, the fix is a
#      larger instance, NOT a smaller recognition model -- swapping w600k_r50
#      invalidates every figure in BENCHMARKS.md and CLAIMS.md.
#
# To remove both, move to a paid instance and add:
#
#   disk:
#     name: imatch-data
#     mountPath: /var/data
#     sizeGB: 1
#
# then set NEXGEN_STORAGE_ROOT and NEXGEN_MODEL_ROOT under /var/data.
#
# ---------------------------------------------------------------------------
# SECRETS
# ---------------------------------------------------------------------------
# NEXGEN_JWT_SECRET and NEXGEN_TEMPLATE_KEY are marked `sync: false`, so Render
# prompts for them in the dashboard and they are never committed here.
#
# LOSING NEXGEN_TEMPLATE_KEY MAKES EVERY STORED TEMPLATE PERMANENTLY
# UNDECRYPTABLE. There is no recovery path. Back it up before first deploy.

databases:
  - name: nexgen-imatch-db
    plan: free
    databaseName: imatch
    user: imatch
    # NOTE: Render's free Postgres instances expire. Check the current
    # retention period on the dashboard -- when it lapses the database is
    # removed and every enrolment, case and audit row goes with it. Treat this
    # deployment as a demo, and export anything that matters.

services:
  - type: web
    name: nexgen-imatch-api
    runtime: python
    plan: free
    region: oregon
    branch: main
    rootDir: .

    # TWO STEPS, and the second is not optional. insightface declares a
    # dependency on opencv-python (the GUI build), which pip installs ALONGSIDE
    # opencv-python-headless -- two complete OpenCV builds, 135 MB of wheels.
    # That duplication is what exceeded the free tier's 512 MB build limit.
    # Installing insightface with --no-deps suppresses it; every runtime
    # dependency it genuinely needs is in requirements-deploy.txt.
    buildCommand: |
      pip install --upgrade pip
      pip install -r backend/requirements-deploy.txt
      pip install --no-deps "insightface>=1.0.1,<2.0"
      python scripts/prefetch_models.py --root ./.insightface --pack buffalo_l

    startCommand: uvicorn imatch_api.main:app --host 0.0.0.0 --port $PORT --app-dir backend

    # Render polls this to decide the service is live. /api/health needs no
    # auth and does not touch the model, so it answers during warm-up.
    healthCheckPath: /api/health

    envVars:
      # --- secrets: Render prompts for these; never commit them ---
      - key: NEXGEN_JWT_SECRET
        sync: false
      - key: NEXGEN_TEMPLATE_KEY
        sync: false

      # --- database: wired automatically from the instance declared above ---
      # Render supplies a postgres:// URL. session.py rewrites the scheme to
      # postgresql+psycopg:// so psycopg 3 is used; pasting it unmodified is
      # correct and intended.
      - key: NEXGEN_DATABASE_URL
        fromDatabase:
          name: nexgen-imatch-db
          property: connectionString

      # --- runtime ---
      - key: NEXGEN_ENGINE_DEVICE
        value: cpu
      - key: NEXGEN_ENV
        value: production
      - key: PYTHON_VERSION
        value: "3.11"

      # Ephemeral on the free tier. Named explicitly so the path is visible
      # rather than defaulting somewhere surprising. See limitation 1 above.
      - key: NEXGEN_STORAGE_ROOT
        value: /tmp/imatch-storage

      # CORS: the deployed Vercel origin. Never a wildcard -- these endpoints
      # carry biometric data behind credentialed requests. This must stay in
      # step with vercel.json's `connect-src`, which names this service in turn;
      # both were shipped with REPLACE-WITH-... placeholders and the result was
      # a login that failed with "Failed to fetch" before a request was sent.
      - key: NEXGEN_CORS_ORIGINS
        value: https://nexgenforensics.vercel.app

      # Where prefetch_models.py put the weights at build time. Must match the
      # --root passed in buildCommand above, or the service re-downloads the
      # pack at startup and OOMs again.
      - key: NEXGEN_MODEL_ROOT
        value: ./.insightface

```

### `backend/requirements-deploy.txt`

```text
# Minimal dependency set to SERVE imatch_api. Used by render.yaml.
#
# Install in TWO steps — the second is not optional:
#
#   pip install -r backend/requirements-deploy.txt
#   pip install --no-deps "insightface>=1.0.1,<2.0"
#
# ---------------------------------------------------------------------------
# WHY THE SEPARATE --no-deps STEP FOR insightface
# ---------------------------------------------------------------------------
# Render's free tier has a hard 512 MB build limit and the full install blew
# through it. The single largest cause was a DUPLICATE OpenCV: `insightface`
# declares a dependency on `opencv-python` (73.8 MB wheel, GUI build), so pip
# installed it ALONGSIDE the `opencv-python-headless` (61.2 MB) this project
# actually wants. Two complete OpenCV builds, one of them pulling in GUI
# libraries a headless API server can never use.
#
# Installing insightface with --no-deps suppresses that. Everything it actually
# needs at runtime is listed below and resolved normally, with
# opencv-python-headless standing in for opencv-python.
#
# An earlier revision of this file applied --no-deps to EVERYTHING and pinned
# the whole transitive tree by hand. That broke immediately: FastAPI had added
# `annotated_doc`, which was not in the hand-written list. Scoping --no-deps to
# the one package that needs it keeps pip's resolver doing the work it is good
# at.
#
# ---------------------------------------------------------------------------
# REMOVED FROM THE DEPLOY SET, AND WHY IT IS SAFE HERE
# ---------------------------------------------------------------------------
#   opencv-python      73.8 MB  duplicate of headless. Substituted, not dropped.
#   faiss-cpu          19.2 MB  gallery_index.py guards the import and falls
#                               back to a numpy matmul. Measured 2.4x faster on
#                               50k templates, but never enabled in production
#                               (SCORECARD L10) and this gallery is far below
#                               the size where it matters.
#   alembic             ~2 MB   a build-time migration tool; the service calls
#                               SQLModel.metadata.create_all at startup.
#                               Verified nothing under imatch_api imports it.
#   httpx               ~1 MB   verified: not imported by imatch_api.
#   sentry-sdk[fastapi] ~0.5MB  verified: not imported by imatch_api. Re-add
#                               deliberately if error reporting is wanted.
#   bcrypt              ~1 MB   verified unused; core/security.py hashes with
#                               Argon2id via argon2-cffi.
#   torch / torchvision  ~2 GB  never in any deploy path. Imported only by the
#                               training pipeline and by the CUDA probe in
#                               runtime.py, which short-circuits when
#                               onnxruntime reports no CUDA provider. Every
#                               such import is already guarded.
#
# KEPT, THOUGH IT MAY LOOK REMOVABLE
#   scikit-image + scipy  51 MB insightface's face_align.py uses
#                               skimage.transform.SimilarityTransform for the
#                               5-point alignment — the hot path of every
#                               encode. Removing it breaks recognition itself,
#                               not merely a script.
#   onnx                  19 MB imported directly by insightface's
#                               arcface_onnx.py.
#   reportlab              2 MB forensic PDF export.
#   psycopg[binary]             Postgres driver; the free tier has no disk for
#                               SQLite. See render.yaml.
# ---------------------------------------------------------------------------

# --- web framework ---
fastapi>=0.115.0
uvicorn[standard]>=0.30.1
python-multipart>=0.0.20
python-dotenv>=1.0.1

# --- data / config ---
sqlmodel>=0.0.21
pydantic[email]>=2.9.0
pydantic-settings>=2.5.0

# --- auth / crypto ---
pyjwt>=2.8.0
argon2-cffi>=23.1.0
cryptography>=44.0.0

# --- database ---
psycopg[binary]>=3.1

# --- recognition engine ---
# insightface is NOT listed here on purpose; install it separately with
# --no-deps, per the header. These are the deps it actually needs at runtime.
onnxruntime>=1.20.0,<1.21
onnx>=1.16
opencv-python-headless>=4.10.0
scikit-image>=0.24.0
numpy>=1.26.4,<3.0
pillow>=11.1.0
tqdm
requests

# --- reporting ---
reportlab>=4.0

```

### `vercel.json`

```json
{
  "framework": "vite",
  "installCommand": "cd frontend && npm install",
  "buildCommand": "cd frontend && npm run build",
  "outputDirectory": "frontend/dist",
  "build": {
    "env": {
      "VITE_IMATCH_API_BASE": "https://nexgen-imatch-api.onrender.com"
    }
  },
  "headers": [
    {
      "source": "/(.*)",
      "headers": [
        {
          "key": "Strict-Transport-Security",
          "value": "max-age=63072000; includeSubDomains; preload"
        },
        {
          "key": "X-Content-Type-Options",
          "value": "nosniff"
        },
        {
          "key": "X-Frame-Options",
          "value": "DENY"
        },
        {
          "key": "Referrer-Policy",
          "value": "strict-origin-when-cross-origin"
        },
        {
          "key": "Permissions-Policy",
          "value": "camera=(), microphone=(), geolocation=(), payment=(), usb=(), browsing-topics=()"
        },
        {
          "key": "Content-Security-Policy",
          "value": "default-src 'self'; base-uri 'self'; object-src 'none'; frame-ancestors 'none'; script-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self' https://nexgen-imatch-api.onrender.com; form-action 'self'; upgrade-insecure-requests"
        }
      ]
    }
  ],
  "rewrites": [
    {
      "source": "/((?!api/).*)",
      "destination": "/index.html"
    }
  ]
}

```

