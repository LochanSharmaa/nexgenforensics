# NexGen iMATCH

Facial recognition platform for forensic investigation: enrol subjects, search a
probe image against your gallery, adjudicate candidates as an examiner, and
produce a case report backed by a tamper-evident audit trail.

---

## What this system does, and what it does not

**It does:** rank enrolled subjects by visual similarity to a probe image, gate
out probes too poor to search, and record who searched for what, when, and on
what stated authority.

**It does not identify anyone.** A similarity score is not the probability that
two images show the same person. Every result is an investigative lead that a
qualified examiner must verify before it is relied upon. The API says so in
every response, the UI says so on every result, and the exported report says so
at the top. That is deliberate — the documented failures of face recognition in
investigations are overwhelmingly failures of a human treating a ranked
candidate as a conclusion.

Structural consequences of that position, which you should not "clean up":

- Only a human can set a candidate to `confirmed`. The engine has no code path
  that writes it.
- A high score that barely beats the runner-up on a large gallery is downgraded
  to `review_required`, because that is the signature of a false match.
- Every search records a lawful basis, and refuses to run without one.
- When the recognition model is missing, the service says so loudly rather than
  returning confident-looking numbers that mean nothing.

Accuracy depends on your imagery, your gallery size, and your threshold. Nothing
here has been independently benchmarked, and no accuracy figure is claimed.
Calibrate before you deploy — see [Calibration](#calibration).

---

## Requirements

| | Minimum | Notes |
|---|---|---|
| Python | 3.11 | Verified on 3.12.10 |
| Node.js | 20 | Verified on 24.18 (frontend only) |
| RAM | 8 GB | The ONNX session holds ~1 GB resident |
| Disk | ~2 GB | ~300 MB model pack, remainder dependencies |
| CPU | x86-64, 4 cores | Inference is CPU-bound by default |
| GPU | Optional | See the GPU note under [Performance](#performance) |
| Database | SQLite (dev) / PostgreSQL 14+ (production) | |

CPU-only is a fully supported configuration, not a degraded one: the model and
the arithmetic are identical, only the throughput differs.

---

## Quick start

### 1. Backend

```bash
cd backend
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt -r requirements-engine.txt
```

`requirements-engine.txt` is what makes recognition actually work. Without it
the service starts in a deterministic fallback mode that **cannot recognize
anyone** — see [Engine modes](#engine-modes).

Create your configuration:

```bash
copy ..\.env.example ..\.env
```

Generate the two secrets and paste them into `.env`:

```bash
python -c "import secrets; print('NEXGEN_JWT_SECRET=' + secrets.token_urlsafe(64))"
```

```bash
python -c "import base64,os; print('NEXGEN_TEMPLATE_KEY=' + base64.b64encode(os.urandom(32)).decode())"
```

> `NEXGEN_TEMPLATE_KEY` encrypts every stored biometric template. **Back it up.**
> Losing or changing it makes every enrolled template permanently unreadable, and
> there is no recovery path — that is what makes the encryption worth having.

Create the first tenant and administrator:

```bash
python scripts/seed.py
```

Start the API:

```bash
uvicorn imatch_api.main:app --host 0.0.0.0 --port 8443 --reload
```

First start downloads the InsightFace `buffalo_l` pack (~350 MB) into
`~/.insightface/models/`. Subsequent starts take a few seconds.

API docs: <http://localhost:8443/docs>

### 2. Frontend

```bash
cd frontend
npm install
```

```bash
npm run dev
```

Open <http://localhost:5173>. The dev server proxies `/api` to port 8443, so the
browser stays same-origin and no CORS grant is needed.

Sign in at `/login` with the credentials `seed.py` printed, then go to
`/workspace`.

### 3. First search

1. **Enrol** at least one subject (`/workspace/enrol`, supervisor role). A search
   against an empty gallery correctly returns nothing.
2. **Open a case** at `/workspace` and record its lawful basis.
3. **Search** at `/workspace/search` with a probe image.
4. **Adjudicate** each candidate — confirm, eliminate, or mark inconclusive.
5. **Export** the case report from the case page.

---

## Proving the engine works

Before trusting anything, run the end-to-end demonstration. It enrols a gallery,
searches a probe, and prints what actually happened:

```bash
cd backend
```

```bash
python test_recognition.py --self-test
```

Measured on this machine (Quadro M1200 host, CPU inference, `buffalo_l`,
25 AgeDB identities):

```
Model loaded          : YES
Recognition network   : w600k_r50
Embedding dimensions  : 512
Search backend        : FAISS IndexFlatIP (exact)

Rank-1 identification : 23/25 = 92.0%
Genuine  pairs        : mean=0.4907
Impostor pairs        : mean=0.0422
Separation (mean gap) : 0.4485

THRESHOLD   TAR       FAR
0.28        0.880     0.0033
0.36        0.800     0.0000
0.42        0.720     0.0000
```

AgeDB is deliberately hard — it varies age across decades. Those numbers are
what this build measured on that dataset, not a product accuracy claim.

Other modes:

```bash
python test_recognition.py --status
```

```bash
python test_recognition.py --enrol path/to/gallery --probe path/to/probe.jpg
```

```bash
python test_recognition.py --compare a.jpg b.jpg
```

## There is no fallback mode

The service loads real recognition weights or refuses to start. An earlier
revision degraded to a "deterministic stub" that hashed pixels into a vector; it
kept the API answering while making every score meaningless. That code is gone,
along with the centre-crop "detector" that asserted a face was present without
looking.

If the model cannot load you get `EngineUnavailableError` with the reason, and
no result. Check what is actually running:

```bash
curl -H "Authorization: Bearer $TOKEN" http://localhost:8443/api/imatch/engine/status
```

## Performance

Measured per image, CPU, `buffalo_l`, on the target hardware:

| Stage | Time |
|---|---|
| Decode | <1 ms |
| Detection (SCRFD) | ~54 ms |
| Alignment | ~2 ms |
| Embedding (ArcFace, flip-TTA) | ~260 ms |
| Search (FAISS, 567 templates) | ~5 ms |
| **Total** | **~320 ms** |

Two optimisations account for a 2.5× speed-up over the first working version:
calling the detection model directly rather than `FaceAnalysis.get()` (which
also ran recognition internally, costing 3× the necessary ArcFace work), and
sizing the detector input to the image instead of always using 640×640.

**GPU note for this hardware:** the Quadro M1200 is Maxwell (compute capability
5.0). The stock `onnxruntime` wheel is CPU-only, and current `onnxruntime-gpu`
builds do not ship sm_50 kernels, so this machine runs on CPU. Setting
`NEXGEN_ENGINE_DEVICE=cuda` is safe — the service checks whether the CUDA
provider is genuinely registered, logs what it chose, and reports the effective
device through the status endpoint. The device never changes the result.

---

## Calibration

The shipped thresholds (`match 0.42`, `review 0.32`) are generic ArcFace
operating points, not validated settings. The false-match rate at a fixed
threshold rises with gallery size and degrades with image quality, so a
threshold that is safe for 500 subjects can be badly wrong for 50,000.

Measure the genuine and impostor distributions on your own imagery:

```bash
python scripts/calibrate_threshold.py path/to/dataset --max-identities 500
```

The dataset is one directory per identity with two or more images each. The
script reports the threshold for each target false-match rate; pick from the FMR
your use of the system can tolerate, not from the equal error rate. In an
investigative context a false match points at the wrong person, so a stricter FMR
is usually correct even though it misses more true matches.

Set `NEXGEN_MATCH_THRESHOLD` accordingly, and keep `NEXGEN_REVIEW_THRESHOLD`
roughly 0.08–0.12 lower so borderline scores reach an examiner instead of being
silently dropped.

---

## Testing

```bash
cd backend
pip install -r requirements-dev.txt
```

```bash
pytest
```

The suite covers alignment maths, quality gating, template encryption, decision
logic, tenant isolation, auth, and audit-chain integrity — all without needing
the model.

`tests/test_recognition_accuracy.py` is the one that proves the engine does its
job: it measures genuine versus impostor score separation and rank-1 accuracy on
real faces. It skips unless both the model and a labelled dataset are present:

```bash
NEXGEN_ENGINE_MODE=auto NEXGEN_TEST_DATASET=/path/to/dataset pytest tests/test_recognition_accuracy.py -v
```

AgeDB under `src_extracted/` is picked up automatically. **Run this before
trusting a deployment** — every other test can pass while the system is
incapable of recognition.

---

## Layout

```
backend/
  nexgen_engine/        Pure recognition. No HTTP, database, or auth.
    detection/          RetinaFace detection, 5-point landmark alignment
    models/             ArcFace r100 recognizer + labelled stub fallback
    inference/          Pipeline, decision engine, score normalization
    search/             Tenant-partitioned gallery index
    security/           Template encryption, liveness/deepfake/morph screens
  imatch_api/           FastAPI service
    core/               Config, auth, RBAC, rate limiting
    db/                 SQLModel schema and session management
    api/routes/         auth, cases, subjects, search, audit, admin, reports
    services/           Engine lifecycle, audit chain, storage, reporting
  scripts/              seed.py, calibrate_threshold.py, dataset_cli.py
  tests/
frontend/
  src/workspace/        Investigator workspace (cases, search, verify, audit)
  src/components/       Public marketing site
  src/services/         API client
```

The engine package carries no service concerns, so it can be embedded,
benchmarked, and tested on its own.

---

## Security posture

- **Templates encrypted at rest** with AES-256-GCM under a per-tenant HKDF
  subkey, with the tenant id bound in as additional authenticated data. Moving a
  row between tenants fails to decrypt rather than silently matching.
- **Tenant isolation is structural.** Each tenant's vectors live in a separate
  matrix and `search()` requires a tenant id, so no code path can compare a probe
  against another tenant's gallery. A post-hoc filter would be one forgotten
  predicate away from a cross-tenant biometric leak.
- **Templates never leave the server.** An ArcFace embedding can be inverted into
  a recognizable approximation of the face, so it is treated as equivalent to the
  biometric itself.
- **Hash-chained audit trail**, mirrored to JSONL. Editing or deleting a record
  breaks verification from that point on. It proves the log has not been altered;
  it does not prove completeness, since someone with database access could
  truncate the tail. Ship the JSONL to write-once storage if you need that.
- **Server-side URL import is disabled.** Fetching a caller-supplied URL from
  inside the service is a server-side request forgery primitive that would reach
  internal addresses and cloud metadata endpoints.
- **Erasure is real.** Deleting a subject removes templates, images, and index
  entries outright. Search history is retained, because deleting it would destroy
  the audit trail.
- Argon2id password hashing, constant-time login regardless of whether an account
  exists, short-lived access tokens, hashed API keys shown once.

Not claimed: the liveness, deepfake, and morphing screens are heuristics, not
certified detection. None has been evaluated against ISO/IEC 30107-3, and a
determined attacker will defeat all three. They prioritise examiner attention;
they do not establish authenticity.

---

## Before production

- [ ] Set `NEXGEN_ENGINE_MODE=real` so a missing model stops the rollout
- [ ] Set and back up `NEXGEN_JWT_SECRET` and `NEXGEN_TEMPLATE_KEY`
- [ ] Calibrate thresholds on representative imagery
- [ ] Run `tests/test_recognition_accuracy.py` against real faces
- [ ] Move to PostgreSQL; SQLite will not survive concurrent load
- [ ] Terminate TLS at the ingress and restrict `NEXGEN_CORS_ORIGINS`
- [ ] Complete legal, privacy, and DPIA review for your jurisdiction
- [ ] Evaluate demographic performance differentials on your own population
- [ ] Define retention and deletion policy; set `NEXGEN_PROBE_RETENTION_DAYS`
- [ ] Ship audit JSONL to append-only storage
- [ ] Establish examiner training and an adjudication standard

The demographic point is not a formality. Face recognition error rates vary
across demographic groups, and a system deployed without measuring that on its
own population will distribute its errors unevenly across the people it is used
on.

---

## Docker

```bash
docker compose up --build
```

Requires `POSTGRES_PASSWORD`, `NEXGEN_JWT_SECRET`, and `NEXGEN_TEMPLATE_KEY` in
the environment or a `.env` file. Compose fails fast rather than starting with
defaults.

---

## Known limitations

Stated plainly, because a forensic tool that oversells itself is worse than one
that does less.

**Accuracy is unbenchmarked.** No NIST FRVT submission, no independent
evaluation. The numbers in this README were measured by this build on AgeDB and
are reproducible with `test_recognition.py --self-test` — they are not a product
accuracy claim, and they will not transfer to your imagery unchanged.

**Demographic performance is unmeasured.** Face recognition error rates are known
to vary across demographic groups. Nothing here measures that, so a deployment
that skips it will distribute its errors unevenly across the people it is used
on, without anyone noticing. Measure it on your own population.

**Liveness, deepfake, and morphing screens are heuristics.** Not evaluated
against ISO/IEC 30107-3. They look for cheap artefacts — texture collapse,
screen moiré, spectral checkerboarding — and a competent attacker defeats all
three. Use them to prioritise examiner attention, never to conclude that media
is authentic.

**Head-pose estimation is approximate.** Yaw, pitch and roll are derived
geometrically from five landmarks, not from a 3-D solver. Good enough to flag a
badly angled probe; not a measured pose value, and pitch is the weakest of the
three.

**The shipped model is ResNet50, not ResNet100.** `buffalo_l` carries
`w600k_r50`. The `antelopev2` pack carries `glintr100` (R100) and is selectable
via `NEXGEN_MODEL_PACK`, but this build has not benchmarked it; expect roughly
double the inference cost.

**Rate limiting is per process.** Behind multiple workers the effective limit
multiplies, and it resets on restart. It is a guard against runaway clients, not
a defence against a distributed attacker — put a shared store or an edge limiter
in front.

**Access tokens are stateless.** Revocation waits for expiry. Keep
`NEXGEN_ACCESS_TOKEN_MINUTES` short, or add a revocation store.

**The audit chain proves integrity, not completeness.** Editing or deleting a
record breaks verification from that point on, but someone with database access
could still truncate the most recent entries. Ship the mirrored JSONL to
append-only storage if you need that guarantee.

**PostgreSQL migrations are not yet wired.** The schema is created from the
SQLModel metadata at startup. SQLite is fine for development; production
deployments should add Alembic before the first schema change.

**Not implemented:** batch/CSV intake, server-side URL import (deliberately
disabled as an SSRF vector), and the dataset-level evaluation utility
(ROC/AUC/CMC curves).

---

## Licence

See [LICENSE](LICENSE).
