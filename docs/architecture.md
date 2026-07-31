# Architecture

[← Back to README](../README.md)

## Layout

```
backend/
  nexgen_engine/          Pure recognition. No HTTP, database, or auth.
    detection/            SCRFD detection, 5-point landmark alignment
    models/               ArcFace recognizer, model-pack registry
    inference/            Pipeline, decision engine, score normalization
    search/               Tenant-partitioned gallery index (FAISS + numpy)
    security/             Template encryption, liveness/deepfake/morph screens
    data/                 Quality filter, dataset manifest, augmentation
  imatch_api/             FastAPI service
    core/                 Config, auth, RBAC, rate limiting
    db/                   SQLModel schema and session management
    api/routes/           auth, cases, subjects, search, audit, admin, reports
    services/             Engine lifecycle, audit chain, storage, reporting
  scripts/                seed.py, calibrate_threshold_suite.py (full suite),
                          calibrate_threshold.py (folders only), dataset_cli.py
  tests/
  test_recognition.py     End-to-end CLI demonstration
frontend/
  src/workspace/          Investigator workspace (cases, search, verify, audit)
  src/components/         Public marketing site
  src/services/           API client
  src/context/            Auth context
docs/
```

The engine package carries no service concerns, so it can be embedded,
benchmarked and tested on its own. That separation is why the recognition tests
can run without a database, and why the pipeline can be profiled without
standing up the API.

---

## Data model

| Table | Holds |
|---|---|
| `tenants` | Isolation boundary |
| `users` | Accounts and roles |
| `api_keys` | Machine credentials (hashed) |
| `cases` | Investigations, with lawful basis |
| `subjects` | Enrolled people |
| `templates` | Encrypted biometric templates |
| `search_runs` | One probe searched against the gallery |
| `candidates` | Ranked results plus examiner adjudication |
| `audit_records` | Hash-chained action log |

Every table carrying data is scoped by `tenant_id`.

The gallery lives in memory for search speed and is rebuilt from the database on
first use per tenant, so the database remains the source of truth and a restart
loses no enrolments.

SQLite connections set `foreign_keys=ON` (off by default, which would let
subjects outlive a deleted tenant) and `journal_mode=WAL` (without it, concurrent
searches block behind any enrolment).

---

## API

Authenticate with `Authorization: Bearer <token>` or `X-API-Key: <key>`.

```text
POST   /api/auth/login                              obtain tokens
POST   /api/auth/refresh                            renew an access token
GET    /api/auth/me                                 current user
POST   /api/auth/users                              create user           (admin)

GET    /api/cases                                   list cases
POST   /api/cases                                   open a case
GET    /api/cases/{id}                              case detail
PATCH  /api/cases/{id}                              update a case
GET    /api/cases/{id}/report?fmt=json|markdown     export a case report

POST   /api/subjects                                enrol                 (supervisor)
GET    /api/subjects                                list subjects
GET    /api/subjects/{id}                           subject detail
GET    /api/subjects/{id}/templates                 template metadata only
DELETE /api/subjects/{id}                           erase a subject       (supervisor)

POST   /api/imatch/search                           search the gallery
POST   /api/imatch/verify                           1:1 comparison
GET    /api/imatch/searches                         search history
GET    /api/imatch/searches/{id}/candidates         ranked candidates
POST   /api/imatch/candidates/{id}/adjudicate       examiner verdict
GET    /api/imatch/engine/status                    loaded model and device

GET    /api/audit                                   audit records
GET    /api/audit/verify                            chain integrity       (admin)

GET    /api/admin/api-keys                          list keys             (admin)
POST   /api/admin/api-keys                          issue a key           (admin)
DELETE /api/admin/api-keys/{id}                     revoke a key          (admin)

GET    /api/health                                  liveness (public)
```

Interactive documentation at `/docs` outside production.

---

## Decision logic

`inference/score_fusion.py` maps a similarity plus probe context onto one of:

| Decision | Meaning |
|---|---|
| `candidate_match` | Above threshold, clear margin, clean probe. An investigative lead |
| `review_required` | In the review band, or above threshold with a caveat |
| `no_match` | Below the review threshold |
| `probe_rejected` | Probe failed the quality gate; never searched |

Thresholds apply to **raw cosine similarity**, never a fused score. Folding
quality into the number compared against a calibrated threshold would silently
move the operating point away from where it was calibrated.

Two cases force review even above threshold:

- **Low margin on a large gallery** (≥50 subjects, margin <0.05). A top hit that
  barely beats the runner-up is the shape of a false match. A *single* candidate
  is explicitly excluded here: its margin of 0.0 means nothing else came close,
  which is the strongest outcome, not a tie.
- **Any probe integrity flag** — liveness, synthetic-media risk, multiple faces.

Quality flags are split into **blocking** and **advisory**. Blocking reasons
(face too small, exposure out of range, severe blur, low detection confidence)
mean the image cannot support a reliable comparison. Advisory reasons (pose,
moderate blur, low contrast) warrant examiner attention but are not grounds for
refusal — treating every flag as fatal caused usable enrolment photographs to be
rejected on a single soft signal, and the pose estimate in particular is an
approximation.
