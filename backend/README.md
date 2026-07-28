# NexGen iMATCH — backend

Two packages, deliberately separated:

- **`nexgen_engine/`** — recognition only. Detection, alignment, template
  extraction, matching, and the integrity screens around them. No HTTP, no
  database, no auth, so it can be embedded, benchmarked, and tested on its own.
- **`imatch_api/`** — the service. FastAPI, persistence, authentication,
  tenancy, audit, reporting.

See the [root README](../README.md) for setup. This document covers internals.

---

## Recognition pipeline

```
bytes → decode (EXIF-aware) → detect → select face → quality gate
      → landmark-align to 112×112 → ArcFace r100 → flip-TTA average
      → L2 normalize → 512-d template
```

Two things the pipeline deliberately does **not** do, because both corrupt
matching and both were present in the original scaffold:

1. **No random-projection fusion across backbones.** A projection matrix never
   trained jointly with the encoders destroys the metric structure ArcFace
   learned. The earlier "eight-backbone ensemble" was eight copies of the same
   hash function behind different class names, fused through a seeded random
   matrix.
2. **No query-dependent state in the template.** Cohort normalization used to
   adjust the embedding itself using statistics from previous queries, which made
   a stored identity depend on unrelated search history — the same photograph
   enrolled twice produced two different people. Normalization now applies to
   *scores*, in `inference/score_fusion.py`.

Test-time augmentation is the horizontal flip and nothing else. ArcFace is
trained with flip augmentation, so a face and its mirror land close together and
averaging cancels some pose noise. Brightness and sharpening variants push the
crop off the training manifold and drag templates toward the dataset mean, which
makes different identities *more* similar, not less.

### Alignment matters more than it looks

`detection/alignment.py` fits a similarity transform (Umeyama) from the five
detected landmarks onto the canonical ArcFace layout. Feeding the network an
unaligned crop costs a large amount of accuracy — the model expects the eye line
in a specific place. When the active detector produces no landmarks, alignment
degrades to a square bounding-box crop and the response carries a
`no_landmark_alignment` flag, so the degradation is visible rather than silent.

### Pre-cropped probes need padding

A detector trained on photographs expects a face to occupy *part* of a scene.
Hand it an image that is only a face — a mugshot, a database thumbnail, a
previously cropped probe — and it finds nothing. Measured on AgeDB (112×112
crops), the first pass detected **0 of 120** faces; re-running with a 40%
replicated border detected all 120 at ~0.8 confidence.

Forensic probes are frequently already cropped, so this is the common case, not
an edge case. Images at or below 200 px skip the unpadded pass entirely rather
than paying for one that will fail. Every coordinate is mapped back to the
original image, and the response carries `padded_detection` so an examiner can
see how the detection was obtained.

### There is no fallback backend

InsightFace loads, or the engine raises `EngineUnavailableError`. Earlier
revisions fell back to an OpenCV Haar cascade (boxes but no landmarks, so no
proper alignment) and finally to a centre crop that asserted a face was present
without looking. Both are gone: a centre crop is a fabricated detection, and a
landmark-less crop silently degrades recognition.

---

## Decision logic

`inference/score_fusion.py` maps a similarity plus probe context onto one of:

| Decision | Meaning |
|---|---|
| `candidate_match` | Above threshold, clear margin, clean probe. An investigative lead |
| `review_required` | In the review band, or above threshold with a caveat |
| `no_match` | Below the review threshold |
| `probe_rejected` | Probe failed the quality gate; never searched |
| `recognition_unavailable` | No model loaded; no conclusion possible |

Thresholds are always applied to **raw cosine similarity**, never to a fused
score. Folding quality into the number compared against a calibrated threshold
would silently move the operating point away from where it was calibrated.

Two cases force review even above threshold:

- **Low margin on a large gallery** (≥50 subjects, margin <0.05). A top hit that
  barely beats the runner-up is the shape of a false match.
- **Any probe integrity flag** — liveness, synthetic-media risk, multiple faces,
  or degraded alignment.

---

## Persistence

SQLModel over SQLite (development) or PostgreSQL (production). Tables: tenants,
users, api_keys, cases, subjects, templates, search_runs, candidates,
audit_records.

The gallery lives in memory for search speed and is rebuilt from the database on
first use per tenant, so the database stays the source of truth and a restart
loses no enrolments. Templates are decrypted only into that index and never
returned through the API.

SQLite connections set `foreign_keys=ON` (off by default, which would let
subjects outlive a deleted tenant) and `journal_mode=WAL` (without it, concurrent
searches block behind any enrolment).

---

## Scaling

Brute-force cosine over an L2-normalized matrix is exact and fast enough well
past 10⁵ templates per tenant on CPU. Beyond that, put an ANN backend behind
`GalleryIndex` and accept the recall/latency trade-off explicitly rather than
inheriting it silently.

Each uvicorn worker loads its own copy of the model (hundreds of MB) and keeps
its own in-memory gallery. Scale with replicas, not `--workers`, until the
gallery moves to a shared vector store.

---

## Modules

| Path | Purpose |
|---|---|
| `nexgen_engine/detection/detector.py` | Detector backends and selection |
| `nexgen_engine/detection/alignment.py` | Umeyama fit, canonical warp, pose estimation |
| `nexgen_engine/models/arcface.py` | ArcFace recognizer and labelled stub |
| `nexgen_engine/data/quality_filter.py` | Laplacian sharpness, exposure, pose gating |
| `nexgen_engine/inference/pipeline.py` | The pipeline above |
| `nexgen_engine/inference/score_fusion.py` | Decision engine and score normalization |
| `nexgen_engine/search/gallery_index.py` | Tenant-partitioned vector gallery |
| `nexgen_engine/security/template_encryption.py` | AES-256-GCM with per-tenant HKDF subkeys |
| `nexgen_engine/security/liveness.py` | Passive frequency-domain screen (heuristic) |
| `nexgen_engine/security/deepfake_detector.py` | Spectral artefact screen (heuristic) |
| `nexgen_engine/security/morphing_detector.py` | Single-image and differential morph screens |
| `imatch_api/services/engine_service.py` | Model lifecycle, gallery, crypto |
| `imatch_api/services/audit_service.py` | Hash-chained audit trail |
| `imatch_api/services/report_service.py` | Case report construction |

---

## Endpoints

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
GET    /api/subjects/{id}/templates                 template metadata only
DELETE /api/subjects/{id}                           erase a subject       (supervisor)

POST   /api/imatch/search                           search the gallery
POST   /api/imatch/verify                           1:1 comparison
GET    /api/imatch/searches                         search history
GET    /api/imatch/searches/{id}/candidates         ranked candidates
POST   /api/imatch/candidates/{id}/adjudicate       examiner verdict
GET    /api/imatch/engine/status                    what the engine actually is

GET    /api/audit                                   audit records
GET    /api/audit/verify                            chain integrity       (admin)

GET    /api/admin/api-keys                          list keys             (admin)
POST   /api/admin/api-keys                          issue a key           (admin)
DELETE /api/admin/api-keys/{id}                     revoke a key          (admin)

GET    /api/health                                  liveness (public)
```

Authenticate with `Authorization: Bearer <token>` or `X-API-Key: <key>`.

---

## Scripts

```bash
python scripts/seed.py                     # first tenant and administrator
python scripts/calibrate_threshold.py DIR  # measure your operating point
python scripts/dataset_cli.py --help       # dataset manifest utilities
```

---

## Honest limitations

- **No independent benchmark.** No NIST FRVT submission, no published accuracy
  figure. Any number you need must come from your own evaluation.
- **Liveness, deepfake, and morphing screens are heuristics.** Not evaluated
  against ISO/IEC 30107-3. A determined attacker defeats all three.
- **Demographic performance is unmeasured.** Error rates vary across demographic
  groups; measure this on your own population before deployment.
- **Rate limiting is per process.** Behind multiple workers the effective limit
  multiplies. Put a shared store or an edge limiter in front.
- **Access tokens are stateless.** Revocation waits for expiry. Keep
  `NEXGEN_ACCESS_TOKEN_MINUTES` short, or add a revocation store.
