# Claims vs. Code

Every user-visible capability claim, matched to the code and test that proves
it — or marked as unbacked. This replaces the current product-page copy.

**Rule applied:** a claim stays only if a specific endpoint or module implements
it *and* something verifies it. Everything else is relabelled to what it
actually does, or removed.

Legend — **Backed**: implemented and tested. **Relabel**: real code exists but
the current wording overstates it. **Remove**: no implementation.

---

> **All figures on this page are for the DEPLOYED model, `w600k_r50`
> (`buffalo_l`), at the deployed threshold 0.2871.** An earlier version of this
> file quoted `glintr100` numbers while `w600k_r50` was in production — the two
> models differ by ~1.5 points on clean sets and by 15 points on TinyFace, so
> the mismatch materially misstated the system. Cross-checked against
> BENCHMARKS.md on 2026-07-31 (item 46).
>
> Re-run that cross-check after ANY model, threshold or fusion change.

## Backed by code and tests

| Claim | Implementation | Proof |
|---|---|---|
| GPU-accelerated inference | `models/cuda_runtime.py` asserts the *post-construction* execution provider; CPU fallback raises | `scripts/verify_gpu.py` — 12/12 checks pass, all 15 ONNX sub-models on `CUDAExecutionProvider` |
| 1:1 face verification | `EngineService.verify()` | `BENCHMARKS.md` §2 — 5 published protocols, 31,000 pairs, 10-fold CV |
| **99.78% verification accuracy (LFW)** | `w600k_r50` (DEPLOYED), flip-TTA, threshold 0.2871 | `BENCHMARKS.md` §2. **Must be quoted with the dataset name** |
| **96.68% (AgeDB-30, cross-age)** | same, at the deployed 0.2871 threshold | `BENCHMARKS.md` §2, §5c |
| Enrollments survive restart | `search/persistence.py` + `EngineService._restore_index()` | `tests_engine/test_service_durability.py::test_enrollment_survives_restart` |
| Encrypted templates at rest | AES-256-GCM per row, key from `NEXGEN_TEMPLATE_KEY` | `test_persistence.py::test_templates_are_encrypted_on_disk` asserts plaintext floats are absent from the DB file |
| Tamper detection on templates | GCM is authenticated — a wrong key raises, never silently decrypts | `test_persistence.py::test_wrong_key_is_detected_not_silently_wrong` |
| Every search is audited | `EngineService._record()` on enroll / identify / verify | `test_service_durability.py::test_identify_writes_audit` |
| Audit hashes are verifiable | `GET /api/biometrics/audit/{audit_hash}` | `test_service_durability.py::test_audit_hash_is_resolvable` |
| Tenant isolation | `GalleryIndex` shards per tenant; `search()` **requires** `tenant_id`, so no code path can compare across tenants | `test_persistence.py::test_tenant_isolation_on_restore` |

> The brief stated "no multi-tenant concept exists anywhere." That is not
> correct — `GalleryIndex` has enforced structural tenant isolation. What was
> missing was tenant-scoped *persistence*, which now exists. The remaining gap
> is that no API surface lets a caller select a tenant; the service pins
> `tenant_id="default"`. Multi-tenant **storage and search** are real;
> multi-tenant **authentication** is not.

---

## Must be relabelled — real code, overstated wording

| Current claim | What the code does | Required wording |
|---|---|---|
| **"Liveness Detection"** | `security/liveness.py` — texture energy, moiré energy, colour spread. A passive single-frame heuristic. It already self-reports `"method": "passive_single_frame_heuristic"` and `"certified": false` | **"Image Quality & Capture Check"**. It is not presentation-attack detection and will not stop a printed photo or replay attack |
| **"Deepfake Check"** | `security/deepfake_detector.py` — FFT mid-band smoothness + Nyquist-corner checkerboard energy. Its own docstring says *"NOT a trained deepfake classifier"* and *"A modern, well-post-processed synthetic face will defeat both"* | **"Synthetic-Media Artifact Screen (advisory)"**. Use to prioritise examiner attention; never to conclude media is authentic |

> The brief said "Deepfake Check — no model exists." A real heuristic detector
> does exist and is honestly documented in its own source. The problem is the
> *UI label*, not the absence of code. Neither module should be deleted — both
> are useful triage signals. Both are misdescribed to users.

---

## Must be removed — no implementation

| Claim | Finding |
|---|---|
| **"Auto-enhance"** | Checkbox in `FaceSearchExperience.jsx:599,604`, **defaults to `true`**. No backend implementation exists anywhere in `backend/`. It is wired to nothing — a control the user believes is processing their image. Remove the checkbox |
| **"URL Import"** | The API accepts multipart uploads only. If re-added, it needs SSRF protection: scheme allowlist, DNS resolution pinning, RFC1918/link-local/loopback blocking, redirect capping, and a response size limit. Not a small task — remove the field until then |

---

## Requires a real project — not a patch

| Claim | Scope |
|---|---|
| **Investigator workspace / case management / roles** | Frontend pages exist (`EnrolPage.jsx`, `ProbeReport.jsx`), and `imatch_api/core/dependencies.py` has `require_admin` / `require_investigator` / `require_supervisor`. Auth scaffolding is real; case linking and workspace state are not. Scope as its own phase |
| **Tenant-aware API** | Storage and search are tenant-scoped; no endpoint exposes tenant selection and no auth binds a caller to a tenant. Needs `tenant_id` on the auth principal and propagation through every route |

---

## Accuracy claims — mandatory phrasing

Never publish a single blended accuracy number.

| Allowed | Not allowed |
|---|---|
| "99.78% 1:1 verification accuracy on LFW (6,000 pairs, 10-fold CV)" | "99.99% accuracy" |
| "96.68% on AgeDB-30 (cross-age)" | "99.99% validation target" |
| "82.45% on TinyFace (degraded, median 32×32 px)" | any average of clean and degraded |
| "87.32% rank-1 identification" — **only** if labelled as identification | quoting the identification number for the verification feature |

Two facts that must accompany any published number:

1. **Degraded-footage performance collapses.** On TinyFace, TAR at FAR=0.1% is
   **33.13%** — roughly one genuine match found in three. Clean-benchmark numbers
   do not describe surveillance footage.
2. **Error rates are not uniform across demographics.** At one global
   threshold, women are falsely rejected 1.7× more often than men (8.45% vs
   4.86%) and falsely matched 5× more often; the under-25 group has 3.8× the
   false-non-match rate of the 41–55 group. See `BENCHMARKS.md` §5.

---

## Removed during this pass

| Item | Reason |
|---|---|
| `DeterministicBackbone` | Hashed image bytes into a pseudo-random vector and returned it as an "embedding" — every similarity score meaningless. Its config type had already been deleted, so it also broke the import of the entire `models` package |
| Hardcoded thresholds 0.28 / 0.42 | Never measured; copied from the README. 0.42 sat above every measured optimum (0.19–0.28), rejecting genuine matches the model scored correctly |
| 3-model ensemble as default | Measured no better than `glintr100` alone on any of 5 protocols, and worse on 3. Now `single_glintr100`, at ⅓ the compute. Override with `NEXGEN_FUSION_METHOD` |
