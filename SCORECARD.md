# SCORECARD.md — Phase 7 items 47 & 50

Written to the same standard as BENCHMARKS.md: plain, factual, no promotional
language. Every "verified" line names the check that produced it. Anything not
verified is listed as a limitation, not omitted.

**As of 2026-07-31.** Deployed model `buffalo_l` / `w600k_r50` on CUDA,
thresholds 0.2871 / 0.2153 / 0.2871.

---

## 1. Verification status right now

| Check | Command | Result |
|---|---|---|
| Engine + API tests | `pytest backend/tests_engine/ backend/tests/` | **161 passed** |
| Accuracy + config regression | `python backend/scripts/regression_check.py` | **PASS** (5 datasets) |
| GPU binding | `python scripts/verify_gpu.py` | **PASS** (12/12, genuinely CUDA) |

---

## 2. Item 47 — every original issue, plus everything found since

### The original 18

| # | Issue | Status |
|---|---|---|
| 1 | Persistence not wired into `imatch_api` | **Was wrong.** It was already wired via its own `Template` table. Proven by enrol → restart → re-query returning a bit-identical 0.494431 |
| 2 | Train/eval overlap unaudited | **Audited. Overlap found in all three archives** (§7c). Fine-tuning closed as a dead end (§6a) |
| 3 | Deployed model vs benchmark winner | **Resolved as deliberate.** `w600k_r50` kept: better on degraded imagery (33.13% vs 17.37% TAR@FAR=0.1%) |
| 4 | No throughput/latency data | **Fixed.** `benchmark_speed.py`; encode p50 14.72 ms, verify 31.04 ms (§7b) |
| 5 | CFP-FP ~1.6 points below published | **Resolved — pack provenance, not accuracy.** Three distinct `cfp_fp.bin` variants (§2b) |
| 6 | Demographic data stale | **Fixed.** Re-measured on the deployed model at the deployed threshold (§5a) |
| 7 | FAISS imported but unused | **Resolved.** Measured; exact `IndexFlatIP` adopted, approximate rejected on recall (§7d) |
| 8 | IJB-B/IJB-C not run | **Still not run.** Corrected label: a 1.57 GB *partial download* exists, not "absent" |
| 9 | Liveness is a heuristic | **Correctly labelled** at API, report and UI layers |
| 10 | Deepfake is a heuristic | **Correctly labelled.** Both self-report `certified: false` |
| 11 | `frontend/dist` stale | **Rebuilt.** See limitation L4 below |
| 12 | `imatchApi.js` on a dead contract | **Rewritten** against the live OpenAPI; proven by browser login → 0.4944 displayed |
| 13 | No batch endpoint | **Built** (3 modes) and **UI wired** (items 20, 21) |
| 14 | No login UI / token handling | **Was wrong.** Both existed; the service layer beneath them did not |
| 15 | `_deprecated_app` quarantined | **Deleted** |
| 16 | numpy version conflict | **Fixed.** Pinned `>=1.26.4,<3.0` — the line every benchmark was produced on |
| 17 | Untracked root files | **Resolved.** ~50 MB of installers and stale logs removed; `node/` gitignored |
| 18 | Unused dependencies | **Removed** after confirming zero imports (8 packages) |

### Found and fixed since (not in the original list)

| Issue | How found | Status |
|---|---|---|
| **Deployed API silently ran ArcFace on CPU** | Startup log inspection | Fixed; `auto` device verified by real ONNX session |
| **Authentication completely broken** without `NEXGEN_JWT_SECRET` — secret regenerated per call | Browser login failing | Fixed; cached at module scope |
| **Core source files never committed** — a fresh clone would not import | Phase 4 hygiene pass | Fixed; 17 files added |
| **No way to create the first operator account** — seed settings read by nothing | Attempting to log in | Fixed; `bootstrap_admin.py` |
| **Thresholds existed in 4 copies and had drifted**; UI stated a rule the engine had stopped applying | Threshold consolidation | Fixed; single source + regression gate |
| **500 on extreme aspect ratios** (`cv2.error` from SCRFD) | Item 33 adversarial tests | Fixed at the decode boundary |
| **500 on any non-image upload to `/search`** | Item 31, driving real traffic | Fixed; now 400 with a usable message |
| **Circular import** in `imatch_api` | Item 33, writing tests | Fixed via PEP 562 lazy resolution |
| **API test suite broken by the threshold change** | Item 49, running both suites together | Fixed; test now derives from config |
| **CLAIMS.md quoted `glintr100` while `w600k_r50` deployed** | Item 46 cross-check | Fixed; provenance header added |
| **Two undeclared dependencies** (`argon2-cffi`, `pydantic[email]`) | Clean-venv rebuild | Declared |
| **`onnxruntime-gpu==1.20.1` has no Windows wheel** | Clean-venv rebuild | Pinned 1.20.2 |
| **`insightface` pinned to a version with no cp311 wheel** | Clean-venv rebuild | Pinned `>=1.0.1,<2.0` |

---

## 3. Item 50 — what is verified, what is not

### Verified working

- **1:1 verification** end to end through the browser against the live API —
  genuine 0.4944 accepted, impostor −0.0662 rejected.
- **1:N search** and **batch** (3 modes), with per-item audit rows and error
  isolation: one corrupt file does not fail a batch.
- **Persistence** — enrolments survive restart, template round-trips bit-exact.
- **PDF forensic report** — 2 pages, real findings, caveats on every page footer.
- **Accuracy**, 31,000 published pairs, 10-fold CV, threshold fitted on 9 folds
  and applied to the held-out fold.
- **GPU execution**, asserted by post-construction session providers.
- **Adversarial input handling** — 23 tests; typed 4xx, never a 500.

### Known limitations

| | Limitation |
|---|---|
| **L1** | **Not deployed and cannot be, as configured.** No hosted backend; `vercel.json`'s catch-all rewrite returns 405 on POST and its CSP blocks `http://`. Everything above is verified locally only. |
| **L2** | **No independent validation.** Every number was produced by the same person and tooling that built the system, on public academic datasets. |
| **L3** | **Degraded-footage performance is weak.** TinyFace 82.45%, and only **33.13% TAR@FAR=0.1%** — about one genuine match in three at a defensible operating point. Leads for human review, never identifications. |
| **L4** | `frontend/dist` is force-added to a gitignored path — tracked copy drifts from source. Decide: untrack and build on deploy, or keep force-adding. |
| **L5** | **Demographic differentials persist.** Women 1.7× the FNMR of men, under-25s 3.8× the 41–55 band. Raising the threshold relocated the errors; it did not remove the gap. |
| **L6** | **CFP-FP absolute figures** in §2 were measured on the outlier pack and understate all configurations by ~1.5 points. Rankings unaffected. |
| **L7** | **Concurrency unmeasured.** All latency figures are single-threaded. **Item 29 (request batching) is now scheduled: after the deploy is proven working, before the Phase 7 stranger test (item 45).** L7 stays open until that measurement exists. |
| **L8** | **Fine-tuning abandoned** — every available training archive is contaminated. No clean corpus on disk. |
| **L9** | **Stranger test (item 45) not run** — blocked by L1. |
| **L10** | **`IndexFlatIP` measured but not enabled in production**; installing faiss activates the existing guarded branch, which needs a deliberate verification run. |

### What would be needed next, in order

1. **Host the backend** (L1). Nothing user-facing is real until this exists, and it blocks the stranger test.
2. **Independent validation** (L2). The single largest credibility gap; nothing internal can close it.
3. **Degraded-footage accuracy** (L3). The measured weakness that matters operationally. AdaFace is the cheapest untested lever — one model swap, measurable on the existing harness.
4. **Concurrency measurement** (L7), then request batching if it is warranted.
5. **A clean training corpus** (L8) if fine-tuning is ever revisited.

### Honest summary

This is a working, locally-verified forensic face-recognition system with a
measurement foundation stronger than its feature completeness. Its accuracy on
clean imagery is at parity with the open-source state of the art because it
*is* the open-source state of the art — stock ArcFace weights, not a trained
contribution. Its distinguishing property is that its claims are measured,
separated by operating condition, and corrected when found wrong.

It is not deployed, not independently validated, and not reliable on
surveillance-quality imagery. None of those are addressed by further internal
work alone.
