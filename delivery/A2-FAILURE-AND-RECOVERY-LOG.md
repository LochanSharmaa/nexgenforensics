# Failure and Recovery Log — NexGen iMATCH

**Document:** A2 · **Period covered:** 2026-07-01 to 2026-08-01
**Companion documents:** A1 Development History (commit-by-commit),
A3 System Architecture

---

## Purpose

A record of **every failure encountered while building this system**: what broke,
how it was found, what the actual cause turned out to be, how it was fixed, and
what evidence proves the fix.

It is kept because a record containing only successes is not a record of the
work. It also has direct operational value: several entries below are failures
that *passed all tests* and were caught only by an independent check. Those are
the ones most likely to recur.

**Structure of each entry**

| Field | Meaning |
|---|---|
| Symptom | What was actually observed |
| Detection | How it surfaced — and whether the test suite caught it |
| Root cause | What was truly wrong, after diagnosis |
| Fix | The change made |
| Evidence | The measurement proving the fix |
| Lesson | The generalisable point |

**Severity**

- **S1 — Silent wrong results.** The system appeared to work and produced
  incorrect output. Worst class: nothing alerts anyone.
- **S2 — Loud failure.** Crash, 500, refusal. Bad but self-announcing.
- **S3 — Methodological.** A measurement or claim was wrong, not the code.
- **S4 — Environmental.** Setup, dependency, configuration.

---

# Part I — Silent wrong results (S1)

These are the entries that matter most. Every one of them passed the tests.

---

## F-01 · GPU silently fell back to CPU

**Severity:** S1 · **Domain:** Inference runtime

**Symptom.** Benchmarks ran to completion with plausible numbers, but far
slower than the hardware should allow. Nothing reported an error.

**Detection.** Noticed by wall-clock time, not by any assertion. The test suite
passed throughout.

**Root cause.** `onnxruntime.get_available_providers()` reports providers the
package was **built** with, not the providers a session actually **bound**. Code
checked the former and concluded CUDA was active. The real binding is only
visible post-construction via `session.get_providers()`.

Compounding it, `insightface` installs plain `onnxruntime` as a dependency
alongside `onnxruntime-gpu`. With both present, the CPU build wins at import
time. So the environment silently degraded itself.

**Fix.**
- `nexgen_engine/models/cuda_runtime.py`: `assert_face_analysis_providers()`
  inspects the real bound provider of every sub-model and raises
  `GpuBindingError` when CUDA was expected but CPU was bound.
- `scripts/setup_gpu.py` uninstalls **both** onnxruntime packages then reinstalls
  the GPU build with `--no-deps`.
- `scripts/verify_gpu.py` — 12 independent checks.

**Evidence.** `verify_gpu.py` reports `provider=CUDAExecutionProvider` per model;
benchmark throughput increased by more than an order of magnitude.

**Lesson.** *A capability check that reads configuration rather than observing
behaviour will report what was intended, not what happened.* This recurred (see
F-02) because the fix was applied in one layer only.

---

## F-02 · The same GPU fallback, again, in the API layer

**Severity:** S1 · **Domain:** API

**Symptom.** After F-01 was fixed and verified, the deployed API was still
running recognition on CPU.

**Detection.** Found by inspecting the running service, not by a test.

**Root cause.** `imatch_api` constructed its own runtime and never called the
Phase 1 assertion. The fix in F-01 protected the benchmark path only. The
engine default was also `device="cpu"`.

**Fix.** Real-session CUDA probe in the API startup path; `EngineConfig.device`
default changed `"cpu"` → `"auto"`.

**Evidence.** Engine banner on the running service reads
`buffalo_l (w600k_r50), 512-d templates on cuda`.

**Lesson.** *Fixing a class of bug in one code path does not fix the class.*
When the same wrong assumption can be made in two places, the guard belongs
where both must pass through it.

---

## F-03 · Threshold drift across four copies, causing a demonstrated false match

**Severity:** S1 · **Domain:** Matching

**Symptom.** The user submitted two images of **different people** and the UI
reported a match at similarity 0.2405.

**Detection.** By the user, in normal use. Not by any test.

**Root cause.** The decision threshold existed as four independent constants in
four files. They had drifted. The lowest value in the chain was 0.20, low enough
to admit a genuine false match.

**Fix.** `ThresholdConfig` in `nexgen_engine/config.py` became the single source
of truth; every other site derives from it via `Field(default_factory=...)`.
Threshold raised to **0.2871**.

**Evidence.** The reported pair now scores below threshold and is refused. The
API test that hard-coded `top_score=0.36` was rewritten to derive from config,
so it can no longer pass against a stale constant.

**Follow-up, and a rejected "improvement".** A later suite-wide calibration
proposed **0.2363**. It was measured, found to re-admit the exact false match
above, and **rejected**. Recorded because a number that optimises an aggregate
metric can still be wrong operationally.

**Lesson.** *A constant duplicated is a constant that will drift.* And: the
value that optimises average accuracy is not automatically the value that should
be deployed.

---

## F-04 · Validation proxy reported improvement while the model got worse

**Severity:** S1/S3 · **Domain:** Model training

**Symptom.** Fine-tuning run #1 reported its validation margin improving from
+0.5474 to +0.6055 — an apparent +0.058 gain. The real benchmarks showed the
model was **worse on every dataset**, worst on TinyFace at −3.07pp.

**Detection.** By refusing to accept the training-time metric and running the
fixed-pair benchmarks.

**Root cause.** The proxy resampled its validation batch on every evaluation, so
its own sampling noise was as large as the effect it claimed to detect. **The run
proves this itself:** during the first 600 steps the backbone was frozen —
learning was impossible — and the margin still swung 0.5474 → 0.4887. That swing
(~0.06) is pure noise, and the final "gain" was +0.058.

**Fix.** Training-time proxies are no longer used for any decision. Run #2
(`finetune_qmul.py`) early-stops on **fixed, published pair lists** only.

**Evidence.** BENCHMARKS §6d records both the proxy's claim and the benchmark
refutation side by side.

**Lesson.** *A metric whose noise floor is unmeasured cannot detect an effect.*
The frozen-backbone phase accidentally provided a perfect noise measurement; it
should be a deliberate control, not an accident.

---

## F-05 · Contamination audit produced a 96.9% false positive

**Severity:** S3 · **Domain:** Dataset integrity

**Symptom.** The QMUL-SurvFace overlap audit reported **96.9% of identities**
above the exclusion threshold, with 78% of nearest neighbours in TinyFace. Read
literally: the dataset was almost entirely contaminated and unusable.

**Detection.** By disbelieving the result and building a control.

**Root cause.** ArcFace embeddings of very low quality faces collapse toward a
common region. A 27×22px face encodes "degraded face" more than "this person".
Both QMUL and TinyFace are native low-resolution, so they appear mutually
similar with **zero** shared identities.

**Fix.** `qmul_overlap_control.py`, exploiting ground truth that was free:
distinct QMUL directories are distinct people by construction.

| Measurement | Median |
|---|---|
| QMUL genuine (same person) | 0.316 |
| Nearest TinyFace (max over 8,171) | 0.522 |
| **Nearest different-person QMUL (max over 6,693)** | **0.600** |

The matched null **exceeds** the TinyFace affinity. A QMUL face resembles a
random different QMUL person more than anything in TinyFace.

**Evidence.** Conclusion reversed: no identity contamination; nothing excluded.

**Lesson.** *A similarity threshold calibrated on clean imagery is meaningless
on degraded imagery.* Quality-induced clustering will masquerade as identity
overlap.

---

## F-06 · The control for F-05 was itself wrong, and nearly reversed the finding

**Severity:** S3 · **Domain:** Methodology

**Symptom.** The first version of the F-05 control printed
**"SPECIFIC — treat the overlap as real"** — the opposite of the correct
conclusion.

**Detection.** By checking whether the comparison was fair before acting on it.

**Root cause.** It compared the nearest TinyFace neighbour — a **maximum over
8,171 candidates** — against a **single random** impostor pair. The maximum of
8,171 draws exceeds a single draw regardless of the underlying distribution. The
test would have declared contamination on entirely unrelated data.

**Fix.** Matched the null: maximum over a comparable number of candidates drawn
from known-different identities. Verdict reversed to ARTEFACT.

**Evidence.** Both versions and the reasoning are retained in the script's
docstring so the error cannot be silently repeated.

**Lesson.** *Extreme-value statistics must be compared against extreme-value
nulls.* A max-of-N versus a single draw is not a comparison.

---

## F-07 · Core source files were never committed

**Severity:** S1 · **Domain:** Repository integrity

**Symptom.** Multiple modules referenced across the codebase did not exist in
version control. A fresh clone could not import the package.

**Detection.** During a Phase 4 audit of what was actually tracked.

**Root cause.** Per-file `git add` habit. Files created and used locally were
never staged. Everything worked on the development machine, so nothing surfaced.

**Fix.** Missing modules created/restored (`nexgen_engine/api/schemas.py`,
`__init__.py`), dead references removed, `BackboneConfig`/`VectorSearchIndex`
ported to `GalleryIndex`.

**Evidence.** CI on a clean checkout now imports and tests the package.

**Lesson.** *"It works on my machine" includes files git has never seen.* CI on a
clean clone is the only check that catches this.

---

# Part II — Loud failures (S2)

---

## F-08 · 500 on extreme aspect ratios

**Symptom.** Uploading an image with an extreme aspect ratio returned HTTP 500.

**Root cause.** SCRFD computed `new_height=0` for very wide/thin inputs; OpenCV
raised `cv2.error` from inside the detector.

**Fix.** Geometry guards at the decode boundary — minimum 16px edge, maximum
50:1 aspect ratio — so malformed input is rejected with a usable 400 before it
reaches the model.

**Lesson.** *Validate at the boundary where the data enters, not where it
eventually breaks.*

---

## F-09 · 500 on any non-image upload to `/search`

**Symptom.** Uploading a non-image file returned 500.

**Root cause.** `storage.store()` sat **outside** the `try` block that handled
decode failures.

**Fix.** Moved inside. Now returns 400 with a readable message.

---

## F-10 · Authentication completely broken by per-call secret regeneration

**Severity:** S1 · **Domain:** Auth

**Symptom.** Tokens issued by the server were rejected by the same server.

**Root cause.** `resolved_jwt_secret()` generated a **new** ephemeral secret on
every call when none was configured. Every token was signed with a key that no
longer existed by verification time.

**Fix.** Cached `_EPHEMERAL_JWT_SECRET` at module scope.

**Lesson.** *A function that returns a fresh secret each call is not a
configuration accessor.*

---

## F-11 · Circular import at startup

**Root cause.** `core/__init__` → `dependencies` → `db.session` → `core.config`.

**Fix.** PEP 562 module-level `__getattr__` for lazy imports in
`core/__init__.py`.

---

## F-12 · Accounts existed that could never authenticate

**Severity:** S2 · **Domain:** Auth

**Symptom.** `investigator@nexgen.local` could not log in. Error claimed the
address was malformed.

**Root cause.** `bootstrap_admin.py` writes **directly to the database** and
accepted `.local`. The login endpoint validated with pydantic `EmailStr`, which
delegates to `email-validator`, which **rejects special-use domains**. The system
could mint an account it would then refuse forever, and the error pointed at the
wrong thing.

**Fix.** `AccountEmail` type on both creation and login: drops the
deliverability rule (internal deployments legitimately use `.local`,
`.internal`, `.lan`), keeps a shape check that still catches typos.

**Evidence.** Same account: HTTP 422 before, **HTTP 200** after.

**Lesson.** *Every account the system lets you create must be an account you can
sign into.* Validation asymmetry between creation and authentication is a
silent account-destruction bug.

---

# Part III — Methodological errors (S3)

---

## F-13 · ANN benchmark measured nothing, because the vectors were synthetic

**Symptom.** Approximate-search recall looked catastrophically poor.

**Root cause.** The benchmark used **random 512-d vectors**. In high dimensions
random vectors are near-equidistant, so nearest-neighbour structure does not
exist and no index can recover it.

**Fix.** Re-ran on **real templates**. Recall approximately doubled.

**Recorded as a method error rather than deleted.** The original conclusion
("ANN is unusable") would have been wrong.

**Lesson.** *Synthetic data with the right shape can have the wrong structure.*

---

## F-14 · CFP-FP anomaly was dataset provenance, not accuracy

**Symptom.** CFP-FP scored ~1.6 points below published figures for the same
architecture.

**Root cause.** The `.bin` packs are **not identical** across distribution
bundles. Three distinct `cfp_fp.bin` variants exist by hash. Two independent
sources agree at ~99.2%; the `faces_webface` variant is the outlier.

**Control.** CFP-**FF** on the same bundle scores 99.91% — correctly saturated —
ruling out a fault in the harness, the flip-TTA, or the model.

**Fix.** Every benchmark figure now records **which pack file produced it, by
hash**.

**Lesson.** *A dataset name is not a unique object.*

---

## F-15 · Fine-tuning attempt #1 — synthetic degradation made the model worse

**Symptom.** After a methodologically clean fine-tune, results were worse on
**all six** benchmarks; worst on TinyFace at −3.07pp accuracy and −10.9pp
TAR@FAR0.1%.

**Root cause (most probable).** Domain gap. The model learned to invert *that
specific* synthetic pipeline — bicubic down/up, Gaussian blur, JPEG — which is
not what a distant camera produces. The drop being **largest on real
low-resolution imagery** is what a domain-gap failure looks like.

Secondary observation: AUC **rose** on four sets while accuracy fell, so what
broke there was calibration, not ranking. On TinyFace AUC fell too, so that loss
was genuine discrimination.

**Outcome.** Not deployed. Reported as a negative result.

**Lesson.** *Simulating a degradation is not the same as reproducing it.*

---

## F-16 · Fine-tuning attempt #2 — right data, still no transfer

**Symptom.** Trained on **real** QMUL surveillance capture. QMUL's own published
verification improved **+12.73pp** (69.00% → 81.74%). The seven reporting
benchmarks showed no accuracy improvement anywhere.

**Root cause.** A domain gap one level up. The model learned QMUL's specific
cameras and crops rather than degraded faces in general.

**Critical distinction from F-04.** This validation was **not** a noisy proxy —
it was a fixed, published, identity-disjoint pair list, and its improvement was
real. It still did not transfer. The lesson is therefore different:
*in-domain gain ≠ cross-domain transfer.*

**Recovery — and this one succeeded.** See F-17.

---

# Part IV — Recovery that produced a genuine improvement

---

## F-17 · Recovering value from a "failed" fine-tune

**Context.** F-16 concluded "no improvement". That was true of the accuracy
column and it concealed the shape of the result. At FAR=0.1%:

| | deployed | QMUL checkpoint |
|---|---|---|
| TinyFace TAR | 33.13% | **38.10%** |
| AgeDB-30 TAR | 96.03% | 88.10% |
| CPLFW TAR | 87.40% | 81.73% |

**Not a worse model — a different one.** Better on degraded capture, worse on
clean-but-hard. A single global choice discards whichever advantage it does not
pick.

**Two facts made routing possible, both measured rather than assumed:**

1. **The embedding spaces are compatible** — same image through both models
   gives median cosine **+0.856**. Training at lr 1e-5 refined the space rather
   than rotating it, so a gallery does **not** need re-enrolling. This was the
   assumption most likely to kill the idea.
2. **The existing quality score separates the conditions** — clean sets median
   0.781, TinyFace 0.502. No new model, no extra inference.

**The trap, and how it was avoided.** A threshold sweep showed 0.50 gave the
best numbers. Using it would have been **fitting the operating point to the test
set** — the same error class as F-04, in a new place. It was refused.

The threshold was re-derived from **QMUL and CASIA quality distributions only**,
both disjoint from every reporting benchmark: crossover at **0.539** (5.8% of
degraded missed, 5.9% of clean misrouted).

**Result at that independently-chosen operating point:**

| Dataset | TAR@FAR0.1% deployed | routed | Δ |
|---|---|---|---|
| **TinyFace** | 33.13% | **37.37%** | **+4.23pp** |
| CPLFW | 87.40% | 87.27% | −0.13 |
| AgeDB-30 | 96.03% | 95.97% | −0.06 |
| LFW / CFP-FF / CALFW | — | unchanged | 0.00 |

**The only unambiguous model improvement in the project** — and it came from
routing between two models, not from either being better.

**Lesson.** *A negative aggregate result can contain a positive conditional one.*
Read the per-condition metrics before shelving a model.

---

# Part V — Environmental and configuration (S4)

| ID | Failure | Root cause | Fix |
|---|---|---|---|
| F-18 | `onnxruntime-gpu==1.20.1` uninstallable | No Windows cp311 wheel for that version | Pinned 1.20.2 |
| F-19 | Deploy OOM at 512 MB | Duplicate OpenCV (`opencv-python` + `opencv-python-headless`) = 281 MB | `requirements-deploy.txt` with headless only |
| F-20 | `--no-deps` on everything broke FastAPI | FastAPI needs `annotated_doc` | Scoped `--no-deps` to `insightface` only |
| F-21 | `RESEND_API_KEY` never loaded | File was UTF-16 with a BOM and contained the bare key with no `KEY=` prefix | Rewritten UTF-8 as `RESEND_API_KEY=...` |
| F-22 | `frontend/dist` perpetually modified | Build artefact tracked in git | `git rm -r --cached frontend/dist` + ignore |
| F-23 | CPU/GPU numeric divergence | 0.494431 vs 0.494213 | Documented; a claim of "identical results" was **corrected** |

---

# Part VI — Failures found only by testing in a browser

These passed the automated suite completely.

---

## F-24 · CSRF middleware sat outside CORS

**Severity:** S2 · Nineteen CSRF tests passed; the feature was broken in a real
browser.

**Root cause.** Starlette runs the **most recently added** middleware first. The
CSRF guard was registered after CORS, making it **outermost**, so its 403
short-circuited before CORS could attach headers. A legitimate cross-origin
client saw an opaque `Failed to fetch` with no way to discover why.

**Why tests missed it.** `TestClient` speaks ASGI directly and never exercises
CORS at all.

**Fix.** CORS registered last, making it outermost, so every response —
including middleware refusals — carries its headers.

**Lesson.** *A test harness that bypasses a layer cannot test that layer.*

---

## F-25 · SameSite blocked the CSRF cookie between localhost and 127.0.0.1

**Root cause.** The client pinned `127.0.0.1:8443` while the page was served
from `localhost:5173`. These are **different sites** to a browser even though
they are one machine, so the `SameSite=Lax` cookie was never sent on POST and
the double-submit check had nothing to compare.

**Fix.** The default API origin now derives from the page's own hostname (ports
are irrelevant to SameSite).

**Lesson.** *`localhost` and `127.0.0.1` are not interchangeable to a browser.*

---

# Failure statistics

| Class | Count |
|---|---|
| S1 — silent wrong results | 6 |
| S2 — loud failures | 6 |
| S3 — methodological | 6 |
| S4 — environmental | 6 |
| Found by browser testing after tests passed | 2 |

**Failures caught by the automated test suite: a minority.** The S1 and S3
classes — the ones that produce confidently wrong answers — were found by
independent checks, controls, and disbelieving results that looked good.

---

# Recurring themes

1. **Configuration checks lie; behaviour checks do not.** F-01, F-02.
2. **Duplicated constants drift.** F-03.
3. **A metric with an unmeasured noise floor cannot detect an effect.** F-04.
4. **Thresholds calibrated on one condition are meaningless on another.** F-05.
5. **Extreme-value statistics need extreme-value nulls.** F-06.
6. **Never choose an operating point on the set you will report against.** F-17.
7. **A passing test suite is evidence about the tested layer only.** F-24.
8. **Negative aggregate results can contain positive conditional ones.** F-17.

---

*Generated from the engineering record. Commit-level detail for every entry is
in A1 Development History; measurements are in BENCHMARKS.md and
`runtime/benchmarks/*.json`.*
