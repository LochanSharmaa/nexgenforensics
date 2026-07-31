# ROADMAP.md — Phase 8 onward

**Owner-approved direction, 2026-08-01.** Written to the same standard as
BENCHMARKS.md and SCORECARD.md: no capability is listed as planned unless the
work that would prove it is named alongside it.

**Optimising for:** world-class reputation via independent validation
(NIST FRTE). Everything else is sequenced to serve or to run alongside that.

**Preserves:** every measured result, every shipped module, and the
measurement discipline itself. Nothing in Phases 1–7 is discarded.

---

## 0. The reputation thesis — stated plainly

SCORECARD L2 says every number here was produced by the same person and tooling
that built the system. That is the largest credibility gap and **no amount of
internal work closes it.** NIST FRTE is the only mechanism that converts
internal measurement into a public, transferable claim.

Two things must be said honestly before planning around it:

**What NIST submission buys.** Independent, adversarial, third-party
measurement on sequestered data, published under your name. It closes L2
outright. It is the difference between "we measured 99.78% on LFW" and "NIST
measured us."

**What it does not buy.** A top rank. The leading FRTE positions are held by
teams training on proprietary datasets orders of magnitude larger than anything
public. The deployed model here is stock `buffalo_l` / `w600k_r50` — SCORECARD
already states the accuracy "is at parity with the open-source state of the art
because it *is* the open-source state of the art." Submitting stock weights
returns a stock rank.

**Therefore the plan targets a defensible niche rather than the overall
leaderboard.** The niche is chosen by the evidence, not by preference: the one
place this project has an original, measured, unpublished result is
**quality-routed model selection on degraded imagery** (BENCHMARKS §6f,
+4.23pp TinyFace TAR@FAR=0.1% for −0.13pp worst-case clean cost, operating
point fixed before measurement). Surveillance-quality imagery is also the
condition forensic customers actually operate in, and the condition where
published performance is weakest across the whole field.

Reputation = independent validation (NIST) + a genuine contribution in a
narrow band (degraded imagery) + the measurement discipline already in place.
Not architecture diagrams.

---

## 1. Blocking finding — the CPU path is unverified

NIST FRTE evaluates submitted libraries on **NIST-controlled hardware under
strict per-image time and template-size budgets**, historically CPU-bound for
the core 1:1 and 1:N tracks. This project is built the other way round:
`cuda_runtime.py` asserts post-construction CUDA binding and raises
`GpuBindingError` if a model silently lands on CPU — deliberately, and it has
caught real bugs.

A CPU path *does* exist, via `_force_cpu()` ([cuda_runtime.py:48](backend/nexgen_engine/models/cuda_runtime.py:48)),
but **it has never been benchmarked for accuracy or latency.** Every number in
BENCHMARKS.md was produced on CUDA.

> **Action A0 (blocks all NIST work):** run the full seven-benchmark suite and
> `benchmark_speed.py` under `NEXGEN_FORCE_CPU=1`. Two questions must be
> answered before any submission planning is real:
> 1. Is CPU accuracy bit-comparable to CUDA, or does provider-dependent
>    numerics move the thresholds? (Expected: negligible. Must be shown, not
>    assumed — this is exactly the class of assumption §6d punished.)
> 2. What is CPU p50/p95 template-generation latency per image, against the
>    published NIST budget?
>
> If CPU latency fails the budget, the submission model changes — that is a
> Phase 8 finding, not a Phase 11 surprise.

**Verify current FRTE requirements against NIST's published API document
before building to them.** The track structure, timing budgets and API
signatures change between evaluation rounds; nothing in this plan should be
built against remembered requirements.

---

## 2. Phase order

| Phase | Name | Gates on | Runs parallel to |
|---|---|---|---|
| **8** | NIST readiness | A0 | — |
| **9** | Ship the measured wins | Phase 8 findings | 12 |
| **10** | The contribution — degraded imagery | 9 | 12 |
| **11** | Submit | 10 | 12 |
| **12** | Product completeness | — | 9, 10, 11 |
| **13** | Publish | 11 | — |

Phase 12 is deliberately unblocked. It is the customer-facing work (video,
evidence layer, XAI) and it must not wait on an evaluation cycle measured in
months.

---

## Phase 8 — NIST readiness

**Exit criterion:** a compiled library that passes NIST's own validation
package on the target platform, with a local harness that reproduces the
submission path end to end.

| # | Task | Notes |
|---|---|---|
| 8.1 | **A0 — CPU accuracy + latency sweep** | Blocks everything below. See §1. |
| 8.2 | Obtain and read the current FRTE API specification | Track selection follows from it, not before it. |
| 8.3 | Implement the required C++ shared-library API over the ONNX runtime | Thin wrapper; the engine stays Python for the product, the submission binary is separate and shares only the weights and preprocessing. |
| 8.4 | Bit-exact preprocessing parity harness | The submission path and `pipeline.py` must produce identical embeddings on a fixed corpus. Assert to float tolerance, in CI. |
| 8.5 | Template size + serialisation budget audit | 512-d float32 = 2 KB before any packing. Confirm against spec. |
| 8.6 | Offline / no-network / static-link compliance check | NIST binaries run air-gapped. |
| 8.7 | Pass NIST's validation package locally | The real gate. |
| 8.8 | Participation agreement, submission encryption, naming | Administrative, but on the critical path — start early. |

**Risk:** 8.3 is the largest single unknown in this plan. If the C++ wrapper
proves disproportionate, the fallback is to submit to a track with a less
demanding harness first and treat it as a learning submission.

---

## Phase 9 — Ship the measured wins

Everything here is **already measured and not yet shipped.** This is the
cheapest accuracy in the project and it strengthens the submission.

| # | Task | Evidence that justifies it |
|---|---|---|
| 9.1 | **Ship quality-routed selection for 1:1** | §6f: +4.23pp TinyFace TAR@FAR0.1%, −0.13pp worst clean, threshold 0.539 derived on QMUL/CASIA only. §6f explicitly says 1:1 can route immediately. |
| 9.2 | Keep 1:N single-model until embedding-space compatibility is verified at gallery scale | §6f's own caveat. Rank ordering is more sensitive to drift than a threshold comparison. **Do not skip this.** |
| 9.3 | Ship the specialist as an opt-in pack with its own version tag | Never replaces `buffalo_l` as default (item 13). |
| 9.4 | Enable `IndexFlatIP` in production | Closes L10. Measured in §7d; the guarded branch already exists. Needs a deliberate verification run, not just `pip install faiss`. |
| 9.5 | Expose `encode_all_faces()` via the API | Already implemented at [pipeline.py:196](backend/nexgen_engine/inference/pipeline.py:196), never exposed. Roadmap item 26. |
| 9.6 | Re-run `regression_check.py` and update CLAIMS.md | Mandatory after any model/threshold/fusion change — CLAIMS.md says so in its own header. |

**Explicitly not doing: multi-model ensembling.** BENCHMARKS §3 is titled *"the
ensemble does not earn its cost"* — the shipped 3-model ensemble lost to the
best single model on 4 of 5 datasets at 3× compute. Both proposed architecture
diagrams call for a 6-model ensemble; our own data rejects it. Reversing that
decision requires new measurement, not a diagram.

---

## Phase 10 — The contribution: degraded imagery

L3 is the operational weakness that matters (TinyFace 33.13% TAR@FAR0.1% —
about one genuine match in three at a defensible operating point). It is also
the niche where an original result is achievable.

| # | Task | Decision rule |
|---|---|---|
| 10.1 | **Evaluate AdaFace** as a drop-in on the existing harness | SCORECARD names it "the cheapest untested lever — one model swap." AdaFace's stated design target is exactly low-quality imagery. Adopt only if it beats the routed engine at the routed engine's own operating point. |
| 10.2 | Evaluate MagFace for quality-aware embedding norms | Its embedding magnitude encodes quality — potentially a better routing signal than the current heuristic score. Test as a *router*, not as a matcher. |
| 10.3 | **Enhancement, measured then decided** (owner decision, 2026-08-01) | Build SwinIR / Real-ESRGAN / GFPGAN behind a flag. Run through the existing TinyFace/QMUL harness. Adopt **only** on measured TAR@FAR improvement. See §3 below for the non-negotiable constraints. |
| 10.4 | Acquire genuinely low-resolution training imagery | §6d/L8: fine-tuning on *synthetically* degraded data made every benchmark worse, worst on TinyFace (33.13% → 22.23%). The evidence points at synthetic degradation not matching real capture. The next attempt needs real low-res data or it repeats a known failure. |
| 10.5 | Re-run the train/eval overlap audit on any new corpus | Non-negotiable. §7c found overlap in all three existing archives. |
| 10.6 | Address demographic differentials (L5) as a first-class result | Women 1.7× the FNMR of men; under-25s 3.8× the 41–55 band. Raising the threshold relocated the errors without removing the gap. NIST reports demographic differentials publicly — arriving with this already characterised is a credibility asset, and arriving without it is a liability. |

---

## 3. Enhancement — the constraints, since it is being built

Owner decision is **measure it, then decide.** The measurement is only valid
under these constraints, and they are not negotiable regardless of outcome:

1. **Never in the evidential image path.** The image shown to an investigator,
   attached to a report, or hashed into chain of custody is always the
   original. A restored image is an analysis intermediate.
2. **Off by default; logged when on.** If a match was produced from a restored
   probe, the audit record and the report must say so, with the model and
   version that restored it.
3. **Judged on TAR@FAR, never on how the output looks.** Restoration is
   optimised for human perceptual quality, which is a *different objective*
   from identity preservation. A face that looks sharper and matches worse is
   the expected failure mode, not a surprise.
4. **A negative result is a shippable result.** If generative restoration
   degrades recognition on TinyFace/QMUL — which §6d's outcome makes
   plausible — that is a publishable finding and directly useful to the field.
   It gets written up either way (Phase 13).

The forensic hazard being controlled for: super-resolution and face
restoration synthesise detail from a learned prior. On a 20-pixel face, a large
fraction of the restored output is the generator's prior rather than the
subject. Matching against it risks matching the generator. Both proposed
architecture diagrams place unconditional enhancement *before* detection and
extraction; that ordering is rejected here.

---

## Phase 11 — Submit

| # | Task |
|---|---|
| 11.1 | FRTE **1:1 Verification** — the primary submission |
| 11.2 | FRTE **1:N Identification** — after 9.2 clears |
| 11.3 | **FATE Quality** (ISO/IEC 29794-5 aligned) — see below |
| 11.4 | Publish the submitted algorithm's provenance in CLAIMS.md before results return |

**11.3 deserves emphasis.** The quality-assessment track is where a small team
can realistically contribute, it is standards-anchored, and this project
already has `ImageQualityFilter` plus a *measured* quality-routing result. It
is the closest existing asset to a track-leading position. Aligning
`ImageQualityFilter` to ISO/IEC 29794-5 is worth doing on its own merits — both
diagrams correctly call for standardised quality metrics, and the current
implementation is bespoke.

**Handling results honestly:** publish the rank whatever it is. A mid-table
NIST result published without spin is worth more than an unvalidated claim of
excellence, and this project's entire differentiator is that it corrects itself
in public.

---

## Phase 12 — Product completeness (parallel, unblocked)

Absorbed from the two proposed architectures, filtered to what survives
scrutiny. Ordered by value per unit cost.

### 12a. Video and acquisition — the largest capability gap
`opencv-python-headless` is already a dependency, so this is nearer than it
looks. Unlocks ~15 roadmap items in one module.

Frame extraction · keyframe selection · per-frame search · face tracking
across frames · timeline playback · snapshot extraction · clip generation ·
RTSP ingestion · multi-camera time sync.

**Constraint:** a face tracked across 300 frames is *one* observation of one
person, not 300 matches. Score aggregation across a track must be designed
deliberately or every confidence figure the system reports becomes inflated.

### 12b. Forensic evidence layer
EXIF extraction (Pillow already does `exif_transpose` and discards the rest) ·
perceptual hashing + near-duplicate detection (SHA-256 already stored per
image; pHash catches what SHA-256 misses) · compression/noise analysis ·
AI-generated-image detection · face redaction and pixelation · CSV and
court-ready chain-of-custody export ([reports.py:31](backend/imatch_api/api/routes/reports.py:31)
already branches on format).

### 12c. Explainable AI — required for court, not optional
Both diagrams call for it and neither specifies it. Minimum viable form:
which facial regions drove the similarity score, the score's position in the
impostor and genuine distributions, and the FMR/FNMR at the operating point
used. A number without a distribution is not explainable.

### 12d. Investigation intelligence
Face clustering / unknown-person grouping (embeddings already in the `Template`
table; agglomerative over cosine similarity, no new dependency) · identity
graph · case linking · timeline analysis.

### 12e. Security and infrastructure gaps
MFA/TOTP · Prometheus `/metrics` (an engine metrics endpoint already exists,
needs reformatting) · Alembic migrations (currently a hand-rolled
`migrate_auth_columns.py`) · close **L1: host the backend**, which also
unblocks **L9: the stranger test** · resolve **L4** (`frontend/dist` tracked in
a gitignored path).

### 12f. Scale — adopt on a measured trigger, not on principle
Milvus, HNSW, IVF-PQ, Elasticsearch, Celery, RabbitMQ. §7d already established
that recall, not speed, decides ANN adoption, and that exact `IndexFlatIP` is
correct at current scale. **Trigger: revisit at >1M templates or when measured
p95 search latency exceeds budget.** Adopting distributed vector search before
then adds unmeasured complexity to a system whose main asset is that its
complexity is measured. Same for Celery/RabbitMQ: adopt when a real workload
needs it — video batch processing (12a) is the likely first genuine trigger.

---

## Phase 13 — Publish

| # | Output | Basis |
|---|---|---|
| 13.1 | Quality-routed model selection for degraded imagery | §6f — original, measured, operating point fixed pre-measurement |
| 13.2 | Synthetic degradation does not transfer to real low-resolution capture | §6d/L8 — a clean negative result the field would benefit from |
| 13.3 | Generative restoration vs. identity preservation | 10.3, whichever way it lands |
| 13.4 | Demographic differentials under threshold adjustment | L5 — threshold changes relocated errors without removing the gap |

---

## 4. Rejected from the proposed architectures, with reasons

| Proposed | Verdict | Reason |
|---|---|---|
| 6-model ensemble (ArcFace + AdaFace + MagFace + CurricularFace + ElasticFace + PartialFC) | **Rejected** | BENCHMARKS §3 measured it and it lost. Also a taxonomy error: these are margin-based *loss functions* sharing near-identical backbones, and PartialFC is a sharded-FC *training strategy*, not a model. Highly correlated members are the condition under which ensembling does not pay. |
| Unconditional enhancement before detection | **Rejected as drawn; rebuilt as 10.3** | Identity hallucination risk in the evidential path. Measured behind a flag instead. |
| Liveness / anti-spoofing / PAD as a pipeline stage | **Rejected in that position** | Liveness asks "is a live person at the sensor now?" — unanswerable of an archived CCTV frame. Belongs at enrolment. Already correctly labelled `certified: false` here. |
| Multi-biometric fusion (gait, voice, tattoo, clothing, vehicle) | **Deferred indefinitely** | Five separate research programmes listed as one row. Clothing is not biometric — that is re-identification, not identity. Fusion without per-modality validated error rates is not defensible in court. |
| SAM 2, CLIP, YOLO in the recognition path | **Rejected** | No stated role in face recognition. SCRFD already handles detection and is measured. |
| Milvus / Elasticsearch now | **Deferred to a measured trigger** | §7d, see 12f. |

**Adopted from them:** ISO/NFIQ-style standardised quality metrics, Explainable
AI, identity graph and case linking, AI-image detection, re-ranking, the whole
acquisition layer, and the Decision Engine framing — which is the strongest
idea on either page and is partially built already
([score_fusion.py:71](backend/nexgen_engine/inference/score_fusion.py:71)).

---

## 4b. Expected outcome — what this plan actually produces

Stated before the work starts, so it can be checked against afterwards.

### Near-certain (in our control, gated only on effort)

| Outcome | Evidence it rests on |
|---|---|
| **L2 closed** — a public, third-party, adversarial measurement published under this project's name | NIST FRTE submission (Phase 11). The only mechanism that does this. |
| **+4.23pp TAR@FAR0.1% on degraded imagery, shipped** | Already measured, §6f. Phase 9 is integration, not research. |
| **The CPU execution path characterised** for accuracy and latency | A0. Currently a blind spot in an otherwise complete measurement record. |
| **A citable negative result** on synthetic degradation not transferring to real low-resolution capture | §6d already produced it; Phase 13 writes it up. |
| **`ImageQualityFilter` aligned to ISO/IEC 29794-5** | 11.3. Replaces a bespoke metric with a standards-anchored one. |
| **A demonstrable product** — video ingestion, evidence layer, XAI | Phase 12, unblocked and parallel. |

### Likely (dependent on results we cannot pre-judge)

- **A mid-table FRTE 1:1 rank.** Stock `buffalo_l` weights return a stock
  position. This is the expected result and it is still worth having — an
  honestly published mid-table NIST entry outranks an unvalidated claim of
  excellence.
- **A stronger position in FATE Quality than in 1:1/1:N.** It is
  standards-anchored, less dominated by proprietary-data scale, and closest to
  an asset this project already has.
- **AdaFace or MagFace displacing or improving the routing arrangement** (10.1,
  10.2). Cheap to test; either result is informative.

### What this plan will *not* deliver — stated so it is not discovered later

- **Not a top-tier FRTE rank on 1:1 or 1:N.** Those positions belong to teams
  training on proprietary corpora orders of magnitude larger than anything
  public. No sequencing of this plan changes that.
- **Not solved degraded imagery.** Routing takes TinyFace TAR@FAR0.1% from
  33.13% to 37.37%. That is a real gain and it still **misses roughly three
  genuine matches in five.** The lead-not-identification rule (L3) survives this
  plan intact and must survive it in the product copy too.
- **Not a trained model contribution** — unless 10.4 succeeds. See below.
- **Not a court-tested system.** No amount of internal or NIST work substitutes
  for a result surviving adversarial examination in an actual proceeding.

### The fork that decides how this ends

**10.4 — acquiring genuinely low-resolution training imagery — is the single
determinant of which outcome this plan reaches.**

- **Without it:** a rigorously validated, well-documented *integrator* of
  open-source face recognition, with an original routing result and unusual
  measurement discipline. Credible, defensible, publishable. Not a model
  contribution.
- **With it:** a plausible path to a genuine contribution in a band the field
  has not solved and where published performance is weakest across every
  vendor. §6d established what does *not* work and why, which is the expensive
  half of that problem already paid for.

Treat 10.4 as the highest-leverage item in the plan, not a late-phase task.
Sourcing real surveillance-quality training data with defensible provenance is
slow and partly outside our control — **start it in Phase 8, in parallel with
A0**, so it is not the thing everything waits on in Phase 10.

### Timeline realism

NIST evaluation cycles run on NIST's schedule; submission to published result is
measured in months, not weeks, and the administrative steps (8.8) gate the
technical ones. This is the reason Phase 12 is explicitly unblocked: the
customer-facing product must not idle waiting on an external clock.

The honest overall shape: **Phases 8–9 are weeks of work with certain payoff.
Phase 10 is the research risk. Phase 11 is mostly waiting. Phase 12 is where
the product becomes demonstrable, and it should be running the entire time.**

### What "highly reputed" looks like concretely, at the end

Not a leaderboard position. A repository where an adversarial reviewer — a
procurement officer, an opposing expert witness, a NIST evaluator — finds:
a published third-party result, a separated per-condition accuracy record,
declared demographic differentials, dataset provenance, documented operating
points, and negative results reported as negative. Almost no one in this field
publishes the last item. That is the differentiator, and it is already half
built.

---

## 5. What neither proposed architecture contained

No box for **evidence**: no independent validation, no demographic differential
reporting, no dataset provenance, no operating-point documentation, no model
versioning or rollback, no negative results.

That is the gap this plan is built around, and it is the one area where this
project is already ahead of both proposals. Preserve it. Every phase above
either produces evidence or is gated on it.

---

## 6. Standing rules carried forward

1. Any model, threshold or fusion change → re-run `regression_check.py` **and**
   the CLAIMS.md ↔ BENCHMARKS.md cross-check (item 46).
2. No operating point is chosen by looking at the benchmarks it will be
   reported against (§6f states this explicitly; 0.539 was derived on disjoint
   data for exactly this reason).
3. Any new training corpus → train/eval overlap audit before any number from it
   is quoted (§7c).
4. Heuristics stay labelled as heuristics. `certified: false` is not a
   placeholder to be removed when the feature matures — it is removed when a
   measurement replaces the heuristic.
5. Degraded-imagery results generate **leads for human review, never
   identifications** (L3).
