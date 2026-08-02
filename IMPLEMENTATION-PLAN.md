# Forensic Identity Intelligence — Implementation Plan

**Programme approval document. Written to be executed, not admired.**
**Date: 2026-08-02 · Supersedes ROADMAP.md Phase 8+ · Companion to NEXTGEN-ARCHITECTURE.md**

---

## 0. What we actually know, measured

Every decision below is grounded in numbers produced on this machine, not on
literature estimates. The grounding run is
`backend/scripts/analyze_evidence_capacity.py`, output in
`runtime/forensics/capacity.json`.

| dataset | Cllr | Cllr_min | Cllr_cal | discrimination share of loss |
|---|---|---|---|---|
| lfw | 0.0149 | 0.0111 | 0.0038 | 74% |
| cfp_ff | 0.0096 | 0.0066 | 0.0030 | 69% |
| agedb_30 | 0.1097 | 0.0937 | 0.0159 | 85% |
| cfp_fp | 0.1601 | 0.1380 | 0.0221 | 86% |
| calfw | 0.1946 | 0.1830 | 0.0116 | 94% |
| cplfw | 0.2616 | 0.2433 | 0.0183 | 93% |
| **tinyface** | **0.7207** | **0.7110** | **0.0097** | **98.7%** |

**Three findings that set the programme's direction:**

1. **Calibration is already close to optimal.** Two-parameter logistic
   calibration on held-out folds leaves Cllr_cal ≤ 0.022 everywhere. There is
   almost nothing left to win by better reporting.

2. **On degraded imagery the system delivers 0.28 bits of a 1-bit question.**
   TinyFace Cllr = 0.72, of which **98.7% is discrimination loss**. This is the
   single most important number in this document: *the evidence layer cannot
   rescue surveillance imagery.* Only better recognition can. It is the
   quantitative licence for everything in Stage 2 and Stage 3.

3. **Capacity measurement is fragile, and we proved it by breaking it twice.**
   First attempt: 99.7% censored on LFW because 3,000 impostor pairs cannot
   resolve a tail below 1/3000. Second attempt with a 20M sampled pool: invalid,
   because "different pair ⇒ different identity" is false — contamination tracks
   identity count (0.202% for AgeDB's 568 identities vs 0.030% for LFW's 5,749),
   and 0.2% contamination is *twice* the FAR being measured at 0.1%. It made
   AgeDB appear to fall from 96.03% to 8.40%. **Reference-population purity is
   the measurement, not a caveat on it.**

The only valid capacity figure we hold is TinyFace, where true identity labels
exist: **bits_median 4.37, bits_p20 1.58, supportable gallery at 80% rank-1 ≈ 1.**
Directionally robust; the absolute value needs a protocol-matched re-run.

---

## 1. Final architecture

```
┌── L0  EVIDENCE INTAKE ──────────────────────────────────────────── NEW ──┐
│  content-addressed store · lineage DAG · signed derivation records       │
│  WHY: in court the question is what the model was shown, not what it said │
└──────────────────────────────┬───────────────────────────────────────────┘
                               ▼
┌── L1  SENSOR & DEGRADATION LAYER ───────────────────────────────── NEW ──┐
│  detect ─ track ─ blind PSF/noise/JPEG estimation ─ MTF passband         │
│  optional: MEASURED camera calibration when the device is available      │
│  WHY: degradation is a known operator. Estimate it; never invert it.     │
└──────────────────────────────┬───────────────────────────────────────────┘
                               ▼
┌── L2  REPRESENTATION ───────────────────── REPLACES cosine embedding ────┐
│  ┌────────────────────────────┐   ┌──────────────────────────────────┐   │
│  │ PROPOSAL   w600k_r50       │──▶│ PERSON MODEL  q(z | O)           │   │
│  │ KEPT, DEMOTED              │   │ posterior, not point             │   │
│  │ fast · retrieval · init    │   │ z_geo / z_refl / φ_age           │   │
│  └────────────────────────────┘   └──────────────────────────────────┘   │
│  WHY: a point cannot express how much identity evidence an image carries │
└──────────────────────────────┬───────────────────────────────────────────┘
                               ▼
┌── L3  INFERENCE ───────────────────────────────────────────── NEW (S3) ──┐
│  band-limited comparison  →  forward render → degrade → likelihood       │
│  ALWAYS in observation space. The hypothesis moves to the evidence.      │
└──────────────────────────────┬───────────────────────────────────────────┘
                               ▼
┌── L4  EVIDENCE ENGINE ──────────────────── PARTLY BUILT (metrics done) ──┐
│  condition-conditional calibration · population model · LR · Cllr        │
│  capacity guard · counter-hypothesis · conformal set · H_unknown         │
└──────────────────────────────┬───────────────────────────────────────────┘
                               ▼
┌── L5  MEMORY ──────────────────────────────────────────────── NEW (S4) ──┐
│  replayable evidence log → posterior is a pure function, never mutated   │
└──────────────────────────────┬───────────────────────────────────────────┘
                               ▼
                     API · PDF · court report
```

### What stays, what goes

| Component | Decision | Reason |
|---|---|---|
| SCRFD detector | **Keep** | Measured, adequate, not the bottleneck |
| `w600k_r50` embedding | **Keep, demote to proposal** | Fast retrieval and inference initialiser. It is not the identity representation |
| Cosine + threshold | **Replace** | Cllr shows the decision layer is not where the loss is; the representation is |
| Exact `IndexFlatIP` | **Keep to 10⁵, then DiskANN/RaBitQ** | §7d already established recall, not speed, decides this |
| `ImageQualityFilter` | **Replace with ISO/IEC 29794-5 / OFIQ 2** | Standards are verifiable by a procurement officer; bespoke metrics are not |
| Hash-chained audit | **Keep, extend to lineage DAG** | Already the right idea, needs to become the data model |
| `forensics/metrics.py` | **Built, keep** | Cllr/PAV/Tippett, validated |
| Threshold config | **Keep as fallback path** | Must survive as the degraded-mode answer when inference is unavailable |

---

## 2. Identity representation

### Formulation

A person is a conditional generative model; the representation is a posterior
over its parameters.

$$z = (z_{\text{geo}},\, z_{\text{refl}},\, \phi_{\text{age}}), \qquad
q(z \mid O) \;\text{for observation set}\; O = \{(I_k, \nu_k)\}$$

with nuisance $\nu = (\text{pose}, \text{illum}, \theta_{\text{deg}})$ where
$\theta_{\text{deg}}$ is the degradation operator's parameters (PSF σ, JPEG Q,
downsample factor, noise level).

Comparison is a marginal likelihood ratio, never a distance:

$$\text{LR} = \frac{\displaystyle\int p(I \mid z_X, \nu)\, p(\nu \mid \text{meta})\, d\nu}
{\displaystyle\int\!\!\!\int p(I \mid z, \nu)\, p(z)\, p(\nu)\, dz\, d\nu}$$

**The denominator is empirical, not generative.** We do not model the population
with the renderer. We *measure* it by scoring against a declared reference cohort.
This is a deliberate risk reduction: the generative model is needed only for the
numerator. Our §0 finding 3 is why — a modelled denominator would inherit every
purity failure we just demonstrated, silently.

### Training objective

$$\mathcal{L} = \underbrace{-\log p(I \mid z, \hat\nu)}_{\text{reconstruct}}
+ \lambda_1 \underbrace{\mathcal{L}_{\text{margin}}(z)}_{\text{proposal stays sharp}}
+ \lambda_2 \underbrace{\|\hat\theta - \theta\|^2}_{\text{recover operator}}
+ \lambda_3 \underbrace{d\big(z(I), z(D_\theta I)\big)}_{\text{invariance}}
+ \lambda_4 \underbrace{C_{llr}}_{\text{proper scoring rule}}
- \lambda_5 \underbrace{\mathcal{L}_{\text{adv}}(\text{cond} \mid z)}_{\text{leakage penalty}}$$

**The design decision that makes this tractable: supervised nuisance, not
unsupervised disentanglement.** For any degradation we apply ourselves, $\theta$
is ground truth. $\lambda_2$ and $\lambda_3$ therefore demand invariance to a
*specific, parameterised, measured* transformation group — a far weaker claim than
general factorisation. Making $\theta$ an output as well as an input prevents the
model from satisfying $\lambda_3$ by ignoring degradation.

$\lambda_4$ is novel: nobody trains face recognition with Cllr as a loss. It is a
proper scoring rule, so it optimises calibration and discrimination jointly, and
it is the deployment metric.

$\lambda_5$ targets the failure this project already documented. The QMUL control
measured condition leakage of **0.600 − 0.522 = +0.078**. That is the quantity to
drive to zero, and the harness exists.

### Inference

1. Proposal: $\hat z, \hat\nu \leftarrow$ encoder (milliseconds)
2. Retrieval: ANN over proposal embeddings → top-K candidates
3. Refinement: gradient descent through the differentiable forward model, top-K only
4. Marginalisation: Laplace approximation or a short SVI run over $\nu$
5. LR against the declared reference cohort

### Limitations, stated

- Posterior is approximate; full marginalisation over $\nu$ is intractable.
- $\phi_{\text{age}}$ needs longitudinal data that barely exists (§3C).
- $z_{\text{refl}}$ is unrecoverable below ~40 px. At CCTV resolution only
  $z_{\text{geo}}$ is constrained, and the posterior must reflect that.
- The factorisation is *assumed*, and identity genuinely affects wrinkles, skin
  and expression habits. We measure leakage rather than assert its absence.

---

## 3. Data strategy

### A) Already on disk — use immediately

| Asset | Location | Purpose |
|---|---|---|
| **IJB-B + IJB-C, complete** | `Downloads/ijb-testsuite.tar` (8.6 GB) | Full suite: `loose_crop`, meta, `IJB_11.py`, reference ArcFace `.npy`. **Our largest unrun benchmark.** Docs wrongly call it a 1.57 GB partial |
| **TinyFace Testing_Set** | `src_extracted/tinyface/` | 8,171 labelled + **153,428 `Gallery_Distractor`** — a real 1:N benchmark, unembedded |
| **TinyFace Training_Set** | same | ~10.4k native-LR training images, **never used** |
| QMUL-SurvFace | `Downloads/QMUL-SurvFace-v1` | 5,319 native-LR identities; research licence |
| Protocol packs ×3 | `src_extracted/faces_*` | LFW/AgeDB/CFP/CALFW/CPLFW; three `cfp_fp.bin` variants |
| Cached embeddings | `runtime/benchmarks/embeddings/` | 7 datasets × models; enables GPU-free analysis |

### B) Must acquire

| Dataset | Purpose | Gate |
|---|---|---|
| **IJB-S** | *The* surveillance benchmark. Our niche is defined against it | Data agreement, months — **start week 1** |
| **BRIAR** | Long-range, multimodal, real government collection | Agreement — start week 1 |
| SCface, UCCS | Real multi-distance surveillance capture | Registration |
| WebFace42M / Glint360K | Proposal-network pretraining at scale | Open |
| VoxCeleb1/2 | The only large paired face+voice video corpus | Open |
| FaceScape, NeRSemble | **Renderer training.** The binding constraint on Stage 3 | Academic agreement |
| Multi-PIE | Controlled pose × illumination — ground-truth nuisance | Licence fee |
| MORPH, CACD, FG-NET | Aging flow $\phi_{\text{age}}$ | Mixed |
| DigiFace-1M | Renderer augmentation **only** — documented synthetic-real gap makes it unfit as identity training data | Open |

### C) Must create ourselves — the differentiated assets

1. **Camera degradation corpus.** Measured PSF, MTF, noise model and compression
   pipeline for the camera models that actually appear in casework. Slanted-edge
   MTF, a test chart, a bench. **No such public corpus exists.** It is the input
   L1 needs, it is achievable with patience rather than compute, and it would be
   cited for a decade.
2. **Forensic reference cohorts.** Demographically declared populations for LR
   denominators. §0 finding 3 says this is the measurement.
3. **Paired HR/LR same-scene capture.** Same subjects, same moment, one HR camera
   and one real CCTV unit at 5/10/20/50 m. This is the only data that lets us
   validate forward-model fidelity against reality rather than against synthesis.

---

## 4. Model development roadmap

### Stage 0 — Cheap falsification (weeks 1–8)

**Objective.** Decide the generative bet before spending years, and repair the
capacity measurement.

**S0.1 — Embed the 153,428 TinyFace distractors.** Gives a genuine open-set 1:N
benchmark with a real reference population. Repairs the measurement everything
else depends on. *~4–8 GPU-hours.*

**S0.2 — Run IJB-B/IJB-C.** Validate our harness against the shipped reference
`.npy` files before trusting our own numbers. *~1–2 GPU-days.*

**S0.3 — Renderer-free observation-space test.** The decisive experiment. No
renderer: **a real HR image is the person model.**

| Arm | Method |
|---|---|
| A | Embed LR probe, embed HR gallery, cosine — *the current paradigm* |
| B1 | Downsample gallery to probe resolution, then compare |
| B2 | Estimate *this probe's* PSF/JPEG/noise, apply **that** operator to the gallery, then compare |
| B3 | **MTF-band-limited comparison** — project both sides onto the spatial-frequency band the imaging chain actually passed |
| C | Likelihood in pixel space under the estimated noise model |

**Success:** B2 − B1 ≥ 2 points TAR@FAR=0.1% on TinyFace **and** QMUL, with
bootstrap CIs excluding zero.
**Failure:** B2 ≈ B1 ≈ A. Then the forward-model thesis is dead, Stage 3 is
cancelled, and the programme becomes Stages 1–2 + 4 only.

**S0.4 — Condition-leakage baseline.** Measure the QMUL statistic across all
corpora. Establishes the $\lambda_5$ target.

### Stage 1 — Upgrade what exists (months 2–5)

**Objective.** Ship a materially better system on the current representation, so
value lands regardless of Stage 0's verdict.

Condition-conditional calibration (quality-binned) · population model and
`H_unknown` · conformal candidate sets · counter-hypothesis engine · OFIQ 2
quality · DiskANN+RaBitQ above 10⁵ · evidence block in API and PDF.

**Success:** open-set TPIR@1% FPIR on TinyFace-distractor beats the top-10 baseline;
conformal coverage within ±2% of nominal; Cllr on IJB-S ≤ 0.85.
**Failure:** conformal coverage cannot be achieved → the population model is wrong,
return to §3C.

### Stage 2 — Representation learning (months 4–14)

**Objective.** Attack the 98.7% discrimination loss directly.

Train the proposal network with the full $\mathcal{L}$ minus reconstruction:
margin + operator regression + invariance + Cllr + leakage penalty. Backbone: ViT-B
or Swin-B initialised from a face-SSL checkpoint. Train on WebFace42M with
*measured* degradation operators from §3C, plus TinyFace `Training_Set` and QMUL.

**Success:** TinyFace TAR@FAR=0.1% ≥ 45% (from 33.13%) with clean-set regression
≤ 0.5 points, and leakage ≤ 0.02 (from 0.078).
**Failure:** degraded gain costs > 2 points clean → revert; the condition-routing
fallback already banks +4.23.

### Stage 3 — Generative inference (months 12–36) — **CONTINGENT ON S0.3**

Rung 1: 3DMM-class differentiable renderer. Rung 2: neural person-conditional
renderer. **Escalate only on a measured gain at the previous rung.**

**Success:** ≥ 5 points TAR@FAR=0.1% on IJB-S over Stage 2, with hallucination
null-tests passing.
**Failure:** rung 1 shows no gain over Stage 2 → stop. Do not build rung 2.

### Stage 4 — Evidence integration (months 6–30, parallel)

Capacity guard wired as a runtime assertion · replayable identity memory ·
retraction propagation · court report generation · ENFSI-format validation.

### Stage 5 — Production (months 24–36)

Tiered cascade: ANN retrieval (ms) → generative refinement on top-K only
(seconds). NIST FRTE 1:1, 1:N, and FATE Quality submissions.

---

## 5. Training plan

| Component | Source | Rationale |
|---|---|---|
| Detector | SCRFD, frozen | Measured, not the bottleneck |
| Backbone init | Face-SSL or DINOv3 ViT-B | FRoundation: fine-tuned foundation models match from-scratch at lower cost |
| Proposal net | Fine-tune from `w600k_r50` | Preserves a known-good starting point; our own fine-tune failure showed training from weaker inits is worse |
| Degradation estimator | **From scratch** | Small CNN, supervised on synthetic operators with ground-truth $\theta$ |
| Renderer | From scratch on 3D data | No suitable pretrained forensic renderer exists |
| Calibrator | Fitted per condition bin | 2 parameters; flexible calibrators overfit and an overconfident LR is the worse failure |

**Self-supervised:** masked-region and multi-view consistency on unlabeled
surveillance video, for the backbone only.
**Synthetic:** degradation operators — yes, with measured parameters. Synthetic
*identities* — renderer augmentation only. Our own §6d result and the published
synthetic-real gap both say synthetic identity training does not transfer.
**Real-world validation:** every claim re-measured on IJB-S / BRIAR / SCface, which
the model never trained on.

---

## 6. Low-resolution: the core solution

At 10–30 px the information is in **low-frequency geometry**. Reflectance and fine
landmarks are gone — not degraded, *absent*. Four components, none of them
super-resolution.

**(a) Estimate the operator.** Blind PSF, noise level, JPEG quantization table
recovery, motion-blur kernel. Feeds the forward model. Estimation is well-posed;
inversion is not.

**(b) MTF-band-limited comparison.** Compute the imaging chain's frequency
cutoff and project *both* probe and gallery onto that passband before comparing.
We never compare information the channel did not pass. This is the sharpest
implementable form of the whole thesis, it needs no renderer, and it is arm B3 of
S0.3.

**(c) Measured camera calibration.** When the physical device is available —
which in casework it often is — measure PSF, MTF and compression rather than
estimating them, and recover absolute face size by scene photogrammetry. This
converts an unknown operator into a documented instrument response and turns
face size into a *constraint on craniofacial geometry*. No vendor does this
because vendors sell software and do not attend scenes.

**(d) Multi-frame, physically justified.** N aliased frames with sub-pixel jitter
genuinely contain more information than one — classical MFSR theory, real
information gain, not prior-borrowing. Must be kept strictly separate in code and
in reporting from generative restoration.

**Uncertainty.** The posterior stays wide when the passband is narrow. The system
reports *"insufficient information"* by construction, not by policy.

**Anti-hallucination, enforced:** comparison in observation space only (type-level
separation) · capacity guard as a runtime assertion · prior-leakage null test per
case.

---

## 7. Forensic evidence engine

| Piece | Status | Note |
|---|---|---|
| Cllr / Cllr_min / Cllr_cal, PAV, Tippett | **Built** | `forensics/metrics.py` |
| Logistic + conditional calibration | **Built** | `forensics/calibration.py` |
| Capacity / bits / gallery bound | **Built, needs valid population** | `forensics/information.py` |
| Population model & RMP | Design | §3C — the critical path |
| `H_unknown` + conformal sets | Stage 1 | |
| Counter-hypothesis | Stage 1 | Doppelgänger search, condition-confound test |
| Court report | Stage 4 | ENFSI verbal scale, declared cohort, stated limits |

**How this differs from face recognition:** a matcher returns "who is closest",
a question that always has an answer. This returns evidential weight against a
declared population, with an explicit not-enrolled hypothesis and the option to
abstain. The rank-1 candidate for an unenrolled probe is structurally
indistinguishable from a true hit in every deployed 1:N system. That is the
property we are eliminating.

---

## 8. Memory and continual learning

**The posterior is a pure function of an append-only evidence log, never mutated
state.** Requirement driving it: if a linkage is disproven in court, every
downstream conclusion must be exactly recomputable without it.

Contamination control: every write carries its own LR · **no autonomous writes**
(automated matches quarantined pending human adjudication, or the system
bootstraps its errors into ground truth) · bimodality monitoring flags dossiers
that may contain two people · retraction propagates and reports which conclusions
changed.

---

## 9. Evaluation plan

| Tier | Sets | Role |
|---|---|---|
| Comparability | LFW, CFP-FF/FP, AgeDB-30, CALFW, CPLFW | Saturated; regression-detection only |
| Scale | IJB-B, IJB-C | On disk, unrun |
| **Decision-grade** | **IJB-S, TinyFace(+distractors), QMUL, SCface, BRIAR** | Where the programme is judged |

**Primary metric is Cllr, not accuracy.** Accuracy measures a decision system;
this is not one. Secondary: TAR@FAR, CMC, open-set TPIR@FPIR, conformal coverage,
capacity in bits, ECE, **hallucination null-tests**, and demographic differentials
as a **CI gate**.

**Success at 3 years:** IJB-S TAR@FAR=0.1% competitive with published leaders;
Cllr ≤ 0.5 on surveillance data; conformal coverage within ±2%; capacity bound
validated as a failure predictor on held-out populations; NIST FRTE submitted;
external forensic-science acceptance of the reporting framework.

---

## 10. Compute, storage, team

| | Prototype (S0–S1) | Research (S2–S3) | Production |
|---|---|---|---|
| GPU | 1× existing | 8–16× A100/H100 | 2–4× inference |
| Storage | ~2 TB | 50–100 TB | ~10 TB |
| Wall-clock | 8 weeks | 18–30 months | 6 months |
| Inference | 15 ms embed | — | ms retrieval → sec refinement |
| **People** | **1–2** | **8–15** | **4–6** |

Stage 0 and Stage 1 are genuinely achievable by one or two people. **Stage 3 is
not.** That gap is the central resourcing fact of this plan.

---

## 11. Risks

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | **Generative inference does not beat discriminative** | Programme-ending for Stage 3 | S0.3 decides it in 8 weeks, before commitment |
| 2 | **Forward-model mismatch produces confidently wrong LRs** | Highest — it is *silent* | Capacity guard, prior-leakage null test, calibration validated on unseen operators, §3C paired HR/LR real capture |
| 3 | **Population model impurity** | High — *already demonstrated twice* | Identity-labelled cohorts only; refuse to emit capacity without them, as the code now does |
| 4 | Data agreements (IJB-S, BRIAR) denied or slow | High | Apply week 1; TinyFace distractors + QMUL as fallback |
| 5 | Disentanglement fails; condition leaks into identity | High | $\lambda_5$ + the QMUL leakage metric as a CI gate, not a hope |
| 6 | Compute for Stage 3 | Medium | Rung-gated escalation; no rung without a measured gain |
| 7 | Legal — population models edge toward ancestry inference; kinship is familial search | High | Declared, auditable strata never inferred from the probe. **Do not build kinship** |
| 8 | Team size vs ambition | **Highest, structurally** | §12 |

---

## 12. Final recommendation

### A) Build

**Stages 0, 1, 2, and 4.** Concretely: repair the capacity measurement, run the
benchmarks already on disk, decide the generative bet with S0.3, ship the evidence
engine, and attack the 98.7% discrimination loss through representation learning
with the operator-supervised objective.

This is a coherent, fundable, publishable programme with a working system at every
milestone. It produces at least four papers — capacity bounds as an admissibility
criterion, Cllr-as-a-loss, correlated-evidence fusion, and the camera degradation
corpus — none of which require the renderer.

### B) Do NOT build

- **A forensic-grade neural renderer in year 1.** It is a face world-model. Even
  with S0.3 positive, go to rung 1 (3DMM) and stop until a lab exists.
- **Kinship / population-substructure modelling.** Scientifically interesting,
  legally radioactive, and it implicates people who never consented.
- **Gait, behaviour, or "movement pattern" biometrics.** The bits are not there.
- **Distributed vector search before 10⁵ templates.** §7d already settled it.
- **Anything for airport gates.** Serving both forensics and access control
  compromises the architecture into neither.

### C) Experiments that must happen first

1. **S0.3 arms A/B1/B2/B3** — decides Stage 3, 8 weeks.
2. **TinyFace distractor embedding** — repairs the measurement everything rests on.
3. **IJB-B/C against the shipped reference results** — validates our harness before
   we trust our own numbers.
4. **Condition-leakage baseline** — sets the $\lambda_5$ target.

### D) First 90 days

| Weeks | Work | Deliverable |
|---|---|---|
| 1 | Apply for IJB-S and BRIAR. Extract `ijb-testsuite.tar` | Applications filed |
| 1–2 | Embed 153,428 TinyFace distractors | Real open-set benchmark; valid capacity |
| 2–4 | Run IJB-B/C, validate against reference `.npy` | Harness trusted; largest gap closed |
| 3–6 | **S0.3** all five arms, TinyFace + QMUL, bootstrap CIs | **Go/no-go on Stage 3** |
| 5–8 | Condition-conditional calibration; leakage baseline | Cllr on unseen surveillance data |
| 7–10 | Population model v1 on TinyFace distractors; `H_unknown` | First defensible LR |
| 9–12 | Conformal sets; counter-hypothesis v1; write up capacity | Paper 1 submitted |

### E) Three-year roadmap

**Year 1** — Stages 0–1 complete. Evidence engine in production. IJB-B/C/S
measured. Papers 1–2 (capacity bounds; calibration on surveillance data). Stage 3
decided.
**Year 2** — Stage 2 representation learning. Camera degradation corpus published.
Memory system. Papers 3–4. NIST FATE Quality submitted.
**Year 3** — Stage 3 rung 1 if licensed by S0.3. NIST FRTE 1:1 and 1:N. External
forensic validation. Production hardening.

### The critical judgement

**The dream architecture is unrealistic at current team size, and pretending
otherwise would be the most expensive mistake available.** Stage 3 needs 8–15
people for 18–30 months. One person with AI assistance cannot build a face
world-model, and attempting it would consume the years that Stages 0–2 and 4 would
have converted into a defensible, published, field-defining position.

So the recommendation is deliberately asymmetric: **be world-class at the
measurement science and merely competitive at the model.** The measured evidence
supports this. Calibration is already near-optimal (Cllr_cal ≤ 0.022) — that work
is nearly done and it is genuinely ahead of the field. The discrimination gap
(98.7% of TinyFace loss) is where NEC and the BRIAR performers have structural
advantages we cannot match with public data and one GPU.

But **nobody is competing on the evidence layer.** The capacity bound, the
population model, correlated-evidence fusion, the retraction-safe memory, and the
camera degradation corpus are all unclaimed, all achievable at this scale, and all
things a court needs and a vendor will not build.

Run S0.3. If it is positive, you have a licence to raise money and hire a lab for
Stage 3. If it is negative, you have saved two years and still hold the strongest
forensic evidence engine in the field.

Either outcome is a win. That is what makes it the right first move.
