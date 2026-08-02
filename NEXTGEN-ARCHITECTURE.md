# HELMHOLTZ — a next-generation forensic identity architecture

**Status: design document. Nothing here is built except where marked.**
**Author: architecture proposal, 2026-08-02. Supersedes no existing document; ROADMAP.md remains the plan of record until this is accepted or rejected.**

Named for Helmholtz's *unconscious inference* — the 1867 proposal that perception
is not pattern matching but inference about the hidden causes of sensation. That
is the entire thesis of this architecture, and the reason it is not another
embedding network.

---

## 0. The claim, stated so it can be attacked

> Current face recognition is a discriminative shortcut around a generative
> process. It learns a direct map from images to a metric space, which works
> when test-time nuisance resembles training and fails structurally when it does
> not. The replacement is Bayesian inference over an explicit generative model of
> persons and imaging, with the discriminative network demoted from *answer* to
> *proposal*.

If that claim is false — if analysis-by-synthesis cannot beat a discriminative
embedding on degraded imagery once both are given equal data and compute — then
this architecture collapses to "a good evidence layer on ArcFace." That is still
worth building, and §13 defines the cheap experiment that decides it. Run that
experiment before committing years.

---

## 1. Why the current pipeline is at its limit

```
Image → detector → alignment → ArcFace embedding → cosine → threshold
```

This is not one pipeline. It is **five consecutive irreversible commitments**,
each of which destroys exactly what forensic inference needs.

| Stage | What it destroys | Why forensics needs it |
|---|---|---|
| Detection | The scene | Absolute scale, camera geometry, other observations of the same person |
| Alignment to 112×112 | Imaging metadata | PSF, motion blur direction, true face size in pixels, sensor MTF, compression history |
| Embedding to fixed 512-d | Information content | A 1000px portrait and a 19×14 crop emit the same 512 floats — the representation asserts they carry equal information, which is false by many bits |
| Cosine similarity | Population structure | Local density around the candidate; how *common* this face is; who else it resembles |
| Threshold | Everything but one bit | Evidential weight, uncertainty, the possibility of "insufficient" |

An astronomer does not discard the point-spread function before doing photometry;
it is the single most important thing they know about the instrument. Stage 2
discards the equivalent as preprocessing, and the field then treats degraded
imagery as an intractable modeling problem rather than as a **known operator
applied in the forward direction**.

### The evidence that this is formulation-limited, not compute-limited

Clean-image recognition is finished: NEC reports 0.06% error against 12M
identities. Surveillance recognition is not: the published IJB-S leaders sit at
**63.44% rank-1** (KP-RPE) and **44.81%** (FaceMoE, surveillance-to-surveillance).
This project measures **33.13% TAR@FAR=0.1%** on TinyFace against 99.70% on LFW.

A decade of architectural progress moved one benchmark three orders of magnitude
and the other by fractions. That is the signature of a formulation problem. More
parameters will not close it.

### Every hidden assumption, and the one beneath them

The seven assumptions are not independent. They are consequences of a single
commitment: **recognition is a discriminative function from images to a metric
space, learned by proxy on a closed set.**

| Assumption | Why it follows from the root | Status in HELMHOLTZ |
|---|---|---|
| Identity is a fixed vector | It is the bottleneck of that function | **Replaced** — §2 |
| Images are independent | A function is memoryless | **Replaced** — §4, §5 |
| Recognition is image matching | The function's domain is images | **Replaced** — §3 |
| One backbone for all conditions | It *is* one function | Replaced, but this is the least interesting one |
| Similarity = evidence | It is the function's output | **Replaced** — §9 |
| Identity is static | A function is stateless | **Replaced** — §5 |
| Only face information matters | The function's domain is crops | Relaxed — §6 |
| Training and deployment are separate | Functions are fitted then frozen | **Replaced** — §5 |

Attacking these individually produces incremental work. MoE attacks "one
backbone." Set aggregation attacks "independent observations." Each leaves the
root intact. HELMHOLTZ attacks the root.

---

## 2. Module 1 — The Person Model (identity representation)

**A person is not a vector. A person is a conditional generative model, and the
"embedding" is a posterior over its parameters.**

```
PersonModel := (  z_geo,     craniofacial geometry — the near-invariant substrate
                  z_refl,    reflectance / texture identity component
                  z_dyn,     idiosyncratic motion signature (video only)
                  φ_age,     an aging flow: dz/dt = f(z, t; φ)
                  q(·),      posterior over all of the above
                  O,         the observation set that produced q
                  L          the append-only evidence log
               )
```

Four properties that no current representation has:

**(a) It is a distribution, not a point.** `q` has support. A 19×14 crop yields a
posterior barely narrower than the population prior — which is the *truth*, and
which the system can then report. A fixed vector cannot express "this observation
constrains identity very little."

**(b) Effective dimensionality is data-dependent.** The posterior's effective rank
is a function of how much the observations actually constrain. Information content
is represented, not assumed constant.

**(c) Identity is factored by physical invariance, not by convenience.** The
identity-bearing signal is concentrated in `z_geo` — bone structure, genuinely
near-constant in adults — while current embeddings are dominated by appearance
statistics that are largely nuisance. The representation weights the signal
roughly *inversely* to its identity content. Explicit factorization fixes this,
and it is what makes low-resolution inference tractable: a 20×20 face carries
almost no reflectance detail but still constrains coarse geometry.

**(d) Identity is a trajectory, not a location.** `φ_age` is a learned flow.
Cross-age comparison is transport along the flow, not distance between endpoints.

### What "understanding a person the way a human recognizes someone familiar" means

Humans exhibit a stark asymmetry: near-perfect on **familiar** faces under extreme
degradation, and **worse than machines** on unfamiliar pairs (20–30% error in
matching studies). The difference is not a better feature extractor. Familiarity
is built from many observations across many conditions, from which the brain
constructs something like a person-specific generative model.

**Every deployed face recognition system performs the task humans are bad at.** It
compares two unfamiliar images and has no mechanism for becoming familiar with
anyone. The Person Model is that mechanism.

---

## 3. Module 2 — Generative identity inference

Recognition becomes: *does there exist a setting of nuisance variables under
which my model of person X could have produced this observation, and how much
more probable is that than under the population model?*

```
                    ┌─── amortized proposal (the old discriminative net) ───┐
                    │   encoder ê(I) → initial ẑ_id, ν̂                      │
                    └───────────────────────┬──────────────────────────────┘
                                            ▼
  hypothesis z_X ──► R(z_X, ν) ──► D_θ(·) ──► Î   ≈?   I_observed
                     renderer      degradation     compare HERE, in the
                     (geometry,    operator        DEGRADED space, always
                      reflectance, (PSF, blur,
                      pose, light)  noise, JPEG)
                                            │
                                            ▼  gradient refinement of ν, z
                                        converged
                                            ▼
        LR = ∫ p(I|z_X,ν)p(ν|meta)dν  /  ∫∫ p(I|z,ν)p(z)p(ν) dz dν
                                             └── population model, §8
```

The discriminative network does not disappear. It becomes the **amortized
inference proposal** — fast, approximate, and refined by gradient descent through
a differentiable renderer. This is what makes the approach tractable in 2026 and
was impossible in 1999.

### Why this did not work in 1999 and may work now

The strongest objection to this entire document is that Blanz & Vetter's 3D
Morphable Model *is* analysis-by-synthesis, and deep discriminative learning
crushed it. Anyone proposing this must answer why it fails differently now.

| 1999 failure cause | Status in 2026 |
|---|---|
| Generative model was linear PCA over ~200 laser scans | Diffusion models, 3D Gaussian splatting, NeRF-class geometry over millions of faces |
| Optimization: minutes per image, local minima | Amortized proposal + differentiable rendering; sub-second |
| Rendering non-differentiable | Fully differentiable rasterization and splatting |
| Could not absorb large data | Trains at WebFace260M scale |

All four causes were engineering, and all four have inverted. The Bitter Lesson is
usually read as "structure loses to scale"; the honest version is "structure that
*cannot absorb* scale loses." This structure now can. **That is the bet.**

### Never hallucinating — three enforced mechanisms

The user requirement is absolute: the system must never invent facial detail that
does not exist in the evidence. Three mechanisms, in increasing strength:

**(1) Architectural invariant — comparison happens in observation space, only.**
The generative model renders *down* to the evidence. The evidence is never
rendered *up*. We evaluate `p(I_obs | z, ν)`; we never construct `Î_hi` and
compare embeddings of it. These are not two ways of doing the same thing:
super-resolution is one-to-many and picks a plausible mode from a learned prior,
so matching against it is *matching against the generator*. Forward rendering is
many-to-one and well-posed.

Enforced in code by type-level separation: restored or upsampled imagery carries
a distinct type that the likelihood path structurally cannot accept. (The existing
`draw_enhanced_pair()` raising `NotImplementedError` is the seed of this
discipline; here it becomes a compile-time property of the whole system.)

**(2) Information-theoretic runtime guard — novel.** §9 estimates the identity
information `Î(q)` available in an observation of quality `q`. The reported LR is
asserted against that bound: a hypothesis cannot be supported more strongly than
the pixels can support. **If the generative prior leaks into the conclusion, the
LR will exceed the channel capacity of the evidence, and the guard fires.** This
converts an abstract safety property into a runtime assertion.

*Honest limitation:* the bound is an expectation over comparisons, not a hard
per-case limit, so it is a strong heuristic guard rather than a proof. It should
be calibrated to fire conservatively and every firing must be logged, not
silently clamped.

**(3) Prior-leakage null test.** Re-run inference with the identity term ablated —
the renderer conditioned on the population mean rather than `z_X`. If substantial
"evidence" survives, it came from the prior. Run per case, reported in the record.

---

## 4. Module 3 — Video-native recognition

The atomic unit of the system is the **observation set**, not the image.

```
video ─► detect+track ─► per-frame condition estimation ν̂_t
                          (PSF, blur, pose, illumination, occlusion)
              │
              ▼
        evidence selection ─── NOT "pick the sharpest frame"
              │                but: pick the subset that maximizes
              │                expected information gain, which favours
              │                DIVERSITY of ν, not quality alone
              ▼
        sequential posterior update  q(z | O_1..T)
              │
              ▼
        one coherent identity observation with one LR
```

### The correlated-evidence problem — subtle, and universally done wrong

A face tracked across 300 frames is **one observation of one person, not 300
matches**. Naive fusion multiplies 300 likelihood ratios and inflates the reported
evidence by orders of magnitude. This is not a rounding error; it is the
difference between LR = 10³ and LR = 10³⁰⁰.

Frames within a track are **conditionally dependent given identity** because they
share nuisance: same camera, same lighting, same seconds, same pose regime. The
correct treatment is a hierarchical model with a per-track latent nuisance
variable that is marginalized out, yielding an **effective number of independent
observations** far below the frame count.

This is a genuine research contribution (§11). It is also the single most likely
place for a competitor system to produce confidently wrong forensic testimony.

---

## 5. Module 4 — Persistent identity memory

```
Person = replayable evidence log  ──►  deterministic posterior
         (append-only, retractable)     (recomputed, never mutated)
```

**The posterior is never mutated in place. It is a pure function of the log.**
This is the design decision that makes the memory forensically usable, and it
follows from a requirement no current system meets:

> If a linkage is disproven in court, every downstream conclusion must be exactly
> recomputable without it.

Mutable template stores cannot do this. A replayable log can: retract observation
`o_i`, recompute, and produce the counterfactual conclusion set.

### Preventing memory corruption — five mechanisms

1. **Every write is itself an inference.** An observation enters a Person Model
   only with a computed LR for "this observation belongs to this person," and that
   LR is stored with it. Weak evidence contributes proportionally, not equally.
2. **No autonomous writes.** Automated matches are quarantined pending human
   adjudication. Only adjudicated links enter the identity prior. Without this the
   system bootstraps its own errors into ground truth — the failure mode that
   destroys any self-training identity system.
3. **Bimodality monitoring.** Periodic test of whether an observation set looks
   like one person or two. Automatic flag: *"this dossier may contain two
   individuals."* Contamination is detected, not assumed away.
4. **Provenance-weighted consolidation.** Consolidation policy is learned but
   constrained: it may reweight, never fabricate, and every consolidation step is
   itself logged.
5. **Retraction propagation.** Retracting an observation invalidates and
   recomputes every conclusion that depended on it, and the system reports which
   ones changed.

Continual learning stops being a defect to suppress and becomes a **consolidation
policy to design** — with a real cognitive-science literature (complementary
learning systems, replay, schema consolidation) behind it.

---

## 6. Module 5 — Multimodal identity, ranked honestly

Most multimodal biometrics papers overclaim. Here is my assessment of what
actually carries identity information, weighted by availability in real forensic
archives:

| Modality | Bits per observation | Degrades with range | Stable over time | Available in archives | Verdict |
|---|---|---|---|---|---|
| **Face** | Highest when >40 px | Fastest | Yes (with aging flow) | Almost always | Primary |
| **Body shape** (anthropometry) | Moderate | Slowly | **No** — weight, clothing | When full body visible | Strong at range; BRIAR demonstrates real gains |
| **Voice** | High | N/A | Yes | Rarely co-occurs with CCTV | Excellent when present; mature LR practice to borrow |
| **Gait** | Low (a few bits) | Slowly | Confounded by shoes, load, surface, injury | Needs clean lateral view | **Overclaimed in the literature.** Marginal |
| **Movement / behaviour** | Near zero | — | No | — | **Not identity.** Reject |
| **Clothing / appearance** | Zero | — | Minutes | Always | **Not biometric.** Re-identification, not identification |

**Fusion must happen at the LR level with modeled correlation, never at score
level.** Modalities observed in the same event share nuisance — same camera, same
lighting, same moment — so independent LR multiplication inflates evidence for
exactly the same reason as the video case in §4. One correlation model serves both.

---

## 7. Module 6 — The low-resolution programme

This is the capability that decides whether the system matters. Four components,
none of which is super-resolution.

### (a) Estimate the degradation operator, do not invert it

Blind estimation of PSF, noise level, motion blur kernel, and JPEG quantization
table from the image itself, plus metadata where available. The operator then
enters the forward model. Inversion is ill-posed and invites hallucination;
forward application is well-posed and safe.

### (b) Physical camera calibration — the forensic asymmetry nobody exploits

**In forensic casework you can often seize the actual camera.** No vendor exploits
this, because vendors sell software and do not attend scenes.

With the physical device you can *measure* rather than estimate: PSF from a test
target, sensor MTF, exact compression pipeline, lens distortion. With scene
photogrammetry you additionally recover absolute scale — the real-world size of a
face that subtends 22 pixels.

This converts the imaging operator from an unknown to a **known, measured,
documented instrument response**, and it converts face size from a nuisance into a
*constraint on craniofacial geometry*. It is the single highest-leverage idea in
this document that requires no new machine learning at all, and it is only
available to a system designed for investigations rather than for access control.

*Requires a casework partner.* Without one this remains theoretical.

### (c) Multi-frame information, physically justified

N frames with sub-pixel jitter genuinely contain more information than one — this
is classical multi-frame super-resolution theory and it is *real information gain*,
not prior-borrowing. It is the one legitimate route to detail recovery and it must
be kept strictly separate, in code and in reporting, from generative restoration.

### (d) Atmospheric turbulence, for long range

At 300–1000 m turbulence dominates. Modeling it in the forward operator is the
approach demonstrated by BRIAR-programme systems.

---

## 8. Module 7 — Open-world intelligence

Replace the ranked list with posterior inference over an open universe:

```
hypotheses:  H_1 ... H_N  (each enrolled person)   +   H_unknown

output:      • posterior over hypotheses
             • conformal candidate set with coverage guarantee
             • explicit P(H_unknown)
             • ABSTAIN when evidence is insufficient
```

> *"With 95% confidence the subject is one of these 47, or is not enrolled at all."*

Nearest-neighbour search answers "who is closest?" — a question that **always has
an answer**. The rank-1 candidate for an unenrolled probe is structurally
indistinguishable from a true hit. This is the most dangerous property of every
deployed 1:N system and it is definitional, not an implementation flaw.

`P(H_unknown)` requires the population model of §9 — this is why the two modules
are inseparable.

---

## 9. Module 8 — The scientific evidence engine

Retained from the prior design, now integrated rather than bolted on.

| Component | Function | Status |
|---|---|---|
| `metrics.py` | Cllr, Cllr_min, Cllr_cal, PAV, Tippett | **Built** |
| `calibration.py` | Condition-conditional score → log₁₀ LR | Specified |
| `information.py` | Identity bits; Fano bounds; max defensible gallery | Specified |
| `population.py` | Typicality — the LR denominator, and `P(H_unknown)` | Design |
| `counter.py` | Doppelgänger search, condition-confound test, disagreement map | Design |
| `evidence.py` | `EvidenceReport` + ENFSI verbal scale + explicit limits | Specified |

### Two integrations that are new here

**Information bounds become a runtime guard on the generative model** (§3), not
merely a reporting statistic. This is the mechanism that makes analysis-by-
synthesis safe enough to use forensically.

**The population model serves three consumers at once:** the LR denominator, the
`H_unknown` likelihood, and the doppelgänger counter-hypothesis. Building it once
resolves three separate gaps.

### Fano bounds — the statement no accuracy metric can make

With identity information `I` bits and gallery size `N`:

```
P_error  ≥  ( log₂N − I − 1 ) / log₂(N−1)
```

Which yields, per capture condition, a **maximum gallery size beyond which correct
identification is impossible for any algorithm, present or future**.

> *"This image contains at most 14 bits of identity information. A gallery of
> 10 million requires 23. Unique identification is information-theoretically
> impossible."*

That is a claim about physics, not about a vendor. It is the difference between a
field that reports benchmark scores and a field that can state its own
impossibility results.

---

## 10. Training strategy

### Datasets

| Purpose | Sets | Notes |
|---|---|---|
| **SSL pretraining** | WebFace260M, unlabeled video | Backbone + proposal network |
| **Supervised identity** | WebFace42M / Glint360K / MS1M-V3 | Trains the *proposal*, not the answer |
| **Surveillance eval** | IJB-S, BRIAR, TinyFace, QMUL-SurvFace, SCface, UCCS | BRIAR and IJB-S need data agreements — start now, they take months |
| **Surveillance train** | TinyFace `Training_Set`, QMUL, BRIAR train | **TinyFace's train split is already on this machine and has never been used** |
| **Video** | IJB-C video, YouTube Faces, VoxCeleb | VoxCeleb is the paired face+voice set |
| **3D / multi-view** | FaceScape, NeRSemble, Multi-PIE, FRGC | **Required for the renderer.** The binding constraint |
| **Longitudinal** | MORPH, CACD, AgeDB, FG-NET | Trains `φ_age`. Genuine decades-long data barely exists — a real field gap |
| **Synthetic** | DigiFace-1M, Arc2Face | Renderer augmentation **only**. Documented synthetic-real gap makes these unsuitable as identity training data |
| **Camera characterization** | **Collect it ourselves** | PSF/MTF/compression measurements from real CCTV units. **A data asset nobody has** |

That last row is worth as much as any model: a public corpus of measured
instrument responses for the camera models that actually appear in casework would
be cited for a decade and is achievable with a lab bench and patience.

### Stages

1. **SSL pretraining** on unlabeled faces and video → general facial representation
2. **Supervised margin training** → the amortized proposal network
3. **Generative model training** on multi-view/3D → person-conditional renderer
4. **Forward-operator training** on paired clean/degraded with *measured* operators
5. **Joint refinement** — end-to-end analysis-by-synthesis through the differentiable renderer
6. **Condition-conditional calibration** on held-out data
7. **Continual consolidation** — memory policy

### Evaluation — and the metric change that matters

**Cllr is the primary metric, not accuracy.** Accuracy measures a decision system;
this is not one. Secondary: standard verification (for comparability), IJB-B/C/S,
BRIAR, TinyFace, QMUL, open-set TPIR@FPIR, **conformal coverage validation**,
**capacity analysis in bits per condition**, hallucination null-tests, and
demographic differentials as a **CI gate rather than a report**.

---

## 11. Research novelty — engineering vs. science

### A) Engineering improvements (valuable, not novel)

MoE backbones · better margin losses · ANN indexing (DiskANN/RaBitQ) · score-level
multimodal fusion · foundation model + LoRA · tracking · template aggregation ·
quality routing. All worth doing. **None of it is a contribution.**

### B) Genuinely new scientific ideas

| # | Idea | Venue | PhD? | Company-defining? |
|---|---|---|---|---|
| 1 | **Information-capacity bounds as forensic admissibility criterion**, and as a runtime hallucination guard | Top-tier ML + forensic science | Yes | **Yes** |
| 2 | **Analysis-by-synthesis with measured, camera-calibrated forward operators** | CVPR/ICCV class | Yes, several | Yes |
| 3 | **Correlated-evidence fusion** — effective independent observation count for video and multimodal LR | Forensic statistics + ML | Yes | Moderate |
| 4 | **Retractable identity memory** — replayable log, exact recomputation on retraction | Systems + forensic science | Yes | **Yes** |
| 5 | **Population model of facial typicality** → true random match probability | Forensic science, high impact | Yes | Yes — but see §14 hazards |
| 6 | **Conformal open-set candidate sets with explicit H_unknown** | ML | Yes | Moderate |

**The company-defining combination is #1 + #4:** a system that states its own
limits and can prove what it concluded without a retracted piece of evidence.
Nobody occupies that category, and it is not reachable by scaling a matcher.

---

## 12. Comparison against the current leaders

| System | Where they win | Where HELMHOLTZ wins | Honest risk |
|---|---|---|---|
| **ArcFace / InsightFace** | Simplicity, throughput, ecosystem | Degraded imagery, uncertainty, evidence | If AbS fails, we are a slower ArcFace |
| **NEC** (FRTE #1, 0.06% @ 12M) | Raw 1:N accuracy at scale — proprietary data, decades of investment | Evidential weight, explainability, open-set honesty, casework integration | **We cannot beat them on accuracy and must not try** |
| **NIST FRTE leaders generally** | Validated accuracy, procurement trust | Everything above the matcher | We have no independent validation until we submit |
| **BRIAR performers (e.g. FarSight)** | Multimodal at range, turbulence, real government data | Evidence layer, memory, capacity bounds | **Closest technical competitor.** Without BRIAR data we cannot match their surveillance results |
| **Foundation-model approaches** | Cheap adaptation, strong general features | We use them — as the proposal network | Risk of being leapfrogged if a face foundation model closes the degraded gap discriminatively |

**Where this architecture may simply fail:** if the degraded-imagery gap turns out
to be closable by discriminative scaling, the generative machinery is expensive
overhead and the field routes around us. §13's experiment is designed to detect
that early.

---

## 13. Implementation roadmap

### Phase 0 — Falsify the core bet cheaply (1–2 months)

**Before anything else.** Build a minimal forward model: take the existing
`w600k_r50`, a simple 3DMM-class renderer, and a measured degradation operator.
On TinyFace and QMUL, ask one question:

> Does rendering a hypothesis down to the observed degradation and comparing in
> observation space beat cosine similarity in embedding space?

A null result here kills §2–§3 and redirects the whole programme to the evidence
layer. **This is the most valuable two months in the plan.** Do not skip it because
the architecture is attractive.

### Phase 1 — Evidence core (1–3 months, parallel, no dependencies)

`metrics.py` ✔ · `calibration.py` · `information.py` · `evidence.py` · fit and
cross-validate on the seven cached embedding sets · run IJB-B/IJB-C (**the full
8.6 GB suite is already on disk**) · publish the capacity analysis.

Ships value regardless of whether Phase 0 succeeds.

### Phase 2 — Population and open-world (3–9 months)

`population.py` · `counter.py` · conformal candidate sets · `H_unknown`.

### Phase 3 — Video-native and memory (6–18 months)

Tracking, evidence selection, correlated-evidence fusion, the replayable
identity log.

### Phase 4 — Generative core (12–36 months, contingent on Phase 0)

Renderer training, differentiable inference, joint refinement, the camera
calibration corpus.

### Phase 5 — Validation and publication (ongoing)

NIST FRTE 1:1 and 1:N, **FATE Quality** (the track where a small team can
realistically lead), and the six papers in §11.

---

## 14. Biggest technical risks, ranked

1. **Analysis-by-synthesis repeats its 1999 failure.** The central bet.
   *Mitigation:* Phase 0, before commitment.
2. **Forward-model mismatch produces confidently wrong LRs.** The most dangerous
   failure because it is **silent** — a miscalibrated score is visibly bad; a
   wrong likelihood from a wrong physical model looks authoritative.
   *Mitigation:* the §9 capacity guard, prior-leakage null tests, and calibration
   validated on data the forward model never saw.
3. **MI estimation is unreliable in high dimensions.** InfoNCE saturates at
   log₂K; a 3-bit error moves the gallery bound by 8×. *Mitigation:* report
   bounds not point estimates; validate against measured error rates.
4. **Data access.** BRIAR and IJB-S require agreements measured in months.
   Without them the surveillance claims cannot be validated. *Start now.*
5. **Compute.** The generative model is a hundreds-of-GPU-month proposition.
6. **Ethical and legal hazard in §9's population model.** A model of facial
   feature frequency by population is one query from ancestry inference; kinship
   structure is functionally familial search — the practice that made forensic
   DNA genuinely controversial because it implicates people who never consented.
   *Mitigation:* population strata must be a declared, auditable, contestable
   parameter, never inferred from the probe. I would leave kinship alone entirely
   regardless of its scientific merit.
7. **Team scale.** This is honestly a 10–20 person, 5-year programme. A smaller
   group should execute Phases 0–2 and publish, not attempt Phase 4.

---

## 15. What would make this genuinely revolutionary

Not accuracy. Three properties, none of which any existing system has:

**1. It states its own limits, provably.** Not "we are 99% accurate" but *"this
evidence supports at most LR = 10³, and here is the information-theoretic reason
it cannot support more."* A field that can state impossibility results has become
a science. Face recognition has never been able to do this.

**2. It can prove what it concluded without a retracted piece of evidence.** The
replayable identity log makes forensic conclusions auditable and reversible in a
way no template store can be.

**3. It never converts a prior into evidence.** The architectural invariant that
comparison happens only in observation space, enforced by type separation, guarded
by capacity bounds, and audited by null tests — so that the thing a court is shown
is always what the sensor recorded, never what the model imagined.

Any system that has those three properties will be trusted in contexts where a
more accurate system is not. That is the durable position, it is unreachable by
scaling a matcher, and it is available to a small group with unusual measurement
discipline.

---

## 16. Relationship to the existing project

**Preserved and load-bearing:** the measurement discipline (contamination audits
reported as floors not totals; the CFP-FP provenance diagnosis; the published
negative fine-tune; the QMUL artefact overturned by a control) — this is the
rarest asset here and every module above depends on it. Also: the hash-chained
audit model, lawful-basis enforcement, the threshold decision record, the
benchmark harness and its 9-fold/held-out protocol, and the architectural
exclusion of generative restoration from the evidential path, which §3 elevates
into the system's central safety invariant.

**Demoted:** `w600k_r50` from *answer* to *proposal network*. The quality router
from *feature* to *empirical evidence that condition-conditioning works on our
data*.

**Replaced:** the fixed embedding, the cosine threshold, the exact index, the
image-as-atomic-unit, and the template store.

**Discarded:** nothing of scientific value.
