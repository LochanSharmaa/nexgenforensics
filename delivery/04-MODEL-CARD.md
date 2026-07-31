# Model Card — NexGen iMATCH Facial Recognition System

**Document:** 04 of 14 · **System version:** 1.0.0 · **Issued:** 2026-08-01
**Applies to:** recognition pack `buffalo_l` / `w600k_r50`, detector SCRFD-10G,
optional degraded-condition pack `arcface_qmul_v2`

> Every figure in this document is reproducible from the delivered source. The
> command that produces each one is given beside it. No number appears here
> that was not measured on this system, on the stated dataset, with the stated
> protocol. Where a capability is unmeasured or unavailable, it is listed as
> such rather than omitted.

---

## 1. Purpose and intended use

### 1.1 What the system does

iMATCH performs **1:1 face verification** (are these two images the same
person?) and **1:N identification search** (which enrolled subjects most
resemble this probe?) over a gallery controlled by the operating
organisation.

### 1.2 Intended use

Generation of **investigative leads** for review by a qualified examiner,
within a documented lawful basis, in an organisation that operates an audit
trail.

### 1.3 Uses the system is NOT validated for

These are not warnings about misuse in the abstract; each corresponds to a
capability that has not been measured on this system and therefore cannot be
defended:

| Not validated for | Why |
|---|---|
| **Positive identification without examiner review** | The system returns a similarity score against a threshold. It does not establish identity. |
| **Fully automated decisions affecting a person** | No measurement supports unattended operation. |
| **Live video / real-time surveillance** | Never benchmarked on video. Latency figures are single-image. |
| **Age, sex, ethnicity or emotion inference** | Not implemented. |
| **Anti-spoofing / presentation-attack detection** | The liveness figure is a **heuristic**, explicitly `certified: false`. It is not a trained PAD classifier and must not be relied on to detect a presentation attack. |
| **Populations unlike the benchmark sets** | See §5 (demographic differentials) and §7. |

### 1.4 Required operating conditions

1. A stated **lawful basis** is recorded verbatim with every operation. The
   system does not evaluate whether a basis is lawful; it ensures one was
   stated and preserved.
2. Every automated result carries the notice: *"Automated face recognition
   returns investigative leads, not identifications. A qualified examiner must
   verify any candidate before it is relied upon."*
3. Audit logging is enabled and its hash chain is verified periodically.

---

## 2. Model architecture and provenance

### 2.1 Components

| Stage | Model | Provenance |
|---|---|---|
| Detection | SCRFD-10G (`det_10g.onnx`) | InsightFace `buffalo_l`, unmodified |
| Alignment | 5-point similarity transform → 112×112 | InsightFace standard |
| Recognition | ResNet-50 ArcFace (`w600k_r50.onnx`) | InsightFace `buffalo_l`, unmodified |
| Quality | `ImageQualityFilter` | Developed for this system |
| Liveness | Capture heuristics | Developed for this system, **not certified** |
| Matching | Cosine similarity, exact search | Developed for this system |
| Degraded pack *(optional)* | `arcface_qmul_v2` | Fine-tuned from `w600k_r50`; see §4.4 |

### 2.2 Statement of origin

**The deployed recognition weights are stock open-source InsightFace weights,
not weights trained by this project.** This is stated plainly because it is
material: the system's clean-image accuracy is the accuracy of the public
state of the art, and its provenance is public and independently verifiable.
The engineering contributions are the pipeline, quality gating, threshold
calibration, audit chain, and the routing described in §4.4.

### 2.3 Embedding

512-dimensional float32, L2-normalised. Comparison is cosine similarity, range
[−1, 1]. Templates are encrypted at rest with AES-256-GCM.

### 2.4 Deliberate architectural exclusion

**No generative restoration model (GFPGAN, CodeFormer, or similar) is present
anywhere in the path that feeds the embedding extractor.** Such models
hallucinate plausible detail that was never captured. In a forensic setting
that means inventing facial features and then matching against the invention.
This exclusion is architectural, not configurable.

---

## 3. Training data

The deployed recognition model was **not trained by this project**. The
following describes the upstream corpora, and the data used for the optional
degraded pack.

| Corpus | Role | Identities | Licence |
|---|---|---|---|
| MS1M / WebFace600K | Upstream training of `w600k_r50` | ~600K | Research use; see InsightFace |
| CASIA-WebFace | Clean anchor for `arcface_qmul_v2` | 9,880 (after exclusion) | Research use |
| QMUL-SurvFace | Degraded data for `arcface_qmul_v2` | 5,319 | **Research purposes only**; images sourced from person re-identification datasets, copyright with original owners |

### 3.1 Contamination control

Training/evaluation overlap was audited rather than assumed.

- **CASIA:** 692 of 10,572 identities (6.5%) matched an evaluation image at
  cosine ≥ 0.40 and were **excluded** before training.
- **QMUL-SurvFace:** audited against all seven evaluation sets. The raw result
  (96.9% above threshold) was **rejected as an artefact** after a control
  showed a QMUL face resembles an arbitrary *different* QMUL person more
  (0.600) than anything in TinyFace (0.522). No identity overlap was
  established; nothing was excluded.

Reproduce: `build_exclusion_list.py`, `audit_qmul_survface.py`,
`qmul_overlap_control.py`.

### 3.2 Licensing constraint carried by the optional pack

`arcface_qmul_v2` was trained on QMUL-SurvFace, whose terms are **research
purposes only**, and whose upstream copyright holders are not enumerated by
the distributor. **Any deployment of that pack inherits this constraint.** The
default deployed pack (`w600k_r50`) does not.

---

## 4. Performance

**Task measured: 1:1 verification.** Protocol: official InsightFace `.bin`
pair lists, counts asserted at load; 10-fold cross-validation with the
threshold fitted on 9 folds and applied to the held-out fold; horizontal-flip
test-time augmentation.

> **No accuracy figure in this document is measured at a threshold tuned on
> the pairs it is reported against.**

### 4.1 Accuracy — deployed model (`w600k_r50`)

| Dataset | Accuracy % (mean ± std) | TAR @ FAR=0.1% | Stresses |
|---|---|---|---|
| LFW | 99.78 ± 0.26 | 99.70 | Frontal, unconstrained (saturated) |
| CFP-FF | 99.87 | 99.86 | Frontal–frontal |
| AgeDB-30 | 98.15 ± 0.61 | 96.03 | 30-year age gap |
| CFP-FP | 97.44 ± 1.07 | 94.69 | Frontal vs profile |
| CALFW | 95.95 ± 1.09 | 92.10 | Cross-age |
| CPLFW | 94.47 ± 1.00 | 87.40 | Cross-pose |
| **TinyFace** | **82.45** | **33.13** | **Native low-resolution (median 32×32 px)** |

`python backend/scripts/benchmark_verification.py`

### 4.2 The figure that matters operationally

**TinyFace TAR@FAR=0.1% is 33.13%.** On genuinely low-resolution imagery, at
the 0.1% false-match operating point, the deployed system finds roughly **one
in three** true matches. This is the single most important limitation in this
document. Degraded-source imagery — the common forensic case — is where the
system is weakest, and no headline accuracy figure should be read as applying
to it.

### 4.3 Precision, recall, F1

These are **operating-point dependent** and are therefore reported as the
threshold sweep rather than as single numbers. At the production threshold
0.2871 on a balanced pair set, precision and recall are derivable from the
per-dataset FMR/FNMR in the Performance Report (document 05). A single
"F1 = x" figure across mixed conditions would be misleading and is deliberately
not given.

### 4.4 Optional degraded-condition routing

The system can route each comparison to a degraded-specialist pack based on
the quality score it already computes. Threshold **0.539**, derived from QMUL
and CASIA quality distributions — **both disjoint from every dataset in the
table above**, so the operating point was fixed before measurement.

| Dataset | TAR@FAR0.1% single-model | routed | Δ |
|---|---|---|---|
| **TinyFace** | 33.13% | **37.37%** | **+4.23** |
| CPLFW | 87.40% | 87.27% | −0.13 |
| AgeDB-30 | 96.03% | 95.97% | −0.06 |
| CFP-FP | 94.69% | 94.66% | −0.03 |
| LFW / CFP-FF / CALFW | — | unchanged | 0.00 |

`python backend/scripts/evaluate_routed_engine.py --threshold 0.539`

**Status: measured and validated, not enabled by default.** Enabling it is an
operator decision that also inherits the licensing constraint in §3.2.

### 4.5 Threshold

| Threshold | Value | Meaning |
|---|---|---|
| Match | 0.2871 | At or above → supports same person |
| Review | 0.2153 | Between review and match → examiner review |

0.2871 was selected after a demonstrated false match at 0.2405. A later
suite-wide calibration suggested 0.2363; it was **rejected** because it would
have re-admitted that exact false match. The higher threshold trades recall
for a lower false-match rate, which is the correct direction for forensic use.

---

## 5. Known limitations

Ordered by operational significance. Each is measured, not speculative.

| # | Limitation | Evidence |
|---|---|---|
| L1 | **Degraded-imagery recall is low** — 33.13% TAR@FAR0.1% on TinyFace | §4.2 |
| L2 | **No external or independent validation.** All figures are self-measured. Not submitted to NIST FRVT or equivalent. | — |
| L3 | **Demographic differentials persist.** Women ~1.7× the FNMR of men; under-25s ~3.8× the 41–55 band. Raising the threshold relocated these errors; it did not remove them. | `benchmark_demographics.py` |
| L4 | **Liveness is a heuristic, not anti-spoofing.** Reported with `certified: false`. | §1.3 |
| L5 | **CFP-FP absolute figures understate** by ~1.5 points due to a dataset-pack provenance artefact; relative ranking is unaffected. | BENCHMARKS §2b |
| L6 | **Single-process rate limiting.** Effective limits multiply by worker count; not a defence against a distributed attacker. | — |
| L7 | **Latency figures are single-threaded.** Threading saturates at ~4 workers (1.86×); batching reaches 2.82×. | `benchmark_concurrency.py` |
| L8 | **Fine-tuning on ~10K identities did not improve the base model.** Two attempts recorded, both reported including the failure. | BENCHMARKS §6d |

---

## 6. Ethical considerations

### 6.1 Demographic differentials

The differentials in L3 are **measured and unresolved**. An operator deploying
this system should assume that false-negative risk is not uniform across the
population, and should not apply a single threshold as though it were.

### 6.2 Irreversible consequences

Face recognition contributes to decisions — arrest, surveillance, exclusion —
that cannot be undone by later correcting a score. The controls in §1.4 exist
for this reason and are not optional.

### 6.3 Consent and provenance of training data

Neither the upstream corpora nor QMUL-SurvFace were collected with the
informed consent of the individuals depicted. This is an industry-wide
condition, not specific to this system, and is disclosed rather than elided.
One dataset commonly used in the re-identification field (DukeMTMC) was
withdrawn by its own authors on ethics grounds; whether it contributed to
QMUL-SurvFace cannot be determined from the distributed files.

### 6.4 What the system does not decide

The system does not determine identity, guilt, or admissibility. It orders
candidates by similarity. Every downstream inference is made by a person, and
the audit chain records who.

---

## 7. Evaluation gaps

Stated so that no reader assumes coverage that does not exist:

- No evaluation on **masked, occluded, or heavily-posed** faces beyond CFP-FP/CPLFW.
- No evaluation on **infrared, thermal, or non-visible spectrum** imagery.
- No evaluation on **children** (benchmark sets are predominantly adult).
- No **video or multi-frame** evaluation.
- No **presentation-attack** evaluation.
- No **adversarial-perturbation** robustness evaluation.

---

## 8. Maintenance and change control

Any change to the recognition pack, detector, alignment, or threshold
**invalidates every figure in this document**. The full benchmark suite must be
re-run and this card re-issued with a new version number before the changed
configuration is used operationally.

`python backend/scripts/regression_check.py` compares a candidate
configuration against the recorded baseline.

---

## 9. Verification of this document

| Claim class | How to reproduce |
|---|---|
| Accuracy, TAR | `benchmark_verification.py`, `benchmark_tinyface.py` |
| Demographics | `benchmark_demographics.py` |
| Routing | `evaluate_routed_engine.py --threshold 0.539` |
| Contamination | `build_exclusion_list.py`, `qmul_overlap_control.py` |
| Thresholds | `calibrate_threshold_suite.py` |

Raw outputs are written to `runtime/benchmarks/*.json`.

---

*Issued 2026-08-01 for system version 1.0.0. Supersedes no prior card.*
