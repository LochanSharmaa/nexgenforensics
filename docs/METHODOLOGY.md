# Methodology and Epistemic Status

Written to paper standard. Every claim below is sorted into one of three bins and
nothing is allowed to drift between them without an experiment.

**PROVEN** — measured on this machine, reproducible by a named command.
**HYPOTHESIS** — reasonable, argued, *not tested*.
**UNKNOWN** — requires an experiment that has not been designed or run.

No marketing language. Where a number is weak, the weakness is stated beside it.

---

## 1. PROVEN

All figures below come from `w600k_r50` on cached embeddings, CPU only, no
inference performed during the CPU phase.

### 1.1 Calibration is close to optimal on every dataset held

`python backend/scripts/evaluate_baseline.py`

| dataset | Cllr | Cllr_min | Cllr_cal | ECE | TAR@FAR=0.1% (95% CI) | misleading same-source |
|---|---|---|---|---|---|---|
| lfw | 0.0149 | 0.0111 | 0.0038 | 0.0007 | 99.70% [99.5, 99.9] | 0.07% |
| cfp_ff | 0.0096 | 0.0066 | 0.0030 | 0.0008 | 99.86% [99.7, 100.0] | 0.03% |
| agedb_30 | 0.1097 | 0.0937 | 0.0159 | 0.0106 | 96.03% [93.8, 96.7] | 0.97% |
| cfp_fp | 0.1601 | 0.1380 | 0.0221 | 0.0223 | 94.89% [94.1, 95.7] | 0.37% |
| calfw | 0.1946 | 0.1830 | 0.0116 | 0.0163 | 92.27% [90.1, 93.2] | 0.87% |
| cplfw | 0.2616 | 0.2433 | 0.0183 | 0.0245 | 87.43% [85.9, 89.6] | 1.33% |
| **tinyface** | **0.5563** | **0.5446** | **0.0117** | 0.0211 | **34.67% [24.2, 40.0]** | **13.20%** |

Method: two-parameter logistic calibration, fitted on 9 of 10 contiguous folds and
applied to the held-out fold, matching the existing verification protocol. Cllr_min
by pool-adjacent-violators. CIs are percentile bootstrap, 400 resamples.

**Finding.** Cllr_cal ≤ 0.022 everywhere. Reporting is nearly exhausted as a source
of improvement.

**Finding, and the one that directs the programme.** On TinyFace **98.7% of the loss
is discrimination**, not calibration. The system delivers 0.44 bits of a 1-bit
question. *No improvement to calibration, thresholds or reporting can fix degraded
imagery.* Only better recognition can.

**Finding, forensically serious.** TinyFace misleading same-source rate is **13.20%**
— on surveillance imagery, 1 in 8 different-source comparisons is reported with an
LR above 1.

### 1.2 Identification and open-set performance

| metric | value |
|---|---|
| gallery | 1,794 identities |
| probes | 1,500 (1,055 mated, 445 non-mated) |
| rank-1 | 44.74% |
| rank-5 / rank-10 | 53.74% / 56.59% |
| TPIR @ FPIR=1% | **11.28%** |
| TPIR @ FPIR=10% | 19.62% |

Non-mated probes come from identities held out of the gallery entirely, so the
unenrolled case is measured rather than assumed.

### 1.3 Identity capacity — one valid measurement only

`python backend/scripts/validate_capacity.py`

| statistic | TinyFace | 95% CI |
|---|---|---|
| bits (median) | 4.33 | [4.32, 4.34] |
| bits (20th pct) | 1.64 | [1.63, 1.65] |
| supportable gallery @80% rank-1 | **≈ 2** | — |

Against a 20M-comparison identity-disjoint reference population built by rejection
sampling on TinyFace's true identity labels.

**Finding.** At surveillance resolution this system does not carry enough identity
information to support 1:N identification at any operationally meaningful gallery
size. This is a joint property of the evidence and the model, not of the threshold.

### 1.4 Capacity measurement has two failure modes, both demonstrated

**Censoring.** Against each pack's own ~3,000 impostor pairs the tail cannot resolve
below 1/3000. LFW returned **99.7% censored** — the ceiling reported as if it were a
measurement.

**Contamination.** A 20M pool built by randomly pairing images from different pairs
assumes *different pair ⇒ different identity*, which is false:

| dataset | identities | suspect fraction |
|---|---|---|
| cfp_ff | 500 | 0.181% |
| cfp_fp | 500 | 0.153% |
| agedb_30 | 568 | 0.202% |
| cplfw | 3,884 | 0.038% |
| calfw | 4,025 | 0.020% |
| lfw | 5,749 | 0.030% |

Contamination tracks identity count exactly, as the mechanism predicts. 0.2% is
**twice** the FAR being measured at 0.1%; it made AgeDB-30 appear to fall from
96.03% to 8.40%. Artefact, not result.

**Consequence.** `ReferencePopulation.from_labelled` raises rather than returning an
unlabelled population, and capacity is computed only where identity labels exist.

### 1.5 A third failure mode, found by our own refusal logic

The first contamination detector used a fixed cosine probe of 0.5 and **refused a
valid TinyFace population**. The 0.18% it flagged was not contamination — the
population was disjoint *by construction*. It was genuine confusability: at 20
pixels different people really do exceed 0.5.

A detector calibrated on clean imagery rejects exactly the degraded measurements
that matter most. The probe is now derived from the corpus's own genuine
distribution (95th percentile), and the verdict is a flag rather than a refusal,
because duplicate labels and genuine confusability are not separable by score on
degraded imagery.

### 1.6 Software properties

288 tests pass (`pytest backend/tests_engine/ backend/tests/`), of which 127 are new
and several encode the failures above as regressions. Verified: Cllr = 1.0 exactly
for an uninformative system; LR is prior-independent under 10:1 class imbalance;
perfect separation does not produce infinite LRs; lineage chain detects tampering
and deletion; identity memory is exactly reversible under retraction; automated
matches never reach the model without adjudication; the degradation package exposes
no inverse operation.

---

## 2. HYPOTHESIS — argued, not tested

| # | Claim | Test that would settle it |
|---|---|---|
| H1 | **Comparing in observation space beats comparing in embedding space on degraded imagery.** The central bet | S0.3 arms A/B1/B2/B3/C. Built, CPU-validated, awaiting GPU |
| H2 | **Modelling the specific operator beats matching resolution alone** (B2 > B1) | The same run. This difference alone decides Stage 3 |
| H3 | Common-passband projection improves cross-resolution matching | S0.3 arm B3 |
| H4 | Supervised nuisance avoids the disentanglement problem that defeats unsupervised factorisation | Stage 2, λ₃/λ₅ ablation against the QMUL leakage metric |
| H5 | Cllr as a training loss improves calibration and discrimination jointly | Stage 2 ablation |
| H6 | The capacity bound predicts *individual* failures, not just dataset ordering | Requires the 153k distractor gallery (G1) |
| H7 | Measured camera calibration beats blind estimation enough to justify the capture programme | Requires C1 corpus + paired capture |

**H1 and H2 are the programme's load-bearing hypotheses.** Everything in Stage 3
rests on them and neither has been tested.

---

## 3. UNKNOWN — no experiment designed

- Whether a forensic-grade renderer is trainable at achievable scale.
- Whether the aging flow φ_age is learnable without longitudinal data that
  effectively does not exist.
- Whether declared reference populations can be assembled without creating a
  demographic-inference hazard.
- Whether conformal coverage holds under the distribution shift between a
  calibration corpus and a real case.
- Whether correlated-evidence fusion for video can be validated without ground-truth
  multi-camera captures.
- What the *true* supportable gallery is once population structure (relatives,
  doppelgängers) is modelled. Current figures assume independence and are therefore
  **optimistic**.

---

## 4. Reproducibility checklist

| Requirement | Status |
|---|---|
| Every figure traceable to one command | Yes — commands given in §1 |
| No inference during CPU phase | Yes — `"inference_run": false` in `baseline_evaluation.json` |
| Deterministic seeds | Yes — all RNG seeded, default 0 |
| Held-out calibration | Yes — 9/10 folds fit, 1 applied; oracle computed and labelled |
| Confidence intervals on published figures | Yes — percentile bootstrap |
| Invalid measurements refused, not degraded | Yes — `PopulationPurityError` |
| Failures recorded as regression tests | Yes — §1.4, §1.5 |
| Raw outputs committed | `runtime/forensics/*.json` |
| Environment pinned | `backend/requirements*.txt`; numpy `>=1.26.4,<3.0` |

### Reproducing everything in §1

```bash
python backend/scripts/audit_assets.py
python backend/scripts/evaluate_baseline.py
python backend/scripts/validate_capacity.py
python backend/scripts/analyze_evidence_capacity.py
python experiments/S0_3/run.py --embedder stub --pairs 40
cd backend && python -m pytest tests_engine/ tests/ -q
```

---

## 5. Standing caveats on every number in this document

1. **Independence is assumed and is false.** Gallery members are treated as
   independent draws; relatives and doppelgängers cluster, so true supportable
   galleries are **smaller** than reported. Optimistic direction.
2. **These are corpora, not forensic reference populations.** They characterise the
   system, not any case.
3. **Capacity is a property of (model, population, condition)**, not of the image.
   It lower-bounds what the pixels contain.
4. **The TinyFace cache lacks flip-TTA** (`emb` only, no `orig`/`flip`), so its
   figures are not directly comparable to published numbers produced with TTA.
5. **No independent validation.** Every figure was produced by the same tooling that
   built the system. Not submitted to NIST FRTE.
6. **QMUL, IJB-B and IJB-C are unmeasured** — images present, embeddings absent.
   Reported as blocked rather than omitted.
