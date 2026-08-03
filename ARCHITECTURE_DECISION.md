# Architecture Decision Record — 2026-08-02

**Authority:** lead research engineer, decision made on measured evidence.
**Scope:** recognition representation, open-set strategy, and what to build next.
**Every number below is measured in this repository.** Artifacts named inline.

---

## 0. Decision summary

| component | decision | evidence |
|---|---|---|
| **ViT-B KP-RPE AdaFace (WebFace12M)** | **REPLACE** the R50 as production recogniser | 3.0× TAR@FAR=0.1% on TinyFace, **no loss on any clean pack** |
| InsightFace R50 + CASIA | **KEEP as permanent baseline**, demote from production | still the reference every claim is measured against |
| SCRFD detector | **KEEP** | not the bottleneck |
| DFA aligner (new) | **ADOPT**, mandatory for non-aligned crops | +51 points on TinyFace |
| Score normalisation (margin / z-norm) | **ADOPT** | 27× TPIR@1% FPIR, free |
| Embedding centering + WCCN | **ADOPT** | 41× TPIR@1% FPIR, free, fitted at enrolment |
| **Multi-frame template averaging** | **REJECT** | **measured to make open-set worse** |
| Degradation-aware / forward-operator training | **REJECT** for surveillance-to-surveillance | S0.3: +0.13 pts, CI spans zero |
| 3DMM / geometry-aware / generative | **REJECT** | no supporting evidence; S0.3 points against |
| Probabilistic identity embedding | **DEFER** | uncertainty already handled in the evidence layer; adds no bits |
| Real low-resolution fine-tuning | **PROPOSE** (GPU approval requested, §6) | the only remaining lever that can add identity information |

---

## 1. The bottleneck, stated precisely

Three independent measurements agree, and they agree on something narrower than
"degraded imagery is hard":

1. **Calibration is exhausted.** Cllr_cal ≤ 0.6% of total Cllr on both
   surveillance corpora. No evidence-layer work moves accuracy.
2. **Capacity is the constraint.** QMUL delivers a **2.92-bit** median
   observation; its 2,965-entry gallery demands **≈11.5 bits**. Short by ~2⁸·⁶.
3. **The deficit is specifically open-set, not verification.** Verification
   improved 2–3× with a better backbone. Stranger rejection did not move.

The failure mechanism, measured directly: mean top-1 gallery score for a
**stranger** vs an **enrolled probe** differs by **0.001** (R50: strangers score
*higher*). The maximum-over-gallery statistic carries essentially no information
about whether the subject is enrolled.

---

## 2. ADOPTED: ViT-B KP-RPE AdaFace replaces the R50 in production

### Evidence — degraded (the target condition)

TinyFace official protocol, 153,428-image reference population, 50M impostor
pairs. `capacity_official_tinyface__{w600k_r50,vit_kprpe_wf12m}.json`.

| metric | R50 | **ViT** | ratio |
|---|---|---|---|
| TAR @ FAR=0.1% | 20.59% | **60.77%** | 3.0× |
| TAR @ FAR=1e-5 | 7.38% | **35.03%** | 4.7× |
| rank-1 (155,997 gallery) | 32.93% | **68.20%** | 2.1× |
| identity bits, median | 4.73 | **12.95** | +8.2 |
| Cllr | 0.6662 | **0.4920** | |

### Evidence — clean (the regression check that decides adoption risk)

`pack_benchmarks__vit_kprpe_wf12m.json` vs BENCHMARKS.md §2.

| pack | R50 | ViT | Δ |
|---|---|---|---|
| LFW | 99.78 | 99.75 | −0.03 |
| CFP-FF | 99.87 | 99.86 | −0.01 |
| CFP-FP | 97.44 | 97.33 | −0.11 |
| AgeDB-30 | 98.15 | 97.90 | −0.25 |
| CALFW | 95.95 | 95.82 | −0.13 |
| CPLFW | 94.47 | 94.40 | −0.07 |

**The ViT is fractionally *behind* on every clean pack** — all within fold
standard deviation (±0.2 to ±1.2), but consistently so, and I am recording it as
a real if small cost rather than rounding it to "equal".

**Why adopt anyway:** the losses are ≤0.25 points on benchmarks that are
saturated and carry no operational information, against a 3–5× gain on the
condition this system exists for. That is the trade the project needs, and it is
the opposite of the usual risk — normally a degraded-imagery gain costs clean
accuracy materially.

### Mandatory consequence: thresholds are recalibrated, not carried over

0.2871 is the FMR=0.1% operating point *for the R50*. Re-derived for the ViT by
the identical rule (FMR=0.1% on AgeDB-30, 12,000 pairs):

| | R50 | **ViT** |
|---|---|---|
| match | 0.2871 | **0.2371** |
| review | 0.2153 | **0.1778** (0.75 × match, preserving the incumbent ratio) |

Carrying the old threshold across would move the operating point to a completely
different place on the ROC while continuing to look calibrated. This is a
blocking requirement for the swap, not a follow-up.

### Two integration defects that had to be fixed first — both silent

| defect | cost | why the obvious gate missed it |
|---|---|---|
| KP-RPE keypoints are [0,1]-normalised, not pixels | LFW 56.70% vs 99.90% | — |
| Canonical keypoints on non-ArcFace-aligned crops | TinyFace **15.19%, below the R50** | `.bin` pack crops *are* aligned, so the LFW gate passed at 99.75% while the assumption was false for every surveillance corpus |

Contract-locked in `test_vit_backbone_contract.py` (7 tests), including an
assertion that alignment stays the **default** path.

---

## 3. REJECTED: multi-frame template averaging

This was on the suggested list. **I measured it and it is counterproductive.**

QMUL, ViT embeddings, probe = mean of N frames, open-set against 2,965 gallery:

| frames | open-set AUC | mated−stranger score gap |
|---|---|---|
| 1 | **0.5399** | **+0.0167** |
| 2 | 0.4957 | −0.0009 |
| 4 | 0.4821 | −0.0070 |
| 8 | 0.4799 | −0.0081 |
| 16 | 0.4785 | −0.0090 |

**More frames makes stranger rejection strictly worse, and drives AUC below
chance.**

**Mechanism, and it follows from an earlier measurement.** Corpus-level
condition leakage is **+0.1088** — far larger than the camera-level term
(+0.0039). Every QMUL embedding carries a large shared acquisition-regime
component. Averaging is a low-pass filter: it suppresses the idiosyncratic
identity signal (which varies frame to frame) while *preserving* the shared
condition component (which does not). The averaged probe converges toward "the
mean QMUL face", and so does every gallery entry.

**This does not reject multi-frame evidence in principle** — it rejects *naive
mean fusion*. Fusion can only help after the common-mode component is removed,
which is §4. Any future fusion work must be gated on that ordering, and must
re-run this exact table.

---

## 4. ADOPTED: condition removal before scoring

Two independent cheap interventions, both attacking the same measured cause.

### (a) Embedding-space: centering + WCCN

Within-class covariance normalisation whitens the directions along which
*same-identity* samples vary — the nuisance/condition axes. Fitted **on the
gallery only**, i.e. data available at enrolment; no probe or test information.

| transform | open AUC | TPIR@1% FPIR | TPIR@10% | rank-1 |
|---|---|---|---|---|
| raw | 0.4958 | 0.03% | 1.07% | **6.30%** |
| centered | 0.5029 | 0.37% | 2.07% | 6.57% |
| **centered + WCCN** | **0.5065** | **1.23%** | **2.80%** | 5.33% |

**41× at FPIR=1%.** Note the trade: WCCN costs ~1 point of rank-1 while buying
open-set rejection. Both must be reported; the correct default depends on
whether the deployment is investigative ranking or watchlist screening, and that
is an operator decision the system should expose rather than hard-code.

### (b) Score-space: margin and per-probe z-norm

| statistic (ViT) | TPIR@1% | TPIR@10% | open AUC |
|---|---|---|---|
| `max` (current) | 0.08% | 1.22% | 0.504 |
| **`margin`** (top1 − top2) | **2.19%** | 3.11% | 0.519 |
| **`znorm`** (per-probe cohort, ranks 50–500 of its own gallery scores) | 1.77% | **3.87%** | **0.524** |

Justified by a prior measurement: per-probe impostor means vary with std
**0.0573** across probes — **57× larger than the 0.001 signal**. Probe-dependent
hubness was drowning the evidence.

**Both are free, need no retraining and no extra data. Both ship.**

---

## 5. The honest limit, and why it is the publishable result

Every intervention above is a large *relative* gain on a near-zero base. The
absolute open-set AUC on real cross-camera surveillance moves from 0.496 to
**~0.52** — against 0.500 for a coin flip.

Structural facts behind that:

- QMUL's identification protocol is **100% cross-camera**: I found **zero**
  same-camera genuine pairs. Probe and gallery never share an imaging condition.
- Genuine cross-camera mean similarity is **+0.2112**, against a corpus-level
  condition leakage of **+0.1088** — the nuisance term is half the signal term.
- Capacity: **2.92 bits** available, **≈11.5 bits** required.

**Nothing tested — better backbone, fusion, normalisation — changes the order of
magnitude.** That is consistent with an information deficit rather than a
modelling deficit, and it was *predicted by the capacity framework before the
backbone swap and confirmed by it*. That predictive success is the strongest
scientific result this project holds, and the finding —
*"single-image open-set identification on cross-camera surveillance is
information-limited, not architecture-limited, and here is the bit budget"* —
is more valuable published than a marginal accuracy claim.

---

## 6. GPU EXPERIMENT PROPOSAL — awaiting approval

**The one remaining lever that can add identity information rather than extract
it more efficiently.**

| | |
|---|---|
| **Objective** | LoRA fine-tune ViT-B KP-RPE on **real** low-resolution imagery |
| **Data** | TinyFace `Training_Set` (7,804 imgs / 2,570 ids) + QMUL `training_set` (220,888 imgs / 5,319 ids). **Contamination audited: zero raw label overlap with either test split** (TinyFace 2,570 vs 2,569; QMUL 5,319 vs 5,265) |
| **Why GPU** | Backpropagation through a 115M-parameter ViT. CPU is not slow here, it is infeasible |
| **Why LoRA, not full FT** | 6 GB cannot hold ViT-B optimiser state at usable batch size. LoRA on `attn.qkv` / `attn.proj` / `mlp.fc1` / `mlp.fc2` (rank 8–16) trains ~1–2% of parameters. Also the correct choice scientifically: the prior full fine-tune failed partly through catastrophic forgetting of calibration |
| **Runtime estimate** | 6–10 GPU-hours for a first rank-8 run at batch 32 |
| **Hypothesis** | Real low-resolution training data raises identity bits on QMUL above the 2.92-bit baseline |
| **Primary metric** | QMUL identity bits (median) and open-set AUC, official protocol |
| **PASS** | bits ≥ 4.0 **and** clean-pack regression ≤ 0.5 points **and** condition leakage not increased above +0.0039 |
| **FAIL** | bits < 3.5, or clean regression > 0.5 pts, or leakage inflated |
| **Decision if PASS** | fine-tuning becomes the programme's main line; scale data and rank |
| **Decision if FAIL** | representation learning is closed on this hardware. The programme's contribution is then the evidence layer plus the capacity/impossibility result, and effort moves to multi-modal evidence or declared-gallery-size limits |

**The precedent that makes this worth running:** the prior fine-tune failed on
*synthetically* degraded data, diagnosed as "synthetic degradation does not match
real capture". This is the documented prerequisite experiment (ROADMAP 10.4) and
the data has been on disk, unused, the entire time.

**Leakage must be measured, not assumed.** A TAR gain that inflates condition
leakage means the model learned the camera. `measure_condition_leakage.py` is the
gate.

---

## 7. Rejected, with reasons

| proposal | why rejected |
|---|---|
| **Multi-frame fusion (naive)** | Measured: AUC 0.5399 → 0.4785 as frames go 1 → 16. §3 |
| **Degradation-aware / forward-operator training** | S0.3, pre-registered rule: +0.13 pts on both surveillance corpora, CI spanning zero. Coherent method, no operator asymmetry to exploit |
| **3DMM / geometry-aware / generative restoration** | No supporting measurement; S0.3 evidence points against; renderer cost is multi-year and 6 GB cannot train one. Architecturally excluded from the evidential path regardless |
| **Probabilistic identity embeddings** | Uncertainty is already handled correctly in the evidence layer (calibrated LRs, conformal sets, abstention). Moving it into the embedding is expensive and adds no bits — it re-expresses the deficit rather than reducing it |
| **Cllr-as-training-loss** | Attacks calibration, which is measured to be within 0.6% of optimal. Optimising a solved problem |
| **Larger backbone / more pretraining data** | 6 GB. And the clean-pack table shows the ViT already matches the R50 there — scale is not the constraint, condition is |

---

## 8. Risks

| risk | severity | mitigation |
|---|---|---|
| ViT clean-pack regression (−0.03 to −0.25 pts) is real, not noise | Low | Recorded, not rounded away. R50 retained as baseline; both remain runnable via `--model` |
| WCCN costs rank-1 (6.30% → 5.33%) | Medium | Expose as an operator-selectable mode, do not hard-code |
| Transforms fitted on the gallery could overfit a small gallery | Medium | Regularised (ε = 1e-3·trace); must be re-validated per deployment gallery |
| LoRA fine-tune repeats the prior failure | Medium | Pre-registered PASS/FAIL in §6; leakage gate; real not synthetic data this time |
| Open-set may be unreachable at any effort | **High, and partly established** | Reframed as the scientific contribution rather than a defect to hide |
| Single-corpus generalisation (QMUL only for open-set) | High | TinyFace structurally cannot measure open-set. **SCface remains the outstanding acquisition** |

---

## 9. Implementation roadmap

**Immediate (no approval needed, CPU/cheap):**
1. Wire `margin` and `znorm` into `forensics/openset.py` as selectable ranking statistics, defaulting to `znorm`, with the AUC ≈ 0.52 caveat carried into every report.
2. Add centering + WCCN as an enrolment-time gallery transform, operator-selectable, off by default until validated on a real gallery.
3. Promote the ViT to default recogniser behind a config flag, with recalibrated thresholds and the R50 retained as `--model w600k_r50`.
4. Re-run the full evidence stack on both backbones; regenerate CLAIMS.md provenance.

**On approval (§6):** LoRA fine-tune on real LR data.

**Blocked on data:** SCface (settles the S0.3 asymmetric-case PASS and gives a
second open-set corpus). IJB-S, BRIAR — applications should already be running.

---

## 10. What I disagree with, explicitly

You listed multi-frame fusion, degradation-aware training, geometry/3DMM, and
probabilistic identity representation as candidates. **On the measured evidence I
reject all four**, one of them (fusion) because I tested it today and it makes
the target metric worse. I have adopted the two you did not list — score
normalisation and WCCN — because they deliver 27× and 41× on the bottleneck
metric for zero training cost.

The one item from your list I endorse without reservation is **ViT-B + KP-RPE +
AdaFace**, and it is adopted with its thresholds re-derived and its clean-pack
cost recorded honestly.
