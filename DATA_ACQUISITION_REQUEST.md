# Data Acquisition Request

**Raised 2026-08-02, immediately after the S0.3 result.**
**One P0 request. Everything else on this page is explicitly NOT requested yet.**

---

## Why this request exists

S0.3 returned a split verdict, and the split is the whole finding:

| condition | B2 − B1 (TAR@FAR=0.1%) | 95% CI | verdict |
|---|---|---|---|
| **HR gallery vs LR probe** (`lfw_synth`, n=20,000) | **+3.44 pts** | [+1.82, +5.62] | **PASS** |
| LR vs LR (TinyFace, 32 px) | −7.53 pts | [−13.61, −1.88] | FAIL |
| LR vs LR (QMUL, 25 px) | −4.80 pts | [−8.84, −1.67] | FAIL |

Forward-operator modelling wins **only where a clean gallery meets a degraded
probe** — which is the forensic case: a custody photograph against CCTV. Where
both sides are already degraded there is no operator asymmetry to exploit and
applying one compounds the degradation.

**The PASS carries a caveat that only new data can remove.** `lfw_synth` degrades
LFW with a *synthetic* operator, and arm B2 estimates a simplified version of that
same operator. Some fraction of +3.44 may be the experiment recovering its own
assumption rather than recovering physics. This project has already published one
negative result caused by exactly this gap — the fine-tune on synthetically
degraded data scored worse on every benchmark, diagnosed as "synthetic
degradation does not match real low-resolution capture."

**A multi-year Stage 3 decision must not rest on a synthetic operator.**

---

## P0 — SCface (REQUESTED)

| | |
|---|---|
| **Name** | SCface — Surveillance Cameras Face Database |
| **Source** | University of Zagreb, FER. `https://www.scface.org/` |
| **Access** | Free for research. Signed licence agreement returned by email; typically days, not months |
| **Size** | ~2.6 GB |
| **Contents** | 130 subjects. One high-quality frontal mugshot each, plus surveillance captures from 5 commercial CCTV cameras at **three fixed distances (4.20 m, 2.60 m, 1.00 m)**, visible and infrared |
| **Storage impact** | Negligible — 145 GB free |
| **GPU cost** | ~4,160 images. Under 1 minute at the measured 330 img/s |

### Why this dataset specifically, and not another

SCface is the only public corpus with the exact structure S0.3 needs:

1. **A genuine HR gallery and a genuinely degraded probe of the same person** —
   mugshot vs real CCTV. TinyFace and QMUL are LR on both sides, which is why
   they cannot test the PASS condition at all.
2. **Real optics.** The degradation comes from actual surveillance cameras, not
   from `apply_forward()`. This is precisely the confound.
3. **Known, labelled standoff distance** — three fixed distances give three
   degradation levels with ground truth. B2's advantage should *grow* with
   distance if the forward-model thesis is right. That is a falsifiable
   prediction, not a single number.
4. **Multiple camera models**, so per-camera operator differences are measurable —
   the input the degradation-estimation layer was built for.

### The experiment it enables, stated before the data arrives

Re-run S0.3 arms A / B1 / B2 / B3 / C on SCface, mugshot-vs-camera, split by
distance.

- **CONFIRM** — B2 − B1 ≥ +2.0 points at 4.20 m with CI excluding zero, and the
  margin ordered by distance (4.20 m > 2.60 m > 1.00 m). Stage 3 licensed for
  the HR/LR case on real optics.
- **REFUTE** — B2 − B1 ≤ +0.5 or CI spanning zero on real imagery. The
  `lfw_synth` PASS was self-fulfilment. Stage 3 cancelled outright, and the
  conclusion becomes: *forward-operator modelling works on synthetic degradation
  and does not transfer* — which is a publishable negative result in its own
  right and closes the question permanently.

Either outcome resolves the programme's largest open decision for under a minute
of GPU. This is the highest information-per-cost request in the project.

---

## P1 — Apply now, expect months (NOT blocking)

| Dataset | Why | Gate |
|---|---|---|
| **IJB-S** | The surveillance benchmark that defines the niche. Published leaders sit at 63.44% rank-1 (KP-RPE) and 44.81% (FaceMoE) — the numbers this project must eventually be measured against | NIST/FBI data agreement. **Apply immediately; it takes months and blocks nothing meanwhile** |
| **BRIAR** | Long-range, multimodal, real government collection | IARPA agreement, same timescale |
| **UCCS** | Real unconstrained surveillance, complements SCface | Registration |

---

## NOT requested, and I want the reasons on record

| Dataset | Why not |
|---|---|
| **Glint360K** (17M images, 500 GB+) | Cannot be trained on a 6 GB RTX A3000 at any batch size or with any parameter-efficient method. The published `w600k_r50` and CVLface weights already encode this training. It would consume a week of bandwidth to produce nothing runnable |
| WebFace260M, MS1M-*, DeepGlint, Celeb500k | Same reason. Pretraining scale is unreachable on this hardware |
| MegaFace testsuite | Withdrawn by its own authors; known distractor-label noise |
| CelebA, IMDB-Face, Asian-Celeb | No role in any planned experiment |
| Kinship / familial corpora | Deliberately out of scope. Familial search implicates people who never consented — see NEXTGEN-ARCHITECTURE.md §14 risk 7 |

**On disk and unused, which is a better use of time than any download:**
TinyFace `Training_Set` (7,804 real native-LR images, never used for training),
MegaFace train (13 GB), CASIA-WebFace (3.1 GB), UMDFaces (2.5 GB).

---

## Summary of the ask

**Please obtain SCface.** Free, ~2.6 GB, days not months, under a minute of GPU,
and it decides whether a multi-year architecture is licensed or cancelled.

**And please file the IJB-S and BRIAR applications now** — they take months, and
starting the clock costs nothing today.
