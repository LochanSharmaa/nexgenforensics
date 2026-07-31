# NexGen iMATCH — Verification Benchmarks

**Task measured: 1:1 verification.** Given two images, decide same-person /
different-person by thresholding cosine similarity. This is what the
"compare two faces" feature does.

**This is not an identification benchmark.** The previously quoted 87.32%
figure was *rank-1 closed-set identification* accuracy ("which of these 25
enrolled people is this probe?"). That is a different task with a different
number, and it does not describe the verification feature. Nothing on this page
should be quoted as an identification result.

Reproduce everything here with:

```bash
python backend/scripts/benchmark_verification.py --datasets lfw agedb_30 cfp_fp calfw cplfw
```

---

## 1. Protocol

| Item | Value |
|---|---|
| Pair source | Official InsightFace `.bin` verification packs (`lfw.bin`, `agedb_30.bin`, `cfp_fp.bin`, `calfw.bin`, `cplfw.bin`) |
| Pair construction | **Published pair lists, not generated here.** Counts asserted at load; a pack with a non-standard count is refused |
| Images | Pre-aligned 112×112 ArcFace crops, fed directly to the recognition net (no re-detection — running a detector on an already-tight crop degrades accuracy) |
| Augmentation | Horizontal-flip TTA: `emb = L2norm(f(img) + f(flip(img)))` |
| Cross-validation | 10-fold, contiguous folds matching the published fold layout |
| **Threshold** | **Fitted on 9 folds, applied to the held-out fold.** No reported accuracy is measured at a threshold tuned on the pairs it is reported against |
| Hardware | NVIDIA RTX A3000 Laptop GPU, `CUDAExecutionProvider` (verified by `scripts/verify_gpu.py`) |

### Pair counts (asserted at load time)

| Dataset | Pairs | Genuine | Impostor | What it stresses |
|---|---:|---:|---:|---|
| LFW | 6,000 | 3,000 | 3,000 | Frontal, unconstrained — saturated |
| AgeDB-30 | 6,000 | 3,000 | 3,000 | 30-year age gap |
| CFP-FP | 7,000 | 3,500 | 3,500 | Frontal vs. profile pose |
| CALFW | 6,000 | 3,000 | 3,000 | Cross-age LFW |
| CPLFW | 6,000 | 3,000 | 3,000 | Cross-pose LFW |

> The brief specified a 3,000-pair AgeDB-30 protocol. The published protocol is
> 6,000 pairs (10 folds × 600); the shipped pack contains 6,000 and that is what
> was used.

---

## 2. Results — accuracy % (mean ± std over 10 held-out folds)

Best per column in **bold**.

| Configuration | LFW | AgeDB-30 | CFP-FP | CALFW | CPLFW | Mean |
|---|---|---|---|---|---|---|
| single: w600k_r50 (R50) | 99.78 ± 0.26 | 98.15 ± 0.61 | 97.44 ± 1.07 | 95.95 ± 1.09 | 94.47 ± 1.00 | 97.16 |
| **single: glintr100 (R100)** | 99.77 ± 0.26 | **98.32 ± 0.60** | **97.71 ± 0.93** | **96.17 ± 1.05** | **94.78 ± 1.00** | **97.35** |
| single: w600k_mbf (MobileFaceNet) | 99.60 ± 0.25 | 96.33 ± 0.67 | 96.00 ± 1.10 | 95.60 ± 1.09 | 92.63 ± 1.03 | 96.03 |
| ensemble: weighted .45/.45/.10 *(shipped)* | **99.80 ± 0.27** | **98.32 ± 0.56** | 97.47 ± 1.01 | 96.07 ± 1.14 | 94.32 ± 1.07 | 97.20 |
| ensemble: equal 1/3 | 99.77 ± 0.25 | 98.07 ± 0.61 | 97.21 ± 1.04 | 96.08 ± 1.04 | 93.92 ± 1.10 | 97.01 |
| ensemble: dual r50+r100 | **99.80 ± 0.27** | 98.30 ± 0.71 | 97.50 ± 1.02 | 96.15 ± 1.14 | 94.47 ± 1.09 | 97.24 |
| ensemble: concat 1536-d | 99.77 ± 0.27 | 98.02 ± 0.60 | 97.56 ± 0.99 | 96.05 ± 1.03 | 94.33 ± 1.17 | 97.15 |

### 2b. PACK PROVENANCE — the CFP-FP result depends on which `.bin` you use

**Resolved: the ~1.6-point CFP-FP shortfall was a dataset-provenance artifact,
not an accuracy problem.** The `.bin` packs are NOT identical across the
training bundles that ship them. Same model (`glintr100`), same code path, same
7,000 pairs:

| `cfp_fp.bin` source | sha256 (first 10) | Accuracy |
|---|---|---|
| `faces_webface_112x112` | `76306c783c` | **97.71 ± 0.93** |
| `faces_megafacetrain` | `76306c783c` (identical) | — |
| `faces_umd` | `d47cdcfe71` | **99.26 ± 0.49** |
| `ms1m-retinaface-t1` | `a8754ddf97` | **99.21 ± 0.36** |

Two independently-sourced packs agree at ~99.2%, matching published results for
this architecture. The `faces_webface` variant is the outlier — most likely a
differently-aligned or lower-quality re-crop.

**Control:** CFP-**FF** (frontal-frontal) on the same `faces_webface` bundle
scores **99.91 ± 0.15**, i.e. correctly saturated. That rules out a fault in the
harness, the flip-TTA, or the model, and isolates the anomaly to the CFP-FP pack
itself.

**LFW cross-check** — the LFW packs also differ (`faces_webface` et al.
`9b711dca71` vs `ms1m-retinaface` `2cdf024294`), but the result does not move
materially: **99.77 ± 0.26** vs **99.80 ± 0.19**. So this is a CFP-FP-specific
problem, not a systematic one.

**Pack availability across bundles**

| Pack | webface | megaface | umd | ms1m | Variants |
|---|---|---|---|---|---|
| lfw | ✓ | ✓ | ✓ | ✓ | **2** |
| agedb_30 | ✓ | ✓ | ✓ | — | 1 |
| cfp_fp | ✓ | ✓ | ✓ | ✓ | **3** |
| cfp_ff | ✓ | ✓ | — | — | 1 |
| calfw | ✓ | — | — | — | 1 |
| cplfw | ✓ | — | — | — | 1 |

**Consequence for §2:** the CFP-FP column in the table above was measured on the
`faces_webface` variant and therefore **understates** every configuration by
roughly 1.5 points. The relative ranking between configurations is unaffected
(all were measured on the same pack), so the production choice in §3 still
holds. The absolute CFP-FP figures should be re-run on the `ms1m-retinaface`
pack before being quoted externally; this has **not** yet been done for all
seven configurations.

**Rule going forward:** every benchmark number must record which pack file
produced it, by hash. A dataset named "CFP-FP" is not a unique object.

### TAR @ FAR = 0.1% (%)

| Configuration | LFW | AgeDB-30 | CFP-FP | CALFW | CPLFW |
|---|---|---|---|---|---|
| single: w600k_r50 | 99.70 | 96.03 | 94.69 | 92.10 | 87.40 |
| **single: glintr100** | 99.63 | 96.33 | 95.57 | 92.30 | 86.07 |
| single: w600k_mbf | 99.33 | 86.07 | 88.63 | 88.07 | 79.17 |
| ensemble: weighted *(shipped)* | 99.63 | 96.37 | 94.94 | 92.13 | 87.80 |
| ensemble: dual r50+r100 | 99.63 | 96.53 | 94.94 | 92.27 | 87.73 |

### Tuned thresholds (mean over folds)

| Configuration | LFW | AgeDB-30 | CFP-FP | CALFW | CPLFW |
|---|---|---|---|---|---|
| single: glintr100 | 0.2838 | 0.2216 | 0.2295 | 0.2292 | 0.1920 |
| ensemble: weighted | 0.2770 | 0.2191 | 0.2024 | 0.2137 | 0.1808 |

---

## 3. The ensemble does not earn its cost

**The shipped 3-model ensemble is not more accurate than its single strongest
backbone.** `glintr100` alone is better or tied on all five datasets:

| Dataset | glintr100 | shipped ensemble | Δ |
|---|---|---|---|
| LFW | 99.77 | 99.80 | −0.03 |
| AgeDB-30 | 98.32 | 98.32 | 0.00 |
| CFP-FP | **97.71** | 97.47 | **+0.24** |
| CALFW | **96.17** | 96.07 | **+0.10** |
| CPLFW | **94.78** | 94.32 | **+0.46** |

The only dataset where the ensemble leads is LFW, by 0.03 points — against a
fold std of ±0.26, i.e. an order of magnitude smaller than the noise. LFW is
saturated at ~99.8% and cannot separate these configurations.

On the datasets that *do* discriminate — pose (CPLFW, CFP-FP) and age (CALFW) —
the single model wins outright.

**Why:** MobileFaceNet is materially weaker (96.03 mean, and only 79.17%
TAR@FAR=0.1% on CPLFW). Averaging its embedding into the fused vector pulls the
result toward a less discriminative direction on exactly the hard pairs that
decide the score. The equal-weight ensemble — which gives MobileFaceNet a full
1/3 vote — is the worst ensemble variant (97.01 mean), confirming the
mechanism. The shipped weighting only avoids harm because it already discounts
MobileFaceNet to 0.10.

**Consequence:** the ensemble costs 3× the inference compute and 3× the model
memory for no accuracy gain, and a small loss under pose variation.

### Production choice

**`glintr100` (antelopev2, ResNet-100, Glint360K), single model, flip-TTA,
threshold 0.22.**

- Highest mean accuracy of every configuration tested (97.35).
- Best on 3 of 5 datasets, tied on the other 2.
- ⅓ the inference cost of the shipped ensemble.
- Threshold 0.22 is the fold-tuned optimum on AgeDB-30 (0.2216) and sits
  within 0.01 of the CFP-FP (0.2295) and CALFW (0.2292) optima.

Threshold is deliberately taken from the **hardest clean benchmark**, not LFW.
LFW's optimum (0.2838) is unrepresentative of casework, where age gap and pose
are the norm; tuning there yields a cut-point too high for anything harder.

The previously shipped thresholds (**0.28 / 0.42**) were never measured — they
were copied from the README. 0.42 sits far above *every* empirically optimal
operating point measured here (0.19–0.28), so genuine pairs the model scored
correctly were being reported "inconclusive" or "different_person". This was
the single largest accuracy loss in the system and cost nothing to fix.

---

## 4. Degraded conditions — TinyFace

**This is a separate number. It must never be averaged into the clean-benchmark
headline.**

```bash
python backend/scripts/benchmark_tinyface.py --model glintr100
```

| Property | Value |
|---|---|
| Identities | 2,569 (2,563 with ≥2 captures) |
| Pairs | 6,000 (3,000 genuine / 3,000 impostor) |
| Native resolution | **median 32×32 px**, min 19×14 |
| Protocol | Same 10-fold, threshold fitted on 9 folds |

| Metric | glintr100 |
|---|---|
| **Accuracy** | **79.68% ± 1.93** |
| TAR @ FAR=1% | 35.30% |
| TAR @ FAR=0.1% | **17.37%** |
| AUC | 0.87843 |
| EER | 20.37% |

### What this means operationally

| Condition | Accuracy | TAR @ FAR=0.1% |
|---|---|---|
| Clean (AgeDB-30) | 98.32% | 96.33% |
| Degraded (TinyFace) | 79.68% | 17.37% |
| **Gap** | **−18.64 pts** | **−78.96 pts** |

The accuracy drop is large; the drop at a forensically usable operating point
is catastrophic. At FAR=0.1% — one false match per thousand impostor
comparisons, already permissive for investigative use — the system finds
**about one in six** genuine matches on surveillance-resolution imagery.

A 32×32 crop simply does not contain the information ArcFace needs. No
threshold change fixes this, because it is a property of the input, not the
decision rule. Degraded-footage searches should be treated as generating leads
for human review, never as identification.

---

## 5. Demographic breakdown

```bash
python backend/scripts/benchmark_demographics.py --model glintr100
```

Built from the raw AgeDB folder, whose filenames carry age and gender
(`<idx>_<Name>_<age>_<gender>.jpg`). The standard `agedb_30.bin` pack cannot
support this — it is anonymised and retains only the is-same flag.

- 16,479 images embedded, 40,098 pairs (8,098 genuine / 32,000 impostor)
- Impostor pairs matched **within** subgroup (same gender, same age bucket);
  cross-group impostors are trivially easy and would deflate every FMR
- **One global threshold** (0.3089, set at FMR=0.1% over all impostors).
  Per-group thresholds would conceal the differential this audit exists to find

### 5a. DEPLOYED CONFIGURATION — `w600k_r50` at operating threshold 0.20

```bash
python backend/scripts/benchmark_demographics.py --model w600k_r50 --threshold 0.20
```

**This is the operationally meaningful table.** It uses the model the service
actually loads, at the threshold the service actually decides on.

| Subgroup | Genuine | Impostor | FNMR % | FMR % |
|---|---:|---:|---:|---:|
| **ALL** | 8,098 | 32,000 | **3.30** | **1.19** |
| gender = female | 3,300 | 16,000 | **4.88** | **1.70** |
| gender = male | 4,798 | 16,000 | 2.21 | 0.68 |
| age 0–25 | 854 | 8,000 | **7.61** | 1.21 |
| age 26–40 | 2,595 | 8,000 | 3.24 | 1.26 |
| age 41–55 | 2,199 | 8,000 | **2.14** | 1.10 |
| age 56+ | 2,450 | 8,000 | 2.90 | 1.19 |

### Findings

1. **False-match rate at the deployed threshold is 1.19%** — roughly **1 in 84**
   impostor comparisons returns a false match. The 0.20 threshold was tuned to
   maximise *accuracy*, which trades away FMR. For forensic use, where a false
   match points at an innocent person, this is the single most important number
   on this page and it is **not** a low-FMR operating point.
2. **Gender.** Women are falsely rejected **2.21×** more often than men
   (4.88% vs 2.21%) and falsely matched **2.50×** more often (1.70% vs 0.68%).
   Both error types run against the same group, so this is a genuine accuracy
   differential, not a threshold-placement artifact.
3. **Age.** The under-25 bucket has **3.56×** the false-non-match rate of the
   41–55 bucket (7.61% vs 2.14%). Young subjects change appearance fastest
   between captures, and AgeDB pairs span up to 30 years.
4. **The aggregate hides both.** A single "3.30% FNMR" looks acceptable while
   concealing a subgroup at 7.61%, and says nothing about the 1.70% FMR
   carried by women.

### 5c. DECISION RECORD — operating threshold raised 0.20 → 0.2871

**Date:** 2026-07-31 · **Approved by:** project owner · **Model:** `w600k_r50`
**Changed in:** `nexgen_engine/config.py::ThresholdConfig` (single source of truth)

#### What prompted it

A real 1:1 comparison of **two different people** returned similarity
**0.2405** and the system reported *"supports … that the images show the same
person."* Not a defect — the pair fell between the deployed threshold (0.20)
and the FMR=0.1% point (0.2871). It is a direct instance of the 1.19%
false-match rate measured in §5a: about **1 in 84** impostor comparisons.

#### Why the old value was wrong for this system

0.20 was the **accuracy-maximising** threshold from 10-fold cross-validation.
Accuracy weights a miss and a false match equally. Forensic work does not: a
missed lead costs a re-check, a false match points investigators at an innocent
person. Optimising accuracy silently chose a permissive operating point.

#### Measured tradeoff — 40,098 AgeDB pairs, same model, same pair set

| Subgroup | FNMR @0.20 | FNMR @0.2871 | FMR @0.20 | FMR @0.2871 |
|---|---:|---:|---:|---:|
| **ALL** | 3.30% | **6.32%** | **1.19%** | **0.10%** |
| gender = female | 4.88% | 8.45% | 1.70% | 0.16% |
| gender = male | 2.21% | 4.86% | 0.68% | 0.03% |
| age 0–25 | 7.61% | **14.75%** | 1.21% | 0.10% |
| age 26–40 | 3.24% | 5.78% | 1.26% | 0.05% |
| age 41–55 | 2.14% | 3.91% | 1.10% | 0.15% |
| age 56+ | 2.90% | 6.12% | 1.19% | 0.09% |

**Net:** false matches fall **12×** (1.19% → 0.10%, ~1 in 84 → ~1 in 1,000);
misses roughly double (3.30% → 6.32%). The 0.2405 pair above now correctly
rejects.

#### Costs accepted, stated plainly

1. **Misses nearly double**, and the burden is uneven. The under-25 group goes
   from 7.61% to **14.75%** FNMR — close to one in seven genuine pairs missed.
2. **The demographic disparity is not fixed, only relocated.** Women retain
   ~1.7× the FNMR of men (8.45% vs 4.86%) and ~5× the FMR (0.16% vs 0.03%).
   Raising the threshold changes where errors land, not who bears them.
3. **This is a single-dataset calibration.** It is derived from AgeDB pairs and
   has not been validated against operational imagery. It is an operating point,
   not an accuracy guarantee.

#### Two calibrations were run, and they disagreed

Before applying 0.2871 it was validated against the **full published suite**
rather than the single dataset it came from. The two methods returned different
answers, and the disagreement is the useful part of this record.

| Calibration | Impostor pairs | FMR=0.1% threshold |
|---|---|---|
| AgeDB, within-subgroup pairs (`benchmark_demographics.py`) | 32,000, matched on gender + age band | **0.2871** |
| Full suite, published protocol pairs (`calibrate_threshold_suite.py`) | 15,500 across LFW/AgeDB-30/CFP-FP/CALFW/CPLFW | **0.2363** |

Per-dataset FMR=0.1% points were tightly clustered — LFW 0.2137, AgeDB-30
0.2388, CFP-FP 0.2094, CALFW 0.2480, CPLFW 0.2404, spread 0.0386 — so the
disagreement is **not** between datasets. It is between *impostor difficulty*.

**Why 0.2363 was rejected.** The published packs pair identities at random, so
most impostors are trivially separable — different sex, decades apart. The
within-subgroup set pairs only same-gender, same-age-band identities. Measured
on that harder distribution:

| Threshold | FMR on protocol pairs | FMR on within-subgroup pairs |
|---|---|---|
| 0.2363 | 0.09% ✓ on target | **0.44%** ✗ 4.4× target |
| 0.2871 | ~0.00% | **0.10%** ✓ on target |

0.2363 reaches the FMR target only on an unrepresentatively easy distribution.
Decisively, the triggering false-match pair scored **0.2405** — *above* 0.2363 —
so adopting the combined-suite number would have reintroduced the exact failure
this change was made to remove.

Real casework impostors resemble the hard set: examiners compare people who are
already plausibly similar, not random strangers. **0.2871 was kept.**

#### Accuracy cost, all datasets, all three thresholds

| Dataset | Acc @0.20 | Acc @0.2363 | Acc @0.2871 |
|---|---|---|---|
| LFW | 99.77% | 99.80% | 99.78% |
| AgeDB-30 | 98.13% | 97.97% | 96.68% |
| CFP-FP | 97.43% | 97.11% | 96.27% |
| CALFW | 96.07% | 96.07% | 95.53% |
| CPLFW | 94.32% | 93.65% | 92.55% |
| **Pooled** | **97.15%** | **96.93%** | **96.17%** |

Pooled FNMR rises 5.40% → 6.06% → 7.65%. **Accuracy is highest at 0.20**, which
is the point: 0.20 *is* the accuracy-optimal threshold and it is the one that
produced the false match. About one point of pooled accuracy was traded for a
12× reduction in false matches. CFP-FF is excluded — no cached embeddings, and
it was not estimated.

#### Verification status of the triggering case — read this before citing it

The 0.2405 pair was confirmed to reject **arithmetically only** (0.2405 <
0.2871). The original two images were not available to re-run, so this has
**not** been re-tested end to end through the engine. The score itself came
from a real run, but the rejection is inferred from that recorded score rather
than observed directly. Re-run it if those images become available.

#### What this threshold is not

It is **not** transferable. The optimum is model-specific (`glintr100` tunes
elsewhere) and condition-specific — TinyFace-grade imagery tunes near 0.22 (§4).
Re-derive with `benchmark_demographics.py` (omit `--threshold` for the FMR=0.1%
point) after any change to the model pack or embedding pipeline.

### 5b. Prior measurement — superseded, retained for comparison

The earlier run used **`glintr100`** (not the deployed model) at a threshold of
**0.3089** anchored to FMR=0.1% on this pair set (not the deployed decision
threshold). Both differences matter, so it does not describe production:

| Subgroup | FNMR % | FMR % |
|---|---:|---:|
| ALL | 4.32 | 0.10 |
| gender = female | 6.09 | 0.14 |
| gender = male | 3.11 | 0.05 |
| age 0–25 | 10.66 | 0.10 |
| age 41–55 | 2.55 | 0.10 |

Moving to the deployed configuration lowered FNMR across the board but raised
FMR **~12×** (0.10% → 1.19%), because 0.20 is a far more permissive cut-point
than 0.3089. The demographic *ratios* stayed broadly stable (gender FNMR 2.0× →
2.21×, gender FMR 2.8× → 2.50×), which indicates the disparity is a property of
the models rather than of where the threshold sits.

> These pairs are locally constructed, so the 4.32% aggregate is **not**
> comparable to the published AgeDB-30 figure in §2 and is not quoted as such.
> The meaningful result is the *relative* differential between subgroups
> measured on one consistent pair set at one threshold (cf. NIST FRVT Part 3).

---

## 6b. REOPENED — the blocker was removable (2026-07-31)

§6a below closed fine-tuning as a dead end. **One of its two premises has since
been disproved, so that conclusion no longer stands as written.**

### Item 39 is achievable after all

§6a stated that the pipeline initialises from ImageNet because only ONNX
inference graphs exist, with no PyTorch ArcFace checkpoint. The first half is
true; the conclusion drawn from it was wrong. `onnx2torch` converts the
deployed recogniser into a trainable module:

```
w600k_r50.onnx -> torch module
  parameters : 43,572,288
  trainable  : 43,572,288
  forward    : (2, 512)
  backward   : 160 tensors received gradient
```

That is initialisation **from ArcFace weights**, which is what item 39 asks for
and what the failed attempt lacked. The 49.38%-at-chance result in §6 is
explained by ImageNet initialisation plus ~2,000× too little data; the first of
those is now fixable.

### Deeper audit: contamination is worse than sampled, as predicted

Re-run at **full identity coverage** — 10 images per identity across all 10,572
CASIA-WebFace identities, 105,631 images:

| Depth | Images | Hits ≥0.70 | Near-dup ≥0.90 |
|---|---:|---:|---:|
| 2/identity | 21,144 | 551 | 20 |
| **10/identity** | **105,631** | **1,508** | **85** |

5× the images found **2.7×** the hits and **4.2×** the near-duplicates. Peak
similarity **0.9888** against AgeDB-30 — the same photograph. This confirms the
method limitation stated in §7c: sampling gives a **floor**, never a ceiling.

Per eval set at the deeper depth: AgeDB-30 417 hits / 58 near-dup; CFP-FP 901 /
26; CALFW 91 / 1; LFW 66 / 0; CPLFW 33 / 0.

### Status: achievable, not attempted

Fine-tuning is **not** a dead end. The path is now clear and unblocked:

1. ✅ **ArcFace initialisation** — demonstrated above.
2. ✅ **Contamination measured** at full identity coverage.
3. ✅ **Exclusion list** — the audit reports which *eval* images are matched; it
   must be extended to emit the contaminated *training* identity IDs so they can
   be dropped. → `build_exclusion_list.py`, results in §6c.
4. ✅ **Clean fine-tune + evaluation** → §6d. **The result is negative.**

**Honest expectation, recorded in advance so the result was not oversold:** even
with ArcFace init and a decontaminated subset, ~10k identities is still far below
the ~360k `glintr100` was trained on. The realistic goal is a modest gain on
degraded imagery, not a new state of the art — and a genuine attempt that fails
to improve anything remains a valid, reportable outcome.

That is what happened. See §6d.

---

## 6c. Training-identity exclusion list (step 3)

**Script:** `backend/scripts/build_exclusion_list.py` · **Date:** 2026-07-31

The overlap audit (§6b) answers *"is there contamination?"* from the eval side:
for each evaluation image, how similar is the nearest training image. That
proves contamination exists but cannot be acted on, because it never names the
training identities responsible.

This runs the search in the opposite direction — for each of the 10,572 CASIA
identities, the similarity of its nearest evaluation image — against a pooled
gallery of **62,000 cached eval embeddings** (LFW, AgeDB-30, CFP-FP, CALFW,
CPLFW), using `faiss.IndexFlatIP` (exact; an approximate index could silently
miss a match and leave a contaminated identity in).

Sampled 105,631 images across all 10,572 identities.

| identity max-similarity | identities | share |
|---|---|---|
| ≥ 0.90 | 32 | 0.3% |
| 0.70–0.90 | 176 | 1.7% |
| 0.50–0.70 | 277 | 2.6% |
| 0.40–0.50 | 207 | 2.0% |
| 0.30–0.40 | 6,654 | 62.9% |
| < 0.30 | 3,226 | 30.5% |

**Threshold 0.40 → exclude 692 identities (6.5%), keep 9,880 (93.5%).**

0.40 sits *below* the ~0.49 mean of genuine same-person pairs, which is
deliberate and asymmetric: excluding a clean identity costs a little training
data, while keeping a contaminated one makes every downstream accuracy number
unfalsifiable.

**Limitation — this list is a floor, not a proof.** Sampling cannot establish
that an identity is clean, only that its sampled images did not match. The true
contaminated set is at least this large.

---

## 6d. RESULT — the clean fine-tune did not work (step 4)

**Date:** 2026-07-31 · **Status:** completed and **negative**
**Scripts:** `finetune_degraded.py` (train), `eval_finetuned_checkpoint.py` (score)

Every methodological objection from §6a/§6b was addressed. It still did not
help. Both facts are recorded here.

### What was run

| requirement | how it was met |
|---|---|
| item 36 contamination | trained only on the 9,880 identities kept by §6c |
| item 37 degraded data | ~50% of every batch downscaled to 16–48px and back up, plus Gaussian blur and JPEG q20–60 |
| item 38 held-out val | 500 identities, **disjoint by identity** from the 9,380 trained on |
| item 39 ArcFace init | backbone converted from the deployed `w600k_r50.onnx` via `onnx2torch` — 43,572,288 params, **not** ImageNet |
| item 40 hard negatives | ArcFace angular margin, plus each degraded view is by construction the hard positive of its clean counterpart |
| item 41 early stopping | on validation, patience 4 |

75,018 training images, batch 48, AdamW, 600 steps of frozen-backbone head
warm-up at lr 1e-4 then 2,400 steps unfrozen at lr 1e-5, 1,573 s on the RTX 3060.

### Result — scored against the deployed model on identical pair lists

Both models were embedded **in the same run**, through the same 10-fold
harness that produced §2 and §4, so a stale cache cannot manufacture a
difference. The only variable is the weights.

| dataset | deployed | fine-tuned | Δ acc | TAR@FAR 0.1% | verdict |
|---|---|---|---|---|---|
| LFW | 99.78% | 99.75% | −0.03pp | 99.70 → 99.67 | no change |
| AgeDB-30 | 98.15% | 97.38% | −0.77pp | 96.03 → **86.97** | **WORSE** |
| CFP-FP | 97.44% | 97.23% | −0.21pp | 94.69 → 93.94 | no change |
| CALFW | 95.95% | 95.62% | −0.33pp | 92.10 → 88.63 | no change |
| CPLFW | 94.47% | 93.88% | −0.58pp | 87.40 → 85.13 | no change |
| **TinyFace** | **82.45%** | **79.38%** | **−3.07pp** | 33.13 → **22.23** | **WORSE** |

**It is worst on TinyFace — the exact condition it was built to improve.** At
the 0.1% false-match operating point, a third of the true matches it previously
found are gone (33.13% → 22.23%).

### Two things worth recording

**1. The internal validation metric lied, and the run itself proves it.**
Training reported its val margin improving +0.5474 → +0.6055. But during the
first 600 steps the backbone was *frozen*, and the margin still swung 0.5474 →
0.4887. Frozen weights cannot learn, so that entire swing is sampling noise —
and its magnitude (~0.06) is the same size as the final "gain" (+0.058). The
proxy never had the resolution to answer the question. Only the fixed-pair-list
benchmarks could, and they say the opposite. Any future run must be judged on
§2/§4, never on a training-time proxy.

**2. AUC rose on four sets while accuracy fell.** LFW, AgeDB-30, CFP-FP and
CALFW all show slightly *better* AUC (e.g. AgeDB-30 0.99130 → 0.99184) alongside
worse accuracy and much worse TAR@FAR. Ranking within a pair list held up; what
broke was calibration — the score distribution shifted so a single global
threshold fits it worse. On TinyFace even AUC fell (0.89217 → 0.86940), so there
the loss is genuine discrimination, not just calibration.

### Why it most likely failed

Not stated as proven — these are the candidate causes, in the order the evidence
supports them:

- **Synthetic degradation ≠ real low-resolution capture.** The model was taught
  to invert *this specific* pipeline (bicubic down/up + Gaussian blur + JPEG).
  Real TinyFace imagery is natively low-resolution from distance and optics. The
  drop being *largest* on TinyFace is what a domain-gap failure looks like.
- **9,380 identities against 600k.** Fine-tuning on ~1.5% of the original
  identity count pulls the embedding space toward a narrow slice of it.
- **Catastrophic forgetting of calibration**, consistent with the AUC-up /
  TAR-down split above.

### Decision

**The deployed model stays `buffalo_l` / `w600k_r50`, unchanged.** The
checkpoint is not shipped and no accuracy claim is made for it. Phase 6 is now
*complete* rather than open: the path was built, executed end to end, and
measured — and the measurement says do not adopt it.

This supersedes §6a's "abandoned for contamination." The barrier was removable;
the fine-tune was run properly and simply did not improve the model. That is a
result, not a dead end.

Reproduce:
```
python backend/scripts/build_exclusion_list.py
python backend/scripts/finetune_degraded.py --steps 3000 --warmup 600
python backend/scripts/eval_finetuned_checkpoint.py
```

---

## 6e. QMUL-SurvFace evaluation — audit before training (2026-07-31)

Real degraded training data was the conclusion of §6d. QMUL-SurvFace was
obtained to test it. **No training has been run.** This section reports steps
1–4 only.

### 1. Licence — a blocking question, not a technical one

The archive ships **no licence file**. `readme.txt` contains structure, protocol
and citation, and nothing else. The published terms say the dataset

> is made available for research purposes

and, materially:

> all the images were collected from the existing person re-identification
> datasets, and the copyright belongs to the original owners

Two consequences, both of which are the project owner's call and neither of
which is a technical matter:

- **"Research purposes" is narrower than this product.** NexGen iMATCH is an
  operational forensic tool. A model whose weights were trained on this data
  carries the restriction into whatever the model is used for. That is a
  question to resolve *before* training, because it cannot be undone afterwards
  by deleting the dataset.
- **Copyright is not QMUL's to grant.** It sits with the original
  re-identification datasets, which are not enumerated on the site. At least one
  dataset commonly used in that space (DukeMTMC) was withdrawn by its own
  authors on ethics grounds. Which sources feed SurvFace cannot be established
  from the files on disk.

Nothing here says the dataset cannot be used. It says the terms are
research-scoped and the provenance is not fully traceable from what was
downloaded, and both facts would be discoverable under cross-examination.

**LICENSING BASIS — DECIDED (2026-07-31).** The project owner reviewed the above
and determined that QMUL-SurvFace's research-purposes terms are **consistent
with this project's current use**. Training proceeded on that basis.

Recorded precisely, because this is the part that would be examined:

| | |
|---|---|
| Terms as published | "made available for research purposes" |
| Upstream copyright | held by the original person re-identification datasets, not by QMUL, and not enumerated in the download |
| Licence file in archive | **none** — `readme.txt` only |
| Determination | research-purposes use, consistent with current project use |
| Decided by | project owner, 2026-07-31 |
| Scope of the decision | covers the *current* use. It is not a finding that the terms permit unrestricted commercial redistribution of a model trained on this data, and it does not resolve the untraceable upstream provenance. Both remain open if the use changes. |

Any checkpoint trained on this data inherits the constraint and must carry it in
its provenance record.

### 2. Identity-overlap audit — the raw result, and why it is wrong

`audit_qmul_survface.py`, same method as §6c, against a gallery of **84,171
embeddings covering all seven evaluation sets** (LFW, AgeDB-30, CFP-FP, CFP-FF,
CALFW, CPLFW, TinyFace — CFP-FF and TinyFace were embedded for this).
78,733 QMUL images sampled at ≤40/identity across all 5,319 identities.

| identity max-similarity | identities | share |
|---|---|---|
| ≥ 0.90 | 1 | 0.0% |
| 0.70–0.90 | 1,490 | 28.0% |
| 0.50–0.70 | 3,391 | 63.8% |
| 0.40–0.50 | 271 | 5.1% |
| < 0.40 | 166 | 3.1% |

At the §6c threshold of 0.40 this excludes **5,153 of 5,319 identities (96.9%)**,
leaving 166. Nearest neighbours land overwhelmingly in TinyFace (78.2%).

Reported as-is that reads as near-total contamination. **It is not.**

### 3. The control that overturns it

Both QMUL and TinyFace are native low-resolution capture, and ArcFace embeddings
of very low quality faces collapse toward a common region of the hypersphere —
a 27×22px face encodes mostly "degraded face", not "this person". A
QMUL↔TinyFace affinity is exactly what that artefact produces with **zero**
shared identities.

`qmul_overlap_control.py` separates the two explanations using ground truth that
is free here: distinct QMUL directories are distinct people by construction.

| measurement | median |
|---|---|
| QMUL genuine (same person, single pair) | 0.316 |
| QMUL impostor (different people, single pair) | 0.151 |
| nearest TinyFace neighbour (max over 8,171) | 0.522 |
| **nearest different-person QMUL (max over 6,693) — matched null** | **0.600** |
| LFW clean impostor, for scale | 0.003 |
| LFW clean genuine, for scale | 0.689 |

**The matched null is HIGHER than the TinyFace affinity (0.600 vs 0.522,
separation −0.078).** A QMUL face resembles an arbitrary *different* QMUL person
more than it resembles anything in TinyFace. And a true same-person QMUL pair
medians 0.316 — well *below* the best unrelated TinyFace match. If identity
drove the affinity, genuine pairs would be the strongest signal present. They
are the weakest.

**Conclusion: no detectable identity contamination between QMUL-SurvFace and any
of the seven evaluation sets. The 96.9% figure is an artefact of quality-induced
embedding collapse and must not be quoted as contamination.**

A methodological note, recorded because it nearly went the other way: the first
version of this control compared the nearest TinyFace neighbour (a *maximum over
8,171 candidates*) against a single random impostor pair (*one draw*), and
printed "SPECIFIC — treat the overlap as real". The maximum of 8,171 draws
exceeds a single draw regardless of the underlying distribution, so that test
would have declared contamination on unrelated data. Comparing max-of-N against
a matched max-of-N reversed the verdict. The 0.40 threshold from §6c is
calibrated for clean-vs-clean and carries no meaning at this image quality.

### 4. Quality characterisation — genuinely native low-resolution

`qmul_quality_stats.py`, 40,000 images sampled of 220,888 on disk.

| | QMUL-SurvFace | TinyFace (target) | CASIA (clean ref) |
|---|---|---|---|
| height px p5/p50/p95 | 11 / **27** / 35 | 31 / **32** / 32 | 112 / 112 / 112 |
| width px p5/p50/p95 | 9 / **22** / 29 | 24 / **32** / 32 | 112 / 112 / 112 |
| range | 7×5 … 117×106 | 14×12 … 32×32 | 112×112 |
| under 32px high | **84.1%** | 6.4% | 0% |

Confirmed native low-resolution capture, not downsampled clean imagery. QMUL is
in fact **harder than the target**: TinyFace is standardised at a 32px cap, while
84% of QMUL sits below 32px and the smallest images are 7×5.

Variance-of-Laplacian was also measured (QMUL 460, TinyFace 771, CASIA 301) but
**is not interpretable across these groups** — the statistic is computed at
native resolution and scales with pixel density, so it compares three different
things. It is recorded for completeness, not used as evidence.

**Detectability:** SCRFD finds a face in **0%** of both QMUL *and* TinyFace
images at native resolution. This is not a QMUL defect — both datasets are
pre-cropped face chips, and the existing TinyFace benchmark already bypasses
detection and resizes straight to 112×112. QMUL enters the pipeline the same
way. No extra pre-processing is required.

### 5. One finding that matters more than the audit

On QMUL imagery the deployed model barely separates identities at all. Genuine
pairs median 0.316 against an impostor median of 0.151, with heavy overlap
(genuine p25 0.180 vs impostor p75 0.268). Compare LFW, where genuine p5 (0.510)
sits cleanly above impostor p95 (0.100).

Read two ways, both true:

- **Opportunity.** This is real headroom — precisely the regime §6d failed to
  reach with synthetic degradation, and the reason this dataset is the right
  next attempt.
- **Risk.** With separation this weak, ArcFace's angular margin may not find
  usable gradient, and any per-identity label noise in the source re-ID data
  will be amplified rather than averaged out.

### Status

Steps 1–4 complete. Overlap audit clean, so no exclusion list needs to be
applied — `runtime/benchmarks/qmul_exclusion_list.json` is retained as the raw
audit record, **not** as a list to act on. Training has not started, pending the
licence decision in §1.

Reproduce:
```
python backend/scripts/audit_qmul_survface.py
python backend/scripts/qmul_overlap_control.py
python backend/scripts/qmul_quality_stats.py
```

---

## 6f. Putting the QMUL checkpoint to use — quality-routed selection

**Date:** 2026-08-01 · **Script:** `backend/scripts/evaluate_routed_engine.py`
**Status:** promising, **not yet adoptable** — the operating point is unvalidated.

§6d/§6e recorded the QMUL fine-tune as "no accuracy improvement". True, but the
accuracy column hides the shape of the result. At FAR=0.1%:

| | deployed | QMUL checkpoint |
|---|---|---|
| TinyFace TAR | 33.13% | **38.10%** |
| AgeDB-30 TAR | 96.03% | 88.10% |
| CPLFW TAR | 87.40% | 81.73% |

That is not a worse model, it is a **different** one: better on degraded
capture, worse on clean-but-hard (age, pose). Picking one globally discards
whichever advantage it does not choose. So the question became whether choosing
**per probe** recovers both.

### Two facts that make routing possible

**The embedding spaces are compatible.** Same image through both models gives a
median cosine of **+0.856**. Fine-tuning at lr 1e-5 refined the space rather
than rotating it. This matters far beyond routing: a gallery enrolled under one
model does **not** have to be re-enrolled to be searched with the other, so 1:N
does not need two templates per subject. That was the assumption most likely to
kill this idea, and it does not hold.

**The pipeline's existing quality score separates the conditions.** No new model
or inference is needed — it is already computed on every request:

| | median | p10 | p90 |
|---|---|---|---|
| clean sets (6) | 0.781 | — | — |
| **TinyFace** | **0.502** | 0.428 | 0.564 |

### The tradeoff curve

Routing a pair to the specialist when **either** image falls below the
threshold (the weaker image limits the comparison):

| threshold | TinyFace TAR@FAR0.1% | worst clean-set TAR change |
|---|---|---|
| **0.50** | **38.63%** (+5.50pp) | **−0.17pp** |
| 0.54 | 37.43% | −0.13pp |
| 0.56 | 37.67% | −1.13pp |
| 0.58 | 37.93% | −1.10pp |
| 0.60–0.64 | 38.13% | −1.03pp |
| 0.68 | 38.10% | −2.00pp |

At 0.50 the routed engine beats **both** single models — above the specialist's
own 38.10% — because it sends only the *worst* imagery to the specialist and
keeps the deployed model for moderately degraded faces. The specialist is not
better at "degraded"; it is better at "very degraded", and routing finds that
line.

### Why this is NOT yet a result to quote

**0.50 was identified by looking at the reporting benchmarks.** Choosing an
operating point because it scores well on the sets it will then be reported
against is fitting to the test set, and every number downstream becomes
unfalsifiable — the same error class as the §6d proxy, in a new place.

The threshold must be derived from data disjoint from the seven benchmarks
(QMUL-SurvFace and CASIA quality distributions are the obvious source, and are
already on disk), and only then measured here. Until that is done, the honest
statement is: *routing looks capable of +5pp TinyFace TAR for ~0.2pp clean
cost, and the operating point has not been independently established.*

The script derives a threshold from the quality distributions by default
(currently 0.581) rather than taking the best sweep value, so it cannot
silently report a test-set-fitted number.

### What would make it adoptable

1. Choose the threshold on QMUL/CASIA quality distributions only, then re-run.
2. If it holds, ship the specialist as an **opt-in** second pack with its own
   version tag, never replacing `buffalo_l` as the default (item 13).
3. 1:1 verification can route immediately. 1:N should stay single-model until
   the +0.856 space compatibility is verified at gallery scale, since a
   rank-ordering is more sensitive to small embedding drift than a single
   threshold comparison is.

Reproduce:
```
python backend/scripts/evaluate_routed_engine.py --sweep
```

---

## 6a. DECISION — fine-tuning abandoned (SUPERSEDED, see §6b)

**Date:** 2026-07-31 · **Decided by:** project owner · **Status:** closed, not deferred

Custom fine-tuning is **not being pursued**. This is a completed investigation
with a negative result, not an unfinished task. Two independent findings led
here, either of which would have been sufficient.

**1. Every available training archive is contaminated** (§7c). All three
overlap the evaluation sets — UMDFaces severely (292 hits per 1k images, 253
near-duplicates, peak similarity 0.9890 against AgeDB-30), CASIA-WebFace and
MegaFace-train comparably to each other (~25–26 per 1k). There is no clean
training corpus on disk. Any accuracy gain measured against these evaluation
sets would be partly memorisation, and in a forensic system that is a number
which could not survive cross-examination.

**2. The one attempt that ran produced a model at chance** (§6). AgeDB-30
49.38% against a 50% baseline, AUC 0.4897. The pipeline initialised from
ImageNet weights rather than ArcFace and trained on 8,738 images against
glintr100's ~17M — roughly 2,000× less data. That gap is not closable by
tuning.

### What would have to change before revisiting

- A training corpus with **verified** zero overlap against the evaluation sets
  — either a full-depth exclusion list (the audit in §7c is sampled, so it
  gives a floor, not a complete list) or a held-out evaluation set built from
  data never present in any training archive.
- Initialisation from **ArcFace** weights, not ImageNet.
- Enough data that fine-tuning can plausibly beat a model trained on 17M images.

None of these are satisfied today.

### What was NOT abandoned

The training pipeline itself is fixed and works: the BatchNorm crash is
resolved, a full epoch completes, gradients are stable, and checkpoints are
versioned. `benchmark_finetuned.py` exists and correctly rejected the one
candidate produced. The machinery is sound; the data is not.

**Deployed model is unchanged: stock `buffalo_l` / `w600k_r50`.** Nothing in
this document was trained on the contaminated archives, so every figure here
stands.

---

## 6. Fine-tuned checkpoint — REJECTED

```bash
python backend/scripts/benchmark_finetuned.py \
  --checkpoint runtime/checkpoints/arcface_ft_v1_20260730.pt
```

Checkpoint: 1 epoch, 8,738 samples, 300 identities, 273 steps.
Loss 39.83 → 37.54, gradient norms stable (mean 73.7, 0 non-finite).

| Dataset | Fine-tuned | glintr100 (stock) | Δ |
|---|---|---|---|
| LFW | 56.62% ± 2.19 | 99.77% | **−43.15** |
| AgeDB-30 | **49.38% ± 1.41** | 98.32% | **−48.94** |

**AgeDB-30 accuracy is 49.38% against a 50% chance baseline, with AUC 0.48971 —
below 0.5. The checkpoint carries no usable identity signal.**

### Why — and why this is not a tuning problem

`train_pipeline.py` does not fine-tune an ArcFace model. It builds a
`torchvision.resnet50` from **ImageNet** classification weights and attaches a
fresh 512-d embedding head. It is training a face recognizer from scratch, from
a starting point that was never trained on faces.

The scale gap is decisive:

| | This run | glintr100 |
|---|---|---|
| Identities | 300 | ~360,000 |
| Images | 8,738 | ~17,000,000 |
| Epochs | 1 | many |
| Init | ImageNet classification | — |

That is ~2,000× fewer images. No learning-rate or schedule change closes it.

**Decision: not shipped, and not merged into the ensemble.** The training
pipeline now runs correctly end-to-end (the Phase 2 crash is fixed and a full
epoch completes), but a working pipeline is not a useful model. Loading this
checkpoint into production would replace a 98.32% backbone with a coin flip.

To make fine-tuning worthwhile it would have to start from the ArcFace weights
themselves — not ImageNet — at a low learning rate, and even then it must beat
the §2 table on this protocol before shipping.

---

## 7. What is not measured

Stated explicitly rather than omitted:

| Item | Status | Reason |
|---|---|---|
| IJB-B / IJB-C | Not run | Datasets not present locally; not downloadable in this environment (see below) |
| Fine-tuned model in-ensemble | Not run | Pointless — the standalone checkpoint is at chance (§6); fusing it can only degrade the result |
| CFP-FF | Measured as a diagnostic control (99.91 ± 0.15), see §2b | Near-saturated as expected; used to confirm the CFP-FP anomaly was not a code fault |
| Identification (rank-1/CMC) | Not re-measured | Out of scope here; this document is verification-only by design |

**Environment note (resolved):** during most of this work, Windows Smart App
Control blocked the interpreter's `_ssl` module and the execution of any newly
created `python.exe`, so `pip` could not reach an index and no fresh virtualenv
could be built. Smart App Control has since been disabled; both are fixed, and
the environment has been rebuilt from the requirements files and re-verified
(`python scripts/setup_gpu.py --check` → 12/12 PASS). IJB-B/IJB-C remain
unmeasured because the datasets are not on disk, not because of tooling.

---

## 7b. Latency and throughput

```bash
python backend/scripts/benchmark_speed.py --iterations 50
```

Measured on: RTX A3000 Laptop, CUDA 12.1, onnxruntime 1.20.1,
`buffalo_l` / `w600k_r50` on `CUDAExecutionProvider`, Windows 11, Python 3.11.15.
Percentiles are **nearest-rank** — every figure is an observation that actually
occurred, never an interpolation.

> **These numbers are for 112×112 pre-cropped AgeDB images.** Latency scales
> with input resolution: a 4000×3000 phone photo costs substantially more in
> decode and detect. Do not quote these as general-purpose figures.

### Single-image encode (decode → detect → align → quality/liveness → embed)

| p50 | p95 | p99 | max | mean ± sd | Throughput |
|---|---|---|---|---|---|
| 14.72 ms | 17.50 ms | 18.98 ms | 18.98 ms | 15.20 ± 1.44 ms | **65.8 img/s** single-threaded |

### Per-stage breakdown (pipeline's own `StageTimings` instrumentation)

| Stage | Mean | Share |
|---|---|---|
| embed | 6.00 ms | 41.1% |
| detect | 5.73 ms | 39.2% |
| align | 0.90 ms | 6.2% |
| quality | 0.36 ms | 2.4% |
| decode | 0.21 ms | 1.4% |
| **total** | **14.61 ms** | 100% |

Embedding and detection together are 80% of the cost. Any optimisation effort
belongs there; the remaining stages are noise.

### Verify 1:1 — one real API request (two encodes + cosine)

| p50 | p95 | p99 | Throughput |
|---|---|---|---|
| 31.04 ms | 32.60 ms | 32.70 ms | **33.1 verifications/s** single-threaded |

### Gallery search — brute-force cosine

| Gallery size | p50 | p95 | Throughput |
|---|---|---|---|
| 100 | 0.198 ms | 0.380 ms | 4,416 /s |
| 1,000 | 0.207 ms | 0.248 ms | 4,987 /s |
| 10,000 | 1.087 ms | 1.365 ms | 890 /s |
| 100,000 | **15.981 ms** | 18.147 ms | **61 /s** |

**This is the quantified answer to "does brute-force scale?"** Search cost is
negligible up to ~10k templates (1 ms, under 8% of one encode). At 100k it
reaches 15.98 ms — **roughly the same cost as encoding the probe image itself**,
so the search stops being free and starts doubling request latency. Scaling is
linear, as expected for a dense matmul.

### 7b-i. Concurrency and request batching (item 29)

```bash
python backend/scripts/benchmark_concurrency.py --total 100
```

**Thread concurrency — full pipeline, RTX A3000**

| Workers | Throughput | p50 | p95 | p99 | Scaling |
|---:|---:|---:|---:|---:|---:|
| 1 | 70.9/s | 13.79 ms | 16.16 ms | 16.65 ms | 1.00× |
| 2 | 112.7/s | 17.21 ms | 21.56 ms | 22.63 ms | 1.59× |
| 4 | **131.9/s** | 29.52 ms | 37.57 ms | 43.45 ms | **1.86×** |
| 8 | 130.3/s | 58.00 ms | 86.79 ms | **99.86 ms** | 1.84× |

**Threading saturates at 4 workers.** Going from 4 to 8 buys *no* throughput
(131.9 → 130.3/s) while p99 latency degrades **2.3×** (43 → 100 ms). More
uvicorn workers past 4 is strictly worse: the same work, much slower tails.

**Request batching — recognition model only**

| Batch | Throughput | ms/image | Speedup |
|---:|---:|---:|---:|
| 1 | 195.4/s | 5.118 | 1.00× |
| 4 | 399.7/s | 2.502 | 2.05× |
| 32 | 472.3/s | 2.118 | 2.42× |
| 64 | **551.2/s** | **1.814** | **2.82×** |

**Batching is the better lever — 2.82× versus threading's 1.86×, and without
the latency penalty.** Per-call overhead (blob construction, session dispatch,
memory transfer) is paid once instead of N times.

**What this means for the build.** If throughput becomes a constraint, the fix
is a request-collecting queue in front of the recogniser, **not** more workers.
Threading is already at its ceiling; batching has headroom.

**Scope, stated honestly:** this measures the engine under load, not the full
HTTP stack. Real end-to-end throughput is additionally bounded by request
parsing, base64 decoding and database writes. Treat these as an upper bound on
what the recognition path can sustain, not a service-level SLO. Batching is
measured on the recogniser alone — detection is per-image and would need
batching separately.

---

## 7c. TRAIN/EVAL IDENTITY OVERLAP AUDIT — overlap found in all three sets

```bash
python backend/scripts/audit_train_eval_overlap.py --per-identity 2
```

**Result: OVERLAP DETECTED. No further fine-tuning may use these archives
against these evaluation sets without an exclusion list.**

Name matching is impossible — the `.bin` packs carry no identity metadata
(verified by scanning pickle opcodes without executing them) and the training
sets carry only numeric IDs (`/CASIA-WebFace/0000045/001.jpg`) or Flickr
handles. Detection is therefore embedding-based.

### Rate-normalised, so different sample depths compare fairly

| Training set | Images | Identities | ≥0.70 hits | **per 1k images** | Near-dup ≥0.90 |
|---|---:|---:|---:|---:|---:|
| CASIA-WebFace | 21,144 | 10,572 (100%) | 551 | **26.1** | 20 |
| **UMDFaces** | 16,554 | 8,277 (100%) | 4,833 | **292.0** | **253** |
| MegaFace-train | 22,000 | 22,000 (3.3%) | 553 | **25.1** | 1 |

**UMDFaces is ~11× worse than either alternative** and overlaps all five
evaluation sets, peaking at similarity **0.9890** against AgeDB-30. A genuine
same-person pair averages ~0.49 in this system; **0.98–0.99 is the same
photograph**, not a similar person.

CASIA-WebFace and MegaFace-train are comparable per-image (26.1 vs 25.1) and
roughly an order of magnitude cleaner than UMDFaces.

### Why the MegaFace numbers are floors, not totals

Only 22,000 of its 657,078 identities were sampled — **3.3% coverage**, versus
100% for the other two. Its raw counts are therefore not comparable to theirs;
only the per-1k-image rate is. At full coverage its absolute count would be
far higher.

### What this does and does not invalidate

- **Current BENCHMARKS.md figures stand.** The deployed model is stock
  `buffalo_l` / `w600k_r50`, trained by InsightFace on WebFace600K — not on
  these local archives. Nothing reported here was trained on contaminated data.
- **It does explain why the rejected fine-tune could never have been trusted.**
  That checkpoint scored 49.38% (chance), so contamination never got a chance
  to inflate it — but had it trained successfully, its numbers would have been
  partly memorisation.

### Method limitation, stated so it is not overclaimed

Sampling proves overlap **exists**; it cannot prove overlap is **absent**. An
identity whose sampled images happen not to resemble the evaluation shots is
missed. The true overlap is necessarily **larger** than the figures above.

The script exits `2` on detection so it can gate a training run in CI.

---

## 7d. ANN SEARCH — latency vs recall (item 28)

```bash
python backend/scripts/benchmark_ann.py --real --sizes 10000 50000
```

**Recall, not speed, decides adoption.** An approximate index that misses a
candidate has silently withheld an investigative lead.

### Measured on REAL ArcFace templates (50,000 gallery, top_k=10)

| Index | p50 | qps | recall@1 | recall@10 |
|---|---|---|---|---|
| exact numpy *(production)* | 4.94 ms | 201 | 1.000 | 1.000 |
| **faiss IndexFlatIP (exact)** | **2.07 ms** | **466** | **1.000** | **1.000** |
| IVF-PQ nprobe=32 | 2.87 ms | 350 | 0.575 | 0.811 |
| HNSW efSearch=256 | 0.72 ms | 1,314 | 0.615 | 0.968 |
| HNSW efSearch=16 | 0.04 ms | 23,180 | 0.610 | 0.956 |

### The synthetic run was invalid, and is retained only as a caution

A first pass used random 512-d unit vectors and showed approximate recall@1
collapsing to 0.005–0.465. **That was an artifact of the data, not a
prediction.** Random high-dimensional vectors are nearly equidistant, so there
is no cluster structure for IVF or HNSW to exploit. Real face embeddings
cluster by identity — which is precisely what these indexes exist to use — and
recall roughly doubled when measured on real templates.

Quoting the synthetic numbers as the adoption answer would have been wrong.

### Decision: adopt faiss `IndexFlatIP`; do NOT adopt an approximate index

**`IndexFlatIP` is a free win.** It is *exact* — recall 1.000 by construction,
not by measurement — and still **2.4× faster** than the numpy path at 50k
(4.94 ms → 2.07 ms). This is the constant-factor speedup predicted in §7b;
the prediction was right that it does not change complexity, and wrong that it
was therefore not worth having.

**Approximate indexes are rejected at current tuning.** HNSW is dramatically
faster (up to 23,000 qps) but recall@1 sits at **0.61** on real data: roughly
two in five searches return a different top-1 than exact search. For lead
generation that is not acceptable without a much stronger recall guarantee.

Caveat on recall@1: it compares gallery *indices*, and these packs contain
near-duplicate images of the same identity, so a differing index does not
always mean a different person. recall@10 (0.968) is the more forgiving and
probably more operationally relevant figure. Even so, neither approaches the
1.000 that exact search gives for free at this scale.

**Revisit if the gallery exceeds ~100k**, where exact search costs ~11 ms and
the trade may become worth measuring again — against real templates, with
identity-level rather than index-level recall.

---

## 7e. FUSION METHOD BY CONDITION (item 35)

```bash
python backend/scripts/benchmark_fusion.py
```

Seven fusion methods scored from the cached embeddings, so these cannot
disagree with §2 — same inputs, same harness.

| Fusion method | LFW | AgeDB-30 | CFP-FP | CALFW | CPLFW | **Clean avg** |
|---|---|---|---|---|---|---|
| single `w600k_r50` *(deployed pack)* | 99.78 | 98.15 | 97.44 | 95.95 | 94.47 | 97.16 |
| **single `glintr100`** *(default)* | 99.77 | **98.32** | **97.71** | **96.17** | **94.78** | **97.35** |
| single `w600k_mbf` | 99.60 | 96.33 | 96.00 | 95.60 | 92.63 | 96.03 |
| dual r50+r100 | **99.80** | 98.30 | 97.50 | 96.15 | 94.47 | 97.24 |
| weighted .45/.45/.10 | **99.80** | 98.32 | 97.47 | 96.07 | 94.32 | 97.19 |
| equal 1/3 | 99.77 | 98.07 | 97.21 | 96.08 | 93.92 | 97.01 |
| concat 1536-d | 99.77 | 98.02 | 97.56 | 96.05 | 94.33 | 97.14 |

**On clean protocols `single_glintr100` wins**, confirming the default. No
ensemble beats the best single model — consistent with §3.

### The winner is condition-dependent

Combining with the TinyFace measurements in §4, for the single-model configs:

| Config | Clean avg | TinyFace |
|---|---|---|
| `glintr100` *(fusion default)* | **97.35** | 79.68 |
| `w600k_r50` *(deployed pack)* | 97.16 | **82.45** |

**They disagree.** `glintr100` is better on clean imagery; `w600k_r50` is
better by 2.8 points on degraded. A single static fusion default is therefore
the wrong shape — the right selector is the probe's measured quality, which the
pipeline already computes.

**Not yet measured:** whether the *multi-model fusion* methods are also
condition-dependent. `benchmark_tinyface.py` does not cache embeddings in the
format `benchmark_fusion.py` reads, so only the single-model comparison above
is evidenced. The script says so and refuses to guess rather than reporting a
clean-only winner as if it were global.

**Current state is coherent but not optimal:** the deployed *pack* is
`w600k_r50` (degraded-optimised) while the fusion *default* names
`single_glintr100`. Since the ensemble classes are not loaded unless selected,
the service runs `w600k_r50` — the right choice for casework. The naming is
confusing and worth reconciling.

---

## 8. Reproducibility

- Embeddings are cached per (dataset, backbone) under
  `runtime/benchmarks/embeddings/`, so re-scoring a new fusion configuration
  costs no GPU time.
- Raw per-fold results: `runtime/benchmarks/verification_results.json`.
- Metric implementations (`_best_threshold`, `_tar_at_far`, `_auc_eer`) are
  unit-tested against known-answer cases: perfectly separable, fully
  overlapping, and null-model inputs where TAR@FAR must equal the target FAR.
- GPU binding is asserted before every run; a CPU fallback aborts the benchmark
  rather than silently producing the same numbers 20× slower.
