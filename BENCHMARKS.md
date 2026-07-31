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

Concurrency is **not** measured: all figures are single-threaded. Throughput
under parallel load requires the request-batching work and has not been run.

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
