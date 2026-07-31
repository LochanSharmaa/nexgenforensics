# A5 — Benchmark and Measurement Record

**Generated:** 2026-08-01 00:52 ·
**Repository state:** `bc7d30e61eef`

Every measurement this system has produced, with the protocol under which it was
taken, the harness that took it, the result, and the complete raw artefact.

The raw JSON is reproduced in full for each measurement. A report that presents
only summary tables asks the reader to trust the summarisation; including the
artefact means the summary can be recomputed. For numbers that may be
challenged, that is the point of the document.

Protocol descriptions are read from the docstrings of the scripts that implement
them, so a protocol cannot drift from the code enforcing it.

---

## Governing protocol

| Item | Value |
|---|---|
| Task | 1:1 verification unless explicitly stated |
| Pair source | Official InsightFace `.bin` packs; counts asserted at load |
| Images | Pre-aligned 112x112 ArcFace crops |
| Augmentation | Horizontal-flip TTA, `L2norm(f(x) + f(flip(x)))` |
| Cross-validation | 10-fold, contiguous folds per published layout |
| **Threshold** | **Fitted on 9 folds, applied to the held-out fold** |
| Similarity | Cosine, range [-1, 1] |

**No accuracy figure in this record is measured at a threshold tuned on the
pairs it is reported against.**

---

## Index of measurements

| Family | Artefacts |
|---|---|
| 1:1 verification — clean protocols | 2 |
| Degraded-condition verification (TinyFace) | 1 |
| Threshold calibration | 2 |
| Demographic differentials | 4 |
| Training/evaluation contamination | 7 |
| Fine-tuning outcomes | 3 |
| Quality-routed model selection | 3 |
| Latency, throughput and concurrency | 2 |
| Approximate nearest-neighbour search | 2 |

---

# 1:1 verification — clean protocols

## Protocol as implemented — `backend/scripts/benchmark_verification.py`

```text
1:1 VERIFICATION benchmark across the standard InsightFace protocol packs.

    python backend/scripts/benchmark_verification.py --datasets lfw agedb_30 cfp_fp calfw cplfw

Embeddings are extracted ONCE per (dataset, backbone) and cached to disk, then
every fusion configuration is scored from the cache. Re-running to add a new
fusion config therefore costs no GPU time.

Reported accuracy always comes from 10-fold cross-validation where the
threshold is fitted on 9 folds and applied to the held-out fold. See
nexgen_engine/benchmarks/verification.py for the protocol rationale.
```

## Protocol as implemented — `backend/nexgen_engine/benchmarks/verification.py`

```text
1:1 face VERIFICATION benchmarking on the standard InsightFace protocol packs.

WHAT THIS MEASURES (and what it deliberately does not)
------------------------------------------------------
This module measures **1:1 verification**: given two images, decide
same-person / different-person by thresholding cosine similarity. That is the
task the product's "compare two faces" feature performs.

It is NOT rank-1 closed-set identification ("which of these N enrolled people
is this probe?"). Those two tasks produce different numbers and are not
interchangeable. Any identification result belongs in a separate report.

PROTOCOL
--------
Pairs come from the published `.bin` verification packs shipped with the
ArcFace/InsightFace training sets (lfw.bin, agedb_30.bin, cfp_fp.bin,
calfw.bin, cplfw.bin). Each pack is a pickle of
`(encoded_images, issame_flags)` where images 2i and 2i+1 form pair i. These
are the standard published pair lists -- 6,000 pairs for LFW / AgeDB-30 /
CALFW / CPLFW and 7,000 for CFP-FP -- not ad hoc pairs generated here, so the
numbers are directly comparable to published results.

Images in these packs are already ArcFace-aligned 112x112 crops. They are fed
straight to the recognition network. Re-running face detection on an
already-tight crop is what the reference implementations avoid, and doing so
measurably degrades accuracy.

THRESHOLD HANDLING (the part that is easy to get wrong)
-------------------------------------------------------
Accuracy is computed with the standard 10-fold cross-validation protocol. For
each fold, the decision threshold is selected on the OTHER 9 FOLDS and then
applied to the held-out fold. The reported accuracy is the mean over the 10
held-out folds, with standard deviation.

This means no reported number is ever measured at a threshold that was tuned
on the same pairs. A single global "best threshold accuracy" over all pairs
is also computed, but only as `oracle_accuracy` -- it is optimistically
biased and must never be quoted as the system's accuracy.
```

## Measurement — `runtime/benchmarks/fusion_selection.json`

### Values

| Field | Value |
|---|---|
| `datasets[0]` | lfw |
| `datasets[1]` | agedb_30 |
| `datasets[2]` | cfp_fp |
| `datasets[3]` | calfw |
| `datasets[4]` | cplfw |
| `note` | Scored from cached embeddings, same inputs as BENCHMARKS.md §2. |
| `accuracy_pct.single_r50 (DEPLOYED pack).lfw` | 99.783 |
| `accuracy_pct.single_r50 (DEPLOYED pack).agedb_30` | 98.15 |
| `accuracy_pct.single_r50 (DEPLOYED pack).cfp_fp` | 97.443 |
| `accuracy_pct.single_r50 (DEPLOYED pack).calfw` | 95.95 |
| `accuracy_pct.single_r50 (DEPLOYED pack).cplfw` | 94.467 |
| `accuracy_pct.single_glintr100 (default).lfw` | 99.767 |
| `accuracy_pct.single_glintr100 (default).agedb_30` | 98.317 |
| `accuracy_pct.single_glintr100 (default).cfp_fp` | 97.714 |
| `accuracy_pct.single_glintr100 (default).calfw` | 96.167 |
| `accuracy_pct.single_glintr100 (default).cplfw` | 94.783 |
| `accuracy_pct.single_mbf.lfw` | 99.6 |
| `accuracy_pct.single_mbf.agedb_30` | 96.333 |
| `accuracy_pct.single_mbf.cfp_fp` | 96.0 |
| `accuracy_pct.single_mbf.calfw` | 95.6 |
| `accuracy_pct.single_mbf.cplfw` | 92.633 |
| `accuracy_pct.dual r50+r100.lfw` | 99.8 |
| `accuracy_pct.dual r50+r100.agedb_30` | 98.3 |
| `accuracy_pct.dual r50+r100.cfp_fp` | 97.5 |
| `accuracy_pct.dual r50+r100.calfw` | 96.15 |
| `accuracy_pct.dual r50+r100.cplfw` | 94.467 |
| `accuracy_pct.weighted .45/.45/.10.lfw` | 99.8 |
| `accuracy_pct.weighted .45/.45/.10.agedb_30` | 98.317 |
| `accuracy_pct.weighted .45/.45/.10.cfp_fp` | 97.471 |
| `accuracy_pct.weighted .45/.45/.10.calfw` | 96.067 |
| `accuracy_pct.weighted .45/.45/.10.cplfw` | 94.317 |
| `accuracy_pct.equal 1/3.lfw` | 99.767 |
| `accuracy_pct.equal 1/3.agedb_30` | 98.067 |
| `accuracy_pct.equal 1/3.cfp_fp` | 97.214 |
| `accuracy_pct.equal 1/3.calfw` | 96.083 |
| `accuracy_pct.equal 1/3.cplfw` | 93.917 |
| `accuracy_pct.concat 1536-d.lfw` | 99.767 |
| `accuracy_pct.concat 1536-d.agedb_30` | 98.017 |
| `accuracy_pct.concat 1536-d.cfp_fp` | 97.557 |
| `accuracy_pct.concat 1536-d.calfw` | 96.05 |
| `accuracy_pct.concat 1536-d.cplfw` | 94.333 |

### Raw artefact

```json
{
  "datasets": [
    "lfw",
    "agedb_30",
    "cfp_fp",
    "calfw",
    "cplfw"
  ],
  "note": "Scored from cached embeddings, same inputs as BENCHMARKS.md \u00a72.",
  "accuracy_pct": {
    "single_r50 (DEPLOYED pack)": {
      "lfw": 99.783,
      "agedb_30": 98.15,
      "cfp_fp": 97.443,
      "calfw": 95.95,
      "cplfw": 94.467
    },
    "single_glintr100 (default)": {
      "lfw": 99.767,
      "agedb_30": 98.317,
      "cfp_fp": 97.714,
      "calfw": 96.167,
      "cplfw": 94.783
    },
    "single_mbf": {
      "lfw": 99.6,
      "agedb_30": 96.333,
      "cfp_fp": 96.0,
      "calfw": 95.6,
      "cplfw": 92.633
    },
    "dual r50+r100": {
      "lfw": 99.8,
      "agedb_30": 98.3,
      "cfp_fp": 97.5,
      "calfw": 96.15,
      "cplfw": 94.467
    },
    "weighted .45/.45/.10": {
      "lfw": 99.8,
      "agedb_30": 98.317,
      "cfp_fp": 97.471,
      "calfw": 96.067,
      "cplfw": 94.317
    },
    "equal 1/3": {
      "lfw": 99.767,
      "agedb_30": 98.067,
      "cfp_fp": 97.214,
      "calfw": 96.083,
      "cplfw": 93.917
    },
    "concat 1536-d": {
      "lfw": 99.767,
      "agedb_30": 98.017,
      "cfp_fp": 97.557,
      "calfw": 96.05,
      "cplfw": 94.333
    }
  }
}
```

## Measurement — `runtime/benchmarks/verification_results.json`

### Per-configuration results (35 rows)

| dataset | config | n_pairs | n_genuine | n_impostor | accuracy_mean | accuracy_std | threshold_mean | threshold_std | oracle_accuracy | oracle_threshold | tar_at_far_1e2 | tar_at_far_1e3 | tar_at_far_1e4 | auc | eer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lfw | single:w600k_r50 (R50) | 6000 | 3000 | 3000 | 0.99783 | 0.00259 | 0.24271 | 0.01348 | 0.99833 | 0.24727 | 0.99767 | 0.99700 | 0.99667 | 0.99943 | 0.00233 |
| lfw | single:glintr100 (R100) | 6000 | 3000 | 3000 | 0.99767 | 0.00260 | 0.28384 | 0.02452 | 0.99817 | 0.28708 | 0.99767 | 0.99633 | 0.99633 | 0.99952 | 0.00267 |
| lfw | single:w600k_mbf (MBF) | 6000 | 3000 | 3000 | 0.99600 | 0.00249 | 0.22025 | 0.00848 | 0.99650 | 0.21504 | 0.99633 | 0.99333 | 0.98800 | 0.99938 | 0.00433 |
| lfw | ensemble:weighted 0.45/0.45/0.10 | 6000 | 3000 | 3000 | 0.99800 | 0.00267 | 0.27695 | 0.01041 | 0.99817 | 0.27380 | 0.99767 | 0.99633 | 0.99633 | 0.99933 | 0.00300 |
| lfw | ensemble:equal 1/3 | 6000 | 3000 | 3000 | 0.99767 | 0.00249 | 0.27118 | 0.00902 | 0.99800 | 0.27534 | 0.99733 | 0.99633 | 0.99500 | 0.99936 | 0.00333 |
| lfw | ensemble:dual r50+r100 | 6000 | 3000 | 3000 | 0.99800 | 0.00267 | 0.26369 | 0.01541 | 0.99817 | 0.25872 | 0.99767 | 0.99633 | 0.99633 | 0.99932 | 0.00300 |
| lfw | ensemble:concat 1536-d | 6000 | 3000 | 3000 | 0.99767 | 0.00271 | 0.24195 | 0.01242 | 0.99800 | 0.23632 | 0.99800 | 0.99667 | 0.99600 | 0.99951 | 0.00267 |
| agedb_30 | single:w600k_r50 (R50) | 6000 | 3000 | 3000 | 0.98150 | 0.00608 | 0.20263 | 0.00042 | 0.98183 | 0.20252 | 0.96933 | 0.96033 | 0.93667 | 0.99130 | 0.02333 |
| agedb_30 | single:glintr100 (R100) | 6000 | 3000 | 3000 | 0.98317 | 0.00603 | 0.22158 | 0.01305 | 0.98450 | 0.23184 | 0.97600 | 0.96333 | 0.95600 | 0.99174 | 0.02100 |
| agedb_30 | single:w600k_mbf (MBF) | 6000 | 3000 | 3000 | 0.96333 | 0.00667 | 0.18391 | 0.00960 | 0.96600 | 0.17417 | 0.93800 | 0.86067 | 0.69733 | 0.98939 | 0.03767 |
| agedb_30 | ensemble:weighted 0.45/0.45/0.10 | 6000 | 3000 | 3000 | 0.98317 | 0.00555 | 0.21914 | 0.00401 | 0.98383 | 0.21768 | 0.97500 | 0.96367 | 0.94500 | 0.99102 | 0.02333 |
| agedb_30 | ensemble:equal 1/3 | 6000 | 3000 | 3000 | 0.98067 | 0.00606 | 0.20783 | 0.00161 | 0.98133 | 0.20712 | 0.97000 | 0.95467 | 0.93900 | 0.99083 | 0.02467 |
| agedb_30 | ensemble:dual r50+r100 | 6000 | 3000 | 3000 | 0.98300 | 0.00706 | 0.20899 | 0.00446 | 0.98383 | 0.20782 | 0.97500 | 0.96533 | 0.94967 | 0.99108 | 0.02367 |
| agedb_30 | ensemble:concat 1536-d | 6000 | 3000 | 3000 | 0.98017 | 0.00603 | 0.20090 | 0.01112 | 0.98217 | 0.19218 | 0.97300 | 0.95967 | 0.94967 | 0.99183 | 0.02233 |
| cfp_fp | single:w600k_r50 (R50) | 7000 | 3500 | 3500 | 0.97443 | 0.01070 | 0.18389 | 0.00181 | 0.97500 | 0.18312 | 0.95486 | 0.94686 | 0.94543 | 0.98023 | 0.04286 |
| cfp_fp | single:glintr100 (R100) | 7000 | 3500 | 3500 | 0.97714 | 0.00930 | 0.22949 | 0.01280 | 0.97757 | 0.23507 | 0.95886 | 0.95571 | 0.95400 | 0.98519 | 0.03800 |
| cfp_fp | single:w600k_mbf (MBF) | 7000 | 3500 | 3500 | 0.96000 | 0.01105 | 0.16144 | 0.00093 | 0.96029 | 0.16100 | 0.92886 | 0.88629 | 0.87514 | 0.97422 | 0.05486 |
| cfp_fp | ensemble:weighted 0.45/0.45/0.10 | 7000 | 3500 | 3500 | 0.97471 | 0.01006 | 0.20241 | 0.00665 | 0.97571 | 0.19749 | 0.95657 | 0.94943 | 0.94600 | 0.98188 | 0.04000 |
| cfp_fp | ensemble:equal 1/3 | 7000 | 3500 | 3500 | 0.97214 | 0.01042 | 0.17690 | 0.01374 | 0.97300 | 0.16908 | 0.95457 | 0.94314 | 0.93314 | 0.97923 | 0.04114 |
| cfp_fp | ensemble:dual r50+r100 | 7000 | 3500 | 3500 | 0.97500 | 0.01023 | 0.20580 | 0.00500 | 0.97571 | 0.20526 | 0.95686 | 0.94943 | 0.94714 | 0.98242 | 0.04029 |
| cfp_fp | ensemble:concat 1536-d | 7000 | 3500 | 3500 | 0.97557 | 0.00995 | 0.18574 | 0.00210 | 0.97600 | 0.18575 | 0.95657 | 0.95143 | 0.94971 | 0.98105 | 0.04086 |
| calfw | single:w600k_r50 (R50) | 6000 | 3000 | 3000 | 0.95950 | 0.01090 | 0.19334 | 0.02047 | 0.96117 | 0.21757 | 0.93133 | 0.92100 | 0.89467 | 0.97755 | 0.05867 |
| calfw | single:glintr100 (R100) | 6000 | 3000 | 3000 | 0.96167 | 0.01051 | 0.22918 | 0.00245 | 0.96233 | 0.23036 | 0.93167 | 0.92300 | 0.61667 | 0.98060 | 0.05667 |
| calfw | single:w600k_mbf (MBF) | 6000 | 3000 | 3000 | 0.95600 | 0.01086 | 0.20034 | 0.00376 | 0.95717 | 0.20015 | 0.92333 | 0.88067 | 0.84267 | 0.97863 | 0.05867 |
| calfw | ensemble:weighted 0.45/0.45/0.10 | 6000 | 3000 | 3000 | 0.96067 | 0.01136 | 0.21370 | 0.00658 | 0.96200 | 0.20888 | 0.93267 | 0.92133 | 0.84033 | 0.97754 | 0.05733 |
| calfw | ensemble:equal 1/3 | 6000 | 3000 | 3000 | 0.96083 | 0.01039 | 0.21530 | 0.00407 | 0.96150 | 0.21381 | 0.93100 | 0.91733 | 0.86133 | 0.97683 | 0.05633 |
| calfw | ensemble:dual r50+r100 | 6000 | 3000 | 3000 | 0.96150 | 0.01136 | 0.21338 | 0.00234 | 0.96217 | 0.21406 | 0.93167 | 0.92267 | 0.83667 | 0.97786 | 0.05633 |
| calfw | ensemble:concat 1536-d | 6000 | 3000 | 3000 | 0.96050 | 0.01028 | 0.20294 | 0.00789 | 0.96183 | 0.20107 | 0.93233 | 0.92100 | 0.84100 | 0.98016 | 0.05700 |
| cplfw | single:w600k_r50 (R50) | 6000 | 3000 | 3000 | 0.94467 | 0.01005 | 0.17394 | 0.00088 | 0.94483 | 0.17368 | 0.89733 | 0.87400 | 0.44000 | 0.96425 | 0.08267 |
| cplfw | single:glintr100 (R100) | 6000 | 3000 | 3000 | 0.94783 | 0.01003 | 0.19203 | 0.00086 | 0.94817 | 0.19162 | 0.90367 | 0.86067 | 0.41700 | 0.97042 | 0.07267 |
| cplfw | single:w600k_mbf (MBF) | 6000 | 3000 | 3000 | 0.92633 | 0.01032 | 0.15928 | 0.00138 | 0.92700 | 0.15884 | 0.85800 | 0.79167 | 0.65233 | 0.95802 | 0.09500 |
| cplfw | ensemble:weighted 0.45/0.45/0.10 | 6000 | 3000 | 3000 | 0.94317 | 0.01071 | 0.18077 | 0.00544 | 0.94500 | 0.18733 | 0.89933 | 0.87800 | 0.39333 | 0.96763 | 0.07867 |
| cplfw | ensemble:equal 1/3 | 6000 | 3000 | 3000 | 0.93917 | 0.01101 | 0.18818 | 0.00922 | 0.94083 | 0.19185 | 0.89067 | 0.84267 | 0.47667 | 0.96603 | 0.08233 |
| cplfw | ensemble:dual r50+r100 | 6000 | 3000 | 3000 | 0.94467 | 0.01090 | 0.19250 | 0.00596 | 0.94567 | 0.19443 | 0.89900 | 0.87733 | 0.39333 | 0.96763 | 0.07767 |
| cplfw | ensemble:concat 1536-d | 6000 | 3000 | 3000 | 0.94333 | 0.01174 | 0.17673 | 0.00390 | 0.94467 | 0.17872 | 0.89800 | 0.87133 | 0.51600 | 0.96897 | 0.07967 |

### Per-fold detail (350 fold records)

The threshold shown for each fold was fitted on the other nine and applied to this one.

#### lfw / single:w600k_r50 (R50)

| Fold | Accuracy | Threshold (fitted on 9) |
|---|---|---|
| 1 | 99.1667% | 0.261869 |
| 2 | 100.0000% | 0.247269 |
| 3 | 100.0000% | 0.247269 |
| 4 | 99.8333% | 0.247269 |
| 5 | 99.5000% | 0.247269 |
| 6 | 99.8333% | 0.217174 |
| 7 | 99.6667% | 0.247269 |
| 8 | 100.0000% | 0.247269 |
| 9 | 100.0000% | 0.247269 |
| 10 | 99.8333% | 0.217174 |
| **mean** | **99.7833%** | std 0.2587pp |

#### lfw / single:glintr100 (R100)

| Fold | Accuracy | Threshold (fitted on 9) |
|---|---|---|
| 1 | 99.1667% | 0.287078 |
| 2 | 99.8333% | 0.326450 |
| 3 | 100.0000% | 0.287078 |
| 4 | 99.8333% | 0.287078 |
| 5 | 99.5000% | 0.287078 |
| 6 | 99.6667% | 0.219628 |
| 7 | 99.6667% | 0.287078 |
| 8 | 100.0000% | 0.287078 |
| 9 | 100.0000% | 0.282758 |
| 10 | 100.0000% | 0.287078 |
| **mean** | **99.7667%** | std 0.2603pp |

#### lfw / single:w600k_mbf (MBF)

| Fold | Accuracy | Threshold (fitted on 9) |
|---|---|---|
| 1 | 99.1667% | 0.225085 |
| 2 | 100.0000% | 0.215040 |
| 3 | 99.6667% | 0.234608 |
| 4 | 99.5000% | 0.237566 |
| 5 | 99.3333% | 0.215040 |
| 6 | 99.6667% | 0.215040 |
| 7 | 99.3333% | 0.215040 |
| 8 | 99.8333% | 0.215040 |
| 9 | 99.8333% | 0.215040 |
| 10 | 99.6667% | 0.215040 |
| **mean** | **99.6000%** | std 0.2494pp |

#### lfw / ensemble:weighted 0.45/0.45/0.10

| Fold | Accuracy | Threshold (fitted on 9) |
|---|---|---|
| 1 | 99.1667% | 0.273799 |
| 2 | 99.8333% | 0.308088 |
| 3 | 100.0000% | 0.273799 |
| 4 | 99.8333% | 0.273799 |
| 5 | 99.5000% | 0.273799 |
| 6 | 100.0000% | 0.271057 |
| 7 | 99.6667% | 0.273799 |
| 8 | 100.0000% | 0.273799 |
| 9 | 100.0000% | 0.273799 |
| 10 | 100.0000% | 0.273799 |
| **mean** | **99.8000%** | std 0.2667pp |

#### lfw / ensemble:equal 1/3

| Fold | Accuracy | Threshold (fitted on 9) |
|---|---|---|
| 1 | 99.1667% | 0.275338 |
| 2 | 100.0000% | 0.275338 |
| 3 | 100.0000% | 0.278016 |
| 4 | 99.8333% | 0.275338 |
| 5 | 99.5000% | 0.275338 |
| 6 | 99.8333% | 0.275338 |
| 7 | 99.6667% | 0.275338 |
| 8 | 100.0000% | 0.275338 |
| 9 | 99.8333% | 0.253205 |
| 10 | 99.8333% | 0.253205 |
| **mean** | **99.7667%** | std 0.2494pp |

#### lfw / ensemble:dual r50+r100

| Fold | Accuracy | Threshold (fitted on 9) |
|---|---|---|
| 1 | 99.1667% | 0.258725 |
| 2 | 99.8333% | 0.309909 |
| 3 | 100.0000% | 0.258725 |
| 4 | 99.8333% | 0.258725 |
| 5 | 99.5000% | 0.258725 |
| 6 | 100.0000% | 0.258725 |
| 7 | 99.6667% | 0.258725 |
| 8 | 100.0000% | 0.258725 |
| 9 | 100.0000% | 0.258725 |
| 10 | 100.0000% | 0.257239 |
| **mean** | **99.8000%** | std 0.2667pp |

#### lfw / ensemble:concat 1536-d

| Fold | Accuracy | Threshold (fitted on 9) |
|---|---|---|
| 1 | 99.1667% | 0.254074 |
| 2 | 100.0000% | 0.236322 |
| 3 | 100.0000% | 0.236322 |
| 4 | 99.8333% | 0.236322 |
| 5 | 99.5000% | 0.236322 |
| 6 | 99.8333% | 0.236322 |
| 7 | 99.5000% | 0.275592 |
| 8 | 100.0000% | 0.235540 |
| 9 | 99.8333% | 0.236322 |
| 10 | 100.0000% | 0.236322 |
| **mean** | **99.7667%** | std 0.2708pp |

#### agedb_30 / single:w600k_r50 (R50)

| Fold | Accuracy | Threshold (fitted on 9) |
|---|---|---|
| 1 | 98.8333% | 0.202520 |
| 2 | 98.8333% | 0.202520 |
| 3 | 99.1667% | 0.203872 |
| 4 | 98.0000% | 0.202520 |
| 5 | 97.8333% | 0.202520 |
| 6 | 98.3333% | 0.202520 |
| 7 | 97.6667% | 0.202273 |
| 8 | 97.3333% | 0.202520 |
| 9 | 97.3333% | 0.202520 |
| 10 | 98.1667% | 0.202520 |
| **mean** | **98.1500%** | std 0.6076pp |

#### agedb_30 / single:glintr100 (R100)

| Fold | Accuracy | Threshold (fitted on 9) |
|---|---|---|
| 1 | 99.0000% | 0.207198 |
| 2 | 99.0000% | 0.207198 |
| 3 | 99.0000% | 0.231845 |
| 4 | 98.6667% | 0.233613 |
| 5 | 97.8333% | 0.231845 |
| 6 | 98.0000% | 0.201329 |
| 7 | 97.6667% | 0.231845 |
| 8 | 97.6667% | 0.207198 |
| 9 | 97.5000% | 0.231845 |
| 10 | 98.8333% | 0.231845 |
| **mean** | **98.3167%** | std 0.6030pp |

#### agedb_30 / single:w600k_mbf (MBF)

| Fold | Accuracy | Threshold (fitted on 9) |
|---|---|---|
| 1 | 96.0000% | 0.192352 |
| 2 | 96.8333% | 0.195059 |
| 3 | 97.3333% | 0.174909 |
| 4 | 96.3333% | 0.195059 |
| 5 | 94.8333% | 0.176093 |
| 6 | 96.6667% | 0.172630 |
| 7 | 96.1667% | 0.192352 |
| 8 | 95.8333% | 0.192352 |
| 9 | 97.0000% | 0.174167 |
| 10 | 96.3333% | 0.174167 |
| **mean** | **96.3333%** | std 0.6667pp |

#### agedb_30 / ensemble:weighted 0.45/0.45/0.10

| Fold | Accuracy | Threshold (fitted on 9) |
|---|---|---|
| 1 | 98.8333% | 0.217684 |
| 2 | 99.1667% | 0.217684 |
| 3 | 99.0000% | 0.217684 |
| 4 | 98.5000% | 0.231125 |
| 5 | 97.8333% | 0.218879 |
| 6 | 98.5000% | 0.217684 |
| 7 | 97.8333% | 0.217684 |
| 8 | 97.5000% | 0.217684 |
| 9 | 97.6667% | 0.217684 |
| 10 | 98.3333% | 0.217619 |
| **mean** | **98.3167%** | std 0.5550pp |

#### agedb_30 / ensemble:equal 1/3

| Fold | Accuracy | Threshold (fitted on 9) |
|---|---|---|
| 1 | 98.5000% | 0.207124 |
| 2 | 99.0000% | 0.207124 |
| 3 | 98.5000% | 0.206530 |
| 4 | 98.5000% | 0.207124 |
| 5 | 97.6667% | 0.207124 |
| 6 | 98.3333% | 0.207124 |
| 7 | 97.6667% | 0.211765 |
| 8 | 96.8333% | 0.210110 |
| 9 | 97.5000% | 0.207124 |
| 10 | 98.1667% | 0.207124 |
| **mean** | **98.0667%** | std 0.6064pp |

#### agedb_30 / ensemble:dual r50+r100

| Fold | Accuracy | Threshold (fitted on 9) |
|---|---|---|
| 1 | 99.0000% | 0.207815 |
| 2 | 99.1667% | 0.207815 |
| 3 | 99.0000% | 0.207815 |
| 4 | 99.0000% | 0.207815 |
| 5 | 97.6667% | 0.207815 |
| 6 | 98.5000% | 0.206461 |
| 7 | 97.5000% | 0.222282 |
| 8 | 97.1667% | 0.207815 |
| 9 | 97.6667% | 0.207815 |
| 10 | 98.3333% | 0.206461 |
| **mean** | **98.3000%** | std 0.7063pp |

#### agedb_30 / ensemble:concat 1536-d

| Fold | Accuracy | Threshold (fitted on 9) |
|---|---|---|
| 1 | 98.8333% | 0.192185 |
| 2 | 98.8333% | 0.192185 |
| 3 | 98.6667% | 0.192185 |
| 4 | 98.0000% | 0.204946 |
| 5 | 97.3333% | 0.215400 |
| 6 | 98.3333% | 0.215400 |
| 7 | 97.3333% | 0.204946 |
| 8 | 97.1667% | 0.185044 |
| 9 | 97.6667% | 0.215400 |
| 10 | 98.0000% | 0.191260 |
| **mean** | **98.0167%** | std 0.6030pp |

#### cfp_fp / single:w600k_r50 (R50)

| Fold | Accuracy | Threshold (fitted on 9) |
|---|---|---|
| 1 | 97.2857% | 0.183123 |
| 2 | 96.8571% | 0.183123 |
| 3 | 98.0000% | 0.182737 |
| 4 | 96.1429% | 0.188871 |
| 5 | 96.1429% | 0.183123 |
| 6 | 99.0000% | 0.185452 |
| 7 | 99.1429% | 0.183123 |
| 8 | 97.7143% | 0.183123 |
| 9 | 96.1429% | 0.183123 |
| 10 | 98.0000% | 0.183123 |
| **mean** | **97.4429%** | std 1.0699pp |

#### cfp_fp / single:glintr100 (R100)

| Fold | Accuracy | Threshold (fitted on 9) |
|---|---|---|
| 1 | 97.5714% | 0.239879 |
| 2 | 97.0000% | 0.235070 |
| 3 | 98.1429% | 0.235070 |
| 4 | 97.2857% | 0.235070 |
| 5 | 96.1429% | 0.235070 |
| 6 | 98.7143% | 0.235070 |
| 7 | 99.2857% | 0.235070 |
| 8 | 98.1429% | 0.235070 |
| 9 | 96.5714% | 0.198070 |
| 10 | 98.2857% | 0.211451 |
| **mean** | **97.7143%** | std 0.9302pp |

#### cfp_fp / single:w600k_mbf (MBF)

| Fold | Accuracy | Threshold (fitted on 9) |
|---|---|---|
| 1 | 94.8571% | 0.161004 |
| 2 | 95.1429% | 0.163306 |
| 3 | 96.4286% | 0.161004 |
| 4 | 94.5714% | 0.161004 |
| 5 | 95.2857% | 0.161004 |
| 6 | 98.0000% | 0.160805 |
| 7 | 97.7143% | 0.161004 |
| 8 | 96.5714% | 0.161004 |
| 9 | 95.7143% | 0.163306 |
| 10 | 95.7143% | 0.161004 |
| **mean** | **96.0000%** | std 1.1047pp |

#### cfp_fp / ensemble:weighted 0.45/0.45/0.10

| Fold | Accuracy | Threshold (fitted on 9) |
|---|---|---|
| 1 | 97.2857% | 0.197492 |
| 2 | 96.7143% | 0.195070 |
| 3 | 98.0000% | 0.211409 |
| 4 | 96.4286% | 0.207388 |
| 5 | 96.1429% | 0.197492 |
| 6 | 98.8571% | 0.197492 |
| 7 | 99.0000% | 0.211409 |
| 8 | 97.7143% | 0.211409 |
| 9 | 96.2857% | 0.197492 |
| 10 | 98.2857% | 0.197492 |
| **mean** | **97.4714%** | std 1.0062pp |

#### cfp_fp / ensemble:equal 1/3

| Fold | Accuracy | Threshold (fitted on 9) |
|---|---|---|
| 1 | 97.0000% | 0.169083 |
| 2 | 96.5714% | 0.169083 |
| 3 | 97.7143% | 0.169083 |
| 4 | 96.2857% | 0.169083 |
| 5 | 95.7143% | 0.169083 |
| 6 | 98.7143% | 0.169083 |
| 7 | 98.8571% | 0.203925 |
| 8 | 97.2857% | 0.203925 |
| 9 | 96.0000% | 0.169083 |
| 10 | 98.0000% | 0.177571 |
| **mean** | **97.2143%** | std 1.0425pp |

#### cfp_fp / ensemble:dual r50+r100

| Fold | Accuracy | Threshold (fitted on 9) |
|---|---|---|
| 1 | 97.2857% | 0.205263 |
| 2 | 96.7143% | 0.194151 |
| 3 | 98.0000% | 0.205263 |
| 4 | 96.4286% | 0.205263 |
| 5 | 96.1429% | 0.205263 |
| 6 | 98.8571% | 0.205263 |
| 7 | 99.1429% | 0.205698 |
| 8 | 97.5714% | 0.213287 |
| 9 | 96.4286% | 0.213287 |
| 10 | 98.4286% | 0.205263 |
| **mean** | **97.5000%** | std 1.0227pp |

#### cfp_fp / ensemble:concat 1536-d

| Fold | Accuracy | Threshold (fitted on 9) |
|---|---|---|
| 1 | 97.4286% | 0.185747 |
| 2 | 96.5714% | 0.190423 |
| 3 | 98.1429% | 0.185747 |
| 4 | 96.7143% | 0.185747 |
| 5 | 96.1429% | 0.185747 |
| 6 | 99.1429% | 0.185747 |
| 7 | 99.0000% | 0.185747 |
| 8 | 98.0000% | 0.185747 |
| 9 | 96.5714% | 0.185747 |
| 10 | 97.8571% | 0.181041 |
| **mean** | **97.5571%** | std 0.9948pp |

#### calfw / single:w600k_r50 (R50)

| Fold | Accuracy | Threshold (fitted on 9) |
|---|---|---|
| 1 | 96.6667% | 0.217571 |
| 2 | 94.1667% | 0.217571 |
| 3 | 96.3333% | 0.174156 |
| 4 | 95.5000% | 0.217571 |
| 5 | 94.3333% | 0.192341 |
| 6 | 97.5000% | 0.174156 |
| 7 | 97.5000% | 0.174156 |
| 8 | 96.3333% | 0.174156 |
| 9 | 95.6667% | 0.174156 |
| 10 | 95.5000% | 0.217571 |
| **mean** | **95.9500%** | std 1.0905pp |

#### calfw / single:glintr100 (R100)

| Fold | Accuracy | Threshold (fitted on 9) |
|---|---|---|
| 1 | 97.0000% | 0.230358 |
| 2 | 94.6667% | 0.230993 |
| 3 | 96.5000% | 0.230358 |
| 4 | 95.8333% | 0.230358 |
| 5 | 94.5000% | 0.230028 |
| 6 | 97.6667% | 0.230358 |
| 7 | 97.6667% | 0.230358 |
| 8 | 96.5000% | 0.224299 |
| 9 | 95.6667% | 0.224299 |
| 10 | 95.6667% | 0.230358 |
| **mean** | **96.1667%** | std 1.0515pp |

#### calfw / single:w600k_mbf (MBF)

| Fold | Accuracy | Threshold (fitted on 9) |
|---|---|---|
| 1 | 95.8333% | 0.200151 |
| 2 | 94.0000% | 0.200151 |
| 3 | 95.5000% | 0.200205 |
| 4 | 95.1667% | 0.195907 |
| 5 | 94.1667% | 0.200151 |
| 6 | 97.0000% | 0.195907 |
| 7 | 97.5000% | 0.200151 |
| 8 | 96.0000% | 0.200151 |
| 9 | 96.1667% | 0.200151 |
| 10 | 94.6667% | 0.210430 |
| **mean** | **95.6000%** | std 1.0858pp |

#### calfw / ensemble:weighted 0.45/0.45/0.10

| Fold | Accuracy | Threshold (fitted on 9) |
|---|---|---|
| 1 | 96.8333% | 0.208880 |
| 2 | 94.5000% | 0.208880 |
| 3 | 96.5000% | 0.220637 |
| 4 | 95.1667% | 0.224481 |
| 5 | 94.6667% | 0.208880 |
| 6 | 97.8333% | 0.220637 |
| 7 | 97.6667% | 0.208880 |
| 8 | 96.6667% | 0.208880 |
| 9 | 95.5000% | 0.206176 |
| 10 | 95.3333% | 0.220637 |
| **mean** | **96.0667%** | std 1.1358pp |

#### calfw / ensemble:equal 1/3

| Fold | Accuracy | Threshold (fitted on 9) |
|---|---|---|
| 1 | 97.0000% | 0.213814 |
| 2 | 94.6667% | 0.213814 |
| 3 | 96.5000% | 0.227183 |
| 4 | 95.5000% | 0.213814 |
| 5 | 94.5000% | 0.213814 |
| 6 | 97.8333% | 0.213814 |
| 7 | 97.1667% | 0.216559 |
| 8 | 96.5000% | 0.212519 |
| 9 | 95.5000% | 0.213814 |
| 10 | 95.6667% | 0.213814 |
| **mean** | **96.0833%** | std 1.0388pp |

#### calfw / ensemble:dual r50+r100

| Fold | Accuracy | Threshold (fitted on 9) |
|---|---|---|
| 1 | 96.8333% | 0.214056 |
| 2 | 94.5000% | 0.213987 |
| 3 | 96.6667% | 0.214056 |
| 4 | 95.5000% | 0.215015 |
| 5 | 94.6667% | 0.214056 |
| 6 | 97.8333% | 0.214056 |
| 7 | 97.8333% | 0.214056 |
| 8 | 96.6667% | 0.214056 |
| 9 | 95.6667% | 0.206425 |
| 10 | 95.3333% | 0.214056 |
| **mean** | **96.1500%** | std 1.1364pp |

#### calfw / ensemble:concat 1536-d

| Fold | Accuracy | Threshold (fitted on 9) |
|---|---|---|
| 1 | 96.5000% | 0.216581 |
| 2 | 94.5000% | 0.202398 |
| 3 | 96.3333% | 0.201072 |
| 4 | 95.5000% | 0.187934 |
| 5 | 94.6667% | 0.201072 |
| 6 | 97.6667% | 0.201072 |
| 7 | 97.6667% | 0.201072 |
| 8 | 96.3333% | 0.201072 |
| 9 | 95.8333% | 0.200509 |
| 10 | 95.5000% | 0.216661 |
| **mean** | **96.0500%** | std 1.0275pp |

#### cplfw / single:w600k_r50 (R50)

| Fold | Accuracy | Threshold (fitted on 9) |
|---|---|---|
| 1 | 94.8333% | 0.173683 |
| 2 | 96.1667% | 0.173683 |
| 3 | 93.3333% | 0.173683 |
| 4 | 94.8333% | 0.173371 |
| 5 | 94.5000% | 0.173683 |
| 6 | 95.1667% | 0.173683 |
| 7 | 95.5000% | 0.173683 |
| 8 | 93.6667% | 0.176573 |
| 9 | 94.0000% | 0.173683 |
| 10 | 92.6667% | 0.173683 |
| **mean** | **94.4667%** | std 1.0050pp |

#### cplfw / single:glintr100 (R100)

| Fold | Accuracy | Threshold (fitted on 9) |
|---|---|---|
| 1 | 94.6667% | 0.191419 |
| 2 | 96.5000% | 0.191624 |
| 3 | 94.3333% | 0.191624 |
| 4 | 95.3333% | 0.191624 |
| 5 | 94.5000% | 0.191624 |
| 6 | 95.6667% | 0.194139 |
| 7 | 96.0000% | 0.191624 |
| 8 | 93.6667% | 0.191703 |
| 9 | 93.8333% | 0.191624 |
| 10 | 93.3333% | 0.193259 |
| **mean** | **94.7833%** | std 1.0029pp |

#### cplfw / single:w600k_mbf (MBF)

| Fold | Accuracy | Threshold (fitted on 9) |
|---|---|---|
| 1 | 92.6667% | 0.158628 |
| 2 | 94.6667% | 0.158844 |
| 3 | 91.6667% | 0.158844 |
| 4 | 93.5000% | 0.158844 |
| 5 | 92.0000% | 0.163430 |
| 6 | 93.3333% | 0.158844 |
| 7 | 93.5000% | 0.158844 |
| 8 | 92.1667% | 0.158874 |
| 9 | 91.3333% | 0.158844 |
| 10 | 91.5000% | 0.158844 |
| **mean** | **92.6333%** | std 1.0323pp |

#### cplfw / ensemble:weighted 0.45/0.45/0.10

| Fold | Accuracy | Threshold (fitted on 9) |
|---|---|---|
| 1 | 94.5000% | 0.187326 |
| 2 | 95.8333% | 0.175689 |
| 3 | 93.3333% | 0.176538 |
| 4 | 94.8333% | 0.187326 |
| 5 | 94.1667% | 0.187326 |
| 6 | 95.3333% | 0.187326 |
| 7 | 95.6667% | 0.175689 |
| 8 | 93.5000% | 0.179068 |
| 9 | 93.6667% | 0.175689 |
| 10 | 92.3333% | 0.175689 |
| **mean** | **94.3167%** | std 1.0710pp |

#### cplfw / ensemble:equal 1/3

| Fold | Accuracy | Threshold (fitted on 9) |
|---|---|---|
| 1 | 94.5000% | 0.191854 |
| 2 | 95.3333% | 0.160959 |
| 3 | 92.6667% | 0.191854 |
| 4 | 94.6667% | 0.191854 |
| 5 | 93.8333% | 0.193401 |
| 6 | 94.5000% | 0.191849 |
| 7 | 95.3333% | 0.188167 |
| 8 | 92.6667% | 0.188167 |
| 9 | 93.6667% | 0.191854 |
| 10 | 92.0000% | 0.191854 |
| **mean** | **93.9167%** | std 1.1011pp |

#### cplfw / ensemble:dual r50+r100

| Fold | Accuracy | Threshold (fitted on 9) |
|---|---|---|
| 1 | 94.5000% | 0.194425 |
| 2 | 96.0000% | 0.174682 |
| 3 | 93.5000% | 0.195295 |
| 4 | 95.0000% | 0.194425 |
| 5 | 94.5000% | 0.194425 |
| 6 | 95.5000% | 0.194425 |
| 7 | 95.8333% | 0.194425 |
| 8 | 93.3333% | 0.194425 |
| 9 | 94.0000% | 0.195295 |
| 10 | 92.5000% | 0.193160 |
| **mean** | **94.4667%** | std 1.0899pp |

#### cplfw / ensemble:concat 1536-d

| Fold | Accuracy | Threshold (fitted on 9) |
|---|---|---|
| 1 | 94.6667% | 0.171487 |
| 2 | 96.3333% | 0.178718 |
| 3 | 93.5000% | 0.180831 |
| 4 | 95.0000% | 0.178718 |
| 5 | 94.1667% | 0.178718 |
| 6 | 95.3333% | 0.178718 |
| 7 | 95.3333% | 0.169764 |
| 8 | 92.8333% | 0.171487 |
| 9 | 93.8333% | 0.180183 |
| 10 | 92.3333% | 0.178718 |
| **mean** | **94.3333%** | std 1.1738pp |

### Raw artefact

```json
{
  "protocol": "10-fold CV, threshold fitted on 9 folds, applied to held-out fold",
  "flip_tta": true,
  "results": [
    {
      "dataset": "lfw",
      "config": "single:w600k_r50 (R50)",
      "n_pairs": 6000,
      "n_genuine": 3000,
      "n_impostor": 3000,
      "accuracy_mean": 0.9978333333333333,
      "accuracy_std": 0.00258736244937666,
      "threshold_mean": 0.2427101734588209,
      "threshold_std": 0.013478517179649862,
      "oracle_accuracy": 0.9983333333333333,
      "oracle_threshold": 0.2472690552642305,
      "tar_at_far_1e2": 0.9976666666666667,
      "tar_at_far_1e3": 0.997,
      "tar_at_far_1e4": 0.9966666666666667,
      "auc": 0.9994280000000001,
      "eer": 0.0023333333333333157,
      "folds": [
        {
          "accuracy": 0.9916666666666667,
          "threshold": 0.2618694062837365
        },
        {
          "accuracy": 1.0,
          "threshold": 0.2472690552642305
        },
        {
          "accuracy": 1.0,
          "threshold": 0.2472690552642305
        },
        {
          "accuracy": 0.9983333333333333,
          "threshold": 0.2472690552642305
        },
        {
          "accuracy": 0.995,
          "threshold": 0.2472690552642305
        },
        {
          "accuracy": 0.9983333333333333,
          "threshold": 0.2171744707274295
        },
        {
          "accuracy": 0.9966666666666667,
          "threshold": 0.2472690552642305
        },
        {
          "accuracy": 1.0,
          "threshold": 0.2472690552642305
        },
        {
          "accuracy": 1.0,
          "threshold": 0.2472690552642305
        },
        {
          "accuracy": 0.9983333333333333,
          "threshold": 0.2171744707274295
        }
      ]
    },
    {
      "dataset": "lfw",
      "config": "single:glintr100 (R100)",
      "n_pairs": 6000,
      "n_genuine": 3000,
      "n_impostor": 3000,
      "accuracy_mean": 0.9976666666666667,
      "accuracy_std": 0.0026034165586355405,
      "threshold_mean": 0.28383816777562937,
      "threshold_std": 0.024522160005770816,
      "oracle_accuracy": 0.9981666666666666,
      "oracle_threshold": 0.2870779074819152,
      "tar_at_far_1e2": 0.9976666666666667,
      "tar_at_far_1e3": 0.9963333333333333,
      "tar_at_far_1e4": 0.9963333333333333,
      "auc": 0.9995202222222223,
      "eer": 0.002666666666666686,
      "folds": [
        {
          "accuracy": 0.9916666666666667,
          "threshold": 0.2870779074819152
        },
        {
          "accuracy": 0.9983333333333333,
          "threshold": 0.3264501876313154
        },
        {
          "accuracy": 1.0,
          "threshold": 0.2870779074819152
        },
        {
          "accuracy": 0.9983333333333333,
          "threshold": 0.2870779074819152
        },
        {
          "accuracy": 0.995,
          "threshold": 0.2870779074819152
        },
        {
          "accuracy": 0.9966666666666667,
          "threshold": 0.21962803506151318
        },
        {
          "accuracy": 0.9966666666666667,
          "threshold": 0.2870779074819152
        },
        {
          "accuracy": 1.0,
          "threshold": 0.2870779074819152
        },
        {
          "accuracy": 1.0,
          "threshold": 0.2827581026900585
        },
        {
          "accuracy": 1.0,
          "threshold": 0.2870779074819152
        }
      ]
    },
    {
      "dataset": "lfw",
      "config": "single:w600k_mbf (MBF)",
      "n_pairs": 6000,
      "n_genuine": 3000,
      "n_impostor": 3000,
      "accuracy_mean": 0.9959999999999999,
      "accuracy_std": 0.002494438257849294,
      "threshold_mean": 0.22025354998655336,
      "threshold_std": 0.008481786508760284,
      "oracle_accuracy": 0.9965,
      "oracle_threshold": 0.21503955260294472,
      "tar_at_far_1e2": 0.9963333333333333,
      "tar_at_far_1e3": 0.9933333333333333,
      "tar_at_far_1e4": 0.988,
      "auc": 0.999380111111111,
      "eer": 0.004333333333333317,
      "folds": [
        {
          "accuracy": 0.9916666666666667,
          "threshold": 0.22508471195550284
        },
        {
          "accuracy": 1.0,
          "threshold": 0.21503955260294472
        },
        {
          "accuracy": 0.9966666666666667,
          "threshold": 0.23460831496762763
        },
        {
          "accuracy": 0.995,
          "threshold": 0.23756560472179017
        },
        {
          "accuracy": 0.9933333333333333,
          "threshold": 0.21503955260294472
        },
        {
          "accuracy": 0.9966666666666667,
          "threshold": 0.21503955260294472
        },
        {
          "accuracy": 0.9933333333333333,
          "threshold": 0.21503955260294472
        },
        {
          "accuracy": 0.9983333333333333,
          "threshold": 0.21503955260294472
        },
        {
          "accuracy": 0.9983333333333333,
          "threshold": 0.21503955260294472
        },
        {
          "accuracy": 0.9966666666666667,
          "threshold": 0.21503955260294472
        }
      ]
    },
    {
      "dataset": "lfw",
      "config": "ensemble:weighted 0.45/0.45/0.10",
      "n_pairs": 6000,
      "n_genuine": 3000,
      "n_impostor": 3000,
      "accuracy_mean": 0.998,
      "accuracy_std": 0.0026666666666666575,
      "threshold_mean": 0.27695360232519006,
      "threshold_std": 0.010410237595208096,
      "oracle_accuracy": 0.9981666666666666,
      "oracle_threshold": 0.27379885325849074,
      "tar_at_far_1e2": 0.9976666666666667,
      "tar_at_far_1e3": 0.9963333333333333,
      "tar_at_far_1e4": 0.9963333333333333,
      "auc": 0.9993286666666666,
      "eer": 0.0030000000000000014,
      "folds": [
        {
          "accuracy": 0.9916666666666667,
          "threshold": 0.27379885325849074
        },
        {
          "accuracy": 0.9983333333333333,
          "threshold": 0.3080879003844673
        },
        {
          "accuracy": 1.0,
          "threshold": 0.27379885325849074
        },
        {
          "accuracy": 0.9983333333333333,
          "threshold": 0.27379885325849074
        },
        {
          "accuracy": 0.995,
          "threshold": 0.27379885325849074
        },
        {
          "accuracy": 1.0,
          "threshold": 0.2710572967995068
        },
        {
          "accuracy": 0.9966666666666667,
          "threshold": 0.27379885325849074
        },
        {
          "accuracy": 1.0,
          "threshold": 0.27379885325849074
        },
        {
          "accuracy": 1.0,
          "threshold": 0.27379885325849074
        },
        {
          "accuracy": 1.0,
          "threshold": 0.27379885325849074
        }
      ]
    },
    {
      "dataset": "lfw",
      "config": "ensemble:equal 1/3",
      "n_pairs": 6000,
      "n_genuine": 3000,
      "n_impostor": 3000,
      "accuracy_mean": 0.9976666666666667,
      "accuracy_std": 0.0024944382578492826,
      "threshold_mean": 0.27117892839444907,
      "threshold_std": 0.009022048046893081,
      "oracle_accuracy": 0.998,
      "oracle_threshold": 0.27533774996316196,
      "tar_at_far_1e2": 0.9973333333333333,
      "tar_at_far_1e3": 0.9963333333333333,
      "tar_at_far_1e4": 0.995,
      "auc": 0.9993602222222222,
      "eer": 0.0033333333333333166,
      "folds": [
        {
          "accuracy": 0.9916666666666667,
          "threshold": 0.27533774996316196
        },
        {
          "accuracy": 1.0,
          "threshold": 0.27533774996316196
        },
        {
          "accuracy": 1.0,
          "threshold": 0.2780159689841505
        },
        {
          "accuracy": 0.9983333333333333,
          "threshold": 0.27533774996316196
        },
        {
          "accuracy": 0.995,
          "threshold": 0.27533774996316196
        },
        {
          "accuracy": 0.9983333333333333,
          "threshold": 0.27533774996316196
        },
        {
          "accuracy": 0.9966666666666667,
          "threshold": 0.27533774996316196
        },
        {
          "accuracy": 1.0,
          "threshold": 0.27533774996316196
        },
        {
          "accuracy": 0.9983333333333333,
          "threshold": 0.25320453260910336
        },
        {
          "accuracy": 0.9983333333333333,
          "threshold": 0.25320453260910336
        }
      ]
    },
    {
      "dataset": "lfw",
      "config": "ensemble:dual r50+r100",
      "n_pairs": 6000,
      "n_genuine": 3000,
      "n_impostor": 3000,
      "accuracy_mean": 0.998,
      "accuracy_std": 0.0026666666666666575,
      "threshold_mean": 0.2636944947590395,
      "threshold_std": 0.015411061687425653,
      "oracle_accuracy": 0.9981666666666666,
      "oracle_threshold": 0.25872467652406367,
      "tar_at_far_1e2": 0.9976666666666667,
      "tar_at_far_1e3": 0.9963333333333333,
      "tar_at_far_1e4": 0.9963333333333333,
      "auc": 0.9993217777777779,
      "eer": 0.0030000000000000014,
      "folds": [
        {
          "accuracy": 0.9916666666666667,
          "threshold": 0.25872467652406367
        },
        {
          "accuracy": 0.9983333333333333,
          "threshold": 0.3099085782335608
        },
        {
          "accuracy": 1.0,
          "threshold": 0.25872467652406367
        },
        {
          "accuracy": 0.9983333333333333,
          "threshold": 0.25872467652406367
        },
        {
          "accuracy": 0.995,
          "threshold": 0.25872467652406367
        },
        {
          "accuracy": 1.0,
          "threshold": 0.25872467652406367
        },
        {
          "accuracy": 0.9966666666666667,
          "threshold": 0.25872467652406367
        },
        {
          "accuracy": 1.0,
          "threshold": 0.25872467652406367
        },
        {
          "accuracy": 1.0,
          "threshold": 0.25872467652406367
        },
        {
          "accuracy": 1.0,
          "threshold": 0.2572389571643247
        }
      ]
    },
    {
      "dataset": "lfw",
      "config": "ensemble:concat 1536-d",
      "n_pairs": 6000,
      "n_genuine": 3000,
      "n_impostor": 3000,
      "accuracy_mean": 0.9976666666666667,
      "accuracy_std": 0.002708012801545312,
      "threshold_mean": 0.24194595077885706,
      "threshold_std": 0.012416129498756577,
      "oracle_accuracy": 0.998,
      "oracle_threshold": 0.23632194501536144,
      "tar_at_far_1e2": 0.998,
      "tar_at_far_1e3": 0.9966666666666667,
      "tar_at_far_1e4": 0.996,
      "auc": 0.9995114444444445,
      "eer": 0.002666666666666686,
      "folds": [
        {
          "accuracy": 0.9916666666666667,
          "threshold": 0.25407403376964766
        },
        {
          "accuracy": 1.0,
          "threshold": 0.23632194501536144
        },
        {
          "accuracy": 1.0,
          "threshold": 0.23632194501536144
        },
        {
          "accuracy": 0.9983333333333333,
          "threshold": 0.23632194501536144
        },
        {
          "accuracy": 0.995,
          "threshold": 0.23632194501536144
        },
        {
          "accuracy": 0.9983333333333333,
          "threshold": 0.23632194501536144
        },
        {
          "accuracy": 0.995,
          "threshold": 0.2755921447266543
        },
        {
          "accuracy": 1.0,
          "threshold": 0.2355397141847389
        },
        {
          "accuracy": 0.9983333333333333,
          "threshold": 0.23632194501536144
        },
        {
          "accuracy": 1.0,
          "threshold": 0.23632194501536144
        }
      ]
    },
    {
      "dataset": "agedb_30",
      "config": "single:w600k_r50 (R50)",
      "n_pairs": 6000,
      "n_genuine": 3000,
      "n_impostor": 3000,
      "accuracy_mean": 0.9815000000000002,
      "accuracy_std": 0.006075908711186037,
      "threshold_mean": 0.20263070784456244,
      "threshold_std": 0.000420427984186282,
      "oracle_accuracy": 0.9818333333333333,
      "oracle_threshold": 0.20252018833810592,
      "tar_at_far_1e2": 0.9693333333333334,
      "tar_at_far_1e3": 0.9603333333333334,
      "tar_at_far_1e4": 0.9366666666666666,
      "auc": 0.9913036666666666,
      "eer": 0.023333333333333324,
      "folds": [
        {
          "accuracy": 0.9883333333333333,
          "threshold": 0.20252018833810592
        },
        {
          "accuracy": 0.9883333333333333,
          "threshold": 0.20252018833810592
        },
        {
          "accuracy": 0.9916666666666667,
          "threshold": 0.2038724778256356
        },
        {
          "accuracy": 0.98,
          "threshold": 0.20252018833810592
        },
        {
          "accuracy": 0.9783333333333334,
          "threshold": 0.20252018833810592
        },
        {
          "accuracy": 0.9833333333333333,
          "threshold": 0.20252018833810592
        },
        {
          "accuracy": 0.9766666666666667,
          "threshold": 0.20227309391514162
        },
        {
          "accuracy": 0.9733333333333334,
          "threshold": 0.20252018833810592
        },
        {
          "accuracy": 0.9733333333333334,
          "threshold": 0.20252018833810592
        },
        {
          "accuracy": 0.9816666666666667,
          "threshold": 0.20252018833810592
        }
      ]
    },
    {
      "dataset": "agedb_30",
      "config": "single:glintr100 (R100)",
      "n_pairs": 6000,
      "n_genuine": 3000,
      "n_impostor": 3000,
      "accuracy_mean": 0.9831666666666667,
      "accuracy_std": 0.006030017505041838,
      "threshold_mean": 0.2215759051254059,
      "threshold_std": 0.013046978080642825,
      "oracle_accuracy": 0.9845,
      "oracle_threshold": 0.23184468881780246,
      "tar_at_far_1e2": 0.976,
      "tar_at_far_1e3": 0.9633333333333334,
      "tar_at_far_1e4": 0.956,
      "auc": 0.991735888888889,
      "eer": 0.02100000000000001,
      "folds": [
        {
          "accuracy": 0.99,
          "threshold": 0.2071980825611516
        },
        {
          "accuracy": 0.99,
          "threshold": 0.2071980825611516
        },
        {
          "accuracy": 0.99,
          "threshold": 0.23184468881780246
        },
        {
          "accuracy": 0.9866666666666667,
          "threshold": 0.2336126387572408
        },
        {
          "accuracy": 0.9783333333333334,
          "threshold": 0.23184468881780246
        },
        {
          "accuracy": 0.98,
          "threshold": 0.20132872072435126
        },
        {
          "accuracy": 0.9766666666666667,
          "threshold": 0.23184468881780246
        },
        {
          "accuracy": 0.9766666666666667,
          "threshold": 0.2071980825611516
        },
        {
          "accuracy": 0.975,
          "threshold": 0.23184468881780246
        },
        {
          "accuracy": 0.9883333333333333,
          "threshold": 0.23184468881780246
        }
      ]
    },
    {
      "dataset": "agedb_30",
      "config": "single:w600k_mbf (MBF)",
      "n_pairs": 6000,
      "n_genuine": 3000,
      "n_impostor": 3000,
      "accuracy_mean": 0.9633333333333333,
      "accuracy_std": 0.006666666666666668,
      "threshold_mean": 0.1839140487309751,
      "threshold_std": 0.009599946647579504,
      "oracle_accuracy": 0.966,
      "oracle_threshold": 0.17416735718477966,
      "tar_at_far_1e2": 0.938,
      "tar_at_far_1e3": 0.8606666666666667,
      "tar_at_far_1e4": 0.6973333333333334,
      "auc": 0.9893865555555555,
      "eer": 0.03766666666666665,
      "folds": [
        {
          "accuracy": 0.96,
          "threshold": 0.19235155779233828
        },
        {
          "accuracy": 0.9683333333333334,
          "threshold": 0.19505934542576037
        },
        {
          "accuracy": 0.9733333333333334,
          "threshold": 0.17490927988766267
        },
        {
          "accuracy": 0.9633333333333334,
          "threshold": 0.19505934542576037
        },
        {
          "accuracy": 0.9483333333333334,
          "threshold": 0.17609333151655934
        },
        {
          "accuracy": 0.9666666666666667,
          "threshold": 0.17262979730743422
        },
        {
          "accuracy": 0.9616666666666667,
          "threshold": 0.19235155779233828
        },
        {
          "accuracy": 0.9583333333333334,
          "threshold": 0.19235155779233828
        },
        {
          "accuracy": 0.97,
          "threshold": 0.17416735718477966
        },
        {
          "accuracy": 0.9633333333333334,
          "threshold": 0.17416735718477966
        }
      ]
    },
    {
      "dataset": "agedb_30",
      "config": "ensemble:weighted 0.45/0.45/0.10",
      "n_pairs": 6000,
      "n_genuine": 3000,
      "n_impostor": 3000,
      "accuracy_mean": 0.9831666666666667,
      "accuracy_std": 0.005550275268448894,
      "threshold_mean": 0.21914133360099597,
      "threshold_std": 0.004010675455835541,
      "oracle_accuracy": 0.9838333333333333,
      "oracle_threshold": 0.21768432383559944,
      "tar_at_far_1e2": 0.975,
      "tar_at_far_1e3": 0.9636666666666667,
      "tar_at_far_1e4": 0.945,
      "auc": 0.9910245555555555,
      "eer": 0.023333333333333324,
      "folds": [
        {
          "accuracy": 0.9883333333333333,
          "threshold": 0.21768432383559944
        },
        {
          "accuracy": 0.9916666666666667,
          "threshold": 0.21768432383559944
        },
        {
          "accuracy": 0.99,
          "threshold": 0.21768432383559944
        },
        {
          "accuracy": 0.985,
          "threshold": 0.2311250054856205
        },
        {
          "accuracy": 0.9783333333333334,
          "threshold": 0.2188792053108426
        },
        {
          "accuracy": 0.985,
          "threshold": 0.21768432383559944
        },
        {
          "accuracy": 0.9783333333333334,
          "threshold": 0.21768432383559944
        },
        {
          "accuracy": 0.975,
          "threshold": 0.21768432383559944
        },
        {
          "accuracy": 0.9766666666666667,
          "threshold": 0.21768432383559944
        },
        {
          "accuracy": 0.9833333333333333,
          "threshold": 0.21761885836430053
        }
      ]
    },
    {
      "dataset": "agedb_30",
      "config": "ensemble:equal 1/3",
      "n_pairs": 6000,
      "n_genuine": 3000,
      "n_impostor": 3000,
      "accuracy_mean": 0.9806666666666668,
      "accuracy_std": 0.006064468466220068,
      "threshold_mean": 0.20782724223790705,
      "threshold_std": 0.0016082877024317951,
      "oracle_accuracy": 0.9813333333333333,
      "oracle_threshold": 0.2071237784515499,
      "tar_at_far_1e2": 0.97,
      "tar_at_far_1e3": 0.9546666666666667,
      "tar_at_far_1e4": 0.939,
      "auc": 0.9908325555555555,
      "eer": 0.024666666666666642,
      "folds": [
        {
          "accuracy": 0.985,
          "threshold": 0.2071237784515499
        },
        {
          "accuracy": 0.99,
          "threshold": 0.2071237784515499
        },
        {
          "accuracy": 0.985,
          "threshold": 0.20653037833269144
        },
        {
          "accuracy": 0.985,
          "threshold": 0.2071237784515499
        },
        {
          "accuracy": 0.9766666666666667,
          "threshold": 0.2071237784515499
        },
        {
          "accuracy": 0.9833333333333333,
          "threshold": 0.2071237784515499
        },
        {
          "accuracy": 0.9766666666666667,
          "threshold": 0.21176511264670253
        },
        {
          "accuracy": 0.9683333333333334,
          "threshold": 0.21011048223882758
        },
        {
          "accuracy": 0.975,
          "threshold": 0.2071237784515499
        },
        {
          "accuracy": 0.9816666666666667,
          "threshold": 0.2071237784515499
        }
      ]
    },
    {
      "dataset": "agedb_30",
      "config": "ensemble:dual r50+r100",
      "n_pairs": 6000,
      "n_genuine": 3000,
      "n_impostor": 3000,
      "accuracy_mean": 0.983,
      "accuracy_std": 0.007063206700139029,
      "threshold_mean": 0.20899117557499935,
      "threshold_std": 0.004462196162990119,
      "oracle_accuracy": 0.9838333333333333,
      "oracle_threshold": 0.2078153292582518,
      "tar_at_far_1e2": 0.975,
      "tar_at_far_1e3": 0.9653333333333334,
      "tar_at_far_1e4": 0.9496666666666667,
      "auc": 0.9910786666666667,
      "eer": 0.02366666666666664,
      "folds": [
        {
          "accuracy": 0.99,
          "threshold": 0.2078153292582518
        },
        {
          "accuracy": 0.9916666666666667,
          "threshold": 0.2078153292582518
        },
        {
          "accuracy": 0.99,
          "threshold": 0.2078153292582518
        },
        {
          "accuracy": 0.99,
          "threshold": 0.2078153292582518
        },
        {
          "accuracy": 0.9766666666666667,
          "threshold": 0.2078153292582518
        },
        {
          "accuracy": 0.985,
          "threshold": 0.2064614414084965
        },
        {
          "accuracy": 0.975,
          "threshold": 0.22228156812523775
        },
        {
          "accuracy": 0.9716666666666667,
          "threshold": 0.2078153292582518
        },
        {
          "accuracy": 0.9766666666666667,
          "threshold": 0.2078153292582518
        },
        {
          "accuracy": 0.9833333333333333,
          "threshold": 0.2064614414084965
        }
      ]
    },
    {
      "dataset": "agedb_30",
      "config": "ensemble:concat 1536-d",
      "n_pairs": 6000,
      "n_genuine": 3000,
      "n_impostor": 3000,
      "accuracy_mean": 0.9801666666666667,
      "accuracy_std": 0.006030017505041822,
      "threshold_mean": 0.2008951315088024,
      "threshold_std": 0.011116283993275309,
      "oracle_accuracy": 0.9821666666666666,
      "oracle_threshold": 0.19218477925573405,
      "tar_at_far_1e2": 0.973,
      "tar_at_far_1e3": 0.9596666666666667,
      "tar_at_far_1e4": 0.9496666666666667,
      "auc": 0.9918285555555555,
      "eer": 0.022333333333333323,
      "folds": [
        {
          "accuracy": 0.9883333333333333,
          "threshold": 0.19218477925573405
        },
        {
          "accuracy": 0.9883333333333333,
          "threshold": 0.19218477925573405
        },
        {
          "accuracy": 0.9866666666666667,
          "threshold": 0.19218477925573405
        },
        {
          "accuracy": 0.98,
          "threshold": 0.20494624982531634
        },
        {
          "accuracy": 0.9733333333333334,
          "threshold": 0.21540018330949956
        },
        {
          "accuracy": 0.9833333333333333,
          "threshold": 0.21540018330949956
        },
        {
          "accuracy": 0.9733333333333334,
          "threshold": 0.20494624982531634
        },
        {
          "accuracy": 0.9716666666666667,
          "threshold": 0.18504409935286747
        },
        {
          "accuracy": 0.9766666666666667,
          "threshold": 0.21540018330949956
        },
        {
          "accuracy": 0.98,
          "threshold": 0.19125982838882255
        }
      ]
    },
    {
      "dataset": "cfp_fp",
      "config": "single:w600k_r50 (R50)",
      "n_pairs": 7000,
      "n_genuine": 3500,
      "n_impostor": 3500,
      "accuracy_mean": 0.9744285714285714,
      "accuracy_std": 0.010699036767630166,
      "threshold_mean": 0.1838923222365063,
      "threshold_std": 0.0018080484429009128,
      "oracle_accuracy": 0.975,
      "oracle_threshold": 0.18312331895619238,
      "tar_at_far_1e2": 0.9548571428571428,
      "tar_at_far_1e3": 0.9468571428571428,
      "tar_at_far_1e4": 0.9454285714285714,
      "auc": 0.9802297959183673,
      "eer": 0.04285714285714284,
      "folds": [
        {
          "accuracy": 0.9728571428571429,
          "threshold": 0.18312331895619238
        },
        {
          "accuracy": 0.9685714285714285,
          "threshold": 0.18312331895619238
        },
        {
          "accuracy": 0.98,
          "threshold": 0.18273723550539478
        },
        {
          "accuracy": 0.9614285714285714,
          "threshold": 0.18887064079388072
        },
        {
          "accuracy": 0.9614285714285714,
          "threshold": 0.18312331895619238
        },
        {
          "accuracy": 0.99,
          "threshold": 0.1854521133724411
        },
        {
          "accuracy": 0.9914285714285714,
          "threshold": 0.18312331895619238
        },
        {
          "accuracy": 0.9771428571428571,
          "threshold": 0.18312331895619238
        },
        {
          "accuracy": 0.9614285714285714,
          "threshold": 0.18312331895619238
        },
        {
          "accuracy": 0.98,
          "threshold": 0.18312331895619238
        }
      ]
    },
    {
      "dataset": "cfp_fp",
      "config": "single:glintr100 (R100)",
      "n_pairs": 7000,
      "n_genuine": 3500,
      "n_impostor": 3500,
      "accuracy_mean": 0.9771428571428571,
      "accuracy_std": 0.009302183197771261,
      "threshold_mean": 0.22948905366922973,
      "threshold_std": 0.012800556995803937,
      "oracle_accuracy": 0.9775714285714285,
      "oracle_threshold": 0.2350701160605514,
      "tar_at_far_1e2": 0.9588571428571429,
      "tar_at_far_1e3": 0.9557142857142857,
      "tar_at_far_1e4": 0.954,
      "auc": 0.9851918367346939,
      "eer": 0.03800000000000002,
      "folds": [
        {
          "accuracy": 0.9757142857142858,
          "threshold": 0.2398790710098026
        },
        {
          "accuracy": 0.97,
          "threshold": 0.2350701160605514
        },
        {
          "accuracy": 0.9814285714285714,
          "threshold": 0.2350701160605514
        },
        {
          "accuracy": 0.9728571428571429,
          "threshold": 0.2350701160605514
        },
        {
          "accuracy": 0.9614285714285714,
          "threshold": 0.2350701160605514
        },
        {
          "accuracy": 0.9871428571428571,
          "threshold": 0.2350701160605514
        },
        {
          "accuracy": 0.9928571428571429,
          "threshold": 0.2350701160605514
        },
        {
          "accuracy": 0.9814285714285714,
          "threshold": 0.2350701160605514
        },
        {
          "accuracy": 0.9657142857142857,
          "threshold": 0.1980695699549535
        },
        {
          "accuracy": 0.9828571428571429,
          "threshold": 0.21145108330368134
        }
      ]
    },
    {
      "dataset": "cfp_fp",
      "config": "single:w600k_mbf (MBF)",
      "n_pairs": 7000,
      "n_genuine": 3500,
      "n_impostor": 3500,
      "accuracy_mean": 0.96,
      "accuracy_std": 0.011047208530960672,
      "threshold_mean": 0.161444337921973,
      "threshold_std": 0.0009327685275817582,
      "oracle_accuracy": 0.9602857142857143,
      "oracle_threshold": 0.16100367096927393,
      "tar_at_far_1e2": 0.9288571428571428,
      "tar_at_far_1e3": 0.8862857142857142,
      "tar_at_far_1e4": 0.8751428571428571,
      "auc": 0.9742179591836735,
      "eer": 0.05485714285714284,
      "folds": [
        {
          "accuracy": 0.9485714285714286,
          "threshold": 0.16100367096927393
        },
        {
          "accuracy": 0.9514285714285714,
          "threshold": 0.1633061807075778
        },
        {
          "accuracy": 0.9642857142857143,
          "threshold": 0.16100367096927393
        },
        {
          "accuracy": 0.9457142857142857,
          "threshold": 0.16100367096927393
        },
        {
          "accuracy": 0.9528571428571428,
          "threshold": 0.16100367096927393
        },
        {
          "accuracy": 0.98,
          "threshold": 0.160805321019657
        },
        {
          "accuracy": 0.9771428571428571,
          "threshold": 0.16100367096927393
        },
        {
          "accuracy": 0.9657142857142857,
          "threshold": 0.16100367096927393
        },
        {
          "accuracy": 0.9571428571428572,
          "threshold": 0.1633061807075778
        },
        {
          "accuracy": 0.9571428571428572,
          "threshold": 0.16100367096927393
        }
      ]
    },
    {
      "dataset": "cfp_fp",
      "config": "ensemble:weighted 0.45/0.45/0.10",
      "n_pairs": 7000,
      "n_genuine": 3500,
      "n_impostor": 3500,
      "accuracy_mean": 0.9747142857142856,
      "accuracy_std": 0.010062052373108754,
      "threshold_mean": 0.20241446120188403,
      "threshold_std": 0.00665202161745545,
      "oracle_accuracy": 0.9757142857142858,
      "oracle_threshold": 0.19749221425390243,
      "tar_at_far_1e2": 0.9565714285714285,
      "tar_at_far_1e3": 0.9494285714285714,
      "tar_at_far_1e4": 0.946,
      "auc": 0.9818825306122448,
      "eer": 0.04000000000000002,
      "folds": [
        {
          "accuracy": 0.9728571428571429,
          "threshold": 0.19749221425390243
        },
        {
          "accuracy": 0.9671428571428572,
          "threshold": 0.19507026511640027
        },
        {
          "accuracy": 0.98,
          "threshold": 0.21140858566201937
        },
        {
          "accuracy": 0.9642857142857143,
          "threshold": 0.20738751864686955
        },
        {
          "accuracy": 0.9614285714285714,
          "threshold": 0.19749221425390243
        },
        {
          "accuracy": 0.9885714285714285,
          "threshold": 0.19749221425390243
        },
        {
          "accuracy": 0.99,
          "threshold": 0.21140858566201937
        },
        {
          "accuracy": 0.9771428571428571,
          "threshold": 0.21140858566201937
        },
        {
          "accuracy": 0.9628571428571429,
          "threshold": 0.19749221425390243
        },
        {
          "accuracy": 0.9828571428571429,
          "threshold": 0.19749221425390243
        }
      ]
    },
    {
      "dataset": "cfp_fp",
      "config": "ensemble:equal 1/3",
      "n_pairs": 7000,
      "n_genuine": 3500,
      "n_impostor": 3500,
      "accuracy_mean": 0.9721428571428572,
      "accuracy_std": 0.010424656799518857,
      "threshold_mean": 0.17690012036926656,
      "threshold_std": 0.013743538061667284,
      "oracle_accuracy": 0.973,
      "oracle_threshold": 0.16908303284960458,
      "tar_at_far_1e2": 0.9545714285714286,
      "tar_at_far_1e3": 0.9431428571428572,
      "tar_at_far_1e4": 0.9331428571428572,
      "auc": 0.9792297959183673,
      "eer": 0.04114285714285715,
      "folds": [
        {
          "accuracy": 0.97,
          "threshold": 0.16908303284960458
        },
        {
          "accuracy": 0.9657142857142857,
          "threshold": 0.16908303284960458
        },
        {
          "accuracy": 0.9771428571428571,
          "threshold": 0.16908303284960458
        },
        {
          "accuracy": 0.9628571428571429,
          "threshold": 0.16908303284960458
        },
        {
          "accuracy": 0.9571428571428572,
          "threshold": 0.16908303284960458
        },
        {
          "accuracy": 0.9871428571428571,
          "threshold": 0.16908303284960458
        },
        {
          "accuracy": 0.9885714285714285,
          "threshold": 0.2039246532076702
        },
        {
          "accuracy": 0.9728571428571429,
          "threshold": 0.2039246532076702
        },
        {
          "accuracy": 0.96,
          "threshold": 0.16908303284960458
        },
        {
          "accuracy": 0.98,
          "threshold": 0.17757066733009325
        }
      ]
    },
    {
      "dataset": "cfp_fp",
      "config": "ensemble:dual r50+r100",
      "n_pairs": 7000,
      "n_genuine": 3500,
      "n_impostor": 3500,
      "accuracy_mean": 0.975,
      "accuracy_std": 0.01022701504519739,
      "threshold_mean": 0.20579989873101007,
      "threshold_std": 0.0049953981954446195,
      "oracle_accuracy": 0.9757142857142858,
      "oracle_threshold": 0.20526283021437075,
      "tar_at_far_1e2": 0.9568571428571429,
      "tar_at_far_1e3": 0.9494285714285714,
      "tar_at_far_1e4": 0.9471428571428572,
      "auc": 0.982423918367347,
      "eer": 0.04028571428571427,
      "folds": [
        {
          "accuracy": 0.9728571428571429,
          "threshold": 0.20526283021437075
        },
        {
          "accuracy": 0.9671428571428572,
          "threshold": 0.19415105154502588
        },
        {
          "accuracy": 0.98,
          "threshold": 0.20526283021437075
        },
        {
          "accuracy": 0.9642857142857143,
          "threshold": 0.20526283021437075
        },
        {
          "accuracy": 0.9614285714285714,
          "threshold": 0.20526283021437075
        },
        {
          "accuracy": 0.9885714285714285,
          "threshold": 0.20526283021437075
        },
        {
          "accuracy": 0.9914285714285714,
          "threshold": 0.2056976286530894
        },
        {
          "accuracy": 0.9757142857142858,
          "threshold": 0.2132866629128804
        },
        {
          "accuracy": 0.9642857142857143,
          "threshold": 0.2132866629128804
        },
        {
          "accuracy": 0.9842857142857143,
          "threshold": 0.20526283021437075
        }
      ]
    },
    {
      "dataset": "cfp_fp",
      "config": "ensemble:concat 1536-d",
      "n_pairs": 7000,
      "n_genuine": 3500,
      "n_impostor": 3500,
      "accuracy_mean": 0.9755714285714285,
      "accuracy_std": 0.009947823062030665,
      "threshold_mean": 0.1857437909526212,
      "threshold_std": 0.0020977652279591166,
      "oracle_accuracy": 0.976,
      "oracle_threshold": 0.18574678143167428,
      "tar_at_far_1e2": 0.9565714285714285,
      "tar_at_far_1e3": 0.9514285714285714,
      "tar_at_far_1e4": 0.9497142857142857,
      "auc": 0.9810523265306121,
      "eer": 0.040857142857142835,
      "folds": [
        {
          "accuracy": 0.9742857142857143,
          "threshold": 0.18574678143167428
        },
        {
          "accuracy": 0.9657142857142857,
          "threshold": 0.19042255562179755
        },
        {
          "accuracy": 0.9814285714285714,
          "threshold": 0.18574678143167428
        },
        {
          "accuracy": 0.9671428571428572,
          "threshold": 0.18574678143167428
        },
        {
          "accuracy": 0.9614285714285714,
          "threshold": 0.18574678143167428
        },
        {
          "accuracy": 0.9914285714285714,
          "threshold": 0.18574678143167428
        },
        {
          "accuracy": 0.99,
          "threshold": 0.18574678143167428
        },
        {
          "accuracy": 0.98,
          "threshold": 0.18574678143167428
        },
        {
          "accuracy": 0.9657142857142857,
          "threshold": 0.18574678143167428
        },
        {
          "accuracy": 0.9785714285714285,
          "threshold": 0.18104110245102012
        }
      ]
    },
    {
      "dataset": "calfw",
      "config": "single:w600k_r50 (R50)",
      "n_pairs": 6000,
      "n_genuine": 3000,
      "n_impostor": 3000,
      "accuracy_mean": 0.9595,
      "accuracy_std": 0.01090489186863706,
      "threshold_mean": 0.19334064583385016,
      "threshold_std": 0.020468809643597416,
      "oracle_accuracy": 0.9611666666666666,
      "oracle_threshold": 0.2175712791531558,
      "tar_at_far_1e2": 0.9313333333333333,
      "tar_at_far_1e3": 0.921,
      "tar_at_far_1e4": 0.8946666666666667,
      "auc": 0.9775483333333335,
      "eer": 0.05866666666666666,
      "folds": [
        {
          "accuracy": 0.9666666666666667,
          "threshold": 0.2175712791531558
        },
        {
          "accuracy": 0.9416666666666667,
          "threshold": 0.2175712791531558
        },
        {
          "accuracy": 0.9633333333333334,
          "threshold": 0.17415613455922577
        },
        {
          "accuracy": 0.955,
          "threshold": 0.2175712791531558
        },
        {
          "accuracy": 0.9433333333333334,
          "threshold": 0.19234066892974938
        },
        {
          "accuracy": 0.975,
          "threshold": 0.17415613455922577
        },
        {
          "accuracy": 0.975,
          "threshold": 0.17415613455922577
        },
        {
          "accuracy": 0.9633333333333334,
          "threshold": 0.17415613455922577
        },
        {
          "accuracy": 0.9566666666666667,
          "threshold": 0.17415613455922577
        },
        {
          "accuracy": 0.955,
          "threshold": 0.2175712791531558
        }
      ]
    },
    {
      "dataset": "calfw",
      "config": "single:glintr100 (R100)",
      "n_pairs": 6000,
      "n_genuine": 3000,
      "n_impostor": 3000,
      "accuracy_mean": 0.9616666666666667,
      "accuracy_std": 0.010514540197058346,
      "threshold_mean": 0.2291768134722008,
      "threshold_std": 0.002448982285746191,
      "oracle_accuracy": 0.9623333333333334,
      "oracle_threshold": 0.2303581845319311,
      "tar_at_far_1e2": 0.9316666666666666,
      "tar_at_far_1e3": 0.923,
      "tar_at_far_1e4": 0.6166666666666667,
      "auc": 0.9806001111111111,
      "eer": 0.05666666666666666,
      "folds": [
        {
          "accuracy": 0.97,
          "threshold": 0.2303581845319311
        },
        {
          "accuracy": 0.9466666666666667,
          "threshold": 0.23099258651124754
        },
        {
          "accuracy": 0.965,
          "threshold": 0.2303581845319311
        },
        {
          "accuracy": 0.9583333333333334,
          "threshold": 0.2303581845319311
        },
        {
          "accuracy": 0.945,
          "threshold": 0.23002782023863777
        },
        {
          "accuracy": 0.9766666666666667,
          "threshold": 0.2303581845319311
        },
        {
          "accuracy": 0.9766666666666667,
          "threshold": 0.2303581845319311
        },
        {
          "accuracy": 0.965,
          "threshold": 0.2242993103902681
        },
        {
          "accuracy": 0.9566666666666667,
          "threshold": 0.2242993103902681
        },
        {
          "accuracy": 0.9566666666666667,
          "threshold": 0.2303581845319311
        }
      ]
    },
    {
      "dataset": "calfw",
      "config": "single:w600k_mbf (MBF)",
      "n_pairs": 6000,
      "n_genuine": 3000,
      "n_impostor": 3000,
      "accuracy_mean": 0.9560000000000001,
      "accuracy_std": 0.010857664983268225,
      "threshold_mean": 0.20033570213804394,
      "threshold_std": 0.0037593847466914476,
      "oracle_accuracy": 0.9571666666666667,
      "oracle_threshold": 0.20015133807606303,
      "tar_at_far_1e2": 0.9233333333333333,
      "tar_at_far_1e3": 0.8806666666666667,
      "tar_at_far_1e4": 0.8426666666666667,
      "auc": 0.9786316666666667,
      "eer": 0.05866666666666666,
      "folds": [
        {
          "accuracy": 0.9583333333333334,
          "threshold": 0.20015133807606303
        },
        {
          "accuracy": 0.94,
          "threshold": 0.20015133807606303
        },
        {
          "accuracy": 0.955,
          "threshold": 0.20020501904481353
        },
        {
          "accuracy": 0.9516666666666667,
          "threshold": 0.19590721361830898
        },
        {
          "accuracy": 0.9416666666666667,
          "threshold": 0.20015133807606303
        },
        {
          "accuracy": 0.97,
          "threshold": 0.19590721361830898
        },
        {
          "accuracy": 0.975,
          "threshold": 0.20015133807606303
        },
        {
          "accuracy": 0.96,
          "threshold": 0.20015133807606303
        },
        {
          "accuracy": 0.9616666666666667,
          "threshold": 0.20015133807606303
        },
        {
          "accuracy": 0.9466666666666667,
          "threshold": 0.21042954664262978
        }
      ]
    },
    {
      "dataset": "calfw",
      "config": "ensemble:weighted 0.45/0.45/0.10",
      "n_pairs": 6000,
      "n_genuine": 3000,
      "n_impostor": 3000,
      "accuracy_mean": 0.9606666666666666,
      "accuracy_std": 0.011357816691600568,
      "threshold_mean": 0.21369696307104258,
      "threshold_std": 0.006582994245062976,
      "oracle_accuracy": 0.962,
      "oracle_threshold": 0.20888017622402383,
      "tar_at_far_1e2": 0.9326666666666666,
      "tar_at_far_1e3": 0.9213333333333333,
      "tar_at_far_1e4": 0.8403333333333334,
      "auc": 0.9775367777777777,
      "eer": 0.05733333333333334,
      "folds": [
        {
          "accuracy": 0.9683333333333334,
          "threshold": 0.20888017622402383
        },
        {
          "accuracy": 0.945,
          "threshold": 0.20888017622402383
        },
        {
          "accuracy": 0.965,
          "threshold": 0.220637211564435
        },
        {
          "accuracy": 0.9516666666666667,
          "threshold": 0.2244807595546075
        },
        {
          "accuracy": 0.9466666666666667,
          "threshold": 0.20888017622402383
        },
        {
          "accuracy": 0.9783333333333334,
          "threshold": 0.220637211564435
        },
        {
          "accuracy": 0.9766666666666667,
          "threshold": 0.20888017622402383
        },
        {
          "accuracy": 0.9666666666666667,
          "threshold": 0.20888017622402383
        },
        {
          "accuracy": 0.955,
          "threshold": 0.2061763553423943
        },
        {
          "accuracy": 0.9533333333333334,
          "threshold": 0.220637211564435
        }
      ]
    },
    {
      "dataset": "calfw",
      "config": "ensemble:equal 1/3",
      "n_pairs": 6000,
      "n_genuine": 3000,
      "n_impostor": 3000,
      "accuracy_mean": 0.9608333333333332,
      "accuracy_std": 0.010388294694831634,
      "threshold_mean": 0.21529574020997772,
      "threshold_std": 0.004074089260590444,
      "oracle_accuracy": 0.9615,
      "oracle_threshold": 0.21381382837819668,
      "tar_at_far_1e2": 0.931,
      "tar_at_far_1e3": 0.9173333333333333,
      "tar_at_far_1e4": 0.8613333333333333,
      "auc": 0.9768275555555556,
      "eer": 0.05633333333333334,
      "folds": [
        {
          "accuracy": 0.97,
          "threshold": 0.21381382837819668
        },
        {
          "accuracy": 0.9466666666666667,
          "threshold": 0.21381382837819668
        },
        {
          "accuracy": 0.965,
          "threshold": 0.22718275791855003
        },
        {
          "accuracy": 0.955,
          "threshold": 0.21381382837819668
        },
        {
          "accuracy": 0.945,
          "threshold": 0.21381382837819668
        },
        {
          "accuracy": 0.9783333333333334,
          "threshold": 0.21381382837819668
        },
        {
          "accuracy": 0.9716666666666667,
          "threshold": 0.21655917757811188
        },
        {
          "accuracy": 0.965,
          "threshold": 0.21251866795573837
        },
        {
          "accuracy": 0.955,
          "threshold": 0.21381382837819668
        },
        {
          "accuracy": 0.9566666666666667,
          "threshold": 0.21381382837819668
        }
      ]
    },
    {
      "dataset": "calfw",
      "config": "ensemble:dual r50+r100",
      "n_pairs": 6000,
      "n_genuine": 3000,
      "n_impostor": 3000,
      "accuracy_mean": 0.9615,
      "accuracy_std": 0.011363929289153885,
      "threshold_mean": 0.21338165405234197,
      "threshold_std": 0.0023368501873181364,
      "oracle_accuracy": 0.9621666666666666,
      "oracle_threshold": 0.21405565726034959,
      "tar_at_far_1e2": 0.9316666666666666,
      "tar_at_far_1e3": 0.9226666666666666,
      "tar_at_far_1e4": 0.8366666666666667,
      "auc": 0.977862111111111,
      "eer": 0.05633333333333334,
      "folds": [
        {
          "accuracy": 0.9683333333333334,
          "threshold": 0.21405565726034959
        },
        {
          "accuracy": 0.945,
          "threshold": 0.2139867817928007
        },
        {
          "accuracy": 0.9666666666666667,
          "threshold": 0.21405565726034959
        },
        {
          "accuracy": 0.955,
          "threshold": 0.21501510992715644
        },
        {
          "accuracy": 0.9466666666666667,
          "threshold": 0.21405565726034959
        },
        {
          "accuracy": 0.9783333333333334,
          "threshold": 0.21405565726034959
        },
        {
          "accuracy": 0.9783333333333334,
          "threshold": 0.21405565726034959
        },
        {
          "accuracy": 0.9666666666666667,
          "threshold": 0.21405565726034959
        },
        {
          "accuracy": 0.9566666666666667,
          "threshold": 0.2064250479810154
        },
        {
          "accuracy": 0.9533333333333334,
          "threshold": 0.21405565726034959
        }
      ]
    },
    {
      "dataset": "calfw",
      "config": "ensemble:concat 1536-d",
      "n_pairs": 6000,
      "n_genuine": 3000,
      "n_impostor": 3000,
      "accuracy_mean": 0.9605,
      "accuracy_std": 0.010275375094532253,
      "threshold_mean": 0.20294436235959248,
      "threshold_std": 0.007893991750003184,
      "oracle_accuracy": 0.9618333333333333,
      "oracle_threshold": 0.20107216966157065,
      "tar_at_far_1e2": 0.9323333333333333,
      "tar_at_far_1e3": 0.921,
      "tar_at_far_1e4": 0.841,
      "auc": 0.9801565555555556,
      "eer": 0.05700000000000002,
      "folds": [
        {
          "accuracy": 0.965,
          "threshold": 0.2165808057348783
        },
        {
          "accuracy": 0.945,
          "threshold": 0.20239804539722436
        },
        {
          "accuracy": 0.9633333333333334,
          "threshold": 0.20107216966157065
        },
        {
          "accuracy": 0.955,
          "threshold": 0.1879340606376439
        },
        {
          "accuracy": 0.9466666666666667,
          "threshold": 0.20107216966157065
        },
        {
          "accuracy": 0.9766666666666667,
          "threshold": 0.20107216966157065
        },
        {
          "accuracy": 0.9766666666666667,
          "threshold": 0.20107216966157065
        },
        {
          "accuracy": 0.9633333333333334,
          "threshold": 0.20107216966157065
        },
        {
          "accuracy": 0.9583333333333334,
          "threshold": 0.20050924987584312
        },
        {
          "accuracy": 0.955,
          "threshold": 0.21666061364248168
        }
      ]
    },
    {
      "dataset": "cplfw",
      "config": "single:w600k_r50 (R50)",
      "n_pairs": 6000,
      "n_genuine": 3000,
      "n_impostor": 3000,
      "accuracy_mean": 0.9446666666666665,
      "accuracy_std": 0.010049875621120896,
      "threshold_mean": 0.1739409617778906,
      "threshold_std": 0.0008822766733571792,
      "oracle_accuracy": 0.9448333333333333,
      "oracle_threshold": 0.17368324298542698,
      "tar_at_far_1e2": 0.8973333333333333,
      "tar_at_far_1e3": 0.874,
      "tar_at_far_1e4": 0.44,
      "auc": 0.964246,
      "eer": 0.08266666666666667,
      "folds": [
        {
          "accuracy": 0.9483333333333334,
          "threshold": 0.17368324298542698
        },
        {
          "accuracy": 0.9616666666666667,
          "threshold": 0.17368324298542698
        },
        {
          "accuracy": 0.9333333333333333,
          "threshold": 0.17368324298542698
        },
        {
          "accuracy": 0.9483333333333334,
          "threshold": 0.17337068701036493
        },
        {
          "accuracy": 0.945,
          "threshold": 0.17368324298542698
        },
        {
          "accuracy": 0.9516666666666667,
          "threshold": 0.17368324298542698
        },
        {
          "accuracy": 0.955,
          "threshold": 0.17368324298542698
        },
        {
          "accuracy": 0.9366666666666666,
          "threshold": 0.1765729868851254
        },
        {
          "accuracy": 0.94,
          "threshold": 0.17368324298542698
        },
        {
          "accuracy": 0.9266666666666666,
          "threshold": 0.17368324298542698
        }
      ]
    },
    {
      "dataset": "cplfw",
      "config": "single:glintr100 (R100)",
      "n_pairs": 6000,
      "n_genuine": 3000,
      "n_impostor": 3000,
      "accuracy_mean": 0.9478333333333333,
      "accuracy_std": 0.010029124255553579,
      "threshold_mean": 0.19202655734418994,
      "threshold_std": 0.0008618243920113055,
      "oracle_accuracy": 0.9481666666666667,
      "oracle_threshold": 0.19162434986529786,
      "tar_at_far_1e2": 0.9036666666666666,
      "tar_at_far_1e3": 0.8606666666666667,
      "tar_at_far_1e4": 0.417,
      "auc": 0.9704245555555556,
      "eer": 0.07266666666666666,
      "folds": [
        {
          "accuracy": 0.9466666666666667,
          "threshold": 0.19141863979673798
        },
        {
          "accuracy": 0.965,
          "threshold": 0.19162434986529786
        },
        {
          "accuracy": 0.9433333333333334,
          "threshold": 0.19162434986529786
        },
        {
          "accuracy": 0.9533333333333334,
          "threshold": 0.19162434986529786
        },
        {
          "accuracy": 0.945,
          "threshold": 0.19162434986529786
        },
        {
          "accuracy": 0.9566666666666667,
          "threshold": 0.19413939348320114
        },
        {
          "accuracy": 0.96,
          "threshold": 0.19162434986529786
        },
        {
          "accuracy": 0.9366666666666666,
          "threshold": 0.19170270407192846
        },
        {
          "accuracy": 0.9383333333333334,
          "threshold": 0.19162434986529786
        },
        {
          "accuracy": 0.9333333333333333,
          "threshold": 0.19325873689824447
        }
      ]
    },
    {
      "dataset": "cplfw",
      "config": "single:w600k_mbf (MBF)",
      "n_pairs": 6000,
      "n_genuine": 3000,
      "n_impostor": 3000,
      "accuracy_mean": 0.9263333333333332,
      "accuracy_std": 0.010322575044801356,
      "threshold_mean": 0.15928412512037463,
      "threshold_std": 0.0013836538269090583,
      "oracle_accuracy": 0.927,
      "oracle_threshold": 0.15884417914485052,
      "tar_at_far_1e2": 0.858,
      "tar_at_far_1e3": 0.7916666666666666,
      "tar_at_far_1e4": 0.6523333333333333,
      "auc": 0.9580217777777777,
      "eer": 0.09499999999999999,
      "folds": [
        {
          "accuracy": 0.9266666666666666,
          "threshold": 0.15862776192959732
        },
        {
          "accuracy": 0.9466666666666667,
          "threshold": 0.15884417914485052
        },
        {
          "accuracy": 0.9166666666666666,
          "threshold": 0.15884417914485052
        },
        {
          "accuracy": 0.935,
          "threshold": 0.15884417914485052
        },
        {
          "accuracy": 0.92,
          "threshold": 0.16343033044133498
        },
        {
          "accuracy": 0.9333333333333333,
          "threshold": 0.15884417914485052
        },
        {
          "accuracy": 0.935,
          "threshold": 0.15884417914485052
        },
        {
          "accuracy": 0.9216666666666666,
          "threshold": 0.1588739048188603
        },
        {
          "accuracy": 0.9133333333333333,
          "threshold": 0.15884417914485052
        },
        {
          "accuracy": 0.915,
          "threshold": 0.15884417914485052
        }
      ]
    },
    {
      "dataset": "cplfw",
      "config": "ensemble:weighted 0.45/0.45/0.10",
      "n_pairs": 6000,
      "n_genuine": 3000,
      "n_impostor": 3000,
      "accuracy_mean": 0.9431666666666668,
      "accuracy_std": 0.010709549217611566,
      "threshold_mean": 0.18076674374434162,
      "threshold_std": 0.005440532839020342,
      "oracle_accuracy": 0.945,
      "oracle_threshold": 0.18732610823080129,
      "tar_at_far_1e2": 0.8993333333333333,
      "tar_at_far_1e3": 0.878,
      "tar_at_far_1e4": 0.3933333333333333,
      "auc": 0.9676258888888889,
      "eer": 0.07866666666666666,
      "folds": [
        {
          "accuracy": 0.945,
          "threshold": 0.18732610823080129
        },
        {
          "accuracy": 0.9583333333333334,
          "threshold": 0.17568921192871473
        },
        {
          "accuracy": 0.9333333333333333,
          "threshold": 0.17653797287544093
        },
        {
          "accuracy": 0.9483333333333334,
          "threshold": 0.18732610823080129
        },
        {
          "accuracy": 0.9416666666666667,
          "threshold": 0.18732610823080129
        },
        {
          "accuracy": 0.9533333333333334,
          "threshold": 0.18732610823080129
        },
        {
          "accuracy": 0.9566666666666667,
          "threshold": 0.17568921192871473
        },
        {
          "accuracy": 0.935,
          "threshold": 0.1790681839299112
        },
        {
          "accuracy": 0.9366666666666666,
          "threshold": 0.17568921192871473
        },
        {
          "accuracy": 0.9233333333333333,
          "threshold": 0.17568921192871473
        }
      ]
    },
    {
      "dataset": "cplfw",
      "config": "ensemble:equal 1/3",
      "n_pairs": 6000,
      "n_genuine": 3000,
      "n_impostor": 3000,
      "accuracy_mean": 0.9391666666666667,
      "accuracy_std": 0.011011357772772623,
      "threshold_mean": 0.18818128674260057,
      "threshold_std": 0.009215190563511076,
      "oracle_accuracy": 0.9408333333333333,
      "oracle_threshold": 0.19185384618707108,
      "tar_at_far_1e2": 0.8906666666666667,
      "tar_at_far_1e3": 0.8426666666666667,
      "tar_at_far_1e4": 0.4766666666666667,
      "auc": 0.9660256666666667,
      "eer": 0.08233333333333334,
      "folds": [
        {
          "accuracy": 0.945,
          "threshold": 0.19185384618707108
        },
        {
          "accuracy": 0.9533333333333334,
          "threshold": 0.16095891852178001
        },
        {
          "accuracy": 0.9266666666666666,
          "threshold": 0.19185384618707108
        },
        {
          "accuracy": 0.9466666666666667,
          "threshold": 0.19185384618707108
        },
        {
          "accuracy": 0.9383333333333334,
          "threshold": 0.19340106702624826
        },
        {
          "accuracy": 0.945,
          "threshold": 0.19184940178430027
        },
        {
          "accuracy": 0.9533333333333334,
          "threshold": 0.18816712457916082
        },
        {
          "accuracy": 0.9266666666666666,
          "threshold": 0.18816712457916082
        },
        {
          "accuracy": 0.9366666666666666,
          "threshold": 0.19185384618707108
        },
        {
          "accuracy": 0.92,
          "threshold": 0.19185384618707108
        }
      ]
    },
    {
      "dataset": "cplfw",
      "config": "ensemble:dual r50+r100",
      "n_pairs": 6000,
      "n_genuine": 3000,
      "n_impostor": 3000,
      "accuracy_mean": 0.9446666666666668,
      "accuracy_std": 0.010898521816181192,
      "threshold_mean": 0.19249858694905625,
      "threshold_std": 0.005964839036478463,
      "oracle_accuracy": 0.9456666666666667,
      "oracle_threshold": 0.1944254625415866,
      "tar_at_far_1e2": 0.899,
      "tar_at_far_1e3": 0.8773333333333333,
      "tar_at_far_1e4": 0.3933333333333333,
      "auc": 0.9676281111111111,
      "eer": 0.07766666666666666,
      "folds": [
        {
          "accuracy": 0.945,
          "threshold": 0.1944254625415866
        },
        {
          "accuracy": 0.96,
          "threshold": 0.17468192938412952
        },
        {
          "accuracy": 0.935,
          "threshold": 0.19529544528439355
        },
        {
          "accuracy": 0.95,
          "threshold": 0.1944254625415866
        },
        {
          "accuracy": 0.945,
          "threshold": 0.1944254625415866
        },
        {
          "accuracy": 0.955,
          "threshold": 0.1944254625415866
        },
        {
          "accuracy": 0.9583333333333334,
          "threshold": 0.1944254625415866
        },
        {
          "accuracy": 0.9333333333333333,
          "threshold": 0.1944254625415866
        },
        {
          "accuracy": 0.94,
          "threshold": 0.19529544528439355
        },
        {
          "accuracy": 0.925,
          "threshold": 0.19316027428812632
        }
      ]
    },
    {
      "dataset": "cplfw",
      "config": "ensemble:concat 1536-d",
      "n_pairs": 6000,
      "n_genuine": 3000,
      "n_impostor": 3000,
      "accuracy_mean": 0.9433333333333334,
      "accuracy_std": 0.011737877907772675,
      "threshold_mean": 0.1767343817128952,
      "threshold_std": 0.003898798385591388,
      "oracle_accuracy": 0.9446666666666667,
      "oracle_threshold": 0.17871823072999451,
      "tar_at_far_1e2": 0.898,
      "tar_at_far_1e3": 0.8713333333333333,
      "tar_at_far_1e4": 0.516,
      "auc": 0.9689746666666667,
      "eer": 0.07966666666666666,
      "folds": [
        {
          "accuracy": 0.9466666666666667,
          "threshold": 0.17148703397453374
        },
        {
          "accuracy": 0.9633333333333334,
          "threshold": 0.17871823072999451
        },
        {
          "accuracy": 0.935,
          "threshold": 0.18083140368034284
        },
        {
          "accuracy": 0.95,
          "threshold": 0.17871823072999451
        },
        {
          "accuracy": 0.9416666666666667,
          "threshold": 0.17871823072999451
        },
        {
          "accuracy": 0.9533333333333334,
          "threshold": 0.17871823072999451
        },
        {
          "accuracy": 0.9533333333333334,
          "threshold": 0.16976435581023216
        },
        {
          "accuracy": 0.9283333333333333,
          "threshold": 0.17148703397453374
        },
        {
          "accuracy": 0.9383333333333334,
          "threshold": 0.18018283603933685
        },
        {
          "accuracy": 0.9233333333333333,
          "threshold": 0.17871823072999451
        }
      ]
    }
  ]
}
```


# Degraded-condition verification (TinyFace)

## Protocol as implemented — `backend/scripts/benchmark_tinyface.py`

```text
TinyFace 1:1 verification — degraded / surveillance-distance conditions.

This is the honest secondary number. TinyFace crops are native low-resolution
surveillance captures (median well under 40x40 px), which is what real
investigative footage looks like. The gap between this and the clean-benchmark
result in BENCHMARKS.md §2 is the operational risk, and the two must never be
averaged into a single headline figure.

Pairs are built from the TinyFace testing set, where the filename prefix is the
identity id (`<identity>_<n>.jpg`). Genuine pairs are two captures of one
identity; impostor pairs are two different identities drawn from the same pool,
so the difficulty comes from resolution rather than from an easy negative pool.

The same 10-fold protocol as the clean benchmark is used: the threshold is
fitted on 9 folds and applied to the held-out fold.
```

## Measurement — `runtime/benchmarks/tinyface.json`

### Values

| Field | Value |
|---|---|
| `dataset` | tinyface |
| `config` | single:w600k_mbf |
| `n_pairs` | 6000 |
| `n_genuine` | 3000 |
| `n_impostor` | 3000 |
| `accuracy_mean` | 0.7968333333333333 |
| `accuracy_std` | 0.017755280904564713 |
| `threshold_mean` | 0.2960047570495532 |
| `threshold_std` | 0.00035164099007737015 |
| `oracle_accuracy` | 0.7971666666666667 |
| `oracle_threshold` | 0.2961800935147163 |
| `tar_at_far_1e2` | 0.44566666666666666 |
| `tar_at_far_1e3` | 0.31433333333333335 |
| `tar_at_far_1e4` | 0.20033333333333334 |
| `auc` | 0.8728208888888889 |
| `eer` | 0.21233333333333337 |
| `folds[0].accuracy` | 0.795 |
| `folds[0].threshold` | 0.2961800935147163 |
| `folds[1].accuracy` | 0.7833333333333333 |
| `folds[1].threshold` | 0.2961800935147163 |
| `folds[2].accuracy` | 0.7683333333333333 |
| `folds[2].threshold` | 0.2953014787861624 |
| `folds[3].accuracy` | 0.79 |
| `folds[3].threshold` | 0.2961800935147163 |
| `folds[4].accuracy` | 0.8183333333333334 |
| `folds[4].threshold` | 0.2953014787861624 |
| `folds[5].accuracy` | 0.7966666666666666 |
| `folds[5].threshold` | 0.2961839583201926 |
| `folds[6].accuracy` | 0.7966666666666666 |
| `folds[6].threshold` | 0.2961800935147163 |
| `folds[7].accuracy` | 0.82 |
| `folds[7].threshold` | 0.2961800935147163 |
| `folds[8].accuracy` | 0.8233333333333334 |
| `folds[8].threshold` | 0.2961800935147163 |
| `folds[9].accuracy` | 0.7766666666666666 |
| `folds[9].threshold` | 0.2961800935147163 |
| `resolution_median_hw[0]` | 32 |
| `resolution_median_hw[1]` | 32 |

### Raw artefact

```json
{
  "dataset": "tinyface",
  "config": "single:w600k_mbf",
  "n_pairs": 6000,
  "n_genuine": 3000,
  "n_impostor": 3000,
  "accuracy_mean": 0.7968333333333333,
  "accuracy_std": 0.017755280904564713,
  "threshold_mean": 0.2960047570495532,
  "threshold_std": 0.00035164099007737015,
  "oracle_accuracy": 0.7971666666666667,
  "oracle_threshold": 0.2961800935147163,
  "tar_at_far_1e2": 0.44566666666666666,
  "tar_at_far_1e3": 0.31433333333333335,
  "tar_at_far_1e4": 0.20033333333333334,
  "auc": 0.8728208888888889,
  "eer": 0.21233333333333337,
  "folds": [
    {
      "accuracy": 0.795,
      "threshold": 0.2961800935147163
    },
    {
      "accuracy": 0.7833333333333333,
      "threshold": 0.2961800935147163
    },
    {
      "accuracy": 0.7683333333333333,
      "threshold": 0.2953014787861624
    },
    {
      "accuracy": 0.79,
      "threshold": 0.2961800935147163
    },
    {
      "accuracy": 0.8183333333333334,
      "threshold": 0.2953014787861624
    },
    {
      "accuracy": 0.7966666666666666,
      "threshold": 0.2961839583201926
    },
    {
      "accuracy": 0.7966666666666666,
      "threshold": 0.2961800935147163
    },
    {
      "accuracy": 0.82,
      "threshold": 0.2961800935147163
    },
    {
      "accuracy": 0.8233333333333334,
      "threshold": 0.2961800935147163
    },
    {
      "accuracy": 0.7766666666666666,
      "threshold": 0.2961800935147163
    }
  ],
  "resolution_median_hw": [
    32,
    32
  ]
}
```


# Threshold calibration

## Protocol as implemented — `backend/scripts/calibrate_threshold_suite.py`

```text
Calibrate the decision threshold across the WHOLE published benchmark suite.

    python backend/scripts/calibrate_threshold_suite.py --model w600k_r50

WHY NOT scripts/calibrate_threshold.py
--------------------------------------
That script calibrates against a folder-per-identity dataset and constructs its
own pairs. It cannot read the `.bin` protocol packs, so it cannot calibrate
across LFW / AgeDB-30 / CFP-FP / CALFW / CPLFW — the published pair lists this
project reports accuracy on. Calibrating on one dataset and deploying the
result across all of them is exactly the mistake this script exists to avoid.

WHAT IT DOES
    1. Loads the cached embeddings for each dataset (same ones the accuracy
       benchmark used, so the calibration and the reported accuracy cannot
       disagree).
    2. Pools every impostor score across all datasets and finds the single
       threshold giving a target FMR on the combined distribution.
    3. Reports each dataset's own FMR=target threshold, so the spread between
       datasets is visible rather than hidden inside an average.
    4. Reports accuracy / FNMR / FMR per dataset at both the incumbent and the
       proposed threshold.

A combined threshold is a compromise. If per-dataset thresholds differ widely,
no single number serves them all, and that must be visible in the output rather
than discovered later in production.
```

## Measurement — `runtime/benchmarks/threshold_calibration.json`

### Values

| Field | Value |
|---|---|
| `model` | w600k_r50 |
| `target_fmr` | 0.001 |
| `combined_threshold` | 0.23628424632584857 |
| `single_dataset_candidate` | 0.2871 |
| `per_dataset_own_threshold.lfw` | 0.21372212912782104 |
| `per_dataset_own_threshold.agedb_30` | 0.23878224862093067 |
| `per_dataset_own_threshold.cfp_fp` | 0.20943365123642504 |
| `per_dataset_own_threshold.calfw` | 0.2480450954955708 |
| `per_dataset_own_threshold.cplfw` | 0.24043860999340935 |
| `spread` | 0.038611444259145755 |
| `per_dataset.lfw.incumbent.threshold` | 0.2 |
| `per_dataset.lfw.incumbent.accuracy` | 0.9976666666666667 |
| `per_dataset.lfw.incumbent.fnmr` | 0.003 |
| `per_dataset.lfw.incumbent.fmr` | 0.0016666666666666668 |
| `per_dataset.lfw.combined.threshold` | 0.23628424632584857 |
| `per_dataset.lfw.combined.accuracy` | 0.998 |
| `per_dataset.lfw.combined.fnmr` | 0.0033333333333333335 |
| `per_dataset.lfw.combined.fmr` | 0.0006666666666666666 |
| `per_dataset.lfw.own_fmr_threshold` | 0.21372212912782104 |
| `per_dataset.agedb_30.incumbent.threshold` | 0.2 |
| `per_dataset.agedb_30.incumbent.accuracy` | 0.9813333333333333 |
| `per_dataset.agedb_30.incumbent.fnmr` | 0.03133333333333333 |
| `per_dataset.agedb_30.incumbent.fmr` | 0.006 |
| `per_dataset.agedb_30.combined.threshold` | 0.23628424632584857 |
| `per_dataset.agedb_30.combined.accuracy` | 0.9796666666666667 |
| `per_dataset.agedb_30.combined.fnmr` | 0.03933333333333333 |
| `per_dataset.agedb_30.combined.fmr` | 0.0013333333333333333 |
| `per_dataset.agedb_30.own_fmr_threshold` | 0.23878224862093067 |
| `per_dataset.cfp_fp.incumbent.threshold` | 0.2 |
| `per_dataset.cfp_fp.incumbent.accuracy` | 0.9742857142857143 |
| `per_dataset.cfp_fp.incumbent.fnmr` | 0.05 |
| `per_dataset.cfp_fp.incumbent.fmr` | 0.0014285714285714286 |
| `per_dataset.cfp_fp.combined.threshold` | 0.23628424632584857 |
| `per_dataset.cfp_fp.combined.accuracy` | 0.9711428571428572 |
| `per_dataset.cfp_fp.combined.fnmr` | 0.05771428571428571 |
| `per_dataset.cfp_fp.combined.fmr` | 0.0 |
| `per_dataset.cfp_fp.own_fmr_threshold` | 0.20943365123642504 |
| `per_dataset.calfw.incumbent.threshold` | 0.2 |
| `per_dataset.calfw.incumbent.accuracy` | 0.9606666666666667 |
| `per_dataset.calfw.incumbent.fnmr` | 0.07466666666666667 |
| `per_dataset.calfw.incumbent.fmr` | 0.004 |
| `per_dataset.calfw.combined.threshold` | 0.23628424632584857 |
| `per_dataset.calfw.combined.accuracy` | 0.9606666666666667 |
| `per_dataset.calfw.combined.fnmr` | 0.07733333333333334 |
| `per_dataset.calfw.combined.fmr` | 0.0013333333333333333 |
| `per_dataset.calfw.own_fmr_threshold` | 0.2480450954955708 |
| `per_dataset.cplfw.incumbent.threshold` | 0.2 |
| `per_dataset.cplfw.incumbent.accuracy` | 0.9431666666666667 |
| `per_dataset.cplfw.incumbent.fnmr` | 0.11166666666666666 |
| `per_dataset.cplfw.incumbent.fmr` | 0.002 |
| `per_dataset.cplfw.combined.threshold` | 0.23628424632584857 |
| `per_dataset.cplfw.combined.accuracy` | 0.9365 |
| `per_dataset.cplfw.combined.fnmr` | 0.12566666666666668 |
| `per_dataset.cplfw.combined.fmr` | 0.0013333333333333333 |
| `per_dataset.cplfw.own_fmr_threshold` | 0.24043860999340935 |

### Raw artefact

```json
{
  "model": "w600k_r50",
  "target_fmr": 0.001,
  "combined_threshold": 0.23628424632584857,
  "single_dataset_candidate": 0.2871,
  "per_dataset_own_threshold": {
    "lfw": 0.21372212912782104,
    "agedb_30": 0.23878224862093067,
    "cfp_fp": 0.20943365123642504,
    "calfw": 0.2480450954955708,
    "cplfw": 0.24043860999340935
  },
  "spread": 0.038611444259145755,
  "datasets_excluded_no_cache": [],
  "per_dataset": {
    "lfw": {
      "incumbent": {
        "threshold": 0.2,
        "accuracy": 0.9976666666666667,
        "fnmr": 0.003,
        "fmr": 0.0016666666666666668
      },
      "combined": {
        "threshold": 0.23628424632584857,
        "accuracy": 0.998,
        "fnmr": 0.0033333333333333335,
        "fmr": 0.0006666666666666666
      },
      "own_fmr_threshold": 0.21372212912782104
    },
    "agedb_30": {
      "incumbent": {
        "threshold": 0.2,
        "accuracy": 0.9813333333333333,
        "fnmr": 0.03133333333333333,
        "fmr": 0.006
      },
      "combined": {
        "threshold": 0.23628424632584857,
        "accuracy": 0.9796666666666667,
        "fnmr": 0.03933333333333333,
        "fmr": 0.0013333333333333333
      },
      "own_fmr_threshold": 0.23878224862093067
    },
    "cfp_fp": {
      "incumbent": {
        "threshold": 0.2,
        "accuracy": 0.9742857142857143,
        "fnmr": 0.05,
        "fmr": 0.0014285714285714286
      },
      "combined": {
        "threshold": 0.23628424632584857,
        "accuracy": 0.9711428571428572,
        "fnmr": 0.05771428571428571,
        "fmr": 0.0
      },
      "own_fmr_threshold": 0.20943365123642504
    },
    "calfw": {
      "incumbent": {
        "threshold": 0.2,
        "accuracy": 0.9606666666666667,
        "fnmr": 0.07466666666666667,
        "fmr": 0.004
      },
      "combined": {
        "threshold": 0.23628424632584857,
        "accuracy": 0.9606666666666667,
        "fnmr": 0.07733333333333334,
        "fmr": 0.0013333333333333333
      },
      "own_fmr_threshold": 0.2480450954955708
    },
    "cplfw": {
      "incumbent": {
        "threshold": 0.2,
        "accuracy": 0.9431666666666667,
        "fnmr": 0.11166666666666666,
        "fmr": 0.002
      },
      "combined": {
        "threshold": 0.23628424632584857,
        "accuracy": 0.9365,
        "fnmr": 0.12566666666666668,
        "fmr": 0.0013333333333333333
      },
      "own_fmr_threshold": 0.24043860999340935
    }
  }
}
```

## Measurement — `runtime/benchmarks/threshold_calibration_at_2871.json`

### Values

| Field | Value |
|---|---|
| `model` | w600k_r50 |
| `target_fmr` | 0.001 |
| `combined_threshold` | 0.23628424632584857 |
| `single_dataset_candidate` | 0.2871 |
| `per_dataset_own_threshold.lfw` | 0.21372212912782104 |
| `per_dataset_own_threshold.agedb_30` | 0.23878224862093067 |
| `per_dataset_own_threshold.cfp_fp` | 0.20943365123642504 |
| `per_dataset_own_threshold.calfw` | 0.2480450954955708 |
| `per_dataset_own_threshold.cplfw` | 0.24043860999340935 |
| `spread` | 0.038611444259145755 |
| `per_dataset.lfw.incumbent.threshold` | 0.2871 |
| `per_dataset.lfw.incumbent.accuracy` | 0.9978333333333333 |
| `per_dataset.lfw.incumbent.fnmr` | 0.004333333333333333 |
| `per_dataset.lfw.incumbent.fmr` | 0.0 |
| `per_dataset.lfw.combined.threshold` | 0.23628424632584857 |
| `per_dataset.lfw.combined.accuracy` | 0.998 |
| `per_dataset.lfw.combined.fnmr` | 0.0033333333333333335 |
| `per_dataset.lfw.combined.fmr` | 0.0006666666666666666 |
| `per_dataset.lfw.own_fmr_threshold` | 0.21372212912782104 |
| `per_dataset.agedb_30.incumbent.threshold` | 0.2871 |
| `per_dataset.agedb_30.incumbent.accuracy` | 0.9668333333333333 |
| `per_dataset.agedb_30.incumbent.fnmr` | 0.06633333333333333 |
| `per_dataset.agedb_30.incumbent.fmr` | 0.0 |
| `per_dataset.agedb_30.combined.threshold` | 0.23628424632584857 |
| `per_dataset.agedb_30.combined.accuracy` | 0.9796666666666667 |
| `per_dataset.agedb_30.combined.fnmr` | 0.03933333333333333 |
| `per_dataset.agedb_30.combined.fmr` | 0.0013333333333333333 |
| `per_dataset.agedb_30.own_fmr_threshold` | 0.23878224862093067 |
| `per_dataset.cfp_fp.incumbent.threshold` | 0.2871 |
| `per_dataset.cfp_fp.incumbent.accuracy` | 0.9627142857142857 |
| `per_dataset.cfp_fp.incumbent.fnmr` | 0.07457142857142857 |
| `per_dataset.cfp_fp.incumbent.fmr` | 0.0 |
| `per_dataset.cfp_fp.combined.threshold` | 0.23628424632584857 |
| `per_dataset.cfp_fp.combined.accuracy` | 0.9711428571428572 |
| `per_dataset.cfp_fp.combined.fnmr` | 0.05771428571428571 |
| `per_dataset.cfp_fp.combined.fmr` | 0.0 |
| `per_dataset.cfp_fp.own_fmr_threshold` | 0.20943365123642504 |
| `per_dataset.calfw.incumbent.threshold` | 0.2871 |
| `per_dataset.calfw.incumbent.accuracy` | 0.9553333333333334 |
| `per_dataset.calfw.incumbent.fnmr` | 0.08866666666666667 |
| `per_dataset.calfw.incumbent.fmr` | 0.0006666666666666666 |
| `per_dataset.calfw.combined.threshold` | 0.23628424632584857 |
| `per_dataset.calfw.combined.accuracy` | 0.9606666666666667 |
| `per_dataset.calfw.combined.fnmr` | 0.07733333333333334 |
| `per_dataset.calfw.combined.fmr` | 0.0013333333333333333 |
| `per_dataset.calfw.own_fmr_threshold` | 0.2480450954955708 |
| `per_dataset.cplfw.incumbent.threshold` | 0.2871 |
| `per_dataset.cplfw.incumbent.accuracy` | 0.9255 |
| `per_dataset.cplfw.incumbent.fnmr` | 0.14866666666666667 |
| `per_dataset.cplfw.incumbent.fmr` | 0.0003333333333333333 |
| `per_dataset.cplfw.combined.threshold` | 0.23628424632584857 |
| `per_dataset.cplfw.combined.accuracy` | 0.9365 |
| `per_dataset.cplfw.combined.fnmr` | 0.12566666666666668 |
| `per_dataset.cplfw.combined.fmr` | 0.0013333333333333333 |
| `per_dataset.cplfw.own_fmr_threshold` | 0.24043860999340935 |

### Raw artefact

```json
{
  "model": "w600k_r50",
  "target_fmr": 0.001,
  "combined_threshold": 0.23628424632584857,
  "single_dataset_candidate": 0.2871,
  "per_dataset_own_threshold": {
    "lfw": 0.21372212912782104,
    "agedb_30": 0.23878224862093067,
    "cfp_fp": 0.20943365123642504,
    "calfw": 0.2480450954955708,
    "cplfw": 0.24043860999340935
  },
  "spread": 0.038611444259145755,
  "datasets_excluded_no_cache": [],
  "per_dataset": {
    "lfw": {
      "incumbent": {
        "threshold": 0.2871,
        "accuracy": 0.9978333333333333,
        "fnmr": 0.004333333333333333,
        "fmr": 0.0
      },
      "combined": {
        "threshold": 0.23628424632584857,
        "accuracy": 0.998,
        "fnmr": 0.0033333333333333335,
        "fmr": 0.0006666666666666666
      },
      "own_fmr_threshold": 0.21372212912782104
    },
    "agedb_30": {
      "incumbent": {
        "threshold": 0.2871,
        "accuracy": 0.9668333333333333,
        "fnmr": 0.06633333333333333,
        "fmr": 0.0
      },
      "combined": {
        "threshold": 0.23628424632584857,
        "accuracy": 0.9796666666666667,
        "fnmr": 0.03933333333333333,
        "fmr": 0.0013333333333333333
      },
      "own_fmr_threshold": 0.23878224862093067
    },
    "cfp_fp": {
      "incumbent": {
        "threshold": 0.2871,
        "accuracy": 0.9627142857142857,
        "fnmr": 0.07457142857142857,
        "fmr": 0.0
      },
      "combined": {
        "threshold": 0.23628424632584857,
        "accuracy": 0.9711428571428572,
        "fnmr": 0.05771428571428571,
        "fmr": 0.0
      },
      "own_fmr_threshold": 0.20943365123642504
    },
    "calfw": {
      "incumbent": {
        "threshold": 0.2871,
        "accuracy": 0.9553333333333334,
        "fnmr": 0.08866666666666667,
        "fmr": 0.0006666666666666666
      },
      "combined": {
        "threshold": 0.23628424632584857,
        "accuracy": 0.9606666666666667,
        "fnmr": 0.07733333333333334,
        "fmr": 0.0013333333333333333
      },
      "own_fmr_threshold": 0.2480450954955708
    },
    "cplfw": {
      "incumbent": {
        "threshold": 0.2871,
        "accuracy": 0.9255,
        "fnmr": 0.14866666666666667,
        "fmr": 0.0003333333333333333
      },
      "combined": {
        "threshold": 0.23628424632584857,
        "accuracy": 0.9365,
        "fnmr": 0.12566666666666668,
        "fmr": 0.0013333333333333333
      },
      "own_fmr_threshold": 0.24043860999340935
    }
  }
}
```


# Demographic differentials

## Protocol as implemented — `backend/scripts/benchmark_demographics.py`

```text
Demographic error-rate breakdown on AgeDB (Phase 3e).

WHY THIS IS SEPARATE FROM benchmark_verification.py
----------------------------------------------------
The standard agedb_30.bin protocol pack contains anonymised image pairs -- it
carries an is-same flag and nothing else. It cannot answer "does this system
fail more often on older subjects, or on women?" because the identities and
attributes were stripped when the pack was built.

The raw AgeDB image folder does carry them: filenames are
`<idx>_<Name>_<age>_<gender>.jpg`. This script therefore builds its OWN
genuine/impostor pairs from the raw folder so every pair keeps its age and
gender labels, then reports false-match and false-non-match rates per subgroup.

Because these are locally constructed pairs, the aggregate accuracy here is NOT
comparable to the published AgeDB-30 number and must not be quoted as such.
What IS meaningful is the *relative* error rate between subgroups measured on
one consistent pair set -- which is the entire point of a bias audit
(cf. NIST FRVT Part 3, which reports demographic differentials this way).

Impostor pairs are matched WITHIN a subgroup (same gender, same age bucket).
Cross-group impostor pairs are systematically easier -- a 25-year-old woman
versus a 70-year-old man is a trivial rejection -- and mixing them in would
deflate the false-match rate of every group.
```

## Measurement — `runtime/benchmarks/demographics.json`

### Values

| Field | Value |
|---|---|
| `model` | glintr100 |
| `threshold_at_fmr_1e3` | 0.3088653841853034 |
| `note` | locally constructed pairs; not comparable to published AgeDB-30 |
| `groups.ALL.genuine` | 8098 |
| `groups.ALL.impostor` | 32000 |
| `groups.ALL.fnmr` | 0.0432205482835268 |
| `groups.ALL.fmr` | 0.00096875 |
| `groups.gender=f.genuine` | 3300 |
| `groups.gender=f.impostor` | 16000 |
| `groups.gender=f.fnmr` | 0.060909090909090906 |
| `groups.gender=f.fmr` | 0.0014375 |
| `groups.gender=m.genuine` | 4798 |
| `groups.gender=m.impostor` | 16000 |
| `groups.gender=m.fnmr` | 0.031054606085869112 |
| `groups.gender=m.fmr` | 0.0005 |
| `groups.age=0-25.genuine` | 854 |
| `groups.age=0-25.impostor` | 8000 |
| `groups.age=0-25.fnmr` | 0.10655737704918032 |
| `groups.age=0-25.fmr` | 0.001 |
| `groups.age=26-40.genuine` | 2595 |
| `groups.age=26-40.impostor` | 8000 |
| `groups.age=26-40.fnmr` | 0.04007707129094412 |
| `groups.age=26-40.fmr` | 0.001 |
| `groups.age=41-55.genuine` | 2199 |
| `groups.age=41-55.impostor` | 8000 |
| `groups.age=41-55.fnmr` | 0.02546612096407458 |
| `groups.age=41-55.fmr` | 0.001 |
| `groups.age=56+.genuine` | 2450 |
| `groups.age=56+.impostor` | 8000 |
| `groups.age=56+.fnmr` | 0.04040816326530612 |
| `groups.age=56+.fmr` | 0.000875 |

### Raw artefact

```json
{
  "model": "glintr100",
  "threshold_at_fmr_1e3": 0.3088653841853034,
  "note": "locally constructed pairs; not comparable to published AgeDB-30",
  "groups": {
    "ALL": {
      "genuine": 8098,
      "impostor": 32000,
      "fnmr": 0.0432205482835268,
      "fmr": 0.00096875
    },
    "gender=f": {
      "genuine": 3300,
      "impostor": 16000,
      "fnmr": 0.060909090909090906,
      "fmr": 0.0014375
    },
    "gender=m": {
      "genuine": 4798,
      "impostor": 16000,
      "fnmr": 0.031054606085869112,
      "fmr": 0.0005
    },
    "age=0-25": {
      "genuine": 854,
      "impostor": 8000,
      "fnmr": 0.10655737704918032,
      "fmr": 0.001
    },
    "age=26-40": {
      "genuine": 2595,
      "impostor": 8000,
      "fnmr": 0.04007707129094412,
      "fmr": 0.001
    },
    "age=41-55": {
      "genuine": 2199,
      "impostor": 8000,
      "fnmr": 0.02546612096407458,
      "fmr": 0.001
    },
    "age=56+": {
      "genuine": 2450,
      "impostor": 8000,
      "fnmr": 0.04040816326530612,
      "fmr": 0.000875
    }
  }
}
```

## Measurement — `runtime/benchmarks/demographics_w600k_r50_thr020.json`

### Values

| Field | Value |
|---|---|
| `model` | w600k_r50 |
| `threshold_used` | 0.2 |
| `threshold_kind` | operating (deployed decision threshold) |
| `threshold_at_fmr_1e3` | 0.2870860145310593 |
| `note` | locally constructed pairs; not comparable to published AgeDB-30 |
| `groups.ALL.genuine` | 8098 |
| `groups.ALL.impostor` | 32000 |
| `groups.ALL.fnmr` | 0.03297110397629044 |
| `groups.ALL.fmr` | 0.01190625 |
| `groups.gender=f.genuine` | 3300 |
| `groups.gender=f.impostor` | 16000 |
| `groups.gender=f.fnmr` | 0.04878787878787879 |
| `groups.gender=f.fmr` | 0.017 |
| `groups.gender=m.genuine` | 4798 |
| `groups.gender=m.impostor` | 16000 |
| `groups.gender=m.fnmr` | 0.02209253855773239 |
| `groups.gender=m.fmr` | 0.0068125 |
| `groups.age=0-25.genuine` | 854 |
| `groups.age=0-25.impostor` | 8000 |
| `groups.age=0-25.fnmr` | 0.07611241217798595 |
| `groups.age=0-25.fmr` | 0.012125 |
| `groups.age=26-40.genuine` | 2595 |
| `groups.age=26-40.impostor` | 8000 |
| `groups.age=26-40.fnmr` | 0.03236994219653179 |
| `groups.age=26-40.fmr` | 0.012625 |
| `groups.age=41-55.genuine` | 2199 |
| `groups.age=41-55.impostor` | 8000 |
| `groups.age=41-55.fnmr` | 0.021373351523419737 |
| `groups.age=41-55.fmr` | 0.011 |
| `groups.age=56+.genuine` | 2450 |
| `groups.age=56+.impostor` | 8000 |
| `groups.age=56+.fnmr` | 0.028979591836734694 |
| `groups.age=56+.fmr` | 0.011875 |

### Raw artefact

```json
{
  "model": "w600k_r50",
  "threshold_used": 0.2,
  "threshold_kind": "operating (deployed decision threshold)",
  "threshold_at_fmr_1e3": 0.2870860145310593,
  "note": "locally constructed pairs; not comparable to published AgeDB-30",
  "groups": {
    "ALL": {
      "genuine": 8098,
      "impostor": 32000,
      "fnmr": 0.03297110397629044,
      "fmr": 0.01190625
    },
    "gender=f": {
      "genuine": 3300,
      "impostor": 16000,
      "fnmr": 0.04878787878787879,
      "fmr": 0.017
    },
    "gender=m": {
      "genuine": 4798,
      "impostor": 16000,
      "fnmr": 0.02209253855773239,
      "fmr": 0.0068125
    },
    "age=0-25": {
      "genuine": 854,
      "impostor": 8000,
      "fnmr": 0.07611241217798595,
      "fmr": 0.012125
    },
    "age=26-40": {
      "genuine": 2595,
      "impostor": 8000,
      "fnmr": 0.03236994219653179,
      "fmr": 0.012625
    },
    "age=41-55": {
      "genuine": 2199,
      "impostor": 8000,
      "fnmr": 0.021373351523419737,
      "fmr": 0.011
    },
    "age=56+": {
      "genuine": 2450,
      "impostor": 8000,
      "fnmr": 0.028979591836734694,
      "fmr": 0.011875
    }
  }
}
```

## Measurement — `runtime/benchmarks/demographics_w600k_r50_thr0236.json`

### Values

| Field | Value |
|---|---|
| `model` | w600k_r50 |
| `threshold_used` | 0.2363 |
| `threshold_kind` | operating (deployed decision threshold) |
| `threshold_at_fmr_1e3` | 0.2870860145310593 |
| `note` | locally constructed pairs; not comparable to published AgeDB-30 |
| `groups.ALL.genuine` | 8098 |
| `groups.ALL.impostor` | 32000 |
| `groups.ALL.fnmr` | 0.041738700913805875 |
| `groups.ALL.fmr` | 0.0044375 |
| `groups.gender=f.genuine` | 3300 |
| `groups.gender=f.impostor` | 16000 |
| `groups.gender=f.fnmr` | 0.058787878787878785 |
| `groups.gender=f.fmr` | 0.006625 |
| `groups.gender=m.genuine` | 4798 |
| `groups.gender=m.impostor` | 16000 |
| `groups.gender=m.fnmr` | 0.030012505210504376 |
| `groups.gender=m.fmr` | 0.00225 |
| `groups.age=0-25.genuine` | 854 |
| `groups.age=0-25.impostor` | 8000 |
| `groups.age=0-25.fnmr` | 0.10772833723653395 |
| `groups.age=0-25.fmr` | 0.00375 |
| `groups.age=26-40.genuine` | 2595 |
| `groups.age=26-40.impostor` | 8000 |
| `groups.age=26-40.fnmr` | 0.038535645472061654 |
| `groups.age=26-40.fmr` | 0.0045 |
| `groups.age=41-55.genuine` | 2199 |
| `groups.age=41-55.impostor` | 8000 |
| `groups.age=41-55.fnmr` | 0.02501136880400182 |
| `groups.age=41-55.fmr` | 0.004875 |
| `groups.age=56+.genuine` | 2450 |
| `groups.age=56+.impostor` | 8000 |
| `groups.age=56+.fnmr` | 0.037142857142857144 |
| `groups.age=56+.fmr` | 0.004625 |

### Raw artefact

```json
{
  "model": "w600k_r50",
  "threshold_used": 0.2363,
  "threshold_kind": "operating (deployed decision threshold)",
  "threshold_at_fmr_1e3": 0.2870860145310593,
  "note": "locally constructed pairs; not comparable to published AgeDB-30",
  "groups": {
    "ALL": {
      "genuine": 8098,
      "impostor": 32000,
      "fnmr": 0.041738700913805875,
      "fmr": 0.0044375
    },
    "gender=f": {
      "genuine": 3300,
      "impostor": 16000,
      "fnmr": 0.058787878787878785,
      "fmr": 0.006625
    },
    "gender=m": {
      "genuine": 4798,
      "impostor": 16000,
      "fnmr": 0.030012505210504376,
      "fmr": 0.00225
    },
    "age=0-25": {
      "genuine": 854,
      "impostor": 8000,
      "fnmr": 0.10772833723653395,
      "fmr": 0.00375
    },
    "age=26-40": {
      "genuine": 2595,
      "impostor": 8000,
      "fnmr": 0.038535645472061654,
      "fmr": 0.0045
    },
    "age=41-55": {
      "genuine": 2199,
      "impostor": 8000,
      "fnmr": 0.02501136880400182,
      "fmr": 0.004875
    },
    "age=56+": {
      "genuine": 2450,
      "impostor": 8000,
      "fnmr": 0.037142857142857144,
      "fmr": 0.004625
    }
  }
}
```

## Measurement — `runtime/benchmarks/demographics_w600k_r50_thr0287.json`

### Values

| Field | Value |
|---|---|
| `model` | w600k_r50 |
| `threshold_used` | 0.2871 |
| `threshold_kind` | operating (deployed decision threshold) |
| `threshold_at_fmr_1e3` | 0.2870860145310593 |
| `note` | locally constructed pairs; not comparable to published AgeDB-30 |
| `groups.ALL.genuine` | 8098 |
| `groups.ALL.impostor` | 32000 |
| `groups.ALL.fnmr` | 0.0632254877747592 |
| `groups.ALL.fmr` | 0.00096875 |
| `groups.gender=f.genuine` | 3300 |
| `groups.gender=f.impostor` | 16000 |
| `groups.gender=f.fnmr` | 0.08454545454545455 |
| `groups.gender=f.fmr` | 0.001625 |
| `groups.gender=m.genuine` | 4798 |
| `groups.gender=m.impostor` | 16000 |
| `groups.gender=m.fnmr` | 0.04856190079199667 |
| `groups.gender=m.fmr` | 0.0003125 |
| `groups.age=0-25.genuine` | 854 |
| `groups.age=0-25.impostor` | 8000 |
| `groups.age=0-25.fnmr` | 0.14754098360655737 |
| `groups.age=0-25.fmr` | 0.001 |
| `groups.age=26-40.genuine` | 2595 |
| `groups.age=26-40.impostor` | 8000 |
| `groups.age=26-40.fnmr` | 0.057803468208092484 |
| `groups.age=26-40.fmr` | 0.0005 |
| `groups.age=41-55.genuine` | 2199 |
| `groups.age=41-55.impostor` | 8000 |
| `groups.age=41-55.fnmr` | 0.03910868576625739 |
| `groups.age=41-55.fmr` | 0.0015 |
| `groups.age=56+.genuine` | 2450 |
| `groups.age=56+.impostor` | 8000 |
| `groups.age=56+.fnmr` | 0.061224489795918366 |
| `groups.age=56+.fmr` | 0.000875 |

### Raw artefact

```json
{
  "model": "w600k_r50",
  "threshold_used": 0.2871,
  "threshold_kind": "operating (deployed decision threshold)",
  "threshold_at_fmr_1e3": 0.2870860145310593,
  "note": "locally constructed pairs; not comparable to published AgeDB-30",
  "groups": {
    "ALL": {
      "genuine": 8098,
      "impostor": 32000,
      "fnmr": 0.0632254877747592,
      "fmr": 0.00096875
    },
    "gender=f": {
      "genuine": 3300,
      "impostor": 16000,
      "fnmr": 0.08454545454545455,
      "fmr": 0.001625
    },
    "gender=m": {
      "genuine": 4798,
      "impostor": 16000,
      "fnmr": 0.04856190079199667,
      "fmr": 0.0003125
    },
    "age=0-25": {
      "genuine": 854,
      "impostor": 8000,
      "fnmr": 0.14754098360655737,
      "fmr": 0.001
    },
    "age=26-40": {
      "genuine": 2595,
      "impostor": 8000,
      "fnmr": 0.057803468208092484,
      "fmr": 0.0005
    },
    "age=41-55": {
      "genuine": 2199,
      "impostor": 8000,
      "fnmr": 0.03910868576625739,
      "fmr": 0.0015
    },
    "age=56+": {
      "genuine": 2450,
      "impostor": 8000,
      "fnmr": 0.061224489795918366,
      "fmr": 0.000875
    }
  }
}
```


# Training/evaluation contamination

## Protocol as implemented — `backend/scripts/audit_train_eval_overlap.py`

```text
Item 36 — audit identity overlap between TRAINING and EVALUATION data.

    python backend/scripts/audit_train_eval_overlap.py --per-identity 2

WHY THIS IS EMBEDDING-BASED AND NOT NAME-BASED
-----------------------------------------------
Name matching is impossible here, established by inspection rather than
assumption:

  * The .bin evaluation packs carry NO identity metadata at all. Verified by
    scanning their pickle opcodes without executing them: the only strings in
    the byte stream are numpy dtype descriptors.
  * The training sets carry no names either. train.lst records original paths
    like /raid5data/dplearn/CASIA-WebFace/0000045/001.jpg — numeric folder IDs.
    MegaFace uses Flickr handles (100001044@N04_identity_0). Neither maps to a
    person's name without an external table that is not on disk.

So overlap can only be detected in embedding space: embed training images,
embed evaluation images, and look for pairs that score high enough to be the
same photograph or the same person.

WHAT THE THRESHOLDS MEAN
------------------------
Calibrated against this project's own measured distributions (BENCHMARKS.md):
genuine AgeDB pairs average ~0.49, impostors ~0.00, and the deployed decision
threshold is 0.2871.

  >= 0.90  near-duplicate  — almost certainly the SAME photograph, or a crop
                             of it. This is the unambiguous leak.
  >= 0.70  probable same identity — well above any genuine-pair average, so a
                             different photo of the same person.
  >= 0.2871 same-person by the system's own deployed rule. Reported for
                             context; at this level false positives are
                             expected and it is NOT evidence of leakage on its
                             own.

HONEST LIMITATION — READ BEFORE QUOTING THE RESULT
--------------------------------------------------
This samples K images per training identity. It can prove overlap EXISTS. It
cannot prove overlap is ABSENT: an identity whose sampled images happen not to
resemble the evaluation shots will be missed. The conclusion wording must say
"no overlap detected at this sampling depth", never "no overlap exists".
```

## Protocol as implemented — `backend/scripts/build_exclusion_list.py`

```text
Phase 6 step 3 — build the training-identity EXCLUSION LIST.

    python backend/scripts/build_exclusion_list.py --per-identity 10

audit_train_eval_overlap.py answers "is there overlap?" from the EVAL side: for
each evaluation image, how similar is the nearest training image. That proves
contamination exists but cannot be acted on, because it never names the
training identities responsible.

This answers the other direction: for each TRAINING identity, how similar is its
nearest evaluation image. Identities above the threshold are written to an
exclusion list so a fine-tune can drop them.

THRESHOLD CHOICE — DELIBERATELY CONSERVATIVE
--------------------------------------------
Default 0.40, which is BELOW the ~0.49 mean of genuine same-person pairs in this
system. That is intentional and asymmetric:

  * Excluding a clean identity costs a little training data. Cheap.
  * Keeping a contaminated one means the model memorises a face it will later
    be tested on, and every downstream accuracy number becomes unfalsifiable.

So the threshold errs toward over-exclusion. A tighter value (0.70, the
"probable same identity" line used in the audit) would leave same-person pairs
scoring 0.40-0.70 in the training set, and those are exactly the cross-age and
cross-pose pairs the benchmarks are built from.

Uses faiss IndexFlatIP for the nearest-neighbour search -- exact, no recall
loss, and the one place BENCHMARKS.md §7d found it genuinely worth using.
```

## Protocol as implemented — `backend/scripts/audit_qmul_survface.py`

```text
QMUL-SurvFace identity-overlap audit + exclusion list.

    python backend/scripts/audit_qmul_survface.py

Same methodology as build_exclusion_list.py (CASIA), with two additions that
this dataset specifically requires.

WHY THIS ONE NEEDS EXTRA CARE
-----------------------------
1. TinyFace and QMUL-SurvFace come from THE SAME LAB (Cheng, Zhu & Gong at
   QMUL). TinyFace is the degraded-condition benchmark this whole exercise is
   trying to improve. If the two share source imagery, training on SurvFace and
   reporting a TinyFace gain would be measuring memorisation. So the nearest
   neighbour is attributed back to WHICH eval set it came from, not just scored.

2. SurvFace images are native low-resolution surveillance crops; the five clean
   eval sets are high-quality portraits. Degraded probes produce systematically
   WEAKER embeddings, which compresses cosine similarity downward for everything
   including true matches. A fixed 0.40 threshold carried over from the CASIA
   audit is therefore a LOOSER filter here in real terms, not a stricter one.
   Counts are reported at several thresholds so that sensitivity is visible
   rather than hidden behind one number.

The gallery covers all seven evaluation sets: LFW, AgeDB-30, CFP-FP, CFP-FF,
CALFW, CPLFW and TinyFace.
```

## Protocol as implemented — `backend/scripts/qmul_overlap_control.py`

```text
CONTROL for the QMUL-SurvFace overlap audit.

    python backend/scripts/qmul_overlap_control.py

The audit reported 96.9% of QMUL identities above the 0.40 exclusion threshold,
with 78% of nearest neighbours landing in TinyFace. Taken at face value that
says the dataset is almost entirely contaminated.

That reading is probably WRONG, and this script exists to find out before any
decision is made on it.

THE COMPETING EXPLANATION
------------------------
ArcFace embeddings of very low quality faces are known to collapse toward a
common region of the hypersphere. A 26x21px blurred face carries little identity
signal, so what the embedding mostly encodes is "degraded face", not "this
person". Two unrelated degraded faces can then sit at cosine 0.6 purely because
both are degraded. QMUL is native surveillance capture and TinyFace is native
low-resolution capture, so a QMUL-to-TinyFace affinity is exactly what this
artefact would produce -- with no shared identities at all.

THE DISCRIMINATING TEST
-----------------------
Measure the IMPOSTOR floor within QMUL itself: similarity between images of
DIFFERENT QMUL identities. Ground truth is known here -- the dataset is ordered
by identity, so different directories are different people by construction.

  If different-person QMUL pairs also score ~0.5-0.7, then 0.5-0.7 is simply the
  noise floor for degraded imagery, the audit threshold is meaningless at that
  scale, and the "96.9% contamination" is an artefact.

  If different-person QMUL pairs score far lower (~0.1-0.2) while the nearest
  TinyFace neighbour scores 0.6+, the eval-set affinity is specific and the
  contamination is real.

A genuine-pair distribution (same identity, different images) is measured too,
to show where a true match actually sits under these conditions.
```

## Measurement — `runtime/benchmarks/exclusion_list.json`

### Values

| Field | Value |
|---|---|
| `train_set` | faces_webface_112x112 |
| `model` | w600k_r50 |
| `threshold` | 0.4 |
| `per_identity_sampled` | 10 |
| `images_sampled` | 105631 |
| `identities_total` | 10572 |
| `identities_excluded` | 692 |
| `identities_kept` | 9880 |
| `eval_sets[0]` | agedb_30 |
| `eval_sets[1]` | calfw |
| `eval_sets[2]` | cfp_fp |
| `eval_sets[3]` | cplfw |
| `eval_sets[4]` | lfw |
| `limitation` | Sampling. An identity whose sampled images happen not to resemble the eval shots is NOT excluded. This list is a floor: the true contaminated set is at least this large. |
| `excluded_labels[0]` | 2 |
| `excluded_labels[1]` | 3 |
| `excluded_labels[2]` | 6 |
| `excluded_labels[3]` | 13 |
| `excluded_labels[4]` | 19 |
| `excluded_labels[5]` | 22 |
| `excluded_labels[6]` | 24 |
| `excluded_labels[7]` | 31 |
| `excluded_labels[8]` | 33 |
| `excluded_labels[9]` | 34 |
| `excluded_labels[10]` | 35 |
| `excluded_labels[11]` | 37 |
| `excluded_labels[12]` | 41 |
| `excluded_labels[13]` | 43 |
| `excluded_labels[14]` | 52 |
| `excluded_labels[15]` | 57 |
| `excluded_labels[16]` | 60 |
| `excluded_labels[17]` | 83 |
| `excluded_labels[18]` | 84 |
| `excluded_labels[19]` | 88 |
| `excluded_labels[20]` | 91 |
| `excluded_labels[21]` | 93 |
| `excluded_labels[22]` | 95 |
| `excluded_labels[23]` | 97 |
| `excluded_labels[24]` | 101 |
| `excluded_labels[25]` | 105 |
| `excluded_labels[26]` | 116 |
| `excluded_labels[27]` | 120 |
| `excluded_labels[28]` | 133 |
| `excluded_labels[29]` | 136 |
| `excluded_labels[30]` | 137 |
| `excluded_labels[31]` | 139 |
| `excluded_labels[32]` | 142 |
| `excluded_labels[33]` | 147 |
| `excluded_labels[34]` | 155 |
| `excluded_labels[35]` | 171 |
| `excluded_labels[36]` | 176 |
| `excluded_labels[37]` | 193 |
| `excluded_labels[38]` | 194 |
| `excluded_labels[39]` | 203 |
| `excluded_labels[40]` | 209 |
| `excluded_labels[41]` | 226 |
| `excluded_labels[42]` | 228 |
| `excluded_labels[43]` | 230 |
| `excluded_labels[44]` | 234 |
| `excluded_labels[45]` | 236 |
| `excluded_labels[46]` | 256 |
| `excluded_labels[47]` | 258 |
| `excluded_labels[48]` | 264 |
| `excluded_labels[49]` | 277 |
| `excluded_labels[50]` | 287 |
| `excluded_labels[51]` | 298 |
| `excluded_labels[52]` | 318 |
| `excluded_labels[53]` | 361 |
| `excluded_labels[54]` | 366 |
| `excluded_labels[55]` | 367 |
| `excluded_labels[56]` | 370 |
| `excluded_labels[57]` | 379 |
| `excluded_labels[58]` | 384 |
| `excluded_labels[59]` | 389 |
| `excluded_labels[60]` | 391 |
| `excluded_labels[61]` | 399 |
| `excluded_labels[62]` | 406 |
| `excluded_labels[63]` | 407 |
| `excluded_labels[64]` | 420 |
| `excluded_labels[65]` | 427 |
| `excluded_labels[66]` | 428 |
| `excluded_labels[67]` | 434 |
| `excluded_labels[68]` | 451 |
| `excluded_labels[69]` | 469 |
| `excluded_labels[70]` | 474 |
| `excluded_labels[71]` | 476 |
| `excluded_labels[72]` | 484 |
| `excluded_labels[73]` | 486 |
| `excluded_labels[74]` | 490 |
| `excluded_labels[75]` | 491 |
| `excluded_labels[76]` | 494 |
| `excluded_labels[77]` | 496 |
| `excluded_labels[78]` | 501 |
| `excluded_labels[79]` | 516 |
| `excluded_labels[80]` | 519 |
| `excluded_labels[81]` | 524 |
| `excluded_labels[82]` | 534 |
| `excluded_labels[83]` | 553 |
| `excluded_labels[84]` | 572 |
| `excluded_labels[85]` | 576 |
| `excluded_labels[86]` | 582 |
| `excluded_labels[87]` | 587 |
| `excluded_labels[88]` | 599 |
| `excluded_labels[89]` | 608 |
| `excluded_labels[90]` | 618 |
| `excluded_labels[91]` | 622 |
| `excluded_labels[92]` | 633 |
| `excluded_labels[93]` | 637 |
| `excluded_labels[94]` | 655 |
| `excluded_labels[95]` | 660 |
| `excluded_labels[96]` | 674 |
| `excluded_labels[97]` | 694 |
| `excluded_labels[98]` | 700 |
| `excluded_labels[99]` | 705 |
| `excluded_labels[100]` | 707 |
| `excluded_labels[101]` | 717 |
| `excluded_labels[102]` | 718 |
| `excluded_labels[103]` | 719 |
| `excluded_labels[104]` | 729 |
| `excluded_labels[105]` | 732 |
| `excluded_labels[106]` | 733 |
| `excluded_labels[107]` | 734 |
| `excluded_labels[108]` | 738 |
| `excluded_labels[109]` | 744 |
| `excluded_labels[110]` | 754 |
| `excluded_labels[111]` | 773 |
| `excluded_labels[112]` | 797 |
| `excluded_labels[113]` | 804 |
| `excluded_labels[114]` | 812 |
| `excluded_labels[115]` | 819 |
| `excluded_labels[116]` | 837 |
| `excluded_labels[117]` | 871 |
| `excluded_labels[118]` | 904 |
| `excluded_labels[119]` | 912 |
| `excluded_labels[120]` | 922 |
| `excluded_labels[121]` | 996 |
| `excluded_labels[122]` | 1024 |
| `excluded_labels[123]` | 1027 |
| `excluded_labels[124]` | 1030 |
| `excluded_labels[125]` | 1039 |
| `excluded_labels[126]` | 1042 |
| `excluded_labels[127]` | 1045 |
| `excluded_labels[128]` | 1051 |
| `excluded_labels[129]` | 1063 |
| `excluded_labels[130]` | 1066 |
| `excluded_labels[131]` | 1071 |
| `excluded_labels[132]` | 1072 |
| `excluded_labels[133]` | 1076 |
| `excluded_labels[134]` | 1077 |
| `excluded_labels[135]` | 1082 |
| `excluded_labels[136]` | 1088 |
| `excluded_labels[137]` | 1092 |
| `excluded_labels[138]` | 1094 |
| `excluded_labels[139]` | 1119 |
| `excluded_labels[140]` | 1172 |
| `excluded_labels[141]` | 1183 |
| `excluded_labels[142]` | 1187 |
| `excluded_labels[143]` | 1199 |
| `excluded_labels[144]` | 1206 |
| `excluded_labels[145]` | 1219 |
| `excluded_labels[146]` | 1226 |
| `excluded_labels[147]` | 1248 |
| `excluded_labels[148]` | 1265 |
| `excluded_labels[149]` | 1279 |
| `excluded_labels[150]` | 1290 |
| `excluded_labels[151]` | 1303 |
| `excluded_labels[152]` | 1314 |
| `excluded_labels[153]` | 1321 |
| `excluded_labels[154]` | 1329 |
| `excluded_labels[155]` | 1350 |
| `excluded_labels[156]` | 1375 |
| `excluded_labels[157]` | 1393 |
| `excluded_labels[158]` | 1407 |
| `excluded_labels[159]` | 1408 |
| `excluded_labels[160]` | 1412 |
| `excluded_labels[161]` | 1425 |
| `excluded_labels[162]` | 1432 |
| `excluded_labels[163]` | 1454 |
| `excluded_labels[164]` | 1460 |
| `excluded_labels[165]` | 1467 |
| `excluded_labels[166]` | 1469 |
| `excluded_labels[167]` | 1498 |
| `excluded_labels[168]` | 1547 |
| `excluded_labels[169]` | 1556 |
| `excluded_labels[170]` | 1558 |
| `excluded_labels[171]` | 1581 |
| `excluded_labels[172]` | 1587 |
| `excluded_labels[173]` | 1595 |
| `excluded_labels[174]` | 1597 |
| `excluded_labels[175]` | 1617 |
| `excluded_labels[176]` | 1646 |
| `excluded_labels[177]` | 1661 |
| `excluded_labels[178]` | 1662 |
| `excluded_labels[179]` | 1676 |
| `excluded_labels[180]` | 1688 |
| `excluded_labels[181]` | 1698 |
| `excluded_labels[182]` | 1699 |
| `excluded_labels[183]` | 1705 |
| `excluded_labels[184]` | 1758 |
| `excluded_labels[185]` | 1768 |
| `excluded_labels[186]` | 1786 |
| `excluded_labels[187]` | 1816 |
| `excluded_labels[188]` | 1832 |
| `excluded_labels[189]` | 1841 |
| `excluded_labels[190]` | 1851 |
| `excluded_labels[191]` | 1858 |
| `excluded_labels[192]` | 1888 |
| `excluded_labels[193]` | 1897 |
| `excluded_labels[194]` | 1925 |
| `excluded_labels[195]` | 1937 |
| `excluded_labels[196]` | 1945 |
| `excluded_labels[197]` | 1948 |
| `excluded_labels[198]` | 1952 |
| `excluded_labels[199]` | 1953 |
| `excluded_labels[200]` | 1969 |
| `excluded_labels[201]` | 1971 |
| `excluded_labels[202]` | 1987 |
| `excluded_labels[203]` | 1994 |
| `excluded_labels[204]` | 2008 |
| `excluded_labels[205]` | 2013 |
| `excluded_labels[206]` | 2023 |
| `excluded_labels[207]` | 2027 |
| `excluded_labels[208]` | 2043 |
| `excluded_labels[209]` | 2075 |
| `excluded_labels[210]` | 2077 |
| `excluded_labels[211]` | 2110 |
| `excluded_labels[212]` | 2118 |
| `excluded_labels[213]` | 2120 |
| `excluded_labels[214]` | 2134 |
| `excluded_labels[215]` | 2140 |
| `excluded_labels[216]` | 2144 |
| `excluded_labels[217]` | 2152 |
| `excluded_labels[218]` | 2171 |
| `excluded_labels[219]` | 2173 |
| `excluded_labels[220]` | 2180 |
| `excluded_labels[221]` | 2193 |
| `excluded_labels[222]` | 2216 |
| `excluded_labels[223]` | 2218 |
| `excluded_labels[224]` | 2224 |
| `excluded_labels[225]` | 2253 |
| `excluded_labels[226]` | 2259 |
| `excluded_labels[227]` | 2300 |
| `excluded_labels[228]` | 2302 |
| `excluded_labels[229]` | 2312 |
| `excluded_labels[230]` | 2315 |
| `excluded_labels[231]` | 2328 |
| `excluded_labels[232]` | 2330 |
| `excluded_labels[233]` | 2336 |
| `excluded_labels[234]` | 2370 |
| `excluded_labels[235]` | 2375 |
| `excluded_labels[236]` | 2446 |
| `excluded_labels[237]` | 2465 |
| `excluded_labels[238]` | 2485 |
| `excluded_labels[239]` | 2496 |
| `excluded_labels[240]` | 2524 |
| `excluded_labels[241]` | 2536 |
| `excluded_labels[242]` | 2553 |
| `excluded_labels[243]` | 2584 |
| `excluded_labels[244]` | 2596 |
| `excluded_labels[245]` | 2614 |
| `excluded_labels[246]` | 2615 |
| `excluded_labels[247]` | 2621 |
| `excluded_labels[248]` | 2623 |
| `excluded_labels[249]` | 2629 |
| `excluded_labels[250]` | 2665 |
| `excluded_labels[251]` | 2690 |
| `excluded_labels[252]` | 2692 |
| `excluded_labels[253]` | 2729 |
| `excluded_labels[254]` | 2759 |
| `excluded_labels[255]` | 2769 |
| `excluded_labels[256]` | 2798 |
| `excluded_labels[257]` | 2805 |
| `excluded_labels[258]` | 2807 |
| `excluded_labels[259]` | 2812 |
| `excluded_labels[260]` | 2822 |
| `excluded_labels[261]` | 2825 |
| `excluded_labels[262]` | 2827 |
| `excluded_labels[263]` | 2838 |
| `excluded_labels[264]` | 2845 |
| `excluded_labels[265]` | 2866 |
| `excluded_labels[266]` | 2869 |
| `excluded_labels[267]` | 2875 |
| `excluded_labels[268]` | 2884 |
| `excluded_labels[269]` | 2896 |
| `excluded_labels[270]` | 2900 |
| `excluded_labels[271]` | 2902 |
| `excluded_labels[272]` | 2918 |
| `excluded_labels[273]` | 2948 |
| `excluded_labels[274]` | 2953 |
| `excluded_labels[275]` | 2959 |
| `excluded_labels[276]` | 2967 |
| `excluded_labels[277]` | 2986 |
| `excluded_labels[278]` | 2990 |
| `excluded_labels[279]` | 2991 |
| `excluded_labels[280]` | 3006 |
| `excluded_labels[281]` | 3010 |
| `excluded_labels[282]` | 3014 |
| `excluded_labels[283]` | 3021 |
| `excluded_labels[284]` | 3024 |
| `excluded_labels[285]` | 3036 |
| `excluded_labels[286]` | 3038 |
| `excluded_labels[287]` | 3053 |
| `excluded_labels[288]` | 3071 |
| `excluded_labels[289]` | 3096 |
| `excluded_labels[290]` | 3126 |
| `excluded_labels[291]` | 3145 |
| `excluded_labels[292]` | 3148 |
| `excluded_labels[293]` | 3152 |
| `excluded_labels[294]` | 3159 |
| `excluded_labels[295]` | 3162 |
| `excluded_labels[296]` | 3165 |
| `excluded_labels[297]` | 3167 |
| `excluded_labels[298]` | 3170 |
| `excluded_labels[299]` | 3192 |
| `excluded_labels[300]` | 3195 |
| `excluded_labels[301]` | 3220 |
| `excluded_labels[302]` | 3253 |
| `excluded_labels[303]` | 3268 |
| `excluded_labels[304]` | 3269 |
| `excluded_labels[305]` | 3270 |
| `excluded_labels[306]` | 3273 |
| `excluded_labels[307]` | 3274 |
| `excluded_labels[308]` | 3315 |
| `excluded_labels[309]` | 3324 |
| `excluded_labels[310]` | 3332 |
| `excluded_labels[311]` | 3347 |
| `excluded_labels[312]` | 3351 |
| `excluded_labels[313]` | 3353 |
| `excluded_labels[314]` | 3356 |
| `excluded_labels[315]` | 3365 |
| `excluded_labels[316]` | 3372 |
| `excluded_labels[317]` | 3376 |
| `excluded_labels[318]` | 3381 |
| `excluded_labels[319]` | 3399 |
| `excluded_labels[320]` | 3403 |
| `excluded_labels[321]` | 3410 |
| `excluded_labels[322]` | 3414 |
| `excluded_labels[323]` | 3446 |
| `excluded_labels[324]` | 3448 |
| `excluded_labels[325]` | 3453 |
| `excluded_labels[326]` | 3492 |
| `excluded_labels[327]` | 3507 |
| `excluded_labels[328]` | 3515 |
| `excluded_labels[329]` | 3578 |
| `excluded_labels[330]` | 3585 |
| `excluded_labels[331]` | 3597 |
| `excluded_labels[332]` | 3632 |
| `excluded_labels[333]` | 3639 |
| `excluded_labels[334]` | 3644 |
| `excluded_labels[335]` | 3645 |
| `excluded_labels[336]` | 3662 |
| `excluded_labels[337]` | 3672 |
| `excluded_labels[338]` | 3690 |
| `excluded_labels[339]` | 3702 |
| `excluded_labels[340]` | 3705 |
| `excluded_labels[341]` | 3708 |
| `excluded_labels[342]` | 3726 |
| `excluded_labels[343]` | 3732 |
| `excluded_labels[344]` | 3746 |
| `excluded_labels[345]` | 3769 |
| `excluded_labels[346]` | 3787 |
| `excluded_labels[347]` | 3794 |
| `excluded_labels[348]` | 3796 |
| `excluded_labels[349]` | 3797 |
| `excluded_labels[350]` | 3830 |
| `excluded_labels[351]` | 3838 |
| `excluded_labels[352]` | 3845 |
| `excluded_labels[353]` | 3848 |
| `excluded_labels[354]` | 3879 |
| `excluded_labels[355]` | 3907 |
| `excluded_labels[356]` | 3914 |
| `excluded_labels[357]` | 3920 |
| `excluded_labels[358]` | 3922 |
| `excluded_labels[359]` | 3924 |
| `excluded_labels[360]` | 3928 |
| `excluded_labels[361]` | 3931 |
| `excluded_labels[362]` | 3936 |
| `excluded_labels[363]` | 3938 |
| `excluded_labels[364]` | 3941 |
| `excluded_labels[365]` | 3965 |
| `excluded_labels[366]` | 4021 |
| `excluded_labels[367]` | 4025 |
| `excluded_labels[368]` | 4049 |
| `excluded_labels[369]` | 4087 |
| `excluded_labels[370]` | 4088 |
| `excluded_labels[371]` | 4092 |
| `excluded_labels[372]` | 4139 |
| `excluded_labels[373]` | 4142 |
| `excluded_labels[374]` | 4146 |
| `excluded_labels[375]` | 4148 |
| `excluded_labels[376]` | 4161 |
| `excluded_labels[377]` | 4164 |
| `excluded_labels[378]` | 4172 |
| `excluded_labels[379]` | 4173 |
| `excluded_labels[380]` | 4177 |
| `excluded_labels[381]` | 4183 |
| `excluded_labels[382]` | 4207 |
| `excluded_labels[383]` | 4264 |
| `excluded_labels[384]` | 4268 |
| `excluded_labels[385]` | 4294 |
| `excluded_labels[386]` | 4311 |
| `excluded_labels[387]` | 4314 |
| `excluded_labels[388]` | 4324 |
| `excluded_labels[389]` | 4328 |
| `excluded_labels[390]` | 4339 |
| `excluded_labels[391]` | 4348 |
| `excluded_labels[392]` | 4360 |
| `excluded_labels[393]` | 4379 |
| `excluded_labels[394]` | 4430 |
| `excluded_labels[395]` | 4447 |
| `excluded_labels[396]` | 4468 |
| `excluded_labels[397]` | 4469 |
| `excluded_labels[398]` | 4479 |
| `excluded_labels[399]` | 4525 |
| `excluded_labels[400]` | 4527 |
| `excluded_labels[401]` | 4532 |
| `excluded_labels[402]` | 4557 |
| `excluded_labels[403]` | 4621 |
| `excluded_labels[404]` | 4630 |
| `excluded_labels[405]` | 4658 |
| `excluded_labels[406]` | 4662 |
| `excluded_labels[407]` | 4664 |
| `excluded_labels[408]` | 4670 |
| `excluded_labels[409]` | 4695 |
| `excluded_labels[410]` | 4712 |
| `excluded_labels[411]` | 4719 |
| `excluded_labels[412]` | 4728 |
| `excluded_labels[413]` | 4730 |
| `excluded_labels[414]` | 4753 |
| `excluded_labels[415]` | 4767 |
| `excluded_labels[416]` | 4780 |
| `excluded_labels[417]` | 4787 |
| `excluded_labels[418]` | 4805 |
| `excluded_labels[419]` | 4808 |
| `excluded_labels[420]` | 4839 |
| `excluded_labels[421]` | 4842 |
| `excluded_labels[422]` | 4853 |
| `excluded_labels[423]` | 4863 |
| `excluded_labels[424]` | 4866 |
| `excluded_labels[425]` | 4871 |
| `excluded_labels[426]` | 4878 |
| `excluded_labels[427]` | 4883 |
| `excluded_labels[428]` | 4888 |
| `excluded_labels[429]` | 4909 |
| `excluded_labels[430]` | 4911 |
| `excluded_labels[431]` | 4929 |
| `excluded_labels[432]` | 4951 |
| `excluded_labels[433]` | 4955 |
| `excluded_labels[434]` | 4964 |
| `excluded_labels[435]` | 4977 |
| `excluded_labels[436]` | 4980 |
| `excluded_labels[437]` | 4998 |
| `excluded_labels[438]` | 5049 |
| `excluded_labels[439]` | 5050 |
| `excluded_labels[440]` | 5065 |
| `excluded_labels[441]` | 5075 |
| `excluded_labels[442]` | 5078 |
| `excluded_labels[443]` | 5088 |
| `excluded_labels[444]` | 5096 |
| `excluded_labels[445]` | 5097 |
| `excluded_labels[446]` | 5100 |
| `excluded_labels[447]` | 5112 |
| `excluded_labels[448]` | 5115 |
| `excluded_labels[449]` | 5122 |
| `excluded_labels[450]` | 5142 |
| `excluded_labels[451]` | 5148 |
| `excluded_labels[452]` | 5152 |
| `excluded_labels[453]` | 5159 |
| `excluded_labels[454]` | 5177 |
| `excluded_labels[455]` | 5189 |
| `excluded_labels[456]` | 5272 |
| `excluded_labels[457]` | 5303 |
| `excluded_labels[458]` | 5308 |
| `excluded_labels[459]` | 5311 |
| `excluded_labels[460]` | 5318 |
| `excluded_labels[461]` | 5324 |
| `excluded_labels[462]` | 5357 |
| `excluded_labels[463]` | 5381 |
| `excluded_labels[464]` | 5387 |
| `excluded_labels[465]` | 5390 |
| `excluded_labels[466]` | 5417 |
| `excluded_labels[467]` | 5432 |
| `excluded_labels[468]` | 5459 |
| `excluded_labels[469]` | 5479 |
| `excluded_labels[470]` | 5500 |
| `excluded_labels[471]` | 5506 |
| `excluded_labels[472]` | 5511 |
| `excluded_labels[473]` | 5512 |
| `excluded_labels[474]` | 5517 |
| `excluded_labels[475]` | 5520 |
| `excluded_labels[476]` | 5522 |
| `excluded_labels[477]` | 5525 |
| `excluded_labels[478]` | 5559 |
| `excluded_labels[479]` | 5570 |
| `excluded_labels[480]` | 5578 |
| `excluded_labels[481]` | 5602 |
| `excluded_labels[482]` | 5624 |
| `excluded_labels[483]` | 5665 |
| `excluded_labels[484]` | 5676 |
| `excluded_labels[485]` | 5682 |
| `excluded_labels[486]` | 5687 |
| `excluded_labels[487]` | 5690 |
| `excluded_labels[488]` | 5718 |
| `excluded_labels[489]` | 5828 |
| `excluded_labels[490]` | 5830 |
| `excluded_labels[491]` | 5843 |
| `excluded_labels[492]` | 5846 |
| `excluded_labels[493]` | 5862 |
| `excluded_labels[494]` | 5873 |
| `excluded_labels[495]` | 5897 |
| `excluded_labels[496]` | 5900 |
| `excluded_labels[497]` | 5925 |
| `excluded_labels[498]` | 5966 |
| `excluded_labels[499]` | 5973 |
| `excluded_labels[500]` | 5984 |
| `excluded_labels[501]` | 6048 |
| `excluded_labels[502]` | 6049 |
| `excluded_labels[503]` | 6058 |
| `excluded_labels[504]` | 6061 |
| `excluded_labels[505]` | 6063 |
| `excluded_labels[506]` | 6065 |
| `excluded_labels[507]` | 6107 |
| `excluded_labels[508]` | 6146 |
| `excluded_labels[509]` | 6161 |
| `excluded_labels[510]` | 6163 |
| `excluded_labels[511]` | 6189 |
| `excluded_labels[512]` | 6204 |
| `excluded_labels[513]` | 6208 |
| `excluded_labels[514]` | 6265 |
| `excluded_labels[515]` | 6280 |
| `excluded_labels[516]` | 6297 |
| `excluded_labels[517]` | 6323 |
| `excluded_labels[518]` | 6328 |
| `excluded_labels[519]` | 6332 |
| `excluded_labels[520]` | 6349 |
| `excluded_labels[521]` | 6384 |
| `excluded_labels[522]` | 6393 |
| `excluded_labels[523]` | 6396 |
| `excluded_labels[524]` | 6458 |
| `excluded_labels[525]` | 6485 |
| `excluded_labels[526]` | 6487 |
| `excluded_labels[527]` | 6558 |
| `excluded_labels[528]` | 6561 |
| `excluded_labels[529]` | 6578 |
| `excluded_labels[530]` | 6579 |
| `excluded_labels[531]` | 6582 |
| `excluded_labels[532]` | 6595 |
| `excluded_labels[533]` | 6621 |
| `excluded_labels[534]` | 6634 |
| `excluded_labels[535]` | 6662 |
| `excluded_labels[536]` | 6707 |
| `excluded_labels[537]` | 6727 |
| `excluded_labels[538]` | 6743 |
| `excluded_labels[539]` | 6759 |
| `excluded_labels[540]` | 6764 |
| `excluded_labels[541]` | 6815 |
| `excluded_labels[542]` | 6817 |
| `excluded_labels[543]` | 6830 |
| `excluded_labels[544]` | 6845 |
| `excluded_labels[545]` | 6901 |
| `excluded_labels[546]` | 6948 |
| `excluded_labels[547]` | 6961 |
| `excluded_labels[548]` | 6962 |
| `excluded_labels[549]` | 6994 |
| `excluded_labels[550]` | 7004 |
| `excluded_labels[551]` | 7031 |
| `excluded_labels[552]` | 7062 |
| `excluded_labels[553]` | 7075 |
| `excluded_labels[554]` | 7103 |
| `excluded_labels[555]` | 7116 |
| `excluded_labels[556]` | 7139 |
| `excluded_labels[557]` | 7142 |
| `excluded_labels[558]` | 7143 |
| `excluded_labels[559]` | 7161 |
| `excluded_labels[560]` | 7162 |
| `excluded_labels[561]` | 7164 |
| `excluded_labels[562]` | 7234 |
| `excluded_labels[563]` | 7311 |
| `excluded_labels[564]` | 7359 |
| `excluded_labels[565]` | 7394 |
| `excluded_labels[566]` | 7439 |
| `excluded_labels[567]` | 7451 |
| `excluded_labels[568]` | 7470 |
| `excluded_labels[569]` | 7484 |
| `excluded_labels[570]` | 7494 |
| `excluded_labels[571]` | 7509 |
| `excluded_labels[572]` | 7512 |
| `excluded_labels[573]` | 7516 |
| `excluded_labels[574]` | 7534 |
| `excluded_labels[575]` | 7562 |
| `excluded_labels[576]` | 7566 |
| `excluded_labels[577]` | 7569 |
| `excluded_labels[578]` | 7597 |
| `excluded_labels[579]` | 7601 |
| `excluded_labels[580]` | 7603 |
| `excluded_labels[581]` | 7675 |
| `excluded_labels[582]` | 7724 |
| `excluded_labels[583]` | 7770 |
| `excluded_labels[584]` | 7775 |
| `excluded_labels[585]` | 7792 |
| `excluded_labels[586]` | 7818 |
| `excluded_labels[587]` | 7881 |
| `excluded_labels[588]` | 7892 |
| `excluded_labels[589]` | 7902 |
| `excluded_labels[590]` | 7948 |
| `excluded_labels[591]` | 7952 |
| `excluded_labels[592]` | 7969 |
| `excluded_labels[593]` | 7997 |
| `excluded_labels[594]` | 8007 |
| `excluded_labels[595]` | 8031 |
| `excluded_labels[596]` | 8055 |
| `excluded_labels[597]` | 8068 |
| `excluded_labels[598]` | 8071 |
| `excluded_labels[599]` | 8100 |
| `excluded_labels[600]` | 8104 |
| `excluded_labels[601]` | 8119 |
| `excluded_labels[602]` | 8131 |
| `excluded_labels[603]` | 8143 |
| `excluded_labels[604]` | 8171 |
| `excluded_labels[605]` | 8222 |
| `excluded_labels[606]` | 8228 |
| `excluded_labels[607]` | 8247 |
| `excluded_labels[608]` | 8277 |
| `excluded_labels[609]` | 8297 |
| `excluded_labels[610]` | 8313 |
| `excluded_labels[611]` | 8319 |
| `excluded_labels[612]` | 8330 |
| `excluded_labels[613]` | 8340 |
| `excluded_labels[614]` | 8403 |
| `excluded_labels[615]` | 8405 |
| `excluded_labels[616]` | 8422 |
| `excluded_labels[617]` | 8431 |
| `excluded_labels[618]` | 8459 |
| `excluded_labels[619]` | 8477 |
| `excluded_labels[620]` | 8536 |
| `excluded_labels[621]` | 8554 |
| `excluded_labels[622]` | 8612 |
| `excluded_labels[623]` | 8627 |
| `excluded_labels[624]` | 8634 |
| `excluded_labels[625]` | 8693 |
| `excluded_labels[626]` | 8700 |
| `excluded_labels[627]` | 8702 |
| `excluded_labels[628]` | 8724 |
| `excluded_labels[629]` | 8781 |
| `excluded_labels[630]` | 8819 |
| `excluded_labels[631]` | 8857 |
| `excluded_labels[632]` | 8890 |
| `excluded_labels[633]` | 8942 |
| `excluded_labels[634]` | 8947 |
| `excluded_labels[635]` | 8982 |
| `excluded_labels[636]` | 9027 |
| `excluded_labels[637]` | 9094 |
| `excluded_labels[638]` | 9095 |
| `excluded_labels[639]` | 9113 |
| `excluded_labels[640]` | 9180 |
| `excluded_labels[641]` | 9190 |
| `excluded_labels[642]` | 9237 |
| `excluded_labels[643]` | 9251 |
| `excluded_labels[644]` | 9325 |
| `excluded_labels[645]` | 9353 |
| `excluded_labels[646]` | 9369 |
| `excluded_labels[647]` | 9370 |
| `excluded_labels[648]` | 9392 |
| `excluded_labels[649]` | 9401 |
| `excluded_labels[650]` | 9416 |
| `excluded_labels[651]` | 9434 |
| `excluded_labels[652]` | 9456 |
| `excluded_labels[653]` | 9463 |
| `excluded_labels[654]` | 9481 |
| `excluded_labels[655]` | 9517 |
| `excluded_labels[656]` | 9532 |
| `excluded_labels[657]` | 9540 |
| `excluded_labels[658]` | 9551 |
| `excluded_labels[659]` | 9570 |
| `excluded_labels[660]` | 9611 |
| `excluded_labels[661]` | 9614 |
| `excluded_labels[662]` | 9649 |
| `excluded_labels[663]` | 9825 |
| `excluded_labels[664]` | 9830 |
| `excluded_labels[665]` | 9833 |
| `excluded_labels[666]` | 9926 |
| `excluded_labels[667]` | 9941 |
| `excluded_labels[668]` | 9969 |
| `excluded_labels[669]` | 9990 |
| `excluded_labels[670]` | 10009 |
| `excluded_labels[671]` | 10048 |
| `excluded_labels[672]` | 10071 |
| `excluded_labels[673]` | 10072 |
| `excluded_labels[674]` | 10080 |
| `excluded_labels[675]` | 10088 |
| `excluded_labels[676]` | 10098 |
| `excluded_labels[677]` | 10123 |
| `excluded_labels[678]` | 10189 |
| `excluded_labels[679]` | 10209 |
| `excluded_labels[680]` | 10237 |
| `excluded_labels[681]` | 10245 |
| `excluded_labels[682]` | 10265 |
| `excluded_labels[683]` | 10321 |
| `excluded_labels[684]` | 10363 |
| `excluded_labels[685]` | 10377 |
| `excluded_labels[686]` | 10382 |
| `excluded_labels[687]` | 10385 |
| `excluded_labels[688]` | 10407 |
| `excluded_labels[689]` | 10480 |
| `excluded_labels[690]` | 10499 |
| `excluded_labels[691]` | 10533 |
| `kept_labels[0]` | 0 |
| `kept_labels[1]` | 1 |
| `kept_labels[2]` | 4 |
| `kept_labels[3]` | 5 |
| `kept_labels[4]` | 7 |
| `kept_labels[5]` | 8 |
| `kept_labels[6]` | 9 |
| `kept_labels[7]` | 10 |
| `kept_labels[8]` | 11 |
| `kept_labels[9]` | 12 |
| `kept_labels[10]` | 14 |
| `kept_labels[11]` | 15 |
| `kept_labels[12]` | 16 |
| `kept_labels[13]` | 17 |
| `kept_labels[14]` | 18 |
| `kept_labels[15]` | 20 |
| `kept_labels[16]` | 21 |
| `kept_labels[17]` | 23 |
| `kept_labels[18]` | 25 |
| `kept_labels[19]` | 26 |
| `kept_labels[20]` | 27 |
| `kept_labels[21]` | 28 |
| `kept_labels[22]` | 29 |
| `kept_labels[23]` | 30 |
| `kept_labels[24]` | 32 |
| `kept_labels[25]` | 36 |
| `kept_labels[26]` | 38 |
| `kept_labels[27]` | 39 |
| `kept_labels[28]` | 40 |
| `kept_labels[29]` | 42 |
| `kept_labels[30]` | 44 |
| `kept_labels[31]` | 45 |
| `kept_labels[32]` | 46 |
| `kept_labels[33]` | 47 |
| `kept_labels[34]` | 48 |
| `kept_labels[35]` | 49 |
| `kept_labels[36]` | 50 |
| `kept_labels[37]` | 51 |
| `kept_labels[38]` | 53 |
| `kept_labels[39]` | 54 |
| `kept_labels[40]` | 55 |
| `kept_labels[41]` | 56 |
| `kept_labels[42]` | 58 |
| `kept_labels[43]` | 59 |
| `kept_labels[44]` | 61 |
| `kept_labels[45]` | 62 |
| `kept_labels[46]` | 63 |
| `kept_labels[47]` | 64 |
| `kept_labels[48]` | 65 |
| `kept_labels[49]` | 66 |
| `kept_labels[50]` | 67 |
| `kept_labels[51]` | 68 |
| `kept_labels[52]` | 69 |
| `kept_labels[53]` | 70 |
| `kept_labels[54]` | 71 |
| `kept_labels[55]` | 72 |
| `kept_labels[56]` | 73 |
| `kept_labels[57]` | 74 |
| `kept_labels[58]` | 75 |
| `kept_labels[59]` | 76 |
| `kept_labels[60]` | 77 |
| `kept_labels[61]` | 78 |
| `kept_labels[62]` | 79 |
| `kept_labels[63]` | 80 |
| `kept_labels[64]` | 81 |
| `kept_labels[65]` | 82 |
| `kept_labels[66]` | 85 |
| `kept_labels[67]` | 86 |
| `kept_labels[68]` | 87 |
| `kept_labels[69]` | 89 |
| `kept_labels[70]` | 90 |
| `kept_labels[71]` | 92 |
| `kept_labels[72]` | 94 |
| `kept_labels[73]` | 96 |
| `kept_labels[74]` | 98 |
| `kept_labels[75]` | 99 |
| `kept_labels[76]` | 100 |
| `kept_labels[77]` | 102 |
| `kept_labels[78]` | 103 |
| `kept_labels[79]` | 104 |
| `kept_labels[80]` | 106 |
| `kept_labels[81]` | 107 |
| `kept_labels[82]` | 108 |
| `kept_labels[83]` | 109 |
| `kept_labels[84]` | 110 |
| `kept_labels[85]` | 111 |
| `kept_labels[86]` | 112 |
| `kept_labels[87]` | 113 |
| `kept_labels[88]` | 114 |
| `kept_labels[89]` | 115 |
| `kept_labels[90]` | 117 |
| `kept_labels[91]` | 118 |
| `kept_labels[92]` | 119 |
| `kept_labels[93]` | 121 |
| `kept_labels[94]` | 122 |
| `kept_labels[95]` | 123 |
| `kept_labels[96]` | 124 |
| `kept_labels[97]` | 125 |
| `kept_labels[98]` | 126 |
| `kept_labels[99]` | 127 |
| `kept_labels[100]` | 128 |
| `kept_labels[101]` | 129 |
| `kept_labels[102]` | 130 |
| `kept_labels[103]` | 131 |
| `kept_labels[104]` | 132 |
| `kept_labels[105]` | 134 |
| `kept_labels[106]` | 135 |
| `kept_labels[107]` | 138 |
| `kept_labels[108]` | 140 |
| `kept_labels[109]` | 141 |
| `kept_labels[110]` | 143 |
| `kept_labels[111]` | 144 |
| `kept_labels[112]` | 145 |
| `kept_labels[113]` | 146 |
| `kept_labels[114]` | 148 |
| `kept_labels[115]` | 149 |
| `kept_labels[116]` | 150 |
| `kept_labels[117]` | 151 |
| `kept_labels[118]` | 152 |
| `kept_labels[119]` | 153 |
| `kept_labels[120]` | 154 |
| `kept_labels[121]` | 156 |
| `kept_labels[122]` | 157 |
| `kept_labels[123]` | 158 |
| `kept_labels[124]` | 159 |
| `kept_labels[125]` | 160 |
| `kept_labels[126]` | 161 |
| `kept_labels[127]` | 162 |
| `kept_labels[128]` | 163 |
| `kept_labels[129]` | 164 |
| `kept_labels[130]` | 165 |
| `kept_labels[131]` | 166 |
| `kept_labels[132]` | 167 |
| `kept_labels[133]` | 168 |
| `kept_labels[134]` | 169 |
| `kept_labels[135]` | 170 |
| `kept_labels[136]` | 172 |
| `kept_labels[137]` | 173 |
| `kept_labels[138]` | 174 |
| `kept_labels[139]` | 175 |
| `kept_labels[140]` | 177 |
| `kept_labels[141]` | 178 |
| `kept_labels[142]` | 179 |
| `kept_labels[143]` | 180 |
| `kept_labels[144]` | 181 |
| `kept_labels[145]` | 182 |
| `kept_labels[146]` | 183 |
| `kept_labels[147]` | 184 |
| `kept_labels[148]` | 185 |
| `kept_labels[149]` | 186 |
| `kept_labels[150]` | 187 |
| `kept_labels[151]` | 188 |
| `kept_labels[152]` | 189 |
| `kept_labels[153]` | 190 |
| `kept_labels[154]` | 191 |
| `kept_labels[155]` | 192 |
| `kept_labels[156]` | 195 |
| `kept_labels[157]` | 196 |
| `kept_labels[158]` | 197 |
| `kept_labels[159]` | 198 |
| `kept_labels[160]` | 199 |
| `kept_labels[161]` | 200 |
| `kept_labels[162]` | 201 |
| `kept_labels[163]` | 202 |
| `kept_labels[164]` | 204 |
| `kept_labels[165]` | 205 |
| `kept_labels[166]` | 206 |
| `kept_labels[167]` | 207 |
| `kept_labels[168]` | 208 |
| `kept_labels[169]` | 210 |
| `kept_labels[170]` | 211 |
| `kept_labels[171]` | 212 |
| `kept_labels[172]` | 213 |
| `kept_labels[173]` | 214 |
| `kept_labels[174]` | 215 |
| `kept_labels[175]` | 216 |
| `kept_labels[176]` | 217 |
| `kept_labels[177]` | 218 |
| `kept_labels[178]` | 219 |
| `kept_labels[179]` | 220 |
| `kept_labels[180]` | 221 |
| `kept_labels[181]` | 222 |
| `kept_labels[182]` | 223 |
| `kept_labels[183]` | 224 |
| `kept_labels[184]` | 225 |
| `kept_labels[185]` | 227 |
| `kept_labels[186]` | 229 |
| `kept_labels[187]` | 231 |
| `kept_labels[188]` | 232 |
| `kept_labels[189]` | 233 |
| `kept_labels[190]` | 235 |
| `kept_labels[191]` | 237 |
| `kept_labels[192]` | 238 |
| `kept_labels[193]` | 239 |
| `kept_labels[194]` | 240 |
| `kept_labels[195]` | 241 |
| `kept_labels[196]` | 242 |
| `kept_labels[197]` | 243 |
| `kept_labels[198]` | 244 |
| `kept_labels[199]` | 245 |
| `kept_labels[200]` | 246 |
| `kept_labels[201]` | 247 |
| `kept_labels[202]` | 248 |
| `kept_labels[203]` | 249 |
| `kept_labels[204]` | 250 |
| `kept_labels[205]` | 251 |
| `kept_labels[206]` | 252 |
| `kept_labels[207]` | 253 |
| `kept_labels[208]` | 254 |
| `kept_labels[209]` | 255 |
| `kept_labels[210]` | 257 |
| `kept_labels[211]` | 259 |
| `kept_labels[212]` | 260 |
| `kept_labels[213]` | 261 |
| `kept_labels[214]` | 262 |
| `kept_labels[215]` | 263 |
| `kept_labels[216]` | 265 |
| `kept_labels[217]` | 266 |
| `kept_labels[218]` | 267 |
| `kept_labels[219]` | 268 |
| `kept_labels[220]` | 269 |
| `kept_labels[221]` | 270 |
| `kept_labels[222]` | 271 |
| `kept_labels[223]` | 272 |
| `kept_labels[224]` | 273 |
| `kept_labels[225]` | 274 |
| `kept_labels[226]` | 275 |
| `kept_labels[227]` | 276 |
| `kept_labels[228]` | 278 |
| `kept_labels[229]` | 279 |
| `kept_labels[230]` | 280 |
| `kept_labels[231]` | 281 |
| `kept_labels[232]` | 282 |
| `kept_labels[233]` | 283 |
| `kept_labels[234]` | 284 |
| `kept_labels[235]` | 285 |
| `kept_labels[236]` | 286 |
| `kept_labels[237]` | 288 |
| `kept_labels[238]` | 289 |
| `kept_labels[239]` | 290 |
| `kept_labels[240]` | 291 |
| `kept_labels[241]` | 292 |
| `kept_labels[242]` | 293 |
| `kept_labels[243]` | 294 |
| `kept_labels[244]` | 295 |
| `kept_labels[245]` | 296 |
| `kept_labels[246]` | 297 |
| `kept_labels[247]` | 299 |
| `kept_labels[248]` | 300 |
| `kept_labels[249]` | 301 |
| `kept_labels[250]` | 302 |
| `kept_labels[251]` | 303 |
| `kept_labels[252]` | 304 |
| `kept_labels[253]` | 305 |
| `kept_labels[254]` | 306 |
| `kept_labels[255]` | 307 |
| `kept_labels[256]` | 308 |
| `kept_labels[257]` | 309 |
| `kept_labels[258]` | 310 |
| `kept_labels[259]` | 311 |
| `kept_labels[260]` | 312 |
| `kept_labels[261]` | 313 |
| `kept_labels[262]` | 314 |
| `kept_labels[263]` | 315 |
| `kept_labels[264]` | 316 |
| `kept_labels[265]` | 317 |
| `kept_labels[266]` | 319 |
| `kept_labels[267]` | 320 |
| `kept_labels[268]` | 321 |
| `kept_labels[269]` | 322 |
| `kept_labels[270]` | 323 |
| `kept_labels[271]` | 324 |
| `kept_labels[272]` | 325 |
| `kept_labels[273]` | 326 |
| `kept_labels[274]` | 327 |
| `kept_labels[275]` | 328 |
| `kept_labels[276]` | 329 |
| `kept_labels[277]` | 330 |
| `kept_labels[278]` | 331 |
| `kept_labels[279]` | 332 |
| `kept_labels[280]` | 333 |
| `kept_labels[281]` | 334 |
| `kept_labels[282]` | 335 |
| `kept_labels[283]` | 336 |
| `kept_labels[284]` | 337 |
| `kept_labels[285]` | 338 |
| `kept_labels[286]` | 339 |
| `kept_labels[287]` | 340 |
| `kept_labels[288]` | 341 |
| `kept_labels[289]` | 342 |
| `kept_labels[290]` | 343 |
| `kept_labels[291]` | 344 |
| `kept_labels[292]` | 345 |
| `kept_labels[293]` | 346 |
| `kept_labels[294]` | 347 |
| `kept_labels[295]` | 348 |
| `kept_labels[296]` | 349 |
| `kept_labels[297]` | 350 |
| `kept_labels[298]` | 351 |
| `kept_labels[299]` | 352 |
| `kept_labels[300]` | 353 |
| `kept_labels[301]` | 354 |
| `kept_labels[302]` | 355 |
| `kept_labels[303]` | 356 |
| `kept_labels[304]` | 357 |
| `kept_labels[305]` | 358 |
| `kept_labels[306]` | 359 |
| `kept_labels[307]` | 360 |
| `kept_labels[308]` | 362 |
| `kept_labels[309]` | 363 |
| `kept_labels[310]` | 364 |
| `kept_labels[311]` | 365 |
| `kept_labels[312]` | 368 |
| `kept_labels[313]` | 369 |
| `kept_labels[314]` | 371 |
| `kept_labels[315]` | 372 |
| `kept_labels[316]` | 373 |
| `kept_labels[317]` | 374 |
| `kept_labels[318]` | 375 |
| `kept_labels[319]` | 376 |
| `kept_labels[320]` | 377 |
| `kept_labels[321]` | 378 |
| `kept_labels[322]` | 380 |
| `kept_labels[323]` | 381 |
| `kept_labels[324]` | 382 |
| `kept_labels[325]` | 383 |
| `kept_labels[326]` | 385 |
| `kept_labels[327]` | 386 |
| `kept_labels[328]` | 387 |
| `kept_labels[329]` | 388 |
| `kept_labels[330]` | 390 |
| `kept_labels[331]` | 392 |
| `kept_labels[332]` | 393 |
| `kept_labels[333]` | 394 |
| `kept_labels[334]` | 395 |
| `kept_labels[335]` | 396 |
| `kept_labels[336]` | 397 |
| `kept_labels[337]` | 398 |
| `kept_labels[338]` | 400 |
| `kept_labels[339]` | 401 |
| `kept_labels[340]` | 402 |
| `kept_labels[341]` | 403 |
| `kept_labels[342]` | 404 |
| `kept_labels[343]` | 405 |
| `kept_labels[344]` | 408 |
| `kept_labels[345]` | 409 |
| `kept_labels[346]` | 410 |
| `kept_labels[347]` | 411 |
| `kept_labels[348]` | 412 |
| `kept_labels[349]` | 413 |
| `kept_labels[350]` | 414 |
| `kept_labels[351]` | 415 |
| `kept_labels[352]` | 416 |
| `kept_labels[353]` | 417 |
| `kept_labels[354]` | 418 |
| `kept_labels[355]` | 419 |
| `kept_labels[356]` | 421 |
| `kept_labels[357]` | 422 |
| `kept_labels[358]` | 423 |
| `kept_labels[359]` | 424 |
| `kept_labels[360]` | 425 |
| `kept_labels[361]` | 426 |
| `kept_labels[362]` | 429 |
| `kept_labels[363]` | 430 |
| `kept_labels[364]` | 431 |
| `kept_labels[365]` | 432 |
| `kept_labels[366]` | 433 |
| `kept_labels[367]` | 435 |
| `kept_labels[368]` | 436 |
| `kept_labels[369]` | 437 |
| `kept_labels[370]` | 438 |
| `kept_labels[371]` | 439 |
| `kept_labels[372]` | 440 |
| `kept_labels[373]` | 441 |
| `kept_labels[374]` | 442 |
| `kept_labels[375]` | 443 |
| `kept_labels[376]` | 444 |
| `kept_labels[377]` | 445 |
| `kept_labels[378]` | 446 |
| `kept_labels[379]` | 447 |
| `kept_labels[380]` | 448 |
| `kept_labels[381]` | 449 |
| `kept_labels[382]` | 450 |
| `kept_labels[383]` | 452 |
| `kept_labels[384]` | 453 |
| `kept_labels[385]` | 454 |
| `kept_labels[386]` | 455 |
| `kept_labels[387]` | 456 |
| `kept_labels[388]` | 457 |
| `kept_labels[389]` | 458 |
| `kept_labels[390]` | 459 |
| `kept_labels[391]` | 460 |
| `kept_labels[392]` | 461 |
| `kept_labels[393]` | 462 |
| `kept_labels[394]` | 463 |
| `kept_labels[395]` | 464 |
| `kept_labels[396]` | 465 |
| `kept_labels[397]` | 466 |
| `kept_labels[398]` | 467 |
| `kept_labels[399]` | 468 |
| `kept_labels[400]` | 470 |
| `kept_labels[401]` | 471 |
| `kept_labels[402]` | 472 |
| `kept_labels[403]` | 473 |
| `kept_labels[404]` | 475 |
| `kept_labels[405]` | 477 |
| `kept_labels[406]` | 478 |
| `kept_labels[407]` | 479 |
| `kept_labels[408]` | 480 |
| `kept_labels[409]` | 481 |
| `kept_labels[410]` | 482 |
| `kept_labels[411]` | 483 |
| `kept_labels[412]` | 485 |
| `kept_labels[413]` | 487 |
| `kept_labels[414]` | 488 |
| `kept_labels[415]` | 489 |
| `kept_labels[416]` | 492 |
| `kept_labels[417]` | 493 |
| `kept_labels[418]` | 495 |
| `kept_labels[419]` | 497 |
| `kept_labels[420]` | 498 |
| `kept_labels[421]` | 499 |
| `kept_labels[422]` | 500 |
| `kept_labels[423]` | 502 |
| `kept_labels[424]` | 503 |
| `kept_labels[425]` | 504 |
| `kept_labels[426]` | 505 |
| `kept_labels[427]` | 506 |
| `kept_labels[428]` | 507 |
| `kept_labels[429]` | 508 |
| `kept_labels[430]` | 509 |
| `kept_labels[431]` | 510 |
| `kept_labels[432]` | 511 |
| `kept_labels[433]` | 512 |
| `kept_labels[434]` | 513 |
| `kept_labels[435]` | 514 |
| `kept_labels[436]` | 515 |
| `kept_labels[437]` | 517 |
| `kept_labels[438]` | 518 |
| `kept_labels[439]` | 520 |
| `kept_labels[440]` | 521 |
| `kept_labels[441]` | 522 |
| `kept_labels[442]` | 523 |
| `kept_labels[443]` | 525 |
| `kept_labels[444]` | 526 |
| `kept_labels[445]` | 527 |
| `kept_labels[446]` | 528 |
| `kept_labels[447]` | 529 |
| `kept_labels[448]` | 530 |
| `kept_labels[449]` | 531 |
| `kept_labels[450]` | 532 |
| `kept_labels[451]` | 533 |
| `kept_labels[452]` | 535 |
| `kept_labels[453]` | 536 |
| `kept_labels[454]` | 537 |
| `kept_labels[455]` | 538 |
| `kept_labels[456]` | 539 |
| `kept_labels[457]` | 540 |
| `kept_labels[458]` | 541 |
| `kept_labels[459]` | 542 |
| `kept_labels[460]` | 543 |
| `kept_labels[461]` | 544 |
| `kept_labels[462]` | 545 |
| `kept_labels[463]` | 546 |
| `kept_labels[464]` | 547 |
| `kept_labels[465]` | 548 |
| `kept_labels[466]` | 549 |
| `kept_labels[467]` | 550 |
| `kept_labels[468]` | 551 |
| `kept_labels[469]` | 552 |
| `kept_labels[470]` | 554 |
| `kept_labels[471]` | 555 |
| `kept_labels[472]` | 556 |
| `kept_labels[473]` | 557 |
| `kept_labels[474]` | 558 |
| `kept_labels[475]` | 559 |
| `kept_labels[476]` | 560 |
| `kept_labels[477]` | 561 |
| `kept_labels[478]` | 562 |
| `kept_labels[479]` | 563 |
| `kept_labels[480]` | 564 |
| `kept_labels[481]` | 565 |
| `kept_labels[482]` | 566 |
| `kept_labels[483]` | 567 |
| `kept_labels[484]` | 568 |
| `kept_labels[485]` | 569 |
| `kept_labels[486]` | 570 |
| `kept_labels[487]` | 571 |
| `kept_labels[488]` | 573 |
| `kept_labels[489]` | 574 |
| `kept_labels[490]` | 575 |
| `kept_labels[491]` | 577 |
| `kept_labels[492]` | 578 |
| `kept_labels[493]` | 579 |
| `kept_labels[494]` | 580 |
| `kept_labels[495]` | 581 |
| `kept_labels[496]` | 583 |
| `kept_labels[497]` | 584 |
| `kept_labels[498]` | 585 |
| `kept_labels[499]` | 586 |
| `kept_labels[500]` | 588 |
| `kept_labels[501]` | 589 |
| `kept_labels[502]` | 590 |
| `kept_labels[503]` | 591 |
| `kept_labels[504]` | 592 |
| `kept_labels[505]` | 593 |
| `kept_labels[506]` | 594 |
| `kept_labels[507]` | 595 |
| `kept_labels[508]` | 596 |
| `kept_labels[509]` | 597 |
| `kept_labels[510]` | 598 |
| `kept_labels[511]` | 600 |
| `kept_labels[512]` | 601 |
| `kept_labels[513]` | 602 |
| `kept_labels[514]` | 603 |
| `kept_labels[515]` | 604 |
| `kept_labels[516]` | 605 |
| `kept_labels[517]` | 606 |
| `kept_labels[518]` | 607 |
| `kept_labels[519]` | 609 |
| `kept_labels[520]` | 610 |
| `kept_labels[521]` | 611 |
| `kept_labels[522]` | 612 |
| `kept_labels[523]` | 613 |
| `kept_labels[524]` | 614 |
| `kept_labels[525]` | 615 |
| `kept_labels[526]` | 616 |
| `kept_labels[527]` | 617 |
| `kept_labels[528]` | 619 |
| `kept_labels[529]` | 620 |
| `kept_labels[530]` | 621 |
| `kept_labels[531]` | 623 |
| `kept_labels[532]` | 624 |
| `kept_labels[533]` | 625 |
| `kept_labels[534]` | 626 |
| `kept_labels[535]` | 627 |
| `kept_labels[536]` | 628 |
| `kept_labels[537]` | 629 |
| `kept_labels[538]` | 630 |
| `kept_labels[539]` | 631 |
| `kept_labels[540]` | 632 |
| `kept_labels[541]` | 634 |
| `kept_labels[542]` | 635 |
| `kept_labels[543]` | 636 |
| `kept_labels[544]` | 638 |
| `kept_labels[545]` | 639 |
| `kept_labels[546]` | 640 |
| `kept_labels[547]` | 641 |
| `kept_labels[548]` | 642 |
| `kept_labels[549]` | 643 |
| `kept_labels[550]` | 644 |
| `kept_labels[551]` | 645 |
| `kept_labels[552]` | 646 |
| `kept_labels[553]` | 647 |
| `kept_labels[554]` | 648 |
| `kept_labels[555]` | 649 |
| `kept_labels[556]` | 650 |
| `kept_labels[557]` | 651 |
| `kept_labels[558]` | 652 |
| `kept_labels[559]` | 653 |
| `kept_labels[560]` | 654 |
| `kept_labels[561]` | 656 |
| `kept_labels[562]` | 657 |
| `kept_labels[563]` | 658 |
| `kept_labels[564]` | 659 |
| `kept_labels[565]` | 661 |
| `kept_labels[566]` | 662 |
| `kept_labels[567]` | 663 |
| `kept_labels[568]` | 664 |
| `kept_labels[569]` | 665 |
| `kept_labels[570]` | 666 |
| `kept_labels[571]` | 667 |
| `kept_labels[572]` | 668 |
| `kept_labels[573]` | 669 |
| `kept_labels[574]` | 670 |
| `kept_labels[575]` | 671 |
| `kept_labels[576]` | 672 |
| `kept_labels[577]` | 673 |
| `kept_labels[578]` | 675 |
| `kept_labels[579]` | 676 |
| `kept_labels[580]` | 677 |
| `kept_labels[581]` | 678 |
| `kept_labels[582]` | 679 |
| `kept_labels[583]` | 680 |
| `kept_labels[584]` | 681 |
| `kept_labels[585]` | 682 |
| `kept_labels[586]` | 683 |
| `kept_labels[587]` | 684 |
| `kept_labels[588]` | 685 |
| `kept_labels[589]` | 686 |
| `kept_labels[590]` | 687 |
| `kept_labels[591]` | 688 |
| `kept_labels[592]` | 689 |
| `kept_labels[593]` | 690 |
| `kept_labels[594]` | 691 |
| `kept_labels[595]` | 692 |
| `kept_labels[596]` | 693 |
| `kept_labels[597]` | 695 |
| `kept_labels[598]` | 696 |
| `kept_labels[599]` | 697 |
| `kept_labels[600]` | 698 |
| `kept_labels[601]` | 699 |
| `kept_labels[602]` | 701 |
| `kept_labels[603]` | 702 |
| `kept_labels[604]` | 703 |
| `kept_labels[605]` | 704 |
| `kept_labels[606]` | 706 |
| `kept_labels[607]` | 708 |
| `kept_labels[608]` | 709 |
| `kept_labels[609]` | 710 |
| `kept_labels[610]` | 711 |
| `kept_labels[611]` | 712 |
| `kept_labels[612]` | 713 |
| `kept_labels[613]` | 714 |
| `kept_labels[614]` | 715 |
| `kept_labels[615]` | 716 |
| `kept_labels[616]` | 720 |
| `kept_labels[617]` | 721 |
| `kept_labels[618]` | 722 |
| `kept_labels[619]` | 723 |
| `kept_labels[620]` | 724 |
| `kept_labels[621]` | 725 |
| `kept_labels[622]` | 726 |
| `kept_labels[623]` | 727 |
| `kept_labels[624]` | 728 |
| `kept_labels[625]` | 730 |
| `kept_labels[626]` | 731 |
| `kept_labels[627]` | 735 |
| `kept_labels[628]` | 736 |
| `kept_labels[629]` | 737 |
| `kept_labels[630]` | 739 |
| `kept_labels[631]` | 740 |
| `kept_labels[632]` | 741 |
| `kept_labels[633]` | 742 |
| `kept_labels[634]` | 743 |
| `kept_labels[635]` | 745 |
| `kept_labels[636]` | 746 |
| `kept_labels[637]` | 747 |
| `kept_labels[638]` | 748 |
| `kept_labels[639]` | 749 |
| `kept_labels[640]` | 750 |
| `kept_labels[641]` | 751 |
| `kept_labels[642]` | 752 |
| `kept_labels[643]` | 753 |
| `kept_labels[644]` | 755 |
| `kept_labels[645]` | 756 |
| `kept_labels[646]` | 757 |
| `kept_labels[647]` | 758 |
| `kept_labels[648]` | 759 |
| `kept_labels[649]` | 760 |
| `kept_labels[650]` | 761 |
| `kept_labels[651]` | 762 |
| `kept_labels[652]` | 763 |
| `kept_labels[653]` | 764 |
| `kept_labels[654]` | 765 |
| `kept_labels[655]` | 766 |
| `kept_labels[656]` | 767 |
| `kept_labels[657]` | 768 |
| `kept_labels[658]` | 769 |
| `kept_labels[659]` | 770 |
| `kept_labels[660]` | 771 |
| `kept_labels[661]` | 772 |
| `kept_labels[662]` | 774 |
| `kept_labels[663]` | 775 |
| `kept_labels[664]` | 776 |
| `kept_labels[665]` | 777 |
| `kept_labels[666]` | 778 |
| `kept_labels[667]` | 779 |
| `kept_labels[668]` | 780 |
| `kept_labels[669]` | 781 |
| `kept_labels[670]` | 782 |
| `kept_labels[671]` | 783 |
| `kept_labels[672]` | 784 |
| `kept_labels[673]` | 785 |
| `kept_labels[674]` | 786 |
| `kept_labels[675]` | 787 |
| `kept_labels[676]` | 788 |
| `kept_labels[677]` | 789 |
| `kept_labels[678]` | 790 |
| `kept_labels[679]` | 791 |
| `kept_labels[680]` | 792 |
| `kept_labels[681]` | 793 |
| `kept_labels[682]` | 794 |
| `kept_labels[683]` | 795 |
| `kept_labels[684]` | 796 |
| `kept_labels[685]` | 798 |
| `kept_labels[686]` | 799 |
| `kept_labels[687]` | 800 |
| `kept_labels[688]` | 801 |
| `kept_labels[689]` | 802 |
| `kept_labels[690]` | 803 |
| `kept_labels[691]` | 805 |
| `kept_labels[692]` | 806 |
| `kept_labels[693]` | 807 |
| `kept_labels[694]` | 808 |
| `kept_labels[695]` | 809 |
| `kept_labels[696]` | 810 |
| `kept_labels[697]` | 811 |
| `kept_labels[698]` | 813 |
| `kept_labels[699]` | 814 |
| `kept_labels[700]` | 815 |
| `kept_labels[701]` | 816 |
| `kept_labels[702]` | 817 |
| `kept_labels[703]` | 818 |
| `kept_labels[704]` | 820 |
| `kept_labels[705]` | 821 |
| `kept_labels[706]` | 822 |
| `kept_labels[707]` | 823 |
| `kept_labels[708]` | 824 |
| `kept_labels[709]` | 825 |
| `kept_labels[710]` | 826 |
| `kept_labels[711]` | 827 |
| `kept_labels[712]` | 828 |
| `kept_labels[713]` | 829 |
| `kept_labels[714]` | 830 |
| `kept_labels[715]` | 831 |
| `kept_labels[716]` | 832 |
| `kept_labels[717]` | 833 |
| `kept_labels[718]` | 834 |
| `kept_labels[719]` | 835 |
| `kept_labels[720]` | 836 |
| `kept_labels[721]` | 838 |
| `kept_labels[722]` | 839 |
| `kept_labels[723]` | 840 |
| `kept_labels[724]` | 841 |
| `kept_labels[725]` | 842 |
| `kept_labels[726]` | 843 |
| `kept_labels[727]` | 844 |
| `kept_labels[728]` | 845 |
| `kept_labels[729]` | 846 |
| `kept_labels[730]` | 847 |
| `kept_labels[731]` | 848 |
| `kept_labels[732]` | 849 |
| `kept_labels[733]` | 850 |
| `kept_labels[734]` | 851 |
| `kept_labels[735]` | 852 |
| `kept_labels[736]` | 853 |
| `kept_labels[737]` | 854 |
| `kept_labels[738]` | 855 |
| `kept_labels[739]` | 856 |
| `kept_labels[740]` | 857 |
| `kept_labels[741]` | 858 |
| `kept_labels[742]` | 859 |
| `kept_labels[743]` | 860 |
| `kept_labels[744]` | 861 |
| `kept_labels[745]` | 862 |
| `kept_labels[746]` | 863 |
| `kept_labels[747]` | 864 |
| `kept_labels[748]` | 865 |
| `kept_labels[749]` | 866 |
| `kept_labels[750]` | 867 |
| `kept_labels[751]` | 868 |
| `kept_labels[752]` | 869 |
| `kept_labels[753]` | 870 |
| `kept_labels[754]` | 872 |
| `kept_labels[755]` | 873 |
| `kept_labels[756]` | 874 |
| `kept_labels[757]` | 875 |
| `kept_labels[758]` | 876 |
| `kept_labels[759]` | 877 |
| `kept_labels[760]` | 878 |
| `kept_labels[761]` | 879 |
| `kept_labels[762]` | 880 |
| `kept_labels[763]` | 881 |
| `kept_labels[764]` | 882 |
| `kept_labels[765]` | 883 |
| `kept_labels[766]` | 884 |
| `kept_labels[767]` | 885 |
| `kept_labels[768]` | 886 |
| `kept_labels[769]` | 887 |
| `kept_labels[770]` | 888 |
| `kept_labels[771]` | 889 |
| `kept_labels[772]` | 890 |
| `kept_labels[773]` | 891 |
| `kept_labels[774]` | 892 |
| `kept_labels[775]` | 893 |
| `kept_labels[776]` | 894 |
| `kept_labels[777]` | 895 |
| `kept_labels[778]` | 896 |
| `kept_labels[779]` | 897 |
| `kept_labels[780]` | 898 |
| `kept_labels[781]` | 899 |
| `kept_labels[782]` | 900 |
| `kept_labels[783]` | 901 |
| `kept_labels[784]` | 902 |
| `kept_labels[785]` | 903 |
| `kept_labels[786]` | 905 |
| `kept_labels[787]` | 906 |
| `kept_labels[788]` | 907 |
| `kept_labels[789]` | 908 |
| `kept_labels[790]` | 909 |
| `kept_labels[791]` | 910 |
| `kept_labels[792]` | 911 |
| `kept_labels[793]` | 913 |
| `kept_labels[794]` | 914 |
| `kept_labels[795]` | 915 |
| `kept_labels[796]` | 916 |
| `kept_labels[797]` | 917 |
| `kept_labels[798]` | 918 |
| `kept_labels[799]` | 919 |
| `kept_labels[800]` | 920 |
| `kept_labels[801]` | 921 |
| `kept_labels[802]` | 923 |
| `kept_labels[803]` | 924 |
| `kept_labels[804]` | 925 |
| `kept_labels[805]` | 926 |
| `kept_labels[806]` | 927 |
| `kept_labels[807]` | 928 |
| `kept_labels[808]` | 929 |
| `kept_labels[809]` | 930 |
| `kept_labels[810]` | 931 |
| `kept_labels[811]` | 932 |
| `kept_labels[812]` | 933 |
| `kept_labels[813]` | 934 |
| `kept_labels[814]` | 935 |
| `kept_labels[815]` | 936 |
| `kept_labels[816]` | 937 |
| `kept_labels[817]` | 938 |
| `kept_labels[818]` | 939 |
| `kept_labels[819]` | 940 |
| `kept_labels[820]` | 941 |
| `kept_labels[821]` | 942 |
| `kept_labels[822]` | 943 |
| `kept_labels[823]` | 944 |
| `kept_labels[824]` | 945 |
| `kept_labels[825]` | 946 |
| `kept_labels[826]` | 947 |
| `kept_labels[827]` | 948 |
| `kept_labels[828]` | 949 |
| `kept_labels[829]` | 950 |
| `kept_labels[830]` | 951 |
| `kept_labels[831]` | 952 |
| `kept_labels[832]` | 953 |
| `kept_labels[833]` | 954 |
| `kept_labels[834]` | 955 |
| `kept_labels[835]` | 956 |
| `kept_labels[836]` | 957 |
| `kept_labels[837]` | 958 |
| `kept_labels[838]` | 959 |
| `kept_labels[839]` | 960 |
| `kept_labels[840]` | 961 |
| `kept_labels[841]` | 962 |
| `kept_labels[842]` | 963 |
| `kept_labels[843]` | 964 |
| `kept_labels[844]` | 965 |
| `kept_labels[845]` | 966 |
| `kept_labels[846]` | 967 |
| `kept_labels[847]` | 968 |
| `kept_labels[848]` | 969 |
| `kept_labels[849]` | 970 |
| `kept_labels[850]` | 971 |
| `kept_labels[851]` | 972 |
| `kept_labels[852]` | 973 |
| `kept_labels[853]` | 974 |
| `kept_labels[854]` | 975 |
| `kept_labels[855]` | 976 |
| `kept_labels[856]` | 977 |
| `kept_labels[857]` | 978 |
| `kept_labels[858]` | 979 |
| `kept_labels[859]` | 980 |
| `kept_labels[860]` | 981 |
| `kept_labels[861]` | 982 |
| `kept_labels[862]` | 983 |
| `kept_labels[863]` | 984 |
| `kept_labels[864]` | 985 |
| `kept_labels[865]` | 986 |
| `kept_labels[866]` | 987 |
| `kept_labels[867]` | 988 |
| `kept_labels[868]` | 989 |
| `kept_labels[869]` | 990 |
| `kept_labels[870]` | 991 |
| `kept_labels[871]` | 992 |
| `kept_labels[872]` | 993 |
| `kept_labels[873]` | 994 |
| `kept_labels[874]` | 995 |
| `kept_labels[875]` | 997 |
| `kept_labels[876]` | 998 |
| `kept_labels[877]` | 999 |
| `kept_labels[878]` | 1000 |
| `kept_labels[879]` | 1001 |
| `kept_labels[880]` | 1002 |
| `kept_labels[881]` | 1003 |
| `kept_labels[882]` | 1004 |
| `kept_labels[883]` | 1005 |
| `kept_labels[884]` | 1006 |
| `kept_labels[885]` | 1007 |
| `kept_labels[886]` | 1008 |
| `kept_labels[887]` | 1009 |
| `kept_labels[888]` | 1010 |
| `kept_labels[889]` | 1011 |
| `kept_labels[890]` | 1012 |
| `kept_labels[891]` | 1013 |
| `kept_labels[892]` | 1014 |
| `kept_labels[893]` | 1015 |
| `kept_labels[894]` | 1016 |
| `kept_labels[895]` | 1017 |
| `kept_labels[896]` | 1018 |
| `kept_labels[897]` | 1019 |
| `kept_labels[898]` | 1020 |
| `kept_labels[899]` | 1021 |
| `kept_labels[900]` | 1022 |
| `kept_labels[901]` | 1023 |
| `kept_labels[902]` | 1025 |
| `kept_labels[903]` | 1026 |
| `kept_labels[904]` | 1028 |
| `kept_labels[905]` | 1029 |
| `kept_labels[906]` | 1031 |
| `kept_labels[907]` | 1032 |
| `kept_labels[908]` | 1033 |
| `kept_labels[909]` | 1034 |
| `kept_labels[910]` | 1035 |
| `kept_labels[911]` | 1036 |
| `kept_labels[912]` | 1037 |
| `kept_labels[913]` | 1038 |
| `kept_labels[914]` | 1040 |
| `kept_labels[915]` | 1041 |
| `kept_labels[916]` | 1043 |
| `kept_labels[917]` | 1044 |
| `kept_labels[918]` | 1046 |
| `kept_labels[919]` | 1047 |
| `kept_labels[920]` | 1048 |
| `kept_labels[921]` | 1049 |
| `kept_labels[922]` | 1050 |
| `kept_labels[923]` | 1052 |
| `kept_labels[924]` | 1053 |
| `kept_labels[925]` | 1054 |
| `kept_labels[926]` | 1055 |
| `kept_labels[927]` | 1056 |
| `kept_labels[928]` | 1057 |
| `kept_labels[929]` | 1058 |
| `kept_labels[930]` | 1059 |
| `kept_labels[931]` | 1060 |
| `kept_labels[932]` | 1061 |
| `kept_labels[933]` | 1062 |
| `kept_labels[934]` | 1064 |
| `kept_labels[935]` | 1065 |
| `kept_labels[936]` | 1067 |
| `kept_labels[937]` | 1068 |
| `kept_labels[938]` | 1069 |
| `kept_labels[939]` | 1070 |
| `kept_labels[940]` | 1073 |
| `kept_labels[941]` | 1074 |
| `kept_labels[942]` | 1075 |
| `kept_labels[943]` | 1078 |
| `kept_labels[944]` | 1079 |
| `kept_labels[945]` | 1080 |
| `kept_labels[946]` | 1081 |
| `kept_labels[947]` | 1083 |
| `kept_labels[948]` | 1084 |
| `kept_labels[949]` | 1085 |
| `kept_labels[950]` | 1086 |
| `kept_labels[951]` | 1087 |
| `kept_labels[952]` | 1089 |
| `kept_labels[953]` | 1090 |
| `kept_labels[954]` | 1091 |
| `kept_labels[955]` | 1093 |
| `kept_labels[956]` | 1095 |
| `kept_labels[957]` | 1096 |
| `kept_labels[958]` | 1097 |
| `kept_labels[959]` | 1098 |
| `kept_labels[960]` | 1099 |
| `kept_labels[961]` | 1100 |
| `kept_labels[962]` | 1101 |
| `kept_labels[963]` | 1102 |
| `kept_labels[964]` | 1103 |
| `kept_labels[965]` | 1104 |
| `kept_labels[966]` | 1105 |
| `kept_labels[967]` | 1106 |
| `kept_labels[968]` | 1107 |
| `kept_labels[969]` | 1108 |
| `kept_labels[970]` | 1109 |
| `kept_labels[971]` | 1110 |
| `kept_labels[972]` | 1111 |
| `kept_labels[973]` | 1112 |
| `kept_labels[974]` | 1113 |
| `kept_labels[975]` | 1114 |
| `kept_labels[976]` | 1115 |
| `kept_labels[977]` | 1116 |
| `kept_labels[978]` | 1117 |
| `kept_labels[979]` | 1118 |
| `kept_labels[980]` | 1120 |
| `kept_labels[981]` | 1121 |
| `kept_labels[982]` | 1122 |
| `kept_labels[983]` | 1123 |
| `kept_labels[984]` | 1124 |
| `kept_labels[985]` | 1125 |
| `kept_labels[986]` | 1126 |
| `kept_labels[987]` | 1127 |
| `kept_labels[988]` | 1128 |
| `kept_labels[989]` | 1129 |
| `kept_labels[990]` | 1130 |
| `kept_labels[991]` | 1131 |
| `kept_labels[992]` | 1132 |
| `kept_labels[993]` | 1133 |
| `kept_labels[994]` | 1134 |
| `kept_labels[995]` | 1135 |
| `kept_labels[996]` | 1136 |
| `kept_labels[997]` | 1137 |
| `kept_labels[998]` | 1138 |
| `kept_labels[999]` | 1139 |
| `kept_labels[1000]` | 1140 |
| `kept_labels[1001]` | 1141 |
| `kept_labels[1002]` | 1142 |
| `kept_labels[1003]` | 1143 |
| `kept_labels[1004]` | 1144 |
| `kept_labels[1005]` | 1145 |
| `kept_labels[1006]` | 1146 |
| `kept_labels[1007]` | 1147 |
| `kept_labels[1008]` | 1148 |
| `kept_labels[1009]` | 1149 |
| `kept_labels[1010]` | 1150 |
| `kept_labels[1011]` | 1151 |
| `kept_labels[1012]` | 1152 |
| `kept_labels[1013]` | 1153 |
| `kept_labels[1014]` | 1154 |
| `kept_labels[1015]` | 1155 |
| `kept_labels[1016]` | 1156 |
| `kept_labels[1017]` | 1157 |
| `kept_labels[1018]` | 1158 |
| `kept_labels[1019]` | 1159 |
| `kept_labels[1020]` | 1160 |
| `kept_labels[1021]` | 1161 |
| `kept_labels[1022]` | 1162 |
| `kept_labels[1023]` | 1163 |
| `kept_labels[1024]` | 1164 |
| `kept_labels[1025]` | 1165 |
| `kept_labels[1026]` | 1166 |
| `kept_labels[1027]` | 1167 |
| `kept_labels[1028]` | 1168 |
| `kept_labels[1029]` | 1169 |
| `kept_labels[1030]` | 1170 |
| `kept_labels[1031]` | 1171 |
| `kept_labels[1032]` | 1173 |
| `kept_labels[1033]` | 1174 |
| `kept_labels[1034]` | 1175 |
| `kept_labels[1035]` | 1176 |
| `kept_labels[1036]` | 1177 |
| `kept_labels[1037]` | 1178 |
| `kept_labels[1038]` | 1179 |
| `kept_labels[1039]` | 1180 |
| `kept_labels[1040]` | 1181 |
| `kept_labels[1041]` | 1182 |
| `kept_labels[1042]` | 1184 |
| `kept_labels[1043]` | 1185 |
| `kept_labels[1044]` | 1186 |
| `kept_labels[1045]` | 1188 |
| `kept_labels[1046]` | 1189 |
| `kept_labels[1047]` | 1190 |
| `kept_labels[1048]` | 1191 |
| `kept_labels[1049]` | 1192 |
| `kept_labels[1050]` | 1193 |
| `kept_labels[1051]` | 1194 |
| `kept_labels[1052]` | 1195 |
| `kept_labels[1053]` | 1196 |
| `kept_labels[1054]` | 1197 |
| `kept_labels[1055]` | 1198 |
| `kept_labels[1056]` | 1200 |
| `kept_labels[1057]` | 1201 |
| `kept_labels[1058]` | 1202 |
| `kept_labels[1059]` | 1203 |
| `kept_labels[1060]` | 1204 |
| `kept_labels[1061]` | 1205 |
| `kept_labels[1062]` | 1207 |
| `kept_labels[1063]` | 1208 |
| `kept_labels[1064]` | 1209 |
| `kept_labels[1065]` | 1210 |
| `kept_labels[1066]` | 1211 |
| `kept_labels[1067]` | 1212 |
| `kept_labels[1068]` | 1213 |
| `kept_labels[1069]` | 1214 |
| `kept_labels[1070]` | 1215 |
| `kept_labels[1071]` | 1216 |
| `kept_labels[1072]` | 1217 |
| `kept_labels[1073]` | 1218 |
| `kept_labels[1074]` | 1220 |
| `kept_labels[1075]` | 1221 |
| `kept_labels[1076]` | 1222 |
| `kept_labels[1077]` | 1223 |
| `kept_labels[1078]` | 1224 |
| `kept_labels[1079]` | 1225 |
| `kept_labels[1080]` | 1227 |
| `kept_labels[1081]` | 1228 |
| `kept_labels[1082]` | 1229 |
| `kept_labels[1083]` | 1230 |
| `kept_labels[1084]` | 1231 |
| `kept_labels[1085]` | 1232 |
| `kept_labels[1086]` | 1233 |
| `kept_labels[1087]` | 1234 |
| `kept_labels[1088]` | 1235 |
| `kept_labels[1089]` | 1236 |
| `kept_labels[1090]` | 1237 |
| `kept_labels[1091]` | 1238 |
| `kept_labels[1092]` | 1239 |
| `kept_labels[1093]` | 1240 |
| `kept_labels[1094]` | 1241 |
| `kept_labels[1095]` | 1242 |
| `kept_labels[1096]` | 1243 |
| `kept_labels[1097]` | 1244 |
| `kept_labels[1098]` | 1245 |
| `kept_labels[1099]` | 1246 |
| `kept_labels[1100]` | 1247 |
| `kept_labels[1101]` | 1249 |
| `kept_labels[1102]` | 1250 |
| `kept_labels[1103]` | 1251 |
| `kept_labels[1104]` | 1252 |
| `kept_labels[1105]` | 1253 |
| `kept_labels[1106]` | 1254 |
| `kept_labels[1107]` | 1255 |
| `kept_labels[1108]` | 1256 |
| `kept_labels[1109]` | 1257 |
| `kept_labels[1110]` | 1258 |
| `kept_labels[1111]` | 1259 |
| `kept_labels[1112]` | 1260 |
| `kept_labels[1113]` | 1261 |
| `kept_labels[1114]` | 1262 |
| `kept_labels[1115]` | 1263 |
| `kept_labels[1116]` | 1264 |
| `kept_labels[1117]` | 1266 |
| `kept_labels[1118]` | 1267 |
| `kept_labels[1119]` | 1268 |
| `kept_labels[1120]` | 1269 |
| `kept_labels[1121]` | 1270 |
| `kept_labels[1122]` | 1271 |
| `kept_labels[1123]` | 1272 |
| `kept_labels[1124]` | 1273 |
| `kept_labels[1125]` | 1274 |
| `kept_labels[1126]` | 1275 |
| `kept_labels[1127]` | 1276 |
| `kept_labels[1128]` | 1277 |
| `kept_labels[1129]` | 1278 |
| `kept_labels[1130]` | 1280 |
| `kept_labels[1131]` | 1281 |
| `kept_labels[1132]` | 1282 |
| `kept_labels[1133]` | 1283 |
| `kept_labels[1134]` | 1284 |
| `kept_labels[1135]` | 1285 |
| `kept_labels[1136]` | 1286 |
| `kept_labels[1137]` | 1287 |
| `kept_labels[1138]` | 1288 |
| `kept_labels[1139]` | 1289 |
| `kept_labels[1140]` | 1291 |
| `kept_labels[1141]` | 1292 |
| `kept_labels[1142]` | 1293 |
| `kept_labels[1143]` | 1294 |
| `kept_labels[1144]` | 1295 |
| `kept_labels[1145]` | 1296 |
| `kept_labels[1146]` | 1297 |
| `kept_labels[1147]` | 1298 |
| `kept_labels[1148]` | 1299 |
| `kept_labels[1149]` | 1300 |
| `kept_labels[1150]` | 1301 |
| `kept_labels[1151]` | 1302 |
| `kept_labels[1152]` | 1304 |
| `kept_labels[1153]` | 1305 |
| `kept_labels[1154]` | 1306 |
| `kept_labels[1155]` | 1307 |
| `kept_labels[1156]` | 1308 |
| `kept_labels[1157]` | 1309 |
| `kept_labels[1158]` | 1310 |
| `kept_labels[1159]` | 1311 |
| `kept_labels[1160]` | 1312 |
| `kept_labels[1161]` | 1313 |
| `kept_labels[1162]` | 1315 |
| `kept_labels[1163]` | 1316 |
| `kept_labels[1164]` | 1317 |
| `kept_labels[1165]` | 1318 |
| `kept_labels[1166]` | 1319 |
| `kept_labels[1167]` | 1320 |
| `kept_labels[1168]` | 1322 |
| `kept_labels[1169]` | 1323 |
| `kept_labels[1170]` | 1324 |
| `kept_labels[1171]` | 1325 |
| `kept_labels[1172]` | 1326 |
| `kept_labels[1173]` | 1327 |
| `kept_labels[1174]` | 1328 |
| `kept_labels[1175]` | 1330 |
| `kept_labels[1176]` | 1331 |
| `kept_labels[1177]` | 1332 |
| `kept_labels[1178]` | 1333 |
| `kept_labels[1179]` | 1334 |
| `kept_labels[1180]` | 1335 |
| `kept_labels[1181]` | 1336 |
| `kept_labels[1182]` | 1337 |
| `kept_labels[1183]` | 1338 |
| `kept_labels[1184]` | 1339 |
| `kept_labels[1185]` | 1340 |
| `kept_labels[1186]` | 1341 |
| `kept_labels[1187]` | 1342 |
| `kept_labels[1188]` | 1343 |
| `kept_labels[1189]` | 1344 |
| `kept_labels[1190]` | 1345 |
| `kept_labels[1191]` | 1346 |
| `kept_labels[1192]` | 1347 |
| `kept_labels[1193]` | 1348 |
| `kept_labels[1194]` | 1349 |
| `kept_labels[1195]` | 1351 |
| `kept_labels[1196]` | 1352 |
| `kept_labels[1197]` | 1353 |
| `kept_labels[1198]` | 1354 |
| `kept_labels[1199]` | 1355 |
| `kept_labels[1200]` | 1356 |
| `kept_labels[1201]` | 1357 |
| `kept_labels[1202]` | 1358 |
| `kept_labels[1203]` | 1359 |
| `kept_labels[1204]` | 1360 |
| `kept_labels[1205]` | 1361 |
| `kept_labels[1206]` | 1362 |
| `kept_labels[1207]` | 1363 |
| `kept_labels[1208]` | 1364 |
| `kept_labels[1209]` | 1365 |
| `kept_labels[1210]` | 1366 |
| `kept_labels[1211]` | 1367 |
| `kept_labels[1212]` | 1368 |
| `kept_labels[1213]` | 1369 |
| `kept_labels[1214]` | 1370 |
| `kept_labels[1215]` | 1371 |
| `kept_labels[1216]` | 1372 |
| `kept_labels[1217]` | 1373 |
| `kept_labels[1218]` | 1374 |
| `kept_labels[1219]` | 1376 |
| `kept_labels[1220]` | 1377 |
| `kept_labels[1221]` | 1378 |
| `kept_labels[1222]` | 1379 |
| `kept_labels[1223]` | 1380 |
| `kept_labels[1224]` | 1381 |
| `kept_labels[1225]` | 1382 |
| `kept_labels[1226]` | 1383 |
| `kept_labels[1227]` | 1384 |
| `kept_labels[1228]` | 1385 |
| `kept_labels[1229]` | 1386 |
| `kept_labels[1230]` | 1387 |
| `kept_labels[1231]` | 1388 |
| `kept_labels[1232]` | 1389 |
| `kept_labels[1233]` | 1390 |
| `kept_labels[1234]` | 1391 |
| `kept_labels[1235]` | 1392 |
| `kept_labels[1236]` | 1394 |
| `kept_labels[1237]` | 1395 |
| `kept_labels[1238]` | 1396 |
| `kept_labels[1239]` | 1397 |
| `kept_labels[1240]` | 1398 |
| `kept_labels[1241]` | 1399 |
| `kept_labels[1242]` | 1400 |
| `kept_labels[1243]` | 1401 |
| `kept_labels[1244]` | 1402 |
| `kept_labels[1245]` | 1403 |
| `kept_labels[1246]` | 1404 |
| `kept_labels[1247]` | 1405 |
| `kept_labels[1248]` | 1406 |
| `kept_labels[1249]` | 1409 |
| `kept_labels[1250]` | 1410 |
| `kept_labels[1251]` | 1411 |
| `kept_labels[1252]` | 1413 |
| `kept_labels[1253]` | 1414 |
| `kept_labels[1254]` | 1415 |
| `kept_labels[1255]` | 1416 |
| `kept_labels[1256]` | 1417 |
| `kept_labels[1257]` | 1418 |
| `kept_labels[1258]` | 1419 |
| `kept_labels[1259]` | 1420 |
| `kept_labels[1260]` | 1421 |
| `kept_labels[1261]` | 1422 |
| `kept_labels[1262]` | 1423 |
| `kept_labels[1263]` | 1424 |
| `kept_labels[1264]` | 1426 |
| `kept_labels[1265]` | 1427 |
| `kept_labels[1266]` | 1428 |
| `kept_labels[1267]` | 1429 |
| `kept_labels[1268]` | 1430 |
| `kept_labels[1269]` | 1431 |
| `kept_labels[1270]` | 1433 |
| `kept_labels[1271]` | 1434 |
| `kept_labels[1272]` | 1435 |
| `kept_labels[1273]` | 1436 |
| `kept_labels[1274]` | 1437 |
| `kept_labels[1275]` | 1438 |
| `kept_labels[1276]` | 1439 |
| `kept_labels[1277]` | 1440 |
| `kept_labels[1278]` | 1441 |
| `kept_labels[1279]` | 1442 |
| `kept_labels[1280]` | 1443 |
| `kept_labels[1281]` | 1444 |
| `kept_labels[1282]` | 1445 |
| `kept_labels[1283]` | 1446 |
| `kept_labels[1284]` | 1447 |
| `kept_labels[1285]` | 1448 |
| `kept_labels[1286]` | 1449 |
| `kept_labels[1287]` | 1450 |
| `kept_labels[1288]` | 1451 |
| `kept_labels[1289]` | 1452 |
| `kept_labels[1290]` | 1453 |
| `kept_labels[1291]` | 1455 |
| `kept_labels[1292]` | 1456 |
| `kept_labels[1293]` | 1457 |
| `kept_labels[1294]` | 1458 |
| `kept_labels[1295]` | 1459 |
| `kept_labels[1296]` | 1461 |
| `kept_labels[1297]` | 1462 |
| `kept_labels[1298]` | 1463 |
| `kept_labels[1299]` | 1464 |
| `kept_labels[1300]` | 1465 |
| `kept_labels[1301]` | 1466 |
| `kept_labels[1302]` | 1468 |
| `kept_labels[1303]` | 1470 |
| `kept_labels[1304]` | 1471 |
| `kept_labels[1305]` | 1472 |
| `kept_labels[1306]` | 1473 |
| `kept_labels[1307]` | 1474 |
| `kept_labels[1308]` | 1475 |
| `kept_labels[1309]` | 1476 |
| `kept_labels[1310]` | 1477 |
| `kept_labels[1311]` | 1478 |
| `kept_labels[1312]` | 1479 |
| `kept_labels[1313]` | 1480 |
| `kept_labels[1314]` | 1481 |
| `kept_labels[1315]` | 1482 |
| `kept_labels[1316]` | 1483 |
| `kept_labels[1317]` | 1484 |
| `kept_labels[1318]` | 1485 |
| `kept_labels[1319]` | 1486 |
| `kept_labels[1320]` | 1487 |
| `kept_labels[1321]` | 1488 |
| `kept_labels[1322]` | 1489 |
| `kept_labels[1323]` | 1490 |
| `kept_labels[1324]` | 1491 |
| `kept_labels[1325]` | 1492 |
| `kept_labels[1326]` | 1493 |
| `kept_labels[1327]` | 1494 |
| `kept_labels[1328]` | 1495 |
| `kept_labels[1329]` | 1496 |
| `kept_labels[1330]` | 1497 |
| `kept_labels[1331]` | 1499 |
| `kept_labels[1332]` | 1500 |
| `kept_labels[1333]` | 1501 |
| `kept_labels[1334]` | 1502 |
| `kept_labels[1335]` | 1503 |
| `kept_labels[1336]` | 1504 |
| `kept_labels[1337]` | 1505 |
| `kept_labels[1338]` | 1506 |
| `kept_labels[1339]` | 1507 |
| `kept_labels[1340]` | 1508 |
| `kept_labels[1341]` | 1509 |
| `kept_labels[1342]` | 1510 |
| `kept_labels[1343]` | 1511 |
| `kept_labels[1344]` | 1512 |
| `kept_labels[1345]` | 1513 |
| `kept_labels[1346]` | 1514 |
| `kept_labels[1347]` | 1515 |
| `kept_labels[1348]` | 1516 |
| `kept_labels[1349]` | 1517 |
| `kept_labels[1350]` | 1518 |
| `kept_labels[1351]` | 1519 |
| `kept_labels[1352]` | 1520 |
| `kept_labels[1353]` | 1521 |
| `kept_labels[1354]` | 1522 |
| `kept_labels[1355]` | 1523 |
| `kept_labels[1356]` | 1524 |
| `kept_labels[1357]` | 1525 |
| `kept_labels[1358]` | 1526 |
| `kept_labels[1359]` | 1527 |
| `kept_labels[1360]` | 1528 |
| `kept_labels[1361]` | 1529 |
| `kept_labels[1362]` | 1530 |
| `kept_labels[1363]` | 1531 |
| `kept_labels[1364]` | 1532 |
| `kept_labels[1365]` | 1533 |
| `kept_labels[1366]` | 1534 |
| `kept_labels[1367]` | 1535 |
| `kept_labels[1368]` | 1536 |
| `kept_labels[1369]` | 1537 |
| `kept_labels[1370]` | 1538 |
| `kept_labels[1371]` | 1539 |
| `kept_labels[1372]` | 1540 |
| `kept_labels[1373]` | 1541 |
| `kept_labels[1374]` | 1542 |
| `kept_labels[1375]` | 1543 |
| `kept_labels[1376]` | 1544 |
| `kept_labels[1377]` | 1545 |
| `kept_labels[1378]` | 1546 |
| `kept_labels[1379]` | 1548 |
| `kept_labels[1380]` | 1549 |
| `kept_labels[1381]` | 1550 |
| `kept_labels[1382]` | 1551 |
| `kept_labels[1383]` | 1552 |
| `kept_labels[1384]` | 1553 |
| `kept_labels[1385]` | 1554 |
| `kept_labels[1386]` | 1555 |
| `kept_labels[1387]` | 1557 |
| `kept_labels[1388]` | 1559 |
| `kept_labels[1389]` | 1560 |
| `kept_labels[1390]` | 1561 |
| `kept_labels[1391]` | 1562 |
| `kept_labels[1392]` | 1563 |
| `kept_labels[1393]` | 1564 |
| `kept_labels[1394]` | 1565 |
| `kept_labels[1395]` | 1566 |
| `kept_labels[1396]` | 1567 |
| `kept_labels[1397]` | 1568 |
| `kept_labels[1398]` | 1569 |
| `kept_labels[1399]` | 1570 |
| `kept_labels[1400]` | 1571 |
| `kept_labels[1401]` | 1572 |
| `kept_labels[1402]` | 1573 |
| `kept_labels[1403]` | 1574 |
| `kept_labels[1404]` | 1575 |
| `kept_labels[1405]` | 1576 |
| `kept_labels[1406]` | 1577 |
| `kept_labels[1407]` | 1578 |
| `kept_labels[1408]` | 1579 |
| `kept_labels[1409]` | 1580 |
| `kept_labels[1410]` | 1582 |
| `kept_labels[1411]` | 1583 |
| `kept_labels[1412]` | 1584 |
| `kept_labels[1413]` | 1585 |
| `kept_labels[1414]` | 1586 |
| `kept_labels[1415]` | 1588 |
| `kept_labels[1416]` | 1589 |
| `kept_labels[1417]` | 1590 |
| `kept_labels[1418]` | 1591 |
| `kept_labels[1419]` | 1592 |
| `kept_labels[1420]` | 1593 |
| `kept_labels[1421]` | 1594 |
| `kept_labels[1422]` | 1596 |
| `kept_labels[1423]` | 1598 |
| `kept_labels[1424]` | 1599 |
| `kept_labels[1425]` | 1600 |
| `kept_labels[1426]` | 1601 |
| `kept_labels[1427]` | 1602 |
| `kept_labels[1428]` | 1603 |
| `kept_labels[1429]` | 1604 |
| `kept_labels[1430]` | 1605 |
| `kept_labels[1431]` | 1606 |
| `kept_labels[1432]` | 1607 |
| `kept_labels[1433]` | 1608 |
| `kept_labels[1434]` | 1609 |
| `kept_labels[1435]` | 1610 |
| `kept_labels[1436]` | 1611 |
| `kept_labels[1437]` | 1612 |
| `kept_labels[1438]` | 1613 |
| `kept_labels[1439]` | 1614 |
| `kept_labels[1440]` | 1615 |
| `kept_labels[1441]` | 1616 |
| `kept_labels[1442]` | 1618 |
| `kept_labels[1443]` | 1619 |
| `kept_labels[1444]` | 1620 |
| `kept_labels[1445]` | 1621 |
| `kept_labels[1446]` | 1622 |
| `kept_labels[1447]` | 1623 |
| `kept_labels[1448]` | 1624 |
| `kept_labels[1449]` | 1625 |
| `kept_labels[1450]` | 1626 |
| `kept_labels[1451]` | 1627 |
| `kept_labels[1452]` | 1628 |
| `kept_labels[1453]` | 1629 |
| `kept_labels[1454]` | 1630 |
| `kept_labels[1455]` | 1631 |
| `kept_labels[1456]` | 1632 |
| `kept_labels[1457]` | 1633 |
| `kept_labels[1458]` | 1634 |
| `kept_labels[1459]` | 1635 |
| `kept_labels[1460]` | 1636 |
| `kept_labels[1461]` | 1637 |
| `kept_labels[1462]` | 1638 |
| `kept_labels[1463]` | 1639 |
| `kept_labels[1464]` | 1640 |
| `kept_labels[1465]` | 1641 |
| `kept_labels[1466]` | 1642 |
| `kept_labels[1467]` | 1643 |
| `kept_labels[1468]` | 1644 |
| `kept_labels[1469]` | 1645 |
| `kept_labels[1470]` | 1647 |
| `kept_labels[1471]` | 1648 |
| `kept_labels[1472]` | 1649 |
| `kept_labels[1473]` | 1650 |
| `kept_labels[1474]` | 1651 |
| `kept_labels[1475]` | 1652 |
| `kept_labels[1476]` | 1653 |
| `kept_labels[1477]` | 1654 |
| `kept_labels[1478]` | 1655 |
| `kept_labels[1479]` | 1656 |
| `kept_labels[1480]` | 1657 |
| `kept_labels[1481]` | 1658 |
| `kept_labels[1482]` | 1659 |
| `kept_labels[1483]` | 1660 |
| `kept_labels[1484]` | 1663 |
| `kept_labels[1485]` | 1664 |
| `kept_labels[1486]` | 1665 |
| `kept_labels[1487]` | 1666 |
| `kept_labels[1488]` | 1667 |
| `kept_labels[1489]` | 1668 |
| `kept_labels[1490]` | 1669 |
| `kept_labels[1491]` | 1670 |
| `kept_labels[1492]` | 1671 |
| `kept_labels[1493]` | 1672 |
| `kept_labels[1494]` | 1673 |
| `kept_labels[1495]` | 1674 |
| `kept_labels[1496]` | 1675 |
| `kept_labels[1497]` | 1677 |
| `kept_labels[1498]` | 1678 |
| `kept_labels[1499]` | 1679 |
| `kept_labels[1500]` | 1680 |
| `kept_labels[1501]` | 1681 |
| `kept_labels[1502]` | 1682 |
| `kept_labels[1503]` | 1683 |
| `kept_labels[1504]` | 1684 |
| `kept_labels[1505]` | 1685 |
| `kept_labels[1506]` | 1686 |
| `kept_labels[1507]` | 1687 |
| `kept_labels[1508]` | 1689 |
| `kept_labels[1509]` | 1690 |
| `kept_labels[1510]` | 1691 |
| `kept_labels[1511]` | 1692 |
| `kept_labels[1512]` | 1693 |
| `kept_labels[1513]` | 1694 |
| `kept_labels[1514]` | 1695 |
| `kept_labels[1515]` | 1696 |
| `kept_labels[1516]` | 1697 |
| `kept_labels[1517]` | 1700 |
| `kept_labels[1518]` | 1701 |
| `kept_labels[1519]` | 1702 |
| `kept_labels[1520]` | 1703 |
| `kept_labels[1521]` | 1704 |
| `kept_labels[1522]` | 1706 |
| `kept_labels[1523]` | 1707 |
| `kept_labels[1524]` | 1708 |
| `kept_labels[1525]` | 1709 |
| `kept_labels[1526]` | 1710 |
| `kept_labels[1527]` | 1711 |
| `kept_labels[1528]` | 1712 |
| `kept_labels[1529]` | 1713 |
| `kept_labels[1530]` | 1714 |
| `kept_labels[1531]` | 1715 |
| `kept_labels[1532]` | 1716 |
| `kept_labels[1533]` | 1717 |
| `kept_labels[1534]` | 1718 |
| `kept_labels[1535]` | 1719 |
| `kept_labels[1536]` | 1720 |
| `kept_labels[1537]` | 1721 |
| `kept_labels[1538]` | 1722 |
| `kept_labels[1539]` | 1723 |
| `kept_labels[1540]` | 1724 |
| `kept_labels[1541]` | 1725 |
| `kept_labels[1542]` | 1726 |
| `kept_labels[1543]` | 1727 |
| `kept_labels[1544]` | 1728 |
| `kept_labels[1545]` | 1729 |
| `kept_labels[1546]` | 1730 |
| `kept_labels[1547]` | 1731 |
| `kept_labels[1548]` | 1732 |
| `kept_labels[1549]` | 1733 |
| `kept_labels[1550]` | 1734 |
| `kept_labels[1551]` | 1735 |
| `kept_labels[1552]` | 1736 |
| `kept_labels[1553]` | 1737 |
| `kept_labels[1554]` | 1738 |
| `kept_labels[1555]` | 1739 |
| `kept_labels[1556]` | 1740 |
| `kept_labels[1557]` | 1741 |
| `kept_labels[1558]` | 1742 |
| `kept_labels[1559]` | 1743 |
| `kept_labels[1560]` | 1744 |
| `kept_labels[1561]` | 1745 |
| `kept_labels[1562]` | 1746 |
| `kept_labels[1563]` | 1747 |
| `kept_labels[1564]` | 1748 |
| `kept_labels[1565]` | 1749 |
| `kept_labels[1566]` | 1750 |
| `kept_labels[1567]` | 1751 |
| `kept_labels[1568]` | 1752 |
| `kept_labels[1569]` | 1753 |
| `kept_labels[1570]` | 1754 |
| `kept_labels[1571]` | 1755 |
| `kept_labels[1572]` | 1756 |
| `kept_labels[1573]` | 1757 |
| `kept_labels[1574]` | 1759 |
| `kept_labels[1575]` | 1760 |
| `kept_labels[1576]` | 1761 |
| `kept_labels[1577]` | 1762 |
| `kept_labels[1578]` | 1763 |
| `kept_labels[1579]` | 1764 |
| `kept_labels[1580]` | 1765 |
| `kept_labels[1581]` | 1766 |
| `kept_labels[1582]` | 1767 |
| `kept_labels[1583]` | 1769 |
| `kept_labels[1584]` | 1770 |
| `kept_labels[1585]` | 1771 |
| `kept_labels[1586]` | 1772 |
| `kept_labels[1587]` | 1773 |
| `kept_labels[1588]` | 1774 |
| `kept_labels[1589]` | 1775 |
| `kept_labels[1590]` | 1776 |
| `kept_labels[1591]` | 1777 |
| `kept_labels[1592]` | 1778 |
| `kept_labels[1593]` | 1779 |
| `kept_labels[1594]` | 1780 |
| `kept_labels[1595]` | 1781 |
| `kept_labels[1596]` | 1782 |
| `kept_labels[1597]` | 1783 |
| `kept_labels[1598]` | 1784 |
| `kept_labels[1599]` | 1785 |
| `kept_labels[1600]` | 1787 |
| `kept_labels[1601]` | 1788 |
| `kept_labels[1602]` | 1789 |
| `kept_labels[1603]` | 1790 |
| `kept_labels[1604]` | 1791 |
| `kept_labels[1605]` | 1792 |
| `kept_labels[1606]` | 1793 |
| `kept_labels[1607]` | 1794 |
| `kept_labels[1608]` | 1795 |
| `kept_labels[1609]` | 1796 |
| `kept_labels[1610]` | 1797 |
| `kept_labels[1611]` | 1798 |
| `kept_labels[1612]` | 1799 |
| `kept_labels[1613]` | 1800 |
| `kept_labels[1614]` | 1801 |
| `kept_labels[1615]` | 1802 |
| `kept_labels[1616]` | 1803 |
| `kept_labels[1617]` | 1804 |
| `kept_labels[1618]` | 1805 |
| `kept_labels[1619]` | 1806 |
| `kept_labels[1620]` | 1807 |
| `kept_labels[1621]` | 1808 |
| `kept_labels[1622]` | 1809 |
| `kept_labels[1623]` | 1810 |
| `kept_labels[1624]` | 1811 |
| `kept_labels[1625]` | 1812 |
| `kept_labels[1626]` | 1813 |
| `kept_labels[1627]` | 1814 |
| `kept_labels[1628]` | 1815 |
| `kept_labels[1629]` | 1817 |
| `kept_labels[1630]` | 1818 |
| `kept_labels[1631]` | 1819 |
| `kept_labels[1632]` | 1820 |
| `kept_labels[1633]` | 1821 |
| `kept_labels[1634]` | 1822 |
| `kept_labels[1635]` | 1823 |
| `kept_labels[1636]` | 1824 |
| `kept_labels[1637]` | 1825 |
| `kept_labels[1638]` | 1826 |
| `kept_labels[1639]` | 1827 |
| `kept_labels[1640]` | 1828 |
| `kept_labels[1641]` | 1829 |
| `kept_labels[1642]` | 1830 |
| `kept_labels[1643]` | 1831 |
| `kept_labels[1644]` | 1833 |
| `kept_labels[1645]` | 1834 |
| `kept_labels[1646]` | 1835 |
| `kept_labels[1647]` | 1836 |
| `kept_labels[1648]` | 1837 |
| `kept_labels[1649]` | 1838 |
| `kept_labels[1650]` | 1839 |
| `kept_labels[1651]` | 1840 |
| `kept_labels[1652]` | 1842 |
| `kept_labels[1653]` | 1843 |
| `kept_labels[1654]` | 1844 |
| `kept_labels[1655]` | 1845 |
| `kept_labels[1656]` | 1846 |
| `kept_labels[1657]` | 1847 |
| `kept_labels[1658]` | 1848 |
| `kept_labels[1659]` | 1849 |
| `kept_labels[1660]` | 1850 |
| `kept_labels[1661]` | 1852 |
| `kept_labels[1662]` | 1853 |
| `kept_labels[1663]` | 1854 |
| `kept_labels[1664]` | 1855 |
| `kept_labels[1665]` | 1856 |
| `kept_labels[1666]` | 1857 |
| `kept_labels[1667]` | 1859 |
| `kept_labels[1668]` | 1860 |
| `kept_labels[1669]` | 1861 |
| `kept_labels[1670]` | 1862 |
| `kept_labels[1671]` | 1863 |
| `kept_labels[1672]` | 1864 |
| `kept_labels[1673]` | 1865 |
| `kept_labels[1674]` | 1866 |
| `kept_labels[1675]` | 1867 |
| `kept_labels[1676]` | 1868 |
| `kept_labels[1677]` | 1869 |
| `kept_labels[1678]` | 1870 |
| `kept_labels[1679]` | 1871 |
| `kept_labels[1680]` | 1872 |
| `kept_labels[1681]` | 1873 |
| `kept_labels[1682]` | 1874 |
| `kept_labels[1683]` | 1875 |
| `kept_labels[1684]` | 1876 |
| `kept_labels[1685]` | 1877 |
| `kept_labels[1686]` | 1878 |
| `kept_labels[1687]` | 1879 |
| `kept_labels[1688]` | 1880 |
| `kept_labels[1689]` | 1881 |
| `kept_labels[1690]` | 1882 |
| `kept_labels[1691]` | 1883 |
| `kept_labels[1692]` | 1884 |
| `kept_labels[1693]` | 1885 |
| `kept_labels[1694]` | 1886 |
| `kept_labels[1695]` | 1887 |
| `kept_labels[1696]` | 1889 |
| `kept_labels[1697]` | 1890 |
| `kept_labels[1698]` | 1891 |
| `kept_labels[1699]` | 1892 |
| `kept_labels[1700]` | 1893 |
| `kept_labels[1701]` | 1894 |
| `kept_labels[1702]` | 1895 |
| `kept_labels[1703]` | 1896 |
| `kept_labels[1704]` | 1898 |
| `kept_labels[1705]` | 1899 |
| `kept_labels[1706]` | 1900 |
| `kept_labels[1707]` | 1901 |
| `kept_labels[1708]` | 1902 |
| `kept_labels[1709]` | 1903 |
| `kept_labels[1710]` | 1904 |
| `kept_labels[1711]` | 1905 |
| `kept_labels[1712]` | 1906 |
| `kept_labels[1713]` | 1907 |
| `kept_labels[1714]` | 1908 |
| `kept_labels[1715]` | 1909 |
| `kept_labels[1716]` | 1910 |
| `kept_labels[1717]` | 1911 |
| `kept_labels[1718]` | 1912 |
| `kept_labels[1719]` | 1913 |
| `kept_labels[1720]` | 1914 |
| `kept_labels[1721]` | 1915 |
| `kept_labels[1722]` | 1916 |
| `kept_labels[1723]` | 1917 |
| `kept_labels[1724]` | 1918 |
| `kept_labels[1725]` | 1919 |
| `kept_labels[1726]` | 1920 |
| `kept_labels[1727]` | 1921 |
| `kept_labels[1728]` | 1922 |
| `kept_labels[1729]` | 1923 |
| `kept_labels[1730]` | 1924 |
| `kept_labels[1731]` | 1926 |
| `kept_labels[1732]` | 1927 |
| `kept_labels[1733]` | 1928 |
| `kept_labels[1734]` | 1929 |
| `kept_labels[1735]` | 1930 |
| `kept_labels[1736]` | 1931 |
| `kept_labels[1737]` | 1932 |
| `kept_labels[1738]` | 1933 |
| `kept_labels[1739]` | 1934 |
| `kept_labels[1740]` | 1935 |
| `kept_labels[1741]` | 1936 |
| `kept_labels[1742]` | 1938 |
| `kept_labels[1743]` | 1939 |
| `kept_labels[1744]` | 1940 |
| `kept_labels[1745]` | 1941 |
| `kept_labels[1746]` | 1942 |
| `kept_labels[1747]` | 1943 |
| `kept_labels[1748]` | 1944 |
| `kept_labels[1749]` | 1946 |
| `kept_labels[1750]` | 1947 |
| `kept_labels[1751]` | 1949 |
| `kept_labels[1752]` | 1950 |
| `kept_labels[1753]` | 1951 |
| `kept_labels[1754]` | 1954 |
| `kept_labels[1755]` | 1955 |
| `kept_labels[1756]` | 1956 |
| `kept_labels[1757]` | 1957 |
| `kept_labels[1758]` | 1958 |
| `kept_labels[1759]` | 1959 |
| `kept_labels[1760]` | 1960 |
| `kept_labels[1761]` | 1961 |
| `kept_labels[1762]` | 1962 |
| `kept_labels[1763]` | 1963 |
| `kept_labels[1764]` | 1964 |
| `kept_labels[1765]` | 1965 |
| `kept_labels[1766]` | 1966 |
| `kept_labels[1767]` | 1967 |
| `kept_labels[1768]` | 1968 |
| `kept_labels[1769]` | 1970 |
| `kept_labels[1770]` | 1972 |
| `kept_labels[1771]` | 1973 |
| `kept_labels[1772]` | 1974 |
| `kept_labels[1773]` | 1975 |
| `kept_labels[1774]` | 1976 |
| `kept_labels[1775]` | 1977 |
| `kept_labels[1776]` | 1978 |
| `kept_labels[1777]` | 1979 |
| `kept_labels[1778]` | 1980 |
| `kept_labels[1779]` | 1981 |
| `kept_labels[1780]` | 1982 |
| `kept_labels[1781]` | 1983 |
| `kept_labels[1782]` | 1984 |
| `kept_labels[1783]` | 1985 |
| `kept_labels[1784]` | 1986 |
| `kept_labels[1785]` | 1988 |
| `kept_labels[1786]` | 1989 |
| `kept_labels[1787]` | 1990 |
| `kept_labels[1788]` | 1991 |
| `kept_labels[1789]` | 1992 |
| `kept_labels[1790]` | 1993 |
| `kept_labels[1791]` | 1995 |
| `kept_labels[1792]` | 1996 |
| `kept_labels[1793]` | 1997 |
| `kept_labels[1794]` | 1998 |
| `kept_labels[1795]` | 1999 |
| `kept_labels[1796]` | 2000 |
| `kept_labels[1797]` | 2001 |
| `kept_labels[1798]` | 2002 |
| `kept_labels[1799]` | 2003 |
| `kept_labels[1800]` | 2004 |
| `kept_labels[1801]` | 2005 |
| `kept_labels[1802]` | 2006 |
| `kept_labels[1803]` | 2007 |
| `kept_labels[1804]` | 2009 |
| `kept_labels[1805]` | 2010 |
| `kept_labels[1806]` | 2011 |
| `kept_labels[1807]` | 2012 |
| `kept_labels[1808]` | 2014 |
| `kept_labels[1809]` | 2015 |
| `kept_labels[1810]` | 2016 |
| `kept_labels[1811]` | 2017 |
| `kept_labels[1812]` | 2018 |
| `kept_labels[1813]` | 2019 |
| `kept_labels[1814]` | 2020 |
| `kept_labels[1815]` | 2021 |
| `kept_labels[1816]` | 2022 |
| `kept_labels[1817]` | 2024 |
| `kept_labels[1818]` | 2025 |
| `kept_labels[1819]` | 2026 |
| `kept_labels[1820]` | 2028 |
| `kept_labels[1821]` | 2029 |
| `kept_labels[1822]` | 2030 |
| `kept_labels[1823]` | 2031 |
| `kept_labels[1824]` | 2032 |
| `kept_labels[1825]` | 2033 |
| `kept_labels[1826]` | 2034 |
| `kept_labels[1827]` | 2035 |
| `kept_labels[1828]` | 2036 |
| `kept_labels[1829]` | 2037 |
| `kept_labels[1830]` | 2038 |
| `kept_labels[1831]` | 2039 |
| `kept_labels[1832]` | 2040 |
| `kept_labels[1833]` | 2041 |
| `kept_labels[1834]` | 2042 |
| `kept_labels[1835]` | 2044 |
| `kept_labels[1836]` | 2045 |
| `kept_labels[1837]` | 2046 |
| `kept_labels[1838]` | 2047 |
| `kept_labels[1839]` | 2048 |
| `kept_labels[1840]` | 2049 |
| `kept_labels[1841]` | 2050 |
| `kept_labels[1842]` | 2051 |
| `kept_labels[1843]` | 2052 |
| `kept_labels[1844]` | 2053 |
| `kept_labels[1845]` | 2054 |
| `kept_labels[1846]` | 2055 |
| `kept_labels[1847]` | 2056 |
| `kept_labels[1848]` | 2057 |
| `kept_labels[1849]` | 2058 |
| `kept_labels[1850]` | 2059 |
| `kept_labels[1851]` | 2060 |
| `kept_labels[1852]` | 2061 |
| `kept_labels[1853]` | 2062 |
| `kept_labels[1854]` | 2063 |
| `kept_labels[1855]` | 2064 |
| `kept_labels[1856]` | 2065 |
| `kept_labels[1857]` | 2066 |
| `kept_labels[1858]` | 2067 |
| `kept_labels[1859]` | 2068 |
| `kept_labels[1860]` | 2069 |
| `kept_labels[1861]` | 2070 |
| `kept_labels[1862]` | 2071 |
| `kept_labels[1863]` | 2072 |
| `kept_labels[1864]` | 2073 |
| `kept_labels[1865]` | 2074 |
| `kept_labels[1866]` | 2076 |
| `kept_labels[1867]` | 2078 |
| `kept_labels[1868]` | 2079 |
| `kept_labels[1869]` | 2080 |
| `kept_labels[1870]` | 2081 |
| `kept_labels[1871]` | 2082 |
| `kept_labels[1872]` | 2083 |
| `kept_labels[1873]` | 2084 |
| `kept_labels[1874]` | 2085 |
| `kept_labels[1875]` | 2086 |
| `kept_labels[1876]` | 2087 |
| `kept_labels[1877]` | 2088 |
| `kept_labels[1878]` | 2089 |
| `kept_labels[1879]` | 2090 |
| `kept_labels[1880]` | 2091 |
| `kept_labels[1881]` | 2092 |
| `kept_labels[1882]` | 2093 |
| `kept_labels[1883]` | 2094 |
| `kept_labels[1884]` | 2095 |
| `kept_labels[1885]` | 2096 |
| `kept_labels[1886]` | 2097 |
| `kept_labels[1887]` | 2098 |
| `kept_labels[1888]` | 2099 |
| `kept_labels[1889]` | 2100 |
| `kept_labels[1890]` | 2101 |
| `kept_labels[1891]` | 2102 |
| `kept_labels[1892]` | 2103 |
| `kept_labels[1893]` | 2104 |
| `kept_labels[1894]` | 2105 |
| `kept_labels[1895]` | 2106 |
| `kept_labels[1896]` | 2107 |
| `kept_labels[1897]` | 2108 |
| `kept_labels[1898]` | 2109 |
| `kept_labels[1899]` | 2111 |
| `kept_labels[1900]` | 2112 |
| `kept_labels[1901]` | 2113 |
| `kept_labels[1902]` | 2114 |
| `kept_labels[1903]` | 2115 |
| `kept_labels[1904]` | 2116 |
| `kept_labels[1905]` | 2117 |
| `kept_labels[1906]` | 2119 |
| `kept_labels[1907]` | 2121 |
| `kept_labels[1908]` | 2122 |
| `kept_labels[1909]` | 2123 |
| `kept_labels[1910]` | 2124 |
| `kept_labels[1911]` | 2125 |
| `kept_labels[1912]` | 2126 |
| `kept_labels[1913]` | 2127 |
| `kept_labels[1914]` | 2128 |
| `kept_labels[1915]` | 2129 |
| `kept_labels[1916]` | 2130 |
| `kept_labels[1917]` | 2131 |
| `kept_labels[1918]` | 2132 |
| `kept_labels[1919]` | 2133 |
| `kept_labels[1920]` | 2135 |
| `kept_labels[1921]` | 2136 |
| `kept_labels[1922]` | 2137 |
| `kept_labels[1923]` | 2138 |
| `kept_labels[1924]` | 2139 |
| `kept_labels[1925]` | 2141 |
| `kept_labels[1926]` | 2142 |
| `kept_labels[1927]` | 2143 |
| `kept_labels[1928]` | 2145 |
| `kept_labels[1929]` | 2146 |
| `kept_labels[1930]` | 2147 |
| `kept_labels[1931]` | 2148 |
| `kept_labels[1932]` | 2149 |
| `kept_labels[1933]` | 2150 |
| `kept_labels[1934]` | 2151 |
| `kept_labels[1935]` | 2153 |
| `kept_labels[1936]` | 2154 |
| `kept_labels[1937]` | 2155 |
| `kept_labels[1938]` | 2156 |
| `kept_labels[1939]` | 2157 |
| `kept_labels[1940]` | 2158 |
| `kept_labels[1941]` | 2159 |
| `kept_labels[1942]` | 2160 |
| `kept_labels[1943]` | 2161 |
| `kept_labels[1944]` | 2162 |
| `kept_labels[1945]` | 2163 |
| `kept_labels[1946]` | 2164 |
| `kept_labels[1947]` | 2165 |
| `kept_labels[1948]` | 2166 |
| `kept_labels[1949]` | 2167 |
| `kept_labels[1950]` | 2168 |
| `kept_labels[1951]` | 2169 |
| `kept_labels[1952]` | 2170 |
| `kept_labels[1953]` | 2172 |
| `kept_labels[1954]` | 2174 |
| `kept_labels[1955]` | 2175 |
| `kept_labels[1956]` | 2176 |
| `kept_labels[1957]` | 2177 |
| `kept_labels[1958]` | 2178 |
| `kept_labels[1959]` | 2179 |
| `kept_labels[1960]` | 2181 |
| `kept_labels[1961]` | 2182 |
| `kept_labels[1962]` | 2183 |
| `kept_labels[1963]` | 2184 |
| `kept_labels[1964]` | 2185 |
| `kept_labels[1965]` | 2186 |
| `kept_labels[1966]` | 2187 |
| `kept_labels[1967]` | 2188 |
| `kept_labels[1968]` | 2189 |
| `kept_labels[1969]` | 2190 |
| `kept_labels[1970]` | 2191 |
| `kept_labels[1971]` | 2192 |
| `kept_labels[1972]` | 2194 |
| `kept_labels[1973]` | 2195 |
| `kept_labels[1974]` | 2196 |
| `kept_labels[1975]` | 2197 |
| `kept_labels[1976]` | 2198 |
| `kept_labels[1977]` | 2199 |
| `kept_labels[1978]` | 2200 |
| `kept_labels[1979]` | 2201 |
| `kept_labels[1980]` | 2202 |
| `kept_labels[1981]` | 2203 |
| `kept_labels[1982]` | 2204 |
| `kept_labels[1983]` | 2205 |
| `kept_labels[1984]` | 2206 |
| `kept_labels[1985]` | 2207 |
| `kept_labels[1986]` | 2208 |
| `kept_labels[1987]` | 2209 |
| `kept_labels[1988]` | 2210 |
| `kept_labels[1989]` | 2211 |
| `kept_labels[1990]` | 2212 |
| `kept_labels[1991]` | 2213 |
| `kept_labels[1992]` | 2214 |
| `kept_labels[1993]` | 2215 |
| `kept_labels[1994]` | 2217 |
| `kept_labels[1995]` | 2219 |
| `kept_labels[1996]` | 2220 |
| `kept_labels[1997]` | 2221 |
| `kept_labels[1998]` | 2222 |
| `kept_labels[1999]` | 2223 |

### Raw artefact

```json
{
  "train_set": "faces_webface_112x112",
  "model": "w600k_r50",
  "threshold": 0.4,
  "per_identity_sampled": 10,
  "images_sampled": 105631,
  "identities_total": 10572,
  "identities_excluded": 692,
  "identities_kept": 9880,
  "eval_sets": [
    "agedb_30",
    "calfw",
    "cfp_fp",
    "cplfw",
    "lfw"
  ],
  "limitation": "Sampling. An identity whose sampled images happen not to resemble the eval shots is NOT excluded. This list is a floor: the true contaminated set is at least this large.",
  "excluded_labels": [
    2,
    3,
    6,
    13,
    19,
    22,
    24,
    31,
    33,
    34,
    35,
    37,
    41,
    43,
    52,
    57,
    60,
    83,
    84,
    88,
    91,
    93,
    95,
    97,
    101,
    105,
    116,
    120,
    133,
    136,
    137,
    139,
    142,
    147,
    155,
    171,
    176,
    193,
    194,
    203,
    209,
    226,
    228,
    230,
    234,
    236,
    256,
    258,
    264,
    277,
    287,
    298,
    318,
    361,
    366,
    367,
    370,
    379,
    384,
    389,
    391,
    399,
    406,
    407,
    420,
    427,
    428,
    434,
    451,
    469,
    474,
    476,
    484,
    486,
    490,
    491,
    494,
    496,
    501,
    516,
    519,
    524,
    534,
    553,
    572,
    576,
    582,
    587,
    599,
    608,
    618,
    622,
    633,
    637,
    655,
    660,
    674,
    694,
    700,
    705,
    707,
    717,
    718,
    719,
    729,
    732,
    733,
    734,
    738,
    744,
    754,
    773,
    797,
    804,
    812,
    819,
    837,
    871,
    904,
    912,
    922,
    996,
    1024,
    1027,
    1030,
    1039,
    1042,
    1045,
    1051,
    1063,
    1066,
    1071,
    1072,
    1076,
    1077,
    1082,
    1088,
    1092,
    1094,
    1119,
    1172,
    1183,
    1187,
    1199,
    1206,
    1219,
    1226,
    1248,
    1265,
    1279,
    1290,
    1303,
    1314,
    1321,
    1329,
    1350,
    1375,
    1393,
    1407,
    1408,
    1412,
    1425,
    1432,
    1454,
    1460,
    1467,
    1469,
    1498,
    1547,
    1556,
    1558,
    1581,
    1587,
    1595,
    1597,
    1617,
    1646,
    1661,
    1662,
    1676,
    1688,
    1698,
    1699,
    1705,
    1758,
    1768,
    1786,
    1816,
    1832,
    1841,
    1851,
    1858,
    1888,
    1897,
    1925,
    1937,
    1945,
    1948,
    1952,
    1953,
    1969,
    1971,
    1987,
    1994,
    2008,
    2013,
    2023,
    2027,
    2043,
    2075,
    2077,
    2110,
    2118,
    2120,
    2134,
    2140,
    2144,
    2152,
    2171,
    2173,
    2180,
    2193,
    2216,
    2218,
    2224,
    2253,
    2259,
    2300,
    2302,
    2312,
    2315,
    2328,
    2330,
    2336,
    2370,
    2375,
    2446,
    2465,
    2485,
    2496,
    2524,
    2536,
    2553,
    2584,
    2596,
    2614,
    2615,
    2621,
    2623,
    2629,
    2665,
    2690,
    2692,
    2729,
    2759,
    2769,
    2798,
    2805,
    2807,
    2812,
    2822,
    2825,
    2827,
    2838,
    2845,
    2866,
    2869,
    2875,
    2884,
    2896,
    2900,
    2902,
    2918,
    2948,
    2953,
    2959,
    2967,
    2986,
    2990,
    2991,
    3006,
    3010,
    3014,
    3021,
    3024,
    3036,
    3038,
    3053,
    3071,
    3096,
    3126,
    3145,
    3148,
    3152,
    3159,
    3162,
    3165,
    3167,
    3170,
    3192,
    3195,
    3220,
    3253,
    3268,
    3269,
    3270,
    3273,
    3274,
    3315,
    3324,
    3332,
    3347,
    3351,
    3353,
    3356,
    3365,
    3372,
    3376,
    3381,
    3399,
    3403,
    3410,
    3414,
    3446,
    3448,
    3453,
    3492,
    3507,
    3515,
    3578,
    3585,
    3597,
    3632,
    3639,
    3644,
    3645,
    3662,
    3672,
    3690,
    3702,
    3705,
    3708,
    3726,
    3732,
    3746,
    3769,
    3787,
    3794,
    3796,
    3797,
    3830,
    3838,
    3845,
    3848,
    3879,
    3907,
    3914,
    3920,
    3922,
    3924,
    3928,
    3931,
    3936,
    3938,
    3941,
    3965,
    4021,
    4025,
    4049,
    4087,
    4088,
    4092,
    4139,
    4142,
    4146,
    4148,
    4161,
    4164,
    4172,
    4173,
    4177,
    4183,
    4207,
    4264,
    4268,
    4294,
    4311,
    4314,
    4324,
    4328,
    4339,
    4348,
    4360,
    4379,
    4430,
    4447,
    4468,
    4469,
    4479,
    4525,
    4527,
    4532,
    4557,
    4621,
    4630,
    4658,
    4662,
    4664,
    4670,
    4695,
    4712,
    4719,
    4728,
    4730,
    4753,
    4767,
    4780,
    4787,
    4805,
    4808,
    4839,
    4842,
    4853,
    4863,
    4866,
    4871,
    4878,
    4883,
    4888,
    4909,
    4911,
    4929,
    4951,
    4955,
    4964,
    4977,
    4980,
    4998,
    5049,
    5050,
    5065,
    5075,
    5078,
    5088,
    5096,
    5097,
    5100,
    5112,
    5115,
    5122,
    5142,
    5148,
    5152,
    5159,
    5177,
    5189,
    5272,
    5303,
    5308,
    5311,
    5318,
    5324,
    5357,
    5381,
    5387,
    5390,
    5417,
    5432,
    5459,
    5479,
    5500,
    5506,
    5511,
    5512,
    5517,
    5520,
    5522,
    5525,
    5559,
    5570,
    5578,
    5602,
    5624,
    5665,
    5676,
    5682,
    5687,
    5690,
    5718,
    5828,
    5830,
    5843,
    5846,
    5862,
    5873,
    5897,
    5900,
    5925,
    5966,
    5973,
    5984,
    6048,
    6049,
    6058,
    6061,
    6063,
    6065,
    6107,
    6146,
    6161,
    6163,
    6189,
    6204,
    6208,
    6265,
    6280,
    6297,
    6323,
    6328,
    6332,
    6349,
    6384,
    6393,
    6396,
    6458,
    6485,
    6487,
    6558,
    6561,
    6578,
    6579,
    6582,
    6595,
    6621,
    6634,
    6662,
    6707,
    6727,
    6743,
    6759,
    6764,
    6815,
    6817,
    6830,
    6845,
    6901,
    6948,
    6961,
    6962,
    6994,
    7004,
    7031,
    7062,
    7075,
    7103,
    7116,
    7139,
    7142,
    7143,
    7161,
    7162,
    7164,
    7234,
    7311,
    7359,
    7394,
    7439,
    7451,
    7470,
    7484,
    7494,
    7509,
    7512,
    7516,
    7534,
    7562,
    7566,
    7569,
    7597,
    7601,
    7603,
    7675,
    7724,
    7770,
    7775,
    7792,
    7818,
    7881,
    7892,
    7902,
    7948,
    7952,
    7969,
    7997,
    8007,
    8031,
    8055,
    8068,
    8071,
    8100,
    8104,
    8119,
    8131,
    8143,
    8171,
    8222,
    8228,
    8247,
    8277,
    8297,
    8313,
    8319,
    8330,
    8340,
    8403,
    8405,
    8422,
    8431,
    8459,
    8477,
    8536,
    8554,
    8612,
    8627,
    8634,
    8693,
    8700,
    8702,
    8724,
    8781,
    8819,
    8857,
    8890,
    8942,
    8947,
    8982,
    9027,
    9094,
    9095,
    9113,
    9180,
    9190,
    9237,
    9251,
    9325,
    9353,
    9369,
    9370,
    9392,
    9401,
    9416,
    9434,
    9456,
    9463,
    9481,
    9517,
    9532,
    9540,
    9551,
    9570,
    9611,
    9614,
    9649,
    9825,
    9830,
    9833,
    9926,
    9941,
    9969,
    9990,
    10009,
    10048,
    10071,
    10072,
    10080,
    10088,
    10098,
    10123,
    10189,
    10209,
    10237,
    10245,
    10265,
    10321,
    10363,
    10377,
    10382,
    10385,
    10407,
    10480,
    10499,
    10533
  ],
  "kept_labels": [
    0,
    1,
    4,
    5,
    7,
    8,
    9,
    10,
    11,
    12,
    14,
    15,
    16,
    17,
    18,
    20,
    21,
    23,
    25,
    26,
    27,
    28,
    29,
    30,
    32,
    36,
    38,
    39,
    40,
    42,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    53,
    54,
    55,
    56,
    58,
    59,
    61,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    85,
    86,
    87,
    89,
    90,
    92,
    94,
    96,
    98,
    99,
    100,
    102,
    103,
    104,
    106,
    107,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    117,
    118,
    119,
    121,
    122,
    123,
    124,
    125,
    126,
    127,
    128,
    129,
    130,
    131,
    132,
    134,
    135,
    138,
    140,
    141,
    143,
    144,
    145,
    146,
    148,
    149,
    150,
    151,
    152,
    153,
    154,
    156,
    157,
    158,
    159,
    160,
    161,
    162,
    163,
    164,
    165,
    166,
    167,
    168,
    169,
    170,
    172,
    173,
    174,
    175,
    177,
    178,
    179,
    180,
    181,
    182,
    183,
    184,
    185,
    186,
    187,
    188,
    189,
    190,
    191,
    192,
    195,
    196,
    197,
    198,
    199,
    200,
    201,
    202,
    204,
    205,
    206,
    207,
    208,
    210,
    211,
    212,
    213,
    214,
    215,
    216,
    217,
    218,
    219,
    220,
    221,
    222,
    223,
    224,
    225,
    227,
    229,
    231,
    232,
    233,
    235,
    237,
    238,
    239,
    240,
    241,
    242,
    243,
    244,
    245,
    246,
    247,
    248,
    249,
    250,
    251,
    252,
    253,
    254,
    255,
    257,
    259,
    260,
    261,
    262,
    263,
    265,
    266,
    267,
    268,
    269,
    270,
    271,
    272,
    273,
    274,
    275,
    276,
    278,
    279,
    280,
    281,
    282,
    283,
    284,
    285,
    286,
    288,
    289,
    290,
    291,
    292,
    293,
    294,
    295,
    296,
    297,
    299,
    300,
    301,
    302,
    303,
    304,
    305,
    306,
    307,
    308,
    309,
    310,
    311,
    312,
    313,
    314,
    315,
    316,
    317,
    319,
    320,
    321,
    322,
    323,
    324,
    325,
    326,
    327,
    328,
    329,
    330,
    331,
    332,
    333,
    334,
    335,
    336,
    337,
    338,
    339,
    340,
    341,
    342,
    343,
    344,
    345,
    346,
    347,
    348,
    349,
    350,
    351,
    352,
    353,
    354,
    355,
    356,
    357,
    358,
    359,
    360,
    362,
    363,
    364,
    365,
    368,
    369,
    371,
    372,
    373,
    374,
    375,
    376,
    377,
    378,
    380,
    381,
    382,
    383,
    385,
    386,
    387,
    388,
    390,
    392,
    393,
    394,
    395,
    396,
    397,
    398,
    400,
    401,
    402,
    403,
    404,
    405,
    408,
    409,
    410,
    411,
    412,
    413,
    414,
    415,
    416,
    417,
    418,
    419,
    421,
    422,
    423,
    424,
    425,
    426,
    429,
    430,
    431,
    432,
    433,
    435,
    436,
    437,
    438,
    439,
    440,
    441,
    442,
    443,
    444,
    445,
    446,
    447,
    448,
    449,
    450,
    452,
    453,
    454,
    455,
    456,
    457,
    458,
    459,
    460,
    461,
    462,
    463,
    464,
    465,
    466,
    467,
    468,
    470,
    471,
    472,
    473,
    475,
    477,
    478,
    479,
    480,
    481,
    482,
    483,
    485,
    487,
    488,
    489,
    492,
    493,
    495,
    497,
    498,
    499,
    500,
    502,
    503,
    504,
    505,
    506,
    507,
    508,
    509,
    510,
    511,
    512,
    513,
    514,
    515,
    517,
    518,
    520,
    521,
    522,
    523,
    525,
    526,
    527,
    528,
    529,
    530,
    531,
    532,
    533,
    535,
    536,
    537,
    538,
    539,
    540,
    541,
    542,
    543,
    544,
    545,
    546,
    547,
    548,
    549,
    550,
    551,
    552,
    554,
    555,
    556,
    557,
    558,
    559,
    560,
    561,
    562,
    563,
    564,
    565,
    566,
    567,
    568,
    569,
    570,
    571,
    573,
    574,
    575,
    577,
    578,
    579,
    580,
    581,
    583,
    584,
    585,
    586,
    588,
    589,
    590,
    591,
    592,
    593,
    594,
    595,
    596,
    597,
    598,
    600,
    601,
    602,
    603,
    604,
    605,
    606,
    607,
    609,
    610,
    611,
    612,
    613,
    614,
    615,
    616,
    617,
    619,
    620,
    621,
    623,
    624,
    625,
    626,
    627,
    628,
    629,
    630,
    631,
    632,
    634,
    635,
    636,
    638,
    639,
    640,
    641,
    642,
    643,
    644,
    645,
    646,
    647,
    648,
    649,
    650,
    651,
    652,
    653,
    654,
    656,
    657,
    658,
    659,
    661,
    662,
    663,
    664,
    665,
    666,
    667,
    668,
    669,
    670,
    671,
    672,
    673,
    675,
    676,
    677,
    678,
    679,
    680,
    681,
    682,
    683,
    684,
    685,
    686,
    687,
    688,
    689,
    690,
    691,
    692,
    693,
    695,
    696,
    697,
    698,
    699,
    701,
    702,
    703,
    704,
    706,
    708,
    709,
    710,
    711,
    712,
    713,
    714,
    715,
    716,
    720,
    721,
    722,
    723,
    724,
    725,
    726,
    727,
    728,
    730,
    731,
    735,
    736,
    737,
    739,
    740,
    741,
    742,
    743,
    745,
    746,
    747,
    748,
    749,
    750,
    751,
    752,
    753,
    755,
    756,
    757,
    758,
    759,
    760,
    761,
    762,
    763,
    764,
    765,
    766,
    767,
    768,
    769,
    770,
    771,
    772,
    774,
    775,
    776,
    777,
    778,
    779,
    780,
    781,
    782,
    783,
    784,
    785,
    786,
    787,
    788,
    789,
    790,
    791,
    792,
    793,
    794,
    795,
    796,
    798,
    799,
    800,
    801,
    802,
    803,
    805,
    806,
    807,
    808,
    809,
    810,
    811,
    813,
    814,
    815,
    816,
    817,
    818,
    820,
    821,
    822,
    823,
    824,
    825,
    826,
    827,
    828,
    829,
    830,
    831,
    832,
    833,
    834,
    835,
    836,
    838,
    839,
    840,
    841,
    842,
    843,
    844,
    845,
    846,
    847,
    848,
    849,
    850,
    851,
    852,
    853,
    854,
    855,
    856,
    857,
    858,
    859,
    860,
    861,
    862,
    863,
    864,
    865,
    866,
    867,
    868,
    869,
    870,
    872,
    873,
    874,
    875,
    876,
    877,
    878,
    879,
    880,
    881,
    882,
    883,
    884,
    885,
    886,
    887,
    888,
    889,
    890,
    891,
    892,
    893,
    894,
    895,
    896,
    897,
    898,
    899,
    900,
    901,
    902,
    903,
    905,
    906,
    907,
    908,
    909,
    910,
    911,
    913,
    914,
    915,
    916,
    917,
    918,
    919,
    920,
    921,
    923,
    924,
    925,
    926,
    927,
    928,
    929,
    930,
    931,
    932,
    933,
    934,
    935,
    936,
    937,
    938,
    939,
    940,
    941,
    942,
    943,
    944,
    945,
    946,
    947,
    948,
    949,
    950,
    951,
    952,
    953,
    954,
    955,
    956,
    957,
    958,
    959,
    960,
    961,
    962,
    963,
    964,
    965,
    966,
    967,
    968,
    969,
    970,
    971,
    972,
    973,
    974,
    975,
    976,
    977,
    978,
    979,
    980,
    981,
    982,
    983,
    984,
    985,
    986,
    987,
    988,
    989,
    990,
    991,
    992,
    993,
    994,
    995,
    997,
    998,
    999,
    1000,
    1001,
    1002,
    1003,
    1004,
    1005,
    1006,
    1007,
    1008,
    1009,
    1010,
    1011,
    1012,
    1013,
    1014,
    1015,
    1016,
    1017,
    1018,
    1019,
    1020,
    1021,
    1022,
    1023,
    1025,
    1026,
    1028,
    1029,
    1031,
    1032,
    1033,
    1034,
    1035,
    1036,
    1037,
    1038,
    1040,
    1041,
    1043,
    1044,
    1046,
    1047,
    1048,
    1049,
    1050,
    1052,
    1053,
    1054,
    1055,
    1056,
    1057,
    1058,
    1059,
    1060,
    1061,
    1062,
    1064,
    1065,
    1067,
    1068,
    1069,
    1070,
    1073,
    1074,
    1075,
    1078,
    1079,
    1080,
    1081,
    1083,
    1084,
    1085,
    1086,
    1087,
    1089,
    1090,
    1091,
    1093,
    1095,
    1096,
    1097,
    1098,
    1099,
    1100,
    1101,
    1102,
    1103,
    1104,
    1105,
    1106,
    1107,
    1108,
    1109,
    1110,
    1111,
    1112,
    1113,
    1114,
    1115,
    1116,
    1117,
    1118,
    1120,
    1121,
    1122,
    1123,
    1124,
    1125,
    1126,
    1127,
    1128,
    1129,
    1130,
    1131,
    1132,
    1133,
    1134,
    1135,
    1136,
    1137,
    1138,
    1139,
    1140,
    1141,
    1142,
    1143,
    1144,
    1145,
    1146,
    1147,
    1148,
    1149,
    1150,
    1151,
    1152,
    1153,
    1154,
    1155,
    1156,
    1157,
    1158,
    1159,
    1160,
    1161,
    1162,
    1163,
    1164,
    1165,
    1166,
    1167,
    1168,
    1169,
    1170,
    1171,
    1173,
    1174,
    1175,
    1176,
    1177,
    1178,
    1179,
    1180,
    1181,
    1182,
    1184,
    1185,
    1186,
    1188,
    1189,
    1190,
    1191,
    1192,
    1193,
    1194,
    1195,
    1196,
    1197,
    1198,
    1200,
    1201,
    1202,
    1203,
    1204,
    1205,
    1207,
    1208,
    1209,
    1210,
    1211,
    1212,
    1213,
    1214,
    1215,
    1216,
    1217,
    1218,
    1220,
    1221,
    1222,
    1223,
    1224,
    1225,
    1227,
    1228,
    1229,
    1230,
    1231,
    1232,
    1233,
    1234,
    1235,
    1236,
    1237,
    1238,
    1239,
    1240,
    1241,
    1242,
    1243,
    1244,
    1245,
    1246,
    1247,
    1249,
    1250,
    1251,
    1252,
    1253,
    1254,
    1255,
    1256,
    1257,
    1258,
    1259,
    1260,
    1261,
    1262,
    1263,
    1264,
    1266,
    1267,
    1268,
    1269,
    1270,
    1271,
    1272,
    1273,
    1274,
    1275,
    1276,
    1277,
    1278,
    1280,
    1281,
    1282,
    1283,
    1284,
    1285,
    1286,
    1287,
    1288,
    1289,
    1291,
    1292,
    1293,
    1294,
    1295,
    1296,
    1297,
    1298,
    1299,
    1300,
    1301,
    1302,
    1304,
    1305,
    1306,
    1307,
    1308,
    1309,
    1310,
    1311,
    1312,
    1313,
    1315,
    1316,
    1317,
    1318,
    1319,
    1320,
    1322,
    1323,
    1324,
    1325,
    1326,
    1327,
    1328,
    1330,
    1331,
    1332,
    1333,
    1334,
    1335,
    1336,
    1337,
    1338,
    1339,
    1340,
    1341,
    1342,
    1343,
    1344,
    1345,
    1346,
    1347,
    1348,
    1349,
    1351,
    1352,
    1353,
    1354,
    1355,
    1356,
    1357,
    1358,
    1359,
    1360,
    1361,
    1362,
    1363,
    1364,
    1365,
    1366,
    1367,
    1368,
    1369,
    1370,
    1371,
    1372,
    1373,
    1374,
    1376,
    1377,
    1378,
    1379,
    1380,
    1381,
    1382,
    1383,
    1384,
    1385,
    1386,
    1387,
    1388,
    1389,
    1390,
    1391,
    1392,
    1394,
    1395,
    1396,
    1397,
    1398,
    1399,
    1400,
    1401,
    1402,
    1403,
    1404,
    1405,
    1406,
    1409,
    1410,
    1411,
    1413,
    1414,
    1415,
    1416,
    1417,
    1418,
    1419,
    1420,
    1421,
    1422,
    1423,
    1424,
    1426,
    1427,
    1428,
    1429,
    1430,
    1431,
    1433,
    1434,
    1435,
    1436,
    1437,
    1438,
    1439,
    1440,
    1441,
    1442,
    1443,
    1444,
    1445,
    1446,
    1447,
    1448,
    1449,
    1450,
    1451,
    1452,
    1453,
    1455,
    1456,
    1457,
    1458,
    1459,
    1461,
    1462,
    1463,
    1464,
    1465,
    1466,
    1468,
    1470,
    1471,
    1472,
    1473,
    1474,
    1475,
    1476,
    1477,
    1478,
    1479,
    1480,
    1481,
    1482,
    1483,
    1484,
    1485,
    1486,
    1487,
    1488,
    1489,
    1490,
    1491,
    1492,
    1493,
    1494,
    1495,
    1496,
    1497,
    1499,
    1500,
    1501,
    1502,
    1503,
    1504,
    1505,
    1506,
    1507,
    1508,
    1509,
    1510,
    1511,
    1512,
    1513,
    1514,
    1515,
    1516,
    1517,
    1518,
    1519,
    1520,
    1521,
    1522,
    1523,
    1524,
    1525,
    1526,
    1527,
    1528,
    1529,
    1530,
    1531,
    1532,
    1533,
    1534,
    1535,
    1536,
    1537,
    1538,
    1539,
    1540,
    1541,
    1542,
    1543,
    1544,
    1545,
    1546,
    1548,
    1549,
    1550,
    1551,
    1552,
    1553,
    1554,
    1555,
    1557,
    1559,
    1560,
    1561,
    1562,
    1563,
    1564,
    1565,
    1566,
    1567,
    1568,
    1569,
    1570,
    1571,
    1572,
    1573,
    1574,
    1575,
    1576,
    1577,
    1578,
    1579,
    1580,
    1582,
    1583,
    1584,
    1585,
    1586,
    1588,
    1589,
    1590,
    1591,
    1592,
    1593,
    1594,
    1596,
    1598,
    1599,
    1600,
    1601,
    1602,
    1603,
    1604,
    1605,
    1606,
    1607,
    1608,
    1609,
    1610,
    1611,
    1612,
    1613,
    1614,
    1615,
    1616,
    1618,
    1619,
    1620,
    1621,
    1622,
    1623,
    1624,
    1625,
    1626,
    1627,
    1628,
    1629,
    1630,
    1631,
    1632,
    1633,
    1634,
    1635,
    1636,
    1637,
    1638,
    1639,
    1640,
    1641,
    1642,
    1643,
    1644,
    1645,
    1647,
    1648,
    1649,
    1650,
    1651,
    1652,
    1653,
    1654,
    1655,
    1656,
    1657,
    1658,
    1659,
    1660,
    1663,
    1664,
    1665,
    1666,
    1667,
    1668,
    1669,
    1670,
    1671,
    1672,
    1673,
    1674,
    1675,
    1677,
    1678,
    1679,
    1680,
    1681,
    1682,
    1683,
    1684,
    1685,
    1686,
    1687,
    1689,
    1690,
    1691,
    1692,
    1693,
    1694,
    1695,
    1696,
    1697,
    1700,
    1701,
    1702,
    1703,
    1704,
    1706,
    1707,
    1708,
    1709,
    1710,
    1711,
    1712,
    1713,
    1714,
    1715,
    1716,
    1717,
    1718,
    1719,
    1720,
    1721,
    1722,
    1723,
    1724,
    1725,
    1726,
    1727,
    1728,
    1729,
    1730,
    1731,
    1732,
    1733,
    1734,
    1735,
    1736,
    1737,
    1738,
    1739,
    1740,
    1741,
    1742,
    1743,
    1744,
    1745,
    1746,
    1747,
    1748,
    1749,
    1750,
    1751,
    1752,
    1753,
    1754,
    1755,
    1756,
    1757,
    1759,
    1760,
    1761,
    1762,
    1763,
    1764,
    1765,
    1766,
    1767,
    1769,
    1770,
    1771,
    1772,
    1773,
    1774,
    1775,
    1776,
    1777,
    1778,
    1779,
    1780,
    1781,
    1782,
    1783,
    1784,
    1785,
    1787,
    1788,
    1789,
    1790,
    1791,
    1792,
    1793,
    1794,
    1795,
    1796,
    1797,
    1798,
    1799,
    1800,
    1801,
    1802,
    1803,
    1804,
    1805,
    1806,
    1807,
    1808,
    1809,
    1810,
    1811,
    1812,
    1813,
    1814,
    1815,
    1817,
    1818,
    1819,
    1820,
    1821,
    1822,
    1823,
    1824,
    1825,
    1826,
    1827,
    1828,
    1829,
    1830,
    1831,
    1833,
    1834,
    1835,
    1836,
    1837,
    1838,
    1839,
    1840,
    1842,
    1843,
    1844,
    1845,
    1846,
    1847,
    1848,
    1849,
    1850,
    1852,
    1853,
    1854,
    1855,
    1856,
    1857,
    1859,
    1860,
    1861,
    1862,
    1863,
    1864,
    1865,
    1866,
    1867,
    1868,
    1869,
    1870,
    1871,
    1872,
    1873,
    1874,
    1875,
    1876,
    1877,
    1878,
    1879,
    1880,
    1881,
    1882,
    1883,
    1884,
    1885,
    1886,
    1887,
    1889,
    1890,
    1891,
    1892,
    1893,
    1894,
    1895,
    1896,
    1898,
    1899,
    1900,
    1901,
    1902,
    1903,
    1904,
    1905,
    1906,
    1907,
    1908,
    1909,
    1910,
    1911,
    1912,
    1913,
    1914,
    1915,
    1916,
    1917,
    1918,
    1919,
    1920,
    1921,
    1922,
    1923,
    1924,
    1926,
    1927,
    1928,
    1929,
    1930,
    1931,
    1932,
    1933,
    1934,
    1935,
    1936,
    1938,
    1939,
    1940,
    1941,
    1942,
    1943,
    1944,
    1946,
    1947,
    1949,
    1950,
    1951,
    1954,
    1955,
    1956,
    1957,
    1958,
    1959,
    1960,
    1961,
    1962,
    1963,
    1964,
    1965,
    1966,
    1967,
    1968,
    1970,
    1972,
    1973,
    1974,
    1975,
    1976,
    1977,
    1978,
    1979,
    1980,
    1981,
    1982,
    1983,
    1984,
    1985,
    1986,
    1988,
    1989,
    1990,
    1991,
    1992,
    1993,
    1995,
    1996,
    1997,
    1998,
    1999,
    2000,
    2001,
    2002,
    2003,
    2004,
    2005,
    2006,
    2007,
    2009,
    2010,
    2011,
    2012,
    2014,
    2015,
    2016,
    2017,
    2018,
    2019,
    2020,
    2021,
    2022,
    2024,
    2025,
    2026,
    2028,
    2029,
    2030,
    2031,
    2032,
    2033,
    2034,
    2035,
    2036,
    2037,
    2038,
    2039,
    2040,
    2041,
    2042,
    2044,
    2045,
    2046,
    2047,
    2048,
    2049,
    2050,
    2051,
    2052,
    2053,
    2054,
    2055,
    2056,
    2057,
    2058,
    2059,
    2060,
    2061,
    2062,
    2063,
    2064,
    2065,
    2066,
    2067,
    2068,
    2069,
    2070,
    2071,
    2072,
    2073,
    2074,
    2076,
    2078,
    2079,
    2080,
    2081,
    2082,
    2083,
    2084,
    2085,
    2086,
    2087,
    2088,
    2089,
    2090,
    2091,
    2092,
    2093,
    2094,
    2095,
    2096,
    2097,
    2098,
    2099,
    2100,
    2101,
    2102,
    2103,
    2104,
    2105,
    2106,
    2107,
    2108,
    2109,
    2111,
    2112,
    2113,
    2114,
    2115,
    2116,
    2117,
    2119,
    2121,
    2122,
    2123,
    2124,
    2125,
    2126,
    2127,
    2128,
    2129,
    2130,
    2131,
    2132,
    2133,
    2135,
    2136,
    2137,
    2138,
    2139,
    2141,
    2142,
    2143,
    2145,
    2146,
    2147,
    2148,
    2149,
    2150,
    2151,
    2153,
    2154,
    2155,
    2156,
    2157,
    2158,
    2159,
    2160,
    2161,
    2162,
    2163,
    2164,
    2165,
    2166,
    2167,
    2168,
    2169,
    2170,
    2172,
    2174,
    2175,
    2176,
    2177,
    2178,
    2179,
    2181,
    2182,
    2183,
    2184,
    2185,
    2186,
    2187,
    2188,
    2189,
    2190,
    2191,
    2192,
    2194,
    2195,
    2196,
    2197,
    2198,
    2199,
    2200,
    2201,
    2202,
    2203,
    2204,
    2205,
    2206,
    2207,
    2208,
    2209,
    2210,
    2211,
    2212,
    2213,
    2214,
    2215,
    2217,
    2219,
    2220,
    2221,
    2222,
    2223,
    2225,
    2226,
    2227,
    2228,
    2229,
    2230,
    2231,
    2232,
    2233,
    2234,
    2235,
    2236,
    2237,
    2238,
    2239,
    2240,
    2241,
    2242,
    2243,
    2244,
    2245,
    2246,
    2247,
    2248,
    2249,
    2250,
    2251,
    2252,
    2254,
    2255,
    2256,
    2257,
    2258,
    2260,
    2261,
    2262,
    2263,
    2264,
    2265,
    2266,
    2267,
    2268,
    2269,
    2270,
    2271,
    2272,
    2273,
    2274,
    2275,
    2276,
    2277,
    2278,
    2279,
    2280,
    2281,
    2282,
    2283,
    2284,
    2285,
    2286,
    2287,
    2288,
    2289,
    2290,
    2291,
    2292,
    2293,
    2294,
    2295,
    2296,
    2297,
    2298,
    2299,
    2301,
    2303,
    2304,
    2305,
    2306,
    2307,
    2308,
    2309,
    2310,
    2311,
    2313,
    2314,
    2316,
    2317,
    2318,
    2319,
    2320,
    2321,
    2322,
    2323,
    2324,
    2325,
    2326,
    2327,
    2329,
    2331,
    2332,
    2333,
    2334,
    2335,
    2337,
    2338,
    2339,
    2340,
    2341,
    2342,
    2343,
    2344,
    2345,
    2346,
    2347,
    2348,
    2349,
    2350,
    2351,
    2352,
    2353,
    2354,
    2355,
    2356,
    2357,
    2358,
    2359,
    2360,
    2361,
    2362,
    2363,
    2364,
    2365,
    2366,
    2367,
    2368,
    2369,
    2371,
    2372,
    2373,
    2374,
    2376,
    2377,
    2378,
    2379,
    2380,
    2381,
    2382,
    2383,
    2384,
    2385,
    2386,
    2387,
    2388,
    2389,
    2390,
    2391,
    2392,
    2393,
    2394,
    2395,
    2396,
    2397,
    2398,
    2399,
    2400,
    2401,
    2402,
    2403,
    2404,
    2405,
    2406,
    2407,
    2408,
    2409,
    2410,
    2411,
    2412,
    2413,
    2414,
    2415,
    2416,
    2417,
    2418,
    2419,
    2420,
    2421,
    2422,
    2423,
    2424,
    2425,
    2426,
    2427,
    2428,
    2429,
    2430,
    2431,
    2432,
    2433,
    2434,
    2435,
    2436,
    2437,
    2438,
    2439,
    2440,
    2441,
    2442,
    2443,
    2444,
    2445,
    2447,
    2448,
    2449,
    2450,
    2451,
    2452,
    2453,
    2454,
    2455,
    2456,
    2457,
    2458,
    2459,
    2460,
    2461,
    2462,
    2463,
    2464,
    2466,
    2467,
    2468,
    2469,
    2470,
    2471,
    2472,
    2473,
    2474,
    2475,
    2476,
    2477,
    2478,
    2479,
    2480,
    2481,
    2482,
    2483,
    2484,
    2486,
    2487,
    2488,
    2489,
    2490,
    2491,
    2492,
    2493,
    2494,
    2495,
    2497,
    2498,
    2499,
    2500,
    2501,
    2502,
    2503,
    2504,
    2505,
    2506,
    2507,
    2508,
    2509,
    2510,
    2511,
    2512,
    2513,
    2514,
    2515,
    2516,
    2517,
    2518,
    2519,
    2520,
    2521,
    2522,
    2523,
    2525,
    2526,
    2527,
    2528,
    2529,
    2530,
    2531,
    2532,
    2533,
    2534,
    2535,
    2537,
    2538,
    2539,
    2540,
    2541,
    2542,
    2543,
    2544,
    2545,
    2546,
    2547,
    2548,
    2549,
    2550,
    2551,
    2552,
    2554,
    2555,
    2556,
    2557,
    2558,
    2559,
    2560,
    2561,
    2562,
    2563,
    2564,
    2565,
    2566,
    2567,
    2568,
    2569,
    2570,
    2571,
    2572,
    2573,
    2574,
    2575,
    2576,
    2577,
    2578,
    2579,
    2580,
    2581,
    2582,
    2583,
    2585,
    2586,
    2587,
    2588,
    2589,
    2590,
    2591,
    2592,
    2593,
    2594,
    2595,
    2597,
    2598,
    2599,
    2600,
    2601,
    2602,
    2603,
    2604,
    2605,
    2606,
    2607,
    2608,
    2609,
    2610,
    2611,
    2612,
    2613,
    2616,
    2617,
    2618,
    2619,
    2620,
    2622,
    2624,
    2625,
    2626,
    2627,
    2628,
    2630,
    2631,
    2632,
    2633,
    2634,
    2635,
    2636,
    2637,
    2638,
    2639,
    2640,
    2641,
    2642,
    2643,
    2644,
    2645,
    2646,
    2647,
    2648,
    2649,
    2650,
    2651,
    2652,
    2653,
    2654,
    2655,
    2656,
    2657,
    2658,
    2659,
    2660,
    2661,
    2662,
    2663,
    2664,
    2666,
    2667,
    2668,
    2669,
    2670,
    2671,
    2672,
    2673,
    2674,
    2675,
    2676,
    2677,
    2678,
    2679,
    2680,
    2681,
    2682,
    2683,
    2684,
    2685,
    2686,
    2687,
    2688,
    2689,
    2691,
    2693,
    2694,
    2695,
    2696,
    2697,
    2698,
    2699,
    2700,
    2701,
    2702,
    2703,
    2704,
    2705,
    2706,
    2707,
    2708,
    2709,
    2710,
    2711,
    2712,
    2713,
    2714,
    2715,
    2716,
    2717,
    2718,
    2719,
    2720,
    2721,
    2722,
    2723,
    2724,
    2725,
    2726,
    2727,
    2728,
    2730,
    2731,
    2732,
    2733,
    2734,
    2735,
    2736,
    2737,
    2738,
    2739,
    2740,
    2741,
    2742,
    2743,
    2744,
    2745,
    2746,
    2747,
    2748,
    2749,
    2750,
    2751,
    2752,
    2753,
    2754,
    2755,
    2756,
    2757,
    2758,
    2760,
    2761,
    2762,
    2763,
    2764,
    2765,
    2766,
    2767,
    2768,
    2770,
    2771,
    2772,
    2773,
    2774,
    2775,
    2776,
    2777,
    2778,
    2779,
    2780,
    2781,
    2782,
    2783,
    2784,
    2785,
    2786,
    2787,
    2788,
    2789,
    2790,
    2791,
    2792,
    2793,
    2794,
    2795,
    2796,
    2797,
    2799,
    2800,
    2801,
    2802,
    2803,
    2804,
    2806,
    2808,
    2809,
    2810,
    2811,
    2813,
    2814,
    2815,
    2816,
    2817,
    2818,
    2819,
    2820,
    2821,
    2823,
    2824,
    2826,
    2828,
    2829,
    2830,
    2831,
    2832,
    2833,
    2834,
    2835,
    2836,
    2837,
    2839,
    2840,
    2841,
    2842,
    2843,
    2844,
    2846,
    2847,
    2848,
    2849,
    2850,
    2851,
    2852,
    2853,
    2854,
    2855,
    2856,
    2857,
    2858,
    2859,
    2860,
    2861,
    2862,
    2863,
    2864,
    2865,
    2867,
    2868,
    2870,
    2871,
    2872,
    2873,
    2874,
    2876,
    2877,
    2878,
    2879,
    2880,
    2881,
    2882,
    2883,
    2885,
    2886,
    2887,
    2888,
    2889,
    2890,
    2891,
    2892,
    2893,
    2894,
    2895,
    2897,
    2898,
    2899,
    2901,
    2903,
    2904,
    2905,
    2906,
    2907,
    2908,
    2909,
    2910,
    2911,
    2912,
    2913,
    2914,
    2915,
    2916,
    2917,
    2919,
    2920,
    2921,
    2922,
    2923,
    2924,
    2925,
    2926,
    2927,
    2928,
    2929,
    2930,
    2931,
    2932,
    2933,
    2934,
    2935,
    2936,
    2937,
    2938,
    2939,
    2940,
    2941,
    2942,
    2943,
    2944,
    2945,
    2946,
    2947,
    2949,
    2950,
    2951,
    2952,
    2954,
    2955,
    2956,
    2957,
    2958,
    2960,
    2961,
    2962,
    2963,
    2964,
    2965,
    2966,
    2968,
    2969,
    2970,
    2971,
    2972,
    2973,
    2974,
    2975,
    2976,
    2977,
    2978,
    2979,
    2980,
    2981,
    2982,
    2983,
    2984,
    2985,
    2987,
    2988,
    2989,
    2992,
    2993,
    2994,
    2995,
    2996,
    2997,
    2998,
    2999,
    3000,
    3001,
    3002,
    3003,
    3004,
    3005,
    3007,
    3008,
    3009,
    3011,
    3012,
    3013,
    3015,
    3016,
    3017,
    3018,
    3019,
    3020,
    3022,
    3023,
    3025,
    3026,
    3027,
    3028,
    3029,
    3030,
    3031,
    3032,
    3033,
    3034,
    3035,
    3037,
    3039,
    3040,
    3041,
    3042,
    3043,
    3044,
    3045,
    3046,
    3047,
    3048,
    3049,
    3050,
    3051,
    3052,
    3054,
    3055,
    3056,
    3057,
    3058,
    3059,
    3060,
    3061,
    3062,
    3063,
    3064,
    3065,
    3066,
    3067,
    3068,
    3069,
    3070,
    3072,
    3073,
    3074,
    3075,
    3076,
    3077,
    3078,
    3079,
    3080,
    3081,
    3082,
    3083,
    3084,
    3085,
    3086,
    3087,
    3088,
    3089,
    3090,
    3091,
    3092,
    3093,
    3094,
    3095,
    3097,
    3098,
    3099,
    3100,
    3101,
    3102,
    3103,
    3104,
    3105,
    3106,
    3107,
    3108,
    3109,
    3110,
    3111,
    3112,
    3113,
    3114,
    3115,
    3116,
    3117,
    3118,
    3119,
    3120,
    3121,
    3122,
    3123,
    3124,
    3125,
    3127,
    3128,
    3129,
    3130,
    3131,
    3132,
    3133,
    3134,
    3135,
    3136,
    3137,
    3138,
    3139,
    3140,
    3141,
    3142,
    3143,
    3144,
    3146,
    3147,
    3149,
    3150,
    3151,
    3153,
    3154,
    3155,
    3156,
    3157,
    3158,
    3160,
    3161,
    3163,
    3164,
    3166,
    3168,
    3169,
    3171,
    3172,
    3173,
    3174,
    3175,
    3176,
    3177,
    3178,
    3179,
    3180,
    3181,
    3182,
    3183,
    3184,
    3185,
    3186,
    3187,
    3188,
    3189,
    3190,
    3191,
    3193,
    3194,
    3196,
    3197,
    3198,
    3199,
    3200,
    3201,
    3202,
    3203,
    3204,
    3205,
    3206,
    3207,
    3208,
    3209,
    3210,
    3211,
    3212,
    3213,
    3214,
    3215,
    3216,
    3217,
    3218,
    3219,
    3221,
    3222,
    3223,
    3224,
    3225,
    3226,
    3227,
    3228,
    3229,
    3230,
    3231,
    3232,
    3233,
    3234,
    3235,
    3236,
    3237,
    3238,
    3239,
    3240,
    3241,
    3242,
    3243,
    3244,
    3245,
    3246,
    3247,
    3248,
    3249,
    3250,
    3251,
    3252,
    3254,
    3255,
    3256,
    3257,
    3258,
    3259,
    3260,
    3261,
    3262,
    3263,
    3264,
    3265,
    3266,
    3267,
    3271,
    3272,
    3275,
    3276,
    3277,
    3278,
    3279,
    3280,
    3281,
    3282,
    3283,
    3284,
    3285,
    3286,
    3287,
    3288,
    3289,
    3290,
    3291,
    3292,
    3293,
    3294,
    3295,
    3296,
    3297,
    3298,
    3299,
    3300,
    3301,
    3302,
    3303,
    3304,
    3305,
    3306,
    3307,
    3308,
    3309,
    3310,
    3311,
    3312,
    3313,
    3314,
    3316,
    3317,
    3318,
    3319,
    3320,
    3321,
    3322,
    3323,
    3325,
    3326,
    3327,
    3328,
    3329,
    3330,
    3331,
    3333,
    3334,
    3335,
    3336,
    3337,
    3338,
    3339,
    3340,
    3341,
    3342,
    3343,
    3344,
    3345,
    3346,
    3348,
    3349,
    3350,
    3352,
    3354,
    3355,
    3357,
    3358,
    3359,
    3360,
    3361,
    3362,
    3363,
    3364,
    3366,
    3367,
    3368,
    3369,
    3370,
    3371,
    3373,
    3374,
    3375,
    3377,
    3378,
    3379,
    3380,
    3382,
    3383,
    3384,
    3385,
    3386,
    3387,
    3388,
    3389,
    3390,
    3391,
    3392,
    3393,
    3394,
    3395,
    3396,
    3397,
    3398,
    3400,
    3401,
    3402,
    3404,
    3405,
    3406,
    3407,
    3408,
    3409,
    3411,
    3412,
    3413,
    3415,
    3416,
    3417,
    3418,
    3419,
    3420,
    3421,
    3422,
    3423,
    3424,
    3425,
    3426,
    3427,
    3428,
    3429,
    3430,
    3431,
    3432,
    3433,
    3434,
    3435,
    3436,
    3437,
    3438,
    3439,
    3440,
    3441,
    3442,
    3443,
    3444,
    3445,
    3447,
    3449,
    3450,
    3451,
    3452,
    3454,
    3455,
    3456,
    3457,
    3458,
    3459,
    3460,
    3461,
    3462,
    3463,
    3464,
    3465,
    3466,
    3467,
    3468,
    3469,
    3470,
    3471,
    3472,
    3473,
    3474,
    3475,
    3476,
    3477,
    3478,
    3479,
    3480,
    3481,
    3482,
    3483,
    3484,
    3485,
    3486,
    3487,
    3488,
    3489,
    3490,
    3491,
    3493,
    3494,
    3495,
    3496,
    3497,
    3498,
    3499,
    3500,
    3501,
    3502,
    3503,
    3504,
    3505,
    3506,
    3508,
    3509,
    3510,
    3511,
    3512,
    3513,
    3514,
    3516,
    3517,
    3518,
    3519,
    3520,
    3521,
    3522,
    3523,
    3524,
    3525,
    3526,
    3527,
    3528,
    3529,
    3530,
    3531,
    3532,
    3533,
    3534,
    3535,
    3536,
    3537,
    3538,
    3539,
    3540,
    3541,
    3542,
    3543,
    3544,
    3545,
    3546,
    3547,
    3548,
    3549,
    3550,
    3551,
    3552,
    3553,
    3554,
    3555,
    3556,
    3557,
    3558,
    3559,
    3560,
    3561,
    3562,
    3563,
    3564,
    3565,
    3566,
    3567,
    3568,
    3569,
    3570,
    3571,
    3572,
    3573,
    3574,
    3575,
    3576,
    3577,
    3579,
    3580,
    3581,
    3582,
    3583,
    3584,
    3586,
    3587,
    3588,
    3589,
    3590,
    3591,
    3592,
    3593,
    3594,
    3595,
    3596,
    3598,
    3599,
    3600,
    3601,
    3602,
    3603,
    3604,
    3605,
    3606,
    3607,
    3608,
    3609,
    3610,
    3611,
    3612,
    3613,
    3614,
    3615,
    3616,
    3617,
    3618,
    3619,
    3620,
    3621,
    3622,
    3623,
    3624,
    3625,
    3626,
    3627,
    3628,
    3629,
    3630,
    3631,
    3633,
    3634,
    3635,
    3636,
    3637,
    3638,
    3640,
    3641,
    3642,
    3643,
    3646,
    3647,
    3648,
    3649,
    3650,
    3651,
    3652,
    3653,
    3654,
    3655,
    3656,
    3657,
    3658,
    3659,
    3660,
    3661,
    3663,
    3664,
    3665,
    3666,
    3667,
    3668,
    3669,
    3670,
    3671,
    3673,
    3674,
    3675,
    3676,
    3677,
    3678,
    3679,
    3680,
    3681,
    3682,
    3683,
    3684,
    3685,
    3686,
    3687,
    3688,
    3689,
    3691,
    3692,
    3693,
    3694,
    3695,
    3696,
    3697,
    3698,
    3699,
    3700,
    3701,
    3703,
    3704,
    3706,
    3707,
    3709,
    3710,
    3711,
    3712,
    3713,
    3714,
    3715,
    3716,
    3717,
    3718,
    3719,
    3720,
    3721,
    3722,
    3723,
    3724,
    3725,
    3727,
    3728,
    3729,
    3730,
    3731,
    3733,
    3734,
    3735,
    3736,
    3737,
    3738,
    3739,
    3740,
    3741,
    3742,
    3743,
    3744,
    3745,
    3747,
    3748,
    3749,
    3750,
    3751,
    3752,
    3753,
    3754,
    3755,
    3756,
    3757,
    3758,
    3759,
    3760,
    3761,
    3762,
    3763,
    3764,
    3765,
    3766,
    3767,
    3768,
    3770,
    3771,
    3772,
    3773,
    3774,
    3775,
    3776,
    3777,
    3778,
    3779,
    3780,
    3781,
    3782,
    3783,
    3784,
    3785,
    3786,
    3788,
    3789,
    3790,
    3791,
    3792,
    3793,
    3795,
    3798,
    3799,
    3800,
    3801,
    3802,
    3803,
    3804,
    3805,
    3806,
    3807,
    3808,
    3809,
    3810,
    3811,
    3812,
    3813,
    3814,
    3815,
    3816,
    3817,
    3818,
    3819,
    3820,
    3821,
    3822,
    3823,
    3824,
    3825,
    3826,
    3827,
    3828,
    3829,
    3831,
    3832,
    3833,
    3834,
    3835,
    3836,
    3837,
    3839,
    3840,
    3841,
    3842,
    3843,
    3844,
    3846,
    3847,
    3849,
    3850,
    3851,
    3852,
    3853,
    3854,
    3855,
    3856,
    3857,
    3858,
    3859,
    3860,
    3861,
    3862,
    3863,
    3864,
    3865,
    3866,
    3867,
    3868,
    3869,
    3870,
    3871,
    3872,
    3873,
    3874,
    3875,
    3876,
    3877,
    3878,
    3880,
    3881,
    3882,
    3883,
    3884,
    3885,
    3886,
    3887,
    3888,
    3889,
    3890,
    3891,
    3892,
    3893,
    3894,
    3895,
    3896,
    3897,
    3898,
    3899,
    3900,
    3901,
    3902,
    3903,
    3904,
    3905,
    3906,
    3908,
    3909,
    3910,
    3911,
    3912,
    3913,
    3915,
    3916,
    3917,
    3918,
    3919,
    3921,
    3923,
    3925,
    3926,
    3927,
    3929,
    3930,
    3932,
    3933,
    3934,
    3935,
    3937,
    3939,
    3940,
    3942,
    3943,
    3944,
    3945,
    3946,
    3947,
    3948,
    3949,
    3950,
    3951,
    3952,
    3953,
    3954,
    3955,
    3956,
    3957,
    3958,
    3959,
    3960,
    3961,
    3962,
    3963,
    3964,
    3966,
    3967,
    3968,
    3969,
    3970,
    3971,
    3972,
    3973,
    3974,
    3975,
    3976,
    3977,
    3978,
    3979,
    3980,
    3981,
    3982,
    3983,
    3984,
    3985,
    3986,
    3987,
    3988,
    3989,
    3990,
    3991,
    3992,
    3993,
    3994,
    3995,
    3996,
    3997,
    3998,
    3999,
    4000,
    4001,
    4002,
    4003,
    4004,
    4005,
    4006,
    4007,
    4008,
    4009,
    4010,
    4011,
    4012,
    4013,
    4014,
    4015,
    4016,
    4017,
    4018,
    4019,
    4020,
    4022,
    4023,
    4024,
    4026,
    4027,
    4028,
    4029,
    4030,
    4031,
    4032,
    4033,
    4034,
    4035,
    4036,
    4037,
    4038,
    4039,
    4040,
    4041,
    4042,
    4043,
    4044,
    4045,
    4046,
    4047,
    4048,
    4050,
    4051,
    4052,
    4053,
    4054,
    4055,
    4056,
    4057,
    4058,
    4059,
    4060,
    4061,
    4062,
    4063,
    4064,
    4065,
    4066,
    4067,
    4068,
    4069,
    4070,
    4071,
    4072,
    4073,
    4074,
    4075,
    4076,
    4077,
    4078,
    4079,
    4080,
    4081,
    4082,
    4083,
    4084,
    4085,
    4086,
    4089,
    4090,
    4091,
    4093,
    4094,
    4095,
    4096,
    4097,
    4098,
    4099,
    4100,
    4101,
    4102,
    4103,
    4104,
    4105,
    4106,
    4107,
    4108,
    4109,
    4110,
    4111,
    4112,
    4113,
    4114,
    4115,
    4116,
    4117,
    4118,
    4119,
    4120,
    4121,
    4122,
    4123,
    4124,
    4125,
    4126,
    4127,
    4128,
    4129,
    4130,
    4131,
    4132,
    4133,
    4134,
    4135,
    4136,
    4137,
    4138,
    4140,
    4141,
    4143,
    4144,
    4145,
    4147,
    4149,
    4150,
    4151,
    4152,
    4153,
    4154,
    4155,
    4156,
    4157,
    4158,
    4159,
    4160,
    4162,
    4163,
    4165,
    4166,
    4167,
    4168,
    4169,
    4170,
    4171,
    4174,
    4175,
    4176,
    4178,
    4179,
    4180,
    4181,
    4182,
    4184,
    4185,
    4186,
    4187,
    4188,
    4189,
    4190,
    4191,
    4192,
    4193,
    4194,
    4195,
    4196,
    4197,
    4198,
    4199,
    4200,
    4201,
    4202,
    4203,
    4204,
    4205,
    4206,
    4208,
    4209,
    4210,
    4211,
    4212,
    4213,
    4214,
    4215,
    4216,
    4217,
    4218,
    4219,
    4220,
    4221,
    4222,
    4223,
    4224,
    4225,
    4226,
    4227,
    4228,
    4229,
    4230,
    4231,
    4232,
    4233,
    4234,
    4235,
    4236,
    4237,
    4238,
    4239,
    4240,
    4241,
    4242,
    4243,
    4244,
    4245,
    4246,
    4247,
    4248,
    4249,
    4250,
    4251,
    4252,
    4253,
    4254,
    4255,
    4256,
    4257,
    4258,
    4259,
    4260,
    4261,
    4262,
    4263,
    4265,
    4266,
    4267,
    4269,
    4270,
    4271,
    4272,
    4273,
    4274,
    4275,
    4276,
    4277,
    4278,
    4279,
    4280,
    4281,
    4282,
    4283,
    4284,
    4285,
    4286,
    4287,
    4288,
    4289,
    4290,
    4291,
    4292,
    4293,
    4295,
    4296,
    4297,
    4298,
    4299,
    4300,
    4301,
    4302,
    4303,
    4304,
    4305,
    4306,
    4307,
    4308,
    4309,
    4310,
    4312,
    4313,
    4315,
    4316,
    4317,
    4318,
    4319,
    4320,
    4321,
    4322,
    4323,
    4325,
    4326,
    4327,
    4329,
    4330,
    4331,
    4332,
    4333,
    4334,
    4335,
    4336,
    4337,
    4338,
    4340,
    4341,
    4342,
    4343,
    4344,
    4345,
    4346,
    4347,
    4349,
    4350,
    4351,
    4352,
    4353,
    4354,
    4355,
    4356,
    4357,
    4358,
    4359,
    4361,
    4362,
    4363,
    4364,
    4365,
    4366,
    4367,
    4368,
    4369,
    4370,
    4371,
    4372,
    4373,
    4374,
    4375,
    4376,
    4377,
    4378,
    4380,
    4381,
    4382,
    4383,
    4384,
    4385,
    4386,
    4387,
    4388,
    4389,
    4390,
    4391,
    4392,
    4393,
    4394,
    4395,
    4396,
    4397,
    4398,
    4399,
    4400,
    4401,
    4402,
    4403,
    4404,
    4405,
    4406,
    4407,
    4408,
    4409,
    4410,
    4411,
    4412,
    4413,
    4414,
    4415,
    4416,
    4417,
    4418,
    4419,
    4420,
    4421,
    4422,
    4423,
    4424,
    4425,
    4426,
    4427,
    4428,
    4429,
    4431,
    4432,
    4433,
    4434,
    4435,
    4436,
    4437,
    4438,
    4439,
    4440,
    4441,
    4442,
    4443,
    4444,
    4445,
    4446,
    4448,
    4449,
    4450,
    4451,
    4452,
    4453,
    4454,
    4455,
    4456,
    4457,
    4458,
    4459,
    4460,
    4461,
    4462,
    4463,
    4464,
    4465,
    4466,
    4467,
    4470,
    4471,
    4472,
    4473,
    4474,
    4475,
    4476,
    4477,
    4478,
    4480,
    4481,
    4482,
    4483,
    4484,
    4485,
    4486,
    4487,
    4488,
    4489,
    4490,
    4491,
    4492,
    4493,
    4494,
    4495,
    4496,
    4497,
    4498,
    4499,
    4500,
    4501,
    4502,
    4503,
    4504,
    4505,
    4506,
    4507,
    4508,
    4509,
    4510,
    4511,
    4512,
    4513,
    4514,
    4515,
    4516,
    4517,
    4518,
    4519,
    4520,
    4521,
    4522,
    4523,
    4524,
    4526,
    4528,
    4529,
    4530,
    4531,
    4533,
    4534,
    4535,
    4536,
    4537,
    4538,
    4539,
    4540,
    4541,
    4542,
    4543,
    4544,
    4545,
    4546,
    4547,
    4548,
    4549,
    4550,
    4551,
    4552,
    4553,
    4554,
    4555,
    4556,
    4558,
    4559,
    4560,
    4561,
    4562,
    4563,
    4564,
    4565,
    4566,
    4567,
    4568,
    4569,
    4570,
    4571,
    4572,
    4573,
    4574,
    4575,
    4576,
    4577,
    4578,
    4579,
    4580,
    4581,
    4582,
    4583,
    4584,
    4585,
    4586,
    4587,
    4588,
    4589,
    4590,
    4591,
    4592,
    4593,
    4594,
    4595,
    4596,
    4597,
    4598,
    4599,
    4600,
    4601,
    4602,
    4603,
    4604,
    4605,
    4606,
    4607,
    4608,
    4609,
    4610,
    4611,
    4612,
    4613,
    4614,
    4615,
    4616,
    4617,
    4618,
    4619,
    4620,
    4622,
    4623,
    4624,
    4625,
    4626,
    4627,
    4628,
    4629,
    4631,
    4632,
    4633,
    4634,
    4635,
    4636,
    4637,
    4638,
    4639,
    4640,
    4641,
    4642,
    4643,
    4644,
    4645,
    4646,
    4647,
    4648,
    4649,
    4650,
    4651,
    4652,
    4653,
    4654,
    4655,
    4656,
    4657,
    4659,
    4660,
    4661,
    4663,
    4665,
    4666,
    4667,
    4668,
    4669,
    4671,
    4672,
    4673,
    4674,
    4675,
    4676,
    4677,
    4678,
    4679,
    4680,
    4681,
    4682,
    4683,
    4684,
    4685,
    4686,
    4687,
    4688,
    4689,
    4690,
    4691,
    4692,
    4693,
    4694,
    4696,
    4697,
    4698,
    4699,
    4700,
    4701,
    4702,
    4703,
    4704,
    4705,
    4706,
    4707,
    4708,
    4709,
    4710,
    4711,
    4713,
    4714,
    4715,
    4716,
    4717,
    4718,
    4720,
    4721,
    4722,
    4723,
    4724,
    4725,
    4726,
    4727,
    4729,
    4731,
    4732,
    4733,
    4734,
    4735,
    4736,
    4737,
    4738,
    4739,
    4740,
    4741,
    4742,
    4743,
    4744,
    4745,
    4746,
    4747,
    4748,
    4749,
    4750,
    4751,
    4752,
    4754,
    4755,
    4756,
    4757,
    4758,
    4759,
    4760,
    4761,
    4762,
    4763,
    4764,
    4765,
    4766,
    4768,
    4769,
    4770,
    4771,
    4772,
    4773,
    4774,
    4775,
    4776,
    4777,
    4778,
    4779,
    4781,
    4782,
    4783,
    4784,
    4785,
    4786,
    4788,
    4789,
    4790,
    4791,
    4792,
    4793,
    4794,
    4795,
    4796,
    4797,
    4798,
    4799,
    4800,
    4801,
    4802,
    4803,
    4804,
    4806,
    4807,
    4809,
    4810,
    4811,
    4812,
    4813,
    4814,
    4815,
    4816,
    4817,
    4818,
    4819,
    4820,
    4821,
    4822,
    4823,
    4824,
    4825,
    4826,
    4827,
    4828,
    4829,
    4830,
    4831,
    4832,
    4833,
    4834,
    4835,
    4836,
    4837,
    4838,
    4840,
    4841,
    4843,
    4844,
    4845,
    4846,
    4847,
    4848,
    4849,
    4850,
    4851,
    4852,
    4854,
    4855,
    4856,
    4857,
    4858,
    4859,
    4860,
    4861,
    4862,
    4864,
    4865,
    4867,
    4868,
    4869,
    4870,
    4872,
    4873,
    4874,
    4875,
    4876,
    4877,
    4879,
    4880,
    4881,
    4882,
    4884,
    4885,
    4886,
    4887,
    4889,
    4890,
    4891,
    4892,
    4893,
    4894,
    4895,
    4896,
    4897,
    4898,
    4899,
    4900,
    4901,
    4902,
    4903,
    4904,
    4905,
    4906,
    4907,
    4908,
    4910,
    4912,
    4913,
    4914,
    4915,
    4916,
    4917,
    4918,
    4919,
    4920,
    4921,
    4922,
    4923,
    4924,
    4925,
    4926,
    4927,
    4928,
    4930,
    4931,
    4932,
    4933,
    4934,
    4935,
    4936,
    4937,
    4938,
    4939,
    4940,
    4941,
    4942,
    4943,
    4944,
    4945,
    4946,
    4947,
    4948,
    4949,
    4950,
    4952,
    4953,
    4954,
    4956,
    4957,
    4958,
    4959,
    4960,
    4961,
    4962,
    4963,
    4965,
    4966,
    4967,
    4968,
    4969,
    4970,
    4971,
    4972,
    4973,
    4974,
    4975,
    4976,
    4978,
    4979,
    4981,
    4982,
    4983,
    4984,
    4985,
    4986,
    4987,
    4988,
    4989,
    4990,
    4991,
    4992,
    4993,
    4994,
    4995,
    4996,
    4997,
    4999,
    5000,
    5001,
    5002,
    5003,
    5004,
    5005,
    5006,
    5007,
    5008,
    5009,
    5010,
    5011,
    5012,
    5013,
    5014,
    5015,
    5016,
    5017,
    5018,
    5019,
    5020,
    5021,
    5022,
    5023,
    5024,
    5025,
    5026,
    5027,
    5028,
    5029,
    5030,
    5031,
    5032,
    5033,
    5034,
    5035,
    5036,
    5037,
    5038,
    5039,
    5040,
    5041,
    5042,
    5043,
    5044,
    5045,
    5046,
    5047,
    5048,
    5051,
    5052,
    5053,
    5054,
    5055,
    5056,
    5057,
    5058,
    5059,
    5060,
    5061,
    5062,
    5063,
    5064,
    5066,
    5067,
    5068,
    5069,
    5070,
    5071,
    5072,
    5073,
    5074,
    5076,
    5077,
    5079,
    5080,
    5081,
    5082,
    5083,
    5084,
    5085,
    5086,
    5087,
    5089,
    5090,
    5091,
    5092,
    5093,
    5094,
    5095,
    5098,
    5099,
    5101,
    5102,
    5103,
    5104,
    5105,
    5106,
    5107,
    5108,
    5109,
    5110,
    5111,
    5113,
    5114,
    5116,
    5117,
    5118,
    5119,
    5120,
    5121,
    5123,
    5124,
    5125,
    5126,
    5127,
    5128,
    5129,
    5130,
    5131,
    5132,
    5133,
    5134,
    5135,
    5136,
    5137,
    5138,
    5139,
    5140,
    5141,
    5143,
    5144,
    5145,
    5146,
    5147,
    5149,
    5150,
    5151,
    5153,
    5154,
    5155,
    5156,
    5157,
    5158,
    5160,
    5161,
    5162,
    5163,
    5164,
    5165,
    5166,
    5167,
    5168,
    5169,
    5170,
    5171,
    5172,
    5173,
    5174,
    5175,
    5176,
    5178,
    5179,
    5180,
    5181,
    5182,
    5183,
    5184,
    5185,
    5186,
    5187,
    5188,
    5190,
    5191,
    5192,
    5193,
    5194,
    5195,
    5196,
    5197,
    5198,
    5199,
    5200,
    5201,
    5202,
    5203,
    5204,
    5205,
    5206,
    5207,
    5208,
    5209,
    5210,
    5211,
    5212,
    5213,
    5214,
    5215,
    5216,
    5217,
    5218,
    5219,
    5220,
    5221,
    5222,
    5223,
    5224,
    5225,
    5226,
    5227,
    5228,
    5229,
    5230,
    5231,
    5232,
    5233,
    5234,
    5235,
    5236,
    5237,
    5238,
    5239,
    5240,
    5241,
    5242,
    5243,
    5244,
    5245,
    5246,
    5247,
    5248,
    5249,
    5250,
    5251,
    5252,
    5253,
    5254,
    5255,
    5256,
    5257,
    5258,
    5259,
    5260,
    5261,
    5262,
    5263,
    5264,
    5265,
    5266,
    5267,
    5268,
    5269,
    5270,
    5271,
    5273,
    5274,
    5275,
    5276,
    5277,
    5278,
    5279,
    5280,
    5281,
    5282,
    5283,
    5284,
    5285,
    5286,
    5287,
    5288,
    5289,
    5290,
    5291,
    5292,
    5293,
    5294,
    5295,
    5296,
    5297,
    5298,
    5299,
    5300,
    5301,
    5302,
    5304,
    5305,
    5306,
    5307,
    5309,
    5310,
    5312,
    5313,
    5314,
    5315,
    5316,
    5317,
    5319,
    5320,
    5321,
    5322,
    5323,
    5325,
    5326,
    5327,
    5328,
    5329,
    5330,
    5331,
    5332,
    5333,
    5334,
    5335,
    5336,
    5337,
    5338,
    5339,
    5340,
    5341,
    5342,
    5343,
    5344,
    5345,
    5346,
    5347,
    5348,
    5349,
    5350,
    5351,
    5352,
    5353,
    5354,
    5355,
    5356,
    5358,
    5359,
    5360,
    5361,
    5362,
    5363,
    5364,
    5365,
    5366,
    5367,
    5368,
    5369,
    5370,
    5371,
    5372,
    5373,
    5374,
    5375,
    5376,
    5377,
    5378,
    5379,
    5380,
    5382,
    5383,
    5384,
    5385,
    5386,
    5388,
    5389,
    5391,
    5392,
    5393,
    5394,
    5395,
    5396,
    5397,
    5398,
    5399,
    5400,
    5401,
    5402,
    5403,
    5404,
    5405,
    5406,
    5407,
    5408,
    5409,
    5410,
    5411,
    5412,
    5413,
    5414,
    5415,
    5416,
    5418,
    5419,
    5420,
    5421,
    5422,
    5423,
    5424,
    5425,
    5426,
    5427,
    5428,
    5429,
    5430,
    5431,
    5433,
    5434,
    5435,
    5436,
    5437,
    5438,
    5439,
    5440,
    5441,
    5442,
    5443,
    5444,
    5445,
    5446,
    5447,
    5448,
    5449,
    5450,
    5451,
    5452,
    5453,
    5454,
    5455,
    5456,
    5457,
    5458,
    5460,
    5461,
    5462,
    5463,
    5464,
    5465,
    5466,
    5467,
    5468,
    5469,
    5470,
    5471,
    5472,
    5473,
    5474,
    5475,
    5476,
    5477,
    5478,
    5480,
    5481,
    5482,
    5483,
    5484,
    5485,
    5486,
    5487,
    5488,
    5489,
    5490,
    5491,
    5492,
    5493,
    5494,
    5495,
    5496,
    5497,
    5498,
    5499,
    5501,
    5502,
    5503,
    5504,
    5505,
    5507,
    5508,
    5509,
    5510,
    5513,
    5514,
    5515,
    5516,
    5518,
    5519,
    5521,
    5523,
    5524,
    5526,
    5527,
    5528,
    5529,
    5530,
    5531,
    5532,
    5533,
    5534,
    5535,
    5536,
    5537,
    5538,
    5539,
    5540,
    5541,
    5542,
    5543,
    5544,
    5545,
    5546,
    5547,
    5548,
    5549,
    5550,
    5551,
    5552,
    5553,
    5554,
    5555,
    5556,
    5557,
    5558,
    5560,
    5561,
    5562,
    5563,
    5564,
    5565,
    5566,
    5567,
    5568,
    5569,
    5571,
    5572,
    5573,
    5574,
    5575,
    5576,
    5577,
    5579,
    5580,
    5581,
    5582,
    5583,
    5584,
    5585,
    5586,
    5587,
    5588,
    5589,
    5590,
    5591,
    5592,
    5593,
    5594,
    5595,
    5596,
    5597,
    5598,
    5599,
    5600,
    5601,
    5603,
    5604,
    5605,
    5606,
    5607,
    5608,
    5609,
    5610,
    5611,
    5612,
    5613,
    5614,
    5615,
    5616,
    5617,
    5618,
    5619,
    5620,
    5621,
    5622,
    5623,
    5625,
    5626,
    5627,
    5628,
    5629,
    5630,
    5631,
    5632,
    5633,
    5634,
    5635,
    5636,
    5637,
    5638,
    5639,
    5640,
    5641,
    5642,
    5643,
    5644,
    5645,
    5646,
    5647,
    5648,
    5649,
    5650,
    5651,
    5652,
    5653,
    5654,
    5655,
    5656,
    5657,
    5658,
    5659,
    5660,
    5661,
    5662,
    5663,
    5664,
    5666,
    5667,
    5668,
    5669,
    5670,
    5671,
    5672,
    5673,
    5674,
    5675,
    5677,
    5678,
    5679,
    5680,
    5681,
    5683,
    5684,
    5685,
    5686,
    5688,
    5689,
    5691,
    5692,
    5693,
    5694,
    5695,
    5696,
    5697,
    5698,
    5699,
    5700,
    5701,
    5702,
    5703,
    5704,
    5705,
    5706,
    5707,
    5708,
    5709,
    5710,
    5711,
    5712,
    5713,
    5714,
    5715,
    5716,
    5717,
    5719,
    5720,
    5721,
    5722,
    5723,
    5724,
    5725,
    5726,
    5727,
    5728,
    5729,
    5730,
    5731,
    5732,
    5733,
    5734,
    5735,
    5736,
    5737,
    5738,
    5739,
    5740,
    5741,
    5742,
    5743,
    5744,
    5745,
    5746,
    5747,
    5748,
    5749,
    5750,
    5751,
    5752,
    5753,
    5754,
    5755,
    5756,
    5757,
    5758,
    5759,
    5760,
    5761,
    5762,
    5763,
    5764,
    5765,
    5766,
    5767,
    5768,
    5769,
    5770,
    5771,
    5772,
    5773,
    5774,
    5775,
    5776,
    5777,
    5778,
    5779,
    5780,
    5781,
    5782,
    5783,
    5784,
    5785,
    5786,
    5787,
    5788,
    5789,
    5790,
    5791,
    5792,
    5793,
    5794,
    5795,
    5796,
    5797,
    5798,
    5799,
    5800,
    5801,
    5802,
    5803,
    5804,
    5805,
    5806,
    5807,
    5808,
    5809,
    5810,
    5811,
    5812,
    5813,
    5814,
    5815,
    5816,
    5817,
    5818,
    5819,
    5820,
    5821,
    5822,
    5823,
    5824,
    5825,
    5826,
    5827,
    5829,
    5831,
    5832,
    5833,
    5834,
    5835,
    5836,
    5837,
    5838,
    5839,
    5840,
    5841,
    5842,
    5844,
    5845,
    5847,
    5848,
    5849,
    5850,
    5851,
    5852,
    5853,
    5854,
    5855,
    5856,
    5857,
    5858,
    5859,
    5860,
    5861,
    5863,
    5864,
    5865,
    5866,
    5867,
    5868,
    5869,
    5870,
    5871,
    5872,
    5874,
    5875,
    5876,
    5877,
    5878,
    5879,
    5880,
    5881,
    5882,
    5883,
    5884,
    5885,
    5886,
    5887,
    5888,
    5889,
    5890,
    5891,
    5892,
    5893,
    5894,
    5895,
    5896,
    5898,
    5899,
    5901,
    5902,
    5903,
    5904,
    5905,
    5906,
    5907,
    5908,
    5909,
    5910,
    5911,
    5912,
    5913,
    5914,
    5915,
    5916,
    5917,
    5918,
    5919,
    5920,
    5921,
    5922,
    5923,
    5924,
    5926,
    5927,
    5928,
    5929,
    5930,
    5931,
    5932,
    5933,
    5934,
    5935,
    5936,
    5937,
    5938,
    5939,
    5940,
    5941,
    5942,
    5943,
    5944,
    5945,
    5946,
    5947,
    5948,
    5949,
    5950,
    5951,
    5952,
    5953,
    5954,
    5955,
    5956,
    5957,
    5958,
    5959,
    5960,
    5961,
    5962,
    5963,
    5964,
    5965,
    5967,
    5968,
    5969,
    5970,
    5971,
    5972,
    5974,
    5975,
    5976,
    5977,
    5978,
    5979,
    5980,
    5981,
    5982,
    5983,
    5985,
    5986,
    5987,
    5988,
    5989,
    5990,
    5991,
    5992,
    5993,
    5994,
    5995,
    5996,
    5997,
    5998,
    5999,
    6000,
    6001,
    6002,
    6003,
    6004,
    6005,
    6006,
    6007,
    6008,
    6009,
    6010,
    6011,
    6012,
    6013,
    6014,
    6015,
    6016,
    6017,
    6018,
    6019,
    6020,
    6021,
    6022,
    6023,
    6024,
    6025,
    6026,
    6027,
    6028,
    6029,
    6030,
    6031,
    6032,
    6033,
    6034,
    6035,
    6036,
    6037,
    6038,
    6039,
    6040,
    6041,
    6042,
    6043,
    6044,
    6045,
    6046,
    6047,
    6050,
    6051,
    6052,
    6053,
    6054,
    6055,
    6056,
    6057,
    6059,
    6060,
    6062,
    6064,
    6066,
    6067,
    6068,
    6069,
    6070,
    6071,
    6072,
    6073,
    6074,
    6075,
    6076,
    6077,
    6078,
    6079,
    6080,
    6081,
    6082,
    6083,
    6084,
    6085,
    6086,
    6087,
    6088,
    6089,
    6090,
    6091,
    6092,
    6093,
    6094,
    6095,
    6096,
    6097,
    6098,
    6099,
    6100,
    6101,
    6102,
    6103,
    6104,
    6105,
    6106,
    6108,
    6109,
    6110,
    6111,
    6112,
    6113,
    6114,
    6115,
    6116,
    6117,
    6118,
    6119,
    6120,
    6121,
    6122,
    6123,
    6124,
    6125,
    6126,
    6127,
    6128,
    6129,
    6130,
    6131,
    6132,
    6133,
    6134,
    6135,
    6136,
    6137,
    6138,
    6139,
    6140,
    6141,
    6142,
    6143,
    6144,
    6145,
    6147,
    6148,
    6149,
    6150,
    6151,
    6152,
    6153,
    6154,
    6155,
    6156,
    6157,
    6158,
    6159,
    6160,
    6162,
    6164,
    6165,
    6166,
    6167,
    6168,
    6169,
    6170,
    6171,
    6172,
    6173,
    6174,
    6175,
    6176,
    6177,
    6178,
    6179,
    6180,
    6181,
    6182,
    6183,
    6184,
    6185,
    6186,
    6187,
    6188,
    6190,
    6191,
    6192,
    6193,
    6194,
    6195,
    6196,
    6197,
    6198,
    6199,
    6200,
    6201,
    6202,
    6203,
    6205,
    6206,
    6207,
    6209,
    6210,
    6211,
    6212,
    6213,
    6214,
    6215,
    6216,
    6217,
    6218,
    6219,
    6220,
    6221,
    6222,
    6223,
    6224,
    6225,
    6226,
    6227,
    6228,
    6229,
    6230,
    6231,
    6232,
    6233,
    6234,
    6235,
    6236,
    6237,
    6238,
    6239,
    6240,
    6241,
    6242,
    6243,
    6244,
    6245,
    6246,
    6247,
    6248,
    6249,
    6250,
    6251,
    6252,
    6253,
    6254,
    6255,
    6256,
    6257,
    6258,
    6259,
    6260,
    6261,
    6262,
    6263,
    6264,
    6266,
    6267,
    6268,
    6269,
    6270,
    6271,
    6272,
    6273,
    6274,
    6275,
    6276,
    6277,
    6278,
    6279,
    6281,
    6282,
    6283,
    6284,
    6285,
    6286,
    6287,
    6288,
    6289,
    6290,
    6291,
    6292,
    6293,
    6294,
    6295,
    6296,
    6298,
    6299,
    6300,
    6301,
    6302,
    6303,
    6304,
    6305,
    6306,
    6307,
    6308,
    6309,
    6310,
    6311,
    6312,
    6313,
    6314,
    6315,
    6316,
    6317,
    6318,
    6319,
    6320,
    6321,
    6322,
    6324,
    6325,
    6326,
    6327,
    6329,
    6330,
    6331,
    6333,
    6334,
    6335,
    6336,
    6337,
    6338,
    6339,
    6340,
    6341,
    6342,
    6343,
    6344,
    6345,
    6346,
    6347,
    6348,
    6350,
    6351,
    6352,
    6353,
    6354,
    6355,
    6356,
    6357,
    6358,
    6359,
    6360,
    6361,
    6362,
    6363,
    6364,
    6365,
    6366,
    6367,
    6368,
    6369,
    6370,
    6371,
    6372,
    6373,
    6374,
    6375,
    6376,
    6377,
    6378,
    6379,
    6380,
    6381,
    6382,
    6383,
    6385,
    6386,
    6387,
    6388,
    6389,
    6390,
    6391,
    6392,
    6394,
    6395,
    6397,
    6398,
    6399,
    6400,
    6401,
    6402,
    6403,
    6404,
    6405,
    6406,
    6407,
    6408,
    6409,
    6410,
    6411,
    6412,
    6413,
    6414,
    6415,
    6416,
    6417,
    6418,
    6419,
    6420,
    6421,
    6422,
    6423,
    6424,
    6425,
    6426,
    6427,
    6428,
    6429,
    6430,
    6431,
    6432,
    6433,
    6434,
    6435,
    6436,
    6437,
    6438,
    6439,
    6440,
    6441,
    6442,
    6443,
    6444,
    6445,
    6446,
    6447,
    6448,
    6449,
    6450,
    6451,
    6452,
    6453,
    6454,
    6455,
    6456,
    6457,
    6459,
    6460,
    6461,
    6462,
    6463,
    6464,
    6465,
    6466,
    6467,
    6468,
    6469,
    6470,
    6471,
    6472,
    6473,
    6474,
    6475,
    6476,
    6477,
    6478,
    6479,
    6480,
    6481,
    6482,
    6483,
    6484,
    6486,
    6488,
    6489,
    6490,
    6491,
    6492,
    6493,
    6494,
    6495,
    6496,
    6497,
    6498,
    6499,
    6500,
    6501,
    6502,
    6503,
    6504,
    6505,
    6506,
    6507,
    6508,
    6509,
    6510,
    6511,
    6512,
    6513,
    6514,
    6515,
    6516,
    6517,
    6518,
    6519,
    6520,
    6521,
    6522,
    6523,
    6524,
    6525,
    6526,
    6527,
    6528,
    6529,
    6530,
    6531,
    6532,
    6533,
    6534,
    6535,
    6536,
    6537,
    6538,
    6539,
    6540,
    6541,
    6542,
    6543,
    6544,
    6545,
    6546,
    6547,
    6548,
    6549,
    6550,
    6551,
    6552,
    6553,
    6554,
    6555,
    6556,
    6557,
    6559,
    6560,
    6562,
    6563,
    6564,
    6565,
    6566,
    6567,
    6568,
    6569,
    6570,
    6571,
    6572,
    6573,
    6574,
    6575,
    6576,
    6577,
    6580,
    6581,
    6583,
    6584,
    6585,
    6586,
    6587,
    6588,
    6589,
    6590,
    6591,
    6592,
    6593,
    6594,
    6596,
    6597,
    6598,
    6599,
    6600,
    6601,
    6602,
    6603,
    6604,
    6605,
    6606,
    6607,
    6608,
    6609,
    6610,
    6611,
    6612,
    6613,
    6614,
    6615,
    6616,
    6617,
    6618,
    6619,
    6620,
    6622,
    6623,
    6624,
    6625,
    6626,
    6627,
    6628,
    6629,
    6630,
    6631,
    6632,
    6633,
    6635,
    6636,
    6637,
    6638,
    6639,
    6640,
    6641,
    6642,
    6643,
    6644,
    6645,
    6646,
    6647,
    6648,
    6649,
    6650,
    6651,
    6652,
    6653,
    6654,
    6655,
    6656,
    6657,
    6658,
    6659,
    6660,
    6661,
    6663,
    6664,
    6665,
    6666,
    6667,
    6668,
    6669,
    6670,
    6671,
    6672,
    6673,
    6674,
    6675,
    6676,
    6677,
    6678,
    6679,
    6680,
    6681,
    6682,
    6683,
    6684,
    6685,
    6686,
    6687,
    6688,
    6689,
    6690,
    6691,
    6692,
    6693,
    6694,
    6695,
    6696,
    6697,
    6698,
    6699,
    6700,
    6701,
    6702,
    6703,
    6704,
    6705,
    6706,
    6708,
    6709,
    6710,
    6711,
    6712,
    6713,
    6714,
    6715,
    6716,
    6717,
    6718,
    6719,
    6720,
    6721,
    6722,
    6723,
    6724,
    6725,
    6726,
    6728,
    6729,
    6730,
    6731,
    6732,
    6733,
    6734,
    6735,
    6736,
    6737,
    6738,
    6739,
    6740,
    6741,
    6742,
    6744,
    6745,
    6746,
    6747,
    6748,
    6749,
    6750,
    6751,
    6752,
    6753,
    6754,
    6755,
    6756,
    6757,
    6758,
    6760,
    6761,
    6762,
    6763,
    6765,
    6766,
    6767,
    6768,
    6769,
    6770,
    6771,
    6772,
    6773,
    6774,
    6775,
    6776,
    6777,
    6778,
    6779,
    6780,
    6781,
    6782,
    6783,
    6784,
    6785,
    6786,
    6787,
    6788,
    6789,
    6790,
    6791,
    6792,
    6793,
    6794,
    6795,
    6796,
    6797,
    6798,
    6799,
    6800,
    6801,
    6802,
    6803,
    6804,
    6805,
    6806,
    6807,
    6808,
    6809,
    6810,
    6811,
    6812,
    6813,
    6814,
    6816,
    6818,
    6819,
    6820,
    6821,
    6822,
    6823,
    6824,
    6825,
    6826,
    6827,
    6828,
    6829,
    6831,
    6832,
    6833,
    6834,
    6835,
    6836,
    6837,
    6838,
    6839,
    6840,
    6841,
    6842,
    6843,
    6844,
    6846,
    6847,
    6848,
    6849,
    6850,
    6851,
    6852,
    6853,
    6854,
    6855,
    6856,
    6857,
    6858,
    6859,
    6860,
    6861,
    6862,
    6863,
    6864,
    6865,
    6866,
    6867,
    6868,
    6869,
    6870,
    6871,
    6872,
    6873,
    6874,
    6875,
    6876,
    6877,
    6878,
    6879,
    6880,
    6881,
    6882,
    6883,
    6884,
    6885,
    6886,
    6887,
    6888,
    6889,
    6890,
    6891,
    6892,
    6893,
    6894,
    6895,
    6896,
    6897,
    6898,
    6899,
    6900,
    6902,
    6903,
    6904,
    6905,
    6906,
    6907,
    6908,
    6909,
    6910,
    6911,
    6912,
    6913,
    6914,
    6915,
    6916,
    6917,
    6918,
    6919,
    6920,
    6921,
    6922,
    6923,
    6924,
    6925,
    6926,
    6927,
    6928,
    6929,
    6930,
    6931,
    6932,
    6933,
    6934,
    6935,
    6936,
    6937,
    6938,
    6939,
    6940,
    6941,
    6942,
    6943,
    6944,
    6945,
    6946,
    6947,
    6949,
    6950,
    6951,
    6952,
    6953,
    6954,
    6955,
    6956,
    6957,
    6958,
    6959,
    6960,
    6963,
    6964,
    6965,
    6966,
    6967,
    6968,
    6969,
    6970,
    6971,
    6972,
    6973,
    6974,
    6975,
    6976,
    6977,
    6978,
    6979,
    6980,
    6981,
    6982,
    6983,
    6984,
    6985,
    6986,
    6987,
    6988,
    6989,
    6990,
    6991,
    6992,
    6993,
    6995,
    6996,
    6997,
    6998,
    6999,
    7000,
    7001,
    7002,
    7003,
    7005,
    7006,
    7007,
    7008,
    7009,
    7010,
    7011,
    7012,
    7013,
    7014,
    7015,
    7016,
    7017,
    7018,
    7019,
    7020,
    7021,
    7022,
    7023,
    7024,
    7025,
    7026,
    7027,
    7028,
    7029,
    7030,
    7032,
    7033,
    7034,
    7035,
    7036,
    7037,
    7038,
    7039,
    7040,
    7041,
    7042,
    7043,
    7044,
    7045,
    7046,
    7047,
    7048,
    7049,
    7050,
    7051,
    7052,
    7053,
    7054,
    7055,
    7056,
    7057,
    7058,
    7059,
    7060,
    7061,
    7063,
    7064,
    7065,
    7066,
    7067,
    7068,
    7069,
    7070,
    7071,
    7072,
    7073,
    7074,
    7076,
    7077,
    7078,
    7079,
    7080,
    7081,
    7082,
    7083,
    7084,
    7085,
    7086,
    7087,
    7088,
    7089,
    7090,
    7091,
    7092,
    7093,
    7094,
    7095,
    7096,
    7097,
    7098,
    7099,
    7100,
    7101,
    7102,
    7104,
    7105,
    7106,
    7107,
    7108,
    7109,
    7110,
    7111,
    7112,
    7113,
    7114,
    7115,
    7117,
    7118,
    7119,
    7120,
    7121,
    7122,
    7123,
    7124,
    7125,
    7126,
    7127,
    7128,
    7129,
    7130,
    7131,
    7132,
    7133,
    7134,
    7135,
    7136,
    7137,
    7138,
    7140,
    7141,
    7144,
    7145,
    7146,
    7147,
    7148,
    7149,
    7150,
    7151,
    7152,
    7153,
    7154,
    7155,
    7156,
    7157,
    7158,
    7159,
    7160,
    7163,
    7165,
    7166,
    7167,
    7168,
    7169,
    7170,
    7171,
    7172,
    7173,
    7174,
    7175,
    7176,
    7177,
    7178,
    7179,
    7180,
    7181,
    7182,
    7183,
    7184,
    7185,
    7186,
    7187,
    7188,
    7189,
    7190,
    7191,
    7192,
    7193,
    7194,
    7195,
    7196,
    7197,
    7198,
    7199,
    7200,
    7201,
    7202,
    7203,
    7204,
    7205,
    7206,
    7207,
    7208,
    7209,
    7210,
    7211,
    7212,
    7213,
    7214,
    7215,
    7216,
    7217,
    7218,
    7219,
    7220,
    7221,
    7222,
    7223,
    7224,
    7225,
    7226,
    7227,
    7228,
    7229,
    7230,
    7231,
    7232,
    7233,
    7235,
    7236,
    7237,
    7238,
    7239,
    7240,
    7241,
    7242,
    7243,
    7244,
    7245,
    7246,
    7247,
    7248,
    7249,
    7250,
    7251,
    7252,
    7253,
    7254,
    7255,
    7256,
    7257,
    7258,
    7259,
    7260,
    7261,
    7262,
    7263,
    7264,
    7265,
    7266,
    7267,
    7268,
    7269,
    7270,
    7271,
    7272,
    7273,
    7274,
    7275,
    7276,
    7277,
    7278,
    7279,
    7280,
    7281,
    7282,
    7283,
    7284,
    7285,
    7286,
    7287,
    7288,
    7289,
    7290,
    7291,
    7292,
    7293,
    7294,
    7295,
    7296,
    7297,
    7298,
    7299,
    7300,
    7301,
    7302,
    7303,
    7304,
    7305,
    7306,
    7307,
    7308,
    7309,
    7310,
    7312,
    7313,
    7314,
    7315,
    7316,
    7317,
    7318,
    7319,
    7320,
    7321,
    7322,
    7323,
    7324,
    7325,
    7326,
    7327,
    7328,
    7329,
    7330,
    7331,
    7332,
    7333,
    7334,
    7335,
    7336,
    7337,
    7338,
    7339,
    7340,
    7341,
    7342,
    7343,
    7344,
    7345,
    7346,
    7347,
    7348,
    7349,
    7350,
    7351,
    7352,
    7353,
    7354,
    7355,
    7356,
    7357,
    7358,
    7360,
    7361,
    7362,
    7363,
    7364,
    7365,
    7366,
    7367,
    7368,
    7369,
    7370,
    7371,
    7372,
    7373,
    7374,
    7375,
    7376,
    7377,
    7378,
    7379,
    7380,
    7381,
    7382,
    7383,
    7384,
    7385,
    7386,
    7387,
    7388,
    7389,
    7390,
    7391,
    7392,
    7393,
    7395,
    7396,
    7397,
    7398,
    7399,
    7400,
    7401,
    7402,
    7403,
    7404,
    7405,
    7406,
    7407,
    7408,
    7409,
    7410,
    7411,
    7412,
    7413,
    7414,
    7415,
    7416,
    7417,
    7418,
    7419,
    7420,
    7421,
    7422,
    7423,
    7424,
    7425,
    7426,
    7427,
    7428,
    7429,
    7430,
    7431,
    7432,
    7433,
    7434,
    7435,
    7436,
    7437,
    7438,
    7440,
    7441,
    7442,
    7443,
    7444,
    7445,
    7446,
    7447,
    7448,
    7449,
    7450,
    7452,
    7453,
    7454,
    7455,
    7456,
    7457,
    7458,
    7459,
    7460,
    7461,
    7462,
    7463,
    7464,
    7465,
    7466,
    7467,
    7468,
    7469,
    7471,
    7472,
    7473,
    7474,
    7475,
    7476,
    7477,
    7478,
    7479,
    7480,
    7481,
    7482,
    7483,
    7485,
    7486,
    7487,
    7488,
    7489,
    7490,
    7491,
    7492,
    7493,
    7495,
    7496,
    7497,
    7498,
    7499,
    7500,
    7501,
    7502,
    7503,
    7504,
    7505,
    7506,
    7507,
    7508,
    7510,
    7511,
    7513,
    7514,
    7515,
    7517,
    7518,
    7519,
    7520,
    7521,
    7522,
    7523,
    7524,
    7525,
    7526,
    7527,
    7528,
    7529,
    7530,
    7531,
    7532,
    7533,
    7535,
    7536,
    7537,
    7538,
    7539,
    7540,
    7541,
    7542,
    7543,
    7544,
    7545,
    7546,
    7547,
    7548,
    7549,
    7550,
    7551,
    7552,
    7553,
    7554,
    7555,
    7556,
    7557,
    7558,
    7559,
    7560,
    7561,
    7563,
    7564,
    7565,
    7567,
    7568,
    7570,
    7571,
    7572,
    7573,
    7574,
    7575,
    7576,
    7577,
    7578,
    7579,
    7580,
    7581,
    7582,
    7583,
    7584,
    7585,
    7586,
    7587,
    7588,
    7589,
    7590,
    7591,
    7592,
    7593,
    7594,
    7595,
    7596,
    7598,
    7599,
    7600,
    7602,
    7604,
    7605,
    7606,
    7607,
    7608,
    7609,
    7610,
    7611,
    7612,
    7613,
    7614,
    7615,
    7616,
    7617,
    7618,
    7619,
    7620,
    7621,
    7622,
    7623,
    7624,
    7625,
    7626,
    7627,
    7628,
    7629,
    7630,
    7631,
    7632,
    7633,
    7634,
    7635,
    7636,
    7637,
    7638,
    7639,
    7640,
    7641,
    7642,
    7643,
    7644,
    7645,
    7646,
    7647,
    7648,
    7649,
    7650,
    7651,
    7652,
    7653,
    7654,
    7655,
    7656,
    7657,
    7658,
    7659,
    7660,
    7661,
    7662,
    7663,
    7664,
    7665,
    7666,
    7667,
    7668,
    7669,
    7670,
    7671,
    7672,
    7673,
    7674,
    7676,
    7677,
    7678,
    7679,
    7680,
    7681,
    7682,
    7683,
    7684,
    7685,
    7686,
    7687,
    7688,
    7689,
    7690,
    7691,
    7692,
    7693,
    7694,
    7695,
    7696,
    7697,
    7698,
    7699,
    7700,
    7701,
    7702,
    7703,
    7704,
    7705,
    7706,
    7707,
    7708,
    7709,
    7710,
    7711,
    7712,
    7713,
    7714,
    7715,
    7716,
    7717,
    7718,
    7719,
    7720,
    7721,
    7722,
    7723,
    7725,
    7726,
    7727,
    7728,
    7729,
    7730,
    7731,
    7732,
    7733,
    7734,
    7735,
    7736,
    7737,
    7738,
    7739,
    7740,
    7741,
    7742,
    7743,
    7744,
    7745,
    7746,
    7747,
    7748,
    7749,
    7750,
    7751,
    7752,
    7753,
    7754,
    7755,
    7756,
    7757,
    7758,
    7759,
    7760,
    7761,
    7762,
    7763,
    7764,
    7765,
    7766,
    7767,
    7768,
    7769,
    7771,
    7772,
    7773,
    7774,
    7776,
    7777,
    7778,
    7779,
    7780,
    7781,
    7782,
    7783,
    7784,
    7785,
    7786,
    7787,
    7788,
    7789,
    7790,
    7791,
    7793,
    7794,
    7795,
    7796,
    7797,
    7798,
    7799,
    7800,
    7801,
    7802,
    7803,
    7804,
    7805,
    7806,
    7807,
    7808,
    7809,
    7810,
    7811,
    7812,
    7813,
    7814,
    7815,
    7816,
    7817,
    7819,
    7820,
    7821,
    7822,
    7823,
    7824,
    7825,
    7826,
    7827,
    7828,
    7829,
    7830,
    7831,
    7832,
    7833,
    7834,
    7835,
    7836,
    7837,
    7838,
    7839,
    7840,
    7841,
    7842,
    7843,
    7844,
    7845,
    7846,
    7847,
    7848,
    7849,
    7850,
    7851,
    7852,
    7853,
    7854,
    7855,
    7856,
    7857,
    7858,
    7859,
    7860,
    7861,
    7862,
    7863,
    7864,
    7865,
    7866,
    7867,
    7868,
    7869,
    7870,
    7871,
    7872,
    7873,
    7874,
    7875,
    7876,
    7877,
    7878,
    7879,
    7880,
    7882,
    7883,
    7884,
    7885,
    7886,
    7887,
    7888,
    7889,
    7890,
    7891,
    7893,
    7894,
    7895,
    7896,
    7897,
    7898,
    7899,
    7900,
    7901,
    7903,
    7904,
    7905,
    7906,
    7907,
    7908,
    7909,
    7910,
    7911,
    7912,
    7913,
    7914,
    7915,
    7916,
    7917,
    7918,
    7919,
    7920,
    7921,
    7922,
    7923,
    7924,
    7925,
    7926,
    7927,
    7928,
    7929,
    7930,
    7931,
    7932,
    7933,
    7934,
    7935,
    7936,
    7937,
    7938,
    7939,
    7940,
    7941,
    7942,
    7943,
    7944,
    7945,
    7946,
    7947,
    7949,
    7950,
    7951,
    7953,
    7954,
    7955,
    7956,
    7957,
    7958,
    7959,
    7960,
    7961,
    7962,
    7963,
    7964,
    7965,
    7966,
    7967,
    7968,
    7970,
    7971,
    7972,
    7973,
    7974,
    7975,
    7976,
    7977,
    7978,
    7979,
    7980,
    7981,
    7982,
    7983,
    7984,
    7985,
    7986,
    7987,
    7988,
    7989,
    7990,
    7991,
    7992,
    7993,
    7994,
    7995,
    7996,
    7998,
    7999,
    8000,
    8001,
    8002,
    8003,
    8004,
    8005,
    8006,
    8008,
    8009,
    8010,
    8011,
    8012,
    8013,
    8014,
    8015,
    8016,
    8017,
    8018,
    8019,
    8020,
    8021,
    8022,
    8023,
    8024,
    8025,
    8026,
    8027,
    8028,
    8029,
    8030,
    8032,
    8033,
    8034,
    8035,
    8036,
    8037,
    8038,
    8039,
    8040,
    8041,
    8042,
    8043,
    8044,
    8045,
    8046,
    8047,
    8048,
    8049,
    8050,
    8051,
    8052,
    8053,
    8054,
    8056,
    8057,
    8058,
    8059,
    8060,
    8061,
    8062,
    8063,
    8064,
    8065,
    8066,
    8067,
    8069,
    8070,
    8072,
    8073,
    8074,
    8075,
    8076,
    8077,
    8078,
    8079,
    8080,
    8081,
    8082,
    8083,
    8084,
    8085,
    8086,
    8087,
    8088,
    8089,
    8090,
    8091,
    8092,
    8093,
    8094,
    8095,
    8096,
    8097,
    8098,
    8099,
    8101,
    8102,
    8103,
    8105,
    8106,
    8107,
    8108,
    8109,
    8110,
    8111,
    8112,
    8113,
    8114,
    8115,
    8116,
    8117,
    8118,
    8120,
    8121,
    8122,
    8123,
    8124,
    8125,
    8126,
    8127,
    8128,
    8129,
    8130,
    8132,
    8133,
    8134,
    8135,
    8136,
    8137,
    8138,
    8139,
    8140,
    8141,
    8142,
    8144,
    8145,
    8146,
    8147,
    8148,
    8149,
    8150,
    8151,
    8152,
    8153,
    8154,
    8155,
    8156,
    8157,
    8158,
    8159,
    8160,
    8161,
    8162,
    8163,
    8164,
    8165,
    8166,
    8167,
    8168,
    8169,
    8170,
    8172,
    8173,
    8174,
    8175,
    8176,
    8177,
    8178,
    8179,
    8180,
    8181,
    8182,
    8183,
    8184,
    8185,
    8186,
    8187,
    8188,
    8189,
    8190,
    8191,
    8192,
    8193,
    8194,
    8195,
    8196,
    8197,
    8198,
    8199,
    8200,
    8201,
    8202,
    8203,
    8204,
    8205,
    8206,
    8207,
    8208,
    8209,
    8210,
    8211,
    8212,
    8213,
    8214,
    8215,
    8216,
    8217,
    8218,
    8219,
    8220,
    8221,
    8223,
    8224,
    8225,
    8226,
    8227,
    8229,
    8230,
    8231,
    8232,
    8233,
    8234,
    8235,
    8236,
    8237,
    8238,
    8239,
    8240,
    8241,
    8242,
    8243,
    8244,
    8245,
    8246,
    8248,
    8249,
    8250,
    8251,
    8252,
    8253,
    8254,
    8255,
    8256,
    8257,
    8258,
    8259,
    8260,
    8261,
    8262,
    8263,
    8264,
    8265,
    8266,
    8267,
    8268,
    8269,
    8270,
    8271,
    8272,
    8273,
    8274,
    8275,
    8276,
    8278,
    8279,
    8280,
    8281,
    8282,
    8283,
    8284,
    8285,
    8286,
    8287,
    8288,
    8289,
    8290,
    8291,
    8292,
    8293,
    8294,
    8295,
    8296,
    8298,
    8299,
    8300,
    8301,
    8302,
    8303,
    8304,
    8305,
    8306,
    8307,
    8308,
    8309,
    8310,
    8311,
    8312,
    8314,
    8315,
    8316,
    8317,
    8318,
    8320,
    8321,
    8322,
    8323,
    8324,
    8325,
    8326,
    8327,
    8328,
    8329,
    8331,
    8332,
    8333,
    8334,
    8335,
    8336,
    8337,
    8338,
    8339,
    8341,
    8342,
    8343,
    8344,
    8345,
    8346,
    8347,
    8348,
    8349,
    8350,
    8351,
    8352,
    8353,
    8354,
    8355,
    8356,
    8357,
    8358,
    8359,
    8360,
    8361,
    8362,
    8363,
    8364,
    8365,
    8366,
    8367,
    8368,
    8369,
    8370,
    8371,
    8372,
    8373,
    8374,
    8375,
    8376,
    8377,
    8378,
    8379,
    8380,
    8381,
    8382,
    8383,
    8384,
    8385,
    8386,
    8387,
    8388,
    8389,
    8390,
    8391,
    8392,
    8393,
    8394,
    8395,
    8396,
    8397,
    8398,
    8399,
    8400,
    8401,
    8402,
    8404,
    8406,
    8407,
    8408,
    8409,
    8410,
    8411,
    8412,
    8413,
    8414,
    8415,
    8416,
    8417,
    8418,
    8419,
    8420,
    8421,
    8423,
    8424,
    8425,
    8426,
    8427,
    8428,
    8429,
    8430,
    8432,
    8433,
    8434,
    8435,
    8436,
    8437,
    8438,
    8439,
    8440,
    8441,
    8442,
    8443,
    8444,
    8445,
    8446,
    8447,
    8448,
    8449,
    8450,
    8451,
    8452,
    8453,
    8454,
    8455,
    8456,
    8457,
    8458,
    8460,
    8461,
    8462,
    8463,
    8464,
    8465,
    8466,
    8467,
    8468,
    8469,
    8470,
    8471,
    8472,
    8473,
    8474,
    8475,
    8476,
    8478,
    8479,
    8480,
    8481,
    8482,
    8483,
    8484,
    8485,
    8486,
    8487,
    8488,
    8489,
    8490,
    8491,
    8492,
    8493,
    8494,
    8495,
    8496,
    8497,
    8498,
    8499,
    8500,
    8501,
    8502,
    8503,
    8504,
    8505,
    8506,
    8507,
    8508,
    8509,
    8510,
    8511,
    8512,
    8513,
    8514,
    8515,
    8516,
    8517,
    8518,
    8519,
    8520,
    8521,
    8522,
    8523,
    8524,
    8525,
    8526,
    8527,
    8528,
    8529,
    8530,
    8531,
    8532,
    8533,
    8534,
    8535,
    8537,
    8538,
    8539,
    8540,
    8541,
    8542,
    8543,
    8544,
    8545,
    8546,
    8547,
    8548,
    8549,
    8550,
    8551,
    8552,
    8553,
    8555,
    8556,
    8557,
    8558,
    8559,
    8560,
    8561,
    8562,
    8563,
    8564,
    8565,
    8566,
    8567,
    8568,
    8569,
    8570,
    8571,
    8572,
    8573,
    8574,
    8575,
    8576,
    8577,
    8578,
    8579,
    8580,
    8581,
    8582,
    8583,
    8584,
    8585,
    8586,
    8587,
    8588,
    8589,
    8590,
    8591,
    8592,
    8593,
    8594,
    8595,
    8596,
    8597,
    8598,
    8599,
    8600,
    8601,
    8602,
    8603,
    8604,
    8605,
    8606,
    8607,
    8608,
    8609,
    8610,
    8611,
    8613,
    8614,
    8615,
    8616,
    8617,
    8618,
    8619,
    8620,
    8621,
    8622,
    8623,
    8624,
    8625,
    8626,
    8628,
    8629,
    8630,
    8631,
    8632,
    8633,
    8635,
    8636,
    8637,
    8638,
    8639,
    8640,
    8641,
    8642,
    8643,
    8644,
    8645,
    8646,
    8647,
    8648,
    8649,
    8650,
    8651,
    8652,
    8653,
    8654,
    8655,
    8656,
    8657,
    8658,
    8659,
    8660,
    8661,
    8662,
    8663,
    8664,
    8665,
    8666,
    8667,
    8668,
    8669,
    8670,
    8671,
    8672,
    8673,
    8674,
    8675,
    8676,
    8677,
    8678,
    8679,
    8680,
    8681,
    8682,
    8683,
    8684,
    8685,
    8686,
    8687,
    8688,
    8689,
    8690,
    8691,
    8692,
    8694,
    8695,
    8696,
    8697,
    8698,
    8699,
    8701,
    8703,
    8704,
    8705,
    8706,
    8707,
    8708,
    8709,
    8710,
    8711,
    8712,
    8713,
    8714,
    8715,
    8716,
    8717,
    8718,
    8719,
    8720,
    8721,
    8722,
    8723,
    8725,
    8726,
    8727,
    8728,
    8729,
    8730,
    8731,
    8732,
    8733,
    8734,
    8735,
    8736,
    8737,
    8738,
    8739,
    8740,
    8741,
    8742,
    8743,
    8744,
    8745,
    8746,
    8747,
    8748,
    8749,
    8750,
    8751,
    8752,
    8753,
    8754,
    8755,
    8756,
    8757,
    8758,
    8759,
    8760,
    8761,
    8762,
    8763,
    8764,
    8765,
    8766,
    8767,
    8768,
    8769,
    8770,
    8771,
    8772,
    8773,
    8774,
    8775,
    8776,
    8777,
    8778,
    8779,
    8780,
    8782,
    8783,
    8784,
    8785,
    8786,
    8787,
    8788,
    8789,
    8790,
    8791,
    8792,
    8793,
    8794,
    8795,
    8796,
    8797,
    8798,
    8799,
    8800,
    8801,
    8802,
    8803,
    8804,
    8805,
    8806,
    8807,
    8808,
    8809,
    8810,
    8811,
    8812,
    8813,
    8814,
    8815,
    8816,
    8817,
    8818,
    8820,
    8821,
    8822,
    8823,
    8824,
    8825,
    8826,
    8827,
    8828,
    8829,
    8830,
    8831,
    8832,
    8833,
    8834,
    8835,
    8836,
    8837,
    8838,
    8839,
    8840,
    8841,
    8842,
    8843,
    8844,
    8845,
    8846,
    8847,
    8848,
    8849,
    8850,
    8851,
    8852,
    8853,
    8854,
    8855,
    8856,
    8858,
    8859,
    8860,
    8861,
    8862,
    8863,
    8864,
    8865,
    8866,
    8867,
    8868,
    8869,
    8870,
    8871,
    8872,
    8873,
    8874,
    8875,
    8876,
    8877,
    8878,
    8879,
    8880,
    8881,
    8882,
    8883,
    8884,
    8885,
    8886,
    8887,
    8888,
    8889,
    8891,
    8892,
    8893,
    8894,
    8895,
    8896,
    8897,
    8898,
    8899,
    8900,
    8901,
    8902,
    8903,
    8904,
    8905,
    8906,
    8907,
    8908,
    8909,
    8910,
    8911,
    8912,
    8913,
    8914,
    8915,
    8916,
    8917,
    8918,
    8919,
    8920,
    8921,
    8922,
    8923,
    8924,
    8925,
    8926,
    8927,
    8928,
    8929,
    8930,
    8931,
    8932,
    8933,
    8934,
    8935,
    8936,
    8937,
    8938,
    8939,
    8940,
    8941,
    8943,
    8944,
    8945,
    8946,
    8948,
    8949,
    8950,
    8951,
    8952,
    8953,
    8954,
    8955,
    8956,
    8957,
    8958,
    8959,
    8960,
    8961,
    8962,
    8963,
    8964,
    8965,
    8966,
    8967,
    8968,
    8969,
    8970,
    8971,
    8972,
    8973,
    8974,
    8975,
    8976,
    8977,
    8978,
    8979,
    8980,
    8981,
    8983,
    8984,
    8985,
    8986,
    8987,
    8988,
    8989,
    8990,
    8991,
    8992,
    8993,
    8994,
    8995,
    8996,
    8997,
    8998,
    8999,
    9000,
    9001,
    9002,
    9003,
    9004,
    9005,
    9006,
    9007,
    9008,
    9009,
    9010,
    9011,
    9012,
    9013,
    9014,
    9015,
    9016,
    9017,
    9018,
    9019,
    9020,
    9021,
    9022,
    9023,
    9024,
    9025,
    9026,
    9028,
    9029,
    9030,
    9031,
    9032,
    9033,
    9034,
    9035,
    9036,
    9037,
    9038,
    9039,
    9040,
    9041,
    9042,
    9043,
    9044,
    9045,
    9046,
    9047,
    9048,
    9049,
    9050,
    9051,
    9052,
    9053,
    9054,
    9055,
    9056,
    9057,
    9058,
    9059,
    9060,
    9061,
    9062,
    9063,
    9064,
    9065,
    9066,
    9067,
    9068,
    9069,
    9070,
    9071,
    9072,
    9073,
    9074,
    9075,
    9076,
    9077,
    9078,
    9079,
    9080,
    9081,
    9082,
    9083,
    9084,
    9085,
    9086,
    9087,
    9088,
    9089,
    9090,
    9091,
    9092,
    9093,
    9096,
    9097,
    9098,
    9099,
    9100,
    9101,
    9102,
    9103,
    9104,
    9105,
    9106,
    9107,
    9108,
    9109,
    9110,
    9111,
    9112,
    9114,
    9115,
    9116,
    9117,
    9118,
    9119,
    9120,
    9121,
    9122,
    9123,
    9124,
    9125,
    9126,
    9127,
    9128,
    9129,
    9130,
    9131,
    9132,
    9133,
    9134,
    9135,
    9136,
    9137,
    9138,
    9139,
    9140,
    9141,
    9142,
    9143,
    9144,
    9145,
    9146,
    9147,
    9148,
    9149,
    9150,
    9151,
    9152,
    9153,
    9154,
    9155,
    9156,
    9157,
    9158,
    9159,
    9160,
    9161,
    9162,
    9163,
    9164,
    9165,
    9166,
    9167,
    9168,
    9169,
    9170,
    9171,
    9172,
    9173,
    9174,
    9175,
    9176,
    9177,
    9178,
    9179,
    9181,
    9182,
    9183,
    9184,
    9185,
    9186,
    9187,
    9188,
    9189,
    9191,
    9192,
    9193,
    9194,
    9195,
    9196,
    9197,
    9198,
    9199,
    9200,
    9201,
    9202,
    9203,
    9204,
    9205,
    9206,
    9207,
    9208,
    9209,
    9210,
    9211,
    9212,
    9213,
    9214,
    9215,
    9216,
    9217,
    9218,
    9219,
    9220,
    9221,
    9222,
    9223,
    9224,
    9225,
    9226,
    9227,
    9228,
    9229,
    9230,
    9231,
    9232,
    9233,
    9234,
    9235,
    9236,
    9238,
    9239,
    9240,
    9241,
    9242,
    9243,
    9244,
    9245,
    9246,
    9247,
    9248,
    9249,
    9250,
    9252,
    9253,
    9254,
    9255,
    9256,
    9257,
    9258,
    9259,
    9260,
    9261,
    9262,
    9263,
    9264,
    9265,
    9266,
    9267,
    9268,
    9269,
    9270,
    9271,
    9272,
    9273,
    9274,
    9275,
    9276,
    9277,
    9278,
    9279,
    9280,
    9281,
    9282,
    9283,
    9284,
    9285,
    9286,
    9287,
    9288,
    9289,
    9290,
    9291,
    9292,
    9293,
    9294,
    9295,
    9296,
    9297,
    9298,
    9299,
    9300,
    9301,
    9302,
    9303,
    9304,
    9305,
    9306,
    9307,
    9308,
    9309,
    9310,
    9311,
    9312,
    9313,
    9314,
    9315,
    9316,
    9317,
    9318,
    9319,
    9320,
    9321,
    9322,
    9323,
    9324,
    9326,
    9327,
    9328,
    9329,
    9330,
    9331,
    9332,
    9333,
    9334,
    9335,
    9336,
    9337,
    9338,
    9339,
    9340,
    9341,
    9342,
    9343,
    9344,
    9345,
    9346,
    9347,
    9348,
    9349,
    9350,
    9351,
    9352,
    9354,
    9355,
    9356,
    9357,
    9358,
    9359,
    9360,
    9361,
    9362,
    9363,
    9364,
    9365,
    9366,
    9367,
    9368,
    9371,
    9372,
    9373,
    9374,
    9375,
    9376,
    9377,
    9378,
    9379,
    9380,
    9381,
    9382,
    9383,
    9384,
    9385,
    9386,
    9387,
    9388,
    9389,
    9390,
    9391,
    9393,
    9394,
    9395,
    9396,
    9397,
    9398,
    9399,
    9400,
    9402,
    9403,
    9404,
    9405,
    9406,
    9407,
    9408,
    9409,
    9410,
    9411,
    9412,
    9413,
    9414,
    9415,
    9417,
    9418,
    9419,
    9420,
    9421,
    9422,
    9423,
    9424,
    9425,
    9426,
    9427,
    9428,
    9429,
    9430,
    9431,
    9432,
    9433,
    9435,
    9436,
    9437,
    9438,
    9439,
    9440,
    9441,
    9442,
    9443,
    9444,
    9445,
    9446,
    9447,
    9448,
    9449,
    9450,
    9451,
    9452,
    9453,
    9454,
    9455,
    9457,
    9458,
    9459,
    9460,
    9461,
    9462,
    9464,
    9465,
    9466,
    9467,
    9468,
    9469,
    9470,
    9471,
    9472,
    9473,
    9474,
    9475,
    9476,
    9477,
    9478,
    9479,
    9480,
    9482,
    9483,
    9484,
    9485,
    9486,
    9487,
    9488,
    9489,
    9490,
    9491,
    9492,
    9493,
    9494,
    9495,
    9496,
    9497,
    9498,
    9499,
    9500,
    9501,
    9502,
    9503,
    9504,
    9505,
    9506,
    9507,
    9508,
    9509,
    9510,
    9511,
    9512,
    9513,
    9514,
    9515,
    9516,
    9518,
    9519,
    9520,
    9521,
    9522,
    9523,
    9524,
    9525,
    9526,
    9527,
    9528,
    9529,
    9530,
    9531,
    9533,
    9534,
    9535,
    9536,
    9537,
    9538,
    9539,
    9541,
    9542,
    9543,
    9544,
    9545,
    9546,
    9547,
    9548,
    9549,
    9550,
    9552,
    9553,
    9554,
    9555,
    9556,
    9557,
    9558,
    9559,
    9560,
    9561,
    9562,
    9563,
    9564,
    9565,
    9566,
    9567,
    9568,
    9569,
    9571,
    9572,
    9573,
    9574,
    9575,
    9576,
    9577,
    9578,
    9579,
    9580,
    9581,
    9582,
    9583,
    9584,
    9585,
    9586,
    9587,
    9588,
    9589,
    9590,
    9591,
    9592,
    9593,
    9594,
    9595,
    9596,
    9597,
    9598,
    9599,
    9600,
    9601,
    9602,
    9603,
    9604,
    9605,
    9606,
    9607,
    9608,
    9609,
    9610,
    9612,
    9613,
    9615,
    9616,
    9617,
    9618,
    9619,
    9620,
    9621,
    9622,
    9623,
    9624,
    9625,
    9626,
    9627,
    9628,
    9629,
    9630,
    9631,
    9632,
    9633,
    9634,
    9635,
    9636,
    9637,
    9638,
    9639,
    9640,
    9641,
    9642,
    9643,
    9644,
    9645,
    9646,
    9647,
    9648,
    9650,
    9651,
    9652,
    9653,
    9654,
    9655,
    9656,
    9657,
    9658,
    9659,
    9660,
    9661,
    9662,
    9663,
    9664,
    9665,
    9666,
    9667,
    9668,
    9669,
    9670,
    9671,
    9672,
    9673,
    9674,
    9675,
    9676,
    9677,
    9678,
    9679,
    9680,
    9681,
    9682,
    9683,
    9684,
    9685,
    9686,
    9687,
    9688,
    9689,
    9690,
    9691,
    9692,
    9693,
    9694,
    9695,
    9696,
    9697,
    9698,
    9699,
    9700,
    9701,
    9702,
    9703,
    9704,
    9705,
    9706,
    9707,
    9708,
    9709,
    9710,
    9711,
    9712,
    9713,
    9714,
    9715,
    9716,
    9717,
    9718,
    9719,
    9720,
    9721,
    9722,
    9723,
    9724,
    9725,
    9726,
    9727,
    9728,
    9729,
    9730,
    9731,
    9732,
    9733,
    9734,
    9735,
    9736,
    9737,
    9738,
    9739,
    9740,
    9741,
    9742,
    9743,
    9744,
    9745,
    9746,
    9747,
    9748,
    9749,
    9750,
    9751,
    9752,
    9753,
    9754,
    9755,
    9756,
    9757,
    9758,
    9759,
    9760,
    9761,
    9762,
    9763,
    9764,
    9765,
    9766,
    9767,
    9768,
    9769,
    9770,
    9771,
    9772,
    9773,
    9774,
    9775,
    9776,
    9777,
    9778,
    9779,
    9780,
    9781,
    9782,
    9783,
    9784,
    9785,
    9786,
    9787,
    9788,
    9789,
    9790,
    9791,
    9792,
    9793,
    9794,
    9795,
    9796,
    9797,
    9798,
    9799,
    9800,
    9801,
    9802,
    9803,
    9804,
    9805,
    9806,
    9807,
    9808,
    9809,
    9810,
    9811,
    9812,
    9813,
    9814,
    9815,
    9816,
    9817,
    9818,
    9819,
    9820,
    9821,
    9822,
    9823,
    9824,
    9826,
    9827,
    9828,
    9829,
    9831,
    9832,
    9834,
    9835,
    9836,
    9837,
    9838,
    9839,
    9840,
    9841,
    9842,
    9843,
    9844,
    9845,
    9846,
    9847,
    9848,
    9849,
    9850,
    9851,
    9852,
    9853,
    9854,
    9855,
    9856,
    9857,
    9858,
    9859,
    9860,
    9861,
    9862,
    9863,
    9864,
    9865,
    9866,
    9867,
    9868,
    9869,
    9870,
    9871,
    9872,
    9873,
    9874,
    9875,
    9876,
    9877,
    9878,
    9879,
    9880,
    9881,
    9882,
    9883,
    9884,
    9885,
    9886,
    9887,
    9888,
    9889,
    9890,
    9891,
    9892,
    9893,
    9894,
    9895,
    9896,
    9897,
    9898,
    9899,
    9900,
    9901,
    9902,
    9903,
    9904,
    9905,
    9906,
    9907,
    9908,
    9909,
    9910,
    9911,
    9912,
    9913,
    9914,
    9915,
    9916,
    9917,
    9918,
    9919,
    9920,
    9921,
    9922,
    9923,
    9924,
    9925,
    9927,
    9928,
    9929,
    9930,
    9931,
    9932,
    9933,
    9934,
    9935,
    9936,
    9937,
    9938,
    9939,
    9940,
    9942,
    9943,
    9944,
    9945,
    9946,
    9947,
    9948,
    9949,
    9950,
    9951,
    9952,
    9953,
    9954,
    9955,
    9956,
    9957,
    9958,
    9959,
    9960,
    9961,
    9962,
    9963,
    9964,
    9965,
    9966,
    9967,
    9968,
    9970,
    9971,
    9972,
    9973,
    9974,
    9975,
    9976,
    9977,
    9978,
    9979,
    9980,
    9981,
    9982,
    9983,
    9984,
    9985,
    9986,
    9987,
    9988,
    9989,
    9991,
    9992,
    9993,
    9994,
    9995,
    9996,
    9997,
    9998,
    9999,
    10000,
    10001,
    10002,
    10003,
    10004,
    10005,
    10006,
    10007,
    10008,
    10010,
    10011,
    10012,
    10013,
    10014,
    10015,
    10016,
    10017,
    10018,
    10019,
    10020,
    10021,
    10022,
    10023,
    10024,
    10025,
    10026,
    10027,
    10028,
    10029,
    10030,
    10031,
    10032,
    10033,
    10034,
    10035,
    10036,
    10037,
    10038,
    10039,
    10040,
    10041,
    10042,
    10043,
    10044,
    10045,
    10046,
    10047,
    10049,
    10050,
    10051,
    10052,
    10053,
    10054,
    10055,
    10056,
    10057,
    10058,
    10059,
    10060,
    10061,
    10062,
    10063,
    10064,
    10065,
    10066,
    10067,
    10068,
    10069,
    10070,
    10073,
    10074,
    10075,
    10076,
    10077,
    10078,
    10079,
    10081,
    10082,
    10083,
    10084,
    10085,
    10086,
    10087,
    10089,
    10090,
    10091,
    10092,
    10093,
    10094,
    10095,
    10096,
    10097,
    10099,
    10100,
    10101,
    10102,
    10103,
    10104,
    10105,
    10106,
    10107,
    10108,
    10109,
    10110,
    10111,
    10112,
    10113,
    10114,
    10115,
    10116,
    10117,
    10118,
    10119,
    10120,
    10121,
    10122,
    10124,
    10125,
    10126,
    10127,
    10128,
    10129,
    10130,
    10131,
    10132,
    10133,
    10134,
    10135,
    10136,
    10137,
    10138,
    10139,
    10140,
    10141,
    10142,
    10143,
    10144,
    10145,
    10146,
    10147,
    10148,
    10149,
    10150,
    10151,
    10152,
    10153,
    10154,
    10155,
    10156,
    10157,
    10158,
    10159,
    10160,
    10161,
    10162,
    10163,
    10164,
    10165,
    10166,
    10167,
    10168,
    10169,
    10170,
    10171,
    10172,
    10173,
    10174,
    10175,
    10176,
    10177,
    10178,
    10179,
    10180,
    10181,
    10182,
    10183,
    10184,
    10185,
    10186,
    10187,
    10188,
    10190,
    10191,
    10192,
    10193,
    10194,
    10195,
    10196,
    10197,
    10198,
    10199,
    10200,
    10201,
    10202,
    10203,
    10204,
    10205,
    10206,
    10207,
    10208,
    10210,
    10211,
    10212,
    10213,
    10214,
    10215,
    10216,
    10217,
    10218,
    10219,
    10220,
    10221,
    10222,
    10223,
    10224,
    10225,
    10226,
    10227,
    10228,
    10229,
    10230,
    10231,
    10232,
    10233,
    10234,
    10235,
    10236,
    10238,
    10239,
    10240,
    10241,
    10242,
    10243,
    10244,
    10246,
    10247,
    10248,
    10249,
    10250,
    10251,
    10252,
    10253,
    10254,
    10255,
    10256,
    10257,
    10258,
    10259,
    10260,
    10261,
    10262,
    10263,
    10264,
    10266,
    10267,
    10268,
    10269,
    10270,
    10271,
    10272,
    10273,
    10274,
    10275,
    10276,
    10277,
    10278,
    10279,
    10280,
    10281,
    10282,
    10283,
    10284,
    10285,
    10286,
    10287,
    10288,
    10289,
    10290,
    10291,
    10292,
    10293,
    10294,
    10295,
    10296,
    10297,
    10298,
    10299,
    10300,
    10301,
    10302,
    10303,
    10304,
    10305,
    10306,
    10307,
    10308,
    10309,
    10310,
    10311,
    10312,
    10313,
    10314,
    10315,
    10316,
    10317,
    10318,
    10319,
    10320,
    10322,
    10323,
    10324,
    10325,
    10326,
    10327,
    10328,
    10329,
    10330,
    10331,
    10332,
    10333,
    10334,
    10335,
    10336,
    10337,
    10338,
    10339,
    10340,
    10341,
    10342,
    10343,
    10344,
    10345,
    10346,
    10347,
    10348,
    10349,
    10350,
    10351,
    10352,
    10353,
    10354,
    10355,
    10356,
    10357,
    10358,
    10359,
    10360,
    10361,
    10362,
    10364,
    10365,
    10366,
    10367,
    10368,
    10369,
    10370,
    10371,
    10372,
    10373,
    10374,
    10375,
    10376,
    10378,
    10379,
    10380,
    10381,
    10383,
    10384,
    10386,
    10387,
    10388,
    10389,
    10390,
    10391,
    10392,
    10393,
    10394,
    10395,
    10396,
    10397,
    10398,
    10399,
    10400,
    10401,
    10402,
    10403,
    10404,
    10405,
    10406,
    10408,
    10409,
    10410,
    10411,
    10412,
    10413,
    10414,
    10415,
    10416,
    10417,
    10418,
    10419,
    10420,
    10421,
    10422,
    10423,
    10424,
    10425,
    10426,
    10427,
    10428,
    10429,
    10430,
    10431,
    10432,
    10433,
    10434,
    10435,
    10436,
    10437,
    10438,
    10439,
    10440,
    10441,
    10442,
    10443,
    10444,
    10445,
    10446,
    10447,
    10448,
    10449,
    10450,
    10451,
    10452,
    10453,
    10454,
    10455,
    10456,
    10457,
    10458,
    10459,
    10460,
    10461,
    10462,
    10463,
    10464,
    10465,
    10466,
    10467,
    10468,
    10469,
    10470,
    10471,
    10472,
    10473,
    10474,
    10475,
    10476,
    10477,
    10478,
    10479,
    10481,
    10482,
    10483,
    10484,
    10485,
    10486,
    10487,
    10488,
    10489,
    10490,
    10491,
    10492,
    10493,
    10494,
    10495,
    10496,
    10497,
    10498,
    10500,
    10501,
    10502,
    10503,
    10504,
    10505,
    10506,
    10507,
    10508,
    10509,
    10510,
    10511,
    10512,
    10513,
    10514,
    10515,
    10516,
    10517,
    10518,
    10519,
    10520,
    10521,
    10522,
    10523,
    10524,
    10525,
    10526,
    10527,
    10528,
    10529,
    10530,
    10531,
    10532,
    10534,
    10535,
    10536,
    10537,
    10538,
    10539,
    10540,
    10541,
    10542,
    10543,
    10544,
    10545,
    10546,
    10547,
    10548,
    10549,
    10550,
    10551,
    10552,
    10553,
    10554,
    10555,
    10556,
    10557,
    10558,
    10559,
    10560,
    10561,
    10562,
    10563,
    10564,
    10565,
    10566,
    10567,
    10568,
    10569,
    10570,
    10571
  ]
}
```

## Measurement — `runtime/benchmarks/overlap_casia_deep.json`

### Values

| Field | Value |
|---|---|
| `model` | w600k_r50 |
| `method` | embedding near-duplicate detection; name matching impossible (training sets carry numeric IDs, .bin packs carry no labels) |
| `thresholds.near_duplicate` | 0.9 |
| `thresholds.probable_same_identity` | 0.7 |
| `thresholds.deployed_decision` | 0.2871 |
| `overlap_found` | True |
| `verdict` | OVERLAP DETECTED - training must not proceed until resolved |
| `limitation` | Sampling proves overlap EXISTS but cannot prove it is ABSENT. |
| `results.faces_webface_112x112.sampled_images` | 105631 |
| `results.faces_webface_112x112.sampled_identities` | 10572 |
| `results.faces_webface_112x112.per_identity_cap` | 10 |
| `results.faces_webface_112x112.per_eval.lfw.near_duplicate_ge_0.90` | 0 |
| `results.faces_webface_112x112.per_eval.lfw.probable_same_id_ge_0.70` | 66 |
| `results.faces_webface_112x112.per_eval.lfw.above_deployed_thr_0.2871` | 6272 |
| `results.faces_webface_112x112.per_eval.lfw.eval_images` | 12000 |
| `results.faces_webface_112x112.per_eval.lfw.max_similarity` | 0.8193030728383814 |
| `results.faces_webface_112x112.per_eval.lfw.mean_max_similarity` | 0.3063682746607373 |
| `results.faces_webface_112x112.per_eval.agedb_30.near_duplicate_ge_0.90` | 58 |
| `results.faces_webface_112x112.per_eval.agedb_30.probable_same_id_ge_0.70` | 417 |
| `results.faces_webface_112x112.per_eval.agedb_30.above_deployed_thr_0.2871` | 8379 |
| `results.faces_webface_112x112.per_eval.agedb_30.eval_images` | 12000 |
| `results.faces_webface_112x112.per_eval.agedb_30.max_similarity` | 0.9887719477803751 |
| `results.faces_webface_112x112.per_eval.agedb_30.mean_max_similarity` | 0.3696646699368547 |
| `results.faces_webface_112x112.per_eval.cfp_fp.near_duplicate_ge_0.90` | 26 |
| `results.faces_webface_112x112.per_eval.cfp_fp.probable_same_id_ge_0.70` | 901 |
| `results.faces_webface_112x112.per_eval.cfp_fp.above_deployed_thr_0.2871` | 10270 |
| `results.faces_webface_112x112.per_eval.cfp_fp.eval_images` | 14000 |
| `results.faces_webface_112x112.per_eval.cfp_fp.max_similarity` | 0.9730783922574902 |
| `results.faces_webface_112x112.per_eval.cfp_fp.mean_max_similarity` | 0.38737383635531114 |
| `results.faces_webface_112x112.per_eval.calfw.near_duplicate_ge_0.90` | 1 |
| `results.faces_webface_112x112.per_eval.calfw.probable_same_id_ge_0.70` | 91 |
| `results.faces_webface_112x112.per_eval.calfw.above_deployed_thr_0.2871` | 6488 |
| `results.faces_webface_112x112.per_eval.calfw.eval_images` | 12000 |
| `results.faces_webface_112x112.per_eval.calfw.max_similarity` | 0.9755425606438681 |
| `results.faces_webface_112x112.per_eval.calfw.mean_max_similarity` | 0.3075146598059553 |
| `results.faces_webface_112x112.per_eval.cplfw.near_duplicate_ge_0.90` | 0 |
| `results.faces_webface_112x112.per_eval.cplfw.probable_same_id_ge_0.70` | 33 |
| `results.faces_webface_112x112.per_eval.cplfw.above_deployed_thr_0.2871` | 6558 |
| `results.faces_webface_112x112.per_eval.cplfw.eval_images` | 12000 |
| `results.faces_webface_112x112.per_eval.cplfw.max_similarity` | 0.8197945699135354 |
| `results.faces_webface_112x112.per_eval.cplfw.mean_max_similarity` | 0.31122959552493884 |

### Raw artefact

```json
{
  "model": "w600k_r50",
  "method": "embedding near-duplicate detection; name matching impossible (training sets carry numeric IDs, .bin packs carry no labels)",
  "thresholds": {
    "near_duplicate": 0.9,
    "probable_same_identity": 0.7,
    "deployed_decision": 0.2871
  },
  "overlap_found": true,
  "verdict": "OVERLAP DETECTED - training must not proceed until resolved",
  "limitation": "Sampling proves overlap EXISTS but cannot prove it is ABSENT.",
  "results": {
    "faces_webface_112x112": {
      "sampled_images": 105631,
      "sampled_identities": 10572,
      "per_identity_cap": 10,
      "per_eval": {
        "lfw": {
          "near_duplicate_ge_0.90": 0,
          "probable_same_id_ge_0.70": 66,
          "above_deployed_thr_0.2871": 6272,
          "eval_images": 12000,
          "max_similarity": 0.8193030728383814,
          "mean_max_similarity": 0.3063682746607373
        },
        "agedb_30": {
          "near_duplicate_ge_0.90": 58,
          "probable_same_id_ge_0.70": 417,
          "above_deployed_thr_0.2871": 8379,
          "eval_images": 12000,
          "max_similarity": 0.9887719477803751,
          "mean_max_similarity": 0.3696646699368547
        },
        "cfp_fp": {
          "near_duplicate_ge_0.90": 26,
          "probable_same_id_ge_0.70": 901,
          "above_deployed_thr_0.2871": 10270,
          "eval_images": 14000,
          "max_similarity": 0.9730783922574902,
          "mean_max_similarity": 0.38737383635531114
        },
        "calfw": {
          "near_duplicate_ge_0.90": 1,
          "probable_same_id_ge_0.70": 91,
          "above_deployed_thr_0.2871": 6488,
          "eval_images": 12000,
          "max_similarity": 0.9755425606438681,
          "mean_max_similarity": 0.3075146598059553
        },
        "cplfw": {
          "near_duplicate_ge_0.90": 0,
          "probable_same_id_ge_0.70": 33,
          "above_deployed_thr_0.2871": 6558,
          "eval_images": 12000,
          "max_similarity": 0.8197945699135354,
          "mean_max_similarity": 0.31122959552493884
        }
      }
    }
  }
}
```

## Measurement — `runtime/benchmarks/qmul_exclusion_list.json`

### Values

| Field | Value |
|---|---|
| `dataset` | QMUL-SurvFace training_set |
| `license` | research purposes only; images sourced from person re-identification datasets, copyright with original owners (qmul-survface.github.io) |
| `model` | w600k_r50 |
| `threshold` | 0.4 |
| `per_identity_sampled` | 40 |
| `images_sampled` | 78733 |
| `eval_sets[0]` | lfw |
| `eval_sets[1]` | agedb_30 |
| `eval_sets[2]` | cfp_fp |
| `eval_sets[3]` | calfw |
| `eval_sets[4]` | cplfw |
| `eval_sets[5]` | cfp_ff |
| `eval_sets[6]` | tinyface |
| `gallery_embeddings` | 84171 |
| `identities_total` | 5319 |
| `identities_excluded` | 5153 |
| `identities_kept` | 166 |
| `peak_similarity` | 0.9153 |
| `nearest_eval_set_tally.tinyface` | 4162 |
| `nearest_eval_set_tally.cplfw` | 965 |
| `nearest_eval_set_tally.cfp_fp` | 184 |
| `nearest_eval_set_tally.lfw` | 4 |
| `nearest_eval_set_tally.calfw` | 3 |
| `nearest_eval_set_tally.cfp_ff` | 1 |
| `threshold_sensitivity.0.30` | 5302 |
| `threshold_sensitivity.0.35` | 5242 |
| `threshold_sensitivity.0.40` | 5153 |
| `threshold_sensitivity.0.45` | 5044 |
| `threshold_sensitivity.0.50` | 4882 |
| `caveat` | Degraded probes yield weaker embeddings, compressing cosine similarity downward for true matches too. A 0.40 threshold carried over from the clean-vs-clean CASIA audit is a LOOSER filter here, not a stricter one. |
| `excluded_labels[0]` | 100 |
| `excluded_labels[1]` | 10000 |
| `excluded_labels[2]` | 10001 |
| `excluded_labels[3]` | 10004 |
| `excluded_labels[4]` | 10005 |
| `excluded_labels[5]` | 10006 |
| `excluded_labels[6]` | 10008 |
| `excluded_labels[7]` | 10009 |
| `excluded_labels[8]` | 10010 |
| `excluded_labels[9]` | 10011 |
| `excluded_labels[10]` | 10012 |
| `excluded_labels[11]` | 10014 |
| `excluded_labels[12]` | 10015 |
| `excluded_labels[13]` | 10016 |
| `excluded_labels[14]` | 10017 |
| `excluded_labels[15]` | 10018 |
| `excluded_labels[16]` | 10019 |
| `excluded_labels[17]` | 1002 |
| `excluded_labels[18]` | 10020 |
| `excluded_labels[19]` | 10023 |
| `excluded_labels[20]` | 10025 |
| `excluded_labels[21]` | 10026 |
| `excluded_labels[22]` | 10027 |
| `excluded_labels[23]` | 10030 |
| `excluded_labels[24]` | 10031 |
| `excluded_labels[25]` | 10032 |
| `excluded_labels[26]` | 10033 |
| `excluded_labels[27]` | 10034 |
| `excluded_labels[28]` | 1004 |
| `excluded_labels[29]` | 10043 |
| `excluded_labels[30]` | 10045 |
| `excluded_labels[31]` | 10046 |
| `excluded_labels[32]` | 10047 |
| `excluded_labels[33]` | 10048 |
| `excluded_labels[34]` | 10049 |
| `excluded_labels[35]` | 10050 |
| `excluded_labels[36]` | 10051 |
| `excluded_labels[37]` | 10054 |
| `excluded_labels[38]` | 10056 |
| `excluded_labels[39]` | 10057 |
| `excluded_labels[40]` | 10062 |
| `excluded_labels[41]` | 10063 |
| `excluded_labels[42]` | 10064 |
| `excluded_labels[43]` | 10065 |
| `excluded_labels[44]` | 10067 |
| `excluded_labels[45]` | 10069 |
| `excluded_labels[46]` | 1007 |
| `excluded_labels[47]` | 10070 |
| `excluded_labels[48]` | 10075 |
| `excluded_labels[49]` | 10076 |
| `excluded_labels[50]` | 10078 |
| `excluded_labels[51]` | 1008 |
| `excluded_labels[52]` | 10081 |
| `excluded_labels[53]` | 10083 |
| `excluded_labels[54]` | 10084 |
| `excluded_labels[55]` | 10085 |
| `excluded_labels[56]` | 1009 |
| `excluded_labels[57]` | 10092 |
| `excluded_labels[58]` | 10095 |
| `excluded_labels[59]` | 10096 |
| `excluded_labels[60]` | 10098 |
| `excluded_labels[61]` | 101 |
| `excluded_labels[62]` | 1010 |
| `excluded_labels[63]` | 10100 |
| `excluded_labels[64]` | 10102 |
| `excluded_labels[65]` | 10103 |
| `excluded_labels[66]` | 10105 |
| `excluded_labels[67]` | 10108 |
| `excluded_labels[68]` | 10109 |
| `excluded_labels[69]` | 1011 |
| `excluded_labels[70]` | 10110 |
| `excluded_labels[71]` | 10111 |
| `excluded_labels[72]` | 10114 |
| `excluded_labels[73]` | 10117 |
| `excluded_labels[74]` | 10121 |
| `excluded_labels[75]` | 10122 |
| `excluded_labels[76]` | 10124 |
| `excluded_labels[77]` | 10125 |
| `excluded_labels[78]` | 10126 |
| `excluded_labels[79]` | 10127 |
| `excluded_labels[80]` | 10129 |
| `excluded_labels[81]` | 1013 |
| `excluded_labels[82]` | 10131 |
| `excluded_labels[83]` | 10133 |
| `excluded_labels[84]` | 10134 |
| `excluded_labels[85]` | 10137 |
| `excluded_labels[86]` | 10141 |
| `excluded_labels[87]` | 10142 |
| `excluded_labels[88]` | 10143 |
| `excluded_labels[89]` | 10144 |
| `excluded_labels[90]` | 10145 |
| `excluded_labels[91]` | 10146 |
| `excluded_labels[92]` | 10147 |
| `excluded_labels[93]` | 10150 |
| `excluded_labels[94]` | 10152 |
| `excluded_labels[95]` | 10153 |
| `excluded_labels[96]` | 10155 |
| `excluded_labels[97]` | 10156 |
| `excluded_labels[98]` | 10157 |
| `excluded_labels[99]` | 10158 |
| `excluded_labels[100]` | 10159 |
| `excluded_labels[101]` | 10160 |
| `excluded_labels[102]` | 10161 |
| `excluded_labels[103]` | 10163 |
| `excluded_labels[104]` | 10165 |
| `excluded_labels[105]` | 10166 |
| `excluded_labels[106]` | 10169 |
| `excluded_labels[107]` | 1017 |
| `excluded_labels[108]` | 10171 |
| `excluded_labels[109]` | 10172 |
| `excluded_labels[110]` | 10173 |
| `excluded_labels[111]` | 10175 |
| `excluded_labels[112]` | 10176 |
| `excluded_labels[113]` | 10177 |
| `excluded_labels[114]` | 10178 |
| `excluded_labels[115]` | 10179 |
| `excluded_labels[116]` | 10184 |
| `excluded_labels[117]` | 10187 |
| `excluded_labels[118]` | 10188 |
| `excluded_labels[119]` | 10189 |
| `excluded_labels[120]` | 1019 |
| `excluded_labels[121]` | 10193 |
| `excluded_labels[122]` | 10198 |
| `excluded_labels[123]` | 10204 |
| `excluded_labels[124]` | 10207 |
| `excluded_labels[125]` | 10208 |
| `excluded_labels[126]` | 1021 |
| `excluded_labels[127]` | 10210 |
| `excluded_labels[128]` | 10213 |
| `excluded_labels[129]` | 10214 |
| `excluded_labels[130]` | 1022 |
| `excluded_labels[131]` | 10220 |
| `excluded_labels[132]` | 10222 |
| `excluded_labels[133]` | 10225 |
| `excluded_labels[134]` | 10227 |
| `excluded_labels[135]` | 1023 |
| `excluded_labels[136]` | 10233 |
| `excluded_labels[137]` | 10234 |
| `excluded_labels[138]` | 10235 |
| `excluded_labels[139]` | 10240 |
| `excluded_labels[140]` | 10241 |
| `excluded_labels[141]` | 10244 |
| `excluded_labels[142]` | 10245 |
| `excluded_labels[143]` | 10249 |
| `excluded_labels[144]` | 1025 |
| `excluded_labels[145]` | 10250 |
| `excluded_labels[146]` | 10251 |
| `excluded_labels[147]` | 10252 |
| `excluded_labels[148]` | 10255 |
| `excluded_labels[149]` | 10257 |
| `excluded_labels[150]` | 10259 |
| `excluded_labels[151]` | 1026 |
| `excluded_labels[152]` | 10260 |
| `excluded_labels[153]` | 10261 |
| `excluded_labels[154]` | 10265 |
| `excluded_labels[155]` | 10266 |
| `excluded_labels[156]` | 10269 |
| `excluded_labels[157]` | 10270 |
| `excluded_labels[158]` | 10271 |
| `excluded_labels[159]` | 10273 |
| `excluded_labels[160]` | 10274 |
| `excluded_labels[161]` | 10276 |
| `excluded_labels[162]` | 10279 |
| `excluded_labels[163]` | 1028 |
| `excluded_labels[164]` | 10281 |
| `excluded_labels[165]` | 10284 |
| `excluded_labels[166]` | 10286 |
| `excluded_labels[167]` | 10287 |
| `excluded_labels[168]` | 10292 |
| `excluded_labels[169]` | 10295 |
| `excluded_labels[170]` | 10297 |
| `excluded_labels[171]` | 10298 |
| `excluded_labels[172]` | 103 |
| `excluded_labels[173]` | 1030 |
| `excluded_labels[174]` | 10300 |
| `excluded_labels[175]` | 10305 |
| `excluded_labels[176]` | 10307 |
| `excluded_labels[177]` | 10308 |
| `excluded_labels[178]` | 10311 |
| `excluded_labels[179]` | 10314 |
| `excluded_labels[180]` | 10316 |
| `excluded_labels[181]` | 10318 |
| `excluded_labels[182]` | 10319 |
| `excluded_labels[183]` | 1032 |
| `excluded_labels[184]` | 10321 |
| `excluded_labels[185]` | 10322 |
| `excluded_labels[186]` | 10324 |
| `excluded_labels[187]` | 10325 |
| `excluded_labels[188]` | 10327 |
| `excluded_labels[189]` | 10328 |
| `excluded_labels[190]` | 10329 |
| `excluded_labels[191]` | 10330 |
| `excluded_labels[192]` | 10333 |
| `excluded_labels[193]` | 10334 |
| `excluded_labels[194]` | 10337 |
| `excluded_labels[195]` | 10338 |
| `excluded_labels[196]` | 10340 |
| `excluded_labels[197]` | 10342 |
| `excluded_labels[198]` | 10343 |
| `excluded_labels[199]` | 10344 |
| `excluded_labels[200]` | 10345 |
| `excluded_labels[201]` | 10349 |
| `excluded_labels[202]` | 10350 |
| `excluded_labels[203]` | 10351 |
| `excluded_labels[204]` | 10352 |
| `excluded_labels[205]` | 10353 |
| `excluded_labels[206]` | 10356 |
| `excluded_labels[207]` | 10357 |
| `excluded_labels[208]` | 10358 |
| `excluded_labels[209]` | 10359 |
| `excluded_labels[210]` | 10361 |
| `excluded_labels[211]` | 10362 |
| `excluded_labels[212]` | 10364 |
| `excluded_labels[213]` | 10365 |
| `excluded_labels[214]` | 10367 |
| `excluded_labels[215]` | 10368 |
| `excluded_labels[216]` | 10369 |
| `excluded_labels[217]` | 1037 |
| `excluded_labels[218]` | 10372 |
| `excluded_labels[219]` | 10377 |
| `excluded_labels[220]` | 10379 |
| `excluded_labels[221]` | 10380 |
| `excluded_labels[222]` | 10381 |
| `excluded_labels[223]` | 10387 |
| `excluded_labels[224]` | 10388 |
| `excluded_labels[225]` | 10389 |
| `excluded_labels[226]` | 10390 |
| `excluded_labels[227]` | 10392 |
| `excluded_labels[228]` | 10393 |
| `excluded_labels[229]` | 10394 |
| `excluded_labels[230]` | 10395 |
| `excluded_labels[231]` | 10396 |
| `excluded_labels[232]` | 10399 |
| `excluded_labels[233]` | 10400 |
| `excluded_labels[234]` | 10403 |
| `excluded_labels[235]` | 10405 |
| `excluded_labels[236]` | 10407 |
| `excluded_labels[237]` | 10409 |
| `excluded_labels[238]` | 1041 |
| `excluded_labels[239]` | 10410 |
| `excluded_labels[240]` | 1042 |
| `excluded_labels[241]` | 10421 |
| `excluded_labels[242]` | 10422 |
| `excluded_labels[243]` | 10423 |
| `excluded_labels[244]` | 10428 |
| `excluded_labels[245]` | 10429 |
| `excluded_labels[246]` | 1043 |
| `excluded_labels[247]` | 10431 |
| `excluded_labels[248]` | 10434 |
| `excluded_labels[249]` | 10435 |
| `excluded_labels[250]` | 10438 |
| `excluded_labels[251]` | 10439 |
| `excluded_labels[252]` | 1044 |
| `excluded_labels[253]` | 10440 |
| `excluded_labels[254]` | 10441 |
| `excluded_labels[255]` | 10442 |
| `excluded_labels[256]` | 10447 |
| `excluded_labels[257]` | 10450 |
| `excluded_labels[258]` | 10453 |
| `excluded_labels[259]` | 10455 |
| `excluded_labels[260]` | 10456 |
| `excluded_labels[261]` | 10457 |
| `excluded_labels[262]` | 10458 |
| `excluded_labels[263]` | 10459 |
| `excluded_labels[264]` | 10460 |
| `excluded_labels[265]` | 10461 |
| `excluded_labels[266]` | 10464 |
| `excluded_labels[267]` | 10465 |
| `excluded_labels[268]` | 1047 |
| `excluded_labels[269]` | 10470 |
| `excluded_labels[270]` | 10473 |
| `excluded_labels[271]` | 10474 |
| `excluded_labels[272]` | 10475 |
| `excluded_labels[273]` | 10476 |
| `excluded_labels[274]` | 10477 |
| `excluded_labels[275]` | 10479 |
| `excluded_labels[276]` | 10480 |
| `excluded_labels[277]` | 10483 |
| `excluded_labels[278]` | 10484 |
| `excluded_labels[279]` | 10489 |
| `excluded_labels[280]` | 1049 |
| `excluded_labels[281]` | 10490 |
| `excluded_labels[282]` | 10491 |
| `excluded_labels[283]` | 10495 |
| `excluded_labels[284]` | 10496 |
| `excluded_labels[285]` | 10498 |
| `excluded_labels[286]` | 105 |
| `excluded_labels[287]` | 10500 |
| `excluded_labels[288]` | 10501 |
| `excluded_labels[289]` | 10502 |
| `excluded_labels[290]` | 10504 |
| `excluded_labels[291]` | 10505 |
| `excluded_labels[292]` | 10506 |
| `excluded_labels[293]` | 10507 |
| `excluded_labels[294]` | 10508 |
| `excluded_labels[295]` | 10511 |
| `excluded_labels[296]` | 10512 |
| `excluded_labels[297]` | 10513 |
| `excluded_labels[298]` | 10517 |
| `excluded_labels[299]` | 10519 |
| `excluded_labels[300]` | 1052 |
| `excluded_labels[301]` | 10520 |
| `excluded_labels[302]` | 10533 |
| `excluded_labels[303]` | 10538 |
| `excluded_labels[304]` | 1054 |
| `excluded_labels[305]` | 10541 |
| `excluded_labels[306]` | 10542 |
| `excluded_labels[307]` | 10546 |
| `excluded_labels[308]` | 10547 |
| `excluded_labels[309]` | 10551 |
| `excluded_labels[310]` | 10552 |
| `excluded_labels[311]` | 10553 |
| `excluded_labels[312]` | 10554 |
| `excluded_labels[313]` | 10558 |
| `excluded_labels[314]` | 10564 |
| `excluded_labels[315]` | 10567 |
| `excluded_labels[316]` | 10569 |
| `excluded_labels[317]` | 10571 |
| `excluded_labels[318]` | 10572 |
| `excluded_labels[319]` | 10573 |
| `excluded_labels[320]` | 10578 |
| `excluded_labels[321]` | 10582 |
| `excluded_labels[322]` | 10583 |
| `excluded_labels[323]` | 10584 |
| `excluded_labels[324]` | 10587 |
| `excluded_labels[325]` | 10588 |
| `excluded_labels[326]` | 10589 |
| `excluded_labels[327]` | 10593 |
| `excluded_labels[328]` | 10596 |
| `excluded_labels[329]` | 10597 |
| `excluded_labels[330]` | 10598 |
| `excluded_labels[331]` | 10600 |
| `excluded_labels[332]` | 10608 |
| `excluded_labels[333]` | 1061 |
| `excluded_labels[334]` | 10612 |
| `excluded_labels[335]` | 10613 |
| `excluded_labels[336]` | 10614 |
| `excluded_labels[337]` | 10618 |
| `excluded_labels[338]` | 10619 |
| `excluded_labels[339]` | 1062 |
| `excluded_labels[340]` | 10620 |
| `excluded_labels[341]` | 10621 |
| `excluded_labels[342]` | 10625 |
| `excluded_labels[343]` | 10626 |
| `excluded_labels[344]` | 1063 |
| `excluded_labels[345]` | 10630 |
| `excluded_labels[346]` | 10631 |
| `excluded_labels[347]` | 10632 |
| `excluded_labels[348]` | 10633 |
| `excluded_labels[349]` | 10634 |
| `excluded_labels[350]` | 10636 |
| `excluded_labels[351]` | 1064 |
| `excluded_labels[352]` | 1065 |
| `excluded_labels[353]` | 1068 |
| `excluded_labels[354]` | 1069 |
| `excluded_labels[355]` | 107 |
| `excluded_labels[356]` | 1075 |
| `excluded_labels[357]` | 1078 |
| `excluded_labels[358]` | 1081 |
| `excluded_labels[359]` | 1091 |
| `excluded_labels[360]` | 1093 |
| `excluded_labels[361]` | 1095 |
| `excluded_labels[362]` | 1096 |
| `excluded_labels[363]` | 1097 |
| `excluded_labels[364]` | 1098 |
| `excluded_labels[365]` | 1099 |
| `excluded_labels[366]` | 11 |
| `excluded_labels[367]` | 1103 |
| `excluded_labels[368]` | 1107 |
| `excluded_labels[369]` | 1108 |
| `excluded_labels[370]` | 111 |
| `excluded_labels[371]` | 1111 |
| `excluded_labels[372]` | 1112 |
| `excluded_labels[373]` | 1114 |
| `excluded_labels[374]` | 1116 |
| `excluded_labels[375]` | 1117 |
| `excluded_labels[376]` | 112 |
| `excluded_labels[377]` | 1121 |
| `excluded_labels[378]` | 1123 |
| `excluded_labels[379]` | 1124 |
| `excluded_labels[380]` | 1125 |
| `excluded_labels[381]` | 1127 |
| `excluded_labels[382]` | 1128 |
| `excluded_labels[383]` | 113 |
| `excluded_labels[384]` | 1131 |
| `excluded_labels[385]` | 1132 |
| `excluded_labels[386]` | 1133 |
| `excluded_labels[387]` | 1134 |
| `excluded_labels[388]` | 1138 |
| `excluded_labels[389]` | 1139 |
| `excluded_labels[390]` | 1140 |
| `excluded_labels[391]` | 1141 |
| `excluded_labels[392]` | 1142 |
| `excluded_labels[393]` | 1145 |
| `excluded_labels[394]` | 1147 |
| `excluded_labels[395]` | 1148 |
| `excluded_labels[396]` | 115 |
| `excluded_labels[397]` | 1155 |
| `excluded_labels[398]` | 1159 |
| `excluded_labels[399]` | 1162 |
| `excluded_labels[400]` | 1166 |
| `excluded_labels[401]` | 1168 |
| `excluded_labels[402]` | 1170 |
| `excluded_labels[403]` | 1171 |
| `excluded_labels[404]` | 1174 |
| `excluded_labels[405]` | 1178 |
| `excluded_labels[406]` | 1179 |
| `excluded_labels[407]` | 1181 |
| `excluded_labels[408]` | 1182 |
| `excluded_labels[409]` | 1187 |
| `excluded_labels[410]` | 1188 |
| `excluded_labels[411]` | 1189 |
| `excluded_labels[412]` | 119 |
| `excluded_labels[413]` | 1190 |
| `excluded_labels[414]` | 1195 |
| `excluded_labels[415]` | 1197 |
| `excluded_labels[416]` | 1199 |
| `excluded_labels[417]` | 120 |
| `excluded_labels[418]` | 1201 |
| `excluded_labels[419]` | 1202 |
| `excluded_labels[420]` | 1204 |
| `excluded_labels[421]` | 1205 |
| `excluded_labels[422]` | 1207 |
| `excluded_labels[423]` | 1209 |
| `excluded_labels[424]` | 121 |
| `excluded_labels[425]` | 1213 |
| `excluded_labels[426]` | 1216 |
| `excluded_labels[427]` | 1221 |
| `excluded_labels[428]` | 1222 |
| `excluded_labels[429]` | 1223 |
| `excluded_labels[430]` | 1224 |
| `excluded_labels[431]` | 1225 |
| `excluded_labels[432]` | 1226 |
| `excluded_labels[433]` | 1227 |
| `excluded_labels[434]` | 1228 |
| `excluded_labels[435]` | 123 |
| `excluded_labels[436]` | 1230 |
| `excluded_labels[437]` | 1232 |
| `excluded_labels[438]` | 1233 |
| `excluded_labels[439]` | 1234 |
| `excluded_labels[440]` | 1240 |
| `excluded_labels[441]` | 1241 |
| `excluded_labels[442]` | 1242 |
| `excluded_labels[443]` | 1243 |
| `excluded_labels[444]` | 1251 |
| `excluded_labels[445]` | 1256 |
| `excluded_labels[446]` | 1257 |
| `excluded_labels[447]` | 1259 |
| `excluded_labels[448]` | 1261 |
| `excluded_labels[449]` | 1263 |
| `excluded_labels[450]` | 1264 |
| `excluded_labels[451]` | 1265 |
| `excluded_labels[452]` | 1266 |
| `excluded_labels[453]` | 127 |
| `excluded_labels[454]` | 1270 |
| `excluded_labels[455]` | 1271 |
| `excluded_labels[456]` | 1273 |
| `excluded_labels[457]` | 1276 |
| `excluded_labels[458]` | 128 |
| `excluded_labels[459]` | 1285 |
| `excluded_labels[460]` | 1286 |
| `excluded_labels[461]` | 1289 |
| `excluded_labels[462]` | 129 |
| `excluded_labels[463]` | 1293 |
| `excluded_labels[464]` | 1298 |
| `excluded_labels[465]` | 1300 |
| `excluded_labels[466]` | 1301 |
| `excluded_labels[467]` | 1305 |
| `excluded_labels[468]` | 1307 |
| `excluded_labels[469]` | 131 |
| `excluded_labels[470]` | 1311 |
| `excluded_labels[471]` | 1312 |
| `excluded_labels[472]` | 1313 |
| `excluded_labels[473]` | 1319 |
| `excluded_labels[474]` | 1322 |
| `excluded_labels[475]` | 1326 |
| `excluded_labels[476]` | 1329 |
| `excluded_labels[477]` | 133 |
| `excluded_labels[478]` | 1330 |
| `excluded_labels[479]` | 1331 |
| `excluded_labels[480]` | 1333 |
| `excluded_labels[481]` | 1336 |
| `excluded_labels[482]` | 1337 |
| `excluded_labels[483]` | 134 |
| `excluded_labels[484]` | 1342 |
| `excluded_labels[485]` | 1343 |
| `excluded_labels[486]` | 1347 |
| `excluded_labels[487]` | 1349 |
| `excluded_labels[488]` | 135 |
| `excluded_labels[489]` | 1350 |
| `excluded_labels[490]` | 1351 |
| `excluded_labels[491]` | 1353 |
| `excluded_labels[492]` | 1354 |
| `excluded_labels[493]` | 1355 |
| `excluded_labels[494]` | 1356 |
| `excluded_labels[495]` | 1359 |
| `excluded_labels[496]` | 136 |
| `excluded_labels[497]` | 1360 |
| `excluded_labels[498]` | 1365 |
| `excluded_labels[499]` | 1367 |
| `excluded_labels[500]` | 1368 |
| `excluded_labels[501]` | 1369 |
| `excluded_labels[502]` | 1370 |
| `excluded_labels[503]` | 1371 |
| `excluded_labels[504]` | 1372 |
| `excluded_labels[505]` | 1373 |
| `excluded_labels[506]` | 1375 |
| `excluded_labels[507]` | 1376 |
| `excluded_labels[508]` | 1384 |
| `excluded_labels[509]` | 1385 |
| `excluded_labels[510]` | 1386 |
| `excluded_labels[511]` | 1387 |
| `excluded_labels[512]` | 1389 |
| `excluded_labels[513]` | 1390 |
| `excluded_labels[514]` | 1392 |
| `excluded_labels[515]` | 1397 |
| `excluded_labels[516]` | 1398 |
| `excluded_labels[517]` | 14 |
| `excluded_labels[518]` | 140 |
| `excluded_labels[519]` | 1401 |
| `excluded_labels[520]` | 1405 |
| `excluded_labels[521]` | 1406 |
| `excluded_labels[522]` | 1407 |
| `excluded_labels[523]` | 1412 |
| `excluded_labels[524]` | 1415 |
| `excluded_labels[525]` | 1421 |
| `excluded_labels[526]` | 1422 |
| `excluded_labels[527]` | 1423 |
| `excluded_labels[528]` | 1426 |
| `excluded_labels[529]` | 143 |
| `excluded_labels[530]` | 1431 |
| `excluded_labels[531]` | 1432 |
| `excluded_labels[532]` | 1434 |
| `excluded_labels[533]` | 1435 |
| `excluded_labels[534]` | 1436 |
| `excluded_labels[535]` | 1438 |
| `excluded_labels[536]` | 1440 |
| `excluded_labels[537]` | 1442 |
| `excluded_labels[538]` | 1443 |
| `excluded_labels[539]` | 1445 |
| `excluded_labels[540]` | 1448 |
| `excluded_labels[541]` | 1449 |
| `excluded_labels[542]` | 145 |
| `excluded_labels[543]` | 1450 |
| `excluded_labels[544]` | 1451 |
| `excluded_labels[545]` | 1452 |
| `excluded_labels[546]` | 1457 |
| `excluded_labels[547]` | 1459 |
| `excluded_labels[548]` | 146 |
| `excluded_labels[549]` | 1461 |
| `excluded_labels[550]` | 1463 |
| `excluded_labels[551]` | 1469 |
| `excluded_labels[552]` | 147 |
| `excluded_labels[553]` | 1470 |
| `excluded_labels[554]` | 1474 |
| `excluded_labels[555]` | 1477 |
| `excluded_labels[556]` | 148 |
| `excluded_labels[557]` | 1480 |
| `excluded_labels[558]` | 1483 |
| `excluded_labels[559]` | 1488 |
| `excluded_labels[560]` | 1489 |
| `excluded_labels[561]` | 149 |
| `excluded_labels[562]` | 1492 |
| `excluded_labels[563]` | 1493 |
| `excluded_labels[564]` | 1494 |
| `excluded_labels[565]` | 1495 |
| `excluded_labels[566]` | 1496 |
| `excluded_labels[567]` | 1498 |
| `excluded_labels[568]` | 1499 |
| `excluded_labels[569]` | 150 |
| `excluded_labels[570]` | 1500 |
| `excluded_labels[571]` | 1501 |
| `excluded_labels[572]` | 1502 |
| `excluded_labels[573]` | 1503 |
| `excluded_labels[574]` | 1504 |
| `excluded_labels[575]` | 1505 |
| `excluded_labels[576]` | 1506 |
| `excluded_labels[577]` | 1507 |
| `excluded_labels[578]` | 1508 |
| `excluded_labels[579]` | 1509 |
| `excluded_labels[580]` | 151 |
| `excluded_labels[581]` | 1511 |
| `excluded_labels[582]` | 1513 |
| `excluded_labels[583]` | 1514 |
| `excluded_labels[584]` | 1515 |
| `excluded_labels[585]` | 1516 |
| `excluded_labels[586]` | 1517 |
| `excluded_labels[587]` | 1518 |
| `excluded_labels[588]` | 1519 |
| `excluded_labels[589]` | 152 |
| `excluded_labels[590]` | 1520 |
| `excluded_labels[591]` | 1521 |
| `excluded_labels[592]` | 1522 |
| `excluded_labels[593]` | 1526 |
| `excluded_labels[594]` | 1527 |
| `excluded_labels[595]` | 1528 |
| `excluded_labels[596]` | 1530 |
| `excluded_labels[597]` | 1531 |
| `excluded_labels[598]` | 1532 |
| `excluded_labels[599]` | 1533 |
| `excluded_labels[600]` | 1534 |
| `excluded_labels[601]` | 1536 |
| `excluded_labels[602]` | 1539 |
| `excluded_labels[603]` | 1546 |
| `excluded_labels[604]` | 1547 |
| `excluded_labels[605]` | 155 |
| `excluded_labels[606]` | 1551 |
| `excluded_labels[607]` | 1556 |
| `excluded_labels[608]` | 1558 |
| `excluded_labels[609]` | 156 |
| `excluded_labels[610]` | 1560 |
| `excluded_labels[611]` | 1563 |
| `excluded_labels[612]` | 1564 |
| `excluded_labels[613]` | 1565 |
| `excluded_labels[614]` | 1566 |
| `excluded_labels[615]` | 1567 |
| `excluded_labels[616]` | 1569 |
| `excluded_labels[617]` | 157 |
| `excluded_labels[618]` | 1571 |
| `excluded_labels[619]` | 1572 |
| `excluded_labels[620]` | 1573 |
| `excluded_labels[621]` | 1579 |
| `excluded_labels[622]` | 1580 |
| `excluded_labels[623]` | 1583 |
| `excluded_labels[624]` | 1587 |
| `excluded_labels[625]` | 1588 |
| `excluded_labels[626]` | 1589 |
| `excluded_labels[627]` | 159 |
| `excluded_labels[628]` | 1595 |
| `excluded_labels[629]` | 1597 |
| `excluded_labels[630]` | 1598 |
| `excluded_labels[631]` | 1599 |
| `excluded_labels[632]` | 16 |
| `excluded_labels[633]` | 160 |
| `excluded_labels[634]` | 1600 |
| `excluded_labels[635]` | 1602 |
| `excluded_labels[636]` | 1604 |
| `excluded_labels[637]` | 1605 |
| `excluded_labels[638]` | 1607 |
| `excluded_labels[639]` | 1610 |
| `excluded_labels[640]` | 1611 |
| `excluded_labels[641]` | 1615 |
| `excluded_labels[642]` | 1616 |
| `excluded_labels[643]` | 162 |
| `excluded_labels[644]` | 1624 |
| `excluded_labels[645]` | 1625 |
| `excluded_labels[646]` | 1627 |
| `excluded_labels[647]` | 163 |
| `excluded_labels[648]` | 1630 |
| `excluded_labels[649]` | 1631 |
| `excluded_labels[650]` | 1632 |
| `excluded_labels[651]` | 1634 |
| `excluded_labels[652]` | 1635 |
| `excluded_labels[653]` | 1636 |
| `excluded_labels[654]` | 1637 |
| `excluded_labels[655]` | 1638 |
| `excluded_labels[656]` | 1639 |
| `excluded_labels[657]` | 164 |
| `excluded_labels[658]` | 1640 |
| `excluded_labels[659]` | 1641 |
| `excluded_labels[660]` | 1643 |
| `excluded_labels[661]` | 1646 |
| `excluded_labels[662]` | 1647 |
| `excluded_labels[663]` | 1649 |
| `excluded_labels[664]` | 165 |
| `excluded_labels[665]` | 1652 |
| `excluded_labels[666]` | 1659 |
| `excluded_labels[667]` | 166 |
| `excluded_labels[668]` | 1663 |
| `excluded_labels[669]` | 1664 |
| `excluded_labels[670]` | 1665 |
| `excluded_labels[671]` | 1670 |
| `excluded_labels[672]` | 1673 |
| `excluded_labels[673]` | 1675 |
| `excluded_labels[674]` | 1676 |
| `excluded_labels[675]` | 1678 |
| `excluded_labels[676]` | 1685 |
| `excluded_labels[677]` | 1686 |
| `excluded_labels[678]` | 1688 |
| `excluded_labels[679]` | 1689 |
| `excluded_labels[680]` | 1690 |
| `excluded_labels[681]` | 1694 |
| `excluded_labels[682]` | 1695 |
| `excluded_labels[683]` | 1699 |
| `excluded_labels[684]` | 17 |
| `excluded_labels[685]` | 170 |
| `excluded_labels[686]` | 1700 |
| `excluded_labels[687]` | 1704 |
| `excluded_labels[688]` | 1706 |
| `excluded_labels[689]` | 1708 |
| `excluded_labels[690]` | 1709 |
| `excluded_labels[691]` | 171 |
| `excluded_labels[692]` | 1710 |
| `excluded_labels[693]` | 1713 |
| `excluded_labels[694]` | 1714 |
| `excluded_labels[695]` | 1715 |
| `excluded_labels[696]` | 1718 |
| `excluded_labels[697]` | 1719 |
| `excluded_labels[698]` | 172 |
| `excluded_labels[699]` | 1720 |
| `excluded_labels[700]` | 1723 |
| `excluded_labels[701]` | 1724 |
| `excluded_labels[702]` | 1726 |
| `excluded_labels[703]` | 1727 |
| `excluded_labels[704]` | 1729 |
| `excluded_labels[705]` | 1733 |
| `excluded_labels[706]` | 1736 |
| `excluded_labels[707]` | 1738 |
| `excluded_labels[708]` | 1739 |
| `excluded_labels[709]` | 174 |
| `excluded_labels[710]` | 1740 |
| `excluded_labels[711]` | 1743 |
| `excluded_labels[712]` | 1745 |
| `excluded_labels[713]` | 1748 |
| `excluded_labels[714]` | 175 |
| `excluded_labels[715]` | 1752 |
| `excluded_labels[716]` | 1753 |
| `excluded_labels[717]` | 1756 |
| `excluded_labels[718]` | 1757 |
| `excluded_labels[719]` | 176 |
| `excluded_labels[720]` | 1763 |
| `excluded_labels[721]` | 1764 |
| `excluded_labels[722]` | 1770 |
| `excluded_labels[723]` | 1772 |
| `excluded_labels[724]` | 1773 |
| `excluded_labels[725]` | 1774 |
| `excluded_labels[726]` | 1783 |
| `excluded_labels[727]` | 1784 |
| `excluded_labels[728]` | 1789 |
| `excluded_labels[729]` | 1792 |
| `excluded_labels[730]` | 1794 |
| `excluded_labels[731]` | 1795 |
| `excluded_labels[732]` | 1796 |
| `excluded_labels[733]` | 1797 |
| `excluded_labels[734]` | 1799 |
| `excluded_labels[735]` | 180 |
| `excluded_labels[736]` | 1800 |
| `excluded_labels[737]` | 1801 |
| `excluded_labels[738]` | 1805 |
| `excluded_labels[739]` | 1807 |
| `excluded_labels[740]` | 1809 |
| `excluded_labels[741]` | 1812 |
| `excluded_labels[742]` | 1815 |
| `excluded_labels[743]` | 1818 |
| `excluded_labels[744]` | 1819 |
| `excluded_labels[745]` | 182 |
| `excluded_labels[746]` | 1823 |
| `excluded_labels[747]` | 1828 |
| `excluded_labels[748]` | 1829 |
| `excluded_labels[749]` | 1830 |
| `excluded_labels[750]` | 1831 |
| `excluded_labels[751]` | 1832 |
| `excluded_labels[752]` | 1833 |
| `excluded_labels[753]` | 1836 |
| `excluded_labels[754]` | 1837 |
| `excluded_labels[755]` | 1838 |
| `excluded_labels[756]` | 1839 |
| `excluded_labels[757]` | 184 |
| `excluded_labels[758]` | 1840 |
| `excluded_labels[759]` | 1843 |
| `excluded_labels[760]` | 1845 |
| `excluded_labels[761]` | 1846 |
| `excluded_labels[762]` | 1847 |
| `excluded_labels[763]` | 1848 |
| `excluded_labels[764]` | 1849 |
| `excluded_labels[765]` | 1851 |
| `excluded_labels[766]` | 1855 |
| `excluded_labels[767]` | 1856 |
| `excluded_labels[768]` | 1857 |
| `excluded_labels[769]` | 186 |
| `excluded_labels[770]` | 1860 |
| `excluded_labels[771]` | 1861 |
| `excluded_labels[772]` | 1864 |
| `excluded_labels[773]` | 1867 |
| `excluded_labels[774]` | 1869 |
| `excluded_labels[775]` | 187 |
| `excluded_labels[776]` | 1870 |
| `excluded_labels[777]` | 1873 |
| `excluded_labels[778]` | 1878 |
| `excluded_labels[779]` | 1879 |
| `excluded_labels[780]` | 1881 |
| `excluded_labels[781]` | 1882 |
| `excluded_labels[782]` | 1885 |
| `excluded_labels[783]` | 1886 |
| `excluded_labels[784]` | 1887 |
| `excluded_labels[785]` | 1889 |
| `excluded_labels[786]` | 1890 |
| `excluded_labels[787]` | 1892 |
| `excluded_labels[788]` | 1895 |
| `excluded_labels[789]` | 1896 |
| `excluded_labels[790]` | 190 |
| `excluded_labels[791]` | 1901 |
| `excluded_labels[792]` | 1903 |
| `excluded_labels[793]` | 1904 |
| `excluded_labels[794]` | 1905 |
| `excluded_labels[795]` | 1910 |
| `excluded_labels[796]` | 1912 |
| `excluded_labels[797]` | 1916 |
| `excluded_labels[798]` | 1917 |
| `excluded_labels[799]` | 1925 |
| `excluded_labels[800]` | 1927 |
| `excluded_labels[801]` | 1928 |
| `excluded_labels[802]` | 193 |
| `excluded_labels[803]` | 1930 |
| `excluded_labels[804]` | 1931 |
| `excluded_labels[805]` | 1934 |
| `excluded_labels[806]` | 1935 |
| `excluded_labels[807]` | 1936 |
| `excluded_labels[808]` | 1937 |
| `excluded_labels[809]` | 1939 |
| `excluded_labels[810]` | 194 |
| `excluded_labels[811]` | 1940 |
| `excluded_labels[812]` | 1941 |
| `excluded_labels[813]` | 1942 |
| `excluded_labels[814]` | 1944 |
| `excluded_labels[815]` | 1949 |
| `excluded_labels[816]` | 195 |
| `excluded_labels[817]` | 1950 |
| `excluded_labels[818]` | 1952 |
| `excluded_labels[819]` | 1956 |
| `excluded_labels[820]` | 1958 |
| `excluded_labels[821]` | 1959 |
| `excluded_labels[822]` | 196 |
| `excluded_labels[823]` | 1967 |
| `excluded_labels[824]` | 1973 |
| `excluded_labels[825]` | 1974 |
| `excluded_labels[826]` | 1976 |
| `excluded_labels[827]` | 1977 |
| `excluded_labels[828]` | 1978 |
| `excluded_labels[829]` | 1983 |
| `excluded_labels[830]` | 1988 |
| `excluded_labels[831]` | 1990 |
| `excluded_labels[832]` | 1994 |
| `excluded_labels[833]` | 1996 |
| `excluded_labels[834]` | 1997 |
| `excluded_labels[835]` | 2000 |
| `excluded_labels[836]` | 2003 |
| `excluded_labels[837]` | 2004 |
| `excluded_labels[838]` | 2005 |
| `excluded_labels[839]` | 2006 |
| `excluded_labels[840]` | 2007 |
| `excluded_labels[841]` | 2009 |
| `excluded_labels[842]` | 2011 |
| `excluded_labels[843]` | 2013 |
| `excluded_labels[844]` | 2014 |
| `excluded_labels[845]` | 2016 |
| `excluded_labels[846]` | 2017 |
| `excluded_labels[847]` | 2018 |
| `excluded_labels[848]` | 2019 |
| `excluded_labels[849]` | 202 |
| `excluded_labels[850]` | 2021 |
| `excluded_labels[851]` | 2025 |
| `excluded_labels[852]` | 2027 |
| `excluded_labels[853]` | 2030 |
| `excluded_labels[854]` | 2036 |
| `excluded_labels[855]` | 2037 |
| `excluded_labels[856]` | 2038 |
| `excluded_labels[857]` | 2041 |
| `excluded_labels[858]` | 2042 |
| `excluded_labels[859]` | 2046 |
| `excluded_labels[860]` | 2049 |
| `excluded_labels[861]` | 205 |
| `excluded_labels[862]` | 2050 |
| `excluded_labels[863]` | 2054 |
| `excluded_labels[864]` | 2057 |
| `excluded_labels[865]` | 2058 |
| `excluded_labels[866]` | 2064 |
| `excluded_labels[867]` | 2072 |
| `excluded_labels[868]` | 2073 |
| `excluded_labels[869]` | 2075 |
| `excluded_labels[870]` | 2078 |
| `excluded_labels[871]` | 2080 |
| `excluded_labels[872]` | 2082 |
| `excluded_labels[873]` | 2083 |
| `excluded_labels[874]` | 2085 |
| `excluded_labels[875]` | 2086 |
| `excluded_labels[876]` | 2087 |
| `excluded_labels[877]` | 2088 |
| `excluded_labels[878]` | 2091 |
| `excluded_labels[879]` | 2092 |
| `excluded_labels[880]` | 2094 |
| `excluded_labels[881]` | 2099 |
| `excluded_labels[882]` | 2105 |
| `excluded_labels[883]` | 2109 |
| `excluded_labels[884]` | 211 |
| `excluded_labels[885]` | 2111 |
| `excluded_labels[886]` | 2116 |
| `excluded_labels[887]` | 2117 |
| `excluded_labels[888]` | 2124 |
| `excluded_labels[889]` | 2127 |
| `excluded_labels[890]` | 213 |
| `excluded_labels[891]` | 2130 |
| `excluded_labels[892]` | 2132 |
| `excluded_labels[893]` | 2133 |
| `excluded_labels[894]` | 2134 |
| `excluded_labels[895]` | 2135 |
| `excluded_labels[896]` | 2136 |
| `excluded_labels[897]` | 2138 |
| `excluded_labels[898]` | 2139 |
| `excluded_labels[899]` | 214 |
| `excluded_labels[900]` | 2140 |
| `excluded_labels[901]` | 2142 |
| `excluded_labels[902]` | 2149 |
| `excluded_labels[903]` | 2150 |
| `excluded_labels[904]` | 2151 |
| `excluded_labels[905]` | 2158 |
| `excluded_labels[906]` | 216 |
| `excluded_labels[907]` | 2160 |
| `excluded_labels[908]` | 2161 |
| `excluded_labels[909]` | 2162 |
| `excluded_labels[910]` | 2163 |
| `excluded_labels[911]` | 2168 |
| `excluded_labels[912]` | 2169 |
| `excluded_labels[913]` | 217 |
| `excluded_labels[914]` | 2170 |
| `excluded_labels[915]` | 2173 |
| `excluded_labels[916]` | 2174 |
| `excluded_labels[917]` | 2175 |
| `excluded_labels[918]` | 2178 |
| `excluded_labels[919]` | 2179 |
| `excluded_labels[920]` | 218 |
| `excluded_labels[921]` | 2180 |
| `excluded_labels[922]` | 2182 |
| `excluded_labels[923]` | 2184 |
| `excluded_labels[924]` | 219 |
| `excluded_labels[925]` | 2190 |
| `excluded_labels[926]` | 2192 |
| `excluded_labels[927]` | 2193 |
| `excluded_labels[928]` | 2194 |
| `excluded_labels[929]` | 2196 |
| `excluded_labels[930]` | 22 |
| `excluded_labels[931]` | 220 |
| `excluded_labels[932]` | 2205 |
| `excluded_labels[933]` | 2209 |
| `excluded_labels[934]` | 221 |
| `excluded_labels[935]` | 2210 |
| `excluded_labels[936]` | 2211 |
| `excluded_labels[937]` | 2215 |
| `excluded_labels[938]` | 2219 |
| `excluded_labels[939]` | 2220 |
| `excluded_labels[940]` | 2224 |
| `excluded_labels[941]` | 2225 |
| `excluded_labels[942]` | 2228 |
| `excluded_labels[943]` | 2229 |
| `excluded_labels[944]` | 223 |
| `excluded_labels[945]` | 2230 |
| `excluded_labels[946]` | 2235 |
| `excluded_labels[947]` | 2238 |
| `excluded_labels[948]` | 2239 |
| `excluded_labels[949]` | 224 |
| `excluded_labels[950]` | 2240 |
| `excluded_labels[951]` | 2241 |
| `excluded_labels[952]` | 2246 |
| `excluded_labels[953]` | 2248 |
| `excluded_labels[954]` | 2249 |
| `excluded_labels[955]` | 2250 |
| `excluded_labels[956]` | 2251 |
| `excluded_labels[957]` | 2253 |
| `excluded_labels[958]` | 2261 |
| `excluded_labels[959]` | 2263 |
| `excluded_labels[960]` | 2264 |
| `excluded_labels[961]` | 2266 |
| `excluded_labels[962]` | 2267 |
| `excluded_labels[963]` | 2268 |
| `excluded_labels[964]` | 227 |
| `excluded_labels[965]` | 2270 |
| `excluded_labels[966]` | 2271 |
| `excluded_labels[967]` | 2277 |
| `excluded_labels[968]` | 2279 |
| `excluded_labels[969]` | 2281 |
| `excluded_labels[970]` | 2283 |
| `excluded_labels[971]` | 2286 |
| `excluded_labels[972]` | 2287 |
| `excluded_labels[973]` | 2288 |
| `excluded_labels[974]` | 229 |
| `excluded_labels[975]` | 2291 |
| `excluded_labels[976]` | 2294 |
| `excluded_labels[977]` | 2295 |
| `excluded_labels[978]` | 2296 |
| `excluded_labels[979]` | 2298 |
| `excluded_labels[980]` | 2299 |
| `excluded_labels[981]` | 230 |
| `excluded_labels[982]` | 2300 |
| `excluded_labels[983]` | 2307 |
| `excluded_labels[984]` | 2309 |
| `excluded_labels[985]` | 2310 |
| `excluded_labels[986]` | 2311 |
| `excluded_labels[987]` | 2314 |
| `excluded_labels[988]` | 2315 |
| `excluded_labels[989]` | 232 |
| `excluded_labels[990]` | 2320 |
| `excluded_labels[991]` | 2321 |
| `excluded_labels[992]` | 2325 |
| `excluded_labels[993]` | 2327 |
| `excluded_labels[994]` | 2330 |
| `excluded_labels[995]` | 2331 |
| `excluded_labels[996]` | 2333 |
| `excluded_labels[997]` | 2334 |
| `excluded_labels[998]` | 2336 |
| `excluded_labels[999]` | 2339 |
| `excluded_labels[1000]` | 2342 |
| `excluded_labels[1001]` | 2344 |
| `excluded_labels[1002]` | 2348 |
| `excluded_labels[1003]` | 2349 |
| `excluded_labels[1004]` | 235 |
| `excluded_labels[1005]` | 2354 |
| `excluded_labels[1006]` | 2355 |
| `excluded_labels[1007]` | 2356 |
| `excluded_labels[1008]` | 2357 |
| `excluded_labels[1009]` | 2358 |
| `excluded_labels[1010]` | 2359 |
| `excluded_labels[1011]` | 236 |
| `excluded_labels[1012]` | 2361 |
| `excluded_labels[1013]` | 2362 |
| `excluded_labels[1014]` | 2363 |
| `excluded_labels[1015]` | 2364 |
| `excluded_labels[1016]` | 2365 |
| `excluded_labels[1017]` | 2367 |
| `excluded_labels[1018]` | 2368 |
| `excluded_labels[1019]` | 2369 |
| `excluded_labels[1020]` | 237 |
| `excluded_labels[1021]` | 2370 |
| `excluded_labels[1022]` | 2373 |
| `excluded_labels[1023]` | 2375 |
| `excluded_labels[1024]` | 2379 |
| `excluded_labels[1025]` | 2381 |
| `excluded_labels[1026]` | 2382 |
| `excluded_labels[1027]` | 2385 |
| `excluded_labels[1028]` | 2388 |
| `excluded_labels[1029]` | 239 |
| `excluded_labels[1030]` | 2390 |
| `excluded_labels[1031]` | 2391 |
| `excluded_labels[1032]` | 2392 |
| `excluded_labels[1033]` | 2394 |
| `excluded_labels[1034]` | 2395 |
| `excluded_labels[1035]` | 2397 |
| `excluded_labels[1036]` | 2403 |
| `excluded_labels[1037]` | 2406 |
| `excluded_labels[1038]` | 241 |
| `excluded_labels[1039]` | 2413 |
| `excluded_labels[1040]` | 2414 |
| `excluded_labels[1041]` | 2416 |
| `excluded_labels[1042]` | 2418 |
| `excluded_labels[1043]` | 242 |
| `excluded_labels[1044]` | 2420 |
| `excluded_labels[1045]` | 2421 |
| `excluded_labels[1046]` | 2423 |
| `excluded_labels[1047]` | 2427 |
| `excluded_labels[1048]` | 2429 |
| `excluded_labels[1049]` | 2430 |
| `excluded_labels[1050]` | 2431 |
| `excluded_labels[1051]` | 2432 |
| `excluded_labels[1052]` | 2433 |
| `excluded_labels[1053]` | 2435 |
| `excluded_labels[1054]` | 2437 |
| `excluded_labels[1055]` | 2439 |
| `excluded_labels[1056]` | 2442 |
| `excluded_labels[1057]` | 2443 |
| `excluded_labels[1058]` | 2446 |
| `excluded_labels[1059]` | 2452 |
| `excluded_labels[1060]` | 2453 |
| `excluded_labels[1061]` | 2454 |
| `excluded_labels[1062]` | 2455 |
| `excluded_labels[1063]` | 2457 |
| `excluded_labels[1064]` | 2458 |
| `excluded_labels[1065]` | 2459 |
| `excluded_labels[1066]` | 2461 |
| `excluded_labels[1067]` | 2462 |
| `excluded_labels[1068]` | 2465 |
| `excluded_labels[1069]` | 2466 |
| `excluded_labels[1070]` | 2467 |
| `excluded_labels[1071]` | 2469 |
| `excluded_labels[1072]` | 247 |
| `excluded_labels[1073]` | 2476 |
| `excluded_labels[1074]` | 2479 |
| `excluded_labels[1075]` | 2480 |
| `excluded_labels[1076]` | 2483 |
| `excluded_labels[1077]` | 2484 |
| `excluded_labels[1078]` | 2486 |
| `excluded_labels[1079]` | 2489 |
| `excluded_labels[1080]` | 249 |
| `excluded_labels[1081]` | 2493 |
| `excluded_labels[1082]` | 2494 |
| `excluded_labels[1083]` | 2499 |
| `excluded_labels[1084]` | 2506 |
| `excluded_labels[1085]` | 2513 |
| `excluded_labels[1086]` | 2514 |
| `excluded_labels[1087]` | 2515 |
| `excluded_labels[1088]` | 2516 |
| `excluded_labels[1089]` | 2517 |
| `excluded_labels[1090]` | 2519 |
| `excluded_labels[1091]` | 2522 |
| `excluded_labels[1092]` | 2523 |
| `excluded_labels[1093]` | 2526 |
| `excluded_labels[1094]` | 2529 |
| `excluded_labels[1095]` | 2532 |
| `excluded_labels[1096]` | 2533 |
| `excluded_labels[1097]` | 2536 |
| `excluded_labels[1098]` | 254 |
| `excluded_labels[1099]` | 2540 |
| `excluded_labels[1100]` | 2543 |
| `excluded_labels[1101]` | 2546 |
| `excluded_labels[1102]` | 2547 |
| `excluded_labels[1103]` | 2548 |
| `excluded_labels[1104]` | 2549 |
| `excluded_labels[1105]` | 255 |
| `excluded_labels[1106]` | 2550 |
| `excluded_labels[1107]` | 2552 |
| `excluded_labels[1108]` | 2557 |
| `excluded_labels[1109]` | 2558 |
| `excluded_labels[1110]` | 2562 |
| `excluded_labels[1111]` | 2563 |
| `excluded_labels[1112]` | 2565 |
| `excluded_labels[1113]` | 2570 |
| `excluded_labels[1114]` | 2571 |
| `excluded_labels[1115]` | 2572 |
| `excluded_labels[1116]` | 2573 |
| `excluded_labels[1117]` | 2576 |
| `excluded_labels[1118]` | 2577 |
| `excluded_labels[1119]` | 2579 |
| `excluded_labels[1120]` | 258 |
| `excluded_labels[1121]` | 2580 |
| `excluded_labels[1122]` | 2582 |
| `excluded_labels[1123]` | 2584 |
| `excluded_labels[1124]` | 2585 |
| `excluded_labels[1125]` | 259 |
| `excluded_labels[1126]` | 2590 |
| `excluded_labels[1127]` | 2591 |
| `excluded_labels[1128]` | 2593 |
| `excluded_labels[1129]` | 2596 |
| `excluded_labels[1130]` | 2598 |
| `excluded_labels[1131]` | 2601 |
| `excluded_labels[1132]` | 2604 |
| `excluded_labels[1133]` | 2609 |
| `excluded_labels[1134]` | 261 |
| `excluded_labels[1135]` | 2610 |
| `excluded_labels[1136]` | 2612 |
| `excluded_labels[1137]` | 2616 |
| `excluded_labels[1138]` | 2617 |
| `excluded_labels[1139]` | 2619 |
| `excluded_labels[1140]` | 2621 |
| `excluded_labels[1141]` | 2623 |
| `excluded_labels[1142]` | 2627 |
| `excluded_labels[1143]` | 2628 |
| `excluded_labels[1144]` | 2631 |
| `excluded_labels[1145]` | 2634 |
| `excluded_labels[1146]` | 2636 |
| `excluded_labels[1147]` | 2637 |
| `excluded_labels[1148]` | 2640 |
| `excluded_labels[1149]` | 2642 |
| `excluded_labels[1150]` | 2643 |
| `excluded_labels[1151]` | 2644 |
| `excluded_labels[1152]` | 2645 |
| `excluded_labels[1153]` | 2646 |
| `excluded_labels[1154]` | 2647 |
| `excluded_labels[1155]` | 2648 |
| `excluded_labels[1156]` | 2649 |
| `excluded_labels[1157]` | 265 |
| `excluded_labels[1158]` | 2650 |
| `excluded_labels[1159]` | 2652 |
| `excluded_labels[1160]` | 2654 |
| `excluded_labels[1161]` | 2655 |
| `excluded_labels[1162]` | 2656 |
| `excluded_labels[1163]` | 2657 |
| `excluded_labels[1164]` | 2659 |
| `excluded_labels[1165]` | 266 |
| `excluded_labels[1166]` | 2668 |
| `excluded_labels[1167]` | 2669 |
| `excluded_labels[1168]` | 267 |
| `excluded_labels[1169]` | 2674 |
| `excluded_labels[1170]` | 2675 |
| `excluded_labels[1171]` | 2676 |
| `excluded_labels[1172]` | 2677 |
| `excluded_labels[1173]` | 2678 |
| `excluded_labels[1174]` | 2679 |
| `excluded_labels[1175]` | 2680 |
| `excluded_labels[1176]` | 2684 |
| `excluded_labels[1177]` | 2685 |
| `excluded_labels[1178]` | 2696 |
| `excluded_labels[1179]` | 2697 |
| `excluded_labels[1180]` | 2699 |
| `excluded_labels[1181]` | 2700 |
| `excluded_labels[1182]` | 2701 |
| `excluded_labels[1183]` | 2703 |
| `excluded_labels[1184]` | 2706 |
| `excluded_labels[1185]` | 2707 |
| `excluded_labels[1186]` | 2708 |
| `excluded_labels[1187]` | 271 |
| `excluded_labels[1188]` | 2711 |
| `excluded_labels[1189]` | 2712 |
| `excluded_labels[1190]` | 2717 |
| `excluded_labels[1191]` | 2720 |
| `excluded_labels[1192]` | 2723 |
| `excluded_labels[1193]` | 2724 |
| `excluded_labels[1194]` | 2725 |
| `excluded_labels[1195]` | 2727 |
| `excluded_labels[1196]` | 2728 |
| `excluded_labels[1197]` | 2729 |
| `excluded_labels[1198]` | 273 |
| `excluded_labels[1199]` | 2730 |
| `excluded_labels[1200]` | 2731 |
| `excluded_labels[1201]` | 2734 |
| `excluded_labels[1202]` | 2736 |
| `excluded_labels[1203]` | 2739 |
| `excluded_labels[1204]` | 2740 |
| `excluded_labels[1205]` | 2744 |
| `excluded_labels[1206]` | 2745 |
| `excluded_labels[1207]` | 2746 |
| `excluded_labels[1208]` | 2748 |
| `excluded_labels[1209]` | 2749 |
| `excluded_labels[1210]` | 2751 |
| `excluded_labels[1211]` | 2758 |
| `excluded_labels[1212]` | 2759 |
| `excluded_labels[1213]` | 2760 |
| `excluded_labels[1214]` | 2763 |
| `excluded_labels[1215]` | 2764 |
| `excluded_labels[1216]` | 2766 |
| `excluded_labels[1217]` | 2767 |
| `excluded_labels[1218]` | 2768 |
| `excluded_labels[1219]` | 2769 |
| `excluded_labels[1220]` | 2770 |
| `excluded_labels[1221]` | 2771 |
| `excluded_labels[1222]` | 2772 |
| `excluded_labels[1223]` | 2774 |
| `excluded_labels[1224]` | 2775 |
| `excluded_labels[1225]` | 2778 |
| `excluded_labels[1226]` | 2779 |
| `excluded_labels[1227]` | 278 |
| `excluded_labels[1228]` | 2780 |
| `excluded_labels[1229]` | 2781 |
| `excluded_labels[1230]` | 2783 |
| `excluded_labels[1231]` | 2791 |
| `excluded_labels[1232]` | 2792 |
| `excluded_labels[1233]` | 2793 |
| `excluded_labels[1234]` | 2795 |
| `excluded_labels[1235]` | 28 |
| `excluded_labels[1236]` | 280 |
| `excluded_labels[1237]` | 2801 |
| `excluded_labels[1238]` | 2802 |
| `excluded_labels[1239]` | 2804 |
| `excluded_labels[1240]` | 2805 |
| `excluded_labels[1241]` | 2806 |
| `excluded_labels[1242]` | 281 |
| `excluded_labels[1243]` | 2813 |
| `excluded_labels[1244]` | 2818 |
| `excluded_labels[1245]` | 2819 |
| `excluded_labels[1246]` | 2821 |
| `excluded_labels[1247]` | 2823 |
| `excluded_labels[1248]` | 2824 |
| `excluded_labels[1249]` | 2829 |
| `excluded_labels[1250]` | 2834 |
| `excluded_labels[1251]` | 2838 |
| `excluded_labels[1252]` | 284 |
| `excluded_labels[1253]` | 2841 |
| `excluded_labels[1254]` | 2842 |
| `excluded_labels[1255]` | 2845 |
| `excluded_labels[1256]` | 2846 |
| `excluded_labels[1257]` | 2847 |
| `excluded_labels[1258]` | 2852 |
| `excluded_labels[1259]` | 2853 |
| `excluded_labels[1260]` | 2854 |
| `excluded_labels[1261]` | 2855 |
| `excluded_labels[1262]` | 2862 |
| `excluded_labels[1263]` | 2864 |
| `excluded_labels[1264]` | 2865 |
| `excluded_labels[1265]` | 2868 |
| `excluded_labels[1266]` | 2869 |
| `excluded_labels[1267]` | 2872 |
| `excluded_labels[1268]` | 2873 |
| `excluded_labels[1269]` | 2874 |
| `excluded_labels[1270]` | 2875 |
| `excluded_labels[1271]` | 2876 |
| `excluded_labels[1272]` | 2877 |
| `excluded_labels[1273]` | 2879 |
| `excluded_labels[1274]` | 2880 |
| `excluded_labels[1275]` | 2884 |
| `excluded_labels[1276]` | 2886 |
| `excluded_labels[1277]` | 2890 |
| `excluded_labels[1278]` | 2891 |
| `excluded_labels[1279]` | 2894 |
| `excluded_labels[1280]` | 2896 |
| `excluded_labels[1281]` | 2900 |
| `excluded_labels[1282]` | 2902 |
| `excluded_labels[1283]` | 2904 |
| `excluded_labels[1284]` | 2905 |
| `excluded_labels[1285]` | 2907 |
| `excluded_labels[1286]` | 291 |
| `excluded_labels[1287]` | 2911 |
| `excluded_labels[1288]` | 2913 |
| `excluded_labels[1289]` | 2916 |
| `excluded_labels[1290]` | 292 |
| `excluded_labels[1291]` | 2924 |
| `excluded_labels[1292]` | 2925 |
| `excluded_labels[1293]` | 2927 |
| `excluded_labels[1294]` | 2928 |
| `excluded_labels[1295]` | 2933 |
| `excluded_labels[1296]` | 2938 |
| `excluded_labels[1297]` | 294 |
| `excluded_labels[1298]` | 2942 |
| `excluded_labels[1299]` | 2944 |
| `excluded_labels[1300]` | 2946 |
| `excluded_labels[1301]` | 2947 |
| `excluded_labels[1302]` | 2949 |
| `excluded_labels[1303]` | 295 |
| `excluded_labels[1304]` | 2952 |
| `excluded_labels[1305]` | 2953 |
| `excluded_labels[1306]` | 2954 |
| `excluded_labels[1307]` | 2956 |
| `excluded_labels[1308]` | 296 |
| `excluded_labels[1309]` | 2961 |
| `excluded_labels[1310]` | 2962 |
| `excluded_labels[1311]` | 2965 |
| `excluded_labels[1312]` | 2966 |
| `excluded_labels[1313]` | 2968 |
| `excluded_labels[1314]` | 2970 |
| `excluded_labels[1315]` | 2971 |
| `excluded_labels[1316]` | 2975 |
| `excluded_labels[1317]` | 2977 |
| `excluded_labels[1318]` | 2985 |
| `excluded_labels[1319]` | 2986 |
| `excluded_labels[1320]` | 2988 |
| `excluded_labels[1321]` | 2989 |
| `excluded_labels[1322]` | 299 |
| `excluded_labels[1323]` | 2994 |
| `excluded_labels[1324]` | 2995 |
| `excluded_labels[1325]` | 2997 |
| `excluded_labels[1326]` | 2999 |
| `excluded_labels[1327]` | 3 |
| `excluded_labels[1328]` | 30 |
| `excluded_labels[1329]` | 300 |
| `excluded_labels[1330]` | 3001 |
| `excluded_labels[1331]` | 3002 |
| `excluded_labels[1332]` | 3005 |
| `excluded_labels[1333]` | 3006 |
| `excluded_labels[1334]` | 3009 |
| `excluded_labels[1335]` | 301 |
| `excluded_labels[1336]` | 3012 |
| `excluded_labels[1337]` | 3013 |
| `excluded_labels[1338]` | 3014 |
| `excluded_labels[1339]` | 3015 |
| `excluded_labels[1340]` | 3017 |
| `excluded_labels[1341]` | 3019 |
| `excluded_labels[1342]` | 3020 |
| `excluded_labels[1343]` | 3022 |
| `excluded_labels[1344]` | 3023 |
| `excluded_labels[1345]` | 3025 |
| `excluded_labels[1346]` | 3026 |
| `excluded_labels[1347]` | 303 |
| `excluded_labels[1348]` | 3030 |
| `excluded_labels[1349]` | 3031 |
| `excluded_labels[1350]` | 3032 |
| `excluded_labels[1351]` | 3033 |
| `excluded_labels[1352]` | 3036 |
| `excluded_labels[1353]` | 3038 |
| `excluded_labels[1354]` | 304 |
| `excluded_labels[1355]` | 3040 |
| `excluded_labels[1356]` | 3044 |
| `excluded_labels[1357]` | 3045 |
| `excluded_labels[1358]` | 3046 |
| `excluded_labels[1359]` | 3047 |
| `excluded_labels[1360]` | 3049 |
| `excluded_labels[1361]` | 3054 |
| `excluded_labels[1362]` | 3055 |
| `excluded_labels[1363]` | 3058 |
| `excluded_labels[1364]` | 306 |
| `excluded_labels[1365]` | 3060 |
| `excluded_labels[1366]` | 3062 |
| `excluded_labels[1367]` | 3064 |
| `excluded_labels[1368]` | 3065 |
| `excluded_labels[1369]` | 3066 |
| `excluded_labels[1370]` | 3068 |
| `excluded_labels[1371]` | 3069 |
| `excluded_labels[1372]` | 3072 |
| `excluded_labels[1373]` | 3073 |
| `excluded_labels[1374]` | 3076 |
| `excluded_labels[1375]` | 3078 |
| `excluded_labels[1376]` | 3079 |
| `excluded_labels[1377]` | 3080 |
| `excluded_labels[1378]` | 3081 |
| `excluded_labels[1379]` | 3082 |
| `excluded_labels[1380]` | 3083 |
| `excluded_labels[1381]` | 3084 |
| `excluded_labels[1382]` | 3085 |
| `excluded_labels[1383]` | 3088 |
| `excluded_labels[1384]` | 3089 |
| `excluded_labels[1385]` | 3091 |
| `excluded_labels[1386]` | 3092 |
| `excluded_labels[1387]` | 3094 |
| `excluded_labels[1388]` | 3096 |
| `excluded_labels[1389]` | 3097 |
| `excluded_labels[1390]` | 310 |
| `excluded_labels[1391]` | 3100 |
| `excluded_labels[1392]` | 3103 |
| `excluded_labels[1393]` | 3107 |
| `excluded_labels[1394]` | 3108 |
| `excluded_labels[1395]` | 3109 |
| `excluded_labels[1396]` | 3110 |
| `excluded_labels[1397]` | 3112 |
| `excluded_labels[1398]` | 3113 |
| `excluded_labels[1399]` | 3114 |
| `excluded_labels[1400]` | 3116 |
| `excluded_labels[1401]` | 3118 |
| `excluded_labels[1402]` | 3120 |
| `excluded_labels[1403]` | 3122 |
| `excluded_labels[1404]` | 3123 |
| `excluded_labels[1405]` | 3126 |
| `excluded_labels[1406]` | 3129 |
| `excluded_labels[1407]` | 3130 |
| `excluded_labels[1408]` | 3133 |
| `excluded_labels[1409]` | 3134 |
| `excluded_labels[1410]` | 3138 |
| `excluded_labels[1411]` | 3142 |
| `excluded_labels[1412]` | 3143 |
| `excluded_labels[1413]` | 3144 |
| `excluded_labels[1414]` | 3147 |
| `excluded_labels[1415]` | 3151 |
| `excluded_labels[1416]` | 3152 |
| `excluded_labels[1417]` | 3155 |
| `excluded_labels[1418]` | 3156 |
| `excluded_labels[1419]` | 3158 |
| `excluded_labels[1420]` | 316 |
| `excluded_labels[1421]` | 3161 |
| `excluded_labels[1422]` | 3162 |
| `excluded_labels[1423]` | 3163 |
| `excluded_labels[1424]` | 3167 |
| `excluded_labels[1425]` | 3169 |
| `excluded_labels[1426]` | 317 |
| `excluded_labels[1427]` | 3170 |
| `excluded_labels[1428]` | 3171 |
| `excluded_labels[1429]` | 3175 |
| `excluded_labels[1430]` | 3176 |
| `excluded_labels[1431]` | 3177 |
| `excluded_labels[1432]` | 3179 |
| `excluded_labels[1433]` | 3180 |
| `excluded_labels[1434]` | 3181 |
| `excluded_labels[1435]` | 3183 |
| `excluded_labels[1436]` | 3188 |
| `excluded_labels[1437]` | 319 |
| `excluded_labels[1438]` | 3195 |
| `excluded_labels[1439]` | 3196 |
| `excluded_labels[1440]` | 3197 |
| `excluded_labels[1441]` | 3198 |
| `excluded_labels[1442]` | 3199 |
| `excluded_labels[1443]` | 32 |
| `excluded_labels[1444]` | 320 |
| `excluded_labels[1445]` | 3201 |
| `excluded_labels[1446]` | 3202 |
| `excluded_labels[1447]` | 3203 |
| `excluded_labels[1448]` | 3206 |
| `excluded_labels[1449]` | 3207 |
| `excluded_labels[1450]` | 321 |
| `excluded_labels[1451]` | 3214 |
| `excluded_labels[1452]` | 3216 |
| `excluded_labels[1453]` | 322 |
| `excluded_labels[1454]` | 3220 |
| `excluded_labels[1455]` | 3221 |
| `excluded_labels[1456]` | 3222 |
| `excluded_labels[1457]` | 3223 |
| `excluded_labels[1458]` | 3224 |
| `excluded_labels[1459]` | 3225 |
| `excluded_labels[1460]` | 3226 |
| `excluded_labels[1461]` | 3231 |
| `excluded_labels[1462]` | 3233 |
| `excluded_labels[1463]` | 3234 |
| `excluded_labels[1464]` | 3237 |
| `excluded_labels[1465]` | 3240 |
| `excluded_labels[1466]` | 3244 |
| `excluded_labels[1467]` | 3245 |
| `excluded_labels[1468]` | 3248 |
| `excluded_labels[1469]` | 325 |
| `excluded_labels[1470]` | 3253 |
| `excluded_labels[1471]` | 3254 |
| `excluded_labels[1472]` | 3255 |
| `excluded_labels[1473]` | 3256 |
| `excluded_labels[1474]` | 3258 |
| `excluded_labels[1475]` | 326 |
| `excluded_labels[1476]` | 3260 |
| `excluded_labels[1477]` | 3262 |
| `excluded_labels[1478]` | 3264 |
| `excluded_labels[1479]` | 3265 |
| `excluded_labels[1480]` | 3266 |
| `excluded_labels[1481]` | 3268 |
| `excluded_labels[1482]` | 3270 |
| `excluded_labels[1483]` | 3271 |
| `excluded_labels[1484]` | 3273 |
| `excluded_labels[1485]` | 3279 |
| `excluded_labels[1486]` | 328 |
| `excluded_labels[1487]` | 3280 |
| `excluded_labels[1488]` | 3282 |
| `excluded_labels[1489]` | 3287 |
| `excluded_labels[1490]` | 3288 |
| `excluded_labels[1491]` | 3296 |
| `excluded_labels[1492]` | 3298 |
| `excluded_labels[1493]` | 3300 |
| `excluded_labels[1494]` | 3302 |
| `excluded_labels[1495]` | 331 |
| `excluded_labels[1496]` | 3314 |
| `excluded_labels[1497]` | 3316 |
| `excluded_labels[1498]` | 3318 |
| `excluded_labels[1499]` | 3319 |
| `excluded_labels[1500]` | 332 |
| `excluded_labels[1501]` | 3323 |
| `excluded_labels[1502]` | 3326 |
| `excluded_labels[1503]` | 3329 |
| `excluded_labels[1504]` | 333 |
| `excluded_labels[1505]` | 3331 |
| `excluded_labels[1506]` | 3332 |
| `excluded_labels[1507]` | 3333 |
| `excluded_labels[1508]` | 3341 |
| `excluded_labels[1509]` | 3343 |
| `excluded_labels[1510]` | 3344 |
| `excluded_labels[1511]` | 3345 |
| `excluded_labels[1512]` | 3346 |
| `excluded_labels[1513]` | 3347 |
| `excluded_labels[1514]` | 3349 |
| `excluded_labels[1515]` | 335 |
| `excluded_labels[1516]` | 3350 |
| `excluded_labels[1517]` | 3352 |
| `excluded_labels[1518]` | 3353 |
| `excluded_labels[1519]` | 3356 |
| `excluded_labels[1520]` | 3357 |
| `excluded_labels[1521]` | 3358 |
| `excluded_labels[1522]` | 336 |
| `excluded_labels[1523]` | 3362 |
| `excluded_labels[1524]` | 3364 |
| `excluded_labels[1525]` | 3367 |
| `excluded_labels[1526]` | 3368 |
| `excluded_labels[1527]` | 337 |
| `excluded_labels[1528]` | 3371 |
| `excluded_labels[1529]` | 3372 |
| `excluded_labels[1530]` | 3373 |
| `excluded_labels[1531]` | 3378 |
| `excluded_labels[1532]` | 338 |
| `excluded_labels[1533]` | 3380 |
| `excluded_labels[1534]` | 3382 |
| `excluded_labels[1535]` | 3383 |
| `excluded_labels[1536]` | 3384 |
| `excluded_labels[1537]` | 3388 |
| `excluded_labels[1538]` | 3389 |
| `excluded_labels[1539]` | 339 |
| `excluded_labels[1540]` | 3390 |
| `excluded_labels[1541]` | 3391 |
| `excluded_labels[1542]` | 3396 |
| `excluded_labels[1543]` | 3398 |
| `excluded_labels[1544]` | 34 |
| `excluded_labels[1545]` | 340 |
| `excluded_labels[1546]` | 3402 |
| `excluded_labels[1547]` | 3403 |
| `excluded_labels[1548]` | 3411 |
| `excluded_labels[1549]` | 3413 |
| `excluded_labels[1550]` | 3414 |
| `excluded_labels[1551]` | 3416 |
| `excluded_labels[1552]` | 3418 |
| `excluded_labels[1553]` | 3420 |
| `excluded_labels[1554]` | 3421 |
| `excluded_labels[1555]` | 3423 |
| `excluded_labels[1556]` | 3424 |
| `excluded_labels[1557]` | 3425 |
| `excluded_labels[1558]` | 3426 |
| `excluded_labels[1559]` | 3427 |
| `excluded_labels[1560]` | 343 |
| `excluded_labels[1561]` | 3432 |
| `excluded_labels[1562]` | 3436 |
| `excluded_labels[1563]` | 3438 |
| `excluded_labels[1564]` | 3439 |
| `excluded_labels[1565]` | 3440 |
| `excluded_labels[1566]` | 3443 |
| `excluded_labels[1567]` | 3444 |
| `excluded_labels[1568]` | 3445 |
| `excluded_labels[1569]` | 345 |
| `excluded_labels[1570]` | 3450 |
| `excluded_labels[1571]` | 3451 |
| `excluded_labels[1572]` | 3454 |
| `excluded_labels[1573]` | 3455 |
| `excluded_labels[1574]` | 3456 |
| `excluded_labels[1575]` | 3458 |
| `excluded_labels[1576]` | 346 |
| `excluded_labels[1577]` | 3460 |
| `excluded_labels[1578]` | 3461 |
| `excluded_labels[1579]` | 3462 |
| `excluded_labels[1580]` | 3465 |
| `excluded_labels[1581]` | 3470 |
| `excluded_labels[1582]` | 3471 |
| `excluded_labels[1583]` | 3472 |
| `excluded_labels[1584]` | 3474 |
| `excluded_labels[1585]` | 3476 |
| `excluded_labels[1586]` | 3479 |
| `excluded_labels[1587]` | 3481 |
| `excluded_labels[1588]` | 3483 |
| `excluded_labels[1589]` | 3484 |
| `excluded_labels[1590]` | 3485 |
| `excluded_labels[1591]` | 3486 |
| `excluded_labels[1592]` | 3487 |
| `excluded_labels[1593]` | 349 |
| `excluded_labels[1594]` | 3490 |
| `excluded_labels[1595]` | 3491 |
| `excluded_labels[1596]` | 3494 |
| `excluded_labels[1597]` | 3499 |
| `excluded_labels[1598]` | 35 |
| `excluded_labels[1599]` | 3502 |
| `excluded_labels[1600]` | 3503 |
| `excluded_labels[1601]` | 3504 |
| `excluded_labels[1602]` | 351 |
| `excluded_labels[1603]` | 3510 |
| `excluded_labels[1604]` | 3513 |
| `excluded_labels[1605]` | 3514 |
| `excluded_labels[1606]` | 3515 |
| `excluded_labels[1607]` | 3516 |
| `excluded_labels[1608]` | 3517 |
| `excluded_labels[1609]` | 3518 |
| `excluded_labels[1610]` | 3519 |
| `excluded_labels[1611]` | 3520 |
| `excluded_labels[1612]` | 3521 |
| `excluded_labels[1613]` | 3529 |
| `excluded_labels[1614]` | 3531 |
| `excluded_labels[1615]` | 3533 |
| `excluded_labels[1616]` | 3534 |
| `excluded_labels[1617]` | 3535 |
| `excluded_labels[1618]` | 3536 |
| `excluded_labels[1619]` | 3539 |
| `excluded_labels[1620]` | 3543 |
| `excluded_labels[1621]` | 3547 |
| `excluded_labels[1622]` | 3549 |
| `excluded_labels[1623]` | 3553 |
| `excluded_labels[1624]` | 3556 |
| `excluded_labels[1625]` | 3557 |
| `excluded_labels[1626]` | 3559 |
| `excluded_labels[1627]` | 3560 |
| `excluded_labels[1628]` | 3564 |
| `excluded_labels[1629]` | 3565 |
| `excluded_labels[1630]` | 3566 |
| `excluded_labels[1631]` | 3567 |
| `excluded_labels[1632]` | 357 |
| `excluded_labels[1633]` | 3570 |
| `excluded_labels[1634]` | 3571 |
| `excluded_labels[1635]` | 3573 |
| `excluded_labels[1636]` | 3577 |
| `excluded_labels[1637]` | 3579 |
| `excluded_labels[1638]` | 358 |
| `excluded_labels[1639]` | 3582 |
| `excluded_labels[1640]` | 3584 |
| `excluded_labels[1641]` | 3585 |
| `excluded_labels[1642]` | 3586 |
| `excluded_labels[1643]` | 3588 |
| `excluded_labels[1644]` | 3589 |
| `excluded_labels[1645]` | 3591 |
| `excluded_labels[1646]` | 3592 |
| `excluded_labels[1647]` | 3595 |
| `excluded_labels[1648]` | 3598 |
| `excluded_labels[1649]` | 360 |
| `excluded_labels[1650]` | 3605 |
| `excluded_labels[1651]` | 3606 |
| `excluded_labels[1652]` | 3607 |
| `excluded_labels[1653]` | 3608 |
| `excluded_labels[1654]` | 361 |
| `excluded_labels[1655]` | 3610 |
| `excluded_labels[1656]` | 3612 |
| `excluded_labels[1657]` | 3613 |
| `excluded_labels[1658]` | 3617 |
| `excluded_labels[1659]` | 3618 |
| `excluded_labels[1660]` | 3620 |
| `excluded_labels[1661]` | 3623 |
| `excluded_labels[1662]` | 3627 |
| `excluded_labels[1663]` | 3629 |
| `excluded_labels[1664]` | 3630 |
| `excluded_labels[1665]` | 3633 |
| `excluded_labels[1666]` | 3636 |
| `excluded_labels[1667]` | 3637 |
| `excluded_labels[1668]` | 3639 |
| `excluded_labels[1669]` | 3640 |
| `excluded_labels[1670]` | 3643 |
| `excluded_labels[1671]` | 3649 |
| `excluded_labels[1672]` | 365 |
| `excluded_labels[1673]` | 3652 |
| `excluded_labels[1674]` | 3655 |
| `excluded_labels[1675]` | 3658 |
| `excluded_labels[1676]` | 3659 |
| `excluded_labels[1677]` | 366 |
| `excluded_labels[1678]` | 3661 |
| `excluded_labels[1679]` | 3662 |
| `excluded_labels[1680]` | 3663 |
| `excluded_labels[1681]` | 3664 |
| `excluded_labels[1682]` | 3669 |
| `excluded_labels[1683]` | 367 |
| `excluded_labels[1684]` | 3670 |
| `excluded_labels[1685]` | 3674 |
| `excluded_labels[1686]` | 3675 |
| `excluded_labels[1687]` | 3676 |
| `excluded_labels[1688]` | 3677 |
| `excluded_labels[1689]` | 3679 |
| `excluded_labels[1690]` | 3680 |
| `excluded_labels[1691]` | 3682 |
| `excluded_labels[1692]` | 3683 |
| `excluded_labels[1693]` | 3687 |
| `excluded_labels[1694]` | 3689 |
| `excluded_labels[1695]` | 369 |
| `excluded_labels[1696]` | 3693 |
| `excluded_labels[1697]` | 3694 |
| `excluded_labels[1698]` | 3695 |
| `excluded_labels[1699]` | 3697 |
| `excluded_labels[1700]` | 3698 |
| `excluded_labels[1701]` | 3704 |
| `excluded_labels[1702]` | 3705 |
| `excluded_labels[1703]` | 3707 |
| `excluded_labels[1704]` | 3716 |
| `excluded_labels[1705]` | 3717 |
| `excluded_labels[1706]` | 3718 |
| `excluded_labels[1707]` | 372 |
| `excluded_labels[1708]` | 3720 |
| `excluded_labels[1709]` | 3723 |
| `excluded_labels[1710]` | 3729 |
| `excluded_labels[1711]` | 373 |
| `excluded_labels[1712]` | 3732 |
| `excluded_labels[1713]` | 3733 |
| `excluded_labels[1714]` | 3735 |
| `excluded_labels[1715]` | 3737 |
| `excluded_labels[1716]` | 3738 |
| `excluded_labels[1717]` | 374 |
| `excluded_labels[1718]` | 3740 |
| `excluded_labels[1719]` | 3742 |
| `excluded_labels[1720]` | 3743 |
| `excluded_labels[1721]` | 375 |
| `excluded_labels[1722]` | 3751 |
| `excluded_labels[1723]` | 3752 |
| `excluded_labels[1724]` | 3754 |
| `excluded_labels[1725]` | 3755 |
| `excluded_labels[1726]` | 3758 |
| `excluded_labels[1727]` | 376 |
| `excluded_labels[1728]` | 3760 |
| `excluded_labels[1729]` | 3761 |
| `excluded_labels[1730]` | 3762 |
| `excluded_labels[1731]` | 3763 |
| `excluded_labels[1732]` | 3764 |
| `excluded_labels[1733]` | 3765 |
| `excluded_labels[1734]` | 3768 |
| `excluded_labels[1735]` | 3769 |
| `excluded_labels[1736]` | 3770 |
| `excluded_labels[1737]` | 3771 |
| `excluded_labels[1738]` | 3772 |
| `excluded_labels[1739]` | 3773 |
| `excluded_labels[1740]` | 3774 |
| `excluded_labels[1741]` | 3775 |
| `excluded_labels[1742]` | 378 |
| `excluded_labels[1743]` | 3781 |
| `excluded_labels[1744]` | 3785 |
| `excluded_labels[1745]` | 3787 |
| `excluded_labels[1746]` | 379 |
| `excluded_labels[1747]` | 3790 |
| `excluded_labels[1748]` | 3792 |
| `excluded_labels[1749]` | 3793 |
| `excluded_labels[1750]` | 3794 |
| `excluded_labels[1751]` | 3795 |
| `excluded_labels[1752]` | 38 |
| `excluded_labels[1753]` | 3807 |
| `excluded_labels[1754]` | 3809 |
| `excluded_labels[1755]` | 381 |
| `excluded_labels[1756]` | 3813 |
| `excluded_labels[1757]` | 3815 |
| `excluded_labels[1758]` | 3818 |
| `excluded_labels[1759]` | 3819 |
| `excluded_labels[1760]` | 382 |
| `excluded_labels[1761]` | 3823 |
| `excluded_labels[1762]` | 3824 |
| `excluded_labels[1763]` | 3826 |
| `excluded_labels[1764]` | 3827 |
| `excluded_labels[1765]` | 3828 |
| `excluded_labels[1766]` | 3832 |
| `excluded_labels[1767]` | 3837 |
| `excluded_labels[1768]` | 3839 |
| `excluded_labels[1769]` | 384 |
| `excluded_labels[1770]` | 3840 |
| `excluded_labels[1771]` | 3841 |
| `excluded_labels[1772]` | 3843 |
| `excluded_labels[1773]` | 3844 |
| `excluded_labels[1774]` | 3845 |
| `excluded_labels[1775]` | 3847 |
| `excluded_labels[1776]` | 3848 |
| `excluded_labels[1777]` | 3850 |
| `excluded_labels[1778]` | 3852 |
| `excluded_labels[1779]` | 3853 |
| `excluded_labels[1780]` | 3857 |
| `excluded_labels[1781]` | 3858 |
| `excluded_labels[1782]` | 3859 |
| `excluded_labels[1783]` | 3863 |
| `excluded_labels[1784]` | 3864 |
| `excluded_labels[1785]` | 3865 |
| `excluded_labels[1786]` | 3866 |
| `excluded_labels[1787]` | 3874 |
| `excluded_labels[1788]` | 3876 |
| `excluded_labels[1789]` | 3878 |
| `excluded_labels[1790]` | 3879 |
| `excluded_labels[1791]` | 388 |
| `excluded_labels[1792]` | 3881 |
| `excluded_labels[1793]` | 3884 |
| `excluded_labels[1794]` | 3886 |
| `excluded_labels[1795]` | 3889 |
| `excluded_labels[1796]` | 3890 |
| `excluded_labels[1797]` | 3891 |
| `excluded_labels[1798]` | 3892 |
| `excluded_labels[1799]` | 3893 |
| `excluded_labels[1800]` | 3895 |
| `excluded_labels[1801]` | 3899 |
| `excluded_labels[1802]` | 390 |
| `excluded_labels[1803]` | 3903 |
| `excluded_labels[1804]` | 3907 |
| `excluded_labels[1805]` | 3908 |
| `excluded_labels[1806]` | 3910 |
| `excluded_labels[1807]` | 3911 |
| `excluded_labels[1808]` | 3912 |
| `excluded_labels[1809]` | 3915 |
| `excluded_labels[1810]` | 3922 |
| `excluded_labels[1811]` | 3924 |
| `excluded_labels[1812]` | 3927 |
| `excluded_labels[1813]` | 3929 |
| `excluded_labels[1814]` | 3930 |
| `excluded_labels[1815]` | 3931 |
| `excluded_labels[1816]` | 3934 |
| `excluded_labels[1817]` | 3937 |
| `excluded_labels[1818]` | 3938 |
| `excluded_labels[1819]` | 394 |
| `excluded_labels[1820]` | 3942 |
| `excluded_labels[1821]` | 3943 |
| `excluded_labels[1822]` | 3944 |
| `excluded_labels[1823]` | 3945 |
| `excluded_labels[1824]` | 3947 |
| `excluded_labels[1825]` | 3950 |
| `excluded_labels[1826]` | 3955 |
| `excluded_labels[1827]` | 3959 |
| `excluded_labels[1828]` | 396 |
| `excluded_labels[1829]` | 3960 |
| `excluded_labels[1830]` | 3962 |
| `excluded_labels[1831]` | 3963 |
| `excluded_labels[1832]` | 3967 |
| `excluded_labels[1833]` | 3968 |
| `excluded_labels[1834]` | 3971 |
| `excluded_labels[1835]` | 3972 |
| `excluded_labels[1836]` | 3973 |
| `excluded_labels[1837]` | 3976 |
| `excluded_labels[1838]` | 3979 |
| `excluded_labels[1839]` | 398 |
| `excluded_labels[1840]` | 3980 |
| `excluded_labels[1841]` | 3982 |
| `excluded_labels[1842]` | 3985 |
| `excluded_labels[1843]` | 3987 |
| `excluded_labels[1844]` | 3989 |
| `excluded_labels[1845]` | 399 |
| `excluded_labels[1846]` | 3990 |
| `excluded_labels[1847]` | 3993 |
| `excluded_labels[1848]` | 3995 |
| `excluded_labels[1849]` | 3996 |
| `excluded_labels[1850]` | 3997 |
| `excluded_labels[1851]` | 3999 |
| `excluded_labels[1852]` | 40 |
| `excluded_labels[1853]` | 400 |
| `excluded_labels[1854]` | 4001 |
| `excluded_labels[1855]` | 4007 |
| `excluded_labels[1856]` | 401 |
| `excluded_labels[1857]` | 4010 |
| `excluded_labels[1858]` | 4014 |
| `excluded_labels[1859]` | 4017 |
| `excluded_labels[1860]` | 4018 |
| `excluded_labels[1861]` | 402 |
| `excluded_labels[1862]` | 4021 |
| `excluded_labels[1863]` | 4022 |
| `excluded_labels[1864]` | 4023 |
| `excluded_labels[1865]` | 4025 |
| `excluded_labels[1866]` | 4027 |
| `excluded_labels[1867]` | 4029 |
| `excluded_labels[1868]` | 4031 |
| `excluded_labels[1869]` | 4034 |
| `excluded_labels[1870]` | 4035 |
| `excluded_labels[1871]` | 4037 |
| `excluded_labels[1872]` | 4038 |
| `excluded_labels[1873]` | 4039 |
| `excluded_labels[1874]` | 404 |
| `excluded_labels[1875]` | 4040 |
| `excluded_labels[1876]` | 4041 |
| `excluded_labels[1877]` | 4046 |
| `excluded_labels[1878]` | 4047 |
| `excluded_labels[1879]` | 4049 |
| `excluded_labels[1880]` | 4053 |
| `excluded_labels[1881]` | 4055 |
| `excluded_labels[1882]` | 4056 |
| `excluded_labels[1883]` | 4062 |
| `excluded_labels[1884]` | 4065 |
| `excluded_labels[1885]` | 4067 |
| `excluded_labels[1886]` | 4070 |
| `excluded_labels[1887]` | 4075 |
| `excluded_labels[1888]` | 4076 |
| `excluded_labels[1889]` | 4077 |
| `excluded_labels[1890]` | 408 |
| `excluded_labels[1891]` | 4080 |
| `excluded_labels[1892]` | 4081 |
| `excluded_labels[1893]` | 4082 |
| `excluded_labels[1894]` | 4085 |
| `excluded_labels[1895]` | 4089 |
| `excluded_labels[1896]` | 409 |
| `excluded_labels[1897]` | 4091 |
| `excluded_labels[1898]` | 4093 |
| `excluded_labels[1899]` | 4096 |
| `excluded_labels[1900]` | 4098 |
| `excluded_labels[1901]` | 4099 |
| `excluded_labels[1902]` | 41 |
| `excluded_labels[1903]` | 410 |
| `excluded_labels[1904]` | 4103 |
| `excluded_labels[1905]` | 4104 |
| `excluded_labels[1906]` | 4105 |
| `excluded_labels[1907]` | 4110 |
| `excluded_labels[1908]` | 4111 |
| `excluded_labels[1909]` | 4115 |
| `excluded_labels[1910]` | 4116 |
| `excluded_labels[1911]` | 4117 |
| `excluded_labels[1912]` | 4118 |
| `excluded_labels[1913]` | 4119 |
| `excluded_labels[1914]` | 4122 |
| `excluded_labels[1915]` | 4126 |
| `excluded_labels[1916]` | 4127 |
| `excluded_labels[1917]` | 4128 |
| `excluded_labels[1918]` | 4129 |
| `excluded_labels[1919]` | 413 |
| `excluded_labels[1920]` | 4131 |
| `excluded_labels[1921]` | 4135 |
| `excluded_labels[1922]` | 4136 |
| `excluded_labels[1923]` | 4138 |
| `excluded_labels[1924]` | 4140 |
| `excluded_labels[1925]` | 4143 |
| `excluded_labels[1926]` | 4144 |
| `excluded_labels[1927]` | 4145 |
| `excluded_labels[1928]` | 4146 |
| `excluded_labels[1929]` | 4149 |
| `excluded_labels[1930]` | 4150 |
| `excluded_labels[1931]` | 4151 |
| `excluded_labels[1932]` | 4157 |
| `excluded_labels[1933]` | 4159 |
| `excluded_labels[1934]` | 4160 |
| `excluded_labels[1935]` | 4162 |
| `excluded_labels[1936]` | 4164 |
| `excluded_labels[1937]` | 4165 |
| `excluded_labels[1938]` | 4167 |
| `excluded_labels[1939]` | 4169 |
| `excluded_labels[1940]` | 4172 |
| `excluded_labels[1941]` | 4173 |
| `excluded_labels[1942]` | 4174 |
| `excluded_labels[1943]` | 4176 |
| `excluded_labels[1944]` | 4177 |
| `excluded_labels[1945]` | 4178 |
| `excluded_labels[1946]` | 418 |
| `excluded_labels[1947]` | 4180 |
| `excluded_labels[1948]` | 4182 |
| `excluded_labels[1949]` | 4183 |
| `excluded_labels[1950]` | 4185 |
| `excluded_labels[1951]` | 4187 |
| `excluded_labels[1952]` | 4188 |
| `excluded_labels[1953]` | 419 |
| `excluded_labels[1954]` | 4192 |
| `excluded_labels[1955]` | 4193 |
| `excluded_labels[1956]` | 4196 |
| `excluded_labels[1957]` | 4197 |
| `excluded_labels[1958]` | 42 |
| `excluded_labels[1959]` | 4201 |
| `excluded_labels[1960]` | 4202 |
| `excluded_labels[1961]` | 4204 |
| `excluded_labels[1962]` | 4206 |
| `excluded_labels[1963]` | 4207 |
| `excluded_labels[1964]` | 4212 |
| `excluded_labels[1965]` | 4213 |
| `excluded_labels[1966]` | 4214 |
| `excluded_labels[1967]` | 4215 |
| `excluded_labels[1968]` | 4218 |
| `excluded_labels[1969]` | 4219 |
| `excluded_labels[1970]` | 422 |
| `excluded_labels[1971]` | 4220 |
| `excluded_labels[1972]` | 4226 |
| `excluded_labels[1973]` | 4228 |
| `excluded_labels[1974]` | 4229 |
| `excluded_labels[1975]` | 423 |
| `excluded_labels[1976]` | 4230 |
| `excluded_labels[1977]` | 4231 |
| `excluded_labels[1978]` | 4232 |
| `excluded_labels[1979]` | 4235 |
| `excluded_labels[1980]` | 4236 |
| `excluded_labels[1981]` | 4237 |
| `excluded_labels[1982]` | 4238 |
| `excluded_labels[1983]` | 4239 |
| `excluded_labels[1984]` | 424 |
| `excluded_labels[1985]` | 4242 |
| `excluded_labels[1986]` | 4243 |
| `excluded_labels[1987]` | 4244 |
| `excluded_labels[1988]` | 425 |
| `excluded_labels[1989]` | 4251 |
| `excluded_labels[1990]` | 4254 |
| `excluded_labels[1991]` | 4256 |
| `excluded_labels[1992]` | 4257 |
| `excluded_labels[1993]` | 4258 |
| `excluded_labels[1994]` | 4259 |
| `excluded_labels[1995]` | 426 |
| `excluded_labels[1996]` | 4262 |
| `excluded_labels[1997]` | 4263 |
| `excluded_labels[1998]` | 4264 |
| `excluded_labels[1999]` | 4267 |
| `kept_labels[0]` | 1077 |
| `kept_labels[1]` | 1088 |
| `kept_labels[2]` | 1122 |
| `kept_labels[3]` | 1153 |
| `kept_labels[4]` | 1193 |
| `kept_labels[5]` | 1217 |
| `kept_labels[6]` | 1220 |
| `kept_labels[7]` | 1248 |
| `kept_labels[8]` | 1321 |
| `kept_labels[9]` | 1418 |
| `kept_labels[10]` | 1544 |
| `kept_labels[11]` | 1550 |
| `kept_labels[12]` | 1574 |
| `kept_labels[13]` | 1581 |
| `kept_labels[14]` | 1582 |
| `kept_labels[15]` | 1622 |
| `kept_labels[16]` | 1650 |
| `kept_labels[17]` | 1657 |
| `kept_labels[18]` | 1750 |
| `kept_labels[19]` | 1759 |
| `kept_labels[20]` | 1782 |
| `kept_labels[21]` | 1802 |
| `kept_labels[22]` | 1803 |
| `kept_labels[23]` | 1806 |
| `kept_labels[24]` | 1841 |
| `kept_labels[25]` | 1877 |
| `kept_labels[26]` | 1933 |
| `kept_labels[27]` | 1960 |
| `kept_labels[28]` | 1965 |
| `kept_labels[29]` | 1979 |
| `kept_labels[30]` | 2020 |
| `kept_labels[31]` | 2051 |
| `kept_labels[32]` | 2063 |
| `kept_labels[33]` | 2068 |
| `kept_labels[34]` | 2108 |
| `kept_labels[35]` | 212 |
| `kept_labels[36]` | 2146 |
| `kept_labels[37]` | 2153 |
| `kept_labels[38]` | 2157 |
| `kept_labels[39]` | 2201 |
| `kept_labels[40]` | 2257 |
| `kept_labels[41]` | 2273 |
| `kept_labels[42]` | 228 |
| `kept_labels[43]` | 2301 |
| `kept_labels[44]` | 2317 |
| `kept_labels[45]` | 238 |
| `kept_labels[46]` | 2402 |
| `kept_labels[47]` | 2426 |
| `kept_labels[48]` | 2436 |
| `kept_labels[49]` | 244 |
| `kept_labels[50]` | 2441 |
| `kept_labels[51]` | 2471 |
| `kept_labels[52]` | 2474 |
| `kept_labels[53]` | 2498 |
| `kept_labels[54]` | 250 |
| `kept_labels[55]` | 2504 |
| `kept_labels[56]` | 2518 |
| `kept_labels[57]` | 2525 |
| `kept_labels[58]` | 2535 |
| `kept_labels[59]` | 2537 |
| `kept_labels[60]` | 2541 |
| `kept_labels[61]` | 2545 |
| `kept_labels[62]` | 2672 |
| `kept_labels[63]` | 269 |
| `kept_labels[64]` | 2737 |
| `kept_labels[65]` | 2810 |
| `kept_labels[66]` | 2815 |
| `kept_labels[67]` | 283 |
| `kept_labels[68]` | 2839 |
| `kept_labels[69]` | 2883 |
| `kept_labels[70]` | 2912 |
| `kept_labels[71]` | 2914 |
| `kept_labels[72]` | 2919 |
| `kept_labels[73]` | 2981 |
| `kept_labels[74]` | 3010 |
| `kept_labels[75]` | 3119 |
| `kept_labels[76]` | 3241 |
| `kept_labels[77]` | 3242 |
| `kept_labels[78]` | 3249 |
| `kept_labels[79]` | 3252 |
| `kept_labels[80]` | 3259 |
| `kept_labels[81]` | 33 |
| `kept_labels[82]` | 3303 |
| `kept_labels[83]` | 3369 |
| `kept_labels[84]` | 3377 |
| `kept_labels[85]` | 3447 |
| `kept_labels[86]` | 3500 |
| `kept_labels[87]` | 3581 |
| `kept_labels[88]` | 3628 |
| `kept_labels[89]` | 3645 |
| `kept_labels[90]` | 3646 |
| `kept_labels[91]` | 3657 |
| `kept_labels[92]` | 3700 |
| `kept_labels[93]` | 3796 |
| `kept_labels[94]` | 3814 |
| `kept_labels[95]` | 3830 |
| `kept_labels[96]` | 3897 |
| `kept_labels[97]` | 3953 |
| `kept_labels[98]` | 397 |
| `kept_labels[99]` | 3986 |
| `kept_labels[100]` | 3998 |
| `kept_labels[101]` | 4043 |
| `kept_labels[102]` | 4057 |
| `kept_labels[103]` | 4066 |
| `kept_labels[104]` | 4074 |
| `kept_labels[105]` | 4079 |
| `kept_labels[106]` | 4100 |
| `kept_labels[107]` | 4108 |
| `kept_labels[108]` | 4209 |
| `kept_labels[109]` | 4240 |
| `kept_labels[110]` | 429 |
| `kept_labels[111]` | 4341 |
| `kept_labels[112]` | 4387 |
| `kept_labels[113]` | 4396 |
| `kept_labels[114]` | 4405 |
| `kept_labels[115]` | 4410 |
| `kept_labels[116]` | 4414 |
| `kept_labels[117]` | 46 |
| `kept_labels[118]` | 467 |
| `kept_labels[119]` | 4851 |
| `kept_labels[120]` | 548 |
| `kept_labels[121]` | 552 |
| `kept_labels[122]` | 558 |
| `kept_labels[123]` | 625 |
| `kept_labels[124]` | 635 |
| `kept_labels[125]` | 655 |
| `kept_labels[126]` | 658 |
| `kept_labels[127]` | 703 |
| `kept_labels[128]` | 706 |
| `kept_labels[129]` | 745 |
| `kept_labels[130]` | 749 |
| `kept_labels[131]` | 754 |
| `kept_labels[132]` | 755 |
| `kept_labels[133]` | 756 |
| `kept_labels[134]` | 792 |
| `kept_labels[135]` | 831 |
| `kept_labels[136]` | 834 |
| `kept_labels[137]` | 835 |
| `kept_labels[138]` | 836 |
| `kept_labels[139]` | 838 |
| `kept_labels[140]` | 849 |
| `kept_labels[141]` | 853 |
| `kept_labels[142]` | 854 |
| `kept_labels[143]` | 856 |
| `kept_labels[144]` | 858 |
| `kept_labels[145]` | 860 |
| `kept_labels[146]` | 863 |
| `kept_labels[147]` | 884 |
| `kept_labels[148]` | 887 |
| `kept_labels[149]` | 8924 |
| `kept_labels[150]` | 893 |
| `kept_labels[151]` | 902 |
| `kept_labels[152]` | 907 |
| `kept_labels[153]` | 909 |
| `kept_labels[154]` | 923 |
| `kept_labels[155]` | 924 |
| `kept_labels[156]` | 931 |
| `kept_labels[157]` | 932 |
| `kept_labels[158]` | 950 |
| `kept_labels[159]` | 951 |
| `kept_labels[160]` | 967 |
| `kept_labels[161]` | 968 |
| `kept_labels[162]` | 980 |
| `kept_labels[163]` | 988 |
| `kept_labels[164]` | 989 |
| `kept_labels[165]` | 998 |

### Raw artefact

```json
{
  "dataset": "QMUL-SurvFace training_set",
  "license": "research purposes only; images sourced from person re-identification datasets, copyright with original owners (qmul-survface.github.io)",
  "model": "w600k_r50",
  "threshold": 0.4,
  "per_identity_sampled": 40,
  "images_sampled": 78733,
  "eval_sets": [
    "lfw",
    "agedb_30",
    "cfp_fp",
    "calfw",
    "cplfw",
    "cfp_ff",
    "tinyface"
  ],
  "gallery_embeddings": 84171,
  "identities_total": 5319,
  "identities_excluded": 5153,
  "identities_kept": 166,
  "peak_similarity": 0.9153,
  "nearest_eval_set_tally": {
    "tinyface": 4162,
    "cplfw": 965,
    "cfp_fp": 184,
    "lfw": 4,
    "calfw": 3,
    "cfp_ff": 1
  },
  "threshold_sensitivity": {
    "0.30": 5302,
    "0.35": 5242,
    "0.40": 5153,
    "0.45": 5044,
    "0.50": 4882
  },
  "caveat": "Degraded probes yield weaker embeddings, compressing cosine similarity downward for true matches too. A 0.40 threshold carried over from the clean-vs-clean CASIA audit is a LOOSER filter here, not a stricter one.",
  "excluded_labels": [
    "100",
    "10000",
    "10001",
    "10004",
    "10005",
    "10006",
    "10008",
    "10009",
    "10010",
    "10011",
    "10012",
    "10014",
    "10015",
    "10016",
    "10017",
    "10018",
    "10019",
    "1002",
    "10020",
    "10023",
    "10025",
    "10026",
    "10027",
    "10030",
    "10031",
    "10032",
    "10033",
    "10034",
    "1004",
    "10043",
    "10045",
    "10046",
    "10047",
    "10048",
    "10049",
    "10050",
    "10051",
    "10054",
    "10056",
    "10057",
    "10062",
    "10063",
    "10064",
    "10065",
    "10067",
    "10069",
    "1007",
    "10070",
    "10075",
    "10076",
    "10078",
    "1008",
    "10081",
    "10083",
    "10084",
    "10085",
    "1009",
    "10092",
    "10095",
    "10096",
    "10098",
    "101",
    "1010",
    "10100",
    "10102",
    "10103",
    "10105",
    "10108",
    "10109",
    "1011",
    "10110",
    "10111",
    "10114",
    "10117",
    "10121",
    "10122",
    "10124",
    "10125",
    "10126",
    "10127",
    "10129",
    "1013",
    "10131",
    "10133",
    "10134",
    "10137",
    "10141",
    "10142",
    "10143",
    "10144",
    "10145",
    "10146",
    "10147",
    "10150",
    "10152",
    "10153",
    "10155",
    "10156",
    "10157",
    "10158",
    "10159",
    "10160",
    "10161",
    "10163",
    "10165",
    "10166",
    "10169",
    "1017",
    "10171",
    "10172",
    "10173",
    "10175",
    "10176",
    "10177",
    "10178",
    "10179",
    "10184",
    "10187",
    "10188",
    "10189",
    "1019",
    "10193",
    "10198",
    "10204",
    "10207",
    "10208",
    "1021",
    "10210",
    "10213",
    "10214",
    "1022",
    "10220",
    "10222",
    "10225",
    "10227",
    "1023",
    "10233",
    "10234",
    "10235",
    "10240",
    "10241",
    "10244",
    "10245",
    "10249",
    "1025",
    "10250",
    "10251",
    "10252",
    "10255",
    "10257",
    "10259",
    "1026",
    "10260",
    "10261",
    "10265",
    "10266",
    "10269",
    "10270",
    "10271",
    "10273",
    "10274",
    "10276",
    "10279",
    "1028",
    "10281",
    "10284",
    "10286",
    "10287",
    "10292",
    "10295",
    "10297",
    "10298",
    "103",
    "1030",
    "10300",
    "10305",
    "10307",
    "10308",
    "10311",
    "10314",
    "10316",
    "10318",
    "10319",
    "1032",
    "10321",
    "10322",
    "10324",
    "10325",
    "10327",
    "10328",
    "10329",
    "10330",
    "10333",
    "10334",
    "10337",
    "10338",
    "10340",
    "10342",
    "10343",
    "10344",
    "10345",
    "10349",
    "10350",
    "10351",
    "10352",
    "10353",
    "10356",
    "10357",
    "10358",
    "10359",
    "10361",
    "10362",
    "10364",
    "10365",
    "10367",
    "10368",
    "10369",
    "1037",
    "10372",
    "10377",
    "10379",
    "10380",
    "10381",
    "10387",
    "10388",
    "10389",
    "10390",
    "10392",
    "10393",
    "10394",
    "10395",
    "10396",
    "10399",
    "10400",
    "10403",
    "10405",
    "10407",
    "10409",
    "1041",
    "10410",
    "1042",
    "10421",
    "10422",
    "10423",
    "10428",
    "10429",
    "1043",
    "10431",
    "10434",
    "10435",
    "10438",
    "10439",
    "1044",
    "10440",
    "10441",
    "10442",
    "10447",
    "10450",
    "10453",
    "10455",
    "10456",
    "10457",
    "10458",
    "10459",
    "10460",
    "10461",
    "10464",
    "10465",
    "1047",
    "10470",
    "10473",
    "10474",
    "10475",
    "10476",
    "10477",
    "10479",
    "10480",
    "10483",
    "10484",
    "10489",
    "1049",
    "10490",
    "10491",
    "10495",
    "10496",
    "10498",
    "105",
    "10500",
    "10501",
    "10502",
    "10504",
    "10505",
    "10506",
    "10507",
    "10508",
    "10511",
    "10512",
    "10513",
    "10517",
    "10519",
    "1052",
    "10520",
    "10533",
    "10538",
    "1054",
    "10541",
    "10542",
    "10546",
    "10547",
    "10551",
    "10552",
    "10553",
    "10554",
    "10558",
    "10564",
    "10567",
    "10569",
    "10571",
    "10572",
    "10573",
    "10578",
    "10582",
    "10583",
    "10584",
    "10587",
    "10588",
    "10589",
    "10593",
    "10596",
    "10597",
    "10598",
    "10600",
    "10608",
    "1061",
    "10612",
    "10613",
    "10614",
    "10618",
    "10619",
    "1062",
    "10620",
    "10621",
    "10625",
    "10626",
    "1063",
    "10630",
    "10631",
    "10632",
    "10633",
    "10634",
    "10636",
    "1064",
    "1065",
    "1068",
    "1069",
    "107",
    "1075",
    "1078",
    "1081",
    "1091",
    "1093",
    "1095",
    "1096",
    "1097",
    "1098",
    "1099",
    "11",
    "1103",
    "1107",
    "1108",
    "111",
    "1111",
    "1112",
    "1114",
    "1116",
    "1117",
    "112",
    "1121",
    "1123",
    "1124",
    "1125",
    "1127",
    "1128",
    "113",
    "1131",
    "1132",
    "1133",
    "1134",
    "1138",
    "1139",
    "1140",
    "1141",
    "1142",
    "1145",
    "1147",
    "1148",
    "115",
    "1155",
    "1159",
    "1162",
    "1166",
    "1168",
    "1170",
    "1171",
    "1174",
    "1178",
    "1179",
    "1181",
    "1182",
    "1187",
    "1188",
    "1189",
    "119",
    "1190",
    "1195",
    "1197",
    "1199",
    "120",
    "1201",
    "1202",
    "1204",
    "1205",
    "1207",
    "1209",
    "121",
    "1213",
    "1216",
    "1221",
    "1222",
    "1223",
    "1224",
    "1225",
    "1226",
    "1227",
    "1228",
    "123",
    "1230",
    "1232",
    "1233",
    "1234",
    "1240",
    "1241",
    "1242",
    "1243",
    "1251",
    "1256",
    "1257",
    "1259",
    "1261",
    "1263",
    "1264",
    "1265",
    "1266",
    "127",
    "1270",
    "1271",
    "1273",
    "1276",
    "128",
    "1285",
    "1286",
    "1289",
    "129",
    "1293",
    "1298",
    "1300",
    "1301",
    "1305",
    "1307",
    "131",
    "1311",
    "1312",
    "1313",
    "1319",
    "1322",
    "1326",
    "1329",
    "133",
    "1330",
    "1331",
    "1333",
    "1336",
    "1337",
    "134",
    "1342",
    "1343",
    "1347",
    "1349",
    "135",
    "1350",
    "1351",
    "1353",
    "1354",
    "1355",
    "1356",
    "1359",
    "136",
    "1360",
    "1365",
    "1367",
    "1368",
    "1369",
    "1370",
    "1371",
    "1372",
    "1373",
    "1375",
    "1376",
    "1384",
    "1385",
    "1386",
    "1387",
    "1389",
    "1390",
    "1392",
    "1397",
    "1398",
    "14",
    "140",
    "1401",
    "1405",
    "1406",
    "1407",
    "1412",
    "1415",
    "1421",
    "1422",
    "1423",
    "1426",
    "143",
    "1431",
    "1432",
    "1434",
    "1435",
    "1436",
    "1438",
    "1440",
    "1442",
    "1443",
    "1445",
    "1448",
    "1449",
    "145",
    "1450",
    "1451",
    "1452",
    "1457",
    "1459",
    "146",
    "1461",
    "1463",
    "1469",
    "147",
    "1470",
    "1474",
    "1477",
    "148",
    "1480",
    "1483",
    "1488",
    "1489",
    "149",
    "1492",
    "1493",
    "1494",
    "1495",
    "1496",
    "1498",
    "1499",
    "150",
    "1500",
    "1501",
    "1502",
    "1503",
    "1504",
    "1505",
    "1506",
    "1507",
    "1508",
    "1509",
    "151",
    "1511",
    "1513",
    "1514",
    "1515",
    "1516",
    "1517",
    "1518",
    "1519",
    "152",
    "1520",
    "1521",
    "1522",
    "1526",
    "1527",
    "1528",
    "1530",
    "1531",
    "1532",
    "1533",
    "1534",
    "1536",
    "1539",
    "1546",
    "1547",
    "155",
    "1551",
    "1556",
    "1558",
    "156",
    "1560",
    "1563",
    "1564",
    "1565",
    "1566",
    "1567",
    "1569",
    "157",
    "1571",
    "1572",
    "1573",
    "1579",
    "1580",
    "1583",
    "1587",
    "1588",
    "1589",
    "159",
    "1595",
    "1597",
    "1598",
    "1599",
    "16",
    "160",
    "1600",
    "1602",
    "1604",
    "1605",
    "1607",
    "1610",
    "1611",
    "1615",
    "1616",
    "162",
    "1624",
    "1625",
    "1627",
    "163",
    "1630",
    "1631",
    "1632",
    "1634",
    "1635",
    "1636",
    "1637",
    "1638",
    "1639",
    "164",
    "1640",
    "1641",
    "1643",
    "1646",
    "1647",
    "1649",
    "165",
    "1652",
    "1659",
    "166",
    "1663",
    "1664",
    "1665",
    "1670",
    "1673",
    "1675",
    "1676",
    "1678",
    "1685",
    "1686",
    "1688",
    "1689",
    "1690",
    "1694",
    "1695",
    "1699",
    "17",
    "170",
    "1700",
    "1704",
    "1706",
    "1708",
    "1709",
    "171",
    "1710",
    "1713",
    "1714",
    "1715",
    "1718",
    "1719",
    "172",
    "1720",
    "1723",
    "1724",
    "1726",
    "1727",
    "1729",
    "1733",
    "1736",
    "1738",
    "1739",
    "174",
    "1740",
    "1743",
    "1745",
    "1748",
    "175",
    "1752",
    "1753",
    "1756",
    "1757",
    "176",
    "1763",
    "1764",
    "1770",
    "1772",
    "1773",
    "1774",
    "1783",
    "1784",
    "1789",
    "1792",
    "1794",
    "1795",
    "1796",
    "1797",
    "1799",
    "180",
    "1800",
    "1801",
    "1805",
    "1807",
    "1809",
    "1812",
    "1815",
    "1818",
    "1819",
    "182",
    "1823",
    "1828",
    "1829",
    "1830",
    "1831",
    "1832",
    "1833",
    "1836",
    "1837",
    "1838",
    "1839",
    "184",
    "1840",
    "1843",
    "1845",
    "1846",
    "1847",
    "1848",
    "1849",
    "1851",
    "1855",
    "1856",
    "1857",
    "186",
    "1860",
    "1861",
    "1864",
    "1867",
    "1869",
    "187",
    "1870",
    "1873",
    "1878",
    "1879",
    "1881",
    "1882",
    "1885",
    "1886",
    "1887",
    "1889",
    "1890",
    "1892",
    "1895",
    "1896",
    "190",
    "1901",
    "1903",
    "1904",
    "1905",
    "1910",
    "1912",
    "1916",
    "1917",
    "1925",
    "1927",
    "1928",
    "193",
    "1930",
    "1931",
    "1934",
    "1935",
    "1936",
    "1937",
    "1939",
    "194",
    "1940",
    "1941",
    "1942",
    "1944",
    "1949",
    "195",
    "1950",
    "1952",
    "1956",
    "1958",
    "1959",
    "196",
    "1967",
    "1973",
    "1974",
    "1976",
    "1977",
    "1978",
    "1983",
    "1988",
    "1990",
    "1994",
    "1996",
    "1997",
    "2000",
    "2003",
    "2004",
    "2005",
    "2006",
    "2007",
    "2009",
    "2011",
    "2013",
    "2014",
    "2016",
    "2017",
    "2018",
    "2019",
    "202",
    "2021",
    "2025",
    "2027",
    "2030",
    "2036",
    "2037",
    "2038",
    "2041",
    "2042",
    "2046",
    "2049",
    "205",
    "2050",
    "2054",
    "2057",
    "2058",
    "2064",
    "2072",
    "2073",
    "2075",
    "2078",
    "2080",
    "2082",
    "2083",
    "2085",
    "2086",
    "2087",
    "2088",
    "2091",
    "2092",
    "2094",
    "2099",
    "2105",
    "2109",
    "211",
    "2111",
    "2116",
    "2117",
    "2124",
    "2127",
    "213",
    "2130",
    "2132",
    "2133",
    "2134",
    "2135",
    "2136",
    "2138",
    "2139",
    "214",
    "2140",
    "2142",
    "2149",
    "2150",
    "2151",
    "2158",
    "216",
    "2160",
    "2161",
    "2162",
    "2163",
    "2168",
    "2169",
    "217",
    "2170",
    "2173",
    "2174",
    "2175",
    "2178",
    "2179",
    "218",
    "2180",
    "2182",
    "2184",
    "219",
    "2190",
    "2192",
    "2193",
    "2194",
    "2196",
    "22",
    "220",
    "2205",
    "2209",
    "221",
    "2210",
    "2211",
    "2215",
    "2219",
    "2220",
    "2224",
    "2225",
    "2228",
    "2229",
    "223",
    "2230",
    "2235",
    "2238",
    "2239",
    "224",
    "2240",
    "2241",
    "2246",
    "2248",
    "2249",
    "2250",
    "2251",
    "2253",
    "2261",
    "2263",
    "2264",
    "2266",
    "2267",
    "2268",
    "227",
    "2270",
    "2271",
    "2277",
    "2279",
    "2281",
    "2283",
    "2286",
    "2287",
    "2288",
    "229",
    "2291",
    "2294",
    "2295",
    "2296",
    "2298",
    "2299",
    "230",
    "2300",
    "2307",
    "2309",
    "2310",
    "2311",
    "2314",
    "2315",
    "232",
    "2320",
    "2321",
    "2325",
    "2327",
    "2330",
    "2331",
    "2333",
    "2334",
    "2336",
    "2339",
    "2342",
    "2344",
    "2348",
    "2349",
    "235",
    "2354",
    "2355",
    "2356",
    "2357",
    "2358",
    "2359",
    "236",
    "2361",
    "2362",
    "2363",
    "2364",
    "2365",
    "2367",
    "2368",
    "2369",
    "237",
    "2370",
    "2373",
    "2375",
    "2379",
    "2381",
    "2382",
    "2385",
    "2388",
    "239",
    "2390",
    "2391",
    "2392",
    "2394",
    "2395",
    "2397",
    "2403",
    "2406",
    "241",
    "2413",
    "2414",
    "2416",
    "2418",
    "242",
    "2420",
    "2421",
    "2423",
    "2427",
    "2429",
    "2430",
    "2431",
    "2432",
    "2433",
    "2435",
    "2437",
    "2439",
    "2442",
    "2443",
    "2446",
    "2452",
    "2453",
    "2454",
    "2455",
    "2457",
    "2458",
    "2459",
    "2461",
    "2462",
    "2465",
    "2466",
    "2467",
    "2469",
    "247",
    "2476",
    "2479",
    "2480",
    "2483",
    "2484",
    "2486",
    "2489",
    "249",
    "2493",
    "2494",
    "2499",
    "2506",
    "2513",
    "2514",
    "2515",
    "2516",
    "2517",
    "2519",
    "2522",
    "2523",
    "2526",
    "2529",
    "2532",
    "2533",
    "2536",
    "254",
    "2540",
    "2543",
    "2546",
    "2547",
    "2548",
    "2549",
    "255",
    "2550",
    "2552",
    "2557",
    "2558",
    "2562",
    "2563",
    "2565",
    "2570",
    "2571",
    "2572",
    "2573",
    "2576",
    "2577",
    "2579",
    "258",
    "2580",
    "2582",
    "2584",
    "2585",
    "259",
    "2590",
    "2591",
    "2593",
    "2596",
    "2598",
    "2601",
    "2604",
    "2609",
    "261",
    "2610",
    "2612",
    "2616",
    "2617",
    "2619",
    "2621",
    "2623",
    "2627",
    "2628",
    "2631",
    "2634",
    "2636",
    "2637",
    "2640",
    "2642",
    "2643",
    "2644",
    "2645",
    "2646",
    "2647",
    "2648",
    "2649",
    "265",
    "2650",
    "2652",
    "2654",
    "2655",
    "2656",
    "2657",
    "2659",
    "266",
    "2668",
    "2669",
    "267",
    "2674",
    "2675",
    "2676",
    "2677",
    "2678",
    "2679",
    "2680",
    "2684",
    "2685",
    "2696",
    "2697",
    "2699",
    "2700",
    "2701",
    "2703",
    "2706",
    "2707",
    "2708",
    "271",
    "2711",
    "2712",
    "2717",
    "2720",
    "2723",
    "2724",
    "2725",
    "2727",
    "2728",
    "2729",
    "273",
    "2730",
    "2731",
    "2734",
    "2736",
    "2739",
    "2740",
    "2744",
    "2745",
    "2746",
    "2748",
    "2749",
    "2751",
    "2758",
    "2759",
    "2760",
    "2763",
    "2764",
    "2766",
    "2767",
    "2768",
    "2769",
    "2770",
    "2771",
    "2772",
    "2774",
    "2775",
    "2778",
    "2779",
    "278",
    "2780",
    "2781",
    "2783",
    "2791",
    "2792",
    "2793",
    "2795",
    "28",
    "280",
    "2801",
    "2802",
    "2804",
    "2805",
    "2806",
    "281",
    "2813",
    "2818",
    "2819",
    "2821",
    "2823",
    "2824",
    "2829",
    "2834",
    "2838",
    "284",
    "2841",
    "2842",
    "2845",
    "2846",
    "2847",
    "2852",
    "2853",
    "2854",
    "2855",
    "2862",
    "2864",
    "2865",
    "2868",
    "2869",
    "2872",
    "2873",
    "2874",
    "2875",
    "2876",
    "2877",
    "2879",
    "2880",
    "2884",
    "2886",
    "2890",
    "2891",
    "2894",
    "2896",
    "2900",
    "2902",
    "2904",
    "2905",
    "2907",
    "291",
    "2911",
    "2913",
    "2916",
    "292",
    "2924",
    "2925",
    "2927",
    "2928",
    "2933",
    "2938",
    "294",
    "2942",
    "2944",
    "2946",
    "2947",
    "2949",
    "295",
    "2952",
    "2953",
    "2954",
    "2956",
    "296",
    "2961",
    "2962",
    "2965",
    "2966",
    "2968",
    "2970",
    "2971",
    "2975",
    "2977",
    "2985",
    "2986",
    "2988",
    "2989",
    "299",
    "2994",
    "2995",
    "2997",
    "2999",
    "3",
    "30",
    "300",
    "3001",
    "3002",
    "3005",
    "3006",
    "3009",
    "301",
    "3012",
    "3013",
    "3014",
    "3015",
    "3017",
    "3019",
    "3020",
    "3022",
    "3023",
    "3025",
    "3026",
    "303",
    "3030",
    "3031",
    "3032",
    "3033",
    "3036",
    "3038",
    "304",
    "3040",
    "3044",
    "3045",
    "3046",
    "3047",
    "3049",
    "3054",
    "3055",
    "3058",
    "306",
    "3060",
    "3062",
    "3064",
    "3065",
    "3066",
    "3068",
    "3069",
    "3072",
    "3073",
    "3076",
    "3078",
    "3079",
    "3080",
    "3081",
    "3082",
    "3083",
    "3084",
    "3085",
    "3088",
    "3089",
    "3091",
    "3092",
    "3094",
    "3096",
    "3097",
    "310",
    "3100",
    "3103",
    "3107",
    "3108",
    "3109",
    "3110",
    "3112",
    "3113",
    "3114",
    "3116",
    "3118",
    "3120",
    "3122",
    "3123",
    "3126",
    "3129",
    "3130",
    "3133",
    "3134",
    "3138",
    "3142",
    "3143",
    "3144",
    "3147",
    "3151",
    "3152",
    "3155",
    "3156",
    "3158",
    "316",
    "3161",
    "3162",
    "3163",
    "3167",
    "3169",
    "317",
    "3170",
    "3171",
    "3175",
    "3176",
    "3177",
    "3179",
    "3180",
    "3181",
    "3183",
    "3188",
    "319",
    "3195",
    "3196",
    "3197",
    "3198",
    "3199",
    "32",
    "320",
    "3201",
    "3202",
    "3203",
    "3206",
    "3207",
    "321",
    "3214",
    "3216",
    "322",
    "3220",
    "3221",
    "3222",
    "3223",
    "3224",
    "3225",
    "3226",
    "3231",
    "3233",
    "3234",
    "3237",
    "3240",
    "3244",
    "3245",
    "3248",
    "325",
    "3253",
    "3254",
    "3255",
    "3256",
    "3258",
    "326",
    "3260",
    "3262",
    "3264",
    "3265",
    "3266",
    "3268",
    "3270",
    "3271",
    "3273",
    "3279",
    "328",
    "3280",
    "3282",
    "3287",
    "3288",
    "3296",
    "3298",
    "3300",
    "3302",
    "331",
    "3314",
    "3316",
    "3318",
    "3319",
    "332",
    "3323",
    "3326",
    "3329",
    "333",
    "3331",
    "3332",
    "3333",
    "3341",
    "3343",
    "3344",
    "3345",
    "3346",
    "3347",
    "3349",
    "335",
    "3350",
    "3352",
    "3353",
    "3356",
    "3357",
    "3358",
    "336",
    "3362",
    "3364",
    "3367",
    "3368",
    "337",
    "3371",
    "3372",
    "3373",
    "3378",
    "338",
    "3380",
    "3382",
    "3383",
    "3384",
    "3388",
    "3389",
    "339",
    "3390",
    "3391",
    "3396",
    "3398",
    "34",
    "340",
    "3402",
    "3403",
    "3411",
    "3413",
    "3414",
    "3416",
    "3418",
    "3420",
    "3421",
    "3423",
    "3424",
    "3425",
    "3426",
    "3427",
    "343",
    "3432",
    "3436",
    "3438",
    "3439",
    "3440",
    "3443",
    "3444",
    "3445",
    "345",
    "3450",
    "3451",
    "3454",
    "3455",
    "3456",
    "3458",
    "346",
    "3460",
    "3461",
    "3462",
    "3465",
    "3470",
    "3471",
    "3472",
    "3474",
    "3476",
    "3479",
    "3481",
    "3483",
    "3484",
    "3485",
    "3486",
    "3487",
    "349",
    "3490",
    "3491",
    "3494",
    "3499",
    "35",
    "3502",
    "3503",
    "3504",
    "351",
    "3510",
    "3513",
    "3514",
    "3515",
    "3516",
    "3517",
    "3518",
    "3519",
    "3520",
    "3521",
    "3529",
    "3531",
    "3533",
    "3534",
    "3535",
    "3536",
    "3539",
    "3543",
    "3547",
    "3549",
    "3553",
    "3556",
    "3557",
    "3559",
    "3560",
    "3564",
    "3565",
    "3566",
    "3567",
    "357",
    "3570",
    "3571",
    "3573",
    "3577",
    "3579",
    "358",
    "3582",
    "3584",
    "3585",
    "3586",
    "3588",
    "3589",
    "3591",
    "3592",
    "3595",
    "3598",
    "360",
    "3605",
    "3606",
    "3607",
    "3608",
    "361",
    "3610",
    "3612",
    "3613",
    "3617",
    "3618",
    "3620",
    "3623",
    "3627",
    "3629",
    "3630",
    "3633",
    "3636",
    "3637",
    "3639",
    "3640",
    "3643",
    "3649",
    "365",
    "3652",
    "3655",
    "3658",
    "3659",
    "366",
    "3661",
    "3662",
    "3663",
    "3664",
    "3669",
    "367",
    "3670",
    "3674",
    "3675",
    "3676",
    "3677",
    "3679",
    "3680",
    "3682",
    "3683",
    "3687",
    "3689",
    "369",
    "3693",
    "3694",
    "3695",
    "3697",
    "3698",
    "3704",
    "3705",
    "3707",
    "3716",
    "3717",
    "3718",
    "372",
    "3720",
    "3723",
    "3729",
    "373",
    "3732",
    "3733",
    "3735",
    "3737",
    "3738",
    "374",
    "3740",
    "3742",
    "3743",
    "375",
    "3751",
    "3752",
    "3754",
    "3755",
    "3758",
    "376",
    "3760",
    "3761",
    "3762",
    "3763",
    "3764",
    "3765",
    "3768",
    "3769",
    "3770",
    "3771",
    "3772",
    "3773",
    "3774",
    "3775",
    "378",
    "3781",
    "3785",
    "3787",
    "379",
    "3790",
    "3792",
    "3793",
    "3794",
    "3795",
    "38",
    "3807",
    "3809",
    "381",
    "3813",
    "3815",
    "3818",
    "3819",
    "382",
    "3823",
    "3824",
    "3826",
    "3827",
    "3828",
    "3832",
    "3837",
    "3839",
    "384",
    "3840",
    "3841",
    "3843",
    "3844",
    "3845",
    "3847",
    "3848",
    "3850",
    "3852",
    "3853",
    "3857",
    "3858",
    "3859",
    "3863",
    "3864",
    "3865",
    "3866",
    "3874",
    "3876",
    "3878",
    "3879",
    "388",
    "3881",
    "3884",
    "3886",
    "3889",
    "3890",
    "3891",
    "3892",
    "3893",
    "3895",
    "3899",
    "390",
    "3903",
    "3907",
    "3908",
    "3910",
    "3911",
    "3912",
    "3915",
    "3922",
    "3924",
    "3927",
    "3929",
    "3930",
    "3931",
    "3934",
    "3937",
    "3938",
    "394",
    "3942",
    "3943",
    "3944",
    "3945",
    "3947",
    "3950",
    "3955",
    "3959",
    "396",
    "3960",
    "3962",
    "3963",
    "3967",
    "3968",
    "3971",
    "3972",
    "3973",
    "3976",
    "3979",
    "398",
    "3980",
    "3982",
    "3985",
    "3987",
    "3989",
    "399",
    "3990",
    "3993",
    "3995",
    "3996",
    "3997",
    "3999",
    "40",
    "400",
    "4001",
    "4007",
    "401",
    "4010",
    "4014",
    "4017",
    "4018",
    "402",
    "4021",
    "4022",
    "4023",
    "4025",
    "4027",
    "4029",
    "4031",
    "4034",
    "4035",
    "4037",
    "4038",
    "4039",
    "404",
    "4040",
    "4041",
    "4046",
    "4047",
    "4049",
    "4053",
    "4055",
    "4056",
    "4062",
    "4065",
    "4067",
    "4070",
    "4075",
    "4076",
    "4077",
    "408",
    "4080",
    "4081",
    "4082",
    "4085",
    "4089",
    "409",
    "4091",
    "4093",
    "4096",
    "4098",
    "4099",
    "41",
    "410",
    "4103",
    "4104",
    "4105",
    "4110",
    "4111",
    "4115",
    "4116",
    "4117",
    "4118",
    "4119",
    "4122",
    "4126",
    "4127",
    "4128",
    "4129",
    "413",
    "4131",
    "4135",
    "4136",
    "4138",
    "4140",
    "4143",
    "4144",
    "4145",
    "4146",
    "4149",
    "4150",
    "4151",
    "4157",
    "4159",
    "4160",
    "4162",
    "4164",
    "4165",
    "4167",
    "4169",
    "4172",
    "4173",
    "4174",
    "4176",
    "4177",
    "4178",
    "418",
    "4180",
    "4182",
    "4183",
    "4185",
    "4187",
    "4188",
    "419",
    "4192",
    "4193",
    "4196",
    "4197",
    "42",
    "4201",
    "4202",
    "4204",
    "4206",
    "4207",
    "4212",
    "4213",
    "4214",
    "4215",
    "4218",
    "4219",
    "422",
    "4220",
    "4226",
    "4228",
    "4229",
    "423",
    "4230",
    "4231",
    "4232",
    "4235",
    "4236",
    "4237",
    "4238",
    "4239",
    "424",
    "4242",
    "4243",
    "4244",
    "425",
    "4251",
    "4254",
    "4256",
    "4257",
    "4258",
    "4259",
    "426",
    "4262",
    "4263",
    "4264",
    "4267",
    "4268",
    "427",
    "4272",
    "4275",
    "4276",
    "4278",
    "4279",
    "4285",
    "4286",
    "4287",
    "4288",
    "4290",
    "4291",
    "4292",
    "4299",
    "4302",
    "4303",
    "4306",
    "4309",
    "4310",
    "4311",
    "4312",
    "4313",
    "4314",
    "4315",
    "4317",
    "4318",
    "4319",
    "4320",
    "4321",
    "4322",
    "4324",
    "4326",
    "4327",
    "4329",
    "4330",
    "4332",
    "4344",
    "4345",
    "4346",
    "435",
    "4352",
    "4354",
    "4358",
    "4360",
    "4362",
    "4366",
    "4367",
    "437",
    "4372",
    "4374",
    "4377",
    "4379",
    "4381",
    "4382",
    "4383",
    "4384",
    "4386",
    "4388",
    "4389",
    "4390",
    "4393",
    "4394",
    "4395",
    "4397",
    "4399",
    "440",
    "4400",
    "4403",
    "4406",
    "441",
    "4411",
    "4413",
    "4416",
    "4418",
    "4419",
    "442",
    "4420",
    "4421",
    "4424",
    "4425",
    "4426",
    "4427",
    "4429",
    "4433",
    "4434",
    "4436",
    "4437",
    "4440",
    "4441",
    "4442",
    "4445",
    "4448",
    "445",
    "4450",
    "4453",
    "4454",
    "4461",
    "4462",
    "4463",
    "4466",
    "4468",
    "4470",
    "4474",
    "4475",
    "4477",
    "4478",
    "4479",
    "448",
    "4481",
    "4483",
    "4484",
    "4485",
    "4487",
    "4488",
    "4489",
    "4491",
    "4493",
    "4495",
    "4496",
    "4497",
    "45",
    "450",
    "4506",
    "4507",
    "4510",
    "4512",
    "4513",
    "4514",
    "4515",
    "4517",
    "4519",
    "452",
    "4521",
    "4528",
    "4529",
    "453",
    "4531",
    "4532",
    "4533",
    "4535",
    "4539",
    "454",
    "4540",
    "4542",
    "4543",
    "4544",
    "4545",
    "4548",
    "4549",
    "455",
    "4550",
    "4553",
    "4554",
    "4558",
    "4559",
    "4561",
    "4562",
    "4563",
    "4567",
    "4568",
    "4569",
    "457",
    "4570",
    "4571",
    "4580",
    "4581",
    "4582",
    "4583",
    "4584",
    "4587",
    "4588",
    "459",
    "4590",
    "4591",
    "4592",
    "4593",
    "4596",
    "4599",
    "460",
    "4601",
    "4602",
    "4603",
    "4609",
    "461",
    "4610",
    "4611",
    "4612",
    "4614",
    "4616",
    "4617",
    "4619",
    "4621",
    "4623",
    "4625",
    "4627",
    "4628",
    "4629",
    "463",
    "4631",
    "4632",
    "4635",
    "4638",
    "464",
    "4640",
    "4641",
    "4644",
    "4647",
    "4648",
    "4649",
    "465",
    "4650",
    "4652",
    "4656",
    "4659",
    "4660",
    "4661",
    "4664",
    "4665",
    "4668",
    "4669",
    "4674",
    "4677",
    "4680",
    "4683",
    "4688",
    "469",
    "4690",
    "4693",
    "4694",
    "4696",
    "4697",
    "4699",
    "47",
    "4700",
    "4701",
    "4705",
    "4707",
    "471",
    "4710",
    "4714",
    "4715",
    "4717",
    "4718",
    "4720",
    "4724",
    "4726",
    "4729",
    "473",
    "4732",
    "4733",
    "4735",
    "4736",
    "4737",
    "474",
    "4740",
    "4743",
    "4744",
    "4745",
    "4747",
    "4749",
    "475",
    "4750",
    "4753",
    "4755",
    "4761",
    "4765",
    "4766",
    "4767",
    "4768",
    "4769",
    "4770",
    "4773",
    "4779",
    "4781",
    "4782",
    "4784",
    "4785",
    "4786",
    "479",
    "4793",
    "4796",
    "4797",
    "4799",
    "4801",
    "4803",
    "4806",
    "4807",
    "4809",
    "4810",
    "4812",
    "4815",
    "4822",
    "4823",
    "4824",
    "4825",
    "4826",
    "4831",
    "4832",
    "4833",
    "4835",
    "4839",
    "4841",
    "4842",
    "4845",
    "4847",
    "4848",
    "4849",
    "485",
    "4850",
    "4853",
    "4855",
    "4856",
    "4858",
    "4859",
    "4864",
    "4866",
    "4868",
    "4869",
    "487",
    "4872",
    "4873",
    "4875",
    "4876",
    "488",
    "4886",
    "4888",
    "4889",
    "4890",
    "4892",
    "4893",
    "4896",
    "4897",
    "4898",
    "4899",
    "490",
    "4904",
    "4906",
    "4909",
    "491",
    "4910",
    "4911",
    "4913",
    "4914",
    "4915",
    "4916",
    "4917",
    "4919",
    "4921",
    "4923",
    "4924",
    "4928",
    "4930",
    "4931",
    "4932",
    "4933",
    "4934",
    "4935",
    "4936",
    "4937",
    "4938",
    "494",
    "4941",
    "4945",
    "4946",
    "4949",
    "4950",
    "4954",
    "4957",
    "4958",
    "4959",
    "496",
    "4961",
    "4962",
    "4965",
    "4966",
    "4968",
    "4970",
    "4974",
    "4978",
    "4981",
    "4983",
    "4985",
    "4986",
    "4987",
    "4989",
    "499",
    "4990",
    "4993",
    "4996",
    "4998",
    "5000",
    "5001",
    "5003",
    "5006",
    "5007",
    "5009",
    "5011",
    "5012",
    "5013",
    "5015",
    "5016",
    "5020",
    "5021",
    "5023",
    "5024",
    "5027",
    "5029",
    "5031",
    "5032",
    "5034",
    "5035",
    "5037",
    "5038",
    "504",
    "5040",
    "5041",
    "5042",
    "5043",
    "5044",
    "5045",
    "5048",
    "5049",
    "505",
    "5054",
    "5056",
    "5057",
    "5058",
    "506",
    "5060",
    "5063",
    "5067",
    "5068",
    "5071",
    "5073",
    "5075",
    "5076",
    "5078",
    "5079",
    "5080",
    "5085",
    "5088",
    "509",
    "5090",
    "5091",
    "5092",
    "5096",
    "5097",
    "5098",
    "5099",
    "51",
    "5101",
    "5102",
    "5104",
    "5105",
    "5111",
    "5112",
    "5113",
    "5118",
    "5119",
    "5120",
    "5121",
    "5123",
    "5127",
    "5129",
    "5130",
    "5132",
    "5133",
    "5135",
    "5137",
    "5138",
    "5139",
    "5145",
    "5147",
    "5149",
    "5152",
    "5154",
    "5155",
    "5159",
    "5160",
    "5161",
    "5162",
    "5165",
    "517",
    "5171",
    "5172",
    "5174",
    "5177",
    "5179",
    "518",
    "5180",
    "5182",
    "5183",
    "5184",
    "5185",
    "5188",
    "519",
    "5193",
    "520",
    "5200",
    "5201",
    "5202",
    "5203",
    "5204",
    "5209",
    "521",
    "5210",
    "5211",
    "5212",
    "5213",
    "5214",
    "5215",
    "5216",
    "5217",
    "5218",
    "522",
    "5220",
    "5222",
    "5223",
    "5228",
    "523",
    "5231",
    "5232",
    "5234",
    "5236",
    "5237",
    "5238",
    "5246",
    "5247",
    "5248",
    "5251",
    "5253",
    "5254",
    "5261",
    "5262",
    "5264",
    "5265",
    "5266",
    "5273",
    "5275",
    "5276",
    "5277",
    "5281",
    "5282",
    "5284",
    "5285",
    "5288",
    "5289",
    "529",
    "5290",
    "5292",
    "5295",
    "530",
    "5300",
    "5301",
    "5302",
    "5303",
    "5304",
    "5305",
    "5307",
    "5310",
    "5311",
    "5312",
    "5314",
    "5316",
    "5317",
    "532",
    "5321",
    "5322",
    "5325",
    "5326",
    "5327",
    "5329",
    "5330",
    "5331",
    "5332",
    "5333",
    "5334",
    "5335",
    "5336",
    "5338",
    "5340",
    "5341",
    "5344",
    "5345",
    "5346",
    "5350",
    "5353",
    "5355",
    "5359",
    "536",
    "5361",
    "5365",
    "5366",
    "5367",
    "537",
    "5370",
    "5372",
    "5373",
    "5374",
    "5378",
    "5379",
    "5380",
    "5381",
    "5383",
    "5384",
    "5386",
    "5390",
    "5394",
    "5396",
    "5399",
    "54",
    "5401",
    "5403",
    "5407",
    "5408",
    "541",
    "5411",
    "5412",
    "5415",
    "5419",
    "542",
    "5420",
    "5421",
    "5423",
    "5425",
    "5426",
    "5430",
    "5431",
    "5432",
    "5439",
    "544",
    "5440",
    "5443",
    "5444",
    "5445",
    "5446",
    "5447",
    "5449",
    "545",
    "5450",
    "5452",
    "5453",
    "5454",
    "5455",
    "5456",
    "5458",
    "5465",
    "5467",
    "5468",
    "5469",
    "547",
    "5470",
    "5471",
    "5473",
    "5477",
    "5478",
    "5479",
    "5481",
    "5482",
    "5483",
    "5487",
    "5489",
    "5490",
    "5492",
    "5495",
    "5496",
    "5497",
    "5498",
    "55",
    "550",
    "5500",
    "5502",
    "5504",
    "5505",
    "5506",
    "5507",
    "5508",
    "5510",
    "5511",
    "5513",
    "5514",
    "5515",
    "5516",
    "5518",
    "5521",
    "5522",
    "5524",
    "5525",
    "5526",
    "5529",
    "5531",
    "5532",
    "5534",
    "5535",
    "5536",
    "5538",
    "5540",
    "5541",
    "5543",
    "5548",
    "5549",
    "555",
    "5550",
    "5552",
    "5554",
    "5555",
    "5559",
    "556",
    "5560",
    "5561",
    "5567",
    "557",
    "5572",
    "5574",
    "5577",
    "5579",
    "5583",
    "5584",
    "5587",
    "5589",
    "559",
    "5592",
    "5593",
    "5595",
    "5596",
    "5597",
    "5598",
    "56",
    "560",
    "5600",
    "5602",
    "5603",
    "5604",
    "5605",
    "5607",
    "5609",
    "5610",
    "5612",
    "5615",
    "5616",
    "5617",
    "5619",
    "562",
    "5620",
    "5623",
    "5625",
    "5629",
    "5630",
    "5633",
    "5634",
    "5635",
    "5636",
    "5637",
    "5638",
    "5639",
    "564",
    "5643",
    "5645",
    "5646",
    "5649",
    "5650",
    "5651",
    "5652",
    "5653",
    "5654",
    "5655",
    "5658",
    "566",
    "5660",
    "5663",
    "5664",
    "5665",
    "5667",
    "5671",
    "5672",
    "5673",
    "5675",
    "5676",
    "5679",
    "5681",
    "5682",
    "5686",
    "5687",
    "569",
    "5690",
    "5692",
    "5693",
    "5695",
    "5697",
    "5698",
    "5699",
    "570",
    "5700",
    "5707",
    "5708",
    "5709",
    "5710",
    "5712",
    "5714",
    "5717",
    "5719",
    "5720",
    "5722",
    "5723",
    "5725",
    "5728",
    "573",
    "5731",
    "5736",
    "5738",
    "5741",
    "5744",
    "5745",
    "5749",
    "575",
    "5750",
    "5753",
    "5757",
    "5758",
    "5759",
    "5760",
    "5761",
    "5763",
    "5764",
    "5767",
    "5771",
    "5772",
    "5773",
    "5774",
    "5775",
    "5776",
    "5778",
    "5779",
    "578",
    "5781",
    "5782",
    "5785",
    "5787",
    "5790",
    "5791",
    "5792",
    "5793",
    "5794",
    "5797",
    "5799",
    "58",
    "5803",
    "5805",
    "5806",
    "5807",
    "5809",
    "5810",
    "5811",
    "5816",
    "5817",
    "5819",
    "5822",
    "5823",
    "5824",
    "5827",
    "5829",
    "5832",
    "5833",
    "5837",
    "5838",
    "584",
    "5841",
    "5844",
    "5845",
    "5846",
    "5847",
    "5848",
    "585",
    "5850",
    "5853",
    "5856",
    "5858",
    "586",
    "5860",
    "5861",
    "5866",
    "5868",
    "5869",
    "5870",
    "5876",
    "5877",
    "5878",
    "5880",
    "5882",
    "5883",
    "5884",
    "5885",
    "5892",
    "5893",
    "5894",
    "5895",
    "5896",
    "5897",
    "5898",
    "590",
    "5900",
    "5901",
    "5902",
    "5903",
    "5905",
    "5906",
    "5908",
    "5909",
    "5910",
    "5912",
    "5913",
    "5915",
    "5918",
    "592",
    "5924",
    "5925",
    "5929",
    "593",
    "5931",
    "5935",
    "5938",
    "5939",
    "594",
    "5940",
    "5943",
    "5945",
    "5946",
    "5947",
    "5948",
    "5950",
    "5951",
    "5954",
    "5956",
    "5959",
    "596",
    "5960",
    "5963",
    "5964",
    "5965",
    "5969",
    "597",
    "5970",
    "5972",
    "5973",
    "5974",
    "5976",
    "5977",
    "598",
    "5981",
    "5985",
    "5987",
    "599",
    "5990",
    "5991",
    "5992",
    "5993",
    "5994",
    "5996",
    "6",
    "60",
    "600",
    "6001",
    "6008",
    "6009",
    "6011",
    "6013",
    "6015",
    "6017",
    "6019",
    "6020",
    "6022",
    "6023",
    "6025",
    "6030",
    "6031",
    "6032",
    "6034",
    "6035",
    "6036",
    "6037",
    "6038",
    "6039",
    "6040",
    "6043",
    "6044",
    "6045",
    "6046",
    "6047",
    "6050",
    "6052",
    "6053",
    "6054",
    "6058",
    "6059",
    "6060",
    "6061",
    "6063",
    "6064",
    "6066",
    "6067",
    "6068",
    "6071",
    "6074",
    "6077",
    "6079",
    "6080",
    "6082",
    "6087",
    "609",
    "6090",
    "6091",
    "6092",
    "6093",
    "6094",
    "6097",
    "6099",
    "6100",
    "6102",
    "6103",
    "6104",
    "6105",
    "6107",
    "6110",
    "6111",
    "6113",
    "6116",
    "6118",
    "6120",
    "6122",
    "6128",
    "6129",
    "6130",
    "6132",
    "6133",
    "6136",
    "6138",
    "6139",
    "614",
    "6140",
    "6141",
    "6142",
    "6143",
    "6144",
    "6147",
    "6153",
    "6156",
    "6157",
    "6159",
    "616",
    "6160",
    "6162",
    "6163",
    "6164",
    "6165",
    "6167",
    "617",
    "6170",
    "6172",
    "6174",
    "6177",
    "6178",
    "6183",
    "6185",
    "6187",
    "6192",
    "6196",
    "6198",
    "6199",
    "62",
    "6209",
    "6210",
    "6211",
    "6214",
    "6215",
    "6216",
    "6217",
    "6223",
    "6224",
    "6225",
    "6228",
    "6229",
    "623",
    "6230",
    "6234",
    "6237",
    "6240",
    "6241",
    "6242",
    "6244",
    "6246",
    "6247",
    "6254",
    "6255",
    "6256",
    "6257",
    "6258",
    "6267",
    "6269",
    "6272",
    "6276",
    "6280",
    "6281",
    "6282",
    "6283",
    "6284",
    "6285",
    "6287",
    "6289",
    "6295",
    "6297",
    "6300",
    "6301",
    "6302",
    "6306",
    "6307",
    "6308",
    "6309",
    "6310",
    "6311",
    "6313",
    "6316",
    "6317",
    "6320",
    "6325",
    "6328",
    "6329",
    "6335",
    "6336",
    "6337",
    "6338",
    "6341",
    "6342",
    "6343",
    "6345",
    "6347",
    "6350",
    "6351",
    "6352",
    "6353",
    "6354",
    "6355",
    "6357",
    "636",
    "6365",
    "6370",
    "6375",
    "6378",
    "638",
    "6380",
    "6381",
    "6384",
    "6386",
    "6387",
    "6388",
    "6389",
    "6390",
    "6391",
    "6392",
    "6393",
    "6396",
    "6397",
    "6398",
    "6399",
    "640",
    "6401",
    "6403",
    "6410",
    "6412",
    "6413",
    "6414",
    "6415",
    "6416",
    "6417",
    "6418",
    "6419",
    "6423",
    "6427",
    "6429",
    "643",
    "6433",
    "6437",
    "6438",
    "6439",
    "6440",
    "6441",
    "6442",
    "6445",
    "6449",
    "645",
    "6451",
    "6454",
    "6457",
    "6458",
    "6459",
    "646",
    "6460",
    "6461",
    "6463",
    "6465",
    "6469",
    "647",
    "6470",
    "6471",
    "6472",
    "6473",
    "6474",
    "6475",
    "6476",
    "6477",
    "6479",
    "648",
    "6480",
    "6483",
    "6484",
    "6485",
    "6486",
    "6487",
    "6488",
    "6489",
    "649",
    "6493",
    "6494",
    "6495",
    "6496",
    "6497",
    "6499",
    "650",
    "6500",
    "6502",
    "6503",
    "6504",
    "6505",
    "6508",
    "651",
    "6510",
    "6513",
    "6514",
    "6516",
    "6517",
    "6518",
    "6519",
    "652",
    "6523",
    "6526",
    "6527",
    "653",
    "6530",
    "6532",
    "6535",
    "6537",
    "6538",
    "6543",
    "6544",
    "6545",
    "6546",
    "6549",
    "6550",
    "6551",
    "6553",
    "6554",
    "6555",
    "6559",
    "6560",
    "6563",
    "6569",
    "6570",
    "6572",
    "6573",
    "6575",
    "6577",
    "6580",
    "6581",
    "6583",
    "6586",
    "6592",
    "6599",
    "660",
    "6601",
    "6602",
    "6603",
    "6604",
    "6606",
    "661",
    "6611",
    "6612",
    "6614",
    "6615",
    "6616",
    "6618",
    "662",
    "6620",
    "6621",
    "6622",
    "6623",
    "6626",
    "6628",
    "6629",
    "6633",
    "6638",
    "6639",
    "6641",
    "6642",
    "6643",
    "665",
    "6650",
    "6651",
    "6657",
    "6658",
    "666",
    "6660",
    "6663",
    "6664",
    "6669",
    "6671",
    "6675",
    "6678",
    "6679",
    "6685",
    "6688",
    "6689",
    "669",
    "6691",
    "6692",
    "6696",
    "6698",
    "670",
    "6700",
    "6703",
    "6704",
    "6706",
    "6713",
    "6714",
    "6716",
    "6718",
    "672",
    "6721",
    "6722",
    "6723",
    "6724",
    "6727",
    "6728",
    "673",
    "6730",
    "6732",
    "6734",
    "6736",
    "6737",
    "674",
    "6740",
    "6742",
    "6746",
    "6749",
    "675",
    "6750",
    "6751",
    "6752",
    "6754",
    "6757",
    "6759",
    "6761",
    "6763",
    "6765",
    "6766",
    "6767",
    "6772",
    "6774",
    "6775",
    "6777",
    "6779",
    "678",
    "6782",
    "6783",
    "6785",
    "6786",
    "6787",
    "6788",
    "6789",
    "6791",
    "6793",
    "6794",
    "6796",
    "6797",
    "68",
    "6800",
    "6802",
    "6805",
    "6808",
    "6809",
    "6810",
    "6811",
    "6814",
    "6816",
    "6818",
    "6819",
    "682",
    "6821",
    "6823",
    "6825",
    "6828",
    "6831",
    "6832",
    "6833",
    "6837",
    "6838",
    "6844",
    "6851",
    "6852",
    "6853",
    "6854",
    "6857",
    "6859",
    "6860",
    "6861",
    "6862",
    "6863",
    "6864",
    "6869",
    "6870",
    "6872",
    "6875",
    "6877",
    "6880",
    "6881",
    "6888",
    "689",
    "6890",
    "6891",
    "6892",
    "6896",
    "6898",
    "69",
    "690",
    "6900",
    "6903",
    "6905",
    "6906",
    "6908",
    "6909",
    "691",
    "6910",
    "6914",
    "6915",
    "6917",
    "6918",
    "6919",
    "692",
    "6920",
    "6923",
    "6924",
    "6925",
    "6926",
    "6928",
    "6930",
    "6931",
    "6934",
    "6938",
    "6939",
    "694",
    "6940",
    "6941",
    "6942",
    "6943",
    "6944",
    "6946",
    "6949",
    "695",
    "6953",
    "6954",
    "6955",
    "6956",
    "6957",
    "6959",
    "696",
    "6960",
    "6961",
    "6962",
    "6964",
    "6965",
    "6966",
    "6968",
    "697",
    "6971",
    "6973",
    "6977",
    "6978",
    "6979",
    "6983",
    "6986",
    "6988",
    "699",
    "6991",
    "6994",
    "6995",
    "6997",
    "7",
    "70",
    "7000",
    "7006",
    "7007",
    "7008",
    "7009",
    "7010",
    "7011",
    "7012",
    "7013",
    "7017",
    "7020",
    "7022",
    "7024",
    "7026",
    "7028",
    "7030",
    "7031",
    "7033",
    "7038",
    "7039",
    "704",
    "7040",
    "7043",
    "7046",
    "7047",
    "7048",
    "705",
    "7051",
    "7053",
    "7058",
    "7060",
    "7061",
    "7063",
    "7064",
    "7066",
    "7068",
    "7069",
    "7072",
    "7073",
    "7074",
    "7076",
    "7077",
    "708",
    "7081",
    "7086",
    "7088",
    "7089",
    "7090",
    "7091",
    "7092",
    "7095",
    "7096",
    "7097",
    "7098",
    "7099",
    "710",
    "7100",
    "7104",
    "7105",
    "7106",
    "7110",
    "7111",
    "7112",
    "7115",
    "7116",
    "7117",
    "7122",
    "7123",
    "7124",
    "7126",
    "7127",
    "7128",
    "7131",
    "7135",
    "7138",
    "7139",
    "7140",
    "7142",
    "7143",
    "7144",
    "7145",
    "7147",
    "7149",
    "715",
    "7150",
    "7151",
    "7152",
    "7155",
    "7157",
    "7159",
    "716",
    "7161",
    "7163",
    "7164",
    "7165",
    "7167",
    "7169",
    "717",
    "7170",
    "7172",
    "7173",
    "7174",
    "7177",
    "7178",
    "7180",
    "7182",
    "7183",
    "7186",
    "7192",
    "7193",
    "7194",
    "7195",
    "7196",
    "7198",
    "72",
    "720",
    "7202",
    "7204",
    "7205",
    "7208",
    "721",
    "7210",
    "7212",
    "7216",
    "7218",
    "722",
    "7221",
    "7222",
    "7225",
    "7227",
    "7228",
    "7229",
    "723",
    "7230",
    "7231",
    "7232",
    "7238",
    "7239",
    "7241",
    "7242",
    "7244",
    "7246",
    "7248",
    "725",
    "7251",
    "7252",
    "7253",
    "7256",
    "726",
    "7260",
    "7262",
    "7265",
    "7266",
    "7267",
    "7269",
    "727",
    "7271",
    "7272",
    "7277",
    "7278",
    "7280",
    "7282",
    "7285",
    "7286",
    "7288",
    "7289",
    "7290",
    "7292",
    "7293",
    "7295",
    "7296",
    "7297",
    "7298",
    "730",
    "7302",
    "7303",
    "7305",
    "7306",
    "7307",
    "7308",
    "7310",
    "7312",
    "7313",
    "7314",
    "7318",
    "7319",
    "732",
    "7320",
    "7321",
    "7322",
    "7324",
    "7327",
    "7329",
    "7330",
    "7331",
    "7335",
    "7337",
    "7341",
    "7342",
    "7345",
    "7346",
    "7347",
    "7349",
    "7350",
    "7352",
    "7353",
    "7357",
    "7358",
    "736",
    "7361",
    "7362",
    "7363",
    "7364",
    "7365",
    "7366",
    "7367",
    "7368",
    "7369",
    "7371",
    "7372",
    "7375",
    "7376",
    "7377",
    "7378",
    "738",
    "7380",
    "7381",
    "7384",
    "7389",
    "739",
    "7391",
    "7392",
    "7393",
    "7394",
    "7395",
    "7398",
    "74",
    "740",
    "7400",
    "7401",
    "7403",
    "7407",
    "7408",
    "7410",
    "7412",
    "7415",
    "7419",
    "742",
    "7420",
    "7422",
    "7423",
    "7428",
    "7429",
    "743",
    "7430",
    "7436",
    "7437",
    "7439",
    "744",
    "7440",
    "7442",
    "7443",
    "7444",
    "7446",
    "7447",
    "7448",
    "7450",
    "7451",
    "7452",
    "7453",
    "7454",
    "746",
    "7460",
    "7465",
    "7466",
    "7467",
    "7469",
    "7470",
    "7472",
    "7475",
    "7478",
    "7479",
    "748",
    "7481",
    "7482",
    "7484",
    "7486",
    "7487",
    "7488",
    "7489",
    "7491",
    "7493",
    "7495",
    "7497",
    "750",
    "7501",
    "7502",
    "7505",
    "7506",
    "7507",
    "751",
    "7510",
    "7513",
    "7514",
    "7515",
    "7516",
    "7518",
    "7519",
    "752",
    "7522",
    "7526",
    "7528",
    "7530",
    "7531",
    "7532",
    "7533",
    "7534",
    "7535",
    "7539",
    "7541",
    "7543",
    "7544",
    "7546",
    "7547",
    "7550",
    "7551",
    "7552",
    "7560",
    "7566",
    "7567",
    "7568",
    "7569",
    "757",
    "7570",
    "7571",
    "7573",
    "7576",
    "7578",
    "758",
    "7581",
    "7584",
    "7586",
    "7588",
    "7591",
    "7593",
    "7596",
    "7598",
    "7599",
    "76",
    "7601",
    "7604",
    "7605",
    "7606",
    "7608",
    "7610",
    "7613",
    "7614",
    "7616",
    "762",
    "7620",
    "7628",
    "7629",
    "763",
    "7630",
    "7631",
    "7633",
    "7634",
    "7635",
    "7636",
    "7638",
    "7639",
    "7640",
    "7641",
    "7643",
    "7646",
    "7649",
    "7651",
    "7653",
    "7654",
    "7656",
    "7657",
    "7658",
    "7659",
    "7660",
    "7665",
    "7667",
    "7669",
    "767",
    "7671",
    "7673",
    "7674",
    "7675",
    "7676",
    "7678",
    "7680",
    "7682",
    "7683",
    "769",
    "7690",
    "7692",
    "7693",
    "7695",
    "7696",
    "7698",
    "7699",
    "77",
    "770",
    "7700",
    "7703",
    "7704",
    "7705",
    "7707",
    "7708",
    "7709",
    "7713",
    "7714",
    "7715",
    "772",
    "7721",
    "7725",
    "7726",
    "7727",
    "7729",
    "773",
    "7732",
    "7733",
    "7736",
    "7737",
    "7740",
    "7742",
    "7743",
    "7744",
    "7745",
    "7748",
    "7749",
    "7750",
    "7751",
    "7753",
    "7757",
    "7760",
    "7762",
    "7763",
    "7766",
    "7767",
    "7772",
    "7773",
    "7774",
    "7777",
    "7779",
    "7780",
    "7781",
    "7783",
    "7784",
    "7786",
    "7793",
    "7797",
    "7799",
    "78",
    "780",
    "7801",
    "7802",
    "7804",
    "7808",
    "7810",
    "7812",
    "7814",
    "7816",
    "7817",
    "7818",
    "7821",
    "7823",
    "7824",
    "7826",
    "7828",
    "783",
    "7833",
    "7834",
    "7836",
    "7838",
    "784",
    "7841",
    "7842",
    "7847",
    "7849",
    "7851",
    "7854",
    "7855",
    "7857",
    "7858",
    "786",
    "7865",
    "7867",
    "7870",
    "7871",
    "7872",
    "7874",
    "7875",
    "7879",
    "7880",
    "7881",
    "7886",
    "7887",
    "7888",
    "7889",
    "789",
    "7890",
    "7891",
    "7894",
    "7895",
    "7897",
    "790",
    "7900",
    "7904",
    "7907",
    "7911",
    "7912",
    "7913",
    "7914",
    "7917",
    "7918",
    "7919",
    "7920",
    "7921",
    "7922",
    "7923",
    "7925",
    "7927",
    "7928",
    "7930",
    "7933",
    "7937",
    "7939",
    "7942",
    "7945",
    "7946",
    "7947",
    "7950",
    "7951",
    "7952",
    "7956",
    "7957",
    "796",
    "7961",
    "7966",
    "7968",
    "7969",
    "797",
    "7970",
    "7971",
    "7972",
    "7974",
    "7975",
    "7976",
    "7977",
    "7979",
    "798",
    "7982",
    "7983",
    "7984",
    "7986",
    "7988",
    "7990",
    "7991",
    "7992",
    "7997",
    "80",
    "8000",
    "8002",
    "8003",
    "8004",
    "8005",
    "8007",
    "8008",
    "8013",
    "8014",
    "8015",
    "8018",
    "802",
    "8022",
    "8029",
    "8030",
    "8031",
    "8033",
    "8035",
    "8036",
    "8037",
    "804",
    "8041",
    "8042",
    "8043",
    "8048",
    "805",
    "8051",
    "8052",
    "8053",
    "8054",
    "8056",
    "8057",
    "8063",
    "8064",
    "8067",
    "8069",
    "807",
    "8071",
    "8074",
    "8075",
    "8083",
    "8085",
    "8089",
    "8090",
    "8091",
    "8096",
    "8097",
    "8098",
    "81",
    "8100",
    "8101",
    "8107",
    "8110",
    "8113",
    "8115",
    "8117",
    "8118",
    "812",
    "8125",
    "8126",
    "8129",
    "8132",
    "8135",
    "8137",
    "8140",
    "8145",
    "8146",
    "8147",
    "8150",
    "8152",
    "8154",
    "8155",
    "8157",
    "8158",
    "8159",
    "816",
    "8160",
    "8166",
    "8168",
    "8169",
    "817",
    "8171",
    "8172",
    "8173",
    "8174",
    "8176",
    "8178",
    "818",
    "8182",
    "8184",
    "8185",
    "8186",
    "8187",
    "8188",
    "8189",
    "8190",
    "8191",
    "8192",
    "8194",
    "8196",
    "8199",
    "8201",
    "8204",
    "8207",
    "8209",
    "8211",
    "8213",
    "8214",
    "8216",
    "8217",
    "8218",
    "8219",
    "822",
    "8224",
    "8225",
    "8227",
    "8229",
    "823",
    "8233",
    "8234",
    "8235",
    "8237",
    "8239",
    "824",
    "8242",
    "8243",
    "8244",
    "8245",
    "8246",
    "8247",
    "8250",
    "8251",
    "8257",
    "8259",
    "826",
    "8260",
    "8264",
    "8265",
    "8266",
    "8267",
    "8269",
    "827",
    "8270",
    "8272",
    "8273",
    "8277",
    "8278",
    "8279",
    "8283",
    "8284",
    "8285",
    "8287",
    "8289",
    "829",
    "8290",
    "8295",
    "8298",
    "8299",
    "8300",
    "8304",
    "8306",
    "8307",
    "8310",
    "8311",
    "8312",
    "8313",
    "8315",
    "8317",
    "8318",
    "8319",
    "832",
    "8320",
    "8321",
    "8322",
    "8323",
    "8324",
    "8325",
    "8326",
    "8327",
    "8329",
    "8331",
    "8337",
    "8338",
    "8339",
    "8340",
    "8342",
    "8346",
    "8347",
    "8349",
    "8353",
    "8354",
    "8356",
    "8359",
    "8364",
    "8365",
    "8366",
    "8368",
    "8369",
    "8372",
    "8377",
    "8378",
    "8379",
    "8382",
    "8383",
    "8384",
    "8385",
    "8386",
    "8387",
    "8389",
    "839",
    "8390",
    "8391",
    "8397",
    "8398",
    "8399",
    "840",
    "8402",
    "8403",
    "8404",
    "8406",
    "8407",
    "8408",
    "841",
    "8410",
    "8411",
    "8412",
    "8413",
    "8415",
    "8417",
    "8419",
    "8420",
    "8424",
    "8426",
    "843",
    "8430",
    "8431",
    "8432",
    "8434",
    "8437",
    "8438",
    "8439",
    "8440",
    "8441",
    "8442",
    "8446",
    "8448",
    "8453",
    "8456",
    "8458",
    "8459",
    "846",
    "8462",
    "8464",
    "8465",
    "8466",
    "8468",
    "8470",
    "8474",
    "8477",
    "8478",
    "8480",
    "8482",
    "8487",
    "8488",
    "8490",
    "8491",
    "8492",
    "8493",
    "8494",
    "8496",
    "8499",
    "8501",
    "8502",
    "8504",
    "8505",
    "8509",
    "851",
    "8510",
    "8512",
    "8516",
    "8517",
    "8518",
    "8520",
    "8521",
    "8523",
    "8525",
    "8526",
    "8531",
    "8532",
    "8533",
    "8535",
    "8537",
    "8541",
    "8542",
    "8544",
    "8546",
    "8547",
    "8549",
    "8551",
    "8552",
    "8554",
    "8555",
    "8560",
    "8563",
    "8564",
    "8565",
    "8566",
    "8568",
    "8570",
    "8572",
    "8576",
    "8578",
    "8579",
    "8582",
    "8584",
    "8585",
    "8586",
    "8587",
    "8588",
    "859",
    "8590",
    "8592",
    "8595",
    "8596",
    "8598",
    "8599",
    "86",
    "8604",
    "8608",
    "8615",
    "8616",
    "8617",
    "862",
    "8621",
    "8622",
    "8623",
    "8625",
    "8626",
    "8627",
    "8629",
    "8632",
    "8633",
    "8634",
    "8635",
    "8637",
    "8638",
    "8641",
    "8643",
    "8644",
    "8648",
    "865",
    "8650",
    "8651",
    "8656",
    "8658",
    "8659",
    "866",
    "8661",
    "8662",
    "8664",
    "8666",
    "8667",
    "8669",
    "8670",
    "8671",
    "8675",
    "8676",
    "8678",
    "8682",
    "8686",
    "8687",
    "8688",
    "8689",
    "869",
    "8690",
    "8691",
    "8694",
    "8695",
    "8698",
    "8699",
    "870",
    "8701",
    "8702",
    "8703",
    "8705",
    "8707",
    "8708",
    "8709",
    "871",
    "8710",
    "8713",
    "8714",
    "8715",
    "8717",
    "8718",
    "872",
    "8721",
    "8723",
    "8725",
    "8729",
    "873",
    "8731",
    "8734",
    "8737",
    "8738",
    "8740",
    "8741",
    "8742",
    "8745",
    "8746",
    "8747",
    "8748",
    "8749",
    "8755",
    "8756",
    "8757",
    "8758",
    "8759",
    "8760",
    "8761",
    "8763",
    "8765",
    "8768",
    "8769",
    "877",
    "8770",
    "8771",
    "8774",
    "8775",
    "8776",
    "8778",
    "8780",
    "8781",
    "8783",
    "8785",
    "8788",
    "879",
    "8790",
    "8795",
    "8797",
    "8798",
    "8799",
    "880",
    "8800",
    "8803",
    "8805",
    "8806",
    "8807",
    "881",
    "8811",
    "8813",
    "8816",
    "8817",
    "8818",
    "8819",
    "882",
    "8821",
    "8822",
    "8823",
    "8827",
    "8828",
    "8829",
    "883",
    "8831",
    "8836",
    "8838",
    "8839",
    "8840",
    "8843",
    "8844",
    "8845",
    "8846",
    "8847",
    "8848",
    "8849",
    "885",
    "8851",
    "8852",
    "8853",
    "8854",
    "8858",
    "8859",
    "886",
    "8862",
    "8864",
    "8865",
    "8866",
    "8867",
    "8869",
    "8871",
    "8872",
    "8873",
    "8875",
    "8877",
    "8878",
    "888",
    "8880",
    "8881",
    "8882",
    "8884",
    "8885",
    "8887",
    "8888",
    "8889",
    "8890",
    "8898",
    "8899",
    "89",
    "8901",
    "8902",
    "8903",
    "8904",
    "8907",
    "8909",
    "8910",
    "8911",
    "8914",
    "8915",
    "892",
    "8921",
    "8922",
    "8925",
    "8931",
    "8934",
    "8937",
    "8938",
    "894",
    "8942",
    "8943",
    "8944",
    "8946",
    "8947",
    "8948",
    "8949",
    "8952",
    "8958",
    "8959",
    "8960",
    "8962",
    "8963",
    "8964",
    "8967",
    "8969",
    "8973",
    "8975",
    "8976",
    "8979",
    "898",
    "8982",
    "8984",
    "8985",
    "8986",
    "8988",
    "8989",
    "899",
    "8990",
    "8992",
    "8993",
    "8994",
    "8995",
    "8996",
    "8997",
    "9003",
    "9005",
    "9006",
    "9008",
    "9009",
    "9010",
    "9011",
    "9012",
    "9013",
    "9016",
    "9019",
    "9020",
    "9023",
    "9025",
    "9026",
    "903",
    "9030",
    "9033",
    "9034",
    "9036",
    "9038",
    "9039",
    "904",
    "9040",
    "9042",
    "9043",
    "9045",
    "9049",
    "905",
    "9053",
    "9055",
    "9056",
    "9061",
    "9063",
    "9064",
    "9065",
    "9066",
    "9067",
    "9068",
    "9070",
    "9072",
    "9077",
    "9079",
    "9080",
    "9081",
    "9082",
    "9084",
    "9087",
    "9088",
    "9089",
    "9090",
    "9091",
    "9093",
    "9094",
    "9095",
    "9096",
    "9097",
    "9098",
    "91",
    "9100",
    "9102",
    "9103",
    "9106",
    "9107",
    "9109",
    "911",
    "9110",
    "9111",
    "9112",
    "9113",
    "9117",
    "9118",
    "9119",
    "912",
    "9120",
    "9122",
    "9123",
    "9125",
    "913",
    "9133",
    "9134",
    "9135",
    "9136",
    "9139",
    "9140",
    "9141",
    "9142",
    "9144",
    "9146",
    "9148",
    "9151",
    "9153",
    "9154",
    "9159",
    "916",
    "9161",
    "9162",
    "9164",
    "9165",
    "9167",
    "9168",
    "9170",
    "9171",
    "9172",
    "9173",
    "9174",
    "9176",
    "9177",
    "9178",
    "918",
    "9184",
    "9190",
    "9192",
    "9194",
    "9195",
    "9196",
    "9198",
    "9199",
    "92",
    "9201",
    "9202",
    "9211",
    "9212",
    "9213",
    "9215",
    "9219",
    "922",
    "9220",
    "9221",
    "9225",
    "9228",
    "9229",
    "9231",
    "9233",
    "9234",
    "9236",
    "9237",
    "9239",
    "9242",
    "9243",
    "9247",
    "9248",
    "9249",
    "925",
    "9250",
    "9251",
    "9255",
    "9256",
    "9258",
    "926",
    "9260",
    "9263",
    "9264",
    "9267",
    "9269",
    "9270",
    "9272",
    "9273",
    "9278",
    "9281",
    "9282",
    "9284",
    "9285",
    "9286",
    "9287",
    "9288",
    "929",
    "9292",
    "9296",
    "9297",
    "9298",
    "930",
    "9300",
    "9303",
    "9305",
    "9307",
    "9310",
    "9315",
    "9316",
    "9317",
    "9320",
    "9323",
    "9324",
    "9325",
    "9326",
    "9327",
    "9328",
    "9329",
    "933",
    "9330",
    "9334",
    "9337",
    "934",
    "9340",
    "9344",
    "9347",
    "9350",
    "9351",
    "9352",
    "9355",
    "9358",
    "9362",
    "9363",
    "9366",
    "9367",
    "9368",
    "9369",
    "9370",
    "9372",
    "9374",
    "9377",
    "9379",
    "938",
    "9384",
    "9386",
    "939",
    "9392",
    "9394",
    "9398",
    "9401",
    "9402",
    "9403",
    "9407",
    "9409",
    "9410",
    "9413",
    "9414",
    "9415",
    "9416",
    "9417",
    "9418",
    "9419",
    "9420",
    "9423",
    "9426",
    "9429",
    "9431",
    "9432",
    "9434",
    "944",
    "9440",
    "9445",
    "9447",
    "9450",
    "9451",
    "9452",
    "9456",
    "9457",
    "946",
    "9460",
    "9463",
    "9465",
    "9466",
    "9467",
    "9468",
    "947",
    "9471",
    "9474",
    "9475",
    "9476",
    "9479",
    "948",
    "9481",
    "9483",
    "9487",
    "9488",
    "9489",
    "9493",
    "9496",
    "9497",
    "9499",
    "9501",
    "9503",
    "9504",
    "9505",
    "9507",
    "9508",
    "9509",
    "9511",
    "9514",
    "9516",
    "9517",
    "9523",
    "9524",
    "9526",
    "9531",
    "9533",
    "9535",
    "9536",
    "9537",
    "9538",
    "9541",
    "9544",
    "9546",
    "9547",
    "9548",
    "9551",
    "9554",
    "9556",
    "9557",
    "9558",
    "9559",
    "956",
    "9560",
    "9561",
    "9564",
    "9566",
    "9569",
    "9574",
    "9575",
    "9576",
    "9577",
    "9579",
    "9580",
    "9583",
    "9584",
    "9585",
    "9586",
    "9588",
    "9592",
    "9593",
    "9594",
    "9595",
    "9596",
    "9597",
    "96",
    "960",
    "9602",
    "9603",
    "9604",
    "9607",
    "9612",
    "9614",
    "9617",
    "9618",
    "9619",
    "9622",
    "9628",
    "963",
    "9633",
    "9634",
    "9635",
    "9636",
    "9639",
    "9641",
    "9646",
    "9647",
    "9648",
    "9649",
    "965",
    "9650",
    "9651",
    "9653",
    "9657",
    "9659",
    "9660",
    "9663",
    "9666",
    "9667",
    "9668",
    "9669",
    "9676",
    "9677",
    "9678",
    "9686",
    "9689",
    "969",
    "9697",
    "9699",
    "970",
    "9701",
    "9703",
    "9705",
    "9706",
    "971",
    "9715",
    "9716",
    "9718",
    "9721",
    "9722",
    "9727",
    "9728",
    "9729",
    "9730",
    "9731",
    "9735",
    "9737",
    "9741",
    "9742",
    "9744",
    "9748",
    "9749",
    "975",
    "9754",
    "9757",
    "9761",
    "9764",
    "9768",
    "9769",
    "9771",
    "9772",
    "9773",
    "9775",
    "9779",
    "9780",
    "9786",
    "9787",
    "9788",
    "9789",
    "979",
    "9791",
    "9792",
    "9793",
    "9795",
    "9796",
    "9797",
    "9798",
    "9799",
    "98",
    "9802",
    "9803",
    "9804",
    "9808",
    "9809",
    "9810",
    "9812",
    "9814",
    "9816",
    "9817",
    "9818",
    "982",
    "9821",
    "9824",
    "9826",
    "9827",
    "983",
    "9832",
    "9835",
    "9838",
    "9839",
    "9841",
    "9842",
    "9843",
    "9845",
    "9849",
    "9850",
    "9864",
    "9866",
    "9867",
    "9869",
    "9870",
    "9871",
    "9872",
    "9874",
    "9875",
    "9876",
    "9878",
    "9879",
    "9881",
    "9882",
    "9889",
    "9891",
    "9893",
    "9894",
    "9896",
    "9898",
    "9899",
    "99",
    "9903",
    "9905",
    "9906",
    "9907",
    "991",
    "9912",
    "9913",
    "9914",
    "9916",
    "9917",
    "9918",
    "9919",
    "9920",
    "9921",
    "9925",
    "9927",
    "9929",
    "9931",
    "9933",
    "9936",
    "9937",
    "9938",
    "9939",
    "9940",
    "9941",
    "9942",
    "9944",
    "9945",
    "9946",
    "9947",
    "9948",
    "9949",
    "9951",
    "9953",
    "9955",
    "9958",
    "9960",
    "9963",
    "9964",
    "9965",
    "9967",
    "9968",
    "9969",
    "9970",
    "9973",
    "9974",
    "9975",
    "9976",
    "9977",
    "9978",
    "9979",
    "9982",
    "9983",
    "9987",
    "9988",
    "999",
    "9990",
    "9994",
    "9996",
    "9999"
  ],
  "kept_labels": [
    "1077",
    "1088",
    "1122",
    "1153",
    "1193",
    "1217",
    "1220",
    "1248",
    "1321",
    "1418",
    "1544",
    "1550",
    "1574",
    "1581",
    "1582",
    "1622",
    "1650",
    "1657",
    "1750",
    "1759",
    "1782",
    "1802",
    "1803",
    "1806",
    "1841",
    "1877",
    "1933",
    "1960",
    "1965",
    "1979",
    "2020",
    "2051",
    "2063",
    "2068",
    "2108",
    "212",
    "2146",
    "2153",
    "2157",
    "2201",
    "2257",
    "2273",
    "228",
    "2301",
    "2317",
    "238",
    "2402",
    "2426",
    "2436",
    "244",
    "2441",
    "2471",
    "2474",
    "2498",
    "250",
    "2504",
    "2518",
    "2525",
    "2535",
    "2537",
    "2541",
    "2545",
    "2672",
    "269",
    "2737",
    "2810",
    "2815",
    "283",
    "2839",
    "2883",
    "2912",
    "2914",
    "2919",
    "2981",
    "3010",
    "3119",
    "3241",
    "3242",
    "3249",
    "3252",
    "3259",
    "33",
    "3303",
    "3369",
    "3377",
    "3447",
    "3500",
    "3581",
    "3628",
    "3645",
    "3646",
    "3657",
    "3700",
    "3796",
    "3814",
    "3830",
    "3897",
    "3953",
    "397",
    "3986",
    "3998",
    "4043",
    "4057",
    "4066",
    "4074",
    "4079",
    "4100",
    "4108",
    "4209",
    "4240",
    "429",
    "4341",
    "4387",
    "4396",
    "4405",
    "4410",
    "4414",
    "46",
    "467",
    "4851",
    "548",
    "552",
    "558",
    "625",
    "635",
    "655",
    "658",
    "703",
    "706",
    "745",
    "749",
    "754",
    "755",
    "756",
    "792",
    "831",
    "834",
    "835",
    "836",
    "838",
    "849",
    "853",
    "854",
    "856",
    "858",
    "860",
    "863",
    "884",
    "887",
    "8924",
    "893",
    "902",
    "907",
    "909",
    "923",
    "924",
    "931",
    "932",
    "950",
    "951",
    "967",
    "968",
    "980",
    "988",
    "989",
    "998"
  ]
}
```

## Measurement — `runtime/benchmarks/qmul_overlap_control.json`

### Values

| Field | Value |
|---|---|
| `verdict` | ARTEFACT |
| `pairs` | 2500 |
| `qmul_genuine.median` | 0.3157 |
| `qmul_genuine.p5` | 0.0396 |
| `qmul_genuine.p95` | 0.6461 |
| `qmul_impostor_control.median` | 0.1507 |
| `qmul_impostor_control.p95` | 0.4343 |
| `nearest_tinyface_max_of_N.median` | 0.5223 |
| `nearest_tinyface_max_of_N.p95` | 0.6916 |
| `matched_null_diff_person_qmul_max_of_N.median` | 0.6001 |
| `matched_null_diff_person_qmul_max_of_N.p95` | 0.7712 |
| `lfw_clean_impostor.median` | 0.0029 |
| `lfw_clean_impostor.p95` | 0.1002 |

### Raw artefact

```json
{
  "verdict": "ARTEFACT",
  "pairs": 2500,
  "qmul_genuine": {
    "median": 0.3157,
    "p5": 0.0396,
    "p95": 0.6461
  },
  "qmul_impostor_control": {
    "median": 0.1507,
    "p95": 0.4343
  },
  "nearest_tinyface_max_of_N": {
    "median": 0.5223,
    "p95": 0.6916
  },
  "matched_null_diff_person_qmul_max_of_N": {
    "median": 0.6001,
    "p95": 0.7712
  },
  "lfw_clean_impostor": {
    "median": 0.0029,
    "p95": 0.1002
  }
}
```

## Measurement — `runtime/benchmarks/qmul_quality.json`

### Values

| Field | Value |
|---|---|
| `qmul_survface.images` | 3000 |
| `qmul_survface.height_px.5` | 11 |
| `qmul_survface.height_px.25` | 16 |
| `qmul_survface.height_px.50` | 27 |
| `qmul_survface.height_px.75` | 30 |
| `qmul_survface.height_px.95` | 35 |
| `qmul_survface.width_px.5` | 8 |
| `qmul_survface.width_px.25` | 14 |
| `qmul_survface.width_px.50` | 22 |
| `qmul_survface.width_px.75` | 24 |
| `qmul_survface.width_px.95` | 29 |
| `qmul_survface.min_hw[0]` | 8 |
| `qmul_survface.min_hw[1]` | 6 |
| `qmul_survface.max_hw[0]` | 75 |
| `qmul_survface.max_hw[1]` | 65 |
| `qmul_survface.median_area_px` | 594 |
| `qmul_survface.sharpness_median` | 481.05 |
| `qmul_survface.pct_under_32px_high` | 83.7 |
| `qmul_survface.pct_under_64px_high` | 99.9 |
| `tinyface.images` | 8171 |
| `tinyface.height_px.5` | 31 |
| `tinyface.height_px.25` | 32 |
| `tinyface.height_px.50` | 32 |
| `tinyface.height_px.75` | 32 |
| `tinyface.height_px.95` | 32 |
| `tinyface.width_px.5` | 24 |
| `tinyface.width_px.25` | 30 |
| `tinyface.width_px.50` | 32 |
| `tinyface.width_px.75` | 32 |
| `tinyface.width_px.95` | 32 |
| `tinyface.min_hw[0]` | 14 |
| `tinyface.min_hw[1]` | 12 |
| `tinyface.max_hw[0]` | 32 |
| `tinyface.max_hw[1]` | 32 |
| `tinyface.median_area_px` | 1024 |
| `tinyface.sharpness_median` | 764.49 |
| `tinyface.pct_under_32px_high` | 6.4 |
| `tinyface.pct_under_64px_high` | 100.0 |
| `casia_clean_reference.images` | 490 |
| `casia_clean_reference.height_px.5` | 112 |
| `casia_clean_reference.height_px.25` | 112 |
| `casia_clean_reference.height_px.50` | 112 |
| `casia_clean_reference.height_px.75` | 112 |
| `casia_clean_reference.height_px.95` | 112 |
| `casia_clean_reference.width_px.5` | 112 |
| `casia_clean_reference.width_px.25` | 112 |
| `casia_clean_reference.width_px.50` | 112 |
| `casia_clean_reference.width_px.75` | 112 |
| `casia_clean_reference.width_px.95` | 112 |
| `casia_clean_reference.min_hw[0]` | 112 |
| `casia_clean_reference.min_hw[1]` | 112 |
| `casia_clean_reference.max_hw[0]` | 112 |
| `casia_clean_reference.max_hw[1]` | 112 |
| `casia_clean_reference.median_area_px` | 12544 |
| `casia_clean_reference.sharpness_median` | 304.76 |
| `casia_clean_reference.pct_under_32px_high` | 0.0 |
| `casia_clean_reference.pct_under_64px_high` | 0.0 |
| `native_detectability.qmul.sampled` | 800 |
| `native_detectability.qmul.detected` | 0 |
| `native_detectability.qmul.detect_rate_pct` | 0.0 |
| `native_detectability.tinyface.sampled` | 800 |
| `native_detectability.tinyface.detected` | 0 |
| `native_detectability.tinyface.detect_rate_pct` | 0.0 |

### Raw artefact

```json
{
  "qmul_survface": {
    "images": 3000,
    "height_px": {
      "5": 11,
      "25": 16,
      "50": 27,
      "75": 30,
      "95": 35
    },
    "width_px": {
      "5": 8,
      "25": 14,
      "50": 22,
      "75": 24,
      "95": 29
    },
    "min_hw": [
      8,
      6
    ],
    "max_hw": [
      75,
      65
    ],
    "median_area_px": 594,
    "sharpness_median": 481.05,
    "pct_under_32px_high": 83.7,
    "pct_under_64px_high": 99.9
  },
  "tinyface": {
    "images": 8171,
    "height_px": {
      "5": 31,
      "25": 32,
      "50": 32,
      "75": 32,
      "95": 32
    },
    "width_px": {
      "5": 24,
      "25": 30,
      "50": 32,
      "75": 32,
      "95": 32
    },
    "min_hw": [
      14,
      12
    ],
    "max_hw": [
      32,
      32
    ],
    "median_area_px": 1024,
    "sharpness_median": 764.49,
    "pct_under_32px_high": 6.4,
    "pct_under_64px_high": 100.0
  },
  "casia_clean_reference": {
    "images": 490,
    "height_px": {
      "5": 112,
      "25": 112,
      "50": 112,
      "75": 112,
      "95": 112
    },
    "width_px": {
      "5": 112,
      "25": 112,
      "50": 112,
      "75": 112,
      "95": 112
    },
    "min_hw": [
      112,
      112
    ],
    "max_hw": [
      112,
      112
    ],
    "median_area_px": 12544,
    "sharpness_median": 304.76,
    "pct_under_32px_high": 0.0,
    "pct_under_64px_high": 0.0
  },
  "native_detectability": {
    "qmul": {
      "sampled": 800,
      "detected": 0,
      "detect_rate_pct": 0.0
    },
    "tinyface": {
      "sampled": 800,
      "detected": 0,
      "detect_rate_pct": 0.0
    }
  }
}
```

## Measurement — `runtime/benchmarks/train_eval_overlap.json`

### Values

| Field | Value |
|---|---|
| `model` | w600k_r50 |
| `method` | embedding near-duplicate detection; name matching impossible (training sets carry numeric IDs, .bin packs carry no labels) |
| `thresholds.near_duplicate` | 0.9 |
| `thresholds.probable_same_identity` | 0.7 |
| `thresholds.deployed_decision` | 0.2871 |
| `overlap_found` | True |
| `verdict` | OVERLAP DETECTED - training must not proceed until resolved |
| `limitation` | Sampling proves overlap EXISTS but cannot prove it is ABSENT. |
| `results.faces_webface_112x112.sampled_images` | 21144 |
| `results.faces_webface_112x112.sampled_identities` | 10572 |
| `results.faces_webface_112x112.per_identity_cap` | 2 |
| `results.faces_webface_112x112.per_eval.lfw.near_duplicate_ge_0.90` | 0 |
| `results.faces_webface_112x112.per_eval.lfw.probable_same_id_ge_0.70` | 13 |
| `results.faces_webface_112x112.per_eval.lfw.above_deployed_thr_0.2871` | 2665 |
| `results.faces_webface_112x112.per_eval.lfw.eval_images` | 12000 |
| `results.faces_webface_112x112.per_eval.lfw.max_similarity` | 0.8044803405643746 |
| `results.faces_webface_112x112.per_eval.lfw.mean_max_similarity` | 0.26862682101407254 |
| `results.faces_webface_112x112.per_eval.agedb_30.near_duplicate_ge_0.90` | 18 |
| `results.faces_webface_112x112.per_eval.agedb_30.probable_same_id_ge_0.70` | 139 |
| `results.faces_webface_112x112.per_eval.agedb_30.above_deployed_thr_0.2871` | 4440 |
| `results.faces_webface_112x112.per_eval.agedb_30.eval_images` | 12000 |
| `results.faces_webface_112x112.per_eval.agedb_30.max_similarity` | 0.9854185557879138 |
| `results.faces_webface_112x112.per_eval.agedb_30.mean_max_similarity` | 0.3157376072012279 |
| `results.faces_webface_112x112.per_eval.cfp_fp.near_duplicate_ge_0.90` | 2 |
| `results.faces_webface_112x112.per_eval.cfp_fp.probable_same_id_ge_0.70` | 361 |
| `results.faces_webface_112x112.per_eval.cfp_fp.above_deployed_thr_0.2871` | 6118 |
| `results.faces_webface_112x112.per_eval.cfp_fp.eval_images` | 14000 |
| `results.faces_webface_112x112.per_eval.cfp_fp.max_similarity` | 0.9650579088311979 |
| `results.faces_webface_112x112.per_eval.cfp_fp.mean_max_similarity` | 0.33615118004982325 |
| `results.faces_webface_112x112.per_eval.calfw.near_duplicate_ge_0.90` | 0 |
| `results.faces_webface_112x112.per_eval.calfw.probable_same_id_ge_0.70` | 44 |
| `results.faces_webface_112x112.per_eval.calfw.above_deployed_thr_0.2871` | 2743 |
| `results.faces_webface_112x112.per_eval.calfw.eval_images` | 12000 |
| `results.faces_webface_112x112.per_eval.calfw.max_similarity` | 0.8484095807387126 |
| `results.faces_webface_112x112.per_eval.calfw.mean_max_similarity` | 0.2701350432318123 |
| `results.faces_webface_112x112.per_eval.cplfw.near_duplicate_ge_0.90` | 0 |
| `results.faces_webface_112x112.per_eval.cplfw.probable_same_id_ge_0.70` | 7 |
| `results.faces_webface_112x112.per_eval.cplfw.above_deployed_thr_0.2871` | 3000 |
| `results.faces_webface_112x112.per_eval.cplfw.eval_images` | 12000 |
| `results.faces_webface_112x112.per_eval.cplfw.max_similarity` | 0.7890674046404245 |
| `results.faces_webface_112x112.per_eval.cplfw.mean_max_similarity` | 0.2745718728864982 |
| `results.faces_umd.sampled_images` | 16554 |
| `results.faces_umd.sampled_identities` | 8277 |
| `results.faces_umd.per_identity_cap` | 2 |
| `results.faces_umd.per_eval.lfw.near_duplicate_ge_0.90` | 9 |
| `results.faces_umd.per_eval.lfw.probable_same_id_ge_0.70` | 792 |
| `results.faces_umd.per_eval.lfw.above_deployed_thr_0.2871` | 4772 |
| `results.faces_umd.per_eval.lfw.eval_images` | 12000 |
| `results.faces_umd.per_eval.lfw.max_similarity` | 0.9863757934925138 |
| `results.faces_umd.per_eval.lfw.mean_max_similarity` | 0.3629139106755039 |
| `results.faces_umd.per_eval.agedb_30.near_duplicate_ge_0.90` | 61 |
| `results.faces_umd.per_eval.agedb_30.probable_same_id_ge_0.70` | 974 |
| `results.faces_umd.per_eval.agedb_30.above_deployed_thr_0.2871` | 9006 |
| `results.faces_umd.per_eval.agedb_30.eval_images` | 12000 |
| `results.faces_umd.per_eval.agedb_30.max_similarity` | 0.9890420639250317 |
| `results.faces_umd.per_eval.agedb_30.mean_max_similarity` | 0.4504648597289361 |
| `results.faces_umd.per_eval.cfp_fp.near_duplicate_ge_0.90` | 116 |
| `results.faces_umd.per_eval.cfp_fp.probable_same_id_ge_0.70` | 1705 |
| `results.faces_umd.per_eval.cfp_fp.above_deployed_thr_0.2871` | 8666 |
| `results.faces_umd.per_eval.cfp_fp.eval_images` | 14000 |
| `results.faces_umd.per_eval.cfp_fp.max_similarity` | 0.9704050292860567 |
| `results.faces_umd.per_eval.cfp_fp.mean_max_similarity` | 0.4431057705002491 |
| `results.faces_umd.per_eval.calfw.near_duplicate_ge_0.90` | 56 |
| `results.faces_umd.per_eval.calfw.probable_same_id_ge_0.70` | 955 |
| `results.faces_umd.per_eval.calfw.above_deployed_thr_0.2871` | 4909 |
| `results.faces_umd.per_eval.calfw.eval_images` | 12000 |
| `results.faces_umd.per_eval.calfw.max_similarity` | 0.9836030557022553 |
| `results.faces_umd.per_eval.calfw.mean_max_similarity` | 0.36111525649046006 |
| `results.faces_umd.per_eval.cplfw.near_duplicate_ge_0.90` | 11 |
| `results.faces_umd.per_eval.cplfw.probable_same_id_ge_0.70` | 407 |
| `results.faces_umd.per_eval.cplfw.above_deployed_thr_0.2871` | 4898 |
| `results.faces_umd.per_eval.cplfw.eval_images` | 12000 |
| `results.faces_umd.per_eval.cplfw.max_similarity` | 0.9799825405378588 |
| `results.faces_umd.per_eval.cplfw.mean_max_similarity` | 0.3472463744285455 |

### Raw artefact

```json
{
  "model": "w600k_r50",
  "method": "embedding near-duplicate detection; name matching impossible (training sets carry numeric IDs, .bin packs carry no labels)",
  "thresholds": {
    "near_duplicate": 0.9,
    "probable_same_identity": 0.7,
    "deployed_decision": 0.2871
  },
  "overlap_found": true,
  "verdict": "OVERLAP DETECTED - training must not proceed until resolved",
  "limitation": "Sampling proves overlap EXISTS but cannot prove it is ABSENT.",
  "results": {
    "faces_webface_112x112": {
      "sampled_images": 21144,
      "sampled_identities": 10572,
      "per_identity_cap": 2,
      "per_eval": {
        "lfw": {
          "near_duplicate_ge_0.90": 0,
          "probable_same_id_ge_0.70": 13,
          "above_deployed_thr_0.2871": 2665,
          "eval_images": 12000,
          "max_similarity": 0.8044803405643746,
          "mean_max_similarity": 0.26862682101407254
        },
        "agedb_30": {
          "near_duplicate_ge_0.90": 18,
          "probable_same_id_ge_0.70": 139,
          "above_deployed_thr_0.2871": 4440,
          "eval_images": 12000,
          "max_similarity": 0.9854185557879138,
          "mean_max_similarity": 0.3157376072012279
        },
        "cfp_fp": {
          "near_duplicate_ge_0.90": 2,
          "probable_same_id_ge_0.70": 361,
          "above_deployed_thr_0.2871": 6118,
          "eval_images": 14000,
          "max_similarity": 0.9650579088311979,
          "mean_max_similarity": 0.33615118004982325
        },
        "calfw": {
          "near_duplicate_ge_0.90": 0,
          "probable_same_id_ge_0.70": 44,
          "above_deployed_thr_0.2871": 2743,
          "eval_images": 12000,
          "max_similarity": 0.8484095807387126,
          "mean_max_similarity": 0.2701350432318123
        },
        "cplfw": {
          "near_duplicate_ge_0.90": 0,
          "probable_same_id_ge_0.70": 7,
          "above_deployed_thr_0.2871": 3000,
          "eval_images": 12000,
          "max_similarity": 0.7890674046404245,
          "mean_max_similarity": 0.2745718728864982
        }
      }
    },
    "faces_umd": {
      "sampled_images": 16554,
      "sampled_identities": 8277,
      "per_identity_cap": 2,
      "per_eval": {
        "lfw": {
          "near_duplicate_ge_0.90": 9,
          "probable_same_id_ge_0.70": 792,
          "above_deployed_thr_0.2871": 4772,
          "eval_images": 12000,
          "max_similarity": 0.9863757934925138,
          "mean_max_similarity": 0.3629139106755039
        },
        "agedb_30": {
          "near_duplicate_ge_0.90": 61,
          "probable_same_id_ge_0.70": 974,
          "above_deployed_thr_0.2871": 9006,
          "eval_images": 12000,
          "max_similarity": 0.9890420639250317,
          "mean_max_similarity": 0.4504648597289361
        },
        "cfp_fp": {
          "near_duplicate_ge_0.90": 116,
          "probable_same_id_ge_0.70": 1705,
          "above_deployed_thr_0.2871": 8666,
          "eval_images": 14000,
          "max_similarity": 0.9704050292860567,
          "mean_max_similarity": 0.4431057705002491
        },
        "calfw": {
          "near_duplicate_ge_0.90": 56,
          "probable_same_id_ge_0.70": 955,
          "above_deployed_thr_0.2871": 4909,
          "eval_images": 12000,
          "max_similarity": 0.9836030557022553,
          "mean_max_similarity": 0.36111525649046006
        },
        "cplfw": {
          "near_duplicate_ge_0.90": 11,
          "probable_same_id_ge_0.70": 407,
          "above_deployed_thr_0.2871": 4898,
          "eval_images": 12000,
          "max_similarity": 0.9799825405378588,
          "mean_max_similarity": 0.3472463744285455
        }
      }
    }
  }
}
```

## Measurement — `runtime/benchmarks/train_eval_overlap_megaface.json`

### Values

| Field | Value |
|---|---|
| `model` | w600k_r50 |
| `method` | embedding near-duplicate detection; name matching impossible (training sets carry numeric IDs, .bin packs carry no labels) |
| `thresholds.near_duplicate` | 0.9 |
| `thresholds.probable_same_identity` | 0.7 |
| `thresholds.deployed_decision` | 0.2871 |
| `overlap_found` | True |
| `verdict` | OVERLAP DETECTED - training must not proceed until resolved |
| `limitation` | Sampling proves overlap EXISTS but cannot prove it is ABSENT. |
| `results.faces_megafacetrain_112x112.sampled_images` | 22000 |
| `results.faces_megafacetrain_112x112.sampled_identities` | 22000 |
| `results.faces_megafacetrain_112x112.per_identity_cap` | 1 |
| `results.faces_megafacetrain_112x112.per_eval.lfw.near_duplicate_ge_0.90` | 0 |
| `results.faces_megafacetrain_112x112.per_eval.lfw.probable_same_id_ge_0.70` | 34 |
| `results.faces_megafacetrain_112x112.per_eval.lfw.above_deployed_thr_0.2871` | 2416 |
| `results.faces_megafacetrain_112x112.per_eval.lfw.eval_images` | 12000 |
| `results.faces_megafacetrain_112x112.per_eval.lfw.max_similarity` | 0.8096097107825015 |
| `results.faces_megafacetrain_112x112.per_eval.lfw.mean_max_similarity` | 0.2698333131557208 |
| `results.faces_megafacetrain_112x112.per_eval.agedb_30.near_duplicate_ge_0.90` | 0 |
| `results.faces_megafacetrain_112x112.per_eval.agedb_30.probable_same_id_ge_0.70` | 5 |
| `results.faces_megafacetrain_112x112.per_eval.agedb_30.above_deployed_thr_0.2871` | 1359 |
| `results.faces_megafacetrain_112x112.per_eval.agedb_30.eval_images` | 12000 |
| `results.faces_megafacetrain_112x112.per_eval.agedb_30.max_similarity` | 0.7730511373128426 |
| `results.faces_megafacetrain_112x112.per_eval.agedb_30.mean_max_similarity` | 0.2572680043825828 |
| `results.faces_megafacetrain_112x112.per_eval.cfp_fp.near_duplicate_ge_0.90` | 1 |
| `results.faces_megafacetrain_112x112.per_eval.cfp_fp.probable_same_id_ge_0.70` | 341 |
| `results.faces_megafacetrain_112x112.per_eval.cfp_fp.above_deployed_thr_0.2871` | 3193 |
| `results.faces_megafacetrain_112x112.per_eval.cfp_fp.eval_images` | 14000 |
| `results.faces_megafacetrain_112x112.per_eval.cfp_fp.max_similarity` | 0.9312711883634887 |
| `results.faces_megafacetrain_112x112.per_eval.cfp_fp.mean_max_similarity` | 0.2889444081353978 |
| `results.faces_megafacetrain_112x112.per_eval.calfw.near_duplicate_ge_0.90` | 0 |
| `results.faces_megafacetrain_112x112.per_eval.calfw.probable_same_id_ge_0.70` | 53 |
| `results.faces_megafacetrain_112x112.per_eval.calfw.above_deployed_thr_0.2871` | 2077 |
| `results.faces_megafacetrain_112x112.per_eval.calfw.eval_images` | 12000 |
| `results.faces_megafacetrain_112x112.per_eval.calfw.max_similarity` | 0.8828490469118752 |
| `results.faces_megafacetrain_112x112.per_eval.calfw.mean_max_similarity` | 0.26608938330093584 |
| `results.faces_megafacetrain_112x112.per_eval.cplfw.near_duplicate_ge_0.90` | 0 |
| `results.faces_megafacetrain_112x112.per_eval.cplfw.probable_same_id_ge_0.70` | 120 |
| `results.faces_megafacetrain_112x112.per_eval.cplfw.above_deployed_thr_0.2871` | 2521 |
| `results.faces_megafacetrain_112x112.per_eval.cplfw.eval_images` | 12000 |
| `results.faces_megafacetrain_112x112.per_eval.cplfw.max_similarity` | 0.833853766536533 |
| `results.faces_megafacetrain_112x112.per_eval.cplfw.mean_max_similarity` | 0.2774835992156923 |

### Raw artefact

```json
{
  "model": "w600k_r50",
  "method": "embedding near-duplicate detection; name matching impossible (training sets carry numeric IDs, .bin packs carry no labels)",
  "thresholds": {
    "near_duplicate": 0.9,
    "probable_same_identity": 0.7,
    "deployed_decision": 0.2871
  },
  "overlap_found": true,
  "verdict": "OVERLAP DETECTED - training must not proceed until resolved",
  "limitation": "Sampling proves overlap EXISTS but cannot prove it is ABSENT.",
  "results": {
    "faces_megafacetrain_112x112": {
      "sampled_images": 22000,
      "sampled_identities": 22000,
      "per_identity_cap": 1,
      "per_eval": {
        "lfw": {
          "near_duplicate_ge_0.90": 0,
          "probable_same_id_ge_0.70": 34,
          "above_deployed_thr_0.2871": 2416,
          "eval_images": 12000,
          "max_similarity": 0.8096097107825015,
          "mean_max_similarity": 0.2698333131557208
        },
        "agedb_30": {
          "near_duplicate_ge_0.90": 0,
          "probable_same_id_ge_0.70": 5,
          "above_deployed_thr_0.2871": 1359,
          "eval_images": 12000,
          "max_similarity": 0.7730511373128426,
          "mean_max_similarity": 0.2572680043825828
        },
        "cfp_fp": {
          "near_duplicate_ge_0.90": 1,
          "probable_same_id_ge_0.70": 341,
          "above_deployed_thr_0.2871": 3193,
          "eval_images": 14000,
          "max_similarity": 0.9312711883634887,
          "mean_max_similarity": 0.2889444081353978
        },
        "calfw": {
          "near_duplicate_ge_0.90": 0,
          "probable_same_id_ge_0.70": 53,
          "above_deployed_thr_0.2871": 2077,
          "eval_images": 12000,
          "max_similarity": 0.8828490469118752,
          "mean_max_similarity": 0.26608938330093584
        },
        "cplfw": {
          "near_duplicate_ge_0.90": 0,
          "probable_same_id_ge_0.70": 120,
          "above_deployed_thr_0.2871": 2521,
          "eval_images": 12000,
          "max_similarity": 0.833853766536533,
          "mean_max_similarity": 0.2774835992156923
        }
      }
    }
  }
}
```


# Fine-tuning outcomes

## Protocol as implemented — `backend/scripts/finetune_degraded.py`

```text
Phase 6 step 4 — fine-tune for DEGRADED imagery, from ArcFace weights,
on a decontaminated subset.

    python backend/scripts/finetune_degraded.py --steps 2000

Everything the previous attempt got wrong is addressed here:

  item 36  training identities come from exclusion_list.json — the 692
           identities that matched an evaluation image are dropped.
  item 37  every batch mixes clean and SIMULATED DEGRADED crops. This is the
           whole point: the target is TinyFace-grade imagery (median 32x32),
           and a model fine-tuned only on clean faces cannot improve it.
  item 38  a held-out validation split, disjoint by IDENTITY, tracked each eval.
  item 39  initialised from the deployed ArcFace ONNX via onnx2torch — NOT
           ImageNet. This is what made the earlier run score at chance.
  item 40  hard-negative aware: ArcFace's angular margin already concentrates
           gradient on hard samples, and the degraded view of an image is by
           construction the hard positive of its clean counterpart.
  item 41  early stopping on validation loss, not a fixed epoch count.

WHAT IS DELIBERATELY NOT CLAIMED
--------------------------------
~9,880 identities is still ~36x fewer than glintr100 saw. The realistic goal
is a modest gain on degraded imagery WITHOUT regressing clean accuracy, not a
new state of the art. A run that improves nothing is a valid result and must be
reported as such — see BENCHMARKS.md §6b.
```

## Protocol as implemented — `backend/scripts/finetune_qmul.py`

```text
Fine-tune for degraded imagery using REAL surveillance capture (QMUL-SurvFace),
not synthetic blur.

    python backend/scripts/finetune_qmul.py --steps 6000

This is the second attempt. The first (BENCHMARKS.md §6d) used synthetic
degradation -- bicubic down/up, Gaussian blur, JPEG -- and made the model WORSE
on every benchmark, worst of all on TinyFace (-3.07pp), the exact condition it
targeted. The diagnosis was a domain gap: the model learned to invert that
specific synthetic pipeline, which is not what a distant camera produces.

WHAT IS DIFFERENT HERE
----------------------
  degraded source   real QMUL-SurvFace capture (median 27x22px, 84% under 32px)
                    instead of synthetically degraded clean photos. NO synthetic
                    blur/JPEG is applied anywhere in this script.
  clean anchor      every batch also carries CASIA clean images, so clean-set
                    accuracy is trained against rather than sacrificed (item 6).
  validation        FIXED, PUBLISHED pair lists -- never a sampled proxy.

THE PROXY RULE (item 8) -- THE MOST IMPORTANT PART OF THIS FILE
---------------------------------------------------------------
The last run's training-time proxy reported +0.058 "improvement" while the real
benchmarks showed regression. The proxy was resampled every evaluation, so its
own noise (~0.06, measurable during the frozen-backbone phase where learning was
impossible) was as large as the effect it claimed to detect.

So nothing here early-stops on a resampled quantity. Both validation signals are
FIXED pair lists, scored through `evaluate_pairs` -- the same 10-fold harness
that produces every number in BENCHMARKS.md §2:

  degraded  QMUL's own published verification protocol: 5,320 positive and
            5,320 negative pairs over 4,888 identities that are VERIFIED
            disjoint from the 5,319 training identities (0 overlap).
  clean     a fixed pair list over 500 CASIA identities held out of training.

Neither is among the seven reporting benchmarks, so early stopping cannot leak
into the reported result. The reported result comes only from
eval_finetuned_checkpoint.py.
```

## Protocol as implemented — `backend/scripts/eval_finetuned_checkpoint.py`

```text
Phase 6 step 4 (item 42) — score a fine-tuned checkpoint on EVERY benchmark,
against the deployed model, on identical inputs.

    python backend/scripts/eval_finetuned_checkpoint.py

The point of this script is that it cannot flatter the checkpoint. The
fine-tuned backbone is wrapped in a shim exposing the same `get_feat(images)`
signature insightface's recogniser has, so the pair lists, the flip
augmentation, the 10-fold cross-validation and the threshold fitting are the
SAME CODE that produced the numbers in BENCHMARKS.md §2 and §4. The only thing
that changes between the two columns is the weights.

Both models are scored in this run rather than reading the baseline from cache,
so a stale cache cannot produce a fake improvement.

REGRESSIONS ARE REPORTED, NOT HIDDEN. A fine-tune that trades clean accuracy
for degraded accuracy is a real and possibly acceptable trade, but it is only
assessable if both halves are printed.
```

## Measurement — `runtime/benchmarks/finetuned.json`

### Per-configuration results (2 rows)

| dataset | config | n_pairs | n_genuine | n_impostor | accuracy_mean | accuracy_std | threshold_mean | threshold_std | oracle_accuracy | oracle_threshold | tar_at_far_1e2 | tar_at_far_1e3 | tar_at_far_1e4 | auc | eer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lfw | finetuned:arcface_ft_v1_20260730.pt | 6000 | 3000 | 3000 | 0.56617 | 0.02191 | 0.33312 | 0.01668 | 0.56917 | 0.32497 | 0.02733 | 0.00667 | 0.00200 | 0.58446 | 0.43167 |
| agedb_30 | finetuned:arcface_ft_v1_20260730.pt | 6000 | 3000 | 3000 | 0.49383 | 0.01410 | 0.10587 | 0.44436 | 0.50250 | -0.16952 | 0.01300 | 0.00133 | 0.00100 | 0.48971 | 0.50533 |

### Raw artefact

```json
[
  {
    "dataset": "lfw",
    "config": "finetuned:arcface_ft_v1_20260730.pt",
    "n_pairs": 6000,
    "n_genuine": 3000,
    "n_impostor": 3000,
    "accuracy_mean": 0.5661666666666667,
    "accuracy_std": 0.02190953622918062,
    "threshold_mean": 0.33311686273422364,
    "threshold_std": 0.016682547414806233,
    "oracle_accuracy": 0.5691666666666667,
    "oracle_threshold": 0.3249741806423183,
    "tar_at_far_1e2": 0.027333333333333334,
    "tar_at_far_1e3": 0.006666666666666667,
    "tar_at_far_1e4": 0.002,
    "auc": 0.584461,
    "eer": 0.43166666666666664
  },
  {
    "dataset": "agedb_30",
    "config": "finetuned:arcface_ft_v1_20260730.pt",
    "n_pairs": 6000,
    "n_genuine": 3000,
    "n_impostor": 3000,
    "accuracy_mean": 0.49383333333333346,
    "accuracy_std": 0.014103781998693353,
    "threshold_mean": 0.10586762794174735,
    "threshold_std": 0.44435667442980176,
    "oracle_accuracy": 0.5025,
    "oracle_threshold": -0.16952267290085915,
    "tar_at_far_1e2": 0.013,
    "tar_at_far_1e3": 0.0013333333333333333,
    "tar_at_far_1e4": 0.001,
    "auc": 0.4897142222222222,
    "eer": 0.5053333333333334
  }
]
```

## Measurement — `runtime/benchmarks/finetuned_qmul_v2.json`

### Values

| Field | Value |
|---|---|
| `checkpoint` | runtime\checkpoints\arcface_qmul_v2.pt |
| `checkpoint_meta.step` | 5500 |
| `checkpoint_meta.degraded_val` | 0.8173872180451127 |
| `checkpoint_meta.clean_val` | 0.9416666666666667 |
| `checkpoint_meta.baseline_degraded` | 0.6900375939849624 |
| `checkpoint_meta.baseline_clean` | 0.943 |
| `checkpoint_meta.n_classes` | 15199 |
| `baseline_model` | w600k_r50 (deployed) |
| `note` | Both models scored in the same run on identical pair lists. |
| `results.lfw.deployed.accuracy_pct` | 99.783 |
| `results.lfw.deployed.tar_far_1e3_pct` | 99.7 |
| `results.lfw.deployed.auc` | 0.99943 |
| `results.lfw.finetuned.accuracy_pct` | 99.717 |
| `results.lfw.finetuned.tar_far_1e3_pct` | 99.667 |
| `results.lfw.finetuned.auc` | 0.99941 |
| `results.agedb_30.deployed.accuracy_pct` | 98.15 |
| `results.agedb_30.deployed.tar_far_1e3_pct` | 96.033 |
| `results.agedb_30.deployed.auc` | 0.9913 |
| `results.agedb_30.finetuned.accuracy_pct` | 97.783 |
| `results.agedb_30.finetuned.tar_far_1e3_pct` | 88.1 |
| `results.agedb_30.finetuned.auc` | 0.99142 |
| `results.cfp_fp.deployed.accuracy_pct` | 97.443 |
| `results.cfp_fp.deployed.tar_far_1e3_pct` | 94.686 |
| `results.cfp_fp.deployed.auc` | 0.98023 |
| `results.cfp_fp.finetuned.accuracy_pct` | 97.171 |
| `results.cfp_fp.finetuned.tar_far_1e3_pct` | 92.829 |
| `results.cfp_fp.finetuned.auc` | 0.97406 |
| `results.cfp_ff.deployed.accuracy_pct` | 99.871 |
| `results.cfp_ff.deployed.tar_far_1e3_pct` | 99.857 |
| `results.cfp_ff.deployed.auc` | 0.99978 |
| `results.cfp_ff.finetuned.accuracy_pct` | 99.857 |
| `results.cfp_ff.finetuned.tar_far_1e3_pct` | 99.8 |
| `results.cfp_ff.finetuned.auc` | 0.99971 |
| `results.calfw.deployed.accuracy_pct` | 95.95 |
| `results.calfw.deployed.tar_far_1e3_pct` | 92.1 |
| `results.calfw.deployed.auc` | 0.97755 |
| `results.calfw.finetuned.accuracy_pct` | 95.883 |
| `results.calfw.finetuned.tar_far_1e3_pct` | 90.533 |
| `results.calfw.finetuned.auc` | 0.97735 |
| `results.cplfw.deployed.accuracy_pct` | 94.467 |
| `results.cplfw.deployed.tar_far_1e3_pct` | 87.4 |
| `results.cplfw.deployed.auc` | 0.96425 |
| `results.cplfw.finetuned.accuracy_pct` | 93.333 |
| `results.cplfw.finetuned.tar_far_1e3_pct` | 81.733 |
| `results.cplfw.finetuned.auc` | 0.94468 |
| `results.tinyface.deployed.accuracy_pct` | 82.45 |
| `results.tinyface.deployed.tar_far_1e3_pct` | 33.133 |
| `results.tinyface.deployed.auc` | 0.89217 |
| `results.tinyface.finetuned.accuracy_pct` | 82.383 |
| `results.tinyface.finetuned.tar_far_1e3_pct` | 38.1 |
| `results.tinyface.finetuned.auc` | 0.901 |

### Raw artefact

```json
{
  "checkpoint": "runtime\\checkpoints\\arcface_qmul_v2.pt",
  "checkpoint_meta": {
    "step": 5500,
    "degraded_val": 0.8173872180451127,
    "clean_val": 0.9416666666666667,
    "baseline_degraded": 0.6900375939849624,
    "baseline_clean": 0.943,
    "n_classes": 15199
  },
  "baseline_model": "w600k_r50 (deployed)",
  "note": "Both models scored in the same run on identical pair lists.",
  "results": {
    "lfw": {
      "deployed": {
        "accuracy_pct": 99.783,
        "tar_far_1e3_pct": 99.7,
        "auc": 0.99943
      },
      "finetuned": {
        "accuracy_pct": 99.717,
        "tar_far_1e3_pct": 99.667,
        "auc": 0.99941
      }
    },
    "agedb_30": {
      "deployed": {
        "accuracy_pct": 98.15,
        "tar_far_1e3_pct": 96.033,
        "auc": 0.9913
      },
      "finetuned": {
        "accuracy_pct": 97.783,
        "tar_far_1e3_pct": 88.1,
        "auc": 0.99142
      }
    },
    "cfp_fp": {
      "deployed": {
        "accuracy_pct": 97.443,
        "tar_far_1e3_pct": 94.686,
        "auc": 0.98023
      },
      "finetuned": {
        "accuracy_pct": 97.171,
        "tar_far_1e3_pct": 92.829,
        "auc": 0.97406
      }
    },
    "cfp_ff": {
      "deployed": {
        "accuracy_pct": 99.871,
        "tar_far_1e3_pct": 99.857,
        "auc": 0.99978
      },
      "finetuned": {
        "accuracy_pct": 99.857,
        "tar_far_1e3_pct": 99.8,
        "auc": 0.99971
      }
    },
    "calfw": {
      "deployed": {
        "accuracy_pct": 95.95,
        "tar_far_1e3_pct": 92.1,
        "auc": 0.97755
      },
      "finetuned": {
        "accuracy_pct": 95.883,
        "tar_far_1e3_pct": 90.533,
        "auc": 0.97735
      }
    },
    "cplfw": {
      "deployed": {
        "accuracy_pct": 94.467,
        "tar_far_1e3_pct": 87.4,
        "auc": 0.96425
      },
      "finetuned": {
        "accuracy_pct": 93.333,
        "tar_far_1e3_pct": 81.733,
        "auc": 0.94468
      }
    },
    "tinyface": {
      "deployed": {
        "accuracy_pct": 82.45,
        "tar_far_1e3_pct": 33.133,
        "auc": 0.89217
      },
      "finetuned": {
        "accuracy_pct": 82.383,
        "tar_far_1e3_pct": 38.1,
        "auc": 0.901
      }
    }
  }
}
```

## Measurement — `runtime/benchmarks/finetuned_v1.json`

### Values

| Field | Value |
|---|---|
| `checkpoint` | C:\Users\hello\Desktop\nexgenforensics\runtime\checkpoints\arcface_degraded_v1.pt |
| `checkpoint_meta.val_margin` | 0.60551618039608 |
| `checkpoint_meta.step` | 3000 |
| `checkpoint_meta.n_classes` | 9380 |
| `baseline_model` | w600k_r50 (deployed) |
| `note` | Both models scored in the same run on identical pair lists. |
| `results.lfw.deployed.accuracy_pct` | 99.783 |
| `results.lfw.deployed.tar_far_1e3_pct` | 99.7 |
| `results.lfw.deployed.auc` | 0.99943 |
| `results.lfw.finetuned.accuracy_pct` | 99.75 |
| `results.lfw.finetuned.tar_far_1e3_pct` | 99.667 |
| `results.lfw.finetuned.auc` | 0.99963 |
| `results.agedb_30.deployed.accuracy_pct` | 98.15 |
| `results.agedb_30.deployed.tar_far_1e3_pct` | 96.033 |
| `results.agedb_30.deployed.auc` | 0.9913 |
| `results.agedb_30.finetuned.accuracy_pct` | 97.383 |
| `results.agedb_30.finetuned.tar_far_1e3_pct` | 86.967 |
| `results.agedb_30.finetuned.auc` | 0.99184 |
| `results.cfp_fp.deployed.accuracy_pct` | 97.443 |
| `results.cfp_fp.deployed.tar_far_1e3_pct` | 94.686 |
| `results.cfp_fp.deployed.auc` | 0.98023 |
| `results.cfp_fp.finetuned.accuracy_pct` | 97.229 |
| `results.cfp_fp.finetuned.tar_far_1e3_pct` | 93.943 |
| `results.cfp_fp.finetuned.auc` | 0.98135 |
| `results.calfw.deployed.accuracy_pct` | 95.95 |
| `results.calfw.deployed.tar_far_1e3_pct` | 92.1 |
| `results.calfw.deployed.auc` | 0.97755 |
| `results.calfw.finetuned.accuracy_pct` | 95.617 |
| `results.calfw.finetuned.tar_far_1e3_pct` | 88.633 |
| `results.calfw.finetuned.auc` | 0.97877 |
| `results.cplfw.deployed.accuracy_pct` | 94.467 |
| `results.cplfw.deployed.tar_far_1e3_pct` | 87.4 |
| `results.cplfw.deployed.auc` | 0.96425 |
| `results.cplfw.finetuned.accuracy_pct` | 93.883 |
| `results.cplfw.finetuned.tar_far_1e3_pct` | 85.133 |
| `results.cplfw.finetuned.auc` | 0.96017 |
| `results.tinyface.deployed.accuracy_pct` | 82.45 |
| `results.tinyface.deployed.tar_far_1e3_pct` | 33.133 |
| `results.tinyface.deployed.auc` | 0.89217 |
| `results.tinyface.finetuned.accuracy_pct` | 79.383 |
| `results.tinyface.finetuned.tar_far_1e3_pct` | 22.233 |
| `results.tinyface.finetuned.auc` | 0.8694 |

### Raw artefact

```json
{
  "checkpoint": "C:\\Users\\hello\\Desktop\\nexgenforensics\\runtime\\checkpoints\\arcface_degraded_v1.pt",
  "checkpoint_meta": {
    "val_margin": 0.60551618039608,
    "step": 3000,
    "n_classes": 9380
  },
  "baseline_model": "w600k_r50 (deployed)",
  "note": "Both models scored in the same run on identical pair lists.",
  "results": {
    "lfw": {
      "deployed": {
        "accuracy_pct": 99.783,
        "tar_far_1e3_pct": 99.7,
        "auc": 0.99943
      },
      "finetuned": {
        "accuracy_pct": 99.75,
        "tar_far_1e3_pct": 99.667,
        "auc": 0.99963
      }
    },
    "agedb_30": {
      "deployed": {
        "accuracy_pct": 98.15,
        "tar_far_1e3_pct": 96.033,
        "auc": 0.9913
      },
      "finetuned": {
        "accuracy_pct": 97.383,
        "tar_far_1e3_pct": 86.967,
        "auc": 0.99184
      }
    },
    "cfp_fp": {
      "deployed": {
        "accuracy_pct": 97.443,
        "tar_far_1e3_pct": 94.686,
        "auc": 0.98023
      },
      "finetuned": {
        "accuracy_pct": 97.229,
        "tar_far_1e3_pct": 93.943,
        "auc": 0.98135
      }
    },
    "calfw": {
      "deployed": {
        "accuracy_pct": 95.95,
        "tar_far_1e3_pct": 92.1,
        "auc": 0.97755
      },
      "finetuned": {
        "accuracy_pct": 95.617,
        "tar_far_1e3_pct": 88.633,
        "auc": 0.97877
      }
    },
    "cplfw": {
      "deployed": {
        "accuracy_pct": 94.467,
        "tar_far_1e3_pct": 87.4,
        "auc": 0.96425
      },
      "finetuned": {
        "accuracy_pct": 93.883,
        "tar_far_1e3_pct": 85.133,
        "auc": 0.96017
      }
    },
    "tinyface": {
      "deployed": {
        "accuracy_pct": 82.45,
        "tar_far_1e3_pct": 33.133,
        "auc": 0.89217
      },
      "finetuned": {
        "accuracy_pct": 79.383,
        "tar_far_1e3_pct": 22.233,
        "auc": 0.8694
      }
    }
  }
}
```


# Quality-routed model selection

## Protocol as implemented — `backend/scripts/evaluate_routed_engine.py`

```text
Can the QMUL checkpoint be USED, rather than shelved?

    python backend/scripts/evaluate_routed_engine.py

BENCHMARKS.md records the fine-tune as "no improvement": accuracy moved nowhere
on any of the seven benchmarks. But the per-metric table says something more
specific than the accuracy column does. At the 0.1% false-match operating point
a forensic deployment actually uses:

    TinyFace   TAR@FAR0.1%   33.13 -> 38.10   (+4.97pp)
    AgeDB-30   TAR@FAR0.1%   96.03 -> 88.10   (-7.93pp)
    CPLFW      TAR@FAR0.1%   87.40 -> 81.73   (-5.67pp)

That is not a worse model. It is a DIFFERENT model: better where images are
degraded, worse where they are clean but hard (age, pose). A single global
choice between them throws away whichever advantage it does not pick.

This script tests whether choosing PER PROBE recovers both, using the quality
score the pipeline already computes on every request -- so routing costs no
extra inference.

THREE QUESTIONS, IN ORDER. The third only matters if the first two hold.

 1. Are the two embedding spaces compatible? If a template enrolled under one
    model can be compared against a probe under the other, routing is free
    everywhere. If not, 1:N search needs BOTH templates stored per subject and
    that is a real cost, not a detail.

 2. Does the quality score actually separate degraded from clean imagery? If it
    does not, there is nothing to route on and the idea dies here.

 3. Does routing beat the deployed model on TinyFace WITHOUT regressing the
    clean sets? This is the only claim worth making, and it is the one that
    would be quoted, so it is measured end to end rather than inferred from the
    two columns above.
```

## Measurement — `runtime/benchmarks/routed_engine.json`

### Values

| Field | Value |
|---|---|
| `cross_model_same_image_median` | 0.8558 |
| `embedding_spaces_compatible` | True |
| `quality_by_dataset.lfw` | 0.7424 |
| `quality_by_dataset.agedb_30` | 0.7957 |
| `quality_by_dataset.cfp_fp` | 0.7753 |
| `quality_by_dataset.cfp_ff` | 0.7859 |
| `quality_by_dataset.calfw` | 0.7863 |
| `quality_by_dataset.cplfw` | 0.7551 |
| `quality_by_dataset.tinyface` | 0.5023 |
| `quality_separation` | 0.2783 |
| `verdict` | DO NOT ADOPT |
| `routing_threshold` | 0.581 |
| `results.lfw.deployed_acc` | 99.783 |
| `results.lfw.specialist_acc` | 99.717 |
| `results.lfw.routed_acc` | 99.767 |
| `results.lfw.deployed_tar_1e3` | 99.7 |
| `results.lfw.specialist_tar_1e3` | 99.667 |
| `results.lfw.routed_tar_1e3` | 99.667 |
| `results.lfw.fraction_routed_to_specialist` | 0.0937 |
| `results.agedb_30.deployed_acc` | 98.15 |
| `results.agedb_30.specialist_acc` | 97.783 |
| `results.agedb_30.routed_acc` | 98.1 |
| `results.agedb_30.deployed_tar_1e3` | 96.033 |
| `results.agedb_30.specialist_tar_1e3` | 88.1 |
| `results.agedb_30.routed_tar_1e3` | 94.933 |
| `results.agedb_30.fraction_routed_to_specialist` | 0.0318 |
| `results.cfp_fp.deployed_acc` | 97.443 |
| `results.cfp_fp.specialist_acc` | 97.171 |
| `results.cfp_fp.routed_acc` | 97.429 |
| `results.cfp_fp.deployed_tar_1e3` | 94.686 |
| `results.cfp_fp.specialist_tar_1e3` | 92.829 |
| `results.cfp_fp.routed_tar_1e3` | 94.657 |
| `results.cfp_fp.fraction_routed_to_specialist` | 0.0353 |
| `results.cfp_ff.deployed_acc` | 99.871 |
| `results.cfp_ff.specialist_acc` | 99.857 |
| `results.cfp_ff.routed_acc` | 99.871 |
| `results.cfp_ff.deployed_tar_1e3` | 99.857 |
| `results.cfp_ff.specialist_tar_1e3` | 99.8 |
| `results.cfp_ff.routed_tar_1e3` | 99.857 |
| `results.cfp_ff.fraction_routed_to_specialist` | 0.0007 |
| `results.calfw.deployed_acc` | 95.95 |
| `results.calfw.specialist_acc` | 95.883 |
| `results.calfw.routed_acc` | 96.017 |
| `results.calfw.deployed_tar_1e3` | 92.1 |
| `results.calfw.specialist_tar_1e3` | 90.533 |
| `results.calfw.routed_tar_1e3` | 92.1 |
| `results.calfw.fraction_routed_to_specialist` | 0.0085 |
| `results.cplfw.deployed_acc` | 94.467 |
| `results.cplfw.specialist_acc` | 93.333 |
| `results.cplfw.routed_acc` | 94.233 |
| `results.cplfw.deployed_tar_1e3` | 87.4 |
| `results.cplfw.specialist_tar_1e3` | 81.733 |
| `results.cplfw.routed_tar_1e3` | 87.233 |
| `results.cplfw.fraction_routed_to_specialist` | 0.1542 |
| `results.tinyface.deployed_acc` | 82.45 |
| `results.tinyface.specialist_acc` | 82.383 |
| `results.tinyface.routed_acc` | 82.45 |
| `results.tinyface.deployed_tar_1e3` | 33.133 |
| `results.tinyface.specialist_tar_1e3` | 38.1 |
| `results.tinyface.routed_tar_1e3` | 37.933 |
| `results.tinyface.fraction_routed_to_specialist` | 0.9878 |

### Raw artefact

```json
{
  "cross_model_same_image_median": 0.8558,
  "embedding_spaces_compatible": true,
  "quality_by_dataset": {
    "lfw": 0.7424,
    "agedb_30": 0.7957,
    "cfp_fp": 0.7753,
    "cfp_ff": 0.7859,
    "calfw": 0.7863,
    "cplfw": 0.7551,
    "tinyface": 0.5023
  },
  "quality_separation": 0.2783,
  "verdict": "DO NOT ADOPT",
  "routing_threshold": 0.581,
  "results": {
    "lfw": {
      "deployed_acc": 99.783,
      "specialist_acc": 99.717,
      "routed_acc": 99.767,
      "deployed_tar_1e3": 99.7,
      "specialist_tar_1e3": 99.667,
      "routed_tar_1e3": 99.667,
      "fraction_routed_to_specialist": 0.0937
    },
    "agedb_30": {
      "deployed_acc": 98.15,
      "specialist_acc": 97.783,
      "routed_acc": 98.1,
      "deployed_tar_1e3": 96.033,
      "specialist_tar_1e3": 88.1,
      "routed_tar_1e3": 94.933,
      "fraction_routed_to_specialist": 0.0318
    },
    "cfp_fp": {
      "deployed_acc": 97.443,
      "specialist_acc": 97.171,
      "routed_acc": 97.429,
      "deployed_tar_1e3": 94.686,
      "specialist_tar_1e3": 92.829,
      "routed_tar_1e3": 94.657,
      "fraction_routed_to_specialist": 0.0353
    },
    "cfp_ff": {
      "deployed_acc": 99.871,
      "specialist_acc": 99.857,
      "routed_acc": 99.871,
      "deployed_tar_1e3": 99.857,
      "specialist_tar_1e3": 99.8,
      "routed_tar_1e3": 99.857,
      "fraction_routed_to_specialist": 0.0007
    },
    "calfw": {
      "deployed_acc": 95.95,
      "specialist_acc": 95.883,
      "routed_acc": 96.017,
      "deployed_tar_1e3": 92.1,
      "specialist_tar_1e3": 90.533,
      "routed_tar_1e3": 92.1,
      "fraction_routed_to_specialist": 0.0085
    },
    "cplfw": {
      "deployed_acc": 94.467,
      "specialist_acc": 93.333,
      "routed_acc": 94.233,
      "deployed_tar_1e3": 87.4,
      "specialist_tar_1e3": 81.733,
      "routed_tar_1e3": 87.233,
      "fraction_routed_to_specialist": 0.1542
    },
    "tinyface": {
      "deployed_acc": 82.45,
      "specialist_acc": 82.383,
      "routed_acc": 82.45,
      "deployed_tar_1e3": 33.133,
      "specialist_tar_1e3": 38.1,
      "routed_tar_1e3": 37.933,
      "fraction_routed_to_specialist": 0.9878
    }
  }
}
```

## Measurement — `runtime/benchmarks/routed_engine_validated.json`

### Values

| Field | Value |
|---|---|
| `cross_model_same_image_median` | 0.8558 |
| `embedding_spaces_compatible` | True |
| `quality_by_dataset.lfw` | 0.7424 |
| `quality_by_dataset.agedb_30` | 0.7957 |
| `quality_by_dataset.cfp_fp` | 0.7753 |
| `quality_by_dataset.cfp_ff` | 0.7859 |
| `quality_by_dataset.calfw` | 0.7863 |
| `quality_by_dataset.cplfw` | 0.7551 |
| `quality_by_dataset.tinyface` | 0.5023 |
| `quality_separation` | 0.2783 |
| `verdict` | ADOPT |
| `routing_threshold` | 0.539 |
| `results.lfw.deployed_acc` | 99.783 |
| `results.lfw.specialist_acc` | 99.717 |
| `results.lfw.routed_acc` | 99.783 |
| `results.lfw.deployed_tar_1e3` | 99.7 |
| `results.lfw.specialist_tar_1e3` | 99.667 |
| `results.lfw.routed_tar_1e3` | 99.7 |
| `results.lfw.fraction_routed_to_specialist` | 0.0275 |
| `results.agedb_30.deployed_acc` | 98.15 |
| `results.agedb_30.specialist_acc` | 97.783 |
| `results.agedb_30.routed_acc` | 98.117 |
| `results.agedb_30.deployed_tar_1e3` | 96.033 |
| `results.agedb_30.specialist_tar_1e3` | 88.1 |
| `results.agedb_30.routed_tar_1e3` | 95.967 |
| `results.agedb_30.fraction_routed_to_specialist` | 0.0138 |
| `results.cfp_fp.deployed_acc` | 97.443 |
| `results.cfp_fp.specialist_acc` | 97.171 |
| `results.cfp_fp.routed_acc` | 97.429 |
| `results.cfp_fp.deployed_tar_1e3` | 94.686 |
| `results.cfp_fp.specialist_tar_1e3` | 92.829 |
| `results.cfp_fp.routed_tar_1e3` | 94.657 |
| `results.cfp_fp.fraction_routed_to_specialist` | 0.0331 |
| `results.cfp_ff.deployed_acc` | 99.871 |
| `results.cfp_ff.specialist_acc` | 99.857 |
| `results.cfp_ff.routed_acc` | 99.871 |
| `results.cfp_ff.deployed_tar_1e3` | 99.857 |
| `results.cfp_ff.specialist_tar_1e3` | 99.8 |
| `results.cfp_ff.routed_tar_1e3` | 99.857 |
| `results.cfp_ff.fraction_routed_to_specialist` | 0.0007 |
| `results.calfw.deployed_acc` | 95.95 |
| `results.calfw.specialist_acc` | 95.883 |
| `results.calfw.routed_acc` | 96.0 |
| `results.calfw.deployed_tar_1e3` | 92.1 |
| `results.calfw.specialist_tar_1e3` | 90.533 |
| `results.calfw.routed_tar_1e3` | 92.1 |
| `results.calfw.fraction_routed_to_specialist` | 0.0038 |
| `results.cplfw.deployed_acc` | 94.467 |
| `results.cplfw.specialist_acc` | 93.333 |
| `results.cplfw.routed_acc` | 94.333 |
| `results.cplfw.deployed_tar_1e3` | 87.4 |
| `results.cplfw.specialist_tar_1e3` | 81.733 |
| `results.cplfw.routed_tar_1e3` | 87.267 |
| `results.cplfw.fraction_routed_to_specialist` | 0.0745 |
| `results.tinyface.deployed_acc` | 82.45 |
| `results.tinyface.specialist_acc` | 82.383 |
| `results.tinyface.routed_acc` | 82.533 |
| `results.tinyface.deployed_tar_1e3` | 33.133 |
| `results.tinyface.specialist_tar_1e3` | 38.1 |
| `results.tinyface.routed_tar_1e3` | 37.367 |
| `results.tinyface.fraction_routed_to_specialist` | 0.9223 |

### Raw artefact

```json
{
  "cross_model_same_image_median": 0.8558,
  "embedding_spaces_compatible": true,
  "quality_by_dataset": {
    "lfw": 0.7424,
    "agedb_30": 0.7957,
    "cfp_fp": 0.7753,
    "cfp_ff": 0.7859,
    "calfw": 0.7863,
    "cplfw": 0.7551,
    "tinyface": 0.5023
  },
  "quality_separation": 0.2783,
  "verdict": "ADOPT",
  "routing_threshold": 0.539,
  "results": {
    "lfw": {
      "deployed_acc": 99.783,
      "specialist_acc": 99.717,
      "routed_acc": 99.783,
      "deployed_tar_1e3": 99.7,
      "specialist_tar_1e3": 99.667,
      "routed_tar_1e3": 99.7,
      "fraction_routed_to_specialist": 0.0275
    },
    "agedb_30": {
      "deployed_acc": 98.15,
      "specialist_acc": 97.783,
      "routed_acc": 98.117,
      "deployed_tar_1e3": 96.033,
      "specialist_tar_1e3": 88.1,
      "routed_tar_1e3": 95.967,
      "fraction_routed_to_specialist": 0.0138
    },
    "cfp_fp": {
      "deployed_acc": 97.443,
      "specialist_acc": 97.171,
      "routed_acc": 97.429,
      "deployed_tar_1e3": 94.686,
      "specialist_tar_1e3": 92.829,
      "routed_tar_1e3": 94.657,
      "fraction_routed_to_specialist": 0.0331
    },
    "cfp_ff": {
      "deployed_acc": 99.871,
      "specialist_acc": 99.857,
      "routed_acc": 99.871,
      "deployed_tar_1e3": 99.857,
      "specialist_tar_1e3": 99.8,
      "routed_tar_1e3": 99.857,
      "fraction_routed_to_specialist": 0.0007
    },
    "calfw": {
      "deployed_acc": 95.95,
      "specialist_acc": 95.883,
      "routed_acc": 96.0,
      "deployed_tar_1e3": 92.1,
      "specialist_tar_1e3": 90.533,
      "routed_tar_1e3": 92.1,
      "fraction_routed_to_specialist": 0.0038
    },
    "cplfw": {
      "deployed_acc": 94.467,
      "specialist_acc": 93.333,
      "routed_acc": 94.333,
      "deployed_tar_1e3": 87.4,
      "specialist_tar_1e3": 81.733,
      "routed_tar_1e3": 87.267,
      "fraction_routed_to_specialist": 0.0745
    },
    "tinyface": {
      "deployed_acc": 82.45,
      "specialist_acc": 82.383,
      "routed_acc": 82.533,
      "deployed_tar_1e3": 33.133,
      "specialist_tar_1e3": 38.1,
      "routed_tar_1e3": 37.367,
      "fraction_routed_to_specialist": 0.9223
    }
  }
}
```

## Measurement — `runtime/benchmarks/routing_threshold.json`

### Values

| Field | Value |
|---|---|
| `derived_from[0]` | QMUL-SurvFace training_set |
| `derived_from[1]` | CASIA-WebFace train.rec |
| `disjoint_from_reporting_benchmarks` | True |
| `qmul_median` | 0.4677 |
| `casia_median` | 0.7547 |
| `threshold` | 0.539 |

### Raw artefact

```json
{
  "derived_from": [
    "QMUL-SurvFace training_set",
    "CASIA-WebFace train.rec"
  ],
  "disjoint_from_reporting_benchmarks": true,
  "qmul_median": 0.4677,
  "casia_median": 0.7547,
  "threshold": 0.539
}
```


# Latency, throughput and concurrency

## Protocol as implemented — `backend/scripts/benchmark_speed.py`

```text
Latency and throughput measurement for the deployed recognition path.

    python backend/scripts/benchmark_speed.py

Closes the "no performance data exists" gap. Every number here is measured on
this host, through the same `FacialRecognitionPipeline` the API serves from --
not a synthetic microbenchmark of the ONNX session in isolation.

WHAT IS MEASURED
    1. Full encode  -- decode -> detect -> align -> quality/liveness/deepfake
                       -> embed. This is what one uploaded image costs.
    2. Per-stage    -- the pipeline already records StageTimings per call, so
                       the breakdown is real instrumentation, not estimation.
    3. Verify (1:1) -- two encodes plus the comparison, i.e. one API request.
    4. Gallery search -- brute-force cosine at several gallery sizes, to show
                       where the current approach stops scaling.

Percentiles are nearest-rank (see speed_benchmark._percentile): every reported
figure is an observation that actually occurred, never an interpolation.

Latency depends on image resolution, so the source image and its dimensions
are recorded alongside the numbers. Quoting these figures without that context
would be meaningless.
```

## Protocol as implemented — `backend/scripts/benchmark_concurrency.py`

```text
Item 29 — concurrency and request batching. Closes SCORECARD limitation L7.

    python backend/scripts/benchmark_concurrency.py

Every latency figure in BENCHMARKS.md §7b is single-threaded. That describes one
operator on an idle machine, not a service under load, so "65 images/second"
has until now been an unwarranted extrapolation. This measures what actually
happens when requests overlap.

TWO DISTINCT QUESTIONS, MEASURED SEPARATELY
-------------------------------------------
1. THREAD CONCURRENCY. FastAPI serves on a thread pool, so N requests hit the
   ONNX session at once. onnxruntime's InferenceSession.run is thread-safe, but
   thread-safe is not the same as parallel: internal locking and a shared intra-
   op thread pool mean throughput may not scale with workers, and p99 latency
   can degrade badly while mean throughput looks fine.

2. REQUEST BATCHING. Feeding N images to the model as ONE batched call instead
   of N separate calls. This is where a real gain is expected, because the
   per-call overhead (blob construction, session dispatch, memory transfer) is
   paid once rather than N times.

The distinction matters for what to build: if batching wins and threading does
not, the fix is a request-collecting queue in front of the model, not more
uvicorn workers.

HONEST SCOPE
------------
This measures the ENGINE under concurrent load, not the full HTTP stack. Real
end-to-end throughput is additionally bounded by request parsing, base64
decoding and database writes. Treat these as an upper bound on what the
recognition path can sustain, not a service-level SLO.
```

## Measurement — `runtime/benchmarks/concurrency.json`

### Values

| Field | Value |
|---|---|
| `device` | cuda |
| `encodes_per_config` | 100 |
| `scope` | engine under load, NOT the full HTTP stack; an upper bound |
| `thread_concurrency[0].workers` | 1 |
| `thread_concurrency[0].encodes` | 100 |
| `thread_concurrency[0].wall_s` | 1.41 |
| `thread_concurrency[0].throughput_per_s` | 70.91 |
| `thread_concurrency[0].mean_ms` | 14.09 |
| `thread_concurrency[0].p50_ms` | 13.79 |
| `thread_concurrency[0].p95_ms` | 16.16 |
| `thread_concurrency[0].p99_ms` | 16.65 |
| `thread_concurrency[1].workers` | 2 |
| `thread_concurrency[1].encodes` | 100 |
| `thread_concurrency[1].wall_s` | 0.887 |
| `thread_concurrency[1].throughput_per_s` | 112.71 |
| `thread_concurrency[1].mean_ms` | 17.68 |
| `thread_concurrency[1].p50_ms` | 17.21 |
| `thread_concurrency[1].p95_ms` | 21.56 |
| `thread_concurrency[1].p99_ms` | 22.63 |
| `thread_concurrency[2].workers` | 4 |
| `thread_concurrency[2].encodes` | 100 |
| `thread_concurrency[2].wall_s` | 0.758 |
| `thread_concurrency[2].throughput_per_s` | 131.93 |
| `thread_concurrency[2].mean_ms` | 29.94 |
| `thread_concurrency[2].p50_ms` | 29.52 |
| `thread_concurrency[2].p95_ms` | 37.57 |
| `thread_concurrency[2].p99_ms` | 43.45 |
| `thread_concurrency[3].workers` | 8 |
| `thread_concurrency[3].encodes` | 100 |
| `thread_concurrency[3].wall_s` | 0.768 |
| `thread_concurrency[3].throughput_per_s` | 130.28 |
| `thread_concurrency[3].mean_ms` | 59.82 |
| `thread_concurrency[3].p50_ms` | 58.0 |
| `thread_concurrency[3].p95_ms` | 86.79 |
| `thread_concurrency[3].p99_ms` | 99.86 |
| `request_batching[0].batch_size` | 1 |
| `request_batching[0].encodes` | 100 |
| `request_batching[0].wall_s` | 0.512 |
| `request_batching[0].throughput_per_s` | 195.38 |
| `request_batching[0].ms_per_image` | 5.118 |
| `request_batching[0].batch_call_p50_ms` | 4.28 |
| `request_batching[1].batch_size` | 4 |
| `request_batching[1].encodes` | 100 |
| `request_batching[1].wall_s` | 0.25 |
| `request_batching[1].throughput_per_s` | 399.73 |
| `request_batching[1].ms_per_image` | 2.502 |
| `request_batching[1].batch_call_p50_ms` | 7.87 |
| `request_batching[2].batch_size` | 16 |
| `request_batching[2].encodes` | 100 |
| `request_batching[2].wall_s` | 0.256 |
| `request_batching[2].throughput_per_s` | 389.99 |
| `request_batching[2].ms_per_image` | 2.564 |
| `request_batching[2].batch_call_p50_ms` | 28.02 |
| `request_batching[3].batch_size` | 32 |
| `request_batching[3].encodes` | 100 |
| `request_batching[3].wall_s` | 0.212 |
| `request_batching[3].throughput_per_s` | 472.25 |
| `request_batching[3].ms_per_image` | 2.118 |
| `request_batching[3].batch_call_p50_ms` | 48.4 |
| `request_batching[4].batch_size` | 64 |
| `request_batching[4].encodes` | 100 |
| `request_batching[4].wall_s` | 0.181 |
| `request_batching[4].throughput_per_s` | 551.17 |
| `request_batching[4].ms_per_image` | 1.814 |
| `request_batching[4].batch_call_p50_ms` | 60.46 |

### Raw artefact

```json
{
  "device": "cuda",
  "encodes_per_config": 100,
  "scope": "engine under load, NOT the full HTTP stack; an upper bound",
  "thread_concurrency": [
    {
      "workers": 1,
      "encodes": 100,
      "wall_s": 1.41,
      "throughput_per_s": 70.91,
      "mean_ms": 14.09,
      "p50_ms": 13.79,
      "p95_ms": 16.16,
      "p99_ms": 16.65
    },
    {
      "workers": 2,
      "encodes": 100,
      "wall_s": 0.887,
      "throughput_per_s": 112.71,
      "mean_ms": 17.68,
      "p50_ms": 17.21,
      "p95_ms": 21.56,
      "p99_ms": 22.63
    },
    {
      "workers": 4,
      "encodes": 100,
      "wall_s": 0.758,
      "throughput_per_s": 131.93,
      "mean_ms": 29.94,
      "p50_ms": 29.52,
      "p95_ms": 37.57,
      "p99_ms": 43.45
    },
    {
      "workers": 8,
      "encodes": 100,
      "wall_s": 0.768,
      "throughput_per_s": 130.28,
      "mean_ms": 59.82,
      "p50_ms": 58.0,
      "p95_ms": 86.79,
      "p99_ms": 99.86
    }
  ],
  "request_batching": [
    {
      "batch_size": 1,
      "encodes": 100,
      "wall_s": 0.512,
      "throughput_per_s": 195.38,
      "ms_per_image": 5.118,
      "batch_call_p50_ms": 4.28
    },
    {
      "batch_size": 4,
      "encodes": 100,
      "wall_s": 0.25,
      "throughput_per_s": 399.73,
      "ms_per_image": 2.502,
      "batch_call_p50_ms": 7.87
    },
    {
      "batch_size": 16,
      "encodes": 100,
      "wall_s": 0.256,
      "throughput_per_s": 389.99,
      "ms_per_image": 2.564,
      "batch_call_p50_ms": 28.02
    },
    {
      "batch_size": 32,
      "encodes": 100,
      "wall_s": 0.212,
      "throughput_per_s": 472.25,
      "ms_per_image": 2.118,
      "batch_call_p50_ms": 48.4
    },
    {
      "batch_size": 64,
      "encodes": 100,
      "wall_s": 0.181,
      "throughput_per_s": 551.17,
      "ms_per_image": 1.814,
      "batch_call_p50_ms": 60.46
    }
  ]
}
```

## Measurement — `runtime/benchmarks/speed.json`

### Values

| Field | Value |
|---|---|
| `host.platform` | Windows-10-10.0.26200-SP0 |
| `host.python` | 3.11.15 |
| `host.torch` | 2.5.1+cu121 |
| `host.gpu` | NVIDIA RTX A3000 Laptop GPU |
| `host.cuda` | 12.1 |
| `host.onnxruntime` | 1.20.1 |
| `host.model_pack` | buffalo_l |
| `host.recognition_network` | w600k_r50 |
| `host.providers[0]` | CUDAExecutionProvider |
| `host.providers[1]` | CPUExecutionProvider |
| `host.device` | cuda |
| `image_resolution` | 112x112 |
| `n_images` | 8 |
| `encode.iterations` | 50 |
| `encode.mean_ms` | 15.203983999672346 |
| `encode.stdev_ms` | 1.4351457315924394 |
| `encode.min_ms` | 13.299299993377645 |
| `encode.p50_ms` | 14.720999999553896 |
| `encode.p95_ms` | 17.496799999207724 |
| `encode.p99_ms` | 18.97570000437554 |
| `encode.max_ms` | 18.97570000437554 |
| `encode.throughput_per_s` | 65.77223443681278 |
| `stages.total_ms.mean_ms` | 14.6118 |
| `stages.total_ms.p50_ms` | 14.08 |
| `stages.embed_ms.mean_ms` | 5.999 |
| `stages.embed_ms.p50_ms` | 5.795 |
| `stages.detect_ms.mean_ms` | 5.7346 |
| `stages.detect_ms.p50_ms` | 5.155 |
| `stages.align_ms.mean_ms` | 0.9044 |
| `stages.align_ms.p50_ms` | 0.88 |
| `stages.quality_ms.mean_ms` | 0.3556 |
| `stages.quality_ms.p50_ms` | 0.34 |
| `stages.decode_ms.mean_ms` | 0.20679999999999998 |
| `stages.decode_ms.p50_ms` | 0.2 |
| `verify_1to1.iterations` | 25 |
| `verify_1to1.mean_ms` | 30.2062919997843 |
| `verify_1to1.stdev_ms` | 1.8323077694388514 |
| `verify_1to1.min_ms` | 26.91959999356186 |
| `verify_1to1.p50_ms` | 31.038100001751445 |
| `verify_1to1.p95_ms` | 32.60020000016084 |
| `verify_1to1.p99_ms` | 32.70460000203457 |
| `verify_1to1.max_ms` | 32.70460000203457 |
| `verify_1to1.throughput_per_s` | 33.10568539849714 |
| `gallery_search.100.iterations` | 20 |
| `gallery_search.100.mean_ms` | 0.22645000026386697 |
| `gallery_search.100.stdev_ms` | 0.08100534387413726 |
| `gallery_search.100.min_ms` | 0.1320000010309741 |
| `gallery_search.100.p50_ms` | 0.19830000383080915 |
| `gallery_search.100.p95_ms` | 0.3804000007221475 |
| `gallery_search.100.p99_ms` | 0.41819999751169235 |
| `gallery_search.100.max_ms` | 0.41819999751169235 |
| `gallery_search.100.throughput_per_s` | 4415.985863699568 |
| `gallery_search.1000.iterations` | 20 |
| `gallery_search.1000.mean_ms` | 0.20051000064995605 |
| `gallery_search.1000.stdev_ms` | 0.03734169411720926 |
| `gallery_search.1000.min_ms` | 0.1370999962091446 |
| `gallery_search.1000.p50_ms` | 0.20700000459328294 |
| `gallery_search.1000.p95_ms` | 0.24790000315988436 |
| `gallery_search.1000.p99_ms` | 0.25680000544525683 |
| `gallery_search.1000.max_ms` | 0.25680000544525683 |
| `gallery_search.1000.throughput_per_s` | 4987.282413637652 |
| `gallery_search.10000.iterations` | 20 |
| `gallery_search.10000.mean_ms` | 1.1235750003834255 |
| `gallery_search.10000.stdev_ms` | 0.1438720996235428 |
| `gallery_search.10000.min_ms` | 0.985200000286568 |
| `gallery_search.10000.p50_ms` | 1.087000004190486 |
| `gallery_search.10000.p95_ms` | 1.3651000044774264 |
| `gallery_search.10000.p99_ms` | 1.5635000017937273 |
| `gallery_search.10000.max_ms` | 1.5635000017937273 |
| `gallery_search.10000.throughput_per_s` | 890.0162424927086 |
| `gallery_search.100000.iterations` | 20 |
| `gallery_search.100000.mean_ms` | 16.367830000672257 |
| `gallery_search.100000.stdev_ms` | 1.0867635639863558 |
| `gallery_search.100000.min_ms` | 14.94519999687327 |
| `gallery_search.100000.p50_ms` | 15.98099999682745 |
| `gallery_search.100000.p95_ms` | 18.147200004023034 |
| `gallery_search.100000.p99_ms` | 18.861399999877904 |
| `gallery_search.100000.max_ms` | 18.861399999877904 |
| `gallery_search.100000.throughput_per_s` | 61.09545370149422 |

### Raw artefact

```json
{
  "host": {
    "platform": "Windows-10-10.0.26200-SP0",
    "python": "3.11.15",
    "torch": "2.5.1+cu121",
    "gpu": "NVIDIA RTX A3000 Laptop GPU",
    "cuda": "12.1",
    "onnxruntime": "1.20.1",
    "model_pack": "buffalo_l",
    "recognition_network": "w600k_r50",
    "providers": [
      "CUDAExecutionProvider",
      "CPUExecutionProvider"
    ],
    "device": "cuda"
  },
  "image_resolution": "112x112",
  "n_images": 8,
  "encode": {
    "iterations": 50,
    "mean_ms": 15.203983999672346,
    "stdev_ms": 1.4351457315924394,
    "min_ms": 13.299299993377645,
    "p50_ms": 14.720999999553896,
    "p95_ms": 17.496799999207724,
    "p99_ms": 18.97570000437554,
    "max_ms": 18.97570000437554,
    "throughput_per_s": 65.77223443681278
  },
  "stages": {
    "total_ms": {
      "mean_ms": 14.6118,
      "p50_ms": 14.08
    },
    "embed_ms": {
      "mean_ms": 5.999,
      "p50_ms": 5.795
    },
    "detect_ms": {
      "mean_ms": 5.7346,
      "p50_ms": 5.155
    },
    "align_ms": {
      "mean_ms": 0.9044,
      "p50_ms": 0.88
    },
    "quality_ms": {
      "mean_ms": 0.3556,
      "p50_ms": 0.34
    },
    "decode_ms": {
      "mean_ms": 0.20679999999999998,
      "p50_ms": 0.2
    }
  },
  "verify_1to1": {
    "iterations": 25,
    "mean_ms": 30.2062919997843,
    "stdev_ms": 1.8323077694388514,
    "min_ms": 26.91959999356186,
    "p50_ms": 31.038100001751445,
    "p95_ms": 32.60020000016084,
    "p99_ms": 32.70460000203457,
    "max_ms": 32.70460000203457,
    "throughput_per_s": 33.10568539849714
  },
  "gallery_search": {
    "100": {
      "iterations": 20,
      "mean_ms": 0.22645000026386697,
      "stdev_ms": 0.08100534387413726,
      "min_ms": 0.1320000010309741,
      "p50_ms": 0.19830000383080915,
      "p95_ms": 0.3804000007221475,
      "p99_ms": 0.41819999751169235,
      "max_ms": 0.41819999751169235,
      "throughput_per_s": 4415.985863699568
    },
    "1000": {
      "iterations": 20,
      "mean_ms": 0.20051000064995605,
      "stdev_ms": 0.03734169411720926,
      "min_ms": 0.1370999962091446,
      "p50_ms": 0.20700000459328294,
      "p95_ms": 0.24790000315988436,
      "p99_ms": 0.25680000544525683,
      "max_ms": 0.25680000544525683,
      "throughput_per_s": 4987.282413637652
    },
    "10000": {
      "iterations": 20,
      "mean_ms": 1.1235750003834255,
      "stdev_ms": 0.1438720996235428,
      "min_ms": 0.985200000286568,
      "p50_ms": 1.087000004190486,
      "p95_ms": 1.3651000044774264,
      "p99_ms": 1.5635000017937273,
      "max_ms": 1.5635000017937273,
      "throughput_per_s": 890.0162424927086
    },
    "100000": {
      "iterations": 20,
      "mean_ms": 16.367830000672257,
      "stdev_ms": 1.0867635639863558,
      "min_ms": 14.94519999687327,
      "p50_ms": 15.98099999682745,
      "p95_ms": 18.147200004023034,
      "p99_ms": 18.861399999877904,
      "max_ms": 18.861399999877904,
      "throughput_per_s": 61.09545370149422
    }
  }
}
```


# Approximate nearest-neighbour search

## Protocol as implemented — `backend/scripts/benchmark_ann.py`

```text
Item 28 — approximate nearest-neighbour search: latency AND recall.

    python backend/scripts/benchmark_ann.py

RE-SCOPED FROM "ADD FAISS". READ THIS BEFORE CHANGING THE SEARCH PATH.
----------------------------------------------------------------------
The original plan was "FAISS-backed search". That would not have achieved its
goal. The branch already in gallery_index.py builds `faiss.IndexFlatIP`, an
EXACT inner-product index -- brute force with better SIMD. It buys a constant
factor, not a change in complexity, so 100k templates would still scale
linearly.

Real scaling needs an APPROXIMATE index. Approximation means missed candidates,
and in a forensic system a missed candidate is an investigative lead that
silently never surfaced. So this benchmark reports RECALL alongside latency,
and recall is the number that decides adoption -- not speed.

WHAT IS MEASURED
    exact      numpy matmul, the current production path (ground truth)
    flat       faiss IndexFlatIP, exact, for the constant-factor comparison
    ivfpq      IVF-PQ, approximate, compressed
    hnsw       HNSW, approximate, graph-based

Recall@1  -- fraction of probes whose TOP-1 matches exact search's top-1.
             This is the one that matters: a wrong rank-1 is a wrong lead.
Recall@10 -- fraction of exact top-10 present in the approximate top-10.
             Matters when an examiner reviews a candidate list rather than
             one name.

Vectors are unit-norm 512-d, matching real ArcFace templates, so inner product
is cosine. Synthetic vectors are used deliberately: real galleries at 100k do
not exist here, and recall behaviour is a property of the index geometry, not
of whose faces are in it. That is a limitation, and it is stated in the output.
```

## Measurement — `runtime/benchmarks/ann_search.json`

### Values

| Field | Value |
|---|---|
| `faiss_version` | 1.14.3 |
| `dim` | 512 |
| `queries` | 200 |
| `top_k` | 10 |
| `limitation` | Synthetic unit vectors. Recall is a property of index geometry, but real ArcFace galleries cluster by identity and may recall differently. Re-measure on real templates before adopting an approximate index in production. |
| `results.1000.exact_numpy.p50_ms` | 0.1142 |
| `results.1000.exact_numpy.p95_ms` | 0.2436 |
| `results.1000.exact_numpy.mean_ms` | 0.1415 |
| `results.1000.exact_numpy.qps` | 7067.6 |
| `results.1000.exact_numpy.recall@1` | 1.0 |
| `results.1000.exact_numpy.recall@10` | 1.0 |
| `results.1000.faiss_flat_exact.p50_ms` | 0.0636 |
| `results.1000.faiss_flat_exact.p95_ms` | 0.0693 |
| `results.1000.faiss_flat_exact.mean_ms` | 0.0661 |
| `results.1000.faiss_flat_exact.qps` | 15137.0 |
| `results.1000.faiss_flat_exact.recall@1` | 1.0 |
| `results.1000.faiss_flat_exact.recall@10` | 1.0 |
| `results.1000.hnsw_ef16.p50_ms` | 0.0447 |
| `results.1000.hnsw_ef16.p95_ms` | 0.0463 |
| `results.1000.hnsw_ef16.mean_ms` | 0.0454 |
| `results.1000.hnsw_ef16.qps` | 22013.5 |
| `results.1000.hnsw_ef16.recall@1` | 0.79 |
| `results.1000.hnsw_ef16.recall@10` | 0.713 |
| `results.1000.hnsw_ef16.bytes_per_vector` | 2048 |
| `results.1000.hnsw_ef64.p50_ms` | 0.0891 |
| `results.1000.hnsw_ef64.p95_ms` | 0.1034 |
| `results.1000.hnsw_ef64.mean_ms` | 0.0917 |
| `results.1000.hnsw_ef64.qps` | 10908.7 |
| `results.1000.hnsw_ef64.recall@1` | 0.98 |
| `results.1000.hnsw_ef64.recall@10` | 0.98 |
| `results.1000.hnsw_ef64.bytes_per_vector` | 2048 |
| `results.1000.hnsw_ef256.p50_ms` | 0.2244 |
| `results.1000.hnsw_ef256.p95_ms` | 0.2368 |
| `results.1000.hnsw_ef256.mean_ms` | 0.2255 |
| `results.1000.hnsw_ef256.qps` | 4435.5 |
| `results.1000.hnsw_ef256.recall@1` | 1.0 |
| `results.1000.hnsw_ef256.recall@10` | 1.0 |
| `results.1000.hnsw_ef256.bytes_per_vector` | 2048 |
| `results.10000.exact_numpy.p50_ms` | 0.7229 |
| `results.10000.exact_numpy.p95_ms` | 0.8937 |
| `results.10000.exact_numpy.mean_ms` | 0.7508 |
| `results.10000.exact_numpy.qps` | 1332.0 |
| `results.10000.exact_numpy.recall@1` | 1.0 |
| `results.10000.exact_numpy.recall@10` | 1.0 |
| `results.10000.faiss_flat_exact.p50_ms` | 0.8264 |
| `results.10000.faiss_flat_exact.p95_ms` | 1.3372 |
| `results.10000.faiss_flat_exact.mean_ms` | 0.9187 |
| `results.10000.faiss_flat_exact.qps` | 1088.5 |
| `results.10000.faiss_flat_exact.recall@1` | 1.0 |
| `results.10000.faiss_flat_exact.recall@10` | 1.0 |
| `results.10000.ivfpq_nprobe1.p50_ms` | 0.095 |
| `results.10000.ivfpq_nprobe1.p95_ms` | 0.1078 |
| `results.10000.ivfpq_nprobe1.mean_ms` | 0.0992 |
| `results.10000.ivfpq_nprobe1.qps` | 10076.6 |
| `results.10000.ivfpq_nprobe1.recall@1` | 0.06 |
| `results.10000.ivfpq_nprobe1.recall@10` | 0.034 |
| `results.10000.ivfpq_nprobe1.bytes_per_vector` | 64 |
| `results.10000.ivfpq_nprobe8.p50_ms` | 0.6976 |
| `results.10000.ivfpq_nprobe8.p95_ms` | 0.7483 |
| `results.10000.ivfpq_nprobe8.mean_ms` | 0.7042 |
| `results.10000.ivfpq_nprobe8.qps` | 1420.0 |
| `results.10000.ivfpq_nprobe8.recall@1` | 0.13 |
| `results.10000.ivfpq_nprobe8.recall@10` | 0.119 |
| `results.10000.ivfpq_nprobe8.bytes_per_vector` | 64 |
| `results.10000.ivfpq_nprobe32.p50_ms` | 2.6783 |
| `results.10000.ivfpq_nprobe32.p95_ms` | 2.7493 |
| `results.10000.ivfpq_nprobe32.mean_ms` | 2.6791 |
| `results.10000.ivfpq_nprobe32.qps` | 373.3 |
| `results.10000.ivfpq_nprobe32.recall@1` | 0.175 |
| `results.10000.ivfpq_nprobe32.recall@10` | 0.2075 |
| `results.10000.ivfpq_nprobe32.bytes_per_vector` | 64 |
| `results.10000.hnsw_ef16.p50_ms` | 0.109 |
| `results.10000.hnsw_ef16.p95_ms` | 0.122 |
| `results.10000.hnsw_ef16.mean_ms` | 0.1109 |
| `results.10000.hnsw_ef16.qps` | 9016.6 |
| `results.10000.hnsw_ef16.recall@1` | 0.29 |
| `results.10000.hnsw_ef16.recall@10` | 0.266 |
| `results.10000.hnsw_ef16.bytes_per_vector` | 2048 |
| `results.10000.hnsw_ef64.p50_ms` | 0.2421 |
| `results.10000.hnsw_ef64.p95_ms` | 0.2591 |
| `results.10000.hnsw_ef64.mean_ms` | 0.245 |
| `results.10000.hnsw_ef64.qps` | 4081.9 |
| `results.10000.hnsw_ef64.recall@1` | 0.64 |
| `results.10000.hnsw_ef64.recall@10` | 0.6095 |
| `results.10000.hnsw_ef64.bytes_per_vector` | 2048 |
| `results.10000.hnsw_ef256.p50_ms` | 0.8927 |
| `results.10000.hnsw_ef256.p95_ms` | 1.329 |
| `results.10000.hnsw_ef256.mean_ms` | 0.9589 |
| `results.10000.hnsw_ef256.qps` | 1042.9 |
| `results.10000.hnsw_ef256.recall@1` | 0.96 |
| `results.10000.hnsw_ef256.recall@10` | 0.9525 |
| `results.10000.hnsw_ef256.bytes_per_vector` | 2048 |
| `results.100000.exact_numpy.p50_ms` | 10.8398 |
| `results.100000.exact_numpy.p95_ms` | 11.46 |
| `results.100000.exact_numpy.mean_ms` | 10.9707 |
| `results.100000.exact_numpy.qps` | 91.2 |
| `results.100000.exact_numpy.recall@1` | 1.0 |
| `results.100000.exact_numpy.recall@10` | 1.0 |
| `results.100000.faiss_flat_exact.p50_ms` | 4.9287 |
| `results.100000.faiss_flat_exact.p95_ms` | 5.552 |
| `results.100000.faiss_flat_exact.mean_ms` | 4.8795 |
| `results.100000.faiss_flat_exact.qps` | 204.9 |
| `results.100000.faiss_flat_exact.recall@1` | 1.0 |
| `results.100000.faiss_flat_exact.recall@10` | 1.0 |
| `results.100000.ivfpq_nprobe1.p50_ms` | 0.1459 |
| `results.100000.ivfpq_nprobe1.p95_ms` | 0.1619 |
| `results.100000.ivfpq_nprobe1.mean_ms` | 0.1487 |
| `results.100000.ivfpq_nprobe1.qps` | 6726.6 |
| `results.100000.ivfpq_nprobe1.recall@1` | 0.005 |
| `results.100000.ivfpq_nprobe1.recall@10` | 0.0105 |
| `results.100000.ivfpq_nprobe1.bytes_per_vector` | 64 |
| `results.100000.ivfpq_nprobe8.p50_ms` | 0.8337 |
| `results.100000.ivfpq_nprobe8.p95_ms` | 0.8679 |
| `results.100000.ivfpq_nprobe8.mean_ms` | 0.8359 |
| `results.100000.ivfpq_nprobe8.qps` | 1196.3 |
| `results.100000.ivfpq_nprobe8.recall@1` | 0.04 |
| `results.100000.ivfpq_nprobe8.recall@10` | 0.0495 |
| `results.100000.ivfpq_nprobe8.bytes_per_vector` | 64 |
| `results.100000.ivfpq_nprobe32.p50_ms` | 3.0677 |
| `results.100000.ivfpq_nprobe32.p95_ms` | 3.256 |
| `results.100000.ivfpq_nprobe32.mean_ms` | 2.9586 |
| `results.100000.ivfpq_nprobe32.qps` | 338.0 |
| `results.100000.ivfpq_nprobe32.recall@1` | 0.065 |
| `results.100000.ivfpq_nprobe32.recall@10` | 0.108 |
| `results.100000.ivfpq_nprobe32.bytes_per_vector` | 64 |
| `results.100000.hnsw_ef16.p50_ms` | 0.1104 |
| `results.100000.hnsw_ef16.p95_ms` | 0.1144 |
| `results.100000.hnsw_ef16.mean_ms` | 0.1113 |
| `results.100000.hnsw_ef16.qps` | 8982.6 |
| `results.100000.hnsw_ef16.recall@1` | 0.045 |
| `results.100000.hnsw_ef16.recall@10` | 0.0485 |
| `results.100000.hnsw_ef16.bytes_per_vector` | 2048 |
| `results.100000.hnsw_ef64.p50_ms` | 0.3668 |
| `results.100000.hnsw_ef64.p95_ms` | 0.3759 |
| `results.100000.hnsw_ef64.mean_ms` | 0.3684 |
| `results.100000.hnsw_ef64.qps` | 2714.7 |
| `results.100000.hnsw_ef64.recall@1` | 0.145 |
| `results.100000.hnsw_ef64.recall@10` | 0.142 |
| `results.100000.hnsw_ef64.bytes_per_vector` | 2048 |
| `results.100000.hnsw_ef256.p50_ms` | 2.2097 |
| `results.100000.hnsw_ef256.p95_ms` | 2.5421 |
| `results.100000.hnsw_ef256.mean_ms` | 2.2428 |
| `results.100000.hnsw_ef256.qps` | 445.9 |
| `results.100000.hnsw_ef256.recall@1` | 0.465 |
| `results.100000.hnsw_ef256.recall@10` | 0.4075 |
| `results.100000.hnsw_ef256.bytes_per_vector` | 2048 |

### Raw artefact

```json
{
  "faiss_version": "1.14.3",
  "dim": 512,
  "queries": 200,
  "top_k": 10,
  "limitation": "Synthetic unit vectors. Recall is a property of index geometry, but real ArcFace galleries cluster by identity and may recall differently. Re-measure on real templates before adopting an approximate index in production.",
  "results": {
    "1000": {
      "exact_numpy": {
        "p50_ms": 0.1142,
        "p95_ms": 0.2436,
        "mean_ms": 0.1415,
        "qps": 7067.6,
        "recall@1": 1.0,
        "recall@10": 1.0
      },
      "faiss_flat_exact": {
        "p50_ms": 0.0636,
        "p95_ms": 0.0693,
        "mean_ms": 0.0661,
        "qps": 15137.0,
        "recall@1": 1.0,
        "recall@10": 1.0
      },
      "hnsw_ef16": {
        "p50_ms": 0.0447,
        "p95_ms": 0.0463,
        "mean_ms": 0.0454,
        "qps": 22013.5,
        "recall@1": 0.79,
        "recall@10": 0.713,
        "bytes_per_vector": 2048
      },
      "hnsw_ef64": {
        "p50_ms": 0.0891,
        "p95_ms": 0.1034,
        "mean_ms": 0.0917,
        "qps": 10908.7,
        "recall@1": 0.98,
        "recall@10": 0.98,
        "bytes_per_vector": 2048
      },
      "hnsw_ef256": {
        "p50_ms": 0.2244,
        "p95_ms": 0.2368,
        "mean_ms": 0.2255,
        "qps": 4435.5,
        "recall@1": 1.0,
        "recall@10": 1.0,
        "bytes_per_vector": 2048
      }
    },
    "10000": {
      "exact_numpy": {
        "p50_ms": 0.7229,
        "p95_ms": 0.8937,
        "mean_ms": 0.7508,
        "qps": 1332.0,
        "recall@1": 1.0,
        "recall@10": 1.0
      },
      "faiss_flat_exact": {
        "p50_ms": 0.8264,
        "p95_ms": 1.3372,
        "mean_ms": 0.9187,
        "qps": 1088.5,
        "recall@1": 1.0,
        "recall@10": 1.0
      },
      "ivfpq_nprobe1": {
        "p50_ms": 0.095,
        "p95_ms": 0.1078,
        "mean_ms": 0.0992,
        "qps": 10076.6,
        "recall@1": 0.06,
        "recall@10": 0.034,
        "bytes_per_vector": 64
      },
      "ivfpq_nprobe8": {
        "p50_ms": 0.6976,
        "p95_ms": 0.7483,
        "mean_ms": 0.7042,
        "qps": 1420.0,
        "recall@1": 0.13,
        "recall@10": 0.119,
        "bytes_per_vector": 64
      },
      "ivfpq_nprobe32": {
        "p50_ms": 2.6783,
        "p95_ms": 2.7493,
        "mean_ms": 2.6791,
        "qps": 373.3,
        "recall@1": 0.175,
        "recall@10": 0.2075,
        "bytes_per_vector": 64
      },
      "hnsw_ef16": {
        "p50_ms": 0.109,
        "p95_ms": 0.122,
        "mean_ms": 0.1109,
        "qps": 9016.6,
        "recall@1": 0.29,
        "recall@10": 0.266,
        "bytes_per_vector": 2048
      },
      "hnsw_ef64": {
        "p50_ms": 0.2421,
        "p95_ms": 0.2591,
        "mean_ms": 0.245,
        "qps": 4081.9,
        "recall@1": 0.64,
        "recall@10": 0.6095,
        "bytes_per_vector": 2048
      },
      "hnsw_ef256": {
        "p50_ms": 0.8927,
        "p95_ms": 1.329,
        "mean_ms": 0.9589,
        "qps": 1042.9,
        "recall@1": 0.96,
        "recall@10": 0.9525,
        "bytes_per_vector": 2048
      }
    },
    "100000": {
      "exact_numpy": {
        "p50_ms": 10.8398,
        "p95_ms": 11.46,
        "mean_ms": 10.9707,
        "qps": 91.2,
        "recall@1": 1.0,
        "recall@10": 1.0
      },
      "faiss_flat_exact": {
        "p50_ms": 4.9287,
        "p95_ms": 5.552,
        "mean_ms": 4.8795,
        "qps": 204.9,
        "recall@1": 1.0,
        "recall@10": 1.0
      },
      "ivfpq_nprobe1": {
        "p50_ms": 0.1459,
        "p95_ms": 0.1619,
        "mean_ms": 0.1487,
        "qps": 6726.6,
        "recall@1": 0.005,
        "recall@10": 0.0105,
        "bytes_per_vector": 64
      },
      "ivfpq_nprobe8": {
        "p50_ms": 0.8337,
        "p95_ms": 0.8679,
        "mean_ms": 0.8359,
        "qps": 1196.3,
        "recall@1": 0.04,
        "recall@10": 0.0495,
        "bytes_per_vector": 64
      },
      "ivfpq_nprobe32": {
        "p50_ms": 3.0677,
        "p95_ms": 3.256,
        "mean_ms": 2.9586,
        "qps": 338.0,
        "recall@1": 0.065,
        "recall@10": 0.108,
        "bytes_per_vector": 64
      },
      "hnsw_ef16": {
        "p50_ms": 0.1104,
        "p95_ms": 0.1144,
        "mean_ms": 0.1113,
        "qps": 8982.6,
        "recall@1": 0.045,
        "recall@10": 0.0485,
        "bytes_per_vector": 2048
      },
      "hnsw_ef64": {
        "p50_ms": 0.3668,
        "p95_ms": 0.3759,
        "mean_ms": 0.3684,
        "qps": 2714.7,
        "recall@1": 0.145,
        "recall@10": 0.142,
        "bytes_per_vector": 2048
      },
      "hnsw_ef256": {
        "p50_ms": 2.2097,
        "p95_ms": 2.5421,
        "mean_ms": 2.2428,
        "qps": 445.9,
        "recall@1": 0.465,
        "recall@10": 0.4075,
        "bytes_per_vector": 2048
      }
    }
  }
}
```

## Measurement — `runtime/benchmarks/ann_search_real.json`

### Values

| Field | Value |
|---|---|
| `faiss_version` | 1.14.3 |
| `dim` | 512 |
| `queries` | 200 |
| `top_k` | 10 |
| `limitation` | Synthetic unit vectors. Recall is a property of index geometry, but real ArcFace galleries cluster by identity and may recall differently. Re-measure on real templates before adopting an approximate index in production. |
| `results.10000.exact_numpy.p50_ms` | 0.7157 |
| `results.10000.exact_numpy.p95_ms` | 0.8275 |
| `results.10000.exact_numpy.mean_ms` | 0.7542 |
| `results.10000.exact_numpy.qps` | 1325.9 |
| `results.10000.exact_numpy.recall@1` | 1.0 |
| `results.10000.exact_numpy.recall@10` | 1.0 |
| `results.10000.faiss_flat_exact.p50_ms` | 0.8129 |
| `results.10000.faiss_flat_exact.p95_ms` | 1.2887 |
| `results.10000.faiss_flat_exact.mean_ms` | 0.8964 |
| `results.10000.faiss_flat_exact.qps` | 1115.6 |
| `results.10000.faiss_flat_exact.recall@1` | 1.0 |
| `results.10000.faiss_flat_exact.recall@10` | 1.0 |
| `results.10000.ivfpq_nprobe1.p50_ms` | 0.0985 |
| `results.10000.ivfpq_nprobe1.p95_ms` | 0.1027 |
| `results.10000.ivfpq_nprobe1.mean_ms` | 0.0993 |
| `results.10000.ivfpq_nprobe1.qps` | 10073.5 |
| `results.10000.ivfpq_nprobe1.recall@1` | 0.555 |
| `results.10000.ivfpq_nprobe1.recall@10` | 0.3305 |
| `results.10000.ivfpq_nprobe1.bytes_per_vector` | 64 |
| `results.10000.ivfpq_nprobe8.p50_ms` | 0.6527 |
| `results.10000.ivfpq_nprobe8.p95_ms` | 0.6964 |
| `results.10000.ivfpq_nprobe8.mean_ms` | 0.6592 |
| `results.10000.ivfpq_nprobe8.qps` | 1517.0 |
| `results.10000.ivfpq_nprobe8.recall@1` | 0.69 |
| `results.10000.ivfpq_nprobe8.recall@10` | 0.5225 |
| `results.10000.ivfpq_nprobe8.bytes_per_vector` | 64 |
| `results.10000.ivfpq_nprobe32.p50_ms` | 2.5898 |
| `results.10000.ivfpq_nprobe32.p95_ms` | 2.6735 |
| `results.10000.ivfpq_nprobe32.mean_ms` | 2.5964 |
| `results.10000.ivfpq_nprobe32.qps` | 385.2 |
| `results.10000.ivfpq_nprobe32.recall@1` | 0.745 |
| `results.10000.ivfpq_nprobe32.recall@10` | 0.616 |
| `results.10000.ivfpq_nprobe32.bytes_per_vector` | 64 |
| `results.10000.hnsw_ef16.p50_ms` | 0.0522 |
| `results.10000.hnsw_ef16.p95_ms` | 0.0555 |
| `results.10000.hnsw_ef16.mean_ms` | 0.0529 |
| `results.10000.hnsw_ef16.qps` | 18908.4 |
| `results.10000.hnsw_ef16.recall@1` | 0.84 |
| `results.10000.hnsw_ef16.recall@10` | 0.856 |
| `results.10000.hnsw_ef16.bytes_per_vector` | 2048 |
| `results.10000.hnsw_ef64.p50_ms` | 0.1801 |
| `results.10000.hnsw_ef64.p95_ms` | 0.1924 |
| `results.10000.hnsw_ef64.mean_ms` | 0.1824 |
| `results.10000.hnsw_ef64.qps` | 5482.3 |
| `results.10000.hnsw_ef64.recall@1` | 0.895 |
| `results.10000.hnsw_ef64.recall@10` | 0.961 |
| `results.10000.hnsw_ef64.bytes_per_vector` | 2048 |
| `results.10000.hnsw_ef256.p50_ms` | 0.6157 |
| `results.10000.hnsw_ef256.p95_ms` | 0.8862 |
| `results.10000.hnsw_ef256.mean_ms` | 0.6699 |
| `results.10000.hnsw_ef256.qps` | 1492.8 |
| `results.10000.hnsw_ef256.recall@1` | 0.91 |
| `results.10000.hnsw_ef256.recall@10` | 0.9895 |
| `results.10000.hnsw_ef256.bytes_per_vector` | 2048 |
| `results.50000.exact_numpy.p50_ms` | 4.9367 |
| `results.50000.exact_numpy.p95_ms` | 5.284 |
| `results.50000.exact_numpy.mean_ms` | 4.9825 |
| `results.50000.exact_numpy.qps` | 200.7 |
| `results.50000.exact_numpy.recall@1` | 1.0 |
| `results.50000.exact_numpy.recall@10` | 1.0 |
| `results.50000.faiss_flat_exact.p50_ms` | 2.0658 |
| `results.50000.faiss_flat_exact.p95_ms` | 2.7795 |
| `results.50000.faiss_flat_exact.mean_ms` | 2.147 |
| `results.50000.faiss_flat_exact.qps` | 465.8 |
| `results.50000.faiss_flat_exact.recall@1` | 1.0 |
| `results.50000.faiss_flat_exact.recall@10` | 1.0 |
| `results.50000.ivfpq_nprobe1.p50_ms` | 0.1325 |
| `results.50000.ivfpq_nprobe1.p95_ms` | 0.145 |
| `results.50000.ivfpq_nprobe1.mean_ms` | 0.1328 |
| `results.50000.ivfpq_nprobe1.qps` | 7528.6 |
| `results.50000.ivfpq_nprobe1.recall@1` | 0.56 |
| `results.50000.ivfpq_nprobe1.recall@10` | 0.6435 |
| `results.50000.ivfpq_nprobe1.bytes_per_vector` | 64 |
| `results.50000.ivfpq_nprobe8.p50_ms` | 0.7953 |
| `results.50000.ivfpq_nprobe8.p95_ms` | 0.8506 |
| `results.50000.ivfpq_nprobe8.mean_ms` | 0.8021 |
| `results.50000.ivfpq_nprobe8.qps` | 1246.7 |
| `results.50000.ivfpq_nprobe8.recall@1` | 0.575 |
| `results.50000.ivfpq_nprobe8.recall@10` | 0.762 |
| `results.50000.ivfpq_nprobe8.bytes_per_vector` | 64 |
| `results.50000.ivfpq_nprobe32.p50_ms` | 2.874 |
| `results.50000.ivfpq_nprobe32.p95_ms` | 3.0761 |
| `results.50000.ivfpq_nprobe32.mean_ms` | 2.8586 |
| `results.50000.ivfpq_nprobe32.qps` | 349.8 |
| `results.50000.ivfpq_nprobe32.recall@1` | 0.575 |
| `results.50000.ivfpq_nprobe32.recall@10` | 0.811 |
| `results.50000.ivfpq_nprobe32.bytes_per_vector` | 64 |
| `results.50000.hnsw_ef16.p50_ms` | 0.0422 |
| `results.50000.hnsw_ef16.p95_ms` | 0.046 |
| `results.50000.hnsw_ef16.mean_ms` | 0.0431 |
| `results.50000.hnsw_ef16.qps` | 23180.3 |
| `results.50000.hnsw_ef16.recall@1` | 0.61 |
| `results.50000.hnsw_ef16.recall@10` | 0.9555 |
| `results.50000.hnsw_ef16.bytes_per_vector` | 2048 |
| `results.50000.hnsw_ef64.p50_ms` | 0.1518 |
| `results.50000.hnsw_ef64.p95_ms` | 0.1791 |
| `results.50000.hnsw_ef64.mean_ms` | 0.1585 |
| `results.50000.hnsw_ef64.qps` | 6308.9 |
| `results.50000.hnsw_ef64.recall@1` | 0.615 |
| `results.50000.hnsw_ef64.recall@10` | 0.961 |
| `results.50000.hnsw_ef64.bytes_per_vector` | 2048 |
| `results.50000.hnsw_ef256.p50_ms` | 0.7238 |
| `results.50000.hnsw_ef256.p95_ms` | 0.9964 |
| `results.50000.hnsw_ef256.mean_ms` | 0.761 |
| `results.50000.hnsw_ef256.qps` | 1314.1 |
| `results.50000.hnsw_ef256.recall@1` | 0.615 |
| `results.50000.hnsw_ef256.recall@10` | 0.9675 |
| `results.50000.hnsw_ef256.bytes_per_vector` | 2048 |

### Raw artefact

```json
{
  "faiss_version": "1.14.3",
  "dim": 512,
  "queries": 200,
  "top_k": 10,
  "limitation": "Synthetic unit vectors. Recall is a property of index geometry, but real ArcFace galleries cluster by identity and may recall differently. Re-measure on real templates before adopting an approximate index in production.",
  "results": {
    "10000": {
      "exact_numpy": {
        "p50_ms": 0.7157,
        "p95_ms": 0.8275,
        "mean_ms": 0.7542,
        "qps": 1325.9,
        "recall@1": 1.0,
        "recall@10": 1.0
      },
      "faiss_flat_exact": {
        "p50_ms": 0.8129,
        "p95_ms": 1.2887,
        "mean_ms": 0.8964,
        "qps": 1115.6,
        "recall@1": 1.0,
        "recall@10": 1.0
      },
      "ivfpq_nprobe1": {
        "p50_ms": 0.0985,
        "p95_ms": 0.1027,
        "mean_ms": 0.0993,
        "qps": 10073.5,
        "recall@1": 0.555,
        "recall@10": 0.3305,
        "bytes_per_vector": 64
      },
      "ivfpq_nprobe8": {
        "p50_ms": 0.6527,
        "p95_ms": 0.6964,
        "mean_ms": 0.6592,
        "qps": 1517.0,
        "recall@1": 0.69,
        "recall@10": 0.5225,
        "bytes_per_vector": 64
      },
      "ivfpq_nprobe32": {
        "p50_ms": 2.5898,
        "p95_ms": 2.6735,
        "mean_ms": 2.5964,
        "qps": 385.2,
        "recall@1": 0.745,
        "recall@10": 0.616,
        "bytes_per_vector": 64
      },
      "hnsw_ef16": {
        "p50_ms": 0.0522,
        "p95_ms": 0.0555,
        "mean_ms": 0.0529,
        "qps": 18908.4,
        "recall@1": 0.84,
        "recall@10": 0.856,
        "bytes_per_vector": 2048
      },
      "hnsw_ef64": {
        "p50_ms": 0.1801,
        "p95_ms": 0.1924,
        "mean_ms": 0.1824,
        "qps": 5482.3,
        "recall@1": 0.895,
        "recall@10": 0.961,
        "bytes_per_vector": 2048
      },
      "hnsw_ef256": {
        "p50_ms": 0.6157,
        "p95_ms": 0.8862,
        "mean_ms": 0.6699,
        "qps": 1492.8,
        "recall@1": 0.91,
        "recall@10": 0.9895,
        "bytes_per_vector": 2048
      }
    },
    "50000": {
      "exact_numpy": {
        "p50_ms": 4.9367,
        "p95_ms": 5.284,
        "mean_ms": 4.9825,
        "qps": 200.7,
        "recall@1": 1.0,
        "recall@10": 1.0
      },
      "faiss_flat_exact": {
        "p50_ms": 2.0658,
        "p95_ms": 2.7795,
        "mean_ms": 2.147,
        "qps": 465.8,
        "recall@1": 1.0,
        "recall@10": 1.0
      },
      "ivfpq_nprobe1": {
        "p50_ms": 0.1325,
        "p95_ms": 0.145,
        "mean_ms": 0.1328,
        "qps": 7528.6,
        "recall@1": 0.56,
        "recall@10": 0.6435,
        "bytes_per_vector": 64
      },
      "ivfpq_nprobe8": {
        "p50_ms": 0.7953,
        "p95_ms": 0.8506,
        "mean_ms": 0.8021,
        "qps": 1246.7,
        "recall@1": 0.575,
        "recall@10": 0.762,
        "bytes_per_vector": 64
      },
      "ivfpq_nprobe32": {
        "p50_ms": 2.874,
        "p95_ms": 3.0761,
        "mean_ms": 2.8586,
        "qps": 349.8,
        "recall@1": 0.575,
        "recall@10": 0.811,
        "bytes_per_vector": 64
      },
      "hnsw_ef16": {
        "p50_ms": 0.0422,
        "p95_ms": 0.046,
        "mean_ms": 0.0431,
        "qps": 23180.3,
        "recall@1": 0.61,
        "recall@10": 0.9555,
        "bytes_per_vector": 2048
      },
      "hnsw_ef64": {
        "p50_ms": 0.1518,
        "p95_ms": 0.1791,
        "mean_ms": 0.1585,
        "qps": 6308.9,
        "recall@1": 0.615,
        "recall@10": 0.961,
        "bytes_per_vector": 2048
      },
      "hnsw_ef256": {
        "p50_ms": 0.7238,
        "p95_ms": 0.9964,
        "mean_ms": 0.761,
        "qps": 1314.1,
        "recall@1": 0.615,
        "recall@10": 0.9675,
        "bytes_per_vector": 2048
      }
    }
  }
}
```

