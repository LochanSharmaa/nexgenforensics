# A7 — Threshold Calibration Record

**Generated:** 2026-07-31 19:39 UTC ·
**Repository state:** `cc96a43f62e1`

| Threshold | Value | Status |
|---|---|---|
| Match | **0.2871** | In force |
| Review band | 0.2153 | In force |
| Routing (degraded specialist) | 0.539 | Measured, not enabled by default |

The decision threshold is the most consequential number in this system: every
match or non-match a user sees is that comparison. A threshold sitting in a
configuration file looks like a tuning parameter, and it is not — it is the
point at which the system asserts two photographs show the same person.

---

# Part I — How the current value was arrived at

## The failure that started it

A user submitted two images of **different people** and the interface reported a
match at similarity **0.2405**.

The cause was not the threshold being wrong in principle. It was that the
threshold existed as **four independent constants in four files**, which had
drifted apart, and the lowest value in the chain was **0.20** — low enough to
admit a genuine false match. The engine, the API, the test suite and the
interface did not agree on what the system's decision rule was.

Recorded in full as A2 / F-03.

## The structural fix

`ThresholdConfig` in `nexgen_engine/config.py` became the single source of
truth. Every other site derives from it rather than restating it:

```python
match: float = 0.2871
review: float = 0.2153
verify: float = 0.2871
```

An API test that had hard-coded `top_score=0.36` was rewritten to derive its
expectation from configuration. That test had been passing against the same
stale constant the code used, so it could not have detected the drift — it was
confirming the bug rather than catching it.

## Why 0.2871

Raising the threshold trades recall for a lower false-match rate. For a system
generating investigative leads about real people that is the correct direction:
a missed lead is recoverable by other means, a false lead attaches a name to a
person who is not in the image.

0.2871 places the demonstrated false pair (0.2405) below threshold with margin,
rather than immediately beneath it.

---

# Part II — Measured effect of each candidate

Same model, same pair sets; only the decision threshold differs. FNMR is the
false-non-match rate (true pairs rejected); FMR the false-match rate (different
people accepted).


## Threshold 0.20

The value in force when a user demonstrated a false match at 0.2405.

| Cohort | Genuine pairs | Impostor pairs | FNMR | FMR |
|---|---|---|---|---|
| ALL | 8,098 | 32,000 | 3.30% | 1.1906% |
| gender=f | 3,300 | 16,000 | 4.88% | 1.7000% |
| gender=m | 4,798 | 16,000 | 2.21% | 0.6813% |
| age=0-25 | 854 | 8,000 | 7.61% | 1.2125% |
| age=26-40 | 2,595 | 8,000 | 3.24% | 1.2625% |
| age=41-55 | 2,199 | 8,000 | 2.14% | 1.1000% |
| age=56+ | 2,450 | 8,000 | 2.90% | 1.1875% |

| Field | Value |
|---|---|
| `model` | w600k_r50 |
| `threshold_used` | 0.2 |
| `threshold_kind` | operating (deployed decision threshold) |
| `threshold_at_fmr_1e3` | 0.2870860145310593 |
| `note` | locally constructed pairs; not comparable to published AgeDB-30 |

<details><summary>Raw artefact — <code>demographics_w600k_r50_thr020.json</code></summary>

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

</details>


## Threshold 0.2363

Suite-wide recalibration candidate. REJECTED — see Part III.

| Cohort | Genuine pairs | Impostor pairs | FNMR | FMR |
|---|---|---|---|---|
| ALL | 8,098 | 32,000 | 4.17% | 0.4437% |
| gender=f | 3,300 | 16,000 | 5.88% | 0.6625% |
| gender=m | 4,798 | 16,000 | 3.00% | 0.2250% |
| age=0-25 | 854 | 8,000 | 10.77% | 0.3750% |
| age=26-40 | 2,595 | 8,000 | 3.85% | 0.4500% |
| age=41-55 | 2,199 | 8,000 | 2.50% | 0.4875% |
| age=56+ | 2,450 | 8,000 | 3.71% | 0.4625% |

| Field | Value |
|---|---|
| `model` | w600k_r50 |
| `threshold_used` | 0.2363 |
| `threshold_kind` | operating (deployed decision threshold) |
| `threshold_at_fmr_1e3` | 0.2870860145310593 |
| `note` | locally constructed pairs; not comparable to published AgeDB-30 |

<details><summary>Raw artefact — <code>demographics_w600k_r50_thr0236.json</code></summary>

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

</details>


## Threshold 0.2871

**Adopted.** In force in the delivered system.

| Cohort | Genuine pairs | Impostor pairs | FNMR | FMR |
|---|---|---|---|---|
| ALL | 8,098 | 32,000 | 6.32% | 0.0969% |
| gender=f | 3,300 | 16,000 | 8.45% | 0.1625% |
| gender=m | 4,798 | 16,000 | 4.86% | 0.0312% |
| age=0-25 | 854 | 8,000 | 14.75% | 0.1000% |
| age=26-40 | 2,595 | 8,000 | 5.78% | 0.0500% |
| age=41-55 | 2,199 | 8,000 | 3.91% | 0.1500% |
| age=56+ | 2,450 | 8,000 | 6.12% | 0.0875% |

| Field | Value |
|---|---|
| `model` | w600k_r50 |
| `threshold_used` | 0.2871 |
| `threshold_kind` | operating (deployed decision threshold) |
| `threshold_at_fmr_1e3` | 0.2870860145310593 |
| `note` | locally constructed pairs; not comparable to published AgeDB-30 |

<details><summary>Raw artefact — <code>demographics_w600k_r50_thr0287.json</code></summary>

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

</details>


---

# Part III — The recalibration that was measured and REJECTED

After 0.2871 was adopted, a suite-wide calibration was run to check the value
against every dataset rather than against the single demonstrated failure. It
proposed **0.2363**.

**It was rejected.** At 0.2363 the originally-reported false pair, at 0.2405,
scores above threshold and is admitted again — the exact failure the change was
made to eliminate.

This is recorded because it is the more instructive outcome of the two. 0.2363
is the value that optimises an aggregate metric across the benchmark suite. It
is also the value that reintroduces a demonstrated, reproducible false match on
real submitted images. **An aggregate optimum is not automatically the correct
operating point**, and a calibration script that returns a number is not
authority to deploy it.

## Calibration output that proposed 0.2363

| Dataset | incumbent | combined | own_fmr_threshold |
|---|---|---|---|
| lfw | {'threshold': 0.2, 'accuracy': 0.9976666666666667, 'fnmr': 0.003, 'fmr': 0.0016666666666666668} | {'threshold': 0.23628424632584857, 'accuracy': 0.998, 'fnmr': 0.0033333333333333335, 'fmr': 0.0006666666666666666} | 0.21372 |
| agedb_30 | {'threshold': 0.2, 'accuracy': 0.9813333333333333, 'fnmr': 0.03133333333333333, 'fmr': 0.006} | {'threshold': 0.23628424632584857, 'accuracy': 0.9796666666666667, 'fnmr': 0.03933333333333333, 'fmr': 0.0013333333333333333} | 0.23878 |
| cfp_fp | {'threshold': 0.2, 'accuracy': 0.9742857142857143, 'fnmr': 0.05, 'fmr': 0.0014285714285714286} | {'threshold': 0.23628424632584857, 'accuracy': 0.9711428571428572, 'fnmr': 0.05771428571428571, 'fmr': 0.0} | 0.20943 |
| calfw | {'threshold': 0.2, 'accuracy': 0.9606666666666667, 'fnmr': 0.07466666666666667, 'fmr': 0.004} | {'threshold': 0.23628424632584857, 'accuracy': 0.9606666666666667, 'fnmr': 0.07733333333333334, 'fmr': 0.0013333333333333333} | 0.24805 |
| cplfw | {'threshold': 0.2, 'accuracy': 0.9431666666666667, 'fnmr': 0.11166666666666666, 'fmr': 0.002} | {'threshold': 0.23628424632584857, 'accuracy': 0.9365, 'fnmr': 0.12566666666666668, 'fmr': 0.0013333333333333333} | 0.24044 |

| Field | Value |
|---|---|
| `model` | w600k_r50 |
| `target_fmr` | 0.001 |
| `combined_threshold` | 0.23628424632584857 |
| `single_dataset_candidate` | 0.2871 |
| `spread` | 0.038611444259145755 |

<details><summary>Raw artefact</summary>

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

</details>

## Suite behaviour at the adopted value

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


---

# Part IV — What the threshold does NOT fix

## Demographic differentials persist

Comparing the cohort tables in Part II across the three values: raising the
threshold **relocated** the errors, it did not remove the disparity between
cohorts. Women show approximately **1.7x** the false-non-match rate of men, and
under-25s approximately **3.8x** the 41-55 band.

A single global threshold applies the same decision rule to populations on which
the system does not perform equally. That is a property of the model, and no
choice of threshold corrects it. It is recorded as an unresolved limitation in
the Model Card (L3) rather than presented as managed.

## One threshold cannot serve both conditions

The adopted value is calibrated on clean-imagery benchmarks. On degraded
imagery the same model achieves **33.13% TAR at FAR=0.1%** — roughly one true
match in three. No threshold recovers information the embedding does not carry.

The measured response to that is not a different threshold but a different
model, selected per probe. See Part V.

---

# Part V — The routing threshold (0.539)

A degraded-condition specialist model is available and can be selected per
comparison using the quality score the pipeline already computes.

**The operating point was chosen before the measurement, not after it.** The
threshold was derived from QMUL-SurvFace and CASIA-WebFace quality
distributions, both of which are **disjoint from every reporting benchmark**:

| | median | tail |
|---|---|---|
| QMUL (degraded, n=3,000) | 0.4677 | p90 0.5250 |
| CASIA (clean, n=2,924) | 0.7547 | p10 0.5671 |

Crossover, where the miss rate on degraded equals the false-route rate on clean,
is **0.539** (5.8% and 5.9% respectively).

A threshold sweep had shown 0.50 producing better benchmark numbers. It was
**not** used, because selecting an operating point on the sets it will then be
reported against is fitting to the test set — the same error class as A2 / F-04,
in a new place.

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

## Measured result at 0.539

| Dataset | TAR@FAR0.1% deployed | routed | Delta | % routed |
|---|---|---|---|---|
| lfw | 99.70% | 99.70% | +0.00pp | 3% |
| agedb_30 | 96.03% | 95.97% | -0.07pp | 1% |
| cfp_fp | 94.69% | 94.66% | -0.03pp | 3% |
| cfp_ff | 99.86% | 99.86% | +0.00pp | 0% |
| calfw | 92.10% | 92.10% | +0.00pp | 0% |
| cplfw | 87.40% | 87.27% | -0.13pp | 7% |
| tinyface | 33.13% | 37.37% | +4.23pp | 92% |


---

# Part VI — Change control

The threshold is part of the validated configuration. Changing it **invalidates
every accuracy figure in this delivery**, because every one of them was measured
under a protocol that fits a threshold on nine folds and applies it to the
tenth.

Before any change:

1. Re-run `calibrate_threshold_suite.py` and record the proposal.
2. Re-run the full verification suite at the candidate value.
3. Re-run `benchmark_demographics.py` at the candidate value — a change that
   improves the aggregate may worsen a cohort.
4. Confirm the candidate still rejects the known false pair at 0.2405.
5. Re-issue the Model Card with a new version.

Step 4 exists because it is the step that stopped 0.2363.
