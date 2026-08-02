# Data Requirements

**Nothing here has been downloaded. This is a request list for approval.**
Generated alongside `docs/DATASET_INVENTORY.md` (what we already hold) on 2026-08-02.

The ordering is by **what unblocks a decision**, not by size or prestige. Several
large, famous datasets are ranked low because nothing currently blocked depends
on them.

---

## A. Already held — no action needed

| Dataset | State | Blocking |
|---|---|---|
| LFW, CFP-FF, CFP-FP, AgeDB-30, CALFW, CPLFW | `.bin` packs + cached embeddings | Nothing. Fully analysed on CPU |
| TinyFace Testing_Set (8,171 imgs, 2,569 ids) | Extracted + cached embeddings | Nothing |
| TinyFace `Gallery_Distractor` (153,428 imgs) | Extracted, **not embedded** | GPU |
| TinyFace `Training_Set` (7,804 imgs, 2,570 ids) | Extracted, **never used** | GPU |
| QMUL-SurvFace (5,319 ids) | Extracted, **not embedded** | GPU |
| IJB-B + IJB-C complete suite (8.6 GB tar) | Downloaded, **not extracted** | CPU extraction + GPU embedding |

**Important:** three of the six rows are already-owned assets that have never been
used. Before requesting anything new, that is where the cheapest gains are.

---

## B. DOWNLOAD REQUIRED — ranked

### Priority 1 — IJB-S (IARPA Janus Surveillance)

| | |
|---|---|
| **Why** | The benchmark that defines the surveillance niche. Published leaders sit at 63.44% rank-1 (KP-RPE) and 44.81% (FaceMoE). Without it we cannot position against anyone |
| **Depends on it** | Stage 1 success criterion (Cllr ≤ 0.85 on IJB-S); Stage 3 success criterion (+5 pts TAR@FAR=0.1%) |
| **Size** | ~150 GB (video + stills) |
| **Storage** | 200 GB working |
| **Licence** | NIST/IARPA data agreement, institutional signature. **Lead time is months — file now even if we never use it** |
| **Compute** | Video decode + embedding: substantial |

### Priority 2 — SCface

| | |
|---|---|
| **Why** | Real multi-distance surveillance capture (1 m / 2.6 m / 4.2 m) with a matched high-quality mugshot per subject. **This is the closest public proxy to the paired HR/LR data S0.3 needs**, and unlike TinyFace it has ground-truth distance |
| **Depends on it** | S0.3 arms B2/B3 validation on real optics rather than synthetic degradation |
| **Size** | ~3 GB, 130 subjects, 4,160 images |
| **Storage** | 5 GB |
| **Licence** | Free for research; email request to University of Zagreb |
| **Compute** | Trivial |

**I would move this to Priority 1 if lead time on IJB-S proves long.** It is small,
fast to obtain, and directly tests the central hypothesis on real cameras.

### Priority 3 — BRIAR

| | |
|---|---|
| **Why** | Long-range (up to 1,000 m), elevated platforms, multimodal face+body+gait. The only corpus for the range regime |
| **Depends on it** | Stage 2 training on real degraded imagery; any multimodal claim |
| **Size** | Multiple TB |
| **Storage** | 5+ TB |
| **Licence** | IARPA agreement, likely US-institution restricted. **May not be obtainable** |
| **Compute** | Large |

### Priority 4 — UCCS (UnConstrained College Students)

| | |
|---|---|
| **Why** | Long-range surveillance with an explicit **open-set** protocol — directly exercises `H_unknown` and conformal coverage, which nothing we hold does well |
| **Depends on it** | Stage 1 open-set validation |
| **Size** | ~15 GB |
| **Licence** | Research registration |

### Priority 5 — Multi-PIE

| | |
|---|---|
| **Why** | Controlled pose × illumination grid. **Ground-truth nuisance labels** — the supervised-nuisance training objective needs exactly this to validate that invariance is learned rather than assumed |
| **Depends on it** | Stage 2 (λ₃ invariance term, λ₅ leakage penalty) |
| **Size** | ~300 GB |
| **Licence** | **Paid** (~USD 1,000, CMU) |
| **Note** | Only worth buying once Stage 2 is actually starting |

### Priority 6 — FaceScape / NeRSemble

| | |
|---|---|
| **Why** | Multi-view 3D face capture. **The binding constraint on Stage 3** — a renderer cannot be trained without them |
| **Depends on it** | Stage 3 only, which is contingent on S0.3 passing |
| **Size** | FaceScape ~500 GB; NeRSemble ~2 TB |
| **Licence** | Academic agreement, non-commercial |
| **Recommendation** | **Do not request until S0.3 returns PASS.** If it fails, these are never needed |

### Priority 7 — WebFace42M / Glint360K

| | |
|---|---|
| **Why** | Proposal-network pretraining at scale |
| **Size** | WebFace42M ~2 TB; Glint360K ~300 GB |
| **Reality check** | **An RTX A3000 6 GB cannot train on these.** Requesting them now would consume days of bandwidth and terabytes of disk for data we cannot process. Defer until a training cluster exists |

### Priority 8 — VoxCeleb1/2

| | |
|---|---|
| **Why** | The only large paired face+voice video corpus |
| **Depends on it** | Multimodal work, which is not in Stages 0–2 |
| **Recommendation** | Defer |

---

## C. Must create ourselves — the differentiated assets

### C1. Camera degradation corpus — highest value, lowest compute

Measured PSF, MTF, noise model and compression pipeline for camera models that
actually appear in casework.

- **Method:** slanted-edge MTF target, flat-field frames for noise, a known chart
  through each camera's full pipeline. Bench work, not compute.
- **Why it matters:** the forward-model layer currently runs on *estimated*
  parameters, and `estimate.py` documents that blur recovery on flat facial
  content is unreliable. Measured parameters remove that error source entirely.
- **Why nobody else has it:** vendors sell software and do not attend scenes.
- **Cost:** a test chart, a tripod, patience. **No GPU.**
- **Status:** could start today with any available camera.

### C2. Paired HR/LR same-scene capture

Same subjects, same moment, one HR camera and one real CCTV unit at 5/10/20/50 m.

- **Why:** the only data that validates forward-model fidelity against reality
  rather than against our own synthesis. Our failed fine-tune already showed that
  synthetic degradation does not transfer.
- **Gate:** requires consent and ethics approval. Not a compute problem.

### C3. Declared forensic reference cohorts

Demographically declared populations for LR denominators.

- **Why:** `docs/CAPACITY_VALIDATION.md` shows reference-population purity *is*
  the measurement. Every current LR is conditional on a corpus, not a population.

---

## Recommendation

**Request now:** SCface (small, fast, directly tests the hypothesis) and file the
IJB-S agreement (long lead time, costs nothing to start).

**Do not request yet:** FaceScape/NeRSemble (contingent on S0.3), Multi-PIE
(contingent on Stage 2 starting), WebFace42M/Glint360K (hardware cannot use them).

**Start regardless:** C1, the camera degradation corpus. It needs no download, no
GPU, and no permission.
