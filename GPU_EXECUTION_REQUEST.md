# GPU Execution Request

**Target hardware: NVIDIA RTX A3000 Laptop GPU, 6 GB VRAM.**
**Status: APPROVED AND EXECUTED 2026-08-02.** G1, G2, G3 and G4 have all run.
Results and their revisions: `docs/MEASUREMENT_RECORD.md`. This file is retained
as the pre-registration of what was requested and why, including the decision
rule fixed before any run; the budget below is what was *requested*.
Raised 2026-08-02 after completing the CPU phase.

Every task below is blocked *only* on inference. All surrounding code, metrics,
statistics, refusal logic and reporting are built, tested (288 passing) and
validated on CPU with a stub embedder.

---

## Hardware reality check, stated first

6 GB VRAM running `w600k_r50` (174 MB ONNX) at batch 32 uses well under 2 GB. All
four tasks below fit comfortably. What does **not** fit on this card: ViT-B/Swin-B
training, renderer training, diffusion, or anything touching WebFace42M. Those are
not requested and should not be attempted on this hardware.

**Total requested: ~5.5–9 GPU-hours across four tasks.**

---

## G1 — Embed the TinyFace `Gallery_Distractor` set

| | |
|---|---|
| **Experiment** | Encode 153,428 distractor images with `w600k_r50` |
| **Why GPU** | 153,428 forward passes. Measured CPU throughput on this machine is roughly 8–12 img/s → **4–5 hours CPU**, versus minutes on the A3000. This is the one task where CPU is merely slow rather than impossible, but it is on the critical path for three others |
| **Runtime (A3000)** | **20–35 min** at batch 32, ~800 MB VRAM |
| **Output** | `runtime/benchmarks/embeddings/tinyface_distractors__w600k_r50.npz` (~300 MB) |
| **Decision enabled** | **Repairs the capacity measurement.** Current supportable-gallery figures rest on a 1,794-entry proxy gallery. A 153k gallery gives the first defensible reference population and the first real open-set 1:N benchmark. `docs/CAPACITY_VALIDATION.md` names this as the top blocker |
| **Priority** | **1** |

---

## G2 — Run S0.3 with the real embedder

| | |
|---|---|
| **Experiment** | `experiments/S0_3/run.py --embedder arcface` — arms A, B1, B2, B3, C on TinyFace and QMUL |
| **Why GPU** | Each arm re-embeds every transformed image. 5 arms × 2 datasets × ~6,000 images × 2 sides = ~120,000 forward passes. The transforms themselves are CPU and already run |
| **Runtime (A3000)** | **1.5–2.5 hours** including QMUL embedding |
| **Output** | `experiments/S0_3/results/s0_3_arcface_*.json` with per-arm TAR@FAR, AUC, and the paired bootstrap on B2−B1 |
| **Decision enabled** | **The programme's central go/no-go.** Decision rule is registered in code *before* the run: PASS if B2−B1 ≥ +2.0 points TAR@FAR=0.1% on both datasets with CI excluding zero; FAIL if ≤ +0.5 or CI includes zero. A PASS licenses Stage 3 (generative core, multi-year). A FAIL cancels it and redirects to the evidence layer — **saving roughly two years** |
| **Priority** | **1** (tied — this is the highest-information experiment in the plan) |

---

## G3 — Embed QMUL-SurvFace

| | |
|---|---|
| **Experiment** | Encode the QMUL-SurvFace training set (5,319 identities) |
| **Why GPU** | ~50k forward passes; also required as S0.3's second dataset |
| **Runtime (A3000)** | **10–20 min** |
| **Output** | Cached embeddings + a second independent capacity measurement |
| **Decision enabled** | Replication. TinyFace is currently our *only* valid capacity measurement, and a single-corpus result should not carry the weight the plan places on it |
| **Priority** | **2** |

---

## G4 — Extract and run IJB-B / IJB-C

| | |
|---|---|
| **Experiment** | Extract the 8.6 GB suite, embed `loose_crop`, run the shipped `IJB_11.py` protocol |
| **Why GPU** | Hundreds of thousands of forward passes |
| **Runtime (A3000)** | Extraction ~15 min (CPU) + **3–5 hours** embedding |
| **Output** | Our largest unrun benchmark, plus harness validation against the reference `.npy` files shipped in the archive |
| **Decision enabled** | **Validates our measurement harness before we trust our own numbers.** The archive ships published ArcFace results; if we cannot reproduce them, every figure we have produced is suspect. Also corrects a stale claim in PROJECT_OVERVIEW/SCORECARD describing this as a 1.57 GB partial download |
| **Priority** | **2** |

---

## Recommended order and why

```
G1 (30 min)  ──►  G3 (20 min)  ──►  G2 (2 h)   ──►  G4 (4 h)
repair the        replicate on       THE DECISION     validate the
measurement       a second corpus                     harness
```

G1 and G3 first because **G2's verdict is only as trustworthy as the data behind
it**, and both are short. G4 last because it is the longest and its value is
confirmatory rather than decisional.

**If only one task is approved, approve G2** — it is the only one that changes what
we build. But its result is stronger with G1 and G3 done first, and those cost
under an hour combined.

---

## Explicitly NOT requested

Forbidden on this hardware and not proposed: ViT/Swin training · renderer
development · diffusion models · large-scale pretraining · WebFace42M processing ·
synthetic identity training · any Stage 2 or Stage 3 training run.

A 6 GB laptop GPU is a good instrument for answering well-posed questions cheaply.
It is not the tool for building a foundation model, and treating it as one would
waste both the hardware and the time.

---

## What happens after

| S0.3 verdict | Consequence |
|---|---|
| **PASS** | Stage 3 is licensed. Request FaceScape/NeRSemble (`DATA_REQUIREMENTS.md` P6). Seek funding and people — Stage 3 needs 8–15, not one |
| **FAIL** | Stage 3 cancelled. Programme redirects to Stages 1, 2 and 4 — the evidence layer, open-set reasoning, and representation learning. Still publishable, still field-leading, two years saved |
| **INCONCLUSIVE** | Re-run on SCface (real optics, ground-truth distance) before deciding. Do not proceed on an ambiguous result |
