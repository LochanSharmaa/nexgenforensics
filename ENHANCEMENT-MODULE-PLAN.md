# Forensic CCTV Image Enhancement — Design for Approval

**Rev 2 · 2026-08-03 · Status: design, not implemented.**
**Companion to ROADMAP.md §10.3/§3, NEXTGEN-ARCHITECTURE.md §7, COMPARISON-REPORT-PLAN.md.**

---

## 0. What this revision changes, and the hardware it must fit

**Priority, per owner direction:** milestone 1 is *investigator value* — an
investigator uploads a poor CCTV image and gets back a significantly more useful
one for visual examination. Recognition benchmarking follows and is measured,
never assumed.

Rev 1 sequenced generative restoration last, behind the deterministic stages.
That ordering optimised for a recognition claim that milestone 1 does not make.
**Generative restoration is now in milestone 1**, because on a 30-pixel face it
is what actually produces a usable image, and the transparency machinery (§5)
is the control rather than deferral.

One correction to Rev 1 worth stating plainly: I deferred diffusion restorers as
"stochastic by design." That was wrong as written. Diffusion is fully
deterministic given a fixed seed, scheduler, step count, and deterministic
kernels. Better still, **seed variance is a sharper hallucination probe than the
cross-model ensemble I proposed** — same architecture, same weights, same input,
different noise draw, so everything that differs is provably the prior. Diffusion
is admitted, seed-pinned, and its variance is exploited. See §5b.

### Hardware envelope — measured on this machine, 2026-08-03

```
GPU      NVIDIA RTX A3000 Laptop · 6144 MiB · compute 8.6 (Ampere) · driver 580.92
torch    2.5.1+cu121 · torch.cuda.is_available() = True
cv2      5.0.0        skimage 0.26.0        numpy pinned <3.0
```

6 GB is the binding constraint and it decides real model choices:

| Model | fp16 VRAM @512² | Fits 6 GB | Verdict |
|---|---|---|---|
| CodeFormer / GFPGAN / RestoreFormer++ | 1.2–1.6 GB | ✅ comfortably | primary face restorers |
| Real-ESRGAN, SwinIR, NAFNet, FFTformer, Retinexformer | 1–3 GB tiled | ✅ | core stages |
| Restormer, HAT, DRCT | 3–4 GB tiled | ✅ with tiling | secondary |
| ResShift, DifFace (own diffusion backbone) | 2.5–4 GB | ✅ | diffusion tier |
| DiffBIR, StableSR, OSEDiff (SD-2.1 based) | 5–7 GB | ⚠️ marginal — fp16 + tiled VAE, must be measured | evaluate in E6, adopt only if it holds |
| **SUPIR (SDXL-based)** | 12–24 GB | ❌ | **ruled out on this hardware.** Best perceptual quality in the field and not available to us. Revisit only on a ≥16 GB card |

It is also a *laptop* GPU on the developer's working machine: sustained batch
load will thermally throttle. Batch sizing is conservative and every stage
supports tiled inference.

**`torch==2.5.1+cu121` is frozen.** `requirements-gpu.txt` documents why — it
supplies the CUDA 12.1 / cuDNN 9.1 DLLs that `onnxruntime-gpu 1.20.2` loads for
the *recognition* engine. A restorer needing a newer torch gets vendored or
rejected; it does not get to move that pin. This is the single easiest way to
silently break face recognition while appearing to improve enhancement.

---

## 1. Existing architecture — what is reusable

The reuse ratio is high. Most forensic requirements are already solved here and
need wiring, not building.

| Requirement | Already exists | Path |
|---|---|---|
| Never overwrite the original | Content-addressed store, SHA-256 filename, write-once | [`storage_service.py:102`](backend/imatch_api/services/storage_service.py:102) |
| Every step logged | Hash-chained, sequence-ordered, JSONL-mirrored audit | [`audit_service.py:87`](backend/imatch_api/services/audit_service.py:87) |
| Reproducibility / lineage | Content-addressed artefact DAG, tamper-evident | [`forensics/provenance.py:78`](backend/nexgen_engine/forensics/provenance.py:78) |
| Blur σ / noise σ / JPEG QF, each **confidence-gated** | Spectral fit, Immerkær MAD, DCT lattice fit | [`degradation/estimate.py:58`](backend/nexgen_engine/degradation/estimate.py:58) |
| Spectral cut-off, passband projection | `effective_cutoff`, `to_passband`, `common_passband` | [`degradation/bandlimit.py:58`](backend/nexgen_engine/degradation/bandlimit.py:58) |
| 8-component quality scoring, blocking vs advisory | `ImageQualityFilter` | [`quality_filter.py:100`](backend/nexgen_engine/data/quality_filter.py:100) |
| Face detect + 5-pt align, padded retry for small faces | SCRFD + `FaceAligner` | [`detection/alignment.py`](backend/nexgen_engine/detection/alignment.py) |
| **106-pt / 68-pt-3D landmarks** | already in the buffalo_l pack, already loaded, unread | COMPARISON-REPORT-PLAN §0 |
| GPU detect with **real binding probe**, CPU fallback | `_cuda_actually_binds()` | [`runtime.py:67`](backend/nexgen_engine/runtime.py:67) |
| Per-stage timing surfaced to the API | `StageTimings` | [`pipeline.py:28`](backend/nexgen_engine/inference/pipeline.py:28) |
| Paired side-by-side plates in PDF | `_image_cell`, and `draw_enhanced_pair` reserved as a hard failure | [`report_pdf.py:267`](backend/imatch_api/services/report_pdf.py:267) |
| Upload / drop / probe UI | `ImageDropZone`, `ProbeReport`, `ProvenancePanel` | `frontend/src/workspace/components/` |
| Feature-flag pattern | `narrative_enabled: bool = False` | [`core/config.py:128`](backend/imatch_api/core/config.py:128) |

**New dependencies for the deterministic core: zero.** cv2 5.0.0, skimage,
numpy, Pillow are installed. Learned restorers need only the already-installed
torch. **One genuinely new dependency is proposed: `av` (PyAV)** — justified in
§2a, because ffmpeg is not on PATH and OpenCV's `VideoCapture` cannot expose the
frame-type information that turns out to be the single largest free quality win
available.

**Caution:** cv2 is on the **5.0** line. `findTransformECC`, `calcOpticalFlowFarneback`
and `createCLAHE` all survive, but OpenCV 5 dropped assorted legacy APIs. Every
cv2 call gets a smoke test in E0 rather than being assumed.

---

## 2. What real CCTV actually looks like — the gap the benchmarks miss

This is the section that decides whether the module is genuinely good or merely
a wrapper around published models. Restoration benchmarks (DIV2K, GoPro, SIDD,
LOL, CelebA-HQ degraded) do not contain the degradations that dominate real
surveillance footage. Seven differences matter, and each has a concrete handler.

### 2a. It is H.264/H.265, not JPEG — and frame type is a free 3–10 dB

The existing `estimate_quality()` fits a **JPEG** DCT lattice. A frame pulled
from a CCTV DVR has different physics:

* 4×4/8×8 **integer** transforms, not the 8×8 float DCT;
* an **in-loop deblocking filter** that smooths block edges — so a JPEG-blocking
  estimator systematically *under-reports* compression severity on video;
* spatially varying macroblock QP, so one image has several effective qualities;
* **inter-frame prediction**: a P/B frame's pixels are largely *copied from a
  reference frame*. Detail visible in a P-frame may have been captured at a
  different instant.

That last point is forensically serious and I have not seen it addressed in any
enhancement product: enhancing a P-frame can sharpen detail that belongs to
another moment, then present it as though observed at the frame's timestamp.

**Handler.** Parse frame type from the bitstream and *prefer I-frames*. On a
typical CCTV GOP (I-frame every 1–4 s) an I-frame is dramatically cleaner than
the P-frames around it — commonly several dB — because it is intra-coded at a
lower QP with no prediction residue. This is the highest quality-per-effort
lever in the entire design and it costs one dependency and no model. Non-I
frames remain usable but are **labelled** with their type and their reference
distance.

`cv2.VideoCapture` cannot report frame type. `av` (PyAV) exposes
`frame.pict_type` directly. That is the justification for the one new dependency.

### 2b. Night mode is infrared — colour is not just wrong, it is absent

Most fixed cameras switch to IR illumination after dark: the IR-cut filter
retracts, the image becomes effectively monochrome, and skin/hair/fabric
reflectances are *IR* reflectances, not visible ones.

Every face restorer in §4 is trained on visible-light RGB faces. Given an IR
frame, they will confidently produce plausible daylight skin tone and texture.
**All of that colour is invented**, and worse, it looks natural.

**Handler.** Detect IR capture from inter-channel correlation (≈1.0) and
near-zero saturation. When detected: force the single-channel path, disable
every colour-restoration stage, and emit a mandatory report line — *"Infrared
capture: the original contains no colour information. Any colour in the enhanced
image is synthetic."* This is a hard rule, not an advisory.

### 2c. Interlacing — still everywhere on analog DVRs

704×576 / 704×480 interlaced is common on installed estates. On motion it
produces comb artifacts. Feed those to a denoiser or SR model and it will treat
comb teeth as genuine high-frequency detail and *sharpen them*.

**Handler.** Detect via row-parity comb energy. Then **separate fields rather
than blend them** — the two fields are 1/50 s apart, giving two genuine
half-height temporal samples. That is strictly better than yadif for our
purposes, because §3's multi-frame fusion can use both. Blending discards a real
observation.

### 2d. Stored resolution routinely overstates true resolution

DVRs commonly upscale before storage, and footage is often re-encoded through
several stages. A "1080p" export can carry no real detail above 480p.

**This codebase can already detect it.** `effective_cutoff()` measures the
highest frequency carrying real energy. If it sits well below 0.5 cycles/px, the
image was upscaled and the extra pixels are empty.

**Handler.** Report **true effective resolution** beside stored pixel
dimensions, and let the planner skip super-resolution when the image is already
upscaled — running ×4 SR on an already-upscaled frame multiplies invention
without adding information. Existing code, new and directly useful output.

### 2e. Backlighting and clipped highlights

A doorway with daylight behind it. The face is in shadow at 10–30 DN; the
background is clipped at 255. Clipped pixels carry **no** recoverable
information — they are a saturated sensor well.

**Handler.** Report clipped fraction inside the face region. Tone-map the shadow
side aggressively; state explicitly that clipped regions are unrecoverable
rather than letting a model invent texture there.

### 2f. OSD burn-in

Timestamps, camera IDs and channel names are burned into the pixels and
routinely overlap faces. Restorers will try to interpret them as facial
structure.

**Handler.** Detect the static overlay by temporal invariance across frames
(it does not move; the scene does), mask it out of restoration, and mark masked
regions in the output. Never inpaint silently.

### 2g. Wide-angle / dome geometric distortion

Faces at the frame edge of a dome camera are geometrically stretched, which
degrades both alignment and restoration.

**Handler.** Optional lens-distortion correction before alignment. Fully
principled only when the camera is available for calibration — which
NEXTGEN-ARCHITECTURE §7b already identifies as the forensic asymmetry worth
exploiting. Ships as an operator-supplied correction with sane defaults.

---

## 3. The pipeline — and the one ordering decision that matters most

```
video ──► [2a] I-frame-preferred demux (PyAV, frame type recorded)
            │
            ├─► [2c] field separation if interlaced   ─┐
            │                                          │  N temporal samples
            ├─► [2b] IR detection → colour policy      │
            │                                          │
            ▼                                          │
     detect + TRACK the face across frames  ◄──────────┘
            │
            ▼
     frame selection — NOT "the sharpest frame"
     but a diverse, well-registered subset of the SAME track
            │
     ┌──────┴──────────────────────────────────────────────┐
     │  per frame:  deblock(codec-aware) → denoise          │
     └──────┬──────────────────────────────────────────────┘
            ▼
     [★] SUB-PIXEL REGISTRATION + ROBUST MULTI-FRAME FUSION
            │        real information gain — detail from the sensor
            ▼
     illumination / tone  →  bounded deconvolution (capped by measured MTF)
            │
            ├────────────────────────► TRACK A OUTPUT
            │                          "processed image" — no invented detail
            ▼
     [TRACK B] generative restoration on the FUSED crop
            │   Real-ESRGAN / SwinIR upscale · CodeFormer face restore
            │   diffusion tier, N seeds → median + variance
            ▼
     TRACK B OUTPUT — "reconstruction", labelled, always beside the original
```

### The ordering decision: fuse first, restore second

Off-the-shelf video enhancement restores each frame and then smooths
temporally. That bakes per-frame hallucination in and then averages the
hallucinations together, which makes them look *more* convincing, not less.

**Fusing first and restoring once is materially better**, for two reasons:

1. The restorer's input is cleaner and higher-resolution, so it has to invent
   less. Generative restorers degrade gracefully with input quality — a
   fused crop from 8 frames pushes CodeFormer from "mostly prior" toward
   "mostly evidence."
2. There is exactly one generative step to disclose, at one point in the
   lineage, with one set of parameters. Restoring 8 frames means 8 sets of
   invented detail to account for.

This is the single biggest quality lever after I-frame selection, and both are
free of new models.

### The planner

Rule-based and versioned — not learned, because §Explainable Processing requires
that every selection state *why*. It emits ordered stages, parameters, a human
rationale per stage, **and the stages considered and skipped with their reason**.
Ordering is fixed by physics: it inverts the capture chain. Deblock before
denoise (blocking is structured; a Gaussian-noise denoiser preserves it while
smoothing real detail around it). Denoise before deblur (deconvolution amplifies
noise by dividing by a small MTF).

Every trigger is gated on the *confidence* of its measurement, mirroring the
`trust_components` conjunction already in `estimate_degradation()` — that
pattern exists because a confidently wrong operator was identified as the most
dangerous output this system can produce, and it applies here verbatim.

The plan is shown to the investigator **before** execution and is overridable;
overrides are audited.

---

## 4. Model comparison — per task, within 6 GB

Selection criteria, in order: real-degradation robustness > fit to a constraint
this codebase already has > VRAM > benchmark PSNR. Where that moves me off the
highest-scoring model I say so.

### Compression artifacts
| Candidate | Assessment |
|---|---|
| **FBCNN** ✅ **selected** | The only one taking an **explicit quality factor as input**. This codebase already estimates QF *with a confidence*, so FBCNN turns an existing measurement into a controllable, reportable parameter instead of an internal guess. Nothing else pairs this cleanly with what is built. |
| SwinIR-CAR | Strong, but fixed-QF models; needs one checkpoint per quality. Secondary. |
| QGAC, ARCNN | Superseded. |
| — | **Gap: all of these are JPEG-trained.** For H.264/H.265 (§2a) they are approximations. Mitigation is I-frame preference plus a codec-aware deblocking pre-pass; a video-codec-native artifact remover is a genuine open problem and is flagged as such rather than papered over. |

### Denoising
| Candidate | Assessment |
|---|---|
| **NAFNet-SIDD** ✅ **primary** | Best real-noise/compute trade currently available; no attention nondeterminism; ~1 GB. |
| **Multi-frame temporal fusion** ✅ **primary when ≥3 frames** | Beats every single-image denoiser on real footage, and does it *without a prior*. Temporal redundancy is the strongest denoising signal in existence. |
| **BM3D / NLM** ✅ **kept** | Fully deterministic, classical, 15 years of literature, no learned prior. This is the answer when a defence expert asks for a method with no neural network in it. Cheap to report alongside. |
| Restormer, SCUNet, MIRNet-v2 | Comparable or heavier; no decisive advantage here. |

### Low light
| Candidate | Assessment |
|---|---|
| **Zero-DCE++** ✅ **Track A** | Estimates a **per-pixel monotone tone curve** and applies it. Structure-preserving by construction — it cannot synthesise an edge, only remap values that exist. That property outweighs its weaker LOL score. ~50 KB model. |
| **Retinexformer** ✅ **Track B** | Substantially better-looking on severe low light; generative enough that it belongs on the reconstruction side. |
| CLAHE / local tone mapping | Deterministic baseline, always available, always reported. |
| SCI, LLFlow, RetinexNet | No decisive advantage over the two above. |

### Motion deblur
| Candidate | Assessment |
|---|---|
| **NAFNet-GoPro** ✅ **selected, bounded** | Regression-trained. Deconvolution strength **capped by the measured MTF**, so it cannot restore beyond the band the sensor recorded — which is what keeps it in Track A. |
| **FFTformer** ✅ **secondary** | Frequency-domain; often better on large blur; useful as a second opinion. |
| DeblurGAN-v2 | ❌ Adversarial objective on a task whose failure mode is invented detail. |
| Restormer, MPRNet, Uformer | Viable, no decisive edge. |

### Defocus deblur
**Restormer-DPDD** ✅ — routed to only when §4's spectral-null analysis shows
radial symmetry (defocus) rather than directional periodic nulls (motion).

### Multi-frame fusion ★ — the legitimate detail recovery
| Candidate | Assessment |
|---|---|
| **Classical robust MFSR** ✅ **selected** | Sub-pixel registration (ECC / Farnebäck optical flow on the face crop), robust L1/median fusion on a common grid, outlier rejection for occlusion and movers. **No learned prior anywhere in the path**, so recovered detail is provably sensor-derived. NEXTGEN §7c requires this be kept strictly separate from generative restoration; classical is how. |
| BasicVSR++, RVRT, VRT | Better-looking, and they hallucinate temporally. Offered in Track B only. |

### Super-resolution (Track B)
| Candidate | Assessment |
|---|---|
| **Real-ESRGAN** ×2/×4 ✅ **primary** | The robustness benchmark for genuinely degraded real input; stable mirrored weights; tiles well inside 6 GB. |
| **SwinIR-Real** ✅ **second opinion** | Architecturally different (transformer regression), so its *disagreement* with Real-ESRGAN is informative for §5c rather than redundant. |
| **ResShift** ✅ **diffusion tier** | Own lightweight backbone, not Stable Diffusion; 4–15 steps; 2.5–4 GB. The diffusion option that comfortably fits this card. |
| DRCT, ATD, HAT | Strong on synthetic ×4; trained on bicubic degradation and transfer poorly to CCTV. Available, not default. |
| DiffBIR, StableSR, OSEDiff | ⚠️ SD-2.1 based, 5–7 GB. Measure in E6 with fp16 + tiled VAE; adopt only if the measurement holds on this card. |
| **SUPIR** | ❌ Ruled out — needs 12–24 GB. Noted honestly as the best output we cannot run. |

### Face restoration (Track B)
| Candidate | Assessment |
|---|---|
| **CodeFormer** ✅ **primary**, fidelity weight *w* exposed | The only major face restorer with an **explicit, documented fidelity↔quality knob**. That means the report can state a *number* for how far the operator was moved toward the prior — exactly what a forensic record needs and what GFPGAN structurally cannot give. Default *w* = 0.7 (fidelity-leaning), not the prettier default. |
| **GFPGAN v1.4**, **RestoreFormer++** ✅ **ensemble members** | Different priors → their disagreement is the §5c hallucination measure. |
| **DifFace** ✅ **diffusion tier** | Own backbone, fits; seed-variance probe (§5b) applies. |
| VQFR, PGDiff | Viable, no decisive edge. |

### Quality assessment
`ImageQualityFilter` (exists) + **CR-FIQA** for ISO/IEC 29794-5-aligned face
utility. ROADMAP §11.3 wants this anyway for the NIST FATE Quality submission,
so it is folded into E1 rather than left optional — answering my own Rev 1 open
question in the affirmative.

---

## 5. Transparency — measuring synthesis, not disclaiming it

You asked that reconstruction be transparent in the interface and reports. Every
product does this with a watermark. This codebase can do it **quantitatively**,
three ways, two of which use code that already exists.

### 5a. Band-limit synthesis mask — physics

`effective_cutoff()` measures the highest frequency carrying real energy in the
original. `to_passband()` projects onto a band. So:

```
synthesised = enhanced − to_passband(enhanced, cutoff_of_original)
```

Everything in that residual is, **by construction**, above the frequency band
the evidence contained. It cannot have come from the sensor. Rendered as an
overlay, and reported as a number:

> **Synthesis fraction 0.31** — 31% of the enhanced image's energy lies above the
> measured spectral cut-off of the original (0.081 cycles/px at 99% energy).
> That content is generated, not observed.

Reproducible by an opposing expert, which a disclaimer is not.

### 5b. Seed variance — for diffusion restorers

Same weights, same input, same everything except the noise draw. Every
difference across N seeds is the prior. Per-pixel variance is a direct
hallucination map, and the median across seeds is a better output than any
single draw. This is why diffusion is admitted rather than deferred: it is the
only model class that hands us a free, exact measure of its own invention.

### 5c. Cross-model disagreement — for deterministic restorers

CodeFormer (two *w* values), GFPGAN, RestoreFormer++ on the same aligned crop.
Where they agree, the evidence constrains the output; where they disagree, each
is filling in from its own prior. Cost: k× inference on one 512² crop — under a
second on this GPU.

### 5d. Multi-frame agreement — the strongest of the four

Restore several *independent frames* of the same track separately. The true face
is shared across them; the noise and the prior's response to that noise are not.
Regions consistent across independently-restored frames are evidence-driven.
This is a real cross-check, available only because §3 tracks faces across
frames.

### 5e. Identity drift

Cosine(embedding(original crop), embedding(enhanced crop)) under the production
recogniser. Large drift means the restorer moved the face toward a *different
identity* — precisely the failure mode ROADMAP §3.3 names. Reported per stage.

**Type separation makes the labelling unforgeable.** `RestoredImage` (Track A)
and `ReconstructedImage` (Track B) are distinct types. `draw_enhanced_pair`
accepts a `ReconstructedImage` only together with its `OriginalImage` parent —
which is what its existing `NotImplementedError` docstring already demands. The
LR/likelihood path accepts neither. The plugin registry forces every new model
to declare a track, so the separation cannot be bypassed by a contributor who
has not read this document.

---

## 6. Module layout

```
backend/nexgen_engine/enhancement/
  __init__.py        # two-track invariant, mirroring degradation/__init__.py
  types.py           # OriginalImage | RestoredImage | ReconstructedImage, Plan, StageResult
  registry.py        # plugin registry; track declaration is mandatory
  analysis.py        # DegradationProfile — extends degradation/estimate.py
  cctv.py            # ★ §2: codec artifacts, IR detect, interlace/field split,
                     #   true-vs-stored resolution, clipping, OSD, distortion
  planner.py         # rules → Plan, versioned, rationale + skip reasons
  runner.py          # deterministic executor: seeds, tiling, timing, hashing, lineage
  attribution.py     # §5a band-limit · 5b seed var · 5c ensemble · 5d multi-frame · 5e drift
  metrics.py         # pre/post scoring and deltas
  weights.py         # cached, SHA-256-verified weight fetch
  video/
    demux.py         # PyAV, I-frame preference, frame-type recording
    track.py         # face tracking across frames
    select.py        # diverse informative subset, not "the sharpest"
  backends/
    classical.py     # deblock, BM3D/NLM, CLAHE, bounded deconv, Lanczos
    mfsr.py          # ★ classical multi-frame — Track A
    fbcnn.py  nafnet.py  restormer.py  zerodce.py  fftformer.py
    realesrgan.py  swinir.py  resshift.py  retinexformer.py
    codeformer.py  gfpgan.py  restoreformer.py  difface.py

backend/imatch_api/
  api/routes/enhancement.py       # analyze | enhance | jobs | recognize
  services/enhancement_service.py
  db/models.py                    # + EnhancementRun, EnhancementStage;
                                  #   SearchRun gains source_kind + enhancement_run_id
  services/report_pdf.py          # implement draw_enhanced_pair() to its existing contract

frontend/src/workspace/
  EnhancementPage.jsx
  components/SplitCompare.jsx      # slider + synchronised pan/zoom
  components/SynthesisOverlay.jsx  # §5a mask, §5b/c/d heatmaps, toggleable
  components/PipelinePlan.jsx      # applied + skipped stages, rationale, timings
  components/MetricDelta.jsx       # before → after
  services/enhancementApi.js
```

**Determinism, and its honest limit.** Fixed seeds,
`torch.use_deterministic_algorithms(True)`, no TTA, weights pinned by SHA-256 and
verified on load. That gives run-to-run determinism on a fixed host. It does
**not** give bit-identical output across GPU architectures or between GPU and
CPU — cuDNN kernel selection and reduction order differ. So the module defines a
**canonical reproduction mode**: CPU, fp32, deterministic kernels. Every run
records device, driver and library versions and which mode produced it. This is
the same honesty `resolve_providers()` already applies: the device is reported,
never assumed.

**Async.** Enhancement is seconds-to-minutes against a ~100 ms search path, so
it is a job table plus an in-process worker, polled by the UI — not Celery until
a measurement says otherwise (ROADMAP §12f).

---

## 7. Roadmap — investigator value first

| Phase | Deliverable | Gate |
|---|---|---|
| **E0** *(2d)* | Types, registry, lineage, DB tables, `enhancement_enabled` flag, weight cache + checksums, **cv2 5.0 API smoke tests**, torch-pin regression test | `ReconstructedImage` provably cannot reach the LR path; `draw_enhanced_pair` cannot render unpaired |
| **E1** *(4d)* | Analysis: motion/defocus discrimination, anisotropic kernel, illumination, occlusion, clipping, **CR-FIQA**, plus **§2 CCTV suite** — IR detect, interlace detect, true-vs-stored resolution, OSD detect | Estimator accuracy measured against known ground truth, same protocol as the existing JPEG-estimator validation |
| **E2** *(4d)* | Planner + Track A classical stages. CPU-only, zero new deps | Deterministic: same input → same SHA-256 over 100 runs |
| **E3** *(5d)* | **Learned single-image stages on GPU**: FBCNN, NAFNet, Zero-DCE++, Real-ESRGAN, SwinIR, **CodeFormer**. Tiled inference, CPU fallback | A 30-px CCTV face produces a visibly and measurably better crop |
| **E4** *(5d)* | **Investigator UI**: split-view, synchronised zoom, plan trace with skip reasons, metric deltas, processing history, side-by-side always | **▶ MILESTONE 1 — an investigator can upload a poor CCTV image and get back a significantly more useful one.** Ships here. |
| **E5** *(4d)* | Attribution: §5a band-limit mask, §5c ensemble, §5e identity drift, wired into UI and PDF | Synthesis fraction ≈ 0 for Track A and clearly > 0 for Track B. If it does not separate, the method is wrong and is reported as such |
| **E6** *(4d)* | Video: PyAV I-frame demux, field separation, tracking, frame selection. Diffusion tier (ResShift, DifFace) + §5b seed variance. Measure DiffBIR/StableSR on 6 GB | I-frame preference measured against naive frame grab |
| **E7** *(5d)* | ★ Multi-frame registration + robust fusion, **fuse-then-restore** ordering, §5d agreement | Measured detail gain over the best single frame on controlled multi-frame capture |
| **E8** *(3d)* | Report integration: `draw_enhanced_pair`, methodology + limitations sections, audit lines, synthesis fraction and identity drift in the PDF | Report states model, version, parameters, synthesis fraction, drift |
| **E9** *(4d)* | **Recognition benchmarking.** Full SCface / TinyFace sweep: original vs each Track A stage vs each Track B model | ROADMAP §10.3 decision point — see the pre-registered gate below |
| **E10** *(3d)* | Async job queue, batch, GPU throughput tuning | p95 latency measured and published |

≈ 43 working days. **Milestone 1 lands at E4 (≈ 20 days).** E5 follows
immediately because shipping generative output without the synthesis measurement
would breach Core Principle 7.

**Pre-registered gate for E9, fixed now so it cannot be tuned to the result:**
adopt an enhancement stage into the *recognition* path only on **≥ +2.0 points
TAR@FAR=0.1% with a bootstrap CI excluding zero**, on a **real** degraded
benchmark (SCface or TinyFace — not synthetically degraded LFW), with **no**
benchmark regressing by more than 0.5 points. Same gate structure S0.3 used.
S0.3 failed its gate and was published as a failure; this may too, and that
result is shippable either way (ROADMAP §13.3).

Note that E9's outcome does not affect milestone 1. Visual examination value and
recognition gain are different claims, and only the second one is gated.

---

## 8. What this design will not do

* Put an enhanced image into the chain-of-custody hash or the LR computation.
* Present enhanced output without the original beside it and the label
  [`report_pdf.py:267`](backend/imatch_api/services/report_pdf.py:267) specifies.
* Claim a recognition improvement before E9 measures one.
* Colorize infrared capture (§2b), or silently inpaint OSD regions (§2f).
* Claim bit-exact reproducibility across heterogeneous hardware — it defines a
  canonical mode instead.
* Move the `torch==2.5.1+cu121` pin.

---

## 9. Open questions

1. **Video ingestion is now on the critical path.** §2a (I-frame preference) and
   §3 (multi-frame fusion) are the two largest quality levers and both need
   video. ROADMAP §12a scopes video ingestion as a separate workstream. Confirm
   folding the minimal demux/track/select slice into this module (as E6/E7
   above), with the full acquisition layer staying in §12a?
2. **PyAV** is the one new dependency, justified by frame-type access that
   OpenCV cannot provide. Approve, or prefer bundling an ffmpeg binary?
3. **Milestone-1 acceptance.** "Significantly more useful" needs a concrete
   test. Proposal: a fixed set of ~20 real degraded CCTV frames, scored blind by
   the owner as better / same / worse against the original, with ≥ 80% "better"
   as the bar. If you have real casework-representative footage, that set is
   worth assembling before E3 so the models are chosen against it rather than
   against DIV2K.
