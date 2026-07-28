# Models & performance

[← Back to README](../README.md)

## The pipeline

```
image bytes
  → decode (EXIF-aware)
  → SCRFD detection            ~54 ms
  → five-point landmarks
  → similarity-transform alignment to the ArcFace layout    ~2 ms
  → ArcFace embedding, flip-TTA averaged                   ~260 ms
  → L2 normalisation
  → cosine similarity via FAISS                              ~5 ms
  → ranked candidates
```

## Model packs

Downloaded on first run into `NEXGEN_MODEL_ROOT` (default `~/.insightface`).

| Pack | Detection | Recognition | Notes |
|---|---|---|---|
| `buffalo_l` **(default)** | SCRFD-10GF | ArcFace `w600k_r50`, 512-d | Verified. Best speed/accuracy balance under 4 GB VRAM |
| `buffalo_s` | SCRFD-500M | `w600k_mbf`, 512-d | Fastest, lowest accuracy |
| `antelopev2` | SCRFD-10GF | `glintr100` (ResNet100), 512-d | More accurate, roughly 2× the cost. **Not benchmarked by this build** |

Switching packs invalidates every stored template. Templates from different
models are not comparable, and comparing across them yields a meaningless score
rather than an error. Keep the original enrolment images and re-enrol before
cutting over.

## There is no fallback mode

The service loads real recognition weights or refuses to start.

An earlier revision degraded to a "deterministic stub" that hashed pixels into a
vector. It kept the API answering while making every score meaningless — the
worst possible failure mode for this system, because nothing downstream looked
wrong. That code is gone, along with the centre-crop "detector" that asserted a
face was present without looking.

If the model cannot load you get `EngineUnavailableError` with the reason, and
no result. Inspect what is actually running:

```bash
curl -H "Authorization: Bearer $TOKEN" http://localhost:8443/api/imatch/engine/status
```

## Measured performance

CPU, `buffalo_l`, on the reference host:

| Stage | Time |
|---|---|
| Decode | <1 ms |
| Detection (SCRFD) | ~54 ms |
| Alignment | ~2 ms |
| Embedding (ArcFace, flip-TTA) | ~260 ms |
| Search (FAISS, 567 templates) | ~5 ms |
| **Total** | **~320 ms** |

Two optimisations account for a ~2.5× speed-up over the first working version:

1. **Calling the detection model directly** rather than `FaceAnalysis.get()`,
   which also runs the recognition network internally and made ArcFace execute
   three times per image.
2. **Sizing the detector input to the image** instead of always using 640×640.
   Running a 640×640 detector over a 112 px thumbnail costs roughly ten times
   what it needs to.

## Pre-cropped probes

A detector trained on photographs expects a face to occupy *part* of a scene.
Given an image that is only a face — a mugshot, a database thumbnail, a
previously cropped probe — it finds nothing.

Measured on AgeDB (112×112 crops), the first detection pass found **0 of 120**
faces. Re-running with a 40 % replicated border found all 120 at ~0.8
confidence.

Forensic probes are frequently already cropped, so this is the common case
rather than an edge case. Images at or below 200 px skip the unpadded pass
entirely instead of paying for one that will fail. Every coordinate is mapped
back to the original image, and responses carry `padded_detection` so an
examiner can see how the detection was obtained.

## Search backend

FAISS `IndexFlatIP` where available, falling back to a numpy matmul. Both are
**exhaustive** over L2-normalised vectors, so the two paths return identical
rankings — FAISS is a speed optimisation, never a different answer.

That distinction matters here. An approximate index would silently drop true
candidates, and a missed lead is invisible to the examiner. Brute-force cosine
is exact and fast enough well past 10⁵ templates per tenant on CPU; beyond that,
put an ANN backend behind `GalleryIndex` and accept the recall/latency trade-off
explicitly rather than inheriting it.

## GPU acceleration

`NEXGEN_ENGINE_DEVICE=cuda` is a *request*, not a guarantee. The service checks
whether `CUDAExecutionProvider` is genuinely registered, logs the device it
chose, and reports the effective device through the status endpoint. **The
device never changes the result** — same weights, same arithmetic.

Two things commonly prevent CUDA from engaging:

- The stock `onnxruntime` wheel is **CPU-only**. CUDA additionally requires
  `onnxruntime-gpu`.
- `onnxruntime-gpu` ships kernels only for the compute capabilities it was built
  against. Maxwell-generation cards (compute capability 5.0, e.g. Quadro M1200)
  fall outside recent builds and will run on CPU regardless.

At ~320 ms per image, CPU inference is entirely workable for investigative
search, which is not a real-time workload. GPU mainly helps bulk enrolment.

## Fine-tuning

Usually unnecessary — see [`backend/FINETUNE_GUIDE.md`](../backend/FINETUNE_GUIDE.md),
which lists the cheaper interventions to try first (threshold calibration, more
images per subject, better enrolment quality) before training anything.
