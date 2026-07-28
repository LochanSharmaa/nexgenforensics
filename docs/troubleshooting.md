# Troubleshooting

[← Back to README](../README.md)

Start here:

```bash
python scripts/check_environment.py
```

Exit `0` recognition works, `1` service runs but cannot recognise, `2` service
cannot run at all.

---

### The service will not start: `EngineUnavailableError`

The recognition dependencies are missing:

```bash
pip install -r backend/requirements-engine.txt
```

This is intentional. There is no fallback recognition mode — a substitute
embedding would produce numbers that look like similarity scores and mean
nothing.

### First start hangs, then fails to load the model

The first run downloads ~300 MB into `~/.insightface/models/`. Check network
access and disk space. To pre-provision, place the pack directory there yourself
and point `NEXGEN_MODEL_ROOT` at its parent.

### `ImportError: email-validator is not installed`

```bash
pip install -r backend/requirements.txt
```

`EmailStr` needs it at schema-build time, so the failure occurs at import rather
than at request time.

---

### "No face was detected" on an image that clearly has a face

Most often the image is an **already-cropped face**. A detector expects a face to
occupy part of a scene; given one that fills the frame it finds nothing.

The pipeline handles this with pad-and-retry, and responses set
`padded_detection: true` when that path was used. If it still fails:

- The face may be below `NEXGEN_MIN_DETECTION_CONFIDENCE` (default `0.50`)
- The face may be smaller than ~60 px on its shorter edge
- Extreme pose or heavy occlusion

### Enrolment rejected with 422

Read `quality.blocking_reasons` in the response. Only blocking reasons refuse an
image:

| Reason | Meaning |
|---|---|
| `face_too_small` | Too few pixels to carry reliable identity detail |
| `brightness_out_of_range` | Exposure unusable |
| `severe_blur` | Effectively no high-frequency detail left |
| `detection_confidence_below_minimum` | Detector was not confident this is a face |

`advisory_reasons` (pose, moderate blur, low contrast) do **not** block; they
flag the result for examiner attention.

Rejecting weak enrolments is deliberate: one poor enrolment image degrades every
future search against that subject.

---

### Searches return no candidates

- **Is anything enrolled?** Check `gallery` in `/api/imatch/engine/status`. An
  empty gallery correctly returns nothing.
- Candidates below `min_candidate_score` (0.20) are never returned.
- If the gallery was enrolled under a **different model pack**, those templates
  are not comparable. Re-enrol.

### Search returns the wrong person at high similarity

Recalibrate. The shipped thresholds are generic operating points, and false-match
rate rises with gallery size:

```bash
python scripts/calibrate_threshold.py path/to/dataset
```

See [Configuration](configuration.md#threshold-calibration).

### Similarity looks low for the same person

Expected across large age gaps, pose differences or poor capture. On AgeDB,
genuine pairs average ~0.49, not ~0.9. Cosine similarity is not a percentage
confidence — treat 0.5 as a strong same-person signal, since impostor pairs
average ~0.04.

---

### Templates fail to decrypt after a restart

`NEXGEN_TEMPLATE_KEY` changed or was unset. If it was unset, an ephemeral key
was generated per process and those templates are permanently unreadable. The
log names the affected count at startup. There is no recovery path — restore the
original key from backup, or re-enrol.

### Audit chain reports `valid: false`

Treat this as a **security incident**, not a data-quality issue. It means records
were altered or removed after being written. `broken_at` names the first bad
record and `reason` distinguishes a content mismatch from a sequence gap.

### `429 Too Many Requests`

Search has its own lower ceiling (`NEXGEN_SEARCH_RATE_LIMIT_PER_MINUTE`, default
30/min) because a bulk gallery scrape looks like a burst of ordinary searches.

---

### CUDA is not being used

Check `device.effective` in `/api/imatch/engine/status`. Two common causes:

1. The stock `onnxruntime` wheel is CPU-only; CUDA needs `onnxruntime-gpu`.
2. `onnxruntime-gpu` ships kernels only for the compute capabilities it was
   built against. Maxwell cards (compute 5.0) fall outside recent builds.

The service logs which device it chose. CPU inference at ~320 ms/image is
workable for investigative search.

### Frontend: "Could not reach the iMATCH service"

The backend is not running on port 8443, or `VITE_IMATCH_API_BASE` points
somewhere else. In development leave it empty to use the Vite proxy.

### Lint reports `motion` is unused

`eslint-plugin-react`'s `jsx-uses-vars` rule must be enabled — core
`no-unused-vars` does not understand JSX. Deleting those imports breaks the build
at runtime.
