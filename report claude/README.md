# Next Gen Forensics — AI-Assisted Facial Comparison Reporting System

This is a working **prototype scaffold** for the reporting layer described in
`docs/build_prompt.md`. It wraps around a facial recognition model (mocked
here with `backend/recognition_adapter/adapter.py`) and produces:

- validated structured JSON findings
- annotated evidence figures (bounding box, landmarks, measurements,
  morphological regions, probe-vs-candidate, differences, quality)
- a professional multi-section forensic PDF report
- a minimal investigator-facing web UI + API

## IMPORTANT — read before using

- **The recognition model in this scaffold is a mock/stub.** Swap
  `backend/recognition_adapter/adapter.py` for a thin wrapper around your
  real model. Do not change any other module's contract when you do —
  everything downstream only consumes the JSON shape defined in
  `backend/schemas/comparison_report.schema.json`.
- **Discrete-feature thresholds are illustrative**, not forensically
  validated. Calibrate `backend/config/discrete_thresholds.yaml` against a
  reference population with a domain expert before any real use.
- **No output in this system claims identity.** Similarity scores are
  investigative leads only. See the disclaimer baked into every generated
  PDF and `backend/report/report_model.py`.

## Quick start

```bash
cd backend
pip install -r requirements.txt --break-system-packages

# 1. Run the full pipeline end-to-end on the bundled sample images
python run_demo.py

# outputs land in: sample_data/output/
#   comparison_report.json
#   figures/fig1..fig8.png
#   ForensicComparisonReport.pdf

# 2. Or start the API + open the frontend
uvicorn api.main:app --reload --port 8000
# then open frontend/index.html in a browser (it calls http://localhost:8000)
```

## Project layout

```
backend/
  recognition_adapter/   # wrapper around YOUR existing model (mock included)
  detection/              # face detection stage (mock, swappable)
  landmarks/              # landmark stage (mock, swappable) + a/b/c layer geometry
  quality/                # image quality assessment
  measurements/           # continuous + discrete anthropometric feature engine
  similarity/             # embedding similarity + soft-feature similarity
  schemas/                # JSON Schema + validator
  visualization/          # Figures 1-8, pure functions of (image, json)
  report/                 # JSON -> report data model
  pdf/                    # report data model -> PDF (reportlab)
  api/                    # FastAPI app
  config/                 # thresholds, model names — no hardcoded logic values
  tests/                  # unit tests for measurements, visualization, schema
  run_demo.py             # end-to-end pipeline runner (no API needed)
frontend/
  index.html              # minimal investigator UI (upload, review, download)
sample_data/
  probe.png, candidate.png (synthetic demo faces + landmark fixtures)
docs/
  build_prompt.md         # the full original spec this scaffold implements
  architecture.md
```

## Integrating your real model

1. Implement `RecognitionAdapter.detect_and_embed(image_path)` in
   `backend/recognition_adapter/adapter.py` to call your actual model and
   return `(bbox, landmarks, embedding_vector, detection_confidence)` in the
   shapes documented in that file's docstring.
2. Leave every other module untouched — they only depend on that adapter's
   output shape and the JSON schema.
3. Run `pytest backend/tests` to confirm nothing downstream broke.
