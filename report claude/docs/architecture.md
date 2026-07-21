# Architecture Guide

## System Overview

NexGen Forensics Comparison Reporting is an AI-assisted investigative tool that performs facial similarity analysis between probe and candidate images. It produces structured JSON findings, annotated figures, and a forensic PDF report.

> **Disclaimer:** All outputs are investigative support material. No output constitutes a determination of identity. A qualified forensic examiner must make the final assessment.

---

## High-Level Architecture

```
┌────────────────────────────────────────────────────────────┐
│                      Frontend (HTML/JS)                    │
│   index.html — 1:1 comparison + 1:N ranking UI            │
└────────────────────────┬───────────────────────────────────┘
                         │ HTTP (FastAPI)
                         ▼
┌────────────────────────────────────────────────────────────┐
│                   API Layer (api/main.py)                  │
│   POST /compare          — 1:1 pipeline                    │
│   POST /compare_ranking  — 1:N ranking                     │
│   GET  /report/{id}/pdf  — PDF download                    │
│   GET  /report/{id}/json — JSON download                   │
└────────────────────────┬───────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────┐
│                  Pipeline (pipeline.py)                    │
│   build_comparison_report()   — 1:1                       │
│   build_candidate_ranking()   — 1:N                       │
└──┬────────┬──────────┬──────────┬──────────┬──────────────┘
   │        │          │          │          │
   ▼        ▼          ▼          ▼          ▼
┌──────┐ ┌──────┐ ┌────────┐ ┌──────┐ ┌──────────┐
│Adapt.│ │Qualit│ │Measure.│ │Simil.│ │  Report  │
│      │ │      │ │Engine  │ │Engine│ │  Layer   │
│detect│ │assess│ │compute │ │embed.│ │PDF+Figs  │
│embed │ │_qual.│ │_contin.│ │_sim. │ │builder   │
└──┬───┘ └──────┘ └────────┘ └──────┘ └──────────┘
   │
   ├─ MediaPipe FaceMesh (landmark detection)
   └─ nexgen_engine.FacialRecognitionPipeline (embedding)
```

---

## Module Reference

### `recognition_adapter/adapter.py`
**The only integration seam to the real recognition model.**

- `detect_and_landmark(image_path)` → `DetectionResult`
  - Runs MediaPipe FaceMesh to extract 468 landmark points in pixel space
  - Returns schema `"468"` with landmark coordinates and confidence scores
  - Falls back to synthetic 68-point layout if no face is detected

- `embed(image_path, detection)` → `EmbeddingResult`
  - Calls `nexgen_engine.FacialRecognitionPipeline.encode_image()` as a black box
  - Returns a 512-dimensional L2-normalized embedding vector
  - Falls back to SHA-256-seeded deterministic vector if PyTorch is unavailable

**No other module imports the recognition model directly.**

---

### `quality/assessment.py`

- `assess_quality(image_bgr, detection_confidence, landmarks)` → dict
  - Blur score: Laplacian variance
  - Noise: median-blur residual standard deviation
  - Compression artifacts: 8×8 block gradient ratio
  - Lighting uniformity: left/right brightness symmetry
  - Pose (yaw/pitch/roll): `cv2.solvePnP` with 6-point generic 3D face model
  - Falls back to geometric asymmetry proxy if key landmarks are missing

- `derive_limitations(quality_metrics)` → list[str]
  - Rule-based limitation strings derived from configured thresholds
  - Thresholds in `config/discrete_thresholds.yaml`

---

### `measurements/engine.py`

- `compute_continuous_measurements(schema, landmarks)` → list[dict]
  - Reads measurement definitions from `config/measurement_definitions.yaml`
  - Supports schema `"68"` and `"468"` (MediaPipe FaceMesh)
  - Each measurement: Euclidean distance between named landmark pairs
  - Angle measurements: `arctan2` of landmark vectors

- `compute_discrete_measurements(continuous)` → list[dict]
  - Reads thresholds from `config/discrete_thresholds.yaml`
  - Normalizes distances by interocular distance
  - Buckets ratios into categorical labels (narrow/medium/wide, etc.)

---

### `similarity/engine.py`

Two independent similarity computations (never merged):

1. **Primary — `embedding_similarity(vec_a, vec_b, threshold)`**
   - Cosine similarity, Euclidean distance, angular distance
   - Decision category based on configured threshold
   - This is the headline metric in every report

2. **Supplementary — `soft_feature_similarity(continuous_a, continuous_b, discrete_a, discrete_b)`**
   - Scale-normalized Euclidean distance over continuous anthropometric features
   - Mahalanobis distance using diagonal covariance matrix (`COVARIANCE_DIAGONAL`)
   - Normalized Hamming distance over categorical attributes
   - Always labeled as corroborating/supplementary — never replaces primary score

---

### `pipeline.py`

- `run_pipeline_single_image(path, adapter, role, request_id)` → dict
  - Runs detection → quality → measurements → embedding for one image
  - Records timing and status for each stage in the audit log
  - Structured logging with request correlation IDs

- `build_comparison_report(probe, candidate, case_id, agency)` → dict
  - Combines two single-image results
  - Computes morphological comparison, both similarity metrics
  - Produces fully-validated JSON report (asserted against schema)

- `build_candidate_ranking(probe, candidates, case_id, top_k)` → dict
  - Embeds probe once, then compares against all candidates
  - Returns ranked list sorted by cosine similarity (descending)
  - Does not produce a PDF — returns lightweight JSON for UI consumption

---

### `pdf/builder.py`

- `build_pdf(report_dict, figure_paths, output_path)` → str
  - Deserializes `report_dict` into typed dataclasses via `report/report_model.py`
  - Uses ReportLab Platypus with `multiBuild` for dynamic Table of Contents
  - Clickable internal hyperlinks (figures, tables, sections)
  - Footer with page numbers and report ID
  - Examiner review page with signature fields

---

### `report/report_model.py`

Typed Python dataclasses mirroring the report JSON schema. Provides:
- `from_dict(d)` → `ComparisonReport` — deserialises a validated JSON dict
- Prevents raw dict access from leaking into the PDF layer

---

### `visualization/`

- `generate_all_figures(report, fig_dir)` → dict[str, str]
  - Produces 8 annotated PNG figures:
    - `fig1_original` — original probe image
    - `fig2_bounding_box` — detected bounding box
    - `fig3_landmarks` — landmark overlay
    - `fig4_measurements` — measurement lines
    - `fig5_morphological_regions` — region segmentation
    - `fig6_probe_vs_candidate` — side-by-side comparison
    - `fig7_difference` — difference visualization
    - `fig8_quality_panel` — quality metric plots

---

### `schemas/`

JSON Schema validation for the report output. Called after every
`build_comparison_report()` to ensure output integrity. Defined in
`config/report_schema.json`.

---

## Data Flow — 1:1 Comparison

```
probe.jpg / candidate.jpg
        │
        ▼
RecognitionAdapter.detect_and_landmark()
  → MediaPipe FaceMesh (468 pts, pixel coordinates)
        │
        ├──→ assess_quality()          → quality_metrics dict
        ├──→ compute_continuous()      → measurements list
        ├──→ compute_discrete()        → categorical attributes
        └──→ RecognitionAdapter.embed()
              → FacialRecognitionPipeline.encode_image()
              → 512-d L2-normalized vector

[probe_data + candidate_data]
        │
        ├──→ compare_morphology()      → morphological_features list
        ├──→ embedding_similarity()    → primary similarity metrics
        └──→ soft_feature_similarity() → supplementary metrics

[merged report dict]
        │
        ├──→ assert_valid_report()     → schema validation
        ├──→ generate_all_figures()    → 8 annotated PNGs
        └──→ build_pdf()              → ForensicComparisonReport.pdf
```

---

## Data Flow — 1:N Ranking

```
probe.jpg + [candidate_1.jpg, ..., candidate_N.jpg]
        │
        ▼ (probe embedded once)
RecognitionAdapter.detect_and_landmark() + embed()
        │
        ▼ (for each candidate)
RecognitionAdapter.detect_and_landmark() + embed()
        │
        ▼
embedding_similarity(probe_vec, candidate_vec, threshold)
        │
        ▼
rankings sorted by cosine_similarity DESC
        │
        ▼
JSON {rankings: [{rank, candidate_id, score, decision_category}]}
```

---

## Configuration Files

| File | Purpose |
|------|---------|
| `config/technical_info.yaml` | Model names, similarity threshold, version metadata |
| `config/measurement_definitions.yaml` | Landmark pairs for each measurement, per schema |
| `config/discrete_thresholds.yaml` | Ratio bounds for categorical bucketing |
| `config/region_definitions.yaml` | Polygon points for morphological region visualization |

---

## Key Design Decisions

1. **Single integration seam** — `adapter.py` is the only file that imports the real model. All other modules depend only on the output shape.

2. **No score fusion** — Primary embedding similarity and supplementary soft-feature similarity are kept strictly separate. They are never combined into a single merged score.

3. **No identity claims** — Every report section is labeled as a similarity assessment. The PDF, JSON, and UI all carry the disclaimer.

4. **Deterministic fallback** — When PyTorch/GPU is unavailable, the adapter uses a SHA-256-seeded deterministic vector so the pipeline can be demonstrated and tested without GPU.

5. **Schema versioning** — All reports carry `schema_version` and `software_version` fields for audit trail integrity.

6. **Structured audit log** — Every pipeline stage records start time, completion time, input/output hashes, and status. Audit log is embedded in every JSON report and PDF.
