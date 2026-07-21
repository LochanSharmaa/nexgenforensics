# Master Build Prompt: AI-Assisted Forensic Facial Comparison Reporting System

Copy everything below into your AI coding agent (Claude Code, Cursor, etc.) as the system/task prompt.

---

## ROLE

You are a senior AI software architect, computer vision engineer, forensic software engineer, and full-stack developer. Build a complete **AI-Assisted Facial Comparison Reporting System** that wraps around an existing, already-trained facial recognition model. You are building the reporting, measurement, visualization, and PDF-generation layer — not a new recognition engine.

## HARD CONSTRAINTS (do not violate)

- Do **NOT** replace, retrain, fine-tune, or modify the existing facial recognition model, its pipeline, or its embeddings. It is the source of truth and is treated as a black box that exposes: face crop, bounding box, landmarks, embedding vector, and a similarity/distance score.
- The recognition model **only returns structured, machine-readable JSON**. It never generates prose, reports, or figures.
- The report generator, visualizer, and PDF builder consume **only** that JSON — never console output, never screenshots, never inferred/hallucinated values.
- Every number in every report **must trace back to a field in the backend JSON**. If a value isn't present, omit it — never fabricate, estimate, or interpolate it.
- The system produces **investigative/research support material**, not a determination of identity. No output may claim or imply confirmed identification. A human examiner makes the final call.

## PIPELINE (fixed order)

```
Evidence Image
  → Face Detection
  → Landmark Detection
  → Image Quality Assessment
  → Anthropometric Measurement Engine
  → Morphological Feature Engine
  → Face Embedding (existing model, untouched)
  → Similarity Engine
  → Structured JSON
  → Visualization Engine
  → Report Generator
  → Professional PDF
```

---

## 1. BACKEND JSON CONTRACT

Define a versioned JSON schema (e.g. `schemas/comparison_report.schema.json`, JSON Schema draft-2020-12) with these top-level sections. Validate every backend response against it before it reaches the report generator.

- **administrative**: case/report ID, requesting agency, date created, software version, schema version
- **evidence**: probe image id/hash, candidate image id(s)/hash(es), chain-of-custody notes (free text field, examiner-filled)
- **image_metadata**: filename, dimensions, format, EXIF (if present), capture date if available
- **quality_metrics**: resolution, blur_score, noise, compression_artifact_score, brightness, contrast, lighting_uniformity, pose {yaw, pitch, roll}, occlusion_map/score, face_visibility_pct, detection_confidence
- **face_detection**: bounding_box {x, y, w, h}, detection_confidence, model_name/version
- **landmarks**: schema {68 | 106 | 468}, array of `{id, x, y, confidence}`, model_name/version
- **measurements_continuous**: array of `{feature_id, name, value, unit ("px" | "ratio" | "mm" if calibrated), landmarks_used[], formula_ref}`
- **measurements_discrete**: array of `{feature_id, name, category, threshold_basis, landmarks_used[]}`
- **morphological_features**: array of `{region, probe_observation, candidate_observation, comparison_label ("similar"|"minor_variation"|"observed_variation"|"not_assessable"), notes}`
- **embedding_metrics**: model_name/version, embedding_dimension, vector_hash (not the raw vector, unless explicitly required)
- **similarity_metrics**: cosine_similarity, euclidean_distance, angular_distance, threshold_used, decision_category (e.g. "below threshold" / "above threshold" — never "match"/"no match" as a factual claim), candidate_rank
- **candidate_ranking**: array of `{candidate_id, score, rank}` for 1:N searches
- **explainable_ai** (optional, only if backend provides it): grad_cam / attention_map / saliency_map / occlusion_map image references
- **limitations**: array of machine-generated caveats tied to quality_metrics (e.g. low resolution, high yaw, partial occlusion) — templated from rules, not free-text LLM generation
- **technical_info**: detection model, recognition model, embedding model, similarity algorithm, landmark model, quality model, hardware, OS, GPU, software version, inference_time_ms, model_hashes
- **audit**: full processing log with timestamps for each pipeline stage, input/output hashes per stage for reproducibility

Reject and log any pipeline stage that returns partial/malformed data rather than silently proceeding with defaults.

---

## 2. LANDMARK & MEASUREMENT ENGINE (grounded in forensic literature)

Base the anthropometric measurement engine on the **facial soft biometric feature methodology of Tome et al. ("Facial Soft Biometric Features for Forensic Face Recognition")**, adapted to whatever landmark set your detector actually outputs (68 / 106 / 468). Implement it as a **pure function library**, decoupled from the detector, that maps `landmarks[] → measurements[]`.

### Landmark layers
- `a_i`: raw landmarks from the detector (the "facial" landmarks)
- `b_i`: derived **geometrical** landmarks computed by simple geometry from `a_i` (e.g. midpoints, projections) — implement as pure functions, not detector output
- `c_i`: **estimated** landmarks requiring light image processing (e.g. nose-root sunken point, mouth upper point, chin left/right limits) — implement as a documented, testable sub-module; if it can't be computed reliably from your landmark set, omit rather than approximate

### Continuous features to implement (name → definition)
Implement each as `distance(landmark_a, landmark_b)` or `angle(landmark_a, landmark_b)` in pixel space by default; convert to a physical unit only if a calibration reference (e.g. known IPD or scale marker) is supplied, otherwise report **pixel distance or a normalized ratio** (e.g. relative to interocular distance) — never fabricate millimeters.

| Region | Continuous measurements |
|---|---|
| Forehead | height, width |
| Eyebrows | separation, inner/outer elevation (L/R), length (L/R), average width (L/R), angle between corners (L/R) |
| Eyes/Orbit | horizontal opening (L/R), interocular distance, angle between corners (L/R) |
| Nose | width, height, nose root width, naso-labial height |
| Mouth | length, average height, angle between corners |
| Chin | width, height |
| Ears | length (L/R), angle between corners (L/R) |
| Contours | average face-line length |

### Discrete (categorical) features
For each continuous measurement that has a forensic categorization equivalent (per the same reference — e.g. forehead height: Short/Average/Large; eyebrow form: Arched/Rectilinear/Sinuous; nose width: Small/Average/Large; mouth orientation: ObliqueL/Neutral/ObliqueR), implement threshold-based bucketing. **Do not hardcode magic thresholds** — load them from a config file (`config/discrete_thresholds.yaml`) so they can be recalibrated per population/dataset without a code change, and document that out-of-the-box thresholds are illustrative, not forensically validated, until calibrated against a reference population.

### Similarity measures for the soft-feature vector (independent of the core embedding similarity)
Implement Euclidean, Mahalanobis, and Hamming (discrete only) distance functions over normalized feature vectors, purely as **supplementary corroborating evidence** — clearly labeled as separate from the primary embedding-based similarity score, never merged into a single opaque number.

---

## 3. IMAGE QUALITY ASSESSMENT MODULE

Independent module, runs after detection, before measurement:
- resolution, blur_score (e.g. variance of Laplacian), noise estimate, compression artifact estimate
- brightness, contrast, lighting uniformity
- pose: yaw / pitch / roll (from landmark geometry or a pose estimator)
- occlusion score / occlusion map
- face visibility %, detection confidence
- Emit **rule-based limitation strings** when metrics cross configurable thresholds (e.g. "yaw exceeds 15°: measurements affected") — templated, not LLM-generated, and always phrased as reliability caveats, never as pass/fail verdicts.

---

## 4. VISUALIZATION ENGINE

Deterministic, code-based rendering only (e.g. OpenCV/PIL/matplotlib/SVG) — **never an image-generation model** — drawing directly on the original uploaded image using only backend JSON coordinates.

Generate, per comparison:

1. **Fig 1** — original evidence image
2. **Fig 2** — bounding box overlay
3. **Fig 3** — landmark overlay, every point numbered
4. **Fig 4** — anthropometric measurement overlay: drawn lines, each labeled with feature name + value
5. **Fig 5** — morphological region map (forehead, eyes, eyebrows, nose, mouth, jaw, chin, ears) as shaded/outlined regions
6. **Fig 6** — probe vs. candidate side-by-side, matched landmark numbering
7. **Fig 7** — difference visualization, color-coded by `comparison_label` (similar / minor variation / observed variation)
8. **Fig 8** — quality visualization panel (blur, pose, lighting, occlusion indicators)
9. **Explainability figures** (Grad-CAM/attention/saliency/occlusion map) **only if** the backend actually provides them — omit entirely otherwise, do not synthesize placeholders

Each figure function should take `(image, json_section) → image_bytes` and be independently unit-testable with synthetic landmark data.

---

## 5. REPORT GENERATOR + PDF BUILDER

Separate the **content assembly** (JSON → structured report model, e.g. Pydantic/dataclasses) from **PDF rendering** (e.g. WeasyPrint/ReportLab/LaTeX via a template engine like Jinja2). No layer should contain hardcoded numeric content — every figure/table/value comes from the report model, which comes only from validated JSON.

### PDF structure
Cover page → Administrative info → Evidence summary → Purpose of examination → Submitted evidence → Methodology → Image quality assessment → Landmark analysis → Anthropometric measurements → Morphological comparison → Similarity analysis (with plain-language explanation of each metric) → Observed similarities → Observed differences → Limitations → AI findings → **Examiner review (blank fields)** → Technical appendix → Audit log → Disclaimer.

### PDF design requirements
A4, page numbers, evidence identifiers in header/footer, auto-numbered figures/tables, running headers/footers, auto-generated table of contents, high-res embedded figures, internal clickable references (figure/table cross-refs).

### Examiner section (blank, never AI-filled)
Examiner name, signature, date, comments, decision (Confirmed / Rejected / Further review required).

### Mandatory disclaimer text (verbatim intent, may be legally reviewed/edited by the org before use)
> This report documents algorithmic facial comparison findings. The AI system provides similarity assessments, not definitive identification. Results must be interpreted by a qualified examiner in conjunction with all other available investigative information.

Also state explicitly, near the similarity section: the system does not claim identity; it reports statistical similarity/distance and quality-adjusted confidence caveats only.

### Technical appendix must include
Detection/recognition/embedding/similarity-algorithm/landmark/quality model names+versions, hardware, software version, inference time, OS, GPU, timestamp, model hashes where available.

---

## 6. SYSTEM ARCHITECTURE / CODE ORGANIZATION

```
/backend
  /recognition_adapter/     # thin wrapper around the EXISTING model — no logic changes
  /detection/
  /landmarks/
  /quality/
  /measurements/            # continuous + discrete feature engine (pure functions, unit-tested)
  /similarity/
  /schemas/                 # JSON Schema definitions + validators
  /visualization/           # figure generators, pure functions of (image, json)
  /report/                  # JSON -> report data model
  /pdf/                     # report data model -> PDF
  /api/                     # FastAPI/Flask endpoints
  /config/                  # thresholds, model paths, feature toggles — NEVER hardcoded in logic
  /tests/
/frontend
  investigator upload/review UI
/docs
  architecture.md, JSON schema docs, calibration/validation guide
```

Requirements:
- Config-driven (no hardcoded thresholds, paths, or model names in code)
- Structured logging at every pipeline stage, with correlation/request IDs
- Explicit error handling per stage; a failure in one stage must not silently produce a partial/misleading report — fail loudly and mark the report as incomplete
- Unit tests for the measurement engine (synthetic landmark fixtures with known expected distances/angles), visualization engine (does it draw at expected coordinates), and JSON schema validation
- Reproducibility: every report should be regenerable byte-for-identical from the same input JSON + config version (store config/model version hashes in the audit section)

---

## 7. FRONTEND (investigator-facing)

- Upload probe image; select comparison database/candidate set
- Run comparison; show progress per pipeline stage
- Review ranked candidates with similarity scores
- Inspect all annotated figures inline
- View quality metrics and similarity metrics panels
- Download PDF report
- Download raw JSON findings
- No client-side logic computes or alters any measurement/similarity value — display only, all computation server-side

---

## 8. VALIDATION / DEPLOYMENT NOTE (include this as a comment/README, not optional)

Before any research or operational use:
- Have the discrete-feature thresholds and measurement engine reviewed/calibrated by a forensic examiner or domain expert against a representative population, per the Tome et al. methodology this system is based on.
- Do not deploy for real casework decisions without human expert validation and, where applicable, legal/ethics review appropriate to your jurisdiction.
- Treat all similarity outputs as investigative leads, not identification evidence.

---

## DELIVERABLES CHECKLIST FOR THE AGENT

- [ ] JSON schema + validator
- [ ] Recognition model adapter (unmodified wrapper)
- [ ] Detection / landmark / quality / measurement / morphology / similarity modules
- [ ] Visualization engine (Figs 1–8 + optional XAI figs)
- [ ] Report data model + PDF builder with TOC, numbering, headers/footers
- [ ] API layer
- [ ] Frontend upload/review/download UI
- [ ] Config files for all thresholds/models/paths
- [ ] Unit tests for measurement + visualization + schema validation
- [ ] README covering architecture, JSON schema, and the validation note in section 8
