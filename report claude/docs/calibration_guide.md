# Calibration Guide

## Overview

This guide explains how to calibrate and validate the forensic facial comparison pipeline for your deployment environment. Calibration involves two distinct activities:

1. **Score threshold calibration** — setting the cosine similarity decision threshold to match your operational context.
2. **Soft-feature covariance calibration** — tuning the Mahalanobis distance covariance diagonal for your target population.

> **Important:** This system produces investigative support material, not determinations of identity. All calibration work should be reviewed by a qualified forensic examiner before operational use.

---

## 1. Score Threshold Calibration

The primary similarity score is a **cosine similarity** between two 512-dimensional embedding vectors. The decision threshold is stored in:

```
backend/config/technical_info.yaml → similarity_threshold
```

### 1.1 Establishing a Decision Threshold

To calibrate:

1. **Obtain a calibration set** — a balanced collection of genuine (same-person) and impostor (different-person) image pairs drawn from your target population. Minimum 200 pairs of each type is recommended.

2. **Run the pipeline on all pairs:**
   ```bash
   cd "report claude/backend"
   python - <<'EOF'
   from pipeline import build_comparison_report
   import json, pathlib

   pairs = [...]  # your list of (probe_path, candidate_path, label) tuples
   results = []
   for probe, cand, label in pairs:
       r = build_comparison_report(probe, cand)
       score = r["similarity_metrics"]["cosine_similarity"]
       results.append({"score": score, "genuine": (label == "genuine")})

   with open("calibration_scores.json", "w") as f:
       json.dump(results, f, indent=2)
   EOF
   ```

3. **Compute ROC and select threshold** using your operational working point (e.g., FAR=0.1%, FMR=1%). Common metrics:
   - **EER** (Equal Error Rate) — threshold where FAR == FRR
   - **FMR100** — threshold where FMR = 1%
   - **Operational point** — selected based on operational risk tolerance

4. **Update the threshold** in `config/technical_info.yaml`:
   ```yaml
   similarity_threshold: 0.76   # replace with your calibrated value
   ```

### 1.2 Validation

After updating the threshold, run the test suite to ensure nothing broke:
```bash
cd "report claude/backend"
python -m pytest tests/ -v
```

---

## 2. Soft-Feature Covariance Calibration

The supplementary Mahalanobis distance uses a **diagonal covariance matrix** defined in `similarity/engine.py` under `COVARIANCE_DIAGONAL`. Each entry is the **population variance** for that feature.

### 2.1 Measuring Population Variance

1. Collect a set of **reference images** from a representative population sample (minimum 500 individuals).
2. Run measurements for all images:
   ```bash
   python - <<'EOF'
   from recognition_adapter.adapter import RecognitionAdapter
   from measurements.engine import compute_continuous_measurements
   import json

   adapter = RecognitionAdapter()
   all_measurements = []
   for img_path in image_paths:
       det = adapter.detect_and_landmark(img_path)
       m = compute_continuous_measurements(det.landmark_schema, det.landmarks)
       all_measurements.append({r["feature_id"]: r["value"] for r in m})

   with open("population_measurements.json", "w") as f:
       json.dump(all_measurements, f, indent=2)
   EOF
   ```
3. Compute variance per feature:
   ```python
   import json, numpy as np, collections

   data = json.load(open("population_measurements.json"))
   by_feature = collections.defaultdict(list)
   for row in data:
       for k, v in row.items():
           by_feature[k].append(v)
   variances = {k: float(np.var(vals)) for k, vals in by_feature.items()}
   print(variances)
   ```
4. Update `COVARIANCE_DIAGONAL` in `similarity/engine.py` with your computed variances.

### 2.2 Interocular Distance Normalization

All distance-based measurements are normalized by the interocular distance before computing similarities. If your population has unusual distributions, verify that `interocular_distance` is reliably detected for your images.

---

## 3. Landmark Model Quality Checks

MediaPipe FaceMesh is configured with:
```python
refine_landmarks = True   # 478-point iris mesh
min_detection_confidence = 0.5
```

For operational forensic use, consider raising `min_detection_confidence` to `0.7–0.85` in `adapter.py` to reject low-confidence detections rather than producing unreliable measurements.

Quality gates are applied via:
```
backend/config/discrete_thresholds.yaml → blur, brightness, contrast limits
```

Adjust these limits based on your image acquisition conditions.

---

## 4. Pose Estimation Parameters

The `solvePnP` estimator uses a generic 3D face model (6 landmark points). The accuracy of yaw/pitch/roll estimates depends on how closely your probe images resemble the assumed focal length:

```python
focal_length = float(image_width)  # approximate assumption
```

For images with known camera parameters, replace this with the actual focal length in pixels for better accuracy.

---

## 5. Regular Re-Calibration

Calibration should be revisited when:
- The recognition backbone model changes
- The target population demographics change significantly
- Landmark detection model is updated
- Significant drift is observed in operational score distributions

---

## 6. Reporting Calibration Decisions

All calibration parameters should be documented in the **Technical Appendix** (Section 12) of each generated report. The threshold used is stored in every JSON report under `similarity_metrics.threshold_used`.
