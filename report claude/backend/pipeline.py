"""
pipeline.py

Orchestrates the fixed pipeline order:
Evidence Image -> Detection -> Landmarks -> Quality -> Measurements ->
Morphology -> Embedding -> Similarity -> Structured JSON

Every stage's timing/hash is recorded to the audit log. This module does
NOT draw figures or build PDFs — it only produces the validated JSON that
those layers consume.

Structured logging: every log line carries request_id and stage for
correlation.  Set LOG_LEVEL=DEBUG to see per-stage detail.
"""
from __future__ import annotations

import hashlib
import logging
import time
import uuid
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2

from recognition_adapter.adapter import RecognitionAdapter
from quality.assessment import assess_quality, derive_limitations
from measurements.engine import compute_continuous_measurements, compute_discrete_measurements
from measurements.morphology import compare_morphology
from similarity.engine import embedding_similarity, soft_feature_similarity
from schemas.validator import assert_valid_report

_CONFIG_DIR = Path(__file__).parent / "config"

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)


class _PipelineLogger:
    """Thin wrapper that injects request_id and stage into every log record."""

    _FMT = logging.Formatter(
        "%(asctime)s %(levelname)-8s [%(request_id)s] %(stage)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )

    def __init__(self, request_id: str):
        self._request_id = request_id
        self._logger = logging.getLogger("pipeline")
        if not self._logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(self._FMT)
            self._logger.addHandler(h)
            self._logger.propagate = False  # avoid double-logging

    def _extra(self, stage: str) -> dict:
        return {"request_id": self._request_id, "stage": stage}

    def info(self, stage: str, msg: str):
        self._logger.info(msg, extra=self._extra(stage))

    def warning(self, stage: str, msg: str):
        self._logger.warning(msg, extra=self._extra(stage))

    def error(self, stage: str, msg: str):
        self._logger.error(msg, extra=self._extra(stage))



def _file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def _load_yaml(name: str) -> dict:
    with open(_CONFIG_DIR / name) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

class AuditLog:
    def __init__(self):
        self.stages = []

    def record(self, stage: str, started_at: float, status: str = "ok",
               input_hash: str = "", output_hash: str = ""):
        self.stages.append({
            "stage": stage,
            "started_at": datetime.fromtimestamp(started_at, tz=timezone.utc).isoformat(),
            "completed_at": datetime.now(tz=timezone.utc).isoformat(),
            "input_hash": input_hash,
            "output_hash": output_hash,
            "status": status,
        })

    def as_dict(self):
        return {"stages": self.stages}


# ---------------------------------------------------------------------------
# Single-image pipeline
# ---------------------------------------------------------------------------

def run_pipeline_single_image(
    image_path: str,
    adapter: RecognitionAdapter,
    role: str = "probe",
    request_id: str = "",
) -> dict:
    """Runs detection -> landmarks -> quality -> measurements -> embedding
    for ONE image.  Returns an intermediate dict (not yet a full report) that
    build_comparison_report() combines for probe + candidate."""
    log = _PipelineLogger(request_id or str(uuid.uuid4()))
    audit = AuditLog()

    # --- Detection & Landmarks ---
    t0 = time.time()
    try:
        detection = adapter.detect_and_landmark(image_path)
        audit.record("detection_and_landmarks", t0, output_hash=_file_hash(image_path))
        log.info("detection", f"[{role}] schema={detection.landmark_schema} conf={detection.detection_confidence:.3f}")
    except Exception as exc:
        audit.record("detection_and_landmarks", t0, status="error")
        log.error("detection", f"[{role}] FAILED: {exc}")
        raise RuntimeError(f"Detection failed for {role}: {exc}") from exc

    # --- Load image ---
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise RuntimeError(f"Could not read image at: {image_path}")
    h, w = image_bgr.shape[:2]

    # --- Quality assessment ---
    t0 = time.time()
    try:
        quality_metrics = assess_quality(image_bgr, detection.detection_confidence, detection.landmarks)
        audit.record("quality_assessment", t0)
        log.info("quality", f"[{role}] blur={quality_metrics['blur_score']} noise={quality_metrics.get('noise')}")
    except Exception as exc:
        audit.record("quality_assessment", t0, status="error")
        log.error("quality", f"[{role}] FAILED: {exc}")
        raise RuntimeError(f"Quality assessment failed for {role}: {exc}") from exc

    # --- Measurements ---
    t0 = time.time()
    try:
        continuous = compute_continuous_measurements(detection.landmark_schema, detection.landmarks)
        discrete = compute_discrete_measurements(continuous)
        audit.record("measurements", t0)
        log.info("measurements", f"[{role}] {len(continuous)} continuous, {len(discrete)} discrete")
    except Exception as exc:
        audit.record("measurements", t0, status="error")
        log.error("measurements", f"[{role}] FAILED: {exc}")
        raise RuntimeError(f"Measurements failed for {role}: {exc}") from exc

    # --- Embedding ---
    t0 = time.time()
    try:
        embedding = adapter.embed(image_path, detection)
        audit.record("embedding", t0)
        log.info("embedding", f"[{role}] model={embedding.model_version} dim={embedding.vector.shape[0]} t={embedding.inference_time_ms:.1f}ms")
    except Exception as exc:
        audit.record("embedding", t0, status="error")
        log.error("embedding", f"[{role}] FAILED: {exc}")
        raise RuntimeError(f"Embedding failed for {role}: {exc}") from exc

    return {
        "role": role,
        "image_path": image_path,
        "image_bgr": image_bgr,
        "width": w,
        "height": h,
        "detection": detection,
        "quality_metrics": quality_metrics,
        "continuous": continuous,
        "discrete": discrete,
        "embedding": embedding,
        "audit_stages": audit.stages,
    }


# ---------------------------------------------------------------------------
# 1:1 Comparison Report
# ---------------------------------------------------------------------------

def build_comparison_report(
    probe_path: str,
    candidate_path: str,
    case_id: str = "",
    requesting_agency: str = "",
) -> dict:
    request_id = str(uuid.uuid4())[:8]
    log = _PipelineLogger(request_id)
    adapter = RecognitionAdapter()
    tech_cfg = _load_yaml("technical_info.yaml")
    discrete_cfg = _load_yaml("discrete_thresholds.yaml")

    log.info("pipeline", f"Starting 1:1 comparison — probe={Path(probe_path).name} candidate={Path(candidate_path).name}")

    probe = run_pipeline_single_image(probe_path, adapter, role="probe", request_id=request_id)
    candidate = run_pipeline_single_image(candidate_path, adapter, role="candidate", request_id=request_id)

    morphological = compare_morphology(probe["discrete"], candidate["discrete"], discrete_cfg)

    sim = embedding_similarity(
        probe["embedding"].vector, candidate["embedding"].vector,
        threshold=tech_cfg["similarity_threshold"],
    )
    sim["soft_feature_similarity"] = soft_feature_similarity(
        probe["continuous"], candidate["continuous"],
        probe.get("discrete"), candidate.get("discrete"),
    )

    limitations = (
        derive_limitations(probe["quality_metrics"]) +
        derive_limitations(candidate["quality_metrics"])
    )

    report_id = str(uuid.uuid4())
    now = datetime.now(tz=timezone.utc).isoformat()

    report = {
        "administrative": {
            "report_id": report_id,
            "case_id": case_id,
            "requesting_agency": requesting_agency,
            "schema_version": "1.0.0",
            "created_at": now,
            "software_version": "nextgen-forensics-scaffold-0.1.0",
        },
        "evidence": {
            "probe_image_id": Path(probe_path).name,
            "probe_image_hash": _file_hash(probe_path),
            "candidate_image_id": Path(candidate_path).name,
            "candidate_image_hash": _file_hash(candidate_path),
            "chain_of_custody_notes": "",
        },
        "image_metadata": {
            "filename": Path(probe_path).name,
            "width": probe["width"],
            "height": probe["height"],
            "format": Path(probe_path).suffix.lstrip("."),
            "capture_date": None,
        },
        "quality_metrics": probe["quality_metrics"],
        "face_detection": {
            "bounding_box": probe["detection"].bbox,
            "detection_confidence": probe["detection"].detection_confidence,
            "model_name": tech_cfg["detection_model"],
            "model_version": "n/a",
        },
        "landmarks": {
            "schema": probe["detection"].landmark_schema,
            "model_name": tech_cfg["landmark_model"],
            "model_version": "n/a",
            "points": probe["detection"].landmarks,
        },
        "measurements_continuous": probe["continuous"],
        "measurements_discrete": probe["discrete"],
        "morphological_features": morphological,
        "embedding_metrics": {
            "model_name": probe["embedding"].model_name,
            "model_version": probe["embedding"].model_version,
            "embedding_dimension": int(probe["embedding"].vector.shape[0]),
            "vector_hash_probe": hashlib.sha256(probe["embedding"].vector.tobytes()).hexdigest()[:16],
            "vector_hash_candidate": hashlib.sha256(candidate["embedding"].vector.tobytes()).hexdigest()[:16],
        },
        "similarity_metrics": sim,
        "candidate_ranking": [
            {"candidate_id": Path(candidate_path).name, "score": sim["cosine_similarity"], "rank": 1}
        ],
        "explainable_ai": None,
        "limitations": limitations,
        "technical_info": {
            "detection_model": tech_cfg["detection_model"],
            "recognition_model": tech_cfg["recognition_model"],
            "embedding_model": tech_cfg["embedding_model"],
            "similarity_algorithm": tech_cfg["similarity_algorithm"],
            "landmark_model": tech_cfg["landmark_model"],
            "quality_model": tech_cfg["quality_model"],
            "hardware": "n/a (scaffold demo)",
            "software_version": "nextgen-forensics-scaffold-0.1.0",
            "inference_time_ms": (
                probe["embedding"].inference_time_ms + candidate["embedding"].inference_time_ms
            ),
            "operating_system": "n/a",
            "gpu": "n/a",
            "timestamp": now,
            "model_hashes": {},
        },
        "audit": {"stages": probe["audit_stages"] + candidate["audit_stages"]},
        # Private fields consumed by visualization layer only (not serialised to JSON)
        "_probe": probe,
        "_candidate": candidate,
    }

    core_report = {k: v for k, v in report.items() if not k.startswith("_")}
    assert_valid_report(core_report)
    log.info("pipeline", f"Report built — id={report_id} similarity={sim['cosine_similarity']:.4f}")

    return report


# ---------------------------------------------------------------------------
# 1:N Candidate Ranking
# ---------------------------------------------------------------------------

def build_candidate_ranking(
    probe_path: str,
    candidate_paths: list[str],
    case_id: str = "",
    top_k: int = 20,
) -> dict:
    """
    Compares a probe image against an ordered list of candidate images and
    returns a ranked results list sorted by cosine similarity (descending).

    Parameters
    ----------
    probe_path      : path to the probe image
    candidate_paths : list of paths to candidate images
    case_id         : optional case identifier
    top_k           : maximum number of ranked results to return

    Returns
    -------
    dict with keys:
        probe_id          — filename of probe image
        probe_image_hash  — SHA-256 (first 16 hex chars)
        case_id           — forwarded case_id
        total_candidates  — number of candidates compared
        rankings          — list[dict] sorted by score descending:
                            {rank, candidate_id, score, decision_category}
        audit             — audit stages for probe embedding only
    """
    request_id = str(uuid.uuid4())[:8]
    log = _PipelineLogger(request_id)
    adapter = RecognitionAdapter()
    tech_cfg = _load_yaml("technical_info.yaml")
    threshold = tech_cfg["similarity_threshold"]

    log.info("ranking", f"1:N probe={Path(probe_path).name} against {len(candidate_paths)} candidates")

    # Embed the probe once
    probe = run_pipeline_single_image(probe_path, adapter, role="probe", request_id=request_id)
    probe_vec = probe["embedding"].vector

    rankings = []
    for cand_path in candidate_paths[:top_k * 5]:  # process at most 5x top_k to bound runtime
        try:
            cand = run_pipeline_single_image(cand_path, adapter, role="candidate", request_id=request_id)
            sim = embedding_similarity(probe_vec, cand["embedding"].vector, threshold=threshold)
            rankings.append({
                "candidate_id": Path(cand_path).name,
                "candidate_path": cand_path,
                "score": sim["cosine_similarity"],
                "decision_category": sim["decision_category"],
            })
        except Exception as exc:
            log.warning("ranking", f"Skipping {Path(cand_path).name}: {exc}")
            rankings.append({
                "candidate_id": Path(cand_path).name,
                "candidate_path": cand_path,
                "score": None,
                "decision_category": "error",
            })

    # Sort by score descending (None scores go last)
    rankings.sort(key=lambda r: (r["score"] is None, -(r["score"] or 0)))

    # Assign final ranks
    for i, r in enumerate(rankings[:top_k], start=1):
        r["rank"] = i

    log.info("ranking", f"Top result: {rankings[0] if rankings else 'none'}")

    return {
        "probe_id": Path(probe_path).name,
        "probe_image_hash": _file_hash(probe_path),
        "case_id": case_id,
        "total_candidates": len(candidate_paths),
        "rankings": rankings[:top_k],
        "audit": {"stages": probe["audit_stages"]},
    }
