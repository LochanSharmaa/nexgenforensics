from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class AdministrativeInfo:
    report_id: str
    schema_version: str
    created_at: str
    software_version: str
    case_id: Optional[str] = None
    requesting_agency: Optional[str] = None


@dataclass
class EvidenceSummary:
    probe_image_id: str
    probe_image_hash: str
    candidate_image_id: str
    candidate_image_hash: str
    chain_of_custody_notes: str = ""


@dataclass
class ImageMetadata:
    filename: str
    width: int
    height: int
    format: str
    capture_date: Optional[str] = None


@dataclass
class PoseEstimate:
    yaw: float
    pitch: float
    roll: float


@dataclass
class QualityMetrics:
    resolution: str
    blur_score: float
    noise: Optional[float]
    compression_artifact_score: Optional[float]
    brightness: float
    contrast: float
    lighting_uniformity: Optional[float]
    pose: PoseEstimate
    occlusion_score: Optional[float]
    face_visibility_pct: float
    detection_confidence: float


@dataclass
class FaceDetectionInfo:
    bounding_box: dict  # {"x": ..., "y": ..., "w": ..., "h": ...}
    detection_confidence: float
    model_name: Optional[str] = None
    model_version: Optional[str] = None


@dataclass
class LandmarkPoint:
    id: int
    x: float
    y: float
    confidence: float


@dataclass
class LandmarksInfo:
    schema: str
    points: List[LandmarkPoint]
    model_name: Optional[str] = None
    model_version: Optional[str] = None


@dataclass
class ContinuousMeasurement:
    feature_id: str
    name: str
    region: str
    value: float
    unit: str
    landmarks_used: List[int] = field(default_factory=list)
    formula_ref: Optional[str] = None


@dataclass
class DiscreteMeasurement:
    feature_id: str
    name: str
    region: str
    category: str
    threshold_basis: Optional[str] = None
    landmarks_used: List[int] = field(default_factory=list)


@dataclass
class MorphologicalFeature:
    region: str
    probe_observation: str
    candidate_observation: str
    comparison_label: str
    notes: Optional[str] = None


@dataclass
class EmbeddingMetrics:
    embedding_dimension: int
    vector_hash_probe: str
    vector_hash_candidate: str
    model_name: Optional[str] = None
    model_version: Optional[str] = None


@dataclass
class SoftFeatureSimilarity:
    euclidean: Optional[float] = None
    mahalanobis: Optional[float] = None
    hamming: Optional[float] = None


@dataclass
class SimilarityMetrics:
    cosine_similarity: float
    euclidean_distance: Optional[float]
    angular_distance: Optional[float]
    threshold_used: Optional[float]
    decision_category: str
    candidate_rank: Optional[int] = None
    soft_feature_similarity: Optional[SoftFeatureSimilarity] = None


@dataclass
class CandidateRankEntry:
    candidate_id: str
    score: float
    rank: int


@dataclass
class ExplainableAI:
    grad_cam: Optional[str] = None
    attention_map: Optional[str] = None
    saliency_map: Optional[str] = None
    occlusion_map: Optional[str] = None


@dataclass
class TechnicalInfo:
    detection_model: Optional[str] = None
    recognition_model: Optional[str] = None
    embedding_model: Optional[str] = None
    similarity_algorithm: Optional[str] = None
    landmark_model: Optional[str] = None
    quality_model: Optional[str] = None
    hardware: Optional[str] = None
    software_version: Optional[str] = None
    inference_time_ms: Optional[float] = None
    operating_system: Optional[str] = None
    gpu: Optional[str] = None
    timestamp: Optional[str] = None
    model_hashes: Dict[str, str] = field(default_factory=dict)


@dataclass
class AuditStage:
    stage: str
    started_at: str
    completed_at: str
    status: str
    input_hash: str = ""
    output_hash: str = ""


@dataclass
class AuditLog:
    stages: List[AuditStage] = field(default_factory=list)


@dataclass
class ComparisonReport:
    administrative: AdministrativeInfo
    evidence: EvidenceSummary
    image_metadata: ImageMetadata
    quality_metrics: QualityMetrics
    face_detection: FaceDetectionInfo
    landmarks: LandmarksInfo
    measurements_continuous: List[ContinuousMeasurement]
    measurements_discrete: List[DiscreteMeasurement]
    morphological_features: List[MorphologicalFeature]
    embedding_metrics: EmbeddingMetrics
    similarity_metrics: SimilarityMetrics
    candidate_ranking: List[CandidateRankEntry]
    limitations: List[str]
    technical_info: TechnicalInfo
    audit: AuditLog
    explainable_ai: Optional[ExplainableAI] = None


def from_dict(d: dict) -> ComparisonReport:
    """Parses a raw dictionary into structured report dataclasses."""
    admin = d["administrative"]
    admin_info = AdministrativeInfo(
        report_id=admin["report_id"],
        schema_version=admin["schema_version"],
        created_at=admin["created_at"],
        software_version=admin["software_version"],
        case_id=admin.get("case_id"),
        requesting_agency=admin.get("requesting_agency")
    )

    ev = d["evidence"]
    evidence = EvidenceSummary(
        probe_image_id=ev["probe_image_id"],
        probe_image_hash=ev["probe_image_hash"],
        candidate_image_id=ev["candidate_image_id"],
        candidate_image_hash=ev["candidate_image_hash"],
        chain_of_custody_notes=ev.get("chain_of_custody_notes", "")
    )

    meta = d["image_metadata"]
    image_metadata = ImageMetadata(
        filename=meta["filename"],
        width=meta["width"],
        height=meta["height"],
        format=meta["format"],
        capture_date=meta.get("capture_date")
    )

    qm = d["quality_metrics"]
    pose_raw = qm.get("pose", {"yaw": 0.0, "pitch": 0.0, "roll": 0.0})
    pose = PoseEstimate(
        yaw=pose_raw.get("yaw", 0.0),
        pitch=pose_raw.get("pitch", 0.0),
        roll=pose_raw.get("roll", 0.0)
    )
    quality_metrics = QualityMetrics(
        resolution=qm["resolution"],
        blur_score=qm["blur_score"],
        noise=qm.get("noise"),
        compression_artifact_score=qm.get("compression_artifact_score"),
        brightness=qm["brightness"],
        contrast=qm["contrast"],
        lighting_uniformity=qm.get("lighting_uniformity"),
        pose=pose,
        occlusion_score=qm.get("occlusion_score"),
        face_visibility_pct=qm["face_visibility_pct"],
        detection_confidence=qm["detection_confidence"]
    )

    fd = d["face_detection"]
    face_detection = FaceDetectionInfo(
        bounding_box=fd["bounding_box"],
        detection_confidence=fd["detection_confidence"],
        model_name=fd.get("model_name"),
        model_version=fd.get("model_version")
    )

    lm = d["landmarks"]
    pts = [LandmarkPoint(id=p["id"], x=p["x"], y=p["y"], confidence=p["confidence"]) for p in lm["points"]]
    landmarks = LandmarksInfo(
        schema=lm["schema"],
        points=pts,
        model_name=lm.get("model_name"),
        model_version=lm.get("model_version")
    )

    measurements_continuous = [
        ContinuousMeasurement(
            feature_id=m["feature_id"],
            name=m["name"],
            region=m["region"],
            value=m["value"],
            unit=m["unit"],
            landmarks_used=m.get("landmarks_used", []),
            formula_ref=m.get("formula_ref")
        )
        for m in d["measurements_continuous"]
    ]

    measurements_discrete = [
        DiscreteMeasurement(
            feature_id=m["feature_id"],
            name=m["name"],
            region=m["region"],
            category=m["category"],
            threshold_basis=m.get("threshold_basis"),
            landmarks_used=m.get("landmarks_used", [])
        )
        for m in d["measurements_discrete"]
    ]

    morphological_features = [
        MorphologicalFeature(
            region=m["region"],
            probe_observation=m.get("probe_observation", "not_assessable"),
            candidate_observation=m.get("candidate_observation", "not_assessable"),
            comparison_label=m["comparison_label"],
            notes=m.get("notes")
        )
        for m in d["morphological_features"]
    ]

    emb = d["embedding_metrics"]
    embedding_metrics = EmbeddingMetrics(
        embedding_dimension=emb["embedding_dimension"],
        vector_hash_probe=emb["vector_hash_probe"],
        vector_hash_candidate=emb["vector_hash_candidate"],
        model_name=emb.get("model_name"),
        model_version=emb.get("model_version")
    )

    sim = d["similarity_metrics"]
    soft_sim_raw = sim.get("soft_feature_similarity")
    soft_sim = None
    if soft_sim_raw:
        soft_sim = SoftFeatureSimilarity(
            euclidean=soft_sim_raw.get("euclidean"),
            mahalanobis=soft_sim_raw.get("mahalanobis"),
            hamming=soft_sim_raw.get("hamming")
        )
    similarity_metrics = SimilarityMetrics(
        cosine_similarity=sim["cosine_similarity"],
        euclidean_distance=sim.get("euclidean_distance"),
        angular_distance=sim.get("angular_distance"),
        threshold_used=sim.get("threshold_used"),
        decision_category=sim["decision_category"],
        candidate_rank=sim.get("candidate_rank"),
        soft_feature_similarity=soft_sim
    )

    candidate_ranking = [
        CandidateRankEntry(
            candidate_id=c["candidate_id"],
            score=c["score"],
            rank=c["rank"]
        )
        for c in d.get("candidate_ranking", [])
    ]

    limitations = d["limitations"]

    tech = d["technical_info"]
    technical_info = TechnicalInfo(
        detection_model=tech.get("detection_model"),
        recognition_model=tech.get("recognition_model"),
        embedding_model=tech.get("embedding_model"),
        similarity_algorithm=tech.get("similarity_algorithm"),
        landmark_model=tech.get("landmark_model"),
        quality_model=tech.get("quality_model"),
        hardware=tech.get("hardware"),
        software_version=tech.get("software_version"),
        inference_time_ms=tech.get("inference_time_ms"),
        operating_system=tech.get("operating_system"),
        gpu=tech.get("gpu"),
        timestamp=tech.get("timestamp"),
        model_hashes=tech.get("model_hashes", {})
    )

    stages = [
        AuditStage(
            stage=s["stage"],
            started_at=s["started_at"],
            completed_at=s["completed_at"],
            status=s["status"],
            input_hash=s.get("input_hash", ""),
            output_hash=s.get("output_hash", "")
        )
        for s in d.get("audit", {}).get("stages", [])
    ]
    audit = AuditLog(stages=stages)

    xai = None
    if d.get("explainable_ai"):
        xai_raw = d["explainable_ai"]
        xai = ExplainableAI(
            grad_cam=xai_raw.get("grad_cam"),
            attention_map=xai_raw.get("attention_map"),
            saliency_map=xai_raw.get("saliency_map"),
            occlusion_map=xai_raw.get("occlusion_map")
        )

    return ComparisonReport(
        administrative=admin_info,
        evidence=evidence,
        image_metadata=image_metadata,
        quality_metrics=quality_metrics,
        face_detection=face_detection,
        landmarks=landmarks,
        measurements_continuous=measurements_continuous,
        measurements_discrete=measurements_discrete,
        morphological_features=morphological_features,
        embedding_metrics=embedding_metrics,
        similarity_metrics=similarity_metrics,
        candidate_ranking=candidate_ranking,
        limitations=limitations,
        technical_info=technical_info,
        audit=audit,
        explainable_ai=xai
    )
