from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Sequence

# MediaPipe Face Mesh landmark indices. refine_landmarks=True supplies 478 points.
LEFT_EYE_OUTER, LEFT_EYE_INNER = 33, 133
RIGHT_EYE_INNER, RIGHT_EYE_OUTER = 362, 263
NOSE_LEFT, NOSE_RIGHT, NOSE_BASE, NOSE_TOP = 129, 358, 2, 168
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
JAW_LEFT, JAW_RIGHT, FOREHEAD, CHIN = 234, 454, 10, 152


@dataclass(frozen=True)
class Measurement:
    name: str
    value: float
    confidence: float
    unit: str = "ratio"


def extract_face_mesh_landmarks(image_rgb) -> list[tuple[float, float, float]]:
    """Extract the complete refined (478 point) Face Mesh landmark set."""
    try:
        import mediapipe as mp
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("mediapipe is required for image landmark extraction") from exc
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True
    ) as mesh:
        result = mesh.process(image_rgb)
    if not result.multi_face_landmarks:
        raise ValueError("No face mesh detected in image")
    landmarks = result.multi_face_landmarks[0].landmark
    if len(landmarks) < 478:
        raise ValueError("Refined 478-point Face Mesh landmarks are required")
    return [(point.x, point.y, point.z) for point in landmarks]


def pose_confidence(yaw: float, pitch: float) -> float:
    """Return 1.0 through 15 degrees, then linearly reduce to a 0.25 floor at 45."""
    excess = max(0.0, max(abs(float(yaw)), abs(float(pitch))) - 15.0)
    return round(max(0.25, 1.0 - (excess / 30.0) * 0.75), 4)


def compute_measurements(
    points: Sequence[Sequence[float]],
    *,
    pose_yaw: float = 0.0,
    pose_pitch: float = 0.0,
    calibration_reference_present: bool = False,
) -> list[Measurement]:
    """Compute deterministic, scale-invariant measurements from Face Mesh points.

    A calibration marker is recorded in the unit label but no absolute distance is
    inferred: marker size must be independently calibrated before using one.
    """
    if len(points) < 478:
        raise ValueError("Face Mesh output must include 478 landmarks")

    def distance(a: int, b: int) -> float:
        return hypot(float(points[a][0]) - float(points[b][0]), float(points[a][1]) - float(points[b][1]))

    interpupillary = distance(LEFT_EYE_INNER, RIGHT_EYE_INNER)
    facial_width = distance(JAW_LEFT, JAW_RIGHT)
    facial_height = distance(FOREHEAD, CHIN)
    nose_length = distance(NOSE_BASE, NOSE_TOP)
    if min(interpupillary, facial_width, facial_height, nose_length) <= 1e-12:
        raise ValueError("Degenerate landmark geometry")

    confidence = pose_confidence(pose_yaw, pose_pitch)
    unit = "ratio (calibration reference present)" if calibration_reference_present else "ratio"
    midline = (float(points[LEFT_EYE_INNER][0]) + float(points[RIGHT_EYE_INNER][0])) / 2.0
    symmetric_pairs = ((LEFT_EYE_OUTER, RIGHT_EYE_OUTER), (MOUTH_LEFT, MOUTH_RIGHT), (NOSE_LEFT, NOSE_RIGHT), (JAW_LEFT, JAW_RIGHT))
    asymmetry = sum(
        abs(abs(float(points[left][0]) - midline) - abs(float(points[right][0]) - midline)) / interpupillary
        for left, right in symmetric_pairs
    ) / len(symmetric_pairs)
    symmetry = max(0.0, min(1.0, 1.0 - asymmetry))

    return [
        Measurement("interpupillary_distance_ratio", interpupillary / facial_width, confidence, unit),
        Measurement("eye_width_ratio", (distance(LEFT_EYE_OUTER, LEFT_EYE_INNER) + distance(RIGHT_EYE_INNER, RIGHT_EYE_OUTER)) / (2 * interpupillary), confidence, unit),
        Measurement("nose_width_to_length_ratio", distance(NOSE_LEFT, NOSE_RIGHT) / nose_length, confidence, unit),
        Measurement("mouth_width_ratio", distance(MOUTH_LEFT, MOUTH_RIGHT) / facial_width, confidence, unit),
        Measurement("jaw_width_ratio", facial_width / interpupillary, confidence, unit),
        Measurement("facial_width_to_height_ratio", facial_width / facial_height, confidence, unit),
        Measurement("symmetry_score", round(symmetry, 6), confidence, "score"),
    ]
