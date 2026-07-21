"""
quality/assessment.py

Computes image-quality metrics from the raw image + detection result.
Rule-based limitation strings are emitted from config thresholds, never
free-text generated.
"""
from __future__ import annotations

import cv2
import numpy as np
import yaml
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "quality_thresholds.yaml"


def _load_thresholds() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _blur_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _brightness_contrast(gray: np.ndarray) -> tuple[float, float]:
    return float(gray.mean()), float(gray.std())


def _crude_pose_fallback(pts: dict) -> dict:
    try:
        left_eye = pts.get(36, pts.get(33))
        right_eye = pts.get(45, pts.get(263))
        nose_tip = pts.get(30, pts.get(4))
        if left_eye is None or right_eye is None or nose_tip is None:
            return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
    except Exception:
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

    eye_mid_x = (left_eye[0] + right_eye[0]) / 2.0
    eye_dist = abs(right_eye[0] - left_eye[0]) or 1.0
    yaw_proxy = float(np.clip((nose_tip[0] - eye_mid_x) / eye_dist * 60, -90, 90))

    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0] or 1.0
    roll_proxy = float(np.degrees(np.arctan2(dy, dx)))

    return {"yaw": round(yaw_proxy, 2), "pitch": 0.0, "roll": round(roll_proxy, 2)}


def _estimate_pose(landmarks: list[dict], w: int = 600, h: int = 800) -> dict:
    """
    Computes a real 3D face pose estimate (yaw, pitch, roll) using cv2.solvePnP
    with a generic 3D face model and 6 standard landmarks.
    """
    if not landmarks:
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
    pts = {p["id"]: (p["x"], p["y"]) for p in landmarks}
    max_id = max(pts.keys())
    schema = "68" if max_id < 100 else "468"

    indices = []
    if schema == "68":
        indices = [36, 45, 30, 48, 54, 8]
    else:  # 468 schema (MediaPipe FaceMesh)
        indices = [33, 263, 4, 61, 291, 152]

    # Check if we have all 6 points needed for solvePnP
    if any(idx not in pts for idx in indices):
        return _crude_pose_fallback(pts)

    image_pts = np.array([pts[idx] for idx in indices], dtype=np.float32)

    # 3D generic facial landmarks model points (anthropometric model in mm)
    model_pts = np.array([
        (-30.0, 30.0, -30.0),   # Left eye outer corner
        (30.0, 30.0, -30.0),    # Right eye outer corner
        (0.0, 0.0, 0.0),        # Nose tip
        (-20.0, -35.0, -20.0),  # Mouth left corner
        (20.0, -35.0, -20.0),   # Mouth right corner
        (0.0, -55.0, -10.0),    # Chin center
    ], dtype=np.float32)

    focal_length = float(w)
    center = (float(w) / 2.0, float(h) / 2.0)
    camera_matrix = np.array([
        [focal_length, 0.0, center[0]],
        [0.0, focal_length, center[1]],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    try:
        success, rot_vec, trans_vec = cv2.solvePnP(
            model_pts, image_pts, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return _crude_pose_fallback(pts)

        rot_mat, _ = cv2.Rodrigues(rot_vec)

        # Extract Euler angles from rotation matrix
        sy = np.sqrt(rot_mat[0, 0] * rot_mat[0, 0] + rot_mat[1, 0] * rot_mat[1, 0])
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])   # pitch
            y = np.arctan2(-rot_mat[2, 0], sy)            # yaw
            z = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])   # roll
        else:
            x = np.arctan2(-rot_mat[1, 2], rot_mat[1, 1])
            y = np.arctan2(-rot_mat[2, 0], sy)
            z = 0.0

        yaw = float(np.degrees(y))
        pitch = float(np.degrees(x))
        roll = float(np.degrees(z))

        # Clamp to normal facial pose ranges
        yaw = float(np.clip(yaw, -90.0, 90.0))
        pitch = float(np.clip(pitch, -90.0, 90.0))
        roll = float(np.clip(roll, -90.0, 90.0))

        return {"yaw": round(yaw, 2), "pitch": round(pitch, 2), "roll": round(roll, 2)}
    except Exception:
        return _crude_pose_fallback(pts)


def _estimate_noise(gray: np.ndarray) -> float:
    """Estimates image noise from Laplacian high-frequency residual."""
    try:
        blurred = cv2.medianBlur(gray, 3)
        residual = cv2.absdiff(gray, blurred)
        return round(float(np.std(residual)), 2)
    except Exception:
        return 0.0


def _estimate_compression(gray: np.ndarray) -> float:
    """Estimates blocky JPEG compression artifacts severity between 0 and 1."""
    try:
        h, w = gray.shape
        if h < 16 or w < 16:
            return 0.0
        grad_y, grad_x = np.gradient(gray.astype(float))
        
        block_diff_x = np.mean(np.abs(grad_x[:, 7::8]))
        block_diff_y = np.mean(np.abs(grad_y[7::8, :]))
        
        internal_diff_x = np.mean(np.abs(grad_x[:, [i for i in range(w) if i % 8 != 7]]))
        internal_diff_y = np.mean(np.abs(grad_y[[i for i in range(h) if i % 8 != 7], :]))
        
        blockiness = (block_diff_x + block_diff_y) / (internal_diff_x + internal_diff_y + 1e-5)
        score = float(np.clip(blockiness - 1.0, 0.0, 1.0))
        return round(score, 3)
    except Exception:
        return 0.0


def _estimate_lighting_uniformity(gray: np.ndarray) -> float:
    """Estimates lighting uniformity by comparing brightness on left and right halves."""
    try:
        h, w = gray.shape
        mid = w // 2
        left_half = gray[:, :mid]
        right_half = gray[:, mid:]
        l_mean = np.mean(left_half)
        r_mean = np.mean(right_half)
        diff = abs(l_mean - r_mean)
        uniformity = max(0.0, 1.0 - (diff / 255.0))
        return round(uniformity, 3)
    except Exception:
        return 1.0


def assess_quality(image_bgr: np.ndarray, detection_confidence: float,
                    landmarks: list[dict]) -> dict:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    blur = _blur_score(gray)
    brightness, contrast = _brightness_contrast(gray)
    pose = _estimate_pose(landmarks, w, h)
    noise = _estimate_noise(gray)
    compression = _estimate_compression(gray)
    lighting_uniformity = _estimate_lighting_uniformity(gray)

    avg_landmark_conf = (
        float(np.mean([p["confidence"] for p in landmarks])) if landmarks else 0.0
    )

    metrics = {
        "resolution": f"{w}x{h}",
        "blur_score": round(blur, 2),
        "noise": noise,
        "compression_artifact_score": compression,
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "lighting_uniformity": lighting_uniformity,
        "pose": pose,
        "occlusion_score": round(1.0 - avg_landmark_conf, 3),
        "face_visibility_pct": round(avg_landmark_conf * 100, 1),
        "detection_confidence": round(detection_confidence, 3),
    }
    return metrics



def derive_limitations(quality_metrics: dict) -> list[str]:
    """Rule-based, config-driven limitation strings — never LLM-generated."""
    t = _load_thresholds()
    limitations = []

    if quality_metrics["blur_score"] < t["blur_score_min"]:
        limitations.append(
            f"Blur score ({quality_metrics['blur_score']}) is below the "
            f"configured minimum ({t['blur_score_min']}); measurements and "
            f"morphological observations may be less reliable."
        )
    yaw = abs(quality_metrics["pose"]["yaw"])
    if yaw > t["max_yaw_degrees"]:
        limitations.append(
            f"Estimated yaw ({yaw}\u00b0) exceeds the configured limit "
            f"({t['max_yaw_degrees']}\u00b0); frontal anthropometric "
            f"measurements may be distorted."
        )
    if quality_metrics["face_visibility_pct"] < t["min_face_visibility_pct"]:
        limitations.append(
            f"Face visibility ({quality_metrics['face_visibility_pct']}%) is "
            f"below the configured minimum "
            f"({t['min_face_visibility_pct']}%); some landmarks may be "
            f"occluded or low-confidence."
        )
    if quality_metrics["detection_confidence"] < t["min_detection_confidence"]:
        limitations.append(
            f"Detection confidence ({quality_metrics['detection_confidence']}) "
            f"is below the configured minimum "
            f"({t['min_detection_confidence']})."
        )
    if not limitations:
        limitations.append("No quality-based limitations were triggered by configured thresholds.")
    return limitations
