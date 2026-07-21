"""
visualization/figures.py

Deterministic, code-based figure generation. Every figure is a pure
function of (image, backend JSON) — no image-generation models, no
estimated/invented coordinates. Uses OpenCV for drawing.
"""
from __future__ import annotations

import cv2
import numpy as np
import yaml
from pathlib import Path

_CONFIG_DIR = Path(__file__).parent.parent / "config"

COLOR_BOX = (0, 200, 255)
COLOR_LANDMARK = (0, 255, 0)
COLOR_LABEL_BG = (30, 30, 30)
COLOR_MEASURE = (255, 140, 0)
COLOR_SIMILAR = (0, 200, 0)
COLOR_MINOR = (0, 200, 255)
COLOR_DIFF = (0, 0, 255)


def _load_regions(schema: str) -> dict:
    with open(_CONFIG_DIR / "region_definitions.yaml") as f:
        return yaml.safe_load(f)[schema]


def _put_label(img, text, org, color=(255, 255, 255), scale=0.4):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, COLOR_LABEL_BG, 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def fig1_original(image_bgr: np.ndarray) -> np.ndarray:
    return image_bgr.copy()


def fig2_bounding_box(image_bgr: np.ndarray, face_detection: dict) -> np.ndarray:
    out = image_bgr.copy()
    bb = face_detection["bounding_box"]
    x, y, w, h = int(bb["x"]), int(bb["y"]), int(bb["w"]), int(bb["h"])
    cv2.rectangle(out, (x, y), (x + w, y + h), COLOR_BOX, 2)
    _put_label(out, f"conf={face_detection['detection_confidence']:.2f}", (x, max(y - 8, 12)))
    return out


def fig3_landmarks(image_bgr: np.ndarray, landmarks: dict) -> np.ndarray:
    out = image_bgr.copy()
    for p in landmarks["points"]:
        pt = (int(p["x"]), int(p["y"]))
        cv2.circle(out, pt, 2, COLOR_LANDMARK, -1)
        cv2.putText(out, str(p["id"]), (pt[0] + 2, pt[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, COLOR_LANDMARK, 1, cv2.LINE_AA)
    return out


def fig4_measurements(image_bgr: np.ndarray, landmarks: dict, measurements_continuous: list[dict]) -> np.ndarray:
    out = image_bgr.copy()
    pts = {p["id"]: (int(p["x"]), int(p["y"])) for p in landmarks["points"]}
    for m in measurements_continuous:
        ids = m["landmarks_used"]
        if len(ids) != 2 or ids[0] not in pts or ids[1] not in pts:
            continue
        p1, p2 = pts[ids[0]], pts[ids[1]]
        cv2.line(out, p1, p2, COLOR_MEASURE, 1, cv2.LINE_AA)
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        _put_label(out, f"{m['name']}: {m['value']:.1f}{m['unit']}", mid, color=(255, 200, 120), scale=0.35)
    return out


def fig5_morphological_regions(image_bgr: np.ndarray, landmarks: dict, schema: str) -> np.ndarray:
    out = image_bgr.copy()
    regions = _load_regions(schema)
    pts = {p["id"]: (int(p["x"]), int(p["y"])) for p in landmarks["points"]}
    palette = [
        (255, 99, 71), (60, 179, 113), (255, 215, 0), (30, 144, 255),
        (238, 130, 238), (255, 165, 0), (0, 206, 209),
    ]
    overlay = out.copy()
    for i, (region, ids) in enumerate(regions.items()):
        region_pts = np.array([pts[i_] for i_ in ids if i_ in pts], dtype=np.int32)
        if len(region_pts) < 3:
            continue
        hull = cv2.convexHull(region_pts)
        color = palette[i % len(palette)]
        cv2.fillPoly(overlay, [hull], color)
        cx, cy = region_pts.mean(axis=0).astype(int)
        _put_label(out, region, (cx, cy), color=color, scale=0.4)
    out = cv2.addWeighted(overlay, 0.28, out, 0.72, 0)
    return out


def fig6_probe_vs_candidate(probe_img: np.ndarray, candidate_img: np.ndarray,
                             probe_landmarks: dict, candidate_landmarks: dict) -> np.ndarray:
    p = fig3_landmarks(probe_img, probe_landmarks)
    c = fig3_landmarks(candidate_img, candidate_landmarks)
    h = max(p.shape[0], c.shape[0])
    p = cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h))
    c = cv2.resize(c, (int(c.shape[1] * h / c.shape[0]), h))
    divider = np.full((h, 4, 3), 255, dtype=np.uint8)
    combined = np.hstack([p, divider, c])
    _put_label(combined, "PROBE", (10, 20), color=(255, 255, 255), scale=0.6)
    _put_label(combined, "CANDIDATE", (p.shape[1] + 14, 20), color=(255, 255, 255), scale=0.6)
    return combined


def fig7_difference(probe_img: np.ndarray, candidate_img: np.ndarray,
                     probe_landmarks: dict, candidate_landmarks: dict,
                     morphological_features: list[dict], schema: str) -> np.ndarray:
    """Colors each region green/yellow/red per its comparison_label, on the
    candidate image, using region hulls from region_definitions.yaml."""
    out = candidate_img.copy()
    regions = _load_regions(schema)
    pts = {p["id"]: (int(p["x"]), int(p["y"])) for p in candidate_landmarks["points"]}
    color_map = {
        "similar": COLOR_SIMILAR,
        "minor_variation": COLOR_MINOR,
        "observed_variation": COLOR_DIFF,
        "not_assessable": (150, 150, 150),
    }
    overlay = out.copy()
    for feat in morphological_features:
        region = feat["region"]
        if region not in regions:
            continue
        ids = regions[region]
        region_pts = np.array([pts[i_] for i_ in ids if i_ in pts], dtype=np.int32)
        if len(region_pts) < 3:
            continue
        hull = cv2.convexHull(region_pts)
        color = color_map.get(feat["comparison_label"], (150, 150, 150))
        cv2.fillPoly(overlay, [hull], color)
    out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
    legend_y = 20
    for label, color in [("similar", COLOR_SIMILAR), ("minor variation", COLOR_MINOR), ("observed variation", COLOR_DIFF)]:
        cv2.rectangle(out, (10, legend_y - 10), (25, legend_y), color, -1)
        _put_label(out, label, (30, legend_y), scale=0.4)
        legend_y += 18
    return out


def fig8_quality_panel(image_bgr: np.ndarray, quality_metrics: dict) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    panel_w = 260
    panel = np.full((h, panel_w, 3), 25, dtype=np.uint8)
    lines = [
        "IMAGE QUALITY",
        f"Resolution: {quality_metrics['resolution']}",
        f"Blur score: {quality_metrics['blur_score']}",
        f"Brightness: {quality_metrics['brightness']}",
        f"Contrast: {quality_metrics['contrast']}",
        f"Yaw: {quality_metrics['pose']['yaw']} deg",
        f"Pitch: {quality_metrics['pose']['pitch']} deg",
        f"Roll: {quality_metrics['pose']['roll']} deg",
        f"Occlusion score: {quality_metrics['occlusion_score']}",
        f"Face visibility: {quality_metrics['face_visibility_pct']}%",
        f"Detect conf.: {quality_metrics['detection_confidence']}",
    ]
    y = 24
    for i, line in enumerate(lines):
        color = (255, 255, 0) if i == 0 else (230, 230, 230)
        scale = 0.5 if i == 0 else 0.42
        cv2.putText(panel, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += 24
    return np.hstack([image_bgr, panel])
