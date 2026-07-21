import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from visualization import figures as fig


def _dummy_image():
    return np.full((200, 200, 3), 200, dtype=np.uint8)


def _dummy_landmarks():
    pts = []
    for i in range(68):
        pts.append({"id": i, "x": 50 + (i % 17) * 5, "y": 50 + (i // 17) * 10, "confidence": 0.9})
    return {"schema": "68", "points": pts}


def test_fig1_returns_same_shape():
    img = _dummy_image()
    out = fig.fig1_original(img)
    assert out.shape == img.shape


def test_fig2_bounding_box_draws_without_error():
    img = _dummy_image()
    detection = {"bounding_box": {"x": 10, "y": 10, "w": 100, "h": 150}, "detection_confidence": 0.9}
    out = fig.fig2_bounding_box(img, detection)
    assert out.shape == img.shape
    assert not np.array_equal(out, img)  # something was drawn


def test_fig3_landmarks_draws_without_error():
    img = _dummy_image()
    out = fig.fig3_landmarks(img, _dummy_landmarks())
    assert not np.array_equal(out, img)


def test_fig4_measurements_handles_missing_landmarks_gracefully():
    img = _dummy_image()
    landmarks = _dummy_landmarks()
    measurements = [{"name": "test", "value": 10.0, "unit": "px", "landmarks_used": [999, 998]}]
    out = fig.fig4_measurements(img, landmarks, measurements)
    assert out.shape == img.shape  # no crash, no line drawn for missing ids
