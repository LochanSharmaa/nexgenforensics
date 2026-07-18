import pytest

from app.services.facial_measurements import compute_measurements, pose_confidence


def fixture_points():
    points = [(0.0, 0.0, 0.0) for _ in range(478)]
    values = {33:(0.20,0.40),133:(0.40,0.40),362:(0.60,0.40),263:(0.80,0.40),129:(0.43,0.55),358:(0.57,0.55),2:(0.50,0.65),168:(0.50,0.45),61:(0.35,0.72),291:(0.65,0.72),234:(0.10,0.55),454:(0.90,0.55),10:(0.50,0.10),152:(0.50,0.90)}
    for index, (x, y) in values.items():
        points[index] = (x, y, 0.0)
    return points


def test_scale_invariant_landmark_measurements():
    measurements = {item.name: item for item in compute_measurements(fixture_points())}
    assert measurements['interpupillary_distance_ratio'].value == pytest.approx(0.25)
    assert measurements['eye_width_ratio'].value == pytest.approx(1.0)
    assert measurements['nose_width_to_length_ratio'].value == pytest.approx(0.7)
    assert measurements['mouth_width_ratio'].value == pytest.approx(0.375)
    assert measurements['jaw_width_ratio'].value == pytest.approx(4.0)
    assert measurements['facial_width_to_height_ratio'].value == pytest.approx(1.0)
    assert measurements['symmetry_score'].value == pytest.approx(1.0)
    assert all(item.confidence == pytest.approx(1.0) for item in measurements.values())


def test_pose_reduces_measurement_confidence_after_fifteen_degrees():
    assert pose_confidence(15, 0) == pytest.approx(1.0)
    assert pose_confidence(30, 0) == 0.625
    assert all(item.confidence == pytest.approx(0.25) for item in compute_measurements(fixture_points(), pose_yaw=60))


