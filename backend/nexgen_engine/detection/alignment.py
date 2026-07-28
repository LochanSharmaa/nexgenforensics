from __future__ import annotations

import math

import numpy as np
from PIL import Image

from .types import DetectedFace, FaceBox, HeadPose

# Canonical five-point layout ArcFace models are trained against, expressed for
# a 112x112 crop: left eye, right eye, nose tip, left mouth, right mouth.
# Feeding an ArcFace network a crop that is not warped onto this layout costs a
# large amount of accuracy, so this transform is not optional.
ARCFACE_REFERENCE_5PT = np.asarray(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float64,
)

_REFERENCE_SIZE = 112

# |nose - eye_centre| / |mouth_centre - eye_centre| for ARCFACE_REFERENCE_5PT,
# i.e. the value a perfectly frontal face produces. Derived, not assumed.
_NEUTRAL_NOSE_RATIO = 0.495


def umeyama_similarity(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (scale, rotation, translation).

    Implements Umeyama (1991). Returns a 3x3 homogeneous matrix ``T`` such that
    ``target ~= T @ [source, 1]``. Raises ``ValueError`` when the point set is
    degenerate, which happens on collinear or duplicated landmarks.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("Source and target must both be (N, 2) point sets of equal length.")

    count, dim = source.shape
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_demean = source - source_mean
    target_demean = target - target_mean

    covariance = target_demean.T @ source_demean / count
    signs = np.ones(dim, dtype=np.float64)
    if np.linalg.det(covariance) < 0:
        signs[dim - 1] = -1.0

    transform = np.eye(dim + 1, dtype=np.float64)
    u_matrix, singular_values, v_matrix = np.linalg.svd(covariance)
    rank = np.linalg.matrix_rank(covariance)

    if rank == 0:
        raise ValueError("Degenerate landmark configuration: covariance has rank 0.")
    if rank == dim - 1:
        if np.linalg.det(u_matrix) * np.linalg.det(v_matrix) > 0:
            transform[:dim, :dim] = u_matrix @ v_matrix
        else:
            saved = signs[dim - 1]
            signs[dim - 1] = -1.0
            transform[:dim, :dim] = u_matrix @ np.diag(signs) @ v_matrix
            signs[dim - 1] = saved
    else:
        transform[:dim, :dim] = u_matrix @ np.diag(signs) @ v_matrix

    variance = source_demean.var(axis=0).sum()
    if variance <= 1e-9:
        raise ValueError("Degenerate landmark configuration: zero variance.")
    scale = float(singular_values @ signs) / variance

    transform[:dim, dim] = target_mean - scale * (transform[:dim, :dim] @ source_mean)
    transform[:dim, :dim] *= scale
    return transform


def estimate_pose(landmarks: np.ndarray) -> HeadPose:
    """Coarse yaw/pitch/roll from the five-point landmark set.

    Yaw comes from how far the nose sits from the eye midpoint relative to the
    inter-ocular distance; roll from the eye-line angle; pitch from where the
    nose sits vertically between the eye line and the mouth line. Approximate by
    construction -- used for gating, not for reporting as a measured value.
    """
    points = np.asarray(landmarks, dtype=np.float64)
    left_eye, right_eye, nose, left_mouth, right_mouth = points

    eye_center = (left_eye + right_eye) / 2.0
    mouth_center = (left_mouth + right_mouth) / 2.0
    eye_delta = right_eye - left_eye
    interocular = float(np.linalg.norm(eye_delta))
    if interocular < 1e-6:
        return HeadPose()

    roll = math.degrees(math.atan2(float(eye_delta[1]), float(eye_delta[0])))

    # Horizontal nose offset from the eye midpoint, normalized by eye spacing.
    yaw_ratio = float(nose[0] - eye_center[0]) / interocular
    yaw = math.degrees(math.atan(yaw_ratio * 2.0))

    vertical_span = float(np.linalg.norm(mouth_center - eye_center))
    if vertical_span < 1e-6:
        return HeadPose(yaw=round(yaw, 2), pitch=0.0, roll=round(roll, 2))

    # Neutral nose position, measured from ARCFACE_REFERENCE_5PT itself rather
    # than guessed: on that canonical frontal layout the nose sits 0.495 of the
    # way from the eye line to the mouth line. The previous constant of 0.45
    # biased every face toward positive pitch and caused good enrolment images
    # to be rejected as pitch_out_of_range.
    #
    # The gain is deliberately modest. Inter-person variation in nose position
    # is comparable to the variation caused by moderate pitch, so this can
    # separate a strongly tilted head from a level one but should not be read
    # as a measured angle.
    nose_ratio = float(np.linalg.norm(nose - eye_center)) / vertical_span
    pitch = math.degrees(math.atan((nose_ratio - _NEUTRAL_NOSE_RATIO) * 1.5))

    return HeadPose(yaw=round(yaw, 2), pitch=round(pitch, 2), roll=round(roll, 2))


def norm_crop(image: Image.Image, landmarks: np.ndarray, output_size: int = _REFERENCE_SIZE) -> Image.Image:
    """Warp a face onto the canonical ArcFace layout using its landmarks."""
    reference = ARCFACE_REFERENCE_5PT.copy()
    if output_size != _REFERENCE_SIZE:
        reference *= output_size / _REFERENCE_SIZE

    transform = umeyama_similarity(np.asarray(landmarks, dtype=np.float64), reference)

    # PIL's AFFINE transform samples the source using an output->input mapping,
    # so the inverse of the fitted forward transform is what gets passed in.
    inverse = np.linalg.inv(transform)
    coefficients = (
        inverse[0, 0], inverse[0, 1], inverse[0, 2],
        inverse[1, 0], inverse[1, 1], inverse[1, 2],
    )
    return image.convert("RGB").transform(
        (output_size, output_size),
        Image.Transform.AFFINE,
        coefficients,
        resample=Image.Resampling.BICUBIC,
    )


def box_crop(image: Image.Image, box: FaceBox, output_size: int = _REFERENCE_SIZE, margin: float = 0.25) -> Image.Image:
    """Fallback crop for detectors that produce no landmarks.

    Expands the box by ``margin``, squares it off so the face is not distorted,
    and resizes. Accuracy is meaningfully worse than a landmark-aligned crop
    because the network never sees the eye line where it expects it.
    """
    center_x = (box.left + box.right) / 2.0
    center_y = (box.top + box.bottom) / 2.0
    side = max(box.width, box.height, 1) * (1.0 + margin)
    half = max(side / 2.0, 1.0)

    crop = image.convert("RGB").crop(
        (
            int(round(center_x - half)),
            int(round(center_y - half)),
            int(round(center_x + half)),
            int(round(center_y + half)),
        )
    )
    if crop.size != (output_size, output_size):
        crop = crop.resize((output_size, output_size), Image.Resampling.BICUBIC)
    return crop


class FaceAligner:
    """Produces recognition-ready crops from detected faces."""

    def __init__(self, output_size: int = _REFERENCE_SIZE) -> None:
        self.output_size = output_size

    def align(self, image: Image.Image, face: DetectedFace) -> Image.Image:
        if face.has_landmarks:
            try:
                return norm_crop(image, face.landmarks, self.output_size)
            except (ValueError, np.linalg.LinAlgError):
                # Degenerate landmarks: fall through to the box crop rather than
                # failing the entire search.
                pass
        return box_crop(image, face.box.clipped(*image.size), self.output_size)


__all__ = [
    "ARCFACE_REFERENCE_5PT",
    "FaceAligner",
    "box_crop",
    "estimate_pose",
    "norm_crop",
    "umeyama_similarity",
]
