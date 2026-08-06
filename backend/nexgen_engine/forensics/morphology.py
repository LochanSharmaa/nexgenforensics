"""Dense landmarks, head pose, and the documented measurement set.

WHAT THIS MODULE IS FOR, AND WHAT IT IS NOT FOR
-----------------------------------------------

The photograph examination report carries a micrometry table: face height,
face width, nasal height, nose width, lip width, and three inter-canthal
measurements, one column per exhibit. This module computes those figures.

It does NOT compute an identification, and the numbers it produces must never
be combined into one. Photo-anthropometry -- deriving identity conclusions from
inter-landmark distances measured in 2D photographs -- is explicitly advised
against by FISWG and is not an accepted identification method. A photograph is
a projection: every distance in it varies with yaw, pitch, camera-to-subject
distance and focal length, and for the same person across two images that
variation routinely exceeds the between-person variation of the same index.

So the measurements are reported as *documented observations*, each with a
confidence interval that says how much of the figure is real and how much is
landmark placement noise, and the report rests its conclusion elsewhere: on
morphological comparison graded by the examiner, on facial-mark correspondence,
and on the calibrated likelihood ratio in ``evidence.py``.

Two consequences are load-bearing and should not be optimised away:

1. **Every measurement carries a CI from a jitter bootstrap.** Without it the
   table prints bare numbers and inherits every criticism above. With it, a
   0.014 difference against a 0.014 interval self-evidently reports
   "indistinguishable from placement noise", and the section is defensible
   precisely because it declines to overclaim.

2. **Per-index differences are never summed, averaged, or scored.** They are
   strongly correlated; combining them manufactures evidence.

SCALE
-----

Distances in a photograph are in pixels, and pixels are not comparable between
images of different resolution. Every measurement is therefore divided by the
inter-pupillary distance of the same face, which makes it dimensionless, and
then multiplied by ``NORMALISED_IPD_INCHES`` so the table can print in inches
like the paper form it replaces. The printed figure is a measurement *on a
normalised image*, not an anatomical measurement of the subject, and the report
says so under the table.

LANDMARK INDICES
----------------

The index constants below were established empirically against this exact model
pack, by locating each of the canonical five points within the 106-point set and
by averaging normalised positions over ten faces -- not copied from
documentation. ``landmark_ids`` travels with every measurement so any printed
number can be traced back to the points that produced it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# --- the 106-point map, verified against this model pack ------------------
#
# "Left" and "right" are the VIEWER's, matching image coordinates and matching
# insightface's own five-point ordering. A subject's left cheek appears on the
# right of the image; naming them by the viewer keeps the code and the pixels
# in agreement.

CONTOUR = tuple(range(0, 33))       #: jaw and cheek outline, 0 = gnathion
EYE_LEFT = tuple(range(33, 43))     #: image-left eye
BROW_LEFT = tuple(range(43, 52))    #: image-left eyebrow
MOUTH = tuple(range(52, 72))        #: lips, outer and inner contour
NOSE = tuple(range(72, 87))         #: bridge, tip, alae, base
EYE_RIGHT = tuple(range(87, 97))    #: image-right eye
BROW_RIGHT = tuple(range(97, 106))  #: image-right eyebrow

GNATHION = 0            #: lowest point of the chin
NASION = 72             #: bridge of the nose, between the eyes
PRONASALE = 86          #: nose tip
SUBNASALE = 80          #: base of the nose at the midline
ALARE_LEFT = 77         #: widest point of the image-left nostril wing
ALARE_RIGHT = 83        #: widest point of the image-right nostril wing
CHEILION_LEFT = 52      #: image-left corner of the mouth
CHEILION_RIGHT = 61     #: image-right corner of the mouth
EXOCANTHION_LEFT = 35   #: outer corner, image-left eye
ENDOCANTHION_LEFT = 39  #: inner corner, image-left eye
ENDOCANTHION_RIGHT = 89  #: inner corner, image-right eye
EXOCANTHION_RIGHT = 93  #: outer corner, image-right eye
PUPIL_LEFT = 38         #: image-left eye centre
PUPIL_RIGHT = 88        #: image-right eye centre

#: Every measurement is scaled so the inter-pupillary distance is exactly this
#: many inches, which is what lets the table print in inches at all. One inch
#: is chosen rather than a population mean IPD so that a reader who wants the
#: dimensionless ratio can read it straight off the page.
NORMALISED_IPD_INCHES = 1.0

#: Deterministic perturbations for the bootstrap: (dx, dy, scale), where the
#: translations are FRACTIONS OF THE FACE BOX, not pixels. A fixed pixel offset
#: is a trivial nudge on a 900 px face and a violent one on a 40 px face, which
#: would make the interval a measure of image resolution rather than of
#: landmark stability. As a fraction it probes the same relative crop change on
#: every exhibit, so intervals from different images are comparable -- which is
#: the entire purpose of printing them side by side in one table.
#:
#: Fixed, not sampled, so the same image yields the same interval on every run
#: and an exhibit can be regenerated on challenge.
_JITTER = (
    (0.000, 0.000, 1.00),
    (0.015, 0.000, 1.00), (-0.015, 0.000, 1.00),
    (0.000, 0.015, 1.00), (0.000, -0.015, 1.00),
    (0.000, 0.000, 1.03), (0.000, 0.000, 0.97),
    (0.012, 0.012, 1.02), (-0.012, -0.012, 0.98),
    (0.012, -0.012, 0.98), (-0.012, 0.012, 1.02),
)


class LandmarksUnavailableError(RuntimeError):
    """Raised when landmarks cannot be produced for an image.

    Never substituted with synthesised or approximated points. A renderer that
    catches this must record the section as unavailable and say why -- an
    invented landmark is an invented measurement.
    """


@dataclass(frozen=True)
class HeadPose:
    """Head orientation in degrees, from the fitted 3D landmark model.

    Distinct from ``detection.types.HeadPose``, which is a five-point geometric
    approximation its own docstring flags as unsuitable for reporting. This one
    comes from ``1k3d68`` and is reportable.
    """

    pitch: float
    yaw: float
    roll: float

    @property
    def profile(self) -> str:
        """The template's vocabulary: a frontal face is "Up", a turned one "Side"."""
        return "Up" if abs(self.yaw) <= 15.0 else "Side"

    def as_dict(self) -> dict[str, float]:
        return {"pitch": round(self.pitch, 2), "yaw": round(self.yaw, 2), "roll": round(self.roll, 2)}


@dataclass(frozen=True)
class LandmarkSet:
    points_106: np.ndarray          #: (106, 2) image pixels
    points_68_3d: np.ndarray | None  #: (68, 3), when 1k3d68 ran
    pose: HeadPose
    per_point_sigma: np.ndarray     #: (106,) pixels, from the jitter bootstrap
    bbox: tuple[float, float, float, float]
    det_score: float
    ipd_px: float

    @property
    def mean_sigma_px(self) -> float:
        return float(np.mean(self.per_point_sigma))

    @property
    def mean_sigma_ipd(self) -> float:
        """Placement noise as a fraction of the inter-pupillary distance.

        The comparable figure. Pixels are not: the same photograph downsampled
        by half halves its per-point sigma in pixels while being no more
        reliable, so a table of pixel sigmas would rank a thumbnail above the
        original it came from.
        """
        return float(np.mean(self.per_point_sigma)) / self.ipd_px

    @property
    def roll_corrected(self) -> np.ndarray:
        """The point set rotated so the inter-pupillary axis is horizontal.

        Measurements between two fixed landmarks are already rotation
        invariant, so this exists for the one that is not: maximum face width
        is defined along the horizontal, and in an image with head roll the
        horizontal is not the face's own axis.
        """
        left, right = self.points_106[PUPIL_LEFT], self.points_106[PUPIL_RIGHT]
        angle = float(np.arctan2(right[1] - left[1], right[0] - left[0]))
        cos, sin = float(np.cos(-angle)), float(np.sin(-angle))
        rotation = np.array([[cos, -sin], [sin, cos]], dtype=np.float64)
        centre = (left + right) / 2.0
        return (self.points_106 - centre) @ rotation.T + centre


@dataclass(frozen=True)
class Measurement:
    """One row of the micrometry table, for one exhibit."""

    key: str                        #: "a" .. "h", the letter on the plate
    label: str
    value_in: float                 #: inches at the declared normalisation
    ci95_in: float                  #: half-width of the 95% interval
    ratio_to_ipd: float
    value_px: float                 #: as measured on the source image
    landmark_ids: tuple[int, ...]

    @property
    def display(self) -> str:
        return f"{self.value_in:.3f} ± {self.ci95_in:.3f}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "value_in": round(self.value_in, 4),
            "ci95_in": round(self.ci95_in, 4),
            "ratio_to_ipd": round(self.ratio_to_ipd, 4),
            "value_px": round(self.value_px, 2),
            "landmark_ids": list(self.landmark_ids),
            "display": self.display,
        }


@dataclass(frozen=True)
class FaceMorphology:
    landmarks: LandmarkSet
    measurements: tuple[Measurement, ...]

    def measurement(self, key: str) -> Measurement | None:
        for item in self.measurements:
            if item.key == key:
                return item
        return None


#: The eight rows of the micrometry table, in the order the paper form uses.
#: Each entry is (letter, printed label, resolver). A resolver returns the two
#: point indices whose separation is the measurement, so the definition of every
#: printed number lives in exactly one place and travels with it into the PDF.
def _widest_contour_pair(points: np.ndarray) -> tuple[int, int]:
    """Bizygomatic width: the contour pair with the largest horizontal gap.

    ``points`` must be roll-corrected, or a tilted head yields the widest gap
    across a diagonal of the face rather than across its width. The paper form
    this table replaces prints the measurement as "maximum face width and
    corresponding point" -- it expects the point to vary by exhibit, which is
    why this resolves per face and reports which indices it used.
    """
    xs = points[list(CONTOUR), 0]
    return int(CONTOUR[int(np.argmin(xs))]), int(CONTOUR[int(np.argmax(xs))])


_MEASUREMENTS: tuple[tuple[str, str, Any], ...] = (
    ("a", "Face height, nasion to gnathion", lambda p: (NASION, GNATHION)),
    ("b", "Maximum face width (bizygomatic)", _widest_contour_pair),
    ("c", "Nasal height, nasion to subnasale", lambda p: (NASION, SUBNASALE)),
    ("d", "Nose width, alare to alare", lambda p: (ALARE_LEFT, ALARE_RIGHT)),
    ("e", "Lip width, cheilion to cheilion", lambda p: (CHEILION_LEFT, CHEILION_RIGHT)),
    ("f", "Exocanthion to endocanthion", lambda p: (EXOCANTHION_LEFT, ENDOCANTHION_LEFT)),
    ("g", "Endocanthion to endocanthion", lambda p: (ENDOCANTHION_LEFT, ENDOCANTHION_RIGHT)),
    ("h", "Exocanthion to exocanthion", lambda p: (EXOCANTHION_LEFT, EXOCANTHION_RIGHT)),
)

MEASUREMENT_KEYS = tuple(key for key, _, _ in _MEASUREMENTS)
MEASUREMENT_LABELS = {key: label for key, label, _ in _MEASUREMENTS}


class _FaceStub(dict):
    """Minimal stand-in for insightface's Face, which is a dict with attributes.

    ``Landmark.get`` reads ``face.bbox`` and writes ``face[taskname]``, so a
    bbox-carrying dict is the whole contract. Building one here lets the
    bootstrap perturb the bbox directly, which is exactly the crop perturbation
    the interval is meant to measure: ``Landmark.get`` derives its affine
    transform from the bbox and nothing else.
    """

    def __init__(self, bbox: np.ndarray) -> None:
        super().__init__()
        self["bbox"] = bbox

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _run_landmarker(model: Any, bgr: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    face = _FaceStub(np.asarray(bbox, dtype=np.float32))
    model.get(bgr, face)
    return np.asarray(face[model.taskname], dtype=np.float64)


def extract_landmarks(
    bgr: np.ndarray,
    bbox: tuple[float, float, float, float],
    landmarkers: dict[str, Any],
    det_score: float = 0.0,
) -> LandmarkSet:
    """Dense landmarks, pose, and per-point uncertainty for one detected face.

    ``bgr`` is an OpenCV-ordered array, ``bbox`` the detector's box for the face
    to measure, and ``landmarkers`` the mapping from ``EngineRuntime.landmarkers``.

    The uncertainty is the point of this function. Landmarks are re-extracted
    under each deterministic perturbation in ``_JITTER``; the per-point standard
    deviation across those runs is the placement noise, and it propagates into
    every measurement's confidence interval. On a sharp frontal photograph it is
    a fraction of a pixel; on a compressed low-resolution one it grows, and the
    intervals in the table widen to match without anyone having to decide that
    the image was poor.
    """
    model_106 = landmarkers.get("landmark_2d_106")
    if model_106 is None:
        raise LandmarksUnavailableError("The 106-point landmark model is not loaded.")

    x1, y1, x2, y2 = (float(v) for v in bbox)
    width, height = x2 - x1, y2 - y1
    if width <= 1.0 or height <= 1.0:
        raise LandmarksUnavailableError(f"Face box {bbox} is degenerate.")
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    span = max(width, height)
    runs: list[np.ndarray] = []
    for fx, fy, scale in _JITTER:
        dx, dy = fx * span, fy * span
        half_w, half_h = width * scale / 2.0, height * scale / 2.0
        jittered = np.array(
            [cx + dx - half_w, cy + dy - half_h, cx + dx + half_w, cy + dy + half_h],
            dtype=np.float32,
        )
        try:
            runs.append(_run_landmarker(model_106, bgr, jittered))
        except Exception as error:  # noqa: BLE001 - one bad perturbation must not lose the set
            logger.debug("Landmark jitter run failed at (%s, %s, %s): %s", dx, dy, scale, error)

    if not runs:
        raise LandmarksUnavailableError("Landmark extraction failed on every perturbation.")

    stack = np.stack(runs)                     # (runs, 106, 2)
    points = stack[0]                          # the unperturbed run is the reported one
    # Euclidean spread per point across perturbations, not per-axis, because a
    # measurement is a distance and cares about displacement in any direction.
    sigma = np.sqrt(np.sum(np.var(stack, axis=0), axis=1)) if len(runs) > 1 else np.zeros(len(points))

    points_68: np.ndarray | None = None
    pose = HeadPose(0.0, 0.0, 0.0)
    model_68 = landmarkers.get("landmark_3d_68")
    if model_68 is not None:
        try:
            face = _FaceStub(np.asarray(bbox, dtype=np.float32))
            model_68.get(bgr, face)
            points_68 = np.asarray(face["landmark_3d_68"], dtype=np.float64)
            raw = face.get("pose")
            if raw is not None:
                pose = HeadPose(pitch=float(raw[0]), yaw=float(raw[1]), roll=float(raw[2]))
        except Exception as error:  # noqa: BLE001
            logger.warning("3D landmark/pose extraction failed: %s", error)

    ipd = float(np.linalg.norm(points[PUPIL_LEFT] - points[PUPIL_RIGHT]))
    if ipd <= 1e-6:
        raise LandmarksUnavailableError("Inter-pupillary distance is zero; the face is degenerate.")

    return LandmarkSet(
        points_106=points,
        points_68_3d=points_68,
        pose=pose,
        per_point_sigma=sigma,
        bbox=(x1, y1, x2, y2),
        det_score=float(det_score),
        ipd_px=ipd,
    )


def measure(landmarks: LandmarkSet) -> tuple[Measurement, ...]:
    """The eight table rows for one face, each with its propagated interval.

    The interval combines the placement noise of the two endpoint landmarks and
    of the two pupils, because the inter-pupillary distance is itself measured
    from noisy points and dividing by it propagates its error into every row.
    """
    # Resolved against the roll-corrected set so "maximum width" means width of
    # the face and not width of a tilted bounding region. Distances between
    # fixed pairs are unaffected -- rotation preserves them.
    points = landmarks.roll_corrected
    sigma = landmarks.per_point_sigma
    ipd = landmarks.ipd_px

    # Relative uncertainty of the divisor, propagated into every measurement.
    ipd_sigma = float(np.hypot(sigma[PUPIL_LEFT], sigma[PUPIL_RIGHT]))
    ipd_rel = ipd_sigma / ipd

    out: list[Measurement] = []
    for key, label, resolver in _MEASUREMENTS:
        i, j = resolver(points)
        distance_px = float(np.linalg.norm(points[i] - points[j]))
        if distance_px <= 1e-9:
            continue

        endpoint_sigma = float(np.hypot(sigma[i], sigma[j]))
        # Quadrature of the numerator's and the divisor's relative errors.
        relative = float(np.hypot(endpoint_sigma / distance_px, ipd_rel))
        ratio = distance_px / ipd
        value_in = ratio * NORMALISED_IPD_INCHES

        out.append(
            Measurement(
                key=key,
                label=label,
                value_in=value_in,
                ci95_in=1.96 * relative * value_in,
                ratio_to_ipd=ratio,
                value_px=distance_px,
                landmark_ids=(int(i), int(j)),
            )
        )
    return tuple(out)


def analyse(
    bgr: np.ndarray,
    bbox: tuple[float, float, float, float],
    landmarkers: dict[str, Any],
    det_score: float = 0.0,
) -> FaceMorphology:
    """Landmarks plus the measurement set, for one face in one image."""
    landmarks = extract_landmarks(bgr, bbox, landmarkers, det_score=det_score)
    return FaceMorphology(landmarks=landmarks, measurements=measure(landmarks))


__all__ = [
    "ALARE_LEFT",
    "ALARE_RIGHT",
    "CHEILION_LEFT",
    "CHEILION_RIGHT",
    "CONTOUR",
    "ENDOCANTHION_LEFT",
    "ENDOCANTHION_RIGHT",
    "EXOCANTHION_LEFT",
    "EXOCANTHION_RIGHT",
    "EYE_LEFT",
    "EYE_RIGHT",
    "GNATHION",
    "MEASUREMENT_KEYS",
    "MEASUREMENT_LABELS",
    "NASION",
    "NORMALISED_IPD_INCHES",
    "PRONASALE",
    "PUPIL_LEFT",
    "PUPIL_RIGHT",
    "SUBNASALE",
    "FaceMorphology",
    "HeadPose",
    "LandmarkSet",
    "LandmarksUnavailableError",
    "Measurement",
    "analyse",
    "extract_landmarks",
    "measure",
]
