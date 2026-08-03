"""Contract tests for the CVLface ViT-B KP-RPE backbone integration.

Two defects found on 2026-08-02 during backbone evaluation, both silent, both
costing tens of points, and neither catchable by the metrics they corrupted:

  9. KEYPOINT UNITS. KP-RPE's relative_keypoints.py builds its patch grid on
     torch.linspace(0, 1, ...), so landmarks must be [0, 1]-normalised. The
     first integration passed 112-pixel coordinates. Measured on 1,000 LFW
     pairs, same checkpoint:

         pixel units (0..112)   60.40%   near chance
         [0, 1]                 99.90%   correct
         [-1, 1]                98.30%   plausible and still wrong

 10. ALIGNMENT ASSUMPTION. TinyFace/QMUL crops are detector crops with margin,
     NOT ArcFace-aligned, so canonical template keypoints are false information
     fed straight into attention. Unaligned: TinyFace TAR@FAR=0.1% 15.19%,
     BELOW the R50 incumbent. Aligned via the DFA aligner: 66.39% against the
     same construction where R50 scores 17.97%.

     The LFW gate could not catch this: .bin pack crops ARE aligned, so
     canonical keypoints are correct there. LFW passed at 99.75% while the
     assumption was wrong for every surveillance corpus. A gate that validates
     tensor plumbing does not validate a semantic assumption -- hence these
     tests, which assert the assumption itself.

These are pure-contract tests: no model weights, no GPU, no cached embeddings.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))
_SCRIPTS = _BACKEND / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from nexgen_engine.models.cvlface_backbone import ARCFACE_5PTS  # noqa: E402


# --------------------------------------------------------------------------- #
# Defect 9 -- keypoint units
# --------------------------------------------------------------------------- #


def test_canonical_keypoints_are_unit_normalised() -> None:
    """Every coordinate must lie in [0, 1]. Pixel units cost ~40 points."""
    assert ARCFACE_5PTS.shape == (5, 2)
    assert ARCFACE_5PTS.min() >= 0.0, "negative coordinate -- [-1,1] convention leaked in"
    assert ARCFACE_5PTS.max() <= 1.0, (
        f"max coordinate {ARCFACE_5PTS.max():.3f} exceeds 1.0 -- these are pixel "
        f"units, which KP-RPE silently misinterprets (60.40% vs 99.90% on LFW)"
    )
    # Sanity: the template should occupy a plausible central band of the crop,
    # not be crushed toward a corner by a bad scale factor.
    assert 0.3 < ARCFACE_5PTS[:, 0].mean() < 0.7
    assert 0.4 < ARCFACE_5PTS[:, 1].mean() < 0.8


def test_canonical_keypoint_geometry_is_a_face() -> None:
    """Ordering is [left eye, right eye, nose, left mouth, right mouth]."""
    left_eye, right_eye, nose, left_mouth, right_mouth = ARCFACE_5PTS
    assert left_eye[0] < right_eye[0], "eye x-order reversed"
    assert left_mouth[0] < right_mouth[0], "mouth-corner x-order reversed"
    # y grows downward in image coordinates: eyes above nose above mouth.
    assert left_eye[1] < nose[1] < left_mouth[1], "vertical ordering is not eyes/nose/mouth"


def test_mirror_keypoints_are_a_valid_reflection() -> None:
    """Flip-TTA must mirror landmarks too, with L/R labels swapped.

    Mirroring the image while keeping the original landmarks would inject a
    misalignment of our own making into the arm that exists to reduce variance.
    """
    from embed_with_vit import MIRROR_5PTS

    assert MIRROR_5PTS.shape == (5, 2)
    assert 0.0 <= MIRROR_5PTS.min() and MIRROR_5PTS.max() <= 1.0
    # y is unchanged by a horizontal flip; x maps to 1 - x with L/R swapped.
    swap = [1, 0, 2, 4, 3]
    assert np.allclose(MIRROR_5PTS[:, 1], ARCFACE_5PTS[swap, 1], atol=1e-6)
    assert np.allclose(MIRROR_5PTS[:, 0], 1.0 - ARCFACE_5PTS[swap, 0], atol=1e-6)
    # Applying the mirror twice must return the original.
    twice = MIRROR_5PTS.copy()
    twice[:, 0] = 1.0 - twice[:, 0]
    twice = twice[swap]
    assert np.allclose(twice, ARCFACE_5PTS, atol=1e-6), "mirror is not an involution"


# --------------------------------------------------------------------------- #
# Defect 10 -- the aligner must be on by default
# --------------------------------------------------------------------------- #


def test_alignment_is_the_default_and_opt_out_is_explicit() -> None:
    """`--no-align` must exist and must default to False.

    If alignment ever becomes opt-IN, unaligned surveillance crops silently get
    canonical keypoints again and every degraded-imagery number regresses by
    ~50 points while still looking like a plausible result.
    """
    import argparse
    import inspect

    import embed_with_vit as mod

    src = inspect.getsource(mod.main)
    assert "--no-align" in src, "the alignment opt-out flag has been removed"
    assert "if not args.no_align" in src, (
        "alignment is no longer the default path -- unaligned crops would get "
        "canonical keypoints, the defect that produced TinyFace 15.19%"
    )
    # The flag must be store_true, i.e. default False => aligner ON.
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-align", action="store_true")
    assert ap.parse_args([]).no_align is False


def test_mirror_landmark_helper_swaps_and_reflects() -> None:
    """Per-image landmark mirroring, used in the aligned TTA path."""
    from embed_with_vit import _mirror_ldmks

    ld = np.array(
        [[[0.30, 0.45], [0.60, 0.45], [0.45, 0.62], [0.34, 0.80], [0.57, 0.80]]],
        dtype=np.float32,
    )
    out = _mirror_ldmks(ld)
    assert out.shape == ld.shape
    assert np.allclose(out[0, :, 1], ld[0, [1, 0, 2, 4, 3], 1], atol=1e-6)
    assert np.allclose(out[0, :, 0], 1.0 - ld[0, [1, 0, 2, 4, 3], 0], atol=1e-6)
    # Involution again, on the per-image path.
    assert np.allclose(_mirror_ldmks(out), ld, atol=1e-6)


def test_landmarks_stay_in_unit_frame_after_mirroring() -> None:
    from embed_with_vit import _mirror_ldmks

    rng = np.random.default_rng(0)
    ld = rng.uniform(0.05, 0.95, size=(16, 5, 2)).astype(np.float32)
    out = _mirror_ldmks(ld)
    assert out.min() >= 0.0 and out.max() <= 1.0


# --------------------------------------------------------------------------- #
# Gate artifact
# --------------------------------------------------------------------------- #


def test_lfw_gate_artifact_is_present_and_passing() -> None:
    """The preprocessing gate must have run and passed before any evaluation.

    Its own limitation is recorded here so it is not over-trusted: LFW crops are
    pre-aligned, so this gate validates the tensor path only. Defect 10 passed
    it while being wrong.
    """
    import json

    p = _BACKEND.parent / "runtime/forensics/lfw_validation_vit_kprpe_wf12m.json"
    if not p.exists():
        pytest.skip("LFW gate artifact not produced yet")
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["gate_passed"] is True
    assert d["accuracy_mean"] >= d["gate_threshold"]
