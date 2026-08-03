"""Forensic image enhancement: a pre-processing layer, never an evidence layer.

READ THIS BEFORE ADDING ANYTHING TO THIS PACKAGE.

The sibling package ``degradation/`` states the rule this one has to live
beside: *always move the hypothesis toward the evidence, never the evidence
toward the hypothesis.* Enhancement moves the evidence. That is the forbidden
direction, and this package exists anyway for two reasons that are worth being
explicit about:

  1. An investigator looking at a 30-pixel face needs the best image the data
     can support. That is a human-examination task, it is legitimate, and it is
     the only thing this package claims to do well.

  2. Whether it also helps *recognition* is an empirical question this project
     refuses to answer by assumption. The answer comes from arms registered in
     experiments/S0_3 against a baseline that is already measured.

So the package is built around one invariant, enforced by type rather than by
discipline:

    ORIGINAL EVIDENCE IS NEVER MODIFIED, NEVER OVERWRITTEN, AND NEVER REPLACED
    BY AN ENHANCED IMAGE ANYWHERE IN THE EVIDENTIAL PATH.

Two output types exist and they are not interchangeable:

    RestoredImage       Track A. Recovers or reveals information the sensor
                        actually captured. Deterministic, classical, no learned
                        prior. May be shown as a "processed image".

    ReconstructedImage  Track B. Synthesises detail from a learned prior. On a
                        low-resolution face a large fraction of the output is
                        the generator, not the subject. May be shown ONLY beside
                        its original and ONLY labelled as a reconstruction.

Neither may enter the likelihood/LR path. ``assert_evidential()`` is the guard,
and it takes the original type only.

CPU works for everything here. GPU is used when it binds and is released between
stages -- see vram.py, which exists because the development card has 6 GB and
cannot hold a restorer and the recogniser at the same time.
"""

from __future__ import annotations

from . import backends as _backends  # noqa: F401  (importing registers every backend)
from .analysis import DegradationProfile, analyze, quality_metrics
from .cache import EnhancementCache
from .planner import plan
from .registry import BackendSpec, Track, available_backends, get_backend, register
from .runner import benchmark_backend, enhance, execute
from .types import (
    EnhancementOutcome,
    EnhancementPlan,
    OriginalImage,
    PlannedStage,
    ReconstructedImage,
    RestoredImage,
    StageResult,
    Task,
    assert_evidential,
)

__all__ = [
    "BackendSpec",
    "DegradationProfile",
    "EnhancementCache",
    "EnhancementOutcome",
    "EnhancementPlan",
    "OriginalImage",
    "PlannedStage",
    "ReconstructedImage",
    "RestoredImage",
    "StageResult",
    "Task",
    "Track",
    "analyze",
    "assert_evidential",
    "available_backends",
    "benchmark_backend",
    "enhance",
    "execute",
    "get_backend",
    "plan",
    "quality_metrics",
    "register",
]
