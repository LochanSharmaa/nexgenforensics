"""Enhancement arms for S0.3. The evidence moves toward the hypothesis here.

Every arm already in arms.py transforms the GALLERY toward the probe. That
direction is the organising principle of nexgen_engine/degradation:

    always move the hypothesis toward the evidence,
    never the evidence toward the hypothesis.

An enhancement arm does the opposite. It transforms the PROBE -- the evidence --
toward what the recogniser was trained to expect. degradation/__init__.py
predicts this will fail, in as many words. That prediction is exactly why these
arms exist: it is a testable claim, it is being tested against a baseline that
is already measured, and the decision rule is fixed before the run.

THE CONVENTIONS OF THIS HARNESS, WHICH ARE NOT THE ENHANCEMENT PACKAGE'S
------------------------------------------------------------------------
run_gpu.py decodes with ``cv2.imdecode(..., IMREAD_COLOR)`` and divides by 255,
so an image here is **float32 BGR in 0..1**. The enhancement package works in
**uint8 RGB**. The adapters below convert both ways.

Getting that wrong is silent: BGR fed to an RGB pipeline still produces a
plausible-looking image, with the red and blue channels swapped, and every
learned model then sees a face with blue skin. Nothing raises. The metric just
comes out low, and the conclusion would be "enhancement does not help".

BATCHING IS STAGE-MAJOR, NOT IMAGE-MAJOR
----------------------------------------
The obvious loop -- for each image, run the whole pipeline -- reloads every
model from disk once per image. At 3,000 pairs that is 3,000 checkpoint loads.

So a batch is processed one STAGE at a time: load the backend once, run it over
every image, release the card, move to the next stage. Peak VRAM stays at one
model, and the checkpoint is read once. This is the "enhance -> release VRAM ->
embed" sequencing the 6 GB budget requires, applied inside the enhancement pass
as well as around it.
"""

from __future__ import annotations

import hashlib
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

_BACKEND = Path(__file__).resolve().parents[2] / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from nexgen_engine.enhancement.analysis import analyze  # noqa: E402
from nexgen_engine.enhancement.cache import EnhancementCache  # noqa: E402
from nexgen_engine.enhancement.registry import get_backend  # noqa: E402
from nexgen_engine.enhancement.types import Task, Track, canonical_digest  # noqa: E402
from nexgen_engine.enhancement.vram import (  # noqa: E402
    free_memory,
    measure_vram,
    resolve_device,
    set_deterministic,
)


# --------------------------------------------------------------------------- #
# harness <-> package adapters
# --------------------------------------------------------------------------- #


def to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """float32 BGR 0..1 (harness) -> uint8 RGB HWC (enhancement package)."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.dtype != np.uint8:
        arr = np.clip(np.asarray(arr, dtype=np.float64), 0.0, 1.0) * 255.0
        arr = arr.round().astype(np.uint8)
    return np.ascontiguousarray(arr[:, :, ::-1])


def to_bgr_float(image: np.ndarray) -> np.ndarray:
    """uint8 RGB HWC (package) -> float32 BGR 0..1 (harness)."""
    return np.ascontiguousarray(image[:, :, ::-1].astype(np.float32) / np.float32(255.0))


def image_digest(image: np.ndarray) -> str:
    arr = np.ascontiguousarray(image)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# per-image adaptive parameters
# --------------------------------------------------------------------------- #


def measured_parameters(rgb: np.ndarray) -> dict[str, dict[str, Any]]:
    """Per-image stage parameters derived from measurement, not from constants.

    An arm names a fixed SEQUENCE of stages; the parameters within it are
    adaptive, because a fixed JPEG-quality or blur-sigma constant across a whole
    corpus would be a different (and worse) experiment.
    """
    profile = analyze(rgb)
    return {
        "classical_deblock": {"quality": profile.jpeg_quality or 50, "monochrome": profile.infrared},
        "classical_denoise": {"sigma": profile.noise_sigma, "monochrome": profile.infrared},
        "nafnet_denoise": {"monochrome": profile.infrared},
        "classical_tone": {"monochrome": profile.infrared, "clip_limit": 2.0, "preserve_clipped": True},
        "classical_deblur": {
            "sigma": max(profile.blur_sigma, 0.4),
            "kind": profile.blur_kind,
            "angle_deg": profile.blur_angle_deg,
            "max_gain": 2.0,
            "cutoff": profile.spectral_cutoff,
        },
        "classical_upscale": {"scale": 4 if profile.short_side < 32 else 2, "monochrome": profile.infrared},
        "realesrgan_x4": {"monochrome": profile.infrared},
        "realesrgan_x2": {"monochrome": profile.infrared},
        "codeformer": {"fidelity_weight": 0.7, "monochrome": profile.infrared, "output_size": 512},
        "gfpgan": {"monochrome": profile.infrared, "output_size": 512},
        "_profile": profile.as_dict(),
    }


# --------------------------------------------------------------------------- #
# arms
# --------------------------------------------------------------------------- #


@dataclass
class EnhancementArm:
    """A named, fixed sequence of enhancement stages applied to the probe side."""

    name: str
    stages: tuple[str, ...]
    description: str
    fixed_parameters: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def track(self) -> Track:
        tracks = {get_backend(s).spec.track for s in self.stages if _registered(s)}
        return Track.RECONSTRUCTION if Track.RECONSTRUCTION in tracks else Track.MEASUREMENT

    def availability(self) -> tuple[bool, str]:
        missing = []
        for stage in self.stages:
            if not _registered(stage):
                missing.append(f"{stage} (not registered)")
                continue
            ok, reason = get_backend(stage).availability()
            if not ok:
                missing.append(f"{stage} ({reason})")
        if missing:
            return False, "; ".join(missing)
        return True, ""

    def plan_key(self, per_image: dict[str, dict[str, Any]]) -> str:
        return canonical_digest(
            {
                "arm": self.name,
                "stages": [
                    {"name": s, "parameters": {**per_image.get(s, {}), **self.fixed_parameters.get(s, {})}}
                    for s in self.stages
                ],
            }
        )

    # -- batch execution ---------------------------------------------------

    def transform_batch(
        self,
        probes: list[np.ndarray],
        *,
        device: str = "auto",
        cache: EnhancementCache | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> tuple[list[np.ndarray], dict[str, Any]]:
        """Enhance every probe, one stage at a time. Returns harness-convention images.

        The card holds exactly one model at any moment and is emptied between
        stages, so the embedder that runs afterwards finds the VRAM it needs.
        """
        set_deterministic(0)
        effective, note = resolve_device(device)
        started = time.perf_counter()

        rgb = [to_rgb_uint8(image) for image in probes]
        params = [measured_parameters(image) for image in rgb]
        keys = [self.plan_key(p) for p in params]
        digests = [image_digest(image) for image in rgb]

        # Cache is consulted per image before any weights are touched: a fully
        # cached batch must not load a single checkpoint.
        pending = list(range(len(rgb)))
        hits = 0
        if cache is not None:
            still: list[int] = []
            for index in pending:
                hit = cache.get(digests[index], keys[index])
                if hit is None:
                    still.append(index)
                else:
                    rgb[index] = hit.pixels
            hits = len(pending) - len(still)
            pending = still
            if progress and hits:
                progress(f"  {self.name}: {hits}/{len(rgb)} served from cache")

        stage_stats: list[dict[str, Any]] = []
        for stage in self.stages:
            if not pending:
                break
            if not _registered(stage):
                stage_stats.append({"stage": stage, "skipped": "not registered"})
                continue
            backend = get_backend(stage)
            ok, reason = backend.availability()
            if not ok:
                stage_stats.append({"stage": stage, "skipped": reason})
                continue

            stage_started = time.perf_counter()
            unchanged = 0
            peak = 0.0
            try:
                backend.load(effective)
                for index in pending:
                    merged = {
                        **backend.spec.default_parameters,
                        **params[index].get(stage, {}),
                        **self.fixed_parameters.get(stage, {}),
                    }
                    before = rgb[index]
                    with measure_vram(effective) as vram:
                        after = backend.apply(before, merged)
                    peak = max(peak, vram.peak_mb)
                    if after.shape == before.shape:
                        delta = float(np.abs(after.astype(np.float64) - before.astype(np.float64)).mean())
                        if delta < 0.05:
                            unchanged += 1
                    rgb[index] = after
            finally:
                backend.release()
                free_memory(effective)

            elapsed = (time.perf_counter() - stage_started) * 1000
            record = {
                "stage": stage,
                "track": backend.spec.track.value,
                "task": backend.spec.task.value,
                "images": len(pending),
                "ms_total": round(elapsed, 1),
                "ms_per_image": round(elapsed / max(len(pending), 1), 2),
                "vram_peak_mb": round(peak, 1),
                "unchanged_outputs": unchanged,
            }
            if unchanged:
                # Distinguish the two reasons an output can equal its input.
                #
                # A classical stage no-op is usually CORRECT: deblocking an image
                # with no detectable JPEG lattice should do nothing, and the
                # adaptive parameters are what make it do nothing. Flagging that
                # as a failure would train the reader to ignore the flag.
                #
                # A learned restorer no-op is a FAILURE. It means the model did
                # not act on the crop -- the documented behaviour of every
                # detect-first wrapper handed a pre-cropped surveillance face --
                # and the arm is then not a measurement of that model at all.
                if backend.spec.track is Track.RECONSTRUCTION or backend.spec.task is Task.FACE_RESTORE:
                    record["warning"] = (
                        f"{unchanged}/{len(pending)} outputs were identical to their input. A learned "
                        "restorer that returns its input did not act on those crops; this arm's result "
                        "is not a measurement of that model."
                    )
                else:
                    record["note"] = (
                        f"{unchanged}/{len(pending)} outputs unchanged. For an adaptively parameterised "
                        "classical stage this is normally correct -- the measurement said there was "
                        "nothing to correct."
                    )
            stage_stats.append(record)

        if cache is not None:
            for index in pending:
                cache.put(digests[index], keys[index], rgb[index], {"arm": self.name, "stages": list(self.stages)})

        report = {
            "arm": self.name,
            "description": self.description,
            "track": self.track.value,
            "stages": stage_stats,
            # Recorded explicitly: a fully cached batch runs no stages, and an
            # empty stage list would otherwise read as "nothing was applied"
            # rather than "nothing needed recomputing".
            "cache_hits": hits,
            "computed": len(pending),
            "device": effective,
            "device_note": note,
            "total_ms": round((time.perf_counter() - started) * 1000, 1),
            "example_profile": params[0]["_profile"] if params else {},
        }
        return [to_bgr_float(image) for image in rgb], report


def _registered(name: str) -> bool:
    try:
        get_backend(name)
    except KeyError:
        return False
    return True


# --------------------------------------------------------------------------- #
# the registered arms
# --------------------------------------------------------------------------- #

ENHANCEMENT_ARMS: dict[str, EnhancementArm] = {
    "E0": EnhancementArm(
        name="E0",
        stages=("classical_upscale",),
        description=(
            "CONTROL. Lanczos upscaling only, no restoration. Separates 'the recogniser prefers a "
            "larger input' from 'restoration recovered identity information'. Without this control an "
            "improvement from any other arm is uninterpretable, because the embedder resamples every "
            "input to 112x112 and the resampling path itself changes."
        ),
        fixed_parameters={"classical_upscale": {"scale": 4}},
    ),
    "E1": EnhancementArm(
        name="E1",
        stages=("classical_deblock", "classical_denoise", "classical_tone", "classical_deblur"),
        description=(
            "Track A only: deterministic, classical, no learned prior. Deblock, denoise, tone, and "
            "band-limited sharpening whose correction is projected onto the measured passband, so it "
            "cannot synthesise frequencies the sensor never recorded."
        ),
    ),
    "E2": EnhancementArm(
        name="E2",
        stages=("realesrgan_x4",),
        description="Real-ESRGAN x4 alone. Generative super-resolution, no preprocessing.",
    ),
    "E3": EnhancementArm(
        name="E3",
        stages=("codeformer",),
        description="CodeFormer alone, fidelity weight 0.7 (fidelity-leaning, not the pretty default).",
    ),
    "E4": EnhancementArm(
        name="E4",
        stages=(
            "classical_deblock",
            "classical_denoise",
            "classical_tone",
            "realesrgan_x4",
            "codeformer",
        ),
        description=(
            "The full adaptive pipeline: clean first, then reconstruct. Restoring a cleaned input is "
            "the ordering argued for in the design -- the restorer has to invent less when its input "
            "carries less noise and fewer block artifacts."
        ),
    ),
    "E5": EnhancementArm(
        name="E5",
        stages=("gfpgan",),
        description="GFPGAN v1.4 alone. A different prior from CodeFormer, so disagreement is informative.",
    ),
}


def arm_from_stages(name: str, stages: tuple[str, ...], description: str = "") -> EnhancementArm:
    """Build an ad-hoc arm, for exploring a stage order without editing this file."""
    return EnhancementArm(name=name, stages=stages, description=description or f"ad-hoc: {' -> '.join(stages)}")


__all__ = [
    "ENHANCEMENT_ARMS",
    "EnhancementArm",
    "Task",
    "Track",
    "arm_from_stages",
    "image_digest",
    "measured_parameters",
    "to_bgr_float",
    "to_rgb_uint8",
]
