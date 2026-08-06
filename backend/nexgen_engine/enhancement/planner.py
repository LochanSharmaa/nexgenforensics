"""Choose the stages, in order, and say why -- including why not.

Rule-based and versioned, deliberately not learned. A learned policy cannot
explain its selection, and an explanation is the deliverable here: the report
has to state which algorithms ran, why they were selected, and what was
considered and rejected.

ORDERING IS PHYSICS, NOT PREFERENCE. The pipeline inverts the capture chain:

    deinterlace  first, always -- comb teeth are not detail, and every stage
                 after this one would treat them as if they were
    deblock      before denoise: blocking is *structured*, and a denoiser
                 trained on Gaussian noise preserves block edges while
                 smoothing the real detail around them
    denoise      before deblur: deconvolution divides by a small MTF, which
                 amplifies whatever noise is present along with the signal
    tone         after denoise: stretching contrast first would stretch noise
    deblur       last of the measurement stages
    ---- Track A ends here; everything above adds no detail that was not there
    upscale      Track B
    face restore Track B

Every trigger is gated on the CONFIDENCE of its own measurement, mirroring the
``trust_components`` conjunction in degradation/estimate.py. A stage fired on an
untrustworthy estimate is worse than a stage not fired at all, because its
output looks processed and is wrong.
"""

from __future__ import annotations

from typing import Any

from .analysis import SMALL_FACE_PIXELS, DegradationProfile
from .registry import get_backend
from .types import EnhancementPlan, PlannedStage, Task, Track

RULESET_VERSION = "1.0"

# Thresholds, all named so a report can quote them.
JPEG_QUALITY_FLOOR = 75          # below this, blocking is visible and worth removing
JPEG_CONFIDENCE_FLOOR = 0.15     # the DCT lattice fit must be decisive
NOISE_SIGMA_FLOOR = 0.008        # below this, denoising costs detail and gains nothing
BLUR_SIGMA_FLOOR = 0.8           # below this, deconvolution is noise amplification
BLUR_CONFIDENCE_FLOOR = 0.5      # the spectral fit must actually describe the falloff
CONTRAST_FLOOR = 0.30            # normalised, matches ImageQualityFilter
LUMA_LOW, LUMA_HIGH = 40.0, 200.0
CLIPPED_LOW_FLOOR = 0.10
UPSCALE_TARGET_PIXELS = SMALL_FACE_PIXELS

# First available backend wins. Order encodes the preference argued in the
# design: deterministic and controllable ahead of best-looking.
PREFERENCES: dict[Task, tuple[str, ...]] = {
    Task.DEINTERLACE: ("classical_deinterlace",),
    Task.DEBLOCK: ("classical_deblock",),
    Task.DENOISE: ("nafnet_denoise", "classical_denoise"),
    Task.TONE: ("classical_tone",),
    Task.DEBLUR: ("classical_deblur",),
    Task.UPSCALE: ("realesrgan_x4", "realesrgan_x2", "classical_upscale"),
    Task.FACE_RESTORE: ("codeformer", "gfpgan"),
}


def _pick(task: Task, allow_reconstruction: bool, override: str | None) -> tuple[str | None, str]:
    """First available backend for a task. Returns ``(name, reason_if_none)``."""
    candidates = (override,) if override else PREFERENCES.get(task, ())
    tried: list[str] = []
    for name in candidates:
        if not name:
            continue
        try:
            backend = get_backend(name)
        except KeyError:
            tried.append(f"{name} (not registered)")
            continue
        if backend.spec.track is Track.RECONSTRUCTION and not allow_reconstruction:
            tried.append(f"{name} (reconstruction not enabled for this run)")
            continue
        ok, reason = backend.availability()
        if ok:
            return name, ""
        tried.append(f"{name} ({reason})")
    if not tried:
        return None, "no backend registered for this task"
    return None, "no usable backend: " + "; ".join(tried)


def _stage(
    name: str,
    task: Task,
    parameters: dict[str, Any],
    rationale: str,
) -> PlannedStage:
    backend = get_backend(name)
    merged = {**backend.spec.default_parameters, **parameters}
    return PlannedStage(
        name=name,
        task=task,
        track=backend.spec.track,
        parameters=merged,
        rationale=rationale,
        selected=True,
    )


def _skip(task: Task, rationale: str, skip_reason: str, name: str = "") -> PlannedStage:
    return PlannedStage(
        name=name or f"<{task.value}>",
        task=task,
        track=Track.MEASUREMENT,
        parameters={},
        rationale=rationale,
        selected=False,
        skip_reason=skip_reason,
    )


def plan(
    profile: DegradationProfile,
    *,
    allow_reconstruction: bool = False,
    overrides: dict[str, str] | None = None,
    disabled: set[str] | None = None,
    ruleset_version: str = RULESET_VERSION,
) -> EnhancementPlan:
    """Decide the pipeline for one measured image.

    ``allow_reconstruction`` is the flag that admits Track B. It is False by
    default and the caller has to opt in, per the standing constraint that
    generative stages are off unless explicitly enabled and logged.

    ``disabled`` lets an examiner drop a stage before execution; the drop is
    recorded as a skip with its reason, so an override is visible in the report
    rather than looking like the planner never considered the stage.
    """
    overrides = overrides or {}
    disabled = disabled or set()
    stages: list[PlannedStage] = []

    def consider(
        task: Task,
        condition: bool,
        parameters: dict[str, Any],
        rationale: str,
        not_triggered: str,
    ) -> None:
        if not condition:
            stages.append(_skip(task, rationale, not_triggered))
            return
        name, reason = _pick(task, allow_reconstruction, overrides.get(task.value))
        if name is None:
            stages.append(_skip(task, rationale, reason))
            return
        if name in disabled:
            stages.append(_skip(task, rationale, "disabled by the examiner before execution", name))
            return
        stages.append(_stage(name, task, parameters, rationale))

    # -- 0. interlacing ---------------------------------------------------
    # First, unconditionally first. Comb teeth are high-contrast periodic
    # structure; a denoiser preserves them and a sharpener amplifies them, so
    # every stage below is wrong until this one has run.
    consider(
        Task.DEINTERLACE,
        profile.interlaced,
        {"method": "field_separation"},
        (
            f"Comb ratio {profile.interlace_ratio:.2f} exceeds 1.05, indicating interlaced capture. "
            "Fields are separated rather than blended: the two fields are 1/50 s apart and are two "
            "genuine temporal samples, which blending would discard."
        ),
        f"comb ratio {profile.interlace_ratio:.2f} indicates progressive capture",
    )

    # -- 1. compression ---------------------------------------------------
    jpeg_decisive = profile.jpeg_quality is not None and profile.jpeg_confidence > JPEG_CONFIDENCE_FLOOR
    blocky = jpeg_decisive and profile.jpeg_quality is not None and profile.jpeg_quality < JPEG_QUALITY_FLOOR
    consider(
        Task.DEBLOCK,
        blocky,
        {"quality": profile.jpeg_quality or 50},
        (
            f"Estimated JPEG quality {profile.jpeg_quality} is below {JPEG_QUALITY_FLOOR} with a "
            f"decisive lattice fit (confidence {profile.jpeg_confidence:.2f}), so block edges are "
            "present and would be preserved by every later stage."
        ),
        (
            "compression not decisive: "
            + (
                f"estimated quality {profile.jpeg_quality} is at or above {JPEG_QUALITY_FLOOR}"
                if jpeg_decisive
                else f"lattice fit confidence {profile.jpeg_confidence:.2f} is below {JPEG_CONFIDENCE_FLOOR}, "
                "so the image was probably never JPEG-compressed or was compressed twice"
            )
        ),
    )

    # -- 2. noise ---------------------------------------------------------
    consider(
        Task.DENOISE,
        profile.noise_sigma > NOISE_SIGMA_FLOOR,
        {"sigma": profile.noise_sigma, "monochrome": profile.infrared},
        (
            f"Estimated noise sigma {profile.noise_sigma:.4f} exceeds {NOISE_SIGMA_FLOOR}. Denoising "
            "runs before deblurring because deconvolution amplifies whatever noise is present."
        ),
        f"noise sigma {profile.noise_sigma:.4f} is at or below {NOISE_SIGMA_FLOOR}; denoising would cost detail",
    )

    # -- 3. illumination and contrast -------------------------------------
    needs_tone = (
        profile.contrast < CONTRAST_FLOOR * 64.0
        or not (LUMA_LOW <= profile.mean_luma <= LUMA_HIGH)
        or profile.clipped_low > CLIPPED_LOW_FLOOR
    )
    consider(
        Task.TONE,
        needs_tone,
        {
            # Never colourise an IR frame. The restorers downstream have no way
            # to know the colour is absent rather than merely muted.
            "monochrome": profile.infrared,
            "clip_limit": 2.0,
            "preserve_clipped": True,
        },
        (
            f"Mean luma {profile.mean_luma:.0f} and contrast {profile.contrast:.0f} DN indicate an "
            f"under-exposed or low-contrast capture ({profile.clipped_low * 100:.1f}% of pixels crushed "
            "black). Tone mapping is applied after denoising so noise is not stretched with the signal."
        ),
        f"exposure is within range (luma {profile.mean_luma:.0f}, contrast {profile.contrast:.0f} DN)",
    )

    # -- 4. blur ----------------------------------------------------------
    blur_trusted = profile.blur_confidence > BLUR_CONFIDENCE_FLOOR
    deblur = blur_trusted and profile.blur_sigma > BLUR_SIGMA_FLOOR and profile.blur_kind != "none"
    consider(
        Task.DEBLUR,
        deblur,
        {
            "sigma": profile.blur_sigma,
            "kind": profile.blur_kind,
            "angle_deg": profile.blur_angle_deg,
            # Hard cap. Deconvolution is not permitted to synthesise beyond the
            # band the sensor recorded -- that is what keeps it in Track A.
            "max_gain": 2.0,
            "cutoff": profile.spectral_cutoff,
        },
        (
            f"{profile.blur_kind.capitalize()} blur, sigma {profile.blur_sigma:.2f}, spectral fit "
            f"R2 {profile.blur_confidence:.2f}, anisotropy {profile.blur_anisotropy:.2f}. Correction is "
            f"capped at the measured spectral cut-off ({profile.spectral_cutoff:.3f} cycles/px) so it "
            "cannot restore frequencies the sensor never recorded."
        ),
        (
            f"blur fit confidence R2 {profile.blur_confidence:.2f} is below {BLUR_CONFIDENCE_FLOOR}; "
            "deconvolution on an untrusted kernel estimate amplifies noise without recovering detail"
            if not blur_trusted
            else f"estimated blur sigma {profile.blur_sigma:.2f} is at or below {BLUR_SIGMA_FLOOR}"
        ),
    )

    # -- 5. upscale (Track B) ---------------------------------------------
    too_small = profile.short_side < UPSCALE_TARGET_PIXELS
    consider(
        Task.UPSCALE,
        too_small,
        {"scale": 4 if profile.short_side < 32 else 2, "monochrome": profile.infrared},
        (
            f"Short side is {profile.short_side}px, below the {UPSCALE_TARGET_PIXELS}px floor. "
            "Super-resolution adds no information -- it synthesises plausible detail from a learned "
            "prior -- but it makes the available structure legible for human examination."
        ),
        f"short side {profile.short_side}px already meets the {UPSCALE_TARGET_PIXELS}px floor",
    )

    # -- 6. face restoration (Track B) ------------------------------------
    consider(
        Task.FACE_RESTORE,
        # Match-scale sharpness, not the native-resolution Laplacian: the raw
        # figure shrinks as resolution grows, which read every large sharp
        # frame as "soft" and queued restoration it did not need.
        too_small or profile.sharpness_match_scale < 60.0,
        {
            "fidelity_weight": 0.7,
            "monochrome": profile.infrared,
            # Surveillance probes arrive pre-cropped. Vendor wrappers that
            # detect-and-align first fail outright on a 24px crop, and they fail
            # by returning the input unchanged. See backends/facerestore.py.
            "assume_aligned": True,
        },
        (
            "Face is small or soft, which is where generative restoration is most useful to a human "
            "examiner and most dangerous to a match. Fidelity weight is set to 0.7 (fidelity-leaning) "
            "rather than the visually pleasing default."
        ),
        "face is large and sharp enough that generative restoration would add prior without adding legibility",
    )

    return EnhancementPlan(
        stages=tuple(stages),
        ruleset_version=ruleset_version,
        profile=profile.as_dict(),
    )


__all__ = [
    "BLUR_CONFIDENCE_FLOOR",
    "BLUR_SIGMA_FLOOR",
    "JPEG_CONFIDENCE_FLOOR",
    "JPEG_QUALITY_FLOOR",
    "NOISE_SIGMA_FLOOR",
    "PREFERENCES",
    "RULESET_VERSION",
    "plan",
]
