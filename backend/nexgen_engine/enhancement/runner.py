"""Execute a plan. One stage at a time, one model resident at a time.

The execution model is sequential on purpose, not for simplicity. The
development card is a 6 GB RTX A3000 that also has to host the recognition
engine. Two restorers resident at once, or a restorer resident while the
recogniser loads, is a CUDA OOM several minutes into a batch.

    for stage in plan:
        load(backend, device)   # weights arrive
        apply()                 # peak VRAM measured across this window
        release()               # weights dropped, cache emptied, card free

Every stage records what it cost and whether it changed anything. That second
part is not bookkeeping: a stage that silently returns its input is the known
failure mode of face restorers on pre-cropped surveillance crops, and it is
indistinguishable from a genuine null result unless something checks.

DETERMINISM, AND ITS HONEST LIMIT. Seeds are pinned, cuDNN is put in
deterministic mode, and no stage uses test-time augmentation. That gives
run-to-run determinism on a fixed host. It does not give bit-identical output
across GPU architectures or between GPU and CPU: cuDNN picks different kernels
and floating-point reduction order differs. Canonical reproduction -- the mode
an examiner uses to regenerate a result exactly -- is CPU, fp32. The device and
library versions are recorded on every run so the mode is never in doubt.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from .analysis import analyze, quality_metrics
from .cache import EnhancementCache
from .planner import plan as build_plan
from .registry import get_backend
from .types import (
    EnhancementOutcome,
    EnhancementPlan,
    OriginalImage,
    ReconstructedImage,
    RestoredImage,
    StageResult,
    Task,
    Track,
)
from .vram import free_memory, measure_vram, resolve_device, set_deterministic

logger = logging.getLogger(__name__)

# Below this mean absolute difference the stage did nothing a viewer could see.
NO_OP_DELTA_DN = 0.05


def _changed(before: np.ndarray, after: np.ndarray) -> tuple[bool, float]:
    if before.shape != after.shape:
        return True, float("nan")
    delta = float(np.abs(after.astype(np.float64) - before.astype(np.float64)).mean())
    return delta >= NO_OP_DELTA_DN, delta


def execute(
    original: OriginalImage,
    plan: EnhancementPlan,
    *,
    device: str = "auto",
    cache: EnhancementCache | None = None,
    seed: int = 0,
) -> EnhancementOutcome:
    """Run a plan against an original. The original is never modified."""
    started = time.perf_counter()
    set_deterministic(seed)
    effective, device_note = resolve_device(device)

    warnings: list[str] = []
    if device_note:
        warnings.append(device_note)

    metrics_before = quality_metrics(original.pixels)
    selected = plan.selected

    # -- cache ------------------------------------------------------------
    if cache is not None:
        hit = cache.get(original.digest, plan.cache_key())
        if hit is not None:
            logger.info("Enhancement cache hit for %s", hit.key[:12])
            track = Track.RECONSTRUCTION if plan.crosses_into_reconstruction else Track.MEASUREMENT
            factory = ReconstructedImage if track is Track.RECONSTRUCTION else RestoredImage
            output = factory.of(
                hit.pixels,
                parent_digest=original.digest,
                stages=tuple(stage.name for stage in selected),
                cached=True,
            )
            stored = hit.metadata.get("stages", [])
            return EnhancementOutcome(
                original=original,
                output=output,
                plan=plan,
                results=(),
                metrics_before=metrics_before,
                metrics_after=hit.metadata.get("metrics_after", quality_metrics(hit.pixels)),
                total_ms=(time.perf_counter() - started) * 1000,
                device=effective,
                warnings=tuple([*warnings, *hit.metadata.get("warnings", []), f"served from cache ({len(stored)} stages)"]),
                cached=True,
            )

    # -- execute ----------------------------------------------------------
    working = original.pixels
    results: list[StageResult] = []

    for planned in selected:
        backend = get_backend(planned.name)
        ok, reason = backend.availability()
        if not ok:
            # Availability is re-checked here as well as in the planner: a
            # weight file can vanish between planning and execution, and a
            # missing stage must degrade the pipeline, never abort it.
            warnings.append(f"{planned.name} became unavailable before execution and was skipped: {reason}")
            continue

        stage_started = time.perf_counter()
        before = working
        try:
            backend.load(effective)
            with measure_vram(effective) as vram:
                produced = backend.apply(before, planned.parameters)
        except Exception as exc:
            logger.exception("Enhancement stage %s failed", planned.name)
            warnings.append(f"{planned.name} failed and was skipped: {exc}")
            continue
        finally:
            try:
                backend.release()
            finally:
                free_memory(effective)

        if produced.dtype != np.uint8 or produced.ndim != 3 or produced.shape[2] != 3:
            warnings.append(
                f"{planned.name} returned {produced.dtype} with shape {produced.shape}; "
                "expected uint8 HxWx3. Stage discarded."
            )
            continue

        changed, delta = _changed(before, produced)
        notes: list[str] = []
        if not changed:
            # A classical stage doing nothing is usually correct -- its
            # parameters are adaptive, so "nothing to correct" is a valid
            # measurement outcome. A learned restorer doing nothing is a
            # failure: it did not act, which on a pre-cropped surveillance face
            # is the documented behaviour of a detect-first wrapper.
            if planned.track is Track.RECONSTRUCTION or planned.task is Task.FACE_RESTORE:
                note = (
                    f"{planned.name} returned its input unchanged (mean |delta| {delta:.4f} DN). A learned "
                    "restorer that returns its input did not act on this image -- the signature of a "
                    "wrapper that failed to find a face in a pre-cropped surveillance crop. This is a "
                    "failure, not a null result."
                )
                warnings.append(note)
            else:
                note = (
                    f"{planned.name} left the image unchanged (mean |delta| {delta:.4f} DN); the measured "
                    "parameters indicated there was nothing for this stage to correct."
                )
            notes.append(note)

        results.append(
            StageResult(
                name=planned.name,
                task=planned.task,
                track=planned.track,
                parameters=dict(planned.parameters),
                rationale=planned.rationale,
                duration_ms=(time.perf_counter() - stage_started) * 1000,
                input_digest="",
                output_digest="",
                changed=changed,
                mean_abs_delta=delta,
                vram_peak_mb=vram.peak_mb,
                device=effective,
                notes=tuple(notes),
            )
        )
        working = produced

    # -- wrap -------------------------------------------------------------
    # The output type follows the plan, not the outcome: if a reconstruction
    # stage was selected, the result is a reconstruction even when that stage
    # later failed, because the operator asked for one and the label must not
    # depend on whether a model happened to load.
    executed_reconstruction = any(r.track is Track.RECONSTRUCTION for r in results)
    track = Track.RECONSTRUCTION if plan.crosses_into_reconstruction else Track.MEASUREMENT
    factory = ReconstructedImage if track is Track.RECONSTRUCTION else RestoredImage
    output = factory.of(
        working,
        parent_digest=original.digest,
        stages=tuple(result.name for result in results),
        ruleset_version=plan.ruleset_version,
        reconstruction_executed=executed_reconstruction,
    )

    metrics_after = quality_metrics(working)

    if cache is not None:
        cache.put(
            original.digest,
            plan.cache_key(),
            working,
            {
                "original_digest": original.digest,
                "plan": plan.as_dict(),
                "stages": [result.as_dict() for result in results],
                "metrics_after": metrics_after,
                "warnings": warnings,
                "device": effective,
                "track": track.value,
            },
        )

    return EnhancementOutcome(
        original=original,
        output=output,
        plan=plan,
        results=tuple(results),
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        total_ms=(time.perf_counter() - started) * 1000,
        device=effective,
        warnings=tuple(warnings),
        cached=False,
    )


def enhance(
    pixels: np.ndarray,
    *,
    source_sha256: str = "",
    allow_reconstruction: bool = False,
    overrides: dict[str, str] | None = None,
    disabled: set[str] | None = None,
    device: str = "auto",
    cache: EnhancementCache | None = None,
    seed: int = 0,
) -> EnhancementOutcome:
    """Analyse, plan and run in one call. The convenience entry point."""
    original = OriginalImage.of(pixels, source_sha256=source_sha256)
    profile = analyze(original.pixels)
    plan = build_plan(
        profile,
        allow_reconstruction=allow_reconstruction,
        overrides=overrides,
        disabled=disabled,
    )
    return execute(original, plan, device=device, cache=cache, seed=seed)


def benchmark_backend(
    name: str,
    pixels: np.ndarray,
    parameters: dict[str, Any] | None = None,
    *,
    device: str = "auto",
    repeats: int = 1,
) -> dict[str, Any]:
    """Measure one backend on one image: runtime, peak VRAM, and whether it acted.

    This is what produces the runtime/VRAM column of the model comparison. It
    loads and releases around the measured window so the reported peak is the
    backend's own cost and not whatever else happened to be resident.
    """
    backend = get_backend(name)
    ok, reason = backend.availability()
    if not ok:
        return {"backend": name, "available": False, "reason": reason}

    effective, note = resolve_device(device)
    merged = {**backend.spec.default_parameters, **(parameters or {})}
    set_deterministic(0)

    durations: list[float] = []
    peak = 0.0
    output: np.ndarray | None = None
    try:
        backend.load(effective)
        for _ in range(max(repeats, 1)):
            started = time.perf_counter()
            with measure_vram(effective) as vram:
                output = backend.apply(pixels, merged)
            durations.append((time.perf_counter() - started) * 1000)
            peak = max(peak, vram.peak_mb)
    finally:
        backend.release()
        free_memory(effective)

    changed, delta = _changed(pixels, output) if output is not None else (False, 0.0)
    return {
        "backend": name,
        "available": True,
        "track": backend.spec.track.value,
        "task": backend.spec.task.value,
        "device": effective,
        "device_note": note,
        "input_shape": list(pixels.shape),
        "output_shape": list(output.shape) if output is not None else None,
        "ms_median": round(float(np.median(durations)), 2) if durations else None,
        "ms_min": round(float(np.min(durations)), 2) if durations else None,
        "vram_peak_mb": round(peak, 1),
        "changed": changed,
        "mean_abs_delta": round(delta, 4) if delta == delta else None,
        "parameters": merged,
    }


__all__ = ["NO_OP_DELTA_DN", "benchmark_backend", "enhance", "execute"]
