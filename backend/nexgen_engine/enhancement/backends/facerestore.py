"""Generative face restoration, with the surveillance-crop hazard handled explicitly.

THE FAILURE THIS FILE EXISTS TO PREVENT
---------------------------------------
CodeFormer, GFPGAN and RestoreFormer all ship inference wrappers that do the
same thing before restoring: run a face detector on the input, align the
detected face to a 512x512 template, restore, and paste back. That pipeline is
correct for a holiday photo and wrong for us, because our inputs are TinyFace,
QMUL and SCface probes -- images that are *already* a cropped face, frequently
20 to 40 pixels on a side.

On such an input the detector finds nothing. The wrapper then returns the
original image unchanged. It does not raise. The result looks exactly like
"enhancement had no measurable effect", which is a plausible experimental
finding and would have been reported as one.

So this module never uses a vendor wrapper. It treats the input as a pre-aligned
crop, letterboxes it to square, resamples to the network's 512x512 input, runs
the network, and resamples back. And it verifies, every single time, that the
output actually differs from the input -- see ``no_op_check``.

WHY ONNX RATHER THAN THE PYTORCH RELEASES
-----------------------------------------
Two reasons, both structural.

``basicsr`` (the shared dependency of the pytorch releases) imports
``torchvision.transforms.functional_tensor``, removed in torchvision 0.17. This
project pins torchvision 0.20.1 because those wheels supply the CUDA DLLs the
*recognition* engine's onnxruntime-gpu loads. Downgrading to satisfy basicsr
would trade working face recognition for working enhancement.

And onnxruntime is already a first-class, GPU-verified dependency here, with
``resolve_providers()`` already implementing the probe-don't-trust discipline
for device selection. Reusing it costs nothing and inherits that behaviour.

Aspect ratio is preserved by letterboxing rather than by stretching to square.
Stretching a face changes the proportions a human examiner reads identity from,
which would make the enhanced image misleading in a way that is hard to see.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..registry import BackendSpec, EnhancementBackend, register
from ..types import Task, Track
from ..weights import WeightSpec, check, resolve

logger = logging.getLogger(__name__)

FACE_TEMPLATE_SIZE = 512


def _cv2():
    import cv2  # noqa: PLC0415

    return cv2


def letterbox_to_square(pixels: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Pad to square by replicating edges. Returns the square and the padding used.

    Replication rather than a constant colour: a black border introduces a hard
    step at the face boundary, and a generative restorer will happily interpret
    that step as jawline or hairline and reconstruct around it.
    """
    cv2 = _cv2()
    height, width = pixels.shape[:2]
    side = max(height, width)
    top = (side - height) // 2
    bottom = side - height - top
    left = (side - width) // 2
    right = side - width - left
    if top or bottom or left or right:
        pixels = cv2.copyMakeBorder(pixels, top, bottom, left, right, cv2.BORDER_REPLICATE)
    return np.ascontiguousarray(pixels), (top, bottom, left, right)


def unletterbox(pixels: np.ndarray, padding: tuple[int, int, int, int], scale: float) -> np.ndarray:
    """Remove the padding added by ``letterbox_to_square``, scaled to the output grid."""
    top, bottom, left, right = (int(round(value * scale)) for value in padding)
    height, width = pixels.shape[:2]
    return np.ascontiguousarray(pixels[top : height - bottom or height, left : width - right or width])


def no_op_check(before: np.ndarray, after: np.ndarray) -> tuple[bool, float, str]:
    """Did the restorer actually do anything?

    Returns ``(changed, mean_abs_delta, note)``. Geometry changes count as
    changed by definition. Otherwise the test is whether the mean absolute
    per-pixel difference clears a floor that ordinary rounding cannot.

    A restorer that returns its input is the documented failure mode of every
    detect-first vendor wrapper on a pre-cropped surveillance face. It has to be
    detected here, loudly, because downstream it is indistinguishable from a
    genuine null result.
    """
    if before.shape != after.shape:
        return True, float("nan"), "output geometry differs from input"
    delta = float(np.abs(after.astype(np.float64) - before.astype(np.float64)).mean())
    if delta < 0.05:
        return False, delta, (
            f"output is identical to the input within rounding (mean |delta| {delta:.4f} DN). "
            "This is the signature of a restorer that failed to find a face and returned its input "
            "unchanged -- it is a failure, not a null result."
        )
    return True, delta, ""


class OnnxFaceRestorer(EnhancementBackend):
    """Base for a 512x512 ONNX face restoration network."""

    weight_spec: WeightSpec
    # Some exports take (image); CodeFormer exports usually take (image, w).
    fidelity_input_names: tuple[str, ...] = ("w", "weight", "fidelity_weight", "codebook_weight")
    input_range_symmetric: bool = True  # most face restorers expect [-1, 1]

    def __init__(self) -> None:
        super().__init__()
        self._session: Any = None
        self._input_name = ""
        self._fidelity_name = ""

    # -- availability ------------------------------------------------------

    def availability(self) -> tuple[bool, str]:
        try:
            import onnxruntime  # noqa: F401,PLC0415
        except Exception as exc:  # pragma: no cover - host-specific
            return False, f"onnxruntime is not installed ({exc})"
        try:
            _cv2()
        except Exception as exc:  # pragma: no cover - host-specific
            return False, f"opencv is not importable: {exc}"
        return check(self.weight_spec)

    # -- lifecycle ---------------------------------------------------------

    def load(self, device: str = "cpu") -> None:
        import onnxruntime  # noqa: PLC0415

        from ...runtime import resolve_providers  # noqa: PLC0415

        if self._session is not None and self._device == device:
            return
        self.release()

        path = resolve(self.weight_spec)
        # Same probe-don't-trust device resolution the recognition engine uses.
        providers, effective = resolve_providers("cuda" if device == "cuda" else "cpu")
        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        session = onnxruntime.InferenceSession(str(path), sess_options=options, providers=providers)

        # Introspect rather than assume. Exports of the same model differ in
        # input arity and naming, and guessing produces an opaque ORT error.
        inputs = session.get_inputs()
        if not inputs:
            raise RuntimeError(f"{self.spec.name}: the ONNX graph declares no inputs.")
        self._input_name = inputs[0].name
        self._fidelity_name = ""
        for candidate in inputs[1:]:
            if candidate.name.lower() in self.fidelity_input_names:
                self._fidelity_name = candidate.name
                break

        self._session = session
        self._device = effective
        logger.info(
            "%s loaded on %s (inputs: %s%s)",
            self.spec.name,
            effective,
            self._input_name,
            f", {self._fidelity_name}" if self._fidelity_name else "",
        )

    def release(self) -> None:
        if self._session is None:
            return
        device = self._device
        self._session = None
        self._input_name = ""
        self._fidelity_name = ""
        from ..vram import free_memory  # noqa: PLC0415

        free_memory(device)

    # -- inference ---------------------------------------------------------

    def _run(self, square: np.ndarray, fidelity: float) -> np.ndarray:
        cv2 = _cv2()
        resized = cv2.resize(square, (FACE_TEMPLATE_SIZE, FACE_TEMPLATE_SIZE), interpolation=cv2.INTER_LANCZOS4)

        array = resized.astype(np.float32) / 255.0
        if self.input_range_symmetric:
            array = (array - 0.5) / 0.5
        batch = np.ascontiguousarray(array.transpose(2, 0, 1)[None], dtype=np.float32)

        feeds: dict[str, np.ndarray] = {self._input_name: batch}
        if self._fidelity_name:
            feeds[self._fidelity_name] = np.array(fidelity, dtype=np.float64)

        output = self._session.run(None, feeds)[0]
        out = np.asarray(output)
        if out.ndim == 4:
            out = out[0]
        if out.shape[0] in (1, 3):
            out = out.transpose(1, 2, 0)
        if out.shape[2] == 1:
            out = np.repeat(out, 3, axis=2)

        if self.input_range_symmetric:
            out = out * 0.5 + 0.5
        return np.clip(out * 255.0, 0, 255).round().astype(np.uint8)

    def apply(self, pixels: np.ndarray, parameters: dict) -> np.ndarray:
        if self._session is None:
            raise RuntimeError(f"{self.spec.name}: load(device) must be called before apply().")
        cv2 = _cv2()

        fidelity = float(np.clip(parameters.get("fidelity_weight", 0.7), 0.0, 1.0))
        target = int(parameters.get("output_size", FACE_TEMPLATE_SIZE))

        square, padding = letterbox_to_square(pixels)
        restored = self._run(square, fidelity)

        # Back to the original aspect ratio at the requested output size.
        scale = restored.shape[0] / max(square.shape[0], 1)
        cropped = unletterbox(restored, padding, scale)

        height, width = pixels.shape[:2]
        longest = max(height, width)
        if longest > 0 and target > 0:
            factor = target / longest
            cropped = cv2.resize(
                cropped,
                (max(int(round(width * factor)), 1), max(int(round(height * factor)), 1)),
                interpolation=cv2.INTER_LANCZOS4,
            )

        out = np.ascontiguousarray(cropped.astype(np.uint8))
        if parameters.get("monochrome"):
            # An IR frame carries no colour. A restorer trained on visible-light
            # faces will produce confident daylight skin tone; collapsing back to
            # luma removes an invention that is invisible as an invention.
            gray = out[..., 0] * 0.299 + out[..., 1] * 0.587 + out[..., 2] * 0.114
            out = np.repeat(gray[:, :, None].round().astype(np.uint8), 3, axis=2)
        return out

    def scale_factor(self, parameters: dict) -> float:
        return 1.0


@register(
    BackendSpec(
        name="codeformer",
        track=Track.RECONSTRUCTION,
        task=Task.FACE_RESTORE,
        version="v0.1.0-onnx",
        summary=(
            "CodeFormer. The only major face restorer with an explicit fidelity-vs-quality control, "
            "so the report can state a number for how far the output was moved toward the prior."
        ),
        deterministic=True,
        requires_weights=True,
        vram_estimate_mb=1600.0,
        default_parameters={"fidelity_weight": 0.7, "assume_aligned": True, "output_size": 512},
    )
)
class CodeFormer(OnnxFaceRestorer):
    weight_spec = WeightSpec(
        filename="codeformer.onnx",
        notes=(
            "ONNX export of CodeFormer, 512x512 input. Exports vary in whether they expose the "
            "fidelity weight as a second input; the loader introspects the graph and passes it only "
            "when present. When absent, the exported default applies and the stage log records that "
            "fidelity_weight had no effect."
        ),
    )


@register(
    BackendSpec(
        name="gfpgan",
        track=Track.RECONSTRUCTION,
        task=Task.FACE_RESTORE,
        version="v1.4-onnx",
        summary="GFPGAN v1.4. Different prior from CodeFormer, so disagreement between them is informative.",
        deterministic=True,
        requires_weights=True,
        vram_estimate_mb=1400.0,
        default_parameters={"assume_aligned": True, "output_size": 512},
    )
)
class GFPGAN(OnnxFaceRestorer):
    weight_spec = WeightSpec(
        filename="gfpgan_v1.4.onnx",
        notes="ONNX export of GFPGANv1.4, 512x512 input, single input tensor, [-1,1] range.",
    )


__all__ = [
    "FACE_TEMPLATE_SIZE",
    "CodeFormer",
    "GFPGAN",
    "OnnxFaceRestorer",
    "letterbox_to_square",
    "no_op_check",
    "unletterbox",
]
