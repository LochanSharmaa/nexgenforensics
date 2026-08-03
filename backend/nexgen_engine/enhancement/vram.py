"""Device selection and VRAM accounting for a card that cannot hold everything.

The development GPU is an RTX A3000 Laptop with 6 GB. The recogniser already
lives on it. A restorer and the recogniser resident at the same time is the
default way this module would fail, and it fails badly: CUDA OOM mid-batch after
several minutes of work.

So the execution model here is strictly sequential and explicit:

    load(device) -> apply() -> release() -> [now the recogniser may load]

``release()`` is not advisory. ``torch.cuda.empty_cache()`` alone does not free
memory still referenced by a live module, so the runner drops the module first
and empties the cache second, in that order.

The device probe follows the same discipline as nexgen_engine/runtime.py: a
capability is confirmed by performing an actual allocation, never by trusting a
build-time flag. That comment exists in runtime.py because reporting a GPU that
silently ran on CPU is a failure this project has already had once.
"""

from __future__ import annotations

import gc
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

logger = logging.getLogger(__name__)

_probe_cache: tuple[bool, str] | None = None


def torch_module() -> Any | None:
    try:
        import torch  # noqa: PLC0415

        return torch
    except Exception:  # pragma: no cover - host-specific
        return None


def _probe_cuda() -> tuple[bool, str]:
    """Confirm CUDA by allocating on it, not by asking whether it exists."""
    torch = torch_module()
    if torch is None:
        return False, "torch is not installed"
    try:
        if not torch.cuda.is_available():
            return False, "torch reports no CUDA device"
        probe = torch.zeros(64, 64, device="cuda")
        # Force a real kernel launch: is_available() has been observed to be
        # true on hosts where the first allocation then fails.
        _ = float((probe * 2).sum().item())
        del probe
        torch.cuda.empty_cache()
        return True, ""
    except Exception as exc:  # pragma: no cover - host-specific
        return False, f"CUDA allocation probe failed: {exc}"


def cuda_binds() -> tuple[bool, str]:
    global _probe_cache
    if _probe_cache is None:
        _probe_cache = _probe_cuda()
        ok, reason = _probe_cache
        logger.info("enhancement CUDA probe: %s%s", "bound" if ok else "unavailable", f" ({reason})" if reason else "")
    return _probe_cache


def resolve_device(requested: str = "auto") -> tuple[str, str]:
    """``(effective_device, note)``. ``auto`` uses CUDA only when it truly binds."""
    if requested == "cpu":
        return "cpu", ""
    ok, reason = cuda_binds()
    if requested == "cuda":
        if ok:
            return "cuda", ""
        return "cpu", f"cuda requested but unavailable ({reason}); running on cpu"
    if ok:
        return "cuda", ""
    return "cpu", f"auto resolved to cpu ({reason})"


def device_report() -> dict[str, Any]:
    """What the host can actually do, for /status and for the run record."""
    torch = torch_module()
    ok, reason = cuda_binds()
    report: dict[str, Any] = {
        "torch_installed": torch is not None,
        "torch_version": getattr(torch, "__version__", None) if torch else None,
        "cuda_available": ok,
        "cuda_unavailable_reason": reason,
        "devices": [],
    }
    if torch is not None and ok:
        try:
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                report["devices"].append(
                    {
                        "index": index,
                        "name": props.name,
                        "total_mb": round(props.total_memory / (1024 * 1024), 1),
                        "compute_capability": f"{props.major}.{props.minor}",
                    }
                )
        except Exception as exc:  # pragma: no cover - host-specific
            report["devices_error"] = str(exc)
    return report


def free_memory(device: str = "cuda") -> None:
    """Release cached device memory. Call AFTER dropping references to modules."""
    gc.collect()
    if device != "cuda":
        return
    torch = torch_module()
    if torch is None:
        return
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    except Exception:  # pragma: no cover - host-specific
        pass


@dataclass
class VramMeasurement:
    """Peak device memory across a measured window, in megabytes."""

    peak_mb: float = 0.0
    device: str = "cpu"


@contextmanager
def measure_vram(device: str) -> Iterator[VramMeasurement]:
    """Measure peak allocation inside the block.

    On CPU this yields zero rather than pretending to measure RSS: process RSS
    is dominated by the interpreter and the loaded recogniser, so reporting it
    as a stage cost would be actively misleading.
    """
    measurement = VramMeasurement(device=device)
    torch = torch_module()
    if device != "cuda" or torch is None:
        yield measurement
        return
    try:
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.max_memory_allocated()
    except Exception:  # pragma: no cover - host-specific
        yield measurement
        return
    try:
        yield measurement
    finally:
        try:
            peak = torch.cuda.max_memory_allocated()
            measurement.peak_mb = max(peak - start, 0) / (1024 * 1024)
        except Exception:  # pragma: no cover - host-specific
            measurement.peak_mb = 0.0


@contextmanager
def loaded(backend: Any, device: str) -> Iterator[Any]:
    """Hold a backend's weights for exactly as long as they are needed.

    Guarantees release on the way out, including on exception. This is the
    contract that lets the S0.3 runner enhance a whole batch, free the card, and
    only then bring the recogniser up.
    """
    backend.load(device)
    try:
        yield backend
    finally:
        try:
            backend.release()
        finally:
            free_memory(device)


def set_deterministic(seed: int = 0) -> None:
    """Pin every source of run-to-run variation we control.

    Honest limit, stated in the module docstring of the runner as well: this
    gives determinism on a fixed host. It does not give bit-identical output
    across GPU architectures or between GPU and CPU, because cuDNN kernel
    selection and floating-point reduction order differ. Canonical reproduction
    is CPU + fp32.
    """
    import random  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    random.seed(seed)
    np.random.seed(seed)
    torch = torch_module()
    if torch is None:
        return
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:  # pragma: no cover - host-specific
        pass


__all__ = [
    "VramMeasurement",
    "cuda_binds",
    "device_report",
    "free_memory",
    "loaded",
    "measure_vram",
    "resolve_device",
    "set_deterministic",
    "torch_module",
]
