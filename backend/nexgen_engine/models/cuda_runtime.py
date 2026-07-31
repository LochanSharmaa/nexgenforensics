"""
CUDA runtime discovery and onnxruntime execution-provider assertions.

WHY THIS MODULE EXISTS
----------------------
onnxruntime-gpu fails *silently*. When it cannot load
`onnxruntime_providers_cuda.dll` (missing CUDA 12 / cuDNN 9 DLLs, or a
CUDA-13-built wheel on a CUDA-12 host) it logs a warning to stderr that is
easy to miss, drops `CUDAExecutionProvider` from the session, and runs the
model on CPU at roughly 20x the latency. Nothing raises. Nothing crashes.

Worse, `onnxruntime.get_available_providers()` still reports
`CUDAExecutionProvider` in that state -- it lists providers the wheel was
*built* with, not providers that actually loaded. Any GPU check based on that
call is a false positive. The only trustworthy signal is
`InferenceSession.get_providers()` on a session that has already been created.

This project has lost the GPU configuration twice to exactly that failure
mode, so the rule here is: resolve the DLLs explicitly, then assert on the
*post-construction* session providers and raise if CUDA was expected but CPU
was bound.

ENVIRONMENT FLAGS
-----------------
NEXGEN_FORCE_CPU=1    Skip CUDA entirely and expect CPU. No assertion failure.
NEXGEN_REQUIRE_GPU=1  Hard-require CUDA even if no GPU is detected (CI guard).
"""

from __future__ import annotations

import logging
import os
import site
import sys
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

CUDA_PROVIDER = "CUDAExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"


class GpuBindingError(RuntimeError):
    """Raised when CUDA was expected but onnxruntime bound the CPU provider."""


def _force_cpu() -> bool:
    return os.environ.get("NEXGEN_FORCE_CPU", "").strip().lower() in {"1", "true", "yes"}


def _require_gpu() -> bool:
    return os.environ.get("NEXGEN_REQUIRE_GPU", "").strip().lower() in {"1", "true", "yes"}


def _candidate_dll_dirs() -> list[Path]:
    """Directories that may hold the CUDA 12 / cuDNN 9 runtime DLLs.

    onnxruntime-gpu does not vendor the CUDA runtime; it resolves
    cudart64_12.dll / cublasLt64_12.dll / cudnn64_9.dll through the normal
    Windows DLL search order. In this project those DLLs are supplied by the
    PyTorch cu121 wheel (torch/lib), and optionally by the standalone
    nvidia-*-cu12 wheels (nvidia/<component>/bin).
    """
    dirs: list[Path] = []

    try:
        import torch  # noqa: PLC0415 - intentionally lazy

        dirs.append(Path(torch.__file__).resolve().parent / "lib")
    except Exception:  # pragma: no cover - torch is optional for CPU-only hosts
        pass

    roots: list[str] = []
    try:
        roots.extend(site.getsitepackages())
    except Exception:
        pass
    user_site = getattr(site, "getusersitepackages", None)
    if callable(user_site):
        try:
            roots.append(user_site())
        except Exception:
            pass

    for root in roots:
        nvidia_root = Path(root) / "nvidia"
        if not nvidia_root.is_dir():
            continue
        for component in sorted(nvidia_root.iterdir()):
            bin_dir = component / "bin"
            if bin_dir.is_dir():
                dirs.append(bin_dir)

    return dirs


@lru_cache(maxsize=1)
def prepare_cuda_dlls() -> list[str]:
    """Expose the vendored CUDA DLLs to onnxruntime's loader. Idempotent.

    Both mechanisms are needed on Windows: `os.add_dll_directory` covers
    LoadLibraryEx calls that use LOAD_LIBRARY_SEARCH_* flags, while prepending
    PATH covers the legacy search order that resolves the *transitive*
    dependencies of onnxruntime_providers_cuda.dll.
    """
    if not sys.platform.startswith("win"):
        return []

    added: list[str] = []
    for directory in _candidate_dll_dirs():
        resolved = str(directory)
        try:
            os.add_dll_directory(resolved)
        except (OSError, AttributeError):
            pass
        if resolved not in os.environ.get("PATH", ""):
            os.environ["PATH"] = resolved + os.pathsep + os.environ.get("PATH", "")
        added.append(resolved)

    if added:
        logger.debug("CUDA DLL search paths registered: %s", added)
    return added


@lru_cache(maxsize=1)
def gpu_hardware_present() -> bool:
    """True when a usable CUDA device exists, independent of onnxruntime."""
    prepare_cuda_dlls()
    try:
        import torch  # noqa: PLC0415

        return bool(torch.cuda.is_available())
    except Exception:
        return False


@lru_cache(maxsize=1)
def init_cuda() -> bool:
    """Initialize the CUDA context before any onnxruntime session is built.

    Creating the torch CUDA context first forces the CUDA driver and the
    cu121 runtime DLLs to be loaded into the process. onnxruntime then finds
    an already-initialized runtime instead of trying (and on a cold process,
    sometimes failing) to bootstrap it from its own provider DLL.
    """
    if _force_cpu():
        logger.info("NEXGEN_FORCE_CPU set - skipping CUDA initialization")
        return False

    prepare_cuda_dlls()
    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            return False
        torch.cuda.init()
        logger.info("CUDA initialized: %s", torch.cuda.get_device_name(0))
        return True
    except Exception as exc:
        logger.warning("CUDA initialization failed, continuing on CPU: %s", exc)
        return False


def detect_duplicate_onnxruntime() -> list[str]:
    """Return installed onnxruntime distributions.

    `onnxruntime` and `onnxruntime-gpu` install into the same import
    namespace. With both present the import resolves to whichever wrote the
    files last, which is a silent, machine-dependent CPU fallback.
    """
    from importlib.metadata import distributions  # noqa: PLC0415

    found = set()
    for dist in distributions():
        name = (dist.metadata["Name"] or "").lower()
        if name in {"onnxruntime", "onnxruntime-gpu", "onnxruntime-directml"}:
            found.add(name)
    return sorted(found)


def resolve_providers() -> tuple[list[str], int]:
    """Return (provider_list, ctx_id) for InsightFace `prepare()`.

    ctx_id is InsightFace's device selector: 0 = first GPU, -1 = CPU.
    """
    if _force_cpu():
        return [CPU_PROVIDER], -1

    init_cuda()

    import onnxruntime as ort  # noqa: PLC0415

    # NOTE: this is a build-time capability list, NOT proof CUDA will bind.
    # The real check happens in assert_session_provider() after load.
    if CUDA_PROVIDER in ort.get_available_providers():
        return [CUDA_PROVIDER, CPU_PROVIDER], 0
    return [CPU_PROVIDER], -1


def cuda_expected() -> bool:
    """Whether this host is supposed to be running on GPU."""
    if _force_cpu():
        return False
    if _require_gpu():
        return True
    return gpu_hardware_present()


def session_provider(model_obj: object) -> str | None:
    """Best-effort read of the execution provider an InsightFace model bound.

    InsightFace model wrappers expose the underlying
    onnxruntime.InferenceSession as `.session`.
    """
    session = getattr(model_obj, "session", None)
    if session is None:
        return None
    try:
        providers = session.get_providers()
    except Exception:
        return None
    return providers[0] if providers else None


def assert_face_analysis_providers(app: object, label: str) -> dict[str, str]:
    """Assert every sub-model of a FaceAnalysis app bound the expected provider.

    Returns {model_name: provider}. Raises GpuBindingError when CUDA was
    expected but any sub-model fell back to CPU.
    """
    bound: dict[str, str] = {}
    for name, model_obj in getattr(app, "models", {}).items():
        provider = session_provider(model_obj)
        if provider:
            bound[name] = provider

    if not bound:
        logger.warning("[%s] could not introspect any onnxruntime session", label)
        return bound

    if not cuda_expected():
        logger.info("[%s] CPU mode (expected). providers=%s", label, bound)
        return bound

    on_cpu = sorted(n for n, p in bound.items() if p != CUDA_PROVIDER)
    if on_cpu:
        raise GpuBindingError(
            f"[{label}] CUDA was expected but these sub-models bound "
            f"{CPU_PROVIDER}: {on_cpu}. Full binding map: {bound}.\n"
            "This is the silent-CPU-fallback failure mode. Diagnose with:\n"
            "  python scripts/verify_gpu.py\n"
            "Most likely causes:\n"
            "  1. torch is a +cpu build, so no CUDA 12 DLLs exist in the venv.\n"
            "     Fix: pip install -r backend/requirements-gpu.txt\n"
            "  2. plain `onnxruntime` is installed alongside `onnxruntime-gpu` "
            "and is shadowing it.\n"
            "     Fix: pip uninstall -y onnxruntime\n"
            "  3. onnxruntime-gpu >= 1.22 (CUDA 13 build) on this CUDA 12 host.\n"
            "     Fix: pin onnxruntime-gpu==1.20.1 per backend/requirements-gpu.txt\n"
            "Set NEXGEN_FORCE_CPU=1 to intentionally run on CPU."
        )

    logger.info("[%s] all sub-models on %s", label, CUDA_PROVIDER)
    return bound
