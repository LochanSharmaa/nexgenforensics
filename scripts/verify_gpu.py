#!/usr/bin/env python
"""
GPU binding smoke test for the NexGen iMATCH recognition engine.

Run this after ANY change to the virtualenv, requirements files, driver, or
CUDA toolkit:

    python scripts/verify_gpu.py

Exit code 0 = every ONNX model is genuinely executing on CUDAExecutionProvider.
Exit code 1 = something fell back to CPU, or the stack is misconfigured.

WHY THIS EXISTS
---------------
onnxruntime-gpu degrades silently. If it cannot load its CUDA provider DLL it
prints an easily-missed warning and runs everything on CPU. Meanwhile
`onnxruntime.get_available_providers()` KEEPS REPORTING CUDAExecutionProvider,
because that call lists providers the wheel was *compiled* with, not providers
that successfully loaded. Any "is the GPU working?" check written against that
function returns a false positive on a broken install.

This script therefore checks the only thing that cannot lie: the providers
reported by a real InferenceSession after it has been constructed.

CHECKS PERFORMED
    1. Exactly one onnxruntime distribution is installed
    2. NVIDIA driver / GPU is visible to torch
    3. torch is a CUDA build (not +cpu) and can allocate on device
    4. A raw onnxruntime session actually binds CUDAExecutionProvider
    5. All 3 InsightFace backbones bind CUDAExecutionProvider
    6. A real inference pass produces a finite, normalized embedding
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

PASS = "PASS"
FAIL = "FAIL"
_results: list[tuple[str, str, str]] = []


def record(name: str, ok: bool, detail: str) -> bool:
    _results.append((PASS if ok else FAIL, name, detail))
    print(f"  [{PASS if ok else FAIL}] {name}: {detail}", flush=True)
    return ok


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}", flush=True)


def check_single_onnxruntime() -> bool:
    section("1. onnxruntime distribution hygiene")
    from nexgen_engine.models.cuda_runtime import detect_duplicate_onnxruntime

    dists = detect_duplicate_onnxruntime()
    if not dists:
        return record("onnxruntime installed", False, "no onnxruntime distribution found")
    if len(dists) > 1:
        return record(
            "single onnxruntime",
            False,
            f"conflicting distributions {dists} share one import namespace; "
            f"uninstall all but onnxruntime-gpu",
        )
    return record("single onnxruntime", True, dists[0])


def check_torch_cuda() -> bool:
    section("2. torch CUDA build")
    from nexgen_engine.models.cuda_runtime import prepare_cuda_dlls

    prepare_cuda_dlls()
    import torch

    ok = record("torch importable", True, f"torch {torch.__version__}")

    if torch.version.cuda is None:
        return record(
            "torch is a CUDA build",
            False,
            f"torch {torch.__version__} is a CPU-only build (torch.version.cuda is None). "
            "It ships no CUDA 12 DLLs, so onnxruntime-gpu cannot bind CUDA either. "
            "Fix: pip install -r backend/requirements-gpu.txt",
        )
    ok &= record("torch is a CUDA build", True, f"compiled against CUDA {torch.version.cuda}")

    if not torch.cuda.is_available():
        return record("torch.cuda.is_available()", False, "no CUDA device visible to torch")
    ok &= record(
        "torch.cuda.is_available()", True, f"{torch.cuda.get_device_name(0)}"
    )

    try:
        x = torch.zeros(64, 64, device="cuda")
        y = (x + 1).sum().item()
        ok &= record("torch device allocation", y == 4096, f"matmul-free sanity sum={y}")
    except Exception as exc:
        ok &= record("torch device allocation", False, f"{exc}")

    return ok


def check_raw_ort_session() -> bool:
    """Build a trivial ONNX graph and confirm CUDA actually binds."""
    section("3. raw onnxruntime CUDA binding")
    from nexgen_engine.models.cuda_runtime import CUDA_PROVIDER, init_cuda

    init_cuda()
    import numpy as np
    import onnxruntime as ort

    record("onnxruntime version", True, ort.__version__)
    record(
        "get_available_providers()",
        True,
        f"{ort.get_available_providers()}  <- build-time list, NOT proof of binding",
    )

    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError:
        return record("raw session", False, "onnx package not installed; cannot build probe graph")

    tmp = Path(os.environ.get("TEMP", ".")) / "nexgen_gpu_probe.onnx"
    inp = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 8])
    out = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8])
    graph = helper.make_graph([helper.make_node("Relu", ["X"], ["Y"])], "probe", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, str(tmp))

    try:
        sess = ort.InferenceSession(
            str(tmp), providers=[CUDA_PROVIDER, "CPUExecutionProvider"]
        )
        bound = sess.get_providers()
        sess.run(None, {"X": np.ones((1, 8), dtype=np.float32)})
    except Exception as exc:
        return record("raw session", False, f"session construction failed: {exc}")
    finally:
        tmp.unlink(missing_ok=True)

    return record(
        "raw session binds CUDA",
        CUDA_PROVIDER in bound,
        f"session.get_providers() = {bound}",
    )


def check_backbones() -> bool:
    section("4. InsightFace backbones (the real test)")
    from nexgen_engine.models.cuda_runtime import (
        CUDA_PROVIDER,
        assert_face_analysis_providers,
        cuda_expected,
    )

    if not cuda_expected():
        return record("cuda expected", False, "NEXGEN_FORCE_CPU set or no GPU detected")

    from nexgen_engine.models.insightface_backbone import InsightFaceEnsembleBackbone

    # Force the full 3-model ensemble regardless of the production fusion
    # setting. Production defaults to single_glintr100 (see BENCHMARKS.md), but
    # this script's job is to prove the CUDA stack works for EVERY backbone --
    # otherwise a GPU regression in an unused model stays hidden until someone
    # switches fusion methods.
    try:
        ensemble = InsightFaceEnsembleBackbone(fusion_method="weighted_avg")
    except Exception as exc:
        record("ensemble load", False, f"{exc}")
        return False

    ok = True
    for label, app in (
        ("buffalo_l", ensemble.buffalo),
        ("antelopev2", ensemble.antelope),
        ("buffalo_s", ensemble.buffalo_s),
    ):
        if app is None:
            ok &= record(label, False, "backbone was not loaded")
            continue
        try:
            bound = assert_face_analysis_providers(app, label)
        except Exception as exc:
            ok &= record(label, False, f"{exc}")
            continue
        all_cuda = bool(bound) and all(p == CUDA_PROVIDER for p in bound.values())
        ok &= record(label, all_cuda, f"{bound}")

    ok &= check_inference(ensemble)
    return ok


def check_inference(ensemble) -> bool:
    section("5. end-to-end inference")
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
    try:
        out = ensemble.encode(img)
    except Exception as exc:
        return record("encode()", False, f"{exc}")

    emb = out.embedding
    norm = float(np.linalg.norm(emb))
    finite = bool(np.all(np.isfinite(emb)))
    return record(
        "encode() output",
        finite and abs(norm - 1.0) < 1e-3,
        f"dim={emb.shape[0]} L2norm={norm:.6f} finite={finite}",
    )


def main() -> int:
    print("=" * 74)
    print("  NexGen iMATCH - GPU EXECUTION PROVIDER VERIFICATION")
    print("=" * 74)

    checks = [
        check_single_onnxruntime,
        check_torch_cuda,
        check_raw_ort_session,
        check_backbones,
    ]
    ok = True
    for check in checks:
        try:
            ok &= bool(check())
        except Exception:
            traceback.print_exc()
            record(check.__name__, False, "raised an unexpected exception")
            ok = False

    section("SUMMARY")
    failures = [r for r in _results if r[0] == FAIL]
    for status, name, detail in _results:
        print(f"  {status:<4} {name}")
    print()
    if failures or not ok:
        print(f"RESULT: FAIL ({len(failures)} failing check(s))")
        print("The engine is NOT running on GPU. See backend/requirements-gpu.txt.")
        return 1
    print(f"RESULT: PASS (all {len(_results)} checks) - engine is genuinely on CUDA")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
