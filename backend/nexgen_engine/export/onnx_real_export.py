from __future__ import annotations

from pathlib import Path


def export_minimal_torch_model_to_onnx(output_path: str | Path, opset: int = 18) -> Path:
    try:
        import torch
        from torch import nn
    except Exception as exc:
        raise RuntimeError("PyTorch and ONNX dependencies are required for real ONNX export.") from exc

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    model = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 2)).eval()
    dummy = torch.randn(1, 16)
    torch.onnx.export(
        model,
        dummy,
        str(target),
        opset_version=opset,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    return target
