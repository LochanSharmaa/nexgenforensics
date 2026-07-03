from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TorchTrainingConfig:
    model_name: str
    checkpoint_dir: Path
    epochs: int = 1
    batch_size: int = 4
    learning_rate: float = 1e-4
    mixed_precision: bool = False


class TorchTrainingRunner:
    def __init__(self, config: TorchTrainingConfig) -> None:
        self.config = config

    def run_synthetic_smoke(self) -> dict[str, float | int | str]:
        try:
            import torch
            from torch import nn
        except Exception as exc:
            raise RuntimeError("PyTorch is required for the production training runner. Install torch to run this path.") from exc

        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(7)
        model = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 2))
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        final_loss = 0.0
        for _ in range(self.config.epochs):
            inputs = torch.randn(self.config.batch_size, 16)
            labels = torch.arange(self.config.batch_size) % 2
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
        checkpoint = self.config.checkpoint_dir / "synthetic_smoke.pt"
        torch.save({"model": model.state_dict(), "loss": final_loss}, checkpoint)
        return {"steps": self.config.epochs, "loss": final_loss, "checkpoint": str(checkpoint)}
