import os
import sys
import time
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models

sys.path.insert(0, str(Path("backend").resolve()))

from nexgen_engine.training.dataset import MultiDatasetFaceDataset
from nexgen_engine.training.arcface_loss import ArcFaceLoss

class ResNet50ArcFaceBackbone(nn.Module):
    """
    ResNet-50 Feature Extractor mapping 112x112 RGB images to 512-d L2-normalized ArcFace embeddings.
    """
    def __init__(self, embedding_dim=512, pretrained=True):
        super(ResNet50ArcFaceBackbone, self).__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        base_model = models.resnet50(weights=weights)
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool

        self.fc = nn.Linear(base_model.fc.in_features, embedding_dim, bias=False)
        self.bn_out = nn.BatchNorm1d(embedding_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.bn_out(x)
        return torch.nn.functional.normalize(x, p=2, dim=1)

def run_training(
    sanity_check=False,
    epochs=5,
    batch_size=32,
    lr=1e-4,
    sanity_steps=100,
    run_tag="v1",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=========================================================================")
    print(f"   STARTING ARCFACE MODEL FINE-TUNING PIPELINE ON DEVICE: {device}       ")
    print(f"=========================================================================\n")

    # 1. Dataset & DataLoader
    max_ids = 50 if sanity_check else 300
    dataset = MultiDatasetFaceDataset(is_train=True, max_identities=max_ids)
    # drop_last=True: the model head ends in nn.BatchNorm1d(512). BatchNorm
    # computes per-feature variance across the batch dimension, which is
    # undefined for a batch of 1 -- PyTorch raises
    # "Expected more than 1 value per channel when training". Whenever
    # len(dataset) % batch_size == 1 the final batch of each epoch is exactly
    # that size, so training crashed at the first epoch boundary.
    #
    # Chosen over swapping BatchNorm1d -> GroupNorm because the BN layer is
    # part of the standard ArcFace head (BN after the embedding FC is what the
    # pretrained ArcFace models use); replacing it would change the head
    # architecture and make our checkpoint incompatible with the reference
    # topology for no benefit. Dropping <=batch_size-1 samples per epoch is a
    # negligible data loss (<0.5% at batch_size=32) and shuffle=True means a
    # different remainder is dropped each epoch.
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    )
    if len(loader) == 0:
        raise ValueError(
            f"dataset has {len(dataset)} samples but batch_size={batch_size}; "
            "with drop_last=True that yields zero batches. Lower --batch-size."
        )

    print(f"Dataset Loaded: {len(dataset)} samples across {dataset.num_classes} identities.")

    # 2. Model & Loss
    model = ResNet50ArcFaceBackbone(embedding_dim=512, pretrained=True).to(device)
    criterion = ArcFaceLoss(in_features=512, out_features=dataset.num_classes, s=64.0, m=0.50).to(device)

    # 3. Optimizer & Scheduler
    optimizer = optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(loader))

    # backend/nexgen_engine/training/train_pipeline.py -> .parent*4 -> project root
    _project_root = Path(__file__).resolve().parent.parent.parent.parent
    checkpoint_dir = _project_root / "runtime" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / "finetuned_resnet50_arcface.pt"

    start_time = time.time()
    step_count = 0
    max_steps = 100 if sanity_check else epochs * len(loader)

    print(f"\n--- TRAINING LOOP RUNNING (Sanity Check = {sanity_check}) ---")
    print(f"{'Epoch':<6} | {'Step':<8} | {'ArcFace Loss':<14} | {'LR':<10} | {'Grad Norm':<10} | {'Step Time':<10}")
    print("-" * 75)

    epoch_stats: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        ep_losses: list[float] = []
        ep_grads: list[float] = []
        for batch_idx, (images, labels) in enumerate(loader):
            step_start = time.time()
            step_count += 1
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            embeddings = model(images)
            loss = criterion(embeddings, labels)
            loss.backward()

            # Compute gradient norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            optimizer.step()
            scheduler.step()

            ep_losses.append(float(loss.item()))
            ep_grads.append(float(total_norm))

            step_time = (time.time() - step_start) * 1000.0  # ms
            current_lr = optimizer.param_groups[0]['lr']

            if step_count % 10 == 0 or step_count == 1 or sanity_check:
                print(f"{epoch:<6} | {step_count:<8} | {loss.item():<14.6f} | {current_lr:<10.6f} | {total_norm:<10.4f} | {step_time:<8.1f} ms")

            if sanity_check and step_count >= sanity_steps:
                first, last = ep_losses[0], ep_losses[-1]
                print(f"\n[SANITY CHECK] completed {step_count} gradient steps without error.")
                print(f"  loss  first={first:.4f}  last={last:.4f}  delta={last - first:+.4f}")
                print(f"  grad norm  min={min(ep_grads):.3f} max={max(ep_grads):.3f}")
                print("  NOTE: a short sanity run proves the loop executes; it does NOT")
                print("        establish convergence and yields no accuracy claim.")
                return model, ckpt_path

        # ---- end of epoch: report the loss curve, not just 'it ran' ----
        n = len(ep_losses)
        head = sum(ep_losses[: max(1, n // 10)]) / max(1, n // 10)
        tail = sum(ep_losses[-max(1, n // 10) :]) / max(1, n // 10)
        grads = torch.tensor(ep_grads)
        stats = {
            "epoch": epoch,
            "steps": n,
            "loss_first_decile_mean": head,
            "loss_last_decile_mean": tail,
            "loss_min": min(ep_losses),
            "loss_max": max(ep_losses),
            "grad_norm_mean": float(grads.mean()),
            "grad_norm_std": float(grads.std()),
            "grad_norm_max": float(grads.max()),
            "grad_norm_nonfinite": int((~torch.isfinite(grads)).sum()),
        }
        epoch_stats.append(stats)
        print(
            f"\n[EPOCH {epoch} COMPLETE] steps={n}  "
            f"loss {head:.4f} -> {tail:.4f} ({tail - head:+.4f})  "
            f"grad_norm mean={stats['grad_norm_mean']:.3f} "
            f"std={stats['grad_norm_std']:.3f} max={stats['grad_norm_max']:.3f} "
            f"nonfinite={stats['grad_norm_nonfinite']}\n"
        )

    # Versioned checkpoint so runs are distinguishable and A/B-able against
    # the stock pretrained weights. Never overwrite a previous run's file.
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    versioned = checkpoint_dir / f"arcface_ft_{run_tag}_{stamp}.pt"

    ckpt_dict = {
        'model_state_dict': model.state_dict(),
        'num_classes': dataset.num_classes,
        'embedding_dim': 512,
        'arch': 'resnet50_arcface',
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'max_identities': max_ids,
        'train_samples': len(dataset),
        'epoch_stats': epoch_stats,
        'created_utc': datetime.now(timezone.utc).isoformat(),
    }
    torch.save(ckpt_dict, versioned)
    torch.save(ckpt_dict, ckpt_path)  # stable "latest" alias

    total_time = time.time() - start_time
    file_size = versioned.stat().st_size
    print(f"\n=========================================================================")
    print(f"   FINE-TUNING COMPLETED IN {total_time:.2f} SECONDS                      ")
    print(f"   VERSIONED CHECKPOINT: {versioned.resolve()}")
    print(f"   LATEST ALIAS:         {ckpt_path.resolve()}")
    print(f"   FILE SIZE: {file_size / 1e6:.2f} MB ({file_size:,} bytes)")
    print(f"   NOTE: no accuracy is claimed for this checkpoint until it is")
    print(f"         evaluated by backend/scripts/benchmark_verification.py")
    print(f"=========================================================================")

    return model, versioned

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity-check", action="store_true", help="Run short 100-step sanity check")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--sanity-steps", type=int, default=100, help="Steps for --sanity-check")
    parser.add_argument("--run-tag", type=str, default="v1", help="Version tag for the checkpoint filename")
    args = parser.parse_args()

    run_training(
        sanity_check=args.sanity_check,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        sanity_steps=args.sanity_steps,
        run_tag=args.run_tag,
    )
