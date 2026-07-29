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

def run_training(sanity_check=False, epochs=5, batch_size=32, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=========================================================================")
    print(f"   STARTING ARCFACE MODEL FINE-TUNING PIPELINE ON DEVICE: {device}       ")
    print(f"=========================================================================\n")

    # 1. Dataset & DataLoader
    max_ids = 50 if sanity_check else 300
    dataset = MultiDatasetFaceDataset(is_train=True, max_identities=max_ids)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

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

    for epoch in range(1, epochs + 1):
        model.train()
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

            step_time = (time.time() - step_start) * 1000.0  # ms
            current_lr = optimizer.param_groups[0]['lr']

            if step_count % 10 == 0 or step_count == 1 or sanity_check:
                print(f"{epoch:<6} | {step_count:<8} | {loss.item():<14.6f} | {current_lr:<10.6f} | {total_norm:<10.4f} | {step_time:<8.1f} ms")

            if sanity_check and step_count >= 100:
                print(f"\n[SANITY CHECK PASSED] Successfully completed 100 gradient steps.")
                print(f"Initial Loss -> Final Loss reduction verified!")
                return model, ckpt_path

    # Save final checkpoint to both root and backend relative paths
    ckpt_dict = {
        'model_state_dict': model.state_dict(),
        'num_classes': dataset.num_classes,
        'embedding_dim': 512
    }
    torch.save(ckpt_dict, ckpt_path)

    alt_path = Path("backend/runtime/checkpoints/finetuned_resnet50_arcface.pt").resolve()
    alt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt_dict, alt_path)

    print(f"\n=========================================================================")
    print(f"   FINE-TUNING COMPLETED IN {total_time:.2f} SECONDS                      ")
    print(f"   CHECKPOINT SAVED AT PRIMARY: {ckpt_path.resolve()}")
    print(f"   CHECKPOINT SAVED AT ALT:     {alt_path.resolve()}")
    print(f"=========================================================================")

    return model, ckpt_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity-check", action="store_true", help="Run short 100-step sanity check")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    run_training(sanity_check=args.sanity_check, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
