import os
import sys
import io
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T

# Force stdout to UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path("backend").resolve()))

from nexgen_engine.training.train_pipeline import ResNet50ArcFaceBackbone
from nexgen_engine.utils import cosine_similarity
from scripts.benchmark_phase1_hard import build_200_identity_hard_dataset

def _safe(s, width=25):
    return s[:width].encode('ascii', errors='replace').decode('ascii').ljust(width)

class FineTunedEvaluator:
    def __init__(self, checkpoint_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Phase 3 Eval] Loading Fine-tuned Weights from: {checkpoint_path} on {self.device}")
        
        self.model = ResNet50ArcFaceBackbone(embedding_dim=512, pretrained=False).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((112, 112)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def get_embedding(self, image_path):
        try:
            with Image.open(image_path).convert("RGB") as img:
                tensor = self.transform(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    emb = self.model(tensor).squeeze(0).cpu().numpy()
                return emb
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None

def run_phase3_benchmark():
    _project_root = Path(__file__).resolve().parent.parent
    ckpt_path = _project_root / "runtime" / "checkpoints" / "finetuned_resnet50_arcface.pt"
    alt_path = _project_root / "backend" / "runtime" / "checkpoints" / "finetuned_resnet50_arcface.pt"

    target_path = None
    if ckpt_path.exists():
        target_path = ckpt_path
    elif alt_path.exists():
        target_path = alt_path

    if target_path is None:
        print(f"ERROR: Checkpoint not found at {ckpt_path} or {alt_path}")
        return

    evaluator = FineTunedEvaluator(str(target_path))

    print("\n=========================================================================")
    print("      PHASE 3 BENCHMARK: FINE-TUNED MODEL EVALUATION (200 IDENTITIES)     ")
    print("=========================================================================\n")

    dataset = build_200_identity_hard_dataset()

    gallery = {}
    probes = []

    for rec in dataset:
        gid = rec["id"]
        ds = rec["dataset"]
        photos = rec["photos"]

        gallery_photo = photos[0]
        probe_photos = photos[1:]

        emb = evaluator.get_embedding(gallery_photo)
        if emb is not None:
            gallery[gid] = {
                "embedding": emb,
                "dataset": ds,
                "path": gallery_photo
            }

        for p in probe_photos:
            probes.append({
                "true_id": gid,
                "dataset": ds,
                "probe_path": p
            })

    print(f"Built Hard Evaluation Benchmark:")
    print(f"  - Gallery Identities (1 per ID): {len(gallery)}")
    print(f"  - Probe Query Images: {len(probes)}\n")

    print(f"{'True Identity':<25} | {'Dataset Category':<30} | {'Result':<6} | {'Top-1 Predicted':<25} | {'Score':<6}")
    print("-" * 105)

    rank1_hits = 0
    per_dataset_results = {}

    for p in probes:
        gid = p["true_id"]
        ds = p["dataset"]
        probe_path = p["probe_path"]

        p_emb = evaluator.get_embedding(probe_path)
        if p_emb is None:
            continue

        best_score = -1.0
        top1_id = None

        for g_id, g_data in gallery.items():
            sim = cosine_similarity(p_emb, g_data["embedding"])
            if sim > best_score:
                best_score = sim
                top1_id = g_id

        is_hit = (top1_id == gid)
        res_str = "HIT" if is_hit else "MISS"
        if is_hit:
            rank1_hits += 1

        ds_stats = per_dataset_results.setdefault(ds, {"hits": 0, "total": 0})
        ds_stats["total"] += 1
        if is_hit:
            ds_stats["hits"] += 1

        print(f"{_safe(gid):<25} | {ds:<30} | {res_str:<6} | {_safe(top1_id):<25} | {best_score:.4f}")

    total_probes = len(probes)
    overall_acc = (rank1_hits / total_probes) * 100.0 if total_probes > 0 else 0.0

    print("\n-------------------------------------------------------------------------")
    print(f"      PHASE 3 FINE-TUNED RANK-1 ACCURACY: {overall_acc:.2f}% ({rank1_hits}/{total_probes})")
    print("-------------------------------------------------------------------------\n")

    print("--- ACCURACY BREAKDOWN BY DATASET CATEGORY ---")
    for ds_name, stats in per_dataset_results.items():
        acc = (stats["hits"] / stats["total"]) * 100.0 if stats["total"] > 0 else 0.0
        print(f"  - {ds_name:<30}: {acc:6.2f}% ({stats['hits']}/{stats['total']})")

    print("\n=========================================================================")
    print("                 HONEST THREE-WAY SIDE-BY-SIDE COMPARISON                ")
    print("=========================================================================")
    print(f" {'Metric / Sub-test':<32} | {'Pretrained Baseline':<20} | {'Dual Ensemble':<20} | {'Fine-Tuned ArcFace':<20}")
    print("-" * 100)
    print(f" {'Overall Rank-1 Accuracy':<32} | {'79.46% (383/482)':<20} | {'81.33% (392/482)':<20} | {overall_acc:.2f}% ({rank1_hits}/{total_probes})")
    for ds_name, stats in per_dataset_results.items():
        acc = (stats["hits"] / stats["total"]) * 100.0 if stats["total"] > 0 else 0.0
        print(f"   - {ds_name:<30} | {'(varies)':<20} | {'(varies)':<20} | {acc:6.2f}%")
    print("=========================================================================\n")

if __name__ == "__main__":
    run_phase3_benchmark()
