import glob
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path("backend").resolve()))

from scripts.benchmark_phase1_hard import build_200_identity_hard_dataset
from nexgen_engine.models.insightface_backbone import InsightFaceArcFaceBackbone, InsightFaceEnsembleBackbone
from nexgen_engine.utils import cosine_similarity

def run_phase2_benchmark():
    records = build_200_identity_hard_dataset()
    print("=========================================================================")
    print(f"   PHASE 2 BENCHMARK: MULTI-MODEL ENSEMBLE ON 200 HARD IDENTITIES      ")
    print("=========================================================================\n")
    print(f"Total Enrolled Identities: {len(records)}")

    # 1. Initialize Single Model Baseline
    print("\n[INIT 1/2] Initializing Single Model Baseline (buffalo_l / w600k_r50)...")
    single_backbone = InsightFaceArcFaceBackbone()

    # 2. Initialize Multi-Model Ensemble
    print("[INIT 2/2] Initializing Multi-Model Ensemble (buffalo_l + antelopev2 + buffalo_s)...")
    ensemble_backbone = InsightFaceEnsembleBackbone()

    def get_embeddings_dict_for_mode(encode_mode_fn):
        all_embs = {}
        for r in records:
            photos = r["photos"]
            for p in photos[:5]:
                if p not in all_embs:
                    with Image.open(p) as img:
                        all_embs[p] = encode_mode_fn(img)
        return all_embs

    print("\n[ENCODING] Extracting embeddings across all 4 ensemble modes...")
    
    modes = {
        "Single Model (buffalo_l)": lambda img: single_backbone.encode(img).embedding,
        "Dual Ensemble (50/50 Avg)": lambda img: ensemble_backbone.fuse_embeddings(
            ensemble_backbone.extract_all_embeddings(np.array(img.convert("RGB"))[:, :, ::-1]), "dual_ensemble"
        ),
        "Triple Ensemble (45/45/10 Avg)": lambda img: ensemble_backbone.fuse_embeddings(
            ensemble_backbone.extract_all_embeddings(np.array(img.convert("RGB"))[:, :, ::-1]), "weighted_avg"
        ),
        "Triple Ensemble (Concat 1536d)": lambda img: ensemble_backbone.fuse_embeddings(
            ensemble_backbone.extract_all_embeddings(np.array(img.convert("RGB"))[:, :, ::-1]), "concat"
        ),
    }

    mode_results = []

    for mode_name, encode_fn in modes.items():
        print(f"\n---> Evaluating: {mode_name}...")
        embs_dict = get_embeddings_dict_for_mode(encode_fn)

        # 1:N Gallery setup
        gallery = {}
        probes = []
        for r in records:
            gid = r["global_id"]
            ds = r["dataset"]
            photos = r["photos"]
            gallery[gid] = embs_dict[photos[0]]
            for probe_path in photos[1:5]:
                probes.append((gid, ds, probe_path))

        # 1:N Evaluation
        rank1_hits = 0
        ds_stats = {}
        for gid, ds, probe_path in probes:
            probe_emb = embs_dict[probe_path]
            scores = []
            for g_id, g_emb in gallery.items():
                sim = float(cosine_similarity(probe_emb, g_emb))
                scores.append((g_id, sim))
            scores.sort(key=lambda x: x[1], reverse=True)

            top1_id, top1_score = scores[0]
            is_hit = (top1_id == gid)
            if is_hit:
                rank1_hits += 1

            st = ds_stats.setdefault(ds, {"hits": 0, "total": 0})
            st["total"] += 1
            if is_hit:
                st["hits"] += 1

        total_probes = len(probes)
        overall_acc = (rank1_hits / total_probes) * 100.0

        # 1:1 Verification Evaluation
        genuine_scores = []
        impostor_scores = []
        for r in records:
            photos = r["photos"][:5]
            for i in range(len(photos)):
                for j in range(i+1, len(photos)):
                    if photos[i] in embs_dict and photos[j] in embs_dict:
                        genuine_scores.append(float(cosine_similarity(embs_dict[photos[i]], embs_dict[photos[j]])))

        for i in range(len(records)):
            for j in range(i+1, min(i+20, len(records))):
                p1 = records[i]["photos"][0]
                p2 = records[j]["photos"][0]
                if p1 in embs_dict and p2 in embs_dict:
                    impostor_scores.append(float(cosine_similarity(embs_dict[p1], embs_dict[p2])))

        gen_mean = float(np.mean(genuine_scores))
        imp_mean = float(np.mean(impostor_scores))

        tar_28 = float(np.mean(np.array(genuine_scores) >= 0.28) * 100.0)
        tar_36 = float(np.mean(np.array(genuine_scores) >= 0.36) * 100.0)
        tar_42 = float(np.mean(np.array(genuine_scores) >= 0.42) * 100.0)

        far_28 = float(np.mean(np.array(impostor_scores) >= 0.28) * 100.0)
        far_36 = float(np.mean(np.array(impostor_scores) >= 0.36) * 100.0)
        far_42 = float(np.mean(np.array(impostor_scores) >= 0.42) * 100.0)

        mode_results.append({
            "name": mode_name,
            "overall_acc": overall_acc,
            "hits": rank1_hits,
            "probes": total_probes,
            "ds_stats": ds_stats,
            "gen_mean": gen_mean,
            "imp_mean": imp_mean,
            "tar_28": tar_28, "far_28": far_28,
            "tar_36": tar_36, "far_36": far_36,
            "tar_42": tar_42, "far_42": far_42,
        })

    print("\n\n=========================================================================================================")
    print("                      PHASE 2 HARD BENCHMARK COMPARISON SUMMARY SUMMARY                                  ")
    print("=========================================================================================================")
    header = f"{'Model / Ensemble Variant':<32} | {'Overall Acc':<11} | {'AgeDB':<8} | {'LFW':<8} | {'CFP 90°':<8} | {'TinyFace':<8} | {'Gen Mean':<8} | {'TAR@0.42':<8}"
    print(header)
    print("-" * len(header))

    for r in mode_results:
        ds = r["ds_stats"]
        acc_agedb = (ds['AgeDB (Age Gap)']['hits'] / ds['AgeDB (Age Gap)']['total']) * 100.0
        acc_lfw = (ds['LFW (In-The-Wild)']['hits'] / ds['LFW (In-The-Wild)']['total']) * 100.0
        acc_cfp = (ds['CFP (Extreme Pose Angle)']['hits'] / ds['CFP (Extreme Pose Angle)']['total']) * 100.0
        acc_tf = (ds['TinyFace (Low-Res Surveillance)']['hits'] / ds['TinyFace (Low-Res Surveillance)']['total']) * 100.0

        print(f"{r['name']:<32} | {r['overall_acc']:>6.2f}%     | {acc_agedb:>6.2f}% | {acc_lfw:>6.2f}% | {acc_cfp:>6.2f}% | {acc_tf:>6.2f}% | {r['gen_mean']:>8.4f} | {r['tar_42']:>6.2f}%")

if __name__ == "__main__":
    run_phase2_benchmark()
