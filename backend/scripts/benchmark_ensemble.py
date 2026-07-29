import glob
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path("backend").resolve()))

from nexgen_engine.models.insightface_backbone import InsightFaceArcFaceBackbone, InsightFaceEnsembleBackbone
from nexgen_engine.utils import cosine_similarity, l2_normalize

def load_agedb_dataset():
    img_files = glob.glob("src_extracted/AgeDB/AgeDB/*.jpg")
    subjects = {}
    for f in sorted(img_files):
        name = os.path.basename(f)
        parts = name.split("_")
        if len(parts) >= 3:
            subj_id = parts[1]
            subjects.setdefault(subj_id, []).append(f)
            
    selected_subjs = {k: v for k, v in list(subjects.items()) if len(v) >= 2}
    selected_keys = sorted(list(selected_subjs.keys()))[:25]
    return selected_subjs, selected_keys

def run_evaluation_for_mode(name, encode_fn, selected_subjs, selected_keys):
    print(f"\n=======================================================")
    print(f"RUNNING BENCHMARK: {name}")
    print(f"=======================================================")
    
    # Pre-extract all embeddings to ensure identical input images are evaluated
    image_embeddings = {}
    for subj in selected_keys:
        for photo_path in selected_subjs[subj]:
            with Image.open(photo_path) as img:
                image_embeddings[photo_path] = encode_fn(img)
                
    # 1. 1:N Identification Gallery Setup (Enroll first photo of each subject)
    gallery = {}
    gallery_records = {}
    for subj in selected_keys:
        photos = selected_subjs[subj]
        enroll_photo = photos[0]
        gallery[subj] = image_embeddings[enroll_photo]
        gallery_records[subj] = photos[1:]  # probes
        
    # 2. 1:N Search Evaluation
    rank1_hits = 0
    total_probes = 0
    hits_detail = []
    misses_detail = []
    
    for subj, probe_photos in gallery_records.items():
        for probe_path in probe_photos:
            total_probes += 1
            probe_emb = image_embeddings[probe_path]
            
            # Compute similarity against all gallery entries
            scores = []
            for g_subj, g_emb in gallery.items():
                sim = float(cosine_similarity(probe_emb, g_emb))
                scores.append((g_subj, sim))
            scores.sort(key=lambda x: x[1], reverse=True)
            
            top1_subj, top1_score = scores[0]
            if top1_subj == subj:
                rank1_hits += 1
                hits_detail.append((subj, os.path.basename(probe_path), top1_score))
            else:
                misses_detail.append((subj, top1_subj, os.path.basename(probe_path), top1_score))
                
    rank1_accuracy = (rank1_hits / total_probes) * 100.0
    print(f"\n--- 1:N BENCHMARK RESULTS ({len(selected_keys)} Identities, {total_probes} Probes) ---")
    print(f"Rank-1 Accuracy: {rank1_accuracy:.2f}% ({rank1_hits}/{total_probes} correct)")
    print(f"Sample Hits (first 3): {hits_detail[:3]}")
    print(f"Sample Misses (first 3): {misses_detail[:3]}")
    
    # 3. 1:1 Genuine vs Impostor Verification Pair Evaluation
    genuine_scores = []
    impostor_scores = []
    
    # Compute genuine scores (all pairs for same subject up to 5 photos)
    for subj in selected_keys:
        photos = selected_subjs[subj][:5]
        for i in range(len(photos)):
            for j in range(i+1, len(photos)):
                e1 = image_embeddings[photos[i]]
                e2 = image_embeddings[photos[j]]
                score = float(cosine_similarity(e1, e2))
                genuine_scores.append(score)
                
    # Compute impostor scores (cross-subject pairs)
    for i in range(len(selected_keys)):
        for j in range(i+1, len(selected_keys)):
            p1 = selected_subjs[selected_keys[i]][0]
            p2 = selected_subjs[selected_keys[j]][0]
            e1 = image_embeddings[p1]
            e2 = image_embeddings[p2]
            score = float(cosine_similarity(e1, e2))
            impostor_scores.append(score)
            
    gen_mean = np.mean(genuine_scores)
    imp_mean = np.mean(impostor_scores)
    
    thresholds = [0.28, 0.36, 0.42]
    print("\n--- 1:1 BIOMETRIC VERIFICATION STATS ---")
    print(f"Genuine Pairs Count: {len(genuine_scores)}, Mean Genuine Score: {gen_mean:.4f}")
    print(f"Impostor Pairs Count: {len(impostor_scores)}, Mean Impostor Score: {imp_mean:.4f}")
    print("\nThreshold Metrics:")
    results_by_threshold = {}
    for th in thresholds:
        tar = np.mean(np.array(genuine_scores) >= th) * 100.0
        far = np.mean(np.array(impostor_scores) >= th) * 100.0
        results_by_threshold[th] = (tar, far)
        print(f"  Threshold >= {th:.2f}: TAR = {tar:.2f}%, FAR = {far:.2f}%")
        
    return {
        "name": name,
        "rank1_acc": rank1_accuracy,
        "rank1_hits": rank1_hits,
        "total_probes": total_probes,
        "gen_mean": gen_mean,
        "imp_mean": imp_mean,
        "thresholds": results_by_threshold,
    }

def main():
    selected_subjs, selected_keys = load_agedb_dataset()
    print(f"[BENCHMARK] Loaded AgeDB dataset: 25 identities selected.")
    
    # 1. Baseline Model: buffalo_l (w600k_r50)
    print("\n[INIT] Initializing Baseline Model: buffalo_l (w600k_r50)...")
    single_model = InsightFaceArcFaceBackbone()
    res_single = run_evaluation_for_mode(
        "Baseline (Single Model: buffalo_l / w600k_r50)",
        lambda img: single_model.encode(img).embedding,
        selected_subjs,
        selected_keys
    )
    
    # 2. Multi-model Ensemble Initializer
    print("\n[INIT] Initializing Multi-Model Ensemble (buffalo_l + antelopev2 + buffalo_s)...")
    ensemble = InsightFaceEnsembleBackbone()
    
    # 2a. Dual Ensemble (buffalo_l + antelopev2)
    res_dual = run_evaluation_for_mode(
        "Dual Ensemble (buffalo_l + antelopev2, Weighted Avg 50/50)",
        lambda img: ensemble.fuse_embeddings(ensemble.extract_all_embeddings(np.array(img.convert("RGB"))[:,:,::-1]), "dual_ensemble"),
        selected_subjs,
        selected_keys
    )
    
    # 2b. Triple Ensemble Weighted Averaging (45/45/10)
    res_triple_weighted = run_evaluation_for_mode(
        "Triple Ensemble (buffalo_l + antelopev2 + buffalo_s, Weighted Avg 45/45/10)",
        lambda img: ensemble.fuse_embeddings(ensemble.extract_all_embeddings(np.array(img.convert("RGB"))[:,:,::-1]), "weighted_avg"),
        selected_subjs,
        selected_keys
    )
    
    # 2c. Triple Ensemble Concatenation (1536-d)
    res_triple_concat = run_evaluation_for_mode(
        "Triple Ensemble (buffalo_l + antelopev2 + buffalo_s, Concatenation Fusion 1536-d)",
        lambda img: ensemble.fuse_embeddings(ensemble.extract_all_embeddings(np.array(img.convert("RGB"))[:,:,::-1]), "concat"),
        selected_subjs,
        selected_keys
    )
    
    # Side-by-side summary table
    print("\n\n=======================================================")
    print("      FINAL HONEST BENCHMARK COMPARISON SUMMARY        ")
    print("=======================================================")
    headers = f"{'Model / Ensemble Variant':<60} | {'Rank-1 Acc':<11} | {'Mean Gen':<9} | {'Mean Imp':<9} | {'TAR @ 0.36':<10} | {'FAR @ 0.36':<10}"
    print(headers)
    print("-" * len(headers))
    
    all_res = [res_single, res_dual, res_triple_weighted, res_triple_concat]
    for r in all_res:
        tar_36, far_36 = r["thresholds"][0.36]
        print(f"{r['name']:<60} | {r['rank1_acc']:>6.2f}%     | {r['gen_mean']:>8.4f}  | {r['imp_mean']:>8.4f}  | {tar_36:>8.2f}%  | {far_36:>8.2f}%")

if __name__ == "__main__":
    main()
