import glob
import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path("backend").resolve()))

from nexgen_engine.api.service import EngineService

def run_benchmark():
    print("[BENCHMARK] Initializing EngineService with Real ArcFace backbone...")
    service = EngineService()
    
    img_files = glob.glob("src_extracted/AgeDB/AgeDB/*.jpg")
    subjects = {}
    for f in img_files:
        name = os.path.basename(f)
        parts = name.split("_")
        if len(parts) >= 3:
            subj_id = parts[1]
            subjects.setdefault(subj_id, []).append(f)
            
    # Select 25 identities with >= 2 photos
    selected_subjs = {k: v for k, v in list(subjects.items()) if len(v) >= 2}
    selected_keys = list(selected_subjs.keys())[:25]
    
    print(f"[BENCHMARK] Selected 25 identities: {selected_keys}")
    
    # 1. 1:N Identification Gallery Setup (Enroll first photo of each subject)
    gallery_records = {}
    for subj in selected_keys:
        photos = selected_subjs[subj]
        enroll_photo = photos[0]
        res = service.enroll(Path(enroll_photo).read_bytes(), subj)
        gallery_records[subj] = photos[1:]  # remaining photos as probes
        
    # 2. 1:N Search Evaluation (Probes against Gallery)
    rank1_hits = 0
    total_probes = 0
    
    for subj, probe_photos in gallery_records.items():
        for probe_path in probe_photos:
            total_probes += 1
            search_res = service.identify(Path(probe_path).read_bytes(), top_k=5)
            if search_res.matches and search_res.matches[0].identity_id == subj:
                rank1_hits += 1
                
    rank1_accuracy = (rank1_hits / total_probes) * 100.0
    print(f"\n--- 1:N BENCHMARK RESULTS (25 Identities, {total_probes} Probes) ---")
    print(f"Rank-1 Accuracy: {rank1_accuracy:.2f}% ({rank1_hits}/{total_probes} correct)")
    
    # 3. 1:1 Genuine vs Impostor Verification Pair Evaluation
    genuine_scores = []
    impostor_scores = []
    
    # Compute genuine scores (all pairs for same subject)
    for subj in selected_keys:
        photos = selected_subjs[subj][:5]  # cap at 5 photos per subject
        for i in range(len(photos)):
            for j in range(i+1, len(photos)):
                res = service.verify(Path(photos[i]).read_bytes(), Path(photos[j]).read_bytes())
                genuine_scores.append(res.score)
                
    # Compute impostor scores (cross-subject pairs)
    for i in range(len(selected_keys)):
        for j in range(i+1, len(selected_keys)):
            p1 = selected_subjs[selected_keys[i]][0]
            p2 = selected_subjs[selected_keys[j]][0]
            res = service.verify(Path(p1).read_bytes(), Path(p2).read_bytes())
            impostor_scores.append(res.score)
            
    gen_mean = np.mean(genuine_scores)
    imp_mean = np.mean(impostor_scores)
    
    # TAR / FAR at thresholds 0.28, 0.36, 0.42
    thresholds = [0.28, 0.36, 0.42]
    
    print("\n--- 1:1 BIOMETRIC VERIFICATION STATS ---")
    print(f"Genuine Pairs Count: {len(genuine_scores)}, Mean Genuine Score: {gen_mean:.4f}")
    print(f"Impostor Pairs Count: {len(impostor_scores)}, Mean Impostor Score: {imp_mean:.4f}")
    print("\nThreshold Metrics:")
    for th in thresholds:
        tar = np.mean(np.array(genuine_scores) >= th) * 100.0
        far = np.mean(np.array(impostor_scores) >= th) * 100.0
        print(f"  Threshold >= {th:.2f}: TAR = {tar:.2f}%, FAR = {far:.2f}%")

if __name__ == "__main__":
    run_benchmark()
