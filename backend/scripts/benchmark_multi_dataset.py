"""
Phase 1 Rigorous Multi-Dataset Benchmark Script
Benchmark targets 200 diverse identities across 4 distinct datasets:
1. AgeDB (50 identities) - Cross-age variation
2. LFW (50 identities) - Varied real-world unconstrained lighting/expression
3. CFP (50 identities) - Extreme pose variation (Frontal vs Profile)
4. TinyFace (50 identities) - Ultra low-resolution surveillance probe evaluation
"""

import glob
import os
import sys
from pathlib import Path
import requests
import numpy as np

BASE = "http://127.0.0.1:8000/api"

def enroll(path, identity_id):
    with open(path, "rb") as f:
        r = requests.post(f"{BASE}/biometrics/enroll",
                          files={"file": f},
                          data={"identity_id": identity_id},
                          timeout=30)
    r.raise_for_status()
    return r.json()

def identify(path, top_k=5):
    with open(path, "rb") as f:
        r = requests.post(f"{BASE}/biometrics/identify",
                          files={"file": f},
                          data={"top_k": str(top_k)},
                          timeout=30)
    r.raise_for_status()
    return r.json()

def verify(path_a, path_b):
    with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
        r = requests.post(f"{BASE}/biometrics/verify",
                          files={"reference": fa, "probe": fb},
                          timeout=30)
    r.raise_for_status()
    return r.json()

def load_dataset_identities():
    dataset_records = {}

    # 1. AgeDB (50 identities, >=3 photos each)
    agedb_files = glob.glob("src_extracted/AgeDB/AgeDB/*.jpg")
    agedb_subjs = {}
    for f in agedb_files:
        name = os.path.basename(f)
        parts = name.split("_")
        if len(parts) >= 3:
            agedb_subjs.setdefault(parts[1], []).append(f)
    agedb_keys = [k for k, v in agedb_subjs.items() if len(v) >= 3][:50]
    dataset_records["AgeDB"] = {f"AgeDB_{k}": agedb_subjs[k] for k in agedb_keys}

    # 2. LFW (50 identities, >=3 photos each)
    lfw_root = r"src_extracted/archive (1)/lfw-deepfunneled/lfw-deepfunneled"
    if os.path.exists(lfw_root):
        lfw_subjs = [d for d in os.listdir(lfw_root) if os.path.isdir(os.path.join(lfw_root, d))]
        lfw_selected = {}
        for s in lfw_subjs:
            s_dir = os.path.join(lfw_root, s)
            photos = sorted(glob.glob(os.path.join(s_dir, "*.jpg")))
            if len(photos) >= 3:
                lfw_selected[f"LFW_{s}"] = photos
                if len(lfw_selected) == 50:
                    break
        dataset_records["LFW"] = lfw_selected

    # 3. CFP (50 identities, frontal gallery + profile probes)
    cfp_root = r"src_extracted/archive (4)/cfp-dataset/Data/Images"
    if os.path.exists(cfp_root):
        cfp_subjs = sorted([d for d in os.listdir(cfp_root) if os.path.isdir(os.path.join(cfp_root, d))])[:50]
        cfp_selected = {}
        for s in cfp_subjs:
            s_dir = os.path.join(cfp_root, s)
            fronts = sorted(glob.glob(os.path.join(s_dir, "frontal", "*.jpg")))
            profiles = sorted(glob.glob(os.path.join(s_dir, "profile", "*.jpg")))
            if fronts and profiles:
                cfp_selected[f"CFP_{s}"] = fronts[:2] + profiles[:2]
        dataset_records["CFP"] = cfp_selected

    # 4. TinyFace (50 identities, match gallery + probe)
    tf_probe_dir = r"src_extracted/tinyface/tinyface/Testing_Set/Probe"
    tf_match_dir = r"src_extracted/tinyface/tinyface/Testing_Set/Gallery_Match"
    if os.path.exists(tf_probe_dir) and os.path.exists(tf_match_dir):
        probe_files = os.listdir(tf_probe_dir)
        tf_subjs = {}
        for f in probe_files:
            sid = f.split("_")[0]
            tf_subjs.setdefault(sid, []).append(os.path.join(tf_probe_dir, f))
        
        tf_selected = {}
        for sid, probes in tf_subjs.items():
            matches = glob.glob(os.path.join(tf_match_dir, f"{sid}_*.jpg"))
            if matches and len(probes) >= 1:
                tf_selected[f"TinyFace_{sid}"] = [matches[0]] + probes[:2]
                if len(tf_selected) == 50:
                    break
        dataset_records["TinyFace"] = tf_selected

    return dataset_records

def main():
    print("================================================================================")
    print("PHASE 1 RIGOROUS BENCHMARK: 200 IDENTITIES ACROSS 4 DATASETS")
    print("================================================================================\n")

    dataset_records = load_dataset_identities()
    total_identities = sum(len(subjs) for subjs in dataset_records.values())
    print(f"Total Enrolled Identities: {total_identities}")
    for dname, subjs in dataset_records.items():
        print(f"  - {dname}: {len(subjs)} identities")

    # 1. Enroll Gallery (First photo of all 200 identities)
    print("\n--------------------------------------------------------------------------------")
    print("STEP 1: ENROLLING ALL 200 IDENTITIES INTO GALLERY")
    print("--------------------------------------------------------------------------------")
    
    all_identities = {}
    for dname, subjs in dataset_records.items():
        for global_id, photos in subjs.items():
            res = enroll(photos[0], global_id)
            all_identities[global_id] = {
                "dataset": dname,
                "gallery_photo": photos[0],
                "probe_photos": photos[1:]
            }

    print(f"Successfully enrolled {len(all_identities)} identities into workspace index.")

    # 2. 1:N Identification Evaluation
    print("\n--------------------------------------------------------------------------------")
    print("STEP 2: 1:N IDENTIFICATION PROBE SEARCH (PER-PROBE DETAILED AUDIT)")
    print("--------------------------------------------------------------------------------")
    
    total_hits = 0
    total_probes = 0
    dataset_stats = {d: {"hits": 0, "total": 0, "errors": []} for d in dataset_records.keys()}

    for global_id, record in all_identities.items():
        dname = record["dataset"]
        for probe_path in record["probe_photos"]:
            total_probes += 1
            dataset_stats[dname]["total"] += 1
            
            pname = os.path.basename(probe_path)
            res = identify(probe_path, top_k=5)
            
            top_match = res["matches"][0]["identity_id"] if res["matches"] else "NONE"
            score = res["matches"][0]["confidence"] if res["matches"] else 0.0
            
            is_hit = (top_match == global_id)
            if is_hit:
                total_hits += 1
                dataset_stats[dname]["hits"] += 1
                status_str = "HIT "
            else:
                status_str = "MISS"
                dataset_stats[dname]["errors"].append({
                    "probe": pname,
                    "target_id": global_id,
                    "top_predicted": top_match,
                    "score": score
                })

            print(f"[{dname:8s}] Target: {global_id:25s} | Probe: {pname:35s} | Status: {status_str} | Top: {top_match:25s} | Score: {score:.4f}")

    # 3. 1:1 Verification Evaluation (Genuine & Impostor)
    print("\n--------------------------------------------------------------------------------")
    print("STEP 3: 1:1 BIOMETRIC VERIFICATION (GENUINE & IMPOSTOR PAIRS)")
    print("--------------------------------------------------------------------------------")

    genuine_scores = []
    impostor_scores = []

    # Genuine pairs (Gallery vs Probes)
    for global_id, record in all_identities.items():
        gallery_photo = record["gallery_photo"]
        for probe_photo in record["probe_photos"]:
            res = verify(gallery_photo, probe_photo)
            genuine_scores.append(res["score"])

    # Impostor pairs (Cross-identity gallery comparisons across different subjects)
    id_list = list(all_identities.keys())
    for i in range(len(id_list) - 1):
        id_a = id_list[i]
        id_b = id_list[i+1]
        photo_a = all_identities[id_a]["gallery_photo"]
        photo_b = all_identities[id_b]["gallery_photo"]
        res = verify(photo_a, photo_b)
        impostor_scores.append(res["score"])

    gen_arr = np.array(genuine_scores)
    imp_arr = np.array(impostor_scores)

    # 4. Final Comprehensive Summary Report
    print("\n================================================================================")
    print("FINAL BENCHMARK ACCURACY & ERROR BREAKDOWN REPORT")
    print("================================================================================")
    
    overall_rank1 = (total_hits / total_probes) * 100.0 if total_probes > 0 else 0.0
    print(f"\nOVERALL 1:N RANK-1 ACCURACY: {total_hits}/{total_probes} = {overall_rank1:.2f}%\n")
    
    print("DATASET ACCURACY BREAKDOWN:")
    for dname, stats in dataset_stats.items():
        d_acc = (stats["hits"] / stats["total"]) * 100.0 if stats["total"] > 0 else 0.0
        print(f"  - {dname:12s}: {stats['hits']:3d} / {stats['total']:3d} correct ({d_acc:6.2f}%)")

    print("\n1:1 VERIFICATION SCORE DISTRIBUTIONS:")
    print(f"  Genuine Pairs  (N={len(gen_arr):3d}): Mean = {gen_arr.mean():.4f}, Std = {gen_arr.std():.4f}, Min = {gen_arr.min():.4f}, Max = {gen_arr.max():.4f}")
    print(f"  Impostor Pairs (N={len(imp_arr):3d}): Mean = {imp_arr.mean():.4f}, Std = {imp_arr.std():.4f}, Min = {imp_arr.min():.4f}, Max = {imp_arr.max():.4f}")

    print("\nDECISION THRESHOLD METRICS (TAR vs FAR):")
    print("  Threshold | TAR (True Accept Rate) | FAR (False Accept Rate)")
    print("  ----------+------------------------+------------------------")
    for th in [0.28, 0.36, 0.42]:
        tar = (gen_arr >= th).mean() * 100.0
        far = (imp_arr >= th).mean() * 100.0
        print(f"    {th:.2f}    |         {tar:5.1f}%          |         {far:5.1f}%")

    print("\nERROR ANALYSIS & FAILURES BY DATASET:")
    for dname, stats in dataset_stats.items():
        errs = stats["errors"]
        print(f"\n[{dname}] Total Errors: {len(errs)}")
        for err in errs[:5]:  # print first 5 sample failures per dataset
            print(f"   Probe: {err['probe']} | Target: {err['target_id']} | Predicted: {err['top_predicted']} | Score: {err['score']:.4f}")

if __name__ == "__main__":
    main()
