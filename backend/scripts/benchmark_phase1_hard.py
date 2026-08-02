import glob
import os
import sys
import io
from pathlib import Path
import numpy as np
from PIL import Image

# Force stdout to UTF-8 to handle non-ASCII characters in identity names (e.g. Cyrillic)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def _safe(s, width=25):
    """Truncate and encode-safe string for printing."""
    return s[:width].encode('ascii', errors='replace').decode('ascii').ljust(width)

sys.path.insert(0, str(Path("backend").resolve()))

from nexgen_engine.models.insightface_backbone import InsightFaceArcFaceBackbone
from nexgen_engine.utils import cosine_similarity

def build_200_identity_hard_dataset():
    dataset_records = []  # list of dicts: {'id': global_id, 'dataset': ds_name, 'photos': [path1, path2, ...]}

    # 1. AgeDB (50 identities)
    agedb_imgs = glob.glob("src_extracted/AgeDB/AgeDB/*.jpg")
    agedb_subjs = {}
    for f in sorted(agedb_imgs):
        parts = os.path.basename(f).split("_")
        if len(parts) >= 3:
            agedb_subjs.setdefault(parts[1], []).append(f)
    agedb_keys = sorted([k for k, v in agedb_subjs.items() if len(v) >= 2])[:50]
    for k in agedb_keys:
        dataset_records.append({
            "global_id": f"agedb_{k}",
            "dataset": "AgeDB (Age Gap)",
            "photos": agedb_subjs[k]
        })

    # 2. LFW (50 identities)
    lfw_dir = Path("src_extracted/lfw_deepfunneled/lfw-deepfunneled/lfw-deepfunneled")
    lfw_subjs = {}
    for d in sorted(list(lfw_dir.iterdir())):
        if d.is_dir():
            imgs = sorted([str(x) for x in d.glob("*.jpg")])
            if len(imgs) >= 2:
                lfw_subjs[d.name] = imgs
    lfw_keys = sorted(list(lfw_subjs.keys()))[:50]
    for k in lfw_keys:
        dataset_records.append({
            "global_id": f"lfw_{k}",
            "dataset": "LFW (In-The-Wild)",
            "photos": lfw_subjs[k]
        })

    # 3. CFP (50 identities - Frontal vs Profile Poses)
    cfp_dir = Path("src_extracted/cfp_dataset/cfp-dataset/Data/Images")
    cfp_subjs = {}
    for d in sorted(list(cfp_dir.iterdir())):
        if d.is_dir():
            f_imgs = sorted([str(x) for x in (d / "frontal").glob("*.jpg")])
            p_imgs = sorted([str(x) for x in (d / "profile").glob("*.jpg")])
            if f_imgs and p_imgs:
                # First frontal photo as enrollment, remaining frontals + profiles as probes
                cfp_subjs[d.name] = f_imgs + p_imgs
    cfp_keys = sorted(list(cfp_subjs.keys()))[:50]
    for k in cfp_keys:
        dataset_records.append({
            "global_id": f"cfp_{k}",
            "dataset": "CFP (Extreme Pose Angle)",
            "photos": cfp_subjs[k]
        })

    # 4. TinyFace (50 identities - Low Resolution Surveillance)
    tf_probe_dir = Path("src_extracted/tinyface/tinyface/Testing_Set/Probe")
    tf_gallery_dir = Path("src_extracted/tinyface/tinyface/Testing_Set/Gallery_Match")
    tf_subjs = {}
    if tf_probe_dir.exists() and tf_gallery_dir.exists():
        for f in sorted(list(tf_probe_dir.glob("*.jpg"))):
            subj_id = f.stem.split("_")[0]
            g_matches = sorted([str(x) for x in tf_gallery_dir.glob(f"{subj_id}_*.jpg")])
            if g_matches and subj_id not in tf_subjs:
                tf_subjs[subj_id] = [g_matches[0], str(f)] + g_matches[1:]
    tf_keys = sorted(list(tf_subjs.keys()))[:50]
    for k in tf_keys:
        dataset_records.append({
            "global_id": f"tinyface_{k}",
            "dataset": "TinyFace (Low-Res Surveillance)",
            "photos": tf_subjs[k]
        })

    return dataset_records

def main():
    print("=========================================================================")
    print("      BUILDING 200-IDENTITY MULTI-DATASET HARD TEST SUITE               ")
    print("=========================================================================")
    records = build_200_identity_hard_dataset()
    print(f"Total Enrolled Identities: {len(records)}")
    
    # Summary per dataset
    ds_counts = {}
    for r in records:
        ds_counts[r["dataset"]] = ds_counts.get(r["dataset"], 0) + 1
    for ds_name, count in ds_counts.items():
        print(f"  - {ds_name}: {count} identities")

    print("\n[INIT] Initializing Baseline ArcFace Model (buffalo_l / w600k_r50)...")
    backbone = InsightFaceArcFaceBackbone()

    print("\n[ENCODING] Extracting embeddings for gallery & probes...")
    all_embeddings = {}
    gallery = {}
    probes = []  # list of (global_id, dataset_name, photo_path)

    for r in records:
        gid = r["global_id"]
        ds = r["dataset"]
        photos = r["photos"]
        
        # Enrollment photo (index 0)
        with Image.open(photos[0]) as img:
            emb_gen = backbone.encode(img).embedding
            all_embeddings[photos[0]] = emb_gen
            gallery[gid] = emb_gen

        # Probe photos (index 1+)
        for probe_path in photos[1:5]:  # cap at max 4 probes per identity to keep balance
            with Image.open(probe_path) as img:
                emb_probe = backbone.encode(img).embedding
                all_embeddings[probe_path] = emb_probe
                probes.append((gid, ds, probe_path))

    print(f"\n[1:N SEARCH EVALUATION] Total Probes: {len(probes)} against 200 Gallery Identities")
    print("-------------------------------------------------------------------------")
    print(f"{'Probe ID':<25} | {'Dataset':<30} | {'Result':<6} | {'Top-1 Match':<25} | {'Score':<6}")
    print("-------------------------------------------------------------------------")

    rank1_hits = 0
    per_dataset_results = {}
    failures_log = []

    for gid, ds, probe_path in probes:
        probe_emb = all_embeddings[probe_path]
        scores = []
        for g_id, g_emb in gallery.items():
            sim = float(cosine_similarity(probe_emb, g_emb))
            scores.append((g_id, sim))
        scores.sort(key=lambda x: x[1], reverse=True)

        top1_id, top1_score = scores[0]
        is_hit = (top1_id == gid)
        
        if is_hit:
            rank1_hits += 1
            res_str = "HIT"
        else:
            res_str = "MISS"
            failures_log.append({
                "true_id": gid,
                "dataset": ds,
                "probe_file": os.path.basename(probe_path),
                "matched_id": top1_id,
                "score": top1_score,
                "probe_path": probe_path
            })

        # Dataset stats tracking
        ds_stats = per_dataset_results.setdefault(ds, {"hits": 0, "total": 0})
        ds_stats["total"] += 1
        if is_hit:
            ds_stats["hits"] += 1

        probe_file_short = os.path.basename(probe_path)
        print(f"{_safe(gid):<25} | {ds:<30} | {res_str:<6} | {_safe(top1_id):<25} | {top1_score:.4f}")

    total_probes = len(probes)
    overall_acc = (rank1_hits / total_probes) * 100.0 if total_probes > 0 else 0.0

    print("\n-------------------------------------------------------------------------")
    print(f"      OVERALL 1:N IDENTIFICATION RANK-1 ACCURACY: {overall_acc:.2f}% ({rank1_hits}/{total_probes})")
    print("-------------------------------------------------------------------------\n")

    print("--- ACCURACY BREAKDOWN BY DATASET CATEGORY ---")
    for ds_name, stats in per_dataset_results.items():
        acc = (stats["hits"] / stats["total"]) * 100.0
        print(f"  * {ds_name:<30}: {acc:>6.2f}% ({stats['hits']}/{stats['total']} correct)")

    # 1:1 Genuine vs Impostor Verification Pair Evaluation
    print("\n[1:1 VERIFICATION EVALUATION]")
    genuine_scores = []
    impostor_scores = []

    # Genuine pairs
    for r in records:
        gid = r["global_id"]
        photos = r["photos"][:5]
        for i in range(len(photos)):
            for j in range(i+1, len(photos)):
                if photos[i] in all_embeddings and photos[j] in all_embeddings:
                    e1 = all_embeddings[photos[i]]
                    e2 = all_embeddings[photos[j]]
                    genuine_scores.append(float(cosine_similarity(e1, e2)))

    # Impostor pairs (cross identity)
    for i in range(len(records)):
        for j in range(i+1, min(i+20, len(records))):
            p1 = records[i]["photos"][0]
            p2 = records[j]["photos"][0]
            if p1 in all_embeddings and p2 in all_embeddings:
                e1 = all_embeddings[p1]
                e2 = all_embeddings[p2]
                impostor_scores.append(float(cosine_similarity(e1, e2)))

    gen_mean = np.mean(genuine_scores) if genuine_scores else 0.0
    imp_mean = np.mean(impostor_scores) if impostor_scores else 0.0

    print(f"Genuine Pairs Count: {len(genuine_scores)}, Mean Genuine Score: {gen_mean:.4f}")
    print(f"Impostor Pairs Count: {len(impostor_scores)}, Mean Impostor Score: {imp_mean:.4f}")

    thresholds = [0.28, 0.36, 0.42]
    print("\nVerification Threshold Stats:")
    for th in thresholds:
        tar = np.mean(np.array(genuine_scores) >= th) * 100.0 if genuine_scores else 0.0
        far = np.mean(np.array(impostor_scores) >= th) * 100.0 if impostor_scores else 0.0
        print(f"  Threshold >= {th:.2f}: TAR = {tar:.2f}%, FAR = {far:.2f}%")

    print("\n=========================================================================")
    print("      FAILURE BREAKDOWN & ROOT CAUSE ANALYSIS                            ")
    print("=========================================================================")
    print(f"Total Missed Probes: {len(failures_log)} out of {total_probes}\n")

    fail_by_ds = {}
    for f in failures_log:
        fail_by_ds.setdefault(f["dataset"], []).append(f)

    for ds_name, fails in fail_by_ds.items():
        print(f"Dataset: {ds_name} ({len(fails)} failures)")
        for f in fails[:5]:
            print(f"  - Probe: {f['probe_file']} (True ID: {f['true_id']}) -> Matched: {f['matched_id']} (Sim: {f['score']:.4f})")
        if len(fails) > 5:
            print(f"  - ... and {len(fails)-5} more failures in {ds_name}")
        print()

if __name__ == "__main__":
    main()
