"""
Benchmark via the LIVE backend HTTP API (avoids ONNX multi-process lock).
25 identities from AgeDB, 1:N rank-1 + 1:1 genuine/impostor stats.
"""
import glob, os, json, requests
import numpy as np

BASE = "http://127.0.0.1:8000/api"

def enroll(path, identity_id):
    with open(path, "rb") as f:
        r = requests.post(f"{BASE}/biometrics/enroll",
                          files={"file": f},
                          data={"identity_id": identity_id})
    r.raise_for_status()
    return r.json()

def identify(path, top_k=5):
    with open(path, "rb") as f:
        r = requests.post(f"{BASE}/biometrics/identify",
                          files={"file": f},
                          data={"top_k": str(top_k)})
    r.raise_for_status()
    return r.json()

def verify(path_a, path_b):
    with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
        r = requests.post(f"{BASE}/biometrics/verify",
                          files={"reference": fa, "probe": fb})
    r.raise_for_status()
    return r.json()

def main():
    imgs = glob.glob("src_extracted/AgeDB/AgeDB/*.jpg")
    subjs = {}
    for f in imgs:
        name = os.path.basename(f)
        parts = name.split("_")
        if len(parts) >= 3:
            subjs.setdefault(parts[1], []).append(f)

    # Select 25 identities with >=2 images
    selected_keys = [k for k, v in subjs.items() if len(v) >= 2][:25]
    print(f"Selected {len(selected_keys)} identities: {selected_keys}")

    # ----- 1:N: Enroll first photo, probe with second -----
    print("\n[1:N] Enrolling gallery...")
    for sid in selected_keys:
        res = enroll(subjs[sid][0], sid)
        print(f"  Enrolled {sid}: {res['decision']}")

    print("\n[1:N] Running probe searches...")
    rank1_hits, total_probes = 0, 0
    for sid in selected_keys:
        for probe_path in subjs[sid][1:3]:   # up to 2 probes each
            total_probes += 1
            res = identify(probe_path, top_k=5)
            top_match = res["matches"][0]["identity_id"] if res["matches"] else "NONE"
            hit = top_match == sid
            rank1_hits += int(hit)
            score = res["matches"][0]["confidence"] if res["matches"] else 0.0
            print(f"  Probe {os.path.basename(probe_path)}: top={top_match} ({'HIT' if hit else 'MISS'}) score={score:.4f}")

    rank1_acc = rank1_hits / total_probes * 100.0
    print(f"\n--- 1:N RANK-1 ACCURACY: {rank1_hits}/{total_probes} = {rank1_acc:.2f}% ---")

    # ----- 1:1: Genuine and impostor pairs -----
    print("\n[1:1] Computing genuine / impostor scores...")
    genuine_scores, impostor_scores = [], []

    # Genuine: consecutive images of same identity
    for sid in selected_keys:
        photos = subjs[sid][:4]
        for i in range(len(photos)):
            for j in range(i+1, len(photos)):
                res = verify(photos[i], photos[j])
                genuine_scores.append(res["score"])

    # Impostor: first photo of adjacent identity pairs
    for i in range(len(selected_keys) - 1):
        a = selected_keys[i]; b = selected_keys[i+1]
        res = verify(subjs[a][0], subjs[b][0])
        impostor_scores.append(res["score"])

    gen_arr  = np.array(genuine_scores)
    imp_arr  = np.array(impostor_scores)

    print(f"\n--- 1:1 VERIFICATION STATS ({len(gen_arr)} genuine / {len(imp_arr)} impostor pairs) ---")
    print(f"Genuine  mean={gen_arr.mean():.4f}  std={gen_arr.std():.4f}  min={gen_arr.min():.4f}  max={gen_arr.max():.4f}")
    print(f"Impostor mean={imp_arr.mean():.4f}  std={imp_arr.std():.4f}  min={imp_arr.min():.4f}  max={imp_arr.max():.4f}")

    print("\nThreshold  TAR      FAR")
    for th in [0.28, 0.36, 0.42]:
        tar = (gen_arr >= th).mean() * 100
        far = (imp_arr >= th).mean() * 100
        print(f"  {th:.2f}     {tar:5.1f}%   {far:5.1f}%")

    print("\n--- README CLAIMED: genuine ~0.4907, impostor ~0.0422, rank-1 ~92% ---")
    print(f"--- TODAY'S REAL:   genuine  {gen_arr.mean():.4f},  impostor  {imp_arr.mean():.4f},  rank-1  {rank1_acc:.1f}% ---")

if __name__ == "__main__":
    main()
