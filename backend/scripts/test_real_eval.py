import sys
from pathlib import Path
sys.path.insert(0, str(Path("backend").resolve()))

from nexgen_engine.api.service import EngineService

def test_pipeline():
    service = EngineService()
    
    # 1:1 Verification Test
    print("\n--- 1:1 VERIFY TEST ---")
    same_img1 = Path("src_extracted/AgeDB/AgeDB/10001_GoldieHawn_23_f.jpg").read_bytes()
    same_img2 = Path("src_extracted/AgeDB/AgeDB/10002_GoldieHawn_24_f.jpg").read_bytes()
    diff_img  = Path("src_extracted/AgeDB/AgeDB/1000_StephenHawking_1_m.jpg").read_bytes()
    
    same_res = service.verify(same_img1, same_img2)
    print(f"Same Person (Goldie Hawn age 23 vs 24): Score = {same_res.score:.6f}, Label = {same_res.label}, Verified = {same_res.verified}")
    
    diff_res = service.verify(same_img1, diff_img)
    print(f"Diff Person (Goldie Hawn vs Stephen Hawking): Score = {diff_res.score:.6f}, Label = {diff_res.label}, Verified = {diff_res.verified}")

    # 1:N Gallery Search Test
    print("\n--- 1:N GALLERY SEARCH TEST ---")
    # Enroll subjects
    gallery_subjects = [
        ("GoldieHawn_1", "src_extracted/AgeDB/AgeDB/10001_GoldieHawn_23_f.jpg"),
        ("GoldieHawn_2", "src_extracted/AgeDB/AgeDB/10005_GoldieHawn_28_f.jpg"),
        ("StephenHawking", "src_extracted/AgeDB/AgeDB/1000_StephenHawking_1_m.jpg"),
        ("MariaCallas", "src_extracted/AgeDB/AgeDB/0_MariaCallas_35_f.jpg"),
        ("GlennClose", "src_extracted/AgeDB/AgeDB/10000_GlennClose_62_f.jpg")
    ]
    
    for subj_id, img_path in gallery_subjects:
        res = service.enroll(Path(img_path).read_bytes(), subj_id)
        print(f"Enrolled {subj_id}: decision={res.decision}, quality={res.quality_score:.4f}")
        
    # Probe search with another Goldie Hawn photo (age 30)
    probe_img = Path("src_extracted/AgeDB/AgeDB/10007_GoldieHawn_30_f.jpg").read_bytes()
    search_res = service.identify(probe_img, top_k=5)
    print(f"\nProbe Search for Goldie Hawn (age 30): Decision = {search_res.decision}")
    print("Ranked candidates:")
    for idx, match in enumerate(search_res.matches, 1):
        print(f"  {idx}. Identity: {match.identity_id}, Score: {match.confidence:.6f}")

if __name__ == "__main__":
    test_pipeline()
