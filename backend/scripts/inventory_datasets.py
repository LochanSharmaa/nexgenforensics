import os
import glob
from pathlib import Path
from PIL import Image

def scan_datasets():
    base_dirs = [Path("src_extracted"), Path("src")]
    print("=========================================================================")
    print("           REAL DISK INVENTORY OF LOCAL FACE DATASETS                    ")
    print("=========================================================================\n")
    
    # 1. AgeDB
    agedb_dir = Path("src_extracted/AgeDB/AgeDB")
    if agedb_dir.exists():
        files = glob.glob(str(agedb_dir / "*.jpg"))
        subjects = {}
        for f in files:
            parts = os.path.basename(f).split("_")
            if len(parts) >= 3:
                subjects.setdefault(parts[1], []).append(f)
        sample_img = Image.open(files[0]) if files else None
        sz = sample_img.size if sample_img else "N/A"
        print("1. AgeDB Dataset:")
        print(f"   - Path: {agedb_dir.resolve()}")
        print(f"   - Real Image Count: {len(files):,}")
        print(f"   - Real Subject Count: {len(subjects):,}")
        print(f"   - Sample Image Resolution: {sz} (Preprocessed/Cropped faces)")
        print(f"   - Sample Filenames:")
        for sf in files[:3]:
            print(f"     * {os.path.basename(sf)}")
        print()

    # 2. WebFace (112x112 aligned)
    wf_dir = Path("src_extracted/faces_webface_112x112")
    if wf_dir.exists():
        subdirs = [d for d in wf_dir.iterdir() if d.is_dir()]
        total_imgs = 0
        samples = []
        sample_img = None
        for d in subdirs:
            imgs = list(d.glob("*.jpg")) + list(d.glob("*.png"))
            total_imgs += len(imgs)
            if not samples and imgs:
                samples = [str(x) for x in imgs[:3]]
                sample_img = Image.open(imgs[0])
        sz = sample_img.size if sample_img else "N/A"
        print("2. MS1M / CASIA-WebFace (faces_webface_112x112):")
        print(f"   - Path: {wf_dir.resolve()}")
        print(f"   - Real Subject Count: {len(subdirs):,}")
        print(f"   - Real Image Count: {total_imgs:,}")
        print(f"   - Alignment Status: Pre-aligned 112x112 ArcFace format ({sz})")
        print(f"   - Sample Filenames:")
        for sf in samples:
            print(f"     * {Path(sf).relative_to(wf_dir)}")
        print()

    # 3. UMDFaces
    umd_dir = Path("src_extracted/faces_umd")
    if umd_dir.exists():
        subdirs = [d for d in umd_dir.glob("*/*") if d.is_dir()] or [d for d in umd_dir.iterdir() if d.is_dir()]
        total_imgs = 0
        samples = []
        sample_img = None
        for root, dirs, files in os.walk(umd_dir):
            jpgs = [f for f in files if f.endswith('.jpg') or f.endswith('.png')]
            total_imgs += len(jpgs)
            if len(samples) < 3 and jpgs:
                samples.append(os.path.join(root, jpgs[0]))
                if not sample_img:
                    sample_img = Image.open(os.path.join(root, jpgs[0]))
        sz = sample_img.size if sample_img else "N/A"
        print("3. UMDFaces (faces_umd):")
        print(f"   - Path: {umd_dir.resolve()}")
        print(f"   - Real Image Count: {total_imgs:,}")
        print(f"   - Sample Image Resolution: {sz} (Unconstrained in-the-wild)")
        print(f"   - Sample Filenames:")
        for sf in samples:
            print(f"     * {Path(sf).relative_to(umd_dir)}")
        print()

    # 4. MegaFace Train (faces_megafacetrain_112x112)
    mf_dir = Path("src_extracted/faces_megafacetrain_112x112")
    if mf_dir.exists():
        subdirs = [d for d in mf_dir.iterdir() if d.is_dir()]
        total_imgs = 0
        samples = []
        sample_img = None
        for d in subdirs[:1000]:
            imgs = list(d.glob("*.jpg"))
            total_imgs += len(imgs)
            if not samples and imgs:
                samples = [str(x) for x in imgs[:3]]
                sample_img = Image.open(imgs[0])
        sz = sample_img.size if sample_img else "N/A"
        print("4. MegaFace Train (faces_megafacetrain_112x112):")
        print(f"   - Path: {mf_dir.resolve()}")
        print(f"   - Real Subject Count (scanned sample): {len(subdirs):,}")
        print(f"   - Alignment Status: Pre-aligned 112x112 ({sz})")
        print(f"   - Sample Filenames:")
        for sf in samples:
            print(f"     * {Path(sf).relative_to(mf_dir)}")
        print()

    # 5. TinyFace
    tf_dir = Path("src_extracted/tinyface")
    if tf_dir.exists():
        samples = []
        total_imgs = 0
        sample_img = None
        for root, dirs, files in os.walk(tf_dir):
            jpgs = [f for f in files if f.endswith('.jpg') or f.endswith('.png')]
            total_imgs += len(jpgs)
            if len(samples) < 3 and jpgs:
                p = os.path.join(root, jpgs[0])
                samples.append(p)
                if not sample_img:
                    sample_img = Image.open(p)
        sz = sample_img.size if sample_img else "N/A"
        print("5. TinyFace Surveillance Dataset:")
        print(f"   - Path: {tf_dir.resolve()}")
        print(f"   - Real Image Count: {total_imgs:,}")
        print(f"   - Resolution Status: Low-resolution surveillance patches ({sz})")
        print(f"   - Sample Filenames:")
        for sf in samples:
            print(f"     * {Path(sf).relative_to(tf_dir)}")
        print()

    # 6. MS1M-RetinaFace
    ms1m_dir = Path("src/ms1m-retinaface-t1")
    if ms1m_dir.exists():
        total_imgs = 0
        samples = []
        sample_img = None
        for root, dirs, files in os.walk(ms1m_dir):
            jpgs = [f for f in files if f.endswith('.jpg') or f.endswith('.png')]
            total_imgs += len(jpgs)
            if len(samples) < 3 and jpgs:
                p = os.path.join(root, jpgs[0])
                samples.append(p)
                if not sample_img:
                    sample_img = Image.open(p)
        sz = sample_img.size if sample_img else "N/A"
        print("6. MS1M-RetinaFace-t1:")
        print(f"   - Path: {ms1m_dir.resolve()}")
        print(f"   - Real Image Count: {total_imgs:,}")
        print(f"   - Image Resolution: {sz}")
        print(f"   - Sample Filenames:")
        for sf in samples:
            print(f"     * {Path(sf).relative_to(ms1m_dir)}")
        print()

if __name__ == "__main__":
    scan_datasets()
