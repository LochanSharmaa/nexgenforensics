import os
import glob
import math
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class MultiDatasetFaceDataset(Dataset):
    """
    Data loader for local face datasets:
    - AgeDB
    - LFW
    - CFP (frontal & profile)
    - TinyFace (surveillance patches)
    - WebFace / Aligned
    """
    def __init__(self, base_dir="src_extracted", image_size=112, is_train=True, max_identities=500):
        self.image_size = image_size
        self.samples = []  # list of (image_path, class_idx)
        self.class_to_idx = {}

        # 1. AgeDB
        agedb_imgs = glob.glob(os.path.join(base_dir, "AgeDB", "AgeDB", "*.jpg"))
        agedb_subjs = {}
        for f in agedb_imgs:
            parts = os.path.basename(f).split("_")
            if len(parts) >= 3:
                agedb_subjs.setdefault(f"agedb_{parts[1]}", []).append(f)

        # 2. CFP
        cfp_dir = Path(base_dir) / "archive (4)" / "cfp-dataset" / "Data" / "Images"
        cfp_subjs = {}
        if cfp_dir.exists():
            for d in cfp_dir.iterdir():
                if d.is_dir():
                    f_imgs = [str(x) for x in (d / "frontal").glob("*.jpg")]
                    p_imgs = [str(x) for x in (d / "profile").glob("*.jpg")]
                    if f_imgs or p_imgs:
                        cfp_subjs[f"cfp_{d.name}"] = f_imgs + p_imgs

        # 3. TinyFace
        tf_probe = Path(base_dir) / "tinyface" / "tinyface" / "Testing_Set" / "Probe"
        tf_gallery = Path(base_dir) / "tinyface" / "tinyface" / "Testing_Set" / "Gallery_Match"
        tf_subjs = {}
        if tf_probe.exists() and tf_gallery.exists():
            for f in tf_probe.glob("*.jpg"):
                subj_id = f.stem.split("_")[0]
                g_matches = [str(x) for x in tf_gallery.glob(f"{subj_id}_*.jpg")]
                if g_matches:
                    tf_subjs[f"tf_{subj_id}"] = [str(f)] + g_matches

        # Merge subjects
        all_subjs = {}
        all_subjs.update(agedb_subjs)
        all_subjs.update(cfp_subjs)
        all_subjs.update(tf_subjs)

        # Filter subjects with >= 2 images
        selected = {k: v for k, v in all_subjs.items() if len(v) >= 2}
        sorted_keys = sorted(list(selected.keys()))[:max_identities]

        for idx, key in enumerate(sorted_keys):
            self.class_to_idx[key] = idx
            for img_path in selected[key]:
                self.samples.append((img_path, idx))

        self.num_classes = len(sorted_keys)

        if is_train:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(),
                T.RandomApply([
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))
                ], p=0.4),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path).convert("RGB") as img:
            tensor_img = self.transform(img)
        return tensor_img, label

if __name__ == "__main__":
    ds = MultiDatasetFaceDataset()
    print(f"Loaded dataset: {len(ds)} samples across {ds.num_classes} identities.")
