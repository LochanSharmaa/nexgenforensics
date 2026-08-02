#!/usr/bin/env python
"""Inventory every dataset, protocol pack and cached embedding on this machine.

    python backend/scripts/audit_assets.py

Writes docs/DATASET_INVENTORY.md. CPU only, no model load, no inference.

The point is not a file listing. It is to answer, per dataset, the three
questions that decide whether an experiment can run today:

    1. Are the IMAGES present?          -> can we embed (needs GPU)
    2. Are the EMBEDDINGS cached?       -> can we analyse now (CPU only)
    3. Are IDENTITY LABELS recoverable? -> can we measure capacity at all

Question 3 is the one that bites. A protocol pack ships pairs, not identities,
which makes an identity-disjoint impostor population impossible to construct
from it -- and without that, a capacity estimate is not merely noisy but wrong.
See nexgen_engine/forensics/information.py for the measured contamination.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

_ROOT = _BACKEND.parent
CACHE = _ROOT / "runtime" / "benchmarks" / "embeddings"
DOCS = _ROOT / "docs"

# Identity counts for the standard packs are published properties of the source
# datasets, not measurable from the .bin files (which carry pairs only).
PACK_FACTS = {
    "lfw": (6000, 5749, "Frontal, unconstrained. Saturated."),
    "cfp_ff": (7000, 500, "Frontal-frontal. Saturated; used as a harness control."),
    "cfp_fp": (7000, 500, "Frontal-profile. Pose stress."),
    "agedb_30": (6000, 568, "30-year age gap."),
    "calfw": (6000, 4025, "Cross-age, hard negatives."),
    "cplfw": (6000, 3884, "Cross-pose, hard negatives."),
}
PACK_DIRS = [
    _ROOT / "src_extracted/faces_webface_112x112/faces_webface_112x112",
    _ROOT / "src_extracted/faces_megafacetrain_112x112/faces_megafacetrain_112x112",
    _ROOT / "src_extracted/faces_umd/faces_umd",
]
ID_RE = re.compile(r"^(\d+)_\d+\.jpg$", re.IGNORECASE)


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} PB"


def dir_size(p: Path, cap: int = 400_000) -> tuple[int, int]:
    total = count = 0
    for f in p.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
            count += 1
            if count >= cap:
                break
    return total, count


def audit_packs() -> list[dict]:
    rows = []
    for name, (pairs, ids, note) in PACK_FACTS.items():
        found = [d / f"{name}.bin" for d in PACK_DIRS if (d / f"{name}.bin").exists()]
        cache = CACHE / f"{name}__w600k_r50.npz"
        rows.append(
            {
                "dataset": name,
                "images": pairs * 2,
                "pairs": pairs,
                "identities": ids,
                "packs_found": len(found),
                "pack_paths": [str(p.relative_to(_ROOT)) for p in found],
                "embeddings_cached": cache.exists(),
                "identity_labels": False,  # .bin packs carry issame only
                "note": note,
            }
        )
    return rows


def audit_tinyface() -> dict | None:
    root = _ROOT / "src_extracted/tinyface/tinyface"
    if not root.is_dir():
        return None
    test = root / "Testing_Set"
    train = root / "Training_Set"

    by_id: dict[str, int] = defaultdict(int)
    labelled = 0
    for sub in ("Gallery_Match", "Probe"):
        d = test / sub
        if d.is_dir():
            for p in d.glob("*.jpg"):
                m = ID_RE.match(p.name)
                if m:
                    by_id[m.group(1)] += 1
                    labelled += 1
    distractors = len(list((test / "Gallery_Distractor").glob("*.jpg"))) if (test / "Gallery_Distractor").is_dir() else 0
    train_ids = [d for d in train.iterdir() if d.is_dir()] if train.is_dir() else []
    train_imgs = sum(len(list(d.glob("*.jpg"))) for d in train_ids)

    return {
        "dataset": "tinyface",
        "labelled_test_images": labelled,
        "test_identities": len(by_id),
        "identities_with_2plus": sum(1 for v in by_id.values() if v >= 2),
        "distractors": distractors,
        "train_identities": len(train_ids),
        "train_images": train_imgs,
        "embeddings_cached": (CACHE / "tinyface_labelled__w600k_r50.npz").exists(),
        "distractors_cached": False,
        "identity_labels": True,
    }


def audit_qmul() -> dict | None:
    for root in (
        Path("C:/Users/hello/Downloads/QMUL-SurvFace-v1/QMUL-SurvFace"),
        _ROOT / "src_extracted/QMUL-SurvFace",
    ):
        if root.is_dir():
            train = root / "training_set"
            ids = [d for d in train.iterdir() if d.is_dir()] if train.is_dir() else []
            return {
                "dataset": "qmul_survface",
                "root": str(root),
                "train_identities": len(ids),
                "train_images": sum(len(list(d.glob("*.jpg"))) for d in ids[:6000]),
                "embeddings_cached": any(CACHE.glob("qmul*")),
                "identity_labels": True,
                "licence": "research purposes only",
            }
    return None


def audit_ijb() -> dict | None:
    tar = Path("C:/Users/hello/Downloads/ijb-testsuite.tar")
    if not tar.exists():
        return None
    return {
        "dataset": "ijb_b_c",
        "archive": str(tar),
        "size_bytes": tar.stat().st_size,
        "extracted": (_ROOT / "src_extracted/ijb").is_dir(),
        "embeddings_cached": any(CACHE.glob("ijb*")),
        "identity_labels": True,
        "note": "Complete suite: IJBB + IJBC loose_crop, meta, IJB_11.py, reference .npy",
    }


def audit_cache() -> list[dict]:
    rows = []
    if not CACHE.is_dir():
        return rows
    for f in sorted(CACHE.glob("*.npz")):
        d = np.load(f)
        rows.append(
            {
                "file": f.name,
                "size_bytes": f.stat().st_size,
                "arrays": {k: list(d[k].shape) for k in d.files},
                "has_labels": "issame" in d.files,
            }
        )
    return rows


def main() -> int:
    DOCS.mkdir(parents=True, exist_ok=True)
    packs = audit_packs()
    tf = audit_tinyface()
    qmul = audit_qmul()
    ijb = audit_ijb()
    cache = audit_cache()

    payload = {"packs": packs, "tinyface": tf, "qmul": qmul, "ijb": ijb, "cache": cache}
    (_ROOT / "runtime" / "forensics").mkdir(parents=True, exist_ok=True)
    (_ROOT / "runtime" / "forensics" / "asset_inventory.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    L = []
    a = L.append
    a("# Dataset Inventory\n")
    a("Generated by `backend/scripts/audit_assets.py`. CPU only, no inference.\n")
    a("**Read the `capacity?` column first.** It is the binding constraint on the\n"
      "science, not disk space. See `docs/CAPACITY_VALIDATION.md`.\n")

    a("\n## 1. Protocol packs (InsightFace `.bin`)\n")
    a("| dataset | images | pairs | identities | packs | cached | identity labels | capacity? |")
    a("|---|---|---|---|---|---|---|---|")
    for r in packs:
        a(
            f"| {r['dataset']} | {r['images']:,} | {r['pairs']:,} | {r['identities']:,} | "
            f"{r['packs_found']} | {'yes' if r['embeddings_cached'] else 'NO'} | "
            f"no (pairs only) | **NO** |"
        )
    a("\nProtocol packs ship an `issame` flag per pair and no identity index. An "
      "identity-disjoint impostor population therefore cannot be constructed from "
      "them, and capacity estimates from these packs are invalid regardless of "
      "sample size. Measured contamination is documented in "
      "`nexgen_engine/forensics/information.py`.\n")

    if tf:
        a("\n## 2. TinyFace\n")
        a("| property | value |")
        a("|---|---|")
        a(f"| labelled test images | {tf['labelled_test_images']:,} |")
        a(f"| test identities | {tf['test_identities']:,} |")
        a(f"| identities with >=2 captures | {tf['identities_with_2plus']:,} |")
        a(f"| **gallery distractors** | **{tf['distractors']:,}** |")
        a(f"| training identities | {tf['train_identities']:,} |")
        a(f"| training images | {tf['train_images']:,} |")
        a(f"| test embeddings cached | {'yes' if tf['embeddings_cached'] else 'NO'} |")
        a(f"| distractor embeddings cached | {'yes' if tf['distractors_cached'] else 'NO'} |")
        a("| identity labels | **yes**, from filename `<id>_<n>.jpg` |")
        a("| capacity? | **YES** -- the only asset here that supports it today |")
        a("\nTwo unexploited assets: the **Training_Set** has never been used for "
          "training (the failed fine-tune used synthetic degradation instead), and "
          "the **Gallery_Distractor** set is an unembedded open-set 1:N benchmark.\n")

    if qmul:
        a("\n## 3. QMUL-SurvFace\n")
        a(f"- root: `{qmul['root']}`")
        a(f"- training identities: {qmul['train_identities']:,}")
        a(f"- identity labels: yes (folder per identity)")
        a(f"- embeddings cached: **{'yes' if qmul['embeddings_cached'] else 'NO -- requires GPU inference'}**")
        a(f"- licence: {qmul['licence']}")
        a("- capacity? **YES once embedded.** Blocked on GPU, not on protocol.\n")

    if ijb:
        a("\n## 4. IJB-B / IJB-C\n")
        a(f"- archive: `{ijb['archive']}` ({human(ijb['size_bytes'])})")
        a(f"- extracted: {'yes' if ijb['extracted'] else 'NO'}")
        a(f"- embeddings cached: **{'yes' if ijb['embeddings_cached'] else 'NO -- requires GPU inference'}**")
        a(f"- {ijb['note']}")
        a("\nCorrects a stale claim in PROJECT_OVERVIEW/SCORECARD, which describe "
          "this as a 1.57 GB partial download. It is the complete suite.\n")

    a("\n## 5. Cached embeddings (what CPU analysis can use today)\n")
    a("| file | size | arrays | pair labels |")
    a("|---|---|---|---|")
    for r in cache:
        arrays = ", ".join(f"{k}{tuple(v)}" for k, v in r["arrays"].items())
        a(f"| {r['file']} | {human(r['size_bytes'])} | {arrays} | {'yes' if r['has_labels'] else 'no'} |")

    a("\n## 6. Missing metadata report\n")
    a("| gap | consequence | resolution |")
    a("|---|---|---|")
    a("| Protocol packs carry no identity index | Capacity uncomputable on 6 of 7 datasets | Not resolvable from the packs. Use TinyFace/QMUL/IJB, which have labels |")
    a("| QMUL embeddings not cached | No CPU analysis of the second native-LR corpus | GPU inference (~5.3k identities) |")
    a("| IJB-B/C embeddings not cached | Largest benchmark unrun | GPU inference |")
    a("| TinyFace distractors not embedded | No open-set 1:N, no valid reference population | GPU inference (153,428 images) |")
    a("| TinyFace cache lacks flip-TTA (`emb` only, no `orig`/`flip`) | TinyFace numbers not comparable to published figures produced with TTA | Re-embed with TTA, or report without and say so |")
    a("| No camera/EXIF metadata in any corpus | Forward degradation model has no ground truth to fit against | Requires the capture programme in DATA_REQUIREMENTS.md |")

    (DOCS / "DATASET_INVENTORY.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {(DOCS / 'DATASET_INVENTORY.md').relative_to(_ROOT)}")
    print(f"wrote runtime/forensics/asset_inventory.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
