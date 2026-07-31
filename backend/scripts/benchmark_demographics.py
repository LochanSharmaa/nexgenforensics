#!/usr/bin/env python
"""
Demographic error-rate breakdown on AgeDB (Phase 3e).

WHY THIS IS SEPARATE FROM benchmark_verification.py
----------------------------------------------------
The standard agedb_30.bin protocol pack contains anonymised image pairs -- it
carries an is-same flag and nothing else. It cannot answer "does this system
fail more often on older subjects, or on women?" because the identities and
attributes were stripped when the pack was built.

The raw AgeDB image folder does carry them: filenames are
`<idx>_<Name>_<age>_<gender>.jpg`. This script therefore builds its OWN
genuine/impostor pairs from the raw folder so every pair keeps its age and
gender labels, then reports false-match and false-non-match rates per subgroup.

Because these are locally constructed pairs, the aggregate accuracy here is NOT
comparable to the published AgeDB-30 number and must not be quoted as such.
What IS meaningful is the *relative* error rate between subgroups measured on
one consistent pair set -- which is the entire point of a bias audit
(cf. NIST FRVT Part 3, which reports demographic differentials this way).

Impostor pairs are matched WITHIN a subgroup (same gender, same age bucket).
Cross-group impostor pairs are systematically easier -- a 25-year-old woman
versus a 70-year-old man is a trivial rejection -- and mixing them in would
deflate the false-match rate of every group.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from nexgen_engine.benchmarks.verification import l2n  # noqa: E402

_ROOT = _BACKEND.parent
AGEDB_DIR = _ROOT / "src_extracted/AgeDB/AgeDB"

# 0_MariaCallas_35_f.jpg
NAME_RE = re.compile(r"^(\d+)_(.+)_(\d+)_([mf])\.jpg$", re.IGNORECASE)

AGE_BUCKETS = [(0, 25), (26, 40), (41, 55), (56, 200)]


def bucket_of(age: int) -> str:
    for lo, hi in AGE_BUCKETS:
        if lo <= age <= hi:
            return f"{lo}-{hi}" if hi < 200 else f"{lo}+"
    return "unknown"


def scan() -> list[dict]:
    records = []
    for p in sorted(AGEDB_DIR.glob("*.jpg")):
        m = NAME_RE.match(p.name)
        if not m:
            continue
        _, name, age, gender = m.groups()
        records.append(
            {
                "path": p,
                "identity": name,
                "age": int(age),
                "gender": gender.lower(),
                "bucket": bucket_of(int(age)),
            }
        )
    return records


def build_pairs(records: list[dict], per_group: int, seed: int = 0):
    """Genuine pairs within an identity; impostor pairs within a subgroup."""
    rng = np.random.default_rng(seed)
    by_identity = defaultdict(list)
    for r in records:
        by_identity[r["identity"]].append(r)

    # subgroup = (gender, age bucket) of the FIRST image in the pair
    genuine = []
    for imgs in by_identity.values():
        if len(imgs) < 2:
            continue
        idx = rng.permutation(len(imgs))
        for i in range(0, len(idx) - 1, 2):
            a, b = imgs[idx[i]], imgs[idx[i + 1]]
            genuine.append((a, b, True))

    by_group = defaultdict(list)
    for r in records:
        by_group[(r["gender"], r["bucket"])].append(r)

    impostor = []
    for group, imgs in by_group.items():
        if len(imgs) < 2:
            continue
        made, guard = 0, 0
        while made < per_group and guard < per_group * 20:
            guard += 1
            i, j = rng.integers(0, len(imgs), 2)
            if imgs[i]["identity"] == imgs[j]["identity"]:
                continue
            impostor.append((imgs[i], imgs[j], False))
            made += 1

    return genuine + impostor


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="glintr100")
    ap.add_argument("--per-group", type=int, default=4000,
                    help="impostor pairs per (gender, age-bucket) subgroup")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Operating threshold to evaluate at (e.g. the deployed 0.20). "
        "Omit to anchor at FMR=0.1% on this pair set instead.",
    )
    ap.add_argument("--out", default=str(_ROOT / "runtime/benchmarks/demographics.json"))
    args = ap.parse_args()

    from benchmark_verification import load_recognizer  # noqa: PLC0415

    records = scan()
    print(f"AgeDB images with parseable labels: {len(records)}")
    print(f"distinct identities: {len({r['identity'] for r in records})}")

    pairs = build_pairs(records, args.per_group)
    print(f"pairs built: {len(pairs)} "
          f"(genuine={sum(1 for p in pairs if p[2])}, "
          f"impostor={sum(1 for p in pairs if not p[2])})")

    # unique images actually referenced
    needed = {}
    for a, b, _ in pairs:
        needed[a["path"]] = a
        needed[b["path"]] = b
    paths = list(needed)
    print(f"unique images to embed: {len(paths)}")

    model = load_recognizer(args.model)
    embs = np.empty((len(paths), 512), dtype=np.float32)
    for i in range(0, len(paths), args.batch_size):
        chunk = paths[i : i + args.batch_size]
        imgs = []
        for p in chunk:
            # cv2.imread() cannot open non-ASCII paths on Windows -- it returns
            # None rather than raising. Decode from bytes so any filename works.
            im = cv2.imdecode(np.frombuffer(Path(p).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            if im is None:
                raise ValueError(f"failed to decode {p}")
            imgs.append(cv2.resize(im, (112, 112)) if im.shape[:2] != (112, 112) else im)
        # flip-TTA, matching the main benchmark
        e = np.asarray(model.get_feat(imgs)) + np.asarray(model.get_feat([x[:, ::-1] for x in imgs]))
        embs[i : i + len(chunk)] = e
        if (i // args.batch_size) % 20 == 0:
            print(f"  {min(i + args.batch_size, len(paths))}/{len(paths)}", end="\r", flush=True)
    embs = l2n(embs.astype(np.float64))
    pos = {p: i for i, p in enumerate(paths)}

    scores = np.array([float(embs[pos[a["path"]]] @ embs[pos[b["path"]]]) for a, b, _ in pairs])
    labels = np.array([lab for _, _, lab in pairs], dtype=bool)

    # One global threshold. A per-group threshold would hide exactly the
    # differential we are trying to expose: in deployment every subgroup is
    # judged by the same cut-point.
    #
    # TWO views are reported, because they answer different questions:
    #   operating -- the threshold the DEPLOYED system actually decides at.
    #                This is the one that describes real-world harm.
    #   fmr_1e3   -- the threshold that would yield FMR=0.1% on this pair set.
    #                Useful for comparing models on equal footing, but it is
    #                NOT what the system does in production.
    imp = np.sort(scores[~labels])[::-1]
    fmr_thr = float(imp[max(0, int(0.001 * imp.size) - 1)])

    if args.threshold is not None:
        thr = float(args.threshold)
        thr_kind = "operating (deployed decision threshold)"
    else:
        thr = fmr_thr
        thr_kind = "anchored at FMR=0.1% on this pair set"
    print(f"\nthreshold in use: {thr:.4f}  [{thr_kind}]")
    print(f"  (for reference, FMR=0.1% on this set falls at {fmr_thr:.4f})")

    def rates(mask):
        g = mask & labels
        i = mask & ~labels
        fnmr = float((scores[g] <= thr).mean()) if g.sum() else float("nan")
        fmr = float((scores[i] > thr).mean()) if i.sum() else float("nan")
        return g.sum(), i.sum(), fnmr, fmr

    groups = {}
    a_gender = np.array([a["gender"] for a, _, _ in pairs])
    a_bucket = np.array([a["bucket"] for a, _, _ in pairs])

    print(f"\n{'subgroup':<22} {'genuine':>8} {'impostor':>9} {'FNMR%':>8} {'FMR%':>8}")
    print("-" * 60)
    for label, mask in [
        ("ALL", np.ones(len(pairs), bool)),
        *[(f"gender={g}", a_gender == g) for g in sorted(set(a_gender))],
        *[(f"age={b}", a_bucket == b) for b in sorted(set(a_bucket))],
    ]:
        ng, ni, fnmr, fmr = rates(mask)
        groups[label] = {"genuine": int(ng), "impostor": int(ni),
                         "fnmr": fnmr, "fmr": fmr}
        print(f"{label:<22} {ng:>8} {ni:>9} {fnmr * 100:>7.2f} {fmr * 100:>7.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"model": args.model, "threshold_used": thr, "threshold_kind": thr_kind,
         "threshold_at_fmr_1e3": fmr_thr,
         "note": "locally constructed pairs; not comparable to published AgeDB-30",
         "groups": groups}, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
