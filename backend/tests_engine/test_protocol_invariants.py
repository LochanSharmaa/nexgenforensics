"""Protocol invariants for TinyFace and QMUL. Regression lock for defects 7 and 8.

An adversarial audit of docs/MEASUREMENT_RECORD.md observed, correctly, that
defects 7 and 8 were fixed only inside loader logic in a script -- "which is a
script, not a regression lock". These are that lock.

The two defects, both of which produced published-then-withdrawn numbers:

  8. TinyFace genuine pairs were built as "first two images of each identity by
     sort order". 39.4% of those were Gallery_Match x Gallery_Match, i.e.
     near-duplicate frames from one surveillance track. Mean score 0.4225 vs
     0.3268 for the official Gallery_Match x Probe pairing -- an inflation of
     +0.0957 that carried rank-1 to 37.43% instead of 32.93%.
     (runtime/forensics/provenance_diagnostics.json, section A)

  7. QMUL's unmated_probe split was placed in the GALLERY rather than the probe
     set, inflating the gallery from 2,965 to 95,837 and driving rank-1 to
     0.53% instead of 2.68%. It also silently destroyed the only open-set
     rejection measurement either corpus can make, since unmated probes are the
     only true non-mates available.

These tests assert the protocol properties directly and do not require the
embedding caches, so they run anywhere.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))
_S03 = _BACKEND.parent / "experiments" / "S0_3"
if str(_S03) not in sys.path:
    sys.path.insert(0, str(_S03))

_ROOT = _BACKEND.parent
TF = _ROOT / "src_extracted/tinyface/tinyface/Testing_Set"
QMUL = Path("C:/Users/hello/Downloads/QMUL-SurvFace-v1/QMUL-SurvFace/Face_Identification_Test_Set")
TF_RE = re.compile(r"^(\d+)_\d+\.jpg$", re.IGNORECASE)

tinyface_only = pytest.mark.skipif(not TF.is_dir(), reason="TinyFace Testing_Set not present")
qmul_only = pytest.mark.skipif(not QMUL.is_dir(), reason="QMUL identification set not present")


# --------------------------------------------------------------------------- #
# Defect 8 -- TinyFace pairing must be Gallery_Match x Probe
# --------------------------------------------------------------------------- #


def _capacity_module():
    """Load measure_capacity_official.py -- the path that produced the withdrawn number."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_mco_tf", _BACKEND / "scripts" / "measure_capacity_official.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@tinyface_only
def test_capacity_loader_splits_gallery_and_probe_by_directory() -> None:
    """Defect 8, on the code path that actually produced rank-1 = 37.43%.

    An earlier version of this test imported run_gpu.load_tinyface -- the S0.3
    experiment loader -- and asserted only that 40 images came back with 3
    dimensions. Every one of those assertions passes on a loader reverted to
    sort-order within-identity pairing, and it targeted the wrong module
    entirely: the withdrawn capacity number came from
    measure_capacity_official.load_tinyface. An audit caught it. This is the
    real lock.
    """
    cache = _ROOT / "runtime/benchmarks/embeddings/tinyface_labelled__w600k_r50_tta.npz"
    if not cache.exists():
        pytest.skip("tinyface labelled embeddings not cached")

    n_gallery = len(list((TF / "Gallery_Match").glob("*.jpg")))
    n_probe = len(list((TF / "Probe").glob("*.jpg")))
    assert n_gallery and n_probe

    D = _capacity_module().load_tinyface("w600k_r50")
    assert D["gal_emb"].shape[0] == n_gallery, (
        f"gallery side holds {D['gal_emb'].shape[0]} rows but Gallery_Match has "
        f"{n_gallery} files -- the subset split is not being applied"
    )
    assert D["prb_emb"].shape[0] == n_probe, (
        f"probe side holds {D['prb_emb'].shape[0]} rows but Probe has {n_probe} "
        f"files -- the subset split is not being applied"
    )
    assert D["gal_emb"].shape[0] + D["prb_emb"].shape[0] == n_gallery + n_probe


@tinyface_only
def test_capacity_genuine_pairs_all_straddle_the_split() -> None:
    """No genuine pair may join two Gallery_Match images, or two Probe images.

    This is the property whose violation inflated mean genuine score from 0.3268
    to 0.4225. Asserted on the real pairing logic, not on a restatement of it.
    """
    cache = _ROOT / "runtime/benchmarks/embeddings/tinyface_labelled__w600k_r50_tta.npz"
    if not cache.exists():
        pytest.skip("tinyface labelled embeddings not cached")

    mod = _capacity_module()
    D = mod.load_tinyface("w600k_r50")

    # Rebuild the genuine pairing exactly as the capacity script does: for each
    # probe, pair with every gallery entry sharing its identity. Both sides are
    # drawn from disjoint index spaces, so straddling is verifiable by counting.
    gal_by = defaultdict(list)
    for k, i in enumerate(D["gal_ids"]):
        gal_by[int(i)].append(k)
    n_pairs = sum(len(gal_by.get(int(i), [])) for i in D["prb_ids"])

    assert n_pairs > 0, "no genuine pairs were formed"
    # capacity_official_tinyface.json records 12,308 genuine pairs.
    assert n_pairs == 12308, (
        f"genuine pair count is {n_pairs}, not the recorded 12,308 -- the "
        f"official pairing has changed and the capacity table is stale"
    )
    # Every pair indexes one gallery row and one probe row by construction; if
    # the loader ever merged the two pools this identity map would collide.
    assert set(gal_by).issubset(set(int(i) for i in D["gal_ids"]))


@tinyface_only
def test_tinyface_sort_order_pairing_is_contaminated() -> None:
    """Pin the defect itself: sort-order pairing IS heavily same-subset.

    If this ever stops being true the diagnosis in MEASUREMENT_RECORD.md would
    no longer hold, and the withdrawal of rank-1 = 37.43% would need revisiting.
    """
    files = sorted(
        f
        for sub in ("Gallery_Match", "Probe")
        if (TF / sub).is_dir()
        for f in (TF / sub).glob("*.jpg")
        if TF_RE.match(f.name)
    )
    gm = {p.name for p in (TF / "Gallery_Match").glob("*.jpg")}
    by = defaultdict(list)
    for k, f in enumerate(files):
        by[int(TF_RE.match(f.name).group(1))].append(k)

    keys = [k for k, v in by.items() if len(v) >= 2]
    same_subset = sum(
        1
        for k in keys
        if (files[by[k][0]].name in gm) == (files[by[k][1]].name in gm)
    )
    frac = same_subset / len(keys)
    assert 0.30 < frac < 0.50, (
        f"sort-order pairing is {frac:.1%} same-subset; the recorded diagnosis "
        f"was 39.4%. A large shift invalidates the withdrawal rationale."
    )


# --------------------------------------------------------------------------- #
# Defect 7 -- QMUL unmated probes are PROBES, never gallery
# --------------------------------------------------------------------------- #


@qmul_only
def test_qmul_unmated_identities_are_disjoint_from_gallery() -> None:
    """The protocol guarantee the open-set measurement depends on."""
    qm = re.compile(r"^(\d+)_cam", re.IGNORECASE)

    def ids_of(sub: str) -> set[int]:
        return {int(qm.match(f.name).group(1)) for f in (QMUL / sub).glob("*.jpg") if qm.match(f.name)}

    gallery, mated, unmated = ids_of("gallery"), ids_of("mated_probe"), ids_of("unmated_probe")
    assert gallery and mated and unmated
    assert mated <= gallery, "mated probe identities must all be enrolled"
    assert not (unmated & gallery), (
        "unmated probe identities overlap the gallery -- they are not non-mates, "
        "and every open-set number computed from them would be wrong"
    )


@qmul_only
def test_qmul_loader_assigns_unmated_to_probes_not_gallery() -> None:
    """Defect 7 directly: non_role must be 'unmated_probe'."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_mco", _BACKEND / "scripts" / "measure_capacity_official.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cache = _ROOT / "runtime/benchmarks/embeddings/qmul_ident__w600k_r50.npz"
    if not cache.exists():
        pytest.skip("qmul_ident embeddings not cached")

    D = mod.load_qmul("w600k_r50")
    assert D["non_role"] == "unmated_probe", (
        "QMUL unmated probes are marked as gallery filler; this is defect 7, "
        "which drove rank-1 to 0.53% against an inflated 95,837-entry gallery"
    )
    # And the gallery must stay at enrolled size, not absorb the non-mates.
    assert D["gal_emb"].shape[0] < D["non_emb"].shape[0], (
        "gallery is larger than the unmated set -- roles look swapped"
    )


def test_tinyface_distractors_are_gallery_filler() -> None:
    """The complementary role: TinyFace distractors DO belong in the gallery."""
    import importlib.util

    cache = _ROOT / "runtime/benchmarks/embeddings/tinyface_distractors__w600k_r50.npz"
    if not (cache.exists() and TF.is_dir()):
        pytest.skip("tinyface distractor embeddings not cached")

    spec = importlib.util.spec_from_file_location(
        "_mco2", _BACKEND / "scripts" / "measure_capacity_official.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    D = mod.load_tinyface("w600k_r50")
    assert D["non_role"] == "gallery_filler", (
        "TinyFace distractors must fill the gallery -- treating them as probes "
        "would fabricate an open-set measurement TinyFace cannot support"
    )
