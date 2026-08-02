"""Forensic evidence layer: score -> calibrated likelihood ratio, with limits.

Three questions the existing engine cannot answer, and this package can:

    calibration.py   what is the weight of this evidence?
    metrics.py       is that weight trustworthy, and by how much? (Cllr)
    information.py   how much could the evidence possibly support? (bits)

The third is the guard on the other two. A reported LR that exceeds what the
observation's information content can support is a sign that the number came from
a model prior rather than from the pixels.
"""

from __future__ import annotations

from .calibration import ConditionalCalibrator, LogisticCalibrator, cross_validated_log10_lr
from .information import (
    CapacityReport,
    capacity_from_pools,
    capacity_report,
    gallery_for_rank1,
    identity_bits,
)
from .metrics import CllrReport, TippettCurve, cllr, cllr_report, pav, pav_log10_lr, tippett

__all__ = [
    "CapacityReport",
    "CllrReport",
    "ConditionalCalibrator",
    "LogisticCalibrator",
    "TippettCurve",
    "capacity_from_pools",
    "capacity_report",
    "cllr",
    "cllr_report",
    "cross_validated_log10_lr",
    "gallery_for_rank1",
    "identity_bits",
    "pav",
    "pav_log10_lr",
    "tippett",
]
