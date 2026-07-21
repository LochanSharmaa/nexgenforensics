"""
visualization/run_figures.py

Takes the report dict produced by pipeline.build_comparison_report() and
generates Figures 1-8 as PNG files. Only reads image bytes + JSON fields —
never re-detects or re-estimates anything.
"""
from __future__ import annotations

import cv2
from pathlib import Path

from visualization import figures as fig


def generate_all_figures(report: dict, output_dir: str) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    probe = report["_probe"]
    candidate = report["_candidate"]
    schema = report["landmarks"]["schema"]

    paths = {}

    def save(name, img):
        p = str(out / f"{name}.png")
        cv2.imwrite(p, img)
        paths[name] = p

    save("fig1_original", fig.fig1_original(probe["image_bgr"]))
    save("fig2_bounding_box", fig.fig2_bounding_box(probe["image_bgr"], report["face_detection"]))
    save("fig3_landmarks", fig.fig3_landmarks(probe["image_bgr"], report["landmarks"]))
    save("fig4_measurements", fig.fig4_measurements(probe["image_bgr"], report["landmarks"], report["measurements_continuous"]))
    save("fig5_morphological_regions", fig.fig5_morphological_regions(probe["image_bgr"], report["landmarks"], schema))

    candidate_landmarks = {
        "schema": schema,
        "points": candidate["detection"].landmarks,
    }
    save("fig6_probe_vs_candidate", fig.fig6_probe_vs_candidate(
        probe["image_bgr"], candidate["image_bgr"], report["landmarks"], candidate_landmarks,
    ))
    save("fig7_difference", fig.fig7_difference(
        probe["image_bgr"], candidate["image_bgr"], report["landmarks"], candidate_landmarks,
        report["morphological_features"], schema,
    ))
    save("fig8_quality_panel", fig.fig8_quality_panel(probe["image_bgr"], report["quality_metrics"]))

    return paths
