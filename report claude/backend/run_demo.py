"""
run_demo.py

End-to-end pipeline demo: generates two synthetic sample face images (if not
already present), runs the full pipeline, writes JSON + figures + PDF to
sample_data/output/.

Run: python run_demo.py   (from inside backend/)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline import build_comparison_report
from visualization.run_figures import generate_all_figures
from pdf.builder import build_pdf

SAMPLE_DIR = Path(__file__).parent.parent / "sample_data"
OUTPUT_DIR = SAMPLE_DIR / "output"


def ensure_sample_images():
    from PIL import Image, ImageDraw
    probe_path = SAMPLE_DIR / "probe.png"
    candidate_path = SAMPLE_DIR / "candidate.png"

    def make_face(path, seed_offset=0):
        img = Image.new("RGB", (600, 800), (222, 210, 195))
        draw = ImageDraw.Draw(img)
        # very simple stylized face placeholder — replace with real evidence images
        draw.ellipse([120, 150, 480, 650], fill=(235, 200, 170))
        draw.ellipse([190 + seed_offset, 320, 240 + seed_offset, 360], fill=(60, 40, 30))
        draw.ellipse([360 - seed_offset, 320, 410 - seed_offset, 360], fill=(60, 40, 30))
        draw.polygon([(300, 380), (280, 480), (320, 480)], fill=(210, 170, 140))
        draw.arc([230, 500, 370, 570], start=20, end=160, fill=(120, 60, 60), width=6)
        img.save(path)

    if not probe_path.exists():
        make_face(probe_path, seed_offset=0)
    if not candidate_path.exists():
        make_face(candidate_path, seed_offset=15)

    return str(probe_path), str(candidate_path)


def main():
    probe_path, candidate_path = ensure_sample_images()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Running pipeline...")
    report = build_comparison_report(
        probe_path, candidate_path,
        case_id="DEMO-0001", requesting_agency="Demo Agency",
    )

    core_report = {k: v for k, v in report.items() if not k.startswith("_")}
    json_path = OUTPUT_DIR / "comparison_report.json"
    with open(json_path, "w") as f:
        json.dump(core_report, f, indent=2, default=str)
    print(f"Wrote JSON: {json_path}")

    fig_dir = OUTPUT_DIR / "figures"
    figure_paths = generate_all_figures(report, str(fig_dir))
    print(f"Wrote {len(figure_paths)} figures to {fig_dir}")

    pdf_path = OUTPUT_DIR / "ForensicComparisonReport.pdf"
    build_pdf(core_report, figure_paths, str(pdf_path))
    print(f"Wrote PDF: {pdf_path}")


if __name__ == "__main__":
    main()
