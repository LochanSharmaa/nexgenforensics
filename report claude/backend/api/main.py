"""
api/main.py

Minimal FastAPI app for the investigator-facing frontend:
  POST /compare              -> upload probe+candidate images, run 1:1 pipeline
  POST /compare_ranking      -> upload probe + multiple candidates, run 1:N ranking
  GET  /report/{id}/json
  GET  /report/{id}/pdf
  GET  /report/{id}/figures/{name}
  GET  /health

Run from backend/: uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
import uuid
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import build_comparison_report, build_candidate_ranking
from visualization.run_figures import generate_all_figures
from pdf.builder import build_pdf

APP_DATA = Path(__file__).parent.parent.parent / "sample_data" / "api_runs"
APP_DATA.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("api")

app = FastAPI(
    title="Next Gen Forensics — Comparison Reporting API",
    description=(
        "AI-assisted facial similarity analysis API. Results are investigative "
        "support material only — not a determination of identity."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_upload(upload: UploadFile, dest: Path) -> None:
    with open(dest, "wb") as f:
        shutil.copyfileobj(upload.file, f)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/compare", summary="1:1 facial comparison")
async def compare(
    probe: UploadFile = File(..., description="Probe face image"),
    candidate: UploadFile = File(..., description="Candidate face image"),
    case_id: str = Form("", description="Optional case identifier"),
    requesting_agency: str = Form("", description="Optional requesting agency name"),
):
    """
    Run the full 1:1 comparison pipeline:
    detection → landmarks → quality → measurements → embedding → similarity → report
    """
    run_id = str(uuid.uuid4())
    run_dir = APP_DATA / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    probe_path = run_dir / f"probe_{probe.filename}"
    candidate_path = run_dir / f"candidate_{candidate.filename}"
    _save_upload(probe, probe_path)
    _save_upload(candidate, candidate_path)

    try:
        report = build_comparison_report(
            str(probe_path), str(candidate_path),
            case_id=case_id, requesting_agency=requesting_agency,
        )
    except Exception as e:
        logger.exception("Pipeline error in /compare run_id=%s", run_id)
        raise HTTPException(status_code=422, detail=f"Pipeline failed: {e}")

    core_report = {k: v for k, v in report.items() if not k.startswith("_")}
    with open(run_dir / "report.json", "w") as f:
        json.dump(core_report, f, indent=2, default=str)

    try:
        fig_dir = run_dir / "figures"
        figure_paths = generate_all_figures(report, str(fig_dir))
    except Exception as e:
        logger.warning("Figure generation failed for run_id=%s: %s", run_id, e)
        figure_paths = {}

    try:
        pdf_path = run_dir / "report.pdf"
        build_pdf(core_report, figure_paths, str(pdf_path))
    except Exception as e:
        logger.warning("PDF generation failed for run_id=%s: %s", run_id, e)
        pdf_path = None

    return JSONResponse({
        "run_id": run_id,
        "report_id": core_report["administrative"]["report_id"],
        "similarity": core_report["similarity_metrics"],
        "figures": list(figure_paths.keys()),
        "json_url": f"/report/{run_id}/json",
        "pdf_url": f"/report/{run_id}/pdf" if pdf_path and pdf_path.exists() else None,
    })


@app.post("/compare_ranking", summary="1:N candidate ranking")
async def compare_ranking(
    probe: UploadFile = File(..., description="Probe face image"),
    candidates: List[UploadFile] = File(..., description="One or more candidate images"),
    case_id: str = Form("", description="Optional case identifier"),
    top_k: int = Form(20, description="Maximum number of ranked results"),
):
    """
    Compare a probe image against multiple candidate images and return a
    ranked similarity list sorted by score (descending).

    This endpoint does NOT generate a PDF — it returns lightweight JSON
    suitable for driving the ranked results UI panel.
    """
    run_id = str(uuid.uuid4())
    run_dir = APP_DATA / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    probe_path = run_dir / f"probe_{probe.filename}"
    _save_upload(probe, probe_path)

    candidate_paths = []
    for i, cand in enumerate(candidates):
        dest = run_dir / f"candidate_{i:03d}_{cand.filename}"
        _save_upload(cand, dest)
        candidate_paths.append(str(dest))

    try:
        ranking = build_candidate_ranking(
            str(probe_path),
            candidate_paths,
            case_id=case_id,
            top_k=top_k,
        )
    except Exception as e:
        logger.exception("Pipeline error in /compare_ranking run_id=%s", run_id)
        raise HTTPException(status_code=422, detail=f"Ranking failed: {e}")

    ranking_path = run_dir / "ranking.json"
    with open(ranking_path, "w") as f:
        json.dump(ranking, f, indent=2, default=str)

    return JSONResponse({
        "run_id": run_id,
        "probe_id": ranking["probe_id"],
        "case_id": ranking["case_id"],
        "total_candidates": ranking["total_candidates"],
        "rankings": ranking["rankings"],
        "json_url": f"/report/{run_id}/ranking",
    })


@app.get("/report/{run_id}/json", summary="Download JSON report")
def get_json(run_id: str):
    path = APP_DATA / run_id / "report.json"
    if not path.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(path, media_type="application/json")


@app.get("/report/{run_id}/ranking", summary="Download ranking JSON")
def get_ranking(run_id: str):
    path = APP_DATA / run_id / "ranking.json"
    if not path.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(path, media_type="application/json")


@app.get("/report/{run_id}/pdf", summary="Download PDF report")
def get_pdf(run_id: str):
    path = APP_DATA / run_id / "report.pdf"
    if not path.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(path, media_type="application/pdf", filename="ForensicComparisonReport.pdf")


@app.get("/report/{run_id}/figures/{name}", summary="Fetch a figure image")
def get_figure(run_id: str, name: str):
    path = APP_DATA / run_id / "figures" / f"{name}.png"
    if not path.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(path, media_type="image/png")


@app.get("/health", summary="Health check")
def health():
    return {"status": "ok", "version": "1.0.0"}
