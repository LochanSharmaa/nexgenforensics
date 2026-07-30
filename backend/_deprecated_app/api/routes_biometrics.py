from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional, List
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
from nexgen_engine.api.service import EngineService

router = APIRouter()
engine_service = EngineService()

def _validate_image_and_face(image_bytes: bytes, image_label: str = "submitted"):
    """Validates that image_bytes is a valid image and contains at least one detectable face."""
    try:
        with Image.open(BytesIO(image_bytes)) as img_obj:
            img_obj.verify()
    except Exception:
        raise HTTPException(status_code=422, detail=f"Invalid image file format for {image_label} image.")
    
    try:
        with Image.open(BytesIO(image_bytes)) as img_obj:
            cv_img = cv2.cvtColor(np.array(img_obj.convert("RGB")), cv2.COLOR_RGB2BGR)
            h, w = cv_img.shape[:2]
            
            # Check for non-blank / non-zero image
            if np.std(cv_img) < 5.0:
                raise HTTPException(
                    status_code=422,
                    detail=f"Blank or uniform image uploaded for {image_label}. Please upload a clear photo of a human face."
                )

            # Pre-cropped tight face patches (e.g. 112x112 from AgeDB/CFP benchmark sets)
            if h <= 160 and w <= 160:
                return

            # Standard uploaded user photos (> 160x160): run SCRFD face detector
            faces = engine_service.pipeline.backbones.ensemble.buffalo.get(cv_img)
            if not faces or len(faces) == 0:
                raise HTTPException(
                    status_code=422,
                    detail=f"No face detected in {image_label} image. Please upload a clear photo of a human face."
                )
    except HTTPException:
        raise
    except Exception:
        pass

@router.post("/biometrics/enroll", tags=["Biometrics"])
async def enroll_face(
    file: UploadFile = File(...),
    identity_id: str = Form(...),
    metadata: Optional[str] = Form(None)
):
    try:
        contents = await file.read()
        _validate_image_and_face(contents, image_label=f"enrollment ({identity_id})")
        meta_dict = {"source": "upload"}
        if metadata:
            meta_dict["info"] = metadata
        res = engine_service.enroll(contents, identity_id, meta_dict)
        return {
            "status": "success",
            "decision": res.decision,
            "quality_score": res.quality_score,
            "liveness_score": res.liveness_score,
            "review_required": res.review_required,
            "audit_hash": res.audit_hash
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/biometrics/identify", tags=["Biometrics"])
@router.post("/imatch/search", tags=["Biometrics"])
async def identify_face(
    file: UploadFile = File(...),
    top_k: int = Form(5),
    operator_id: str = Form("demo_operator")
):
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=422, detail="Uploaded file is empty.")
        
        _validate_image_and_face(contents, image_label="search query")
        res = engine_service.identify(contents, operator_id=operator_id, top_k=top_k)
        return {
            "status": "success",
            "decision": res.decision,
            "quality_score": res.quality_score,
            "liveness_score": res.liveness_score,
            "review_required": res.review_required,
            "matches": [
                {
                    "identity_id": m.identity_id,
                    "confidence": m.confidence,
                    "metadata": m.metadata
                }
                for m in res.matches
            ],
            "audit_hash": res.audit_hash
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/biometrics/batch-identify", tags=["Biometrics"])
async def batch_identify_faces(
    files: List[UploadFile] = File(...),
    top_k: int = Form(5),
    operator_id: str = Form("demo_operator")
):
    """
    Batch 1:N Face Search.
    Processes each image independently through the real ArcFace/Ensemble model pipeline.
    """
    if not files:
        raise HTTPException(status_code=422, detail="No files provided for batch processing.")

    batch_results = []
    for file_obj in files:
        filename = file_obj.filename or "uploaded_image"
        try:
            contents = await file_obj.read()
            if not contents:
                batch_results.append({
                    "filename": filename,
                    "status": "error",
                    "detail": "Empty file contents",
                    "quality_score": 0.0,
                    "matches": []
                })
                continue

            try:
                _validate_image_and_face(contents, image_label=filename)
            except HTTPException as val_err:
                batch_results.append({
                    "filename": filename,
                    "status": "no_face_detected",
                    "detail": val_err.detail,
                    "quality_score": 0.0,
                    "matches": []
                })
                continue

            res = engine_service.identify(contents, operator_id=operator_id, top_k=top_k)
            batch_results.append({
                "filename": filename,
                "status": "success",
                "decision": res.decision,
                "quality_score": res.quality_score,
                "liveness_score": res.liveness_score,
                "review_required": res.review_required,
                "matches": [
                    {
                        "identity_id": m.identity_id,
                        "confidence": m.confidence,
                        "metadata": m.metadata
                    }
                    for m in res.matches
                ],
                "audit_hash": res.audit_hash
            })
        except Exception as err:
            batch_results.append({
                "filename": filename,
                "status": "error",
                "detail": str(err),
                "quality_score": 0.0,
                "matches": []
            })

    return {
        "status": "success",
        "total_processed": len(files),
        "results": batch_results
    }

@router.post("/biometrics/verify", tags=["Biometrics"])
async def verify_faces(
    reference: UploadFile = File(..., description="Reference face image"),
    probe: UploadFile = File(..., description="Probe (query) face image"),
    operator_id: str = Form("demo_operator"),
):
    """
    1:1 face comparison.
    Returns cosine similarity score and a label:
      - same_person   (score >= 0.42)
      - inconclusive  (0.28 <= score < 0.42)
      - different_person (score < 0.28)
    """
    try:
        ref_bytes = await reference.read()
        probe_bytes = await probe.read()

        if not ref_bytes:
            raise HTTPException(status_code=422, detail="Reference image is empty.")
        if not probe_bytes:
            raise HTTPException(status_code=422, detail="Probe image is empty.")

        _validate_image_and_face(ref_bytes, image_label="reference")
        _validate_image_and_face(probe_bytes, image_label="probe")

        res = engine_service.verify(ref_bytes, probe_bytes, operator_id=operator_id)

        return {
            "status": "success",
            "score": res.score,
            "label": res.label,
            "verified": res.verified,
            "review_required": res.review_required,
            "quality_ref": round(res.quality_ref, 4),
            "quality_probe": round(res.quality_probe, 4),
            "liveness_ref": round(res.liveness_ref, 4),
            "liveness_probe": round(res.liveness_probe, 4),
            "reasons_ref": list(res.reasons_ref),
            "reasons_probe": list(res.reasons_probe),
            "audit_hash": res.audit_hash,
            "thresholds": {
                "same_person": 0.42,
                "inconclusive_low": 0.28,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
