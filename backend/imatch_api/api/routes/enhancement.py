"""Forensic image enhancement endpoints.

    GET  /api/imatch/enhance/status              what this host can run, and why not
    POST /api/imatch/enhance/analyze             measure only; produces no pixels
    POST /api/imatch/enhance                     run, store separately, audit
    GET  /api/imatch/enhance/{id}                the run record
    GET  /api/imatch/enhance/{id}/image          original | enhanced, as stored
    POST /api/imatch/enhance/{id}/recognise      A/B: original and enhanced searched

THE ORIGINAL IS ALWAYS PRIMARY. ``/recognise`` runs the existing recogniser
twice and returns both ranked lists. The original's result is marked ``primary``
and is the only one written to the case record without an examiner explicitly
adjudicating it. ``EngineService.encode()`` is not touched: both sides call it
with different bytes, which is what keeps enhancement explicit and keeps
enrolment out of it entirely.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from sqlmodel import Session, select

from nexgen_engine.inference.pipeline import InvalidImageError, NoFaceDetectedError

from ...core.config import Settings, get_settings
from ...core.dependencies import Principal, client_context, enforce_search_rate_limit
from ...db.models import Case, EnhancementRun, SearchRun
from ...db.session import get_session
from ...services.audit_service import ACTION_ENHANCE, ACTION_SEARCH, AuditService
from ...services.engine_service import EngineService, get_engine_service
from ...services.enhancement_service import (
    EnhancementDisabledError,
    EnhancementService,
    get_enhancement_service,
)
from ...services.storage_service import (
    ImageTooLargeError,
    StorageService,
    UnsupportedImageError,
    decode_base64_image,
)
from ..schemas import (
    EnhancementAnalysisResponse,
    EnhancementRecogniseRequest,
    EnhancementRecogniseResponse,
    EnhancementRequest,
    EnhancementResponse,
    RecognitionSide,
)
from .auth import get_audit_service
from .search import _candidate_response, _persist_candidates, _probe_assessment

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/imatch/enhance", tags=["enhancement"])

CAUTION = (
    "The original image is the evidence. The enhanced result is an analysis intermediate shown for "
    "comparison only. A candidate returned from the enhanced image is not corroboration of a candidate "
    "returned from the original: both were produced by the same recogniser from the same underlying "
    "capture, so they are not independent observations."
)


def get_storage(settings: Settings = Depends(get_settings)) -> StorageService:
    return StorageService(settings.storage_path, settings.probe_retention_days)


def _load_run(session: Session, tenant_id: str, enhancement_id: str) -> EnhancementRun:
    run = session.get(EnhancementRun, enhancement_id)
    # Tenant is re-checked rather than trusted from the id: an id is guessable
    # and there is no endpoint that should read across tenants.
    if run is None or run.tenant_id != tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Enhancement run not found.")
    return run


@router.get("/status")
def enhancement_status(
    principal: Principal = Depends(enforce_search_rate_limit),
    service: EnhancementService = Depends(get_enhancement_service),
) -> dict:
    """Which backends are usable here, and for the rest, exactly why not.

    A missing checkpoint is a normal state on a fresh install, not an error, so
    it is reported as an unavailable backend with the path the file belongs at.
    """
    return service.status()


@router.post("/analyze", response_model=EnhancementAnalysisResponse)
def analyze_image(
    payload: EnhancementRequest,
    principal: Principal = Depends(enforce_search_rate_limit),
    settings: Settings = Depends(get_settings),
    service: EnhancementService = Depends(get_enhancement_service),
) -> EnhancementAnalysisResponse:
    """Measure the degradation and show the plan without executing it.

    Deliberately produces no image and writes no record: this is the call an
    investigator makes to see what the system thinks is wrong before deciding
    whether to process anything.
    """
    try:
        raw = decode_base64_image(payload.image_base64, settings.max_upload_bytes)
        result = service.analyze_bytes(raw)
    except (UnsupportedImageError, ImageTooLargeError, InvalidImageError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    return EnhancementAnalysisResponse(**result)


@router.post("", response_model=EnhancementResponse, status_code=status.HTTP_201_CREATED)
def enhance(
    payload: EnhancementRequest,
    request: Request,
    principal: Principal = Depends(enforce_search_rate_limit),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
    service: EnhancementService = Depends(get_enhancement_service),
    storage: StorageService = Depends(get_storage),
    audit: AuditService = Depends(get_audit_service),
) -> EnhancementResponse:
    """Enhance one image. The original is stored unchanged and remains primary."""
    ip_address, user_agent = client_context(request)

    lawful_basis = payload.lawful_basis.strip()
    case = None
    if payload.case_id:
        case = session.exec(
            select(Case).where(Case.id == payload.case_id, Case.tenant_id == principal.tenant_id)
        ).first()
        if case is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Case not found.")
        if not lawful_basis:
            lawful_basis = case.lawful_basis

    try:
        raw = decode_base64_image(payload.image_base64, settings.max_upload_bytes)
    except (UnsupportedImageError, ImageTooLargeError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    try:
        outcome = service.enhance_bytes(
            raw,
            allow_reconstruction=payload.allow_reconstruction,
            overrides=payload.overrides or None,
            disabled=set(payload.disabled_stages) if payload.disabled_stages else None,
        )
    except EnhancementDisabledError as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from exc
    except (UnsupportedImageError, InvalidImageError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    run = service.persist(
        session,
        tenant_id=principal.tenant_id,
        operator_id=principal.id,
        outcome=outcome,
        storage=storage,
        original_bytes=raw,
        case_id=case.id if case else None,
        lawful_basis=lawful_basis,
    )

    record = audit.record(
        session,
        tenant_id=principal.tenant_id,
        action=ACTION_ENHANCE,
        actor_id=principal.id,
        actor_label=principal.label,
        resource_type="enhancement_run",
        resource_id=run.id,
        outcome=run.track,
        lawful_basis=lawful_basis,
        detail={
            "case_id": run.case_id,
            "original_sha256": run.original_sha256,
            "enhanced_sha256": run.enhanced_sha256,
            "track": run.track,
            "ruleset_version": run.ruleset_version,
            # The stages and their versions, so the chain answers "what was done
            # to this image" without needing the run row.
            "stages": [
                {"name": s.name, "task": s.task.value, "track": s.track.value, "parameters": s.parameters}
                for s in outcome.results
            ],
            "device": outcome.device,
            "warnings": list(outcome.warnings),
        },
        ip_address=ip_address,
        user_agent=user_agent,
    )
    run.audit_hash = record.entry_hash
    session.add(run)
    session.commit()
    session.refresh(run)

    return EnhancementResponse(**service.hydrate(run))


@router.get("/{enhancement_id}", response_model=EnhancementResponse)
def get_enhancement(
    enhancement_id: str,
    principal: Principal = Depends(enforce_search_rate_limit),
    session: Session = Depends(get_session),
    service: EnhancementService = Depends(get_enhancement_service),
) -> EnhancementResponse:
    return EnhancementResponse(**service.hydrate(_load_run(session, principal.tenant_id, enhancement_id)))


@router.get("/{enhancement_id}/image")
def get_enhancement_image(
    enhancement_id: str,
    variant: str = Query(default="enhanced", pattern="^(original|enhanced)$"),
    principal: Principal = Depends(enforce_search_rate_limit),
    session: Session = Depends(get_session),
    storage: StorageService = Depends(get_storage),
) -> Response:
    """Serve either side of the pair.

    Both variants are served from the same endpoint on purpose: the UI that
    shows one can always show the other, so a client cannot end up displaying an
    enhanced image with no way to fetch its original.
    """
    run = _load_run(session, principal.tenant_id, enhancement_id)
    path = run.original_path if variant == "original" else run.enhanced_path
    payload = storage.read(path) if path else None
    if payload is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"The {variant} image is no longer in storage.")

    media_type = "image/png" if variant == "enhanced" else "application/octet-stream"
    headers = {
        "Cache-Control": "private, max-age=3600",
        "X-Image-Variant": variant,
        "X-Image-SHA256": run.original_sha256 if variant == "original" else run.enhanced_sha256,
    }
    if variant == "enhanced":
        # Travels with the bytes. A client that saves this image off and shows it
        # elsewhere still carries the statement of what it is.
        headers["X-Enhancement-Track"] = run.track
        headers["X-Enhancement-Label"] = (
            "AI-enhanced preview - not evidentiary, for visual reference only"
            if run.track == "reconstructed"
            else "Processed image - deterministic operations only, no synthesised detail"
        )
    return Response(content=payload, media_type=media_type, headers=headers)


@router.post("/{enhancement_id}/recognise", response_model=EnhancementRecogniseResponse)
def recognise_both(
    enhancement_id: str,
    payload: EnhancementRecogniseRequest,
    request: Request,
    principal: Principal = Depends(enforce_search_rate_limit),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
    engine: EngineService = Depends(get_engine_service),
    storage: StorageService = Depends(get_storage),
    audit: AuditService = Depends(get_audit_service),
) -> EnhancementRecogniseResponse:
    """Search the gallery with the original and with the enhanced image.

    Two independent searches through the UNMODIFIED recogniser. Nothing about
    the model, its thresholds or the gallery changes; the only difference
    between the two calls is the bytes handed to ``encode()``.

    Both runs are persisted with ``source_kind`` set, so the case record can
    always answer which image produced which candidate list.
    """
    run = _load_run(session, principal.tenant_id, enhancement_id)
    ip_address, user_agent = client_context(request)

    lawful_basis = payload.lawful_basis.strip() or run.lawful_basis
    if settings.require_lawful_basis and not lawful_basis:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "A lawful basis must be stated for every biometric search, including a comparison run.",
        )

    sides: dict[str, RecognitionSide] = {}
    for variant, path, source_kind in (
        ("original", run.original_path, "original"),
        ("enhanced", run.enhanced_path, run.track),
    ):
        raw = storage.read(path) if path else None
        if raw is None:
            sides[variant] = RecognitionSide(
                source_kind=source_kind,
                decision="unavailable",
                confidence=0.0,
                explanation=f"The {variant} image is no longer in storage.",
                candidates=[],
                error="image_missing",
            )
            continue

        try:
            result = engine.encode(raw)
        except NoFaceDetectedError as exc:
            # Very common on the ORIGINAL side with real surveillance frames,
            # and it is a finding rather than an error: "the detector could not
            # find a face until the image was processed" is exactly the kind of
            # thing this comparison exists to surface.
            sides[variant] = RecognitionSide(
                source_kind=source_kind,
                decision="no_face_detected",
                confidence=0.0,
                explanation=str(exc),
                candidates=[],
                error="no_face_detected",
            )
            continue
        except InvalidImageError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

        outcome, decision, normalized = engine.search(
            session, principal.tenant_id, result, payload.top_k
        )

        search_run = SearchRun(
            tenant_id=principal.tenant_id,
            case_id=run.case_id,
            operator_id=principal.id,
            mode="compare",
            lawful_basis=lawful_basis,
            purpose=payload.purpose.strip(),
            decision=decision.label,
            top_score=outcome.top_score,
            margin=outcome.margin,
            gallery_size=outcome.gallery_size,
            candidate_count=len(outcome.matches),
            quality_score=result.quality.score,
            liveness_score=result.liveness.score,
            deepfake_risk=result.deepfake_risk,
            faces_detected=result.faces_detected,
            review_required=True,
            recognition_capable=result.recognition_capable,
            reasons=json.dumps(list(decision.reasons)),
            explanation=decision.explanation,
            recognizer_backend=engine.runtime.recognizer.info.backend,
            recognizer_pack=engine.runtime.recognizer.info.model_pack,
            match_threshold=engine.config.thresholds.match,
            review_threshold=engine.config.thresholds.review,
            probe_sha256=run.original_sha256 if variant == "original" else run.enhanced_sha256,
            probe_path=path,
            source_kind=source_kind,
            enhancement_run_id=None if variant == "original" else run.id,
        )
        session.add(search_run)
        session.flush()
        candidates = _persist_candidates(session, principal.tenant_id, search_run.id, outcome, normalized)

        record = audit.record(
            session,
            tenant_id=principal.tenant_id,
            action=ACTION_SEARCH,
            actor_id=principal.id,
            actor_label=principal.label,
            resource_type="search_run",
            resource_id=search_run.id,
            outcome=decision.label,
            lawful_basis=lawful_basis,
            detail={
                "case_id": run.case_id,
                "source_kind": source_kind,
                "enhancement_run_id": None if variant == "original" else run.id,
                "probe_sha256": search_run.probe_sha256,
                "gallery_size": outcome.gallery_size,
                "top_score": outcome.top_score,
                "candidate_subjects": [match.subject_id for match in outcome.matches],
                "model_pack": engine.runtime.recognizer.info.model_pack,
            },
            ip_address=ip_address,
            user_agent=user_agent,
        )
        search_run.audit_hash = record.entry_hash
        session.add(search_run)

        sides[variant] = RecognitionSide(
            source_kind=source_kind,
            search_id=search_run.id,
            decision=decision.label,
            confidence=decision.confidence,
            explanation=decision.explanation,
            candidates=candidates,
            probe=_probe_assessment(result),
        )

    session.commit()

    return EnhancementRecogniseResponse(
        enhancement_id=run.id,
        primary="original",
        original=sides["original"],
        enhanced=sides["enhanced"],
        caution=CAUTION,
    )


__all__ = ["router"]
