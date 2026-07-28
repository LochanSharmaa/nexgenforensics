from __future__ import annotations

import json
import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlmodel import Session, select

from nexgen_engine.inference.pipeline import InvalidImageError, NoFaceDetectedError
from nexgen_engine.security.morphing_detector import MorphingDetector

from ...core.config import Settings, get_settings
from ...core.dependencies import (
    Principal,
    client_context,
    enforce_search_rate_limit,
    get_current_principal,
)
from ...db.models import Adjudication, Candidate, Case, SearchRun, Subject, utcnow
from ...db.session import get_session
from ...services.audit_service import (
    ACTION_ADJUDICATE,
    ACTION_SEARCH,
    ACTION_VERIFY,
    AuditService,
)
from ...services.engine_service import EngineService, get_engine_service
from ...services.storage_service import (
    ImageTooLargeError,
    StorageService,
    UnsupportedImageError,
    decode_base64_image,
)
from ..schemas import (
    AdjudicateRequest,
    CandidateResponse,
    ProbeAssessment,
    SearchRequest,
    SearchResponse,
    SearchRunResponse,
    VerifyRequest,
    VerifyResponse,
)
from .auth import get_audit_service

logger = logging.getLogger(__name__)

# Mounted at /api/imatch so POST /api/imatch/search matches the contract the
# web console already speaks.
router = APIRouter(prefix="/api/imatch", tags=["search"])


def get_storage(settings: Settings = Depends(get_settings)) -> StorageService:
    return StorageService(settings.storage_path, settings.probe_retention_days)


@router.post("/search", response_model=SearchResponse)
def search(
    payload: SearchRequest,
    request: Request,
    principal: Principal = Depends(enforce_search_rate_limit),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
    engine: EngineService = Depends(get_engine_service),
    storage: StorageService = Depends(get_storage),
    audit: AuditService = Depends(get_audit_service),
) -> SearchResponse:
    """Search a probe image against the caller's tenant gallery.

    The gallery searched is always the authenticated principal's tenant. There
    is no parameter to search another tenant, by design.
    """
    started = time.perf_counter()
    ip_address, user_agent = client_context(request)

    lawful_basis = payload.lawful_basis.strip()
    if settings.require_lawful_basis and not lawful_basis:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "A lawful basis must be stated for every biometric search. "
            "Set NEXGEN_REQUIRE_LAWFUL_BASIS=false only for non-production testing.",
        )

    case = _resolve_case(session, principal.tenant_id, payload.case_id)
    if case is not None and not lawful_basis:
        lawful_basis = case.lawful_basis

    raw = _load_probe_bytes(payload, settings)
    stored = storage.store(principal.tenant_id, raw, category="probes")

    try:
        result = engine.encode(raw)
    except NoFaceDetectedError as exc:
        _audit_failure(session, audit, principal, "no_face_detected", lawful_basis, ip_address, user_agent)
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except InvalidImageError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    outcome, decision, normalized = engine.search(session, principal.tenant_id, result, payload.top_k)
    duration_ms = int((time.perf_counter() - started) * 1000)

    run = SearchRun(
        tenant_id=principal.tenant_id,
        case_id=case.id if case else None,
        operator_id=principal.id,
        mode=payload.mode,
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
        review_required=decision.label != "candidate_match",
        recognition_capable=result.recognition_capable,
        reasons=json.dumps(list(decision.reasons)),
        explanation=decision.explanation,
        recognizer_backend=engine.runtime.recognizer.info.backend,
        recognizer_pack=engine.runtime.recognizer.info.model_pack,
        match_threshold=engine.config.thresholds.match,
        review_threshold=engine.config.thresholds.review,
        probe_sha256=stored.sha256,
        probe_path=stored.path,
        duration_ms=duration_ms,
    )
    session.add(run)
    session.flush()

    candidates = _persist_candidates(session, principal.tenant_id, run.id, outcome, normalized)

    record = audit.record(
        session,
        tenant_id=principal.tenant_id,
        action=ACTION_SEARCH,
        actor_id=principal.id,
        actor_label=principal.label,
        resource_type="search_run",
        resource_id=run.id,
        outcome=decision.label,
        lawful_basis=lawful_basis,
        detail={
            "case_id": case.id if case else None,
            "probe_sha256": stored.sha256,
            "gallery_size": outcome.gallery_size,
            "top_score": outcome.top_score,
            "candidate_subjects": [match.subject_id for match in outcome.matches],
            "recognition_capable": result.recognition_capable,
            "model_pack": engine.runtime.recognizer.info.model_pack,
        },
        ip_address=ip_address,
        user_agent=user_agent,
    )
    run.audit_hash = record.entry_hash
    session.add(run)
    session.commit()

    return SearchResponse(
        search_id=run.id,
        decision=decision.label,
        confidence=decision.confidence,
        explanation=decision.explanation,
        review_required=run.review_required,
        recognition_capable=result.recognition_capable,
        reasons=list(decision.reasons),
        candidates=candidates,
        probe=_probe_assessment(result),
        gallery_size=outcome.gallery_size,
        margin=round(outcome.margin, 6),
        thresholds={
            "match": engine.config.thresholds.match,
            "review": engine.config.thresholds.review,
        },
        model=engine.runtime.recognizer.info.as_dict(),
        duration_ms=duration_ms,
        audit_hash=record.entry_hash,
    )


@router.post("/verify", response_model=VerifyResponse)
def verify(
    payload: VerifyRequest,
    request: Request,
    principal: Principal = Depends(enforce_search_rate_limit),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
    engine: EngineService = Depends(get_engine_service),
    audit: AuditService = Depends(get_audit_service),
) -> VerifyResponse:
    """One-to-one comparison of two supplied images. Nothing is enrolled."""
    ip_address, user_agent = client_context(request)

    lawful_basis = payload.lawful_basis.strip()
    if settings.require_lawful_basis and not lawful_basis:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "A lawful basis must be stated for every biometric comparison.",
        )

    try:
        reference_bytes = decode_base64_image(payload.reference_image_base64, settings.max_upload_bytes)
        probe_bytes = decode_base64_image(payload.probe_image_base64, settings.max_upload_bytes)
        reference = engine.encode(reference_bytes)
        probe = engine.encode(probe_bytes)
    except (UnsupportedImageError, ImageTooLargeError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except NoFaceDetectedError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except InvalidImageError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    similarity = engine.compare(reference.embedding, probe.embedding)
    threshold = engine.config.thresholds.verify
    capable = engine.recognition_capable
    verified = bool(capable and similarity >= threshold)

    if not capable:
        explanation = (
            "No recognition model is loaded, so this comparison carries no meaning. "
            "Install backend/requirements-engine.txt."
        )
    elif verified:
        explanation = (
            f"Similarity {similarity:.3f} meets the verification threshold {threshold:.2f}. "
            "This supports, but does not establish, that the images show the same person."
        )
    else:
        explanation = (
            f"Similarity {similarity:.3f} is below the verification threshold {threshold:.2f}. "
            "The images are not supported as showing the same person."
        )

    morphing = MorphingDetector().differential_risk(probe.embedding, reference.embedding)

    record = audit.record(
        session,
        tenant_id=principal.tenant_id,
        action=ACTION_VERIFY,
        actor_id=principal.id,
        actor_label=principal.label,
        resource_type="case" if payload.case_id else "",
        resource_id=payload.case_id or "",
        outcome="verified" if verified else "not_verified",
        lawful_basis=lawful_basis,
        detail={"similarity": round(similarity, 6), "threshold": threshold, "recognition_capable": capable},
        ip_address=ip_address,
        user_agent=user_agent,
    )
    session.commit()

    return VerifyResponse(
        similarity=round(similarity, 6),
        verified=verified,
        threshold=threshold,
        explanation=explanation,
        recognition_capable=capable,
        reference=_probe_assessment(reference),
        probe=_probe_assessment(probe),
        morphing=morphing,
        audit_hash=record.entry_hash,
    )


@router.get("/searches", response_model=list[SearchRunResponse])
def list_searches(
    case_id: str | None = None,
    limit: int = 50,
    principal: Principal = Depends(get_current_principal),
    session: Session = Depends(get_session),
) -> list[SearchRunResponse]:
    statement = select(SearchRun).where(SearchRun.tenant_id == principal.tenant_id)
    if case_id:
        statement = statement.where(SearchRun.case_id == case_id)
    statement = statement.order_by(SearchRun.created_at.desc()).limit(max(1, min(limit, 200)))
    return [SearchRunResponse.model_validate(run) for run in session.exec(statement).all()]


@router.get("/searches/{search_id}/candidates", response_model=list[CandidateResponse])
def list_candidates(
    search_id: str,
    principal: Principal = Depends(get_current_principal),
    session: Session = Depends(get_session),
) -> list[CandidateResponse]:
    run = session.get(SearchRun, search_id)
    if run is None or run.tenant_id != principal.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Search not found.")

    rows = session.exec(
        select(Candidate)
        .where(Candidate.search_run_id == search_id, Candidate.tenant_id == principal.tenant_id)
        .order_by(Candidate.rank)
    ).all()
    return [_candidate_response(session, row) for row in rows]


@router.post("/candidates/{candidate_id}/adjudicate", response_model=CandidateResponse)
def adjudicate(
    candidate_id: str,
    payload: AdjudicateRequest,
    request: Request,
    principal: Principal = Depends(get_current_principal),
    session: Session = Depends(get_session),
    audit: AuditService = Depends(get_audit_service),
) -> CandidateResponse:
    """Record an examiner's verdict on a candidate.

    This is the only place a candidate becomes "confirmed", and only a human can
    do it. The engine never writes this field.
    """
    candidate = session.get(Candidate, candidate_id)
    if candidate is None or candidate.tenant_id != principal.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Candidate not found.")

    candidate.adjudication = payload.adjudication
    candidate.examiner_notes = payload.examiner_notes.strip()
    candidate.adjudicated_by = principal.id
    candidate.adjudicated_at = utcnow()
    session.add(candidate)

    ip_address, user_agent = client_context(request)
    audit.record(
        session,
        tenant_id=principal.tenant_id,
        action=ACTION_ADJUDICATE,
        actor_id=principal.id,
        actor_label=principal.label,
        resource_type="candidate",
        resource_id=candidate.id,
        outcome=payload.adjudication.value,
        detail={
            "search_run_id": candidate.search_run_id,
            "subject_id": candidate.subject_id,
            "score": candidate.score,
            "notes_supplied": bool(candidate.examiner_notes),
        },
        ip_address=ip_address,
        user_agent=user_agent,
    )
    session.commit()
    session.refresh(candidate)
    return _candidate_response(session, candidate)


# ------------------------------------------------------------------ helpers --


def _load_probe_bytes(payload: SearchRequest, settings: Settings) -> bytes:
    if payload.image_base64:
        try:
            return decode_base64_image(payload.image_base64, settings.max_upload_bytes)
        except (UnsupportedImageError, ImageTooLargeError) as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    if payload.source_url:
        # Deliberately not implemented. Fetching a client-supplied URL from
        # inside the service is a server-side request forgery primitive: it
        # would let a caller reach internal addresses and cloud metadata
        # endpoints using the server's network position. Enabling this needs an
        # allow-list of hosts, DNS-rebinding protection, and egress limits.
        raise HTTPException(
            status.HTTP_501_NOT_IMPLEMENTED,
            "Server-side URL import is disabled. Fetch the image client-side and submit it as "
            "image_base64.",
        )

    raise HTTPException(status.HTTP_400_BAD_REQUEST, "Supply image_base64.")


def _resolve_case(session: Session, tenant_id: str, case_id: str | None) -> Case | None:
    if not case_id:
        return None
    case = session.get(Case, case_id)
    if case is None or case.tenant_id != tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Case not found.")
    return case


def _persist_candidates(
    session: Session,
    tenant_id: str,
    search_run_id: str,
    outcome,  # noqa: ANN001 - SearchOutcome
    normalized,  # noqa: ANN001 - np.ndarray
) -> list[CandidateResponse]:
    responses: list[CandidateResponse] = []
    for rank, match in enumerate(outcome.matches, start=1):
        candidate = Candidate(
            tenant_id=tenant_id,
            search_run_id=search_run_id,
            subject_id=match.subject_id,
            template_id=match.template_id,
            rank=rank,
            score=match.score,
            normalized_score=round(float(normalized[rank - 1]), 6) if len(normalized) >= rank else 0.0,
        )
        session.add(candidate)
        session.flush()
        responses.append(_candidate_response(session, candidate))
    return responses


def _candidate_response(session: Session, candidate: Candidate) -> CandidateResponse:
    subject = session.get(Subject, candidate.subject_id)
    return CandidateResponse(
        id=candidate.id,
        rank=candidate.rank,
        subject_id=candidate.subject_id,
        template_id=candidate.template_id,
        subject_name=subject.display_name if subject else "",
        external_ref=subject.external_ref if subject else "",
        score=candidate.score,
        normalized_score=candidate.normalized_score,
        adjudication=candidate.adjudication or Adjudication.PENDING,
        examiner_notes=candidate.examiner_notes,
    )


def _probe_assessment(result) -> ProbeAssessment:  # noqa: ANN001 - RecognitionResult
    return ProbeAssessment(
        quality=result.quality.as_dict(),
        liveness=result.liveness.as_dict(),
        deepfake_risk=result.deepfake_risk,
        faces_detected=result.faces_detected,
        detector=result.detector_name,
        box={
            "left": result.face.box.left,
            "top": result.face.box.top,
            "right": result.face.box.right,
            "bottom": result.face.box.bottom,
        },
        pose={
            "yaw": result.face.pose.yaw,
            "pitch": result.face.pose.pitch,
            "roll": result.face.pose.roll,
        },
    )


def _audit_failure(
    session: Session,
    audit: AuditService,
    principal: Principal,
    reason: str,
    lawful_basis: str,
    ip_address: str,
    user_agent: str,
) -> None:
    audit.record(
        session,
        tenant_id=principal.tenant_id,
        action=ACTION_SEARCH,
        actor_id=principal.id,
        actor_label=principal.label,
        outcome="rejected",
        lawful_basis=lawful_basis,
        detail={"reason": reason},
        ip_address=ip_address,
        user_agent=user_agent,
    )
    session.commit()


__all__ = ["get_storage", "router"]
