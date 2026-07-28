from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlmodel import Session, func, select

from ...core.dependencies import Principal, client_context, get_current_principal
from ...db.models import (
    Adjudication,
    Candidate,
    Case,
    CaseStatus,
    Role,
    SearchRun,
    Subject,
    utcnow,
)
from ...db.session import get_session
from ...services.audit_service import ACTION_CASE_CREATE, ACTION_CASE_UPDATE, AuditService
from ..schemas import CaseDetailResponse, CaseResponse, CreateCaseRequest, UpdateCaseRequest
from .auth import get_audit_service

router = APIRouter(prefix="/api/cases", tags=["cases"])


@router.get("", response_model=list[CaseResponse])
def list_cases(
    status_filter: CaseStatus | None = None,
    principal: Principal = Depends(get_current_principal),
    session: Session = Depends(get_session),
) -> list[CaseResponse]:
    """List cases in the caller's tenant.

    Investigators see only their own cases; supervisors and admins see the whole
    tenant. A case file names people who may never be charged, so read access is
    not granted tenant-wide by default.
    """
    statement = select(Case).where(Case.tenant_id == principal.tenant_id)
    if not principal.has_role(Role.SUPERVISOR):
        statement = statement.where(Case.owner_id == principal.id)
    if status_filter is not None:
        statement = statement.where(Case.status == status_filter)

    cases = session.exec(statement.order_by(Case.updated_at.desc())).all()
    return [CaseResponse.model_validate(case) for case in cases]


@router.post("", response_model=CaseResponse, status_code=status.HTTP_201_CREATED)
def create_case(
    payload: CreateCaseRequest,
    request: Request,
    principal: Principal = Depends(get_current_principal),
    session: Session = Depends(get_session),
    audit: AuditService = Depends(get_audit_service),
) -> CaseResponse:
    reference = payload.reference.strip()
    existing = session.exec(
        select(Case).where(Case.tenant_id == principal.tenant_id, Case.reference == reference)
    ).first()
    if existing is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, f"Case reference {reference!r} already exists.")

    case = Case(
        tenant_id=principal.tenant_id,
        reference=reference,
        title=payload.title.strip(),
        description=payload.description.strip(),
        lawful_basis=payload.lawful_basis.strip(),
        owner_id=principal.id,
    )
    session.add(case)
    session.flush()

    ip_address, user_agent = client_context(request)
    audit.record(
        session,
        tenant_id=principal.tenant_id,
        action=ACTION_CASE_CREATE,
        actor_id=principal.id,
        actor_label=principal.label,
        resource_type="case",
        resource_id=case.id,
        lawful_basis=case.lawful_basis,
        detail={"reference": reference},
        ip_address=ip_address,
        user_agent=user_agent,
    )
    session.commit()
    session.refresh(case)
    return CaseResponse.model_validate(case)


@router.get("/{case_id}", response_model=CaseDetailResponse)
def get_case(
    case_id: str,
    principal: Principal = Depends(get_current_principal),
    session: Session = Depends(get_session),
) -> CaseDetailResponse:
    case = _load_case(session, principal, case_id)

    search_count = session.exec(
        select(func.count()).select_from(SearchRun).where(SearchRun.case_id == case.id)
    ).one()
    subject_count = session.exec(
        select(func.count()).select_from(Subject).where(Subject.case_id == case.id)
    ).one()
    confirmed_count = session.exec(
        select(func.count())
        .select_from(Candidate)
        .join(SearchRun, SearchRun.id == Candidate.search_run_id)
        .where(SearchRun.case_id == case.id, Candidate.adjudication == Adjudication.CONFIRMED)
    ).one()

    return CaseDetailResponse(
        **CaseResponse.model_validate(case).model_dump(),
        search_count=int(search_count),
        subject_count=int(subject_count),
        confirmed_count=int(confirmed_count),
    )


@router.patch("/{case_id}", response_model=CaseResponse)
def update_case(
    case_id: str,
    payload: UpdateCaseRequest,
    request: Request,
    principal: Principal = Depends(get_current_principal),
    session: Session = Depends(get_session),
    audit: AuditService = Depends(get_audit_service),
) -> CaseResponse:
    case = _load_case(session, principal, case_id)
    changes: dict[str, object] = {}

    if payload.title is not None:
        case.title = payload.title.strip()
        changes["title"] = case.title
    if payload.description is not None:
        case.description = payload.description.strip()
        changes["description"] = "updated"
    if payload.lawful_basis is not None:
        case.lawful_basis = payload.lawful_basis.strip()
        changes["lawful_basis"] = case.lawful_basis
    if payload.status is not None and payload.status != case.status:
        case.status = payload.status
        case.closed_at = utcnow() if payload.status == CaseStatus.CLOSED else None
        changes["status"] = payload.status.value

    case.updated_at = utcnow()
    session.add(case)

    ip_address, user_agent = client_context(request)
    audit.record(
        session,
        tenant_id=principal.tenant_id,
        action=ACTION_CASE_UPDATE,
        actor_id=principal.id,
        actor_label=principal.label,
        resource_type="case",
        resource_id=case.id,
        detail=changes,
        ip_address=ip_address,
        user_agent=user_agent,
    )
    session.commit()
    session.refresh(case)
    return CaseResponse.model_validate(case)


def _load_case(session: Session, principal: Principal, case_id: str) -> Case:
    case = session.get(Case, case_id)
    # Same 404 for "does not exist" and "belongs to another tenant": a
    # distinguishable 403 would confirm that a given case id is real.
    if case is None or case.tenant_id != principal.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Case not found.")
    if not principal.has_role(Role.SUPERVISOR) and case.owner_id != principal.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Case not found.")
    return case


__all__ = ["router"]
