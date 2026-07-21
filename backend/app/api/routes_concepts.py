from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select
from typing import List

from app.db.database import get_session
from app.core.auth import get_current_user
from app.models.user import User
from app.models.project import Project
from app.models.reasoning import ConceptGenealogy, MissingOpportunityReport, ConceptScore
from app.schemas.concept import (
    ConceptRead,
    ConceptGenerateRequest,
    ConceptRegenerateRequest,
    ConceptCombineRequest,
    DiversityCheckResponse,
)
from app.services.concept_service import ConceptService
from app.services.diversity_service import DiversityService

router = APIRouter()


def _verify_project_ownership(session: Session, project_id: int, user: User) -> Project:
    project = session.get(Project, project_id)
    if not project or project.user_id != user.id:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.post("/concepts/generate", response_model=List[ConceptRead], tags=["Concepts"])
async def generate_concepts(
    request: ConceptGenerateRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    _verify_project_ownership(session, request.project_id, current_user)
    try:
        return ConceptService.generate_concepts(session, request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Concept generation failed: {str(e)}"
        )


@router.post("/concepts/regenerate", response_model=ConceptRead, tags=["Concepts"])
async def regenerate_concept(
    request: ConceptRegenerateRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    _verify_project_ownership(session, request.project_id, current_user)
    try:
        return ConceptService.regenerate_concept(session, request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/concepts/combine", response_model=ConceptRead, tags=["Concepts"])
async def combine_concepts(
    request: ConceptCombineRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    _verify_project_ownership(session, request.project_id, current_user)
    try:
        return ConceptService.combine_concepts(session, request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/concepts/save", response_model=ConceptRead, tags=["Concepts"])
async def save_concept(
    request: ConceptRegenerateRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    _verify_project_ownership(session, request.project_id, current_user)
    db_c = ConceptService.update_concept_status(session, request.concept_id, "saved")
    if not db_c:
        raise HTTPException(status_code=404, detail="Concept not found")
    return db_c


@router.post("/concepts/reject", response_model=ConceptRead, tags=["Concepts"])
async def reject_concept(
    request: ConceptRegenerateRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    _verify_project_ownership(session, request.project_id, current_user)
    db_c = ConceptService.update_concept_status(session, request.concept_id, "rejected")
    if not db_c:
        raise HTTPException(status_code=404, detail="Concept not found")
    return db_c


@router.get(
    "/concepts/{project_id}",
    response_model=List[ConceptRead],
    tags=["Concepts"],
)
async def get_concepts(
    project_id: int,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    _verify_project_ownership(session, project_id, current_user)
    return ConceptService.get_concepts_by_project(session, project_id)


@router.get("/concepts/{concept_id}/genealogy", tags=["Concepts"])
async def get_concept_genealogy(
    concept_id: int,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    entries = session.exec(
        select(ConceptGenealogy).where(
            (ConceptGenealogy.parent_concept_id == concept_id)
            | (ConceptGenealogy.child_concept_id == concept_id)
        )
    ).all()
    return [e.model_dump() for e in entries]


@router.get("/projects/{project_id}/missing-opportunities", tags=["Concepts"])
async def get_missing_opportunities(
    project_id: int,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    _verify_project_ownership(session, project_id, current_user)
    reports = session.exec(
        select(MissingOpportunityReport).where(
            MissingOpportunityReport.project_id == project_id
        )
    ).all()
    return [r.model_dump() for r in reports]


@router.get("/concepts/{concept_id}/scores", tags=["Concepts"])
async def get_concept_scores(
    concept_id: int,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    scores = session.exec(
        select(ConceptScore).where(ConceptScore.concept_id == concept_id)
    ).all()
    if not scores:
        raise HTTPException(status_code=404, detail="Scores not found")
    return [s.model_dump() for s in scores]
