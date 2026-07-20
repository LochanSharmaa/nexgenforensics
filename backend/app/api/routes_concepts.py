from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session
from typing import List
from app.db.database import get_session
from app.schemas.concept import (
    ConceptRead,
    ConceptGenerateRequest,
    ConceptRegenerateRequest,
    ConceptCombineRequest,
    DiversityCheckResponse
)
from app.services.concept_service import ConceptService
from app.services.diversity_service import DiversityService

router = APIRouter()

@router.post("/concepts/generate", response_model=List[ConceptRead], tags=["Concepts"])
def generate_concepts(request: ConceptGenerateRequest, session: Session = Depends(get_session)):
    try:
        return ConceptService.generate_concepts(session, request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Concept generation failed: {str(e)}")

@router.get("/projects/{project_id}/concepts", response_model=List[ConceptRead], tags=["Concepts"])
def get_concepts(project_id: int, session: Session = Depends(get_session)):
    return ConceptService.get_concepts_by_project(session, project_id)

@router.post("/concepts/regenerate", response_model=ConceptRead, tags=["Concepts"])
def regenerate_concept(request: ConceptRegenerateRequest, session: Session = Depends(get_session)):
    try:
        return ConceptService.regenerate_concept(session, request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/concepts/combine", response_model=ConceptRead, tags=["Concepts"])
def combine_concepts(request: ConceptCombineRequest, session: Session = Depends(get_session)):
    try:
        return ConceptService.combine_concepts(session, request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/concepts/diversity-check", response_model=DiversityCheckResponse, tags=["Concepts"])
def check_diversity(request: ConceptGenerateRequest, session: Session = Depends(get_session)):
    concepts = ConceptService.get_concepts_by_project(session, request.project_id)
    if not concepts:
        raise HTTPException(status_code=404, detail="No concepts found for this project")
    return DiversityService.calculate_pairwise_similarities(concepts)

@router.post("/concepts/save", response_model=ConceptRead, tags=["Concepts"])
def save_concept(request: ConceptRegenerateRequest, session: Session = Depends(get_session)):
    # Reuses schema (project_id, concept_id as concept_id)
    db_c = ConceptService.update_concept_status(session, request.concept_id, "saved")
    if not db_c:
        raise HTTPException(status_code=404, detail="Concept not found")
    return db_c

@router.post("/concepts/reject", response_model=ConceptRead, tags=["Concepts"])
def reject_concept(request: ConceptRegenerateRequest, session: Session = Depends(get_session)):
    db_c = ConceptService.update_concept_status(session, request.concept_id, "rejected")
    if not db_c:
        raise HTTPException(status_code=404, detail="Concept not found")
    return db_c
