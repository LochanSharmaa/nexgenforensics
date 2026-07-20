from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session
from typing import List
from app.db.database import get_session
from app.schemas.brief import (
    CreativeBriefRead,
    BriefExtractionRequest,
    ClarifyingQuestionRead,
    ClarifyingQuestionAnswer
)
from app.services.brief_service import BriefService
from app.services.question_service import QuestionService

router = APIRouter()

@router.post("/brief/extract", response_model=CreativeBriefRead, tags=["Brief"])
def extract_brief(request: BriefExtractionRequest, session: Session = Depends(get_session)):
    try:
        return BriefService.extract_brief(session, request)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract brief: {str(e)}")

@router.get("/brief/{project_id}", response_model=CreativeBriefRead, tags=["Brief"])
def get_brief(project_id: int, session: Session = Depends(get_session)):
    brief = BriefService.get_brief_by_project(session, project_id)
    if not brief:
        raise HTTPException(status_code=404, detail="Brief not found for this project")
    return brief

@router.post("/brief/questions", response_model=List[ClarifyingQuestionRead], tags=["Brief Questions"])
def generate_questions(request: BriefExtractionRequest, session: Session = Depends(get_session)):
    try:
        return QuestionService.generate_questions(session, request.project_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/projects/{project_id}/questions", response_model=List[ClarifyingQuestionRead], tags=["Brief Questions"])
def get_questions(project_id: int, session: Session = Depends(get_session)):
    return QuestionService.get_questions_by_project(session, project_id)

@router.post("/brief/questions/{question_id}/answer", response_model=ClarifyingQuestionRead, tags=["Brief Questions"])
def answer_question(question_id: int, answer_in: ClarifyingQuestionAnswer, session: Session = Depends(get_session)):
    db_q = QuestionService.answer_question(session, question_id, answer_in)
    if not db_q:
        raise HTTPException(status_code=404, detail="Question not found")
    return db_q
