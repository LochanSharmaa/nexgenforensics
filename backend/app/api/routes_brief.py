from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session
from typing import List

from app.db.database import get_session
from app.core.auth import get_current_user
from app.models.user import User
from app.models.project import Project
from app.schemas.brief import (
    CreativeBriefRead,
    BriefExtractionRequest,
    ClarifyingQuestionRead,
    ClarifyingQuestionAnswer,
)
from app.services.brief_service import BriefService
from app.services.question_service import QuestionService

router = APIRouter()


def _verify_project_ownership(session: Session, project_id: int, user: User) -> Project:
    """Verify the project belongs to the current user."""
    project = session.get(Project, project_id)
    if not project or project.user_id != user.id:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.post("/brief/extract", response_model=CreativeBriefRead, tags=["Brief"])
async def extract_brief(
    request: BriefExtractionRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    _verify_project_ownership(session, request.project_id, current_user)
    try:
        return BriefService.extract_brief(session, request)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract brief: {str(e)}")


@router.post(
    "/brief/questions",
    response_model=List[ClarifyingQuestionRead],
    tags=["Brief Questions"],
)
async def generate_questions(
    request: BriefExtractionRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    _verify_project_ownership(session, request.project_id, current_user)
    try:
        return QuestionService.generate_questions(session, request.project_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/brief/clarify", response_model=ClarifyingQuestionRead, tags=["Brief Questions"])
async def clarify_question(
    answer_in: ClarifyingQuestionAnswer,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    db_q = QuestionService.answer_question(session, answer_in.question_id, answer_in)
    if not db_q:
        raise HTTPException(status_code=404, detail="Question not found")
    return db_q


@router.get(
    "/projects/{project_id}/questions",
    response_model=List[ClarifyingQuestionRead],
    tags=["Brief Questions"],
)
async def get_questions(
    project_id: int,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    _verify_project_ownership(session, project_id, current_user)
    return QuestionService.get_questions_by_project(session, project_id)
