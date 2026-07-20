from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session
from app.db.database import get_session
from app.schemas.feedback import FeedbackCreate, FeedbackRead
from app.services.feedback_service import FeedbackService
from app.models.style_profile import StyleProfile

router = APIRouter()

@router.post("/feedback", response_model=FeedbackRead, tags=["Feedback"])
def submit_feedback(feedback_in: FeedbackCreate, session: Session = Depends(get_session)):
    try:
        # Defaults to user_id=1 for local development
        return FeedbackService.submit_feedback(session, feedback_in, user_id=1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback submission failed: {str(e)}")

@router.get("/style-profile/{user_id}", response_model=None, tags=["Feedback"])
def get_style_profile(user_id: int, session: Session = Depends(get_session)):
    profile = FeedbackService.get_style_profile(session, user_id)
    if not profile:
        # Return a blank profile schema
        return {
            "user_id": user_id,
            "liked_styles": [],
            "disliked_styles": [],
            "preferred_moods": [],
            "preferred_colors": [],
            "preferred_typography": []
        }
    return profile
