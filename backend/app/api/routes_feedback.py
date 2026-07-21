from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from app.db.database import get_session
from app.core.auth import get_current_user
from app.models.user import User
from app.schemas.feedback import FeedbackCreate, FeedbackRead
from app.services.feedback_service import FeedbackService

router = APIRouter()


@router.post("/feedback", response_model=FeedbackRead, tags=["Feedback"])
async def submit_feedback(
    feedback_in: FeedbackCreate,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    try:
        return FeedbackService.submit_feedback(session, feedback_in, user_id=current_user.id)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Feedback submission failed: {str(e)}"
        )


@router.get("/style-profile/{user_id}", tags=["Feedback"])
async def get_style_profile(
    user_id: int,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    # Users can only view their own style profile
    if user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    profile = FeedbackService.get_style_profile(session, user_id)
    if not profile:
        return {
            "user_id": user_id,
            "liked_styles": [],
            "disliked_styles": [],
            "preferred_moods": [],
            "preferred_colors": [],
            "preferred_typography": [],
        }
    return profile
