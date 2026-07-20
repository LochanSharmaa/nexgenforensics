from typing import List, Optional
from sqlmodel import Session, select
from app.models.feedback import Feedback
from app.models.concept import Concept
from app.models.style_profile import StyleProfile
from app.schemas.feedback import FeedbackCreate
from datetime import datetime

class FeedbackService:
    @staticmethod
    def submit_feedback(session: Session, feedback_in: FeedbackCreate, user_id: int = 1) -> Feedback:
        # Save feedback record
        db_fb = Feedback(
            user_id=user_id,
            project_id=feedback_in.project_id,
            concept_id=feedback_in.concept_id,
            rating=feedback_in.rating,
            liked=feedback_in.liked,
            notes=feedback_in.notes
        )
        session.add(db_fb)

        # Retrieve concept details to update user's StyleProfile
        concept = session.get(Concept, feedback_in.concept_id)
        if concept:
            # Look up or create style profile for the user
            statement = select(StyleProfile).where(StyleProfile.user_id == user_id)
            profile = session.exec(statement).first()
            if not profile:
                profile = StyleProfile(
                    user_id=user_id,
                    liked_styles=[],
                    disliked_styles=[],
                    preferred_moods=[],
                    preferred_colors=[],
                    preferred_typography=[]
                )
                session.add(profile)
            
            # Update lists
            if feedback_in.rating >= 4 or feedback_in.liked:
                if concept.style_category not in profile.liked_styles:
                    profile.liked_styles.append(concept.style_category)
                # Cleanup from dislikes
                if concept.style_category in profile.disliked_styles:
                    profile.disliked_styles.remove(concept.style_category)
                
                # Capture colors
                for color in concept.color_palette:
                    if color not in profile.preferred_colors:
                        profile.preferred_colors.append(color)
                        
                # Capture typography style
                if concept.typography_direction not in profile.preferred_typography:
                    profile.preferred_typography.append(concept.typography_direction)
            else:
                # Poor feedback
                if concept.style_category not in profile.disliked_styles:
                    profile.disliked_styles.append(concept.style_category)
                if concept.style_category in profile.liked_styles:
                    profile.liked_styles.remove(concept.style_category)

            profile.updated_at = datetime.utcnow()
            session.add(profile)

        session.commit()
        session.refresh(db_fb)
        return db_fb

    @staticmethod
    def get_style_profile(session: Session, user_id: int) -> Optional[StyleProfile]:
        statement = select(StyleProfile).where(StyleProfile.user_id == user_id)
        return session.exec(statement).first()
