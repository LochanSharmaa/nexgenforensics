from typing import Optional
from sqlmodel import Session, select
from app.models.project import Project
from app.models.creative_brief import CreativeBrief
from app.models.style_profile import StyleProfile
from app.services.provider_router import get_llm_provider
from app.schemas.brief import BriefExtractionRequest

class BriefService:
    @staticmethod
    def extract_brief(session: Session, request: BriefExtractionRequest) -> CreativeBrief:
        project = session.get(Project, request.project_id)
        if not project:
            raise ValueError("Project not found")

        # Get style profile if exists
        style_profile_data = None
        statement = select(StyleProfile).where(StyleProfile.user_id == (project.user_id or 1))
        profile = session.exec(statement).first()
        if profile:
            style_profile_data = {
                "liked_styles": profile.liked_styles,
                "disliked_styles": profile.disliked_styles,
                "preferred_moods": profile.preferred_moods,
                "preferred_colors": profile.preferred_colors,
                "preferred_typography": profile.preferred_typography
            }

        provider = get_llm_provider()
        brief_data = provider.extract_brief(project.raw_imagination, style_profile=style_profile_data)

        # Delete existing brief for the project if any
        statement = select(CreativeBrief).where(CreativeBrief.project_id == project.id)
        existing = session.exec(statement).first()
        if existing:
            session.delete(existing)

        db_brief = CreativeBrief(
            project_id=project.id,
            main_subject=brief_data.get("main_subject", "Product"),
            design_type=brief_data.get("design_type", project.design_type or "Poster"),
            target_audience=brief_data.get("target_audience", "General public"),
            mood=brief_data.get("mood", []),
            colors=brief_data.get("colors", []),
            fixed_elements=brief_data.get("fixed_elements", []),
            flexible_elements=brief_data.get("flexible_elements", []),
            avoid_elements=brief_data.get("avoid_elements", []),
            tensions=brief_data.get("tensions", [])
        )

        session.add(db_brief)
        
        project.status = "brief_extracted"
        session.add(project)
        
        session.commit()
        session.refresh(db_brief)
        return db_brief

    @staticmethod
    def get_brief_by_project(session: Session, project_id: int) -> Optional[CreativeBrief]:
        statement = select(CreativeBrief).where(CreativeBrief.project_id == project_id)
        return session.exec(statement).first()
