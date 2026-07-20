from typing import List, Optional
from sqlmodel import Session, select
from app.models.project import Project
from app.schemas.project import ProjectCreate, ProjectUpdate
from datetime import datetime

class ProjectService:
    @staticmethod
    def create_project(session: Session, project_in: ProjectCreate) -> Project:
        db_project = Project(
            title=project_in.title,
            raw_imagination=project_in.raw_imagination,
            design_type=project_in.design_type,
            user_id=project_in.user_id or 1,  # Default demo user ID
            status="pending"
        )
        session.add(db_project)
        session.commit()
        session.refresh(db_project)
        return db_project

    @staticmethod
    def get_project(session: Session, project_id: int) -> Optional[Project]:
        return session.get(Project, project_id)

    @staticmethod
    def get_projects(session: Session, skip: int = 0, limit: int = 100) -> List[Project]:
        statement = select(Project).offset(skip).limit(limit).order_by(Project.created_at.desc())
        return session.exec(statement).all()

    @staticmethod
    def update_project(session: Session, project: Project, project_in: ProjectUpdate) -> Project:
        project_data = project_in.model_dump(exclude_unset=True)
        for key, value in project_data.items():
            setattr(project, key, value)
        project.updated_at = datetime.utcnow()
        session.add(project)
        session.commit()
        session.refresh(project)
        return project
