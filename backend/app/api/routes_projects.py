from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session
from typing import List
from app.db.database import get_session
from app.schemas.project import ProjectCreate, ProjectRead
from app.services.project_service import ProjectService

router = APIRouter()

@router.post("/projects", response_model=ProjectRead, status_code=status.HTTP_201_CREATED, tags=["Projects"])
def create_project(project_in: ProjectCreate, session: Session = Depends(get_session)):
    try:
        return ProjectService.create_project(session, project_in)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/projects", response_model=List[ProjectRead], tags=["Projects"])
def list_projects(skip: int = 0, limit: int = 100, session: Session = Depends(get_session)):
    return ProjectService.get_projects(session, skip, limit)

@router.get("/projects/{project_id}", response_model=ProjectRead, tags=["Projects"])
def get_project(project_id: int, session: Session = Depends(get_session)):
    project = ProjectService.get_project(session, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project
