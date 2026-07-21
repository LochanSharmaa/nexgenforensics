from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse, Response
from sqlmodel import Session

from app.db.database import get_session
from app.core.auth import get_current_user
from app.models.user import User
from app.models.project import Project
from app.services.export_service import ExportService

router = APIRouter()


def _verify_ownership(session: Session, project_id: int, user: User) -> Project:
    project = session.get(Project, project_id)
    if not project or project.user_id != user.id:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.get("/export/{project_id}/json", tags=["Export"])
async def export_json(
    project_id: int,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    _verify_ownership(session, project_id, current_user)
    try:
        json_content = ExportService.export_json(session, project_id)
        return Response(
            content=json_content,
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename=project_{project_id}_board.json"
            },
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/{project_id}/markdown", tags=["Export"])
async def export_markdown(
    project_id: int,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    _verify_ownership(session, project_id, current_user)
    try:
        markdown_content = ExportService.export_markdown(session, project_id)
        return Response(
            content=markdown_content,
            media_type="text/markdown",
            headers={
                "Content-Disposition": f"attachment; filename=project_{project_id}_board.md"
            },
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/{project_id}/present", tags=["Export"])
async def export_presentation(
    project_id: int,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Presentation Mode export — reasoning/scores/dissent hidden, clean output."""
    _verify_ownership(session, project_id, current_user)
    try:
        present_data = ExportService.export_presentation(session, project_id)
        return present_data
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
