from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse, Response
from sqlmodel import Session
from app.db.database import get_session
from app.services.export_service import ExportService

router = APIRouter()

@router.get("/export/{project_id}/json", tags=["Export"])
def export_json(project_id: int, session: Session = Depends(get_session)):
    try:
        json_content = ExportService.export_json(session, project_id)
        # Return as downloadable JSON file
        return Response(
            content=json_content,
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=project_{project_id}_board.json"}
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/export/{project_id}/markdown", tags=["Export"])
def export_markdown(project_id: int, session: Session = Depends(get_session)):
    try:
        markdown_content = ExportService.export_markdown(session, project_id)
        # Return as plain text or downloadable markdown file
        return Response(
            content=markdown_content,
            media_type="text/markdown",
            headers={"Content-Disposition": f"attachment; filename=project_{project_id}_board.md"}
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
