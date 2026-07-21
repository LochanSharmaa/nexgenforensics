import threading
import time
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from sqlmodel import Session
from typing import List

from app.db.database import get_session, engine
from app.core.auth import get_current_user
from app.models.user import User
from app.models.project import Project
from app.schemas.image import (
    ImageGenerateRequest,
    ImageGenerateBatchRequest,
    ImageResponse,
    ImageJobRead,
)
from app.services.image_service import ImageService
from app.models.image_job import ImageJob
from app.models.generated_image import GeneratedImage

router = APIRouter()


def simulate_image_worker(job_id: str, concept_id: int):
    """Thread-based background worker for image generation (pre-Celery)."""
    time.sleep(2.0)
    with Session(engine) as session:
        job = session.get(ImageJob, job_id)
        if not job:
            return

        job.status = "running"
        session.add(job)
        session.commit()

        time.sleep(2.0)
        try:
            req = ImageGenerateRequest(concept_id=concept_id)
            img = ImageService.generate_image_sync(session, req)

            job = session.get(ImageJob, job_id)
            job.status = "completed"
            job.image_url = img.image_url
            session.add(job)
            session.commit()
        except Exception as e:
            job = session.get(ImageJob, job_id)
            job.status = "failed"
            job.error_message_sanitized = str(e)
            session.add(job)
            session.commit()


@router.post("/images/generate", response_model=ImageJobRead, tags=["Images"])
async def generate_image(
    request: ImageGenerateRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    try:
        job = ImageService.create_image_job(session, request)
        background_tasks.add_task(simulate_image_worker, job.id, request.concept_id)
        return job
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/images/jobs/{job_id}", response_model=ImageJobRead, tags=["Images"])
async def get_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    job = ImageService.get_job_status(session, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get(
    "/projects/{project_id}/images",
    response_model=List[ImageResponse],
    tags=["Images"],
)
async def get_project_images(
    project_id: int,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    # Verify ownership
    project = session.get(Project, project_id)
    if not project or project.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Project not found")

    images = ImageService.get_images_by_project(session, project_id)
    return [
        ImageResponse(
            image_url=img.image_url,
            prompt=img.prompt,
            provider=img.provider,
            reference_only=img.reference_only,
            notice="Reference only \u2014 final artwork belongs to the designer.",
        )
        for img in images
    ]
