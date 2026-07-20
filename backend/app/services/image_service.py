import uuid
from datetime import datetime
from typing import List, Optional
from sqlmodel import Session, select
from app.models.concept import Concept
from app.models.generated_image import GeneratedImage
from app.models.image_job import ImageJob
from app.services.provider_router import get_image_provider
from app.schemas.image import ImageGenerateRequest, ImageGenerateBatchRequest
from app.core.config import settings

class ImageService:
    @staticmethod
    def generate_image_sync(session: Session, request: ImageGenerateRequest) -> GeneratedImage:
        concept = session.get(Concept, request.concept_id)
        if not concept:
            raise ValueError("Concept not found")

        provider = get_image_provider()
        res = provider.generate_reference_image(
            concept_prompt=concept.reference_image_prompt,
            style=concept.style_category,
            size=request.size or "1024x1024"
        )

        db_img = GeneratedImage(
            project_id=concept.project_id,
            concept_id=concept.id,
            image_url=res["image_url"],
            prompt=res["prompt"],
            provider=res["provider"],
            reference_only=res["reference_only"]
        )
        session.add(db_img)
        session.commit()
        session.refresh(db_img)
        return db_img

    @staticmethod
    def create_image_job(session: Session, request: ImageGenerateRequest) -> ImageJob:
        concept = session.get(Concept, request.concept_id)
        if not concept:
            raise ValueError("Concept not found")

        job_id = str(uuid.uuid4())
        db_job = ImageJob(
            id=job_id,
            project_id=concept.project_id,
            concept_id=concept.id,
            status="queued",
            provider=settings.IMAGE_PROVIDER,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        session.add(db_job)
        session.commit()
        session.refresh(db_job)

        # In Phase 1/2, to simulate background worker without Redis dependency:
        # We can launch a background task or run a simulated execution.
        # We will implement real Redis RQ queuing in Phase 4.
        return db_job

    @staticmethod
    def create_batch_jobs(session: Session, request: ImageGenerateBatchRequest) -> List[ImageJob]:
        jobs = []
        for cid in request.concept_ids:
            req = ImageGenerateRequest(concept_id=cid, size=request.size)
            job = ImageService.create_image_job(session, req)
            jobs.append(job)
        return jobs

    @staticmethod
    def get_job_status(session: Session, job_id: str) -> Optional[ImageJob]:
        return session.get(ImageJob, job_id)

    @staticmethod
    def get_images_by_project(session: Session, project_id: int) -> List[GeneratedImage]:
        statement = select(GeneratedImage).where(GeneratedImage.project_id == project_id)
        return session.exec(statement).all()
