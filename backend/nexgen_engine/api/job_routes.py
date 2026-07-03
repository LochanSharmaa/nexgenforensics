from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from ..auth import AuthService, require_role
from ..jobs import JobQueue


class JobCreateRequest(BaseModel):
    job_type: str
    payload: dict = {}


def build_job_router(queue: JobQueue, auth: AuthService) -> APIRouter:
    router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])

    @router.post("")
    def create_job(payload: JobCreateRequest, authorization: str | None = Header(default=None)) -> dict:
        principal = _principal(auth, authorization)
        require_role(principal.role, "operator")
        job_id = queue.enqueue(principal.tenant_id, payload.job_type, payload.payload)
        return {"job_id": job_id, "status": "queued"}

    @router.get("/{job_id}")
    def get_job(job_id: str, authorization: str | None = Header(default=None)) -> dict:
        principal = _principal(auth, authorization)
        job = queue.status(job_id)
        if not job or job["tenant_id"] != principal.tenant_id:
            raise HTTPException(status_code=404, detail="Job not found.")
        return job

    return router


def _principal(auth: AuthService, authorization: str | None):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Bearer token required.")
    try:
        return auth.verify_token(authorization.split(" ", 1)[1])
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid token.") from exc
