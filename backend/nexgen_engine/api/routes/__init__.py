from __future__ import annotations

import base64

from fastapi import APIRouter, HTTPException, Request

from ..service import EngineService


def build_engine_router(service: EngineService) -> APIRouter:
    router = APIRouter(prefix="/api/v1/engine", tags=["engine"])

    @router.post("/enroll")
    async def enroll(request: Request) -> dict:
        identity_id, workspace, payload = await _read_image_request(request, require_identity=True)
        assert identity_id is not None
        if not payload:
            raise HTTPException(status_code=400, detail="Image is required.")
        return service.enroll(payload, identity_id, {"workspace": workspace}).model_dump()

    @router.post("/identify")
    async def identify(request: Request) -> dict:
        operator_id, _, payload = await _read_image_request(request, require_identity=False)
        if not payload:
            raise HTTPException(status_code=400, detail="Image is required.")
        return service.identify(payload, operator_id=operator_id or "operator").model_dump()

    return router


async def _read_image_request(request: Request, require_identity: bool) -> tuple[str | None, str, bytes]:
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        data = await request.json()
        image_b64 = data.get("image_base64") or data.get("image")
        if not image_b64:
            raise HTTPException(status_code=400, detail="image_base64 is required.")
        try:
            payload = base64.b64decode(str(image_b64), validate=True)
        except Exception as exc:
            raise HTTPException(status_code=400, detail="image_base64 is invalid.") from exc
        identity_or_operator = data.get("identity_id") if require_identity else data.get("operator_id", "operator")
        if require_identity and not identity_or_operator:
            raise HTTPException(status_code=400, detail="identity_id is required.")
        return identity_or_operator, str(data.get("workspace", "default")), payload

    try:
        form = await request.form()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=415,
            detail="Multipart upload support requires python-multipart; send JSON with image_base64 or install backend requirements.",
        ) from exc
    image = form.get("image")
    if image is None or not hasattr(image, "read"):
        raise HTTPException(status_code=400, detail="Image is required.")
    payload = await image.read()
    identity_or_operator = str(form.get("identity_id") or form.get("operator_id") or "operator")
    if require_identity and not form.get("identity_id"):
        raise HTTPException(status_code=400, detail="identity_id is required.")
    return identity_or_operator, str(form.get("workspace") or "default"), payload


__all__ = ["build_engine_router"]
