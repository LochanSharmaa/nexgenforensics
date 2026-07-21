from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional

from app.core.auth import get_current_user
from app.models.user import User
from app.schemas.settings import ProviderSettingsRead, TestProviderRequest, TestProviderResponse
from app.services.settings_service import SettingsService

router = APIRouter()


@router.get("/settings/providers", response_model=ProviderSettingsRead, tags=["Settings"])
async def get_provider_settings(
    current_user: User = Depends(get_current_user),
):
    return SettingsService.get_active_settings()


@router.post("/settings/test-llm", response_model=TestProviderResponse, tags=["Settings"])
async def test_llm_connection(
    request: TestProviderRequest,
    current_user: User = Depends(get_current_user),
):
    return SettingsService.test_llm_connection(request)


@router.post("/settings/test-image", response_model=TestProviderResponse, tags=["Settings"])
async def test_image_connection(
    request: TestProviderRequest,
    current_user: User = Depends(get_current_user),
):
    return SettingsService.test_image_connection(request)
