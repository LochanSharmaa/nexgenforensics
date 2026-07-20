from fastapi import APIRouter, Depends
from app.schemas.settings import ProviderSettingsRead, TestProviderRequest, TestProviderResponse
from app.services.settings_service import SettingsService

router = APIRouter()

@router.get("/settings/providers", response_model=ProviderSettingsRead, tags=["Settings"])
def get_provider_settings():
    return SettingsService.get_active_settings()

@router.post("/settings/test-llm", response_model=TestProviderResponse, tags=["Settings"])
def test_llm_connection(request: TestProviderRequest):
    return SettingsService.test_llm_connection(request)

@router.post("/settings/test-image", response_model=TestProviderResponse, tags=["Settings"])
def test_image_connection(request: TestProviderRequest):
    return SettingsService.test_image_connection(request)
