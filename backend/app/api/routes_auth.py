"""
Auth routes: /api/me and related user endpoints.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

from app.core.auth import get_current_user
from app.models.user import User

router = APIRouter()


class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    plan_tier: str
    created_at: datetime

    class Config:
        from_attributes = True


@router.get("/me", response_model=UserResponse, tags=["Auth"])
async def get_me(current_user: User = Depends(get_current_user)):
    """Return the currently authenticated user's profile."""
    return current_user
