from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field

class User(SQLModel, table=True):
    __tablename__ = "users"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    email: str = Field(unique=True, index=True)
    auth_provider_id: Optional[str] = Field(default=None, index=True)
    plan_tier: str = Field(default="free")  # free, premium, business
    created_at: datetime = Field(default_factory=datetime.utcnow)
