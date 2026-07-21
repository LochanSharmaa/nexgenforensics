from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field

class Subscription(SQLModel, table=True):
    __tablename__ = "subscriptions"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    stripe_customer_id: str
    stripe_subscription_id: str
    plan: str  # free, premium, business
    status: str  # active, trialing, past_due, canceled
    current_period_end: datetime

class UsageEvent(SQLModel, table=True):
    __tablename__ = "usage_events"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    event_type: str  # concept_generation, image_generation
    quantity: int = Field(default=1)
    cost_estimate_usd: float = Field(default=0.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class AuditLog(SQLModel, table=True):
    __tablename__ = "audit_logs"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[int] = Field(default=None, foreign_key="users.id", index=True)
    action: str  # generate, save, reject, export, billing
    resource_type: str  # project, concept, billing
    resource_id: Optional[int] = None
    ip_hash: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
