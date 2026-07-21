"""
Billing routes: Stripe checkout session creation and webhook handling.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlmodel import Session, select
from pydantic import BaseModel

from app.core.config import settings
from app.core.auth import get_current_user
from app.db.database import get_session
from app.models.user import User
from app.models.billing import Subscription, AuditLog

logger = logging.getLogger(__name__)

router = APIRouter()


class CreateCheckoutSessionRequest(BaseModel):
    plan: str  # "premium" or "business"
    success_url: str = "http://localhost:3000/settings/billing?success=true"
    cancel_url: str = "http://localhost:3000/settings/billing?canceled=true"


class CheckoutSessionResponse(BaseModel):
    checkout_url: str
    session_id: str


@router.post(
    "/billing/create-checkout-session",
    response_model=CheckoutSessionResponse,
    tags=["Billing"],
)
async def create_checkout_session(
    request: CreateCheckoutSessionRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    if not settings.STRIPE_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Stripe is not configured",
        )

    try:
        import stripe

        stripe.api_key = settings.STRIPE_SECRET_KEY

        # Determine which price to use
        price_id = (
            settings.STRIPE_PRICE_PREMIUM
            if request.plan == "premium"
            else settings.STRIPE_PRICE_BUSINESS
        )
        if not price_id:
            raise HTTPException(
                status_code=400,
                detail=f"Stripe price not configured for plan: {request.plan}",
            )

        # Check for existing Stripe customer
        existing_sub = session.exec(
            select(Subscription).where(Subscription.user_id == current_user.id)
        ).first()

        customer_id = existing_sub.stripe_customer_id if existing_sub else None

        # Create or reuse Stripe customer
        if not customer_id:
            customer = stripe.Customer.create(
                email=current_user.email,
                name=current_user.name,
                metadata={"sifs_user_id": str(current_user.id)},
            )
            customer_id = customer.id

        # Create checkout session
        checkout_session = stripe.checkout.Session.create(
            customer=customer_id,
            payment_method_types=["card"],
            line_items=[{"price": price_id, "quantity": 1}],
            mode="subscription",
            success_url=request.success_url,
            cancel_url=request.cancel_url,
            metadata={"sifs_user_id": str(current_user.id), "plan": request.plan},
        )

        # Audit log
        audit = AuditLog(
            user_id=current_user.id,
            action="billing",
            resource_type="billing",
            resource_id=None,
        )
        session.add(audit)
        session.commit()

        return CheckoutSessionResponse(
            checkout_url=checkout_session.url,
            session_id=checkout_session.id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stripe checkout error: {e}")
        raise HTTPException(status_code=500, detail="Failed to create checkout session")


@router.post("/billing/webhook", tags=["Billing"])
async def stripe_webhook(
    request: Request,
    session: Session = Depends(get_session),
):
    """Handle Stripe webhook events. No auth required — verified by Stripe signature."""
    if not settings.STRIPE_SECRET_KEY or not settings.STRIPE_WEBHOOK_SECRET:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Stripe webhooks not configured",
        )

    try:
        import stripe

        stripe.api_key = settings.STRIPE_SECRET_KEY

        body = await request.body()
        sig_header = request.headers.get("stripe-signature", "")

        event = stripe.Webhook.construct_event(
            body, sig_header, settings.STRIPE_WEBHOOK_SECRET
        )
    except Exception as e:
        logger.error(f"Stripe webhook signature verification failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid webhook signature")

    event_type = event.get("type", "")
    data_object = event.get("data", {}).get("object", {})

    if event_type == "checkout.session.completed":
        _handle_checkout_completed(session, data_object)
    elif event_type == "customer.subscription.updated":
        _handle_subscription_updated(session, data_object)
    elif event_type == "customer.subscription.deleted":
        _handle_subscription_deleted(session, data_object)

    return {"received": True}


def _handle_checkout_completed(session: Session, checkout_data: dict):
    """Process a successful checkout — create or update subscription."""
    metadata = checkout_data.get("metadata", {})
    sifs_user_id = metadata.get("sifs_user_id")
    plan = metadata.get("plan", "premium")

    if not sifs_user_id:
        logger.warning("Checkout completed but no sifs_user_id in metadata")
        return

    user_id = int(sifs_user_id)
    user = session.get(User, user_id)
    if not user:
        logger.warning(f"User {user_id} not found for checkout")
        return

    stripe_customer_id = checkout_data.get("customer", "")
    stripe_subscription_id = checkout_data.get("subscription", "")

    # Upsert subscription
    existing = session.exec(
        select(Subscription).where(Subscription.user_id == user_id)
    ).first()

    if existing:
        existing.stripe_customer_id = stripe_customer_id
        existing.stripe_subscription_id = stripe_subscription_id
        existing.plan = plan
        existing.status = "active"
        session.add(existing)
    else:
        from datetime import datetime, timedelta

        sub = Subscription(
            user_id=user_id,
            stripe_customer_id=stripe_customer_id,
            stripe_subscription_id=stripe_subscription_id,
            plan=plan,
            status="active",
            current_period_end=datetime.utcnow() + timedelta(days=30),
        )
        session.add(sub)

    # Update user plan tier
    user.plan_tier = plan
    session.add(user)
    session.commit()

    logger.info(f"User {user_id} upgraded to {plan}")


def _handle_subscription_updated(session: Session, sub_data: dict):
    """Handle subscription updates (plan change, renewal)."""
    stripe_sub_id = sub_data.get("id", "")
    sub = session.exec(
        select(Subscription).where(
            Subscription.stripe_subscription_id == stripe_sub_id
        )
    ).first()
    if not sub:
        return

    status_val = sub_data.get("status", sub.status)
    sub.status = status_val

    # Update period end
    period_end = sub_data.get("current_period_end")
    if period_end:
        from datetime import datetime

        sub.current_period_end = datetime.utcfromtimestamp(period_end)

    session.add(sub)
    session.commit()


def _handle_subscription_deleted(session: Session, sub_data: dict):
    """Handle subscription cancellation — downgrade user to free."""
    stripe_sub_id = sub_data.get("id", "")
    sub = session.exec(
        select(Subscription).where(
            Subscription.stripe_subscription_id == stripe_sub_id
        )
    ).first()
    if not sub:
        return

    sub.status = "canceled"
    session.add(sub)

    user = session.get(User, sub.user_id)
    if user:
        user.plan_tier = "free"
        session.add(user)

    session.commit()
    logger.info(f"User {sub.user_id} subscription canceled, downgraded to free")
