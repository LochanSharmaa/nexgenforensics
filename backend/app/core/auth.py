"""
Authentication middleware for SIFS Creative Reasoning Engine.

Uses Clerk JWT verification in production.
Falls back to a default dev user when CLERK_SECRET_KEY is empty (local dev).
"""

import logging
import httpx
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlmodel import Session, select

from app.core.config import settings
from app.db.database import get_session
from app.models.user import User

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)

# Cache for JWKS keys
_jwks_cache: Optional[dict] = None


def _get_or_create_dev_user(session: Session) -> User:
    """Get or create a default dev user for local development."""
    dev_user = session.exec(
        select(User).where(User.email == "dev@sifs.local")
    ).first()
    if not dev_user:
        dev_user = User(
            name="Dev Designer",
            email="dev@sifs.local",
            auth_provider_id="dev_local",
            plan_tier="premium",  # Premium for dev convenience
        )
        session.add(dev_user)
        session.commit()
        session.refresh(dev_user)
    return dev_user


async def _verify_clerk_token(token: str) -> dict:
    """Verify a Clerk JWT and return its claims."""
    try:
        from jose import jwt, JWTError

        global _jwks_cache

        # Fetch JWKS if not cached
        if _jwks_cache is None:
            jwks_url = settings.CLERK_JWKS_URL
            if not jwks_url:
                # Derive from Clerk publishable key or fall back
                raise ValueError("CLERK_JWKS_URL not configured")

            async with httpx.AsyncClient() as client:
                resp = await client.get(jwks_url)
                resp.raise_for_status()
                _jwks_cache = resp.json()

        # Decode without verification first to get the key ID
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")

        # Find the matching key
        rsa_key = None
        for key in _jwks_cache.get("keys", []):
            if key.get("kid") == kid:
                rsa_key = key
                break

        if not rsa_key:
            raise JWTError("Unable to find matching key")

        # Verify and decode
        claims = jwt.decode(
            token,
            rsa_key,
            algorithms=["RS256"],
            options={"verify_aud": False},
        )
        return claims

    except Exception as e:
        logger.error(f"Clerk token verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token",
        )


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    session: Session = Depends(get_session),
) -> User:
    """
    FastAPI dependency that returns the authenticated User.

    - Production (CLERK_SECRET_KEY set): verifies Clerk JWT, looks up/creates user
    - Development (CLERK_SECRET_KEY empty): returns a default dev user
    """
    # Dev mode fallback
    if not settings.CLERK_SECRET_KEY:
        return _get_or_create_dev_user(session)

    # Production: require token
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    claims = await _verify_clerk_token(credentials.credentials)

    # Extract user info from Clerk claims
    clerk_user_id = claims.get("sub", "")
    email = claims.get("email", claims.get("email_addresses", [{}])[0].get("email_address", ""))
    name = claims.get("name", claims.get("first_name", "Designer"))

    if not clerk_user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token claims",
        )

    # Find or create the user
    user = session.exec(
        select(User).where(User.auth_provider_id == clerk_user_id)
    ).first()

    if not user:
        user = User(
            name=name,
            email=email or f"{clerk_user_id}@clerk.user",
            auth_provider_id=clerk_user_id,
            plan_tier="free",
        )
        session.add(user)
        session.commit()
        session.refresh(user)

    return user
