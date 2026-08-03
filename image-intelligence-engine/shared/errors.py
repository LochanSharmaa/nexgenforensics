"""Typed exception hierarchy.

Retry logic differs by class, so the hierarchy exists to be branched on rather
than to decorate messages. A provider rate-limit is retryable; a malformed
lawful basis is not; an illegal state transition never will be.

Every error carries a stable ``type_slug`` used as the RFC 9457
``application/problem+json`` ``type``. Clients match on the slug, never on
message text, so wording can improve without breaking consumers.
"""

from __future__ import annotations

from typing import Any


class IIEError(Exception):
    """Root of every deliberate failure in the platform."""

    type_slug: str = "internal-error"
    http_status: int = 500
    retryable: bool = False

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context = context

    def as_problem(self, instance: str = "") -> dict[str, Any]:
        body: dict[str, Any] = {
            "type": f"https://iie.invalid/problems/{self.type_slug}",
            "title": self.__class__.__name__,
            "status": self.http_status,
            "detail": self.message,
        }
        if instance:
            body["instance"] = instance
        if self.context:
            body["context"] = self.context
        return body


# -- configuration ---------------------------------------------------------


class ConfigurationError(IIEError):
    type_slug = "configuration-error"
    http_status = 500


# -- request-shaped --------------------------------------------------------


class ValidationError(IIEError):
    type_slug = "validation-error"
    http_status = 422


class NotFoundError(IIEError):
    type_slug = "not-found"
    http_status = 404


class ConflictError(IIEError):
    type_slug = "conflict"
    http_status = 409


class AuthenticationError(IIEError):
    type_slug = "authentication-required"
    http_status = 401


class AuthorizationError(IIEError):
    type_slug = "forbidden"
    http_status = 403


# -- policy ----------------------------------------------------------------


class PolicyViolation(IIEError):
    """A request the platform refuses on policy grounds.

    Distinct from ValidationError: the request is well-formed, and we decline it.
    Refusals are audited, so this class is caught explicitly by the middleware.
    """

    type_slug = "policy-violation"
    http_status = 422


class LawfulBasisRequired(PolicyViolation):
    type_slug = "lawful-basis-required"


class RetentionHoldActive(PolicyViolation):
    """Purge blocked by an unreleased hold. Holds always win over policy."""

    type_slug = "retention-hold-active"
    http_status = 409


class BiometricProcessingRefused(PolicyViolation):
    """Tripped if any code path attempts facial identification.

    Should be unreachable — the lockfile guard and architecture tests prevent a
    face library from ever entering the build. It exists as a last backstop so
    the failure is loud and named rather than a confusing ImportError.
    """

    type_slug = "biometric-processing-refused"
    http_status = 403


# -- domain ----------------------------------------------------------------


class DomainError(IIEError):
    type_slug = "domain-error"
    http_status = 409


class StateTransitionError(DomainError):
    type_slug = "illegal-state-transition"


class EvidenceIntegrityError(DomainError):
    """A hash chain failed verification, or a fact lacks supporting evidence."""

    type_slug = "evidence-integrity-error"
    http_status = 500


# -- infrastructure --------------------------------------------------------


class InfrastructureError(IIEError):
    type_slug = "infrastructure-error"
    retryable = True


class StorageError(InfrastructureError):
    type_slug = "storage-error"


class ProviderError(InfrastructureError):
    type_slug = "provider-error"
    http_status = 502


class ProviderNotConfigured(ProviderError):
    """Credentials absent. Reported distinctly from 'returned no results' —
    conflating them puts a false negative in a report."""

    type_slug = "provider-not-configured"
    http_status = 503
    retryable = False


class ProviderRateLimited(ProviderError):
    type_slug = "provider-rate-limited"
    http_status = 429


class FetchError(InfrastructureError):
    type_slug = "fetch-error"
    http_status = 502


class BlockedTargetError(FetchError):
    """SSRF guard refused the target. Never retryable — the address will not
    become public on a second attempt."""

    type_slug = "blocked-target"
    http_status = 400
    retryable = False


__all__ = [
    "AuthenticationError",
    "AuthorizationError",
    "BiometricProcessingRefused",
    "BlockedTargetError",
    "ConfigurationError",
    "ConflictError",
    "DomainError",
    "EvidenceIntegrityError",
    "FetchError",
    "IIEError",
    "InfrastructureError",
    "LawfulBasisRequired",
    "NotFoundError",
    "PolicyViolation",
    "ProviderError",
    "ProviderNotConfigured",
    "ProviderRateLimited",
    "RetentionHoldActive",
    "StateTransitionError",
    "StorageError",
    "ValidationError",
]
