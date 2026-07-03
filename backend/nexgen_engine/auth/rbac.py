from __future__ import annotations


ROLE_ORDER = {"viewer": 1, "operator": 2, "tenant_admin": 3, "super_admin": 4}


def require_role(actual: str, required: str) -> None:
    if ROLE_ORDER.get(actual, 0) < ROLE_ORDER.get(required, 0):
        raise PermissionError(f"Role '{actual}' does not satisfy required role '{required}'.")
