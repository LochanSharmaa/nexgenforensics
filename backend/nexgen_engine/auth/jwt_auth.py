from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Principal:
    user_id: str
    tenant_id: str
    role: str
    email: str


class AuthService:
    def __init__(self, secret_key: str, issuer: str = "nexgen") -> None:
        self.secret_key = secret_key.encode("utf-8")
        self.issuer = issuer

    def hash_password(self, password: str, salt: str) -> str:
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 200_000)
        return f"pbkdf2_sha256$200000${salt}${base64.urlsafe_b64encode(digest).decode('ascii')}"

    def verify_password(self, password: str, encoded: str) -> bool:
        algorithm, iterations, salt, digest = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        expected = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), int(iterations))
        return hmac.compare_digest(base64.urlsafe_b64encode(expected).decode("ascii"), digest)

    def issue_token(self, principal: Principal, ttl_seconds: int = 3600) -> str:
        header = {"alg": "HS256", "typ": "JWT"}
        payload = {
            "iss": self.issuer,
            "sub": principal.user_id,
            "tenant_id": principal.tenant_id,
            "role": principal.role,
            "email": principal.email,
            "exp": int(time.time()) + ttl_seconds,
        }
        signing_input = f"{_b64(header)}.{_b64(payload)}"
        signature = hmac.new(self.secret_key, signing_input.encode("ascii"), hashlib.sha256).digest()
        return f"{signing_input}.{_b64_bytes(signature)}"

    def verify_token(self, token: str) -> Principal:
        header_b64, payload_b64, signature_b64 = token.split(".", 2)
        signing_input = f"{header_b64}.{payload_b64}"
        expected = hmac.new(self.secret_key, signing_input.encode("ascii"), hashlib.sha256).digest()
        if not hmac.compare_digest(_b64_bytes(expected), signature_b64):
            raise PermissionError("Invalid token signature.")
        payload = json.loads(base64.urlsafe_b64decode(_pad(payload_b64)).decode("utf-8"))
        if int(payload["exp"]) < int(time.time()):
            raise PermissionError("Token expired.")
        return Principal(payload["sub"], payload["tenant_id"], payload["role"], payload["email"])


def _b64(payload: dict) -> str:
    return _b64_bytes(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))


def _b64_bytes(payload: bytes) -> str:
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _pad(value: str) -> bytes:
    return (value + "=" * (-len(value) % 4)).encode("ascii")
