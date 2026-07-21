from __future__ import annotations

import base64
import os
from dataclasses import dataclass

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes


@dataclass(frozen=True)
class EncryptedTemplate:
    salt: str
    nonce: str
    ciphertext: str
    dimensions: int


class TemplateEncryptor:
    def __init__(self, iterations: int = 600_000) -> None:
        self.iterations = iterations

    def encrypt(self, embedding: np.ndarray, secret: str) -> EncryptedTemplate:
        salt = os.urandom(32)
        nonce = os.urandom(12)
        key = self._derive_key(secret, salt)
        payload = np.asarray(embedding, dtype=np.float32).tobytes()
        ciphertext = AESGCM(key).encrypt(nonce, payload, None)
        return EncryptedTemplate(
            salt=base64.b64encode(salt).decode("ascii"),
            nonce=base64.b64encode(nonce).decode("ascii"),
            ciphertext=base64.b64encode(ciphertext).decode("ascii"),
            dimensions=int(np.asarray(embedding).shape[0]),
        )

    def decrypt(self, encrypted: EncryptedTemplate, secret: str) -> np.ndarray:
        salt = base64.b64decode(encrypted.salt)
        nonce = base64.b64decode(encrypted.nonce)
        ciphertext = base64.b64decode(encrypted.ciphertext)
        key = self._derive_key(secret, salt)
        payload = AESGCM(key).decrypt(nonce, ciphertext, None)
        return np.frombuffer(payload, dtype=np.float32).copy()

    def _derive_key(self, secret: str, salt: bytes) -> bytes:
        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=self.iterations)
        return kdf.derive(secret.encode("utf-8"))
