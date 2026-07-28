from __future__ import annotations

import base64
import os
from dataclasses import dataclass

import numpy as np
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

TEMPLATE_KEY_BYTES = 32
_NONCE_BYTES = 12


class TemplateDecryptionError(RuntimeError):
    """Raised when a stored template cannot be decrypted or has been tampered with."""


@dataclass(frozen=True)
class EncryptedTemplate:
    """An AES-256-GCM sealed biometric template.

    ``tenant_id`` is bound in as additional authenticated data, so a ciphertext
    moved between tenant rows in the database fails to decrypt instead of
    silently matching under the wrong tenant.
    """

    nonce: str
    ciphertext: str
    dimensions: int
    tenant_id: str

    def as_dict(self) -> dict[str, object]:
        return {
            "nonce": self.nonce,
            "ciphertext": self.ciphertext,
            "dimensions": self.dimensions,
            "tenant_id": self.tenant_id,
        }


class TemplateEncryptor:
    """Encrypts biometric templates at rest with a single master key.

    Biometrics are irrevocable: a leaked template cannot be reissued like a
    password. Templates are therefore never stored in the clear.

    The master key is supplied once (``NEXGEN_TEMPLATE_KEY``, 32 random bytes,
    base64) and a per-tenant subkey is derived from it with HKDF. Deriving with
    HKDF rather than running PBKDF2 per record matters: the previous version ran
    600k PBKDF2 iterations on every encrypt AND every decrypt, which would add
    roughly a second of CPU to each template touched and make a gallery load of
    any size unusable. PBKDF2 is still used, but only on the ``from_passphrase``
    path where the input is genuinely low-entropy.
    """

    def __init__(self, master_key: bytes) -> None:
        if len(master_key) != TEMPLATE_KEY_BYTES:
            raise ValueError(f"Master key must be exactly {TEMPLATE_KEY_BYTES} bytes, got {len(master_key)}.")
        self._master_key = master_key
        self._subkeys: dict[str, bytes] = {}

    # ------------------------------------------------------------ factories --

    @classmethod
    def from_base64(cls, encoded: str) -> "TemplateEncryptor":
        try:
            return cls(base64.b64decode(encoded, validate=True))
        except Exception as exc:
            raise ValueError(
                "NEXGEN_TEMPLATE_KEY must be 32 random bytes, base64-encoded. Generate with: "
                'python -c "import base64,os; print(base64.b64encode(os.urandom(32)).decode())"'
            ) from exc

    @classmethod
    def from_passphrase(cls, passphrase: str, salt: bytes, iterations: int = 600_000) -> "TemplateEncryptor":
        """Derive a master key from a human-chosen passphrase.

        Only for operators who cannot manage a random key. The salt must be
        stored and reused, or every previously sealed template becomes
        unreadable.
        """
        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=TEMPLATE_KEY_BYTES, salt=salt, iterations=iterations)
        return cls(kdf.derive(passphrase.encode("utf-8")))

    @classmethod
    def generate(cls) -> "TemplateEncryptor":
        """Ephemeral key, for tests only. Templates do not survive the process."""
        return cls(os.urandom(TEMPLATE_KEY_BYTES))

    # -------------------------------------------------------------- crypto ---

    def encrypt(self, embedding: np.ndarray, tenant_id: str) -> EncryptedTemplate:
        vector = np.ascontiguousarray(np.asarray(embedding, dtype=np.float32))
        if vector.ndim != 1:
            raise ValueError(f"Expected a 1-D template, got shape {vector.shape}.")

        nonce = os.urandom(_NONCE_BYTES)
        ciphertext = AESGCM(self._subkey(tenant_id)).encrypt(nonce, vector.tobytes(), tenant_id.encode("utf-8"))
        return EncryptedTemplate(
            nonce=base64.b64encode(nonce).decode("ascii"),
            ciphertext=base64.b64encode(ciphertext).decode("ascii"),
            dimensions=int(vector.shape[0]),
            tenant_id=tenant_id,
        )

    def decrypt(self, encrypted: EncryptedTemplate) -> np.ndarray:
        try:
            payload = AESGCM(self._subkey(encrypted.tenant_id)).decrypt(
                base64.b64decode(encrypted.nonce),
                base64.b64decode(encrypted.ciphertext),
                encrypted.tenant_id.encode("utf-8"),
            )
        except (InvalidTag, ValueError) as exc:
            raise TemplateDecryptionError(
                "Template failed authenticated decryption. The master key is wrong, or the "
                "stored record was altered or moved between tenants."
            ) from exc

        vector = np.frombuffer(payload, dtype=np.float32).copy()
        if vector.shape[0] != encrypted.dimensions:
            raise TemplateDecryptionError(
                f"Template length mismatch: expected {encrypted.dimensions}, decoded {vector.shape[0]}."
            )
        return vector

    def _subkey(self, tenant_id: str) -> bytes:
        """Per-tenant key derived from the master key.

        Compromising one tenant's derived key does not expose another's, and the
        HKDF info binding makes cross-tenant ciphertext reuse fail loudly.
        """
        cached = self._subkeys.get(tenant_id)
        if cached is not None:
            return cached
        subkey = HKDF(
            algorithm=hashes.SHA256(),
            length=TEMPLATE_KEY_BYTES,
            salt=None,
            info=b"nexgen-imatch-template:" + tenant_id.encode("utf-8"),
        ).derive(self._master_key)
        self._subkeys[tenant_id] = subkey
        return subkey


__all__ = [
    "TEMPLATE_KEY_BYTES",
    "EncryptedTemplate",
    "TemplateDecryptionError",
    "TemplateEncryptor",
]
