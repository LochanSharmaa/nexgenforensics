"""Object storage port and a filesystem implementation.

Screenshots, HTML snapshots and images do not belong in PostgreSQL, so they live
behind this port. The filesystem implementation is what a local-first
single-user deployment actually needs; an S3/MinIO adapter slots in behind the
same protocol without touching a caller.

Keys are content-addressed by SHA256. Two consequences worth stating:

* Uploading the same image twice writes one object, and the second write is a
  no-op rather than a duplicate.
* A stored object's key *is* a checksum of its bytes, so verifying that a file
  was not altered after collection needs no separate manifest — the name proves
  it. That property is load-bearing for the reproducibility package.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Protocol

from .errors import StorageError
from .hashing import sha256_hex

# Segmenting by the first two hex pairs keeps directories from growing to
# hundreds of thousands of entries, which some filesystems handle badly.
_FANOUT = 2
_SEGMENT = 2


class ObjectStore(Protocol):
    """Blob storage. Implementations must be safe to call concurrently."""

    def put(self, data: bytes, *, prefix: str, extension: str = "") -> str: ...
    def get(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def delete(self, key: str) -> bool: ...
    def path_for(self, key: str) -> Path | None: ...


def content_key(data: bytes, *, prefix: str, extension: str = "") -> str:
    """Deterministic, content-addressed key."""
    digest = sha256_hex(data)
    segments = [digest[i * _SEGMENT : (i + 1) * _SEGMENT] for i in range(_FANOUT)]
    suffix = f".{extension.lstrip('.')}" if extension else ""
    return "/".join([prefix.strip("/"), *segments, f"{digest}{suffix}"])


class FilesystemObjectStore:
    """Stores objects under a root directory."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        """Resolve a key to a path, refusing anything outside the root.

        Keys are generated internally today, but this is the boundary where a
        traversal would land if one ever arrived from a request, so it is
        checked here rather than assumed upstream.
        """
        candidate = (self.root / key).resolve()
        if not candidate.is_relative_to(self.root):
            raise StorageError(f"Refusing to access {key!r}: outside the storage root.")
        return candidate

    def put(self, data: bytes, *, prefix: str, extension: str = "") -> str:
        key = content_key(data, prefix=prefix, extension=extension)
        destination = self._resolve(key)
        if destination.exists():
            return key   # content-addressed: identical bytes, nothing to do

        destination.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temporary neighbour and rename, so a crash mid-write cannot
        # leave a truncated file sitting at a key that claims to hash to it.
        staging = destination.with_suffix(destination.suffix + ".partial")
        try:
            staging.write_bytes(data)
            staging.replace(destination)
        except OSError as exc:
            staging.unlink(missing_ok=True)
            raise StorageError(f"Could not store object {key!r}: {exc}") from exc
        return key

    def get(self, key: str) -> bytes:
        path = self._resolve(key)
        try:
            return path.read_bytes()
        except FileNotFoundError as exc:
            raise StorageError(f"Object {key!r} not found.") from exc
        except OSError as exc:
            raise StorageError(f"Could not read object {key!r}: {exc}") from exc

    def exists(self, key: str) -> bool:
        return self._resolve(key).exists()

    def delete(self, key: str) -> bool:
        path = self._resolve(key)
        if not path.exists():
            return False
        path.unlink()
        return True

    def path_for(self, key: str) -> Path | None:
        path = self._resolve(key)
        return path if path.exists() else None

    def usage_bytes(self) -> int:
        return sum(f.stat().st_size for f in self.root.rglob("*") if f.is_file())

    def clear(self) -> None:
        """Tests only. Never call this from application code."""
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)


__all__ = ["FilesystemObjectStore", "ObjectStore", "content_key"]
