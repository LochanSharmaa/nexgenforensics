"""Model weight resolution: cached, checksum-verified, and never surprising.

Three rules, each learned from a failure mode this project has already hit or
has explicitly designed against elsewhere:

1. **Never download implicitly.** A forensic tool that reaches the network
   during a case, on its own initiative, is a tool that can behave differently
   on two runs. Downloads happen when ``NEXGEN_ENHANCEMENT_ALLOW_DOWNLOAD`` is
   set, and are logged. Otherwise a missing weight file makes a backend
   *unavailable*, with a message saying exactly which file to place where.

2. **Verify the checksum, always.** ``runtime.py`` learned the equivalent lesson
   about CUDA: a component that reports itself present while being something
   else is worse than one that is absent. A silently corrupted or substituted
   checkpoint produces plausible output from the wrong model.

3. **One root, relocatable.** Defaults beside the InsightFace pack so an
   air-gapped install can be seeded by copying one directory.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

ENV_ROOT = "NEXGEN_ENHANCEMENT_MODEL_ROOT"
ENV_ALLOW_DOWNLOAD = "NEXGEN_ENHANCEMENT_ALLOW_DOWNLOAD"


def model_root() -> Path:
    configured = os.environ.get(ENV_ROOT)
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".nexgen" / "enhancement"


def downloads_allowed() -> bool:
    return os.environ.get(ENV_ALLOW_DOWNLOAD, "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class WeightSpec:
    """One checkpoint: where it lives, what it should hash to, where it came from."""

    filename: str
    sha256: str = ""
    url: str = ""
    notes: str = ""

    @property
    def path(self) -> Path:
        return model_root() / self.filename


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def check(spec: WeightSpec) -> tuple[bool, str]:
    """``(usable, reason_if_not)``. Never raises, never downloads."""
    path = spec.path
    if not path.is_file():
        hint = f" Download it from {spec.url} and place it there." if spec.url else ""
        if not downloads_allowed():
            hint += f" Or set {ENV_ALLOW_DOWNLOAD}=1 to fetch it automatically."
        return False, f"weights not present at {path}.{hint}"
    if spec.sha256:
        actual = sha256_file(path)
        if actual != spec.sha256:
            return False, (
                f"checksum mismatch for {path}: expected {spec.sha256[:16]}..., got {actual[:16]}.... "
                "The file is corrupt or is not the checkpoint this backend was validated against."
            )
    return True, ""


def resolve(spec: WeightSpec) -> Path:
    """Path to a verified checkpoint, fetching it only if explicitly permitted."""
    ok, reason = check(spec)
    if ok:
        return spec.path
    if not spec.path.is_file() and spec.url and downloads_allowed():
        _download(spec)
        ok, reason = check(spec)
        if ok:
            return spec.path
    raise FileNotFoundError(reason)


def _download(spec: WeightSpec) -> None:  # pragma: no cover - requires network
    import urllib.request  # noqa: PLC0415

    target = spec.path
    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading enhancement weights %s from %s", spec.filename, spec.url)
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(spec.url, timeout=120) as response, open(tmp, "wb") as handle:
            while block := response.read(1 << 20):
                handle.write(block)
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    logger.info("Downloaded %s (%d bytes)", spec.filename, target.stat().st_size)


def catalogue(specs: dict[str, WeightSpec]) -> list[dict[str, object]]:
    """Status of every declared checkpoint, for /status and for operators."""
    rows = []
    for name, spec in sorted(specs.items()):
        ok, reason = check(spec)
        rows.append(
            {
                "backend": name,
                "filename": spec.filename,
                "path": str(spec.path),
                "present": spec.path.is_file(),
                "verified": ok,
                "reason": reason,
                "sha256_expected": spec.sha256,
                "url": spec.url,
                "notes": spec.notes,
            }
        )
    return rows


__all__ = [
    "ENV_ALLOW_DOWNLOAD",
    "ENV_ROOT",
    "WeightSpec",
    "catalogue",
    "check",
    "downloads_allowed",
    "model_root",
    "resolve",
    "sha256_file",
]
