"""L0 -- evidence intake, content addressing, and the lineage DAG.

In court the contested question is rarely "what did the model output". It is
"what exactly was the model shown, and what was done to it first". A crop, a
re-encode, a rotation, a colour conversion -- each is a step someone can
challenge, and a system that cannot enumerate them cannot defend its output.

So lineage is not logging *about* the pipeline. It is the pipeline's data model.
Every artefact -- source media, extracted frame, aligned crop, embedding, score,
likelihood ratio -- is a node addressed by the SHA-256 of its own bytes, with an
edge recording the operation that produced it from its parents.

Two properties follow, and both matter:

    REPRODUCIBLE  -- an artefact's identity is its content, so an independent
                     party regenerating the pipeline lands on the same hashes or
                     discovers exactly which step diverged.

    TAMPER-EVIDENT -- the ledger is hash-chained, so altering or removing any
                      record invalidates every record after it.

This deliberately mirrors the existing audit chain in imatch_api rather than
replacing it: that chain records *actions*, this records *artefacts*.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

GENESIS = "0" * 64


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def canonical_digest(obj: object) -> str:
    """Digest of a structure, stable across runs and Python versions."""
    return sha256_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode())


@dataclass(frozen=True)
class Artifact:
    """A node in the lineage DAG."""

    digest: str
    kind: str  # source_media | frame | crop | embedding | score | lr | report
    operation: str  # what produced it
    parents: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)
    recorded_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def as_dict(self) -> dict:
        return {
            "digest": self.digest,
            "kind": self.kind,
            "operation": self.operation,
            "parents": list(self.parents),
            "metadata": self.metadata,
            "recorded_utc": self.recorded_utc,
        }


@dataclass
class LineageLedger:
    """Append-only, hash-chained record of artefacts.

    Chaining is over the ordered sequence of records: each entry incorporates the
    digest of its predecessor, so a removed or edited entry breaks every
    subsequent link and :meth:`verify` localises the break.
    """

    path: Path | None = None
    entries: list[dict] = field(default_factory=list)

    def _tip(self) -> str:
        return self.entries[-1]["chain"] if self.entries else GENESIS

    def record(self, artifact: Artifact) -> str:
        prev = self._tip()
        body = artifact.as_dict()
        entry = {"prev": prev, "artifact": body}
        entry["chain"] = canonical_digest(entry)
        self.entries.append(entry)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry["chain"]

    def verify(self) -> tuple[bool, str]:
        prev = GENESIS
        for i, entry in enumerate(self.entries):
            if entry["prev"] != prev:
                return False, f"chain broken at entry {i}: parent link mismatch"
            expected = canonical_digest({"prev": entry["prev"], "artifact": entry["artifact"]})
            if expected != entry["chain"]:
                return False, f"chain broken at entry {i}: content was altered after recording"
            prev = entry["chain"]
        return True, f"{len(self.entries)} entries verified"

    def ancestors(self, digest: str) -> list[dict]:
        """Full derivation history of an artefact, oldest first."""
        by_digest = {e["artifact"]["digest"]: e["artifact"] for e in self.entries}
        seen: list[dict] = []
        stack = [digest]
        visited = set()
        while stack:
            d = stack.pop()
            if d in visited or d not in by_digest:
                continue
            visited.add(d)
            art = by_digest[d]
            seen.append(art)
            stack.extend(art["parents"])
        return list(reversed(seen))

    @classmethod
    def load(cls, path: str | Path) -> "LineageLedger":
        p = Path(path)
        entries = []
        if p.exists():
            with open(p, encoding="utf-8") as fh:
                entries = [json.loads(line) for line in fh if line.strip()]
        return cls(path=p, entries=entries)


def ingest_file(ledger: LineageLedger, path: str | Path, **metadata) -> Artifact:
    """Admit source media, addressed by the SHA-256 of its bytes."""
    p = Path(path)
    art = Artifact(
        digest=sha256_file(p),
        kind="source_media",
        operation="ingest",
        metadata={"filename": p.name, "size_bytes": p.stat().st_size, **metadata},
    )
    ledger.record(art)
    return art


def derive(
    ledger: LineageLedger,
    kind: str,
    operation: str,
    parents: list[Artifact],
    payload: object,
    **metadata,
) -> Artifact:
    """Record an artefact derived from others, addressed by its own content."""
    art = Artifact(
        digest=canonical_digest(payload),
        kind=kind,
        operation=operation,
        parents=tuple(p.digest for p in parents),
        metadata=metadata,
    )
    ledger.record(art)
    return art


__all__ = [
    "GENESIS",
    "Artifact",
    "LineageLedger",
    "canonical_digest",
    "derive",
    "ingest_file",
    "sha256_bytes",
    "sha256_file",
]
