from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AuditEntry:
    timestamp: str
    operator_id: str
    operation: str
    decision: str
    confidence: float
    metadata: dict[str, Any]
    previous_hash: str
    entry_hash: str


class AuditLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, operator_id: str, operation: str, decision: str, confidence: float, metadata: dict[str, Any]) -> AuditEntry:
        previous_hash = self._last_hash()
        timestamp = datetime.now(timezone.utc).isoformat()
        body = {
            "timestamp": timestamp,
            "operator_id": operator_id,
            "operation": operation,
            "decision": decision,
            "confidence": round(float(confidence), 6),
            "metadata": metadata,
            "previous_hash": previous_hash,
        }
        entry_hash = hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()
        entry = AuditEntry(**body, entry_hash=entry_hash)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(entry), sort_keys=True) + "\n")
        return entry

    def verify(self) -> bool:
        previous = ""
        if not self.path.exists():
            return True
        for line in self.path.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            entry_hash = entry.pop("entry_hash")
            if entry["previous_hash"] != previous:
                return False
            if hashlib.sha256(json.dumps(entry, sort_keys=True).encode("utf-8")).hexdigest() != entry_hash:
                return False
            previous = entry_hash
        return True

    def _last_hash(self) -> str:
        if not self.path.exists():
            return ""
        lines = [line for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            return ""
        return str(json.loads(lines[-1])["entry_hash"])
