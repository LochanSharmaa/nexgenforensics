from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class ClientPackageManifest:
    client_id: str
    package_version: str
    generated_at: str
    includes: list[str]
    validation_required: bool


def package_for_client(client_id: str, output_dir: str | Path, package_version: str = "0.1.0") -> ClientPackageManifest:
    target_dir = Path(output_dir) / client_id
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest = ClientPackageManifest(
        client_id=client_id,
        package_version=package_version,
        generated_at=datetime.now(timezone.utc).isoformat(),
        includes=["api", "engine_config", "security_config", "deployment_config"],
        validation_required=True,
    )
    (target_dir / "client_package_manifest.json").write_text(json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8")
    return manifest
