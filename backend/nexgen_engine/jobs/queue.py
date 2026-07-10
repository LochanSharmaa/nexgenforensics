from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from ..storage import Database


@dataclass(frozen=True)
class JobStatus:
    job_id: str
    tenant_id: str
    job_type: str
    status: str
    payload: dict
    error: str = ""
    progress: float = 0.0


class JobQueue:
    def __init__(self, database: Database) -> None:
        self.database = database
        self.database.migrate()

    def enqueue(self, tenant_id: str, job_type: str, payload: dict, priority: str = "normal") -> str:
        job_id = f"job_{uuid.uuid4().hex[:16]}"
        enriched_payload = dict(payload)
        enriched_payload.setdefault("priority", priority)
        enriched_payload.setdefault("progress", 0.0)
        self.database.create_job(job_id, tenant_id, job_type, json.dumps(enriched_payload, sort_keys=True))
        return job_id

    def status(self, job_id: str) -> dict | None:
        job = self.database.job_status(job_id)
        if not job:
            return None
        try:
            payload = json.loads(job.get("payload_json", "{}"))
        except json.JSONDecodeError:
            payload = {}
        job["progress"] = float(payload.get("progress", 0.0))
        job["priority"] = str(payload.get("priority", "normal"))
        return job

    def update_progress(self, job_id: str, progress: float) -> None:
        job = self.database.job_status(job_id)
        if not job:
            raise KeyError(job_id)
        payload = json.loads(job.get("payload_json", "{}"))
        payload["progress"] = max(0.0, min(1.0, float(progress)))
        self.database.update_job(job_id, job["status"], job.get("error", ""))
        with self.database.connect() as db:
            db.execute("UPDATE jobs SET payload_json = ?, updated_at = CURRENT_TIMESTAMP WHERE job_id = ?", (json.dumps(payload, sort_keys=True), job_id))


class JobWorker:
    def __init__(self, database: Database, handlers: dict[str, Callable[[dict], None]]) -> None:
        self.database = database
        self.handlers = handlers

    def run_job(self, job_id: str) -> None:
        job = self.database.job_status(job_id)
        if not job:
            raise KeyError(job_id)
        handler = self.handlers.get(job["job_type"])
        if handler is None:
            self.database.update_job(job_id, "failed", f"No handler for {job['job_type']}")
            return
        self.database.update_job(job_id, "running")
        try:
            handler(json.loads(job["payload_json"]))
        except Exception as exc:
            self.database.update_job(job_id, "failed", str(exc))
            return
        self.database.update_job(job_id, "succeeded")
