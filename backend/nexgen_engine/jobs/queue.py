from __future__ import annotations

import json
import uuid
from collections.abc import Callable

from ..storage import Database


class JobQueue:
    def __init__(self, database: Database) -> None:
        self.database = database
        self.database.migrate()

    def enqueue(self, tenant_id: str, job_type: str, payload: dict) -> str:
        job_id = f"job_{uuid.uuid4().hex[:16]}"
        self.database.create_job(job_id, tenant_id, job_type, json.dumps(payload, sort_keys=True))
        return job_id

    def status(self, job_id: str) -> dict | None:
        return self.database.job_status(job_id)


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
