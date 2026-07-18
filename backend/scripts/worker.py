from __future__ import annotations

import argparse
import json

from nexgen_engine.jobs import JobQueue, JobWorker
from nexgen_engine.settings import Settings
from nexgen_engine.storage import Database
from app.batch_reports import run_batch_report_job


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a NexGen local worker job")
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    settings = Settings.from_env()
    db_path = settings.database_url.replace("sqlite:///", "", 1) if settings.database_url.startswith("sqlite:///") else settings.runtime_dir / "nexgen.db"
    database = Database(db_path)
    queue = JobQueue(database)
    worker = JobWorker(
        database,
        {
            "dataset_ingestion": lambda payload: None,
            "enrollment_processing": lambda payload: None,
            "embedding_generation": lambda payload: None,
            "index_build": lambda payload: None,
            "model_export": lambda payload: None,
            "training": lambda payload: None,
            "noop": lambda payload: None,
            "forensic_report_batch": run_batch_report_job,
        },
    )
    worker.run_job(args.job_id)
    print(json.dumps(queue.status(args.job_id), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

