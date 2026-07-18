from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .forensic_report_routes import generate_automated_report
from .schemas import ModelSimilarityScore, SourceImageFindings
from nexgen_engine.security.audit_logger import AuditLogger


def run_batch_report_job(payload: dict[str, Any]) -> None:
    """Worker handler; summary names use the job's batch id for stable retrieval."""
    root = Path(payload['root'])
    tenant_id = payload['tenant_id']
    batch_id = payload['batch_id']
    audit = AuditLogger(payload['audit_path'])
    rows: list[dict[str, Any]] = []
    for pair in payload['pairs']:
        result = generate_automated_report(
            audit=audit, root=root, case_id=pair['pair_id'], tenant_id=tenant_id,
            examiner_id=payload['examiner_id'],
            source_images=[SourceImageFindings.model_validate(item) for item in pair['source_images']],
            model_scores=[ModelSimilarityScore.model_validate(item) for item in pair['model_scores']],
            fused_score=float(pair['fused_score']), threshold_value=float(pair.get('threshold_value', .72)),
        )
        rows.append({'image_pair_ids': pair['pair_id'], 'decision': result['findings']['decision'], 'calibrated_match_probability': result['findings']['calibrated_match_probability'], 'report_url': result['report_path']})
    directory = root / tenant_id
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / f'{batch_id}-summary.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['image_pair_ids', 'decision', 'calibrated_match_probability', 'report_url'])
        writer.writeheader(); writer.writerows(rows)
    counts = {key: sum(row['decision'] == key for row in rows) for key in ('match', 'no_match', 'inconclusive')}
    json_path = directory / f'{batch_id}-summary.json'
    json_path.write_text(json.dumps({'batch_id': batch_id, 'tenant': tenant_id, 'timestamp': datetime.now(timezone.utc).isoformat(), 'total_pairs': len(rows), **counts, 'pairs': rows}, indent=2), encoding='utf-8')
