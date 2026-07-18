import csv, json
from pathlib import Path
from app.batch_reports import run_batch_report_job

def source(v): return {'sha256':v*64,'quality_score':.9,'pose_yaw':0,'pose_pitch':0,'liveness_score':.9}
def test_batch_worker_creates_tenant_scoped_summary_files(tmp_path: Path):
 p={'batch_id':'batch-test','tenant_id':'tenant-a','examiner_id':'worker','root':str(tmp_path),'audit_path':str(tmp_path/'audit.jsonl'),'pairs':[{'pair_id':'pair-a','source_images':[source('a')],'model_scores':[{'model_name':'m','score':.9}],'fused_score':.9},{'pair_id':'pair-b','source_images':[source('b')],'model_scores':[{'model_name':'m','score':.2}],'fused_score':.2}]}
 run_batch_report_job(p); folder=tmp_path/'tenant-a'; c=folder/'batch-test-summary.csv'; j=folder/'batch-test-summary.json'
 assert c.exists() and j.exists()
 with c.open() as h: assert len(list(csv.DictReader(h)))==2
 summary=json.loads(j.read_text()); assert summary['total_pairs']==2 and len(summary['pairs'])==2 and summary['match']==1 and summary['no_match']==1
