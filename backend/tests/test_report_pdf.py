from datetime import datetime, timezone
from pathlib import Path



from app.schemas import ForensicFindings, MeasurementFinding, ModelSimilarityScore, SourceImageFindings
from app.services.report_pdf import TenantReportStorage, render_and_store_report, render_report_html


def test_pdf_contains_required_sections_scope_and_audit(tmp_path: Path):
    findings = ForensicFindings(case_id='CASE-9', examiner_id='examiner', timestamp=datetime.now(timezone.utc), tenant_id='tenant-9', source_images=[SourceImageFindings(sha256='a'*64, quality_score=.9, pose_yaw=0, pose_pitch=0, liveness_score=.9)], model_scores=[ModelSimilarityScore(model_name='model-a', score=.93)], fused_score=.93, calibrated_match_probability=93, decision='match', threshold_value=.8, false_match_rate=.001, measurements=[MeasurementFinding(name='symmetry_score', value=.97, confidence=1)], audit_hash='b'*64, config_version='test-config')
    path = render_and_store_report(findings, TenantReportStorage(tmp_path))
    assert path.exists() and path.parent.name == 'tenant-9'
    text = ''.join(page.extract_text() or '' for page in PdfReader(str(path)).pages)
    for section in ('Automated Facial Comparison Report', 'Landmark-based Geometric Measurements', 'Per-model Similarity Scores', 'Fused Score and Decision', 'Scope Statement', findings.audit_hash):
        assert section in text

