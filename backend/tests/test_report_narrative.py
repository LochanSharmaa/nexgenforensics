from datetime import datetime, timezone

from app.schemas import ForensicFindings, ModelSimilarityScore, SourceImageFindings
from app.services.report_narrative import STANDARD_SCOPE_TEXT, build_narrative


def findings(quality=0.9, yaw=0.0, liveness=0.9):
    return ForensicFindings(case_id='CASE-1', examiner_id='examiner', timestamp=datetime.now(timezone.utc), tenant_id='tenant', source_images=[SourceImageFindings(sha256='a'*64, quality_score=quality, pose_yaw=yaw, pose_pitch=0, liveness_score=liveness)], model_scores=[ModelSimilarityScore(model_name='model-a', score=.9)], fused_score=.9, calibrated_match_probability=90, decision='match', threshold_value=.8, false_match_rate=.001, audit_hash='b'*64, config_version='config-7')


def test_narrative_good_quality_discloses_measured_values():
    output = build_narrative(findings())
    assert '1 comparison model(s)' in output['methodology_summary']
    assert 'quality score 0.90' in output['quality_notes']
    assert 'No material quality' in output['quality_notes']


def test_narrative_discloses_low_quality_and_liveness():
    output = build_narrative(findings(quality=.4, liveness=.5))
    assert 'quality is below' in output['quality_notes']
    assert 'liveness is below' in output['quality_notes']


def test_narrative_discloses_high_pose_and_scope_is_immutable():
    output = build_narrative(findings(yaw=20))
    assert 'pose angle exceeds 15 degrees' in output['quality_notes']
    assert output['scope_statement'] == STANDARD_SCOPE_TEXT.replace('[config_version]', 'config-7')
