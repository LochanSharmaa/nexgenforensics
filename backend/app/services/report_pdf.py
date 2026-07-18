from __future__ import annotations
import html
from pathlib import Path
from app.schemas import ForensicFindings
from .report_narrative import build_narrative, STANDARD_SCOPE_TEXT
def render_report_pdf(findings: ForensicFindings, output_path: Path) -> Path:
    narrative=build_narrative(findings)
    assert narrative['scope_statement'] and findings.audit_hash
    rows=''.join(f'<tr><td>{html.escape(m.name)}</td><td>{m.value:.4f}</td><td>{m.confidence:.2f}</td></tr>' for m in findings.measurements)
    scores=''.join(f'<tr><td>{html.escape(s.model_name)}</td><td>{s.score:.4f}</td></tr>' for s in findings.model_scores)
    images=''.join(f'<li>SHA-256: {html.escape(i.sha256)} | Quality: {i.quality_score:.2f} | Pose: {i.pose_yaw:.1f}/{i.pose_pitch:.1f} | Liveness: {i.liveness_score:.2f}</li>' for i in findings.source_images)
    document=f'''<!doctype html><html><head><style>@page{{size:A4;margin:18mm}}body{{font-family:Arial;color:#17202a;font-size:10pt}}h1{{color:#13263d}}h2{{color:#204667;border-bottom:1px solid #9aabb8;padding-bottom:3px}}.header{{background:#13263d;color:white;padding:16px}}table{{width:100%;border-collapse:collapse}}td,th{{border:1px solid #aab7c4;padding:5px;text-align:left}}th{{background:#dce6ec}}.decision{{font-size:16pt;font-weight:bold}}</style></head><body><section class="header"><h1>Automated Facial Comparison Report</h1><p>Case: {html.escape(findings.case_id)} | Examiner: {html.escape(findings.examiner_id)} | Tenant: {html.escape(findings.tenant_id)}</p><p>Audit hash: {findings.audit_hash}</p></section><h2>Source Images and Landmark Overlay Data</h2><ul>{images}</ul><h2>Landmark-based Geometric Measurements</h2><table><tr><th>Measurement</th><th>Value</th><th>Confidence</th></tr>{rows}</table><h2>Per-model Similarity Scores</h2><table><tr><th>Model</th><th>Score</th></tr>{scores}</table><h2>Fused Score and Decision</h2><p class="decision">{findings.decision.upper()} — {findings.fused_score:.4f} ({findings.calibrated_match_probability:.1f}%)</p><p>Threshold: {findings.threshold_value:.3f}; False-match rate: {findings.false_match_rate:.6f}</p><h2>Methodology</h2><p>{html.escape(narrative['methodology_summary'])}</p><h2>Quality and Limiting Factors</h2><p>{html.escape(narrative['quality_notes'])}</p><h2>Scope Statement</h2><p>{html.escape(narrative['scope_statement'])}</p><footer>NexGen Forensics | Audit hash {findings.audit_hash}</footer></body></html>'''
    output_path.parent.mkdir(parents=True,exist_ok=True)
    try:
        from weasyprint import HTML
        HTML(string=document).write_pdf(str(output_path))
    except ImportError as exc:
        raise RuntimeError('weasyprint is required to render forensic PDFs') from exc
    return output_path

def tenant_report_path(root: Path, tenant_id: str, case_id: str, audit_hash: str) -> Path:
    safe=lambda value: ''.join(c if c.isalnum() or c in '-_' else '_' for c in value)
    return root / safe(tenant_id) / f'{safe(case_id)}-{audit_hash[:12]}.pdf'
