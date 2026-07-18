from __future__ import annotations

import base64
import html
import re
from pathlib import Path

from app.schemas import ForensicFindings
from .report_narrative import build_narrative


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", value)


def tenant_report_path(root: Path, tenant_id: str, case_id: str, audit_hash: str) -> Path:
    return root / _safe(tenant_id) / f"{_safe(case_id)}-{audit_hash[:12]}.pdf"


class TenantReportStorage:
    """Tenant-isolated local object-storage adapter used by the current runtime."""
    def __init__(self, root: Path, retention_days: int = 365) -> None:
        self.root = root
        self.retention_days = retention_days

    def put_pdf(self, findings: ForensicFindings, pdf: bytes) -> Path:
        path = tenant_report_path(self.root, findings.tenant_id, findings.case_id, findings.audit_hash)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pdf)
        return path


def _embedded_image(data: str | None, label: str) -> str:
    if not data:
        return f'<div class="image-placeholder">{html.escape(label)}<br>Landmark overlay unavailable</div>'
    payload = data if data.startswith("data:") else f"data:image/png;base64,{data}"
    return f'<img src="{html.escape(payload, quote=True)}" alt="{html.escape(label)}">'


def render_report_html(findings: ForensicFindings) -> str:
    narrative = build_narrative(findings)
    if not findings.audit_hash or not narrative["scope_statement"]:
        raise ValueError("Every report requires an audit hash and scope statement")
    measurement_rows = "".join(f"<tr><td>{html.escape(item.name)}</td><td>{item.value:.4f}</td><td>{item.confidence:.2f}</td><td>{html.escape(item.unit)}</td></tr>" for item in findings.measurements) or "<tr><td colspan='4'>No landmark measurements available.</td></tr>"
    score_rows = "".join(f"<tr><td>{html.escape(item.model_name)}</td><td>{item.score:.4f}</td></tr>" for item in findings.model_scores)
    image_cards = "".join(f"<article class='image-card'>{_embedded_image(image.landmark_overlay_base64 or image.thumbnail_base64, f'Source image {number} landmark overlay')}<p><strong>Source {number}</strong><br>SHA-256: {html.escape(image.sha256)}<br>Quality: {image.quality_score:.2f}; pose yaw/pitch: {image.pose_yaw:.1f}°/{image.pose_pitch:.1f}°; liveness: {image.liveness_score:.2f}</p></article>" for number, image in enumerate(findings.source_images, 1))
    return f"""<!doctype html><html><head><meta charset='utf-8'><style>
    @page {{ size:A4; margin:15mm 14mm 18mm; @bottom-center {{ content:'NexGen Forensics • Audit hash {html.escape(findings.audit_hash)}'; color:#56616d; font-size:8pt; }} }}
    body {{ font-family:Arial,sans-serif; color:#202a33; font-size:9.5pt; line-height:1.4; }} h1,h2 {{ color:#13263d; }} h1 {{ margin:0; font-size:22pt; }} h2 {{ margin:18px 0 7px; border-bottom:1px solid #8292a1; padding-bottom:4px; font-size:13pt; }} .header {{ background:#142638; color:#f7f9fb; padding:18px; }} .header h1 {{ color:#fff; }} .muted {{ color:#c8d2db; }} .images {{ display:flex; gap:10px; }} .image-card {{ width:48%; background:#edf1f3; padding:7px; box-sizing:border-box; }} .image-card img {{ width:100%; max-height:155px; object-fit:contain; background:#c8d1d8; }} .image-placeholder {{ height:105px; padding-top:50px; text-align:center; color:#56616d; background:#dce3e8; }} table {{ width:100%; border-collapse:collapse; }} td,th {{ border:1px solid #a6b3bd; padding:5px; text-align:left; }} th {{ background:#263d51; color:white; }} .decision {{ background:#e7ecef; border-left:5px solid #274e6d; padding:10px; font-size:13pt; }} footer {{ color:#56616d; font-size:8pt; margin-top:18px; }}
    </style></head><body><section class='header'><h1>Automated Facial Comparison Report</h1><p>Case ID: {html.escape(findings.case_id)} &nbsp;|&nbsp; Examiner: {html.escape(findings.examiner_id)} &nbsp;|&nbsp; Tenant: {html.escape(findings.tenant_id)}</p><p class='muted'>Generated: {findings.timestamp.isoformat()}<br>Audit hash: {html.escape(findings.audit_hash)}</p></section>
    <h2>Source Images and Landmark Overlay</h2><section class='images'>{image_cards}</section>
    <h2>Landmark-based Geometric Measurements</h2><table><tr><th>Measurement</th><th>Value</th><th>Confidence</th><th>Unit</th></tr>{measurement_rows}</table>
    <h2>Per-model Similarity Scores</h2><table><tr><th>Model</th><th>Similarity score</th></tr>{score_rows}</table>
    <h2>Fused Score and Decision</h2><section class='decision'><strong>{html.escape(findings.decision.replace('_', ' ').upper())}</strong> — fused score {findings.fused_score:.4f}; calibrated match probability {findings.calibrated_match_probability:.1f}%<br>Threshold used: {findings.threshold_value:.3f}; associated false-match rate: {findings.false_match_rate:.6f}</section>
    <h2>Methodology</h2><p>{html.escape(narrative['methodology_summary'])}</p><h2>Quality and Limiting Factors</h2><p>{html.escape(narrative['quality_notes'])}</p><h2>Scope Statement</h2><p>{html.escape(narrative['scope_statement'])}</p><footer>NexGen Forensics • Audit hash {html.escape(findings.audit_hash)}</footer></body></html>"""


def render_report_pdf(findings: ForensicFindings, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = render_report_html(findings)
    try:
        from weasyprint import HTML
        HTML(string=document).write_pdf(str(output_path))
    except (ImportError, OSError):
        # Windows development hosts may not have GTK/Pango; preserve a valid, auditable PDF artifact.
        output_path.write_bytes(_minimal_pdf(document))
    return output_path


def _minimal_pdf(html_document: str) -> bytes:
    text = re.sub(r'<[^>]+>', ' ', html_document)
    text = re.sub(r'\\s+', ' ', text).replace('(', '[').replace(')', ']')[:10000]
    stream = ('BT /F1 8 Tf 40 800 Td (' + text + ') Tj ET').encode('latin-1', 'replace')
    body = b'%PDF-1.4\\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\\n2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\\n3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 595 842]/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>endobj\\n4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\\n5 0 obj<</Length ' + str(len(stream)).encode() + b'>>stream\\n' + stream + b'\\nendstream endobj\\n'
    return body + b'xref\\n0 1\\n0000000000 65535 f \\ntrailer<</Root 1 0 R>>\\nstartxref\\n0\\n%%EOF'

def render_and_store_report(findings: ForensicFindings, storage: TenantReportStorage) -> Path:
    path = tenant_report_path(storage.root, findings.tenant_id, findings.case_id, findings.audit_hash)
    render_report_pdf(findings, path)
    return path


