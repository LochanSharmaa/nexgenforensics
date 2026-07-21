"""
pdf/builder.py

Builds the final forensic PDF report from the validated JSON report +
generated figures. Uses reportlab Platypus. No content here is
hand-typed data — every number/label comes from the report dataclass.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
    Image as RLImage,
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.pdfgen import canvas as rl_canvas

from report.report_model import (
    from_dict, ComparisonReport, ContinuousMeasurement, MorphologicalFeature,
    CandidateRankEntry
)

DISCLAIMER_TEXT = (
    "This report documents algorithmic facial comparison findings. The AI system "
    "provides similarity assessments, not definitive identification. Results must be "
    "interpreted by a qualified examiner in conjunction with all other available "
    "investigative information. Similarity scores reported herein are statistical "
    "measures of resemblance between extracted representations and do not, on their "
    "own, establish or refute personal identity."
)

_FIG_COUNTER = {"n": 0}
_TABLE_COUNTER = {"n": 0}


def _next_fig_num() -> int:
    _FIG_COUNTER["n"] += 1
    return _FIG_COUNTER["n"]


def _next_table_num() -> int:
    _TABLE_COUNTER["n"] += 1
    return _TABLE_COUNTER["n"]


class _NumberedCanvas(rl_canvas.Canvas):
    """Adds page numbers + a running footer with the report ID."""
    def __init__(self, *args, report_id: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self._report_id = report_id
        self._saved_states = []

    def showPage(self):
        self._saved_states.append(dict(self.__dict__))
        super().showPage()

    def save(self):
        page_count = len(self._saved_states)
        for i, state in enumerate(self._saved_states):
            self.__dict__.update(state)
            self._draw_footer(i + 1, page_count)
            super().showPage()
        super().save()

    def _draw_footer(self, page_num, page_count):
        self.setFont("Helvetica", 7)
        self.setFillColor(colors.grey)
        self.drawString(20 * mm, 12 * mm, f"Report ID: {self._report_id}")
        self.drawRightString(190 * mm, 12 * mm, f"Page {page_num} of {page_count}")
        self.drawCentredString(105 * mm, 12 * mm, "CONFIDENTIAL - FORENSIC WORK PRODUCT")


class TOCDocTemplate(SimpleDocTemplate):
    """Custom SimpleDocTemplate to dynamically construct TOC and Outline entries."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.toc = TableOfContents()
        # Custom outline levels styling
        self.toc.levelStyles = [
            ParagraphStyle(name="TOC_L0", fontSize=10, leading=14, fontName="Helvetica-Bold"),
            ParagraphStyle(name="TOC_L1", fontSize=9, leading=12, leftIndent=12, fontName="Helvetica")
        ]

    def afterFlowable(self, flowable):
        if flowable.__class__.__name__ == 'Paragraph':
            text = flowable.getPlainText()
            style = flowable.style.name
            if style == 'H1F':
                # Skip Cover Page or TOC heading from appearing in TOC
                if "Forensic Facial Comparison" in text or "Table of Contents" in text:
                    return
                # Create a target destination from the heading text
                key = f"h1_{hash(text)}"
                self.canv.bookmarkPage(key)
                self.canv.addOutlineEntry(text, key, 0, False)
                self.toc.addEntry(0, text, self.page, key)
            elif style == 'H2F':
                key = f"h2_{hash(text)}"
                self.canv.bookmarkPage(key)
                self.canv.addOutlineEntry(text, key, 1, False)
                self.toc.addEntry(1, text, self.page, key)


def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1F", parent=styles["Heading1"], spaceBefore=14, spaceAfter=8))
    styles.add(ParagraphStyle(name="H2F", parent=styles["Heading2"], spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="BodyF", parent=styles["Normal"], fontSize=9.5, leading=13))
    styles.add(ParagraphStyle(name="Caption", parent=styles["Normal"], fontSize=8, textColor=colors.grey, spaceAfter=10))
    styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], fontSize=22, spaceAfter=6))
    styles.add(ParagraphStyle(name="Disclaimer", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#444444")))
    return styles


def _kv_table(rows: list[tuple[str, str]], styles):
    data = [[Paragraph(f"<b>{k}</b>", styles["BodyF"]), Paragraph(str(v), styles["BodyF"])] for k, v in rows]
    t = Table(data, colWidths=[55 * mm, 105 * mm])
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f2f2f2")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _figure_block(path: str, caption: str, styles, max_width=160 * mm, max_height=90 * mm, fig_id: str = ""):
    n = _next_fig_num()
    img = RLImage(path)
    iw, ih = img.imageWidth, img.imageHeight
    scale = min(max_width / iw, max_height / ih)
    img.drawWidth = iw * scale
    img.drawHeight = ih * scale
    anchor = f'<a name="fig_{n}"/>' if not fig_id else f'<a name="{fig_id}"/>'
    cap = Paragraph(f"{anchor}<b>Figure {n}.</b> {caption}", styles["Caption"])
    return [img, cap, Spacer(1, 6)]


def _measurements_table(measurements: List[ContinuousMeasurement], styles):
    n = _next_table_num()
    header = ["Feature", "Region", "Value", "Unit", "Landmarks"]
    data = [header]
    for m in measurements:
        data.append([m.name, m.region, f"{m.value:.2f}", m.unit, str(m.landmarks_used)])
    t = Table(data, colWidths=[45 * mm, 25 * mm, 20 * mm, 15 * mm, 35 * mm], repeatRows=1)
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    caption = Paragraph(f'<a name="table_{n}"/><b>Table {n}.</b> Continuous anthropometric measurements (probe image).', styles["Caption"])
    return [t, caption, Spacer(1, 8)]


def _morphology_table(features: List[MorphologicalFeature], styles):
    n = _next_table_num()
    header = ["Region", "Probe", "Candidate", "Comparison"]
    color_map = {
        "similar": colors.HexColor("#d4f7d4"),
        "minor_variation": colors.HexColor("#fff3cd"),
        "observed_variation": colors.HexColor("#f8d7da"),
        "not_assessable": colors.HexColor("#e2e2e2"),
    }
    data = [header]
    row_colors = []
    for f in features:
        data.append([f.region, f.probe_observation, f.candidate_observation, f.comparison_label])
        row_colors.append(color_map.get(f.comparison_label, colors.white))

    t = Table(data, colWidths=[35 * mm, 35 * mm, 35 * mm, 35 * mm], repeatRows=1)
    style = [
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]
    for i, c in enumerate(row_colors, start=1):
        style.append(("BACKGROUND", (0, i), (-1, i), c))
    t.setStyle(TableStyle(style))
    caption = Paragraph(f'<a name="table_{n}"/><b>Table {n}.</b> Morphological feature comparison, probe vs. candidate.', styles["Caption"])
    return [t, caption, Spacer(1, 8)]


def build_pdf(report: dict, figure_paths: dict[str, str], output_path: str) -> str:
    # Deserialize report using report_model
    report_obj = from_dict(report)
    styles = _styles()
    
    _FIG_COUNTER["n"] = 0
    _TABLE_COUNTER["n"] = 0

    doc = TOCDocTemplate(
        output_path, pagesize=A4,
        topMargin=20 * mm, bottomMargin=20 * mm, leftMargin=18 * mm, rightMargin=18 * mm,
        title="Forensic Facial Comparison Report",
    )

    story = []

    # ---- Cover page ----
    admin = report_obj.administrative
    story.append(Spacer(1, 40 * mm))
    story.append(Paragraph("Forensic Facial Comparison Report", styles["CoverTitle"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph("AI-Assisted Similarity Analysis &mdash; Investigative Support Document", styles["BodyF"]))
    story.append(Spacer(1, 20))
    story.append(_kv_table([
        ("Report ID", admin.report_id),
        ("Case ID", admin.case_id or "&mdash;"),
        ("Requesting Agency", admin.requesting_agency or "&mdash;"),
        ("Created", admin.created_at),
        ("Software Version", admin.software_version),
    ], styles))
    story.append(Spacer(1, 30))
    story.append(Paragraph(DISCLAIMER_TEXT, styles["Disclaimer"]))
    story.append(PageBreak())

    # ---- Table of Contents Page ----
    story.append(Paragraph("Table of Contents", styles["H1F"]))
    story.append(Spacer(1, 5))
    story.append(doc.toc)
    story.append(PageBreak())

    # ---- Administrative / Evidence ----
    story.append(Paragraph("1. Administrative Information", styles["H1F"]))
    story.append(_kv_table([
        ("Report ID", admin.report_id),
        ("Case ID", admin.case_id or "&mdash;"),
        ("Requesting Agency", admin.requesting_agency or "&mdash;"),
        ("Created At", admin.created_at),
        ("Software Version", admin.software_version),
        ("Schema Version", admin.schema_version),
    ], styles))
    story.append(Spacer(1, 10))

    story.append(Paragraph("2. Evidence Summary", styles["H1F"]))
    ev = report_obj.evidence
    story.append(_kv_table([
        ("Probe Image ID", ev.probe_image_id),
        ("Probe Image Hash", ev.probe_image_hash),
        ("Candidate Image ID", ev.candidate_image_id),
        ("Candidate Image Hash", ev.candidate_image_hash),
        ("Chain of Custody Notes", ev.chain_of_custody_notes or "&mdash;"),
    ], styles))
    story.append(Spacer(1, 6))
    story.append(Paragraph("2.1 Purpose of Examination", styles["H2F"]))
    story.append(Paragraph(
        "Algorithmic comparison of a probe facial image against a candidate facial "
        "image to support investigative review. This document does not constitute "
        "a determination of identity.", styles["BodyF"]))
    story.append(PageBreak())

    # ---- Methodology ----
    story.append(Paragraph("3. Methodology", styles["H1F"]))
    story.append(Paragraph(
        "Processing followed a fixed pipeline: face detection, landmark localization, "
        "image quality assessment, anthropometric measurement, morphological feature "
        "comparison, embedding extraction (unmodified recognition model), and similarity "
        "scoring. Anthropometric measurement definitions follow the soft-biometric "
        "methodology of Tome et al. (2015), &ldquo;Facial Soft Biometric Features for "
        "Forensic Face Recognition.&rdquo; Landmarks were localized using the model specified "
        "in the appendix.<br/><br/>"
        "Clickable references:<br/>"
        "&bull; Go to <a href='#fig_3'>Landmarks Overlay (Figure 3)</a><br/>"
        "&bull; Go to <a href='#fig_4'>Anthropometric Measurements Diagram (Figure 4)</a><br/>"
        "&bull; Go to <a href='#table_1'>Continuous Measurements Table (Table 1)</a><br/>"
        "&bull; Go to <a href='#table_2'>Morphological Variations Table (Table 2)</a>", styles["BodyF"]))
    story.append(Spacer(1, 10))
    story.append(PageBreak())

    # ---- Image quality ----
    story.append(Paragraph("4. Image Quality Assessment", styles["H1F"]))
    qm = report_obj.quality_metrics
    story.append(_kv_table([
        ("Resolution", qm.resolution),
        ("Blur Score (Laplacian variance)", str(qm.blur_score)),
        ("Noise Estimate", str(qm.noise) if qm.noise is not None else "Not computed"),
        ("Compression Artifact Score", str(qm.compression_artifact_score) if qm.compression_artifact_score is not None else "Not computed"),
        ("Brightness", str(qm.brightness)),
        ("Contrast", str(qm.contrast)),
        ("Lighting Uniformity Variance", str(qm.lighting_uniformity) if qm.lighting_uniformity is not None else "Not computed"),
        ("Pose Yaw", f"{qm.pose.yaw}&deg;"),
        ("Pose Pitch", f"{qm.pose.pitch}&deg;"),
        ("Pose Roll", f"{qm.pose.roll}&deg;"),
        ("Occlusion Score", str(qm.occlusion_score) if qm.occlusion_score is not None else "Not computed"),
        ("Face Visibility", f"{qm.face_visibility_pct}%"),
        ("Detection Confidence", str(qm.detection_confidence)),
    ], styles))
    story.extend(_figure_block(figure_paths["fig8_quality_panel"], "Quality visualization panel.", styles, fig_id="fig_8"))
    story.append(Paragraph("See <a href='#fig_8'>Figure 8</a> for regional visual plots of the quality attributes.", styles["BodyF"]))
    story.append(PageBreak())

    # ---- Landmark analysis ----
    story.append(Paragraph("5. Landmark Analysis", styles["H1F"]))
    story.append(Paragraph(f"Landmark schema: {report_obj.landmarks.schema} points. "
                            f"Model: {report_obj.landmarks.model_name or 'N/A'}.", styles["BodyF"]))
    story.extend(_figure_block(figure_paths["fig1_original"], "Original evidence image (probe).", styles, fig_id="fig_1"))
    story.extend(_figure_block(figure_paths["fig2_bounding_box"], "Detected bounding box.", styles, fig_id="fig_2"))
    story.extend(_figure_block(figure_paths["fig3_landmarks"], "Landmark overlay, numbered.", styles, fig_id="fig_3"))
    story.append(PageBreak())

    # ---- Anthropometric measurements ----
    story.append(Paragraph("6. Anthropometric Measurements", styles["H1F"]))
    story.extend(_figure_block(figure_paths["fig4_measurements"], "Anthropometric measurement overlay.", styles, fig_id="fig_4"))
    story.extend(_measurements_table(report_obj.measurements_continuous, styles))
    story.append(PageBreak())

    # ---- Morphological comparison ----
    story.append(Paragraph("7. Morphological Comparison", styles["H1F"]))
    story.extend(_figure_block(figure_paths["fig5_morphological_regions"], "Morphological region segmentation.", styles, fig_id="fig_5"))
    story.extend(_figure_block(figure_paths["fig6_probe_vs_candidate"], "Probe vs. candidate, matched landmark numbering.", styles, fig_id="fig_6"))
    story.extend(_figure_block(figure_paths["fig7_difference"], "Difference visualization by region.", styles, fig_id="fig_7"))
    story.extend(_morphology_table(report_obj.morphological_features, styles))
    story.append(PageBreak())

    # ---- Similarity analysis ----
    story.append(Paragraph("8. Similarity Analysis", styles["H1F"]))
    sim = report_obj.similarity_metrics
    soft_sim = sim.soft_feature_similarity
    story.append(_kv_table([
        ("Cosine similarity", str(sim.cosine_similarity)),
        ("Euclidean distance", str(sim.euclidean_distance) if sim.euclidean_distance is not None else "&mdash;"),
        ("Angular distance (deg)", str(sim.angular_distance) if sim.angular_distance is not None else "&mdash;"),
        ("Threshold used", str(sim.threshold_used) if sim.threshold_used is not None else "&mdash;"),
        ("Decision category", sim.decision_category),
        ("Soft-feature Euclidean (supplementary)", str(soft_sim.euclidean) if soft_sim and soft_sim.euclidean is not None else "&mdash;"),
        ("Soft-feature Mahalanobis (supplementary)", str(soft_sim.mahalanobis) if soft_sim and soft_sim.mahalanobis is not None else "&mdash;"),
        ("Soft-feature Hamming (supplementary)", str(soft_sim.hamming) if soft_sim and soft_sim.hamming is not None else "&mdash;"),
    ], styles))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "<b>Plain-language notes:</b> Cosine similarity measures how closely the two "
        "images' embedding directions align (1.0 = identical direction, 0 = unrelated). "
        "Euclidean and angular distance are alternative geometric measures of the same "
        "underlying comparison. The decision category reflects only whether the score "
        "crossed a configured threshold &mdash; it is not a determination of identity. The "
        "supplementary soft-feature distances are derived independently from this system's "
        "own anthropometric measurements and corroborate, but do not replace, the "
        "primary embedding-based score.", styles["BodyF"]))
    story.append(Spacer(1, 10))

    similar = [f.region for f in report_obj.morphological_features if f.comparison_label == "similar"]
    diff = [f.region for f in report_obj.morphological_features if f.comparison_label == "observed_variation"]
    story.append(Paragraph("8.1 Observed Similarities", styles["H2F"]))
    story.append(Paragraph(
        ", ".join(similar) or "None at the configured category resolution.",
        styles["BodyF"]))
    story.append(Paragraph("8.2 Observed Differences", styles["H2F"]))
    story.append(Paragraph(
        ", ".join(diff) or "None at the configured category resolution.",
        styles["BodyF"]))
    story.append(PageBreak())

    # ---- Limitations / AI findings ----
    story.append(Paragraph("9. Limitations", styles["H1F"]))
    for lim in report_obj.limitations:
        story.append(Paragraph(f"&bull; {lim}", styles["BodyF"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("10. AI Findings Summary", styles["H1F"]))
    story.append(Paragraph(
        f"The system computed a cosine similarity of {sim.cosine_similarity} against a "
        f"configured threshold of {sim.threshold_used}, yielding a decision category of "
        f"&ldquo;{sim.decision_category}&rdquo;. This is a statistical similarity assessment, "
        f"not a determination of identity.", styles["BodyF"]))
    story.append(PageBreak())

    # ---- Examiner section ----
    story.append(Paragraph("11. Examiner Review", styles["H1F"]))
    story.append(_kv_table([
        ("Examiner", "________________________________"),
        ("Signature", "________________________________"),
        ("Date", "________________________________"),
        ("Decision", "&nbsp;&nbsp;&#9744; Confirmed   &nbsp;&nbsp;&#9744; Rejected   &nbsp;&nbsp;&#9744; Further review required"),
        ("Comments", ""),
    ], styles))
    story.append(Spacer(1, 40))
    story.append(Spacer(1, 60))
    story.append(PageBreak())

    # ---- Technical appendix ----
    story.append(Paragraph("12. Technical Appendix", styles["H1F"]))
    tech = report_obj.technical_info
    story.append(_kv_table([
        ("Detection Model", tech.detection_model or "&mdash;"),
        ("Recognition Model", tech.recognition_model or "&mdash;"),
        ("Embedding Model", tech.embedding_model or "&mdash;"),
        ("Similarity Algorithm", tech.similarity_algorithm or "&mdash;"),
        ("Landmark Model", tech.landmark_model or "&mdash;"),
        ("Quality Model", tech.quality_model or "&mdash;"),
        ("Hardware", tech.hardware or "&mdash;"),
        ("Operating System", tech.operating_system or "&mdash;"),
        ("GPU", tech.gpu or "&mdash;"),
        ("Software Version", tech.software_version or "&mdash;"),
        ("Inference Time (ms)", str(tech.inference_time_ms) if tech.inference_time_ms is not None else "&mdash;"),
        ("Timestamp", tech.timestamp or "&mdash;"),
    ], styles))
    story.append(PageBreak())

    # ---- Audit log ----
    story.append(Paragraph("13. Audit Log", styles["H1F"]))
    audit_rows = [("Stage", "Started", "Completed", "Status")]
    for s in report_obj.audit.stages:
        audit_rows.append((s.stage, s.started_at, s.completed_at, s.status))
    t = Table(audit_rows, colWidths=[45 * mm, 45 * mm, 45 * mm, 25 * mm], repeatRows=1)
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
    ]))
    story.append(t)
    story.append(Spacer(1, 16))

    story.append(Paragraph("14. Disclaimer", styles["H1F"]))
    story.append(Paragraph(DISCLAIMER_TEXT, styles["BodyF"]))

    def _canvas_factory(*args, **kwargs):
        return _NumberedCanvas(*args, report_id=admin.report_id, **kwargs)

    # Use multiBuild to build the document while resolving page numbers for TOC
    doc.multiBuild(story, canvasmaker=_canvas_factory)
    return output_path
