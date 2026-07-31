"""
Forensic case report as PDF.

Renders the SAME dict that ReportService.build() produces for JSON and
Markdown, so the three formats cannot drift apart or disagree about a finding.
There is no second data path here — if a number is wrong in the PDF it is
wrong in the JSON too, which is the property you want when a report is
disclosed alongside its machine-readable source.

DELIBERATE CONTENT DECISIONS

Every page carries the investigative-lead notice in the footer, not just the
first page. Reports get printed, split, and photocopied; a caveat that appears
once is a caveat that gets separated from the finding it qualifies.

Similarity scores are printed to 4 decimal places next to the threshold that
judged them, never as a bare "match". A reader must be able to see how close
to the line a decision fell.

Liveness and synthetic-media figures are always accompanied by their heuristic
qualifier. They are image-quality signals, not presentation-attack detection,
and a PDF is exactly where that distinction gets lost.

Enhanced imagery, if ever added, must be labelled per the constraint in the
project brief and shown beside the original — see draw_enhanced_pair().
"""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

INVESTIGATIVE_NOTICE = (
    "Automated face recognition returns investigative leads, not "
    "identifications. A qualified examiner must verify any candidate before it "
    "is relied upon."
)

HEURISTIC_NOTICE = (
    "Liveness and synthetic-media figures are heuristics, not certified "
    "detection. They have not been evaluated against ISO/IEC 30107-3. A pass "
    "means nothing obvious was wrong, not that the media is authentic."
)

_INK = colors.HexColor("#1a1a1a")
_MUTED = colors.HexColor("#6b6b6b")
_RULE = colors.HexColor("#c9c9c9")
_ALERT = colors.HexColor("#8a2b2b")


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "t", parent=base["Title"], fontSize=17, leading=21, textColor=_INK, alignment=TA_LEFT
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontSize=11.5, leading=14,
            spaceBefore=10, spaceAfter=4, textColor=_INK,
        ),
        "body": ParagraphStyle("b", parent=base["BodyText"], fontSize=9, leading=12.5, textColor=_INK),
        "muted": ParagraphStyle("m", parent=base["BodyText"], fontSize=7.8, leading=10.5, textColor=_MUTED),
        "alert": ParagraphStyle("a", parent=base["BodyText"], fontSize=8.2, leading=11.5, textColor=_ALERT),
    }


def _kv_table(rows: list[tuple[str, Any]], widths=(52 * mm, 118 * mm)) -> Table:
    data = [[Paragraph(f"<b>{k}</b>", _styles()["body"]), Paragraph(str(v), _styles()["body"])] for k, v in rows]
    t = Table(data, colWidths=widths, hAlign="LEFT")
    t.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, -2), 0.25, _RULE),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ])
    )
    return t


def _footer(canvas, doc):  # noqa: ANN001
    """Notice + page number on EVERY page. See module docstring."""
    canvas.saveState()
    canvas.setStrokeColor(_RULE)
    canvas.setLineWidth(0.3)
    canvas.line(20 * mm, 16 * mm, 190 * mm, 16 * mm)
    canvas.setFont("Helvetica", 6.4)
    canvas.setFillColor(_MUTED)
    text = canvas.beginText(20 * mm, 12.5 * mm)
    for line in (
        "Investigative lead only - not an identification. A qualified examiner must verify any candidate.",
        "Liveness / synthetic-media figures are heuristics, not certified detection (not ISO/IEC 30107-3 evaluated).",
    ):
        text.textLine(line)
    canvas.drawText(text)
    canvas.setFont("Helvetica", 7)
    canvas.drawRightString(190 * mm, 12.5 * mm, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


def draw_enhanced_pair(*_args, **_kwargs):
    """Placeholder for enhanced-image rendering.

    NOT IMPLEMENTED, and deliberately left as a hard failure rather than a
    silent no-op. The project constraint is explicit: an enhanced image may
    appear only when labelled "AI-enhanced preview - not evidentiary, for
    visual reference only" AND shown beside the unmodified original. A partial
    implementation that rendered an enhanced image without that pairing would
    breach the evidentiary rule, so it raises instead.
    """
    raise NotImplementedError(
        "Enhanced-image rendering is not implemented. It must show the original "
        "alongside, labelled 'AI-enhanced preview - not evidentiary, for visual "
        "reference only'."
    )


def render_case_report_pdf(report: dict[str, Any]) -> bytes:
    """Render a ReportService.build() dict to PDF bytes."""
    st = _styles()
    buf = BytesIO()
    doc = BaseDocTemplate(
        buf, pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=22 * mm,
        title=f"Case report {report.get('case', {}).get('reference', '')}",
        author="NexGen iMATCH",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=_footer)])

    case = report.get("case", {}) or {}
    summary = report.get("summary", {}) or {}
    searches = report.get("searches", []) or []

    story: list[Any] = [
        Paragraph("Forensic Face Comparison Report", st["title"]),
        Spacer(1, 2 * mm),
        Paragraph(
            f"Case {case.get('reference', '-')} &mdash; {case.get('title', '-')}", st["body"]
        ),
        Spacer(1, 4 * mm),
        Paragraph(INVESTIGATIVE_NOTICE, st["alert"]),
        Spacer(1, 4 * mm),
        Paragraph("Case", st["h2"]),
        _kv_table([
            ("Reference", case.get("reference", "-")),
            ("Title", case.get("title", "-")),
            ("Status", case.get("status", "-")),
            ("Lawful basis", case.get("lawful_basis") or "(not recorded)"),
            ("Generated (UTC)", report.get("generated_at") or datetime.now(timezone.utc).isoformat()),
            ("Generated by", report.get("generated_by", "-")),
        ]),
        Paragraph("Summary", st["h2"]),
        _kv_table([
            ("Searches run", summary.get("searches_run", 0)),
            ("Candidates returned", summary.get("candidates_returned", 0)),
            ("Confirmed by examiner", summary.get("confirmed_by_examiner", 0)),
            ("Awaiting adjudication", summary.get("awaiting_adjudication", 0)),
            ("Searches with no result", summary.get("searches_with_no_result", 0)),
        ]),
    ]

    if not searches:
        story += [
            Paragraph("Searches", st["h2"]),
            Paragraph("No searches have been run against this case.", st["muted"]),
        ]

    for i, s in enumerate(searches, 1):
        block: list[Any] = [
            Paragraph(f"Search {i}", st["h2"]),
            _kv_table([
                ("Performed at (UTC)", s.get("performed_at", "-")),
                ("Operator", s.get("operator", "-")),
                ("Lawful basis", s.get("lawful_basis") or "(not recorded)"),
                ("Probe SHA-256", f"<font size=7>{s.get('probe_sha256', '-')}</font>"),
                ("Decision", s.get("decision", "-")),
                ("Top score", s.get("top_score", "-")),
                ("Thresholds", f"match {(s.get('thresholds') or {}).get('match', '-')} / "
                               f"review {(s.get('thresholds') or {}).get('review', '-')}"),
                ("Model", f"{(s.get('model') or {}).get('pack', '-')} "
                          f"({(s.get('model') or {}).get('backend', '-')})"),
                ("Gallery size", s.get("gallery_size", "-")),
                ("Probe quality", s.get("probe_quality", "-")),
                # The qualifier travels with the number, always.
                ("Liveness (heuristic)",
                 f"{s.get('probe_liveness', '-')} &mdash; "
                 f"{s.get('probe_liveness_method', 'heuristic')}, "
                 f"certified: {s.get('probe_liveness_certified', False)}"),
                ("Recognition capable", s.get("recognition_capable", "-")),
                ("Audit hash", f"<font size=7>{s.get('audit_hash', '-')}</font>"),
            ]),
        ]
        if s.get("explanation"):
            block.append(Spacer(1, 1.5 * mm))
            block.append(Paragraph(s["explanation"], st["body"]))
        block.append(Spacer(1, 1.5 * mm))
        block.append(Paragraph(HEURISTIC_NOTICE, st["muted"]))

        cands = s.get("candidates", []) or []
        if cands:
            head = ["Rank", "Subject", "Score", "Adjudication"]
            data = [[Paragraph(f"<b>{h}</b>", st["body"]) for h in head]]
            for c in cands:
                data.append([
                    Paragraph(str(c.get("rank", "-")), st["body"]),
                    Paragraph(str(c.get("subject_name") or c.get("subject_id", "-")), st["body"]),
                    Paragraph(f"{c.get('score', '-')}", st["body"]),
                    Paragraph(str(c.get("adjudication", "pending")), st["body"]),
                ])
            tbl = Table(data, colWidths=(16 * mm, 82 * mm, 28 * mm, 44 * mm), hAlign="LEFT")
            tbl.setStyle(TableStyle([
                ("LINEBELOW", (0, 0), (-1, 0), 0.5, _INK),
                ("LINEBELOW", (0, 1), (-1, -1), 0.25, _RULE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            block += [Spacer(1, 2 * mm), tbl]
        else:
            block += [Spacer(1, 1.5 * mm),
                      Paragraph("No candidates above the reporting floor.", st["muted"])]

        story.append(KeepTogether(block))

    trail = report.get("audit_trail", []) or []
    if trail:
        story.append(Paragraph("Audit trail", st["h2"]))
        data = [[Paragraph(f"<b>{h}</b>", st["body"]) for h in ("Timestamp (UTC)", "Action", "Actor", "Outcome")]]
        for r in trail[:60]:
            data.append([
                Paragraph(f"<font size=7.5>{r.get('timestamp', '-')}</font>", st["body"]),
                Paragraph(str(r.get("action", "-")), st["body"]),
                Paragraph(str(r.get("actor", "-")), st["body"]),
                Paragraph(str(r.get("outcome", "-")), st["body"]),
            ])
        t = Table(data, colWidths=(44 * mm, 40 * mm, 46 * mm, 40 * mm), hAlign="LEFT")
        t.setStyle(TableStyle([
            ("LINEBELOW", (0, 0), (-1, 0), 0.5, _INK),
            ("LINEBELOW", (0, 1), (-1, -1), 0.25, _RULE),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(t)
        if len(trail) > 60:
            story.append(Paragraph(
                f"Showing 60 of {len(trail)} audit entries. The complete chain is "
                f"available via the JSON export and the audit API.", st["muted"]))

    story += [
        Paragraph("Examiner sign-off", st["h2"]),
        Paragraph(
            "This report records an automated comparison. It is not an expert "
            "conclusion until an examiner has reviewed the imagery and signed below.",
            st["body"]),
        Spacer(1, 8 * mm),
        _kv_table([("Examiner name", "&nbsp;"), ("Signature", "&nbsp;"), ("Date", "&nbsp;")]),
    ]

    doc.build(story)
    return buf.getvalue()
