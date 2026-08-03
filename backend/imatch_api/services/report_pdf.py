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

Imagery shown here is the unmodified stored file, scaled to fit the page and
nothing else. Every plate carries the SHA-256 of the bytes it was rendered from,
so a reader can tie the picture on the page to the file the score was computed
against. A missing or unreadable image prints as a stated absence, never as a
blank space — the reader must be able to tell "no image" from "image omitted".

NARRATIVE SECTIONS. Executive Summary, Findings and Similarity Explanation may
be drafted by a language model (services/narrative_service.py) and are printed
under a disclosure that names the model. Methodology, Limitations and Conclusion
come from ReportService templates and are never generated.

This renderer does not call, import, or depend on the narrative layer. If the
narrative is absent — disabled, unreachable, or rejected by its validator — the
report renders every factual section unchanged and prints why the prose is
missing. Report generation must never be blocked on a third-party API.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timezone
from io import BytesIO
from typing import Any

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger(__name__)

#: Resolves a storage-relative path to image bytes. Injected rather than
#: imported so this module stays free of storage and database concerns, and so
#: a caller with no image access can still render a complete report.
ImageLoader = Callable[[str], bytes | None]

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


def _prose(paragraphs: Any, style: ParagraphStyle) -> list[Any]:
    """Render a string or list of strings as spaced paragraphs."""
    if not paragraphs:
        return []
    if isinstance(paragraphs, str):
        paragraphs = [paragraphs]
    out: list[Any] = []
    for text in paragraphs:
        if not str(text).strip():
            continue
        out += [Paragraph(str(text).replace("\n", "<br/>"), style), Spacer(1, 2 * mm)]
    return out


def _image_flowable(raw: bytes, max_width: float, max_height: float) -> Image | None:
    """Decode to a page-sized flowable, or None if the bytes are not an image.

    Rendered at three times the placed size so the plate survives printing
    rather than looking like a screen thumbnail on paper. The image content is
    unaltered: this is a resize for display, stated as such in the caption.
    """
    try:
        with PILImage.open(BytesIO(raw)) as decoded:
            frame = decoded.convert("RGB")
            frame.thumbnail((int(max_width * 3), int(max_height * 3)), PILImage.LANCZOS)
            buffer = BytesIO()
            frame.save(buffer, format="PNG")
            width_px, height_px = frame.size
    except Exception as error:  # noqa: BLE001 - a bad image must not kill the report
        logger.warning("Report image could not be decoded: %s", error)
        return None

    buffer.seek(0)
    scale = min(max_width / width_px, max_height / height_px)
    return Image(buffer, width=width_px * scale, height=height_px * scale)


def _image_cell(
    loader: ImageLoader | None,
    path: str,
    sha256: str,
    heading: str,
    caption: str,
    st: dict[str, ParagraphStyle],
    box: tuple[float, float],
) -> list[Any]:
    """One labelled image, or an explicit statement of why there isn't one."""
    cell: list[Any] = [Paragraph(f"<b>{heading}</b>", st["body"]), Spacer(1, 1 * mm)]

    reason = ""
    if loader is None:
        reason = "Image not available to this export."
    elif not path:
        reason = "No image was retained for this record."
    else:
        raw = loader(path)
        if raw is None:
            # The retention window is the usual explanation, and a reader who is
            # told that will not mistake it for evidence being withheld.
            reason = "Image no longer held. It may have passed the retention window."
        else:
            flowable = _image_flowable(raw, box[0], box[1])
            if flowable is None:
                reason = "Stored file could not be decoded as an image."
            else:
                cell.append(flowable)

    if reason:
        cell.append(Paragraph(reason, st["alert"]))

    cell.append(Spacer(1, 1 * mm))
    cell.append(Paragraph(caption, st["muted"]))
    if sha256:
        cell.append(Paragraph(f"SHA-256 <font size=6.4>{sha256}</font>", st["muted"]))
    return cell


def _comparison_plate(
    loader: ImageLoader | None,
    search: dict[str, Any],
    candidate: dict[str, Any] | None,
    st: dict[str, ParagraphStyle],
) -> Table:
    """Probe beside the enrolment image the score was actually computed against."""
    box = (78 * mm, 78 * mm)
    probe = _image_cell(
        loader,
        search.get("probe_path", ""),
        search.get("probe_sha256", ""),
        "Probe image",
        "Submitted image, unmodified. Scaled to fit the page.",
        st,
        box,
    )
    if candidate is None:
        other = [
            Paragraph("<b>Candidate image</b>", st["body"]),
            Spacer(1, 1 * mm),
            Paragraph("No candidate was returned for this search.", st["muted"]),
        ]
    else:
        other = _image_cell(
            loader,
            candidate.get("template_image_path", ""),
            candidate.get("template_image_sha256", ""),
            f"Candidate rank {candidate.get('rank', '-')} — {candidate.get('subject_name', '-')}",
            "Enrolment image behind the compared template, unmodified. Scaled to fit the page.",
            st,
            box,
        )

    table = Table([[probe, other]], colWidths=(85 * mm, 85 * mm), hAlign="LEFT")
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return table


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


def render_case_report_pdf(
    report: dict[str, Any],
    image_loader: ImageLoader | None = None,
) -> bytes:
    """Render a ReportService.build() dict to PDF bytes.

    ``image_loader`` resolves storage-relative paths to bytes. Omit it and the
    report renders in full with each plate replaced by a stated absence.
    """
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
    standard = report.get("standard_sections", {}) or {}
    narrative = report.get("narrative", {}) or {}
    drafted = narrative.get("sections", {}) or {}

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

    # The disclosure sits above the first drafted paragraph, not in a footnote.
    # A reader must know a section was model-drafted before they read it, not
    # after. When no narrative is present, the reason is printed instead --
    # silence would read as though the section had simply been forgotten.
    if narrative.get("available") and narrative.get("disclosure"):
        story += [Spacer(1, 3 * mm), Paragraph(narrative["disclosure"], st["muted"])]
    elif narrative.get("withheld_notice"):
        story += [Spacer(1, 3 * mm), Paragraph(narrative["withheld_notice"], st["muted"])]

    if drafted.get("executive_summary"):
        story.append(Paragraph("Executive summary", st["h2"]))
        story += _prose(drafted["executive_summary"], st["body"])

    if standard.get("methodology"):
        story.append(Paragraph("Methodology", st["h2"]))
        story += _prose(standard["methodology"], st["body"])

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
                # Resolved by ReportService, which owns the mapping. Two request
                # vocabularies feed SearchRun.mode and they overlap misleadingly
                # -- deciding it here would put a second, driftable copy of that
                # mapping in the renderer.
                ("Comparison type", s.get("mode_label") or s.get("mode") or "-"),
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
            head = ["Rank", "Subject", "Similarity", "Normalised", "Adjudication", "Examiner"]
            data = [[Paragraph(f"<b>{h}</b>", st["body"]) for h in head]]
            for c in cands:
                # Read "similarity", the key ReportService actually emits. This
                # was previously "score", a key that has never existed in the
                # report dict, so every candidate similarity in every exported
                # PDF printed as "-" while the JSON and Markdown carried the
                # real figure.
                similarity = c.get("similarity")
                normalised = c.get("normalized_score")
                data.append([
                    Paragraph(str(c.get("rank", "-")), st["body"]),
                    Paragraph(str(c.get("subject_name") or c.get("subject_id", "-")), st["body"]),
                    Paragraph(f"{similarity:.4f}" if isinstance(similarity, (int, float)) else "-", st["body"]),
                    Paragraph(f"{normalised:.4f}" if isinstance(normalised, (int, float)) else "-", st["body"]),
                    Paragraph(str(c.get("adjudication", "pending")), st["body"]),
                    Paragraph(str(c.get("adjudicated_by") or "-"), st["body"]),
                ])
            tbl = Table(
                data,
                colWidths=(12 * mm, 48 * mm, 22 * mm, 22 * mm, 30 * mm, 36 * mm),
                hAlign="LEFT",
            )
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

        # Image plates follow the block rather than joining it: a KeepTogether
        # holding two plates plus a table cannot fit on one page and reportlab
        # would push the whole search onto a fresh one, leaving half a page
        # blank on every search in the report.
        #
        # Which candidates get a plate: rank 1, plus every candidate an examiner
        # has adjudicated. A candidate someone made a decision about is one a
        # reader will want to see. The cap is stated below rather than applied
        # silently, because a truncated exhibit list that does not say so reads
        # as a complete one.
        plated = [c for c in cands if c.get("rank") == 1 or c.get("adjudicated_at")]
        story.append(Spacer(1, 3 * mm))
        if not plated:
            story.append(_comparison_plate(image_loader, s, None, st))
        for c in plated:
            story.append(KeepTogether([
                Paragraph(
                    f"Search {i} — probe and candidate rank {c.get('rank', '-')}", st["body"]
                ),
                Spacer(1, 2 * mm),
                _comparison_plate(image_loader, s, c, st),
            ]))
            story.append(Spacer(1, 3 * mm))

        if len(plated) < len(cands):
            story.append(Paragraph(
                f"Images are shown for {len(plated)} of {len(cands)} candidate(s): rank 1 and "
                f"any candidate an examiner has adjudicated. The remaining candidates appear "
                f"in the table above with their scores, and their imagery is available in the "
                f"case record.", st["muted"]))

    if drafted.get("findings"):
        story.append(Paragraph("Findings", st["h2"]))
        story += _prose(drafted["findings"], st["body"])

    if drafted.get("similarity_explanation"):
        story.append(Paragraph("Similarity explanation", st["h2"]))
        story += _prose(drafted["similarity_explanation"], st["body"])

    # Limitations are rendered in the alert style and as a complete list. They
    # are assembled by ReportService from fixed text plus conditionals on the
    # record, so nothing on this page can be dropped or softened here.
    if standard.get("limitations"):
        story.append(Paragraph("Limitations", st["h2"]))
        for item in standard["limitations"]:
            story += [Paragraph(f"&bull; {item}", st["alert"]), Spacer(1, 1.5 * mm)]

    if standard.get("conclusion"):
        story.append(Paragraph("Conclusion", st["h2"]))
        story += _prose(standard["conclusion"], st["body"])

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
