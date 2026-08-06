"""The photograph examination report, as PDF.

Renders the dict ``ExaminationService.build()`` produces, in the layout of the
examination report form this replaces: covering letter, description of
questioned and specimen exhibits with the photographs themselves, observation
with the micrometry table, the annotated measurement plates, the secondary
identification findings with their comparison grids, and the result.

EVERY STRING FROM THE RECORD IS ESCAPED. reportlab's ``Paragraph`` parses a
mini-markup language, so a filename containing ``<`` or ``&`` -- which a
submitter chooses, not us -- would either break the render or inject markup
into the document. ``_t()`` is applied to all of it. Literal markup in this
module is written directly and is the only markup that reaches the page.

WHAT THIS RENDERER WILL NOT PRINT

An examiner's finding that no examiner made. Where the record holds no
question, no observation and no result, the section prints "Not recorded"
against a note of what is missing. A blank looks like an oversight and
generated prose would be fabricated expert evidence; a stated absence is
neither.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from io import BytesIO
from typing import Any
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger(__name__)

_INK = colors.HexColor("#111111")
_MUTED = colors.HexColor("#6b6b6b")
_RULE = colors.HexColor("#b9b9b9")
_ALERT = colors.HexColor("#8a2b2b")
_HEAD_BG = colors.HexColor("#ececec")

_PAGE_WIDTH = 170 * mm  #: A4 less the 20 mm margins either side


def _t(value: Any) -> str:
    """Escape anything that came from the record before it meets the parser."""
    return escape("" if value is None else str(value))


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "t", parent=base["Title"], fontName="Times-Bold", fontSize=13.5,
            leading=17, textColor=_INK, alignment=TA_CENTER, spaceAfter=2,
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontName="Times-Bold", fontSize=11.5,
            leading=14, spaceBefore=11, spaceAfter=5, textColor=_INK,
        ),
        "h3": ParagraphStyle(
            "h3", parent=base["Heading3"], fontName="Times-Bold", fontSize=10.5,
            leading=13, spaceBefore=8, spaceAfter=4, textColor=_INK,
        ),
        # Times at 10.5/15 justified: the report form this replaces is a letter,
        # and a letter set in a sans face at 9pt does not read as one.
        "body": ParagraphStyle(
            "b", parent=base["BodyText"], fontName="Times-Roman", fontSize=10.5,
            leading=15, textColor=_INK, alignment=TA_JUSTIFY,
        ),
        "body_left": ParagraphStyle(
            "bl", parent=base["BodyText"], fontName="Times-Roman", fontSize=10.5,
            leading=15, textColor=_INK, alignment=TA_LEFT,
        ),
        "cell": ParagraphStyle(
            "c", parent=base["BodyText"], fontName="Times-Roman", fontSize=8.2,
            leading=10.5, textColor=_INK, alignment=TA_CENTER,
        ),
        "cell_left": ParagraphStyle(
            "cl", parent=base["BodyText"], fontName="Times-Roman", fontSize=8.2,
            leading=10.5, textColor=_INK, alignment=TA_LEFT,
        ),
        "muted": ParagraphStyle(
            "m", parent=base["BodyText"], fontName="Times-Roman", fontSize=8,
            leading=11, textColor=_MUTED, alignment=TA_JUSTIFY,
        ),
        "alert": ParagraphStyle(
            "a", parent=base["BodyText"], fontName="Times-Roman", fontSize=9,
            leading=12.5, textColor=_ALERT, alignment=TA_JUSTIFY,
        ),
        "mark": ParagraphStyle(
            "mk", parent=base["BodyText"], fontName="Times-Bold", fontSize=11,
            leading=14, textColor=_INK, alignment=TA_CENTER,
        ),
    }


def _footer(reference: str):
    def draw(canvas, doc):  # noqa: ANN001
        canvas.saveState()
        canvas.setStrokeColor(_RULE)
        canvas.setLineWidth(0.3)
        canvas.line(20 * mm, 15 * mm, 190 * mm, 15 * mm)
        canvas.setFont("Times-Roman", 7.5)
        canvas.setFillColor(_MUTED)
        canvas.drawString(20 * mm, 11 * mm, f"Photograph examination report — case {reference}")
        canvas.drawRightString(190 * mm, 11 * mm, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()

    return draw


def _image(raw: bytes, max_width: float, max_height: float) -> Image | None:
    """Decode to a page-sized flowable, or None if the bytes are not an image.

    Rasterised at twice the placed size so a plate survives printing rather
    than looking like a screen thumbnail on paper.
    """
    try:
        with PILImage.open(BytesIO(raw)) as decoded:
            frame = decoded.convert("RGB")
            frame.thumbnail((int(max_width * 2), int(max_height * 2)), PILImage.LANCZOS)
            buffer = BytesIO()
            frame.save(buffer, format="PNG")
            width, height = frame.size
    except Exception as error:  # noqa: BLE001 - a bad image must not kill the report
        logger.warning("Examination image could not be decoded: %s", error)
        return None

    buffer.seek(0)
    scale = min(max_width / width, max_height / height)
    return Image(buffer, width=width * scale, height=height * scale)


def _pil_flowable(image: PILImage.Image, max_width: float, max_height: float) -> Image | None:
    buffer = BytesIO()
    frame = image.convert("RGB")
    frame.thumbnail((int(max_width * 2), int(max_height * 2)), PILImage.LANCZOS)
    frame.save(buffer, format="PNG")
    buffer.seek(0)
    width, height = frame.size
    scale = min(max_width / width, max_height / height)
    return Image(buffer, width=width * scale, height=height * scale)


def _prose(text: Any, style: ParagraphStyle, gap: float = 2.4) -> list[Any]:
    """A string or list of strings as spaced, escaped paragraphs."""
    if not text:
        return []
    blocks = [text] if isinstance(text, str) else list(text)
    out: list[Any] = []
    for block in blocks:
        for part in str(block).split("\n\n"):
            if not part.strip():
                continue
            out += [Paragraph(_t(part).replace("\n", "<br/>"), style), Spacer(1, gap * mm)]
    return out


def _not_recorded(what: str, st: dict[str, ParagraphStyle]) -> list[Any]:
    return [
        Paragraph(
            f"<i>Not recorded.</i> {_t(what)} This section is printed as absent rather "
            "than composed, because it is the examiner's to write.",
            st["muted"],
        ),
        Spacer(1, 2.4 * mm),
    ]


# ------------------------------------------------------------- sections ---


def _exhibit_block(
    photo: dict[str, Any],
    image: PILImage.Image | None,
    index: int,
    st: dict[str, ParagraphStyle],
) -> list[Any]:
    """One numbered exhibit: its description, then the photograph itself."""
    marks = photo.get("marks") or []
    if marks:
        marked = "marked as " + " and ".join(f"&ldquo;{_t(m)}&rdquo;" for m in marks)
    else:
        marked = "not marked (no face was detected in it)"

    description = (
        f"{index}. The photograph sample named as &ldquo;{_t(photo['display_name'])}&rdquo; "
        f"having size &ldquo;{_t(photo['size_text'])}&rdquo;, size on disk "
        f"&ldquo;{_t(photo['size_on_disk_text'])}&rdquo;, {marked}."
    )
    block: list[Any] = [
        Paragraph(description, st["body"]),
        Spacer(1, 1.5 * mm),
        Paragraph(f"SHA-256 {_t(photo['sha256'])}", st["muted"]),
        Spacer(1, 2 * mm),
    ]

    if photo.get("subject_name"):
        block.append(
            Paragraph(f"Enrolled against subject: {_t(photo['subject_name'])}", st["muted"])
        )
        block.append(Spacer(1, 1.5 * mm))

    if image is not None:
        flowable = _pil_flowable(image, 108 * mm, 92 * mm)
        if flowable is not None:
            block += [flowable, Spacer(1, 2 * mm)]
    if photo.get("missing_reason"):
        block.append(Paragraph(_t(photo["missing_reason"]), st["alert"]))
        block.append(Spacer(1, 2 * mm))

    block.append(Spacer(1, 2 * mm))
    return block


def _micrometry_table(table: dict[str, Any], st: dict[str, ParagraphStyle]) -> Table:
    """The measurement table, laid out as the paper form has it.

    Two header rows -- the filename above the exhibit mark -- because a reader
    needs to tie a column both to the file it came from and to the mark used
    everywhere else in the report.
    """
    header = table["header"]
    columns = len(header)

    label_width = 58 * mm
    value_width = (_PAGE_WIDTH - label_width) / max(1, columns)

    data: list[list[Any]] = [
        [Paragraph("<b>Parameter</b>", st["cell_left"])]
        + [Paragraph(f"<b>{_t(item['filename'])}</b>", st["cell"]) for item in header],
        [Paragraph("", st["cell_left"])]
        + [Paragraph(f"<b>{_t(item['mark'])}</b>", st["cell"]) for item in header],
    ]

    group_rows: list[int] = []
    seen_group: set[str] = set()
    for row in table["rows"]:
        group = row.get("group")
        if group and group not in seen_group:
            seen_group.add(group)
            group_rows.append(len(data))
            data.append(
                [Paragraph(f"<b>{_t(group)}</b>", st["cell_left"])]
                + [Paragraph("", st["cell"]) for _ in header]
            )
        data.append(
            [Paragraph(_t(row["label"]), st["cell_left"])]
            + [Paragraph(_t(value), st["cell"]) for value in row["values"]]
        )

    flowable = Table(
        data,
        colWidths=[label_width] + [value_width] * columns,
        hAlign="LEFT",
        repeatRows=2,
    )
    style = [
        ("GRID", (0, 0), (-1, -1), 0.4, _RULE),
        ("BACKGROUND", (0, 0), (-1, 1), _HEAD_BG),
        ("LINEBELOW", (0, 1), (-1, 1), 0.8, _INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    for row_index in group_rows:
        style.append(("BACKGROUND", (0, row_index), (-1, row_index), _HEAD_BG))
        style.append(("SPAN", (0, row_index), (-1, row_index)))
    flowable.setStyle(TableStyle(style))
    return flowable


def _comparison_grid(
    grid: list[dict[str, Any]],
    crops: dict[str, bytes],
    st: dict[str, ParagraphStyle],
) -> Table | None:
    """The upper-face grid under a secondary-identification finding."""
    cells: list[list[Any]] = []
    row_images: list[Any] = []
    row_labels: list[Any] = []
    per_row = 4
    box_width = min(38 * mm, _PAGE_WIDTH / per_row - 4 * mm)

    for entry in grid:
        raw = crops.get(entry["mark"])
        cell = _image(raw, box_width, 46 * mm) if raw else None
        row_images.append(cell if cell is not None else Paragraph("(no crop)", st["muted"]))
        row_labels.append(Paragraph(f"<b>{_t(entry['mark'])}</b>", st["mark"]))
        if len(row_images) == per_row:
            cells += [row_images, row_labels]
            row_images, row_labels = [], []

    if row_images:
        while len(row_images) < per_row:
            row_images.append("")
            row_labels.append("")
        cells += [row_images, row_labels]

    if not cells:
        return None

    table = Table(cells, colWidths=[_PAGE_WIDTH / per_row] * per_row, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


# --------------------------------------------------------------- render ---


def render_examination_pdf(report: dict[str, Any]) -> bytes:
    """Render an ``ExaminationService.build()`` dict to PDF bytes."""
    st = _styles()
    case = report.get("case", {}) or {}
    letter = report.get("letter", {}) or {}
    exhibits = report.get("exhibits", {}) or {}
    observation = report.get("observation", {}) or {}
    photographs = report.get("_photographs", []) or []

    by_sha = {photo.sha256: photo for photo in photographs}
    plates: dict[str, bytes] = {}
    crops: dict[str, bytes] = {}
    for photo in photographs:
        for face in photo.faces:
            if face.plate is not None:
                plates[face.mark] = face.plate.png
            if face.upper_face is not None:
                crops[face.mark] = face.upper_face.png

    buffer = BytesIO()
    doc = BaseDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=20 * mm,
        title=f"Photograph examination report {case.get('reference', '')}",
        author="NexGen iMATCH",
    )
    doc.addPageTemplates([
        PageTemplate(
            id="main",
            frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")],
            onPage=_footer(case.get("reference", "-")),
        )
    ])

    story: list[Any] = [
        Paragraph(f"Sub: {_t(letter.get('subject', 'Photograph Examination Report'))}", st["title"]),
        Spacer(1, 1.5 * mm),
        Paragraph(
            f"Case {_t(case.get('reference', '-'))} &mdash; {_t(case.get('title', '-'))}",
            st["mark"],
        ),
        Spacer(1, 5 * mm),
        Paragraph(_t(letter.get("salutation", "Sir,")), st["body_left"]),
        Spacer(1, 2 * mm),
    ]
    story += _prose(letter.get("covering"), st["body"])

    if letter.get("receipt"):
        story += _prose(letter["receipt"], st["body"])
    else:
        story += _not_recorded(
            "How and from whom the material was received has not been entered.", st
        )

    story.append(Paragraph("Question put for examination", st["h3"]))
    if letter.get("question"):
        story += _prose(letter["question"], st["body"])
    else:
        story += _not_recorded("No question has been entered against this case.", st)

    story.append(Paragraph("Purpose of examination", st["h3"]))
    if letter.get("purpose"):
        story += _prose(letter["purpose"], st["body"])
    else:
        story += _not_recorded("No purpose has been entered against this case.", st)

    story.append(Paragraph(_t(report.get("notice", "")), st["alert"]))
    story.append(Spacer(1, 3 * mm))

    # -- description of exhibits ------------------------------------------
    for role, heading, lead in (
        ("questioned", "Description of exhibits — questioned photographs",
         "The names of the questioned photographs with their details are as follows:"),
        ("specimen", "Description of exhibits — specimen photographs",
         "The names of the specimen photographs with their details are as follows:"),
    ):
        entries = exhibits.get(role, []) or []
        story.append(Paragraph(heading, st["h2"]))
        if not entries:
            story.append(Paragraph(
                f"No {role} photographs are held against this case.", st["muted"]
            ))
            story.append(Spacer(1, 2 * mm))
            continue
        story.append(Paragraph(lead, st["body"]))
        story.append(Spacer(1, 2.5 * mm))
        for index, photo in enumerate(entries, 1):
            source = by_sha.get(photo["sha256"])
            story.append(KeepTogether(
                _exhibit_block(photo, source.image if source else None, index, st)
            ))

    # -- observation -------------------------------------------------------
    all_marks = [
        mark
        for role in ("questioned", "specimen")
        for photo in (exhibits.get(role, []) or [])
        for mark in (photo.get("marks") or [])
    ]
    story.append(PageBreak())
    story.append(Paragraph("Observation", st["h2"]))
    if all_marks:
        story.append(Paragraph(
            "The following features of the photograph samples marked as "
            f"&ldquo;{_t(all_marks[0])}&rdquo; to &ldquo;{_t(all_marks[-1])}&rdquo; "
            "were examined, and the respective characteristics are as follows:",
            st["body"],
        ))
    else:
        story.append(Paragraph(
            "No exhibit in this case yielded a face that could be examined.", st["alert"]
        ))
    story.append(Spacer(1, 2.5 * mm))

    tables = observation.get("tables", []) or []
    if tables:
        story.append(Paragraph("Primary micrometry and morphology features", st["h3"]))
        for table in tables:
            story.append(Spacer(1, 2 * mm))
            story.append(_micrometry_table(table, st))
            story.append(Spacer(1, 1.5 * mm))
            story.append(Paragraph(_t(table["caption"]), st["muted"]))
            story.append(Spacer(1, 2.5 * mm))

        if observation.get("normalisation"):
            story.append(Paragraph(_t(observation["normalisation"]), st["muted"]))
            story.append(Spacer(1, 2 * mm))
        if observation.get("caveat"):
            story.append(Paragraph(_t(observation["caveat"]), st["alert"]))
            story.append(Spacer(1, 3 * mm))

        legend = observation.get("legend", []) or []
        if legend:
            story.append(Paragraph(
                "Key to the measurement plates: "
                + "; ".join(f"<b>{_t(item['key'])}</b> {_t(item['label'])}" for item in legend)
                + ".",
                st["muted"],
            ))
            story.append(Spacer(1, 2 * mm))

    # -- measurement plates ------------------------------------------------
    if plates:
        story.append(PageBreak())
        story.append(Paragraph("Measurement plates", st["h2"]))
        story.append(Paragraph(
            "Each plate shows the measured distances drawn between the landmark points "
            "that produced them. The unmodified photograph each plate derives from "
            "appears in the description of exhibits above.",
            st["body"],
        ))
        story.append(Spacer(1, 3 * mm))
        for mark, png in plates.items():
            flowable = _image(png, 118 * mm, 150 * mm)
            if flowable is None:
                continue
            story.append(KeepTogether([
                Paragraph(f"Exhibit {_t(mark)}", st["mark"]),
                Spacer(1, 1.5 * mm),
                flowable,
                Spacer(1, 4 * mm),
            ]))

    # -- secondary identification -----------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("Secondary identification", st["h2"]))
    findings = report.get("secondary", []) or []
    if not findings:
        story += _not_recorded(
            "No secondary-identification observation has been recorded for this case. "
            "Morphological comparison is the examiner's method and its findings are "
            "entered by the examiner.",
            st,
        )
    for position, finding in enumerate(findings, 1):
        block: list[Any] = [
            Paragraph(f"{position}. {_t(finding['body'])}", st["body"]),
            Spacer(1, 2 * mm),
        ]
        if finding.get("unknown_marks"):
            block.append(Paragraph(
                "This observation cites exhibit(s) "
                + ", ".join(_t(m) for m in finding["unknown_marks"])
                + ", which are not in the inventory above.",
                st["alert"],
            ))
            block.append(Spacer(1, 1.5 * mm))
        grid = _comparison_grid(finding.get("grid", []) or [], crops, st)
        if grid is not None:
            block += [grid, Spacer(1, 4 * mm)]
        story.append(KeepTogether(block))

    # -- result ------------------------------------------------------------
    story.append(Paragraph("Result of examination", st["h2"]))
    result = report.get("result", []) or []
    if result:
        story += _prose(result, st["body"])
    else:
        story += _not_recorded(
            "No result has been recorded. A result of examination is an expert "
            "conclusion and is entered and signed by the examiner.",
            st,
        )

    # -- limitations, suppressed sections, sign-off ------------------------
    limitations = report.get("limitations", []) or []
    if limitations:
        story.append(Paragraph("Limitations", st["h2"]))
        for item in limitations:
            story += [Paragraph(f"&bull; {_t(item)}", st["alert"]), Spacer(1, 1.5 * mm)]

    unavailable = report.get("unavailable", []) or []
    if unavailable:
        story.append(Paragraph("Sections not produced", st["h2"]))
        story.append(Paragraph(
            "The following could not be produced from the material held. They are "
            "listed rather than omitted, so that an absence in this report is "
            "readable as an absence and not as a finding.",
            st["muted"],
        ))
        story.append(Spacer(1, 1.5 * mm))
        for item in unavailable:
            story += [Paragraph(f"&bull; {_t(item)}", st["muted"]), Spacer(1, 1.2 * mm)]

    story += [
        Paragraph("Examiner sign-off", st["h2"]),
        Paragraph(
            "The measurements, plates and inventory in this report were produced by an "
            "automated system from the images held against this case. The findings and "
            "the result are the examiner's, and this report is not an expert conclusion "
            "until signed below.",
            st["body"],
        ),
        Spacer(1, 4 * mm),
        Paragraph(
            f"Report generated {_t(report.get('generated_at') or datetime.now(timezone.utc).isoformat())} "
            f"by {_t(report.get('generated_by', '-'))}.",
            st["muted"],
        ),
        Spacer(1, 10 * mm),
    ]
    signature = Table(
        [
            [Paragraph("Examiner name", st["cell_left"]), Paragraph("", st["cell_left"])],
            [Paragraph("Signature", st["cell_left"]), Paragraph("", st["cell_left"])],
            [Paragraph("Date", st["cell_left"]), Paragraph("", st["cell_left"])],
        ],
        colWidths=(45 * mm, 100 * mm),
        rowHeights=(11 * mm, 11 * mm, 11 * mm),
        hAlign="LEFT",
    )
    signature.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, _RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(signature)

    doc.build(story)
    return buffer.getvalue()


__all__ = ["render_examination_pdf"]
