"""Assembles the photograph examination report.

This is the second report this system produces and it answers a different
question from the first. ``ReportService`` documents *what the system did*: a
case log of searches, scores, thresholds and adjudications. This one documents
*what the photographs show*: an inventory of every image held against a case,
each face in each image marked as an exhibit, a table of measurements, plates,
and the examiner's findings.

THE MARKING CONVENTION
----------------------

Questioned photographs -- the images submitted for examination -- are marked
Q-1, Q-2 ... Specimen photographs -- the known references enrolled against the
case -- are marked S-1, S-2 ...

Marks are assigned per FACE, not per file. A photograph containing two people
yields two marks, because the examination is of a face and a report that gave
one mark to a two-person photograph could not say which person a measurement
belonged to.

WHAT THIS SERVICE WILL NOT DO
-----------------------------

It will not write the examiner's opinion. The question put to the examiner, the
purpose, the secondary-identification observations and the result of
examination are read from ``ExaminationNote`` rows and printed verbatim, or
printed as "not recorded". Nothing here composes them, and nothing here infers
a conclusion from the measurements -- see the module docstring of
``forensics/morphology.py`` for why the measurements cannot carry one.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image as PILImage
from sqlmodel import Session, select

from nexgen_engine.forensics import exhibits as exhibit_render
from nexgen_engine.forensics.morphology import (
    MEASUREMENT_KEYS,
    MEASUREMENT_LABELS,
    NORMALISED_IPD_INCHES,
    FaceMorphology,
    LandmarksUnavailableError,
)
from nexgen_engine.forensics.morphology import analyse as analyse_face

from ..db.models import (
    Case,
    ExaminationNote,
    ExaminationSection,
    SearchRun,
    Subject,
    Template,
)

logger = logging.getLogger(__name__)

#: Columns per micrometry table. Eight exhibits across an A4 portrait page is
#: already tight; beyond that the figures stop being readable, so the table is
#: split and each part is captioned with the exhibits it covers.
MAX_COLUMNS_PER_TABLE = 6

#: NTFS allocates in 4 KiB clusters. "Size on disk" in the description of
#: exhibits is that rounded figure, which is what an examiner reading the file
#: properties on a Windows machine will see -- the point of printing it is that
#: it matches what they can check.
_CLUSTER_BYTES = 4096

EXAMINATION_NOTICE = (
    "This examination compares photographic images. It does not establish "
    "identity by itself: a similarity between two photographs is evidence to be "
    "weighed, and the conclusion recorded here is the examiner's, made on their "
    "own professional judgement."
)

#: Printed under every micrometry table. The table exists because the report
#: form calls for it; this text is why the table is not the basis of the
#: conclusion. Removing it would leave the numbers making a claim that FISWG
#: has advised against for two decades.
MEASUREMENT_CAVEAT = (
    "These measurements are recorded observations, not grounds for "
    "identification. A photograph is a projection: every distance in it varies "
    "with head pose, camera-to-subject distance and lens focal length, and for "
    "one person photographed twice that variation routinely exceeds the "
    "variation between two different people. FISWG advises against deriving "
    "identification from photographic measurement, and no conclusion in this "
    "report rests on the figures in this table. Each figure is followed by the "
    "half-width of its 95% interval, obtained by re-measuring under controlled "
    "perturbations of the crop: where a difference between two exhibits is "
    "smaller than the intervals either side of it, that difference is not "
    "distinguishable from landmark placement noise. Individual differences are "
    "not combined into a composite score, because the indices are strongly "
    "correlated and combining them would manufacture evidence."
)


@dataclass
class ExhibitFace:
    """One marked face: the unit the report actually examines."""

    mark: str
    morphology: FaceMorphology | None
    unavailable_reason: str = ""
    plate: exhibit_render.Exhibit | None = None
    upper_face: exhibit_render.Exhibit | None = None

    @property
    def usable(self) -> bool:
        return self.morphology is not None


@dataclass
class ExhibitPhotograph:
    """One source photograph and every marked face in it."""

    role: str                       #: "questioned" | "specimen"
    filename: str
    sha256: str
    path: str
    byte_size: int
    created_at: datetime
    faces: list[ExhibitFace] = field(default_factory=list)
    image: PILImage.Image | None = None
    missing_reason: str = ""
    subject_name: str = ""

    @property
    def marks(self) -> list[str]:
        return [face.mark for face in self.faces]

    @property
    def size_on_disk(self) -> int:
        return int(math.ceil(self.byte_size / _CLUSTER_BYTES) * _CLUSTER_BYTES)

    @property
    def display_name(self) -> str:
        if self.filename:
            return self.filename
        return f"(no filename recorded; stored as {self.sha256[:16]}…)"


def _size_text(value: int) -> str:
    """File size in the unit a reader would see in their own file properties.

    Scaled rather than fixed at MB: a 6 KB exhibit printed as "0.01 MB" reads
    as a rounding error, and the exact byte count is what makes the line
    checkable against the file itself.
    """
    if value >= 1024 * 1024:
        return f"{value / (1024 * 1024):.2f} MB ({value:,} bytes)"
    if value >= 1024:
        return f"{value / 1024:.2f} KB ({value:,} bytes)"
    return f"{value:,} bytes"


class ExaminationService:
    """Builds the photograph examination report dict.

    One dict, rendered to PDF by ``examination_pdf`` and returned verbatim as
    JSON, so the two cannot disagree about a finding -- the same invariant
    ``report_pdf`` holds for the case log.
    """

    def __init__(self, engine: Any, image_loader: Any) -> None:
        self._engine = engine
        self._load = image_loader

    # ------------------------------------------------------------ build ---

    def build(
        self,
        session: Session,
        tenant_id: str,
        case_id: str,
        generated_by: str,
    ) -> dict[str, Any]:
        case = session.get(Case, case_id)
        if case is None or case.tenant_id != tenant_id:
            raise ValueError("Case not found.")

        unavailable: list[str] = []
        landmarkers = self._resolve_landmarkers(unavailable)

        photographs = self._collect(session, tenant_id, case_id)
        self._analyse(photographs, landmarkers, unavailable)

        notes = self._notes(session, tenant_id, case_id)
        faces = [face for photo in photographs for face in photo.faces]
        by_mark = {face.mark: (photo, face) for photo in photographs for face in photo.faces}

        return {
            "kind": "photograph_examination",
            "notice": EXAMINATION_NOTICE,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generated_by": generated_by,
            "case": {
                "id": case.id,
                "reference": case.reference,
                "title": case.title,
                "description": case.description,
                "status": case.status.value,
                "lawful_basis": case.lawful_basis,
                "opened_at": case.created_at.isoformat(),
            },
            "letter": self._letter(notes),
            "exhibits": {
                "questioned": [self._photo_dict(p) for p in photographs if p.role == "questioned"],
                "specimen": [self._photo_dict(p) for p in photographs if p.role == "specimen"],
            },
            "observation": {
                "tables": self._tables(photographs),
                "caveat": MEASUREMENT_CAVEAT,
                "normalisation": (
                    "Every face was geometrically scaled to a common "
                    f"inter-pupillary distance of {NORMALISED_IPD_INCHES:.3f} in before "
                    "measurement, so that figures from photographs of different "
                    "resolution are dimensionally comparable. Each figure is therefore "
                    "a measurement on a normalised image, not an anatomical measurement "
                    "of the subject, and is equal to that distance expressed as a "
                    "multiple of the inter-pupillary distance."
                ),
                "plates": [
                    {
                        "mark": face.mark,
                        "caption": face.plate.caption,
                        "sha256": face.plate.sha256,
                        "source_sha256": list(face.plate.source_sha256),
                    }
                    for face in faces
                    if face.plate is not None
                ],
                "legend": [
                    {"key": key, "label": MEASUREMENT_LABELS[key], "colour": list(colour)}
                    for key, colour in exhibit_render.measurement_legend()
                    if key in MEASUREMENT_LABELS
                ],
            },
            "secondary": self._secondary(notes, by_mark),
            "result": [note.body for note in notes.get(ExaminationSection.RESULT, [])],
            "limitations": self._limitations(photographs, notes),
            "unavailable": unavailable,
            # Kept out of the serialisable dict's own structures: the renderer
            # needs the pixels, a JSON export must not carry them.
            "_photographs": photographs,
        }

    # ---------------------------------------------------------- gather ----

    def _resolve_landmarkers(self, unavailable: list[str]) -> dict[str, Any] | None:
        """The dense-landmark models, or None with the reason recorded.

        A failure here suppresses the measurement table and the plates. It does
        NOT fall back to the five-point set or to any approximation: an
        approximate landmark produces an exact-looking number, and the reader
        has no way to tell one from the other.
        """
        try:
            return self._engine.runtime.landmarkers
        except Exception as error:  # noqa: BLE001
            logger.warning("Dense landmarks unavailable: %s", error)
            unavailable.append(
                "Micrometry table and measurement plates: the dense-landmark models "
                f"could not be loaded ({error}). No measurement is reported rather "
                "than an approximate one."
            )
            return None

    def _collect(self, session: Session, tenant_id: str, case_id: str) -> list[ExhibitPhotograph]:
        """Every image held against the case, deduplicated, in arrival order.

        Deduplication is by SHA-256, not by row: the same photograph searched
        three times is one exhibit, and listing it three times would inflate the
        inventory and give one face three different marks.
        """
        photographs: list[ExhibitPhotograph] = []
        seen: set[str] = set()

        runs = session.exec(
            select(SearchRun)
            .where(SearchRun.tenant_id == tenant_id, SearchRun.case_id == case_id)
            .order_by(SearchRun.created_at)
        ).all()
        for run in runs:
            if not run.probe_sha256 or run.probe_sha256 in seen:
                continue
            seen.add(run.probe_sha256)
            photographs.append(
                ExhibitPhotograph(
                    role="questioned",
                    filename=run.probe_filename,
                    sha256=run.probe_sha256,
                    path=run.probe_path,
                    byte_size=0,
                    created_at=run.created_at,
                )
            )

        subject_ids = session.exec(
            select(Subject.id).where(Subject.tenant_id == tenant_id, Subject.case_id == case_id)
        ).all()
        if subject_ids:
            templates = session.exec(
                select(Template)
                .where(Template.tenant_id == tenant_id, Template.subject_id.in_(subject_ids))
                .order_by(Template.created_at)
            ).all()
            names = {
                subject.id: subject.display_name
                for subject in session.exec(
                    select(Subject).where(Subject.id.in_(subject_ids))
                ).all()
            }
            for template in templates:
                if not template.image_sha256 or template.image_sha256 in seen:
                    continue
                seen.add(template.image_sha256)
                photographs.append(
                    ExhibitPhotograph(
                        role="specimen",
                        filename=template.image_filename,
                        sha256=template.image_sha256,
                        path=template.image_path,
                        byte_size=0,
                        created_at=template.created_at,
                        subject_name=names.get(template.subject_id, ""),
                    )
                )
        return photographs

    def inventory(
        self, session: Session, tenant_id: str, case_id: str
    ) -> list[dict[str, Any]]:
        """The exhibit marks for a case, without measuring anything.

        The notes editor needs to offer the examiner a list of marks to cite,
        and building the full report to get one would run the jitter bootstrap
        and render every plate -- seconds of work to populate a picker.

        Detection still has to run, because a mark is per face and only the
        detector knows how many faces a photograph holds. That is one pass per
        image rather than the twelve the measurement path costs, and no plate
        is rendered.

        Marks are assigned by exactly the same rule as ``build``: same
        collection order, same left-to-right ordering within a photograph. If
        the two ever disagreed, the examiner would cite marks that the report
        then reports as absent.
        """
        case = session.get(Case, case_id)
        if case is None or case.tenant_id != tenant_id:
            raise ValueError("Case not found.")

        photographs = self._collect(session, tenant_id, case_id)
        self._analyse(photographs, landmarkers=None, unavailable=[], measure=False)
        return [
            {
                "mark": face.mark,
                "role": photo.role,
                "filename": photo.display_name,
                "sha256": photo.sha256,
                "subject_name": photo.subject_name,
            }
            for photo in photographs
            for face in photo.faces
        ]

    def _analyse(
        self,
        photographs: list[ExhibitPhotograph],
        landmarkers: dict[str, Any] | None,
        unavailable: list[str],
        measure: bool = True,
    ) -> None:
        """Load each photograph, mark every face in it, and measure each face.

        With ``measure=False`` the marks are assigned and nothing else is
        computed -- the inventory path. That is a different thing from
        ``landmarkers is None``, which means measurement was wanted and could
        not be done, and which records a reason against every face.
        """
        counters = {"questioned": 0, "specimen": 0}
        prefix = {"questioned": "Q", "specimen": "S"}

        for photo in photographs:
            raw = self._load(photo.path) if photo.path else None
            if raw is None:
                photo.missing_reason = (
                    "The stored image is no longer held. It may have passed the "
                    "retention window."
                )
                continue
            photo.byte_size = len(raw)

            try:
                image = PILImage.open(BytesIO(raw))
                image.load()
                photo.image = image.convert("RGB")
            except Exception as error:  # noqa: BLE001
                photo.missing_reason = f"The stored file could not be decoded as an image ({error})."
                continue

            try:
                detected = self._engine.runtime.detector.detect(photo.image)
            except Exception as error:  # noqa: BLE001
                photo.missing_reason = f"Face detection failed on this photograph ({error})."
                continue

            if not detected:
                photo.missing_reason = "No face was detected in this photograph."
                continue

            # Left to right, so the marks in a group photograph run in the order
            # a reader's eye crosses the picture.
            detected = sorted(detected, key=lambda face: face.box.left)
            bgr = None if not measure else np.asarray(photo.image)[:, :, ::-1].copy()

            for face in detected:
                counters[photo.role] += 1
                mark = f"{prefix[photo.role]}-{counters[photo.role]}"
                box = (face.box.left, face.box.top, face.box.right, face.box.bottom)

                if not measure:
                    photo.faces.append(ExhibitFace(mark=mark, morphology=None))
                    continue

                if landmarkers is None:
                    photo.faces.append(
                        ExhibitFace(mark=mark, morphology=None,
                                    unavailable_reason="Dense-landmark models unavailable.")
                    )
                    continue

                try:
                    morphology = analyse_face(bgr, box, landmarkers, det_score=face.confidence)
                except Exception as error:  # noqa: BLE001 - includes LandmarksUnavailableError
                    logger.warning("Morphology failed for %s: %s", mark, error)
                    photo.faces.append(
                        ExhibitFace(mark=mark, morphology=None,
                                    unavailable_reason=f"Landmarks could not be placed ({error}).")
                    )
                    unavailable.append(f"Exhibit {mark}: no measurement ({error}).")
                    continue

                entry = ExhibitFace(mark=mark, morphology=morphology)
                try:
                    entry.plate = exhibit_render.measurement_plate(
                        photo.image, morphology, mark, photo.sha256
                    )
                    entry.upper_face = exhibit_render.upper_face_plate(
                        photo.image, morphology.landmarks, mark, photo.sha256
                    )
                except Exception as error:  # noqa: BLE001
                    logger.warning("Plate rendering failed for %s: %s", mark, error)
                    unavailable.append(f"Exhibit {mark}: plate could not be rendered ({error}).")
                photo.faces.append(entry)

    # --------------------------------------------------------- sections ---

    def _notes(
        self, session: Session, tenant_id: str, case_id: str
    ) -> dict[ExaminationSection, list[ExaminationNote]]:
        rows = session.exec(
            select(ExaminationNote)
            .where(ExaminationNote.tenant_id == tenant_id, ExaminationNote.case_id == case_id)
            .order_by(ExaminationNote.section, ExaminationNote.ordinal)
        ).all()
        grouped: dict[ExaminationSection, list[ExaminationNote]] = {}
        for row in rows:
            grouped.setdefault(row.section, []).append(row)
        return grouped

    def _letter(self, notes: dict[ExaminationSection, list[ExaminationNote]]) -> dict[str, Any]:
        def body(section: ExaminationSection) -> str:
            entries = notes.get(section, [])
            return "\n\n".join(note.body for note in entries if note.body.strip())

        return {
            "subject": "Photograph Examination Report",
            "salutation": "Sir,",
            "covering": (
                "Following your request to examine the samples described below, the "
                "forensic examination of the photographs is submitted herewith. The "
                "photographs have been examined against the parameters set out in this "
                "report and the findings are recorded below."
            ),
            "receipt": body(ExaminationSection.RECEIPT),
            "question": body(ExaminationSection.QUESTION),
            "purpose": body(ExaminationSection.PURPOSE),
        }

    def _photo_dict(self, photo: ExhibitPhotograph) -> dict[str, Any]:
        return {
            "marks": photo.marks,
            "filename": photo.filename,
            "display_name": photo.display_name,
            "sha256": photo.sha256,
            "byte_size": photo.byte_size,
            "size_text": _size_text(photo.byte_size) if photo.byte_size else "(size unavailable)",
            "size_on_disk_text": (
                _size_text(photo.size_on_disk) if photo.byte_size else "(size unavailable)"
            ),
            "received_at": photo.created_at.isoformat(),
            "subject_name": photo.subject_name,
            "missing_reason": photo.missing_reason,
            "faces": [
                {
                    "mark": face.mark,
                    "measured": face.usable,
                    "unavailable_reason": face.unavailable_reason,
                    "pose": face.morphology.landmarks.pose.as_dict() if face.usable else None,
                    "profile": face.morphology.landmarks.pose.profile if face.usable else None,
                }
                for face in photo.faces
            ],
        }

    def _tables(self, photographs: list[ExhibitPhotograph]) -> list[dict[str, Any]]:
        """The micrometry table, split into page-width chunks.

        Rows mirror the paper form: photograph quality, photograph profile, then
        the eight measurements. A face whose landmarks could not be placed is
        still given a column, filled with "not measurable" -- dropping it would
        make the table look complete when an exhibit is missing from it.
        """
        columns: list[tuple[ExhibitPhotograph, ExhibitFace]] = [
            (photo, face) for photo in photographs for face in photo.faces
        ]
        if not columns:
            return []

        tables: list[dict[str, Any]] = []
        for start in range(0, len(columns), MAX_COLUMNS_PER_TABLE):
            chunk = columns[start : start + MAX_COLUMNS_PER_TABLE]
            header = [
                {
                    "mark": face.mark,
                    "filename": photo.display_name,
                    "role": photo.role,
                }
                for photo, face in chunk
            ]

            rows: list[dict[str, Any]] = [
                {
                    "label": "Photograph quality",
                    "group": None,
                    "values": [_quality_word(face) for _photo, face in chunk],
                },
                {
                    "label": "Photograph profile",
                    "group": None,
                    "values": [
                        (
                            f"{face.morphology.landmarks.pose.profile} "
                            f"(yaw {face.morphology.landmarks.pose.yaw:+.1f}°)"
                            if face.usable
                            else "—"
                        )
                        for _photo, face in chunk
                    ],
                },
            ]

            for key in MEASUREMENT_KEYS:
                values = []
                for _photo, face in chunk:
                    if not face.usable:
                        values.append("not measurable")
                        continue
                    measurement = face.morphology.measurement(key)
                    values.append(measurement.display if measurement else "—")
                rows.append(
                    {
                        "label": f"{MEASUREMENT_LABELS[key]} ({key})",
                        "group": "Measurements",
                        "values": values,
                    }
                )

            tables.append(
                {
                    "header": header,
                    "rows": rows,
                    "caption": (
                        "Exhibits " + ", ".join(item["mark"] for item in header)
                        + ". Figures in inches at the stated normalisation, "
                        "followed by the 95% interval half-width."
                    ),
                }
            )
        return tables

    def _secondary(
        self,
        notes: dict[ExaminationSection, list[ExaminationNote]],
        by_mark: dict[str, tuple[ExhibitPhotograph, ExhibitFace]],
    ) -> list[dict[str, Any]]:
        """The examiner's findings, each with the exhibits it names.

        The grid under a finding is built from the marks the examiner cited, so
        the pictures a reader sees are always the exhibits the finding is about.
        A cited mark that does not exist is reported as such rather than
        silently dropped -- it means the finding refers to something not in the
        inventory, which a reader needs to know.
        """
        out: list[dict[str, Any]] = []
        for note in notes.get(ExaminationSection.OBSERVATION, []):
            try:
                marks = [str(mark) for mark in json.loads(note.marks or "[]")]
            except (ValueError, TypeError):
                marks = []

            grid: list[dict[str, Any]] = []
            unknown: list[str] = []
            for mark in marks:
                found = by_mark.get(mark)
                if found is None:
                    unknown.append(mark)
                    continue
                _photo, face = found
                grid.append(
                    {
                        "mark": mark,
                        "has_crop": face.upper_face is not None,
                        "sha256": face.upper_face.sha256 if face.upper_face else "",
                    }
                )
            out.append(
                {
                    "ordinal": note.ordinal,
                    "body": note.body,
                    "marks": marks,
                    "grid": grid,
                    "unknown_marks": unknown,
                    "recorded_at": note.updated_at.isoformat(),
                }
            )
        return out

    def _limitations(
        self,
        photographs: list[ExhibitPhotograph],
        notes: dict[ExaminationSection, list[ExaminationNote]],
    ) -> list[str]:
        """Fixed caveats, then conditionals driven by what the material actually is.

        Assembled by code so nothing here can be omitted during generation.
        Items are appended, never removed.
        """
        limitations = [
            "Photographic comparison is affected by head pose, illumination, "
            "camera-to-subject distance, lens characteristics, image resolution and "
            "lossy compression. Each of these alters the apparent shape of a face "
            "without any change in the face itself.",
            "Measurements taken from photographs are recorded in this report as "
            "observations. They are not a basis for identification and no conclusion "
            "here is derived from them.",
            "A conclusion in this report is the examiner's professional judgement. "
            "The system records, measures and renders; it does not decide.",
        ]

        faces = [face for photo in photographs for face in photo.faces]
        unmeasured = [face for face in faces if not face.usable]
        if unmeasured:
            limitations.append(
                f"{len(unmeasured)} of {len(faces)} marked exhibit(s) could not be "
                "measured. They appear in the table as 'not measurable' and are "
                "excluded from every figure quoted."
            )

        off_angle = [
            face for face in faces
            if face.usable and abs(face.morphology.landmarks.pose.yaw) > 15.0
        ]
        if off_angle:
            marks = ", ".join(face.mark for face in off_angle)
            limitations.append(
                f"Exhibit(s) {marks} are turned more than 15° from frontal. Comparison "
                "across a pose difference of this size is materially less reliable, and "
                "measurements from these exhibits are affected by projection to a "
                "degree the interval alongside them does not capture."
            )

        low_resolution = [
            face for face in faces
            if face.usable and face.morphology.landmarks.ipd_px < 40.0
        ]
        if low_resolution:
            marks = ", ".join(face.mark for face in low_resolution)
            limitations.append(
                f"Exhibit(s) {marks} carry fewer than 40 pixels between the pupils. "
                "At that resolution facial detail is at or below the level at which "
                "comparison has been validated."
            )

        missing = [photo for photo in photographs if photo.missing_reason]
        if missing:
            limitations.append(
                f"{len(missing)} photograph(s) held against this case could not be "
                "examined; the reason is printed against each in the description of "
                "exhibits."
            )

        if not notes.get(ExaminationSection.RESULT):
            limitations.append(
                "No result of examination has been recorded by an examiner. This "
                "report is incomplete until that is entered and signed."
            )
        return limitations


def _quality_word(face: ExhibitFace) -> str:
    """The paper form's vocabulary, driven by measurable resolution.

    Inter-pupillary distance in pixels, not image dimensions: a 12-megapixel
    photograph of a crowd can carry a face smaller than a cropped thumbnail, and
    it is the face that is being examined. The thresholds follow the published
    guidance that comparison degrades sharply below roughly 40 px between the
    pupils and is severely compromised well before 20 px.
    """
    if not face.usable:
        return "Not assessable"
    ipd = face.morphology.landmarks.ipd_px
    if ipd >= 90:
        return f"Good ({ipd:.0f} px IPD)"
    if ipd >= 40:
        return f"Fair ({ipd:.0f} px IPD)"
    return f"Poor ({ipd:.0f} px IPD)"


__all__ = [
    "EXAMINATION_NOTICE",
    "MAX_COLUMNS_PER_TABLE",
    "MEASUREMENT_CAVEAT",
    "ExaminationService",
    "ExhibitFace",
    "ExhibitPhotograph",
]
