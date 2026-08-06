"""Tests for the photograph examination report.

The properties checked here are the ones that make the document defensible
rather than merely printable:

* every image held against the case reaches the inventory, and each face in
  each image gets its own mark;
* the same photograph submitted twice is one exhibit, not two;
* measurements are invariant under rescaling and rotation of the same image,
  and every figure carries an interval;
* nothing the examiner did not write is printed as their opinion;
* a filename from outside cannot inject markup into the PDF.
"""

from __future__ import annotations

import json
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from sqlmodel import Session, select

from imatch_api.core.config import get_settings
from imatch_api.db.models import (
    Case,
    ExaminationNote,
    ExaminationSection,
    Role,
    SearchRun,
    Subject,
    Template,
)
from imatch_api.db.session import get_engine
from imatch_api.services.examination_pdf import render_examination_pdf
from imatch_api.services.examination_service import ExaminationService
from imatch_api.services.storage_service import StorageService, safe_filename


# --------------------------------------------------------------- fixtures ---


@pytest.fixture
def session():
    with Session(get_engine()) as active:
        yield active


@pytest.fixture
def storage() -> StorageService:
    settings = get_settings()
    return StorageService(settings.storage_path, settings.probe_retention_days)


@pytest.fixture
def tenant(tenant_factory):
    return tenant_factory("exam-tenant")


@pytest.fixture
def examiner(tenant, user_factory):
    return user_factory(tenant, email="examiner@example.com", role=Role.SUPERVISOR)


@pytest.fixture
def api_headers(tenant_factory, user_factory, auth_headers):
    """A logged-in supervisor for the endpoint tests.

    Separate from the ``examiner`` fixture above, which seeds rows directly:
    these tests go through HTTP and need the default tenant the login path
    resolves.
    """
    api_tenant = tenant_factory("default")
    user_factory(api_tenant, email="supervisor@example.com", role=Role.SUPERVISOR)
    return auth_headers("supervisor@example.com")


# ---------------------------------------------------------------- helpers ---


def _pdf_text(pdf: bytes) -> str:
    """Extracted page text. A raw byte search would miss it -- streams are compressed."""
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(pdf))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _seed(session, storage, tenant_id, owner_id, questioned, specimen, notes=()):
    """Build a case holding the given image files, and return its id."""
    case = Case(
        tenant_id=tenant_id,
        reference="EXAM-1",
        title="Photograph comparison",
        owner_id=owner_id,
        lawful_basis="Court-directed examination",
    )
    session.add(case)
    session.commit()

    for path in questioned:
        stored = storage.store(tenant_id, path.read_bytes(), category="probes")
        session.add(
            SearchRun(
                tenant_id=tenant_id,
                case_id=case.id,
                operator_id=owner_id,
                mode="single",
                probe_sha256=stored.sha256,
                probe_path=stored.path,
                probe_filename=path.name,
                recognition_capable=True,
            )
        )
    session.commit()

    if specimen:
        subject = Subject(
            tenant_id=tenant_id,
            display_name="Reference subject",
            case_id=case.id,
            enrolled_by=owner_id,
        )
        session.add(subject)
        session.commit()
        for path in specimen:
            stored = storage.store(tenant_id, path.read_bytes(), category="enrolments")
            session.add(
                Template(
                    tenant_id=tenant_id,
                    subject_id=subject.id,
                    nonce="n",
                    ciphertext="c",
                    image_sha256=stored.sha256,
                    image_path=stored.path,
                    image_filename=path.name,
                    created_by=owner_id,
                )
            )
        session.commit()

    for section, body, marks in notes:
        session.add(
            ExaminationNote(
                tenant_id=tenant_id,
                case_id=case.id,
                section=section,
                body=body,
                marks=json.dumps(marks),
                author_id=owner_id,
            )
        )
    session.commit()
    return case.id


@pytest.fixture
def two_identities(face_paths):
    """Two identities with two usable photographs each."""
    names = [name for name, paths in sorted(face_paths.items()) if len(paths) >= 2]
    if len(names) < 2:
        pytest.skip("Two identities with two usable images each are required.")
    return face_paths[names[0]][:2], face_paths[names[1]][:2]


# -------------------------------------------------------------- inventory ---


def test_every_image_reaches_the_inventory_with_its_own_mark(
    session, storage, tenant, examiner, engine_service, two_identities
):
    questioned, specimen = two_identities
    case_id = _seed(session, storage, tenant.id, examiner.id, questioned, specimen)

    report = ExaminationService(engine_service, storage.read).build(
        session, tenant.id, case_id, "examiner"
    )

    assert len(report["exhibits"]["questioned"]) == len(questioned)
    assert len(report["exhibits"]["specimen"]) == len(specimen)

    marks = [
        mark
        for role in ("questioned", "specimen")
        for photo in report["exhibits"][role]
        for mark in photo["marks"]
    ]
    # Every mark is unique, and the two roles are numbered independently.
    assert len(marks) == len(set(marks))
    assert [m for m in marks if m.startswith("Q-")] == ["Q-1", "Q-2"]
    assert [m for m in marks if m.startswith("S-")] == ["S-1", "S-2"]


def test_the_same_photograph_twice_is_one_exhibit(
    session, storage, tenant, examiner, engine_service, two_identities
):
    """A probe searched repeatedly must not appear as several exhibits.

    Otherwise one photograph collects three marks and the inventory overstates
    how much material the examination actually rests on.
    """
    questioned, _ = two_identities
    repeated = [questioned[0], questioned[0], questioned[0]]
    case_id = _seed(session, storage, tenant.id, examiner.id, repeated, [])

    report = ExaminationService(engine_service, storage.read).build(
        session, tenant.id, case_id, "examiner"
    )
    assert len(report["exhibits"]["questioned"]) == 1
    assert report["exhibits"]["questioned"][0]["marks"] == ["Q-1"]


def test_unreadable_image_is_reported_not_skipped(
    session, storage, tenant, examiner, engine_service, two_identities
):
    questioned, _ = two_identities
    case_id = _seed(session, storage, tenant.id, examiner.id, questioned[:1], [])

    run = session.exec(select(SearchRun).where(SearchRun.case_id == case_id)).one()
    run.probe_path = "does/not/exist.jpg"
    session.add(run)
    session.commit()

    report = ExaminationService(engine_service, storage.read).build(
        session, tenant.id, case_id, "examiner"
    )
    photo = report["exhibits"]["questioned"][0]
    assert photo["marks"] == []
    assert "no longer held" in photo["missing_reason"]


# ----------------------------------------------------------- measurements ---


def test_measurements_are_scale_and_rotation_invariant(engine_service, two_identities):
    """The same face, rescaled and rotated, must measure the same.

    A measurement that moves when the camera is turned sideways is measuring
    the photograph, not the face, and cannot appear in a comparison table.
    """
    from nexgen_engine.forensics.morphology import analyse

    (source, _), _ = two_identities
    landmarkers = engine_service.runtime.landmarkers
    detector = engine_service.runtime.detector

    original = Image.open(source).convert("RGB")
    variants = {
        "original": original,
        "half": original.resize((original.width // 2, original.height // 2), Image.LANCZOS),
        "rotated": original.rotate(12, resample=Image.BICUBIC, expand=True),
    }

    ratios: dict[str, dict[str, float]] = {}
    for name, image in variants.items():
        faces = detector.detect(image)
        if not faces:
            pytest.skip(f"Detector found no face in the {name} variant.")
        box = max(faces, key=lambda f: f.box.area).box
        bgr = np.asarray(image)[:, :, ::-1].copy()
        morphology = analyse(
            bgr, (box.left, box.top, box.right, box.bottom), landmarkers
        )
        ratios[name] = {m.key: m.ratio_to_ipd for m in morphology.measurements}

    for key in ratios["original"]:
        base = ratios["original"][key]
        for variant in ("half", "rotated"):
            if key not in ratios[variant]:
                continue
            drift = abs(ratios[variant][key] - base) / base
            assert drift < 0.10, (
                f"Measurement {key} moved {drift:.1%} under {variant}; a geometric "
                "transform of the same photograph must not change it."
            )


def test_every_measurement_carries_an_interval(
    session, storage, tenant, examiner, engine_service, two_identities
):
    """A bare number in this table would be an unsupported claim."""
    questioned, _ = two_identities
    case_id = _seed(session, storage, tenant.id, examiner.id, questioned[:1], [])

    report = ExaminationService(engine_service, storage.read).build(
        session, tenant.id, case_id, "examiner"
    )
    tables = report["observation"]["tables"]
    assert tables, "A measurable exhibit must produce a micrometry table."

    measurement_rows = [row for row in tables[0]["rows"] if row["group"] == "Measurements"]
    assert len(measurement_rows) == 8
    for row in measurement_rows:
        for value in row["values"]:
            assert "±" in value, f"{row['label']} printed {value!r} with no interval."


def test_the_caveat_travels_with_the_table(
    session, storage, tenant, examiner, engine_service, two_identities
):
    questioned, _ = two_identities
    case_id = _seed(session, storage, tenant.id, examiner.id, questioned[:1], [])

    report = ExaminationService(engine_service, storage.read).build(
        session, tenant.id, case_id, "examiner"
    )
    caveat = report["observation"]["caveat"]
    assert "not grounds for identification" in caveat
    assert "FISWG" in caveat


# ------------------------------------------------------- examiner opinion ---


def test_no_examiner_text_means_the_section_says_so(
    session, storage, tenant, examiner, engine_service, two_identities
):
    """The report must never compose a finding nobody made."""
    questioned, specimen = two_identities
    case_id = _seed(session, storage, tenant.id, examiner.id, questioned, specimen)

    report = ExaminationService(engine_service, storage.read).build(
        session, tenant.id, case_id, "examiner"
    )
    assert report["result"] == []
    assert report["secondary"] == []
    assert report["letter"]["question"] == ""

    assert "Not recorded" in _pdf_text(render_examination_pdf(report))
    assert any("No result of examination has been recorded" in item
               for item in report["limitations"])


def test_examiner_text_is_printed_verbatim(
    session, storage, tenant, examiner, engine_service, two_identities
):
    questioned, specimen = two_identities
    finding = "The hairline in Q-1 corresponds with that in S-1."
    case_id = _seed(
        session, storage, tenant.id, examiner.id, questioned, specimen,
        notes=[
            (ExaminationSection.QUESTION, "Are these the same person?", []),
            (ExaminationSection.OBSERVATION, finding, ["Q-1", "S-1"]),
            (ExaminationSection.RESULT, "Q-1 corresponds with S-1.", []),
        ],
    )

    report = ExaminationService(engine_service, storage.read).build(
        session, tenant.id, case_id, "examiner"
    )
    assert report["letter"]["question"] == "Are these the same person?"
    assert report["result"] == ["Q-1 corresponds with S-1."]
    assert report["secondary"][0]["body"] == finding
    assert [item["mark"] for item in report["secondary"][0]["grid"]] == ["Q-1", "S-1"]


def test_a_finding_citing_an_absent_exhibit_says_so(
    session, storage, tenant, examiner, engine_service, two_identities
):
    """Silently dropping the citation would hide that the finding is unsupported."""
    questioned, specimen = two_identities
    case_id = _seed(
        session, storage, tenant.id, examiner.id, questioned, specimen,
        notes=[(ExaminationSection.OBSERVATION, "Refers to a mark that is not here.",
                ["Q-1", "Z-9"])],
    )

    report = ExaminationService(engine_service, storage.read).build(
        session, tenant.id, case_id, "examiner"
    )
    assert report["secondary"][0]["unknown_marks"] == ["Z-9"]


# -------------------------------------------------------------- rendering ---


def test_pdf_renders_with_plates(
    session, storage, tenant, examiner, engine_service, two_identities
):
    questioned, specimen = two_identities
    case_id = _seed(session, storage, tenant.id, examiner.id, questioned, specimen)

    report = ExaminationService(engine_service, storage.read).build(
        session, tenant.id, case_id, "examiner"
    )
    pdf = render_examination_pdf(report)

    assert pdf.startswith(b"%PDF")
    assert b"/Image" in pdf, "The report must embed the exhibit imagery."
    # Four exhibit photographs plus four measurement plates: a report this size
    # cannot be a few kilobytes unless the imagery silently failed to render.
    assert len(pdf) > 100_000
    assert len(report["observation"]["plates"]) == 4


def test_a_hostile_filename_cannot_inject_markup(
    session, storage, tenant, examiner, engine_service, two_identities
):
    """reportlab's Paragraph parses markup, and the filename comes from outside."""
    questioned, _ = two_identities
    case_id = _seed(session, storage, tenant.id, examiner.id, questioned[:1], [])

    run = session.exec(select(SearchRun).where(SearchRun.case_id == case_id)).one()
    run.probe_filename = '<font color="red">&pwn;</font>.jpg'
    session.add(run)
    session.commit()

    report = ExaminationService(engine_service, storage.read).build(
        session, tenant.id, case_id, "examiner"
    )
    pdf = render_examination_pdf(report)
    assert pdf.startswith(b"%PDF")

    # The tag must reach the page as literal text. If it had been parsed as
    # markup the angle brackets would be gone and the filename would render in
    # red, which is how a hostile name smuggles emphasis into an exhibit list.
    text = _pdf_text(pdf)
    assert '<font color="red">' in text
    assert "&pwn;" in text


@pytest.mark.parametrize(
    ("supplied", "expected"),
    [
        ("../../etc/passwd", "passwd"),
        ("C:\\Users\\me\\DSC_0001.JPG", "DSC_0001.JPG"),
        ("photo\x00.jpg", "photo.jpg"),
        ("..", ""),
        ("", ""),
        (None, ""),
        ("holiday (1).jpeg", "holiday (1).jpeg"),
    ],
)
def test_safe_filename(supplied, expected):
    assert safe_filename(supplied) == expected


def test_safe_filename_is_length_capped():
    assert len(safe_filename("x" * 500 + ".jpg")) == 120


# ------------------------------------------------------------- json export --


class TestEndpoint:
    """The report reached through the API, filenames and all."""

    @pytest.fixture
    def api_case(self, client, api_headers, face_paths) -> str:
        import base64

        names = [name for name, paths in sorted(face_paths.items()) if len(paths) >= 2]
        if len(names) < 2:
            pytest.skip("Two identities with two usable images each are required.")
        questioned = face_paths[names[0]]
        specimen = face_paths[names[1]]

        case = client.post(
            "/api/cases",
            headers=api_headers,
            json={
                "reference": "OP-EXAM-1",
                "title": "Photograph examination coverage",
                "lawful_basis": "Warrant 2026/115",
            },
        )
        assert case.status_code == 201, case.text
        case_id = case.json()["id"]

        # case_id matters: an enrolment not linked to the case is not a
        # specimen exhibit of it, and would not appear in the inventory.
        enrol = client.post(
            "/api/subjects",
            headers=api_headers,
            json={
                "display_name": "Reference subject",
                "case_id": case_id,
                "image_base64": base64.b64encode(specimen[0].read_bytes()).decode(),
                "filename": specimen[0].name,
                "lawful_basis": "Warrant 2026/115",
            },
        )
        assert enrol.status_code == 201, enrol.text

        search = client.post(
            "/api/imatch/search",
            headers=api_headers,
            json={
                "image_base64": base64.b64encode(questioned[0].read_bytes()).decode(),
                "filename": questioned[0].name,
                "lawful_basis": "Warrant 2026/115",
                "case_id": case_id,
                "mode": "single",
            },
        )
        assert search.status_code == 200, search.text
        return case_id

    def test_examination_pdf_downloads(self, client, api_headers, api_case):
        response = client.get(
            f"/api/cases/{api_case}/report",
            headers=api_headers,
            params={"fmt": "examination"},
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "application/pdf"
        assert "examination-OP-EXAM-1.pdf" in response.headers["content-disposition"]
        assert response.content.startswith(b"%PDF")
        assert b"/Image" in response.content

    def test_submitted_filenames_reach_the_exhibit_list(
        self, client, api_headers, api_case, face_paths
    ):
        """The description of exhibits has to name files the submitter recognises."""
        body = client.get(
            f"/api/cases/{api_case}/report",
            headers=api_headers,
            params={"fmt": "examination-json"},
        ).json()

        questioned = body["exhibits"]["questioned"]
        specimen = body["exhibits"]["specimen"]
        assert questioned and specimen, "both roles must be populated"
        assert questioned[0]["filename"].endswith(".jpg")
        assert specimen[0]["filename"].endswith(".jpg")
        assert questioned[0]["marks"] == ["Q-1"]
        assert specimen[0]["marks"] == ["S-1"]

    def test_notes_round_trip_into_the_report(self, client, api_headers, api_case):
        """The examiner's text is the input path for the sections nothing generates."""
        put = client.put(
            f"/api/cases/{api_case}/examination-notes",
            headers=api_headers,
            json=[
                {"section": "question", "body": "Are these the same person?"},
                {"section": "observation", "ordinal": 1,
                 "body": "The hairline corresponds.", "marks": ["Q-1", "S-1"]},
                {"section": "result", "body": "Q-1 corresponds with S-1."},
            ],
        )
        assert put.status_code == 200, put.text
        assert {note["section"] for note in put.json()} == {"question", "observation", "result"}

        body = client.get(
            f"/api/cases/{api_case}/report",
            headers=api_headers,
            params={"fmt": "examination-json"},
        ).json()
        assert body["letter"]["question"] == "Are these the same person?"
        assert body["result"] == ["Q-1 corresponds with S-1."]
        assert body["secondary"][0]["marks"] == ["Q-1", "S-1"]
        assert body["secondary"][0]["unknown_marks"] == []

    def test_notes_replace_rather_than_accumulate(self, client, api_headers, api_case):
        for body in ("First attempt.", "Second attempt."):
            response = client.put(
                f"/api/cases/{api_case}/examination-notes",
                headers=api_headers,
                json=[{"section": "result", "body": body}],
            )
            assert response.status_code == 200, response.text

        notes = client.get(
            f"/api/cases/{api_case}/examination-notes", headers=api_headers
        ).json()
        assert [note["body"] for note in notes] == ["Second attempt."]

    def test_unknown_note_section_is_refused(self, client, api_headers, api_case):
        response = client.put(
            f"/api/cases/{api_case}/examination-notes",
            headers=api_headers,
            json=[{"section": "conclusion", "body": "Not a valid section."}],
        )
        assert response.status_code == 422

    def test_unknown_format_is_refused(self, client, api_headers, api_case):
        response = client.get(
            f"/api/cases/{api_case}/report",
            headers=api_headers,
            params={"fmt": "exam"},
        )
        assert response.status_code == 400


def test_json_export_is_serialisable_and_carries_no_pixels(
    session, storage, tenant, examiner, engine_service, two_identities
):
    questioned, specimen = two_identities
    case_id = _seed(session, storage, tenant.id, examiner.id, questioned, specimen)

    report = ExaminationService(engine_service, storage.read).build(
        session, tenant.id, case_id, "examiner"
    )
    public = {key: value for key, value in report.items() if not key.startswith("_")}
    blob = json.dumps(public)

    assert "_photographs" not in blob
    # Plates are referenced by hash; the bytes stay in the PDF.
    for plate in public["observation"]["plates"]:
        assert len(plate["sha256"]) == 64
