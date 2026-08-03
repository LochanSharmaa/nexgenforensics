# Forensic Face Comparison Report — Implementation Plan

**Written to be executed. Companion to IMPLEMENTATION-PLAN.md and NEXTGEN-ARCHITECTURE.md.**
**Date: 2026-08-03 · Scope: turn the case-log PDF into an examiner-grade comparison report**

---

## 0. What exists today, and what is missing

`backend/imatch_api/services/report_pdf.py` produces a **case log**: case metadata,
a candidate table, an audit trail, a signature block. It is a correct and honest
document, and it contains no imagery, no measurements, and none of the evidence
machinery this codebase already has.

Already built and **not reaching the report**:

| Capability | Where | Status |
|---|---|---|
| Calibrated log10 LR + ENFSI verbal bands | `nexgen_engine/forensics/evidence.py` | built, unwired |
| Logistic / conditional calibration | `nexgen_engine/forensics/calibration.py` | built, unwired |
| Reference population, random-match probability | `nexgen_engine/forensics/population.py` | built, unwired |
| Cllr, PAV, Tippett | `nexgen_engine/forensics/metrics.py` | built, unwired |
| Open-set handling | `nexgen_engine/forensics/openset.py` | built, unwired |
| Probe and enrolment images on disk | `SearchRun.probe_path`, `Template.image_path` | persisted, unread by the report |

Missing entirely: dense landmarks, morphometric observations, morphological
comparison grid, facial-mark correspondence, image comparability assessment,
model saliency, and any rendering of imagery into the PDF.

**Two facts discovered during survey that shape every phase below:**

1. `FaceAnalysis(name="buffalo_l")` is constructed with no `allowed_modules`
   filter (`nexgen_engine/models/insightface_backbone.py:63`). The pack's
   `2d106det` (106 landmarks) and `1k3d68` (68 3D landmarks, real pose) models
   are **already downloaded and already loaded into memory**, and nothing reads
   them. Dense landmarks cost zero new dependencies and zero new model assets.
2. `opencv-python-headless` and `scikit-image` are already declared in
   `requirements-engine.txt`. Mark detection needs no new dependency either.

The only genuinely new dependency in this whole plan is **none**. That is
unusual and it should be preserved — see §9.

---

## 1. The framing decision, made once, binding on every phase

The obvious build is: detect landmarks, draw calipers on both faces, tabulate
inter-landmark distances, and title the section *"why these images match."*

**We are not doing that.** Photo-anthropometry — deriving identity conclusions
from inter-landmark distances measured in 2D photographs — has been explicitly
rejected as an identification method by both FISWG and ENFSI. The reasons are
not implementation defects and cannot be engineered away:

- A photograph is a projection. Any inter-landmark distance varies with yaw,
  pitch, camera-to-subject distance, and lens focal length.
- The resulting measurement variance for **the same person across two images**
  routinely exceeds the between-person variance of the same index.
- Landmark placement error on degraded imagery — which is our operating
  condition, see the TinyFace Cllr of 0.72 in IMPLEMENTATION-PLAN.md §0 — is
  itself large relative to the differences being reported.

If the report asserts that measurements demonstrate the match, an opposing
expert cites that literature, and the **calibrated LR — our actually defensible
number — is discredited alongside it.** The weakest claim in the document sets
the ceiling for the whole document.

**Therefore:**

- Measurements are printed as **documented observations with error bars and an
  explicit pose caveat**, never as grounds for the conclusion.
- The conclusion rests on (a) the calibrated LR from `evidence.py`,
  (b) morphological comparison graded by the examiner, and (c) facial-mark
  correspondence. (b) and (c) are the FISWG-accepted method; (a) is ours.
- Per-index differences are **never combined into a composite score.** They are
  strongly correlated; multiplying them manufactures evidence. This is the same
  failure mode the capacity guard in `evidence.py` already defends against, and
  the report must not reintroduce it through the back door.

This costs nothing visually. The plates still show two faces with landmarks,
lines, and numbers. Only the claim attached to them changes.

---

## 2. Target document structure

One report per case; one **comparison plate set** per adjudicated candidate.
Section numbering follows evaluative-reporting convention (ENFSI 2015).

| § | Section | Source | Phase |
|---|---|---|---|
| 1 | Request and scope | `Case`, `SearchRun.lawful_basis`, `purpose` | exists |
| 2 | Material examined | images + SHA-256 + capture metadata | P2, P6 |
| 3 | Image comparability | pose, effective IPD, illumination, compression | P1 |
| 4 | Comparison plate | normalised side-by-side, annotated + unmodified | P2 |
| 5 | Morphometric observations | normalised indices with CI and percentile | P1, P4b |
| 6 | Morphological comparison | FISWG feature grid, examiner-graded | P5, P8 |
| 7 | Facial marks and blemishes | detected marks, correspondence map | P3 |
| 8 | Model evidence | LR, verbal band, RMP, rank-1 margin, saliency | P4 |
| 9 | Limitations and conclusion | suppressed sections + reasons, sign-off | P6, P7 |

**§3 is a gate, not a preamble.** It decides which later sections are permitted
to render at all. If probe and candidate yaw differ by more than the configured
tolerance, §5 does not print with a caveat — §5 does not print, and §9 records
why. A caveat under a printed table is read as decoration; an absent table is
read as a finding.

**Non-matches get the identical structure.** For an eliminated candidate, §5
highlights indices whose delta falls outside the same-source distribution, §7
lists marks present in one image and absent in the other, and §8 reports
LR < 1 with its verbal band. "Why not" receives the same rigour as "why". The
symmetry is itself a credibility signal, and its absence is the first thing a
reviewer notices.

---

## 3. Phase 1 — Dense landmarks, pose, and comparability

**New file: `backend/nexgen_engine/forensics/morphology.py`**

Expose what the loaded pack already computes.

```python
@dataclass(frozen=True)
class LandmarkSet:
    points_106: np.ndarray          # (106, 2) image pixels
    points_68_3d: np.ndarray | None # (68, 3) from 1k3d68, when available
    pose: HeadPose                  # from 1k3d68, NOT the 5-point approximation
    detector: str
    per_point_sigma: np.ndarray     # (106,) px, from the jitter bootstrap

@dataclass(frozen=True)
class MorphometricIndex:
    name: str
    value: float          # normalised by IPD -> scale-free
    ci95: float           # propagated from per_point_sigma
    landmark_ids: tuple[int, ...]   # so any number is traceable to its points

@dataclass(frozen=True)
class Comparability:
    usable: bool
    yaw_delta: float
    pitch_delta: float
    roll_delta: float
    effective_ipd_px: tuple[float, float]
    suppressed_sections: tuple[str, ...]
    reasons: tuple[str, ...]
```

Work items:

1. **`extract_landmarks(image, face) -> LandmarkSet`.** Call `2d106det` and
   `1k3d68` from the shared `FaceAnalysis`. Prefer `1k3d68` for pose: it is a
   fitted 3D model, materially better than the geometric approximation in
   `alignment.py::estimate_pose`, which its own docstring already flags as
   unsuitable for reporting to an analyst. Keep `estimate_pose` for the
   detection-time reject path; do not change its behaviour.

2. **Jitter bootstrap for `per_point_sigma`.** Re-run landmark extraction over
   N deterministic perturbations of the crop (±2 px translation, ±2° rotation,
   ±3% scale; fixed seed). The per-point standard deviation of the returned
   landmarks is the landmark uncertainty. Propagate it through each index to
   get `ci95`.

   *This is the single most important item in Phase 1.* Without it §5 prints
   bare numbers and inherits every criticism in §1. With it, a delta of
   0.014 ± 0.014 self-evidently reports "indistinguishable from noise", and
   the section becomes defensible precisely because it declines to overclaim.

3. **Index set**, all divided by inter-pupillary distance: intercanthal width,
   nose width (alare–alare), nose length (nasion–subnasale), mouth width
   (cheilion–cheilion), eye-to-mouth vertical, bizygomatic width, philtrum
   length. Each carries its `landmark_ids` so any printed number is traceable
   back to specific points on the plate.

4. **`assess_comparability(a, b, policy) -> Comparability`.** Yaw/pitch/roll
   deltas, effective IPD in pixels (the resolution proxy that actually matters
   — not image dimensions), and the suppression list. Tolerances live in
   `nexgen_engine/config.py`, defaulted at yaw Δ ≤ 15°, pitch Δ ≤ 15°,
   effective IPD ≥ 40 px, and recorded in the report so a reader can see the
   policy that governed their document.

5. **Fallback discipline.** When the engine is in deterministic-fallback mode
   (`recognition_capable == false`), `extract_landmarks` **raises**. It must
   never return synthesised landmarks. Every downstream renderer treats the
   raise as "section unavailable" and records it in §9. This follows the
   existing precedent of `report_pdf.py::draw_enhanced_pair`, which fails hard
   rather than emitting a partially-compliant exhibit.

**Tests** (`backend/tests_engine/test_morphology.py`): indices invariant under
synthetic rescale and in-plane rotation of a known image; `per_point_sigma`
grows monotonically as JPEG quality drops (reuse `nexgen_engine/degradation/`);
comparability suppresses §5 at 25° yaw delta; fallback mode raises.

---

## 4. Phase 2 — Deterministic exhibit rendering

**New file: `backend/nexgen_engine/forensics/exhibits.py`**

```python
@dataclass(frozen=True)
class Exhibit:
    png: bytes
    sha256: str              # of these exact bytes
    source_sha256: tuple[str, ...]   # every image that fed it
    kind: str                # "comparison_plate" | "marks" | "saliency" | ...
    params: dict             # every knob, sufficient to regenerate bit-identically
    engine_version: str
```

Renderers: `comparison_plate()`, `unmodified_pair()`, `marks_plate()`,
`saliency_plate()`.

Rules, all non-negotiable:

- **PIL only.** No matplotlib, no external font files. Fonts and colours are
  pinned constants; a font substitution changes the bytes and therefore the
  hash. Determinism is the property that makes an exhibit reproducible on
  challenge, and it is easy to lose by accident.
- **Every annotated plate is emitted with its unmodified pair**, same scale,
  same page. Extends the `draw_enhanced_pair` rule from enhanced imagery to all
  derived exhibits.
- **Normalisation for the plate**: roll-correct both faces, scale both to a
  common IPD, place on a shared measurement grid. State the applied transform
  in the caption — the reader must know the images were geometrically
  normalised for display and that this is why they align.
- Regenerating an exhibit from the same inputs and params yields an identical
  `sha256`. Asserted in tests, not assumed.

---

## 5. Phase 3 — Facial marks and blemishes

**New file: `backend/nexgen_engine/forensics/marks.py`**

Detect scale-space blobs (`skimage.feature.blob_doh` / `blob_log`) on the
aligned, illumination-normalised crop; express positions in the aligned
reference frame so probe and candidate are directly comparable; match by
position with a tolerance derived from `per_point_sigma`, not a magic constant.

Report **correspondence and non-correspondence both** — a mark clearly present
in one image and clearly absent in a comparable region of the other is
probative, and omitting it would make the section an argument rather than an
observation.

Guard against the obvious false positives: JPEG blocking artefacts, specular
highlights, dust, hair. Each detected mark carries a confidence and a
"comparable region" flag; marks in regions occluded or below resolution in
either image are listed as *not comparable* rather than silently dropped.

This section is, in practice, the strongest single line of evidence available
in this discipline. It deserves more care than §5, and gets it.

---

## 6. Phase 4 — Model evidence

### 4a. Wire the existing evidence stack

**New file: `backend/imatch_api/services/evidence_service.py`**

Load the fitted calibrator and reference population from
`runtime/forensics/`, produce an `EvidenceReport` per candidate: log10 LR, CI,
ENFSI verbal band, reference population identity, random-match probability at
the actual gallery size, capacity-guard status. All of this already exists in
`nexgen_engine/forensics/`; this phase is wiring and persistence, not new
science.

Also report **rank-1 margin** and the rank-1 score plotted against the score
histogram of the rest of the gallery. A rank-1 score of 0.62 means something
very different when rank-2 is 0.61 than when rank-2 is 0.31, and the histogram
shows that at a glance in a way no scalar does.

### 4b. Same-source difference distributions

**New script: `backend/scripts/build_morphometric_reference.py`**

For §5's percentile column, compute the distribution of each index's delta over
same-source and different-source pairs from a labelled dataset, stratified by
pose bin and effective IPD bin. Output to `runtime/forensics/morphometric_reference.json`.

**This is a data task on the critical path.** Until it exists, §5's percentile
column cannot render and the section prints values, CIs, and deltas only. Ship
it that way rather than inventing tolerances — an arbitrary "within 5%" threshold
is exactly the unsupported claim §1 exists to prevent.

### 4c. Saliency

**New file: `backend/nexgen_engine/forensics/saliency.py`**

Occlusion sensitivity over the cosine similarity: slide a fixed occluder across
the probe on a deterministic grid, record the score drop, render as a heatmap.
Seed and grid recorded in `Exhibit.params`.

Labelled in the PDF as **model attention, not human-interpretable feature
evidence.** It shows which pixels the network used; it does not show which
facial features an examiner should credit. Losing that distinction turns a
useful diagnostic into a misleading exhibit.

---

## 7. Phase 5 — Persistence

No Alembic in this project — `SQLModel.metadata.create_all` at
`imatch_api/db/session.py:108` plus targeted scripts in `backend/scripts/`
(precedent: `migrate_auth_columns.py`). Follow that pattern.

**New tables in `imatch_api/db/models.py`:**

```python
class ComparisonExhibit(SQLModel, table=True):
    # tenant_id, candidate_id, kind, sha256, source_sha256, params (JSON),
    # engine_version, created_at, created_by, storage_path
```

```python
class MorphologicalGrading(SQLModel, table=True):
    # tenant_id, candidate_id, feature (FISWG feature-list id),
    # grade: correspondence | difference | not_comparable | not_visible,
    # note, graded_by, graded_at
```

**Rule: the PDF may not print a grading no examiner made.** §6 renders only
persisted `MorphologicalGrading` rows. Ungraded features print as *not
assessed*, never as blank and never as a default. A report that silently
defaults a feature to "correspondence" is a fabricated expert opinion.

Migration script: `backend/scripts/migrate_comparison_tables.py`.
Exhibit generation is recorded through `AuditService` with the existing
`ACTION_EXPORT` treatment — a derived exhibit leaving the system is the same
class of event as a report export.

---

## 8. Phase 6–8 — Assembly, PDF, UI

### Phase 6 — `backend/imatch_api/services/comparison_service.py`

Reads `SearchRun.probe_path` and `Template.image_path` via `StorageService`,
runs morphology → comparability → marks → evidence → exhibits, persists
exhibits, returns the `comparison` dict.

Then extend `ReportService._search_section()` with a `comparison` block per
candidate. **The one-dict-three-formats invariant is preserved**: JSON,
Markdown and PDF continue to render from the same dict, exactly as
`report_pdf.py`'s module docstring requires. Exhibits appear in JSON as hashes
plus retrieval URLs, and as embedded plates in the PDF.

Retention interacts here: `StorageService` purges at 90 days. Exhibits are
derived from images that may be purged before the report is re-exported. Either
exhibits inherit the source retention (report becomes non-regenerable after
purge, and says so) or exhibits are retained under a separate case-bound policy.
**Decision required before Phase 6 lands** — flagged in §10.

### Phase 7 — `backend/imatch_api/services/report_pdf.py`

Add `platypus.Image` plates from `BytesIO`, a `KeepTogether` block per
comparison, and §9's suppressed-section list. Replace `draw_enhanced_pair`'s
`NotImplementedError` with a real implementation built on `exhibits.py` — it
can finally satisfy its own contract, since `unmodified_pair()` provides
exactly the pairing it demands.

Page budget: one comparison ≈ 3 pages. Cap embedded comparisons at the
adjudicated candidates plus rank-1; the rest stay in the existing table. Any
cap is stated in the document — a silent truncation reads as completeness.

### Phase 8 — `frontend/src/workspace/VerifyPage.jsx`

New `components/MorphologyPanel.jsx`: comparability banner, plate viewer,
morphometric table, marks overlay toggle, and the FISWG grading grid that
writes `MorphologicalGrading` rows. Follow the `ProvenancePanel.jsx` structure —
organised around the examiner's questions, technical detail collapsed.

**§6 cannot be produced without this UI.** The PDF renders examiner gradings;
the examiner needs somewhere to make them. Phase 8 is not polish, it is the
input path for the section that makes this a forensic report rather than a
system printout.

---

## 9. Sequencing

**Milestone 1 — "publishable, no new science" (report §2, §3, §4, §8).**
Phases 1, 2, 4a, 6, 7. Material examined, comparability gate, comparison plate,
and the LR that already exists. This alone converts a case log into a report
recognisable to a forensic reader. No new dependencies, no new data collection.

**Milestone 2 — "examiner method" (report §6, §7).** Phases 3, 5, 8.
Mark correspondence and the FISWG grading path.

**Milestone 3 — "observations" (report §5 percentile column, §8 saliency).**
Phases 4b, 4c. Gated behind Milestone 1's comparability check and behind the
reference-distribution data task.

§5 renders from Milestone 1 with values, CIs and deltas; it gains its
percentile column in Milestone 3. It is gated behind §3 in every milestone.

---

## 10. Open decisions

1. **Exhibit retention vs. the 90-day image purge** (§8, Phase 6). Blocking for
   Phase 6.
2. **Which labelled dataset backs the morphometric reference distributions**
   (Phase 4b), and whether pose/IPD stratification is fine enough to be
   meaningful at the available sample size. Blocking for §5's percentile column
   only.
3. **FISWG feature-list version to encode** in `MorphologicalGrading.feature`.
   Affects the schema; cheap now, migration later.

## 11. Explicitly out of scope

- Any composite score derived from morphometric indices (§1).
- Automated morphological grading. The system lays out §6 and records who
  graded it; it does not decide it.
- Automated identification conclusions. Every existing notice in
  `report_service.py::REPORT_NOTICE` and `report_pdf.py` stands unchanged.
- Enhanced or super-resolved imagery in evidentiary sections, beyond the
  existing labelled-preview rule.
