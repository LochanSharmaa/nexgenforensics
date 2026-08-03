from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlmodel import Session, select

from ..db.models import (
    Adjudication,
    AuditRecord,
    Candidate,
    Case,
    SearchRun,
    Subject,
    Template,
    User,
)

# Repeated verbatim in every export. A report that leaves this out invites the
# reader to treat a similarity score as an identification.
REPORT_NOTICE = (
    "This report documents automated facial recognition searches. Automated face recognition "
    "produces investigative leads, not identifications. Candidate scores express similarity "
    "between images, not the probability that two images show the same person. No entry in this "
    "report may be relied upon as an identification unless a qualified examiner has independently "
    "verified it, and that verification is recorded here as an examiner adjudication."
)


# SearchRun.mode carries values from TWO different request vocabularies, and
# they overlap misleadingly.
#
#   /api/imatch/search  -> single | compare | batch | url
#       Always searched against the enrolled gallery. "single" describes the
#       SUBMISSION (one image), not the comparison: it is still 1:N. Reading it
#       as a 1:1 verification would understate the gallery every score was
#       ranked against.
#
#   /api/imatch/batch   -> one_to_many | pair | gallery
#       "pair" is the only true 1:1 verification in the system.
#
# Unknown values fall through to the raw string rather than a guess, because a
# wrong comparison type in a forensic report is worse than an unfamiliar one.
_MODE_LABELS = {
    "single": "Identification (1:N, enrolled gallery)",
    "compare": "Identification (1:N, enrolled gallery)",
    "batch": "Identification (1:N, enrolled gallery)",
    "url": "Identification (1:N, enrolled gallery)",
    "gallery": "Identification (1:N, enrolled gallery)",
    "one_to_many": "Comparison (1:N, against submitted images)",
    "pair": "Verification (1:1, nominated pair)",
}

#: The only mode that is a 1:1 verification.
VERIFICATION_MODE = "pair"


def mode_label(mode: str) -> str:
    return _MODE_LABELS.get(mode, mode or "(not recorded)")


class ReportService:
    """Builds a defensible case report from stored records.

    Everything in the output is read back from persisted rows -- searches,
    candidates, adjudications, and the audit chain -- rather than recomputed, so
    the report reflects what actually happened and when, including searches that
    returned nothing and candidates an examiner eliminated. Omitting those would
    make the report an argument rather than a record.
    """

    def build(self, session: Session, tenant_id: str, case_id: str, generated_by: str) -> dict[str, Any]:
        case = session.get(Case, case_id)
        if case is None or case.tenant_id != tenant_id:
            raise ValueError("Case not found.")

        runs = session.exec(
            select(SearchRun)
            .where(SearchRun.tenant_id == tenant_id, SearchRun.case_id == case_id)
            .order_by(SearchRun.created_at)
        ).all()

        searches = [self._search_section(session, run) for run in runs]

        audit_records = session.exec(
            select(AuditRecord)
            .where(AuditRecord.tenant_id == tenant_id, AuditRecord.resource_id == case_id)
            .order_by(AuditRecord.sequence)
        ).all()

        confirmed = sum(
            1
            for search in searches
            for candidate in search["candidates"]
            if candidate["adjudication"] == Adjudication.CONFIRMED.value
        )
        pending = sum(
            1
            for search in searches
            for candidate in search["candidates"]
            if candidate["adjudication"] == Adjudication.PENDING.value
        )

        return {
            "notice": REPORT_NOTICE,
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
                "closed_at": case.closed_at.isoformat() if case.closed_at else None,
            },
            "summary": {
                "searches_run": len(searches),
                "candidates_returned": sum(len(search["candidates"]) for search in searches),
                "confirmed_by_examiner": confirmed,
                "awaiting_adjudication": pending,
                "searches_with_no_result": sum(1 for search in searches if not search["candidates"]),
            },
            "searches": searches,
            # Methodology, limitations and conclusion, rendered from the record
            # by fixed templates. These are deliberately NOT model-generated --
            # see the module docstring of services/narrative_service.py.
            "standard_sections": self._standard_sections(case, searches),
            "audit_trail": [
                {
                    "timestamp": record.created_at.isoformat(),
                    "actor": record.actor_label or record.actor_id,
                    "action": record.action,
                    "outcome": record.outcome,
                    "lawful_basis": record.lawful_basis,
                    "entry_hash": record.entry_hash,
                }
                for record in audit_records
            ],
        }

    def _search_section(self, session: Session, run: SearchRun) -> dict[str, Any]:
        operator = session.get(User, run.operator_id)
        candidates = session.exec(
            select(Candidate).where(Candidate.search_run_id == run.id).order_by(Candidate.rank)
        ).all()

        return {
            "search_id": run.id,
            "performed_at": run.created_at.isoformat(),
            # Recorded verbatim, plus the resolved label. 1:1 and 1:N answer
            # different questions and carry different weight, and the
            # distinction is invisible once both are printed as "score 0.61".
            # See _MODE_LABELS for why "single" is not the 1:1 case.
            "mode": run.mode,
            "mode_label": mode_label(run.mode),
            "operator": operator.email if operator else run.operator_id,
            "lawful_basis": run.lawful_basis,
            "purpose": run.purpose,
            "probe_sha256": run.probe_sha256,
            # Storage-relative, content-addressed, and readable only by a
            # principal already authorised for this case. It reveals nothing the
            # SHA-256 beside it does not, and the PDF renderer needs it to place
            # the probe image on the page.
            "probe_path": run.probe_path,
            "decision": run.decision,
            "explanation": run.explanation,
            "top_score": run.top_score,
            "margin": run.margin,
            "gallery_size": run.gallery_size,
            "probe_quality": run.quality_score,
            "probe_liveness": run.liveness_score,
            # The liveness score is a passive single-frame HEURISTIC, not
            # presentation-attack detection. Emitting the bare number invited a
            # reader to treat it as a spoof determination, so the qualifier
            # travels with it into the report exactly as it does in the API
            # response (nexgen_engine/security/liveness.py sets the same fields).
            # Do not remove this without removing the score too.
            "probe_liveness_method": "passive_single_frame_heuristic",
            "probe_liveness_certified": False,
            "probe_liveness_caveat": (
                "Heuristic image-quality signal, not certified presentation-attack "
                "detection. It does not establish that a live person was present "
                "and will not reliably detect a printed photo or replayed screen."
            ),
            "review_required": run.review_required,
            # Recorded per search: a run made while the stub was loaded carries
            # no evidential weight, and that must stay visible in the report.
            "recognition_capable": run.recognition_capable,
            "reasons": json.loads(run.reasons) if run.reasons else [],
            "model": {"backend": run.recognizer_backend, "pack": run.recognizer_pack},
            "thresholds": {"match": run.match_threshold, "review": run.review_threshold},
            "audit_hash": run.audit_hash,
            "candidates": [self._candidate_section(session, candidate) for candidate in candidates],
        }

    def _candidate_section(self, session: Session, candidate: Candidate) -> dict[str, Any]:
        subject = session.get(Subject, candidate.subject_id)
        examiner = session.get(User, candidate.adjudicated_by) if candidate.adjudicated_by else None
        # The enrolment image behind the template, so the report can show the
        # examiner the exact image the score was computed against rather than
        # some other photograph of the same subject.
        template = session.get(Template, candidate.template_id)
        return {
            "rank": candidate.rank,
            "subject_id": candidate.subject_id,
            "subject_name": subject.display_name if subject else "(subject deleted)",
            "external_ref": subject.external_ref if subject else "",
            "template_id": candidate.template_id,
            "template_image_sha256": template.image_sha256 if template else "",
            "template_image_path": template.image_path if template else "",
            "similarity": candidate.score,
            "normalized_score": candidate.normalized_score,
            "adjudication": candidate.adjudication.value,
            "adjudicated_by": examiner.email if examiner else None,
            "adjudicated_at": candidate.adjudicated_at.isoformat() if candidate.adjudicated_at else None,
            "examiner_notes": candidate.examiner_notes,
        }

    # -- deterministic sections -------------------------------------------
    #
    # Methodology, limitations and conclusion are assembled here, from the
    # record, by code. They are the three sections a language model must not
    # write: a generated methodology can misdescribe the system, a generated
    # limitation can quietly soften a caveat, and a generated conclusion would
    # be the model making the decision. See services/narrative_service.py.
    #
    # These render identically whether or not the narrative layer is configured,
    # reachable, or enabled, so a report is never blocked on a third party.

    def _standard_sections(self, case: Case, searches: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "methodology": self._methodology(searches),
            "limitations": self._limitations(case, searches),
            "conclusion": self._conclusion(searches),
        }

    def _methodology(self, searches: list[dict[str, Any]]) -> list[str]:
        if not searches:
            return ["No searches were run against this case, so no comparison method was applied."]

        packs = sorted({f"{s['model']['pack']} ({s['model']['backend']})" for s in searches if s.get("model")})
        match_thresholds = sorted({s["thresholds"]["match"] for s in searches if s.get("thresholds")})
        review_thresholds = sorted({s["thresholds"]["review"] for s in searches if s.get("thresholds")})
        verifications = [s for s in searches if s.get("mode") == VERIFICATION_MODE]
        identifications = [s for s in searches if s.get("mode") != VERIFICATION_MODE]
        galleries = sorted({s.get("gallery_size", 0) for s in identifications})

        paragraphs = [
            "Each submitted image was processed by an automated face recognition pipeline: "
            "face detection, geometric alignment to the recognition model's reference "
            "landmark layout, and conversion to a fixed-length numerical template. "
            "Comparison is performed between templates, not between images.",
            "Recognition model: " + ", ".join(packs) + ". Templates are compared by cosine "
            "similarity, which yields a figure expressing likeness between two images. It is "
            "not a probability, and it is specific to this model -- a score from one model "
            "cannot be compared against a score from another.",
        ]

        if match_thresholds:
            paragraphs.append(
                "Decision thresholds applied: match at "
                + ", ".join(f"{t:.4f}" for t in match_thresholds)
                + "; review at "
                + ", ".join(f"{t:.4f}" for t in review_thresholds)
                + ". These are operating points selected by benchmark measurement on this "
                "model pack, not natural boundaries. A score marginally above a threshold "
                "and one far above it are both reported as a match while differing "
                "substantially in weight."
            )

        if identifications:
            paragraphs.append(
                f"{len(identifications)} search(es) were run in identification mode (1:N), "
                "comparing the probe against every enrolled template and returning a ranked "
                "candidate list"
                + (f" from a gallery of {', '.join(str(g) for g in galleries)} template(s)." if galleries else ".")
                + " Ranking is relative to the enrolled population only. A rank-1 result is "
                "the closest template in this gallery, which does not establish that the "
                "person sought is enrolled in it at all."
            )
        if verifications:
            paragraphs.append(
                f"{len(verifications)} comparison(s) were run in verification mode (1:1), "
                "comparing two nominated images directly against the verification threshold "
                "without reference to a gallery."
            )

        paragraphs.append(
            "Every search, adjudication and export is written to a hash-chained audit log at "
            "the time it occurs. The chain is reproduced in this report and can be verified "
            "independently."
        )
        return paragraphs

    def _limitations(self, case: Case, searches: list[dict[str, Any]]) -> list[str]:
        """Fixed caveats first, then conditionals driven by what actually happened.

        Assembled by code precisely so that nothing here can be omitted or
        softened during report generation. Items are appended, never removed.
        """
        limitations = [
            "Automated face recognition produces investigative leads, not identifications. "
            "No entry in this report may be relied upon as an identification unless a "
            "qualified examiner has independently verified it and that verification is "
            "recorded here as an adjudication.",
            "A similarity score expresses likeness between two images. It is not the "
            "probability that the images show the same person, and it must not be reported, "
            "converted, or paraphrased as a percentage confidence.",
            "Candidate ranking is relative to the enrolled gallery only. The absence of a "
            "high-scoring candidate does not establish that the person sought does not "
            "exist; it establishes that no close template is enrolled.",
            "Recognition accuracy degrades substantially on low-resolution, off-angle, "
            "poorly lit and compressed imagery. Surveillance-grade footage is materially "
            "harder than the benchmark conditions under which thresholds were set.",
        ]

        incapable = [s for s in searches if not s.get("recognition_capable", True)]
        if incapable:
            limitations.append(
                f"WARNING: {len(incapable)} of {len(searches)} search(es) in this report ran "
                "without a recognition model loaded. Those results carry no evidential "
                "weight whatsoever and must be disregarded."
            )

        if any(s.get("probe_liveness") is not None for s in searches):
            limitations.append(
                "Liveness and synthetic-media figures quoted in this report are passive "
                "single-frame heuristics, not certified presentation-attack detection. They "
                "have not been evaluated against ISO/IEC 30107-3. A pass means nothing "
                "obvious was wrong; it does not establish that a live person was present, "
                "and it will not reliably detect a printed photograph or a replayed screen."
            )

        pending = [c for s in searches for c in s["candidates"] if c["adjudication"] == Adjudication.PENDING.value]
        if pending:
            limitations.append(
                f"{len(pending)} candidate(s) in this report have not yet been adjudicated by "
                "an examiner. They are machine output only and no weight should be placed on "
                "them in their current state."
            )

        if not case.lawful_basis:
            limitations.append(
                "No lawful basis is recorded against this case. The processing documented "
                "here should not be relied upon until that omission is resolved."
            )

        low_quality = [s for s in searches if (s.get("probe_quality") or 1.0) < 0.5]
        if low_quality:
            limitations.append(
                f"{len(low_quality)} probe image(s) scored below 0.5 on the automated quality "
                "assessment. Scores derived from low-quality probes are less reliable in both "
                "directions -- they raise the chance of a missed match and of a spurious one."
            )
        return limitations

    def _conclusion(self, searches: list[dict[str, Any]]) -> list[str]:
        """The decision, restated from the record. Never generated.

        This section exists to state what the system reported and what remains
        outstanding. It draws no inference the record does not already contain.
        """
        if not searches:
            return ["No searches have been run against this case. There is nothing to conclude."]

        confirmed = [c for s in searches for c in s["candidates"] if c["adjudication"] == Adjudication.CONFIRMED.value]
        eliminated = [c for s in searches for c in s["candidates"] if c["adjudication"] == Adjudication.ELIMINATED.value]
        pending = [c for s in searches for c in s["candidates"] if c["adjudication"] == Adjudication.PENDING.value]
        incapable = [s for s in searches if not s.get("recognition_capable", True)]

        lines = [
            f"{len(searches)} search(es) were run against this case, returning "
            f"{sum(len(s['candidates']) for s in searches)} candidate(s) above the reporting floor."
        ]

        if incapable:
            lines.append(
                f"{len(incapable)} search(es) ran without a recognition model loaded and are "
                "excluded from any conclusion drawn here."
            )

        if confirmed:
            lines.append(
                f"An examiner has confirmed {len(confirmed)} candidate(s) as a match on their "
                "own professional judgement, recorded in this report with the examiner's "
                "identity and the time of that judgement. That adjudication, not the "
                "similarity score, is the finding."
            )
        else:
            lines.append(
                "No candidate in this report has been confirmed as a match by an examiner."
            )

        if eliminated:
            lines.append(f"{len(eliminated)} candidate(s) were examined and eliminated.")
        if pending:
            lines.append(
                f"{len(pending)} candidate(s) remain awaiting adjudication. This report is "
                "not complete in respect of those candidates."
            )

        lines.append(
            "This report records the output of an automated system and the adjudications "
            "made upon it. It is not an expert conclusion until an examiner has reviewed the "
            "imagery and signed below."
        )
        return lines

    def to_markdown(self, report: dict[str, Any]) -> str:
        case = report["case"]
        summary = report["summary"]
        lines = [
            f"# Case Report: {case['reference']}",
            "",
            f"**{case['title']}**",
            "",
            "> " + report["notice"].replace("\n", " "),
            "",
            "## Case details",
            "",
            f"- Status: {case['status']}",
            f"- Opened: {case['opened_at']}",
            f"- Lawful basis: {case['lawful_basis'] or '(not recorded)'}",
            f"- Report generated: {report['generated_at']} by {report['generated_by']}",
            "",
            "## Summary",
            "",
            f"- Searches run: {summary['searches_run']}",
            f"- Candidates returned: {summary['candidates_returned']}",
            f"- Confirmed by examiner: {summary['confirmed_by_examiner']}",
            f"- Awaiting adjudication: {summary['awaiting_adjudication']}",
            f"- Searches returning nothing: {summary['searches_with_no_result']}",
            "",
        ]

        narrative = report.get("narrative") or {}
        sections = report.get("standard_sections") or {}

        if narrative.get("available"):
            lines += ["> " + narrative["disclosure"], ""]
        elif narrative.get("withheld_notice"):
            lines += ["> " + narrative["withheld_notice"], ""]

        if narrative.get("sections", {}).get("executive_summary"):
            lines += ["## Executive summary", "", narrative["sections"]["executive_summary"], ""]
        if sections.get("methodology"):
            lines += ["## Methodology", ""]
            lines += [paragraph + "\n" for paragraph in sections["methodology"]]

        lines += ["## Searches", ""]

        for search in report["searches"]:
            lines.append(f"### Search {search['search_id'][:8]} - {search['performed_at']}")
            lines.append("")
            lines.append(f"- Operator: {search['operator']}")
            lines.append(f"- Lawful basis: {search['lawful_basis'] or '(not recorded)'}")
            lines.append(f"- Probe SHA-256: `{search['probe_sha256']}`")
            lines.append(f"- Decision: **{search['decision']}**")
            lines.append(f"- {search['explanation']}")
            lines.append(f"- Gallery searched: {search['gallery_size']} templates")
            lines.append(f"- Model: {search['model']['backend']} ({search['model']['pack']})")
            if not search["recognition_capable"]:
                lines.append(
                    "- **WARNING: this search ran without a recognition model loaded. "
                    "Its results carry no evidential weight.**"
                )
            lines.append("")

            if search["candidates"]:
                lines.append("| Rank | Subject | Similarity | Adjudication | Examiner |")
                lines.append("| ---: | --- | ---: | --- | --- |")
                for candidate in search["candidates"]:
                    lines.append(
                        f"| {candidate['rank']} | {candidate['subject_name']} "
                        f"| {candidate['similarity']:.4f} | {candidate['adjudication']} "
                        f"| {candidate['adjudicated_by'] or '-'} |"
                    )
            else:
                lines.append("_No candidates were returned._")
            lines.append("")

        if narrative.get("sections", {}).get("findings"):
            lines += ["## Findings", "", narrative["sections"]["findings"], ""]
        if narrative.get("sections", {}).get("similarity_explanation"):
            lines += ["## Similarity explanation", "", narrative["sections"]["similarity_explanation"], ""]

        if sections.get("limitations"):
            lines += ["## Limitations", ""]
            lines += [f"- {item}" for item in sections["limitations"]]
            lines.append("")
        if sections.get("conclusion"):
            lines += ["## Conclusion", ""]
            lines += [paragraph + "\n" for paragraph in sections["conclusion"]]

        lines.extend(["## Audit trail", "", "| Timestamp | Actor | Action | Outcome |", "| --- | --- | --- | --- |"])
        for entry in report["audit_trail"]:
            lines.append(f"| {entry['timestamp']} | {entry['actor']} | {entry['action']} | {entry['outcome']} |")
        lines.append("")

        return "\n".join(lines)


__all__ = ["REPORT_NOTICE", "VERIFICATION_MODE", "ReportService", "mode_label"]
