from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlmodel import Session, select

from ..db.models import Adjudication, AuditRecord, Candidate, Case, SearchRun, Subject, User

# Repeated verbatim in every export. A report that leaves this out invites the
# reader to treat a similarity score as an identification.
REPORT_NOTICE = (
    "This report documents automated facial recognition searches. Automated face recognition "
    "produces investigative leads, not identifications. Candidate scores express similarity "
    "between images, not the probability that two images show the same person. No entry in this "
    "report may be relied upon as an identification unless a qualified examiner has independently "
    "verified it, and that verification is recorded here as an examiner adjudication."
)


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
            .order_by(AuditRecord.created_at)
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
            "operator": operator.email if operator else run.operator_id,
            "lawful_basis": run.lawful_basis,
            "purpose": run.purpose,
            "probe_sha256": run.probe_sha256,
            "decision": run.decision,
            "explanation": run.explanation,
            "top_score": run.top_score,
            "margin": run.margin,
            "gallery_size": run.gallery_size,
            "probe_quality": run.quality_score,
            "probe_liveness": run.liveness_score,
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
        return {
            "rank": candidate.rank,
            "subject_id": candidate.subject_id,
            "subject_name": subject.display_name if subject else "(subject deleted)",
            "external_ref": subject.external_ref if subject else "",
            "similarity": candidate.score,
            "normalized_score": candidate.normalized_score,
            "adjudication": candidate.adjudication.value,
            "adjudicated_by": examiner.email if examiner else None,
            "adjudicated_at": candidate.adjudicated_at.isoformat() if candidate.adjudicated_at else None,
            "examiner_notes": candidate.examiner_notes,
        }

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
            "## Searches",
            "",
        ]

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

        lines.extend(["## Audit trail", "", "| Timestamp | Actor | Action | Outcome |", "| --- | --- | --- | --- |"])
        for entry in report["audit_trail"]:
            lines.append(f"| {entry['timestamp']} | {entry['actor']} | {entry['action']} | {entry['outcome']} |")
        lines.append("")

        return "\n".join(lines)


__all__ = ["REPORT_NOTICE", "ReportService"]
