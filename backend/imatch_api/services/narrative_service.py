"""Model-drafted prose for the case report, and the guards that make it safe.

WHAT THIS LAYER IS. The recognition engine has already decided everything. This
takes the findings it produced and asks a language model to write three
DESCRIPTIVE sections around them: an executive summary, a findings narrative,
and an explanation of what the similarity figures mean. That is the whole job.

WHAT THIS LAYER IS NOT. It does not decide, re-rank, re-score, or observe. Three
sections a reader might expect to find here are deliberately absent:

  Methodology  - a statement of fact about which model, thresholds and gallery
                 actually ran. It is already fully determined by the record. A
                 model paraphrasing it can only introduce error, and error about
                 our own system in a legal document is the worst kind.
  Limitations  - a generated limitation can be a SOFTENED limitation, and that
                 failure is silent. The caveats are fixed text plus conditionals
                 driven by the data.
  Conclusion   - the conclusion IS the decision. Generating it would be the
                 model deciding, which is the one thing it must not do.

All three are rendered from templates in ReportService.

THREE GUARDS, IN ORDER OF IMPORTANCE

1. NUMERIC WHITELIST. Every numeral in the generated text must already appear in
   the evidence payload. This blocks the specific failure that matters most: a
   cosine similarity of 0.62 written up as "62% confidence". Those are unrelated
   quantities, the conversion is meaningless, and it is exactly the sentence a
   reader would quote.

2. APPEARANCE VOCABULARY. The model is sent numbers and never sees an image, so
   ANY sentence describing a nose, jawline, or hairline is fabricated by
   construction. Rejecting the vocabulary outright is a cheap, complete guard
   against invented observation -- there is no legitimate use of these words on
   this code path.

3. ASSERTION PHRASES. Language that converts a similarity score into an
   identification is rejected regardless of how it is hedged.

A failed generation is retried once with the complaint fed back, then abandoned.
The report renders without the narrative and says so. A report missing a section
is recoverable; a report containing an unchecked generated claim is not.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import requests
from sqlmodel import Session, select

from ..core.config import Settings
from ..db.models import ReportNarrative

logger = logging.getLogger(__name__)

_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

#: The only sections a model is permitted to author. See the module docstring
#: for why methodology, limitations and conclusion are not on this list.
NARRATIVE_SECTIONS: tuple[str, ...] = (
    "executive_summary",
    "findings",
    "similarity_explanation",
)

#: Printed verbatim in the report. A reader is entitled to know which paragraphs
#: were drafted by a model before deciding how much weight to give them.
DISCLOSURE = (
    "The Executive Summary, Findings and Similarity Explanation sections below were "
    "drafted by an automated language model ({model}) from the structured findings in "
    "this report, and checked against them. The model performed no facial comparison, "
    "saw no imagery, and made no decision. Every figure it cites is reproduced from the "
    "recorded results. Methodology, Limitations and Conclusion are generated from the "
    "record by fixed templates, not by the model. The examiner signing this report "
    "adopts the wording as their own."
)

WITHHELD_NOTICE = (
    "Narrative sections were not included in this report. The findings, scores, "
    "thresholds and audit trail below are unaffected -- they are read directly from "
    "the record and never depend on the narrative layer."
)

# Words describing physical appearance. The model receives no imagery, so any of
# these indicates invented observation rather than explanation.
_APPEARANCE_VOCABULARY: tuple[str, ...] = (
    "nose", "nasal", "jaw", "jawline", "eyebrow", "brow", "eyelid", "lip", "lips",
    "chin", "cheek", "cheekbone", "forehead", "hairline", "earlobe", "mole",
    "scar", "freckle", "wrinkle", "dimple", "complexion", "skin tone",
    "eye shape", "facial hair", "beard", "moustache", "mustache", "philtrum",
    "visual inspection", "we observed", "i observed", "visibly", "appears to show",
)

# Language that converts a similarity figure into an identification.
_ASSERTION_PHRASES: tuple[str, ...] = (
    "same person", "same individual", "is the suspect", "proves", "proven",
    "conclusive", "conclusively", "definitively", "definitely", "beyond doubt",
    "confirms the identity", "positive identification", "identified as",
    "we identify", "guilty", "perpetrator",
)

_NUMBER_TOKEN = re.compile(r"\d+(?:\.\d+)?")


class NarrativeUnavailable(RuntimeError):
    """Generation could not complete. Never raised to the caller of attach()."""


@dataclass(frozen=True)
class NarrativeResult:
    available: bool
    sections: dict[str, str]
    model: str
    reason: str = ""
    validator_status: str = "passed"
    validator_notes: str = ""
    attempts: int = 0
    reused: bool = False
    evidence_digest: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "sections": self.sections,
            "disclosure": DISCLOSURE.format(model=self.model) if self.available else "",
            "withheld_notice": "" if self.available else WITHHELD_NOTICE,
            "model": self.model,
            "reason": self.reason,
            "validator_status": self.validator_status,
            "validator_notes": self.validator_notes,
            "attempts": self.attempts,
            "reused": self.reused,
            "evidence_digest": self.evidence_digest,
        }


# --------------------------------------------------------------------- payload


def build_payload(report: dict[str, Any]) -> dict[str, Any]:
    """Project the report into the pseudonymised evidence the model may see.

    Two rules govern what crosses this boundary.

    IDENTIFIERS ARE STRIPPED. Subject names, external references, operator and
    examiner e-mail addresses, the case reference and title, and the free-text
    lawful basis are all removed. Candidates become "Candidate 1", "Candidate 2".
    The model needs the shape of the result to write about it; it does not need
    to know who anyone is, and sending case data to a third party is a decision
    the deployment makes once, in config, not a decision this function makes per
    field.

    HASHES ARE TRUNCATED. A full SHA-256 is an identifier for the exact image
    file. Twelve characters are enough for the narrative to refer to an exhibit;
    the full digest is printed in the report from the record, not from here.
    """
    searches = []
    for index, search in enumerate(report.get("searches", []) or [], 1):
        candidates = [
            {
                "label": f"Candidate {candidate.get('rank', position)}",
                "rank": candidate.get("rank"),
                "similarity": candidate.get("similarity"),
                "normalized_score": candidate.get("normalized_score"),
                "adjudication": candidate.get("adjudication"),
                "adjudicated": bool(candidate.get("adjudicated_at")),
            }
            for position, candidate in enumerate(search.get("candidates", []) or [], 1)
        ]
        searches.append({
            "search_number": index,
            "mode": search.get("mode", "identification"),
            "decision": search.get("decision"),
            "top_score": search.get("top_score"),
            "margin": search.get("margin"),
            "gallery_size": search.get("gallery_size"),
            "thresholds": search.get("thresholds"),
            "model": search.get("model"),
            "probe_quality": search.get("probe_quality"),
            "probe_liveness": search.get("probe_liveness"),
            "probe_liveness_certified": search.get("probe_liveness_certified"),
            "recognition_capable": search.get("recognition_capable"),
            "review_required": search.get("review_required"),
            "reasons": search.get("reasons"),
            "probe_sha256_short": (search.get("probe_sha256") or "")[:12],
            "candidate_count": len(candidates),
            "candidates": candidates,
        })

    return {
        "summary": report.get("summary", {}),
        "case_status": (report.get("case", {}) or {}).get("status"),
        "lawful_basis_recorded": bool((report.get("case", {}) or {}).get("lawful_basis")),
        "searches": searches,
        "audit_entry_count": len(report.get("audit_trail", []) or []),
    }


def payload_digest(payload: dict[str, Any]) -> str:
    """Cache key and invalidation rule in one value. See ReportNarrative."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------- validator


def _allowed_numerals(payload: dict[str, Any]) -> set[str]:
    """Every numeral the model is permitted to write.

    Numeric values contribute their whole formatted forms only -- NOT their
    digit runs. That distinction is the percentage guard: a similarity of 0.20
    permits "0.2" and "0.20" but never a bare "20", so "20% confidence" is
    rejected. Strings contribute their digit runs, which is what makes
    timestamps and truncated hashes quotable.
    """
    allowed: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, bool) or node is None:
            return
        if isinstance(node, (int, float)):
            value = float(node)
            allowed.add(str(node))
            for places in (0, 1, 2, 3, 4):
                allowed.add(f"{value:.{places}f}")
            if value.is_integer():
                allowed.add(str(int(value)))
            # 0.2000 and 0.2 are the same number; accept the trimmed form too.
            allowed.add(f"{value:.4f}".rstrip("0").rstrip("."))
            return
        if isinstance(node, str):
            allowed.update(_NUMBER_TOKEN.findall(node))
            return
        if isinstance(node, dict):
            for item in node.values():
                walk(item)
            return
        if isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    walk(payload)
    allowed.discard("")
    return allowed


def validate_sections(sections: dict[str, str], payload: dict[str, Any]) -> list[str]:
    """Return the complaints. Empty list means the text may be printed."""
    complaints: list[str] = []
    allowed = _allowed_numerals(payload)

    for name in NARRATIVE_SECTIONS:
        text = (sections.get(name) or "").strip()
        if not text:
            complaints.append(f"Section '{name}' is empty.")
            continue

        lowered = text.lower()

        invented = sorted({token for token in _NUMBER_TOKEN.findall(text) if token not in allowed})
        if invented:
            complaints.append(
                f"Section '{name}' contains figures that do not appear in the supplied "
                f"findings: {', '.join(invented)}. Reproduce figures exactly as given and "
                f"never convert a similarity score into a percentage."
            )

        appearance = sorted({w for w in _APPEARANCE_VOCABULARY if re.search(rf"\b{re.escape(w)}\b", lowered)})
        if appearance:
            complaints.append(
                f"Section '{name}' describes physical appearance ({', '.join(appearance)}). "
                f"No imagery was supplied, so such a description is unsupported. Write only "
                f"about the supplied figures and their meaning."
            )

        assertions = sorted({p for p in _ASSERTION_PHRASES if p in lowered})
        if assertions:
            complaints.append(
                f"Section '{name}' asserts an identification ({', '.join(assertions)}). "
                f"Similarity is not identity. Describe what the system reported and what it "
                f"does and does not establish."
            )

    return complaints


# --------------------------------------------------------------------- service


_SYSTEM_RULES = """You are drafting three descriptive sections of a forensic face
comparison report for a UK-style investigative audience. You are an explanation
layer only.

WHAT YOU ARE GIVEN
A JSON object of findings already produced by a face recognition system. It is
the complete and only source of fact available to you.

ABSOLUTE RULES
1. Do not make, revise, or imply a match decision. The decision is already in
   the data; report it as the system's output, never as your own.
2. Do not invent evidence. You have not seen any image. Never describe a face,
   a facial feature, or anything about how the people look. There is no
   photograph in your input.
3. Do not introduce any numeral that is not already in the input. Reproduce
   figures exactly as given.
4. Never express a similarity score as a percentage, probability, or confidence
   level. A cosine similarity is not a probability and the conversion is
   meaningless.
5. Similarity is not identity. Never state or imply that two images show the
   same person.
6. If the input says recognition_capable is false, say plainly that the search
   ran without a recognition model loaded and carries no evidential weight.
   This outranks everything else in the summary.
7. Write plain professional prose. No markdown, no headings, no bullet lists.

THE SECTIONS
executive_summary: 100-150 words. What was searched, what came back, what state
  the case is in. Neutral register.
findings: 150-250 words. Walk through each search: the decision, the top score
  against the threshold that judged it, the gallery size, the ranking, and the
  examiner adjudication status. Say explicitly where a candidate is still
  awaiting adjudication.
similarity_explanation: 120-200 words. Explain what the similarity figures mean
  and what they do not. Cover: the score measures likeness between images, not
  probability of identity; the threshold is an operating point chosen by
  measurement, not a natural boundary; a score near the threshold is weaker
  evidence than one far above it; and the ranking is relative to this gallery
  only, so the true subject may simply be absent from it.
"""

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {name: {"type": "string"} for name in NARRATIVE_SECTIONS},
    "required": list(NARRATIVE_SECTIONS),
}


class NarrativeService:
    """Generates once, persists, and reuses. Never raises into the report path."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def enabled(self) -> bool:
        return bool(self.settings.narrative_enabled and self.settings.gemini_api_key)

    def attach(
        self,
        session: Session,
        report: dict[str, Any],
        *,
        tenant_id: str,
        case_id: str,
        generated_by: str,
    ) -> NarrativeResult:
        """Add ``report["narrative"]`` and return what happened, for the audit log.

        Every failure path here is non-fatal by design. A report that cannot be
        exported because a third-party API is slow would be a worse outcome than
        a report without a summary paragraph.
        """
        if not self.enabled:
            reason = (
                "Narrative generation is disabled for this deployment."
                if not self.settings.narrative_enabled
                else "Narrative generation is enabled but no GEMINI_API_KEY is configured."
            )
            result = NarrativeResult(False, {}, self.settings.gemini_model, reason=reason)
            report["narrative"] = result.as_dict()
            return result

        payload = build_payload(report)
        digest = payload_digest(payload)

        existing = session.exec(
            select(ReportNarrative)
            .where(
                ReportNarrative.tenant_id == tenant_id,
                ReportNarrative.case_id == case_id,
                ReportNarrative.evidence_digest == digest,
                ReportNarrative.validator_status == "passed",
            )
            .order_by(ReportNarrative.generated_at)
        ).first()
        if existing is not None:
            result = NarrativeResult(
                True, json.loads(existing.sections_json), existing.model,
                attempts=existing.attempts, reused=True, evidence_digest=digest,
            )
            report["narrative"] = result.as_dict()
            return result

        try:
            sections, attempts, complaints = self._generate(payload)
        except NarrativeUnavailable as error:
            result = NarrativeResult(
                False, {}, self.settings.gemini_model,
                reason=str(error), validator_status="not_generated", evidence_digest=digest,
            )
            report["narrative"] = result.as_dict()
            return result

        passed = not complaints
        record = ReportNarrative(
            tenant_id=tenant_id,
            case_id=case_id,
            evidence_digest=digest,
            sections_json=json.dumps(sections, sort_keys=True),
            model=self.settings.gemini_model,
            prompt_sha256=hashlib.sha256(_SYSTEM_RULES.encode("utf-8")).hexdigest(),
            validator_status="passed" if passed else "rejected",
            validator_notes=" | ".join(complaints),
            attempts=attempts,
            generated_by=generated_by,
        )
        # Rejected generations are kept too. "The model was asked and its answer
        # was refused" is a fact a reviewer may need, and discarding it would
        # make a suppressed narrative indistinguishable from one never attempted.
        session.add(record)

        result = NarrativeResult(
            available=passed,
            sections=sections if passed else {},
            model=self.settings.gemini_model,
            reason="" if passed else "Generated text failed validation and was withheld.",
            validator_status=record.validator_status,
            validator_notes=record.validator_notes,
            attempts=attempts,
            evidence_digest=digest,
        )
        report["narrative"] = result.as_dict()
        return result

    # -- generation -------------------------------------------------------

    def _generate(self, payload: dict[str, Any]) -> tuple[dict[str, str], int, list[str]]:
        complaints: list[str] = []
        sections: dict[str, str] = {}
        attempts = max(1, int(self.settings.narrative_max_attempts))

        for attempt in range(1, attempts + 1):
            prompt = self._prompt(payload, complaints)
            raw = self._call(prompt)
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as error:
                raise NarrativeUnavailable(f"Model returned unparseable output: {error}") from error

            sections = {name: str(parsed.get(name, "")).strip() for name in NARRATIVE_SECTIONS}
            complaints = validate_sections(sections, payload)
            if not complaints:
                return sections, attempt, []
            logger.warning("Narrative attempt %d rejected: %s", attempt, "; ".join(complaints))

        return sections, attempts, complaints

    def _prompt(self, payload: dict[str, Any], complaints: list[str]) -> str:
        parts = [
            _SYSTEM_RULES,
            "FINDINGS (the only facts available to you):",
            json.dumps(payload, indent=2, sort_keys=True, default=str),
        ]
        if complaints:
            parts += [
                "YOUR PREVIOUS DRAFT WAS REJECTED. Fix every point and return the "
                "full set of sections again:",
                "\n".join(f"- {c}" for c in complaints),
            ]
        parts.append("Return a JSON object with exactly these keys: " + ", ".join(NARRATIVE_SECTIONS))
        return "\n\n".join(parts)

    def _call(self, prompt: str) -> str:
        url = _ENDPOINT.format(model=self.settings.gemini_model)
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                # Temperature 0 is a reproducibility aid, not a guarantee -- the
                # guarantee is that ReportNarrative is written once and reused.
                "temperature": 0,
                "candidateCount": 1,
                "responseMimeType": "application/json",
                "responseSchema": _RESPONSE_SCHEMA,
            },
        }
        try:
            response = requests.post(
                url,
                params={"key": self.settings.gemini_api_key},
                json=body,
                timeout=self.settings.gemini_timeout_seconds,
            )
        except requests.RequestException as error:
            raise NarrativeUnavailable(f"Narrative provider unreachable: {error}") from error

        if response.status_code >= 400:
            # The key is passed as a query parameter, so never echo the URL.
            raise NarrativeUnavailable(
                f"Narrative provider returned HTTP {response.status_code}."
            )

        try:
            candidates = response.json()["candidates"]
            return "".join(part.get("text", "") for part in candidates[0]["content"]["parts"])
        except (KeyError, IndexError, ValueError) as error:
            raise NarrativeUnavailable(f"Unexpected provider response shape: {error}") from error


__all__ = [
    "DISCLOSURE",
    "NARRATIVE_SECTIONS",
    "WITHHELD_NOTICE",
    "NarrativeResult",
    "NarrativeService",
    "NarrativeUnavailable",
    "build_payload",
    "payload_digest",
    "validate_sections",
]
