"""One live call to the narrative provider, with everything shown.

Run this before relying on the narrative layer in casework. It answers three
questions the unit tests deliberately cannot, because they stub the provider:

  1. Does the configuration actually resolve? (enabled, key present, model)
  2. What EXACTLY leaves this machine? The full outbound payload is printed
     before the call, so the pseudonymisation can be inspected rather than
     trusted.
  3. Does the provider's response shape match what NarrativeService parses,
     and does the returned prose survive the validators?

The findings below are synthetic. No case data is read, and nothing is written
to the database -- the session is in-memory and discarded on exit.

    python scripts/verify_narrative.py

Exit status is 0 only if a narrative was generated AND passed validation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlmodel import Session, SQLModel, create_engine  # noqa: E402

from imatch_api.core.config import get_settings  # noqa: E402
from imatch_api.services.narrative_service import (  # noqa: E402
    NarrativeService,
    build_payload,
)

# Deliberately loaded with identifiers, so the printed payload demonstrates that
# they are stripped rather than merely absent.
SYNTHETIC_REPORT = {
    "generated_at": "2026-08-03T09:15:00+00:00",
    "generated_by": "dc.hargreaves@example.police.uk",
    "case": {
        "id": "case-smoke",
        "reference": "OP-NIGHTJAR-2291",
        "title": "Operation Nightjar",
        "status": "open",
        "lawful_basis": "PACE s.64A, prevention and detection of crime",
    },
    "summary": {
        "searches_run": 1,
        "candidates_returned": 2,
        "confirmed_by_examiner": 0,
        "awaiting_adjudication": 2,
        "searches_with_no_result": 0,
    },
    "searches": [{
        "search_id": "run-smoke",
        "performed_at": "2026-08-02T14:30:00+00:00",
        "mode": "single",
        "mode_label": "Identification (1:N, enrolled gallery)",
        "operator": "dc.hargreaves@example.police.uk",
        "lawful_basis": "PACE s.64A, prevention and detection of crime",
        "probe_sha256": "a" * 64,
        "decision": "review",
        "top_score": 0.7421,
        "margin": 0.0912,
        "gallery_size": 4820,
        "probe_quality": 0.61,
        "probe_liveness": 0.88,
        "probe_liveness_certified": False,
        "review_required": True,
        "recognition_capable": True,
        "reasons": [],
        "model": {"backend": "insightface", "pack": "buffalo_l"},
        "thresholds": {"match": 0.20, "review": 0.15},
        "candidates": [
            {"rank": 1, "subject_name": "Priya Raghunathan", "external_ref": "PNC-9931",
             "similarity": 0.7421, "normalized_score": 0.8110, "adjudication": "pending",
             "adjudicated_at": None},
            {"rank": 2, "subject_name": "Tomasz Wieczorek", "external_ref": "PNC-4417",
             "similarity": 0.6509, "normalized_score": 0.7002, "adjudication": "pending",
             "adjudicated_at": None},
        ],
    }],
    "audit_trail": [],
}

IDENTIFIERS = (
    "Priya Raghunathan", "Tomasz Wieczorek", "dc.hargreaves@example.police.uk",
    "OP-NIGHTJAR-2291", "Operation Nightjar", "PNC-9931", "PACE",
)


def main() -> int:
    settings = get_settings()

    print("=" * 72)
    print("CONFIGURATION")
    print("=" * 72)
    print(f"  narrative_enabled : {settings.narrative_enabled}")
    # The key itself is never printed, here or anywhere else.
    print(f"  GEMINI_API_KEY    : {'set' if settings.gemini_api_key else 'NOT SET'}")
    print(f"  model             : {settings.gemini_model}")
    print(f"  timeout           : {settings.gemini_timeout_seconds}s")
    print(f"  max attempts      : {settings.narrative_max_attempts}")

    service = NarrativeService(settings)
    if not service.enabled:
        print(
            "\nNarrative generation is not enabled. Set NEXGEN_NARRATIVE_ENABLED=true "
            "and GEMINI_API_KEY in .env, then run this again."
        )
        return 1

    payload = build_payload(SYNTHETIC_REPORT)

    print("\n" + "=" * 72)
    print("OUTBOUND PAYLOAD -- this, and nothing else, is sent to the provider")
    print("=" * 72)
    print(json.dumps(payload, indent=2, sort_keys=True))

    serialised = json.dumps(payload)
    leaked = [item for item in IDENTIFIERS if item in serialised]
    print("\n  identifier check  : ", end="")
    print("FAILED -- " + ", ".join(leaked) if leaked else "clean, no identifiers present")
    if leaked:
        return 1

    print("\n" + "=" * 72)
    print(f"CALLING {settings.gemini_model} ...")
    print("=" * 72)

    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        result = service.attach(
            session, dict(SYNTHETIC_REPORT),
            tenant_id="smoke", case_id="case-smoke", generated_by="verify_narrative.py",
        )

    print(f"  available         : {result.available}")
    print(f"  attempts          : {result.attempts}")
    print(f"  validator         : {result.validator_status}")
    if result.reason:
        print(f"  reason            : {result.reason}")
    if result.validator_notes:
        print("\n  VALIDATOR COMPLAINTS:")
        for note in result.validator_notes.split(" | "):
            print(f"    - {note}")

    if not result.available:
        print(
            "\nNo narrative was produced. The report would still export in full, "
            "with its factual sections intact and the reason printed on the page."
        )
        return 1

    print("\n" + "=" * 72)
    print("GENERATED SECTIONS")
    print("=" * 72)
    for name, text in result.sections.items():
        print(f"\n--- {name} ---\n{text}")

    print("\nPassed. The provider response parsed and the prose cleared every validator.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
