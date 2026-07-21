"""
Validates backend report dicts against comparison_report.schema.json.
Every pipeline output MUST pass through validate_report() before it is
handed to the visualization or report-generation layers.
"""
import json
from pathlib import Path
from jsonschema import Draft202012Validator

_SCHEMA_PATH = Path(__file__).parent / "comparison_report.schema.json"


def load_schema() -> dict:
    with open(_SCHEMA_PATH) as f:
        return json.load(f)


def validate_report(report: dict) -> list[str]:
    """Returns a list of validation error strings. Empty list == valid."""
    schema = load_schema()
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(report), key=lambda e: e.path)
    return [f"{'/'.join(str(p) for p in e.path)}: {e.message}" for e in errors]


def assert_valid_report(report: dict) -> None:
    errors = validate_report(report)
    if errors:
        raise ValueError(
            "Report JSON failed schema validation:\n" + "\n".join(errors)
        )
