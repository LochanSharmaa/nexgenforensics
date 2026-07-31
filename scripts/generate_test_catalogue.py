#!/usr/bin/env python
"""
Generate A9 — Test Suite Catalogue.

    python scripts/generate_test_catalogue.py

Every automated test in the system: what it asserts, why it exists, and its
complete source.

WHY THE SOURCE IS INCLUDED RATHER THAN A DESCRIPTION
----------------------------------------------------
A catalogue that describes tests in prose can drift from what the tests
actually check, and a test that has silently stopped asserting anything still
reads as coverage. Parsing the files with `ast` and reproducing the source means
each entry is the test itself. If an assertion is weak, that is visible here.

WHAT A PASSING SUITE DOES AND DOES NOT DEMONSTRATE
--------------------------------------------------
Recorded in A2: the majority of this system's most serious defects — the ones
producing confidently wrong answers — were NOT caught by these tests. They were
caught by independent controls, by disbelieving results that looked correct,
and twice only by opening a browser. This catalogue is evidence of what is
checked automatically. It is not evidence that the system is correct.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
TEST_DIRS = [
    (_ROOT / "backend" / "tests", "API, authentication, governance and workflow"),
    (_ROOT / "backend" / "tests_engine", "Recognition engine, persistence, adversarial input"),
]


def first_sentence(doc: str | None) -> str:
    if not doc:
        return ""
    text = " ".join(doc.strip().split())
    for stop in (". ", "! ", "? "):
        if stop in text:
            return text[: text.index(stop) + 1]
    return text


def assertion_count(node: ast.AST) -> int:
    """Count every mechanism that can fail a test, not just bare `assert`.

    Counting only `ast.Assert` undercounts badly and misrepresents the suite:
    `np.testing.assert_allclose(...)` and `with pytest.raises(...)` are full
    assertions that raise on failure, and several tests here use nothing else.
    An earlier version of this script reported one module as having 11 tests and
    6 assertions, which read as though five tests checked nothing. They all
    check something.
    """
    count = 0
    for n in ast.walk(node):
        if isinstance(n, ast.Assert):
            count += 1
        elif isinstance(n, ast.Call):
            fn = n.func
            name = getattr(fn, "attr", None) or getattr(fn, "id", None) or ""
            if name.startswith("assert"):          # np.testing.*, self.assert*
                count += 1
            elif name in {"raises", "warns", "approx"}:   # pytest.raises(...)
                count += 1
    return count


def collect(path: Path) -> tuple[str, list[tuple[str, str, ast.AST]]]:
    """Return (module docstring, [(class_or_none, test_name, node)])."""
    source = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "", []
    module_doc = ast.get_docstring(tree) or ""
    found: list[tuple[str, str, ast.AST]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            found.append(("", node.name, node))
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name.startswith("test"):
                    found.append((node.name, sub.name, sub))
    return module_doc, found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_ROOT / "delivery/A9-TEST-SUITE-CATALOGUE.md"))
    args = ap.parse_args()

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_ROOT,
                            capture_output=True, text=True).stdout.strip()[:12]

    modules: list[tuple[Path, str, str, list]] = []
    for directory, label in TEST_DIRS:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("test_*.py")):
            doc, tests = collect(path)
            if tests:
                modules.append((path, label, doc, tests))

    total = sum(len(t) for _, _, _, t in modules)
    total_asserts = sum(assertion_count(n) for _, _, _, ts in modules for _, _, n in ts)
    classes = {c for _, _, _, ts in modules for c, _, _ in ts if c}

    parts: list[str] = [f"""# A9 — Test Suite Catalogue

**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} ·
**Repository state:** `{commit}`

| | |
|---|---|
| Test modules | {len(modules)} |
| Test functions | {total} |
| Test classes | {len(classes)} |
| Assertion statements | {total_asserts} |

Each entry below reproduces the test's complete source. A catalogue that
describes tests in prose can drift from what they actually check, and a test
that has silently stopped asserting anything still reads as coverage. These are
the tests themselves.

---

## What a passing suite does and does not demonstrate

Recorded in A2 Failure and Recovery Log: **the majority of this system's most
serious defects were not caught by these tests.** The silent-wrong-result and
methodological classes — the failures that produce confidently incorrect output
— were found by independent controls, by disbelieving results that looked
correct, and in two cases only by opening a browser and watching a real request
fail.

Specific examples, each of which passed the entire suite at the time:

| Defect | Why the suite could not catch it |
|---|---|
| GPU silently ran on CPU (F-01, F-02) | Results were correct, only slower. Nothing asserts on throughput. |
| Threshold drift causing a false match (F-03) | The test hard-coded the same stale constant the code used. |
| Training proxy reported a gain while the model got worse (F-04) | No test compares a checkpoint against the deployed model. |
| CSRF guard outside CORS (F-24) | `TestClient` speaks ASGI directly and never exercises CORS. |
| SameSite blocked the cookie (F-25) | No test uses a browser, and `localhost` vs `127.0.0.1` only differs in one. |

This catalogue is evidence of what is checked automatically. It is not evidence
that the system is correct.

---

## Index

| Module | Tests | Assertions |
|---|---|---|
"""]
    for path, _, _, tests in modules:
        rel = path.relative_to(_ROOT).as_posix()
        parts.append(f"| `{rel}` | {len(tests)} | "
                     f"{sum(assertion_count(n) for _, _, n in tests)} |\n")
    parts.append("\n---\n")

    current_label = None
    for path, label, module_doc, tests in modules:
        if label != current_label:
            parts.append(f"\n# {label}\n\n")
            current_label = label

        rel = path.relative_to(_ROOT).as_posix()
        source_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        parts.append(f"\n## `{rel}`\n\n")
        parts.append(f"**{len(tests)} tests · "
                     f"{sum(assertion_count(n) for _, _, n in tests)} assertions**\n\n")

        if module_doc:
            parts.append(f"### Purpose of this module, as recorded in it\n\n"
                         f"```text\n{module_doc.strip()}\n```\n\n")

        # summary table for the module
        parts.append("| Class | Test | Asserts | What it checks |\n|---|---|---|---|\n")
        for cls, name, node in tests:
            summary = first_sentence(ast.get_docstring(node)) or "*(no docstring)*"
            summary = summary.replace("|", "\\|")[:160]
            parts.append(f"| {cls or '—'} | `{name}` | {assertion_count(node)} | {summary} |\n")
        parts.append("\n")

        # full source per test
        for cls, name, node in tests:
            heading = f"{cls}.{name}" if cls else name
            parts.append(f"### `{heading}`\n\n")
            doc = ast.get_docstring(node)
            if doc:
                parts.append(f"**Rationale as recorded in the test**\n\n"
                             f"```text\n{doc.strip()}\n```\n\n")
            start = node.lineno - 1
            end = getattr(node, "end_lineno", start + 1)
            # include decorators, which carry parametrisation and markers
            for dec in getattr(node, "decorator_list", []):
                start = min(start, dec.lineno - 1)
            body = "\n".join(source_lines[start:end])
            parts.append(f"```python\n{body}\n```\n\n")

    text = "".join(parts)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    words = len(text.split())
    print(f"wrote {out}")
    print(f"  {len(modules)} modules   {total} tests   {total_asserts} assertions")
    print(f"  {len(text.splitlines()):,} lines   {words:,} words")
    print(f"  ~{words / 450:.0f} pages @450 w/p   ~{words / 250:.0f} pages @250 w/p")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
