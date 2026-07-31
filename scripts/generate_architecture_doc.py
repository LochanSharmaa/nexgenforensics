#!/usr/bin/env python
"""
Generate A3 — System Architecture.

    python scripts/generate_architecture_doc.py

Describes the delivered system: its modules, which of them are actually reached
from the running service, the HTTP surface, the persisted data model, and the
request paths through the recognition pipeline.

REACHABILITY IS COMPUTED, NOT ASSERTED
--------------------------------------
The source tree contains more modules than the service uses. Some are
instrumentation, some are earlier approaches that were superseded, and some
implement capabilities the product does not claim. An architecture document that
lists every file as though each were part of the delivered system is misleading
in a specific and damaging way: a reader concludes that a capability exists
because a file named after it is present.

This walks the import graph from the real entry points and separates modules
that are **reachable** from those that are **not**. The unreachable list is
published rather than hidden — for a maintainer it is the more useful half,
and for anyone assessing claims it is the half that matters.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
BACKEND = _ROOT / "backend"
ENTRY_POINTS = ["imatch_api.main"]
PACKAGES = ("imatch_api", "nexgen_engine")


def module_name(path: Path) -> str:
    rel = path.relative_to(BACKEND).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def all_modules() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for pkg in PACKAGES:
        for p in (BACKEND / pkg).rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            out[module_name(p)] = p
    return out


def imports_of(path: Path, own: str) -> set[str]:
    """Absolute and relative imports, resolved to module names."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return set()
    found: set[str] = set()
    pkg_parts = own.split(".")[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                found.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:                       # relative
                base = pkg_parts[: len(pkg_parts) - node.level + 1]
                target = ".".join(base + ([node.module] if node.module else []))
            else:
                target = node.module or ""
            if target:
                found.add(target)
                for a in node.names:
                    found.add(f"{target}.{a.name}")
    return found


def reachable(modules: dict[str, Path]) -> set[str]:
    seen: set[str] = set()
    queue = [e for e in ENTRY_POINTS if e in modules]
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        seen.add(name)
        for imp in imports_of(modules[name], name):
            # an import of `a.b.c` may name a module or a symbol inside `a.b`
            for candidate in (imp, imp.rsplit(".", 1)[0] if "." in imp else imp):
                if candidate in modules and candidate not in seen:
                    queue.append(candidate)
    return seen


def routes() -> list[tuple[str, str, str]]:
    """(method, path, function) from the FastAPI decorators."""
    out: list[tuple[str, str, str]] = []
    for p in sorted((BACKEND / "imatch_api" / "api" / "routes").glob("*.py")):
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        prefix = ""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "APIRouter":
                for kw in node.keywords:
                    if kw.arg == "prefix" and isinstance(kw.value, ast.Constant):
                        prefix = kw.value.value
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for dec in node.decorator_list:
                if not isinstance(dec, ast.Call):
                    continue
                fn = dec.func
                verb = getattr(fn, "attr", "")
                if verb in {"get", "post", "put", "patch", "delete"} and dec.args:
                    arg = dec.args[0]
                    if isinstance(arg, ast.Constant):
                        out.append((verb.upper(), prefix + arg.value, node.name))
    return sorted(out, key=lambda r: (r[1], r[0]))


def db_models() -> list[tuple[str, list[str]]]:
    p = BACKEND / "imatch_api" / "db" / "models.py"
    if not p.exists():
        return []
    tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
    out = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            fields = [s.target.id for s in node.body
                      if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)]
            if fields:
                out.append((node.name, fields))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_ROOT / "delivery/A3-SYSTEM-ARCHITECTURE.md"))
    args = ap.parse_args()

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_ROOT,
                            capture_output=True, text=True).stdout.strip()[:12]
    mods = all_modules()
    live = reachable(mods)
    dead = sorted(set(mods) - live)

    parts: list[str] = [f"""# A3 — System Architecture

**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} ·
**Repository state:** `{commit}`

| | |
|---|---|
| Python modules in tree | {len(mods)} |
| **Reachable from the running service** | **{len(live)}** |
| Present but not reached | {len(dead)} |
| HTTP endpoints | {len(routes())} |
| Persisted tables | {len(db_models())} |

---

## How to read this document

Module reachability is **computed** by walking the import graph from
`imatch_api.main`, not asserted from the directory listing.

That distinction is load-bearing. The source tree contains more modules than the
service uses: some are instrumentation, some are approaches that were superseded
and left in place, and some implement capabilities the product does not claim.
Listing every file as part of the delivered system would let a reader conclude
that a capability exists because a file named after it is present — for a
forensic system, that is exactly the wrong inference to invite.

**Both halves are published.** For a maintainer the unreachable list is the more
useful one, and for anyone assessing what the system actually does it is the
half that matters.

---

# Part I — Runtime architecture

## Process model

```
  Browser (React SPA, Vite)
        |  HTTPS, Bearer token or HTTPOnly cookie
        v
  FastAPI application  (imatch_api.main:app)
        |
        +-- middleware, outermost first
        |     CORSMiddleware            headers on every response, incl. refusals
        |     csrf_guard                double-submit on cookie-borne changes
        |     request_context           correlation id + security headers
        |
        +-- routers   auth, account, cases, subjects, search, audit, reports, admin, health
        |
        +-- services  engine_service, audit_service, accounts, mail, report_pdf
        |
        +-- db        SQLModel over SQLite or PostgreSQL
        |
        v
  nexgen_engine   (in-process, not a separate service)
        detection -> alignment -> quality -> embedding -> matching
        |
        v
  ONNX Runtime, CUDA execution provider
```

The recognition engine runs **in-process**. There is no model server: a request
thread calls the ONNX session directly. This bounds throughput (see A5,
concurrency: threading saturates at about four workers) and is the reason
batching, rather than more workers, is the scaling lever.

## Request path — 1:1 verification

1. `POST /api/imatch/verify` with two base64 images and a lawful basis.
2. Authentication resolves a principal from `Authorization` or `X-API-Key`.
3. Rate limit checked against the principal.
4. Lawful basis required — refused if absent when enforcement is on.
5. Each image: decode → **geometry guard** (min 16px edge, max 50:1 aspect) →
   detect → align to 112×112 → quality assessment → embed.
6. Cosine similarity, compared against the configured threshold.
7. Audit record written and hash-chained before the response is returned.

The geometry guard at step 5 exists because malformed input previously reached
the detector and raised from inside OpenCV, producing a 500 (A2, F-08).

## Request path — 1:N identification

As above through step 5, then the probe embedding is compared against the
tenant's gallery shard via exact inner-product search, ranked, and truncated to
`top_k`. Exact search is deliberate: approximate indexing was measured and its
recall loss was not judged acceptable for the lead-generation task (A5, ANN).

---

# Part II — Module inventory

## Reachable from the running service

These modules are executed, directly or transitively, by the delivered service.

| Module | File |
|---|---|
"""]
    for name in sorted(live):
        parts.append(f"| `{name}` | `{mods[name].relative_to(_ROOT).as_posix()}` |\n")

    parts.append(f"""
## Present in the tree but NOT reached from the service

**{len(dead)} modules.** Reasons vary and are not interchangeable:

- **Instrumentation** — benchmark and audit scripts invoked directly from the
  command line rather than by the service. These are part of the delivery and
  their outputs are in A5; they simply are not in the request path.
- **Superseded approaches** — earlier implementations retained in history.
- **Unclaimed capability** — a module whose name suggests a feature the product
  does not claim. The presence of a file is not a claim that the capability is
  delivered, and CLAIMS.md is the authority on what is claimed.

A reader assessing what this system does should treat this list as **not part of
the running system** unless a specific module is shown to be invoked by an
operator-facing workflow.

""")

    # Categorise rather than list flat: the reason a module is unreached is
    # what a reader needs, and the categories are not interchangeable.
    def category(name: str) -> str:
        if ".benchmarks" in name or ".analytics" in name:
            return ("Instrumentation — invoked from the command line, not the "
                    "service. Part of the delivery; outputs are in A5.")
        if ".training" in name or ".losses" in name or ".data." in name:
            return ("Training and dataset tooling — used to produce and audit "
                    "models offline. Never in a request path by design.")
        if ".export" in name:
            return "Packaging and model-export tooling — offline use."
        if name.startswith("nexgen_engine.api"):
            return ("Superseded — an earlier in-engine API layer, replaced by "
                    "`imatch_api`. Retained in history.")
        if "presentation_attack" in name:
            return ("**Unclaimed capability — deliberately not wired in.** The "
                    "product does NOT claim presentation-attack detection; the "
                    "liveness signal it does report is a heuristic marked "
                    "`certified: false`. This module being unreached is "
                    "consistent with that claim, not an oversight.")
        if "audit_logger" in name:
            return ("Superseded — the service uses "
                    "`imatch_api.services.audit_service` instead.")
        return "Not reached from the entry point; no specific role identified."

    buckets: dict[str, list[str]] = {}
    for name in dead:
        buckets.setdefault(category(name), []).append(name)

    for reason, names in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        parts.append(f"### {len(names)} module(s)\n\n{reason}\n\n| Module | File |\n|---|---|\n")
        for name in sorted(names):
            parts.append(f"| `{name}` | `{mods[name].relative_to(_ROOT).as_posix()}` |\n")
        parts.append("\n")

    # ------------------------------------------------------------ routes --
    parts.append("\n---\n\n# Part III — HTTP surface\n\n")
    parts.append("Extracted from the FastAPI route decorators.\n\n")
    parts.append("| Method | Path | Handler |\n|---|---|---|\n")
    for verb, path, fn in routes():
        parts.append(f"| {verb} | `{path}` | `{fn}` |\n")

    # ------------------------------------------------------------ models --
    parts.append("\n---\n\n# Part IV — Persisted data model\n\n")
    for name, fields in db_models():
        parts.append(f"## `{name}`\n\n")
        parts.append("| Field |\n|---|\n")
        for f in fields:
            parts.append(f"| `{f}` |\n")
        parts.append("\n")

    # ------------------------------------------------- key source files ---
    parts.append("\n---\n\n# Part V — Core implementation\n\n")
    parts.append("The modules on which the recognition result depends, in full.\n\n")
    for rel in ["backend/nexgen_engine/config.py",
                "backend/nexgen_engine/inference/pipeline.py",
                "backend/nexgen_engine/search/gallery_index.py",
                "backend/nexgen_engine/models/cuda_runtime.py",
                "backend/imatch_api/main.py",
                "backend/imatch_api/core/dependencies.py"]:
        p = _ROOT / rel
        if p.exists():
            parts.append(f"## `{rel}`\n\n```python\n"
                         f"{p.read_text(encoding='utf-8', errors='replace')}\n```\n\n")

    text = "".join(parts)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    words = len(text.split())
    print(f"wrote {out}")
    print(f"  {len(mods)} modules, {len(live)} reachable, {len(dead)} not reached")
    print(f"  {len(routes())} endpoints, {len(db_models())} tables")
    print(f"  {len(text.splitlines()):,} lines   {words:,} words")
    print(f"  ~{words / 450:.0f} pages @450 w/p   ~{words / 250:.0f} pages @250 w/p")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
