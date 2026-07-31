#!/usr/bin/env python
"""
Generate A6, A8, A10 and A11 of the delivery package.

    python scripts/generate_remaining_docs.py

  A6  Dataset Provenance and Integrity
  A8  API and Interface Specification
  A10 Security Architecture and Controls
  A11 Deployment and Operations

Each is assembled from the repository so it cannot drift from the system it
describes: A8 reads the FastAPI routes and Pydantic schemas, A10 reads the
security modules, A11 reads the real deployment configuration, and A6 reads the
contamination audits that were actually run.

Where a section cannot be written truthfully it says so. Sections marked NOT
DELIVERABLE are not gaps in this generator; they are capabilities the system
does not yet have, and stating that is the document's job.
"""

from __future__ import annotations

import ast
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
BACKEND = _ROOT / "backend"
BENCH = _ROOT / "runtime" / "benchmarks"


def head(title: str, subtitle: str = "") -> str:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_ROOT,
                            capture_output=True, text=True).stdout.strip()[:12]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (f"# {title}\n\n**Generated:** {stamp} · **Repository state:** "
            f"`{commit}`\n\n{subtitle}\n\n---\n\n")


def source(rel: str) -> str:
    p = _ROOT / rel
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def docstring(rel: str) -> str:
    t = source(rel)
    if '"""' not in t:
        return ""
    s = t.index('"""') + 3
    e = t.find('"""', s)
    return t[s:e].strip() if e > s else ""


def artefact(name: str):
    p = BENCH / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def embed_json(name: str, data) -> str:
    return (f"<details><summary>Raw artefact — <code>{name}</code></summary>\n\n"
            f"```json\n{json.dumps(data, indent=2)}\n```\n\n</details>\n\n")


# ============================================================== A6 =========

def build_a6() -> str:
    p = [head("A6 — Dataset Provenance and Integrity",
              "Every corpus this system was trained or evaluated on: where it came "
              "from, what licence governs it, what was done to establish that "
              "training and evaluation data do not overlap, and what remains "
              "unverifiable.")]

    p.append("""## Corpora

| Corpus | Role | Identities | Licence | Provenance traceable? |
|---|---|---|---|---|
| MS1M / WebFace600K | Upstream training of the deployed `w600k_r50` | ~600,000 | Research use (InsightFace) | Public, verifiable |
| CASIA-WebFace | Clean anchor for the experimental degraded pack | 10,572 (9,880 after exclusion) | Research use | Public, verifiable |
| QMUL-SurvFace | Real degraded capture for the experimental pack | 5,319 | **Research purposes only** | **No** — see below |
| LFW, AgeDB-30, CFP-FP, CFP-FF, CALFW, CPLFW | Evaluation | — | Research use | Public, verifiable |
| TinyFace | Degraded-condition evaluation | — | Research use | Public, verifiable |

## The provenance that cannot be established

QMUL-SurvFace ships **no licence file**; `readme.txt` contains structure and a
citation only. The published terms state the dataset is "made available for
research purposes", and — materially — that *"all the images were collected from
the existing person re-identification datasets, and the copyright belongs to the
original owners"*.

Those upstream owners are **not enumerated** by the distributor. One dataset
commonly used in that field (DukeMTMC) was withdrawn by its own authors on
ethics grounds; whether it contributed cannot be determined from the files
supplied.

**Consequence.** Any model trained on QMUL-SurvFace inherits a research-only
restriction and an untraceable upstream chain. The deployed recognition pack
does **not** carry this; the experimental degraded pack does, and enabling it is
therefore a licensing decision as well as a technical one.

Neither the upstream corpora nor QMUL-SurvFace were collected with the informed
consent of the individuals depicted. This is an industry-wide condition rather
than one specific to this system, and it is disclosed rather than elided.

---

## Contamination control

A model evaluated on identities it was trained on reports memorisation as
accuracy. Overlap was therefore audited rather than assumed, in both directions.

""")

    for name, title, note in [
        ("train_eval_overlap.json", "CASIA — evaluation-side audit",
         "For each evaluation image, similarity to the nearest training image."),
        ("overlap_casia_deep.json", "CASIA — full-depth audit",
         "Repeated at full identity coverage (105,631 images, all 10,572 identities). "
         "Five times the images found 2.7x the hits, confirming that sampling gives "
         "a floor rather than a complete answer."),
        ("exclusion_list.json", "CASIA — exclusion list",
         "The training-side inverse: for each training identity, similarity to the "
         "nearest evaluation image. **692 of 10,572 identities (6.5%) excluded** at "
         "threshold 0.40, chosen deliberately below the ~0.49 genuine-pair mean "
         "because excluding a clean identity is cheap and keeping a contaminated one "
         "makes every downstream number unfalsifiable."),
        ("qmul_exclusion_list.json", "QMUL-SurvFace — raw audit result",
         "Reported 96.9% of identities above threshold. **This figure is an artefact "
         "and must not be quoted as contamination** — see the control below."),
        ("qmul_overlap_control.json", "QMUL-SurvFace — the control that overturned it",
         "Distinct QMUL directories are distinct people by construction, which makes "
         "a matched null directly measurable. The nearest DIFFERENT-PERSON QMUL image "
         "(0.600) scores HIGHER than the nearest TinyFace image (0.522), and a true "
         "same-person QMUL pair medians only 0.316. Degraded embeddings cluster by "
         "quality, not identity. **No identity overlap was established; nothing was "
         "excluded.**"),
        ("qmul_quality.json", "QMUL-SurvFace — capture characteristics",
         "Confirms native low-resolution capture rather than downsampled clean "
         "imagery: median 27x22px, 84.1% below 32px high, smallest 7x5."),
    ]:
        data = artefact(name)
        p.append(f"### {title}\n\n{note}\n\n")
        if data is None:
            p.append(f"*Artefact `{name}` not present in this working tree.*\n\n")
        else:
            p.append(embed_json(name, data))

    p.append("""---

## Method, as implemented

""")
    for rel in ["backend/scripts/audit_train_eval_overlap.py",
                "backend/scripts/build_exclusion_list.py",
                "backend/scripts/audit_qmul_survface.py",
                "backend/scripts/qmul_overlap_control.py"]:
        d = docstring(rel)
        if d:
            p.append(f"### `{rel}`\n\n```text\n{d}\n```\n\n")

    p.append("""---

## Limitations of these audits

1. **Sampling gives a floor, not a proof.** An identity whose sampled images
   happen not to resemble an evaluation image is not excluded. The true
   contaminated set is at least as large as the one found, never smaller.
2. **Similarity is not identity.** The audits detect embedding proximity. On
   degraded imagery that proximity is driven by quality as much as by identity,
   which is precisely what produced the QMUL false positive above.
3. **Upstream corpora were not audited against each other.** The deployed model's
   training set is not distributed in a form that permits it.
""")
    return "".join(p)


# ============================================================== A8 =========

def build_a8() -> str:
    routes: list[tuple[str, str, str, str]] = []
    for f in sorted((BACKEND / "imatch_api" / "api" / "routes").glob("*.py")):
        try:
            tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        prefix = ""
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "APIRouter":
                for kw in n.keywords:
                    if kw.arg == "prefix" and isinstance(kw.value, ast.Constant):
                        prefix = kw.value.value
        for n in ast.walk(tree):
            if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for dec in n.decorator_list:
                if isinstance(dec, ast.Call) and getattr(dec.func, "attr", "") in {
                        "get", "post", "put", "patch", "delete"} and dec.args:
                    a = dec.args[0]
                    if isinstance(a, ast.Constant):
                        routes.append((dec.func.attr.upper(), prefix + a.value,
                                       n.name, ast.get_docstring(n) or ""))
    routes.sort(key=lambda r: (r[1], r[0]))

    p = [head("A8 — API and Interface Specification",
              f"The complete HTTP surface: {len(routes)} endpoints, their handlers, "
              "the request and response schemas, and the authentication and "
              "governance rules that apply to them.")]

    p.append("""## Conventions

| Aspect | Rule |
|---|---|
| Base URL | Configured per deployment; `http://127.0.0.1:8443` in development |
| Authentication | `Authorization: Bearer <access token>` or `X-API-Key: <key>` |
| Content type | `application/json`; images as base64 strings, not multipart |
| Errors | `{"detail": "..."}`, with `request_id` on 500 |
| CSRF | Required on state-changing requests that do NOT carry a header credential |
| Rate limiting | Per-principal, and per-IP on unauthenticated auth endpoints |
| Caching | `Cache-Control: no-store` on every response |

**Lawful basis.** Every biometric operation requires a stated lawful basis when
`NEXGEN_REQUIRE_LAWFUL_BASIS` is on. The system does not evaluate whether a
basis is lawful; it ensures one was stated and records it verbatim in the audit
chain.

---

## Endpoint index

| Method | Path | Handler |
|---|---|---|
""")
    for verb, path, fn, _ in routes:
        p.append(f"| {verb} | `{path}` | `{fn}` |\n")

    p.append("\n---\n\n## Endpoint detail\n\n")
    for verb, path, fn, doc in routes:
        p.append(f"### {verb} `{path}`\n\n**Handler:** `{fn}`\n\n")
        if doc:
            p.append(f"```text\n{doc.strip()}\n```\n\n")
        else:
            p.append("*No handler docstring.*\n\n")

    p.append("---\n\n## Request and response schemas\n\n")
    p.append("Pydantic models defining every documented request and response "
             "shape, in full.\n\n")
    p.append(f"```python\n{source('backend/imatch_api/api/schemas.py')}\n```\n\n")
    p.append("## Machine-readable specification\n\n"
             "A complete OpenAPI 3 document is served by the running service at "
             "`/openapi.json`, with interactive documentation at `/docs` when not "
             "in production. That document is generated from the same type "
             "annotations reproduced above, so the two cannot disagree.\n")
    return "".join(p)


# ============================================================= A10 =========

def build_a10() -> str:
    p = [head("A10 — Security Architecture and Controls",
              "Implemented controls, the reasoning behind each, and the threats "
              "this system does not defend against.")]

    p.append("""## Implemented controls

| Control | Implementation | Notes |
|---|---|---|
| Password hashing | Argon2id | Rehashed transparently when cost parameters change |
| Password strength | 12+ characters, 3 of 4 character classes | Length-and-variety floor, not a composition maze |
| Session tokens | JWT access + refresh, typed claims | Refresh rotated on every use; replay of a used token fails |
| Session revocation | Refresh hash stored; cleared on logout and on password reset | `session_epoch` bumped on reset so already-issued access tokens can be rejected |
| E-mail verification | 6-digit OTP, SHA-256 hashed, 10-minute expiry, 5-attempt cap | SHA-256 rather than Argon2 deliberately — see below |
| Account lockout | 5 failed logins, 15-minute lock | Checked BEFORE the credential verdict |
| Rate limiting | Sliding window, per-principal and per-IP per-flow | Single-process; see limitations |
| Template encryption | AES-256-GCM at rest | Authenticated encryption; tampering is detected, not just undetected |
| Transport | HTTPS required in production, HSTS | `Strict-Transport-Security: max-age=63072000` |
| CSRF | Signed double-submit | Scoped to cookie-borne state changes; header-credentialed requests exempt |
| Security headers | CSP, COOP, CORP, Permissions-Policy, X-Frame-Options, nosniff | CSP is `default-src 'none'` for API responses |
| Access control | Role hierarchy, enforced server-side on every request | UI gating is usability, not security |
| Audit trail | Hash-chained, append-only | Tampering with an earlier record invalidates the chain |
| Model integrity | SHA-256 published for every weight file | A4 Part I |

---

## Reasoning behind the non-obvious choices

### OTP is hashed with SHA-256, not Argon2

Unlike a password, a 6-digit code has only 10^6 of entropy. No key-derivation
function saves it from an offline attack against a stolen database — a million
guesses succeeds regardless of cost parameters. What protects it is the
**10-minute expiry** and the **5-attempt cap**, both enforced online. Spending
Argon2 cost on the hash would slow a hot path while changing nothing about the
actual threat.

### CSRF exempts header-credentialed requests

A browser attaches cookies to cross-site requests automatically; it does not
attach `Authorization` or `X-API-Key`, and CORS preflight blocks a cross-origin
page from setting them. A request carrying a header credential is therefore
already immune, and demanding a token would break every API client while
protecting nothing.

What the guard does protect is **login CSRF** — an attacker forcing a victim's
browser into an account the attacker controls, so the victim's subsequent
searches are recorded in the audit chain against the wrong person. On a system
whose audit trail is intended as evidence, that is more serious than the usual
framing of CSRF suggests.

### Enumeration is closed on every unauthenticated endpoint

`register`, `resend-otp` and `forgot-password` return identical responses
whether or not an address exists, and the unverified-email refusal on login is
raised only **after** the password check. An attacker must not be able to use a
forensic system's auth endpoints to discover who holds an account on it.

### Login timing is flattened

A login for a non-existent account still performs an Argon2 verification against
a dummy hash, so response time does not distinguish "no such user" from "wrong
password".

---

## Threats NOT defended against

Stated so that no reader assumes coverage that does not exist.

| Threat | Status |
|---|---|
| **Presentation attack / spoofing** | **Not defended.** The liveness figure is a heuristic reported with `certified: false`. It is not a trained PAD classifier and the `presentation_attack` module is not wired into the service. |
| Distributed rate-limit evasion | Rate limiting is per-process. Behind multiple workers the effective limit multiplies by worker count, and it resets on restart. A shared store or edge limiter is required for a real defence. |
| Adversarial perturbation of input images | Not evaluated. No robustness measurement exists. |
| Model extraction via query volume | Not defended beyond ordinary rate limiting. |
| Insider misuse by an authorised user | Detectable after the fact through the audit chain; not prevented. |
| Compromise of the host | Out of scope. Template encryption protects data at rest, not a running process holding keys. |

---

## Audit chain

Every biometric operation, authentication event and administrative action is
recorded with timestamp, actor, IP address, user agent, outcome and the stated
lawful basis. Records are hash-chained: each entry incorporates the digest of
its predecessor, so altering or removing an earlier record invalidates every
record after it. Verification is available to administrators through
`/api/audit/verify`.

---

## Implementation

""")
    for rel in ["backend/imatch_api/core/security.py",
                "backend/imatch_api/core/csrf.py",
                "backend/nexgen_engine/security/template_encryption.py"]:
        s = source(rel)
        if s:
            p.append(f"### `{rel}`\n\n```python\n{s}\n```\n\n")
    return "".join(p)


# ============================================================= A11 =========

def build_a11() -> str:
    p = [head("A11 — Deployment and Operations",
              "How the delivered system is installed, configured, run and "
              "monitored, and what has and has not been validated in deployment.")]

    p.append("""## Validation status

**The system has not been validated in a live deployment.** All verification in
this package was performed locally. There is no hosted instance whose behaviour
under real traffic, real concurrency or real failure conditions has been
observed. This is stated first because every operational figure below should be
read with it in mind.

---

## Requirements

| Component | Requirement |
|---|---|
| Python | 3.11 |
| GPU (recommended) | NVIDIA with CUDA 12.x; ~2 GB VRAM for the recognition pack |
| GPU (verified on) | RTX A3000 Laptop / RTX 3060, `CUDAExecutionProvider` |
| CPU fallback | Supported and correct, but roughly an order of magnitude slower |
| Database | SQLite for single-node; PostgreSQL for multi-node |
| Disk | ~350 MB models, plus gallery templates |
| Memory | ~2 GB service, plus model residency |

**CPU and GPU results differ in the last decimal places** (0.494431 vs
0.494213 on a reference pair). This is ordinary floating-point divergence
between execution providers. An earlier claim that results were identical was
incorrect and has been corrected.

---

## Configuration

All configuration is by environment variable, prefixed `NEXGEN_` except where a
third party fixes the name.

| Variable | Purpose | Production requirement |
|---|---|---|
| `NEXGEN_JWT_SECRET` | Token signing | **Must be set.** Absent, an ephemeral key is generated and every restart invalidates all sessions |
| `NEXGEN_TEMPLATE_KEY` | Template encryption at rest | **Must be set**, or templates cannot be decrypted after restart |
| `NEXGEN_DATABASE_URL` | Database | `postgresql+psycopg://` for multi-node |
| `NEXGEN_CORS_ORIGINS` | Allowed browser origins | Never `*` with credentials |
| `NEXGEN_COOKIE_SECURE` | Secure cookie flag | **`true` behind HTTPS** |
| `NEXGEN_ENGINE_DEVICE` | `auto`, `cuda` or `cpu` | `auto` fails loudly if CUDA was expected and not bound |
| `NEXGEN_REQUIRE_LAWFUL_BASIS` | Governance enforcement | `true` |
| `NEXGEN_ALLOW_SELF_REGISTRATION` | Public signup | **`false`** unless deliberately opened |
| `RESEND_API_KEY` | Transactional e-mail | Backend only. Never expose to a browser bundle |

---

## Installation

```bash
python -m venv .venv && .venv/bin/pip install -r backend/requirements.txt
python scripts/setup_gpu.py            # GPU only; deconflicts onnxruntime builds
python scripts/verify_gpu.py           # 12 checks; refuse to proceed if it fails
python backend/scripts/migrate_auth_columns.py
python backend/scripts/bootstrap_admin.py --password '<strong password>'
python -m uvicorn imatch_api.main:app --host 0.0.0.0 --port 8443 --app-dir backend
```

`setup_gpu.py` exists because `insightface` installs a CPU `onnxruntime`
alongside `onnxruntime-gpu`, and the CPU build wins at import. Without it the
service runs on CPU while reporting no error at all — a failure that produced
correct results at a tenth of the speed and went unnoticed until throughput was
questioned.

---

## Operational limits

| Property | Measured value | Source |
|---|---|---|
| Thread concurrency | Saturates at ~4 workers (1.86x) | A5, concurrency |
| Request batching | 2.82x at batch 32 | A5, concurrency |
| Scaling lever | **Batching, not more workers** | The engine runs in-process |

---

## Monitoring

| Signal | Where |
|---|---|
| Liveness | `GET /api/health` |
| Engine status and bound provider | `GET /api/imatch/engine/status` |
| Latency percentiles | `LATENCY` collector, bounded ring buffer |
| Audit chain integrity | `GET /api/audit/verify` (admin) |
| Correlation | `X-Request-ID` on every response |

**Verify the bound execution provider after every deployment.** A service that
has silently fallen back to CPU is the highest-frequency operational failure
this system has had, and it does not announce itself.

---

## Backup and recovery

| Asset | Consequence of loss |
|---|---|
| Database | Gallery, cases and audit chain lost |
| `NEXGEN_TEMPLATE_KEY` | **Every enrolled template is permanently unrecoverable.** Back up separately from the database, or the backup and the key are lost together |
| `NEXGEN_JWT_SECRET` | All sessions invalidated; recoverable by re-authentication |
| Model weights | Re-downloadable; verify against the SHA-256 digests in A4 Part I |

---

## Deployment configuration as delivered

""")
    for rel in ["render.yaml", "backend/requirements-deploy.txt", "vercel.json"]:
        s = source(rel)
        if s:
            lang = "yaml" if rel.endswith(".yaml") else (
                "json" if rel.endswith(".json") else "text")
            p.append(f"### `{rel}`\n\n```{lang}\n{s}\n```\n\n")
    return "".join(p)


def main() -> int:
    outputs = [
        ("delivery/A6-DATASET-PROVENANCE.md", build_a6),
        ("delivery/A8-API-SPECIFICATION.md", build_a8),
        ("delivery/A10-SECURITY-ARCHITECTURE.md", build_a10),
        ("delivery/A11-DEPLOYMENT-AND-OPERATIONS.md", build_a11),
    ]
    total = 0
    for rel, builder in outputs:
        text = builder()
        p = _ROOT / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        w = len(text.split())
        total += w
        print(f"  {rel:44s} {w:7,} words  ~{w / 450:5.0f}p  ~{w / 250:5.0f}p")
    print(f"  {'TOTAL':44s} {total:7,} words  ~{total / 450:5.0f}p  ~{total / 250:5.0f}p")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
