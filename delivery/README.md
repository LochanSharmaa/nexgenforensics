# NexGen iMATCH — Delivery Package

**System version:** 1.0.0 · **Package issued:** 2026-08-01

This package documents the delivered facial-recognition system: its
architecture, implementation, deployment, operation, validation, maintenance
and usage.

---

## Principle governing this package

**Every claim is traceable to a command that reproduces it.** For a system
whose outputs may be relied upon in decisions about real people — and examined
adversarially — documentation volume is not the measure of quality. A single
unsupported sentence in a validation report is worth more to opposing counsel
than a thousand pages of correct ones are worth to the operator.

Accordingly:

- No performance figure appears anywhere in this package that was not measured
  on this system, on a named dataset, under a stated protocol.
- Where a capability is **unmeasured**, it is listed as unmeasured.
- Where a result was **negative**, it is reported as negative.
- Sections that cannot yet be written truthfully are marked **NOT YET
  DELIVERABLE** with the reason, rather than filled with plausible text.

---

## Contents

| # | Document | Status |
|---|---|---|
| 01 | Installation Guide | Planned |
| 02 | User Manual | Planned |
| 03 | API Documentation | Partially exists — OpenAPI at `/docs`, `/openapi.json` |
| **04** | **Model Card** | **Delivered** — `04-MODEL-CARD.md` |
| 05 | Performance Report | Source data complete (`BENCHMARKS.md`, `runtime/benchmarks/*.json`); formal document planned |
| 06 | Validation Report | Partial — internal testing and edge cases measured; **external validation NOT YET DELIVERABLE** (see below) |
| 07 | Dataset Documentation | Source data complete (BENCHMARKS §2b, §6c, §6e); formal document planned |
| 08 | Deployment Package | Partial — `render.yaml`, `requirements-deploy.txt`, `Dockerfile` exist; **not yet validated on a live deployment** |
| 09 | Inference Pipeline Spec | Partially exists — OpenAPI schemas; worked examples planned |
| 10 | Security Documentation | Partial — implemented controls listed below; formal document planned |
| 11 | Maintenance Package | Partial — `regression_check.py` exists; retraining pipeline documented in BENCHMARKS §6 |
| 12 | Test Suite | **Delivered** — 209 automated tests, `backend/tests/`, `backend/tests_engine/` |
| 13 | Licensing and Legal | **NOT YET DELIVERABLE** — requires legal counsel, not engineering |
| 14 | Knowledge Transfer | **NOT YET DELIVERABLE** — requires scheduled sessions with the receiving organisation |

---

## Component inventory (forensic-specific)

| Component | Delivered | Notes |
|---|---|---|
| Face detection | Yes | SCRFD-10G |
| Face alignment | Yes | 5-point similarity transform → 112×112 |
| Face enhancement / super-resolution | **Deliberately absent** | Generative restoration is architecturally excluded from the embedding path — see Model Card §2.4 |
| Feature extraction / embedding | Yes | ArcFace ResNet-50, 512-d |
| Face matching engine | Yes | Cosine similarity, exact search |
| Quality assessment | Yes | `ImageQualityFilter`, 0–1 aggregate + sub-scores |
| Anti-spoofing / liveness | **Heuristic only** | Reported `certified: false`; NOT presentation-attack detection |
| Search / index database | Yes | SQLite/PostgreSQL + encrypted templates; exact search |
| Threshold calibration report | Yes | BENCHMARKS §5a/§5c |
| Known failure cases | Yes | Model Card §5, §7 |
| Chain-of-custody / audit guidance | Partial | Hash-chained audit log implemented; operational guidance planned |
| Validation on forensic datasets | Yes — with a caveat | TinyFace (native low-res) measured; **33.13% TAR@FAR0.1%** is the headline limitation |

---

## What cannot be claimed, and why

These are stated at the front of the package rather than buried:

1. **No independent validation.** Every figure is self-measured. The system has
   not been submitted to NIST FRVT or any external evaluation. This is the
   single largest credibility gap and no internal work can close it.

2. **Not validated in live deployment.** All verification is local. There is no
   hosted instance whose behaviour under real traffic has been observed.

3. **Degraded-imagery recall is low** — 33.13% TAR at FAR=0.1% on TinyFace,
   improvable to 37.37% with optional routing. Both figures should be read
   before any operational reliance on low-quality source imagery.

4. **Demographic differentials are measured and unresolved** — women ~1.7× the
   false-non-match rate of men; under-25s ~3.8× the 41–55 band.

5. **The liveness signal is not anti-spoofing.**

---

## Security controls implemented

| Control | Status |
|---|---|
| Password hashing | Argon2id |
| Session tokens | JWT access + refresh, refresh rotated on use and revoked on logout |
| Template encryption at rest | AES-256-GCM |
| Transport | HTTPS required in production; HSTS set |
| Audit trail | Hash-chained, append-only, verifiable |
| Access control | Role hierarchy (investigator < supervisor < admin), enforced server-side |
| Rate limiting | Per-principal and per-IP, per-flow |
| Account lockout | 5 failures → 15 minutes |
| E-mail verification | 6-digit OTP, hashed, 10-minute expiry, 5-attempt cap |
| CSRF | Signed double-submit on cookie-borne state changes |
| Security headers | CSP, HSTS, COOP, CORP, Permissions-Policy, X-Frame-Options, nosniff |
| Model integrity | **Delivered** — SHA-256 for every weight file, A4 Part I |

---

## Reproducing every figure in this package

```bash
python backend/scripts/benchmark_verification.py
python backend/scripts/benchmark_tinyface.py
python backend/scripts/benchmark_demographics.py
python backend/scripts/evaluate_routed_engine.py --threshold 0.539
python -m pytest backend/tests backend/tests_engine
```

Raw outputs land in `runtime/benchmarks/*.json`. The narrative record,
including negative results and superseded decisions, is `BENCHMARKS.md`.
