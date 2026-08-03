# Image Intelligence & OSINT Investigation Platform

**An uploaded image is the entry point to a structured public-information investigation.** The platform discovers where an image appears publicly, collects what those pages say, correlates it across sources, and presents evidence with the chain back to its origin intact.

> **This platform performs no facial recognition.** It does not identify people from facial features, compare faces, rank people by facial similarity, or build biometric databases. Identity information is reported only where a discovered public page explicitly publishes it, with attribution. See [ARCHITECTURE §1.1](docs/ARCHITECTURE.md) — the prohibition is enforced by CI guards and schema design, not by policy alone.

---

## Status

| | |
|---|---|
| **Phase** | 4 of 15 complete, plus image ingest |
| **Tests** | 161 passing (42 tables, 28 endpoints) |
| **Next** | Phase 5 — surface inside the NexGen iMATCH investigator workspace |

Design documents: [Architecture](docs/ARCHITECTURE.md) · [Data model](docs/DATA_MODEL.md) · [API](docs/API.md) · [Roadmap](docs/ROADMAP.md) · [Revision 3](docs/REVISION_3.md)

---

## Quick start

```bash
docker compose up -d --wait
```

Then seed the local investigator account — the password is printed once and is not recoverable:

```bash
docker compose exec api python -m scripts.bootstrap --email you@example.com
```

API docs at `http://localhost:8000/api/docs`, health at `/health`, readiness at `/health/ready`.

### Without Docker

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

```bash
alembic upgrade head && uvicorn api.main:app --reload
```

---

## The design principle

Everything is **evidence-first**. Every statement, entity, relationship and timeline event links to one or more supporting observations, and the system never outputs an unsupported conclusion. Each finding answers five questions, and every one has a home in the schema rather than being reconstructed at render time:

- Where did this come from?
- Which pages support it?
- When was it observed?
- How many *independent* sources support it?
- Are there conflicting sources?

That fourth question is the subtle one. Forty outlets republishing one wire story is **one** journalistic source, not forty; `example.com` and `blog.example.com` are one operator. Independence is counted over distinct registrable domains after collapsing duplicate-content clusters, because the naive count would systematically overstate confidence.

When a trade-off arises, the binding priority order is: **evidence traceability → explainability → investigator workflow → reproducibility → extensibility → maintainability → performance**. Faster or simpler loses to traceable and reproducible.

---

## What is built

| Component | Detail |
|---|---|
| `shared/` | Typed config validated at startup, structured logging, Prometheus metrics, typed error hierarchy, UUIDv7 ids, injected clock, hash-chain primitives, and every state machine as pure data |
| `database/` | 42 tables across workspace, sources, evidence, graph, annotation and operations; five DB-level invariants; repositories per bounded context; 2 Alembic migrations with tested downgrades |
| `api/` | 25 endpoints: auth, investigation CRUD and workflow, pipeline runs with resume, SSE progress, retention holds, review queue, evidence chain, audit verification, RFC 9457 problem responses |
| `image_discovery/` | Ingest: SHA256, pHash/dHash/wHash, dimensions, EXIF with GPS flagging. File content only — no facial analysis |
| `worker/` | arq skeleton with a database round-trip task; heavy runtime lands from Phase 6 |
| CI | Forbidden-dependency guard, architecture boundary tests, migration up/down/`alembic check` against real PostgreSQL, Compose build and health verification |

### Guards that run before any feature code

The compliance job runs **first and alone** in CI — if a face-recognition library is present, nothing else about the build matters.

- **Lockfile guard** (`scripts/check_forbidden_deps.py`) inspects what is actually installed, catching a transitive pull-in that no source file imports. It also runs at Docker build time, so an image containing one cannot be produced.
- **Architecture tests** assert the domain layer imports no infrastructure, persistence never imports the interface layer, routers construct no infrastructure, no module imports a face library, and no ANN/vector index exists anywhere — a face gallery must have nowhere to live.

These went in at Phase 2 on purpose. A boundary added after the code it constrains is a boundary already crossed.

---

## Integrity guarantees

**Two hash chains, one construction.** The audit log answers *"what did the system do?"*; the custody chain answers *"what happened to this artifact?"* — an examiner challenging a screenshot asks the second. Each record's hash covers its content plus its predecessor, so editing history breaks every subsequent hash and `verify()` reports the first divergent index. Both are insert-only at the privilege level (`REVOKE UPDATE, DELETE`), so tampering needs superuser access *and* still breaks the chain.

**Refusals are audited too.** A request rejected for a missing lawful basis is exactly what an auditor asks about later, and the entry is committed independently of the failed transaction so the rollback cannot erase it.

**Human input never becomes machine evidence.** `observations` are `CHECK`-constrained to `EXTRACTED` origin and `notes` to `HUMAN`. A note *cites* evidence; it can never turn into it. An investigator's hypothesis must not become indistinguishable from a crawled fact when the report is challenged months later.

**States are validated, never assigned.** Every transition routes through a pure transition table in the domain layer. Backward moves — reopening a completed case — require a recorded reason.

---

## Testing

```bash
pytest -q
```

Tests run against SQLite and need no running services; PostgreSQL-specific guarantees are covered by the migration job in CI.

---

## Honest limits

- An image never published online returns nothing. There is no index of unpublished images and there lawfully cannot be one.
- Discovery recall is bounded by the providers. What Google and TinEye have not crawled, this cannot find.
- A name on a page is a **claim by that page**. Pages are stale, miscaptioned, syndicated, and occasionally deliberately false.
- Domain classification is descriptive, never evaluative. `.gov` means the domain is governmental; it does not mean the page is true, and classification is barred from influencing any confidence score.

## Licence

MIT.
