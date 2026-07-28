<div align="center">

# NexGen iMATCH

**Facial recognition for forensic investigation.**

Enrol subjects, search a probe image against your gallery, adjudicate candidates
as an examiner, and produce a case report backed by a tamper-evident audit trail.

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![React](https://img.shields.io/badge/react-19-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-async-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Model](https://img.shields.io/badge/model-ArcFace%20%2F%20InsightFace-FF6F00)](docs/models.md)
[![Tests](https://img.shields.io/badge/tests-128%20passing-4c1)](docs/testing.md)
[![Licence](https://img.shields.io/badge/licence-MIT-blue)](LICENSE)

[Install](docs/installation.md) ·
[Configure](docs/configuration.md) ·
[Models](docs/models.md) ·
[Architecture](docs/architecture.md) ·
[Security](docs/security.md) ·
[Testing](docs/testing.md) ·
[Deploy](docs/deployment.md) ·
[Troubleshoot](docs/troubleshooting.md)

</div>

---

## It actually recognises people

No placeholder embeddings, no simulated scores. Run this yourself:

```bash
python backend/test_recognition.py --self-test
```

```text
Model loaded          : YES
Recognition network   : w600k_r50
Embedding dimensions  : 512
Search backend        : FAISS IndexFlatIP (exact)
Device in use         : cpu

Rank-1 identification : 23/25 = 92.0%
Genuine  pairs        : mean=0.4907
Impostor pairs        : mean=0.0422
Separation (mean gap) : 0.4485

THRESHOLD   TAR       FAR
0.28        0.880     0.0033
0.36        0.800     0.0000
0.42        0.720     0.0000
```

Measured on 25 AgeDB identities, CPU, `buffalo_l`. AgeDB is deliberately hard —
it varies age across decades. **These are this build's measurements on that
dataset, not a product accuracy claim.** Yours will differ; see
[calibration](docs/configuration.md#threshold-calibration).

---

## What it does, and what it does not

**It does:** rank enrolled subjects by visual similarity to a probe, gate out
probes too poor to search, and record who searched for what, when, and on what
stated authority.

**It does not identify anyone.** A similarity score is not the probability that
two images show the same person. Every result is an investigative lead requiring
examiner verification.

That position is enforced structurally, not just documented:

- Only a **human** can mark a candidate `confirmed`. The engine has no code path
  that writes it.
- A high score that **barely beats the runner-up** on a large gallery is
  downgraded to review — that pattern is the signature of a false match.
- Every search **requires a stated lawful basis**, recorded verbatim.
- If the recognition model cannot load, the service **refuses to start** rather
  than returning confident-looking numbers that mean nothing.

The documented failures of face recognition in investigations are overwhelmingly
failures of a human treating a ranked candidate as a conclusion.

---

## Pipeline

```
image → decode → SCRFD detection → 5-point landmark alignment
      → ArcFace embedding (flip-TTA) → L2 normalise
      → FAISS cosine search → ranked candidates → examiner adjudication
```

~320 ms per image end to end on CPU. [Details](docs/models.md#measured-performance).

---

## Quick start

```bash
cd backend && python -m venv .venv && .venv\Scripts\activate
```

```bash
pip install -r requirements.txt -r requirements-engine.txt
```

```bash
copy ..\.env.example ..\.env    # then generate the two secrets
```

```bash
python scripts/seed.py && uvicorn imatch_api.main:app --port 8443
```

```bash
cd ../frontend && npm install && npm run dev
```

Open <http://localhost:5173>, sign in, and go to `/workspace`.

Full walkthrough, including secret generation and your first search:
**[Installation](docs/installation.md)**.

---

## Layout

```
backend/nexgen_engine/   Pure recognition — no HTTP, database, or auth
backend/imatch_api/      FastAPI service, persistence, auth, audit
frontend/src/workspace/  Investigator workspace
docs/                    Full documentation
```

The engine carries no service concerns, so it can be embedded, benchmarked and
tested on its own. [Architecture](docs/architecture.md).

---

## Known limitations

Stated plainly, because a forensic tool that oversells itself is worse than one
that does less.

- **Accuracy is unbenchmarked.** No NIST FRVT submission, no independent
  evaluation. The figures above are reproducible measurements, not a claim.
- **Demographic performance is unmeasured.** Error rates are known to vary
  across demographic groups. A deployment that skips measuring this will
  distribute its errors unevenly across the people it is used on, and nobody
  will notice.
- **Liveness, deepfake and morphing screens are heuristics**, not evaluated
  against ISO/IEC 30107-3. A competent attacker defeats all three.
- **Head pose is approximate** — derived from five landmarks, not a 3-D solver.
- **The shipped model is ResNet50** (`w600k_r50`), not R100. `antelopev2`
  carries R100 and is selectable, but is unbenchmarked here.
- **Rate limiting is per process**; behind multiple workers the effective limit
  multiplies. Put a shared store or edge limiter in front.
- **Access tokens are stateless** — revocation waits for expiry.
- **The audit chain proves integrity, not completeness.** Someone with database
  access could truncate the tail; ship the JSONL to append-only storage.
- **PostgreSQL migrations are not wired.** Schema is created from SQLModel
  metadata at startup; add Alembic before your first production schema change.
- **Not implemented:** batch/CSV intake, server-side URL import (deliberately
  disabled as an SSRF vector), and dataset-level ROC/AUC/CMC evaluation.

Pre-production checklist: [Security & governance](docs/security.md#before-production).

---

## Licence

[MIT](LICENSE).

Licensing does not grant lawful authority to process biometric data. That is
yours to establish, per your jurisdiction and use case.
