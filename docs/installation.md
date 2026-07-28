# Installation

[← Back to README](../README.md)

## Requirements

| | Minimum | Notes |
|---|---|---|
| Python | 3.11 | Verified on 3.12.10 |
| Node.js | 20 | Verified on 24.18 (frontend only) |
| RAM | 8 GB | The ONNX session holds ~1 GB resident |
| Disk | ~2 GB | ~300 MB model pack, remainder dependencies |
| CPU | x86-64, 4 cores | Inference is CPU-bound by default |
| GPU | Optional | See [Models & performance](models.md#gpu-acceleration) |
| Database | SQLite (dev) / PostgreSQL 14+ (production) | |

CPU-only is a fully supported configuration, not a degraded one: the model and
the arithmetic are identical, only throughput differs.

---

## 1. Backend

```bash
cd backend
```

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

On macOS or Linux use `source .venv/bin/activate`.

```bash
pip install -r requirements.txt -r requirements-engine.txt
```

`requirements-engine.txt` carries InsightFace, ONNX Runtime, OpenCV and FAISS.
Without it the service will not start at all — there is no fallback recognition
mode. See [Models & performance](models.md).

### Configuration

```bash
copy ..\.env.example ..\.env
```

Generate the two required secrets and paste them into `.env`:

```bash
python -c "import secrets; print('NEXGEN_JWT_SECRET=' + secrets.token_urlsafe(64))"
```

```bash
python -c "import base64,os; print('NEXGEN_TEMPLATE_KEY=' + base64.b64encode(os.urandom(32)).decode())"
```

> **Back up `NEXGEN_TEMPLATE_KEY`.** It encrypts every stored biometric
> template. Losing or changing it makes every enrolled template permanently
> unreadable, with no recovery path — which is precisely what makes the
> encryption worth having.

Full variable reference: [Configuration](configuration.md).

### First run

```bash
python scripts/seed.py
```

Creates the first tenant and administrator, printing a generated password once.

```bash
uvicorn imatch_api.main:app --host 0.0.0.0 --port 8443 --reload
```

The first start downloads the InsightFace `buffalo_l` pack (~300 MB) into
`~/.insightface/models/`. Later starts take a few seconds.

API documentation: <http://localhost:8443/docs>

---

## 2. Frontend

```bash
cd frontend
```

```bash
npm install
```

```bash
npm run dev
```

Open <http://localhost:5173>. The dev server proxies `/api` to port 8443, so the
browser stays same-origin and no CORS grant is needed.

Sign in at `/login` with the credentials `seed.py` printed, then go to
`/workspace`.

---

## 3. Verify the install

Before trusting anything, confirm the engine actually recognizes people:

```bash
cd backend
```

```bash
python test_recognition.py --status
```

```bash
python test_recognition.py --self-test
```

The self-test needs a labelled face dataset. See
[Testing](testing.md#the-recognition-tests).

---

## 4. First search

1. **Enrol** at least one subject (`/workspace/enrol`, supervisor role). A search
   against an empty gallery correctly returns nothing.
2. **Open a case** at `/workspace` and record its lawful basis.
3. **Search** at `/workspace/search` with a probe image.
4. **Adjudicate** each candidate — confirm, eliminate, or mark inconclusive.
5. **Export** the case report from the case page.

---

## Troubleshooting

Common failures and their causes: [Troubleshooting](troubleshooting.md).
