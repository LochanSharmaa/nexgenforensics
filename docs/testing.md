# Testing

[← Back to README](../README.md)

```bash
cd backend
```

```bash
pip install -r requirements-dev.txt
```

```bash
pytest
```

**128 tests**, run against the real recognition model rather than a stub.

| File | Covers |
|---|---|
| `test_engine.py` | Umeyama transform, pose estimation, quality metrics, decoding, template encryption, decision logic |
| `test_gallery_index.py` | Tenant isolation, ranking, subject collapsing, mutation, validation |
| `test_recognition_engine.py` | **Real recognition**: separation, rank-1 identification, FAISS/numpy agreement |
| `test_api_auth.py` | Login, tokens, RBAC, API keys |
| `test_api_workflow.py` | Cases, enrolment, search governance, end-to-end identification |
| `test_audit.py` | Hash-chain integrity, tamper detection, chain ordering |

---

## The recognition tests

`test_recognition_engine.py` is the file that proves the engine does its job.
Every other test can pass while the system is incapable of recognising anyone —
which is exactly the state this codebase was previously in.

It asserts measured behaviour, never a fixed accuracy figure:

- same-person pairs outscore different-person pairs by a clear margin
- false-match rate stays below 2 % at the configured threshold
- rank-1 identification finds the right subject in a gallery
- FAISS and numpy return identical scores
- embeddings are **not** hashes — asserted directly, since that is the specific
  defect this repository once shipped
- pre-cropped faces are detected via pad-and-retry

These tests need real photographs and skip cleanly without them:

```bash
NEXGEN_TEST_DATASET=/path/to/dataset pytest tests/test_recognition_engine.py -v
```

The dataset is one directory per identity with two or more images each, or a
flat AgeDB-style layout (`<index>_<Name>_<age>_<sex>.jpg`). AgeDB under
`src_extracted/` is picked up automatically.

**Run these before trusting a deployment.**

### Why there are no synthetic faces

Test fixtures use real photographs and verify detectability up front. A drawn
oval is not a face, the detector correctly refuses it, and a test that passed on
one would be testing nothing.

Identities are also filtered before use: an identity is only offered to tests if
at least two of its photographs genuinely produce a template. Any real dataset
contains unusable images — profile shots, occlusion, motion blur — and a test
that happened to draw one would fail for reasons unrelated to the behaviour it
checks.

---

## End-to-end demonstration

The human-verifiable proof, outside pytest:

```bash
python test_recognition.py --self-test
```

```bash
python test_recognition.py --enrol path/to/gallery --probe probe.jpg
```

```bash
python test_recognition.py --compare a.jpg b.jpg
```

```bash
python test_recognition.py --status
```

Prints model status, faces detected, embedding dimensions, ranked candidates
with similarity scores, and per-stage timings.

---

## Frontend

```bash
cd frontend
```

```bash
npm run lint
```

```bash
npm run build
```

Note on lint: core `no-unused-vars` does not understand JSX, so an identifier
used only as `<motion.div>` gets reported as unused. `eslint-plugin-react`'s
`jsx-uses-vars` rule is enabled to prevent that — acting on those reports by
deleting imports breaks the build at runtime.

---

## Environment check

Verifies the toolchain before anything else runs:

```bash
python scripts/check_environment.py
```

Exit codes: `0` recognition works, `1` service runs but cannot recognise,
`2` service cannot run at all.
