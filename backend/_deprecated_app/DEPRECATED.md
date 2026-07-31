# DEPRECATED — NOT THE NEXGEN iMATCH BACKEND

**Do not build on this directory. Do not restore it. Do not point the frontend at it.**

This was `backend/app/`. It is a leftover from a *different product* and has
never been part of the facial recognition system.

## Evidence

1. **It cannot start.** `app/core/config.py` was never committed, so
   `uvicorn app.main:app` dies with
   `ModuleNotFoundError: No module named 'app.core.config'`. Only stale
   `__pycache__/*.pyc` files remain in `app/core/`.
2. **Its routes belong to another product** — `routes_brief`, `routes_concepts`,
   `routes_images`, `routes_billing`, `routes_projects`. None of these are
   biometric.
3. **The Dockerfile does not deploy it.** `backend/deployment/Dockerfile` ends
   with:
   `CMD ["uvicorn", "imatch_api.main:app", "--host", "0.0.0.0", "--port", "8443", ...]`

## The real backend

`backend/imatch_api/` on port **8443** — cases, reports, subjects, search,
audit, admin. That is canonical. See the repository README.

The one genuinely biometric file here, `api/routes_biometrics.py`, was an
unauthenticated multipart duplicate of `imatch_api`'s
`POST /api/imatch/verify`. It carried a third, stale copy of the decision
thresholds (0.42/0.28) that did not match either the engine or the calibrated
values in `BENCHMARKS.md`. Thresholds now live in exactly one place:
`nexgen_engine/config.py::ThresholdConfig`.

This directory is kept only so a future reader can confirm the above rather
than rediscovering the fork and re-deciding it. It can be deleted outright.
