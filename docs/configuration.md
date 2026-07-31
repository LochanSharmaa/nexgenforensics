# Configuration

[← Back to README](../README.md)

All configuration comes from environment variables, read from `.env` at the
repository root or from the process environment. Every variable is prefixed
`NEXGEN_`. The authoritative definitions, including validation, live in
[`backend/imatch_api/core/config.py`](../backend/imatch_api/core/config.py);
[`.env.example`](../.env.example) is the annotated template.

---

## Service

| Variable | Default | Purpose |
|---|---|---|
| `NEXGEN_ENV` | `development` | `development`, `staging`, `production`, `test`. Production enforces secret requirements and disables `/docs` |
| `NEXGEN_API_HOST` | `0.0.0.0` | Bind address |
| `NEXGEN_API_PORT` | `8443` | Bind port |
| `NEXGEN_LOG_LEVEL` | `INFO` | Standard Python levels |
| `NEXGEN_CORS_ORIGINS` | `http://localhost:5173` | Comma-separated. Never `*` — these endpoints carry biometric data behind credentialed requests |

## Database

| Variable | Default |
|---|---|
| `NEXGEN_DATABASE_URL` | `sqlite:///./runtime/imatch.db` |

PostgreSQL: `postgresql+psycopg://user:password@host:5432/imatch`. SQLite is fine
for development but will not survive concurrent production load.

## Security

| Variable | Default | Purpose |
|---|---|---|
| `NEXGEN_JWT_SECRET` | *(empty)* | **Required in production**, minimum 32 characters |
| `NEXGEN_JWT_ALGORITHM` | `HS256` | |
| `NEXGEN_ACCESS_TOKEN_MINUTES` | `60` | Tokens are stateless, so this is the revocation window |
| `NEXGEN_REFRESH_TOKEN_DAYS` | `7` | |
| `NEXGEN_TEMPLATE_KEY` | *(empty)* | **Required in production**, 32 random bytes base64-encoded |
| `NEXGEN_RATE_LIMIT_PER_MINUTE` | `120` | General endpoints |
| `NEXGEN_SEARCH_RATE_LIMIT_PER_MINUTE` | `30` | Biometric search specifically |

Both secrets are deliberately empty in `.env.example`. A placeholder long enough
to satisfy the production length check would let a copied example file ship a
publicly known signing key. Empty fails fast in production and generates an
ephemeral per-process value in development.

Search gets its own lower ceiling because a bulk gallery scrape looks exactly
like a burst of ordinary searches.

## Engine

| Variable | Default | Purpose |
|---|---|---|
| `NEXGEN_MODEL_PACK` | `buffalo_l` | `buffalo_l`, `buffalo_s`, `antelopev2` |
| `NEXGEN_MODEL_ROOT` | *(empty)* | Defaults to `~/.insightface` |
| `NEXGEN_ENGINE_DEVICE` | `cpu` | `cpu` or `cuda` |
| `NEXGEN_MATCH_THRESHOLD` | `0.42` | Above this, a candidate match |
| `NEXGEN_REVIEW_THRESHOLD` | `0.32` | Between review and match, examiner adjudicates |
| `NEXGEN_VERIFY_THRESHOLD` | `0.42` | 1:1 comparison |
| `NEXGEN_MIN_QUALITY` | `0.35` | Aggregate quality floor |
| `NEXGEN_MIN_DETECTION_CONFIDENCE` | `0.50` | Below this, no face is considered found |

`NEXGEN_REVIEW_THRESHOLD` must not exceed `NEXGEN_MATCH_THRESHOLD`; the service
refuses to start otherwise, because an empty review band silently auto-accepts
borderline scores.

`cuda` is a request, not a guarantee — see
[Models & performance](models.md#gpu-acceleration).

## Governance

| Variable | Default | Purpose |
|---|---|---|
| `NEXGEN_REQUIRE_LAWFUL_BASIS` | `true` | Refuses any search without a stated basis |
| `NEXGEN_PROBE_RETENTION_DAYS` | `90` | `0` disables automatic purge |
| `NEXGEN_AUDIT_LOG_PATH` | `./runtime/audit.jsonl` | Mirror of the hash chain |

Leave `NEXGEN_REQUIRE_LAWFUL_BASIS` on outside testing. The system cannot judge
whether a search is lawful, but it can require that somebody states a reason and
preserve that statement alongside the result.

## Storage

| Variable | Default |
|---|---|
| `NEXGEN_STORAGE_ROOT` | `./runtime/storage` |
| `NEXGEN_MAX_UPLOAD_MB` | `15` |

## Bootstrap

Used only by `scripts/seed.py`: `NEXGEN_SEED_TENANT`,
`NEXGEN_SEED_ADMIN_EMAIL`, `NEXGEN_SEED_ADMIN_PASSWORD`. If the password is
empty a strong one is generated and printed once.

---

## Threshold calibration

The shipped thresholds are generic ArcFace operating points, not validated
settings. False-match rate at a fixed threshold rises with gallery size and
degrades with image quality, so a threshold that is safe for 500 subjects can be
badly wrong for 50,000.

Measure the genuine and impostor distributions on your own imagery:

```bash
# Full published benchmark suite (LFW/AgeDB-30/CFP-FP/CALFW/CPLFW) -- use this
# to pick the deployed threshold. See BENCHMARKS.md section 5c.
python scripts/calibrate_threshold_suite.py --model w600k_r50

# Your OWN operational imagery, one directory per identity (folders only,
# cannot read the .bin protocol packs):
python scripts/calibrate_threshold.py path/to/dataset --max-identities 500
```

The dataset is one directory per identity, two or more images each. The script
reports the threshold for each target false-match rate.

**Pick from the FMR your use of the system can tolerate, not from the equal error
rate.** In an investigative context a false match points at the wrong person, so
a stricter FMR is usually correct even though it misses more true matches.

Set `NEXGEN_MATCH_THRESHOLD` accordingly and keep `NEXGEN_REVIEW_THRESHOLD`
roughly 0.08–0.12 lower, so borderline scores reach an examiner instead of being
silently dropped.
