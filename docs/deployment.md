# Deployment

[← Back to README](../README.md)

## Docker Compose

```bash
docker compose up --build
```

Requires `POSTGRES_PASSWORD`, `NEXGEN_JWT_SECRET` and `NEXGEN_TEMPLATE_KEY` in
the environment or a `.env` file. Compose fails fast rather than starting on
defaults.

Two named volumes matter:

- `model_cache` → `/models`. Without it the ~300 MB pack downloads on every
  container start.
- `api_data` → `/data`. Holds probe/enrolment storage and the audit JSONL.

## The image

Built from `backend/deployment/Dockerfile`:

```bash
cd backend
```

```bash
docker build -f deployment/Dockerfile -t nexgen-imatch-api:1.0.0 .
```

- Runs as an unprivileged user (uid 10001)
- `libgomp1` and `libglib2.0-0` are runtime requirements of onnxruntime and
  OpenCV respectively
- Healthcheck with a 90 s start period, because model loading is not instant
- **One worker by default.** Each worker loads its own copy of the model
  (hundreds of MB) and keeps its own in-memory gallery. Scale with replicas, not
  `--workers`, until the gallery moves to a shared vector store.

For a fully offline image, copy a pre-downloaded pack into `/models` at build
time.

## Kubernetes

Manifests in `backend/deployment/k8s/`:

| File | Purpose |
|---|---|
| `deployment.yaml` | Pod spec, probes, resource limits, secret wiring |
| `service.yaml` | ClusterIP |
| `ingress.yaml` | External routing |
| `hpa.yaml` | Autoscaling on CPU |
| `network_policy.yaml` | Default-deny with narrow egress |

Notable choices:

- **`runAsNonRoot`, `readOnlyRootFilesystem`, all capabilities dropped.**
- **A startup probe with 30 failures allowed** gives model loading room without
  slackening the liveness probe once running.
- **Egress is narrow** — DNS and PostgreSQL only. The blast radius of a
  compromised dependency in a service holding biometric templates is the point;
  it must not be able to open arbitrary outbound connections and exfiltrate a
  gallery. Pre-provision the model pack rather than allowing runtime download.
- **Ingress is restricted to the ingress-controller namespace.** A bare
  `namespaceSelector: {}` would admit every pod in the cluster.

Secrets expected in `nexgen-imatch-secrets`: `jwt-secret`, `template-key`,
`database-url`.

> Rotating `template-key` makes every previously enrolled template permanently
> unreadable. Back it up before any key change.

## Reverse proxy

`backend/deployment/nginx.conf` sets timeouts above nginx's 60 s default (a
timeout mid-search leaves the client unable to distinguish failure from a
genuine no-match), a 24 MB body limit above the application's own 15 MB ceiling
so the API can return a proper error, and a per-source-address rate limit in
front of the application's per-principal limiter.

It **overwrites** `X-Forwarded-For` rather than appending. The usual
`$proxy_add_x_forwarded_for` preserves a client-supplied header, which would
then be recorded as the caller's address in the audit trail.

## Frontend

```bash
cd frontend
```

```bash
npm run build
```

Outputs to `dist/`. `vercel.json` configures SPA rewrites and security headers,
including a CSP that blocks inline scripts.

Set `VITE_IMATCH_API_BASE` to the API's absolute `https://` origin. The client
refuses a non-HTTPS endpoint in a production build.

## Database

SQLite is the development default. For production set `NEXGEN_DATABASE_URL` to
PostgreSQL.

The schema is currently created from SQLModel metadata at startup. **Migrations
are not yet wired** — add Alembic before your first production schema change.
