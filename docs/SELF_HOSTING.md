# Running NexGen iMATCH yourself — laptop now, personal server later

The 512 MB Render free tier cannot hold a 191 MB ONNX model plus Python, and
split hosting (frontend on Vercel, API on Render) made frontend and API
**different sites** to the browser, which is what kept breaking login (CORS
grants, CSP `connect-src`, and cross-site CSRF cookies all had to agree, and
any drift produced "Failed to fetch" or 403). This document is the supported
path instead: everything on hardware you control, behind **one origin**.

There are two modes, and the second is a superset of the first:

| | Laptop (development) | Personal server (production) |
|---|---|---|
| Backend | uvicorn from the repo, `.venv` | Docker container, built from the same code |
| Frontend | Vite dev server on :5173 | Built SPA served by nginx |
| Database | SQLite at `<repo>/runtime/imatch.db` | Postgres 16 in a container |
| Model weights | `~/.insightface` (already present) | Docker volume, downloaded once |
| Origins | localhost:5173 + localhost:8443 (same-site) | ONE origin via nginx `/api` proxy |
| Secrets | `.env` at repo root (already present) | `.env` next to the compose file |

---

## 1. Laptop mode (what you run today)

### Start the backend

```bash
.venv/Scripts/python.exe -m uvicorn imatch_api.main:app --app-dir backend --host 127.0.0.1 --port 8443
```

(Or use the `backend` entry in `.claude/launch.json`.) Startup loads the
recognition model and takes a few seconds; the service logs
`iMATCH ready: buffalo_l ...` when it is serving. Weights load from
`~/.insightface`, which this machine already has.

### Start the frontend

```bash
npm run dev --prefix frontend
```

Open http://localhost:5173. The dev client calls `localhost:8443` directly
(same-site, so the CSRF cookie works) and the backend's default
`NEXGEN_CORS_ORIGINS` already allows the dev origin.

### First sign-in

An admin exists for this machine's database (seeded 2026-08-02):

- email `nikhilljatt@gmail.com`, tenant `nexgen-demo` (leave the tenant field
  blank — it is only needed when one email exists in several tenants).
- The generated password was printed once by `seed.py`. Change it after first
  login. To create another account:

```bash
NEXGEN_SEED_ADMIN_EMAIL=you@example.com .venv/Scripts/python.exe backend/scripts/seed.py
```

### Things that used to go wrong here, now fixed in code

- `seed.py` created admins with `email_verified=False`, and login rejects
  unverified accounts — a freshly seeded system had no account that could log
  in. Seeded admins are now verified on creation.
- The SQLite path was relative to the *current directory*, so starting uvicorn
  from the repo root vs from `backend/` silently opened two different
  databases with different users. Relative paths are now anchored to the repo
  root; `<repo>/runtime/imatch.db` is the only database regardless of where
  you start the server. (`backend/runtime/imatch.db` is the obsolete
  accidental twin — delete it once you are sure nothing you need is in it.)

### Secrets

`<repo>/.env` already holds `NEXGEN_JWT_SECRET` and `NEXGEN_TEMPLATE_KEY`.
**Back up `NEXGEN_TEMPLATE_KEY`** somewhere safe: it encrypts stored biometric
templates, and losing it makes them permanently undecryptable.

---

## 2. Personal server mode

One command once Docker is installed on the server:

```bash
docker compose -f docker-compose.selfhost.yml up -d --build
```

The stack is three containers — `web` (nginx: SPA + `/api` reverse proxy),
`api` (FastAPI + recognition engine), `postgres`. The browser talks **only to
`web`**, so frontend and API are the same origin: no CORS configuration, no
cross-site cookies, no CSP allowlist to keep in step. The frontend image is
built with `VITE_IMATCH_API_BASE=same-origin`, which makes the bundle use
relative `/api` paths (this value is recognised by
`frontend/src/services/imatchApi.js`).

### Before first start

Create `.env` next to `docker-compose.selfhost.yml`:

```bash
POSTGRES_PASSWORD=<strong password>
NEXGEN_JWT_SECRET=<python -c "import secrets; print(secrets.token_urlsafe(64))">
NEXGEN_TEMPLATE_KEY=<python -c "import base64,os; print(base64.b64encode(os.urandom(32)).decode())">
```

Generate fresh values for the server — do not reuse the laptop's. Back up
`NEXGEN_TEMPLATE_KEY` (see above; this is the one unrecoverable secret).

### First start

The api container downloads the ~280 MB model pack once into the
`model_cache` volume (first boot takes a few minutes; watch
`docker compose -f docker-compose.selfhost.yml logs -f api` for
`iMATCH ready`). Then seed the first admin:

```bash
docker compose -f docker-compose.selfhost.yml exec api python scripts/seed.py
```

Open `http://<server>:8080` and sign in with the printed credentials.

### HTTPS

Put any TLS terminator in front of `web:8080` — Caddy is the least effort:

```
your.domain.com {
    reverse_proxy localhost:8080
}
```

Once HTTPS is on, set `NEXGEN_COOKIE_SECURE=true` in the server's `.env` and
`docker compose ... up -d` again. Do **not** set it while still on plain HTTP:
browsers refuse `Secure` cookies over http on non-localhost hosts, which
breaks the CSRF cookie and with it every login.

### Moving data from the laptop (when the time comes)

Nothing forces a migration — the server starts empty and you can simply
re-enrol. If you do want the laptop data: templates are encrypted with the
laptop's `NEXGEN_TEMPLATE_KEY`, so a copied database is only readable if the
server uses the **same** key. Copy `runtime/imatch.db` + the laptop `.env`
keys together, or start clean with fresh keys — not a mix.

### GPU later

`NEXGEN_ENGINE_DEVICE=cpu` is the compose default and is correct for most
servers. If the server has an NVIDIA GPU, install the NVIDIA container
toolkit, add a `deploy.resources.reservations.devices` block (or
`gpus: all`) to the `api` service, use `backend/requirements-gpu.txt` in the
image, and set `NEXGEN_ENGINE_DEVICE=auto`. Scores differ from CPU only at
the 4th decimal place (see render.yaml's header note); no decisions change.

---

## 3. The invariant that keeps login working

Every login failure this project has had in deployment came from one source:
**the frontend and the API being different sites**. If you ever split them
again (CDN frontend + API elsewhere), you must keep four things in agreement:
the frontend's `VITE_IMATCH_API_BASE`, the frontend CSP's `connect-src`, the
API's `NEXGEN_CORS_ORIGINS`, and cross-site cookie settings
(`NEXGEN_COOKIE_SAMESITE=none` + `NEXGEN_COOKIE_SECURE=true` — and Safari
still blocks third-party cookies entirely). Behind one origin, none of these
exist. Prefer one origin.
